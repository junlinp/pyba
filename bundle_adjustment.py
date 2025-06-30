#!/usr/bin/env python3
"""
Bundle Adjustment using pyceres and pycolmap

This module provides bundle adjustment functionality using pyceres for optimization
and pycolmap for cost functions and reconstruction data structures.
"""

import numpy as np
import pyceres
from typing import List, Tuple, Optional
import torch
import einops


class CameraModel:
    def __init__(self,  K: np.ndarray):
        self.K = K
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]

    def project(self, X: np.ndarray) -> np.ndarray:
        """
        Project 3D points to 2D points.
        """
        X = X.reshape(-1, 3)
        X = X @ self.K.T
        X = X / X[:, 2:3]
        return X[:, :2]

    # return function value and jacobian
    def project_tensor(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project 3D points to 2D points.
        """
        # X should be requested gradient
        X = X.view(-1, 3)
        X.retain_grad()  # Enable gradient retention for non-leaf tensor
        
        # Convert K to tensor with same device and dtype as X
        K_tensor = torch.tensor(self.K, device=X.device, dtype=X.dtype)
        X_proj = X @ K_tensor.T
        X_proj = X_proj / X_proj[:, 2:3]
        
        # Compute Jacobians for each component
        jacobians = []
        for i in range(2):  # x and y components
            if X.grad is not None:
                X.grad.zero_()
            X_proj[0, i].backward(retain_graph=True)
            jacobians.append(X.grad.clone())
        
        return X_proj[:, :2], torch.stack(jacobians, dim=0).squeeze(1)

def skew_symmetric(v):
    # v: (3,)
    return torch.stack([
        torch.stack([torch.zeros_like(v[0]), -v[2], v[1]]),
        torch.stack([v[2], torch.zeros_like(v[0]), -v[0]]),
        torch.stack([-v[1], v[0], torch.zeros_like(v[0])])
    ])

def transform_tensor(pose_camera_to_world: torch.Tensor, X: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Transform 3D points to 3D points.
    pose_camera_to_world: [6] vector, [t, R]
    X: [3] tensor
    """
    translation = pose_camera_to_world[:3]
    lie_algebra = pose_camera_to_world[3:]

    # Convert Lie algebra to rotation matrix using differentiable operations
    theta = torch.norm(lie_algebra)
    I = torch.eye(3, device=lie_algebra.device, dtype=lie_algebra.dtype)
    if theta < 1e-6:
        K = skew_symmetric(lie_algebra)
        R = I + K
    else:
        k = lie_algebra / theta
        K = skew_symmetric(k)
        R = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)

    # Transform point from world to camera coordinates
    point_in_camera = R.T @ (X - translation)
    print(f"point_in_camera: {point_in_camera.shape}")
    
    jacobian_pose_camera_to_world = []
    jacobian_X = []

    pose_camera_to_world.retain_grad()
    X.retain_grad()

    for i in range(3):
        if pose_camera_to_world.grad is not None:
            pose_camera_to_world.grad.zero_()
        if X.grad is not None:
            X.grad.zero_()
        
        point_in_camera[i].backward(retain_graph=True)

        assert pose_camera_to_world.grad is not None
        assert X.grad is not None
        jacobian_pose_camera_to_world.append(pose_camera_to_world.grad.clone())
        jacobian_X.append(X.grad.clone())

    # should be (3, 6)
    jacobian_pose_camera_to_world = torch.stack(jacobian_pose_camera_to_world, dim=0)
    assert jacobian_pose_camera_to_world.shape == (3, 6)

    # should be (3, 3)
    jacobian_X = torch.stack(jacobian_X, dim=0)
    assert jacobian_X.shape == (3, 3)

    return point_in_camera, [jacobian_pose_camera_to_world, jacobian_X]

class ReprojErrorCost(pyceres.CostFunction):
    def __init__(self, x_2d: torch.Tensor, camera_model: CameraModel):
        pyceres.CostFunction.__init__(self)
        self.set_num_residuals(2)
        self.set_parameter_block_sizes([6, 3])
        self.x_2d = x_2d
        self.camera_model = camera_model

    def Evaluate(self, pose_parameters:np.ndarray, point_3d_parameters:np.ndarray, residuals:np.ndarray, jacobians:np.ndarray):
        pose_parameters = torch.tensor(pose_parameters, requires_grad=True)
        point_3d_parameters = torch.tensor(point_3d_parameters, requires_grad=True)

        point_in_camera, jacobians_transform = transform_tensor(pose_parameters, point_3d_parameters)
        x_2d_pred, jacobians_camera_model = self.camera_model.project_tensor(point_in_camera)
        
        # Compute residuals: predicted - observed
        residual_tensor = x_2d_pred - self.x_2d
        residuals[:] = residual_tensor.detach().numpy()
        
        if jacobians is not None:
            # Jacobian chain rule: d(residual)/d(pose) = d(residual)/d(point_in_camera) * d(point_in_camera)/d(pose)
            # jacobians_camera_model: [2, 3], jacobians_transform[0]: [3, 6]
            if jacobians[0] is not None:
                jacobians[0][:] = (jacobians_camera_model @ jacobians_transform[0]).detach().numpy()
            
            # d(residual)/d(point_3d) = d(residual)/d(point_in_camera) * d(point_in_camera)/d(point_3d)
            # jacobians_camera_model: [2, 3], jacobians_transform[1]: [3, 3]
            if jacobians[1] is not None:
                jacobians[1][:] = (jacobians_camera_model @ jacobians_transform[1]).detach().numpy()
        
        return True

class BundleAdjuster:
    """
    Bundle adjustment optimizer using pyceres and pycolmap.
    """
    
    def __init__(self, fix_first_pose: bool = True, fix_intrinsics: bool = True):
        """
        Initialize the bundle adjuster.
        
        Args:
            fix_first_pose: Whether to fix the first camera pose
            fix_intrinsics: Whether to fix camera intrinsics
        """
        self.fix_first_pose = fix_first_pose
        self.fix_intrinsics = fix_intrinsics
    
    def create_reconstruction_from_data(self, points_3d: np.ndarray, 
                                      observations: List[Tuple[int, int, np.ndarray]], 
                                      camera_poses: List[np.ndarray], 
                                      intrinsics: np.ndarray) :
        """
        Create a pycolmap reconstruction from the input data.
        
        Args:
            points_3d: 3D points as (N, 3) array
            observations: List of (point_idx, frame_idx, point_2d) tuples
            camera_poses: List of 4x4 camera pose matrices
            intrinsics: 3x3 camera intrinsics matrix
            
        Returns:
            pycolmap.Reconstruction object
        """
        rec = pycolmap.Reconstruction()
        
        # Add camera
        w, h = 1242, 375  # KITTI image dimensions
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        cam = pycolmap.Camera(
            model="SIMPLE_PINHOLE",
            width=w,
            height=h,
            params=np.array([fx, cx, cy]),
            camera_id=0,
        )
        rec.add_camera(cam)
        
        # Add 3D points
        for i, point_3d in enumerate(points_3d):
            rec.add_point3D(point_3d, pycolmap.Track(), np.zeros(3))
        
        # Add images and observations
        for frame_idx, pose in enumerate(camera_poses):
            # Convert 4x4 pose to rotation and translation
            R = pose[:3, :3]
            t = pose[:3, 3]
            
            # Create rotation and translation objects
            rotation = pycolmap.Rotation3d(R)
            translation = t
            
            # Create rigid transform
            cam_from_world = pycolmap.Rigid3d(rotation, translation)
            
            # Create image
            im = pycolmap.Image(
                id=frame_idx,
                name=str(frame_idx),
                camera_id=cam.camera_id,
                cam_from_world=cam_from_world,
            )
            
            # Add observations for this image
            points2d = []
            for point_idx, obs_frame_idx, point_2d in observations:
                if obs_frame_idx == frame_idx:
                    points2d.append(pycolmap.Point2D(point_2d, point_idx))
            
            im.points2D = pycolmap.ListPoint2D(points2d)
            rec.add_image(im)
        
        return rec
    
    def define_problem(self, rec) -> pyceres.Problem:
        """
        Define the bundle adjustment problem.
        
        Args:
            rec: pycolmap.Reconstruction object
            
        Returns:
            pyceres.Problem object
        """
        prob = pyceres.Problem()
        loss = pyceres.TrivialLoss()
        
        for im in rec.images.values():
            cam = rec.cameras[im.camera_id]
            for p in im.points2D:
                if p.point3D_id in rec.points3D:
                    # Create cost function for reprojection error
                    cost = pycolmap.cost_functions.ReprojErrorCost(
                        cam.model, p.xy
                    )
                    # Add residual block with pose and point parameters
                    pose = im.cam_from_world
                    params = [
                        pose.rotation.quat,
                        pose.translation,
                        rec.points3D[p.point3D_id].xyz,
                        cam.params,
                    ]
                    prob.add_residual_block(cost, loss, params)
                    
                    # Set quaternion manifold for rotation
                    prob.set_manifold(
                        pose.rotation.quat, pyceres.EigenQuaternionManifold()
                    )
        
        # Fix camera intrinsics if requested
        if self.fix_intrinsics:
            for cam in rec.cameras.values():
                prob.set_parameter_block_constant(cam.params)
        
        # Fix first pose if requested
        if self.fix_first_pose and len(rec.images) > 0:
            first_image = rec.images[0]
            prob.set_parameter_block_constant(first_image.cam_from_world.rotation.quat)
            prob.set_parameter_block_constant(first_image.cam_from_world.translation)
        
        return prob
    
    def solve(self, prob: pyceres.Problem) -> pyceres.SolverSummary:
        """
        Solve the bundle adjustment problem.
        
        Args:
            prob: pyceres.Problem object
            
        Returns:
            pyceres.SolverSummary object
        """
        print(f"Problem: {prob.num_parameter_blocks()} parameter blocks, "
              f"{prob.num_parameters()} parameters, "
              f"{prob.num_residual_blocks()} residual blocks, "
              f"{prob.num_residuals()} residuals")
        
        options = pyceres.SolverOptions()
        options.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
        options.minimizer_progress_to_stdout = True
        options.num_threads = -1
        
        summary = pyceres.SolverSummary()
        pyceres.solve(options, prob, summary)
        
        print(summary.BriefReport())
        return summary
    
    def run(self, points_3d: np.ndarray, 
            observations: List[Tuple[int, int, np.ndarray]], 
            camera_poses: List[np.ndarray], 
            intrinsics: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray, float]:
        """
        Run bundle adjustment.
        
        Args:
            points_3d: Initial 3D points as (N, 3) array
            observations: List of (point_idx, frame_idx, point_2d) tuples
            camera_poses: Initial camera poses as list of 4x4 matrices
            intrinsics: Camera intrinsics as 3x3 matrix
            
        Returns:
            Tuple of (optimized_camera_poses, optimized_points_3d, final_reprojection_error)
        """
        print("Bundle adjustment: {} cameras, {} points, {} observations".format(
            len(camera_poses), len(points_3d), len(observations)))
        
        # Create reconstruction
        rec = self.create_reconstruction_from_data(points_3d, observations, camera_poses, intrinsics)
        
        # Define problem
        problem = self.define_problem(rec)
        
        # Solve
        summary = self.solve(problem)
        
        # Extract results
        optimized_camera_poses = []
        for i in range(len(rec.images)):
            im = rec.images[i]
            pose = np.eye(4)
            pose[:3, :3] = im.cam_from_world.rotation.rotation_matrix()
            pose[:3, 3] = im.cam_from_world.translation
            optimized_camera_poses.append(pose)
        
        optimized_points_3d = np.array([p.xyz for p in rec.points3D.values()])
        
        # Calculate final reprojection error
        final_reprojection_error = summary.final_cost / summary.num_residuals if summary.num_residuals > 0 else 0.0
        
        return optimized_camera_poses, optimized_points_3d, final_reprojection_error


def run_bundle_adjustment_pyceres(points_3d, points_2d, camera_poses, intrinsics):
    """
    Legacy function for backward compatibility.
    """
    ba = BundleAdjuster(fix_first_pose=True, fix_intrinsics=True)
    
    # Convert points_2d to observations format
    observations = []
    for i, point_2d_list in enumerate(points_2d):
        for j, point_2d in enumerate(point_2d_list):
            observations.append((j, i, point_2d))
    
    return ba.run(points_3d, observations, camera_poses, intrinsics) 