#!/usr/bin/env python3
"""
Bundle Adjustment using pyceres

This module provides bundle adjustment functionality using pyceres for optimization.
"""

import numpy as np
import pyceres
from typing import List, Tuple
from rotation import rotation_matrix_to_angle_axis, angle_axis_to_rotation_matrix, so3_right_jacobian, skew_symmetric
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

class ReprojErrorCost(pyceres.CostFunction):
    """
    Reprojection error cost function for bundle adjustment.
    Compatible with pyceres - uses only NumPy operations.
    """
    def __init__(self, x_2d: np.ndarray, camera_model: CameraModel):
        super().__init__()
        self.x_2d = np.array(x_2d, dtype=np.float64)
        self.camera_model = camera_model
        self.set_num_residuals(2)
        self.set_parameter_block_sizes([6, 3])

    def Evaluate(self, parameters, residuals, jacobians):
        pose_parameters = parameters[0]
        point_3d_parameters = parameters[1]
        t = pose_parameters[:3]
        w = pose_parameters[3:]
        R = angle_axis_to_rotation_matrix(w)
        point_in_camera = R.T @ (point_3d_parameters - t)
        x_2d_pred = self.camera_model.project(point_in_camera.reshape(1, 3)).flatten()
        residuals[:] = x_2d_pred - self.x_2d

        # compute the jacobian
        fx = self.camera_model.K[0, 0]
        fy = self.camera_model.K[1, 1]
        x, y, z = point_in_camera
        z_inv = 1.0 / z if z != 0 else 1e-10
        J_camera = np.array([
            [fx * z_inv, 0, -fx * x * z_inv**2],
            [0, fy * z_inv, -fy * y * z_inv**2]
        ])
        J_t = -R.T
        point_world_minus_t = point_3d_parameters - t
        J_w = R.T @ skew_symmetric(point_world_minus_t) @ so3_right_jacobian(-w)
        J_pose = np.zeros((2, 6), dtype=np.float64)
        J_pose[:, :3] = J_camera @ J_t
        J_pose[:, 3:] = J_camera @ J_w
        J_point = J_camera @ R.T

        if jacobians is not None:
            if jacobians[0] is not None:
                jacobians[0][:] = J_pose.flatten('C')
            if jacobians[1] is not None:
                jacobians[1][:] = J_point.flatten('C')
        return True
    

class BundleAdjuster:
    """
    Bundle adjustment optimizer using pyceres.
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
    
    def solve_bundle_adjustment(self, points_3d: dict[int, np.ndarray], 
                                observations: List[Tuple[int, int, np.ndarray]], 
                                camera_poses: dict[int, np.ndarray], 
                                intrinsics: np.ndarray):
        """
        Solve the bundle adjustment problem.
        """
        problem, pose_params_dict, point_params_dict = self.define_problem(points_3d, observations, camera_poses, intrinsics)
        summary = self.solve(problem)
        
        # Convert updated pose_params back to camera poses
        optimized_camera_poses = {}

        for timestamp, pose_param in pose_params_dict.items():
            t = pose_param[:3]
            w = pose_param[3:]
            R = angle_axis_to_rotation_matrix(w)
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = t
            optimized_camera_poses[timestamp] = pose
        
        return summary, optimized_camera_poses, point_params_dict
    
    def define_problem(self, points_3d: dict[int, np.ndarray], 
                       observations: List[Tuple[int, int, np.ndarray]], 
                       camera_poses: dict[int, np.ndarray], 
                       intrinsics: np.ndarray):
        """
        Define the bundle adjustment problem.
        
        Args:
            rec: Dictionary containing reconstruction data
            
        Returns:
            pyceres.Problem object
        """
        prob = pyceres.Problem()
        loss = pyceres.HuberLoss(2.0)
        
        # Extract data
        # Create camera model
        camera_model = CameraModel(intrinsics)
        
        # Convert camera poses to parameter format [t_x, t_y, t_z, w_x, w_y, w_z]
        pose_params = {}
        for timestamp, pose in camera_poses.items():
            t = pose[:3, 3]  # translation
            R = pose[:3, :3]  # rotation matrix
            # Convert rotation matrix to quaternion, then to angle-axis
            w = rotation_matrix_to_angle_axis(R)  # angle-axis representation
            pose_param = np.concatenate([t, w]).astype(np.float64)
            pose_params[timestamp] = pose_param
        
        # Store 3D points as parameter blocks (one array per point)
        point_params = {landmark_id: pt.astype(np.float64) for landmark_id, pt in points_3d.items()}
        
        # Add residual blocks for each observation
        for landmark_id, timestamp, point_2d in observations:
            cost = ReprojErrorCost(point_2d, camera_model)
            pose_param = pose_params[timestamp]  # always the same object
            point_param = point_params[landmark_id]  # always the same object
            prob.add_residual_block(cost, loss, [pose_param, point_param])
        
        # Fix first pose if requested
        if self.fix_first_pose and len(pose_params) > 0:
            first_timestamp = np.min(np.array(list(pose_params.keys())))
            prob.set_parameter_block_constant(pose_params[first_timestamp])
        
        # Fix camera intrinsics if requested
        if self.fix_intrinsics:
            # Note: intrinsics are not currently parameterized in this setup
            pass
        
        print(f"Created problem with {len(observations)} observations")
        return prob, pose_params, point_params
    
    def solve(self, prob: pyceres.Problem):
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
        options.linear_solver_type = pyceres.LinearSolverType.SPARSE_SCHUR
        options.num_threads = 16
        options.minimizer_progress_to_stdout = True
        options.max_num_iterations = 50
        options.function_tolerance = 1e-6
        
        summary = pyceres.SolverSummary()
        pyceres.solve(options, prob, summary)
        print(summary.FullReport())
        return summary
    
    def run(self, points_3d: dict[int, np.ndarray], 
            observations: List[Tuple[int, int, np.ndarray]], 
            camera_poses: dict[int, np.ndarray], 
            intrinsics: np.ndarray):
        """
        Run bundle adjustment.
        
        Args:
            points_3d: Initial 3D points as dict[int, np.ndarray]
            observations: List of (point_idx, frame_idx, point_2d) tuples
            camera_poses: Initial camera poses as dict[int, np.ndarray]
            intrinsics: Camera intrinsics as 3x3 matrix
            
        Returns:
            Tuple of (optimized_camera_poses, optimized_points_3d, final_reprojection_error)
        """
        print("Bundle adjustment: {} cameras, {} points, {} observations".format(
            len(camera_poses), len(points_3d.keys()), len(observations)))
        
        # Define problem
        problem, pose_params, point_params = self.define_problem(points_3d, observations, camera_poses, intrinsics)
        
        # Store initial values for comparison
        initial_pose_params = {timestamp: pose_param.copy() for timestamp, pose_param in pose_params.items()}
        initial_point_params = {timestamp: point_param.copy() for timestamp, point_param in point_params.items()}
        
        # Solve
        summary = self.solve(problem)

        # Check parameter changes
        pose_changes = []
        for timestamp, (init_pose, final_pose) in enumerate(zip(initial_pose_params, pose_params)):
            change = np.linalg.norm(init_pose - final_pose)
            pose_changes.append(change)
            if timestamp < 3:  # Print first 3 poses
                print(f"Pose {timestamp} change: {change:.6f}")
        
        point_changes = []
        for timestamp, (init_point, final_point) in enumerate(zip(initial_point_params, point_params)):
            change = np.linalg.norm(init_point - final_point)
            point_changes.append(change)
            if timestamp < 5:  # Print first 5 points
                print(f"Point {timestamp} change: {change:.6f}")
        
        print(f"Max pose change: {max(pose_changes):.6f}")
        print(f"Max point change: {max(point_changes):.6f}")
        print(f"Mean pose change: {np.mean(pose_changes):.6f}")
        print(f"Mean point change: {np.mean(point_changes):.6f}")

        # Convert updated pose_params to camera poses
        optimized_camera_poses = {}
        for timestamp, pose_param in pose_params.items():
            t = pose_param[:3]
            w = pose_param[3:]
            R = angle_axis_to_rotation_matrix(w)
            # matrix 4x4
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = t
            optimized_camera_poses[timestamp] = pose
        
        # Convert point_params back to numpy array
        optimized_points_3d = {timestamp: point_param for timestamp, point_param in point_params.items()}
        
        return summary, optimized_camera_poses, optimized_points_3d
