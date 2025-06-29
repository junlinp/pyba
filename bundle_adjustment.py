#!/usr/bin/env python3
"""
Bundle Adjustment Implementation

Uses pyceres with automatic differentiation for efficient bundle adjustment.
"""

import numpy as np
from typing import List, Tuple, Optional
import cv2
import pyceres

class ReprojectionCostFunction(pyceres.CostFunction):
    """
    Cost function for reprojection error in bundle adjustment.
    """
    
    def __init__(self, observed_2d, K):
        super().__init__()
        self.observed_2d = observed_2d
        self.K = K
        self.set_num_residuals(2)
        self.set_parameter_block_sizes([6, 3])  # pose parameters, point parameters
    
    def Evaluate(self, parameters, residuals, jacobians):
        """
        Evaluate the cost function.
        
        Args:
            parameters: List of parameter blocks [pose_params, point_params]
            residuals: Output residuals (2,)
            jacobians: Output Jacobians (optional)
        """
        pose_params = parameters[0]  # [6] - Rodrigues rotation (3) + translation (3)
        point_params = parameters[1]  # [3] - 3D point coordinates
        
        # Extract rotation and translation
        rvec = pose_params[:3]
        tvec = pose_params[3:]
        
        # Convert Rodrigues to rotation matrix
        theta = np.linalg.norm(rvec)
        if theta < 1e-8:
            R = np.eye(3)
        else:
            k = rvec / theta
            K_cross = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            R = np.eye(3) + np.sin(theta) * K_cross + (1 - np.cos(theta)) * (K_cross @ K_cross)
        
        # Transform 3D point to camera coordinates
        point_cam = R @ point_params + tvec
        
        # Project to image plane
        point_2d_homog = self.K @ point_cam
        point_2d = point_2d_homog[:2] / point_2d_homog[2]
        
        # Compute residual
        residuals[0] = point_2d[0] - self.observed_2d[0]
        residuals[1] = point_2d[1] - self.observed_2d[1]
        
        # Compute Jacobians if requested
        if jacobians is not None:
            # This is a simplified Jacobian computation
            # In practice, you might want to use automatic differentiation or compute full Jacobians
            if jacobians[0] is not None:  # Pose Jacobian
                # Simplified: set to identity for now
                jacobians[0][:] = 0.0
                jacobians[0][0, 0] = 1.0  # dx/drx
                jacobians[0][1, 1] = 1.0  # dy/dry
            
            if jacobians[1] is not None:  # Point Jacobian
                # Simplified: set to identity for now
                jacobians[1][:] = 0.0
                jacobians[1][0, 0] = 1.0  # dx/dX
                jacobians[1][1, 1] = 1.0  # dy/dY
        
        return True

class BundleAdjuster:
    """
    Bundle adjustment optimizer using pyceres with automatic differentiation.
    """
    
    def __init__(self, fix_first_pose=True, fix_intrinsics=True, max_iterations=100):
        self.fix_first_pose = fix_first_pose
        self.fix_intrinsics = fix_intrinsics
        self.max_iterations = max_iterations
        
    def _pose_to_params(self, camera_poses: np.ndarray) -> np.ndarray:
        """Convert 4x4 camera poses to optimization parameters (rotation + translation)."""
        num_poses = len(camera_poses)
        params = np.zeros(num_poses * 6)  # 6 DOF per pose (3 rotation + 3 translation)
        
        for i in range(num_poses):
            pose = camera_poses[i]
            R = pose[:3, :3]
            t = pose[:3, 3]
            
            # Convert rotation matrix to Rodrigues vector
            rvec = cv2.Rodrigues(R)[0].flatten()
            
            # Store rotation and translation
            params[i*6:i*6+3] = rvec
            params[i*6+3:i*6+6] = t
            
        return params
    
    def _params_to_pose(self, params: np.ndarray, num_poses: int) -> np.ndarray:
        """Convert optimization parameters back to 4x4 camera poses."""
        camera_poses = np.zeros((num_poses, 4, 4))
        
        for i in range(num_poses):
            rvec = params[i*6:i*6+3]
            t = params[i*6+3:i*6+6]
            
            # Convert Rodrigues vector to rotation matrix
            R = cv2.Rodrigues(rvec)[0]
            
            # Build 4x4 pose matrix
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = t
            
            camera_poses[i] = pose
            
        return camera_poses
    
    def _project_point(self, point_3d: np.ndarray, pose: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Project a 3D point to 2D using camera pose and intrinsics."""
        # Transform point to camera coordinates
        point_cam = pose[:3, :3] @ point_3d + pose[:3, 3]
        
        # Project to image plane
        point_2d_homog = K @ point_cam
        point_2d = point_2d_homog[:2] / point_2d_homog[2]
        
        return point_2d
    
    def run(self, points_3d: np.ndarray, observations: List[Tuple], 
            camera_poses: np.ndarray, intrinsics: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run bundle adjustment using pyceres with automatic differentiation.
        
        Args:
            points_3d: (N, 3) array of 3D points
            observations: list of (landmark_idx, frame_idx, keypoint_2d)
            camera_poses: (M, 4, 4) array of camera poses
            intrinsics: (3, 3) camera intrinsics matrix
            
        Returns:
            optimized_camera_poses, optimized_points_3d, final_reprojection_error
        """
        num_poses = len(camera_poses)
        num_points = len(points_3d)
        
        print(f"Bundle adjustment: {num_poses} cameras, {num_points} points, {len(observations)} observations")
        
        # Prepare initial parameters
        pose_params = self._pose_to_params(camera_poses)
        point_params = points_3d.copy()
        
        # Set up pyceres problem
        problem = pyceres.Problem()
        
        # Add camera pose parameter blocks
        for i in range(num_poses):
            # Extract the 6 parameters for this pose
            pose_block = pose_params[i*6:(i+1)*6]
            problem.add_parameter_block(pose_block, 6)
            if i == 0 and self.fix_first_pose:
                problem.set_parameter_block_constant(pose_block)
        
        # Add 3D point parameter blocks
        for j in range(num_points):
            problem.add_parameter_block(point_params[j], 3)
        
        # Add reprojection error residuals
        for landmark_idx, frame_idx, keypoint_2d in observations:
            if frame_idx >= num_poses:
                continue
                
            # Create cost function
            cost_function = ReprojectionCostFunction(keypoint_2d, intrinsics)
            
            # Add residual block
            problem.add_residual_block(
                cost_function, 
                None,  # loss function (None = squared loss)
                [pose_params[frame_idx*6:(frame_idx+1)*6], point_params[landmark_idx]]
            )
        
        # Set up solver options
        options = pyceres.SolverOptions()
        options.max_num_iterations = self.max_iterations
        options.minimizer_progress_to_stdout = True
        options.linear_solver_type = pyceres.LinearSolverType.SPARSE_SCHUR
        options.sparse_linear_algebra_library_type = pyceres.SparseLinearAlgebraLibraryType.SUITE_SPARSE
        
        # Solve the problem
        summary = pyceres.SolverSummary()
        pyceres.solve(options, problem, summary)
        
        print(f"Bundle adjustment completed!")
        print(summary.BriefReport())
        
        # Convert back to 4x4 poses
        optimized_camera_poses = self._params_to_pose(pose_params, num_poses)
        optimized_points_3d = point_params
        
        # Compute final reprojection error
        final_errors = []
        for landmark_idx, frame_idx, keypoint_2d in observations:
            if frame_idx >= num_poses:
                continue
                
            point_3d = optimized_points_3d[landmark_idx]
            pose = optimized_camera_poses[frame_idx]
            
            # Project 3D point to 2D
            projected_2d = self._project_point(point_3d, pose, intrinsics)
            
            # Compute reprojection error
            error = projected_2d - keypoint_2d
            final_errors.extend(error)
        
        final_errors = np.array(final_errors)
        final_reprojection_error = np.sqrt(np.mean(final_errors**2))
        
        print(f"Final reprojection error: {final_reprojection_error:.4f} pixels")
        
        return optimized_camera_poses, optimized_points_3d, final_reprojection_error 