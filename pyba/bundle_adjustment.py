#!/usr/bin/env python3
"""
Bundle Adjustment using pyceres

This module provides bundle adjustment functionality using pyceres for optimization.
"""

import numpy as np
import pyceres
from typing import List, Tuple, Dict
from pyba.rotation import rotation_matrix_to_angle_axis, angle_axis_to_rotation_matrix, so3_right_jacobian, skew_symmetric

from pyba.pyceres_bind import ba_solve

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
    
    def solve_bundle_adjustment(self, points_3d: Dict[int, np.ndarray], 
                                observations: List[Tuple[int, int, np.ndarray]], 
                                camera_poses: Dict[int, np.ndarray], 
                                intrinsics: np.ndarray,
                                constant_pose_index: Dict[int, bool] = None,
                                relative_pose_constraints: List[Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]] = None):
        """
        Solve the bundle adjustment problem.
        """
        #problem, pose_params_dict, point_params_dict = self.define_problem(points_3d, observations, camera_poses, intrinsics)
        #summary = self.solve(problem)
        print(f"use ba_solve")

        if constant_pose_index is not None:
            py_constant_pose_index = {timestamp: is_constant for timestamp, is_constant in constant_pose_index.items()}
        else:
            py_constant_pose_index = {}

        if relative_pose_constraints is not None:
            py_relative_pose_constraints = []
            for cam_idx_i, cam_idx_j, relative_pose_j_i, translation_weight, rotation_weight in relative_pose_constraints:
                py_relative_pose_constraints.append((cam_idx_i, cam_idx_j, relative_pose_j_i, translation_weight, rotation_weight))
        else:
            py_relative_pose_constraints = []

        optimized_camera_poses, optimized_points_3d = ba_solve(camera_poses, points_3d, observations, intrinsics, py_constant_pose_index, py_relative_pose_constraints)
        return optimized_camera_poses, optimized_points_3d
    
    
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
    
    def run(self, points_3d: Dict[int, np.ndarray], 
            observations: List[Tuple[int, int, np.ndarray]], 
            camera_poses: Dict[int, np.ndarray], 
            intrinsics: np.ndarray,
            constant_pose_index: Dict[int, bool] = None,
            relative_pose_constraints: List[Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]] = None):
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

        constant_pose_index_dict = {}
        if constant_pose_index is not None:
            for timestamp, index in constant_pose_index.items():
                constant_pose_index_dict[timestamp] = index

        if self.fix_first_pose:
            min_timestamp = min(camera_poses.keys())
            constant_pose_index_dict[min_timestamp] = True

        
        optimized_camera_poses, optimized_points_3d = self.solve_bundle_adjustment(points_3d, observations, camera_poses, intrinsics, constant_pose_index_dict, relative_pose_constraints)
        return optimized_camera_poses, optimized_points_3d

       
