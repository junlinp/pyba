#!/usr/bin/env python3
"""
pyba - Python Bundle Adjustment and Landmark Tracking

A comprehensive Python library for Structure from Motion (SFM), 
landmark tracking, and bundle adjustment using the KITTI dataset.
"""

__version__ = "0.1.0"
__author__ = "Assistant"
__email__ = "assistant@example.com"

# Import main classes and functions
from .landmark_tracker import LandmarkTracker, Landmark
from .bundle_adjustment import BundleAdjuster, ReprojErrorCost, CameraModel
from .rotation import (
    rotation_matrix_to_angle_axis, 
    angle_axis_to_rotation_matrix, 
    so3_right_jacobian, 
    skew_symmetric
)
from .kitti_reader import KITTIOdometryReader

# Import utility functions
from .kitti_sfm import (
    extract_superpoint_features,
    match_keypoints,
    multiview_triangulation,
    project_point_to_image,
    poses_to_transforms,
    compute_relative_pose_error
)

__all__ = [
    # Main classes
    "LandmarkTracker",
    "Landmark", 
    "BundleAdjuster",
    "ReprojErrorCost",
    "CameraModel",
    "KITTIOdometryReader",
    
    # Rotation utilities
    "rotation_matrix_to_angle_axis",
    "angle_axis_to_rotation_matrix", 
    "so3_right_jacobian",
    "skew_symmetric",
    
    # SFM utilities
    "extract_superpoint_features",
    "match_keypoints",
    "triangulate_points_multiview",
    "project_point_to_image",
    "poses_to_transforms",
    "compute_relative_pose_error",
] 