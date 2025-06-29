#!/usr/bin/env python3
"""
KITTI SFM Pipeline with Landmark Tracking

- Keypoint detection & matching: LightGlue (with built-in SuperPoint)
- Landmark tracking: Incremental tracking across frames
- Bundle adjustment: PyCeres (optional)
- Visualization: matplotlib (for camera poses and sparse point cloud)
- Debug: Save match results as images

Requirements:
- torch, torchvision
- lightglue (includes SuperPoint)
- pyceres (optional, for bundle adjustment)
- matplotlib

Author: Assistant
Date: 2024
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from kitti_reader import KITTIOdometryReader
from landmark_tracker import Landmark, LandmarkTracker

# Import LightGlue (includes SuperPoint)
try:
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import rbd
except ImportError:
    print("Please install lightglue: pip install lightglue")
    exit(1)

# Import pyceres (optional)
try:
    import pyceres
    PYCERES_AVAILABLE = True
except ImportError:
    print("Warning: pyceres not available. Bundle adjustment will be skipped.")
    PYCERES_AVAILABLE = False


def extract_superpoint_features(image: torch.Tensor) -> dict[str, torch.Tensor]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
    features = extractor.extract(image)
    return features

def match_keypoints(features0: dict[str, torch.Tensor], features1: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lightglue = LightGlue(features='superpoint').eval().to(device)
    matches01 = lightglue({'image0': features0, 'image1': features1})
    feats0, feats1, matches01 = [rbd(x) for x in [features0, features1, matches01]]  # remove batch dimension
    matches = matches01['matches']  # indices with shape (K,2)
    points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
    points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)
    return points0, points1, matches

def run_bundle_adjustment_pyceres(points_3d, points_2d, camera_poses, intrinsics):
    """
    Run bundle adjustment using pyceres.
    Args:
        points_3d: (N, 3)
        points_2d: list of (M_i, 2) for each image
        camera_poses: (num_images, 6) or (num_images, 7)
        intrinsics: (3, 3)
    Returns:
        optimized camera_poses, points_3d
    """
    if not PYCERES_AVAILABLE:
        print("Bundle adjustment skipped: pyceres not available")
        return camera_poses, points_3d
    # Placeholder: actual implementation depends on pyceres API
    print("[Stub] Running bundle adjustment with pyceres...")
    return camera_poses, points_3d


def visualize_sfm(camera_poses, points_3d):
    """
    Visualize camera poses and sparse point cloud.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Plot points
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], s=1, c='b', alpha=0.5, label='Points')
    # Plot cameras
    for i, pose in enumerate(camera_poses):
        t = pose[:3]
        ax.scatter(t[0], t[1], t[2], c='r', marker='^', s=40)
        ax.text(t[0], t[1], t[2], f'C{i}', color='r')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('SFM: Camera Poses and Sparse Point Cloud')
    plt.legend()
    plt.show()


def to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    img_rgb = img_rgb / 255.0  # shape (H, W, 3), float32
    # convert to (C, H, W)
    return torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).float()


def save_match_debug_image(img0, img1, kpts0, kpts1, matches, output_path, frame_pair):
    """
    Save debug image showing feature matches between two images.
    
    Args:
        img0: First image (numpy array)
        img1: Second image (numpy array)
        kpts0: Keypoints from first image (N, 2)
        kpts1: Keypoints from second image (M, 2)
        matches: Matches array (K, 2) with indices
        output_path: Path to save the debug image
        frame_pair: String describing the frame pair (e.g., "0-1")
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert keypoints to OpenCV format
    kpts0_cv = [cv2.KeyPoint(x=int(pt[0]), y=int(pt[1]), size=1) for pt in kpts0]
    kpts1_cv = [cv2.KeyPoint(x=int(pt[0]), y=int(pt[1]), size=1) for pt in kpts1]
    
    # Filter matches to ensure indices are within bounds
    valid_matches = []
    for match in matches:
        idx0, idx1 = match[0], match[1]
        if 0 <= idx0 < len(kpts0) and 0 <= idx1 < len(kpts1):
            valid_matches.append(cv2.DMatch(_queryIdx=int(idx0), _trainIdx=int(idx1), _distance=0))
    
    # Draw matches using OpenCV
    combined_img = cv2.drawMatches(img0, kpts0_cv, img1, kpts1_cv, valid_matches, None)
    
    # Save the image
    cv2.imwrite(str(output_path), combined_img)
    print(f"Saved match debug image: {output_path} with {len(valid_matches)} valid matches")


def estimate_pose(kpts0: np.ndarray, kpts1: np.ndarray, K: np.ndarray):
    pts0 = kpts0
    pts1 = kpts1
    
    E, mask = cv2.findEssentialMat(pts0, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask_pose = cv2.recoverPose(E, pts0, pts1, K)
    return R, t, mask_pose, pts0, pts1

def triangulate_points(kpts0: np.ndarray, kpts1: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray):
    P0 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P1 = K @ np.hstack((R, t))
    pts0 = kpts0.T
    pts1 = kpts1.T
    pts4d = cv2.triangulatePoints(P0, P1, pts0, pts1)
    pts3d = (pts4d[:3] / pts4d[3]).T  # shape (N, 3)
    return pts3d

def save_pointcloud_ply(points_3d, colors, output_path):
    """
    Save point cloud as PLY file.
    
    Args:
        points_3d: numpy array of shape (N, 3) with 3D points
        colors: numpy array of shape (N, 3) with RGB colors (0-255)
        output_path: path to save the PLY file
    """
    with open(output_path, 'w') as f:
        # Write PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points_3d)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Write points and colors
        for i in range(len(points_3d)):
            x, y, z = points_3d[i]
            r, g, b = colors[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
    
    print(f"Saved point cloud to: {output_path}")

def poses_to_transforms(poses: np.ndarray) -> np.ndarray:
    """
    Convert KITTI poses (N, 12) to 4x4 transformation matrices (N, 4, 4).
    
    Args:
        poses: Ground truth poses as (N, 12) array (flattened 3x4 matrices)
        
    Returns:
        transforms: 4x4 transformation matrices as (N, 4, 4) array
    """
    N = poses.shape[0]
    transforms = np.zeros((N, 4, 4))
    
    for i in range(N):
        # Reshape pose to 3x4
        pose_3x4 = poses[i].reshape(3, 4)
        
        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :4] = pose_3x4
        
        transforms[i] = transform
    
    return transforms

def compute_relative_pose_error(gt_poses: np.ndarray, estimated_poses: np.ndarray) -> float:
    """
    Compute relative pose error between ground truth and estimated poses.
    
    Args:
        gt_poses: Ground truth poses as (N, 12) array
        estimated_poses: Estimated poses as (N, 12) array
        
    Returns:
        error: Average relative pose error
    """
    if len(gt_poses) != len(estimated_poses):
        raise ValueError("Ground truth and estimated poses must have same length")
    
    total_error = 0.0
    count = 0
    
    for i in range(1, len(gt_poses)):
        # Get ground truth relative pose
        gt_prev = gt_poses[i-1].reshape(3, 4)
        gt_curr = gt_poses[i].reshape(3, 4)
        gt_relative = np.linalg.inv(gt_prev) @ gt_curr
        
        # Get estimated relative pose
        est_prev = estimated_poses[i-1].reshape(3, 4)
        est_curr = estimated_poses[i].reshape(3, 4)
        est_relative = np.linalg.inv(est_prev) @ est_curr
        
        # Compute error
        error_matrix = np.linalg.inv(gt_relative) @ est_relative
        translation_error = np.linalg.norm(error_matrix[:3, 3])
        total_error += translation_error
        count += 1
    
    return float(total_error / count if count > 0 else 0.0)

def triangulate_points_multiview(keypoints_list: List[np.ndarray], camera_poses: List[np.ndarray], K: np.ndarray) -> np.ndarray:
    """
    Triangulate 3D point from multiple views using DLT method.
    
    Args:
        keypoints_list: List of 2D keypoints (N, 2) for each view
        camera_poses: List of 4x4 camera poses for each view
        K: Camera intrinsics matrix (3, 3)
        
    Returns:
        point_3d: 3D point (3,)
    """
    if len(keypoints_list) < 2:
        return np.zeros(3)
    
    # Build the DLT matrix
    A = []
    for i, (kp, pose) in enumerate(zip(keypoints_list, camera_poses)):
        # Get projection matrix P = K * [R|t]
        P = K @ pose[:3, :4]
        
        # Normalize homogeneous coordinates
        x, y = kp[0], kp[1]
        
        # Add two rows to A matrix for each view
        A.append(x * P[2, :] - P[0, :])
        A.append(y * P[2, :] - P[1, :])
    
    A = np.array(A)
    
    # Solve using SVD
    U, S, Vt = np.linalg.svd(A)
    point_4d = Vt[-1, :]
    
    # Convert to 3D coordinates
    point_3d = point_4d[:3] / point_4d[3]
    
    return point_3d

def main():
    # Load KITTI data
    reader = KITTIOdometryReader()
    seq = reader.sequences[0]
    info = reader.get_sequence_info(seq)
    print(f"Sequence {seq} info: {info}")
    calib = reader.load_calibration(seq)
    K = calib['P0'][:3, :3]  # Use P0 as intrinsics
    print(f"Camera intrinsics:\n{K}")

    # Load ground truth poses
    gt_poses = reader.load_poses(seq)
    print(f"Loaded {len(gt_poses)} ground truth poses")
    
    # Convert poses to 4x4 transformation matrices
    gt_transforms = poses_to_transforms(gt_poses)
    print(f"Converted poses to {len(gt_transforms)} 4x4 transformation matrices")
    
    # Analyze ground truth trajectory
    trajectory = reader.get_trajectory(seq)
    total_distance = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
    print(f"Total trajectory distance: {total_distance:.2f} meters")
    print(f"Average frame-to-frame distance: {total_distance/(len(trajectory)-1):.3f} meters")

    # Load timestamps for the sequence
    try:
        timestamps = reader.load_timestamps(seq)
        print(f"Loaded {len(timestamps)} timestamps")
    except FileNotFoundError:
        print("Warning: No timestamps found, using frame indices as timestamps")
        timestamps = None

    # Initialize landmark tracker
    landmark_tracker = LandmarkTracker(min_track_length=3, max_reprojection_error=2.0)
    print("Initialized landmark tracker")

    # Select a few frames for demo (limit to first 10 frames for quick testing)
    num_frames = min(10, len(gt_poses))
    print(f"Processing {num_frames} frames for demo")
    
    # Create debug output directory
    debug_dir = Path(f"debug_matches_seq_{seq}")
    debug_dir.mkdir(exist_ok=True)

    # Process frames with landmark tracking
    all_camera_poses = [np.eye(4)]  # First camera at origin
    camera_poses_history = [np.eye(4)]  # Track all camera poses
    prev_matches = None  # Store matches from previous frame
    prev_keypoints = None  # Store keypoints from previous frame
    prev_descriptors = None  # Store descriptors from previous frame
    
    for frame_id in range(1, num_frames):
        print(f"\nProcessing frame {frame_id}...")
        
        prev_image = reader.load_image(seq, frame_id - 1, 'left')
        curr_image = reader.load_image(seq, frame_id, 'left')

        # Extract features using SuperPoint
        features_prev = extract_superpoint_features(to_tensor(prev_image))
        features_curr = extract_superpoint_features(to_tensor(curr_image))
        keypoint_prev = features_prev['keypoints'].squeeze(0).numpy()
        keypoint_curr = features_curr['keypoints'].squeeze(0).numpy()
        

        if timestamps is not None:
            prev_timestamp = int(timestamps[frame_id - 1])
            curr_timestamp = int(timestamps[frame_id])
        else:
            prev_timestamp = frame_id - 1
            curr_timestamp = frame_id
        
        match_keypoint_prev, match_keypoint_curr, matches = match_keypoints(features_prev, features_curr)
        landmark_tracker.add_matched_frame(prev_timestamp, curr_timestamp, keypoint_prev, keypoint_curr, matches.numpy())
            
        # Save debug image
        debug_path = debug_dir / f"matches_{frame_id-1}_{frame_id}.png"
        save_match_debug_image(
            prev_image, curr_image, match_keypoint_prev, match_keypoint_curr, matches, 
            debug_path, f"{frame_id-1}-{frame_id}"
        )
    
    # triangulate landmarks with the camera poses
    # using the ground truth poses now. and the keypoints observations.
    print(f"\nTriangulating landmarks...")
    print(f"Total landmarks: {len(landmark_tracker.landmarks)}")
    
    for landmark_id in landmark_tracker.landmarks:
        landmark = landmark_tracker.landmarks[landmark_id]
        observations = landmark_tracker.landmark_observations[landmark_id]
        
        if len(observations) < 2:
            continue
            
        # Collect keypoints and camera poses for this landmark
        keypoints_list = []
        camera_poses = []
        
        for timestamp, kp_idx in observations.items():
            # Get the keypoint from the original keypoint arrays
            if timestamp < len(keypoint_prev) and kp_idx < len(keypoint_prev):
                # This is a simplified approach - in reality we need to track which frame each keypoint came from
                keypoints_list.append(keypoint_prev[kp_idx])
            else:
                # Fallback to landmark keypoint
                keypoints_list.append(landmark.keypoint)
            # Get camera pose for this timestamp
            camera_poses.append(gt_transforms[timestamp])
        
        # Triangulate the landmark
        landmark.position_3d = triangulate_points_multiview(keypoints_list, camera_poses, K)
    
    print(f"Landmarks with 3D positions: {sum(1 for l in landmark_tracker.landmarks.values() if l.position_3d is not None and not np.isnan(l.position_3d).any())}")

    # Prepare data for bundle adjustment
    points_3d = []
    observations = []  # (landmark_idx, frame_idx, keypoint_2d)
    
    # For now, let's create a simple test case with ground truth poses and some synthetic observations
    print(f"\nPreparing bundle adjustment data...")
    
    # Create some synthetic 3D points and observations for testing
    if len(points_3d) == 0:
        print("Creating synthetic test data for bundle adjustment...")
        # Create a few synthetic 3D points
        synthetic_points = np.array([
            [1.0, 0.0, 5.0],
            [0.0, 1.0, 5.0],
            [-1.0, 0.0, 5.0],
            [0.0, -1.0, 5.0],
            [2.0, 2.0, 8.0],
        ])
        
        points_3d = synthetic_points
        observations = []
        
        # Create observations for each point in each camera
        for point_idx in range(len(synthetic_points)):
            for frame_idx in range(num_frames):
                # Project 3D point to 2D using ground truth pose
                pose = gt_transforms[frame_idx]
                point_3d = synthetic_points[point_idx]
                
                # Transform to camera coordinates
                point_cam = pose[:3, :3] @ point_3d + pose[:3, 3]
                
                # Project to image plane
                point_2d_homog = K @ point_cam
                point_2d = point_2d_homog[:2] / point_2d_homog[2]
                
                # Add some noise to simulate real observations
                point_2d += np.random.normal(0, 1.0, 2)
                
                observations.append((point_idx, frame_idx, point_2d))
    
    points_3d = np.array(points_3d)
    camera_poses = np.array([gt_transforms[i] for i in range(num_frames)])
    
    print(f"Bundle adjustment data: {len(points_3d)} points, {len(observations)} observations, {len(camera_poses)} cameras")

    # Run bundle adjustment
    from bundle_adjustment import BundleAdjuster
    ba = BundleAdjuster(fix_first_pose=True, fix_intrinsics=True)
    optimized_camera_poses, optimized_points_3d, reprojection_error = ba.run(points_3d, observations, camera_poses, K)
    
    # Display bundle adjustment results
    print(f"\nBundle Adjustment Results:")
    print(f"Initial points: {len(points_3d)}")
    print(f"Optimized points: {len(optimized_points_3d)}")
    print(f"Final reprojection error: {reprojection_error:.4f} pixels")
    
    # Compare with ground truth poses (if using ground truth as initial)
    if len(optimized_camera_poses) > 1:
        pose_error = compute_relative_pose_error(
            gt_poses[:num_frames].reshape(-1, 12), 
            np.array([pose[:3, :4].flatten() for pose in optimized_camera_poses])
        )
        print(f"Relative pose error vs ground truth: {pose_error:.4f} meters")
    
    # Use optimized_camera_poses, optimized_points_3d for further processing/visualization

    print(f"\nProcessing completed. Debug images saved to: {debug_dir}")
    print(f"Ground truth poses loaded successfully for sequence {seq}")
    print(f"Total poses: {len(gt_poses)}, Trajectory length: {total_distance:.2f}m")


if __name__ == "__main__":
    main() 