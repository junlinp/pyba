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
    tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).float()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return tensor.to(device)


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

def save_pointcloud_ply(points_3d: dict[int, np.ndarray], colors: dict[int, np.ndarray], output_path: str):
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
        for landmark_id, point_3d in points_3d.items():
            x, y, z = point_3d
            r, g, b = colors[landmark_id]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
    
    print(f"Saved point cloud to: {output_path}")

def poses_to_transforms(poses: list[np.ndarray]) -> list[np.ndarray]:
    """
    Convert KITTI poses (N, 12) to 4x4 transformation matrices (N, 4, 4).
    
    Args:
        poses: Ground truth poses as (N, 12) array (flattened 3x4 matrices)
        
    Returns:
        transforms: 4x4 transformation matrices as (N, 4, 4) array
    """
    N = len(poses)
    transforms = []
    
    for i in range(N):
        # Reshape pose to 3x4
        pose_3x4 = poses[i]
        
        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :4] = pose_3x4
        
        transforms.append(transform)
    
    return transforms

def compute_relative_pose_error(gt_poses: np.ndarray, estimated_poses: np.ndarray) -> float:
    """
    Compute relative pose error between ground truth and estimated poses.
    
    Args:
        gt_poses: Ground truth poses as (N, 3, 4) array
        estimated_poses: Estimated poses as (N, 3, 4) array
        
    Returns:
        Average translation error in meters
    """
    total_error = 0.0
    count = 0
    
    for i in range(1, len(gt_poses)):
        # Convert 3x4 poses to 4x4 transformation matrices
        gt_prev_3x4 = gt_poses[i-1].reshape(3, 4)
        gt_curr_3x4 = gt_poses[i].reshape(3, 4)
        
        # Create 4x4 transformation matrices
        gt_prev_4x4 = np.eye(4)
        gt_prev_4x4[:3, :4] = gt_prev_3x4
        
        gt_curr_4x4 = np.eye(4)
        gt_curr_4x4[:3, :4] = gt_curr_3x4
        
        # Compute relative pose
        gt_relative = np.linalg.inv(gt_prev_4x4) @ gt_curr_4x4
        
        # Get estimated relative pose
        est_prev_3x4 = estimated_poses[i-1].reshape(3, 4)
        est_curr_3x4 = estimated_poses[i].reshape(3, 4)
        
        # Create 4x4 transformation matrices for estimated poses
        est_prev_4x4 = np.eye(4)
        est_prev_4x4[:3, :4] = est_prev_3x4
        
        est_curr_4x4 = np.eye(4)
        est_curr_4x4[:3, :4] = est_curr_3x4
        
        est_relative = np.linalg.inv(est_prev_4x4) @ est_curr_4x4
        
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
        camera_poses: List of 4x4 camera poses for each view. from camera to world.
        K: Camera intrinsics matrix (3, 3)
        
    Returns:
        point_3d: 3D point (3,)
    """
    if len(keypoints_list) < 2:
        return np.zeros(3)
    
    # Build the DLT matrix
    A = []
    for i, (kp, pose) in enumerate(zip(keypoints_list, camera_poses)):
        pose_inv = np.linalg.inv(pose)
        # Get projection matrix P = K * [R|t]
        P = K @ pose_inv[:3, :4]
        
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

def project_point_to_image(point_3d: np.ndarray, camera_pose: np.ndarray, K: np.ndarray):
    rotation = camera_pose[:3, :3]
    translation = camera_pose[:3, 3]
    point_3d_camera = rotation.T @ (point_3d - translation)
    point_2d_homo = K @ point_3d_camera
    point_2d = point_2d_homo[:2] / point_2d_homo[2]
    return point_2d

def project_landmarks_to_images(images: dict[int, np.ndarray], landmark_tracker: LandmarkTracker, camera_poses: dict[int, np.ndarray], K: np.ndarray):
    for landmark_id, landmark in landmark_tracker.landmarks.items():
        for timestamp, kp_idx in landmark_tracker.landmark_observations[landmark_id].items():
            image = images[timestamp]
            keypoint_2d = landmark_tracker.landmark_keypoints[landmark_id][timestamp][kp_idx]
            camera_pose = camera_poses[timestamp]
            point_3d = landmark.position_3d 
            projected_keypoint = project_point_to_image(point_3d, camera_pose, K)
            cv2.circle(image, (int(projected_keypoint[0]), int(projected_keypoint[1])), 2, (0, 0, 255), -1)
            cv2.circle(image, (int(keypoint_2d[0]), int(keypoint_2d[1])), 2, (0, 255, 0), -1)
    return images

def main():
    # Load KITTI data
    reader = KITTIOdometryReader()
    seq = reader.sequences[5]
    info = reader.get_sequence_info(seq)
    print(f"Sequence {seq} info: {info}")
    calib = reader.load_calibration(seq)
    K = calib['P0'][:3, :3]  # Use P0 as intrinsics
    print(f"Camera intrinsics:\n{K}")

    # Load ground truth poses
    gt_poses = reader.load_poses(seq)
    print(f"Loaded {len(gt_poses)} ground truth poses")
    
    
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

    FRAME_STEP = 10
    # Select a few frames for demo (limit to first 10 frames for quick testing)
    # num_frames = min(FRAME_STEP * 2, len(gt_poses))
    num_frames = len(gt_poses)

    print(f"Processing {num_frames} frames for demo")
    
    # Create debug output directory
    debug_dir = Path(f"debug_matches_seq_{seq}")
    debug_dir.mkdir(exist_ok=True)

    # Process frames with landmark tracking
    timestamp_to_gt_poses = {}
    for i, pose in enumerate(gt_poses):
        timestamp_to_gt_poses[int(timestamps[i])] = pose

    
    prev_matches = None  # Store matches from previous frame
    prev_keypoints = None  # Store keypoints from previous frame
    prev_descriptors = None  # Store descriptors from previous frame
    
    # Store keypoints for each frame for later triangulation
    frame_keypoints = {}  # timestamp -> keypoints array
    
    camera_poses = {}
    images = {}

    for frame_id in range(FRAME_STEP, num_frames, FRAME_STEP):
        prev_frame_timestamp = timestamps[frame_id - FRAME_STEP]
        frame_timestamp = timestamps[frame_id]
        camera_poses[prev_frame_timestamp] = gt_poses[frame_id - FRAME_STEP]
        camera_poses[frame_timestamp] = gt_poses[frame_id]
        images[prev_frame_timestamp] = reader.load_image(seq, frame_id - FRAME_STEP, 'left')
        images[frame_timestamp] = reader.load_image(seq, frame_id, 'left')

        print(f"\nProcessing frame {frame_id}...")
        
        prev_image = reader.load_image(seq, frame_id - FRAME_STEP, 'left')
        curr_image = reader.load_image(seq, frame_id, 'left')

        # Extract features using SuperPoint
        features_prev = extract_superpoint_features(to_tensor(prev_image))
        features_curr = extract_superpoint_features(to_tensor(curr_image))
        keypoint_prev = features_prev['keypoints'].squeeze(0).cpu().numpy()
        keypoint_curr = features_curr['keypoints'].squeeze(0).cpu().numpy()
        
        # Store keypoints for each frame
        if timestamps is not None:
            prev_timestamp = int(timestamps[frame_id - FRAME_STEP])
            curr_timestamp = int(timestamps[frame_id])
        else:
            prev_timestamp = frame_id - FRAME_STEP
            curr_timestamp = frame_id
        
        frame_keypoints[prev_timestamp] = keypoint_prev
        frame_keypoints[curr_timestamp] = keypoint_curr
        
        match_keypoint_prev, match_keypoint_curr, matches = match_keypoints(features_prev, features_curr)
        print(f"  Frame {frame_id}: {len(matches)} matches, {len(landmark_tracker.landmarks)} landmarks before adding")
        landmark_tracker.add_matched_frame(prev_timestamp, curr_timestamp, keypoint_prev, keypoint_curr, matches.cpu().numpy())
        print(f"  Frame {frame_id}: {len(landmark_tracker.landmarks)} landmarks after adding")

        for match in matches:
            kp_idx0, kp_idx1 = match[0], match[1]
            kp0 = keypoint_prev[kp_idx0]
            kp1 = keypoint_curr[kp_idx1]
            cv2.circle(prev_image, (int(kp0[0]), int(kp0[1])), 2, (0, 0, 255), -1)
            cv2.circle(curr_image, (int(kp1[0]), int(kp1[1])), 2, (0, 0, 255), -1)
        cv2.imwrite(f"debug_matches_seq_{seq}/matches_{frame_id-FRAME_STEP}_{frame_id}_cycle.png", np.concatenate((prev_image, curr_image), axis=1))
            
        # Save debug image
        debug_path = debug_dir / f"matches_{frame_id-FRAME_STEP}_{frame_id}.png"
        save_match_debug_image(
            prev_image, curr_image, match_keypoint_prev, match_keypoint_curr, matches, 
            debug_path, f"{frame_id-FRAME_STEP}-{frame_id}"
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
        camera_poses_list = []
        
        for timestamp, kp_idx in observations.items():
            # Get the keypoint from the correct frame
            keypoints_list.append(landmark_tracker.landmark_keypoints[landmark_id][timestamp][kp_idx])
            camera_poses_list.append(timestamp_to_gt_poses[timestamp])
        # Triangulate the landmark
        landmark.position_3d = triangulate_points_multiview(keypoints_list, camera_poses_list, K)
    print(f"Landmarks with 3D positions: {sum(1 for l in landmark_tracker.landmarks.values() if l.position_3d is not None and not np.isnan(l.position_3d).any())}")

    project_landmarks_to_images(images, landmark_tracker, camera_poses, K)
    # save the images with the landmarks projected to them
    for timestamp, image in images.items():
        cv2.imwrite(f"debug_matches_seq_{seq}/landmarks_{timestamp}.png", image)

    # Prepare data for bundle adjustment
    points_3d = {}
    observations = []  # (landmark_id, timestamp, keypoint_2d)
    
    # Use real landmark data instead of synthetic data
    print(f"\nPreparing bundle adjustment data from landmarks...")
    
    # Collect landmarks with 3D positions and multiple observations
    valid_landmarks = []

    for landmark_id in landmark_tracker.landmarks:
        landmark = landmark_tracker.landmarks[landmark_id]
        observations_list = landmark_tracker.landmark_observations[landmark_id]
        if len(observations_list) >= 2:
            valid_landmarks.append((landmark_id, landmark, observations_list))

    print(f"Total landmarks: {len(landmark_tracker.landmarks)}")
    print(f"Found {len(valid_landmarks)} valid landmarks for bundle adjustment")
    
    if len(valid_landmarks) > 0:
        # Use real landmark data
        # Create observations for each landmark
        for landmark_id, landmark, observations_list in valid_landmarks:
            for timestamp, kp_idx in observations_list.items():
                # Get the 2D keypoint from the correct frame
                keypoint_2d = frame_keypoints[timestamp][kp_idx]
                # Convert timestamp to frame index for bundle adjustment
                observations.append((landmark_id, timestamp, keypoint_2d))
                points_3d[landmark_id] = landmark.position_3d
        
        print(f"Bundle adjustment data: {len(points_3d)} points, {len(observations)} observations, {len(camera_poses)} cameras")

    # Run bundle adjustment
    from bundle_adjustment import BundleAdjuster
    from rotation import angle_axis_to_rotation_matrix
    ba = BundleAdjuster(fix_first_pose=True, fix_intrinsics=True)
    
    # Add some noise to initial poses to test optimization (except first pose which is fixed)
    noisy_poses = camera_poses.copy()
    for timestamp in noisy_poses:  # Skip first pose (fixed)
        if timestamp == 0:
            continue
        # Add small random noise to translation
        noisy_poses[timestamp][:3, 3] += np.random.normal(0, 0.1, 3)
        # Add small random noise to rotation (simplified)
        angle_noise = np.random.normal(0, 0.01, 3)  # Small angle noise
        R_noise = angle_axis_to_rotation_matrix(angle_noise)
        noisy_poses[timestamp][:3, :3] = R_noise @ noisy_poses[timestamp][:3, :3]
    
    print(f"Added noise to initial poses for testing optimization")
    
    summary_obj, optimized_camera_poses, optimized_points_3d = ba.run(points_3d, observations, noisy_poses, K)

    print(f"Optimized camera poses: {len(optimized_camera_poses)}")
    print(f"Optimized points: {len(optimized_points_3d)}")
    print(f"Final reprojection error: {summary_obj.final_cost:.4f} pixels")
    
    # Debug: Check if points actually changed
    print(f"\nChecking if 3D points changed:")
    print(f"Initial points shape: {len(points_3d)}")
    print(f"Optimized points shape: {len(optimized_points_3d)}")

    
    # Debug: Check if poses actually changed
    # compare the optimized camera poses with the ground truth camera poses
    print(f"Comparing optimized camera poses with ground truth camera poses...")
    for timestamp, optimized_pose in optimized_camera_poses.items():
        gt_pose = timestamp_to_gt_poses[timestamp]
        print(f"Optimized pose {timestamp}: {optimized_pose}")
        print(f"Ground truth pose {timestamp}: {gt_pose}")
        print(f"Relative pose error: {np.linalg.norm(optimized_pose[:3, 3] - gt_pose[:3, 3])} meters")


    print(f"\nProcessing completed. Debug images saved to: {debug_dir}")
    print(f"Ground truth poses loaded successfully for sequence {seq}")
    print(f"Total poses: {len(gt_poses)}, Trajectory length: {total_distance:.2f}m")

    # draw ground truth trajectory x-z plane with green, and draw optimized trajectory x-z plane with red in the same figure
    plt.figure(figsize=(10, 10))
    optimized_trajectory = np.array([optimized_camera_poses[timestamp][:3, 3] for timestamp in optimized_camera_poses])
    plt.plot(trajectory[:, 0], trajectory[:, 2], color='green', label='Ground Truth Trajectory')
    plt.plot(optimized_trajectory[:, 0], optimized_trajectory[:, 2], color='red', label='Optimized Trajectory')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('Ground Truth Trajectory')
    plt.legend()
    plt.show()


    print(f"Optimized camera poses: {len(optimized_camera_poses)}")
    # output the optimized points 3d to a ply file
    save_pointcloud_ply(optimized_points_3d, {landmark_id: np.ones(3) for landmark_id in optimized_points_3d}, "optimized_points_3d.ply")


if __name__ == "__main__":
    main() 