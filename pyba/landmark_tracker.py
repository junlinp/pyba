#!/usr/bin/env python3
"""
Landmark Tracking System

A class to track landmarks incrementally across a sequence of images
using feature matching and triangulation.

Author: Assistant
Date: 2024
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict

def multiview_triangulation(keypoints_list: List[np.ndarray], camera_poses: List[np.ndarray], K: np.ndarray) -> Tuple[bool, np.ndarray]:
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
        return False, np.zeros(3)
    
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

    # depth in each camera should be positive
    for pose in camera_poses:
        pose_inv = np.linalg.inv(pose)
        point_in_camera = pose_inv[:3, :3] @ point_3d + pose_inv[:3, 3]
        if point_in_camera[2] <= 0.1:
            return False, np.zeros(3)
    return True, point_3d


@dataclass
class Landmark:
    """Represents a 3D landmark point with tracking information."""
    def __init__(self, position_3d: np.ndarray, triangulated: bool):
        self.position_3d = position_3d
        self.triangulated = triangulated

class LandmarkTracker:
    """
    Tracks landmarks incrementally across image sequences using feature matching.
    """
    
    def __init__(self, min_track_length: int = 3, max_reprojection_error: float = 2.0):
        """
        Initialize the landmark tracker.
        
        Args:
            min_track_length: Minimum number of frames a landmark must be tracked
            max_reprojection_error: Maximum reprojection error for valid landmarks
        """
        # landmark_id -> Landmark
        self.landmarks: Dict[int, Landmark] = {}

        # next landmark id to assign
        self.next_landmark_id = 0
        self.min_track_length = min_track_length
        self.max_reprojection_error = max_reprojection_error

        # frame_id -> {kp_idx -> landmark_id}
        self.frame_landmarks: Dict[int, Dict[int, int]] = {}  # frame_id -> {kp_idx -> landmark_id}

        # landmark_id -> {frame_id -> kp_idx}
        self.landmark_observations: Dict[int, Dict[int, int]] = {}  # landmark_id -> {frame_id -> kp_idx}
        # landmark_id -> {timestamp -> kp_idx -> keypoint}
        self.landmark_keypoints: Dict[int, Dict[int, Dict[int, np.ndarray]]] = {}

    def _get_landmark_id_or_generate(self, image_timestamp: int, keypoint_index:int, keypoint: np.ndarray) -> int:
        """
        Get the landmark ID for a given image timestamp and keypoint index.
        If no landmark is found, generate a new one.
        """
        if image_timestamp not in self.frame_landmarks:
            self.frame_landmarks[image_timestamp] = {}

        if keypoint_index not in self.frame_landmarks[image_timestamp]: 
            self.frame_landmarks[image_timestamp][keypoint_index] = self.next_landmark_id
            self.landmarks[self.next_landmark_id] = Landmark(np.zeros(3), False)
            self.landmark_observations[self.next_landmark_id] = {image_timestamp: keypoint_index}
            self.landmark_keypoints[self.next_landmark_id] = {image_timestamp: {keypoint_index: keypoint}}
            self.next_landmark_id += 1

        return self.frame_landmarks[image_timestamp][keypoint_index]

        
    def add_matched_frame(self, timestamp0: int, timestamp1:int, keypoints0: np.ndarray, keypoints1: np.ndarray, matches: np.ndarray):
        """
        Add a matched frame to the landmark tracker.
        Args:
            timestamp0: Timestamp of first frame
            timestamp1: Timestamp of second frame
            keypoints0: Matched keypoints from first frame (N, 2)
            keypoints1: Matched keypoints from second frame (N, 2)
            matches: Match indices (N, 2) where matches[i] = [idx0, idx1]
        """
        for match in matches:
            kp0_idx, kp1_idx = match[0], match[1]
            # Check if keypoints are already associated with landmarks
            landmark_id0 = self._get_landmark_id_or_generate(timestamp0, kp0_idx, keypoints0[kp0_idx, :])
            landmark_id1 = self._get_landmark_id_or_generate(timestamp1, kp1_idx, keypoints1[kp1_idx, :])

            if landmark_id0 != landmark_id1:
                self._merge_landmarks(landmark_id0, landmark_id1)
            else:
                # else: already the same landmark, nothing to do
                self.landmark_keypoints[landmark_id0][timestamp0][kp0_idx] = keypoints0[kp0_idx, :]
                self.landmark_keypoints[landmark_id1][timestamp1][kp1_idx] = keypoints1[kp1_idx, :]


    def observation_relations_for_ba(self) -> List[Tuple[int, int, np.ndarray]]:
        '''
        Get the observation relations for bundle adjustment.
        The relations are used to construct the observation matrix for bundle adjustment.
        The observation matrix is a sparse matrix, where each row corresponds to a landmark,
        and each column corresponds to a camera pose.
        The value of the observation matrix is the keypoint observation.
        The observation matrix is used to solve the bundle adjustment problem.
        
        Returns:
            relations: List of tuples (timestamp, landmark_id, keypoint)
        '''
        relations = []
        for landmark_id in self.landmarks:
            landmark = self.landmarks[landmark_id]
            if landmark.triangulated:
                for timestamp, kp_idx in self.landmark_observations[landmark_id].items():
                    keypoint = self.landmark_keypoints[landmark_id][timestamp][kp_idx]
                    relations.append((timestamp, landmark_id, keypoint))
        return relations

    def get_landmark_point3ds(self) -> Dict[int, np.ndarray]:
        '''
        Get the 3D points of the landmarks.
        '''
        return {landmark_id: landmark.position_3d for landmark_id, landmark in self.landmarks.items() if landmark.triangulated}

    def _merge_landmarks(self, landmark_id0: int, landmark_id1: int):
        """
        Merge two landmarks into one.
        """
        if landmark_id0 == landmark_id1:
            return

        # Check if both landmarks still exist
        if landmark_id0 not in self.landmark_observations or landmark_id1 not in self.landmark_observations:
            # One or both landmarks have already been merged, skip this merge
            return

        # Choose the smaller ID as the target to keep
        target_id = min(landmark_id0, landmark_id1)
        source_id = max(landmark_id0, landmark_id1)

        # Only change the source landmark ID to the target ID
        self._change_landmark_id(source_id, target_id)

    def _change_landmark_id(self, landmark_id: int, new_landmark_id: int):
        """
        Change the landmark ID of a landmark, merging all observations and updating references.
        """
        if landmark_id == new_landmark_id:
            return

        # Merge observations
        obs_from = self.landmark_observations[landmark_id]
        obs_to = self.landmark_observations.get(new_landmark_id, {})
        merged_obs = obs_to.copy()
        merged_obs.update(obs_from)
        self.landmark_observations[new_landmark_id] = merged_obs
        del self.landmark_observations[landmark_id]

        # Update frame_landmarks to point to new_landmark_id
        for frame_id, kp_idx in obs_from.items():
            self.frame_landmarks[frame_id][kp_idx] = new_landmark_id
            # Ensure the sub-dictionaries exist
            if frame_id not in self.landmark_keypoints[new_landmark_id]:
                self.landmark_keypoints[new_landmark_id][frame_id] = {}
            if frame_id in self.landmark_keypoints[landmark_id] and kp_idx in self.landmark_keypoints[landmark_id][frame_id]:
                self.landmark_keypoints[new_landmark_id][frame_id][kp_idx] = self.landmark_keypoints[landmark_id][frame_id][kp_idx]
                del self.landmark_keypoints[landmark_id][frame_id][kp_idx]
                # Clean up empty sub-dicts
                if not self.landmark_keypoints[landmark_id][frame_id]:
                    del self.landmark_keypoints[landmark_id][frame_id]
        # Clean up old landmark_id if empty
        if landmark_id in self.landmark_keypoints and not self.landmark_keypoints[landmark_id]:
            del self.landmark_keypoints[landmark_id]

        # Optionally, merge 3D position/keypoint (keep the one with more observations or just keep new_landmark_id's)
        # Here, we keep the one with more observations
        if new_landmark_id in self.landmarks and landmark_id in self.landmarks:
            if len(merged_obs) >= 2:
                # Prefer the one with more observations
                self.landmarks[new_landmark_id] = self.landmarks[new_landmark_id]
            else:
                self.landmarks[new_landmark_id] = self.landmarks[landmark_id]
        elif landmark_id in self.landmarks:
            self.landmarks[new_landmark_id] = self.landmarks[landmark_id]
        # Remove the old landmark
        if landmark_id in self.landmarks:
            del self.landmarks[landmark_id]
