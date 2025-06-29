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


@dataclass
class Landmark:
    """Represents a 3D landmark point with tracking information."""
    def __init__(self, keypoint: np.ndarray, position_3d: np.ndarray):
        self.position_3d = position_3d
        self.keypoint = keypoint

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

    def get_landmark_id_or_generate(self, image_timestamp: int, keypoint_index:int, keypoint: np.ndarray) -> int:
        """
        Get the landmark ID for a given image timestamp and keypoint index.
        If no landmark is found, generate a new one.
        """
        if image_timestamp not in self.frame_landmarks:
            self.frame_landmarks[image_timestamp] = {}

        if keypoint_index not in self.frame_landmarks[image_timestamp]: 
            self.frame_landmarks[image_timestamp][keypoint_index] = self.next_landmark_id
            self.landmarks[self.next_landmark_id] = Landmark(keypoint, np.zeros(3))
            self.landmark_observations[self.next_landmark_id] = {image_timestamp: keypoint_index}
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
            landmark_id0 = self.get_landmark_id_or_generate(timestamp0, kp0_idx, keypoints0[kp0_idx, :])
            landmark_id1 = self.get_landmark_id_or_generate(timestamp1, kp1_idx, keypoints1[kp1_idx, :])

            #  merge two landmarks into one since they are the same landmark
            self.merge_landmarks(landmark_id0, landmark_id1)

    def merge_landmarks(self, landmark_id0: int, landmark_id1: int):
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
        self.change_landmark_id(source_id, target_id)

    def change_landmark_id(self, landmark_id: int, new_landmark_id: int):
        """
        Change the landmark ID of a landmark.
        """
        if landmark_id == new_landmark_id:
            return

        observations = self.landmark_observations[landmark_id].copy()
        del self.landmark_observations[landmark_id]
        self.landmark_observations[new_landmark_id] = observations.copy()

        for frame_id, kp_idx in observations.items():
            self.frame_landmarks[frame_id][kp_idx] = new_landmark_id
        
        self.landmarks[new_landmark_id] = self.landmarks[landmark_id]
        del self.landmarks[landmark_id]
