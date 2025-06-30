#!/usr/bin/env python3
"""
Unit tests for LandmarkTracker class.
"""

import numpy as np
import unittest
from landmark_tracker import LandmarkTracker, Landmark


class TestLandmarkTracker(unittest.TestCase):
    """Test cases for LandmarkTracker class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tracker = LandmarkTracker(min_track_length=3, max_reprojection_error=2.0)
    
    def test_initialization(self):
        """Test LandmarkTracker initialization."""
        self.assertEqual(len(self.tracker.landmarks), 0)
        self.assertEqual(self.tracker.next_landmark_id, 0)
        self.assertEqual(self.tracker.min_track_length, 3)
        self.assertEqual(self.tracker.max_reprojection_error, 2.0)
        self.assertEqual(len(self.tracker.frame_landmarks), 0)
        self.assertEqual(len(self.tracker.landmark_observations), 0)
    
    def test_get_landmark_id_or_generate_new(self):
        """Test generating new landmark ID for unassociated keypoint."""
        keypoint = np.array([100.0, 200.0])
        landmark_id = self.tracker.get_landmark_id_or_generate(0, 5, keypoint)
        
        self.assertEqual(landmark_id, 0)
        self.assertEqual(self.tracker.next_landmark_id, 1)
        self.assertIn(0, self.tracker.landmarks)
        self.assertIn(0, self.tracker.landmark_observations)
        self.assertIn(0, self.tracker.frame_landmarks)
        self.assertIn(5, self.tracker.frame_landmarks[0])
        
        # Check landmark data
        landmark = self.tracker.landmarks[0]
        np.testing.assert_array_equal(landmark.keypoint, keypoint)
        np.testing.assert_array_equal(landmark.position_3d, np.zeros(3))
        
        # Check observations
        self.assertEqual(self.tracker.landmark_observations[0], {0: 5})
    
    def test_get_landmark_id_or_generate_existing(self):
        """Test retrieving existing landmark ID."""
        keypoint = np.array([100.0, 200.0])
        
        # Generate landmark
        landmark_id1 = self.tracker.get_landmark_id_or_generate(0, 5, keypoint)
        
        # Retrieve same landmark
        landmark_id2 = self.tracker.get_landmark_id_or_generate(0, 5, keypoint)
        
        self.assertEqual(landmark_id1, landmark_id2)
        self.assertEqual(self.tracker.next_landmark_id, 1)  # Should not increment
    
    def test_add_matched_frame_new_landmarks(self):
        """Test adding matched frame with new landmarks."""
        keypoints0 = np.array([[100, 200], [150, 250], [200, 300]])
        keypoints1 = np.array([[110, 210], [160, 260], [210, 310]])
        matches = np.array([[0, 0], [1, 1], [2, 2]])
        
        self.tracker.add_matched_frame(0, 1, keypoints0, keypoints1, matches)
        
        # Should have 3 landmarks
        self.assertEqual(len(self.tracker.landmarks), 3)
        # Each landmark should have 2 observations (frames 0 and 1)
        for obs in self.tracker.landmark_observations.values():
            self.assertEqual(len(obs), 2)
            self.assertIn(0, obs)
            self.assertIn(1, obs)
    
    def test_add_matched_frame_extend_landmarks(self):
        """Test extending existing landmarks."""
        # First frame: create landmarks
        keypoints0 = np.array([[100, 200], [150, 250]])
        keypoints1 = np.array([[110, 210], [160, 260]])
        matches1 = np.array([[0, 0], [1, 1]])
        
        self.tracker.add_matched_frame(0, 1, keypoints0, keypoints1, matches1)
        
        # Second frame: extend landmarks
        keypoints2 = np.array([[120, 220], [170, 270]])
        matches2 = np.array([[0, 0], [1, 1]])
        
        self.tracker.add_matched_frame(1, 2, keypoints1, keypoints2, matches2)
        
        # Should have 2 landmarks, each with 3 observations (frames 0, 1, 2)
        self.assertEqual(len(self.tracker.landmarks), 2)
        for obs in self.tracker.landmark_observations.values():
            self.assertEqual(len(obs), 3)
            self.assertIn(0, obs)
            self.assertIn(1, obs)
            self.assertIn(2, obs)
    
    def test_merge_landmarks(self):
        """Test merging two landmarks."""
        # Create two separate landmarks
        keypoints0 = np.array([[100, 200]])
        keypoints1 = np.array([[110, 210]])
        matches1 = np.array([[0, 0]])
        
        self.tracker.add_matched_frame(0, 1, keypoints0, keypoints1, matches1)
        
        # Create another landmark
        keypoints2 = np.array([[150, 250]])
        keypoints3 = np.array([[160, 260]])
        matches2 = np.array([[0, 0]])
        
        self.tracker.add_matched_frame(2, 3, keypoints2, keypoints3, matches2)
        
        # Now merge them by matching the same keypoints
        keypoints4 = np.array([[110, 210]])  # Same as keypoints1
        keypoints5 = np.array([[160, 260]])  # Same as keypoints3
        matches3 = np.array([[0, 0]])
        
        self.tracker.add_matched_frame(1, 3, keypoints4, keypoints5, matches3)
        
        # Should have only 1 landmark after merging
        self.assertEqual(len(self.tracker.landmarks), 1)
        self.assertIn(0, self.tracker.landmarks)  # Should keep the smaller ID
        self.assertNotIn(1, self.tracker.landmarks)  # Should be merged
    
    def test_change_landmark_id(self):
        """Test changing landmark ID."""
        # Create a landmark
        keypoint = np.array([100.0, 200.0])
        landmark_id = self.tracker.get_landmark_id_or_generate(0, 5, keypoint)
        
        # Add another observation
        self.tracker.landmark_observations[landmark_id][1] = 10
        self.tracker.frame_landmarks.setdefault(1, {})[10] = landmark_id
        
        # Change the ID
        self.tracker.change_landmark_id(landmark_id, 5)
        
        # Check that the landmark moved to new ID
        self.assertNotIn(landmark_id, self.tracker.landmarks)
        self.assertNotIn(landmark_id, self.tracker.landmark_observations)
        self.assertIn(5, self.tracker.landmarks)
        self.assertIn(5, self.tracker.landmark_observations)
        
        # Check observations are preserved
        self.assertEqual(self.tracker.landmark_observations[5], {0: 5, 1: 10})
        
        # Check frame landmarks are updated
        self.assertEqual(self.tracker.frame_landmarks[0][5], 5)
        self.assertEqual(self.tracker.frame_landmarks[1][10], 5)
    
    def test_merge_landmarks_same_id(self):
        """Test merging landmarks with same ID (should do nothing)."""
        keypoint = np.array([100.0, 200.0])
        landmark_id = self.tracker.get_landmark_id_or_generate(0, 5, keypoint)
        
        initial_count = len(self.tracker.landmarks)
        self.tracker.merge_landmarks(landmark_id, landmark_id)
        
        # Should not change anything
        self.assertEqual(len(self.tracker.landmarks), initial_count)
        self.assertIn(landmark_id, self.tracker.landmarks)
    
    def test_merge_landmarks_nonexistent(self):
        """Test merging with nonexistent landmarks."""
        # Create one landmark
        keypoint = np.array([100.0, 200.0])
        landmark_id = self.tracker.get_landmark_id_or_generate(0, 5, keypoint)
        
        # Try to merge with nonexistent landmark
        self.tracker.merge_landmarks(landmark_id, 999)
        
        # Should not change anything
        self.assertEqual(len(self.tracker.landmarks), 1)
        self.assertIn(landmark_id, self.tracker.landmarks)
    
    def test_landmark_observations_consistency(self):
        """Test that landmark observations are consistent across data structures."""
        keypoints0 = np.array([[100, 200], [150, 250]])
        keypoints1 = np.array([[110, 210], [160, 260]])
        matches = np.array([[0, 0], [1, 1]])
        
        self.tracker.add_matched_frame(0, 1, keypoints0, keypoints1, matches)
        
        # Check consistency
        for landmark_id in self.tracker.landmarks:
            observations = self.tracker.landmark_observations[landmark_id]
            
            for timestamp, kp_idx in observations.items():
                # Check frame_landmarks consistency
                self.assertIn(timestamp, self.tracker.frame_landmarks)
                self.assertIn(kp_idx, self.tracker.frame_landmarks[timestamp])
                self.assertEqual(self.tracker.frame_landmarks[timestamp][kp_idx], landmark_id)
    
    def test_multiple_frames_tracking(self):
        """Test tracking across multiple frames."""
        # Frame 0-1
        keypoints0 = np.array([[100, 200], [150, 250]])
        keypoints1 = np.array([[110, 210], [160, 260]])
        matches1 = np.array([[0, 0], [1, 1]])
        self.tracker.add_matched_frame(0, 1, keypoints0, keypoints1, matches1)
        
        # Frame 1-2 (extend existing tracks)
        keypoints2 = np.array([[120, 220], [170, 270]])
        matches2 = np.array([[0, 0], [1, 1]])
        self.tracker.add_matched_frame(1, 2, keypoints1, keypoints2, matches2)
        
        # Frame 2-3 (extend existing tracks)
        keypoints3 = np.array([[130, 230], [180, 280]])
        matches3 = np.array([[0, 0], [1, 1]])
        self.tracker.add_matched_frame(2, 3, keypoints2, keypoints3, matches3)
        
        # Should have 2 landmarks, each with 4 observations (frames 0, 1, 2, 3)
        self.assertEqual(len(self.tracker.landmarks), 2)
        for obs in self.tracker.landmark_observations.values():
            self.assertEqual(len(obs), 4)
            for frame in range(4):
                self.assertIn(frame, obs)


class TestLandmark(unittest.TestCase):
    """Test cases for Landmark class."""
    
    def test_landmark_initialization(self):
        """Test Landmark initialization."""
        keypoint = np.array([100.0, 200.0])
        position_3d = np.array([1.0, 2.0, 3.0])
        
        landmark = Landmark(keypoint, position_3d)
        
        np.testing.assert_array_equal(landmark.keypoint, keypoint)
        np.testing.assert_array_equal(landmark.position_3d, position_3d)
    
    def test_landmark_default_position(self):
        """Test Landmark with default zero position."""
        keypoint = np.array([100.0, 200.0])
        landmark = Landmark(keypoint, np.zeros(3))
        
        np.testing.assert_array_equal(landmark.keypoint, keypoint)
        np.testing.assert_array_equal(landmark.position_3d, np.zeros(3))


if __name__ == '__main__':
    unittest.main() 