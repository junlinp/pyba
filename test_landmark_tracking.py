#!/usr/bin/env python3
"""
Test Landmark Tracking with Realistic Features

This script tests the LandmarkTracker with more realistic feature matching
to demonstrate the tracking capabilities.
"""

import numpy as np
import cv2
from landmark_tracker import LandmarkTracker


def create_synthetic_features(num_features=50):
    """Create synthetic features that can be tracked across frames."""
    # Create base keypoints
    keypoints = np.random.rand(num_features, 2) * 100
    
    # Create descriptors with some similarity for tracking
    descriptors = np.random.rand(num_features, 128).astype(np.float32)
    
    return keypoints, descriptors


def simulate_camera_motion(keypoints, translation_factor=5.0):
    """Simulate camera motion by translating keypoints."""
    # Add some translation and noise
    translation = np.random.randn(2) * translation_factor
    noise = np.random.randn(*keypoints.shape) * 0.5
    
    new_keypoints = keypoints + translation + noise
    
    # Keep keypoints within bounds
    new_keypoints = np.clip(new_keypoints, 0, 100)
    
    return new_keypoints


def test_realistic_tracking():
    """Test landmark tracking with realistic feature simulation."""
    print("Testing LandmarkTracker with realistic features...")
    
    tracker = LandmarkTracker(min_track_length=2, max_reprojection_error=2.0)
    
    # Create initial features
    keypoints, descriptors = create_synthetic_features(50)
    
    # Process multiple frames
    num_frames = 5
    camera_matrix = np.array([[1000, 0, 500], [0, 1000, 500], [0, 0, 1]])
    
    for frame_id in range(num_frames):
        print(f"\nProcessing frame {frame_id}")
        
        if frame_id > 0:
            # Simulate camera motion
            keypoints = simulate_camera_motion(keypoints)
            # Add some new features and remove some old ones
            if frame_id % 2 == 0:
                # Add new features
                new_kps, new_descs = create_synthetic_features(10)
                keypoints = np.vstack([keypoints, new_kps])
                descriptors = np.vstack([descriptors, new_descs])
        
        # Create camera pose (simple translation)
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = np.array([frame_id * 2.0, 0, 0])
        
        # Add frame to tracker
        landmarks = tracker.add_frame(frame_id, keypoints, descriptors, camera_pose, camera_matrix)
        
        print(f"  Frame {frame_id}: {len(landmarks)} landmarks processed")
        print(f"  Keypoints in frame: {len(keypoints)}")
    
    # Get statistics
    stats = tracker.get_statistics()
    print(f"\nTracking Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Get stable landmarks
    stable_landmarks = tracker.get_stable_landmarks()
    print(f"\nStable landmarks: {len(stable_landmarks)}")
    
    # Show some example tracks
    if stable_landmarks:
        print(f"\nExample landmark tracks:")
        for i, landmark in enumerate(stable_landmarks[:3]):
            print(f"  Landmark {landmark.id}:")
            print(f"    Track length: {landmark.track_length}")
            print(f"    First frame: {landmark.first_frame}")
            print(f"    Last frame: {landmark.last_frame}")
            print(f"    Position: {landmark.position_3d}")
            print(f"    Observations: {len(landmark.observations)}")
    
    return tracker


def test_feature_matching():
    """Test the feature matching functionality specifically."""
    print("\nTesting feature matching...")
    
    # Create two sets of similar features
    kp1, desc1 = create_synthetic_features(30)
    kp2, desc2 = create_synthetic_features(30)
    
    # Make some features similar for matching
    num_similar = 15
    desc2[:num_similar] = desc1[:num_similar] + np.random.randn(num_similar, 128) * 0.1
    
    # Test matching
    tracker = LandmarkTracker()
    matches = tracker._match_features(desc2, [desc1[i] for i in range(len(desc1))])
    
    print(f"Found {len(matches)} matches between {len(desc1)} and {len(desc2)} features")
    
    if len(matches) > 0:
        print(f"Match ratio: {len(matches) / min(len(desc1), len(desc2)):.2f}")


if __name__ == "__main__":
    # Test realistic tracking
    tracker = test_realistic_tracking()
    
    # Test feature matching
    test_feature_matching()
    
    print("\nLandmark tracking test completed!") 