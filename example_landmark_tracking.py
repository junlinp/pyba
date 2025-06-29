#!/usr/bin/env python3
"""
Example: Landmark Tracking with KITTI Dataset

This script demonstrates how to use the LandmarkTracker class
with KITTI odometry data and SuperPoint features.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add current directory to path for imports
sys.path.append('.')

from kitti_reader import KITTIOdometryReader
from kitti_sfm import extract_superpoint_features, match_features_lightglue
from landmark_tracker import LandmarkTracker


def run_landmark_tracking_example():
    """Run landmark tracking example with KITTI data."""
    
    # Initialize KITTI reader
    dataset_path = "/mnt/rbd0/KITTI/odometry"
    reader = KITTIOdometryReader(dataset_path)
    
    # Get available sequences
    sequences = reader.get_available_sequences()
    print(f"Available sequences: {sequences}")
    
    if not sequences:
        print("No sequences found!")
        return
    
    # Use sequence 0
    sequence_id = 0
    print(f"Processing sequence {sequence_id}")
    
    # Get sequence data
    sequence_data = reader.load_sequence(sequence_id)
    if not sequence_data:
        print(f"Failed to load sequence {sequence_id}")
        return
    
    # Initialize landmark tracker
    tracker = LandmarkTracker(min_track_length=3, max_reprojection_error=2.0)
    
    # Process frames
    num_frames = min(10, len(sequence_data['left_images']))  # Process first 10 frames
    print(f"Processing {num_frames} frames...")
    
    camera_matrix = sequence_data['calibration']['P_rect_02'][:3, :3]
    
    for frame_id in range(num_frames):
        print(f"Processing frame {frame_id}")
        
        # Load image
        img = sequence_data['left_images'][frame_id]
        
        # Extract features using SuperPoint
        keypoints, descriptors = extract_superpoint_features(img)
        
        if len(keypoints) == 0:
            print(f"No features found in frame {frame_id}")
            continue
        
        # Convert keypoints to numpy array
        kpts_array = np.array([kp.pt for kp in keypoints])
        
        # For simplicity, use identity pose for now
        # In a real application, you'd estimate camera pose from previous frames
        camera_pose = np.eye(4)
        if frame_id > 0:
            # Simple translation for demonstration
            camera_pose[:3, 3] = np.array([frame_id * 0.5, 0, 0])
        
        # Add frame to tracker
        landmarks = tracker.add_frame(frame_id, kpts_array, descriptors, camera_pose, camera_matrix)
        
        print(f"Frame {frame_id}: {len(landmarks)} landmarks processed")
    
    # Get tracking statistics
    stats = tracker.get_statistics()
    print("\nTracking Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Get stable landmarks
    stable_landmarks = tracker.get_stable_landmarks()
    print(f"\nStable landmarks: {len(stable_landmarks)}")
    
    # Visualize landmark positions
    if len(stable_landmarks) > 0:
        positions = tracker.get_landmark_positions()
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot landmark positions
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                  c='red', s=10, alpha=0.6, label='Landmarks')
        
        # Plot camera trajectory
        camera_positions = []
        for frame_id in range(num_frames):
            if frame_id in tracker.frame_landmarks:
                # Get camera pose (simplified)
                pose = np.eye(4)
                if frame_id > 0:
                    pose[:3, 3] = np.array([frame_id * 0.5, 0, 0])
                camera_positions.append(pose[:3, 3])
        
        if camera_positions:
            camera_positions = np.array(camera_positions)
            ax.plot(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], 
                   'b-', linewidth=2, label='Camera Trajectory')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Landmark Tracking Results')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('landmark_tracking_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("Visualization saved as 'landmark_tracking_results.png'")
    
    # Save landmark data
    save_landmark_data(tracker, f"landmarks_sequence_{sequence_id}.npz")
    
    return tracker


def save_landmark_data(tracker, filename):
    """Save landmark tracking data to file."""
    stable_landmarks = tracker.get_stable_landmarks()
    
    if len(stable_landmarks) == 0:
        print("No stable landmarks to save")
        return
    
    # Prepare data for saving
    landmark_ids = []
    positions = []
    track_lengths = []
    first_frames = []
    last_frames = []
    
    for landmark in stable_landmarks:
        landmark_ids.append(landmark.id)
        positions.append(landmark.position_3d)
        track_lengths.append(landmark.track_length)
        first_frames.append(landmark.first_frame)
        last_frames.append(landmark.last_frame)
    
    # Convert to numpy arrays
    landmark_ids = np.array(landmark_ids)
    positions = np.array(positions)
    track_lengths = np.array(track_lengths)
    first_frames = np.array(first_frames)
    last_frames = np.array(last_frames)
    
    # Save to file
    np.savez(filename,
             landmark_ids=landmark_ids,
             positions=positions,
             track_lengths=track_lengths,
             first_frames=first_frames,
             last_frames=last_frames)
    
    print(f"Landmark data saved to {filename}")


def analyze_landmark_tracks(tracker):
    """Analyze and print detailed information about landmark tracks."""
    stable_landmarks = tracker.get_stable_landmarks()
    
    if len(stable_landmarks) == 0:
        print("No stable landmarks to analyze")
        return
    
    print(f"\nDetailed Landmark Analysis:")
    print(f"Total stable landmarks: {len(stable_landmarks)}")
    
    # Analyze track lengths
    track_lengths = [lm.track_length for lm in stable_landmarks]
    print(f"Track length statistics:")
    print(f"  Min: {min(track_lengths)}")
    print(f"  Max: {max(track_lengths)}")
    print(f"  Mean: {np.mean(track_lengths):.2f}")
    print(f"  Median: {np.median(track_lengths):.2f}")
    
    # Show some example landmarks
    print(f"\nExample landmarks:")
    for i, landmark in enumerate(stable_landmarks[:5]):
        print(f"  Landmark {landmark.id}:")
        print(f"    Position: {landmark.position_3d}")
        print(f"    Track length: {landmark.track_length}")
        print(f"    First frame: {landmark.first_frame}")
        print(f"    Last frame: {landmark.last_frame}")
        print(f"    Observations: {len(landmark.observations)}")


if __name__ == "__main__":
    try:
        tracker = run_landmark_tracking_example()
        if tracker:
            analyze_landmark_tracks(tracker)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc() 