#!/usr/bin/env python3
"""
KITTI Odometry Dataset Reader

This module provides functionality to read and process KITTI odometry dataset
from /mnt/rbd0/KITTI/odometry directory.

The KITTI odometry dataset contains:
- Stereo images (left and right cameras)
- Calibration data
- Ground truth poses
- Timestamps

Author: Assistant
Date: 2024
"""

import os
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt


class KITTIOdometryReader:
    """
    A class to read and process KITTI odometry dataset.
    """
    
    def __init__(self, dataset_path: str = "/mnt/rbd0/KITTI/odometry/dataset/sequences"):
        """
        Initialize the KITTI odometry reader.
        
        Args:
            dataset_path (str): Path to the KITTI odometry dataset
        """
        self.dataset_path = Path(dataset_path)
        self.sequences = []
        self.calibration_data = {}
        
        # Check if dataset path exists
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")
        
        # Get available sequences
        self._load_sequences()
        
    def _load_sequences(self):
        """Load available sequence directories."""
        for item in self.dataset_path.iterdir():
            if item.is_dir() and item.name.isdigit():
                self.sequences.append(int(item.name))
        self.sequences.sort()
        print(f"Found {len(self.sequences)} sequences: {self.sequences}")
    
    def get_sequence_path(self, sequence_id: int) -> Path:
        """
        Get the path to a specific sequence.
        
        Args:
            sequence_id (int): Sequence ID
            
        Returns:
            Path: Path to the sequence directory
        """
        if sequence_id not in self.sequences:
            raise ValueError(f"Sequence {sequence_id} not found. Available: {self.sequences}")
        return self.dataset_path / f"{sequence_id:02d}"
    
    def load_calibration(self, sequence_id: int) -> Dict[str, np.ndarray]:
        """
        Load calibration data for a sequence.
        
        Args:
            sequence_id (int): Sequence ID
            
        Returns:
            Dict[str, np.ndarray]: Calibration matrices
        """
        if sequence_id in self.calibration_data:
            return self.calibration_data[sequence_id]
        
        calib_file = self.get_sequence_path(sequence_id) / "calib.txt"
        if not calib_file.exists():
            raise FileNotFoundError(f"Calibration file not found: {calib_file}")
        
        calibration = {}
        with open(calib_file, 'r') as f:
            for line in f:
                key, value = line.split(':', 1)
                calibration[key.strip()] = np.array([float(x) for x in value.split()]).reshape(3, 4)
        
        self.calibration_data[sequence_id] = calibration
        return calibration
    
    def load_image(self, sequence_id: int, frame_id: int, camera: str = 'left') -> np.ndarray:
        """
        Load an image from the dataset.
        
        Args:
            sequence_id (int): Sequence ID
            frame_id (int): Frame ID
            camera (str): Camera type ('left' or 'right')
            
        Returns:
            np.ndarray: Image as numpy array (H, W, 3) in RGB format
        """
        if camera == 'left':
            image_dir = self.get_sequence_path(sequence_id) / "image_2"
        elif camera == 'right':
            image_dir = self.get_sequence_path(sequence_id) / "image_3"
        else:
            raise ValueError("Camera must be 'left' or 'right'")
        image_file = image_dir / f"{frame_id:06d}.png"
        if not image_file.exists():
            raise FileNotFoundError(f"Image file not found: {image_file}")
        image = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load image: {image_file}")
        return image
    
    def load_poses(self, sequence_id: int) -> np.ndarray:
        """
        Load ground truth poses for a sequence.
        
        Args:
            sequence_id (int): Sequence ID
            
        Returns:
            np.ndarray: Poses as numpy array (N, 12) - flattened 3x4 transformation matrices
        """
        poses_file = Path("/mnt/rbd0/KITTI/odometry/dataset/poses") / f"{sequence_id:02d}.txt"
        if not poses_file.exists():
            raise FileNotFoundError(f"Poses file not found: {poses_file}")
        poses = np.loadtxt(str(poses_file))
        return poses
    
    def get_sequence_info(self, sequence_id: int) -> Dict[str, int]:
        """
        Get information about a sequence.
        
        Args:
            sequence_id (int): Sequence ID
            
        Returns:
            Dict[str, int]: Sequence information
        """
        sequence_path = self.get_sequence_path(sequence_id)
        
        # Count images
        left_images = len(list((sequence_path / "image_2").glob("*.png")))
        right_images = len(list((sequence_path / "image_3").glob("*.png")))
        
        # Count poses
        poses_file = Path("/mnt/rbd0/KITTI/odometry/dataset/poses") / f"{sequence_id:02d}.txt"
        num_poses = 0
        if poses_file.exists():
            with open(poses_file, 'r') as f:
                num_poses = sum(1 for line in f)
        
        return {
            'left_images': left_images,
            'right_images': right_images,
            'poses': num_poses
        }
    
    def visualize_frame(self, sequence_id: int, frame_id: int) -> None:
        """
        Visualize a frame from the dataset.
        
        Args:
            sequence_id (int): Sequence ID
            frame_id (int): Frame ID
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Load stereo images
        left_img = self.load_image(sequence_id, frame_id, 'left')
        right_img = self.load_image(sequence_id, frame_id, 'right')
        
        # Display stereo images
        axes[0].imshow(left_img)
        axes[0].set_title(f'Left Camera - Sequence {sequence_id}, Frame {frame_id}')
        axes[0].axis('off')
        
        axes[1].imshow(right_img)
        axes[1].set_title(f'Right Camera - Sequence {sequence_id}, Frame {frame_id}')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_trajectory(self, sequence_id: int) -> np.ndarray:
        """
        Extract trajectory from poses.
        
        Args:
            sequence_id (int): Sequence ID
            
        Returns:
            np.ndarray: Trajectory as numpy array (N, 3) - x, y, z positions
        """
        poses = self.load_poses(sequence_id)
        trajectory = poses[:, [3, 7, 11]]  # Extract translation components
        return trajectory
    
    def visualize_trajectory(self, sequence_id: int) -> None:
        """
        Visualize the trajectory for a sequence.
        
        Args:
            sequence_id (int): Sequence ID
        """
        trajectory = self.get_trajectory(sequence_id)
        
        plt.figure(figsize=(12, 8))
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Trajectory')
        plt.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Start')
        plt.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=10, label='End')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title(f'Trajectory - Sequence {sequence_id}')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()
    
    def load_timestamps(self, sequence_id: int) -> np.ndarray:
        """
        Load timestamps for a sequence.
        
        Args:
            sequence_id (int): Sequence ID
            
        Returns:
            np.ndarray: Timestamps as numpy array (N,) in seconds from sequence start
        """
        times_file = self.get_sequence_path(sequence_id) / "times.txt"
        if not times_file.exists():
            raise FileNotFoundError(f"Times file not found: {times_file}")
        
        timestamps = np.loadtxt(str(times_file))
        return timestamps
    
    def get_timestamp(self, sequence_id: int, frame_id: int) -> float:
        """
        Get timestamp for a specific frame.
        
        Args:
            sequence_id (int): Sequence ID
            frame_id (int): Frame ID
            
        Returns:
            float: Timestamp in seconds from sequence start
        """
        timestamps = self.load_timestamps(sequence_id)
        if frame_id >= len(timestamps):
            raise ValueError(f"Frame {frame_id} not found. Available frames: 0-{len(timestamps)-1}")
        return timestamps[frame_id]


def main():
    """
    Example usage of the KITTI odometry reader.
    """
    try:
        # Initialize reader
        reader = KITTIOdometryReader()
        
        print("KITTI Odometry Dataset Reader")
        print("=" * 40)
        
        # Print available sequences
        print(f"Available sequences: {reader.sequences}")
        
        if not reader.sequences:
            print("No sequences found in the dataset.")
            return
        
        # Example: Get info for first sequence
        first_sequence = reader.sequences[0]
        info = reader.get_sequence_info(first_sequence)
        print(f"\nSequence {first_sequence} info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Example: Load calibration data
        print(f"\nLoading calibration for sequence {first_sequence}...")
        calib = reader.load_calibration(first_sequence)
        print("Calibration matrices:")
        for key, matrix in calib.items():
            print(f"  {key}:")
            print(f"    {matrix}")
        
        # Example: Visualize first frame
        print(f"\nVisualizing first frame of sequence {first_sequence}...")
        reader.visualize_frame(first_sequence, 0)
        
        # Example: Visualize trajectory
        print(f"\nVisualizing trajectory of sequence {first_sequence}...")
        reader.visualize_trajectory(first_sequence)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check if the dataset path is correct and the dataset is properly installed.")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main() 