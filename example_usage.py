#!/usr/bin/env python3
"""
Example usage of the KITTI Odometry Reader

This script demonstrates various ways to use the KITTI reader class.
"""

from kitti_reader import KITTIOdometryReader
import numpy as np


def basic_usage_example():
    """Basic usage example showing how to load different data types."""
    print("=== Basic Usage Example ===")
    
    # Initialize the reader
    reader = KITTIOdometryReader()
    
    if not reader.sequences:
        print("No sequences found. Please check the dataset path.")
        return None, None
    
    # Get the first available sequence
    sequence_id = reader.sequences[0]
    print(f"Using sequence {sequence_id}")
    
    # Get sequence information
    info = reader.get_sequence_info(sequence_id)
    print(f"Sequence info: {info}")
    
    # Load calibration data
    calib = reader.load_calibration(sequence_id)
    print(f"Calibration keys: {list(calib.keys())}")
    
    # Load first frame data
    frame_id = 0
    
    # Load stereo images
    left_image = reader.load_image(sequence_id, frame_id, 'left')
    right_image = reader.load_image(sequence_id, frame_id, 'right')
    print(f"Left image shape: {left_image.shape}")
    print(f"Right image shape: {right_image.shape}")
    
    # Load poses
    poses = reader.load_poses(sequence_id)
    print(f"Number of poses: {len(poses)}")
    
    return reader, sequence_id


def data_processing_example():
    """Example of processing KITTI data for computer vision tasks."""
    print("\n=== Data Processing Example ===")
    
    reader, sequence_id = basic_usage_example()
    if reader is None:
        return
    
    frame_id = 0
    
    # Load data
    left_image = reader.load_image(sequence_id, frame_id, 'left')
    calib = reader.load_calibration(sequence_id)
    
    # Example: Extract trajectory
    trajectory = reader.get_trajectory(sequence_id)
    print(f"Trajectory length: {len(trajectory)} poses")
    print(f"Total distance: {np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)):.2f} meters")
    
    # Example: Analyze image statistics
    print(f"Image statistics:")
    print(f"  Mean RGB values: {np.mean(left_image, axis=(0,1))}")
    print(f"  Image range: [{left_image.min()}, {left_image.max()}]")


def visualization_example():
    """Example of visualizing KITTI data."""
    print("\n=== Visualization Example ===")
    
    reader, sequence_id = basic_usage_example()
    if reader is None:
        return
    
    # Visualize first frame
    print("Visualizing first frame...")
    reader.visualize_frame(sequence_id, 0)
    
    # Visualize trajectory
    print("Visualizing trajectory...")
    reader.visualize_trajectory(sequence_id)


def batch_processing_example():
    """Example of processing multiple frames."""
    print("\n=== Batch Processing Example ===")
    
    reader, sequence_id = basic_usage_example()
    if reader is None:
        return
    
    # Process first 10 frames
    num_frames = min(10, reader.get_sequence_info(sequence_id)['left_images'])
    
    print(f"Processing {num_frames} frames...")
    
    for frame_id in range(num_frames):
        try:
            # Load data
            left_image = reader.load_image(sequence_id, frame_id, 'left')
            
            # Simple processing: calculate image brightness
            brightness = np.mean(left_image)
            
            print(f"Frame {frame_id:03d}: Brightness = {brightness:.2f}")
            
        except FileNotFoundError:
            print(f"Frame {frame_id} not found, skipping...")
            break


if __name__ == "__main__":
    print("KITTI Odometry Reader - Example Usage")
    print("=" * 50)
    
    try:
        # Run examples
        basic_usage_example()
        data_processing_example()
        visualization_example()
        batch_processing_example()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the KITTI dataset is available at /mnt/rbd0/KITTI/odometry")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc() 