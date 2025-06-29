# KITTI Odometry Dataset Reader

A comprehensive Python library for reading and processing the KITTI odometry dataset from `/mnt/rbd0/KITTI/odometry`.

## Overview

The KITTI odometry dataset is a popular benchmark for visual odometry and SLAM (Simultaneous Localization and Mapping) research. This library provides easy-to-use functionality to:

- Load stereo images (left and right cameras)
- Load LiDAR point clouds
- Load calibration data
- Load ground truth poses
- Visualize data and trajectories
- Process data for computer vision tasks

## Features

- **Easy Data Loading**: Simple interface to load different data types
- **Visualization**: Built-in visualization for images, point clouds, and trajectories
- **Data Processing**: Utilities for filtering and processing point clouds
- **Batch Processing**: Support for processing multiple frames
- **Error Handling**: Robust error handling for missing files or invalid data

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure the KITTI dataset is available at `/mnt/rbd0/KITTI/odometry`

## Usage

### Basic Usage

```python
from kitti_reader import KITTIOdometryReader

# Initialize the reader
reader = KITTIOdometryReader()

# Get available sequences
print(f"Available sequences: {reader.sequences}")

# Load data from the first sequence
sequence_id = reader.sequences[0]
frame_id = 0

# Load stereo images
left_image = reader.load_image(sequence_id, frame_id, 'left')
right_image = reader.load_image(sequence_id, frame_id, 'right')

# Load point cloud
point_cloud = reader.load_point_cloud(sequence_id, frame_id)

# Load calibration data
calib = reader.load_calibration(sequence_id)

# Load poses
poses = reader.load_poses(sequence_id)
```

### Visualization

```python
# Visualize a frame (stereo images + point cloud)
reader.visualize_frame(sequence_id, frame_id)

# Visualize trajectory
reader.visualize_trajectory(sequence_id)
```

### Data Processing

```python
# Get sequence information
info = reader.get_sequence_info(sequence_id)
print(f"Sequence info: {info}")

# Extract trajectory
trajectory = reader.get_trajectory(sequence_id)

# Filter point cloud
mask = (point_cloud[:, 2] > -1.5) & (np.linalg.norm(point_cloud[:, :3], axis=1) < 50.0)
filtered_points = point_cloud[mask]
```

## Dataset Structure

The KITTI odometry dataset should have the following structure:

```
/mnt/rbd0/KITTI/odometry/
├── 00/
│   ├── calib.txt
│   ├── poses.txt
│   ├── image_left/
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...
│   ├── image_right/
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...
│   └── velodyne/
│       ├── 000000.bin
│       ├── 000001.bin
│       └── ...
├── 01/
└── ...
```

## Data Format

### Images
- Format: PNG
- Resolution: 1242x375 pixels
- Color space: RGB (converted from BGR)

### Point Clouds
- Format: Binary (.bin)
- Structure: Nx4 array (x, y, z, intensity)
- Coordinate system: Velodyne coordinate system

### Calibration
- Format: Text file
- Contains: Camera intrinsics, extrinsics, and stereo parameters

### Poses
- Format: Text file
- Structure: Nx12 array (flattened 3x4 transformation matrices)
- Reference: First frame of the sequence

## Examples

Run the example script to see various usage patterns:

```bash
python example_usage.py
```

Or run the main script for a complete demonstration:

```bash
python kitti_reader.py
```

## API Reference

### KITTIOdometryReader

#### Methods

- `__init__(dataset_path)`: Initialize the reader
- `load_image(sequence_id, frame_id, camera)`: Load stereo image
- `load_point_cloud(sequence_id, frame_id)`: Load LiDAR point cloud
- `load_calibration(sequence_id)`: Load calibration data
- `load_poses(sequence_id)`: Load ground truth poses
- `get_sequence_info(sequence_id)`: Get sequence information
- `visualize_frame(sequence_id, frame_id)`: Visualize frame data
- `visualize_trajectory(sequence_id)`: Visualize trajectory
- `get_trajectory(sequence_id)`: Extract trajectory from poses

## Error Handling

The library includes comprehensive error handling for:
- Missing dataset directory
- Missing sequence directories
- Missing data files
- Invalid file formats
- Corrupted data

## Dependencies

- `numpy`: Numerical computing
- `opencv-python`: Image processing
- `matplotlib`: Visualization
- `pathlib`: Path handling

## License

This code is provided as-is for educational and research purposes.

## References

- KITTI Dataset: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
- KITTI Paper: "Are we ready for autonomous driving? The KITTI vision benchmark suite" 