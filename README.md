# pyba - Python Bundle Adjustment and Landmark Tracking

A comprehensive Python library for Structure from Motion (SFM), landmark tracking, and bundle adjustment using the KITTI dataset.

## Overview

The pyba library provides easy-to-use functionality for:

- **Landmark Tracking**: Incremental tracking of features across image sequences
- **Bundle Adjustment**: Optimization of camera poses and 3D points using pyceres
- **KITTI Dataset Support**: Easy loading and processing of KITTI odometry data
- **SFM Pipeline**: Complete Structure from Motion pipeline with LightGlue features
- **Visualization**: Built-in visualization for trajectories and point clouds

## Features

- **Modern Feature Matching**: Uses LightGlue with SuperPoint for robust feature detection and matching
- **Incremental Tracking**: LandmarkTracker class for tracking features across multiple frames
- **Bundle Adjustment**: Optimize camera poses and 3D points using pyceres
- **KITTI Integration**: Direct support for KITTI odometry dataset
- **Visualization Tools**: Plot trajectories, point clouds, and debug images

## Installation

### Basic Installation

```bash
# Install from local directory
pip install -e .

# Or install with optional dependencies
pip install -e .[full]  # Includes pyceres for bundle adjustment
```

### From GitHub (if you have a repository)

```bash
pip install git+https://github.com/yourusername/pyba.git

# With optional dependencies
pip install git+https://github.com/yourusername/pyba.git#egg=pyba[full]
```

### Dependencies

The package requires:
- `numpy>=1.21.0`
- `opencv-python>=4.5.0`
- `matplotlib>=3.5.0`
- `torch>=2.0.0`
- `torchvision>=0.15.0`
- `kornia>=0.6.11`
- `lightglue` (from GitHub)
- `pyceres>=0.0.1` (optional, for bundle adjustment)

## Quick Start

### Basic Usage

```python
import pyba
from pyba import LandmarkTracker, KITTIOdometryReader

# Initialize KITTI reader
reader = KITTIOdometryReader()
sequences = reader.sequences
print(f"Available sequences: {sequences}")

# Initialize landmark tracker
tracker = LandmarkTracker(min_track_length=3)

# Load data from sequence
seq = sequences[0]
calib = reader.load_calibration(seq)
poses = reader.load_poses(seq)
```

### Landmark Tracking

```python
from pyba import LandmarkTracker, extract_superpoint_features, match_keypoints

# Initialize tracker
tracker = LandmarkTracker()

# Process frames
for frame_id in range(10):
    # Load images
    img0 = reader.load_image(seq, frame_id, 'left')
    img1 = reader.load_image(seq, frame_id + 1, 'left')
    
    # Extract features
    features0 = extract_superpoint_features(img0)
    features1 = extract_superpoint_features(img1)
    
    # Match features
    kpts0, kpts1, matches = match_keypoints(features0, features1)
    
    # Add to tracker
    tracker.add_matched_frame(frame_id, frame_id + 1, kpts0, kpts1, matches)
```

### Bundle Adjustment

```python
from pyba import BundleAdjuster

# Initialize bundle adjuster
ba = BundleAdjuster(fix_first_pose=True)

# Get observations from tracker
observations = tracker.observation_relations_for_ba()
points_3d = tracker.get_landmark_point3ds()

# Run bundle adjustment
summary, optimized_poses, optimized_points = ba.run(
    points_3d, observations, camera_poses, K
)
```

## API Reference

### Main Classes

#### `LandmarkTracker`
Tracks landmarks incrementally across image sequences.

```python
tracker = LandmarkTracker(min_track_length=3, max_reprojection_error=2.0)
tracker.add_matched_frame(timestamp0, timestamp1, keypoints0, keypoints1, matches)
observations = tracker.observation_relations_for_ba()
points_3d = tracker.get_landmark_point3ds()
```

#### `BundleAdjuster`
Optimizes camera poses and 3D points using bundle adjustment.

```python
ba = BundleAdjuster(fix_first_pose=True, fix_intrinsics=True)
summary, optimized_poses, optimized_points = ba.run(points_3d, observations, camera_poses, K)
```

#### `KITTIOdometryReader`
Loads and manages KITTI odometry dataset.

```python
reader = KITTIOdometryReader()
calib = reader.load_calibration(sequence_id)
poses = reader.load_poses(sequence_id)
images = reader.load_image(sequence_id, frame_id, 'left')
```

### Utility Functions

#### Feature Extraction and Matching
- `extract_superpoint_features(image)`: Extract SuperPoint features
- `match_keypoints(features0, features1)`: Match features using LightGlue

#### Geometry Utilities
- `triangulate_points_multiview(keypoints_list, camera_poses, K)`: Multi-view triangulation
- `project_point_to_image(point_3d, camera_pose, K)`: Project 3D point to image
- `poses_to_transforms(poses)`: Convert KITTI poses to 4x4 matrices

#### Rotation Utilities
- `rotation_matrix_to_angle_axis(R)`: Convert rotation matrix to angle-axis
- `angle_axis_to_rotation_matrix(w)`: Convert angle-axis to rotation matrix

## Examples

### Complete SFM Pipeline

Run the main SFM pipeline:

```bash
python -m pyba.kitti_sfm
```

This will:
1. Load KITTI sequence data
2. Extract and match features
3. Track landmarks across frames
4. Triangulate 3D points
5. Run bundle adjustment
6. Save results and visualizations

### Development

For development, install with dev dependencies:

```bash
pip install -e .[dev]
```

Run tests:
```bash
pytest
```

Format code:
```bash
black .
```

## Dataset Structure

The KITTI odometry dataset should have the following structure:

```
/path/to/kitti/odometry/
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

## License

This code is provided as-is for educational and research purposes.

## References

- KITTI Dataset: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
- LightGlue: https://github.com/cvg/LightGlue
- SuperPoint: https://github.com/magicleap/SuperPointPretrainedNetwork 