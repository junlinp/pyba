# Loading Ground Truth Poses from KITTI Dataset

This document explains how to load and use ground truth poses from the KITTI odometry dataset.

## Overview

The KITTI odometry dataset provides ground truth poses for each sequence, which are essential for:
- Evaluating visual odometry/SLAM algorithms
- Computing trajectory errors
- Benchmarking pose estimation methods
- Training and validation of learning-based approaches

## Data Format

### Pose File Location
Ground truth poses are stored in text files at:
```
/mnt/rbd0/KITTI/odometry/dataset/poses/{sequence_id:02d}.txt
```

### Pose Format
Each pose file contains:
- **Format**: Text file with one pose per line
- **Structure**: Each line contains 12 values representing a flattened 3×4 transformation matrix
- **Reference**: Poses are relative to the first frame of the sequence
- **Units**: Meters for translation, radians for rotation

### Pose Matrix Structure
Each pose is stored as a flattened 3×4 matrix:
```
[R11 R12 R13 t1]
[R21 R22 R23 t2]  -> [R11 R12 R13 t1 R21 R22 R23 t2 R31 R32 R33 t3]
[R31 R32 R33 t3]
```

Where:
- `R` is the 3×3 rotation matrix
- `t` is the 3×1 translation vector

## Loading Poses

### Using KITTIOdometryReader

```python
from kitti_reader import KITTIOdometryReader

# Initialize reader
reader = KITTIOdometryReader()

# Load poses for a sequence
sequence_id = 0
gt_poses = reader.load_poses(sequence_id)

print(f"Loaded {len(gt_poses)} poses")
print(f"Pose array shape: {gt_poses.shape}")  # (N, 12)
```

### Converting to 4×4 Transformation Matrices

```python
def poses_to_transforms(poses: np.ndarray) -> np.ndarray:
    """Convert KITTI poses (N, 12) to 4x4 transformation matrices (N, 4, 4)."""
    N = poses.shape[0]
    transforms = np.zeros((N, 4, 4))
    
    for i in range(N):
        # Reshape pose to 3x4
        pose_3x4 = poses[i].reshape(3, 4)
        
        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :4] = pose_3x4
        
        transforms[i] = transform
    
    return transforms

# Convert poses
gt_transforms = poses_to_transforms(gt_poses)
```

## Working with Poses

### Extracting Trajectory

```python
# Get trajectory (translation components only)
trajectory = reader.get_trajectory(sequence_id)
print(f"Trajectory shape: {trajectory.shape}")  # (N, 3)

# Compute total distance
total_distance = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
print(f"Total trajectory distance: {total_distance:.2f} meters")
```

### Computing Relative Poses

```python
def get_relative_pose(pose1, pose2):
    """Compute relative pose from pose1 to pose2."""
    # Reshape to 3x4 matrices
    T1 = pose1.reshape(3, 4)
    T2 = pose2.reshape(3, 4)
    
    # Convert to 4x4 homogeneous matrices
    H1 = np.eye(4)
    H1[:3, :4] = T1
    
    H2 = np.eye(4)
    H2[:3, :4] = T2
    
    # Compute relative pose: H2 * inv(H1)
    relative_pose = H2 @ np.linalg.inv(H1)
    
    return relative_pose[:3, :4].flatten()
```

### Pose Evaluation

```python
def compute_relative_pose_error(gt_poses, estimated_poses):
    """Compute relative pose error between ground truth and estimated poses."""
    if len(gt_poses) != len(estimated_poses):
        raise ValueError("Pose arrays must have same length")
    
    total_error = 0.0
    count = 0
    
    for i in range(1, len(gt_poses)):
        # Get ground truth relative pose
        gt_prev = gt_poses[i-1].reshape(3, 4)
        gt_curr = gt_poses[i].reshape(3, 4)
        gt_relative = np.linalg.inv(gt_prev) @ gt_curr
        
        # Get estimated relative pose
        est_prev = estimated_poses[i-1].reshape(3, 4)
        est_curr = estimated_poses[i].reshape(3, 4)
        est_relative = np.linalg.inv(est_prev) @ est_curr
        
        # Compute error
        error_matrix = np.linalg.inv(gt_relative) @ est_relative
        translation_error = np.linalg.norm(error_matrix[:3, 3])
        total_error += translation_error
        count += 1
    
    return total_error / count if count > 0 else 0.0
```

## Example Usage

### Complete Example

```python
#!/usr/bin/env python3
import numpy as np
from kitti_reader import KITTIOdometryReader

def main():
    # Load KITTI data
    reader = KITTIOdometryReader()
    seq = reader.sequences[0]
    
    # Load ground truth poses
    gt_poses = reader.load_poses(seq)
    print(f"Loaded {len(gt_poses)} ground truth poses")
    
    # Convert to 4x4 transformation matrices
    gt_transforms = poses_to_transforms(gt_poses)
    
    # Analyze trajectory
    trajectory = reader.get_trajectory(seq)
    total_distance = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
    print(f"Total trajectory distance: {total_distance:.2f} meters")
    
    # Show first few poses
    for i in range(min(3, len(gt_poses))):
        pose_3x4 = gt_poses[i].reshape(3, 4)
        print(f"Pose {i}:")
        print(pose_3x4)
        print()

if __name__ == "__main__":
    main()
```

### Test Script

Run the test script to verify pose loading:

```bash
python test_gt_poses.py
```

## Integration with SFM Pipeline

The ground truth poses are now integrated into the SFM pipeline (`kitti_sfm.py`):

1. **Loading**: Poses are loaded using `reader.load_poses(seq)`
2. **Conversion**: Converted to 4×4 matrices using `poses_to_transforms()`
3. **Analysis**: Trajectory analysis and distance computation
4. **Evaluation**: Ready for comparison with estimated poses

## Notes

- **Coordinate System**: KITTI uses a right-handed coordinate system
- **Units**: All distances are in meters
- **Reference Frame**: Poses are relative to the first frame (frame 0)
- **File Format**: ASCII text files with space-separated values
- **Precision**: Double precision floating point values

## Troubleshooting

### Common Issues

1. **File Not Found**: Ensure the dataset path is correct
2. **Shape Mismatch**: Verify pose array has shape (N, 12)
3. **Invalid Poses**: Check that rotation matrices are orthogonal
4. **Memory Issues**: For large sequences, consider loading poses in chunks

### Validation

```python
# Validate rotation matrices
for i, pose in enumerate(gt_poses):
    R = pose.reshape(3, 4)[:3, :3]
    R_RT = R @ R.T
    if not np.allclose(R_RT, np.eye(3), atol=1e-6):
        print(f"Warning: Pose {i} has non-orthogonal rotation matrix")
``` 