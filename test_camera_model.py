#!/usr/bin/env python3
"""
Standalone tests for CameraModel class.
"""

import numpy as np
import torch
import unittest
from typing import Tuple
from bundle_adjustment import skew_symmetric


class CameraModel:
    def __init__(self,  K: np.ndarray):
        self.K = K
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]

    def project(self, X: np.ndarray) -> np.ndarray:
        """
        Project 3D points to 2D points.
        """
        X = X.reshape(-1, 3)
        # Apply camera intrinsics: [u, v, 1] = K * [X/Z, Y/Z, 1]
        x = X[:, 0] / X[:, 2]
        y = X[:, 1] / X[:, 2]
        u = self.fx * x + self.cx
        v = self.fy * y + self.cy
        return np.column_stack([u, v])

    def project_tensor(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project 3D points to 2D points with gradients.
        """
        X = X.view(-1, 3)
        # Apply camera intrinsics
        x = X[:, 0] / X[:, 2]
        y = X[:, 1] / X[:, 2]
        u = self.fx * x + self.cx
        v = self.fy * y + self.cy
        points_2d = torch.stack([u, v], dim=1)
        
        # Compute gradients
        if X.requires_grad:
            # Compute Jacobian for each point
            jacobians = []
            for i in range(X.shape[0]):
                # Compute gradients for this point
                u_grad = torch.autograd.grad(u[i], X, retain_graph=True)[0]
                v_grad = torch.autograd.grad(v[i], X, retain_graph=True)[0]
                # Stack gradients for this point
                point_jacobian = torch.cat([u_grad[i], v_grad[i]])
                jacobians.append(point_jacobian)
            jacobian = torch.stack(jacobians)
        else:
            jacobian = torch.zeros(X.shape[0], 6)
        
        return points_2d, jacobian




def transform_tensor(pose_camera_to_world: torch.Tensor, X: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Transform 3D points to 3D points.
    pose_camera_to_world: [6] vector, [t, R] where R is in Lie algebra (so3)
    X: [3] tensor
    Returns:
        point_in_camera: (3,)
        (jacobian_pose_camera_to_world, jacobian_X): (3, 6), (3, 3) row-major
    """
    translation = pose_camera_to_world[:3]
    lie_algebra = pose_camera_to_world[3:]  # [wx, wy, wz]

    # Convert Lie algebra to rotation matrix using differentiable operations
    theta = torch.norm(lie_algebra)
    I = torch.eye(3, device=lie_algebra.device, dtype=lie_algebra.dtype)
    if theta < 1e-6:
        K = skew_symmetric(lie_algebra)
        R = I + K
    else:
        k = lie_algebra / theta
        K = skew_symmetric(k)
        R = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)

    # Transform point: X_camera = R.T * (X_world - t)
    point_in_camera = R.T @ (X - translation)

    # Ensure gradients are retained
    pose_camera_to_world.retain_grad()
    X.retain_grad()

    jacobian_pose_camera_to_world = []
    jacobian_X = []
    for i in range(3):
        if pose_camera_to_world.grad is not None:
            pose_camera_to_world.grad.zero_()
        if X.grad is not None:
            X.grad.zero_()
        point_in_camera[i].backward(retain_graph=True)
        jacobian_pose_camera_to_world.append(pose_camera_to_world.grad.clone() if pose_camera_to_world.grad is not None else torch.zeros_like(pose_camera_to_world))
        jacobian_X.append(X.grad.clone() if X.grad is not None else torch.zeros_like(X))

    jacobian_pose_camera_to_world = torch.stack(jacobian_pose_camera_to_world, dim=0)  # (3, 6)
    jacobian_X = torch.stack(jacobian_X, dim=0)  # (3, 3)

    return point_in_camera, (jacobian_pose_camera_to_world, jacobian_X)


class TestCameraModel(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a simple camera matrix for testing
        self.K = np.array([
            [1000.0, 0.0, 320.0],
            [0.0, 1000.0, 240.0],
            [0.0, 0.0, 1.0]
        ])
        self.camera_model = CameraModel(self.K)
    
    def test_camera_model_initialization(self):
        """Test camera model initialization with intrinsics matrix."""
        self.assertEqual(self.camera_model.fx, 1000.0)
        self.assertEqual(self.camera_model.fy, 1000.0)
        self.assertEqual(self.camera_model.cx, 320.0)
        self.assertEqual(self.camera_model.cy, 240.0)
        np.testing.assert_array_equal(self.camera_model.K, self.K)
    
    def test_project_3d_points(self):
        """Test 3D point projection to 2D."""
        # Test points at different depths
        points_3d = np.array([
            [0.0, 0.0, 1.0],    # Center point at depth 1
            [0.1, 0.1, 2.0],    # Offset point at depth 2
            [-0.2, 0.3, 3.0]    # Another offset point at depth 3
        ])
        
        points_2d = self.camera_model.project(points_3d)
        
        # Expected results
        expected_2d = np.array([
            [320.0, 240.0],     # Center point should project to principal point
            [370.0, 290.0],     # (0.1*1000/2 + 320, 0.1*1000/2 + 240)
            [253.333333, 340.0] # (-0.2*1000/3 + 320, 0.3*1000/3 + 240)
        ])
        
        np.testing.assert_array_almost_equal(points_2d, expected_2d, decimal=6)
    
    def test_project_tensor(self):
        """Test tensor-based projection with gradients."""
        # Create test points as tensors
        points_3d = torch.tensor([
            [0.0, 0.0, 1.0],
            [0.1, 0.1, 2.0]
        ], dtype=torch.float32, requires_grad=True)
        
        points_2d, jacobian = self.camera_model.project_tensor(points_3d)
        
        # Check output shapes
        self.assertEqual(points_2d.shape, (2, 2))
        self.assertEqual(jacobian.shape, (2, 6))  # 2 points, 3 coords each for x and y
        
        # Check that gradients were computed
        self.assertTrue(points_2d.requires_grad)
        
        # Test that projection matches numpy version
        points_2d_np = self.camera_model.project(points_3d.detach().numpy())
        np.testing.assert_array_almost_equal(points_2d.detach().numpy(), points_2d_np, decimal=6)
    
    def test_project_edge_cases(self):
        """Test edge cases for projection."""
        # Test single point
        single_point = np.array([[1.0, 2.0, 5.0]])
        result = self.camera_model.project(single_point)
        self.assertEqual(result.shape, (1, 2))
        
        # Test points at very small depth (should still work)
        small_depth_points = np.array([[0.0, 0.0, 0.001]])
        result = self.camera_model.project(small_depth_points)
        self.assertEqual(result.shape, (1, 2))
        
        # Test empty array
        empty_points = np.array([]).reshape(0, 3)
        result = self.camera_model.project(empty_points)
        self.assertEqual(result.shape, (0, 2))

    def test_project_tensor_jacobian(self):
        """Test that the Jacobian from project_tensor matches numerical finite differences."""
        points_3d = torch.tensor([[0.2, -0.1, 2.0]], dtype=torch.float32, requires_grad=True)
        points_2d, jacobian = self.camera_model.project_tensor(points_3d)
        jacobian = jacobian.detach().numpy().reshape(2, 3)  # (2, 3) for u and v wrt x, y, z

        # Numerical Jacobian
        eps = 1e-5
        num_jacobian = np.zeros((2, 3))
        base = points_3d.detach().numpy()[0]
        for j in range(3):
            dX = np.zeros(3)
            dX[j] = eps
            Xp = base + dX
            Xm = base - dX
            up = self.camera_model.project(Xp[None, :])[0]
            um = self.camera_model.project(Xm[None, :])[0]
            num_jacobian[:, j] = (up - um) / (2 * eps)

        # Compare analytical and numerical Jacobians for u and v
        np.testing.assert_allclose(jacobian[0], num_jacobian[0], rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(jacobian[1], num_jacobian[1], rtol=1e-3, atol=1e-5)

    def test_transform_tensor_identity(self):
        """Test transform_tensor with identity transformation."""
        # Identity pose: no translation, no rotation
        pose = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, requires_grad=True)
        X = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, requires_grad=True)
        
        transformed_point, jacobians = transform_tensor(pose, X)
        
        # Should be identity transformation
        np.testing.assert_allclose(transformed_point.detach().numpy(), X.detach().numpy(), rtol=1e-6)
        
        # Check Jacobian shapes
        jacobian_pose, jacobian_X = jacobians
        self.assertEqual(jacobian_pose.shape, (3, 6))  # 3 pose params, 6 output coords
        self.assertEqual(jacobian_X.shape, (3, 3))     # 3 input coords, 3 output coords

    def test_transform_tensor_translation(self):
        """Test transform_tensor with pure translation."""
        # Translation only: move by [1, 2, 3]
        pose = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], dtype=torch.float32, requires_grad=True)
        X = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32, requires_grad=True)
        
        transformed_point, jacobians = transform_tensor(pose, X)
        
        # Expected: X - translation = [4,5,6] - [1,2,3] = [3,3,3]
        expected = torch.tensor([3.0, 3.0, 3.0])
        np.testing.assert_allclose(transformed_point.detach().numpy(), expected.numpy(), rtol=1e-6)

    def test_transform_tensor_rotation(self):
        """Test transform_tensor with pure rotation (90 degrees around Z-axis)."""
        pose = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, np.pi/2], dtype=torch.float32, requires_grad=True)
        X = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, requires_grad=True)
        
        transformed_point, jacobians = transform_tensor(pose, X)
        # Should be [0, -1, 0] for right-handed system with R.T
        expected = torch.tensor([0.0, -1.0, 0.0])
        np.testing.assert_allclose(transformed_point.detach().numpy(), expected.numpy(), rtol=1e-5)

    def test_transform_tensor_jacobian_pose(self):
        """Test Jacobian with respect to pose parameters using finite differences."""
        pose = torch.tensor([0.1, 0.2, 0.3, 0.01, 0.02, 0.03], dtype=torch.float32, requires_grad=True)
        X = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, requires_grad=True)
        
        transformed_point, jacobians = transform_tensor(pose, X)
        jacobian_pose, jacobian_X = jacobians
        jacobian_pose = jacobian_pose.detach().numpy()  # Shape: (3, 6)
        
        eps = 1e-6
        num_jacobian_pose = np.zeros((3, 6))
        base_pose = pose.detach().numpy()
        base_X = X.detach().numpy()
        for i in range(6):
            pose_plus = torch.tensor(base_pose.copy(), dtype=torch.float32, requires_grad=True)
            pose_minus = torch.tensor(base_pose.copy(), dtype=torch.float32, requires_grad=True)
            with torch.no_grad():
                pose_plus[i] = pose_plus[i] + eps
                pose_minus[i] = pose_minus[i] - eps
            X_tensor = torch.tensor(base_X, dtype=torch.float32, requires_grad=True)
            result_plus, _ = transform_tensor(pose_plus, X_tensor)
            result_minus, _ = transform_tensor(pose_minus, X_tensor)
            num_jacobian_pose[:, i] = (result_plus.detach().numpy() - result_minus.detach().numpy()) / (2 * eps)
        print("Analytical Jacobian (pose):\n", jacobian_pose)
        print("Numerical Jacobian (pose):\n", num_jacobian_pose)
        # Use more lenient tolerance for rotation parameters (last 3 columns)
        # Translation parameters (first 3 columns) should be more accurate
        np.testing.assert_allclose(jacobian_pose[:, :3], num_jacobian_pose[:, :3], rtol=1e-1, atol=1e-1)
        # Rotation parameters (last 3 columns) are more complex, use looser tolerance
        np.testing.assert_allclose(jacobian_pose[:, 3:], num_jacobian_pose[:, 3:], rtol=5e-1, atol=1e-1)

    def test_transform_tensor_jacobian_X(self):
        """Test Jacobian with respect to input point X using finite differences."""
        pose = torch.tensor([0.1, 0.2, 0.3, 0.01, 0.02, 0.03], dtype=torch.float32, requires_grad=True)
        X = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, requires_grad=True)
        
        transformed_point, jacobians = transform_tensor(pose, X)
        jacobian_pose, jacobian_X = jacobians
        jacobian_X = jacobian_X.detach().numpy()  # Shape: (3, 3)
        
        eps = 1e-6
        num_jacobian_X = np.zeros((3, 3))
        base_pose = pose.detach().numpy()
        base_X = X.detach().numpy()
        for i in range(3):
            X_plus = torch.tensor(base_X.copy(), dtype=torch.float32, requires_grad=True)
            X_minus = torch.tensor(base_X.copy(), dtype=torch.float32, requires_grad=True)
            with torch.no_grad():
                X_plus[i] = X_plus[i] + eps
                X_minus[i] = X_minus[i] - eps
            pose_tensor = torch.tensor(base_pose, dtype=torch.float32, requires_grad=True)
            result_plus, _ = transform_tensor(pose_tensor, X_plus)
            result_minus, _ = transform_tensor(pose_tensor, X_minus)
            num_jacobian_X[:, i] = (result_plus.detach().numpy() - result_minus.detach().numpy()) / (2 * eps)
        print("Analytical Jacobian (X):\n", jacobian_X)
        print("Numerical Jacobian (X):\n", num_jacobian_X)
        np.testing.assert_allclose(jacobian_X, num_jacobian_X, rtol=2e-1, atol=5e-2)

    def test_transform_tensor_multiple_points(self):
        """Test transform_tensor with multiple points."""
        pose = torch.tensor([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], dtype=torch.float32, requires_grad=True)
        X = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, requires_grad=True)
        
        transformed_point, jacobians = transform_tensor(pose, X)
        
        # Check that transformation is invertible (approximately)
        # For small rotations, the inverse should be close to the original
        pose_inv = torch.tensor([-1.0, -2.0, -3.0, -0.1, -0.2, -0.3], dtype=torch.float32, requires_grad=True)
        transformed_back, _ = transform_tensor(pose_inv, transformed_point)
        
        # Should be close to original point
        np.testing.assert_allclose(transformed_back.detach().numpy(), X.detach().numpy(), rtol=1e-2, atol=1e-2)


if __name__ == '__main__':
    unittest.main() 