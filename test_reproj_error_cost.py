#!/usr/bin/env python3
"""
Tests for ReprojErrorCost class.
"""

import numpy as np
import torch
import unittest
from typing import Tuple, List
from bundle_adjustment import CameraModel, transform_tensor, ReprojErrorCost
import pyceres

def skew_symmetric(v):
    # v: (3,)
    return torch.stack([
        torch.stack([torch.zeros_like(v[0]), -v[2], v[1]]),
        torch.stack([v[2], torch.zeros_like(v[0]), -v[0]]),
        torch.stack([-v[1], v[0], torch.zeros_like(v[0])])
    ])


class TestReprojErrorCost(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a simple camera matrix for testing
        self.K = np.array([
            [1000.0, 0.0, 320.0],
            [0.0, 1000.0, 240.0],
            [0.0, 0.0, 1.0]
        ])
        self.camera_model = CameraModel(self.K)
        
        # Test data
        self.x_2d = torch.tensor([320.0, 240.0], dtype=torch.float32)  # Principal point
        self.pose_parameters = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Identity pose
        self.point_3d_parameters = np.array([0.0, 0.0, 2.0], dtype=np.float32)  # Point at depth 2
    
    def test_reproj_error_cost_initialization(self):
        """Test ReprojErrorCost initialization."""
        cost = ReprojErrorCost(self.x_2d, self.camera_model)
        self.assertEqual(cost.x_2d.shape, (2,))
        self.assertIsInstance(cost.camera_model, CameraModel)
    
    def test_evaluate_identity_pose(self):
        """Test evaluation with identity pose (should have zero residual)."""
        cost = ReprojErrorCost(self.x_2d, self.camera_model)
        
        # Point at [0, 0, 2] with identity pose should project to principal point
        residuals = np.zeros(2, dtype=np.float32)
        jacobians = [np.zeros((2, 6), dtype=np.float32), np.zeros((2, 3), dtype=np.float32)]
        
        success = cost.Evaluate(self.pose_parameters, self.point_3d_parameters, residuals, jacobians)
        
        self.assertTrue(success)
        # Residual should be very small (close to zero)
        np.testing.assert_array_almost_equal(residuals, np.zeros(2), decimal=6)
    
    def test_evaluate_with_translation(self):
        """Test evaluation with translation pose."""
        cost = ReprojErrorCost(self.x_2d, self.camera_model)
        
        # Pose with translation [0.1, 0.1, 0]
        pose_with_translation = np.array([0.1, 0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        residuals = np.zeros(2, dtype=np.float32)
        jacobians = [np.zeros((2, 6), dtype=np.float32), np.zeros((2, 3), dtype=np.float32)]
        
        success = cost.Evaluate(pose_with_translation, self.point_3d_parameters, residuals, jacobians)
        
        self.assertTrue(success)
        # Should have some residual due to translation
        self.assertTrue(np.any(np.abs(residuals) > 1e-6))
    
    def test_evaluate_with_rotation(self):
        """Test evaluation with rotation pose."""
        cost = ReprojErrorCost(self.x_2d, self.camera_model)
        
        # Pose with small rotation around Z-axis
        pose_with_rotation = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1], dtype=np.float32)
        # Use a point off the principal axis
        point_3d_parameters = np.array([1.0, 0.0, 2.0], dtype=np.float32)
        residuals = np.zeros(2, dtype=np.float32)
        jacobians = [np.zeros((2, 6), dtype=np.float32), np.zeros((2, 3), dtype=np.float32)]
        
        success = cost.Evaluate(pose_with_rotation, point_3d_parameters, residuals, jacobians)
        
        self.assertTrue(success)
        # Should have some residual due to rotation
        self.assertTrue(np.any(np.abs(residuals) > 1e-6))
    
    def test_jacobian_computation(self):
        """Test that Jacobians are computed correctly."""
        cost = ReprojErrorCost(self.x_2d, self.camera_model)
        
        residuals = np.zeros(2, dtype=np.float32)
        jacobians = [np.zeros((2, 6), dtype=np.float32), np.zeros((2, 3), dtype=np.float32)]
        
        success = cost.Evaluate(self.pose_parameters, self.point_3d_parameters, residuals, jacobians)
        
        self.assertTrue(success)
        
        # Check Jacobian shapes
        self.assertEqual(jacobians[0].shape, (2, 6))  # Pose Jacobian
        self.assertEqual(jacobians[1].shape, (2, 3))  # Point Jacobian
        
        # Check that Jacobians are not all zero
        self.assertTrue(np.any(np.abs(jacobians[0]) > 1e-6))
        self.assertTrue(np.any(np.abs(jacobians[1]) > 1e-6))
    
    def test_jacobian_numerical_check(self):
        """Test Jacobians using finite differences."""
        cost = ReprojErrorCost(self.x_2d, self.camera_model)
        
        # Base evaluation
        residuals_base = np.zeros(2, dtype=np.float32)
        jacobians_base = [np.zeros((2, 6), dtype=np.float32), np.zeros((2, 3), dtype=np.float32)]
        cost.Evaluate(self.pose_parameters, self.point_3d_parameters, residuals_base, jacobians_base)
        
        # Numerical Jacobian for pose parameters
        eps = 1e-6
        num_jacobian_pose = np.zeros((2, 6), dtype=np.float32)
        
        for i in range(6):
            pose_plus = self.pose_parameters.copy()
            pose_minus = self.pose_parameters.copy()
            pose_plus[i] += eps
            pose_minus[i] -= eps
            
            residuals_plus = np.zeros(2, dtype=np.float32)
            residuals_minus = np.zeros(2, dtype=np.float32)
            jacobians_dummy = [None, None]
            
            cost.Evaluate(pose_plus, self.point_3d_parameters, residuals_plus, jacobians_dummy)
            cost.Evaluate(pose_minus, self.point_3d_parameters, residuals_minus, jacobians_dummy)
            
            num_jacobian_pose[:, i] = (residuals_plus - residuals_minus) / (2 * eps)
        
        # Compare analytical and numerical Jacobians for pose
        np.testing.assert_allclose(jacobians_base[0], num_jacobian_pose, rtol=1e-2, atol=1e-4)
    
    def test_edge_cases(self):
        """Test edge cases."""
        cost = ReprojErrorCost(self.x_2d, self.camera_model)
        
        # Test with very small depth
        point_small_depth = np.array([0.0, 0.0, 0.001], dtype=np.float32)
        residuals = np.zeros(2, dtype=np.float32)
        jacobians = [np.zeros((2, 6), dtype=np.float32), np.zeros((2, 3), dtype=np.float32)]
        
        success = cost.Evaluate(self.pose_parameters, point_small_depth, residuals, jacobians)
        self.assertTrue(success)
        
        # Test with large translation
        pose_large_translation = np.array([10.0, 10.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        residuals = np.zeros(2, dtype=np.float32)
        jacobians = [np.zeros((2, 6), dtype=np.float32), np.zeros((2, 3), dtype=np.float32)]
        
        success = cost.Evaluate(pose_large_translation, self.point_3d_parameters, residuals, jacobians)
        self.assertTrue(success)


if __name__ == '__main__':
    unittest.main() 