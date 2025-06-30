#!/usr/bin/env python3
"""
Comprehensive unit tests for Bundle Adjustment functionality.

This module tests the complete bundle adjustment pipeline including:
- Transform functions
- Camera models
- Cost functions
- Bundle adjustment optimization
- Integration tests
"""

import numpy as np
import torch
import unittest
from typing import List, Tuple
import sys
import os
import pyceres

# Add the current directory to the path to import bundle_adjustment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bundle_adjustment import (
    CameraModel, 
    transform_tensor, 
    ReprojErrorCost, 
    BundleAdjuster,
    skew_symmetric
)


class TestTransformFunctions(unittest.TestCase):
    """Test transform functions used in bundle adjustment."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.dtype = torch.float32
    
    def test_skew_symmetric(self):
        """Test skew symmetric matrix creation."""
        v = torch.tensor([1.0, 2.0, 3.0], dtype=self.dtype)
        S = skew_symmetric(v)
        
        # Check shape
        self.assertEqual(S.shape, (3, 3))
        
        # Check that S is skew-symmetric (S^T = -S)
        S_transpose = S.T
        np.testing.assert_array_almost_equal(S.detach().numpy(), -S_transpose.detach().numpy())
        
        # Check specific values
        expected = torch.tensor([
            [0.0, -3.0, 2.0],
            [3.0, 0.0, -1.0],
            [-2.0, 1.0, 0.0]
        ], dtype=self.dtype)
        np.testing.assert_array_almost_equal(S.detach().numpy(), expected.detach().numpy())
    
    def test_transform_tensor_identity_pose(self):
        """Test transform with identity pose (no transformation)."""
        # Identity pose: [t_x, t_y, t_z, w_x, w_y, w_z] = [0, 0, 0, 0, 0, 0]
        pose = torch.zeros(6, dtype=self.dtype, requires_grad=True)
        point = torch.tensor([1.0, 2.0, 3.0], dtype=self.dtype, requires_grad=True)
        
        point_camera, (jacobian_pose, jacobian_point) = transform_tensor(pose, point)
        
        # With identity pose, point should remain the same
        np.testing.assert_array_almost_equal(point_camera.detach().numpy(), point.detach().numpy())
        
        # Check Jacobian shapes
        self.assertEqual(jacobian_pose.shape, (3, 6))
        self.assertEqual(jacobian_point.shape, (3, 3))
    
    def test_transform_tensor_translation_only(self):
        """Test transform with translation only."""
        # Translation [1, 2, 3], no rotation
        pose = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], dtype=self.dtype, requires_grad=True)
        point = torch.tensor([4.0, 5.0, 6.0], dtype=self.dtype, requires_grad=True)
        
        point_camera, (jacobian_pose, jacobian_point) = transform_tensor(pose, point)
        
        # Expected: point_camera = point - translation
        expected = point - pose[:3]
        np.testing.assert_array_almost_equal(point_camera.detach().numpy(), expected.detach().numpy())
    
    def test_transform_tensor_rotation_only(self):
        """Test transform with rotation only."""
        # Small rotation around Z-axis
        pose = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.1], dtype=self.dtype, requires_grad=True)
        point = torch.tensor([1.0, 0.0, 0.0], dtype=self.dtype, requires_grad=True)
        
        point_camera, (jacobian_pose, jacobian_point) = transform_tensor(pose, point)
        
        # With small rotation around Z, point should be slightly rotated
        # For small angle θ, rotation around Z: [cos(θ), sin(θ), 0; -sin(θ), cos(θ), 0; 0, 0, 1]
        # For θ = 0.1, cos(θ) ≈ 0.995, sin(θ) ≈ 0.1
        expected_x = 0.995  # cos(0.1)
        expected_y = -0.1   # -sin(0.1)
        expected_z = 0.0
        
        np.testing.assert_array_almost_equal(
            point_camera.detach().numpy(), 
            [expected_x, expected_y, expected_z], 
            decimal=3
        )


class TestCameraModel(unittest.TestCase):
    """Test camera model functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.K = np.array([
            [1000.0, 0.0, 320.0],
            [0.0, 1000.0, 240.0],
            [0.0, 0.0, 1.0]
        ])
        self.camera_model = CameraModel(self.K)
    
    def test_camera_model_initialization(self):
        """Test camera model initialization."""
        self.assertEqual(self.camera_model.fx, 1000.0)
        self.assertEqual(self.camera_model.fy, 1000.0)
        self.assertEqual(self.camera_model.cx, 320.0)
        self.assertEqual(self.camera_model.cy, 240.0)
        np.testing.assert_array_equal(self.camera_model.K, self.K)
    
    def test_project_numpy(self):
        """Test numpy-based projection."""
        points_3d = np.array([
            [0.0, 0.0, 1.0],
            [0.1, 0.1, 2.0],
            [-0.2, 0.3, 3.0]
        ])
        
        points_2d = self.camera_model.project(points_3d)
        
        expected_2d = np.array([
            [320.0, 240.0],
            [370.0, 290.0],
            [253.333333, 340.0]
        ])
        
        np.testing.assert_array_almost_equal(points_2d, expected_2d, decimal=6)
    
    def test_project_tensor(self):
        """Test tensor-based projection with gradients for a single point."""
        point_3d = torch.tensor([0.1, 0.1, 2.0], dtype=torch.float32, requires_grad=True)
        points_2d, jacobian = self.camera_model.project_tensor(point_3d)
        self.assertEqual(points_2d.shape, (2,))
        self.assertEqual(jacobian.shape, (2, 3))
        # Test that projection matches numpy version
        points_2d_np = self.camera_model.project(point_3d.detach().numpy().reshape(1, 3))[0]
        np.testing.assert_array_almost_equal(points_2d.detach().numpy(), points_2d_np, decimal=6)


class TestReprojErrorCost(unittest.TestCase):
    """Test reprojection error cost function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.K = np.array([
            [1000.0, 0.0, 320.0],
            [0.0, 1000.0, 240.0],
            [0.0, 0.0, 1.0]
        ])
        self.camera_model = CameraModel(self.K)
        self.x_2d = torch.tensor([320.0, 240.0], dtype=torch.float32)
        self.cost = ReprojErrorCost(self.x_2d, self.camera_model)
    
    def test_cost_function_initialization(self):
        """Test cost function initialization."""
        self.assertEqual(self.cost.x_2d.shape, (2,))
        self.assertIsInstance(self.cost.camera_model, CameraModel)
    
    def test_evaluate_identity_pose(self):
        """Test evaluation with identity pose."""
        pose_parameters = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        point_3d_parameters = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        residuals = np.zeros(2, dtype=np.float32)
        jacobians = [np.zeros(12, dtype=np.float32), np.zeros(6, dtype=np.float32)]
        
        success = self.cost.Evaluate([pose_parameters, point_3d_parameters], residuals, jacobians)
        
        self.assertTrue(success)
        # Residual should be very small for identity pose
        np.testing.assert_array_almost_equal(residuals, np.zeros(2), decimal=6)
    
    def test_evaluate_with_translation(self):
        """Test evaluation with translation."""
        pose_parameters = np.array([0.1, 0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        point_3d_parameters = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        residuals = np.zeros(2, dtype=np.float32)
        jacobians = [np.zeros(12, dtype=np.float32), np.zeros(6, dtype=np.float32)]
        
        success = self.cost.Evaluate([pose_parameters, point_3d_parameters], residuals, jacobians)
        
        self.assertTrue(success)
        # Should have non-zero residual due to translation
        self.assertTrue(np.any(np.abs(residuals) > 1e-6))
    
    def test_evaluate_with_rotation(self):
        """Test evaluation with rotation."""
        pose_parameters = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1], dtype=np.float32)
        point_3d_parameters = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        residuals = np.zeros(2, dtype=np.float32)
        jacobians = [np.zeros(12, dtype=np.float32), np.zeros(6, dtype=np.float32)]
        
        success = self.cost.Evaluate([pose_parameters, point_3d_parameters], residuals, jacobians)
        
        self.assertTrue(success)
        # Should have non-zero residual due to rotation
        self.assertTrue(np.any(np.abs(residuals) > 1e-6))
    
    def test_jacobian_computation(self):
        """Test that Jacobians are computed correctly."""
        pose_parameters = np.array([0.1, 0.2, 0.3, 0.01, 0.02, 0.03], dtype=np.float32)
        point_3d_parameters = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        residuals = np.zeros(2, dtype=np.float32)
        jacobians = [np.zeros(12, dtype=np.float32), np.zeros(6, dtype=np.float32)]
        
        success = self.cost.Evaluate([pose_parameters, point_3d_parameters], residuals, jacobians)
        
        self.assertTrue(success)
        self.assertEqual(jacobians[0].shape, (12,))
        self.assertEqual(jacobians[1].shape, (6,))
        
        # Check that Jacobians are not all zero
        self.assertTrue(np.any(np.abs(jacobians[0]) > 1e-6))
        self.assertTrue(np.any(np.abs(jacobians[1]) > 1e-6))


class TestBundleAdjuster(unittest.TestCase):
    """Test BundleAdjuster class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ba = BundleAdjuster(fix_first_pose=True, fix_intrinsics=True)
        
        # Create test data
        self.points_3d = np.array([
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 2.0],
            [1.0, 1.0, 2.0]
        ], dtype=np.float32)
        
        self.camera_poses = [
            np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ], dtype=np.float32),
            np.array([
                [1.0, 0.0, 0.0, 0.1],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ], dtype=np.float32)
        ]
        
        self.intrinsics = np.array([
            [1000.0, 0.0, 320.0],
            [0.0, 1000.0, 240.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        # Create observations: (point_idx, frame_idx, point_2d)
        self.observations = [
            (0, 0, np.array([320.0, 240.0])),
            (1, 0, np.array([370.0, 240.0])),
            (2, 0, np.array([320.0, 290.0])),
            (3, 0, np.array([370.0, 290.0])),
            (0, 1, np.array([320.0, 240.0])),
            (1, 1, np.array([370.0, 240.0])),
            (2, 1, np.array([320.0, 290.0])),
            (3, 1, np.array([370.0, 290.0]))
        ]
    
    def test_bundle_adjuster_initialization(self):
        """Test BundleAdjuster initialization."""
        ba = BundleAdjuster(fix_first_pose=True, fix_intrinsics=True)
        self.assertTrue(ba.fix_first_pose)
        self.assertTrue(ba.fix_intrinsics)
        
        ba = BundleAdjuster(fix_first_pose=False, fix_intrinsics=False)
        self.assertFalse(ba.fix_first_pose)
        self.assertFalse(ba.fix_intrinsics)
    
    def test_create_reconstruction_from_data(self):
        """Test reconstruction creation from data."""
        rec = self.ba.create_reconstruction_from_data(
            self.points_3d, self.observations, self.camera_poses, self.intrinsics
        )
        
        # Check that reconstruction was created
        self.assertIsNotNone(rec)
        self.assertIn('points_3d', rec)
        self.assertIn('camera_poses', rec)
        self.assertIn('intrinsics', rec)
        self.assertIn('observations', rec)
        self.assertEqual(len(rec['points_3d']), 4)
        self.assertEqual(len(rec['camera_poses']), 2)
    
    def test_define_problem(self):
        """Test problem definition."""
        rec = self.ba.create_reconstruction_from_data(
            self.points_3d, self.observations, self.camera_poses, self.intrinsics
        )
        
        problem = self.ba.define_problem(rec)
        
        # Check that problem was created
        self.assertIsNotNone(problem)
        self.assertIsInstance(problem, pyceres.Problem)


class TestBundleAdjustmentIntegration(unittest.TestCase):
    """Integration tests for complete bundle adjustment pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ba = BundleAdjuster(fix_first_pose=True, fix_intrinsics=True)
        
        # Create synthetic data
        self.points_3d = np.array([
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 2.0],
            [1.0, 1.0, 2.0]
        ], dtype=np.float32)
        
        self.camera_poses = [
            np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ], dtype=np.float32),
            np.array([
                [1.0, 0.0, 0.0, 0.1],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ], dtype=np.float32)
        ]
        
        self.intrinsics = np.array([
            [1000.0, 0.0, 320.0],
            [0.0, 1000.0, 240.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        # Create observations
        self.observations = [
            (0, 0, np.array([320.0, 240.0])),
            (1, 0, np.array([370.0, 240.0])),
            (2, 0, np.array([320.0, 290.0])),
            (3, 0, np.array([370.0, 290.0])),
            (0, 1, np.array([320.0, 240.0])),
            (1, 1, np.array([370.0, 240.0])),
            (2, 1, np.array([320.0, 290.0])),
            (3, 1, np.array([370.0, 290.0]))
        ]
    
    def test_complete_bundle_adjustment_pipeline(self):
        """Test complete bundle adjustment pipeline."""
        # Add some noise to initial estimates
        noisy_points_3d = self.points_3d + 0.01 * np.random.randn(*self.points_3d.shape)
        noisy_camera_poses = []
        for pose in self.camera_poses:
            noisy_pose = pose.copy()
            noisy_pose[:3, 3] += 0.01 * np.random.randn(3)  # Add noise to translation
            noisy_camera_poses.append(noisy_pose)
        
        # Run bundle adjustment
        optimized_poses, optimized_points, final_error = self.ba.run(
            noisy_points_3d, self.observations, noisy_camera_poses, self.intrinsics
        )
        
        # Check that optimization completed
        self.assertIsNotNone(optimized_poses)
        self.assertIsNotNone(optimized_points)
        self.assertIsInstance(final_error, float)
        self.assertGreaterEqual(final_error, 0.0)
        
        # Check that we have the expected number of poses and points
        self.assertEqual(len(optimized_poses), len(self.camera_poses))
        self.assertEqual(len(optimized_points), len(self.points_3d))
    
    def test_bundle_adjustment_with_noise(self):
        """Test bundle adjustment with noisy observations."""
        # Add noise to observations
        noisy_observations = []
        for point_idx, frame_idx, point_2d in self.observations:
            noisy_point_2d = point_2d + 2.0 * np.random.randn(2)  # 2 pixel noise
            noisy_observations.append((point_idx, frame_idx, noisy_point_2d))
        
        # Run bundle adjustment
        optimized_poses, optimized_points, final_error = self.ba.run(
            self.points_3d, noisy_observations, self.camera_poses, self.intrinsics
        )
        
        # Check that optimization completed
        self.assertIsNotNone(optimized_poses)
        self.assertIsNotNone(optimized_points)
        self.assertIsInstance(final_error, float)
        self.assertGreaterEqual(final_error, 0.0)
        print(f"Final error: {final_error}")



def run_all_tests():
    """Run all bundle adjustment tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestTransformFunctions,
        TestCameraModel,
        TestReprojErrorCost,
        TestBundleAdjuster,
        TestBundleAdjustmentIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
