import numpy as np
import unittest
from rotation import (
    rotation_matrix_to_quaternion,
    quaternion_to_rotation_matrix,
    quaternion_to_angle_axis,
    angle_axis_to_quaternion,
    rotation_matrix_to_angle_axis,
    angle_axis_to_rotation_matrix,
    skew_symmetric,
    so3_right_jacobian
)


class TestRotationConversions(unittest.TestCase):
    """Test rotation conversion functions."""
    
    def test_identity_rotation(self):
        """Test identity rotation conversions."""
        # Identity rotation matrix
        R_identity = np.eye(3)
        
        # Convert to quaternion
        q = rotation_matrix_to_quaternion(R_identity)
        expected_q = np.array([0.0, 0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(q, expected_q)
        
        # Convert back to rotation matrix
        R_back = quaternion_to_rotation_matrix(q)
        np.testing.assert_array_almost_equal(R_back, R_identity)
        
        # Convert to angle-axis
        angle_axis = quaternion_to_angle_axis(q)
        expected_angle_axis = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(angle_axis, expected_angle_axis)
        
        # Convert back to quaternion
        q_back = angle_axis_to_quaternion(angle_axis)
        np.testing.assert_array_almost_equal(q_back, expected_q)
        
        # Test direct rotation matrix to angle-axis conversion
        angle_axis_direct = rotation_matrix_to_angle_axis(R_identity)
        np.testing.assert_array_almost_equal(angle_axis_direct, expected_angle_axis)
        
        # Test direct angle-axis to rotation matrix conversion
        R_direct = angle_axis_to_rotation_matrix(angle_axis_direct)
        np.testing.assert_array_almost_equal(R_direct, R_identity)
    
    def test_rotation_around_x_axis(self):
        """Test rotation around X-axis."""
        angle = np.pi / 4  # 45 degrees
        
        # Create rotation matrix around X-axis
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R_x = np.array([
            [1, 0, 0],
            [0, cos_a, -sin_a],
            [0, sin_a, cos_a]
        ])
        
        # Convert to quaternion
        q = rotation_matrix_to_quaternion(R_x)
        
        # Expected quaternion for rotation around X-axis
        expected_q = np.array([np.sin(angle/2), 0, 0, np.cos(angle/2)])
        np.testing.assert_array_almost_equal(q, expected_q)
        
        # Convert back to rotation matrix
        R_back = quaternion_to_rotation_matrix(q)
        np.testing.assert_array_almost_equal(R_back, R_x)
        
        # Convert to angle-axis
        angle_axis = quaternion_to_angle_axis(q)
        expected_angle_axis = np.array([angle, 0, 0])
        np.testing.assert_array_almost_equal(angle_axis, expected_angle_axis)
        
        # Convert back to quaternion
        q_back = angle_axis_to_quaternion(angle_axis)
        np.testing.assert_array_almost_equal(q_back, expected_q)
        
        # Test direct conversions
        angle_axis_direct = rotation_matrix_to_angle_axis(R_x)
        np.testing.assert_array_almost_equal(angle_axis_direct, expected_angle_axis)
        
        R_direct = angle_axis_to_rotation_matrix(angle_axis_direct)
        np.testing.assert_array_almost_equal(R_direct, R_x)
    
    def test_rotation_around_y_axis(self):
        """Test rotation around Y-axis."""
        angle = np.pi / 3  # 60 degrees
        
        # Create rotation matrix around Y-axis
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R_y = np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ])
        
        # Convert to quaternion
        q = rotation_matrix_to_quaternion(R_y)
        
        # Expected quaternion for rotation around Y-axis
        expected_q = np.array([0, np.sin(angle/2), 0, np.cos(angle/2)])
        np.testing.assert_array_almost_equal(q, expected_q)
        
        # Convert back to rotation matrix
        R_back = quaternion_to_rotation_matrix(q)
        np.testing.assert_array_almost_equal(R_back, R_y)
        
        # Test direct conversions
        angle_axis_direct = rotation_matrix_to_angle_axis(R_y)
        expected_angle_axis = np.array([0, angle, 0])
        np.testing.assert_array_almost_equal(angle_axis_direct, expected_angle_axis)
        
        R_direct = angle_axis_to_rotation_matrix(angle_axis_direct)
        np.testing.assert_array_almost_equal(R_direct, R_y)
    
    def test_rotation_around_z_axis(self):
        """Test rotation around Z-axis."""
        angle = np.pi / 2  # 90 degrees
        
        # Create rotation matrix around Z-axis
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R_z = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        
        # Convert to quaternion
        q = rotation_matrix_to_quaternion(R_z)
        
        # Expected quaternion for rotation around Z-axis
        expected_q = np.array([0, 0, np.sin(angle/2), np.cos(angle/2)])
        np.testing.assert_array_almost_equal(q, expected_q)
        
        # Convert back to rotation matrix
        R_back = quaternion_to_rotation_matrix(q)
        np.testing.assert_array_almost_equal(R_back, R_z)
        
        # Test direct conversions
        angle_axis_direct = rotation_matrix_to_angle_axis(R_z)
        expected_angle_axis = np.array([0, 0, angle])
        np.testing.assert_array_almost_equal(angle_axis_direct, expected_angle_axis)
        
        R_direct = angle_axis_to_rotation_matrix(angle_axis_direct)
        np.testing.assert_array_almost_equal(R_direct, R_z)
    
    def test_arbitrary_rotation(self):
        """Test arbitrary rotation."""
        # Create a rotation matrix from Euler angles (ZYX convention)
        alpha, beta, gamma = np.pi/6, np.pi/4, np.pi/3  # 30, 45, 60 degrees
        
        # Rotation around Z
        cos_a, sin_a = np.cos(alpha), np.sin(alpha)
        R_z = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
        
        # Rotation around Y
        cos_b, sin_b = np.cos(beta), np.sin(beta)
        R_y = np.array([[cos_b, 0, sin_b], [0, 1, 0], [-sin_b, 0, cos_b]])
        
        # Rotation around X
        cos_g, sin_g = np.cos(gamma), np.sin(gamma)
        R_x = np.array([[1, 0, 0], [0, cos_g, -sin_g], [0, sin_g, cos_g]])
        
        # Combined rotation
        R = R_z @ R_y @ R_x
        
        # Convert to quaternion
        q = rotation_matrix_to_quaternion(R)
        
        # Convert back to rotation matrix
        R_back = quaternion_to_rotation_matrix(q)
        np.testing.assert_array_almost_equal(R_back, R, decimal=10)
        
        # Convert to angle-axis and back
        angle_axis = quaternion_to_angle_axis(q)
        q_back = angle_axis_to_quaternion(angle_axis)
        np.testing.assert_array_almost_equal(q_back, q, decimal=10)
        
        # Test direct conversions
        angle_axis_direct = rotation_matrix_to_angle_axis(R)
        R_direct = angle_axis_to_rotation_matrix(angle_axis_direct)
        np.testing.assert_array_almost_equal(R_direct, R, decimal=10)
    
    def test_quaternion_normalization(self):
        """Test that quaternions are properly normalized."""
        # Non-normalized quaternion
        q_unnorm = np.array([1.0, 2.0, 3.0, 4.0])
        
        # Convert to rotation matrix (should normalize internally)
        R = quaternion_to_rotation_matrix(q_unnorm)
        
        # Convert back to quaternion
        q_norm = rotation_matrix_to_quaternion(R)
        
        # Check that the result is normalized
        norm = np.linalg.norm(q_norm)
        self.assertAlmostEqual(norm, 1.0, places=10)
    
    def test_edge_cases(self):
        """Test edge cases and numerical stability."""
        # Very small rotation
        small_angle = 1e-8
        angle_axis_small = np.array([small_angle, 0, 0])
        q_small = angle_axis_to_quaternion(angle_axis_small)
        
        # Should be close to identity quaternion
        expected_q_small = np.array([small_angle/2, 0, 0, 1.0])
        np.testing.assert_array_almost_equal(q_small, expected_q_small, decimal=10)
        
        # Large rotation (should wrap around)
        large_angle = 2 * np.pi + np.pi/4
        angle_axis_large = np.array([large_angle, 0, 0])
        q_large = angle_axis_to_quaternion(angle_axis_large)
        
        # Should be equivalent to pi/4 rotation (account for sign ambiguity)
        expected_q_large = np.array([np.sin(np.pi/8), 0, 0, np.cos(np.pi/8)])
        # Check if quaternions are equal or negative of each other
        if np.dot(q_large, expected_q_large) < 0:
            expected_q_large = -expected_q_large
        np.testing.assert_array_almost_equal(q_large, expected_q_large, decimal=10)
        
        # Test very small angle-axis to rotation matrix
        R_small = angle_axis_to_rotation_matrix(angle_axis_small)
        # Use lower precision for very small rotations due to numerical issues
        np.testing.assert_array_almost_equal(R_small, np.eye(3), decimal=6)
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Invalid rotation matrix shape
        with self.assertRaises(ValueError):
            rotation_matrix_to_quaternion(np.eye(2))
        
        # Invalid quaternion length
        with self.assertRaises(ValueError):
            quaternion_to_rotation_matrix(np.array([1, 2, 3]))
        
        with self.assertRaises(ValueError):
            quaternion_to_angle_axis(np.array([1, 2, 3]))
        
        # Invalid angle-axis length
        with self.assertRaises(ValueError):
            angle_axis_to_quaternion(np.array([1, 2]))
        
        # Test direct conversion error handling
        with self.assertRaises(ValueError):
            rotation_matrix_to_angle_axis(np.eye(2))
        
        with self.assertRaises(ValueError):
            angle_axis_to_rotation_matrix(np.array([1, 2]))


class TestSkewSymmetric(unittest.TestCase):
    """Test skew-symmetric matrix function."""
    
    def test_skew_symmetric_basic(self):
        """Test basic skew-symmetric matrix creation."""
        v = np.array([1.0, 2.0, 3.0])
        S = skew_symmetric(v)
        
        expected_S = np.array([
            [0, -3, 2],
            [3, 0, -1],
            [-2, 1, 0]
        ])
        
        np.testing.assert_array_equal(S, expected_S)
    
    def test_skew_symmetric_properties(self):
        """Test properties of skew-symmetric matrices."""
        v = np.array([1.0, 2.0, 3.0])
        S = skew_symmetric(v)
        
        # Skew-symmetric matrix should be antisymmetric: S^T = -S
        np.testing.assert_array_almost_equal(S.T, -S)
        
        # Diagonal should be zero
        np.testing.assert_array_almost_equal(np.diag(S), np.zeros(3))
        
        # For any vector v, S @ v should be equivalent to v × v (cross product)
        # This should be zero for the same vector
        cross_product = np.cross(v, v)
        matrix_product = S @ v
        np.testing.assert_array_almost_equal(cross_product, matrix_product)
    
    def test_skew_symmetric_zero_vector(self):
        """Test skew-symmetric matrix for zero vector."""
        v = np.array([0.0, 0.0, 0.0])
        S = skew_symmetric(v)
        
        expected_S = np.zeros((3, 3))
        np.testing.assert_array_equal(S, expected_S)
    
    def test_skew_symmetric_unit_vectors(self):
        """Test skew-symmetric matrices for unit vectors."""
        # X-axis
        v_x = np.array([1.0, 0.0, 0.0])
        S_x = skew_symmetric(v_x)
        expected_S_x = np.array([
            [0, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        np.testing.assert_array_equal(S_x, expected_S_x)
        
        # Y-axis
        v_y = np.array([0.0, 1.0, 0.0])
        S_y = skew_symmetric(v_y)
        expected_S_y = np.array([
            [0, 0, 1],
            [0, 0, 0],
            [-1, 0, 0]
        ])
        np.testing.assert_array_equal(S_y, expected_S_y)
        
        # Z-axis
        v_z = np.array([0.0, 0.0, 1.0])
        S_z = skew_symmetric(v_z)
        expected_S_z = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 0]
        ])
        np.testing.assert_array_equal(S_z, expected_S_z)


class TestSO3RightJacobian(unittest.TestCase):
    """Test SO(3) right Jacobian function."""
    
    def test_so3_right_jacobian_zero(self):
        """Test SO(3) right Jacobian for zero rotation."""
        phi = np.array([0.0, 0.0, 0.0])
        J = so3_right_jacobian(phi)
        
        # For zero rotation, Jacobian should be identity
        np.testing.assert_array_almost_equal(J, np.eye(3))
    
    def test_so3_right_jacobian_small_angle(self):
        """Test SO(3) right Jacobian for small angles."""
        phi = np.array([1e-6, 0.0, 0.0])
        J = so3_right_jacobian(phi)
        
        # For small angles, should be close to identity minus skew-symmetric
        expected_J = np.eye(3) - 0.5 * skew_symmetric(phi) + (1.0/6.0) * skew_symmetric(phi) @ skew_symmetric(phi)
        np.testing.assert_array_almost_equal(J, expected_J, decimal=10)
    
    def test_so3_right_jacobian_axis_aligned(self):
        """Test SO(3) right Jacobian for axis-aligned rotations."""
        # X-axis rotation
        phi_x = np.array([np.pi/4, 0.0, 0.0])
        J_x = so3_right_jacobian(phi_x)
        
        # Check that J_x is invertible (determinant should be non-zero)
        det_J_x = np.linalg.det(J_x)
        self.assertGreater(abs(det_J_x), 1e-10)
        
        # Y-axis rotation
        phi_y = np.array([0.0, np.pi/3, 0.0])
        J_y = so3_right_jacobian(phi_y)
        
        det_J_y = np.linalg.det(J_y)
        self.assertGreater(abs(det_J_y), 1e-10)
        
        # Z-axis rotation
        phi_z = np.array([0.0, 0.0, np.pi/2])
        J_z = so3_right_jacobian(phi_z)
        
        det_J_z = np.linalg.det(J_z)
        self.assertGreater(abs(det_J_z), 1e-10)
    
    def test_so3_right_jacobian_properties(self):
        """Test properties of SO(3) right Jacobian."""
        phi = np.array([1.0, 2.0, 3.0])
        J = so3_right_jacobian(phi)
        
        # Jacobian should be invertible for non-zero rotations
        det_J = np.linalg.det(J)
        self.assertGreater(abs(det_J), 1e-10)
        
        # Test that J @ phi gives a reasonable result
        # This is related to the exponential map
        result = J @ phi
        self.assertEqual(result.shape, (3,))
    
    def test_so3_right_jacobian_arbitrary(self):
        """Test SO(3) right Jacobian for arbitrary rotation."""
        phi = np.array([1.5, -0.8, 2.1])
        J = so3_right_jacobian(phi)
        
        # Check basic properties
        self.assertEqual(J.shape, (3, 3))
        
        # Check that J is invertible
        det_J = np.linalg.det(J)
        self.assertGreater(abs(det_J), 1e-10)
        
        # Test continuity: small changes in phi should result in small changes in J
        phi_small = phi + np.array([1e-8, 0, 0])
        J_small = so3_right_jacobian(phi_small)
        
        diff = np.linalg.norm(J - J_small)
        self.assertLess(diff, 1e-6)


class TestRoundTripConversions(unittest.TestCase):
    """Test round-trip conversions between different representations."""
    
    def test_rotation_matrix_round_trip(self):
        """Test round-trip conversion: R -> angle_axis -> R."""
        # Create a rotation matrix
        angle = np.pi / 3
        axis = np.array([1.0, 1.0, 1.0])
        axis = axis / np.linalg.norm(axis)
        
        # Create rotation matrix using Rodrigues' formula
        K = skew_symmetric(axis)
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        
        # Round-trip conversion
        angle_axis = rotation_matrix_to_angle_axis(R)
        R_back = angle_axis_to_rotation_matrix(angle_axis)
        
        np.testing.assert_array_almost_equal(R_back, R, decimal=10)
    
    def test_quaternion_angle_axis_round_trip(self):
        """Test round-trip conversion: quaternion -> angle_axis -> quaternion."""
        # Create a quaternion
        q = np.array([0.5, 0.5, 0.5, 0.5])
        q = q / np.linalg.norm(q)  # normalize
        
        # Round-trip conversion
        angle_axis = quaternion_to_angle_axis(q)
        q_back = angle_axis_to_quaternion(angle_axis)
        
        # Quaternions can differ by sign, so check both possibilities
        if np.dot(q, q_back) < 0:
            q_back = -q_back
        np.testing.assert_array_almost_equal(q_back, q, decimal=10)
    
    def test_all_representations_round_trip(self):
        """Test round-trip conversion through all representations."""
        # Start with a rotation matrix (using exact values)
        angle = np.pi / 6  # 30 degrees
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        R = np.array([
            [cos_angle, -sin_angle, 0],
            [sin_angle, cos_angle, 0],
            [0, 0, 1]
        ])
        
        # R -> quaternion -> angle_axis -> quaternion -> R
        q1 = rotation_matrix_to_quaternion(R)
        angle_axis = quaternion_to_angle_axis(q1)
        q2 = angle_axis_to_quaternion(angle_axis)
        R_back = quaternion_to_rotation_matrix(q2)
        
        np.testing.assert_array_almost_equal(R_back, R, decimal=10)
        
        # Also test direct conversion
        angle_axis_direct = rotation_matrix_to_angle_axis(R)
        R_direct = angle_axis_to_rotation_matrix(angle_axis_direct)
        np.testing.assert_array_almost_equal(R_direct, R, decimal=10)


if __name__ == '__main__':
    unittest.main() 