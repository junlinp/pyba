import numpy as np



# quaternion: [x, y, z, w]

def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to quaternion.
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        Quaternion as [x, y, z, w]
    """
    # Ensure R is 3x3
    if R.shape != (3, 3):
        raise ValueError("R must be a 3x3 matrix")
    
    # Method from http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    trace = np.trace(R)
    
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2  # S = 4 * qw
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S = 4 * qx
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S = 4 * qy
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S = 4 * qz
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    
    return np.array([x, y, z, w])


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to rotation matrix.
    
    Args:
        q: Quaternion as [x, y, z, w]
        
    Returns:
        3x3 rotation matrix
    """
    # Ensure q is a 4-vector
    if len(q) != 4:
        raise ValueError("q must be a 4-vector [x, y, z, w]")
    
    x, y, z, w = q
    
    # Normalize quaternion
    norm = np.sqrt(x*x + y*y + z*z + w*w)
    if norm > 0:
        x, y, z, w = x/norm, y/norm, z/norm, w/norm
    
    # Convert to rotation matrix
    # R = [1-2y²-2z², 2xy-2zw, 2xz+2yw;
    #      2xy+2zw, 1-2x²-2z², 2yz-2xw;
    #      2xz-2yw, 2yz+2xw, 1-2x²-2y²]
    
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    
    return R


def quaternion_to_angle_axis(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to angle-axis representation.
    
    Args:
        q: Quaternion as [x, y, z, w]
        
    Returns:
        Angle-axis as [x, y, z] where magnitude is the angle in radians
    """
    # Ensure q is a 4-vector
    if len(q) != 4:
        raise ValueError("q must be a 4-vector [x, y, z, w]")
    
    x, y, z, w = q
    
    # Normalize quaternion
    norm = np.sqrt(x*x + y*y + z*z + w*w)
    if norm > 0:
        x, y, z, w = x/norm, y/norm, z/norm, w/norm
    
    # Handle the case where w = ±1 (no rotation)
    if abs(w) > 0.9999:
        return np.array([0.0, 0.0, 0.0])
    
    # Convert to angle-axis
    # angle = 2 * acos(w)
    # axis = [x, y, z] / sin(angle/2)
    angle = 2 * np.arccos(w)
    sin_half_angle = np.sqrt(1 - w*w)
    
    if sin_half_angle > 1e-6:
        axis = np.array([x, y, z]) / sin_half_angle
    else:
        axis = np.array([x, y, z])
    
    return axis * angle


def angle_axis_to_quaternion(angle_axis: np.ndarray) -> np.ndarray:
    """
    Convert angle-axis representation to quaternion.
    
    Args:
        angle_axis: Angle-axis as [x, y, z] where magnitude is the angle in radians
        
    Returns:
        Quaternion as [x, y, z, w]
    """
    # Ensure angle_axis is a 3-vector
    if len(angle_axis) != 3:
        raise ValueError("angle_axis must be a 3-vector [x, y, z]")
    
    angle = np.linalg.norm(angle_axis)
    
    # Handle zero rotation
    if angle < 1e-12:  # Use smaller threshold for better precision
        return np.array([0.0, 0.0, 0.0, 1.0])
    
    # Convert to quaternion
    # w = cos(angle/2)
    # [x, y, z] = sin(angle/2) * axis
    half_angle = angle / 2.0
    axis = angle_axis / angle
    
    w = np.cos(half_angle)
    xyz = np.sin(half_angle) * axis
    
    return np.array([xyz[0], xyz[1], xyz[2], w])

def rotation_matrix_to_angle_axis(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to angle-axis representation.
    """
    q = rotation_matrix_to_quaternion(R)
    return quaternion_to_angle_axis(q)

def angle_axis_to_rotation_matrix(angle_axis: np.ndarray) -> np.ndarray:
    """
    Convert angle-axis representation to rotation matrix.
    """
    q = angle_axis_to_quaternion(angle_axis)
    return quaternion_to_rotation_matrix(q)


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """
    Convert a vector to a skew-symmetric matrix.
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def so3_right_jacobian(phi):
    theta = np.linalg.norm(phi)
    I = np.eye(3)

    if theta < 1e-5:
        return I - 0.5 * skew_symmetric(phi) + (1.0 / 6.0) * skew_symmetric(phi) @ skew_symmetric(phi)
    else:
        K = skew_symmetric(phi)
        theta2 = theta**2
        theta3 = theta**3
        A = (1 - np.cos(theta)) / theta2
        B = (theta - np.sin(theta)) / theta3
        return I - A * K + B * K @ K