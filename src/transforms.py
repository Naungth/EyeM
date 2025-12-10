"""
Spatial Transforms and Adjoint Operations

Utilities for transforming twists between frames (e.g., camera to end-effector).
Based on patterns from real-world IBVS implementation.
"""

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    """
    Compute skew-symmetric matrix from 3D vector.
    
    For vector v = [x, y, z], returns:
        [ 0  -z   y ]
        [ z   0  -x ]
        [-y   x   0 ]
    
    Args:
        v: 3D vector
    
    Returns:
        3x3 skew-symmetric matrix
    """
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def adjoint_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Compute adjoint transform matrix for transforming twists between frames.
    
    Given rotation R and translation t from frame A to frame B:
        Ad_B^A = [ R        [t]×R ]
                 [ 0            R ]
    
    This transforms a twist expressed at frame B's origin into frame A
    using the twist ordering [v; w] (linear velocity followed by angular).
    Translation only affects the linear component (shifting the reference
    point), not the angular component.
    
    Args:
        R: 3x3 rotation matrix
        t: 3D translation vector
    
    Returns:
        6x6 adjoint transform matrix
    """
    upper = np.hstack((R, skew(t) @ R))
    lower = np.hstack((np.zeros((3, 3)), R))
    return np.vstack((upper, lower))


def euler_rpy_to_rotation(rx_rad: float, ry_rad: float, rz_rad: float) -> np.ndarray:
    """
    Convert Euler angles (roll-pitch-yaw) to rotation matrix.
    
    Uses ZYX convention: R = Rz @ Ry @ Rx
    
    Args:
        rx_rad: Roll angle in radians (rotation about x-axis)
        ry_rad: Pitch angle in radians (rotation about y-axis)
        rz_rad: Yaw angle in radians (rotation about z-axis)
    
    Returns:
        3x3 rotation matrix
    """
    cx, sx = np.cos(rx_rad), np.sin(rx_rad)
    cy, sy = np.cos(ry_rad), np.sin(ry_rad)
    cz, sz = np.cos(rz_rad), np.sin(rz_rad)
    
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=float)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
    
    return Rz @ Ry @ Rx


def se3_exp_body(twist6: np.ndarray, dt: float) -> np.ndarray:
    """
    Integrate body-frame twist to SE(3) transformation using exponential map.
    
    Computes T = exp([v] dt) where [v] is the twist in se(3).
    
    Args:
        twist6: 6D twist [vx, vy, vz, wx, wy, wz] in body frame
        dt: Time step
    
    Returns:
        4x4 homogeneous transformation matrix
    """
    v = np.asarray(twist6[:3], dtype=float)  # Linear velocity
    w = np.asarray(twist6[3:], dtype=float)   # Angular velocity
    
    theta = np.linalg.norm(w) * dt
    
    if theta < 1e-9:
        # Small angle approximation
        R = np.eye(3) + skew(w) * dt
        t = v * dt
    else:
        # Full exponential map
        w_unit = w / np.linalg.norm(w)
        w_hat = skew(w_unit)
        
        # Rotation: Rodrigues' formula
        R = (
            np.eye(3)
            + np.sin(theta) * w_hat
            + (1.0 - np.cos(theta)) * (w_hat @ w_hat)
        )
        
        # Translation: V matrix
        I = np.eye(3)
        V = (
            I * dt
            + (1 - np.cos(theta)) / (np.linalg.norm(w) ** 2) * skew(w)
            + (theta - np.sin(theta)) / (np.linalg.norm(w) ** 3) * (skew(w) @ skew(w))
        )
        t = V @ v
    
    # Build 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    
    return T
