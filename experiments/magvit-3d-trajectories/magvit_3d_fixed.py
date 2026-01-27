#!/usr/bin/env python3
"""
MAGVIT 3D Trajectory Generator - TDD Implementation (REFACTORED)

Implementation to pass tests in test_magvit_3d_fixed.py
Following RED → GREEN → REFACTOR cycle

REFACTOR improvements:
- Added comprehensive type hints
- Improved documentation
- Added constants for magic numbers
- Extracted validation helper
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import Optional, Tuple

# ==============================================================================
# CONSTANTS
# ==============================================================================

DEFAULT_SIGMA = 1.5
DEFAULT_SEQ_LENGTH = 16
DEFAULT_FOCAL_LENGTH = 600
DEFAULT_IMG_SIZE = (480, 640)  # (height, width)
CAMERA_DEPTH_THRESHOLD = 0.1  # Minimum Y distance for valid projection


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def _validate_trajectory_shape(trajectory: np.ndarray) -> None:
    """
    Validate trajectory has correct shape (N, 3).
    
    Args:
        trajectory: Trajectory array to validate
        
    Raises:
        ValueError: If shape is invalid
    """
    if trajectory.ndim != 2 or trajectory.shape[1] != 3:
        raise ValueError(f"Trajectory must be (N, 3), got {trajectory.shape}")


# ==============================================================================
# CORE FUNCTIONS
# ==============================================================================

def smooth_trajectory(
    trajectory: np.ndarray, 
    sigma: float = DEFAULT_SIGMA
) -> np.ndarray:
    """
    Apply Gaussian smoothing to trajectory.
    
    Smooths each spatial dimension independently using a Gaussian filter.
    Reduces high-frequency noise while preserving overall trajectory shape.
    
    Args:
        trajectory: Input trajectory (N, 3) with [x, y, z] coordinates
        sigma: Gaussian smoothing parameter (higher = more smoothing)
        
    Returns:
        Smoothed trajectory (N, 3) with same shape as input
        
    Raises:
        ValueError: If trajectory shape is invalid
        
    Example:
        >>> traj = np.array([[0, 1, 2], [0.1, 1.1, 2.1], [0.2, 1.2, 2.2]])
        >>> smoothed = smooth_trajectory(traj, sigma=1.0)
    """
    _validate_trajectory_shape(trajectory)
    
    smoothed = np.zeros_like(trajectory)
    for dim in range(3):  # x, y, z
        smoothed[:, dim] = gaussian_filter1d(trajectory[:, dim], sigma=sigma)
    
    return smoothed


def generate_smooth_linear(seq_length: int = DEFAULT_SEQ_LENGTH) -> np.ndarray:
    """
    Generate smooth linear trajectory.
    
    Creates a straight-line path from start to end point.
    Trajectory moves diagonally forward and upward.
    
    Args:
        seq_length: Number of frames in trajectory
        
    Returns:
        Trajectory array (seq_length, 3)
        
    Trajectory characteristics:
        - Start: [0.0, 1.2, 2.5] (left, forward, up)
        - End: [0.6, 2.0, 2.7] (right, more forward, higher)
        - Motion: Diagonal forward and upward
    """
    start = np.array([0.0, 1.2, 2.5])
    end = np.array([0.6, 2.0, 2.7])
    t = np.linspace(0, 1, seq_length)
    trajectory = start[None, :] + t[:, None] * (end - start)[None, :]
    return trajectory


def generate_smooth_circular(seq_length: int = DEFAULT_SEQ_LENGTH) -> np.ndarray:
    """
    Generate smooth circular trajectory in XZ plane.
    
    Creates a circular path at constant depth (Y).
    Circle is in the XZ plane (lateral and vertical motion).
    
    Args:
        seq_length: Number of frames in trajectory
        
    Returns:
        Smoothed trajectory array (seq_length, 3)
        
    Trajectory characteristics:
        - Center: [0, 1.7, 2.55]
        - Radius: 0.35m
        - Plane: XZ (constant Y)
        - Smoothing: sigma=0.8
    """
    t = np.linspace(0, 2 * np.pi, seq_length + 1)[:-1]  # Avoid duplicate endpoint
    radius = 0.35
    center_y = 1.7
    center_z = 2.55
    
    x = radius * np.cos(t)
    y = np.ones(seq_length) * center_y
    z = center_z + radius * np.sin(t)
    
    trajectory = np.stack([x, y, z], axis=1)
    return smooth_trajectory(trajectory, sigma=0.8)


def generate_smooth_helical(seq_length: int = DEFAULT_SEQ_LENGTH) -> np.ndarray:
    """
    Generate smooth helical trajectory.
    
    Creates a spiral path that moves forward while rotating.
    Combines circular motion in XZ with linear motion in Y.
    
    Args:
        seq_length: Number of frames in trajectory
        
    Returns:
        Smoothed trajectory array (seq_length, 3)
        
    Trajectory characteristics:
        - Radius: 0.25m
        - Y range: 1.3 to 2.1 (moves forward)
        - Z oscillation: ±0.2m
        - Smoothing: sigma=1.2
    """
    t = np.linspace(0, 2 * np.pi, seq_length)
    radius = 0.25
    
    x = radius * np.cos(t)
    y = 1.3 + t / (2 * np.pi) * 0.8
    z = 2.55 + 0.2 * np.sin(t)
    
    trajectory = np.stack([x, y, z], axis=1)
    return smooth_trajectory(trajectory, sigma=1.2)


def project_3d_to_2d(
    point_3d: np.ndarray, 
    camera_pos: np.ndarray, 
    focal_length: float = DEFAULT_FOCAL_LENGTH, 
    img_size: Tuple[int, int] = DEFAULT_IMG_SIZE
) -> Optional[np.ndarray]:
    """
    Project 3D point to 2D using pinhole camera model.
    
    Implements standard pinhole projection with the coordinate system defined
    in COORDINATE_SYSTEM_DOCUMENTATION.md.
    
    Camera coordinate system:
        - Origin: Camera optical center
        - +X: Right (image columns increase)
        - +Y: Forward (depth, into the scene)
        - +Z: Up (image rows increase downward)
    
    Projection formula:
        x_image = focal_length * X_cam / Y_cam + center_x
        y_image = focal_length * Z_cam / Y_cam + center_y
    
    Args:
        point_3d: 3D point in world coordinates [x, y, z]
        camera_pos: Camera position in world coordinates [x, y, z]
        focal_length: Camera focal length in pixels
        img_size: Image size as (height, width) in pixels
        
    Returns:
        2D point [x, y] in image coordinates (pixels), or None if behind camera
        
    Example:
        >>> cam = np.array([0, 0, 2.5])
        >>> point = np.array([0.3, 1.5, 2.8])
        >>> projected = project_3d_to_2d(point, cam)
        >>> print(projected)  # [440.0, 360.0]
    """
    # Transform to camera frame
    point_cam = point_3d - camera_pos
    
    # Behind camera check (Y is depth)
    if point_cam[1] <= CAMERA_DEPTH_THRESHOLD:
        return None
    
    # Pinhole projection
    depth = point_cam[1]
    x_2d = focal_length * point_cam[0] / depth + img_size[1] / 2
    y_2d = focal_length * point_cam[2] / depth + img_size[0] / 2
    
    return np.array([x_2d, y_2d])
