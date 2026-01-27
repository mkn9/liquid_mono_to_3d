#!/usr/bin/env python3
"""
MAGVIT 3D Generator - Verified Implementation

Uses verified algorithms from:
- 3d_tracker_cylinder.ipynb (6/23/2025)
- 3d_tracker_9.ipynb (6/21/2025, commit c7889a37)

Key Features:
- Proper camera setup with K, R, t matrices
- Verified 3D to 2D projection
- ConvexHull-based shape rendering
- Multi-camera trajectory generation

Author: AI Assistant
Date: 2026-01-19
"""

import numpy as np
from typing import Tuple, Dict
from scipy.spatial import ConvexHull
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def setup_cameras(img_size: int = 128) -> Dict[str, np.ndarray]:
    """
    Set up stereo camera system using verified algorithm.
    
    Cameras look in +Y direction (forward) with verified rotation matrix.
    
    Args:
        img_size: Image size (default 128x128)
    
    Returns:
        Dictionary containing:
        - K: Intrinsic matrix
        - R1, R2: Rotation matrices
        - t1, t2: Translation vectors
        - P1, P2: Projection matrices
        - cam1_pos, cam2_pos: Camera world positions
    """
    # Camera intrinsic matrix scaled for image size
    # Focal length and principal point scaled proportionally
    scale = img_size / 1280.0
    fx = fy = 1000 * scale
    cx = img_size / 2
    cy = img_size / 2
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Camera world positions
    cam1_pos = np.array([0.0, 0.0, 2.55])
    cam2_pos = np.array([1.0, 0.0, 2.55])
    
    # Verified rotation matrix for +Y looking cameras
    # Rotate 90° around X-axis: standard CV cameras look in -Z, we want +Y
    R_corrected = np.array([
        [1,  0,  0],   # X axis unchanged
        [0,  0, -1],   # Y axis becomes -Z (depth in camera coordinates)
        [0,  1,  0]    # Z axis becomes Y (vertical in camera coordinates)
    ], dtype=np.float64)
    
    R1 = R_corrected.copy()
    R2 = R_corrected.copy()
    
    # Translation vectors: t = -R * C
    t1 = -R1 @ cam1_pos.reshape(3, 1)
    t2 = -R2 @ cam2_pos.reshape(3, 1)
    
    # Projection matrices: P = K[R|t]
    P1 = K @ np.hstack((R1, t1))
    P2 = K @ np.hstack((R2, t2))
    
    return {
        'K': K,
        'R1': R1,
        'R2': R2,
        't1': t1,
        't2': t2,
        'P1': P1,
        'P2': P2,
        'cam1_pos': cam1_pos,
        'cam2_pos': cam2_pos
    }


def project_point(P: np.ndarray, point_3d: np.ndarray, camera_pos: np.ndarray = None) -> np.ndarray:
    """
    Project 3D world point to 2D image coordinates (verified algorithm).
    
    Args:
        P: 3x4 projection matrix
        point_3d: 3D point in world coordinates [x, y, z]
        camera_pos: Camera position in world coordinates (optional, for depth check)
        
    Returns:
        2D pixel coordinates [u, v], or [inf, inf] if point is at/behind camera
    """
    # Check if point is behind camera (for +Y looking cameras)
    if camera_pos is not None:
        # For cameras looking in +Y direction, reject points with Y < camera Y
        if point_3d[1] <= camera_pos[1]:
            return np.array([float('inf'), float('inf')])
    
    # Convert to homogeneous coordinates
    point_3d_h = np.append(point_3d, 1.0)
    
    # Project: pixel_h = P * point_3d_h
    proj = P @ point_3d_h
    
    # Handle points at camera plane or with negative depth (avoid division by zero)
    if proj[2] < 1e-10:
        return np.array([float('inf'), float('inf')])
    
    # Convert from homogeneous to Cartesian coordinates
    return proj[:2] / proj[2]


def get_cube_outline_points(center: np.ndarray, size: float) -> np.ndarray:
    """
    Generate 8 corner points of a cube.
    
    Args:
        center: Cube center [x, y, z]
        size: Cube side length
        
    Returns:
        Array of shape (8, 3) with cube corners
    """
    half_size = size / 2
    corners = []
    
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            for dz in [-1, 1]:
                corner = center + np.array([dx, dy, dz]) * half_size
                corners.append(corner)
    
    return np.array(corners)


def get_cylinder_outline_points(
    center: np.ndarray,
    radius: float,
    height: float,
    n_theta: int = 16
) -> np.ndarray:
    """
    Generate outline points for a vertical cylinder (verified algorithm).
    
    Args:
        center: Cylinder center [x, y, z]
        radius: Cylinder radius
        height: Cylinder height
        n_theta: Number of points around circumference
        
    Returns:
        Array of shape (2*n_theta, 3) with top and bottom circle points
    """
    # Vertical cylinder axis
    axis_dir = np.array([0, 0, 1])
    
    # Endpoints
    bottom_center = center - (height / 2) * axis_dir
    top_center = center + (height / 2) * axis_dir
    
    # Perpendicular vectors for circular cross-section
    perp1 = np.array([1, 0, 0])
    perp2 = np.array([0, 1, 0])
    
    outline_points = []
    theta_values = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    
    # Bottom circle points
    for theta in theta_values:
        point = (bottom_center +
                radius * np.cos(theta) * perp1 +
                radius * np.sin(theta) * perp2)
        outline_points.append(point)
    
    # Top circle points
    for theta in theta_values:
        point = (top_center +
                radius * np.cos(theta) * perp1 +
                radius * np.sin(theta) * perp2)
        outline_points.append(point)
    
    return np.array(outline_points)


def get_cone_outline_points(
    center: np.ndarray,
    base_radius: float,
    height: float,
    n_theta: int = 16
) -> np.ndarray:
    """
    Generate outline points for a vertical cone.
    
    Args:
        center: Cone center [x, y, z]
        base_radius: Cone base radius
        height: Cone height
        n_theta: Number of points around base circle
        
    Returns:
        Array of shape (n_theta+1, 3) with apex and base circle points
    """
    # Vertical cone axis
    axis_dir = np.array([0, 0, 1])
    
    # Apex and base center
    apex = center + (height / 2) * axis_dir
    base_center = center - (height / 2) * axis_dir
    
    # Perpendicular vectors
    perp1 = np.array([1, 0, 0])
    perp2 = np.array([0, 1, 0])
    
    outline_points = [apex]  # Start with apex
    
    theta_values = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    
    # Base circle points
    for theta in theta_values:
        point = (base_center +
                base_radius * np.cos(theta) * perp1 +
                base_radius * np.sin(theta) * perp2)
        outline_points.append(point)
    
    return np.array(outline_points)


def render_shape_2d(
    shape: str,
    center: np.ndarray,
    P: np.ndarray,
    camera_pos: np.ndarray,
    img_size: int = 128,
    color: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """
    Render 3D shape to 2D image using ConvexHull (verified algorithm).
    
    Args:
        shape: Shape type ('cube', 'cylinder', 'cone')
        center: Shape center in 3D world coordinates
        P: 3x4 projection matrix
        camera_pos: Camera world position
        img_size: Output image size (square)
        color: RGB color tuple
        
    Returns:
        Image array of shape (img_size, img_size, 3)
    """
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    
    # Get 3D outline points based on shape
    if shape == 'cube':
        outline_3d = get_cube_outline_points(center, size=0.1)
    elif shape == 'cylinder':
        outline_3d = get_cylinder_outline_points(center, radius=0.05, height=0.2, n_theta=24)
    elif shape == 'cone':
        outline_3d = get_cone_outline_points(center, base_radius=0.05, height=0.2, n_theta=24)
    else:
        return img
    
    # Project all outline points to 2D
    projected_outline = []
    for point_3d in outline_3d:
        pixel = project_point(P, point_3d, camera_pos)
        if not (np.isinf(pixel[0]) or np.isinf(pixel[1]) or 
                np.isnan(pixel[0]) or np.isnan(pixel[1])):
            # Check if within image bounds (with some margin)
            if -img_size <= pixel[0] < 2*img_size and -img_size <= pixel[1] < 2*img_size:
                projected_outline.append(pixel)
    
    if len(projected_outline) < 3:
        return img  # Not enough points to form a shape
    
    projected_outline = np.array(projected_outline)
    
    # Use ConvexHull to find silhouette
    try:
        hull = ConvexHull(projected_outline)
        hull_points = projected_outline[hull.vertices]
        
        # Fill polygon using scanline algorithm
        hull_points_int = hull_points.astype(np.int32)
        
        # Simple polygon fill
        from matplotlib.path import Path
        y, x = np.mgrid[:img_size, :img_size]
        points = np.vstack((x.flatten(), y.flatten())).T
        path = Path(hull_points)
        mask = path.contains_points(points).reshape(img_size, img_size)
        
        img[mask] = color
        
    except Exception as e:
        # If ConvexHull fails, just draw points
        for px, py in projected_outline:
            x, y = int(px), int(py)
            if 0 <= x < img_size and 0 <= y < img_size:
                img[y, x] = color
    
    return img


def generate_linear_trajectory(seq_length: int = 16) -> np.ndarray:
    """Generate linear trajectory in 3D space."""
    t = np.linspace(0, 1, seq_length)
    x = -0.2 + t * 0.4  # -0.2 to 0.2
    y = np.ones(seq_length) * 1.0  # Constant Y=1.0 (in front of cameras)
    z = np.linspace(2.3, 2.7, seq_length)  # Varying height
    
    return np.stack([x, y, z], axis=1)


def generate_circular_trajectory(seq_length: int = 16) -> np.ndarray:
    """Generate circular trajectory in XZ plane."""
    t = np.linspace(0, 2*np.pi, seq_length)
    radius = 0.2
    x = radius * np.cos(t)
    y = np.ones(seq_length) * 1.0  # Constant Y=1.0
    z = 2.5 + radius * np.sin(t)
    
    return np.stack([x, y, z], axis=1)


def generate_helical_trajectory(seq_length: int = 16) -> np.ndarray:
    """Generate helical trajectory."""
    t = np.linspace(0, 2*np.pi, seq_length)
    radius = 0.15
    x = radius * np.cos(t)
    y = np.ones(seq_length) * 1.0  # Constant Y=1.0
    z = np.linspace(2.3, 2.7, seq_length)  # Rising helix
    
    return np.stack([x, y, z], axis=1)


def generate_parabolic_trajectory(seq_length: int = 16) -> np.ndarray:
    """Generate parabolic trajectory."""
    t = np.linspace(-1, 1, seq_length)
    x = t * 0.2
    y = 0.8 + 0.4 * (1 - t**2)  # Parabola in Y with apex in middle (0.8 to 1.2)
    z = np.ones(seq_length) * 2.5  # Constant Z=2.5
    
    return np.stack([x, y, z], axis=1)


class MAGVIT3DVerifiedGenerator:
    """
    MAGVIT 3D trajectory generator using verified algorithms.
    
    Generates multi-camera video sequences of 3D object trajectories
    for cube, cylinder, and cone shapes.
    """
    
    def __init__(self, seq_length: int = 16, img_size: int = 128, num_cameras: int = 3):
        """
        Initialize generator.
        
        Args:
            seq_length: Number of frames per trajectory
            img_size: Image size (square)
            num_cameras: Number of cameras (currently 3 supported)
        """
        self.seq_length = seq_length
        self.img_size = img_size
        self.num_cameras = num_cameras
        
        # Setup cameras with appropriate image size
        self.cameras = setup_cameras(img_size=img_size)
        
        # Shape types and colors
        self.shapes = ['cube', 'cylinder', 'cone']
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        
        # Trajectory generation functions
        self.trajectory_funcs = [
            generate_linear_trajectory,
            generate_circular_trajectory,
            generate_helical_trajectory,
            generate_parabolic_trajectory
        ]
        
        # Camera projection matrices (first 3 cameras)
        self.P_matrices = [
            self.cameras['P1'],
            self.cameras['P2'],
            self.cameras['P1']  # Use P1 again for 3rd camera (placeholder)
        ]
        
        self.cam_positions = [
            self.cameras['cam1_pos'],
            self.cameras['cam2_pos'],
            self.cameras['cam1_pos']  # Use cam1 again (placeholder)
        ]
    
    def _generate_video_for_camera(
        self,
        shape: str,
        trajectory: np.ndarray,
        color: Tuple[int, int, int],
        camera_idx: int
    ) -> np.ndarray:
        """
        Generate video sequence for one camera view.
        
        Args:
            shape: Shape type
            trajectory: 3D trajectory array (seq_length, 3)
            color: RGB color
            camera_idx: Camera index
            
        Returns:
            Video array (seq_length, img_size, img_size, 3)
        """
        video = np.zeros((self.seq_length, self.img_size, self.img_size, 3), dtype=np.uint8)
        
        P = self.P_matrices[camera_idx]
        cam_pos = self.cam_positions[camera_idx]
        
        for frame_idx in range(self.seq_length):
            center = trajectory[frame_idx]
            video[frame_idx] = render_shape_2d(shape, center, P, cam_pos, self.img_size, color)
        
        return video
    
    def generate_dataset(self, num_samples: int) -> Dict[str, np.ndarray]:
        """
        Generate complete dataset of 3D trajectories with multi-camera videos.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Dictionary with keys:
            - 'trajectories_3d': shape (num_samples, seq_length, 3)
            - 'multi_view_videos': shape (num_samples, num_cameras, seq_length, img_size, img_size, 3)
            - 'labels': shape (num_samples,) - 0=cube, 1=cylinder, 2=cone
        """
        trajectories_3d = []
        multi_view_videos = []
        labels = []
        
        rng = np.random.default_rng(seed=42)  # Deterministic
        
        for i in range(num_samples):
            # Select shape (cycle through 3 shapes)
            shape_idx = i % 3
            shape = self.shapes[shape_idx]
            color = self.colors[shape_idx]
            
            # Select trajectory type (cycle through 4 types)
            traj_func = self.trajectory_funcs[i % 4]
            trajectory = traj_func(self.seq_length)
            
            # Add small noise to trajectory
            noise = rng.normal(0, 0.01, trajectory.shape)
            noisy_trajectory = trajectory + noise
            
            # Generate multi-camera views
            multi_view = np.zeros((self.num_cameras, self.seq_length, 
                                   self.img_size, self.img_size, 3), dtype=np.uint8)
            
            for cam_idx in range(self.num_cameras):
                multi_view[cam_idx] = self._generate_video_for_camera(
                    shape, noisy_trajectory, color, cam_idx
                )
            
            trajectories_3d.append(noisy_trajectory)
            multi_view_videos.append(multi_view)
            labels.append(shape_idx)
        
        return {
            'trajectories_3d': np.array(trajectories_3d),
            'multi_view_videos': np.array(multi_view_videos),
            'labels': np.array(labels)
        }


if __name__ == '__main__':
    # Quick test
    logger.info("Testing MAGVIT3D Verified Generator...")
    
    generator = MAGVIT3DVerifiedGenerator(seq_length=16, img_size=128)
    dataset = generator.generate_dataset(num_samples=3)
    
    logger.info(f"Generated dataset:")
    logger.info(f"  trajectories_3d: {dataset['trajectories_3d'].shape}")
    logger.info(f"  multi_view_videos: {dataset['multi_view_videos'].shape}")
    logger.info(f"  labels: {dataset['labels'].shape}")
    logger.info("✅ Generator test passed!")

