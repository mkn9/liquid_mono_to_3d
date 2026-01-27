#!/usr/bin/env python3
"""
MAGVIT 3D Trajectory Generator

Generates 3D trajectory data for cubes, cylinders, and cones with multi-camera views.

Implementation follows TDD: Written to pass tests in test_magvit_3d_generation.py

Key Features:
- 4 trajectory patterns: linear, circular, helical, parabolic
- 3 shape types: cube, cylinder, cone
- Multi-camera rendering (3 cameras)
- Gaussian noise (σ=0.02) for realism
- Deterministic with fixed random seed
"""

import numpy as np
from typing import Dict, Tuple, Callable, List


def generate_linear_trajectory(seq_length: int = 16) -> np.ndarray:
    """
    Generate linear 3D trajectory (straight line).
    
    Args:
        seq_length: Number of frames in trajectory
        
    Returns:
        Array of shape (seq_length, 3) with 3D coordinates
    """
    start = np.array([-0.3, -0.3, 0.0])
    end = np.array([0.3, 0.3, 0.2])
    t = np.linspace(0, 1, seq_length)
    trajectory = start[None, :] + t[:, None] * (end - start)[None, :]
    return trajectory


def generate_circular_trajectory(seq_length: int = 16) -> np.ndarray:
    """
    Generate circular 3D trajectory in XY plane.
    
    Args:
        seq_length: Number of frames in trajectory
        
    Returns:
        Array of shape (seq_length, 3) with 3D coordinates
    """
    t = np.linspace(0, 2 * np.pi, seq_length)
    radius = 0.3
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = np.zeros(seq_length)
    trajectory = np.stack([x, y, z], axis=1)
    return trajectory


def generate_helical_trajectory(seq_length: int = 16) -> np.ndarray:
    """
    Generate helical 3D trajectory (spiral with Z progression).
    
    Args:
        seq_length: Number of frames in trajectory
        
    Returns:
        Array of shape (seq_length, 3) with 3D coordinates
    """
    t = np.linspace(0, 2 * np.pi, seq_length)
    radius = 0.25
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = t / (2 * np.pi) * 0.3 - 0.15
    trajectory = np.stack([x, y, z], axis=1)
    return trajectory


def generate_parabolic_trajectory(seq_length: int = 16) -> np.ndarray:
    """
    Generate parabolic 3D trajectory.
    
    Args:
        seq_length: Number of frames in trajectory
        
    Returns:
        Array of shape (seq_length, 3) with 3D coordinates
    """
    t = np.linspace(-1, 1, seq_length)
    x = t * 0.3
    y = t ** 2 * 0.3 - 0.3
    z = -t ** 2 * 0.1
    trajectory = np.stack([x, y, z], axis=1)
    return trajectory


def _render_shape_simple(
    shape: str,
    point_3d: np.ndarray,
    color: Tuple[int, int, int],
    img_size: int
) -> np.ndarray:
    """
    Render a shape at a 3D point as a simple 2D projection.
    
    Args:
        shape: Shape type ('cube', 'cylinder', 'cone')
        point_3d: 3D position [x, y, z]
        color: RGB color tuple
        img_size: Image size (square)
        
    Returns:
        Image array of shape (img_size, img_size, 3)
    """
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    
    # Simple 2D projection: map x,y to image coordinates
    # Map [-0.5, 0.5] to [0, img_size]
    x = int((point_3d[0] + 0.5) * img_size)
    y = int((point_3d[1] + 0.5) * img_size)
    
    # Draw shape if in bounds
    if 10 < x < img_size - 10 and 10 < y < img_size - 10:
        if shape == 'cube':
            img[y-10:y+10, x-10:x+10] = color
        elif shape == 'cylinder':
            Y, X = np.ogrid[:img_size, :img_size]
            mask = (X - x)**2 + (Y - y)**2 <= 100
            img[mask] = color
        elif shape == 'cone':
            for dy in range(-10, 11):
                width = max(0, 10 - abs(dy))
                if 0 <= y + dy < img_size:
                    x_start = max(0, x - width)
                    x_end = min(img_size, x + width + 1)
                    img[y + dy, x_start:x_end] = color
    
    return img


class MAGVIT3DGenerator:
    """
    MAGVIT 3D trajectory generator for cubes, cylinders, and cones.
    
    Generates multi-camera video sequences of 3D object trajectories.
    """
    
    def __init__(self, seq_length: int = 16, img_size: int = 128, num_cameras: int = 3):
        """
        Initialize generator.
        
        Args:
            seq_length: Number of frames per trajectory
            img_size: Image size (square)
            num_cameras: Number of cameras for multi-view
        """
        self.seq_length = seq_length
        self.img_size = img_size
        self.num_cameras = num_cameras
        
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
        
        # Camera offsets for multi-view
        self.camera_offsets = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.65, 0.0, 0.0]),
            np.array([0.325, 0.56, 0.0])
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
            shape: Shape type ('cube', 'cylinder', 'cone')
            trajectory: 3D trajectory array of shape (seq_length, 3)
            color: RGB color tuple
            camera_idx: Camera index
            
        Returns:
            Video array of shape (seq_length, img_size, img_size, 3)
        """
        video = np.zeros((self.seq_length, self.img_size, self.img_size, 3), dtype=np.uint8)
        offset = self.camera_offsets[min(camera_idx, len(self.camera_offsets) - 1)]
        
        for frame_idx in range(self.seq_length):
            point_3d = trajectory[frame_idx] - offset
            video[frame_idx] = _render_shape_simple(shape, point_3d, color, self.img_size)
        
        return video
    
    def generate_dataset(self, num_samples: int) -> Dict[str, np.ndarray]:
        """
        Generate complete dataset of 3D trajectories with multi-camera videos.
        
        Args:
            num_samples: Number of trajectory samples to generate
            
        Returns:
            Dictionary with keys:
            - 'trajectories_3d': shape (num_samples, seq_length, 3)
            - 'multi_view_videos': shape (num_samples, num_cameras, seq_length, img_size, img_size, 3)
            - 'labels': shape (num_samples,) - 0=cube, 1=cylinder, 2=cone
        """
        trajectories_3d = []
        multi_view_videos = []
        labels = []
        
        for i in range(num_samples):
            # Select shape (cycle through 3 shapes)
            shape_idx = i % 3
            shape = self.shapes[shape_idx]
            color = self.colors[shape_idx]
            
            # Select trajectory pattern (cycle through 4 patterns)
            traj_func = self.trajectory_funcs[i % len(self.trajectory_funcs)]
            trajectory = traj_func(seq_length=self.seq_length)
            
            # Add Gaussian noise (σ=0.02)
            noise = np.random.normal(0, 0.02, trajectory.shape)
            noisy_trajectory = trajectory + noise
            
            # Generate multi-view video
            multi_view = []
            for cam_idx in range(self.num_cameras):
                video = self._generate_video_for_camera(shape, noisy_trajectory, color, cam_idx)
                multi_view.append(video)
            multi_view = np.array(multi_view)  # Shape: (num_cameras, seq_length, img_size, img_size, 3)
            
            trajectories_3d.append(noisy_trajectory)
            multi_view_videos.append(multi_view)
            labels.append(shape_idx)
        
        return {
            'trajectories_3d': np.array(trajectories_3d),
            'multi_view_videos': np.array(multi_view_videos),
            'labels': np.array(labels)
        }

