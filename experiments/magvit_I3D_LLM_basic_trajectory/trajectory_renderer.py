"""
Trajectory renderer: Converts 3D trajectories to video frames (IMAGES).

CRITICAL: This module renders IMAGES (tensors with shape T,C,H,W),
NOT coordinate arrays. This ensures we're building a TRUE vision model.

Following TDD: Implementation created after tests (GREEN phase).
"""

import numpy as np
import torch
import cv2
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class CameraParams:
    """Camera parameters for 3D-to-2D projection.
    
    Attributes:
        position: 3D position of camera (x, y, z)
        focal_length: Camera focal length in pixels
        image_center: Image center point (cx, cy) in pixels
    """
    position: np.ndarray
    focal_length: float
    image_center: Tuple[int, int]


class TrajectoryRenderer:
    """Render 3D trajectories as video frames (RGB images).
    
    This class ensures we process IMAGES, not coordinate shortcuts.
    Output is always (T, 3, H, W) tensor representing RGB video.
    
    Args:
        image_size: Tuple of (height, width) for output frames
        style: Rendering style - 'dot' or 'trail'
        dot_radius: Radius of dots when using 'dot' style
        trail_length: Number of previous frames to show in 'trail' style
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (64, 64),
        style: str = 'dot',
        dot_radius: int = 5,  # Increased from 3 to 5 for more visible differences
        trail_length: int = 5
    ):
        """Initialize trajectory renderer.
        
        Args:
            image_size: (height, width) of output frames
            style: 'dot' or 'trail'
            dot_radius: Radius for dot rendering
            trail_length: Number of frames to show in trail
        """
        self.image_size = image_size
        self.style = style
        self.dot_radius = dot_radius
        self.trail_length = trail_length
    
    def render_video(
        self,
        trajectory_3d: np.ndarray,
        camera_params: CameraParams
    ) -> torch.Tensor:
        """Render 3D trajectory as video frames (IMAGES, not coordinates).
        
        This is the core method enforcing TRUE VISION modeling.
        Output is RGB video tensor, not coordinate array.
        
        Args:
            trajectory_3d: 3D trajectory points, shape (T, 3)
            camera_params: Camera configuration for projection
        
        Returns:
            torch.Tensor: Video frames, shape (T, 3, H, W)
                - T: number of frames
                - 3: RGB channels
                - H, W: image dimensions
        
        Raises:
            ValueError: If trajectory is empty or has wrong shape
        """
        # Validate input
        if trajectory_3d.size == 0:
            raise ValueError("Trajectory cannot be empty")
        
        if trajectory_3d.ndim != 2 or trajectory_3d.shape[1] != 3:
            raise ValueError(
                f"Trajectory must have shape (T, 3), got {trajectory_3d.shape}"
            )
        
        num_frames = len(trajectory_3d)
        height, width = self.image_size
        
        # Initialize frames list
        frames = []
        
        # Project all 3D points to 2D
        points_2d = self._project_trajectory(trajectory_3d, camera_params)
        
        # Render each frame
        for t in range(num_frames):
            # Create blank white canvas (RGB)
            frame = np.ones((height, width, 3), dtype=np.uint8) * 255
            
            if self.style == 'dot':
                # Render single dot at current position
                self._draw_dot(frame, points_2d[t])
            
            elif self.style == 'trail':
                # Render trail of previous positions
                start_idx = max(0, t - self.trail_length + 1)
                for prev_t in range(start_idx, t + 1):
                    # Fade older points
                    alpha = (prev_t - start_idx + 1) / self.trail_length
                    self._draw_dot(frame, points_2d[prev_t], alpha=alpha)
            
            frames.append(frame)
        
        # Convert to tensor: (T, H, W, 3) → (T, 3, H, W)
        frames_array = np.stack(frames)  # (T, H, W, 3)
        frames_tensor = torch.from_numpy(frames_array).permute(0, 3, 1, 2).float() / 255.0
        
        return frames_tensor
    
    def _project_trajectory(
        self,
        trajectory_3d: np.ndarray,
        camera_params: CameraParams
    ) -> np.ndarray:
        """Project 3D trajectory points to 2D image coordinates.
        
        Uses pinhole camera model:
            x_2d = f * (X - Cx) / (Z - Cz) + cx
            y_2d = f * (Y - Cy) / (Z - Cz) + cy
        
        Args:
            trajectory_3d: 3D points, shape (T, 3)
            camera_params: Camera configuration
        
        Returns:
            np.ndarray: 2D points, shape (T, 2)
        """
        # Camera parameters
        cam_pos = camera_params.position
        focal_length = camera_params.focal_length
        cx, cy = camera_params.image_center
        
        # Translate points to camera coordinate system
        points_cam = trajectory_3d - cam_pos
        
        # Extract coordinates
        X = points_cam[:, 0]
        Y = points_cam[:, 1]
        Z = points_cam[:, 2]
        
        # Handle points behind camera (Z <= 0)
        # Clamp to small positive value to avoid division by zero
        Z = np.maximum(Z, 0.01)
        
        # Project to 2D
        x_2d = focal_length * X / Z + cx
        y_2d = focal_length * Y / Z + cy
        
        # Clamp to image boundaries
        height, width = self.image_size
        x_2d = np.clip(x_2d, 0, width - 1)
        y_2d = np.clip(y_2d, 0, height - 1)
        
        points_2d = np.stack([x_2d, y_2d], axis=1)
        
        return points_2d
    
    def _draw_dot(
        self,
        frame: np.ndarray,
        point_2d: np.ndarray,
        alpha: float = 1.0
    ):
        """Draw a dot on the frame at specified 2D location.
        
        Args:
            frame: Image to draw on, shape (H, W, 3), modified in-place
            point_2d: 2D point coordinates (x, y)
            alpha: Opacity (0.0 to 1.0) for fading effect
        """
        x, y = point_2d.astype(int)
        
        # Color: Red with alpha-based intensity (more visible than blue)
        # (Using BGR for OpenCV)
        color_intensity = int(255 * alpha)
        color = (0, 0, color_intensity)  # BGR: red
        
        # Draw filled circle
        cv2.circle(frame, (x, y), self.dot_radius, color, thickness=-1)
        
        # Add outer ring for more visibility
        cv2.circle(frame, (x, y), self.dot_radius + 1, (0, 0, 0), thickness=1)

