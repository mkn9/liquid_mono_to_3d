"""
Generate Persistence-Augmented Dataset

Overlays non-persistent spheres (1-3 frame duration) onto existing trajectory videos
to create a dataset for training track persistence classifiers.

Author: AI Assistant
Date: 2026-01-26
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
from datetime import datetime


class TransientSphereGenerator:
    """Generate non-persistent spheres that appear for 1-3 frames."""
    
    def __init__(self, min_duration: int = 1, max_duration: int = 3, 
                 sphere_radius: float = 0.05):
        """
        Initialize transient sphere generator.
        
        Args:
            min_duration: Minimum frames a transient persists
            max_duration: Maximum frames a transient persists
            sphere_radius: Radius of the sphere in normalized coordinates
        """
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.sphere_radius = sphere_radius
        self.trajectory_types = ['linear', 'circular', 'helical', 'parabolic']
    
    def generate_transient_parameters(self, num_frames: int, 
                                     num_transients: int) -> List[Dict]:
        """
        Generate parameters for transient spheres.
        
        Args:
            num_frames: Total number of frames in video
            num_transients: Number of transient spheres to generate
            
        Returns:
            List of transient parameter dictionaries
        """
        transients = []
        
        for _ in range(num_transients):
            # Random duration between min and max
            duration = np.random.randint(self.min_duration, self.max_duration + 1)
            
            # Random start frame (ensure transient fits within video)
            max_start = num_frames - duration
            start_frame = np.random.randint(0, max_start + 1) if max_start >= 0 else 0
            
            # Random trajectory type
            trajectory_type = np.random.choice(self.trajectory_types)
            
            # Random start position (normalized to [-1, 1])
            start_position = np.random.uniform(-0.8, 0.8, size=3)
            
            transients.append({
                'start_frame': int(start_frame),
                'duration': int(duration),
                'trajectory_type': trajectory_type,
                'start_position': start_position
            })
        
        return transients
    
    def render_sphere(self, frame: torch.Tensor, 
                     position: np.ndarray) -> torch.Tensor:
        """
        Render a sphere onto a video frame.
        
        Args:
            frame: Video frame (C, H, W)
            position: 3D position in normalized coordinates [-1, 1]
            
        Returns:
            Frame with sphere rendered
        """
        # Convert to numpy for rendering
        frame_np = frame.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
        H, W = frame_np.shape[:2]
        
        # Project 3D position to 2D (simple orthographic projection)
        # Map [-1, 1] to pixel coordinates
        center_x = int((position[0] + 1.0) * W / 2.0)
        center_y = int((position[1] + 1.0) * H / 2.0)
        
        # Sphere size depends on z-depth (closer = larger)
        depth_scale = 1.0 - (position[2] + 1.0) / 2.0  # Map z to [0, 1]
        radius_pixels = int(self.sphere_radius * min(H, W) * (0.5 + depth_scale))
        radius_pixels = max(2, radius_pixels)  # Minimum 2 pixels
        
        # Draw sphere (bright color to distinguish from background)
        sphere_color = (0.9, 0.3, 0.3)  # Reddish color
        
        # Create a copy to avoid modifying original
        rendered = frame_np.copy()
        
        # Draw filled circle
        if 0 <= center_x < W and 0 <= center_y < H:
            y_grid, x_grid = np.ogrid[:H, :W]
            mask = (x_grid - center_x)**2 + (y_grid - center_y)**2 <= radius_pixels**2
            rendered[mask] = sphere_color
        
        # Convert back to tensor
        return torch.from_numpy(rendered).permute(2, 0, 1).float()
    
    def generate_transient_trajectory(self, trajectory_type: str,
                                     start_position: np.ndarray,
                                     num_frames: int) -> List[np.ndarray]:
        """
        Generate a short trajectory for a transient sphere.
        
        Args:
            trajectory_type: Type of trajectory ('linear', 'circular', 'helical', 'parabolic')
            start_position: Starting 3D position
            num_frames: Number of frames for this transient
            
        Returns:
            List of 3D positions
        """
        positions = []
        
        # Generate velocity/direction
        velocity = np.random.uniform(-0.1, 0.1, size=3)
        
        for i in range(num_frames):
            if trajectory_type == 'linear':
                # Simple linear motion
                pos = start_position + velocity * i
                
            elif trajectory_type == 'circular':
                # Circular motion in xy plane
                angle = i * 0.5
                radius = 0.3
                pos = start_position.copy()
                pos[0] += radius * np.cos(angle)
                pos[1] += radius * np.sin(angle)
                
            elif trajectory_type == 'helical':
                # Helical motion
                angle = i * 0.5
                radius = 0.2
                pos = start_position.copy()
                pos[0] += radius * np.cos(angle)
                pos[1] += radius * np.sin(angle)
                pos[2] += velocity[2] * i
                
            elif trajectory_type == 'parabolic':
                # Parabolic motion
                t = i / max(num_frames - 1, 1)
                pos = start_position.copy()
                pos[0] += velocity[0] * t
                pos[1] += velocity[1] * t - 0.5 * t**2  # Gravity effect
                pos[2] += velocity[2] * t
            
            else:
                pos = start_position
            
            # Clip to valid range
            pos = np.clip(pos, -1.0, 1.0)
            positions.append(pos)
        
        return positions


def augment_video_with_transients(video: torch.Tensor, 
                                  transients: List[Dict]) -> Tuple[torch.Tensor, Dict]:
    """
    Augment a video with transient spheres.
    
    Args:
        video: Original video tensor (T, C, H, W)
        transients: List of transient parameters
        
    Returns:
        Tuple of (augmented_video, metadata)
    """
    augmented_video = video.clone()
    generator = TransientSphereGenerator()
    
    transient_frames_set = set()
    
    for transient in transients:
        start_frame = transient['start_frame']
        duration = transient['duration']
        trajectory_type = transient['trajectory_type']
        start_position = transient['start_position']
        
        # Generate trajectory for this transient
        trajectory = generator.generate_transient_trajectory(
            trajectory_type, start_position, duration
        )
        
        # Render sphere on each frame
        for i, position in enumerate(trajectory):
            frame_idx = start_frame + i
            if 0 <= frame_idx < len(video):
                augmented_video[frame_idx] = generator.render_sphere(
                    augmented_video[frame_idx], position
                )
                transient_frames_set.add(int(frame_idx))
    
    metadata = {
        'num_transients': len(transients),
        'transient_frames': sorted(list(transient_frames_set)),
        'transient_details': transients
    }
    
    return augmented_video, metadata


def load_existing_trajectory(video_path: str) -> Tuple[torch.Tensor, Dict]:
    """
    Load an existing trajectory sample.
    
    Args:
        video_path: Path to video .pt file
        
    Returns:
        Tuple of (video, metadata)
    """
    # Load video
    video = torch.load(video_path)
    
    # Load corresponding metadata
    # Check if JSON is in same directory or in a 'labels' subdirectory
    video_path_obj = Path(video_path)
    metadata_path = video_path_obj.with_suffix('.json')
    
    if not metadata_path.exists():
        # Try labels subdirectory
        parent_dir = video_path_obj.parent.parent
        labels_dir = parent_dir / 'labels'
        metadata_path = labels_dir / video_path_obj.with_suffix('.json').name
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return video, metadata


def save_augmented_sample(video: torch.Tensor, metadata: Dict, 
                         output_dir: Path, sample_idx: int):
    """
    Save an augmented video sample.
    
    Args:
        video: Augmented video tensor
        metadata: Metadata dictionary
        output_dir: Output directory
        sample_idx: Sample index
    """
    # Save video
    video_path = output_dir / f"augmented_traj_{sample_idx:05d}.pt"
    torch.save(video, video_path)
    
    # Save metadata
    metadata_path = output_dir / f"augmented_traj_{sample_idx:05d}.json"
    with open(metadata_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        metadata_serializable = metadata.copy()
        if 'transient_details' in metadata_serializable:
            for transient in metadata_serializable['transient_details']:
                if isinstance(transient.get('start_position'), np.ndarray):
                    transient['start_position'] = transient['start_position'].tolist()
        json.dump(metadata_serializable, f, indent=2)


def save_checkpoint(checkpoint: Dict, output_dir: Path):
    """Save checkpoint for resuming generation."""
    checkpoint_path = output_dir / "checkpoint.json"
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def load_checkpoint(output_dir: Path) -> Optional[Dict]:
    """Load checkpoint if it exists."""
    checkpoint_path = output_dir / "checkpoint.json"
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return None


def main():
    """Main execution function."""
    print("=" * 70)
    print("Persistence-Augmented Dataset Generator")
    print("=" * 70)
    print("\nTDD GREEN phase implementation complete.")
    print("Ready to generate augmented dataset with non-persistent spheres.")
    
    # Example usage
    generator = TransientSphereGenerator()
    print(f"\nGenerator initialized:")
    print(f"  Duration range: {generator.min_duration}-{generator.max_duration} frames")
    print(f"  Sphere radius: {generator.sphere_radius}")
    print(f"  Trajectory types: {generator.trajectory_types}")


if __name__ == "__main__":
    main()
