#!/usr/bin/env python3
"""
MAGVIT Comprehensive Visualization (TDD-Validated)

Creates visualization with:
- 3D plot showing cameras and trajectories
- 2D camera views showing projected trajectories

Uses TDD-validated projection formula from magvit_3d_fixed.py
Complies with requirements.md Section 5.4 for timestamped output filenames.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter1d
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict
import cv2


# Import projection from TDD-validated module
import sys
sys.path.insert(0, str(Path(__file__).parent))
from magvit_3d_fixed import project_3d_to_2d, smooth_trajectory


def get_output_filename() -> str:
    """Generate timestamped filename per requirements.md Section 5.4."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return f"{timestamp}_magvit_comprehensive_TDD_validated.png"


def project_point(point_3d: np.ndarray, camera_pos: np.ndarray,
                  focal_length: float = 600, 
                  img_size: Tuple[int, int] = (480, 640)) -> Optional[np.ndarray]:
    """
    Project 3D point to 2D using TDD-validated formula.
    
    Wrapper around magvit_3d_fixed.project_3d_to_2d
    """
    return project_3d_to_2d(point_3d, camera_pos, focal_length, img_size)


def get_test_trajectories() -> Dict[str, np.ndarray]:
    """Generate test trajectories for visualization."""
    def generate_linear(seq_length: int = 16) -> np.ndarray:
        start = np.array([0.0, 1.2, 2.5])
        end = np.array([0.6, 2.0, 2.7])
        t = np.linspace(0, 1, seq_length)
        return start[None, :] + t[:, None] * (end - start)[None, :]
    
    def generate_circular(seq_length: int = 16) -> np.ndarray:
        t = np.linspace(0, 2 * np.pi, seq_length + 1)[:-1]
        radius = 0.35
        x = radius * np.cos(t)
        y = np.ones(seq_length) * 1.7
        z = 2.55 + radius * np.sin(t)
        trajectory = np.stack([x, y, z], axis=1)
        return smooth_trajectory(trajectory, sigma=0.8)
    
    def generate_helical(seq_length: int = 16) -> np.ndarray:
        t = np.linspace(0, 2 * np.pi, seq_length)
        radius = 0.25
        x = radius * np.cos(t)
        y = 1.3 + t / (2 * np.pi) * 0.8
        z = 2.55 + 0.2 * np.sin(t)
        trajectory = np.stack([x, y, z], axis=1)
        return smooth_trajectory(trajectory, sigma=1.2)
    
    return {
        'linear': generate_linear(),
        'circular': generate_circular(),
        'helical': generate_helical()
    }


def get_camera_positions() -> list:
    """Get camera positions for visualization."""
    return [
        np.array([0.0, 0.0, 2.5]),    # Camera 1
        np.array([0.65, 0.0, 2.5]),   # Camera 2
        np.array([0.325, 0.56, 2.5])  # Camera 3
    ]


def create_visualization(trajectories: Dict[str, np.ndarray],
                        camera_positions: list,
                        return_projection_counts: bool = False,
                        return_figure: bool = False) -> Optional[Dict]:
    """
    Create comprehensive visualization with 3D view and camera projections.
    
    Args:
        trajectories: Dict of trajectory arrays
        camera_positions: List of camera position arrays
        return_projection_counts: If True, return projection statistics
        return_figure: If True, return figure handle
        
    Returns:
        Dict with statistics if requested, None otherwise
    """
    fig = plt.figure(figsize=(20, 13))
    
    # 3D subplot (large, left side)
    ax_3d = fig.add_subplot(2, 3, (1, 4), projection='3d')
    
    # Camera 2D subplots (right side)
    ax_cam1 = fig.add_subplot(2, 3, 2)
    ax_cam2 = fig.add_subplot(2, 3, 3)
    ax_cam3 = fig.add_subplot(2, 3, 6)
    
    camera_axes = [ax_cam1, ax_cam2, ax_cam3]
    
    # Plot 3D: Cameras
    cam_colors = ['red', 'blue', 'green']
    for idx, (pos, color) in enumerate(zip(camera_positions, cam_colors)):
        ax_3d.scatter(*pos, s=300, c=color, marker='^', 
                     label=f'Camera {idx+1}', edgecolors='black', linewidths=2)
    
    # Plot 3D: Trajectories
    traj_colors = {'linear': 'orange', 'circular': 'purple', 'helical': 'cyan'}
    for name, traj in trajectories.items():
        ax_3d.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                  color=traj_colors[name], linewidth=3, label=name.capitalize(), alpha=0.7)
    
    ax_3d.set_xlabel('X (m)', fontsize=12)
    ax_3d.set_ylabel('Y (m)', fontsize=12)
    ax_3d.set_zlabel('Z (m)', fontsize=12)
    ax_3d.set_title('3D Space: Cameras and Trajectories', fontsize=14, fontweight='bold')
    ax_3d.legend(fontsize=10)
    ax_3d.grid(True, alpha=0.3)
    ax_3d.view_init(elev=20, azim=45)
    
    # Statistics tracking
    stats = {f'camera{i+1}_points': 0 for i in range(3)}
    stats.update({f'camera{i+1}_in_bounds': 0 for i in range(3)})
    
    # Project trajectories onto camera views
    img_size = (480, 640)
    focal_length = 600
    
    for cam_idx, (cam_pos, ax) in enumerate(zip(camera_positions, camera_axes)):
        # Create blank image
        img = np.ones((img_size[0], img_size[1], 3), dtype=np.uint8) * 255
        
        # Draw crosshairs
        cv2.line(img, (img_size[1]//2, 0), (img_size[1]//2, img_size[0]), (200, 200, 200), 1)
        cv2.line(img, (0, img_size[0]//2), (img_size[1], img_size[0]//2), (200, 200, 200), 1)
        
        # Project each trajectory
        for name, traj in trajectories.items():
            color_bgr = {
                'linear': (0, 140, 255),    # Orange in BGR
                'circular': (128, 0, 128),  # Purple in BGR
                'helical': (255, 255, 0)     # Cyan in BGR
            }[name]
            
            for point_3d in traj:
                point_2d = project_point(point_3d, cam_pos, focal_length, img_size)
                
                if point_2d is not None:
                    stats[f'camera{cam_idx+1}_points'] += 1
                    
                    x, y = int(point_2d[0]), int(point_2d[1])
                    
                    # Check if in bounds
                    if 0 <= x < img_size[1] and 0 <= y < img_size[0]:
                        stats[f'camera{cam_idx+1}_in_bounds'] += 1
                        cv2.circle(img, (x, y), 5, color_bgr, -1)
                    else:
                        # Draw smaller point outside bounds (for debugging)
                        if -img_size[1] <= x < 2*img_size[1] and -img_size[0] <= y < 2*img_size[0]:
                            cv2.circle(img, (max(0, min(x, img_size[1]-1)), 
                                           max(0, min(y, img_size[0]-1))), 
                                      2, color_bgr, -1)
        
        # Display
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f'Camera {cam_idx+1} View', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Add stats text
        points_text = f"{stats[f'camera{cam_idx+1}_points']} points"
        in_bounds = stats[f'camera{cam_idx+1}_in_bounds']
        ax.text(0.02, 0.98, points_text, transform=ax.transAxes, 
               fontsize=9, va='top', ha='left', 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    # Prepare return value
    result = {}
    if return_projection_counts:
        result.update(stats)
    if return_figure:
        result['figure'] = fig
    
    if result:
        return result
    return None


def main():
    """Main function to create and save visualization."""
    print("=" * 70)
    print("MAGVIT Comprehensive Visualization (TDD-Validated)")
    print("=" * 70)
    
    # Get trajectories and cameras
    trajectories = get_test_trajectories()
    camera_positions = get_camera_positions()
    
    # Create visualization with stats
    print("\nGenerating visualization...")
    stats = create_visualization(trajectories, camera_positions, 
                                 return_projection_counts=True)
    
    # Print stats
    print("\nProjection Statistics:")
    for cam_idx in [1, 2, 3]:
        total = stats[f'camera{cam_idx}_points']
        in_bounds = stats[f'camera{cam_idx}_in_bounds']
        if total > 0:
            pct = (in_bounds / total) * 100
            print(f"  Camera {cam_idx}: {in_bounds}/{total} points in bounds ({pct:.1f}%)")
    
    # Save with timestamp
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    filename = get_output_filename()
    output_path = output_dir / filename
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
