#!/usr/bin/env python3
"""
Visualize MAGVIT 3D Verified Dataset

Creates comprehensive visualizations similar to the verified notebooks:
- 3D trajectory plots
- Multi-camera 2D views
- Shape rendering examples
- Camera setup visualization

Author: AI Assistant
Date: 2026-01-19
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from magvit_verified_generator import (
    MAGVIT3DVerifiedGenerator,
    setup_cameras,
    generate_linear_trajectory,
    generate_circular_trajectory,
    generate_helical_trajectory,
    generate_parabolic_trajectory
)
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def visualize_trajectory_types(output_dir: Path):
    """Visualize the four trajectory types in 3D."""
    logger.info("Creating trajectory types visualization...")
    
    fig = plt.figure(figsize=(16, 12))
    
    trajectories = {
        'Linear': generate_linear_trajectory(16),
        'Circular': generate_circular_trajectory(16),
        'Helical': generate_helical_trajectory(16),
        'Parabolic': generate_parabolic_trajectory(16)
    }
    
    colors = ['red', 'green', 'blue', 'purple']
    
    for idx, (name, traj) in enumerate(trajectories.items()):
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
        
        # Plot trajectory
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
               color=colors[idx], linewidth=3, label=name)
        
        # Mark start and end
        ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], 
                  color='green', s=200, marker='o', label='Start', edgecolors='black', linewidths=2)
        ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], 
                  color='red', s=200, marker='s', label='End', edgecolors='black', linewidths=2)
        
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_zlabel('Z (m)', fontsize=10)
        ax.set_title(f'{name} Trajectory', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.view_init(elev=20, azim=45)
    
    plt.suptitle('Four Trajectory Types in MAGVIT 3D Dataset', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'magvit_verified_trajectory_types.png', dpi=150, bbox_inches='tight')
    logger.info(f"  ✅ Saved: magvit_verified_trajectory_types.png")
    plt.close()


def visualize_camera_setup(output_dir: Path):
    """Visualize the camera setup and viewing directions."""
    logger.info("Creating camera setup visualization...")
    
    cameras = setup_cameras(img_size=128)
    cam1_pos = cameras['cam1_pos']
    cam2_pos = cameras['cam2_pos']
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot cameras
    ax.scatter(cam1_pos[0], cam1_pos[1], cam1_pos[2], 
              s=500, c='red', marker='^', edgecolors='black', linewidths=2, label='Camera 1')
    ax.scatter(cam2_pos[0], cam2_pos[1], cam2_pos[2], 
              s=500, c='blue', marker='^', edgecolors='black', linewidths=2, label='Camera 2')
    
    # Plot viewing directions (lookingafter +Y)
    for cam_pos, color, name in [(cam1_pos, 'red', 'Cam1'), (cam2_pos, 'blue', 'Cam2')]:
        look_end = cam_pos + np.array([0, 2, 0])  # Look in +Y direction
        ax.plot([cam_pos[0], look_end[0]], [cam_pos[1], look_end[1]], [cam_pos[2], look_end[2]],
               color=color, linewidth=3, linestyle='--', alpha=0.7)
    
    # Plot sample trajectory
    traj = generate_circular_trajectory(16)
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
           color='green', linewidth=2, alpha=0.6, label='Sample Trajectory')
    
    # Plot ground plane
    x_range = np.linspace(-0.5, 1.5, 10)
    y_range = np.linspace(0, 2, 10)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    ax.plot_surface(X, Y, Z, alpha=0.1, color='gray')
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title('Camera Setup: Stereo System Looking in +Y Direction', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'magvit_verified_camera_setup.png', dpi=150, bbox_inches='tight')
    logger.info(f"  ✅ Saved: magvit_verified_camera_setup.png")
    plt.close()


def visualize_multicamera_views(dataset: dict, output_dir: Path):
    """Visualize multi-camera views for sample trajectories."""
    logger.info("Creating multi-camera views visualization...")
    
    multi_view_videos = dataset['multi_view_videos']
    labels = dataset['labels']
    shape_names = ['Cube', 'Cylinder', 'Cone']
    
    # Select one sample of each shape
    sample_indices = [0, 1, 2]  # First cube, cylinder, cone
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    for row, sample_idx in enumerate(sample_indices):
        shape_name = shape_names[labels[sample_idx]]
        
        for col in range(3):
            ax = axes[row, col]
            
            # Show middle frame (frame 8)
            frame = multi_view_videos[sample_idx, col, 8]
            
            ax.imshow(frame)
            ax.set_title(f'{shape_name} - Camera {col+1}', fontsize=12, fontweight='bold')
            ax.axis('off')
    
    plt.suptitle('Multi-Camera Views: Different Shapes at Mid-Trajectory', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'magvit_verified_multicamera_views.png', dpi=150, bbox_inches='tight')
    logger.info(f"  ✅ Saved: magvit_verified_multicamera_views.png")
    plt.close()


def visualize_trajectory_sequence(dataset: dict, output_dir: Path):
    """Visualize trajectory sequence from single camera."""
    logger.info("Creating trajectory sequence visualization...")
    
    multi_view_videos = dataset['multi_view_videos']
    labels = dataset['labels']
    shape_names = ['Cube', 'Cylinder', 'Cone']
    
    # Show cube from camera 1
    sample_idx = 0
    cam_idx = 0
    frame_indices = [0, 3, 6, 9, 12, 15]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, frame_idx in enumerate(frame_indices):
        ax = axes[i]
        frame = multi_view_videos[sample_idx, cam_idx, frame_idx]
        ax.imshow(frame)
        ax.set_title(f'Frame {frame_idx}', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle(f'{shape_names[labels[sample_idx]]} Trajectory Sequence (Camera 1)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'magvit_verified_sequence.png', dpi=150, bbox_inches='tight')
    logger.info(f"  ✅ Saved: magvit_verified_sequence.png")
    plt.close()


def visualize_3d_trajectories(dataset: dict, output_dir: Path):
    """Visualize 3D trajectories by shape."""
    logger.info("Creating 3D trajectories visualization...")
    
    trajectories_3d = dataset['trajectories_3d']
    labels = dataset['labels']
    shape_names = ['Cube', 'Cylinder', 'Cone']
    colors = ['red', 'green', 'blue']
    
    fig = plt.figure(figsize=(18, 6))
    
    for i, (shape_name, color) in enumerate(zip(shape_names, colors)):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        
        # Plot all trajectories for this shape
        shape_trajs = trajectories_3d[labels == i]
        for traj in shape_trajs:
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                   color=color, alpha=0.6, linewidth=2)
        
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_zlabel('Z (m)', fontsize=10)
        ax.set_title(f'{shape_name} Trajectories', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.view_init(elev=20, azim=45)
    
    plt.suptitle('3D Trajectories by Shape (All Samples)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'magvit_verified_3d_trajectories.png', dpi=150, bbox_inches='tight')
    logger.info(f"  ✅ Saved: magvit_verified_3d_trajectories.png")
    plt.close()


def main():
    """Generate all visualizations."""
    logger.info("="*70)
    logger.info("MAGVIT 3D Verified Dataset Visualization")
    logger.info("="*70)
    
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    # Generate dataset
    logger.info("\nGenerating dataset...")
    generator = MAGVIT3DVerifiedGenerator(seq_length=16, img_size=128, num_cameras=3)
    dataset = generator.generate_dataset(num_samples=12)  # 4 of each shape
    
    logger.info(f"  Dataset generated:")
    logger.info(f"    trajectories_3d: {dataset['trajectories_3d'].shape}")
    logger.info(f"    multi_view_videos: {dataset['multi_view_videos'].shape}")
    logger.info(f"    labels: {dataset['labels'].shape}")
    
    # Create visualizations
    logger.info("\nCreating visualizations...")
    visualize_trajectory_types(output_dir)
    visualize_camera_setup(output_dir)
    visualize_multicamera_views(dataset, output_dir)
    visualize_trajectory_sequence(dataset, output_dir)
    visualize_3d_trajectories(dataset, output_dir)
    
    # Save dataset
    dataset_file = output_dir / 'magvit_3d_verified_dataset.npz'
    np.savez_compressed(dataset_file, **dataset)
    logger.info(f"\n✅ Dataset saved: {dataset_file}")
    
    logger.info("\n" + "="*70)
    logger.info("Summary of Generated Visualizations:")
    logger.info("  1. magvit_verified_trajectory_types.png - Four trajectory types")
    logger.info("  2. magvit_verified_camera_setup.png - Camera configuration")
    logger.info("  3. magvit_verified_multicamera_views.png - Multi-camera views")
    logger.info("  4. magvit_verified_sequence.png - Trajectory sequence")
    logger.info("  5. magvit_verified_3d_trajectories.png - 3D trajectories by shape")
    logger.info("="*70)


if __name__ == '__main__':
    main()

