#!/usr/bin/env python3
"""
Generate MAGVIT 3D dataset with 50 samples.

This script uses the TESTED magvit_3d_generator module.
All code has passed comprehensive test suite.

Updated to use timestamped filenames per requirements.md Section 5.4.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import logging

from magvit_3d_generator import MAGVIT3DGenerator
from output_utils_shared import save_figure

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_visualizations(dataset: dict, output_dir: Path):
    """
    Create visualization plots for the dataset.
    
    Args:
        dataset: Dictionary with trajectories_3d, multi_view_videos, labels
        output_dir: Directory to save plots
    """
    logger.info("Creating visualizations...")
    
    trajectories_3d = dataset['trajectories_3d']
    labels = dataset['labels']
    shape_names = ['Cube', 'Cylinder', 'Cone']
    colors = ['red', 'green', 'blue']
    
    # 1. Trajectory Type Visualization (shows 4 distinct types)
    from magvit_3d_generator import (
        generate_linear_trajectory,
        generate_circular_trajectory,
        generate_helical_trajectory,
        generate_parabolic_trajectory
    )
    
    # Generate one example of each trajectory type
    traj_types = [
        generate_linear_trajectory(16),
        generate_circular_trajectory(16),
        generate_helical_trajectory(16),
        generate_parabolic_trajectory(16)
    ]
    traj_names = ['Linear', 'Circular', 'Helical', 'Parabolic']
    traj_colors = ['red', 'green', 'blue', 'orange']
    
    fig = plt.figure(figsize=(14, 12))
    
    for i, (traj, name, color) in enumerate(zip(traj_types, traj_names, traj_colors)):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        
        # Plot trajectory
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                color=color, linewidth=3, marker='o', markersize=4, alpha=0.8)
        
        # Mark start and end points
        ax.scatter(*traj[0], s=200, c='green', marker='o', edgecolors='black', 
                   linewidths=2, label='Start', zorder=5)
        ax.scatter(*traj[-1], s=200, c='red', marker='s', edgecolors='black', 
                   linewidths=2, label='End', zorder=5)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'{name} Trajectory', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.view_init(elev=20, azim=45)
        
        # Set consistent axis limits
        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([-0.5, 0.5])
        ax.set_zlim([-0.3, 0.3])
    
    plt.suptitle('Four Trajectory Types: Linear, Circular, Helical, Parabolic', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    saved_path = save_figure(fig, output_dir.parent, "magvit_3d_trajectory_types.png", dpi=150)
    logger.info(f"  Saved: {saved_path.name}")
    plt.close()
    
    # 1b. Sample trajectories by shape (supplementary view)
    fig = plt.figure(figsize=(15, 5))
    
    for i, (shape_name, color) in enumerate(zip(shape_names, colors)):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        
        # Plot first 5 trajectories for this shape
        shape_trajs = trajectories_3d[labels == i]
        for traj in shape_trajs[:5]:
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, alpha=0.6, linewidth=2)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'{shape_name} Sample Trajectories (mixed types)')
        ax.grid(True, alpha=0.3)
        ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    saved_path = save_figure(fig, output_dir.parent, "magvit_3d_trajectories_by_shape.png", dpi=150)
    logger.info(f"  Saved: {saved_path.name}")
    plt.close()
    
    # 2. 2D Error Analysis (path length statistics)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (shape_name, color, ax) in enumerate(zip(shape_names, colors, axes)):
        shape_trajs = trajectories_3d[labels == i]
        
        # Calculate path lengths
        lengths = []
        for traj in shape_trajs:
            diffs = np.diff(traj, axis=0)
            path_length = np.sum(np.linalg.norm(diffs, axis=1))
            lengths.append(path_length)
        
        ax.hist(lengths, bins=15, color=color, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Total Path Length (m)')
        ax.set_ylabel('Count')
        ax.set_title(f'{shape_name} Path Lengths')
        ax.grid(True, alpha=0.3)
        ax.axvline(np.mean(lengths), color='black', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(lengths):.3f}m')
        ax.legend()
    
    plt.tight_layout()
    saved_path = save_figure(fig, output_dir.parent, "magvit_3d_errors_2d.png", dpi=150)
    logger.info(f"  Saved: {saved_path.name}")
    plt.close()
    
    # 3. Camera Position Visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Camera positions (z=2.55m looking at origin)
    cam_positions = np.array([
        [0.0, 0.0, 2.55],
        [0.65, 0.0, 2.55],
        [0.325, 0.56, 2.55]
    ])
    
    # Plot cameras
    for i, pos in enumerate(cam_positions):
        ax.scatter(*pos, s=300, c=f'C{i}', marker='^', edgecolors='black', linewidths=2,
                   label=f'Camera {i+1}', zorder=5)
        # Viewing direction to origin
        ax.plot([pos[0], 0], [pos[1], 0], [pos[2], 0],
                f'C{i}--', alpha=0.6, linewidth=2)
    
    # Plot origin (target point)
    ax.scatter(0, 0, 0, s=200, c='black', marker='o', label='Target (origin)', zorder=5)
    
    # Plot sample trajectory region
    sample_traj = trajectories_3d[0]
    ax.plot(sample_traj[:, 0], sample_traj[:, 1], sample_traj[:, 2],
            'gray', alpha=0.3, linestyle=':', linewidth=1, label='Sample trajectory')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Camera Setup and Viewing Directions')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=20, azim=45)
    
    # Set equal aspect ratio
    max_range = 1.5
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([0, 3])
    
    saved_path = save_figure(fig, output_dir.parent, "magvit_3d_cameras.png", dpi=150)
    logger.info(f"  Saved: {saved_path.name}")
    plt.close()


def main():
    """Main function to generate MAGVIT 3D dataset."""
    logger.info("="*70)
    logger.info("MAGVIT 3D Dataset Generation (50 samples) - TDD VERIFIED")
    logger.info("="*70)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Output directory
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    # Generate dataset using TESTED generator
    logger.info("Generating dataset...")
    generator = MAGVIT3DGenerator(seq_length=16, img_size=128, num_cameras=3)
    dataset = generator.generate_dataset(num_samples=50)
    
    # Save dataset
    output_file = output_dir / 'magvit_3d_dataset.npz'
    np.savez_compressed(output_file, **dataset)
    
    logger.info("")
    logger.info("Dataset saved successfully:")
    logger.info(f"  File: {output_file}")
    logger.info(f"  trajectories_3d shape: {dataset['trajectories_3d'].shape}")
    logger.info(f"  multi_view_videos shape: {dataset['multi_view_videos'].shape}")
    logger.info(f"  labels shape: {dataset['labels'].shape}")
    logger.info("")
    
    # Create visualizations
    create_visualizations(dataset, output_dir)
    
    logger.info("")
    logger.info("="*70)
    logger.info("Generation complete!")
    logger.info("="*70)
    logger.info(f"Output directory: {output_dir}")
    logger.info("Files created:")
    for f in sorted(output_dir.glob('*')):
        size_kb = f.stat().st_size / 1024
        logger.info(f"  - {f.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()

