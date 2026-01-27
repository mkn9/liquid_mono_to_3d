#!/usr/bin/env python3
"""
Generate MAGVIT 3D dataset with 50 samples (cube, cylinder, cone trajectories).

Simple version based on working proof-of-concept code.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_linear_trajectory(seq_length=16):
    """Generate linear 3D trajectory."""
    start = np.array([-0.3, -0.3, 0.0])
    end = np.array([0.3, 0.3, 0.2])
    t = np.linspace(0, 1, seq_length)
    trajectory = start[None, :] + t[:, None] * (end - start)[None, :]
    return trajectory


def generate_circular_trajectory(seq_length=16):
    """Generate circular 3D trajectory."""
    t = np.linspace(0, 2 * np.pi, seq_length)
    radius = 0.3
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = np.zeros(seq_length)
    trajectory = np.stack([x, y, z], axis=1)
    return trajectory


def generate_helical_trajectory(seq_length=16):
    """Generate helical 3D trajectory."""
    t = np.linspace(0, 2 * np.pi, seq_length)
    radius = 0.25
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = t / (2 * np.pi) * 0.3 - 0.15
    trajectory = np.stack([x, y, z], axis=1)
    return trajectory


def generate_parabolic_trajectory(seq_length=16):
    """Generate parabolic 3D trajectory."""
    t = np.linspace(-1, 1, seq_length)
    x = t * 0.3
    y = t ** 2 * 0.3 - 0.3
    z = -t ** 2 * 0.1
    trajectory = np.stack([x, y, z], axis=1)
    return trajectory


def generate_simple_video(shape, trajectory, camera_idx=0, img_size=128):
    """Generate simple 2D projection video for a camera."""
    seq_length = len(trajectory)
    video = np.zeros((seq_length, img_size, img_size, 3), dtype=np.uint8)
    
    # Camera offset based on camera_idx
    camera_offsets = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.65, 0.0, 0.0]),
        np.array([0.325, 0.56, 0.0])
    ]
    offset = camera_offsets[min(camera_idx, len(camera_offsets) - 1)]
    
    colors = {'cube': (255, 0, 0), 'cylinder': (0, 255, 0), 'cone': (0, 0, 255)}
    color = colors.get(shape, (255, 255, 255))
    
    for frame_idx in range(seq_length):
        # Simple 2D projection (just x, y, ignore z)
        point_3d = trajectory[frame_idx] - offset
        
        # Map to image coordinates
        x = int((point_3d[0] + 0.5) * img_size)
        y = int((point_3d[1] + 0.5) * img_size)
        
        # Draw shape if in bounds
        if 10 < x < img_size - 10 and 10 < y < img_size - 10:
            if shape == 'cube':
                video[frame_idx, y-10:y+10, x-10:x+10] = color
            elif shape == 'cylinder':
                Y, X = np.ogrid[:img_size, :img_size]
                mask = (X - x)**2 + (Y - y)**2 <= 100
                video[frame_idx][mask] = color
            elif shape == 'cone':
                for dy in range(-10, 11):
                    width = max(0, 10 - abs(dy))
                    if 0 <= y + dy < img_size:
                        x_start = max(0, x - width)
                        x_end = min(img_size, x + width + 1)
                        video[frame_idx, y + dy, x_start:x_end] = color
    
    return video


def generate_dataset(num_samples=50):
    """Generate MAGVIT 3D dataset."""
    logger.info(f"Generating {num_samples} 3D trajectory samples...")
    
    shapes = ['cube', 'cylinder', 'cone']
    trajectory_funcs = [
        generate_linear_trajectory,
        generate_circular_trajectory,
        generate_helical_trajectory,
        generate_parabolic_trajectory
    ]
    
    trajectories_3d = []
    multi_view_videos = []
    labels = []
    
    for i in range(num_samples):
        # Select shape (cycle through 3 shapes)
        shape_idx = i % 3
        shape = shapes[shape_idx]
        
        # Select trajectory pattern (cycle through 4 patterns)
        traj_func = trajectory_funcs[i % len(trajectory_funcs)]
        trajectory = traj_func(seq_length=16)
        
        # Add small noise
        noise = np.random.normal(0, 0.02, trajectory.shape)
        noisy_trajectory = trajectory + noise
        
        # Generate multi-view video (3 cameras)
        multi_view = []
        for cam_idx in range(3):
            video = generate_simple_video(shape, noisy_trajectory, cam_idx, img_size=128)
            multi_view.append(video)
        multi_view = np.array(multi_view)  # Shape: (3, 16, 128, 128, 3)
        
        trajectories_3d.append(noisy_trajectory)
        multi_view_videos.append(multi_view)
        labels.append(shape_idx)
        
        if (i + 1) % 10 == 0:
            logger.info(f"  Generated {i + 1}/{num_samples} samples")
    
    return {
        'trajectories_3d': np.array(trajectories_3d),
        'multi_view_videos': np.array(multi_view_videos),
        'labels': np.array(labels)
    }


def create_visualizations(dataset, output_dir):
    """Create visualization plots."""
    logger.info("Creating visualizations...")
    
    trajectories_3d = dataset['trajectories_3d']
    labels = dataset['labels']
    shape_names = ['Cube', 'Cylinder', 'Cone']
    colors = ['red', 'green', 'blue']
    
    # 1. 3D Trajectory Plot
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
        ax.set_title(f'{shape_name} 3D Trajectories')
        ax.grid(True, alpha=0.3)
        ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'magvit_3d_trajectories.png', dpi=150, bbox_inches='tight')
    logger.info(f"  Saved: magvit_3d_trajectories.png")
    plt.close()
    
    # 2. 2D Error Analysis (trajectory statistics)
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
    plt.savefig(output_dir / 'magvit_3d_errors_2d.png', dpi=150, bbox_inches='tight')
    logger.info(f"  Saved: magvit_3d_errors_2d.png")
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
    
    plt.savefig(output_dir / 'magvit_3d_cameras.png', dpi=150, bbox_inches='tight')
    logger.info(f"  Saved: magvit_3d_cameras.png")
    plt.close()


def main():
    """Main function to generate MAGVIT 3D dataset."""
    logger.info("="*60)
    logger.info("MAGVIT 3D Dataset Generation (50 samples)")
    logger.info("="*60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Output directory
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    # Generate dataset
    dataset = generate_dataset(num_samples=50)
    
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
    logger.info("="*60)
    logger.info("Generation complete!")
    logger.info("="*60)
    logger.info(f"Output directory: {output_dir}")
    logger.info("Files created:")
    for f in sorted(output_dir.glob('*')):
        size_kb = f.stat().st_size / 1024
        logger.info(f"  - {f.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()

