#!/usr/bin/env python3
"""
Visualize 2D camera views from the MAGVIT 3D dataset.

Shows examples of the multi-camera rendered views for different shapes and trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 70)
    logger.info("MAGVIT 3D Camera Views Visualization")
    logger.info("=" * 70)
    
    # Load dataset
    dataset_path = Path(__file__).parent / 'results' / 'magvit_3d_dataset.npz'
    data = np.load(dataset_path)
    
    multi_view_videos = data['multi_view_videos']  # (50, 3, 16, 128, 128, 3)
    trajectories_3d = data['trajectories_3d']      # (50, 16, 3)
    labels = data['labels']                        # (50,)
    
    shape_names = ['Cube', 'Cylinder', 'Cone']
    output_dir = Path(__file__).parent / 'results'
    
    logger.info(f"Dataset shape: {multi_view_videos.shape}")
    logger.info(f"  Samples: {multi_view_videos.shape[0]}")
    logger.info(f"  Cameras: {multi_view_videos.shape[1]}")
    logger.info(f"  Frames: {multi_view_videos.shape[2]}")
    logger.info(f"  Image size: {multi_view_videos.shape[3]}x{multi_view_videos.shape[4]}")
    logger.info("")
    
    # Visualization 1: Single sample across 3 cameras at different time steps
    logger.info("Creating visualization 1: Multi-camera views at different time steps...")
    sample_idx = 0  # First cube sample
    time_steps = [0, 5, 10, 15]  # Beginning, middle, near end, end
    
    fig, axes = plt.subplots(len(time_steps), 3, figsize=(12, 16))
    fig.suptitle(f'Multi-Camera Views: {shape_names[labels[sample_idx]]} (Sample {sample_idx})', 
                 fontsize=16, fontweight='bold')
    
    for t_idx, t in enumerate(time_steps):
        for cam_idx in range(3):
            ax = axes[t_idx, cam_idx]
            img = multi_view_videos[sample_idx, cam_idx, t]
            ax.imshow(img.astype(np.uint8))
            ax.set_title(f'Camera {cam_idx+1}, Frame {t}')
            ax.axis('off')
            
            # Add 3D position info
            pos_3d = trajectories_3d[sample_idx, t]
            ax.text(0.5, -0.05, f'3D: ({pos_3d[0]:.2f}, {pos_3d[1]:.2f}, {pos_3d[2]:.2f})',
                   transform=ax.transAxes, ha='center', va='top', fontsize=8)
    
    plt.tight_layout()
    output_file = output_dir / 'camera_views_multitime.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"✅ Saved: {output_file}")
    plt.close()
    
    # Visualization 2: Different shapes from Camera 1 at a single time step
    logger.info("Creating visualization 2: Different shapes from Camera 1...")
    time_step = 8  # Middle frame
    sample_indices = [0, 1, 2]  # One of each shape (cube, cylinder, cone)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Camera 1 Views: Different Shapes at Frame {time_step}', 
                 fontsize=16, fontweight='bold')
    
    for i, sample_idx in enumerate(sample_indices):
        ax = axes[i]
        img = multi_view_videos[sample_idx, 0, time_step]  # Camera 1 (index 0)
        ax.imshow(img.astype(np.uint8))
        ax.set_title(f'{shape_names[labels[sample_idx]]} (Sample {sample_idx})')
        ax.axis('off')
        
        pos_3d = trajectories_3d[sample_idx, time_step]
        ax.text(0.5, -0.05, f'3D: ({pos_3d[0]:.2f}, {pos_3d[1]:.2f}, {pos_3d[2]:.2f})',
               transform=ax.transAxes, ha='center', va='top', fontsize=10)
    
    plt.tight_layout()
    output_file = output_dir / 'camera_views_shapes.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"✅ Saved: {output_file}")
    plt.close()
    
    # Visualization 3: All 3 cameras for a single shape at a single time step
    logger.info("Creating visualization 3: All cameras for a single shape...")
    sample_idx = 1  # Cylinder
    time_step = 10
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'{shape_names[labels[sample_idx]]} from 3 Camera Perspectives (Frame {time_step})', 
                 fontsize=16, fontweight='bold')
    
    for cam_idx in range(3):
        ax = axes[cam_idx]
        img = multi_view_videos[sample_idx, cam_idx, time_step]
        ax.imshow(img.astype(np.uint8))
        ax.set_title(f'Camera {cam_idx+1}')
        ax.axis('off')
    
    pos_3d = trajectories_3d[sample_idx, time_step]
    fig.text(0.5, 0.02, f'Object 3D Position: ({pos_3d[0]:.2f}, {pos_3d[1]:.2f}, {pos_3d[2]:.2f})',
            ha='center', fontsize=12)
    
    plt.tight_layout()
    output_file = output_dir / 'camera_views_perspectives.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"✅ Saved: {output_file}")
    plt.close()
    
    # Visualization 4: Trajectory sequence montage (single camera, multiple frames)
    logger.info("Creating visualization 4: Trajectory sequence montage...")
    sample_idx = 2  # Cone
    cam_idx = 0  # Camera 1
    frame_indices = [0, 3, 6, 9, 12, 15]  # Sample frames
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'{shape_names[labels[sample_idx]]} Trajectory Sequence (Camera 1)', 
                 fontsize=16, fontweight='bold')
    
    axes_flat = axes.flatten()
    for i, frame_idx in enumerate(frame_indices):
        ax = axes_flat[i]
        img = multi_view_videos[sample_idx, cam_idx, frame_idx]
        ax.imshow(img.astype(np.uint8))
        ax.set_title(f'Frame {frame_idx}')
        ax.axis('off')
        
        pos_3d = trajectories_3d[sample_idx, frame_idx]
        ax.text(0.5, -0.05, f'({pos_3d[0]:.2f}, {pos_3d[1]:.2f}, {pos_3d[2]:.2f})',
               transform=ax.transAxes, ha='center', va='top', fontsize=9)
    
    plt.tight_layout()
    output_file = output_dir / 'camera_views_sequence.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"✅ Saved: {output_file}")
    plt.close()
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("Summary:")
    logger.info(f"  Total samples: {multi_view_videos.shape[0]}")
    logger.info(f"  Cameras per sample: {multi_view_videos.shape[1]}")
    logger.info(f"  Frames per camera: {multi_view_videos.shape[2]}")
    logger.info(f"  Total images in dataset: {np.prod(multi_view_videos.shape[:3])}")
    logger.info("")
    logger.info("Generated visualizations:")
    logger.info("  1. camera_views_multitime.png - Multi-camera views at different time steps")
    logger.info("  2. camera_views_shapes.png - Different shapes from Camera 1")
    logger.info("  3. camera_views_perspectives.png - All cameras for a single shape")
    logger.info("  4. camera_views_sequence.png - Trajectory sequence montage")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()

