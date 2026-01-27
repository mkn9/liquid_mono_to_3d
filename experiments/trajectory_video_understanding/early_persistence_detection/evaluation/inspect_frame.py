#!/usr/bin/env python3
"""Inspect specific frames from a sample to debug rendering issues."""

import torch
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def inspect_sample_frame(sample_id, frame_idx, data_dir):
    """Load and visualize a specific frame from a sample."""
    # Load video
    video_path = data_dir / f"augmented_traj_{sample_id:05d}.pt"
    json_path = data_dir / f"augmented_traj_{sample_id:05d}.json"
    
    video = torch.load(video_path)
    if video.dim() == 5:
        video = video.squeeze(0)
    
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    
    # Get the specific frame
    frame = video[frame_idx]  # (C, H, W)
    frame_np = frame.permute(1, 2, 0).cpu().numpy()
    
    # Analyze color channels
    r_channel = frame[0].cpu().numpy()
    g_channel = frame[1].cpu().numpy()
    b_channel = frame[2].cpu().numpy()
    
    print(f"Sample {sample_id}, Frame {frame_idx}")
    print(f"{'='*60}")
    print(f"Shape: {frame.shape}")
    print(f"Min/Max: {frame.min():.3f} / {frame.max():.3f}")
    print()
    
    # Find bright pixels (potential spheres)
    brightness = frame.mean(dim=0).cpu().numpy()
    bright_pixels = brightness > 0.5
    
    if bright_pixels.any():
        print(f"Found {bright_pixels.sum()} bright pixels")
        
        # Analyze colors of bright pixels
        bright_r = r_channel[bright_pixels]
        bright_g = g_channel[bright_pixels]
        bright_b = b_channel[bright_pixels]
        
        print(f"\nBright pixel colors:")
        print(f"  Red   (mean): {bright_r.mean():.3f}, (min/max): {bright_r.min():.3f}/{bright_r.max():.3f}")
        print(f"  Green (mean): {bright_g.mean():.3f}, (min/max): {bright_g.min():.3f}/{bright_g.max():.3f}")
        print(f"  Blue  (mean): {bright_b.mean():.3f}, (min/max): {bright_b.min():.3f}/{bright_b.max():.3f}")
        
        # Check if there are distinct color clusters
        print(f"\nColor analysis:")
        white_pixels = (bright_r > 0.7) & (bright_g > 0.7) & (bright_b > 0.7)
        red_pixels = (bright_r > 0.7) & (bright_g < 0.5) & (bright_b < 0.5)
        
        print(f"  White-ish pixels: {white_pixels.sum()}")
        print(f"  Red-ish pixels: {red_pixels.sum()}")
    
    # Check metadata
    print(f"\nMetadata:")
    print(f"  Transient frames: {metadata.get('transient_frames', [])}")
    print(f"  Frame {frame_idx} has transient: {frame_idx in metadata.get('transient_frames', [])}")
    
    # Visualize
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # RGB image
    axes[0].imshow(np.clip(frame_np, 0, 1))
    axes[0].set_title(f'Frame {frame_idx} - RGB')
    axes[0].axis('off')
    
    # Individual channels
    axes[1].imshow(r_channel, cmap='Reds', vmin=0, vmax=1)
    axes[1].set_title('Red Channel')
    axes[1].axis('off')
    
    axes[2].imshow(g_channel, cmap='Greens', vmin=0, vmax=1)
    axes[2].set_title('Green Channel')
    axes[2].axis('off')
    
    axes[3].imshow(b_channel, cmap='Blues', vmin=0, vmax=1)
    axes[3].set_title('Blue Channel')
    axes[3].axis('off')
    
    plt.tight_layout()
    output_path = Path(f'sample_{sample_id:05d}_frame_{frame_idx}_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    plt.close()
    
    return frame_np


if __name__ == "__main__":
    data_dir = Path('/Users/mike/Dropbox/Documents/Machine_Learning/LLM_and_Multimodal_Models/repos/mono_to_3d/experiments/trajectory_video_understanding/persistence_augmented_dataset/output_samples')
    
    # Inspect sample 576, frame 31
    print("Analyzing Sample 576, Frame 31 (the 'two white spheres' frame)")
    print("="*60)
    frame = inspect_sample_frame(576, 31, data_dir)

