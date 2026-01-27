#!/usr/bin/env python3
"""
Comprehensive dataset quality validation.

Checks:
1. Visual quality of rendered videos
2. Label correctness 
3. Trajectory patterns match expected shapes
4. Data integrity
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print("="*70)
print("DATASET QUALITY VALIDATION")
print("="*70)
print()

# Load dataset
dataset_path = Path("results/20260124_1546_full_dataset.npz")
print(f"Loading: {dataset_path}")
print(f"Generated: 2026-01-24 15:46 (Jan 24, 3:46 PM)")
print()

data = np.load(dataset_path)
videos = data['videos']  # (200, 16, 3, 64, 64)
labels = data['labels']
trajectory_3d = data['trajectory_3d']  # (200, 16, 3)
equations = data['equations']
descriptions = data['descriptions']

print(f"Videos shape: {videos.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Trajectories shape: {trajectory_3d.shape}")
print()

# Class names
class_names = ['Linear', 'Circular', 'Helical', 'Parabolic']

print("="*70)
print("CHECK 1: VISUAL QUALITY")
print("="*70)
print()

# Create visualization of first sample from each class
fig, axes = plt.subplots(4, 16, figsize=(24, 8))
fig.suptitle('Dataset Visual Inspection - First Sample per Class (All 16 Frames)', fontsize=16)

for class_id in range(4):
    # Get first sample of this class
    indices = np.where(labels == class_id)[0]
    idx = indices[0]
    video = videos[idx]  # (16, 3, 64, 64)
    
    print(f"Class {class_id} ({class_names[class_id]}):")
    print(f"  Total samples: {len(indices)}")
    print(f"  Video range: [{video.min():.3f}, {video.max():.3f}]")
    
    # Check if video has variation across frames
    frame_means = [video[t].mean() for t in range(16)]
    frame_variation = np.std(frame_means)
    print(f"  Frame variation (std of means): {frame_variation:.4f}")
    
    # Plot all 16 frames
    for t in range(16):
        frame = video[t].transpose(1, 2, 0)  # (H, W, 3)
        axes[class_id, t].imshow(frame, vmin=0, vmax=1)
        axes[class_id, t].axis('off')
        
        if t == 0:
            axes[class_id, t].set_ylabel(
                f'Class {class_id}\n{class_names[class_id]}',
                fontsize=10
            )
        
        if class_id == 0:
            axes[class_id, t].set_title(f'Frame {t}', fontsize=8)
    
    print()

plt.tight_layout()
output_path = Path("results/dataset_visual_inspection.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✅ Saved: {output_path}")
print()

print("="*70)
print("CHECK 2: TRAJECTORY PATTERNS")
print("="*70)
print()

# Analyze trajectory patterns
fig, axes = plt.subplots(4, 3, figsize=(15, 12))
fig.suptitle('3D Trajectory Patterns (First Sample per Class)', fontsize=16)

for class_id in range(4):
    indices = np.where(labels == class_id)[0]
    idx = indices[0]
    traj = trajectory_3d[idx]  # (16, 3)
    
    print(f"Class {class_id} ({class_names[class_id]}):")
    print(f"  Equation: {equations[idx]}")
    print(f"  X range: [{traj[:, 0].min():.3f}, {traj[:, 0].max():.3f}]")
    print(f"  Y range: [{traj[:, 1].min():.3f}, {traj[:, 1].max():.3f}]")
    print(f"  Z range: [{traj[:, 2].min():.3f}, {traj[:, 2].max():.3f}]")
    
    # Calculate motion metrics
    displacement = np.linalg.norm(traj[-1] - traj[0])
    path_length = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
    print(f"  Displacement: {displacement:.3f}")
    print(f"  Path length: {path_length:.3f}")
    
    # XY projection
    axes[class_id, 0].plot(traj[:, 0], traj[:, 1], 'b-o', markersize=3)
    axes[class_id, 0].plot(traj[0, 0], traj[0, 1], 'go', markersize=8, label='Start')
    axes[class_id, 0].plot(traj[-1, 0], traj[-1, 1], 'ro', markersize=8, label='End')
    axes[class_id, 0].set_xlabel('X')
    axes[class_id, 0].set_ylabel('Y')
    axes[class_id, 0].set_title(f'{class_names[class_id]} - XY Plane')
    axes[class_id, 0].legend(fontsize=8)
    axes[class_id, 0].grid(True, alpha=0.3)
    axes[class_id, 0].axis('equal')
    
    # XZ projection
    axes[class_id, 1].plot(traj[:, 0], traj[:, 2], 'b-o', markersize=3)
    axes[class_id, 1].plot(traj[0, 0], traj[0, 2], 'go', markersize=8, label='Start')
    axes[class_id, 1].plot(traj[-1, 0], traj[-1, 2], 'ro', markersize=8, label='End')
    axes[class_id, 1].set_xlabel('X')
    axes[class_id, 1].set_ylabel('Z')
    axes[class_id, 1].set_title(f'{class_names[class_id]} - XZ Plane')
    axes[class_id, 1].legend(fontsize=8)
    axes[class_id, 1].grid(True, alpha=0.3)
    
    # 3D trajectory
    ax = fig.add_subplot(4, 3, class_id*3 + 3, projection='3d')
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'b-o', markersize=3)
    ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], c='g', s=100, label='Start')
    ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], c='r', s=100, label='End')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{class_names[class_id]} - 3D')
    ax.legend(fontsize=8)
    
    print()

plt.tight_layout()
output_path = Path("results/trajectory_patterns_validation.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✅ Saved: {output_path}")
print()

print("="*70)
print("CHECK 3: CLASS SEPARABILITY")
print("="*70)
print()

# Check if classes have distinct characteristics
print("Analyzing class separability...")
print()

for class_id in range(4):
    indices = np.where(labels == class_id)[0]
    trajs = trajectory_3d[indices]  # (N, 16, 3)
    
    # Calculate statistics
    displacements = np.array([
        np.linalg.norm(traj[-1] - traj[0]) 
        for traj in trajs
    ])
    path_lengths = np.array([
        np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
        for traj in trajs
    ])
    
    print(f"Class {class_id} ({class_names[class_id]}):")
    print(f"  Displacement: {displacements.mean():.3f} ± {displacements.std():.3f}")
    print(f"  Path length:  {path_lengths.mean():.3f} ± {path_lengths.std():.3f}")
    print()

print("="*70)
print("CHECK 4: DATA INTEGRITY")
print("="*70)
print()

# Check for NaN or Inf
has_nan = np.isnan(videos).any() or np.isnan(trajectory_3d).any()
has_inf = np.isinf(videos).any() or np.isinf(trajectory_3d).any()

print(f"NaN values: {'❌ FOUND' if has_nan else '✅ None'}")
print(f"Inf values: {'❌ FOUND' if has_inf else '✅ None'}")
print()

# Check for duplicate samples
print("Checking for duplicates...")
video_hashes = [videos[i].tobytes() for i in range(len(videos))]
unique_hashes = len(set(video_hashes))
print(f"Unique samples: {unique_hashes}/{len(videos)}")
if unique_hashes < len(videos):
    print(f"⚠️  Found {len(videos) - unique_hashes} duplicate samples!")
else:
    print("✅ No duplicates found")
print()

# Check value ranges
print("Value range checks:")
print(f"  Videos in [0, 1]: {'✅ Yes' if (videos.min() >= 0 and videos.max() <= 1) else '❌ No'}")
print(f"  Labels in [0, 3]: {'✅ Yes' if (labels.min() >= 0 and labels.max() <= 3) else '❌ No'}")
print()

print("="*70)
print("SUMMARY")
print("="*70)
print()

# Overall assessment
issues = []

if has_nan or has_inf:
    issues.append("Contains NaN/Inf values")

if unique_hashes < len(videos):
    issues.append(f"{len(videos) - unique_hashes} duplicate samples")

if videos.min() < 0 or videos.max() > 1:
    issues.append("Video values outside [0, 1] range")

# Check frame variation
low_variation_classes = []
for class_id in range(4):
    indices = np.where(labels == class_id)[0]
    video = videos[indices[0]]
    frame_means = [video[t].mean() for t in range(16)]
    frame_variation = np.std(frame_means)
    if frame_variation < 0.001:
        low_variation_classes.append(class_id)

if low_variation_classes:
    issues.append(f"Classes with low temporal variation: {low_variation_classes}")

if len(issues) == 0:
    print("✅ DATA QUALITY: GOOD")
    print()
    print("No major issues detected. Data appears suitable for training.")
    print()
    print("Visual inspection files created:")
    print("  - results/dataset_visual_inspection.png")
    print("  - results/trajectory_patterns_validation.png")
    print()
    print("⚠️  IMPORTANT: Please visually inspect the PNG files to verify:")
    print("   1. Trajectories are visible in video frames")
    print("   2. Motion is smooth across frames")
    print("   3. Trajectory patterns match expected shapes")
else:
    print("⚠️  DATA QUALITY: ISSUES FOUND")
    print()
    for issue in issues:
        print(f"  ❌ {issue}")
    print()
    print("Recommendation: Regenerate dataset with current code")

print()
print("="*70)
print("VALIDATION COMPLETE - REVIEW VISUALIZATIONS")
print("="*70)

