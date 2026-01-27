#!/usr/bin/env python3
"""
Generate dataset using the validated three-layer system with auto-framing.

This uses the CORRECT, most recent code:
- auto_camera_framing.py (auto framing)
- multi_camera_validation.py (three-layer validation)
- Proper file naming convention (YYYYMMDD_HHMM_description)
"""

import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict

from dataset_generator import (
    generate_linear_trajectory,
    generate_circular_trajectory,
    generate_helical_trajectory,
    generate_parabolic_trajectory,
    augment_trajectory,
    get_trajectory_equation,
    get_trajectory_description
)
from auto_camera_framing import compute_camera_params, validate_camera_framing
from trajectory_renderer import TrajectoryRenderer


def generate_validated_dataset(
    num_samples: int = 200,
    frames_per_video: int = 16,
    image_size: Tuple[int, int] = (64, 64),
    coverage_ratio: float = 0.7,
    augmentation: bool = True,
    augmentation_prob: float = 0.8,
    noise_scale: float = 0.2,
    seed: int = 42
) -> Dict[str, any]:
    """
    Generate dataset with automatic camera framing validation.
    
    Args:
        num_samples: Total samples to generate (will be balanced across 4 classes)
        frames_per_video: Number of frames per trajectory
        image_size: (height, width) of video frames
        coverage_ratio: Fraction of frame to fill with trajectory (0.7 = 70%)
        augmentation: Whether to apply noise/rotation/translation augmentation
        augmentation_prob: Probability of augmenting each sample (default 0.8)
        noise_scale: Scale factor for noise level (1.0 = full, 0.2 = 20% of original)
        seed: Random seed for reproducibility
    
    Returns:
        Dict containing:
            - videos: torch.Tensor (N, T, 3, H, W) - RGB video frames
            - labels: torch.Tensor (N,) - class labels
            - trajectory_3d: np.ndarray (N, T, 3) - 3D ground truth
            - equations: List[str] - symbolic equations
            - descriptions: List[str] - natural language descriptions
            - camera_params: List[dict] - camera parameters used for each sample
            - framing_validation: List[dict] - validation results for each sample
    """
    rng = np.random.default_rng(seed)
    
    # Trajectory generators
    generators = [
        generate_linear_trajectory,
        generate_circular_trajectory,
        generate_helical_trajectory,
        generate_parabolic_trajectory
    ]
    class_names = ['Linear', 'Circular', 'Helical', 'Parabolic']
    
    # Storage
    all_videos = []
    all_labels = []
    all_trajectories_3d = []
    all_equations = []
    all_descriptions = []
    all_camera_params = []
    all_framing_validation = []
    
    # Renderer
    renderer = TrajectoryRenderer(image_size=image_size, style='dot')
    
    # Generate samples per class
    samples_per_class = num_samples // 4
    
    print(f"Generating {num_samples} samples ({samples_per_class} per class)")
    print(f"Settings: {frames_per_video} frames, {image_size[0]}×{image_size[1]} pixels")
    print(f"Coverage ratio: {coverage_ratio} (trajectory fills {coverage_ratio*100:.0f}% of frame)")
    if augmentation:
        print(f"Augmentation: ENABLED ({augmentation_prob*100:.0f}% of samples)")
        base_noise = "0.01-0.03"
        scaled_min = 0.01 * noise_scale
        scaled_max = 0.03 * noise_scale
        print(f"  - Noise: Gaussian std {scaled_min:.4f}-{scaled_max:.4f} (scale {noise_scale:.1f}×)")
        print(f"  - Rotation: ±15°")
        print(f"  - Translation: ±0.1")
    else:
        print(f"Augmentation: DISABLED (perfect curves)")
    print()
    
    failed_samples = 0
    
    for class_id in range(4):
        print(f"Class {class_id} ({class_names[class_id]}):", end=" ", flush=True)
        
        generator = generators[class_id]
        class_samples = 0
        attempts = 0
        max_attempts = samples_per_class * 3  # Allow retries
        
        while class_samples < samples_per_class and attempts < max_attempts:
            attempts += 1
            
            # Generate trajectory
            trajectory_3d = generator(num_frames=frames_per_video, rng=rng)
            
            # Apply augmentation if requested
            if augmentation and rng.random() < augmentation_prob:
                # Custom augmentation with scaled noise
                augmented = trajectory_3d.copy()
                
                # Add scaled Gaussian noise
                base_noise_std = rng.uniform(0.01, 0.03)
                noise_std = base_noise_std * noise_scale  # Scale down noise
                noise = rng.normal(0, noise_std, size=augmented.shape)
                augmented += noise
                
                # Small rotation (around Z axis) - keep full rotation
                angle = rng.uniform(-np.pi/12, np.pi/12)  # ±15 degrees
                rotation_z = np.array([
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1]
                ])
                augmented = augmented @ rotation_z.T
                
                # Small translation - keep full translation
                translation = rng.uniform(-0.1, 0.1, size=3)
                augmented += translation
                
                trajectory_3d = augmented
            
            # Compute optimal camera parameters with auto-framing
            # (AFTER augmentation so framing adapts to augmented trajectory)
            camera_params = compute_camera_params(
                trajectory_3d,
                image_size=image_size,
                coverage_ratio=coverage_ratio,
                focal_length_strategy="auto"
            )
            
            # Validate framing
            validation = validate_camera_framing(
                trajectory_3d,
                camera_params,
                image_size,
                min_visible_ratio=0.9  # 90% of points must be visible
            )
            
            # Only accept if validation passes
            if not validation['is_valid']:
                failed_samples += 1
                continue
            
            # Render video with validated camera
            video = renderer.render_video(trajectory_3d, camera_params)
            
            # Store
            all_videos.append(video)
            all_labels.append(class_id)
            all_trajectories_3d.append(torch.from_numpy(trajectory_3d))
            all_equations.append(get_trajectory_equation(class_id))
            all_descriptions.append(get_trajectory_description(class_id))
            all_camera_params.append({
                'position': camera_params.position.tolist(),
                'focal_length': camera_params.focal_length,
                'image_center': camera_params.image_center
            })
            all_framing_validation.append(validation)
            
            class_samples += 1
            
            if class_samples % 10 == 0:
                print(f"{class_samples}", end="...", flush=True)
        
        print(f" ✓ {class_samples} samples ({attempts} attempts, {attempts - class_samples} rejected)")
    
    print()
    print(f"Total failed validations: {failed_samples}")
    print()
    
    # Stack into tensors
    videos_tensor = torch.stack(all_videos)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    traj_tensor = torch.stack(all_trajectories_3d)
    
    # Shuffle
    indices = torch.randperm(len(videos_tensor))
    videos_tensor = videos_tensor[indices]
    labels_tensor = labels_tensor[indices]
    traj_tensor = traj_tensor[indices]
    all_equations = [all_equations[i] for i in indices]
    all_descriptions = [all_descriptions[i] for i in indices]
    all_camera_params = [all_camera_params[i] for i in indices]
    all_framing_validation = [all_framing_validation[i] for i in indices]
    
    return {
        'videos': videos_tensor,
        'labels': labels_tensor,
        'trajectory_3d': traj_tensor.numpy(),
        'equations': np.array(all_equations),
        'descriptions': np.array(all_descriptions),
        'camera_params': all_camera_params,
        'framing_validation': all_framing_validation
    }


if __name__ == "__main__":
    import time
    
    print("="*70)
    print("VALIDATED DATASET GENERATION")
    print("="*70)
    print()
    
    start = time.time()
    
    # Generate dataset with augmentation (reduced noise)
    dataset = generate_validated_dataset(
        num_samples=200,
        frames_per_video=16,
        image_size=(64, 64),
        coverage_ratio=0.7,
        augmentation=True,
        augmentation_prob=0.8,
        noise_scale=0.2,  # 20% of original noise
        seed=42
    )
    
    elapsed = time.time() - start
    
    print(f"✅ Generation complete in {elapsed:.1f} seconds")
    print()
    
    # Save with proper naming convention
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = f"results/{timestamp}_dataset_200_validated.npz"
    
    # Save (exclude camera_params and framing_validation from npz for size)
    np.savez_compressed(
        output_path,
        videos=dataset['videos'].numpy(),
        labels=dataset['labels'].numpy(),
        trajectory_3d=dataset['trajectory_3d'],
        equations=dataset['equations'],
        descriptions=dataset['descriptions']
    )
    
    print(f"✅ Saved: {output_path}")
    print()
    
    # Print summary statistics
    print("Dataset summary:")
    print(f"  Videos: {dataset['videos'].shape}")
    print(f"  Labels: {dataset['labels'].shape}")
    print(f"  Unique samples: {len(torch.unique(dataset['videos'].reshape(len(dataset['videos']), -1), dim=0))}")
    print()
    
    # Framing validation summary
    visible_ratios = [v['visible_ratio'] for v in dataset['framing_validation']]
    print(f"Framing quality:")
    print(f"  Min visible ratio: {min(visible_ratios):.3f}")
    print(f"  Mean visible ratio: {np.mean(visible_ratios):.3f}")
    print(f"  All samples >90% visible: {all(r >= 0.9 for r in visible_ratios)}")
    print()
    
    print("✅ Dataset ready for MAGVIT training!")

