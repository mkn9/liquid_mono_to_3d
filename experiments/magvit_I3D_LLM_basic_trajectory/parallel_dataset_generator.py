"""
Parallel dataset generator for trajectory videos.

Generates datasets using multiprocessing for 3-4× speedup.
Each worker generates samples for one trajectory class independently.

Following TDD: Implementation created after tests (GREEN phase).
"""

import numpy as np
import torch
from typing import Dict, Tuple, List
from multiprocessing import Pool, cpu_count
import functools

from dataset_generator import (
    generate_linear_trajectory,
    generate_circular_trajectory,
    generate_helical_trajectory,
    generate_parabolic_trajectory,
    augment_trajectory,
    get_trajectory_equation,
    get_trajectory_description
)
from trajectory_renderer import TrajectoryRenderer, CameraParams


def generate_single_class_dataset(
    class_id: int,
    num_samples: int,
    frames_per_video: int = 16,
    image_size: Tuple[int, int] = (64, 64),
    augmentation: bool = True,
    seed: int = 42
) -> Dict[str, any]:
    """Generate dataset for a single trajectory class.
    
    This function is designed to be called in parallel for each class.
    
    Args:
        class_id: Trajectory class (0=linear, 1=circular, 2=helical, 3=parabolic)
        num_samples: Number of samples to generate for this class
        frames_per_video: Number of frames per trajectory
        image_size: (height, width) of video frames
        augmentation: Whether to apply data augmentation
        seed: Random seed (will be offset by class_id for independence)
    
    Returns:
        Dict containing:
            - videos: torch.Tensor (N, T, 3, H, W) - RGB video frames
            - labels: torch.Tensor (N,) - class labels (all same class_id)
            - trajectory_3d: np.ndarray (N, T, 3) - 3D ground truth
            - equations: List[str] - symbolic equations
            - descriptions: List[str] - natural language descriptions
    """
    # Offset seed by class_id to ensure independence between workers
    rng = np.random.default_rng(seed + class_id * 10000)
    
    # Select generator for this class
    generators = [
        generate_linear_trajectory,
        generate_circular_trajectory,
        generate_helical_trajectory,
        generate_parabolic_trajectory
    ]
    generator = generators[class_id]
    
    # Initialize lists
    all_videos = []
    all_labels = []
    all_trajectories_3d = []
    all_equations = []
    all_descriptions = []
    
    # Setup renderer
    renderer = TrajectoryRenderer(image_size=image_size, style='dot')
    
    # Setup camera
    camera_params = CameraParams(
        position=np.array([0.0, 0.0, 0.0]),
        focal_length=800,
        image_center=(image_size[0]//2, image_size[1]//2)
    )
    
    # Generate samples for this class
    for _ in range(num_samples):
        # Generate 3D trajectory
        trajectory_3d = generator(num_frames=frames_per_video, rng=rng)
        
        # Apply augmentation if requested
        if augmentation:
            if rng.random() < 0.8:  # 80% of samples get augmented
                trajectory_3d = augment_trajectory(trajectory_3d, rng=rng)
        
        # Render to video frames
        video = renderer.render_video(trajectory_3d, camera_params)
        
        # Store
        all_videos.append(video)
        all_labels.append(class_id)
        all_trajectories_3d.append(trajectory_3d)
        all_equations.append(get_trajectory_equation(class_id))
        all_descriptions.append(get_trajectory_description(class_id))
    
    # Convert to tensors/arrays
    videos_tensor = torch.stack(all_videos)  # (N, T, 3, H, W)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)  # (N,)
    trajectories_array = np.array(all_trajectories_3d)  # (N, T, 3)
    
    return {
        'videos': videos_tensor,
        'labels': labels_tensor,
        'trajectory_3d': trajectories_array,
        'equations': all_equations,
        'descriptions': all_descriptions
    }


def merge_class_datasets(
    class_datasets: List[Dict[str, any]],
    shuffle_seed: int = 42
) -> Dict[str, any]:
    """Merge datasets from multiple classes into single dataset.
    
    Args:
        class_datasets: List of datasets (one per class)
        shuffle_seed: Seed for shuffling merged dataset
    
    Returns:
        Merged dataset with all classes combined and shuffled
    """
    # Concatenate all datasets
    all_videos = torch.cat([ds['videos'] for ds in class_datasets], dim=0)
    all_labels = torch.cat([ds['labels'] for ds in class_datasets], dim=0)
    all_trajectories = np.concatenate([ds['trajectory_3d'] for ds in class_datasets], axis=0)
    
    # Flatten lists
    all_equations = []
    all_descriptions = []
    for ds in class_datasets:
        all_equations.extend(ds['equations'])
        all_descriptions.extend(ds['descriptions'])
    
    # Shuffle dataset (maintain correspondence)
    num_samples = len(all_videos)
    shuffle_indices = torch.randperm(num_samples, generator=torch.Generator().manual_seed(shuffle_seed))
    
    all_videos = all_videos[shuffle_indices]
    all_labels = all_labels[shuffle_indices]
    all_trajectories = all_trajectories[shuffle_indices.numpy()]
    all_equations = [all_equations[i] for i in shuffle_indices]
    all_descriptions = [all_descriptions[i] for i in shuffle_indices]
    
    return {
        'videos': all_videos,
        'labels': all_labels,
        'trajectory_3d': all_trajectories,
        'equations': all_equations,
        'descriptions': all_descriptions
    }


def validate_merged_dataset(
    dataset: Dict[str, any],
    expected_samples: int
) -> bool:
    """Validate merged dataset for consistency and completeness.
    
    Args:
        dataset: Merged dataset to validate
        expected_samples: Expected number of samples
    
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check sample count
        num_videos = len(dataset['videos'])
        if num_videos != expected_samples:
            print(f"Sample count mismatch: {num_videos} != {expected_samples}")
            return False
        
        # Check shape consistency
        num_labels = len(dataset['labels'])
        num_trajectories = len(dataset['trajectory_3d'])
        num_equations = len(dataset['equations'])
        num_descriptions = len(dataset['descriptions'])
        
        if not (num_videos == num_labels == num_trajectories == num_equations == num_descriptions):
            print(f"Shape mismatch: videos={num_videos}, labels={num_labels}, "
                  f"trajectories={num_trajectories}, equations={num_equations}, "
                  f"descriptions={num_descriptions}")
            return False
        
        # Check class distribution (should be balanced)
        labels = dataset['labels'].numpy()
        for class_id in range(4):
            count = (labels == class_id).sum()
            expected_per_class = expected_samples // 4
            if abs(count - expected_per_class) > 1:  # Allow ±1 for rounding
                print(f"Class {class_id} imbalance: {count} != {expected_per_class}")
                return False
        
        # Check for finite values
        if not torch.all(torch.isfinite(dataset['videos'])):
            print("Videos contain NaN or Inf")
            return False
        
        return True
    
    except Exception as e:
        print(f"Validation error: {e}")
        return False


def generate_dataset_parallel(
    num_samples: int = 1200,
    frames_per_video: int = 16,
    image_size: Tuple[int, int] = (64, 64),
    augmentation: bool = True,
    seed: int = 42,
    num_workers: int = None
) -> Dict[str, any]:
    """Generate complete dataset using parallel processing.
    
    Distributes generation across multiple workers (one per trajectory class).
    Provides 3-4× speedup compared to sequential generation.
    
    Args:
        num_samples: Total number of samples to generate
        frames_per_video: Number of frames per trajectory
        image_size: (height, width) of video frames
        augmentation: Whether to apply data augmentation
        seed: Random seed for reproducibility
        num_workers: Number of parallel workers (default: min(4, cpu_count))
    
    Returns:
        Dict containing:
            - videos: torch.Tensor (N, T, 3, H, W) - RGB video frames
            - labels: torch.Tensor (N,) - class labels (0-3)
            - trajectory_3d: np.ndarray (N, T, 3) - 3D ground truth
            - equations: List[str] - symbolic equations
            - descriptions: List[str] - natural language descriptions
    """
    if num_workers is None:
        num_workers = min(4, cpu_count())  # 4 classes max
    
    # Calculate samples per class (balanced)
    samples_per_class = num_samples // 4
    
    # Create partial function with fixed parameters
    generate_func = functools.partial(
        generate_single_class_dataset,
        num_samples=samples_per_class,
        frames_per_video=frames_per_video,
        image_size=image_size,
        augmentation=augmentation,
        seed=seed
    )
    
    # Generate datasets in parallel (one worker per class)
    with Pool(processes=num_workers) as pool:
        class_datasets = pool.map(generate_func, range(4))
    
    # Merge all class datasets
    merged_dataset = merge_class_datasets(class_datasets, shuffle_seed=seed)
    
    # Validate merged dataset
    expected_samples = samples_per_class * 4
    if not validate_merged_dataset(merged_dataset, expected_samples):
        raise ValueError("Dataset validation failed after merge")
    
    return merged_dataset

