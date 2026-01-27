"""
Dataset generator for trajectory classification and forecasting.

Generates 4 types of 3D trajectories (linear, circular, helical, parabolic),
renders them as video frames (IMAGES, not coordinates), and packages with
metadata (equations, descriptions).

Following TDD: Implementation created after tests (GREEN phase).
"""

import numpy as np
import torch
from typing import Dict, Tuple, List
from trajectory_renderer import TrajectoryRenderer, CameraParams


def generate_linear_trajectory(
    num_frames: int = 16,
    rng: np.random.Generator = None
) -> np.ndarray:
    """Generate linear (straight-line) 3D trajectory.
    
    Equation: p(t) = start + t * direction
    
    Args:
        num_frames: Number of frames in trajectory
        rng: Random number generator for reproducibility
    
    Returns:
        np.ndarray: 3D trajectory points, shape (num_frames, 3)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Random start point
    start = rng.uniform(-0.5, 0.5, size=3)
    start[2] += 2.0  # Keep Z positive (in front of camera)
    
    # Random direction (ensure 3D variation to avoid camera degeneracies)
    direction = rng.uniform(-0.3, 0.3, size=3)
    # Ensure minimum variation in each dimension (0.11 units minimum)
    for i in range(3):
        if abs(direction[i]) < 0.11:
            direction[i] = 0.11 * (1 if direction[i] >= 0 else -1)
    
    # Generate smooth linear trajectory (no added noise)
    t = np.linspace(0, 1, num_frames)
    trajectory = start + np.outer(t, direction)
    
    return trajectory


def generate_circular_trajectory(
    num_frames: int = 16,
    rng: np.random.Generator = None
) -> np.ndarray:
    """Generate circular 3D trajectory.
    
    Equation: x = r*cos(θ), y = r*sin(θ), z = constant
    
    Args:
        num_frames: Number of frames in trajectory
        rng: Random number generator for reproducibility
    
    Returns:
        np.ndarray: 3D trajectory points, shape (num_frames, 3)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Random radius
    radius = rng.uniform(0.3, 0.6)
    
    # Random center in XY plane only
    center_xy = rng.uniform(-0.2, 0.2, size=2)
    # Fixed Z (constant for all points in circle)
    center_z = rng.uniform(1.8, 2.2)
    
    # Generate perfect circle (smooth, no noise)
    theta = np.linspace(0, 2*np.pi, num_frames)
    x = radius * np.cos(theta) + center_xy[0]
    y = radius * np.sin(theta) + center_xy[1]
    z = np.full(num_frames, center_z)  # Constant Z!
    
    trajectory = np.stack([x, y, z], axis=1)
    
    return trajectory


def generate_helical_trajectory(
    num_frames: int = 16,
    rng: np.random.Generator = None
) -> np.ndarray:
    """Generate helical (spiral) 3D trajectory.
    
    Equation: x = r*cos(θ), y = r*sin(θ), z = a*t + b
    
    Args:
        num_frames: Number of frames in trajectory
        rng: Random number generator for reproducibility
    
    Returns:
        np.ndarray: 3D trajectory points, shape (num_frames, 3)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Random radius
    radius = rng.uniform(0.3, 0.5)
    
    # Random center and Z progression
    center_xy = rng.uniform(-0.2, 0.2, size=2)
    z_start = rng.uniform(1.5, 2.0)
    z_end = rng.uniform(2.5, 3.0)
    
    # Generate smooth helix (no added noise)
    theta = np.linspace(0, 4*np.pi, num_frames)  # 2 full rotations
    x = radius * np.cos(theta) + center_xy[0]
    y = radius * np.sin(theta) + center_xy[1]
    z = np.linspace(z_start, z_end, num_frames)
    
    trajectory = np.stack([x, y, z], axis=1)
    
    return trajectory


def generate_parabolic_trajectory(
    num_frames: int = 16,
    rng: np.random.Generator = None
) -> np.ndarray:
    """Generate parabolic 3D trajectory (projectile motion).
    
    Equation: 
        x(t) = x0 + vx*t  (linear horizontal)
        y(t) = y0 + vy*t  (linear lateral)
        z(t) = z0 + vz*t - 0.5*g*t²  (parabolic vertical with gravity)
    
    Args:
        num_frames: Number of frames in trajectory
        rng: Random number generator for reproducibility
    
    Returns:
        np.ndarray: 3D trajectory points, shape (num_frames, 3)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    t = np.linspace(0, 1, num_frames)
    
    # Random initial position
    x0 = rng.uniform(-0.3, 0.3)
    y0 = rng.uniform(-0.3, 0.3)
    z0 = rng.uniform(2.0, 2.3)  # Start height
    
    # Random initial velocities
    vx = rng.uniform(0.3, 0.8)  # Horizontal velocity
    vy = rng.uniform(-0.2, 0.2)  # Lateral velocity
    vz = rng.uniform(0.2, 0.6)  # Initial upward velocity
    
    # Gravity constant
    g = rng.uniform(0.8, 1.5)  # Varies gravity strength
    
    # Generate smooth projectile motion (no added noise)
    x = x0 + vx * t
    y = y0 + vy * t
    z = z0 + vz * t - 0.5 * g * (t ** 2)
    
    trajectory = np.stack([x, y, z], axis=1)
    
    return trajectory


def augment_trajectory(
    trajectory: np.ndarray,
    rng: np.random.Generator = None
) -> np.ndarray:
    """Apply augmentation to trajectory.
    
    Augmentations:
    - Add Gaussian noise
    - Small rotation
    - Small translation
    
    Args:
        trajectory: Original trajectory, shape (T, 3)
        rng: Random number generator
    
    Returns:
        np.ndarray: Augmented trajectory, shape (T, 3)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    augmented = trajectory.copy()
    
    # Add noise (different noise level than generation)
    noise_std = rng.uniform(0.01, 0.03)
    noise = rng.normal(0, noise_std, size=augmented.shape)
    augmented += noise
    
    # Small rotation (around Z axis)
    angle = rng.uniform(-np.pi/12, np.pi/12)  # ±15 degrees
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_z = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    augmented = augmented @ rotation_z.T
    
    # Small translation
    translation = rng.uniform(-0.1, 0.1, size=3)
    augmented += translation
    
    return augmented


def get_trajectory_equation(trajectory_type: int) -> str:
    """Get symbolic equation for trajectory type.
    
    Args:
        trajectory_type: 0=linear, 1=circular, 2=helical, 3=parabolic
    
    Returns:
        str: Symbolic equation
    """
    equations = {
        0: "p(t) = p₀ + v·t",  # Linear
        1: "x = r·cos(θ), y = r·sin(θ), z = c",  # Circular
        2: "x = r·cos(θ), y = r·sin(θ), z = a·t + b",  # Helical
        3: "x = a·t² + b·t + c (for each dimension)"  # Parabolic
    }
    return equations.get(trajectory_type, "Unknown")


def get_trajectory_description(trajectory_type: int) -> str:
    """Get natural language description for trajectory type.
    
    Args:
        trajectory_type: 0=linear, 1=circular, 2=helical, 3=parabolic
    
    Returns:
        str: Natural language description
    """
    descriptions = {
        0: "The object moves in a straight line with constant velocity.",
        1: "The object follows a circular path, maintaining a constant radius around a center point.",
        2: "The object traces a helical spiral, combining circular motion in the XY plane with linear motion in the Z direction.",
        3: "The object follows a parabolic trajectory with quadratic acceleration in each dimension, similar to projectile motion."
    }
    return descriptions.get(trajectory_type, "Unknown trajectory type")


def generate_dataset(
    num_samples: int = 1200,
    frames_per_video: int = 16,
    image_size: Tuple[int, int] = (64, 64),
    augmentation: bool = True,
    seed: int = 42
) -> Dict[str, any]:
    """Generate complete dataset of trajectory videos.
    
    CRITICAL: Returns IMAGES (videos as tensors), not coordinate shortcuts.
    
    Args:
        num_samples: Total number of samples to generate
        frames_per_video: Number of frames per trajectory
        image_size: (height, width) of video frames
        augmentation: Whether to apply data augmentation
        seed: Random seed for reproducibility
    
    Returns:
        Dict containing:
            - videos: torch.Tensor (N, T, 3, H, W) - RGB video frames
            - labels: torch.Tensor (N,) - class labels (0-3)
            - trajectory_3d: np.ndarray (N, T, 3) - 3D ground truth
            - equations: List[str] - symbolic equations
            - descriptions: List[str] - natural language descriptions
    """
    rng = np.random.default_rng(seed)
    
    # Trajectory generators
    generators = [
        generate_linear_trajectory,
        generate_circular_trajectory,
        generate_helical_trajectory,
        generate_parabolic_trajectory
    ]
    
    # Calculate samples per class (balanced)
    samples_per_class = num_samples // len(generators)
    
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
    
    # Generate samples for each class
    for class_id, generator in enumerate(generators):
        for _ in range(samples_per_class):
            # Generate 3D trajectory
            trajectory_3d = generator(num_frames=frames_per_video, rng=rng)
            
            # Apply augmentation if requested
            if augmentation:
                # Randomly decide whether to augment this sample
                if rng.random() < 0.8:  # 80% of samples get augmented
                    trajectory_3d = augment_trajectory(trajectory_3d, rng=rng)
            
            # Render to video frames (IMAGES, not coordinates!)
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
    
    # Shuffle dataset (maintain correspondence)
    actual_num_samples = len(all_videos)
    shuffle_indices = torch.randperm(actual_num_samples, generator=torch.Generator().manual_seed(seed))
    videos_tensor = videos_tensor[shuffle_indices]
    labels_tensor = labels_tensor[shuffle_indices]
    trajectories_array = trajectories_array[shuffle_indices.numpy()]
    all_equations = [all_equations[i] for i in shuffle_indices]
    all_descriptions = [all_descriptions[i] for i in shuffle_indices]
    
    return {
        'videos': videos_tensor,
        'labels': labels_tensor,
        'trajectory_3d': trajectories_array,
        'equations': all_equations,
        'descriptions': all_descriptions
    }

