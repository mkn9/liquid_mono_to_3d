"""
Automatic Camera Framing System

Automatically computes optimal camera parameters to frame 3D trajectories.
Includes validation and augmentation capabilities.
"""
import numpy as np
from typing import Tuple, Dict, Any, List
from trajectory_renderer import CameraParams, TrajectoryRenderer


def compute_camera_params(
    trajectory_3d: np.ndarray,
    image_size: Tuple[int, int] = (64, 64),
    coverage_ratio: float = 0.7,
    focal_length_strategy: str = "auto"
) -> CameraParams:
    """
    Automatically compute camera parameters to optimally frame a trajectory.
    
    Args:
        trajectory_3d: 3D trajectory points, shape (T, 3)
        image_size: Output image dimensions (height, width)
        coverage_ratio: Fraction of frame to fill (0.5-0.9 recommended)
        focal_length_strategy: "auto" to calculate, "fixed" for default
    
    Returns:
        CameraParams with optimal position and focal length
    """
    # 1. Compute trajectory bounding box
    bbox_min = trajectory_3d.min(axis=0)
    bbox_max = trajectory_3d.max(axis=0)
    center = (bbox_min + bbox_max) / 2
    size = bbox_max - bbox_min
    
    # 2. Get maximum extent in X-Y plane
    max_extent = max(size[0], size[1], 0.1)  # At least 0.1 to avoid division by zero
    
    # 3. Position camera to look at trajectory center
    # Move camera back in Z to see the entire trajectory
    depth_extent = size[2]
    camera_z = bbox_min[2] - max(depth_extent * 1.5, 1.0)  # Behind trajectory
    camera_pos = np.array([center[0], center[1], camera_z])
    
    # 4. Calculate focal length for desired coverage
    if focal_length_strategy == "auto":
        # Distance from camera to trajectory center
        distance = center[2] - camera_z
        # Calculate focal length so trajectory fills coverage_ratio of frame
        # tan(FOV/2) = (extent/2) / distance
        # focal_length = (image_extent * distance) / trajectory_extent
        desired_image_extent = min(image_size) * coverage_ratio
        focal_length = (desired_image_extent * distance) / max_extent
        # Clamp to reasonable range
        focal_length = np.clip(focal_length, 50, 500)
    else:
        focal_length = 150.0  # Conservative default
    
    return CameraParams(
        position=camera_pos,
        focal_length=float(focal_length),
        image_center=(image_size[1]//2, image_size[0]//2)  # (cx, cy)
    )


def validate_camera_framing(
    trajectory_3d: np.ndarray,
    camera_params: CameraParams,
    image_size: Tuple[int, int],
    min_visible_ratio: float = 0.9
) -> Dict[str, Any]:
    """
    Validate that camera parameters provide good framing.
    
    Args:
        trajectory_3d: 3D trajectory points, shape (T, 3)
        camera_params: Camera configuration to validate
        image_size: Output image dimensions (height, width)
        min_visible_ratio: Minimum fraction of points that should be visible
    
    Returns:
        Dictionary with validation metrics:
        - is_valid: bool
        - visible_points: int
        - visible_ratio: float
        - coverage: float (0-1, how much of frame is used)
        - clipping_ratio: float
        - center_offset_normalized: float
        - recommendations: List[str]
    """
    renderer = TrajectoryRenderer(image_size=image_size)
    
    # Project all points
    projected = renderer._project_trajectory(trajectory_3d, camera_params)
    
    # Check visibility (points within image bounds)
    in_bounds = (
        (projected[:, 0] >= 0) & (projected[:, 0] < image_size[1]) &
        (projected[:, 1] >= 0) & (projected[:, 1] < image_size[0])
    )
    visible_points = int(in_bounds.sum())
    visible_ratio = visible_points / len(trajectory_3d)
    
    # Check coverage (how much of frame is used)
    if in_bounds.any():
        valid_points = projected[in_bounds]
        x_range = valid_points[:, 0].max() - valid_points[:, 0].min()
        y_range = valid_points[:, 1].max() - valid_points[:, 1].min()
        coverage = (x_range * y_range) / (image_size[0] * image_size[1])
    else:
        coverage = 0.0
    
    # Check centering
    center_x, center_y = image_size[1]//2, image_size[0]//2
    if in_bounds.any():
        valid_points = projected[in_bounds]
        mean_x = valid_points[:, 0].mean()
        mean_y = valid_points[:, 1].mean()
        center_offset = np.sqrt((mean_x - center_x)**2 + (mean_y - center_y)**2)
        center_offset_norm = float(center_offset / (min(image_size) / 2))
    else:
        center_offset_norm = 1.0
    
    # Generate recommendations
    recommendations = _generate_recommendations(visible_ratio, coverage, center_offset_norm)
    
    # Overall validation
    is_valid = (visible_ratio >= min_visible_ratio) and (coverage > 0.05)
    
    return {
        "is_valid": is_valid,
        "visible_points": visible_points,
        "visible_ratio": float(visible_ratio),
        "coverage": float(coverage),
        "clipping_ratio": float(1.0 - visible_ratio),
        "center_offset_normalized": center_offset_norm,
        "recommendations": recommendations
    }


def _generate_recommendations(
    visible_ratio: float,
    coverage: float,
    center_offset: float
) -> List[str]:
    """Generate recommendations for improving framing."""
    recs = []
    
    if visible_ratio < 0.9:
        recs.append("Move camera back or reduce focal length (clipping detected)")
    if coverage < 0.1:
        recs.append("Move camera closer or increase focal length (frame underutilized)")
    if coverage > 0.8:
        recs.append("Reduce focal length or move back (too zoomed in)")
    if center_offset > 0.3:
        recs.append("Reposition camera to center trajectory")
    
    return recs if recs else ["Framing is good!"]


def augment_camera_params(
    base_params: CameraParams,
    augmentation_level: str = "moderate",
    seed: int = None
) -> CameraParams:
    """
    Add controlled variation to camera parameters for data augmentation.
    
    Args:
        base_params: Base camera parameters to perturb
        augmentation_level: "light", "moderate", or "heavy"
        seed: Random seed for reproducibility
    
    Returns:
        CameraParams with perturbed parameters
    """
    rng = np.random.default_rng(seed)
    
    # Define augmentation ranges
    aug_ranges = {
        "light": {"pos": 0.1, "focal": 0.1},
        "moderate": {"pos": 0.3, "focal": 0.2},
        "heavy": {"pos": 0.5, "focal": 0.3}
    }
    ranges = aug_ranges.get(augmentation_level, aug_ranges["moderate"])
    
    # Perturb position
    pos_noise = rng.normal(0, ranges["pos"], size=3)
    new_position = base_params.position + pos_noise
    
    # Perturb focal length (multiplicative)
    focal_mult = 1.0 + rng.normal(0, ranges["focal"])
    new_focal = base_params.focal_length * focal_mult
    
    # Clamp focal length to reasonable range
    new_focal = np.clip(new_focal, 50, 500)
    
    return CameraParams(
        position=new_position,
        focal_length=float(new_focal),
        image_center=base_params.image_center
    )
