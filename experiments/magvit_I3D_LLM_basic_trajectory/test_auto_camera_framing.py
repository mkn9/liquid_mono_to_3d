"""
Test automatic camera framing system.

Following TDD per cursorrules - RED phase.
"""
import numpy as np
import pytest
import torch

from dataset_generator import (
    generate_linear_trajectory,
    generate_circular_trajectory,
    generate_helical_trajectory,
    generate_parabolic_trajectory
)


def test_compute_camera_params_exists():
    """Function compute_camera_params should exist."""
    from auto_camera_framing import compute_camera_params
    assert callable(compute_camera_params)


def test_compute_camera_params_returns_camera_params():
    """Should return CameraParams object."""
    from auto_camera_framing import compute_camera_params
    from trajectory_renderer import CameraParams
    
    trajectory = generate_linear_trajectory(num_frames=16)
    params = compute_camera_params(trajectory, image_size=(64, 64))
    
    assert isinstance(params, CameraParams)
    assert hasattr(params, "position")
    assert hasattr(params, "focal_length")
    assert hasattr(params, "image_center")


def test_compute_camera_params_position_shape():
    """Camera position should be 3D vector."""
    from auto_camera_framing import compute_camera_params
    
    trajectory = generate_circular_trajectory(num_frames=16)
    params = compute_camera_params(trajectory)
    
    assert params.position.shape == (3,)
    assert isinstance(params.position, np.ndarray)


def test_compute_camera_params_focal_length_positive():
    """Focal length should be positive."""
    from auto_camera_framing import compute_camera_params
    
    trajectory = generate_helical_trajectory(num_frames=16)
    params = compute_camera_params(trajectory)
    
    assert params.focal_length > 0
    assert params.focal_length < 2000  # Reasonable upper bound


def test_compute_camera_params_centers_on_trajectory():
    """Camera should be positioned to look at trajectory center."""
    from auto_camera_framing import compute_camera_params
    
    trajectory = generate_parabolic_trajectory(num_frames=16)
    traj_center = trajectory.mean(axis=0)
    
    params = compute_camera_params(trajectory)
    
    # Camera X, Y should be close to trajectory X, Y center
    assert abs(params.position[0] - traj_center[0]) < 1.0
    assert abs(params.position[1] - traj_center[1]) < 1.0
    # Camera Z should be behind trajectory
    assert params.position[2] < traj_center[2]


def test_compute_camera_params_coverage_ratio():
    """Coverage ratio should affect focal length."""
    from auto_camera_framing import compute_camera_params
    
    trajectory = generate_linear_trajectory(num_frames=16)
    
    params_tight = compute_camera_params(trajectory, coverage_ratio=0.9)
    params_loose = compute_camera_params(trajectory, coverage_ratio=0.5)
    
    # Tighter coverage needs higher focal length (more zoom)
    assert params_tight.focal_length > params_loose.focal_length


def test_validate_camera_framing_exists():
    """Function validate_camera_framing should exist."""
    from auto_camera_framing import validate_camera_framing
    assert callable(validate_camera_framing)


def test_validate_camera_framing_returns_dict():
    """Should return dictionary with validation metrics."""
    from auto_camera_framing import validate_camera_framing, compute_camera_params
    
    trajectory = generate_circular_trajectory(num_frames=16)
    params = compute_camera_params(trajectory)
    
    validation = validate_camera_framing(trajectory, params, image_size=(64, 64))
    
    assert isinstance(validation, dict)
    assert "is_valid" in validation
    assert "visible_ratio" in validation
    assert "coverage" in validation
    assert "recommendations" in validation


def test_validate_camera_framing_detects_good_framing():
    """Should mark good framing as valid."""
    from auto_camera_framing import validate_camera_framing, compute_camera_params
    
    trajectory = generate_helical_trajectory(num_frames=16)
    params = compute_camera_params(trajectory, coverage_ratio=0.7)
    
    validation = validate_camera_framing(trajectory, params, image_size=(64, 64))
    
    assert validation["is_valid"] == True
    assert validation["visible_ratio"] > 0.9


def test_validate_camera_framing_detects_poor_framing():
    """Should detect when trajectory is poorly framed."""
    from auto_camera_framing import validate_camera_framing
    from trajectory_renderer import CameraParams
    
    trajectory = generate_parabolic_trajectory(num_frames=16)
    
    # Intentionally VERY bad params (camera way too far, extreme focal length)
    bad_params = CameraParams(
        position=np.array([0.0, 0.0, -20.0]),  # Way behind
        focal_length=1500.0,  # Extreme telephoto
        image_center=(32, 32)
    )
    
    validation = validate_camera_framing(trajectory, bad_params, image_size=(64, 64))
    
    # Should detect clipping (most points off-screen)
    assert validation["is_valid"] == False or validation["visible_ratio"] < 0.5


def test_augment_camera_params_exists():
    """Function augment_camera_params should exist."""
    from auto_camera_framing import augment_camera_params
    assert callable(augment_camera_params)


def test_augment_camera_params_returns_camera_params():
    """Should return perturbed CameraParams."""
    from auto_camera_framing import augment_camera_params, compute_camera_params
    
    trajectory = generate_linear_trajectory(num_frames=16)
    base_params = compute_camera_params(trajectory)
    
    augmented = augment_camera_params(base_params, augmentation_level="moderate")
    
    assert hasattr(augmented, "position")
    assert hasattr(augmented, "focal_length")
    # Should be different from base
    assert not np.allclose(augmented.position, base_params.position, atol=0.01)


def test_augment_camera_params_stays_reasonable():
    """Augmentation should keep parameters in reasonable range."""
    from auto_camera_framing import augment_camera_params, compute_camera_params
    
    trajectory = generate_circular_trajectory(num_frames=16)
    base_params = compute_camera_params(trajectory)
    
    for _ in range(10):  # Test multiple augmentations
        augmented = augment_camera_params(base_params, augmentation_level="heavy")
        
        # Focal length should stay in reasonable range
        assert 50 <= augmented.focal_length <= 500
        # Position should not move too far
        distance = np.linalg.norm(augmented.position - base_params.position)
        assert distance < 3.0  # Within 3 units


def test_different_trajectories_get_different_params():
    """Different trajectory types should get appropriate camera settings."""
    from auto_camera_framing import compute_camera_params
    
    linear = generate_linear_trajectory(num_frames=16)
    circular = generate_circular_trajectory(num_frames=16)
    helical = generate_helical_trajectory(num_frames=16)
    
    params_linear = compute_camera_params(linear)
    params_circular = compute_camera_params(circular)
    params_helical = compute_camera_params(helical)
    
    # Should adapt to different trajectory sizes/positions
    # At least one parameter should be substantially different
    focal_diffs = [
        abs(params_linear.focal_length - params_circular.focal_length),
        abs(params_circular.focal_length - params_helical.focal_length),
        abs(params_helical.focal_length - params_linear.focal_length)
    ]
    
    assert max(focal_diffs) > 10  # At least 10 units difference
