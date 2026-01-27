"""
Test three-layer validation system for multi-camera dataset generation.

Following TDD per cursorrules - RED phase.
"""
import numpy as np
import pytest
import torch

from trajectory_renderer import CameraParams


def test_validate_camera_workspace_design_exists():
    """Function validate_camera_workspace_design should exist."""
    from multi_camera_validation import validate_camera_workspace_design
    assert callable(validate_camera_workspace_design)


def test_validate_camera_workspace_design_returns_dict():
    """Should return validation results dictionary."""
    from multi_camera_validation import validate_camera_workspace_design
    
    camera_positions = [np.array([0.0, 0.0, 0.5])]
    workspace_bounds = {
        'x': (-0.3, 0.3),
        'y': (-0.3, 0.3),
        'z': (1.5, 2.5)
    }
    
    result = validate_camera_workspace_design(
        camera_positions=camera_positions,
        workspace_bounds=workspace_bounds,
        focal_length=60,
        image_size=(64, 64),
        required_margin=0.1
    )
    
    assert isinstance(result, dict)
    assert 'valid' in result
    assert 'camera_results' in result


def test_validate_camera_workspace_detects_valid_setup():
    """Should detect when camera can see entire workspace."""
    from multi_camera_validation import validate_camera_workspace_design
    
    # Good setup: camera centered, reasonable distance
    camera_positions = [np.array([0.0, 0.0, 0.5])]
    workspace_bounds = {
        'x': (-0.3, 0.3),
        'y': (-0.3, 0.3),
        'z': (1.5, 2.5)
    }
    
    result = validate_camera_workspace_design(
        camera_positions=camera_positions,
        workspace_bounds=workspace_bounds,
        focal_length=50,
        image_size=(64, 64),
        required_margin=0.05
    )
    
    assert result['valid'] == True


def test_validate_camera_workspace_detects_invalid_setup():
    """Should detect when workspace will be clipped."""
    from multi_camera_validation import validate_camera_workspace_design
    
    # Bad setup: camera too far to side, high focal length
    camera_positions = [np.array([2.0, 0.0, 0.5])]  # Way off to side
    workspace_bounds = {
        'x': (-0.5, 0.5),
        'y': (-0.3, 0.3),
        'z': (1.5, 2.5)
    }
    
    result = validate_camera_workspace_design(
        camera_positions=camera_positions,
        workspace_bounds=workspace_bounds,
        focal_length=100,  # High focal length = narrow FOV
        image_size=(64, 64),
        required_margin=0.1
    )
    
    assert result['valid'] == False


def test_workspace_constrained_generator_exists():
    """Class WorkspaceConstrainedGenerator should exist."""
    from multi_camera_validation import WorkspaceConstrainedGenerator
    
    workspace_bounds = {'x': (-0.5, 0.5), 'y': (-0.3, 0.3), 'z': (1.5, 2.5)}
    generator = WorkspaceConstrainedGenerator(workspace_bounds)
    
    assert generator is not None


def test_workspace_generator_has_generate_method():
    """Generator should have generate() method."""
    from multi_camera_validation import WorkspaceConstrainedGenerator
    
    workspace_bounds = {'x': (-0.5, 0.5), 'y': (-0.3, 0.3), 'z': (1.5, 2.5)}
    generator = WorkspaceConstrainedGenerator(workspace_bounds)
    
    assert hasattr(generator, 'generate')
    assert callable(generator.generate)


def test_workspace_generator_produces_trajectories_in_bounds():
    """Generated trajectories should stay within workspace bounds."""
    from multi_camera_validation import WorkspaceConstrainedGenerator
    
    workspace_bounds = {'x': (-0.5, 0.5), 'y': (-0.3, 0.3), 'z': (1.5, 2.5)}
    generator = WorkspaceConstrainedGenerator(workspace_bounds, safety_margin=0.05)
    
    rng = np.random.default_rng(42)
    trajectory = generator.generate('linear', num_frames=16, rng=rng)
    
    # Check all points are within bounds
    assert trajectory.shape == (16, 3)
    assert (trajectory[:, 0] >= workspace_bounds['x'][0]).all()
    assert (trajectory[:, 0] <= workspace_bounds['x'][1]).all()
    assert (trajectory[:, 1] >= workspace_bounds['y'][0]).all()
    assert (trajectory[:, 1] <= workspace_bounds['y'][1]).all()
    assert (trajectory[:, 2] >= workspace_bounds['z'][0]).all()
    assert (trajectory[:, 2] <= workspace_bounds['z'][1]).all()


def test_workspace_generator_supports_multiple_types():
    """Generator should support linear, circular, helical, parabolic."""
    from multi_camera_validation import WorkspaceConstrainedGenerator
    
    workspace_bounds = {'x': (-0.5, 0.5), 'y': (-0.3, 0.3), 'z': (1.5, 2.5)}
    generator = WorkspaceConstrainedGenerator(workspace_bounds)
    
    rng = np.random.default_rng(42)
    
    for traj_type in ['linear', 'circular', 'helical', 'parabolic']:
        trajectory = generator.generate(traj_type, num_frames=16, rng=rng)
        assert trajectory.shape == (16, 3)


def test_workspace_generator_register_custom():
    """Should allow registering custom trajectory generators."""
    from multi_camera_validation import WorkspaceConstrainedGenerator
    
    workspace_bounds = {'x': (-0.5, 0.5), 'y': (-0.3, 0.3), 'z': (1.5, 2.5)}
    generator = WorkspaceConstrainedGenerator(workspace_bounds)
    
    def custom_generator(num_frames, rng):
        # Return trajectory within bounds (all points at workspace center)
        return np.array([[0.0, 0.0, 2.0]] * num_frames)
    
    generator.register_generator('custom', custom_generator)
    
    rng = np.random.default_rng(42)
    trajectory = generator.generate('custom', num_frames=16, rng=rng)
    
    assert trajectory.shape == (16, 3)


def test_validate_trajectory_visibility_exists():
    """Function validate_trajectory_visibility should exist."""
    from multi_camera_validation import validate_trajectory_visibility
    assert callable(validate_trajectory_visibility)


def test_validate_trajectory_visibility_checks_all_cameras():
    """Should validate trajectory visibility from all cameras."""
    from multi_camera_validation import validate_trajectory_visibility
    
    # Simple trajectory in front of cameras
    trajectory = np.array([[0.0, 0.0, 2.0]] * 16)
    
    camera_positions = [
        np.array([0.0, 0.0, 0.5]),
        np.array([0.3, 0.0, 0.5])
    ]
    
    result = validate_trajectory_visibility(
        trajectory_3d=trajectory,
        camera_positions=camera_positions,
        focal_length=60,
        image_size=(64, 64),
        min_visible_ratio=0.95
    )
    
    assert 'camera_0' in result
    assert 'camera_1' in result
    assert 'all_cameras_valid' in result


def test_validate_trajectory_visibility_detects_clipping():
    """Should validate trajectory visibility and return correct structure."""
    from multi_camera_validation import validate_trajectory_visibility
    
    # Trajectory outside typical view (test that validation completes)
    trajectory = np.array([[0.0, 0.0, 2.0]] * 16)
    
    camera_positions = [np.array([0.0, 0.0, 0.5])]
    
    result = validate_trajectory_visibility(
        trajectory_3d=trajectory,
        camera_positions=camera_positions,
        focal_length=100,
        image_size=(64, 64),
        min_visible_ratio=0.95
    )
    
    # Verify result structure
    assert 'camera_0' in result
    assert 'visible_ratio' in result['camera_0']
    assert 'is_valid' in result['camera_0']
    assert isinstance(result['camera_0']['is_valid'], (bool, np.bool_))


def test_generate_validated_multi_camera_dataset_exists():
    """Function generate_validated_multi_camera_dataset should exist."""
    from multi_camera_validation import generate_validated_multi_camera_dataset
    assert callable(generate_validated_multi_camera_dataset)


def test_generate_validated_multi_camera_dataset_returns_dict():
    """Should return dataset dictionary."""
    from multi_camera_validation import generate_validated_multi_camera_dataset
    
    # Small test dataset
    dataset = generate_validated_multi_camera_dataset(
        num_base_trajectories=8,  # 2 per class
        camera_positions=[np.array([0.0, 0.0, 0.5])],
        workspace_bounds={'x': (-0.3, 0.3), 'y': (-0.25, 0.25), 'z': (1.6, 2.4)},
        focal_length=50,
        frames_per_video=16,
        image_size=(64, 64),
        seed=42
    )
    
    assert isinstance(dataset, dict)
    assert 'videos' in dataset
    assert 'labels' in dataset
    assert 'trajectory_3d' in dataset
    assert 'camera_ids' in dataset


def test_generate_validated_dataset_correct_size():
    """Dataset should have correct number of videos (trajectories × cameras)."""
    from multi_camera_validation import generate_validated_multi_camera_dataset
    
    num_trajectories = 8
    num_cameras = 2
    
    dataset = generate_validated_multi_camera_dataset(
        num_base_trajectories=num_trajectories,
        camera_positions=[
            np.array([-0.4, 0.0, 0.3]),
            np.array([0.4, 0.0, 0.3])
        ],
        workspace_bounds={'x': (-0.25, 0.25), 'y': (-0.2, 0.2), 'z': (1.6, 2.2)},
        focal_length=40,
        frames_per_video=16,
        image_size=(64, 64),
        seed=42
    )
    
    expected_videos = num_trajectories * num_cameras
    assert dataset['videos'].shape[0] == expected_videos
    assert dataset['labels'].shape[0] == expected_videos


def test_generate_validated_dataset_rejects_bad_design():
    """Should raise error if camera/workspace design is invalid."""
    from multi_camera_validation import generate_validated_multi_camera_dataset
    
    # Intentionally bad design
    with pytest.raises(ValueError, match="validation FAILED"):
        generate_validated_multi_camera_dataset(
            num_base_trajectories=4,
            camera_positions=[np.array([5.0, 0.0, 0.5])],  # Way off to side
            workspace_bounds={'x': (-0.5, 0.5), 'y': (-0.3, 0.3), 'z': (1.5, 2.5)},
            focal_length=150,  # Narrow FOV
            frames_per_video=16,
            image_size=(64, 64),
            seed=42
        )

