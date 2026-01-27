#!/usr/bin/env python3
"""
Test suite for MAGVIT 3D trajectory generation.

Following DETERMINISTIC TDD REQUIREMENT from requirements.md Section 3.3:
- Tests written FIRST (Red phase)
- Deterministic with fixed seeds
- Explicit numeric tolerances
- No float == comparisons

Test organization:
- Invariant tests: Properties that must always hold
- Golden tests: Canonical scenarios with known outputs
- Unit tests: Individual function behavior
"""

import pytest
import numpy as np
from pathlib import Path


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def fixed_seed():
    """Fixed random seed for deterministic testing."""
    return 42


@pytest.fixture
def test_data_dir(tmp_path):
    """Temporary directory for test data."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir


# ==============================================================================
# INVARIANT TESTS
# ==============================================================================

@pytest.mark.invariant
def test_dataset_no_nans_or_infs(fixed_seed):
    """
    Invariant: Generated dataset must contain no NaN or Inf values.
    
    Rationale: NaN/Inf values indicate numerical instability and will cause
    downstream processing to fail.
    """
    from magvit_3d_generator import MAGVIT3DGenerator
    
    np.random.seed(fixed_seed)
    generator = MAGVIT3DGenerator(seq_length=16, img_size=128, num_cameras=3)
    dataset = generator.generate_dataset(num_samples=5)
    
    # Check trajectories
    assert not np.any(np.isnan(dataset['trajectories_3d'])), \
        "Trajectories contain NaN values"
    assert not np.any(np.isinf(dataset['trajectories_3d'])), \
        "Trajectories contain Inf values"
    
    # Check videos
    assert not np.any(np.isnan(dataset['multi_view_videos'])), \
        "Videos contain NaN values"
    assert not np.any(np.isinf(dataset['multi_view_videos'])), \
        "Videos contain Inf values"
    
    # Check labels
    assert not np.any(np.isnan(dataset['labels'])), \
        "Labels contain NaN values"


@pytest.mark.invariant
def test_output_shapes_match_spec(fixed_seed):
    """
    Invariant: Output shapes must exactly match specification.
    
    Spec:
    - trajectories_3d: (num_samples, 16, 3)
    - multi_view_videos: (num_samples, 3, 16, 128, 128, 3)
    - labels: (num_samples,)
    """
    from magvit_3d_generator import MAGVIT3DGenerator
    
    num_samples = 5
    seq_length = 16
    img_size = 128
    num_cameras = 3
    
    np.random.seed(fixed_seed)
    generator = MAGVIT3DGenerator(
        seq_length=seq_length,
        img_size=img_size,
        num_cameras=num_cameras
    )
    dataset = generator.generate_dataset(num_samples=num_samples)
    
    # Check trajectories shape
    expected_traj_shape = (num_samples, seq_length, 3)
    assert dataset['trajectories_3d'].shape == expected_traj_shape, \
        f"Trajectory shape {dataset['trajectories_3d'].shape} != {expected_traj_shape}"
    
    # Check videos shape
    expected_video_shape = (num_samples, num_cameras, seq_length, img_size, img_size, 3)
    assert dataset['multi_view_videos'].shape == expected_video_shape, \
        f"Video shape {dataset['multi_view_videos'].shape} != {expected_video_shape}"
    
    # Check labels shape
    expected_label_shape = (num_samples,)
    assert dataset['labels'].shape == expected_label_shape, \
        f"Label shape {dataset['labels'].shape} != {expected_label_shape}"


@pytest.mark.invariant
def test_trajectory_bounds(fixed_seed):
    """
    Invariant: All trajectory coordinates must be within ±1.0 meters.
    
    Rationale: Trajectories outside this range indicate coordinate system errors
    or incorrect parameter scaling.
    """
    from magvit_3d_generator import MAGVIT3DGenerator
    
    np.random.seed(fixed_seed)
    generator = MAGVIT3DGenerator(seq_length=16, img_size=128, num_cameras=3)
    dataset = generator.generate_dataset(num_samples=10)
    
    trajectories = dataset['trajectories_3d']
    
    # Check bounds
    assert trajectories.min() >= -1.0, \
        f"Trajectory min {trajectories.min():.3f} < -1.0 (out of bounds)"
    assert trajectories.max() <= 1.0, \
        f"Trajectory max {trajectories.max():.3f} > 1.0 (out of bounds)"


@pytest.mark.invariant
def test_label_values_valid(fixed_seed):
    """
    Invariant: Labels must be in range [0, 2] for 3 shape types.
    
    Spec: 0=cube, 1=cylinder, 2=cone
    """
    from magvit_3d_generator import MAGVIT3DGenerator
    
    np.random.seed(fixed_seed)
    generator = MAGVIT3DGenerator(seq_length=16, img_size=128, num_cameras=3)
    dataset = generator.generate_dataset(num_samples=10)
    
    labels = dataset['labels']
    
    # Check range
    assert labels.min() >= 0, f"Label min {labels.min()} < 0"
    assert labels.max() <= 2, f"Label max {labels.max()} > 2"
    
    # Check all values are integers
    assert np.all(labels == labels.astype(int)), "Labels must be integers"


@pytest.mark.invariant
def test_video_pixel_values_valid(fixed_seed):
    """
    Invariant: Video pixel values must be in range [0, 255] and uint8 type.
    
    Rationale: Videos are RGB images with standard 8-bit pixel range.
    """
    from magvit_3d_generator import MAGVIT3DGenerator
    
    np.random.seed(fixed_seed)
    generator = MAGVIT3DGenerator(seq_length=16, img_size=128, num_cameras=3)
    dataset = generator.generate_dataset(num_samples=3)
    
    videos = dataset['multi_view_videos']
    
    # Check dtype
    assert videos.dtype == np.uint8, f"Video dtype {videos.dtype} != uint8"
    
    # Check range (for uint8, this is automatic, but verify)
    assert videos.min() >= 0, f"Video min {videos.min()} < 0"
    assert videos.max() <= 255, f"Video max {videos.max()} > 255"


@pytest.mark.deterministic
def test_reproducibility_with_fixed_seed(fixed_seed):
    """
    Deterministic Test: Same seed must produce identical results.
    
    This is critical for debugging and regression testing.
    """
    from magvit_3d_generator import MAGVIT3DGenerator
    
    # First run
    np.random.seed(fixed_seed)
    generator1 = MAGVIT3DGenerator(seq_length=16, img_size=128, num_cameras=3)
    dataset1 = generator1.generate_dataset(num_samples=3)
    
    # Second run with same seed
    np.random.seed(fixed_seed)
    generator2 = MAGVIT3DGenerator(seq_length=16, img_size=128, num_cameras=3)
    dataset2 = generator2.generate_dataset(num_samples=3)
    
    # Must be exactly identical (bit-for-bit)
    np.testing.assert_array_equal(
        dataset1['trajectories_3d'],
        dataset2['trajectories_3d'],
        err_msg="Trajectories not reproducible with same seed"
    )
    
    np.testing.assert_array_equal(
        dataset1['multi_view_videos'],
        dataset2['multi_view_videos'],
        err_msg="Videos not reproducible with same seed"
    )
    
    np.testing.assert_array_equal(
        dataset1['labels'],
        dataset2['labels'],
        err_msg="Labels not reproducible with same seed"
    )


# ==============================================================================
# GOLDEN TESTS
# ==============================================================================

@pytest.mark.golden
def test_canonical_50_samples(fixed_seed):
    """
    Golden Test: Generate exactly 50 samples with expected properties.
    
    This is the primary use case: generating the full dataset of 50 samples
    with balanced shape distribution.
    """
    from magvit_3d_generator import MAGVIT3DGenerator
    
    np.random.seed(fixed_seed)
    generator = MAGVIT3DGenerator(seq_length=16, img_size=128, num_cameras=3)
    dataset = generator.generate_dataset(num_samples=50)
    
    # Must have exactly 50 samples
    assert len(dataset['trajectories_3d']) == 50, \
        f"Expected 50 samples, got {len(dataset['trajectories_3d'])}"
    assert len(dataset['multi_view_videos']) == 50, \
        f"Expected 50 videos, got {len(dataset['multi_view_videos'])}"
    assert len(dataset['labels']) == 50, \
        f"Expected 50 labels, got {len(dataset['labels'])}"
    
    # Check label distribution is roughly balanced
    label_counts = np.bincount(dataset['labels'])
    assert len(label_counts) == 3, \
        f"Expected 3 shape types, got {len(label_counts)}"
    assert label_counts.min() >= 15, \
        f"Label distribution too unbalanced: {label_counts} (min < 15)"
    assert label_counts.max() <= 18, \
        f"Label distribution too unbalanced: {label_counts} (max > 18)"
    
    # Store summary statistics as golden reference
    mean_traj_length = np.mean([
        np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
        for traj in dataset['trajectories_3d']
    ])
    
    # Golden expectation: mean path length should be ~1.4-1.5 meters
    # (based on trajectory patterns: linear=~0.8m, circular=~1.9m, helical=~2.0m, parabolic=~1.4m)
    assert 1.2 <= mean_traj_length <= 1.8, \
        f"Mean trajectory length {mean_traj_length:.3f}m outside expected range [1.2, 1.8]m"


@pytest.mark.golden
def test_noise_applied_correctly(fixed_seed):
    """
    Golden Test: Verify Gaussian noise (σ=0.02) is applied to trajectories.
    
    Without noise, trajectories would be perfectly deterministic.
    With noise, we should see small perturbations.
    """
    from magvit_3d_generator import MAGVIT3DGenerator, generate_linear_trajectory
    
    # Generate clean trajectory
    clean_traj = generate_linear_trajectory(seq_length=16)
    
    # Generate noisy trajectory through generator
    np.random.seed(fixed_seed)
    generator = MAGVIT3DGenerator(seq_length=16, img_size=128, num_cameras=3)
    dataset = generator.generate_dataset(num_samples=1)
    noisy_traj = dataset['trajectories_3d'][0]
    
    # Noise should be small but non-zero
    # Standard deviation should be approximately 0.02
    noise = noisy_traj - clean_traj
    noise_std = np.std(noise)
    
    # With σ=0.02 and 48 points (16 frames × 3 coords), expect std ≈ 0.02
    # Allow 50% tolerance due to sampling variation
    np.testing.assert_allclose(
        noise_std, 0.02,
        rtol=0.5,  # 50% tolerance
        atol=0.01,
        err_msg=f"Noise std {noise_std:.4f} != 0.02 (±50%)"
    )


# ==============================================================================
# UNIT TESTS - Trajectory Generation Functions
# ==============================================================================

@pytest.mark.unit
def test_linear_trajectory_is_linear():
    """
    Unit Test: Linear trajectory should be a straight line.
    
    Verification: Direction vectors should be constant.
    """
    from magvit_3d_generator import generate_linear_trajectory
    
    traj = generate_linear_trajectory(seq_length=16)
    
    # Check shape
    assert traj.shape == (16, 3), f"Wrong shape: {traj.shape}"
    
    # Check it's actually linear (direction vector constant)
    diffs = np.diff(traj, axis=0)
    norms = np.linalg.norm(diffs, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized_diffs = diffs / norms
    
    # All direction vectors should be nearly identical
    for i in range(1, len(normalized_diffs)):
        np.testing.assert_allclose(
            normalized_diffs[i], normalized_diffs[0],
            rtol=0.01, atol=0.01,
            err_msg=f"Linear trajectory not linear at step {i}"
        )


@pytest.mark.unit
def test_circular_trajectory_has_constant_radius():
    """
    Unit Test: Circular trajectory should have constant radius in XY plane.
    
    Spec: radius = 0.3m, Z should be constant.
    """
    from magvit_3d_generator import generate_circular_trajectory
    
    traj = generate_circular_trajectory(seq_length=16)
    
    # Check shape
    assert traj.shape == (16, 3), f"Wrong shape: {traj.shape}"
    
    # Check constant radius in XY plane
    radii = np.sqrt(traj[:, 0]**2 + traj[:, 1]**2)
    expected_radius = 0.3
    
    np.testing.assert_allclose(
        radii, expected_radius,
        rtol=0.01, atol=0.01,
        err_msg=f"Circular trajectory has non-constant radius: {radii}"
    )
    
    # Check Z is constant (should be 0)
    np.testing.assert_allclose(
        traj[:, 2], 0.0,
        rtol=0.01, atol=0.01,
        err_msg=f"Circular trajectory Z not constant: {traj[:, 2]}"
    )


@pytest.mark.unit
def test_helical_trajectory_has_linear_z():
    """
    Unit Test: Helical trajectory should have linear Z progression.
    
    Spec: Spiral in XY plane with linear Z motion.
    """
    from magvit_3d_generator import generate_helical_trajectory
    
    traj = generate_helical_trajectory(seq_length=16)
    
    # Check shape
    assert traj.shape == (16, 3), f"Wrong shape: {traj.shape}"
    
    # Check Z progression is linear
    z_diffs = np.diff(traj[:, 2])
    
    # All Z increments should be approximately equal
    np.testing.assert_allclose(
        z_diffs, z_diffs[0],
        rtol=0.01, atol=0.01,
        err_msg=f"Helical trajectory Z not linear: {z_diffs}"
    )


@pytest.mark.unit
def test_parabolic_trajectory_shape():
    """
    Unit Test: Parabolic trajectory should follow parabolic equation.
    
    Spec: y = ax² + bx + c form
    """
    from magvit_3d_generator import generate_parabolic_trajectory
    
    traj = generate_parabolic_trajectory(seq_length=16)
    
    # Check shape
    assert traj.shape == (16, 3), f"Wrong shape: {traj.shape}"
    
    # For parabolic motion, second derivative should be approximately constant
    # (this is the defining property of a parabola)
    x_second_diff = np.diff(np.diff(traj[:, 0]))
    y_second_diff = np.diff(np.diff(traj[:, 1]))
    
    # Second differences should be approximately zero (constant acceleration)
    # Allow small tolerance for numerical precision
    assert np.abs(np.std(x_second_diff)) < 0.01, \
        "X coordinate doesn't follow parabolic motion"
    assert np.abs(np.std(y_second_diff)) < 0.01, \
        "Y coordinate doesn't follow parabolic motion"


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================

@pytest.mark.integration
def test_dataset_can_be_saved_and_loaded(fixed_seed, test_data_dir):
    """
    Integration Test: Dataset should be saveable to npz and loadable.
    
    This tests the complete workflow: generate → save → load.
    """
    from magvit_3d_generator import MAGVIT3DGenerator
    
    # Generate dataset
    np.random.seed(fixed_seed)
    generator = MAGVIT3DGenerator(seq_length=16, img_size=128, num_cameras=3)
    original_dataset = generator.generate_dataset(num_samples=5)
    
    # Save to file
    output_file = test_data_dir / "test_dataset.npz"
    np.savez_compressed(output_file, **original_dataset)
    
    # Verify file exists
    assert output_file.exists(), "Dataset file not created"
    
    # Load from file
    loaded_data = np.load(output_file)
    loaded_dataset = {
        'trajectories_3d': loaded_data['trajectories_3d'],
        'multi_view_videos': loaded_data['multi_view_videos'],
        'labels': loaded_data['labels']
    }
    
    # Verify data matches exactly
    np.testing.assert_array_equal(
        original_dataset['trajectories_3d'],
        loaded_dataset['trajectories_3d'],
        err_msg="Loaded trajectories don't match original"
    )
    
    np.testing.assert_array_equal(
        original_dataset['multi_view_videos'],
        loaded_dataset['multi_view_videos'],
        err_msg="Loaded videos don't match original"
    )
    
    np.testing.assert_array_equal(
        original_dataset['labels'],
        loaded_dataset['labels'],
        err_msg="Loaded labels don't match original"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

