#!/usr/bin/env python3
"""
TDD Tests for MAGVIT Comprehensive Visualization Script

Following requirements.md Section 3.3:
- Tests written FIRST (RED phase)
- Tests verify camera projections work correctly
"""

import pytest
import numpy as np
from pathlib import Path


# ==============================================================================
# INVARIANT TESTS
# ==============================================================================

def test_all_cameras_render_points():
    """
    INVARIANT: All 3 cameras must render trajectory points.
    
    SPECIFICATION:
    - Camera 1 must show projected points
    - Camera 2 must show projected points
    - Camera 3 must show projected points
    - Points must be within image bounds for properly positioned cameras/objects
    """
    from magvit_comprehensive_viz import create_visualization, get_test_trajectories, get_camera_positions
    
    trajectories = get_test_trajectories()
    camera_positions = get_camera_positions()
    
    # This will create the visualization
    result = create_visualization(trajectories, camera_positions, return_projection_counts=True)
    
    # Verify all cameras have projections
    assert result['camera1_points'] > 0, "Camera 1 should render trajectory points"
    assert result['camera2_points'] > 0, "Camera 2 should render trajectory points"
    assert result['camera3_points'] > 0, "Camera 3 should render trajectory points"


def test_projections_within_image_bounds():
    """
    INVARIANT: Majority of projected points must be within image bounds.
    
    SPECIFICATION:
    - At least 40% of points should project within [0, img_width] x [0, img_height]
    - This ensures cameras and objects are positioned reasonably
    """
    from magvit_comprehensive_viz import create_visualization, get_test_trajectories, get_camera_positions
    
    trajectories = get_test_trajectories()
    camera_positions = get_camera_positions()
    
    result = create_visualization(trajectories, camera_positions, return_projection_counts=True)
    
    # Check each camera
    for cam_idx in [1, 2, 3]:
        in_bounds = result[f'camera{cam_idx}_in_bounds']
        total = result[f'camera{cam_idx}_points']
        
        if total > 0:
            ratio = in_bounds / total
            assert ratio >= 0.40, f"Camera {cam_idx}: Only {ratio*100:.1f}% in bounds (need ≥40%)"


def test_output_file_has_timestamp():
    """
    INVARIANT: Output filename must have timestamp prefix.
    
    SPECIFICATION:
    - Format: YYYYMMDD_HHMM_*comprehensive*.png
    - Ensures compliance with requirements.md Section 5.4
    """
    from magvit_comprehensive_viz import get_output_filename
    
    filename = get_output_filename()
    
    # Check format
    parts = filename.split('_')
    assert len(parts) >= 3, f"Filename should have timestamp: {filename}"
    assert len(parts[0]) == 8 and parts[0].isdigit(), f"Should have YYYYMMDD: {parts[0]}"
    assert len(parts[1]) == 4 and parts[1].isdigit(), f"Should have HHMM: {parts[1]}"
    assert 'comprehensive' in filename, "Filename should contain 'comprehensive'"


# ==============================================================================
# GOLDEN TESTS
# ==============================================================================

def test_projection_uses_correct_formula():
    """
    GOLDEN TEST: Verify projection uses TDD-validated formula.
    
    SPECIFICATION:
    - Must use: x_2d = f * X_cam / Y_cam + center_x
    - Must use: y_2d = f * Z_cam / Y_cam + center_y
    - Y is depth (forward direction)
    """
    from magvit_comprehensive_viz import project_point
    
    camera_pos = np.array([0.0, 0.0, 2.5])
    point_3d = np.array([0.3, 1.5, 2.8])
    
    projected = project_point(point_3d, camera_pos, focal_length=600, img_size=(480, 640))
    
    # Verify projection worked
    assert projected is not None, "Point should project (in front of camera)"
    
    # Manual calculation
    point_cam = point_3d - camera_pos
    expected_x = 600 * point_cam[0] / point_cam[1] + 640 / 2
    expected_y = 600 * point_cam[2] / point_cam[1] + 480 / 2
    
    np.testing.assert_allclose(projected, [expected_x, expected_y], atol=1.0,
                               err_msg="Projection formula incorrect")


def test_visualization_structure():
    """
    GOLDEN TEST: Visualization must have correct subplot structure.
    
    SPECIFICATION:
    - 1 large 3D plot (showing cameras and trajectories)
    - 3 smaller 2D plots (one for each camera view)
    - Total: 4 subplots arranged properly
    """
    from magvit_comprehensive_viz import create_visualization, get_test_trajectories, get_camera_positions
    
    trajectories = get_test_trajectories()
    camera_positions = get_camera_positions()
    
    # Create viz and get figure handle
    result = create_visualization(
        trajectories, 
        camera_positions,
        return_figure=True
    )
    
    fig = result['figure']
    axes = fig.get_axes()
    
    # Should have 4 subplots: 1 3D + 3 2D camera views
    assert len(axes) >= 4, f"Should have ≥4 subplots, got {len(axes)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
