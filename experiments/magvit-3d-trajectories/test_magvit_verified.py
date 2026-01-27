#!/usr/bin/env python3
"""
Comprehensive Test Suite for Verified MAGVIT 3D Generator

Tests the MAGVIT 3D generator using verified algorithms from:
- 3d_tracker_cylinder.ipynb (6/23/2025)
- 3d_tracker_9.ipynb (6/21/2025, commit c7889a37)

Test Categories:
1. Camera Setup - K, R, t, P matrices
2. 3D to 2D Projection
3. Shape Outline Generation (cube, cylinder, cone)
4. ConvexHull-based Rendering
5. Multi-camera Views
6. Trajectory Generation
7. Dataset Generation
8. Visualizations
9. Integration Tests

Author: AI Assistant
Date: 2026-01-19
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Pytest markers
pytestmark = pytest.mark.unit


class TestCameraSetup:
    """Test camera setup with verified algorithm."""
    
    @pytest.mark.unit
    def test_camera_intrinsics_matrix(self):
        """Camera intrinsics should be correctly scaled for image size."""
        from magvit_verified_generator import setup_cameras
        
        # Test with default 128x128 image size
        cameras_128 = setup_cameras(img_size=128)
        K_128 = cameras_128['K']
        
        # Focal length should be scaled: 1000 * (128/1280) = 100
        expected_fx = expected_fy = 1000 * (128/1280)
        expected_cx = expected_cy = 64  # Center of 128x128 image
        
        np.testing.assert_allclose(K_128[0, 0], expected_fx, atol=1e-10)
        np.testing.assert_allclose(K_128[1, 1], expected_fy, atol=1e-10)
        np.testing.assert_allclose(K_128[0, 2], expected_cx, atol=1e-10)
        np.testing.assert_allclose(K_128[1, 2], expected_cy, atol=1e-10)
        assert K_128.dtype == np.float64, "K should be float64"
        
        # Test with 1280x720 image size (original verified size)
        cameras_1280 = setup_cameras(img_size=1280)
        K_1280 = cameras_1280['K']
        
        np.testing.assert_allclose(K_1280[0, 0], 1000, atol=1e-10)
        np.testing.assert_allclose(K_1280[1, 1], 1000, atol=1e-10)
        np.testing.assert_allclose(K_1280[0, 2], 640, atol=1e-10)
        np.testing.assert_allclose(K_1280[1, 2], 640, atol=1e-10)
    
    @pytest.mark.unit
    def test_camera_positions(self):
        """Camera positions should be at correct locations."""
        from magvit_verified_generator import setup_cameras
        
        cameras = setup_cameras()
        
        expected_cam1 = np.array([0.0, 0.0, 2.55])
        expected_cam2 = np.array([1.0, 0.0, 2.55])
        
        np.testing.assert_allclose(cameras['cam1_pos'], expected_cam1, atol=1e-10)
        np.testing.assert_allclose(cameras['cam2_pos'], expected_cam2, atol=1e-10)
    
    @pytest.mark.unit
    def test_camera_rotation_matrices(self):
        """Rotation matrices should make cameras look in +Y direction."""
        from magvit_verified_generator import setup_cameras
        
        cameras = setup_cameras()
        
        # Verified rotation matrix for +Y looking cameras
        expected_R = np.array([
            [1,  0,  0],
            [0,  0, -1],
            [0,  1,  0]
        ], dtype=np.float64)
        
        np.testing.assert_allclose(cameras['R1'], expected_R, atol=1e-10)
        np.testing.assert_allclose(cameras['R2'], expected_R, atol=1e-10)
    
    @pytest.mark.unit
    def test_translation_vectors(self):
        """Translation vectors should follow t = -R * C formula."""
        from magvit_verified_generator import setup_cameras
        
        cameras = setup_cameras()
        R1, R2 = cameras['R1'], cameras['R2']
        cam1_pos, cam2_pos = cameras['cam1_pos'], cameras['cam2_pos']
        t1, t2 = cameras['t1'], cameras['t2']
        
        # Verify t = -R * C
        expected_t1 = -R1 @ cam1_pos.reshape(3, 1)
        expected_t2 = -R2 @ cam2_pos.reshape(3, 1)
        
        np.testing.assert_allclose(t1, expected_t1, atol=1e-10)
        np.testing.assert_allclose(t2, expected_t2, atol=1e-10)
    
    @pytest.mark.unit
    def test_projection_matrices_shape(self):
        """Projection matrices should be 3x4."""
        from magvit_verified_generator import setup_cameras
        
        cameras = setup_cameras()
        P1, P2 = cameras['P1'], cameras['P2']
        
        assert P1.shape == (3, 4), f"P1 shape should be (3, 4), got {P1.shape}"
        assert P2.shape == (3, 4), f"P2 shape should be (3, 4), got {P2.shape}"
    
    @pytest.mark.unit
    def test_projection_matrices_construction(self):
        """Projection matrices should follow P = K[R|t]."""
        from magvit_verified_generator import setup_cameras
        
        cameras = setup_cameras()
        K = cameras['K']
        R1, R2 = cameras['R1'], cameras['R2']
        t1, t2 = cameras['t1'], cameras['t2']
        P1, P2 = cameras['P1'], cameras['P2']
        
        expected_P1 = K @ np.hstack((R1, t1))
        expected_P2 = K @ np.hstack((R2, t2))
        
        np.testing.assert_allclose(P1, expected_P1, atol=1e-10)
        np.testing.assert_allclose(P2, expected_P2, atol=1e-10)
    
    @pytest.mark.unit
    def test_baseline_distance(self):
        """Baseline distance should be 1.0 meter."""
        from magvit_verified_generator import setup_cameras
        
        cameras = setup_cameras()
        cam1_pos, cam2_pos = cameras['cam1_pos'], cameras['cam2_pos']
        
        baseline = np.linalg.norm(cam2_pos - cam1_pos)
        np.testing.assert_allclose(baseline, 1.0, atol=1e-10)


class TestProjection:
    """Test 3D to 2D projection using verified algorithm."""
    
    @pytest.mark.unit
    def test_project_point_in_front_of_camera(self):
        """Point in front of camera should project to valid pixel."""
        from magvit_verified_generator import setup_cameras, project_point
        
        cameras = setup_cameras()
        P1 = cameras['P1']
        cam1_pos = cameras['cam1_pos']
        
        # Point at Y=1.0 (in front of camera at Y=0)
        point_3d = np.array([0.5, 1.0, 2.5])
        pixel = project_point(P1, point_3d, cam1_pos)
        
        assert not np.isinf(pixel[0]), "X pixel should not be inf"
        assert not np.isinf(pixel[1]), "Y pixel should not be inf"
        assert not np.isnan(pixel[0]), "X pixel should not be NaN"
        assert not np.isnan(pixel[1]), "Y pixel should not be NaN"
        
        # Should be within image bounds (1280x720)
        assert 0 <= pixel[0] <= 1280, f"X pixel {pixel[0]} out of bounds"
        assert 0 <= pixel[1] <= 720, f"Y pixel {pixel[1]} out of bounds"
    
    @pytest.mark.unit
    def test_project_point_at_camera_plane_returns_inf(self):
        """Point at camera depth (Y=0) should return infinity."""
        from magvit_verified_generator import setup_cameras, project_point
        
        cameras = setup_cameras()
        P1 = cameras['P1']
        cam1_pos = cameras['cam1_pos']
        
        # Point at camera Y position
        point_3d = np.array([0.5, 0.0, 2.5])
        pixel = project_point(P1, point_3d, cam1_pos)
        
        assert np.isinf(pixel[0]) or np.isinf(pixel[1]), "Should return inf for point at camera plane"
    
    @pytest.mark.unit
    def test_project_point_behind_camera_returns_inf(self):
        """Point behind camera (Y<0) should return infinity."""
        from magvit_verified_generator import setup_cameras, project_point
        
        cameras = setup_cameras()
        P1 = cameras['P1']
        cam1_pos = cameras['cam1_pos']
        
        # Point behind camera
        point_3d = np.array([0.5, -1.0, 2.5])
        pixel = project_point(P1, point_3d, cam1_pos)
        
        assert np.isinf(pixel[0]) or np.isinf(pixel[1]), "Should return inf for point behind camera"
    
    @pytest.mark.unit
    def test_projection_homogeneous_normalization(self):
        """Projection should properly normalize homogeneous coordinates."""
        from magvit_verified_generator import setup_cameras, project_point
        
        cameras = setup_cameras()
        P1 = cameras['P1']
        cam1_pos = cameras['cam1_pos']
        
        point_3d = np.array([0.3, 1.5, 2.7])
        point_3d_h = np.append(point_3d, 1.0)
        
        proj_h = P1 @ point_3d_h
        expected_pixel = proj_h[:2] / proj_h[2]
        
        actual_pixel = project_point(P1, point_3d, cam1_pos)
        
        np.testing.assert_allclose(actual_pixel, expected_pixel, atol=1e-10)
    
    @pytest.mark.unit
    def test_projection_consistency_across_cameras(self):
        """Same point should project differently to camera 1 and 2."""
        from magvit_verified_generator import setup_cameras, project_point
        
        cameras = setup_cameras()
        P1, P2 = cameras['P1'], cameras['P2']
        cam1_pos, cam2_pos = cameras['cam1_pos'], cameras['cam2_pos']
        
        # Point between cameras
        point_3d = np.array([0.5, 1.0, 2.55])
        
        pixel1 = project_point(P1, point_3d, cam1_pos)
        pixel2 = project_point(P2, point_3d, cam2_pos)
        
        # Pixels should be different (disparity)
        assert not np.allclose(pixel1, pixel2), "Pixels should differ due to disparity"
        
        # But both should be valid
        assert not np.isinf(pixel1[0]) and not np.isinf(pixel1[1])
        assert not np.isinf(pixel2[0]) and not np.isinf(pixel2[1])


class TestShapeOutlines:
    """Test shape outline generation for cube, cylinder, cone."""
    
    @pytest.mark.unit
    def test_cube_outline_has_8_corners(self):
        """Cube outline should have 8 corner points."""
        from magvit_verified_generator import get_cube_outline_points
        
        center = np.array([0.5, 1.0, 2.5])
        size = 0.1
        
        outline = get_cube_outline_points(center, size)
        
        assert outline.shape == (8, 3), f"Cube should have 8 corners, got {outline.shape}"
    
    @pytest.mark.unit
    def test_cube_corners_correct_distance_from_center(self):
        """Cube corners should be correct distance from center."""
        from magvit_verified_generator import get_cube_outline_points
        
        center = np.array([0.5, 1.0, 2.5])
        size = 0.1
        
        outline = get_cube_outline_points(center, size)
        
        # Distance from center to corner should be (size/2) * sqrt(3)
        expected_dist = (size / 2) * np.sqrt(3)
        
        for corner in outline:
            dist = np.linalg.norm(corner - center)
            np.testing.assert_allclose(dist, expected_dist, atol=1e-10)
    
    @pytest.mark.unit
    def test_cylinder_outline_circular_cross_section(self):
        """Cylinder outline should have circular cross-sections."""
        from magvit_verified_generator import get_cylinder_outline_points
        
        center = np.array([0.5, 1.0, 2.5])
        radius = 0.05
        height = 0.2
        
        outline = get_cylinder_outline_points(center, radius, height, n_theta=16)
        
        # Should have 32 points (16 on top circle + 16 on bottom circle)
        assert outline.shape == (32, 3), f"Expected 32 points, got {outline.shape}"
        
        # Check top and bottom circles have correct radius
        axis_dir = np.array([0, 0, 1])
        bottom_center = center - (height / 2) * axis_dir
        top_center = center + (height / 2) * axis_dir
        
        # First 16 points are bottom circle
        for i in range(16):
            dist = np.linalg.norm(outline[i] - bottom_center)
            np.testing.assert_allclose(dist, radius, atol=1e-10)
        
        # Next 16 points are top circle
        for i in range(16, 32):
            dist = np.linalg.norm(outline[i] - top_center)
            np.testing.assert_allclose(dist, radius, atol=1e-10)
    
    @pytest.mark.unit
    def test_cone_outline_has_apex_and_base(self):
        """Cone outline should have apex and base circle."""
        from magvit_verified_generator import get_cone_outline_points
        
        center = np.array([0.5, 1.0, 2.5])
        base_radius = 0.05
        height = 0.2
        
        outline = get_cone_outline_points(center, base_radius, height, n_theta=16)
        
        # Should have 17 points (1 apex + 16 on base circle)
        assert outline.shape == (17, 3), f"Expected 17 points, got {outline.shape}"
        
        # First point should be apex
        axis_dir = np.array([0, 0, 1])
        expected_apex = center + (height / 2) * axis_dir
        np.testing.assert_allclose(outline[0], expected_apex, atol=1e-10)
        
        # Remaining points should be on base circle
        base_center = center - (height / 2) * axis_dir
        for i in range(1, 17):
            dist = np.linalg.norm(outline[i] - base_center)
            np.testing.assert_allclose(dist, base_radius, atol=1e-10)


class TestConvexHullRendering:
    """Test ConvexHull-based 2D rendering."""
    
    @pytest.mark.unit
    def test_render_shape_returns_valid_image(self):
        """Render shape should return valid image array."""
        from magvit_verified_generator import render_shape_2d, setup_cameras
        
        cameras = setup_cameras()
        P1 = cameras['P1']
        cam1_pos = cameras['cam1_pos']
        
        center = np.array([0.5, 1.0, 2.5])
        img = render_shape_2d('cube', center, P1, cam1_pos, img_size=128)
        
        assert img.shape == (128, 128, 3), f"Image shape should be (128, 128, 3), got {img.shape}"
        assert img.dtype == np.uint8, f"Image dtype should be uint8, got {img.dtype}"
    
    @pytest.mark.unit
    def test_rendered_shape_has_nonzero_pixels(self):
        """Rendered shape should have non-zero pixels."""
        from magvit_verified_generator import render_shape_2d, setup_cameras
        
        cameras = setup_cameras()
        P1 = cameras['P1']
        cam1_pos = cameras['cam1_pos']
        
        center = np.array([0.5, 1.0, 2.5])
        img = render_shape_2d('cube', center, P1, cam1_pos, img_size=128, color=(255, 0, 0))
        
        # Should have red pixels
        red_pixels = np.where((img[:, :, 0] > 0) & (img[:, :, 1] == 0) & (img[:, :, 2] == 0))
        assert len(red_pixels[0]) > 0, "Rendered shape should have colored pixels"
    
    @pytest.mark.unit
    def test_out_of_bounds_shape_renders_empty(self):
        """Shape outside field of view should render as empty."""
        from magvit_verified_generator import render_shape_2d, setup_cameras
        
        cameras = setup_cameras()
        P1 = cameras['P1']
        cam1_pos = cameras['cam1_pos']
        
        # Point far outside FOV
        center = np.array([10.0, 1.0, 2.5])
        img = render_shape_2d('cube', center, P1, cam1_pos, img_size=128)
        
        # Should be all zeros
        assert np.all(img == 0), "Out of bounds shape should render as empty"


class TestTrajectoryGeneration:
    """Test trajectory generation functions."""
    
    @pytest.mark.unit
    def test_linear_trajectory_endpoints(self):
        """Linear trajectory should connect start and end points."""
        from magvit_verified_generator import generate_linear_trajectory
        
        traj = generate_linear_trajectory(seq_length=16)
        
        assert traj.shape == (16, 3), f"Trajectory shape should be (16, 3), got {traj.shape}"
        
        # Check endpoints are at expected locations
        assert traj[0, 0] < traj[-1, 0], "X should increase"
    
    @pytest.mark.unit
    def test_circular_trajectory_returns_to_start(self):
        """Circular trajectory should form a closed loop."""
        from magvit_verified_generator import generate_circular_trajectory
        
        traj = generate_circular_trajectory(seq_length=16)
        
        assert traj.shape == (16, 3), f"Trajectory shape should be (16, 3), got {traj.shape}"
        
        # First and last points should be close (closed loop)
        dist = np.linalg.norm(traj[0] - traj[-1])
        assert dist < 0.2, f"Circular trajectory should close, distance: {dist}"
    
    @pytest.mark.unit
    def test_helical_trajectory_has_constant_radius(self):
        """Helical trajectory should maintain constant radius from axis."""
        from magvit_verified_generator import generate_helical_trajectory
        
        traj = generate_helical_trajectory(seq_length=16)
        
        # Project to XY plane and check radius
        radii = np.sqrt(traj[:, 0]**2 + traj[:, 1]**2)
        
        # All radii should be approximately equal
        np.testing.assert_allclose(radii, radii[0], atol=0.05)
    
    @pytest.mark.unit
    def test_parabolic_trajectory_has_apex(self):
        """Parabolic trajectory should have maximum Y value in middle."""
        from magvit_verified_generator import generate_parabolic_trajectory
        
        traj = generate_parabolic_trajectory(seq_length=16)
        
        # Find index of maximum Y
        max_idx = np.argmax(traj[:, 1])
        
        # Should be in middle portion (not at edges)
        assert 4 < max_idx < 12, f"Parabola apex should be in middle, got index {max_idx}"


class TestDatasetGeneration:
    """Test complete dataset generation."""
    
    @pytest.mark.integration
    def test_generate_dataset_returns_correct_keys(self):
        """Dataset should have correct keys."""
        from magvit_verified_generator import MAGVIT3DVerifiedGenerator
        
        generator = MAGVIT3DVerifiedGenerator(seq_length=16, img_size=128)
        dataset = generator.generate_dataset(num_samples=3)
        
        assert 'trajectories_3d' in dataset
        assert 'multi_view_videos' in dataset
        assert 'labels' in dataset
    
    @pytest.mark.integration
    def test_dataset_shapes_correct(self):
        """Dataset arrays should have correct shapes."""
        from magvit_verified_generator import MAGVIT3DVerifiedGenerator
        
        generator = MAGVIT3DVerifiedGenerator(seq_length=16, img_size=128, num_cameras=3)
        dataset = generator.generate_dataset(num_samples=6)
        
        assert dataset['trajectories_3d'].shape == (6, 16, 3)
        assert dataset['multi_view_videos'].shape == (6, 3, 16, 128, 128, 3)
        assert dataset['labels'].shape == (6,)
    
    @pytest.mark.integration
    def test_dataset_labels_correct(self):
        """Dataset labels should cycle through shapes."""
        from magvit_verified_generator import MAGVIT3DVerifiedGenerator
        
        generator = MAGVIT3DVerifiedGenerator(seq_length=16, img_size=128)
        dataset = generator.generate_dataset(num_samples=9)
        
        # Should cycle: 0, 1, 2, 0, 1, 2, 0, 1, 2
        expected_labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        np.testing.assert_array_equal(dataset['labels'], expected_labels)
    
    @pytest.mark.integration
    def test_dataset_videos_have_nonzero_pixels(self):
        """Generated videos should have non-zero pixels."""
        from magvit_verified_generator import MAGVIT3DVerifiedGenerator
        
        generator = MAGVIT3DVerifiedGenerator(seq_length=16, img_size=128)
        dataset = generator.generate_dataset(num_samples=3)
        
        videos = dataset['multi_view_videos']
        
        # Check that at least some frames have non-zero pixels
        for i in range(3):
            for cam in range(3):
                for frame in range(16):
                    img = videos[i, cam, frame]
                    if np.any(img > 0):
                        break
                else:
                    continue
                break
            else:
                continue
            break
        else:
            pytest.fail("No non-zero pixels found in any video")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

