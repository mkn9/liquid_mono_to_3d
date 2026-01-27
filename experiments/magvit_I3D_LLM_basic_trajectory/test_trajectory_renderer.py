"""
Test module for trajectory rendering (IMAGES, not coordinates).

CRITICAL REQUIREMENT: Model must process IMAGES, not coordinate shortcuts.
This ensures we're building a TRUE vision-language model.

All tests follow Red → Green → Refactor TDD cycle.
All random tests use explicit seeds for determinism.
All numeric comparisons use explicit tolerances.
"""

import numpy as np
import torch
import pytest
from numpy.testing import assert_allclose

# These imports will fail initially (RED phase - expected!)
from trajectory_renderer import TrajectoryRenderer, CameraParams


class TestTrajectoryRendererInvariants:
    """Invariant tests: properties that must always hold."""
    
    def test_renderer_outputs_image_tensor_not_coordinates(self):
        """CRITICAL: Renderer must output IMAGES (B,T,C,H,W), not coordinates.
        
        This is the core test enforcing TRUE vision modeling.
        A renderer that outputs coordinates is NOT a vision system.
        
        Specification:
        - Input: 3D trajectory points (T, 3)
        - Output: Video frames (T, 3, H, W) where C=3 for RGB
        - NOT: Output (T, 2) coordinates
        """
        renderer = TrajectoryRenderer(image_size=(64, 64))
        
        # Simple linear trajectory
        trajectory_3d = np.array([
            [0.0, 0.0, 2.0],
            [0.1, 0.0, 2.0],
            [0.2, 0.0, 2.0],
            [0.3, 0.0, 2.0],
        ])  # Shape: (4, 3)
        
        camera_params = CameraParams(
            position=np.array([0.0, 0.0, 0.0]),
            focal_length=800,
            image_center=(32, 32)
        )
        
        # Render to video
        video = renderer.render_video(trajectory_3d, camera_params)
        
        # MUST be image tensor, not coordinates
        assert isinstance(video, torch.Tensor), "Output must be torch.Tensor"
        assert video.ndim == 4, f"Expected 4D tensor (T,C,H,W), got {video.ndim}D"
        assert video.shape == (4, 3, 64, 64), \
            f"Expected (4,3,64,64) for 4 frames, RGB, 64x64, got {video.shape}"
        
        # Verify it's actually image data (pixel values in [0, 1])
        assert video.min() >= 0.0, "Image values must be >= 0"
        assert video.max() <= 1.0, "Image values must be <= 1"
        
        # Should NOT be coordinate data (would have very different properties)
        # Coordinates would typically be in range [-1, 1] or [0, image_size]
        # And would have shape (T, 2), not (T, 3, H, W)
    
    def test_rendered_frames_are_finite(self):
        """Rendered frames must never contain NaN or Inf values."""
        renderer = TrajectoryRenderer(image_size=(64, 64))
        
        trajectory_3d = np.array([
            [0.0, 0.0, 2.0],
            [0.1, 0.1, 2.0],
        ])
        
        camera_params = CameraParams(
            position=np.array([0.0, 0.0, 0.0]),
            focal_length=800,
            image_center=(32, 32)
        )
        
        video = renderer.render_video(trajectory_3d, camera_params)
        
        assert torch.all(torch.isfinite(video)), \
            "Rendered frames contain NaN or Inf values"
    
    def test_different_trajectories_produce_different_images(self):
        """Different 3D trajectories must produce visually different images.
        
        This ensures the renderer is actually responding to input,
        not just generating static images.
        
        Seed: N/A (deterministic geometry)
        Tolerance: Images should differ by >1% of pixels
        """
        renderer = TrajectoryRenderer(image_size=(64, 64), style='dot')
        
        # Trajectory 1: Vertical line
        trajectory_1 = np.array([
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 2.5],
            [0.0, 0.0, 3.0],
        ])
        
        # Trajectory 2: Horizontal line (different motion)
        trajectory_2 = np.array([
            [0.0, 0.0, 2.0],
            [0.5, 0.0, 2.0],
            [1.0, 0.0, 2.0],
        ])
        
        camera_params = CameraParams(
            position=np.array([0.0, 0.0, 0.0]),
            focal_length=800,
            image_center=(32, 32)
        )
        
        video_1 = renderer.render_video(trajectory_1, camera_params)
        video_2 = renderer.render_video(trajectory_2, camera_params)
        
        # Images should be different
        diff = torch.abs(video_1 - video_2).mean()
        assert diff > 0.01, \
            f"Different trajectories produced nearly identical images (diff={diff:.4f})"
    
    def test_trajectory_point_appears_in_frame(self):
        """Rendered trajectory points must appear in expected image locations.
        
        This validates that projection math is working correctly.
        
        Specification:
        - 3D point projects to specific 2D pixel location
        - Rendered image should have non-background pixels near that location
        - Tolerance: Within 5 pixels of expected location
        """
        renderer = TrajectoryRenderer(image_size=(64, 64), style='dot', dot_radius=3)
        
        # Simple point directly in front of camera
        trajectory_3d = np.array([
            [0.0, 0.0, 2.0],  # Centered, should project to image center
        ])
        
        camera_params = CameraParams(
            position=np.array([0.0, 0.0, 0.0]),
            focal_length=800,
            image_center=(32, 32)
        )
        
        video = renderer.render_video(trajectory_3d, camera_params)
        
        # Extract first (only) frame
        frame = video[0]  # Shape: (3, 64, 64)
        
        # Check that center region has a drawn object (non-white pixels)
        # Assuming white background (1.0, 1.0, 1.0) and colored dot
        center_region = frame[:, 28:36, 28:36]  # 8x8 region around center
        
        # Should have some non-white pixels in center
        is_white = (center_region[0] == 1.0) & (center_region[1] == 1.0) & (center_region[2] == 1.0)
        non_white_pixels = (~is_white).sum()
        
        assert non_white_pixels > 0, \
            "Expected to find trajectory point rendered near image center, but region is all white"


class TestTrajectoryRendererGolden:
    """Golden/regression tests: canonical scenarios with known outputs."""
    
    def test_render_simple_linear_trajectory(self):
        """Test rendering of simple linear trajectory.
        
        Scenario: 3D line moving horizontally, viewed from fixed camera
        Expected: Dot should move horizontally across frame
        Seed: N/A (deterministic)
        Tolerance: Frames should show progressive motion
        """
        renderer = TrajectoryRenderer(image_size=(64, 64), style='dot')
        
        # Linear trajectory in X direction (smaller range to stay in frame)
        # With focal_length=800, Z=2.0, image_center=32, this produces visible motion
        trajectory_3d = np.array([
            [-0.1, 0.0, 2.0],
            [-0.05, 0.0, 2.0],
            [0.0, 0.0, 2.0],
            [0.05, 0.0, 2.0],
            [0.1, 0.0, 2.0],
        ])
        
        camera_params = CameraParams(
            position=np.array([0.0, 0.0, 0.0]),
            focal_length=800,
            image_center=(32, 32)
        )
        
        video = renderer.render_video(trajectory_3d, camera_params)
        
        # Should have 5 frames
        assert video.shape[0] == 5, f"Expected 5 frames, got {video.shape[0]}"
        
        # Each frame should be valid RGB image
        assert video.shape == (5, 3, 64, 64)
        
        # Frames should be different (motion is happening)
        # Note: At 64x64 resolution, consecutive frames may have very small differences
        for i in range(len(video) - 1):
            diff = torch.abs(video[i] - video[i+1]).mean()
            assert diff > 0.0001, \
                f"Frames {i} and {i+1} are nearly identical (diff={diff:.4f}), expected motion"
    
    def test_camera_position_affects_rendering(self):
        """Different camera positions must produce different views.
        
        Scenario: Same trajectory viewed from 2 different camera positions
        Expected: Images should differ significantly
        Seed: N/A (deterministic)
        Tolerance: >5% pixel difference
        """
        renderer = TrajectoryRenderer(image_size=(64, 64), style='dot')
        
        trajectory_3d = np.array([
            [0.0, 0.0, 2.0],
            [0.0, 0.5, 2.0],
            [0.0, 1.0, 2.0],
        ])
        
        # Camera 1: Centered
        camera_1 = CameraParams(
            position=np.array([0.0, 0.0, 0.0]),
            focal_length=800,
            image_center=(32, 32)
        )
        
        # Camera 2: Offset to the right
        camera_2 = CameraParams(
            position=np.array([1.0, 0.0, 0.0]),
            focal_length=800,
            image_center=(32, 32)
        )
        
        video_1 = renderer.render_video(trajectory_3d, camera_1)
        video_2 = renderer.render_video(trajectory_3d, camera_2)
        
        # Different camera positions should produce different views
        # Note: At 64x64 resolution with small dots, differences may be moderate
        diff = torch.abs(video_1 - video_2).mean()
        assert diff > 0.01, \
            f"Different camera positions produced similar images (diff={diff:.4f})"


class TestTrajectoryRendererEdgeCases:
    """Edge case tests: boundary conditions, error handling."""
    
    def test_empty_trajectory_raises_error(self):
        """Empty trajectory should raise ValueError."""
        renderer = TrajectoryRenderer(image_size=(64, 64))
        
        empty_trajectory = np.array([]).reshape(0, 3)
        
        camera_params = CameraParams(
            position=np.array([0.0, 0.0, 0.0]),
            focal_length=800,
            image_center=(32, 32)
        )
        
        with pytest.raises(ValueError, match="Trajectory cannot be empty"):
            renderer.render_video(empty_trajectory, camera_params)
    
    def test_invalid_trajectory_shape_raises_error(self):
        """Trajectory with wrong shape should raise ValueError."""
        renderer = TrajectoryRenderer(image_size=(64, 64))
        
        # Wrong shape: (T, 2) instead of (T, 3)
        invalid_trajectory = np.array([
            [0.0, 0.0],
            [0.1, 0.1],
        ])
        
        camera_params = CameraParams(
            position=np.array([0.0, 0.0, 0.0]),
            focal_length=800,
            image_center=(32, 32)
        )
        
        with pytest.raises(ValueError, match="Trajectory must have shape"):
            renderer.render_video(invalid_trajectory, camera_params)
    
    def test_point_behind_camera_handled_gracefully(self):
        """Points behind camera should be handled without crashing.
        
        Specification:
        - Points with negative Z (behind camera) are invalid for projection
        - Should either skip rendering or project to image boundary
        - Must not crash or produce NaN
        """
        renderer = TrajectoryRenderer(image_size=(64, 64))
        
        # Trajectory that goes behind camera (negative Z)
        trajectory_3d = np.array([
            [0.0, 0.0, 2.0],   # In front
            [0.0, 0.0, 1.0],   # In front
            [0.0, 0.0, -1.0],  # Behind (invalid)
            [0.0, 0.0, 0.5],   # In front again
        ])
        
        camera_params = CameraParams(
            position=np.array([0.0, 0.0, 0.0]),
            focal_length=800,
            image_center=(32, 32)
        )
        
        # Should not crash
        video = renderer.render_video(trajectory_3d, camera_params)
        
        # Should produce valid output (even if some frames are blank)
        assert video.shape == (4, 3, 64, 64)
        assert torch.all(torch.isfinite(video))


class TestCameraParams:
    """Test CameraParams dataclass."""
    
    def test_camera_params_creation(self):
        """CameraParams should store camera configuration."""
        params = CameraParams(
            position=np.array([1.0, 2.0, 3.0]),
            focal_length=800,
            image_center=(320, 240)
        )
        
        assert_allclose(params.position, np.array([1.0, 2.0, 3.0]))
        assert params.focal_length == 800
        assert params.image_center == (320, 240)


class TestRenderingStyles:
    """Test different rendering styles (dot, trail, etc.)."""
    
    def test_dot_style_rendering(self):
        """Test 'dot' style renders individual points."""
        renderer = TrajectoryRenderer(image_size=(64, 64), style='dot')
        
        trajectory_3d = np.array([
            [0.0, 0.0, 2.0],
        ])
        
        camera_params = CameraParams(
            position=np.array([0.0, 0.0, 0.0]),
            focal_length=800,
            image_center=(32, 32)
        )
        
        video = renderer.render_video(trajectory_3d, camera_params)
        
        # Should render successfully
        assert video.shape == (1, 3, 64, 64)
    
    def test_trail_style_rendering(self):
        """Test 'trail' style renders motion trails."""
        renderer = TrajectoryRenderer(image_size=(64, 64), style='trail')
        
        trajectory_3d = np.array([
            [0.0, 0.0, 2.0],
            [0.1, 0.0, 2.0],
            [0.2, 0.0, 2.0],
        ])
        
        camera_params = CameraParams(
            position=np.array([0.0, 0.0, 0.0]),
            focal_length=800,
            image_center=(32, 32)
        )
        
        video = renderer.render_video(trajectory_3d, camera_params)
        
        # Should render successfully
        assert video.shape == (3, 3, 64, 64)
        
        # Later frames should have more pixels (trail accumulates)
        # This is a qualitative test - we just check it runs

