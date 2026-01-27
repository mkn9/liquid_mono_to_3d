#!/usr/bin/env python3
"""
Test shape rendering to verify 2D projections are correct.

Tests that cube/cylinder/cone render with correct visual appearance.
"""

import numpy as np
import pytest
from magvit_3d_generator import _render_shape_simple


class TestShapeRendering:
    """Test that shapes render correctly in 2D."""
    
    def test_cube_renders_as_square(self):
        """
        Cube should render as a square (rectangle with equal sides).
        
        Expected: 20x20 pixel square (400 pixels total)
        Tolerance: ±10 pixels for edge effects
        """
        img = _render_shape_simple('cube', np.array([0.0, 0.0, 0.0]), (255, 0, 0), 128)
        
        # Find the red pixels (cube color)
        red_pixels = np.where((img[:, :, 0] == 255) & 
                              (img[:, :, 1] == 0) & 
                              (img[:, :, 2] == 0))
        
        # Check pixel count (should be 20x20 = 400)
        pixel_count = len(red_pixels[0])
        assert 390 <= pixel_count <= 410, \
            f"Cube has {pixel_count} pixels, expected ~400 (20x20)"
        
        # Check it's actually square-shaped (not circular or triangular)
        y_range = red_pixels[0].max() - red_pixels[0].min() + 1
        x_range = red_pixels[1].max() - red_pixels[1].min() + 1
        
        assert abs(y_range - x_range) <= 2, \
            f"Cube not square: y_range={y_range}, x_range={x_range}"
        assert 19 <= y_range <= 21, f"Cube y_range={y_range}, expected ~20"
        assert 19 <= x_range <= 21, f"Cube x_range={x_range}, expected ~20"
        
        # Check it's filled (square should be solid, not outline)
        # Density should be 1.0 for a filled square
        density = pixel_count / (y_range * x_range)
        assert density > 0.95, f"Cube density={density:.2f}, expected ~1.0 (filled square)"
    
    def test_cylinder_renders_as_circle(self):
        """
        Cylinder should render as a circle in 2D.
        
        Expected: Circle with radius 10 pixels (π*10² ≈ 314 pixels)
        Tolerance: ±20 pixels for discretization
        """
        img = _render_shape_simple('cylinder', np.array([0.0, 0.0, 0.0]), (0, 255, 0), 128)
        
        # Find green pixels (cylinder color)
        green_pixels = np.where((img[:, :, 0] == 0) & 
                                (img[:, :, 1] == 255) & 
                                (img[:, :, 2] == 0))
        
        # Check pixel count (π * 10² ≈ 314)
        pixel_count = len(green_pixels[0])
        expected_count = np.pi * 10**2
        assert 290 <= pixel_count <= 340, \
            f"Cylinder has {pixel_count} pixels, expected ~{expected_count:.0f} (π*10²)"
        
        # Check it's circular: all pixels should be roughly same distance from center
        center_y = img.shape[0] // 2
        center_x = img.shape[1] // 2
        
        distances = np.sqrt((green_pixels[0] - center_y)**2 + 
                           (green_pixels[1] - center_x)**2)
        
        # All pixels should be within radius 10 (±1 for discretization)
        assert distances.max() <= 11, \
            f"Cylinder max radius={distances.max():.1f}, expected ≤10"
        assert distances.min() >= 0, \
            f"Cylinder min radius={distances.min():.1f}, expected ≥0"
        
        # Standard deviation should be small (all pixels near radius 10)
        # For a filled circle, we expect distances from 0 to 10
        assert distances.mean() < 7, \
            f"Cylinder mean radius={distances.mean():.1f}, expected <7 for filled circle"
    
    def test_cone_renders_as_triangle(self):
        """
        Cone should render as a triangle in 2D.
        
        Expected: Triangle with base 20 pixels, height 20 pixels
        Area ≈ 0.5 * 20 * 20 = 200 pixels
        Tolerance: ±20 pixels for shape discretization
        """
        img = _render_shape_simple('cone', np.array([0.0, 0.0, 0.0]), (0, 0, 255), 128)
        
        # Find blue pixels (cone color)
        blue_pixels = np.where((img[:, :, 0] == 0) & 
                               (img[:, :, 1] == 0) & 
                               (img[:, :, 2] == 255))
        
        # Check pixel count (triangle area ≈ 0.5 * base * height)
        pixel_count = len(blue_pixels[0])
        expected_count = 0.5 * 20 * 20
        assert 180 <= pixel_count <= 240, \
            f"Cone has {pixel_count} pixels, expected ~{expected_count:.0f} (triangle)"
        
        # Check triangular shape: width should vary with y position
        # Triangle should get narrower as you move away from center
        y_range = blue_pixels[0].max() - blue_pixels[0].min() + 1
        x_range = blue_pixels[1].max() - blue_pixels[1].min() + 1
        
        assert 18 <= y_range <= 22, f"Cone height={y_range}, expected ~20"
        assert 18 <= x_range <= 22, f"Cone base={x_range}, expected ~20"
        
        # Check density < 1.0 (triangle should not be as dense as square)
        # Triangle has area 0.5*base*height, square has area base*height
        density = pixel_count / (y_range * x_range)
        assert 0.4 <= density <= 0.6, \
            f"Cone density={density:.2f}, expected ~0.5 (triangle)"
    
    def test_shapes_centered_correctly(self):
        """
        All shapes should be centered at the same position.
        
        This verifies the projection math is consistent.
        """
        point = np.array([0.0, 0.0, 0.0])
        img_size = 128
        
        # Render all three shapes
        img_cube = _render_shape_simple('cube', point, (255, 0, 0), img_size)
        img_cylinder = _render_shape_simple('cylinder', point, (0, 255, 0), img_size)
        img_cone = _render_shape_simple('cone', point, (0, 0, 255), img_size)
        
        # Find centers of mass
        def get_center_of_mass(img, color_channel):
            pixels = np.where(img[:, :, color_channel] > 0)
            if len(pixels[0]) == 0:
                return None, None
            center_y = np.mean(pixels[0])
            center_x = np.mean(pixels[1])
            return center_y, center_x
        
        cube_y, cube_x = get_center_of_mass(img_cube, 0)  # Red channel
        cyl_y, cyl_x = get_center_of_mass(img_cylinder, 1)  # Green channel
        cone_y, cone_x = get_center_of_mass(img_cone, 2)  # Blue channel
        
        # All centers should be at image center (64, 64)
        expected_center = img_size / 2
        
        assert abs(cube_y - expected_center) < 2, \
            f"Cube center Y={cube_y:.1f}, expected {expected_center}"
        assert abs(cube_x - expected_center) < 2, \
            f"Cube center X={cube_x:.1f}, expected {expected_center}"
        
        assert abs(cyl_y - expected_center) < 2, \
            f"Cylinder center Y={cyl_y:.1f}, expected {expected_center}"
        assert abs(cyl_x - expected_center) < 2, \
            f"Cylinder center X={cyl_x:.1f}, expected {expected_center}"
        
        assert abs(cone_y - expected_center) < 2, \
            f"Cone center Y={cone_y:.1f}, expected {expected_center}"
        assert abs(cone_x - expected_center) < 2, \
            f"Cone center X={cone_x:.1f}, expected {expected_center}"
    
    def test_shapes_at_different_positions(self):
        """
        Shapes should render at correct positions when moved.
        
        Tests that the 3D to 2D projection works correctly.
        """
        img_size = 128
        
        # Test point at different location (offset from center)
        point1 = np.array([0.2, 0.1, 0.0])  # Offset in positive x, positive y
        img1 = _render_shape_simple('cube', point1, (255, 0, 0), img_size)
        
        point2 = np.array([-0.2, -0.1, 0.0])  # Offset in negative x, negative y
        img2 = _render_shape_simple('cube', point2, (255, 0, 0), img_size)
        
        # Find centers
        def get_center(img):
            pixels = np.where(img[:, :, 0] > 0)
            if len(pixels[0]) == 0:
                return None, None
            return np.mean(pixels[0]), np.mean(pixels[1])
        
        center1_y, center1_x = get_center(img1)
        center2_y, center2_x = get_center(img2)
        
        # Center1 should be to the right and down from image center
        # Center2 should be to the left and up from image center
        img_center = img_size / 2
        
        # X-axis: positive x → right (larger x coordinate)
        assert center1_x > img_center, \
            f"Point at x=0.2 should be right of center: {center1_x:.1f} > {img_center}"
        assert center2_x < img_center, \
            f"Point at x=-0.2 should be left of center: {center2_x:.1f} < {img_center}"
        
        # Y-axis: positive y → down (larger y coordinate)
        assert center1_y > img_center, \
            f"Point at y=0.1 should be down from center: {center1_y:.1f} > {img_center}"
        assert center2_y < img_center, \
            f"Point at y=-0.1 should be up from center: {center2_y:.1f} < {img_center}"
    
    def test_shapes_have_correct_colors(self):
        """
        Shapes should render in the specified colors only.
        
        No color blending or artifacts.
        """
        img_size = 128
        
        # Render cube in pure red
        img_red = _render_shape_simple('cube', np.array([0.0, 0.0, 0.0]), 
                                       (255, 0, 0), img_size)
        
        # Find all non-black pixels
        non_black = np.where((img_red[:, :, 0] > 0) | 
                            (img_red[:, :, 1] > 0) | 
                            (img_red[:, :, 2] > 0))
        
        # All non-black pixels should be pure red (255, 0, 0)
        for i in range(len(non_black[0])):
            y, x = non_black[0][i], non_black[1][i]
            assert img_red[y, x, 0] == 255, f"Red channel not 255 at ({y}, {x})"
            assert img_red[y, x, 1] == 0, f"Green channel not 0 at ({y}, {x})"
            assert img_red[y, x, 2] == 0, f"Blue channel not 0 at ({y}, {x})"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v'])

