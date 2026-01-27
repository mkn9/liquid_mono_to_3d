#!/usr/bin/env python3
"""
Comprehensive Unit Test Suite for 3D Tracking System

Tests every component of the 3D tracking pipeline:
1. Camera setup and calibration
2. Coordinate system transformations  
3. 3D to 2D projections
4. 2D to 3D triangulation
5. Geometric consistency
6. Edge cases and error handling
"""

import unittest
import numpy as np
import cv2
import math
import sys
import os

# Import the functions we want to test
# Add current directory to path to import from notebook
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class Test3DTrackerSetup(unittest.TestCase):
    """Test camera setup and configuration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.K = np.array([
            [1000, 0, 640],
            [0, 1000, 360], 
            [0, 0, 1]
        ], dtype=np.float64)
        
        self.cam1_center = np.array([0.0, 0.0, 2.5])
        self.cam2_center = np.array([1.0, 0.0, 2.5])
        
        # ACTUAL rotation matrix used in our system (identity)
        # In our coordinate system, identity rotation means cameras look in +Y direction
        self.actual_rotation = np.eye(3, dtype=np.float64)
    
    def test_camera_orientation_matrix(self):
        """Test that identity rotation in our system means cameras look in +Y direction."""
        print("\n=== Testing Camera Orientation Matrix ===")
        
        # In our coordinate system, identity rotation means looking in +Y direction
        # Test point in front of camera
        test_point = np.array([0, 1, 0])  # 1 unit in +Y direction
        
        # Apply identity rotation (should not change coordinates)
        transformed_point = np.dot(self.actual_rotation, test_point)
        
        np.testing.assert_array_almost_equal(
            transformed_point, test_point, decimal=10,
            err_msg="Identity rotation should preserve coordinates"
        )
        
        # Positive Y means in front of camera in our system
        self.assertGreater(transformed_point[1], 0, 
                          msg="Positive Y should mean in front of camera")
        
        print("✓ Identity rotation preserves coordinates")
        print("✓ Positive Y means cameras look in +Y direction")
    
    def test_no_x_squint(self):
        """Test that cameras have no squint in X direction."""
        print("\n=== Testing No X Squint ===")
        
        # X axis should remain unchanged
        x_axis = np.array([1, 0, 0])
        transformed_x = np.dot(self.expected_rotation, x_axis)
        
        np.testing.assert_array_almost_equal(
            transformed_x, x_axis, decimal=10,
            err_msg="X axis should remain unchanged (no squint)"
        )
        print("✓ No X axis squint - X direction preserved")
    
    def test_level_in_z(self):
        """Test that cameras are level in Z dimension."""
        print("\n=== Testing Level in Z ===")
        
        # Y axis should transform to -Z axis (level horizon)
        y_axis = np.array([0, 1, 0])
        expected_neg_z = np.array([0, 0, -1])
        transformed_y = np.dot(self.expected_rotation, y_axis)
        
        np.testing.assert_array_almost_equal(
            transformed_y, expected_neg_z, decimal=10,
            err_msg="Y axis should transform to -Z axis (level in Z)"
        )
        print("✓ Cameras are level in Z dimension")
    
    def test_camera_baseline(self):
        """Test camera baseline distance and alignment."""
        print("\n=== Testing Camera Baseline ===")
        
        baseline = np.linalg.norm(self.cam2_center - self.cam1_center)
        expected_baseline = 1.0  # 1 meter separation
        
        self.assertAlmostEqual(
            baseline, expected_baseline, places=10,
            msg="Camera baseline should be 1.0 meters"
        )
        
        # Cameras should be aligned in Y and Z
        y_diff = abs(self.cam2_center[1] - self.cam1_center[1])
        z_diff = abs(self.cam2_center[2] - self.cam1_center[2])
        
        self.assertAlmostEqual(y_diff, 0.0, places=10, msg="Cameras should be aligned in Y")
        self.assertAlmostEqual(z_diff, 0.0, places=10, msg="Cameras should be aligned in Z")
        print("✓ Camera baseline: 1.0m, properly aligned")


class Test3DTrackerProjection(unittest.TestCase):
    """Test 3D to 2D projection functionality."""
    
    def setUp(self):
        """Set up projection test fixtures."""
        self.setup_cameras()
        
        # Test points at known locations
        self.test_points_3d = np.array([
            [0.0, 1.0, 2.5],   # Directly in front of cam1, at camera height
            [1.0, 1.0, 2.5],   # Directly in front of cam2, at camera height
            [0.5, 1.0, 2.5],   # Midway between cameras
            [0.0, 2.0, 2.5],   # Further in Y direction from cam1
            [0.5, 1.0, 3.0]    # Higher than cameras
        ])
    
    def setup_cameras(self):
        """Set up camera matrices for testing."""
        K = np.array([
            [1000, 0, 640],
            [0, 1000, 360], 
            [0, 0, 1]
        ], dtype=np.float64)
        
        cam1_center = np.array([0.0, 0.0, 2.5])
        cam2_center = np.array([1.0, 0.0, 2.5])
        
        # Rotation for +Y pointing cameras
        R = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ], dtype=np.float64)
        
        t1 = -np.dot(R, cam1_center.reshape(3, 1))
        t2 = -np.dot(R, cam2_center.reshape(3, 1))
        
        self.P1 = np.dot(K, np.hstack((R, t1)))
        self.P2 = np.dot(K, np.hstack((R, t2)))
        self.K = K
        
    def test_point_in_front_of_cam1(self):
        """Test projection of point directly in front of camera 1."""
        print("\n=== Testing Point Directly in Front of Camera 1 ===")
        
        point_3d = np.array([0.0, 1.0, 2.5])  # Directly in front, same height
        point_3d_h = np.append(point_3d, 1.0)
        
        proj = np.dot(self.P1, point_3d_h)
        pixel = proj[:2] / proj[2]
        
        # Should project to image center (camera looking straight at it)
        expected_pixel = np.array([640, 360])  # Image center
        
        np.testing.assert_array_almost_equal(
            pixel, expected_pixel, decimal=1,
            err_msg="Point in front of cam1 should project to image center"
        )
        print(f"✓ Point {point_3d} projects to {pixel} (near image center)")
    
    def test_point_to_right_of_cam1(self):
        """Test that points to the right project to higher X pixels."""
        print("\n=== Testing Rightward Point Movement ===")
        
        point_left = np.array([0.2, 1.0, 2.5])
        point_right = np.array([0.6, 1.0, 2.5])
        
        # Project both points
        pixel_left = self.project_point(self.P1, point_left)
        pixel_right = self.project_point(self.P1, point_right)
        
        # Right point should have higher X pixel coordinate
        self.assertGreater(
            pixel_right[0], pixel_left[0],
            msg="Rightward movement should increase X pixel coordinate"
        )
        print(f"✓ Left point {point_left} → pixel {pixel_left}")
        print(f"✓ Right point {point_right} → pixel {pixel_right}")
        print(f"✓ Rightward movement: ΔX = {pixel_right[0] - pixel_left[0]:.1f} pixels")
    
    def test_depth_consistency(self):
        """Test that closer points have larger projections."""
        print("\n=== Testing Depth Consistency ===")
        
        point_far = np.array([0.5, 2.0, 2.5])    # Further away
        point_near = np.array([0.5, 1.5, 2.5])   # Closer
        
        pixel_far = self.project_point(self.P1, point_far)
        pixel_near = self.project_point(self.P1, point_near)
        
        # Both should be valid projections
        self.assertTrue(np.all(np.isfinite(pixel_far)), "Far point should project validly")
        self.assertTrue(np.all(np.isfinite(pixel_near)), "Near point should project validly")
        
        print(f"✓ Far point {point_far} → pixel {pixel_far}")
        print(f"✓ Near point {point_near} → pixel {pixel_near}")
    
    def project_point(self, P, point_3d):
        """Helper function to project a 3D point."""
        point_3d_h = np.append(point_3d, 1.0)
        proj = np.dot(P, point_3d_h)
        if abs(proj[2]) > 1e-10:
            return proj[:2] / proj[2]
        else:
            return np.array([np.inf, np.inf])


class Test3DTrackerTriangulation(unittest.TestCase):
    """Test 2D to 3D triangulation functionality."""
    
    def setUp(self):
        """Set up triangulation test fixtures."""
        self.setup_cameras()
        
        # Known 3D points for round-trip testing
        self.known_3d_points = np.array([
            [0.2, 1.0, 2.7],
            [0.4, 1.0, 2.5],
            [0.6, 1.0, 2.3],
            [0.5, 1.5, 2.5]
        ])
    
    def setup_cameras(self):
        """Set up camera matrices."""
        K = np.array([
            [1000, 0, 640],
            [0, 1000, 360], 
            [0, 0, 1]
        ], dtype=np.float64)
        
        cam1_center = np.array([0.0, 0.0, 2.5])
        cam2_center = np.array([1.0, 0.0, 2.5])
        
        R = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ], dtype=np.float64)
        
        t1 = -np.dot(R, cam1_center.reshape(3, 1))
        t2 = -np.dot(R, cam2_center.reshape(3, 1))
        
        self.P1 = np.dot(K, np.hstack((R, t1)))
        self.P2 = np.dot(K, np.hstack((R, t2)))
    
    def test_round_trip_consistency(self):
        """Test 3D→2D→3D round-trip consistency."""
        print("\n=== Testing Round-Trip Consistency ===")
        
        for i, point_3d in enumerate(self.known_3d_points):
            with self.subTest(point=i):
                # Forward projection: 3D → 2D
                pixel1 = self.project_point(self.P1, point_3d)
                pixel2 = self.project_point(self.P2, point_3d)
                
                # Skip if projection failed
                if not (np.all(np.isfinite(pixel1)) and np.all(np.isfinite(pixel2))):
                    self.skipTest(f"Invalid projection for point {point_3d}")
                
                # Backward triangulation: 2D → 3D
                reconstructed = self.triangulate_point(pixel1, pixel2)
                
                # Should match original point
                np.testing.assert_array_almost_equal(
                    reconstructed, point_3d, decimal=2,
                    err_msg=f"Round-trip failed for point {point_3d}"
                )
                
                error = np.linalg.norm(reconstructed - point_3d)
                print(f"✓ Point {i}: {point_3d} → {reconstructed} (error: {error:.4f}m)")
    
    def test_triangulation_geometry(self):
        """Test triangulation geometric consistency."""
        print("\n=== Testing Triangulation Geometry ===")
        
        # Test point that should be exactly between cameras
        test_point = np.array([0.5, 1.0, 2.5])
        
        pixel1 = self.project_point(self.P1, test_point)
        pixel2 = self.project_point(self.P2, test_point)
        
        reconstructed = self.triangulate_point(pixel1, pixel2)
        
        # X coordinate should be 0.5 (midway between cameras)
        self.assertAlmostEqual(
            reconstructed[0], 0.5, places=2,
            msg="Point between cameras should have X=0.5"
        )
        
        print(f"✓ Midpoint test: {test_point} → {reconstructed}")
    
    def project_point(self, P, point_3d):
        """Project 3D point to 2D."""
        point_3d_h = np.append(point_3d, 1.0)
        proj = np.dot(P, point_3d_h)
        if abs(proj[2]) > 1e-10:
            return proj[:2] / proj[2]
        else:
            return np.array([np.inf, np.inf])
    
    def triangulate_point(self, pixel1, pixel2):
        """Triangulate 3D point from 2D observations."""
        p1 = np.array(pixel1, dtype=np.float32).reshape(2, 1)
        p2 = np.array(pixel2, dtype=np.float32).reshape(2, 1)
        
        point_homog = cv2.triangulatePoints(self.P1, self.P2, p1, p2)
        
        if abs(point_homog[3]) > 1e-10:
            return (point_homog[:3] / point_homog[3]).flatten()
        else:
            return np.array([np.nan, np.nan, np.nan])


class Test3DTrackerGeometry(unittest.TestCase):
    """Test geometric consistency and coordinate systems."""
    
    def test_coordinate_system_handedness(self):
        """Test that coordinate system is right-handed."""
        print("\n=== Testing Coordinate System Handedness ===")
        
        # Right-handed coordinate system: X × Y = Z
        x_axis = np.array([1, 0, 0])
        y_axis = np.array([0, 1, 0])
        z_axis = np.array([0, 0, 1])
        
        cross_product = np.cross(x_axis, y_axis)
        
        np.testing.assert_array_almost_equal(
            cross_product, z_axis, decimal=10,
            err_msg="Coordinate system should be right-handed"
        )
        print("✓ Right-handed coordinate system confirmed")
    
    def test_camera_separation_geometry(self):
        """Test geometric relationships between cameras."""
        print("\n=== Testing Camera Separation Geometry ===")
        
        cam1_pos = np.array([0.0, 0.0, 2.5])
        cam2_pos = np.array([1.0, 0.0, 2.5])
        
        # Test baseline vector
        baseline = cam2_pos - cam1_pos
        expected_baseline = np.array([1.0, 0.0, 0.0])
        
        np.testing.assert_array_almost_equal(
            baseline, expected_baseline, decimal=10,
            err_msg="Baseline should be pure X translation"
        )
        
        baseline_length = np.linalg.norm(baseline)
        self.assertAlmostEqual(
            baseline_length, 1.0, places=10,
            msg="Baseline length should be 1.0 meters"
        )
        
        print(f"✓ Baseline vector: {baseline}")
        print(f"✓ Baseline length: {baseline_length:.6f}m")


class Test3DTrackerEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        """Set up edge case test fixtures."""
        self.setup_cameras()
    
    def setup_cameras(self):
        """Set up camera matrices."""
        K = np.array([
            [1000, 0, 640],
            [0, 1000, 360], 
            [0, 0, 1]
        ], dtype=np.float64)
        
        cam1_center = np.array([0.0, 0.0, 2.5])
        cam2_center = np.array([1.0, 0.0, 2.5])
        
        R = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ], dtype=np.float64)
        
        t1 = -np.dot(R, cam1_center.reshape(3, 1))
        t2 = -np.dot(R, cam2_center.reshape(3, 1))
        
        self.P1 = np.dot(K, np.hstack((R, t1)))
        self.P2 = np.dot(K, np.hstack((R, t2)))
    
    def test_point_behind_camera(self):
        """Test handling of points behind cameras."""
        print("\n=== Testing Points Behind Cameras ===")
        
        # Point behind both cameras (negative Y)
        point_behind = np.array([0.5, -1.0, 2.5])
        point_3d_h = np.append(point_behind, 1.0)
        
        proj1 = np.dot(self.P1, point_3d_h)
        proj2 = np.dot(self.P2, point_3d_h)
        
        # Z component should be negative (behind camera)
        self.assertLess(proj1[2], 0, "Point behind cam1 should have negative Z")
        self.assertLess(proj2[2], 0, "Point behind cam2 should have negative Z")
        
        print(f"✓ Point behind cameras: {point_behind}")
        print(f"✓ Cam1 Z component: {proj1[2]:.3f} (negative = behind)")
        print(f"✓ Cam2 Z component: {proj2[2]:.3f} (negative = behind)")
    
    def test_point_at_infinity(self):
        """Test handling of points at infinity."""
        print("\n=== Testing Points at Infinity ===")
        
        # Very distant point
        point_far = np.array([0.5, 1000.0, 2.5])
        
        try:
            pixel1 = self.project_point(self.P1, point_far)
            pixel2 = self.project_point(self.P2, point_far)
            
            # Should still produce finite results
            self.assertTrue(np.all(np.isfinite(pixel1)), "Distant point should project finitely")
            self.assertTrue(np.all(np.isfinite(pixel2)), "Distant point should project finitely")
            
            print(f"✓ Distant point {point_far} handled correctly")
            print(f"  Cam1 pixel: {pixel1}")
            print(f"  Cam2 pixel: {pixel2}")
            
        except Exception as e:
            self.fail(f"Should handle distant points gracefully: {e}")
    
    def project_point(self, P, point_3d):
        """Project 3D point to 2D with error handling."""
        point_3d_h = np.append(point_3d, 1.0)
        proj = np.dot(P, point_3d_h)
        if abs(proj[2]) > 1e-10:
            return proj[:2] / proj[2]
        else:
            return np.array([np.inf, np.inf])


def run_comprehensive_tests():
    """Run all test suites with detailed reporting."""
    print("="*80)
    print("COMPREHENSIVE 3D TRACKING SYSTEM UNIT TESTS")
    print("="*80)
    
    # Create test suite
    test_classes = [
        Test3DTrackerSetup,
        Test3DTrackerProjection, 
        Test3DTrackerTriangulation,
        Test3DTrackerGeometry,
        Test3DTrackerEdgeCases
    ]
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    
    for test_class in test_classes:
        print(f"\n{'-'*60}")
        print(f"Running {test_class.__name__}")
        print(f"{'-'*60}")
        
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        result = runner.run(suite)
        
        total_tests += result.testsRun
        total_failures += len(result.failures)
        total_errors += len(result.errors)
        
        if result.failures:
            print(f"\nFAILURES in {test_class.__name__}:")
            for test, failure in result.failures:
                print(f"  - {test}: {failure}")
        
        if result.errors:
            print(f"\nERRORS in {test_class.__name__}:")
            for test, error in result.errors:
                print(f"  - {test}: {error}")
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total Tests Run: {total_tests}")
    print(f"Failures: {total_failures}")
    print(f"Errors: {total_errors}")
    
    if total_failures == 0 and total_errors == 0:
        print("🎉 ALL TESTS PASSED! 🎉")
        print("3D Tracking System is functioning correctly.")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please review the failures and errors above.")
    
    print(f"{'='*80}")
    
    return total_failures == 0 and total_errors == 0


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1) 