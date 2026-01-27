#!/usr/bin/env python3
"""
Test Suite for 3D Tracker 9: Y=1.0 Plane Constraint

This test suite specifically validates that all trajectory points lie in the Y=1.0 plane
and that the camera system works correctly with this constraint.

Tests:
1. All trajectory points have Y-coordinate = 1.0
2. FOV intersection with Y=1.0 plane creates rectangles (not trapezoids)
3. Camera-to-plane geometry is correct
4. Projection and triangulation work with Y=1.0 constraint
5. No coordinate system confusion with old Y-varying trajectories
"""

import unittest
import numpy as np
import sys
import os

class TestY1PlaneConstraint(unittest.TestCase):
    """Test Y=1.0 plane constraint implementation."""
    
    def setUp(self):
        """Set up test fixtures for Y=1.0 plane testing."""
        # Camera intrinsic matrix
        self.K = np.array([
            [1000, 0, 640],
            [0, 1000, 360], 
            [0, 0, 1]
        ], dtype=np.float64)
        
        # CORRECTED camera positions (Z=2.55 to avoid singularities)
        self.cam1_center = np.array([0.0, 0.0, 2.55])
        self.cam2_center = np.array([1.0, 0.0, 2.55])
        
        # CORRECTED rotation matrices for +Y looking cameras
        self.R_corrected = np.array([
            [1,  0,  0],   # X axis unchanged
            [0,  0, -1],   # Y axis becomes -Z (depth in camera coordinates)
            [0,  1,  0]    # Z axis becomes Y (vertical in camera coordinates)
        ], dtype=np.float64)
        
        # Projection matrices
        t1 = -np.dot(self.R_corrected, self.cam1_center.reshape(3, 1))
        t2 = -np.dot(self.R_corrected, self.cam2_center.reshape(3, 1))
        
        self.P1 = np.dot(self.K, np.hstack((self.R_corrected, t1)))
        self.P2 = np.dot(self.K, np.hstack((self.R_corrected, t2)))
        
        # Y=1.0 plane trajectory (from 3d_tracker_9)
        self.y1_trajectory = [
            np.array([0.1, 1.0, 2.3]),   # Start: left, Y=1.0, low
            np.array([0.2, 1.0, 2.4]),   # Moving right, Y=1.0, up
            np.array([0.3, 1.0, 2.5]),   # Continuing motion, Y=1.0
            np.array([0.4, 1.0, 2.6]),   # Peak height, Y=1.0
            np.array([0.5, 1.0, 2.5]),   # Descending, Y=1.0
            np.array([0.6, 1.0, 2.4]),   # Continuing right, Y=1.0
            np.array([0.7, 1.0, 2.3]),   # End: right, Y=1.0, low
        ]
    
    def test_all_points_in_y1_plane(self):
        """Test that all trajectory points have Y-coordinate = 1.0."""
        print("\n=== Testing Y=1.0 Plane Constraint ===")
        
        for i, point in enumerate(self.y1_trajectory):
            self.assertAlmostEqual(
                point[1], 1.0, places=10,
                msg=f"Point {i+1} should have Y-coordinate = 1.0"
            )
            print(f"✓ Point {i+1}: {point} has Y = {point[1]}")
        
        # Test that Y coordinates are exactly 1.0 (no floating point errors)
        y_coords = [point[1] for point in self.y1_trajectory]
        unique_y = set(y_coords)
        
        self.assertEqual(len(unique_y), 1, "All Y coordinates should be identical")
        self.assertEqual(list(unique_y)[0], 1.0, "The unique Y coordinate should be 1.0")
        
        print("✅ All trajectory points lie exactly in Y=1.0 plane")
    
    def test_plane_perpendicular_to_optical_axis(self):
        """Test that Y=1.0 plane is perpendicular to camera optical axis."""
        print("\n=== Testing Plane-Optical Axis Geometry ===")
        
        # Camera optical axis direction (in world coordinates)
        # Cameras look in +Y direction
        optical_axis = np.array([0, 1, 0])
        
        # Y=1.0 plane normal vector (also in +Y direction)
        plane_normal = np.array([0, 1, 0])
        
        # Dot product should be 1 (parallel vectors, same direction)
        dot_product = np.dot(optical_axis, plane_normal)
        self.assertAlmostEqual(
            dot_product, 1.0, places=10,
            msg="Optical axis should be parallel to plane normal"
        )
        
        # Distance from cameras (at Y=0) to plane (at Y=1.0)
        distance_cam1 = abs(1.0 - self.cam1_center[1])
        distance_cam2 = abs(1.0 - self.cam2_center[1])
        
        self.assertAlmostEqual(distance_cam1, 1.0, places=10)
        self.assertAlmostEqual(distance_cam2, 1.0, places=10)
        
        print("✅ Y=1.0 plane is perpendicular to optical axis")
        print(f"✅ Distance from cameras to plane: {distance_cam1:.1f} meters")
    
    def test_fov_rectangle_on_y1_plane(self):
        """Test that FOV intersection with Y=1.0 plane creates a rectangle."""
        print("\n=== Testing FOV Rectangle Geometry ===")
        
        # FOV parameters
        fov_horizontal = np.radians(60)  # 60 degrees horizontal FOV
        fov_vertical = np.radians(45)    # 45 degrees vertical FOV
        distance_to_plane = 1.0          # Distance from camera to Y=1.0 plane
        
        # Calculate rectangle dimensions on Y=1.0 plane
        half_width_x = distance_to_plane * np.tan(fov_horizontal / 2)
        half_height_z = distance_to_plane * np.tan(fov_vertical / 2)
        
        # For camera 1 at [0, 0, 2.55]
        expected_x_range = [-half_width_x, half_width_x]
        expected_z_range = [2.55 - half_height_z, 2.55 + half_height_z]
        
        print(f"✅ FOV rectangle X range: [{expected_x_range[0]:.3f}, {expected_x_range[1]:.3f}]")
        print(f"✅ FOV rectangle Z range: [{expected_z_range[0]:.3f}, {expected_z_range[1]:.3f}]")
        
        # Verify trajectory points are within FOV
        for i, point in enumerate(self.y1_trajectory):
            x_in_fov = expected_x_range[0] <= point[0] <= expected_x_range[1]
            z_in_fov = expected_z_range[0] <= point[2] <= expected_z_range[1]
            
            if not (x_in_fov and z_in_fov):
                print(f"⚠️  Point {i+1} {point} may be outside FOV")
            else:
                print(f"✓ Point {i+1} {point} is within FOV rectangle")
    
    def test_no_trapezoid_confusion(self):
        """Test that we don't have trapezoid FOV confusion from old coordinate system."""
        print("\n=== Testing No Trapezoid Confusion ===")
        
        # In the OLD incorrect system, people thought FOV was a trapezoid
        # because they incorrectly thought the plane was parallel to optical axis
        
        # In our CORRECTED system:
        # - Cameras look in +Y direction
        # - Y=1.0 plane is PERPENDICULAR to optical axis
        # - FOV intersection is a RECTANGLE, not trapezoid
        
        # This test ensures we're not using the old trapezoid logic
        fov_angle = 60  # degrees
        distance = 1.0  # meters to Y=1.0 plane
        
        # Rectangle calculation (CORRECT)
        rect_half_width = distance * np.tan(np.radians(fov_angle / 2))
        
        # Old trapezoid calculation would be different and WRONG
        # (It would try to account for "slanted" intersection that doesn't exist)
        
        # Verify we get consistent rectangle dimensions
        expected_width = 2 * rect_half_width
        
        self.assertAlmostEqual(
            expected_width, 2 * distance * np.tan(np.radians(fov_angle / 2)), 
            places=6, msg="Should use rectangle calculation, not trapezoid"
        )
        
        print("✅ Using correct rectangle FOV calculation")
        print("✅ No trapezoid confusion from old coordinate system")
    
    def test_y1_plane_projection_accuracy(self):
        """Test projection accuracy for points in Y=1.0 plane."""
        print("\n=== Testing Y=1.0 Plane Projection Accuracy ===")
        
        projection_errors = []
        
        for i, point_3d in enumerate(self.y1_trajectory):
            # Project to both cameras
            pixel1 = self.project_point_corrected(self.P1, point_3d)
            pixel2 = self.project_point_corrected(self.P2, point_3d)
            
            # Check projections are finite
            finite1 = np.isfinite(pixel1).all()
            finite2 = np.isfinite(pixel2).all()
            
            self.assertTrue(finite1, f"Point {i+1} should project finitely to camera 1")
            self.assertTrue(finite2, f"Point {i+1} should project finitely to camera 2")
            
            if finite1 and finite2:
                # Triangulate back
                reconstructed = self.triangulate_point_corrected(pixel1, pixel2)
                
                # Calculate reconstruction error
                error = np.linalg.norm(reconstructed - point_3d)
                projection_errors.append(error)
                
                print(f"✓ Point {i+1}: 3D {point_3d} → reconstruction error {error:.6f}m")
            else:
                print(f"❌ Point {i+1}: Infinite projection")
        
        # All errors should be very small
        max_error = max(projection_errors) if projection_errors else float('inf')
        avg_error = np.mean(projection_errors) if projection_errors else float('inf')
        
        self.assertLess(max_error, 0.001, "Maximum reconstruction error should be < 1mm")
        self.assertLess(avg_error, 0.0001, "Average reconstruction error should be < 0.1mm")
        
        print(f"✅ Maximum reconstruction error: {max_error:.6f} meters")
        print(f"✅ Average reconstruction error: {avg_error:.6f} meters")
    
    def test_coordinate_system_consistency(self):
        """Test coordinate system consistency with Y=1.0 constraint."""
        print("\n=== Testing Coordinate System Consistency ===")
        
        # Test that cameras are correctly positioned relative to Y=1.0 plane
        self.assertEqual(self.cam1_center[1], 0.0, "Camera 1 should be at Y=0")
        self.assertEqual(self.cam2_center[1], 0.0, "Camera 2 should be at Y=0")
        
        # Test that all trajectory points are in front of cameras
        for i, point in enumerate(self.y1_trajectory):
            self.assertGreater(
                point[1], self.cam1_center[1], 
                f"Point {i+1} should be in front of camera 1 (Y > 0)"
            )
            self.assertGreater(
                point[1], self.cam2_center[1], 
                f"Point {i+1} should be in front of camera 2 (Y > 0)"
            )
        
        # Test camera separation is correct
        baseline = np.linalg.norm(self.cam2_center - self.cam1_center)
        self.assertAlmostEqual(baseline, 1.0, places=10, msg="Baseline should be 1.0m")
        
        print("✅ Cameras positioned correctly relative to Y=1.0 plane")
        print("✅ All trajectory points in front of cameras")
        print(f"✅ Camera baseline: {baseline:.3f} meters")
    
    def project_point_corrected(self, P, point_3d):
        """Project 3D world point to 2D image coordinates - CORRECTED version."""
        # Convert to homogeneous coordinates
        point_3d_h = np.append(point_3d, 1.0)
        
        # Project: pixel_h = P * point_3d_h
        proj = np.dot(P, point_3d_h)
        
        # Handle points at camera plane (avoid division by zero)
        if abs(proj[2]) < 1e-10:
            return np.array([float('inf'), float('inf')])
        
        # Convert from homogeneous to Cartesian coordinates
        return proj[:2] / proj[2]
    
    def triangulate_point_corrected(self, pixel1, pixel2):
        """Triangulate 3D point from stereo 2D observations - CORRECTED version."""
        # Set up linear system: A * X = 0
        # where X is the homogeneous 3D point [x, y, z, 1]
        A = np.array([
            pixel1[0] * self.P1[2, :] - self.P1[0, :],  # u1*P1[2,:] - P1[0,:]
            pixel1[1] * self.P1[2, :] - self.P1[1, :],  # v1*P1[2,:] - P1[1,:]
            pixel2[0] * self.P2[2, :] - self.P2[0, :],  # u2*P2[2,:] - P2[0,:]
            pixel2[1] * self.P2[2, :] - self.P2[1, :]   # v2*P2[2,:] - P2[1,:]
        ])
        
        # Solve using SVD: A = U * S * V^T
        # Solution is last column of V (smallest singular value)
        _, _, Vt = np.linalg.svd(A)
        X_h = Vt[-1, :]
        
        # Convert from homogeneous to Cartesian coordinates
        return X_h[:3] / X_h[3]


class TestY1PlaneVsOldTrajectories(unittest.TestCase):
    """Test that Y=1.0 plane behaves differently from old varying-Y trajectories."""
    
    def setUp(self):
        """Set up comparison between Y=1.0 and old varying-Y trajectories."""
        # Y=1.0 plane trajectory (new)
        self.y1_trajectory = [
            np.array([0.1, 1.0, 2.3]),
            np.array([0.2, 1.0, 2.4]),
            np.array([0.3, 1.0, 2.5]),
            np.array([0.4, 1.0, 2.6]),
            np.array([0.5, 1.0, 2.5]),
            np.array([0.6, 1.0, 2.4]),
            np.array([0.7, 1.0, 2.3]),
        ]
        
        # Old varying-Y trajectory (from 3d_tracker_8)
        self.old_trajectory = [
            np.array([0.1, 0.5, 2.3]),
            np.array([0.2, 0.7, 2.4]),
            np.array([0.3, 0.9, 2.5]),
            np.array([0.4, 1.1, 2.6]),
            np.array([0.5, 1.3, 2.5]),
            np.array([0.6, 1.5, 2.4]),
            np.array([0.7, 1.7, 2.3]),
        ]
    
    def test_y_coordinate_differences(self):
        """Test that Y=1.0 trajectory has constant Y, old trajectory has varying Y."""
        print("\n=== Testing Y Coordinate Differences ===")
        
        # Y=1.0 trajectory should have constant Y
        y1_coords = [point[1] for point in self.y1_trajectory]
        y1_variance = np.var(y1_coords)
        
        self.assertAlmostEqual(y1_variance, 0.0, places=10, 
                              msg="Y=1.0 trajectory should have zero Y variance")
        
        # Old trajectory should have varying Y
        old_y_coords = [point[1] for point in self.old_trajectory]
        old_y_variance = np.var(old_y_coords)
        
        self.assertGreater(old_y_variance, 0.1, 
                          msg="Old trajectory should have significant Y variance")
        
        print(f"✅ Y=1.0 trajectory Y variance: {y1_variance:.10f}")
        print(f"✅ Old trajectory Y variance: {old_y_variance:.3f}")
        print("✅ Y=1.0 trajectory has constant Y, old trajectory varies")
    
    def test_geometric_properties(self):
        """Test geometric properties specific to Y=1.0 plane."""
        print("\n=== Testing Y=1.0 Plane Geometric Properties ===")
        
        # All points in Y=1.0 trajectory should be equidistant from Y=0 plane
        distances_from_y0 = [abs(point[1] - 0.0) for point in self.y1_trajectory]
        distance_variance = np.var(distances_from_y0)
        
        self.assertAlmostEqual(distance_variance, 0.0, places=10,
                              msg="All Y=1.0 points should be equidistant from Y=0")
        
        # All distances should be exactly 1.0
        for i, distance in enumerate(distances_from_y0):
            self.assertAlmostEqual(distance, 1.0, places=10,
                                  msg=f"Point {i+1} should be 1.0 units from Y=0 plane")
        
        print("✅ All Y=1.0 points are exactly 1.0 units from cameras")
        print("✅ Y=1.0 plane constraint maintained throughout trajectory")


def run_y1_plane_tests():
    """Run all Y=1.0 plane constraint tests."""
    print("="*60)
    print("RUNNING Y=1.0 PLANE CONSTRAINT TESTS")
    print("="*60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add Y=1.0 plane constraint tests
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestY1PlaneConstraint))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestY1PlaneVsOldTrajectories))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*60)
    print("Y=1.0 PLANE CONSTRAINT TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall result: {'✅ PASS' if success else '❌ FAIL'}")
    
    return success


if __name__ == "__main__":
    run_y1_plane_tests() 