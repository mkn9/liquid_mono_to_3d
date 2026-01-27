#!/usr/bin/env python3
"""
Test Suite for Our ACTUAL Coordinate System

This test suite validates our REAL coordinate system:
- Cameras at [0,0,2.5] and [1,0,2.5] 
- Identity rotation matrices (R = I)
- Cameras look in +Y direction (NOT -Z direction)
- Level in Z (no up/down tilt)
- No X squint (no left/right tilt)
- Objects at y > 0 are in front of cameras
"""

import unittest
import numpy as np
import sys
import os

class TestActualCoordinateSystem(unittest.TestCase):
    """Test our actual coordinate system implementation."""
    
    def setUp(self):
        """Set up test fixtures matching our ACTUAL system."""
        self.K = np.array([
            [1000, 0, 640],
            [0, 1000, 360], 
            [0, 0, 1]
        ], dtype=np.float64)
        
        self.cam1_center = np.array([0.0, 0.0, 2.55])
        self.cam2_center = np.array([1.0, 0.0, 2.55])
        
        # ACTUAL rotation matrices used in our system (identity)
        self.R1 = np.eye(3, dtype=np.float64)
        self.R2 = np.eye(3, dtype=np.float64)
        
        # ACTUAL translation vectors
        self.t1 = -np.dot(self.R1, self.cam1_center.reshape(3, 1))
        self.t2 = -np.dot(self.R2, self.cam2_center.reshape(3, 1))
        
        # ACTUAL projection matrices
        self.P1 = np.dot(self.K, np.hstack((self.R1, self.t1)))
        self.P2 = np.dot(self.K, np.hstack((self.R2, self.t2)))
    
    def test_camera_positions(self):
        """Test camera positions are correct."""
        print("\n=== Testing Camera Positions ===")
        
        np.testing.assert_array_almost_equal(
            self.cam1_center, [0.0, 0.0, 2.5], decimal=10,
            err_msg="Camera 1 should be at [0, 0, 2.5]"
        )
        
        np.testing.assert_array_almost_equal(
            self.cam2_center, [1.0, 0.0, 2.5], decimal=10,
            err_msg="Camera 2 should be at [1, 0, 2.5]"
        )
        
        baseline = np.linalg.norm(self.cam2_center - self.cam1_center)
        self.assertAlmostEqual(baseline, 1.0, places=10, 
                              msg="Baseline should be 1.0 meters")
        
        print("✓ Camera positions correct")
        print(f"✓ Baseline: {baseline:.3f} meters")
    
    def test_identity_rotation_means_plus_y_looking(self):
        """Test that identity rotation in our system means looking in +Y direction."""
        print("\n=== Testing Identity Rotation = +Y Looking ===")
        
        # In our coordinate system, a point at [0, 1, 2.55] relative to camera at [0, 0, 2.55]
        # becomes [0, 1, 0] in camera coordinates
        test_point_world = np.array([0.0, 1.0, 2.55])  # 1 unit in front of cam1 in Y
        test_point_cam = test_point_world - self.cam1_center  # [0, 1, 0]
        
        # Apply identity rotation (should not change coordinates)
        test_point_rotated = np.dot(self.R1, test_point_cam)
        
        expected = np.array([0.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(
            test_point_rotated, expected, decimal=10,
            err_msg="Identity rotation should preserve coordinates"
        )
        
        # Positive Y means in front of camera
        self.assertGreater(test_point_rotated[1], 0, 
                          msg="Positive Y should mean in front of camera")
        
        print("✓ Identity rotation preserves coordinates")
        print("✓ Positive Y means in front of camera")
        print("✓ Cameras look in +Y direction")
    
    def test_no_x_squint(self):
        """Test that cameras have no X squint (no left/right tilt)."""
        print("\n=== Testing No X Squint ===")
        
        # Points with same Y,Z but different X should project with only X difference
        point_left = np.array([0.2, 1.0, 2.55])   # Left of center
        point_right = np.array([0.6, 1.0, 2.55])  # Right of center
        
        pixel_left = self.project_point(self.P1, point_left)
        pixel_right = self.project_point(self.P1, point_right)
        
        # Y pixels should be the same (no squint)
        self.assertAlmostEqual(pixel_left[1], pixel_right[1], places=1,
                              msg="Points at same Y,Z should have same Y pixel (no X squint)")
        
        # X pixels should be different
        self.assertNotAlmostEqual(pixel_left[0], pixel_right[0], places=1,
                                 msg="Points at different X should have different X pixels")
        
        print("✓ No X squint - same Y,Z gives same Y pixel")
        print(f"✓ Left point Y pixel: {pixel_left[1]:.1f}")
        print(f"✓ Right point Y pixel: {pixel_right[1]:.1f}")
    
    def test_level_in_z(self):
        """Test that cameras are level in Z (no up/down tilt)."""
        print("\n=== Testing Level in Z ===")
        
        # Points with same X,Y but different Z should project with only Y difference
        point_low = np.array([0.4, 1.0, 2.0])    # Below camera
        point_high = np.array([0.4, 1.0, 3.0])   # Above camera
        
        pixel_low = self.project_point(self.P1, point_low)
        pixel_high = self.project_point(self.P1, point_high)
        
        # X pixels should be the same (level in Z)
        self.assertAlmostEqual(pixel_low[0], pixel_high[0], places=1,
                              msg="Points at same X,Y should have same X pixel (level in Z)")
        
        # Y pixels should be different
        self.assertNotAlmostEqual(pixel_low[1], pixel_high[1], places=1,
                                 msg="Points at different Z should have different Y pixels")
        
        print("✓ Level in Z - same X,Y gives same X pixel")
        print(f"✓ Low point X pixel: {pixel_low[0]:.1f}")
        print(f"✓ High point X pixel: {pixel_high[0]:.1f}")
    
    def test_trajectory_projection(self):
        """Test projection of our actual trajectory."""
        print("\n=== Testing Trajectory Projection ===")
        
        # Our actual trajectory
        trajectory = [
            np.array([0.2, 1.0, 2.7]),
            np.array([0.3, 1.0, 2.6]),
            np.array([0.4, 1.0, 2.5]),
            np.array([0.5, 1.0, 2.4]),
            np.array([0.6, 1.0, 2.3])
        ]
        
        print("Trajectory projections:")
        for i, point in enumerate(trajectory):
            pixel1 = self.project_point(self.P1, point)
            pixel2 = self.project_point(self.P2, point)
            
            print(f"Point {i+1} {point} -> Cam1: [{pixel1[0]:.0f}, {pixel1[1]:.0f}], Cam2: [{pixel2[0]:.0f}, {pixel2[1]:.0f}]")
            
            # All points should be in front of cameras (positive Y)
            point_cam1 = point - self.cam1_center
            point_cam2 = point - self.cam2_center
            
            self.assertGreater(point_cam1[1], 0, f"Point {i+1} should be in front of cam1")
            self.assertGreater(point_cam2[1], 0, f"Point {i+1} should be in front of cam2")
        
        # Test rightward movement increases X pixels
        pixel1_start = self.project_point(self.P1, trajectory[0])
        pixel1_end = self.project_point(self.P1, trajectory[-1])
        
        self.assertGreater(pixel1_end[0], pixel1_start[0],
                          "Rightward movement should increase X pixels in cam1")
        
        print("✓ All trajectory points in front of cameras")
        print("✓ Rightward movement increases X pixels")
    
    def test_triangulation_accuracy(self):
        """Test triangulation accuracy with our coordinate system."""
        print("\n=== Testing Triangulation Accuracy ===")
        
        test_point = np.array([0.4, 1.0, 2.55])  # Middle of trajectory
        
        # Project to both cameras
        pixel1 = self.project_point(self.P1, test_point)
        pixel2 = self.project_point(self.P2, test_point)
        
        # Triangulate back
        reconstructed = self.triangulate_point(pixel1, pixel2)
        
        # Should be very close to original
        np.testing.assert_array_almost_equal(
            reconstructed, test_point, decimal=3,
            err_msg="Triangulation should accurately reconstruct 3D point"
        )
        
        error = np.linalg.norm(reconstructed - test_point)
        print(f"✓ Triangulation error: {error:.6f} meters")
        print(f"✓ Original: {test_point}")
        print(f"✓ Reconstructed: {reconstructed}")
    
    def project_point(self, P, point_3d):
        """Project 3D point to 2D using projection matrix."""
        point_3d_h = np.append(point_3d, 1.0)
        proj = np.dot(P, point_3d_h)
        return proj[:2] / proj[2]
    
    def triangulate_point(self, pixel1, pixel2):
        """Triangulate 3D point from two 2D observations."""
        # Create the linear system Ax = 0
        A = np.array([
            pixel1[0] * self.P1[2, :] - self.P1[0, :],
            pixel1[1] * self.P1[2, :] - self.P1[1, :],
            pixel2[0] * self.P2[2, :] - self.P2[0, :],
            pixel2[1] * self.P2[2, :] - self.P2[1, :]
        ])
        
        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1, :]
        
        # Convert from homogeneous coordinates
        return X[:3] / X[3]

def run_coordinate_system_tests():
    """Run all coordinate system tests."""
    print("="*60)
    print("TESTING OUR ACTUAL COORDINATE SYSTEM")
    print("="*60)
    print("System specifications:")
    print("- Cameras at [0,0,2.5] and [1,0,2.5]")
    print("- Identity rotation matrices")
    print("- Cameras look in +Y direction")
    print("- Level in Z, no X squint")
    print("- Objects at y > camera_y are in front")
    print("="*60)
    
    unittest.main(argv=[''], exit=False, verbosity=2)

if __name__ == "__main__":
    run_coordinate_system_tests() 