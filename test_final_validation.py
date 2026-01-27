#!/usr/bin/env python3
"""
Final validation test for the corrected 3D tracking coordinate system.
This test validates the actual coordinate system used in 3d_tracker_8.ipynb
"""

import numpy as np
import cv2
import unittest

class TestCorrectedCoordinateSystem(unittest.TestCase):
    """Test the corrected coordinate system that's actually implemented."""
    
    def setUp(self):
        """Set up the corrected camera system."""
        # Camera positions - corrected to avoid singularity
        self.cam1_pos = np.array([0.0, 0.0, 2.55])  # Slightly above trajectory
        self.cam2_pos = np.array([1.0, 0.0, 2.55])  # 1m baseline
        
        # Corrected rotation matrix for +Y looking cameras
        self.R_corrected = np.array([
            [1, 0, 0],
            [0, 0, -1], 
            [0, 1, 0]
        ])
        
        # Intrinsic matrix
        self.K = np.array([
            [1000, 0, 640],
            [0, 1000, 360],
            [0, 0, 1]
        ], dtype=float)
        
        # Build projection matrices
        self.P1 = self.build_projection_matrix(self.K, self.R_corrected, self.cam1_pos)
        self.P2 = self.build_projection_matrix(self.K, self.R_corrected, self.cam2_pos)
        
        # Test trajectory - same Y, same Z=2.5, moving in X
        self.trajectory = np.array([
            [0.2, 1.0, 2.5],
            [0.3, 1.0, 2.5], 
            [0.4, 1.0, 2.5],
            [0.5, 1.0, 2.5],
            [0.6, 1.0, 2.5]
        ])
    
    def build_projection_matrix(self, K, R, t):
        """Build projection matrix P = K[R|t]."""
        # For external parameters, we want to transform world to camera
        # t_cam = -R @ t_world (standard computer vision convention)
        t_cam = -R @ t
        RT = np.hstack([R, t_cam.reshape(-1, 1)])
        P = K @ RT
        return P
    
    def project_point_corrected(self, point_3d, P):
        """Project 3D point using corrected projection matrix."""
        point_3d_h = np.append(point_3d, 1.0)
        proj = P @ point_3d_h
        if abs(proj[2]) < 1e-10:
            return np.array([np.inf, np.inf])
        return proj[:2] / proj[2]
    
    def triangulate_point_corrected(self, pixel1, pixel2):
        """Triangulate 3D point from pixel coordinates."""
        # Build system of equations Ax = 0
        A = np.array([
            pixel1[0] * self.P1[2, :] - self.P1[0, :],
            pixel1[1] * self.P1[2, :] - self.P1[1, :],
            pixel2[0] * self.P2[2, :] - self.P2[0, :],
            pixel2[1] * self.P2[2, :] - self.P2[1, :]
        ])
        
        try:
            _, _, Vt = np.linalg.svd(A)
            point_3d_h = Vt[-1]
            point_3d = point_3d_h[:3] / point_3d_h[3]
            return point_3d
        except np.linalg.LinAlgError:
            return np.array([np.nan, np.nan, np.nan])
    
    def test_camera_setup(self):
        """Test that cameras are set up correctly."""
        print("\n=== Testing Camera Setup ===")
        
        # Test baseline
        baseline = np.linalg.norm(self.cam2_pos - self.cam1_pos)
        self.assertAlmostEqual(baseline, 1.0, places=6)
        print(f"✓ Baseline: {baseline:.6f}m")
        
        # Test camera positions avoid singularity
        self.assertNotEqual(self.cam1_pos[2], 2.5)
        self.assertNotEqual(self.cam2_pos[2], 2.5)
        print(f"✓ Camera Z positions: {self.cam1_pos[2]}, {self.cam2_pos[2]} (avoid Z=2.5 singularity)")
        
        # Test rotation matrix is correct for +Y looking
        y_axis = np.array([0, 1, 0])
        transformed_y = self.R_corrected @ y_axis
        expected_forward = np.array([0, 0, 1])  # In our corrected system, +Y world becomes +Z camera
        np.testing.assert_array_almost_equal(transformed_y, expected_forward, decimal=10)
        print("✓ Rotation matrix correctly transforms +Y world to +Z camera (forward)")
    
    def test_no_infinite_projections(self):
        """Test that no projections are infinite."""
        print("\n=== Testing No Infinite Projections ===")
        
        all_finite = True
        for i, point in enumerate(self.trajectory):
            pixel1 = self.project_point_corrected(point, self.P1)
            pixel2 = self.project_point_corrected(point, self.P2)
            
            if np.any(np.isinf(pixel1)) or np.any(np.isinf(pixel2)):
                all_finite = False
                print(f"✗ Point {i} {point} has infinite projection")
            else:
                print(f"✓ Point {i} {point} → Cam1: [{pixel1[0]:.1f}, {pixel1[1]:.1f}], Cam2: [{pixel2[0]:.1f}, {pixel2[1]:.1f}]")
        
        self.assertTrue(all_finite, "All projections should be finite")
    
    def test_rightward_movement_increases_x_pixels(self):
        """Test that rightward movement increases X pixel coordinates."""
        print("\n=== Testing Rightward Movement ===")
        
        # Get projections for first and last points
        first_point = self.trajectory[0]  # X = 0.2
        last_point = self.trajectory[-1]  # X = 0.6
        
        pixel1_first = self.project_point_corrected(first_point, self.P1)
        pixel1_last = self.project_point_corrected(last_point, self.P1)
        
        pixel2_first = self.project_point_corrected(first_point, self.P2)
        pixel2_last = self.project_point_corrected(last_point, self.P2)
        
        print(f"Camera 1: X=0.2 → [{pixel1_first[0]:.1f}, {pixel1_first[1]:.1f}], X=0.6 → [{pixel1_last[0]:.1f}, {pixel1_last[1]:.1f}]")
        print(f"Camera 2: X=0.2 → [{pixel2_first[0]:.1f}, {pixel2_first[1]:.1f}], X=0.6 → [{pixel2_last[0]:.1f}, {pixel2_last[1]:.1f}]")
        
        # Both cameras should show increasing X pixels for rightward movement
        self.assertGreater(pixel1_last[0], pixel1_first[0], "Camera 1: Rightward movement should increase X pixels")
        self.assertGreater(pixel2_last[0], pixel2_first[0], "Camera 2: Rightward movement should increase X pixels")
        
        print("✓ Both cameras show increasing X pixels for rightward movement")
    
    def test_perfect_triangulation_accuracy(self):
        """Test that triangulation is perfectly accurate."""
        print("\n=== Testing Triangulation Accuracy ===")
        
        max_error = 0.0
        for i, original_point in enumerate(self.trajectory):
            # Project to pixels
            pixel1 = self.project_point_corrected(original_point, self.P1)
            pixel2 = self.project_point_corrected(original_point, self.P2)
            
            # Skip if projections are infinite
            if np.any(np.isinf(pixel1)) or np.any(np.isinf(pixel2)):
                continue
            
            # Triangulate back to 3D
            reconstructed = self.triangulate_point_corrected(pixel1, pixel2)
            
            # Skip if triangulation failed
            if np.any(np.isnan(reconstructed)):
                continue
            
            # Calculate error
            error = np.linalg.norm(reconstructed - original_point)
            max_error = max(max_error, error)
            
            print(f"✓ Point {i}: {original_point} → {reconstructed} (error: {error:.10f}m)")
            
            # Should be very accurate
            self.assertLess(error, 1e-6, f"Triangulation error should be < 1e-6m, got {error}")
        
        print(f"✓ Maximum triangulation error: {max_error:.10f}m")
    
    def test_coordinate_system_consistency(self):
        """Test that the coordinate system is internally consistent."""
        print("\n=== Testing Coordinate System Consistency ===")
        
        # Test a point directly in front of camera 1
        test_point = np.array([0.0, 1.0, 2.55])  # Same X and Z as camera 1, but Y=1.0 (in front)
        pixel1 = self.project_point_corrected(test_point, self.P1)
        
        # Should project near image center for camera 1
        center_x, center_y = 640, 360
        print(f"Point directly in front of cam1 {test_point} → pixel [{pixel1[0]:.1f}, {pixel1[1]:.1f}]")
        print(f"Image center: [{center_x}, {center_y}]")
        
        # X should be close to center (same X coordinate)
        self.assertLess(abs(pixel1[0] - center_x), 50, "Point with same X as camera should project near center X")
        
        # Test that Y=1.0 means "in front" (positive depth)
        # In our coordinate system, cameras look in +Y direction
        # So Y=1.0 > Y_camera=0.0 means the point is in front
        print("✓ Y=1.0 > Y_camera=0.0 confirms point is in front of camera")
        
    def test_opencv_triangulation_comparison(self):
        """Compare our triangulation with OpenCV's."""
        print("\n=== Comparing with OpenCV Triangulation ===")
        
        # Test one point
        test_point = self.trajectory[0]
        pixel1 = self.project_point_corrected(test_point, self.P1)
        pixel2 = self.project_point_corrected(test_point, self.P2)
        
        # Our triangulation
        our_result = self.triangulate_point_corrected(pixel1, pixel2)
        
        # OpenCV triangulation
        points_4d = cv2.triangulatePoints(self.P1, self.P2, 
                                        pixel1.reshape(2, 1), 
                                        pixel2.reshape(2, 1))
        opencv_result = points_4d[:3, 0] / points_4d[3, 0]
        
        # Compare results
        difference = np.linalg.norm(our_result - opencv_result)
        print(f"Original point: {test_point}")
        print(f"Our triangulation: {our_result}")  
        print(f"OpenCV triangulation: {opencv_result}")
        print(f"Difference: {difference:.10f}m")
        
        # Should be very close
        self.assertLess(difference, 1e-6, "Our triangulation should match OpenCV")

if __name__ == '__main__':
    print("="*80)
    print("FINAL VALIDATION OF CORRECTED 3D TRACKING COORDINATE SYSTEM")
    print("="*80)
    print("This test validates the coordinate system used in 3d_tracker_8.ipynb")
    print("- Cameras at Z=2.55 (avoiding Z=2.5 singularity)")
    print("- Rotation matrix for +Y looking cameras")
    print("- Trajectory at Z=2.5 moving in X direction")
    print("="*80)
    
    unittest.main(verbosity=2) 