#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Cylinder Tracking System
Tests all core functionality of the 3D cylinder tracking implementation.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import sys

class Cylinder3D:
    """Represents a 3D cylinder object with position, orientation, and dimensions."""
    
    def __init__(self, center, radius=0.05, height=0.2, axis_direction=None):
        self.center = np.array(center, dtype=float)
        self.radius = radius
        self.height = height
        
        if axis_direction is None:
            self.axis_direction = np.array([0.0, 0.0, 1.0])
        else:
            self.axis_direction = np.array(axis_direction, dtype=float)
            self.axis_direction = self.axis_direction / np.linalg.norm(self.axis_direction)
    
    def get_endpoints(self):
        """Get the top and bottom center points of the cylinder."""
        half_height = self.height / 2
        offset = half_height * self.axis_direction
        bottom_center = self.center - offset
        top_center = self.center + offset
        return bottom_center, top_center
    
    def get_visible_points(self, camera_position):
        """Get key points on the cylinder that are visible from a camera position."""
        bottom_center, top_center = self.get_endpoints()
        
        # Calculate perpendicular vectors to cylinder axis
        if abs(self.axis_direction[2]) < 0.9:
            perp1 = np.cross(self.axis_direction, [0, 0, 1])
        else:
            perp1 = np.cross(self.axis_direction, [1, 0, 0])
        
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(self.axis_direction, perp1)
        perp2 = perp2 / np.linalg.norm(perp2)
        
        # Direction from cylinder center to camera
        to_camera = camera_position - self.center
        to_camera_norm = to_camera / np.linalg.norm(to_camera)
        
        # Project camera direction onto the perpendicular plane
        camera_proj = to_camera_norm - np.dot(to_camera_norm, self.axis_direction) * self.axis_direction
        if np.linalg.norm(camera_proj) > 1e-6:
            camera_proj = camera_proj / np.linalg.norm(camera_proj)
        else:
            camera_proj = perp1
        
        # Key points: center, top, bottom, and visible edges
        points = [
            self.center,  # Center point
            bottom_center,  # Bottom center
            top_center,  # Top center
            self.center + self.radius * camera_proj,  # Closest edge point
            self.center - self.radius * camera_proj,  # Farthest edge point
            bottom_center + self.radius * camera_proj,  # Bottom closest edge
            bottom_center - self.radius * camera_proj,  # Bottom farthest edge
            top_center + self.radius * camera_proj,  # Top closest edge
            top_center - self.radius * camera_proj,  # Top farthest edge
        ]
        
        return np.array(points)

def set_up_cameras():
    """Set up corrected camera system - cameras look in +Y direction."""
    # Camera intrinsic matrices
    K = np.array([
        [1000, 0, 640],
        [0, 1000, 360], 
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Camera positions
    cam1_center = np.array([0.0, 0.0, 2.55])
    cam2_center = np.array([1.0, 0.0, 2.55])
    
    # Corrected rotation matrices for +Y looking cameras
    R_corrected = np.array([
        [1,  0,  0],
        [0,  0, -1],
        [0,  1,  0]
    ], dtype=np.float64)
    
    R1 = R_corrected.copy()
    R2 = R_corrected.copy()
    
    # Translation vectors
    t1 = -np.dot(R1, cam1_center.reshape(3, 1))
    t2 = -np.dot(R2, cam2_center.reshape(3, 1))
    
    # Projection matrices
    P1 = np.dot(K, np.hstack((R1, t1)))
    P2 = np.dot(K, np.hstack((R2, t2)))
    
    return P1, P2, cam1_center, cam2_center

def project_point_corrected(P, point_3d):
    """Project 3D world point to 2D image coordinates."""
    point_3d_h = np.append(point_3d, 1.0)
    proj = np.dot(P, point_3d_h)
    
    if abs(proj[2]) < 1e-10:
        return np.array([float('inf'), float('inf')])
    
    return proj[:2] / proj[2]

def triangulate_point_corrected(pixel1, pixel2, P1, P2):
    """Triangulate 3D point from stereo 2D observations."""
    A = np.array([
        pixel1[0] * P1[2, :] - P1[0, :],
        pixel1[1] * P1[2, :] - P1[1, :],
        pixel2[0] * P2[2, :] - P2[0, :],
        pixel2[1] * P2[2, :] - P2[1, :]
    ])
    
    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1, :]
    
    return X_h[:3] / X_h[3]

def project_cylinder(cylinder, P, camera_pos):
    """Project cylinder to 2D image coordinates."""
    visible_points = cylinder.get_visible_points(camera_pos)
    
    projected_points = []
    for point in visible_points:
        pixel = project_point_corrected(P, point)
        projected_points.append(pixel)
    
    return np.array(projected_points)

def generate_cylinder_trajectory():
    """Generate a trajectory of cylinder objects moving through 3D space."""
    trajectory_centers = [
        np.array([0.2, 1.0, 2.7]),
        np.array([0.3, 1.0, 2.6]), 
        np.array([0.4, 1.0, 2.5]),
        np.array([0.5, 1.0, 2.4]),
        np.array([0.6, 1.0, 2.3])
    ]
    
    cylinders = []
    for center in trajectory_centers:
        cylinder = Cylinder3D(
            center=center,
            radius=0.05,
            height=0.2,
            axis_direction=[0, 0, 1]
        )
        cylinders.append(cylinder)
    
    return cylinders

def project_cylinder_trajectory(cylinders, P1, P2, cam1_pos, cam2_pos):
    """Project cylinder trajectory to both camera views."""
    camera1_projections = []
    camera2_projections = []
    
    for cylinder in cylinders:
        proj1 = project_cylinder(cylinder, P1, cam1_pos)
        proj2 = project_cylinder(cylinder, P2, cam2_pos)
        camera1_projections.append(proj1)
        camera2_projections.append(proj2)
    
    return camera1_projections, camera2_projections

def reconstruct_cylinder_trajectory(camera1_projections, camera2_projections, P1, P2):
    """Reconstruct 3D cylinder trajectory from 2D projections."""
    reconstructed_cylinders = []
    
    for proj1, proj2 in zip(camera1_projections, camera2_projections):
        reconstructed_points = []
        
        for p1, p2 in zip(proj1, proj2):
            if (np.isinf(p1[0]) or np.isinf(p1[1]) or 
                np.isinf(p2[0]) or np.isinf(p2[1])):
                reconstructed_points.append(np.array([np.nan, np.nan, np.nan]))
                continue
            
            point_3d = triangulate_point_corrected(p1, p2, P1, P2)
            reconstructed_points.append(point_3d)
        
        reconstructed_points = np.array(reconstructed_points)
        
        if not np.isnan(reconstructed_points[0]).any():
            reconstructed_center = reconstructed_points[0]
            reconstructed_cylinder = Cylinder3D(
                center=reconstructed_center,
                radius=0.05,
                height=0.2,
                axis_direction=[0, 0, 1]
            )
            reconstructed_cylinders.append(reconstructed_cylinder)
        else:
            reconstructed_cylinders.append(None)
    
    return reconstructed_cylinders

class CylinderTrackingTests:
    """Comprehensive unit tests for the cylinder tracking system."""
    
    def __init__(self):
        self.test_results = []
        self.passed = 0
        self.failed = 0
    
    def assert_equal(self, actual, expected, tolerance=1e-6, test_name=""):
        """Assert that two values are equal within tolerance."""
        if isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray):
            if np.allclose(actual, expected, atol=tolerance):
                self.test_results.append(f"✅ PASS: {test_name}")
                self.passed += 1
                return True
            else:
                self.test_results.append(f"❌ FAIL: {test_name} - Expected {expected}, got {actual}")
                self.failed += 1
                return False
        elif isinstance(actual, tuple) and isinstance(expected, tuple):
            if actual == expected:
                self.test_results.append(f"✅ PASS: {test_name}")
                self.passed += 1
                return True
            else:
                self.test_results.append(f"❌ FAIL: {test_name} - Expected {expected}, got {actual}")
                self.failed += 1
                return False
        else:
            try:
                if abs(actual - expected) <= tolerance:
                    self.test_results.append(f"✅ PASS: {test_name}")
                    self.passed += 1
                    return True
                else:
                    self.test_results.append(f"❌ FAIL: {test_name} - Expected {expected}, got {actual}")
                    self.failed += 1
                    return False
            except TypeError:
                # Handle cases where subtraction is not supported
                if actual == expected:
                    self.test_results.append(f"✅ PASS: {test_name}")
                    self.passed += 1
                    return True
                else:
                    self.test_results.append(f"❌ FAIL: {test_name} - Expected {expected}, got {actual}")
                    self.failed += 1
                    return False
    
    def assert_true(self, condition, test_name=""):
        """Assert that condition is true."""
        if condition:
            self.test_results.append(f"✅ PASS: {test_name}")
            self.passed += 1
            return True
        else:
            self.test_results.append(f"❌ FAIL: {test_name} - Condition was False")
            self.failed += 1
            return False
    
    def test_cylinder_creation(self):
        """Test cylinder object creation and basic properties."""
        print("Testing cylinder creation...")
        
        # Test default cylinder
        cylinder = Cylinder3D([0, 0, 0])
        self.assert_equal(cylinder.center, np.array([0, 0, 0]), test_name="Default cylinder center")
        self.assert_equal(cylinder.radius, 0.05, test_name="Default cylinder radius")
        self.assert_equal(cylinder.height, 0.2, test_name="Default cylinder height")
        self.assert_equal(cylinder.axis_direction, np.array([0, 0, 1]), test_name="Default cylinder axis")
        
        # Test custom cylinder
        custom_cylinder = Cylinder3D([1, 2, 3], radius=0.1, height=0.5, axis_direction=[0, 1, 0])
        self.assert_equal(custom_cylinder.center, np.array([1, 2, 3]), test_name="Custom cylinder center")
        self.assert_equal(custom_cylinder.radius, 0.1, test_name="Custom cylinder radius")
        self.assert_equal(custom_cylinder.height, 0.5, test_name="Custom cylinder height")
        self.assert_equal(custom_cylinder.axis_direction, np.array([0, 1, 0]), test_name="Custom cylinder axis")
    
    def test_cylinder_endpoints(self):
        """Test cylinder endpoint calculation."""
        print("Testing cylinder endpoints...")
        
        # Vertical cylinder
        cylinder = Cylinder3D([0, 0, 0], height=2.0)
        bottom, top = cylinder.get_endpoints()
        self.assert_equal(bottom, np.array([0, 0, -1]), test_name="Vertical cylinder bottom")
        self.assert_equal(top, np.array([0, 0, 1]), test_name="Vertical cylinder top")
        
        # Horizontal cylinder
        horizontal_cylinder = Cylinder3D([0, 0, 0], height=2.0, axis_direction=[1, 0, 0])
        bottom, top = horizontal_cylinder.get_endpoints()
        self.assert_equal(bottom, np.array([-1, 0, 0]), test_name="Horizontal cylinder bottom")
        self.assert_equal(top, np.array([1, 0, 0]), test_name="Horizontal cylinder top")
    
    def test_camera_setup(self):
        """Test camera setup and projection matrices."""
        print("Testing camera setup...")
        
        P1, P2, cam1_pos, cam2_pos = set_up_cameras()
        
        # Test camera positions
        self.assert_equal(cam1_pos, np.array([0, 0, 2.55]), test_name="Camera 1 position")
        self.assert_equal(cam2_pos, np.array([1, 0, 2.55]), test_name="Camera 2 position")
        
        # Test projection matrix shapes
        self.assert_equal(P1.shape, (3, 4), test_name="Camera 1 projection matrix shape")
        self.assert_equal(P2.shape, (3, 4), test_name="Camera 2 projection matrix shape")
        
        # Test baseline distance
        baseline = np.linalg.norm(cam2_pos - cam1_pos)
        self.assert_equal(baseline, 1.0, test_name="Camera baseline distance")
    
    def test_projection_triangulation(self):
        """Test 3D to 2D projection and 2D to 3D triangulation."""
        print("Testing projection and triangulation...")
        
        P1, P2, cam1_pos, cam2_pos = set_up_cameras()
        
        # Test point in front of cameras
        test_point = np.array([0.5, 1.0, 2.5])
        
        # Project to both cameras
        pixel1 = project_point_corrected(P1, test_point)
        pixel2 = project_point_corrected(P2, test_point)
        
        # Check that projections are valid
        self.assert_true(not np.isinf(pixel1[0]) and not np.isinf(pixel1[1]), 
                        "Camera 1 projection is valid")
        self.assert_true(not np.isinf(pixel2[0]) and not np.isinf(pixel2[1]), 
                        "Camera 2 projection is valid")
        
        # Triangulate back to 3D
        reconstructed = triangulate_point_corrected(pixel1, pixel2, P1, P2)
        
        # Check reconstruction accuracy
        self.assert_equal(reconstructed, test_point, tolerance=1e-3, 
                         test_name="3D point reconstruction accuracy")
    
    def test_cylinder_projection(self):
        """Test cylinder projection to camera views."""
        print("Testing cylinder projection...")
        
        P1, P2, cam1_pos, cam2_pos = set_up_cameras()
        cylinder = Cylinder3D([0.5, 1.0, 2.5])
        
        # Project cylinder to both cameras
        proj1 = project_cylinder(cylinder, P1, cam1_pos)
        proj2 = project_cylinder(cylinder, P2, cam2_pos)
        
        # Check that we get expected number of points
        self.assert_equal(len(proj1), 9, test_name="Camera 1 projection point count")
        self.assert_equal(len(proj2), 9, test_name="Camera 2 projection point count")
        
        # Check that center point projections are valid
        center_proj1 = proj1[0]  # First point is center
        center_proj2 = proj2[0]
        
        self.assert_true(not np.isinf(center_proj1[0]) and not np.isinf(center_proj1[1]), 
                        "Camera 1 center projection is valid")
        self.assert_true(not np.isinf(center_proj2[0]) and not np.isinf(center_proj2[1]), 
                        "Camera 2 center projection is valid")
    
    def test_trajectory_generation(self):
        """Test cylinder trajectory generation."""
        print("Testing trajectory generation...")
        
        cylinders = generate_cylinder_trajectory()
        
        # Check number of cylinders
        self.assert_equal(len(cylinders), 5, test_name="Trajectory cylinder count")
        
        # Check that all cylinders have correct properties
        for i, cylinder in enumerate(cylinders):
            self.assert_equal(cylinder.radius, 0.05, test_name=f"Cylinder {i+1} radius")
            self.assert_equal(cylinder.height, 0.2, test_name=f"Cylinder {i+1} height")
            self.assert_equal(cylinder.axis_direction, np.array([0, 0, 1]), test_name=f"Cylinder {i+1} axis")
        
        # Check trajectory progression (X should increase, Z should decrease)
        x_coords = [c.center[0] for c in cylinders]
        z_coords = [c.center[2] for c in cylinders]
        
        self.assert_true(all(x_coords[i] <= x_coords[i+1] for i in range(len(x_coords)-1)),
                        "X coordinates increase along trajectory")
        self.assert_true(all(z_coords[i] >= z_coords[i+1] for i in range(len(z_coords)-1)),
                        "Z coordinates decrease along trajectory")
    
    def test_full_pipeline(self):
        """Test the complete tracking pipeline."""
        print("Testing full tracking pipeline...")
        
        # Generate trajectory and set up cameras
        cylinders = generate_cylinder_trajectory()
        P1, P2, cam1_pos, cam2_pos = set_up_cameras()
        
        # Project to camera views
        cam1_projections, cam2_projections = project_cylinder_trajectory(
            cylinders, P1, P2, cam1_pos, cam2_pos)
        
        # Reconstruct trajectory
        reconstructed_cylinders = reconstruct_cylinder_trajectory(
            cam1_projections, cam2_projections, P1, P2)
        
        # Check that we reconstructed the right number of cylinders
        valid_reconstructions = [c for c in reconstructed_cylinders if c is not None]
        self.assert_true(len(valid_reconstructions) >= 4, 
                        f"At least 4 cylinders reconstructed (got {len(valid_reconstructions)})")
        
        # Check reconstruction accuracy for valid cylinders
        for i, (original, reconstructed) in enumerate(zip(cylinders, reconstructed_cylinders)):
            if reconstructed is not None:
                error = np.linalg.norm(original.center - reconstructed.center)
                self.assert_true(error < 0.01, 
                               f"Cylinder {i+1} reconstruction error < 1cm (error: {error:.4f}m)")
    
    def run_all_tests(self):
        """Run all tests and display results."""
        print("=" * 60)
        print("RUNNING CYLINDER TRACKING UNIT TESTS")
        print("=" * 60)
        
        # Run all test methods
        self.test_cylinder_creation()
        self.test_cylinder_endpoints()
        self.test_camera_setup()
        self.test_projection_triangulation()
        self.test_cylinder_projection()
        self.test_trajectory_generation()
        self.test_full_pipeline()
        
        # Display results
        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        
        for result in self.test_results:
            print(result)
        
        print(f"\nSUMMARY: {self.passed} passed, {self.failed} failed")
        
        if self.failed == 0:
            print("🎉 ALL TESTS PASSED!")
        else:
            print(f"⚠️  {self.failed} TESTS FAILED")
        
        return self.failed == 0

def main():
    """Main function to run all tests."""
    print("EXECUTING COMPREHENSIVE CYLINDER TRACKING UNIT TESTS")
    print("=" * 80)
    
    test_suite = CylinderTrackingTests()
    all_tests_passed = test_suite.run_all_tests()
    
    if all_tests_passed:
        print("\n🎉 ALL TESTS PASSED! Cylinder tracking system is working correctly.")
        print("\n📋 Validated functionality:")
        print("   • Cylinder object creation and property management")
        print("   • Geometric calculations (endpoints, visible points)")
        print("   • Camera system setup and projection matrices")
        print("   • 3D to 2D projection and 2D to 3D triangulation")
        print("   • Cylinder projection to camera views")
        print("   • Trajectory generation and progression")
        print("   • Complete end-to-end tracking pipeline")
        print("   • Sub-centimeter reconstruction accuracy")
        return 0
    else:
        print("\n⚠️ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 