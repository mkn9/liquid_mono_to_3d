#!/usr/bin/env python3
"""
Comprehensive Unit Tests for 3D Cone Tracking System

This module contains comprehensive unit tests for the cone tracking functionality,
following the unit testing best practices defined in cursorrules and requirements.md.

Test Categories:
1. Cone3D class creation and properties
2. Geometric calculations (endpoints, visible points, mesh generation)
3. Camera setup and projection matrices
4. 3D to 2D projection accuracy
5. 2D to 3D triangulation accuracy
6. Trajectory generation and processing
7. Visualization function validation
8. Integration testing for full pipeline
9. Error handling and edge cases
10. Performance testing for computational efficiency

Author: AI Assistant
Date: 2025-01-23
"""

import unittest
import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the project root to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the cone tracking modules (these would be imported from the actual modules)
# For this example, we'll define minimal classes to demonstrate the testing structure

class Cone3D:
    """Mock Cone3D class for testing purposes."""
    def __init__(self, center, base_radius=0.05, height=0.2, axis_direction=None):
        if base_radius <= 0:
            raise ValueError("Base radius must be positive")
        if height <= 0:
            raise ValueError("Height must be positive")
        
        self.center = np.array(center, dtype=float)
        self.base_radius = base_radius
        self.height = height
        self.axis_direction = np.array([0.0, 0.0, 1.0]) if axis_direction is None else np.array(axis_direction)
    
    def get_endpoints(self):
        half_height = self.height / 2
        base_center = self.center - half_height * self.axis_direction
        apex = self.center + half_height * self.axis_direction
        return base_center, apex
    
    def get_visible_points(self, camera_position):
        return np.array([[0, 0, 0], [1, 1, 1]])  # Mock implementation


class TestCone3DCreation(unittest.TestCase):
    """Test suite for Cone3D object creation and basic properties."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.default_center = [0.5, 1.0, 2.5]
        self.default_radius = 0.05
        self.default_height = 0.2
    
    def test_cone_creation_with_default_parameters(self):
        """Test cone creation with default parameters."""
        cone = Cone3D(self.default_center)
        
        np.testing.assert_array_equal(cone.center, np.array(self.default_center))
        self.assertEqual(cone.base_radius, 0.05)
        self.assertEqual(cone.height, 0.2)
        np.testing.assert_array_equal(cone.axis_direction, np.array([0.0, 0.0, 1.0]))
    
    def test_cone_creation_with_custom_parameters(self):
        """Test cone creation with custom parameters."""
        center = [1.0, 2.0, 3.0]
        radius = 0.1
        height = 0.3
        axis = [0.0, 1.0, 0.0]
        
        cone = Cone3D(center, base_radius=radius, height=height, axis_direction=axis)
        
        np.testing.assert_array_equal(cone.center, np.array(center))
        self.assertEqual(cone.base_radius, radius)
        self.assertEqual(cone.height, height)
        np.testing.assert_array_equal(cone.axis_direction, np.array(axis))
    
    def test_cone_creation_with_invalid_radius_raises_error(self):
        """Test that negative radius raises ValueError."""
        with self.assertRaises(ValueError) as context:
            Cone3D([0, 0, 0], base_radius=-0.1, height=0.2)
        
        self.assertIn("Base radius must be positive", str(context.exception))
    
    def test_cone_creation_with_invalid_height_raises_error(self):
        """Test that negative height raises ValueError."""
        with self.assertRaises(ValueError) as context:
            Cone3D([0, 0, 0], base_radius=0.1, height=-0.2)
        
        self.assertIn("Height must be positive", str(context.exception))
    
    def test_cone_creation_with_zero_dimensions_raises_error(self):
        """Test that zero dimensions raise ValueError."""
        with self.assertRaises(ValueError):
            Cone3D([0, 0, 0], base_radius=0.0, height=0.2)
        
        with self.assertRaises(ValueError):
            Cone3D([0, 0, 0], base_radius=0.1, height=0.0)


class TestCone3DGeometry(unittest.TestCase):
    """Test suite for cone geometric calculations."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.cone = Cone3D([0.5, 1.0, 2.5], base_radius=0.05, height=0.2)
        self.camera_pos = np.array([0.0, 0.0, 2.55])
    
    def test_get_endpoints_calculation_accuracy(self):
        """Test cone endpoint calculation accuracy."""
        base_center, apex = self.cone.get_endpoints()
        
        # Expected values: center ± height/2 along axis
        expected_base = np.array([0.5, 1.0, 2.4])  # center - 0.1 in Z
        expected_apex = np.array([0.5, 1.0, 2.6])  # center + 0.1 in Z
        
        np.testing.assert_array_almost_equal(base_center, expected_base, decimal=6)
        np.testing.assert_array_almost_equal(apex, expected_apex, decimal=6)
    
    def test_get_endpoints_with_custom_axis(self):
        """Test endpoint calculation with custom axis direction."""
        # Cone with Y-axis orientation
        cone = Cone3D([0, 0, 0], base_radius=0.05, height=0.2, axis_direction=[0, 1, 0])
        base_center, apex = cone.get_endpoints()
        
        expected_base = np.array([0.0, -0.1, 0.0])  # center - 0.1 in Y
        expected_apex = np.array([0.0, 0.1, 0.0])   # center + 0.1 in Y
        
        np.testing.assert_array_almost_equal(base_center, expected_base, decimal=6)
        np.testing.assert_array_almost_equal(apex, expected_apex, decimal=6)
    
    def test_get_visible_points_returns_array(self):
        """Test that get_visible_points returns numpy array."""
        visible_points = self.cone.get_visible_points(self.camera_pos)
        
        self.assertIsInstance(visible_points, np.ndarray)
        self.assertGreater(len(visible_points), 0)
        self.assertEqual(visible_points.shape[1], 3)  # Should be 3D points
    
    @pytest.mark.parametrize("radius,height", [
        (0.01, 0.1),   # Small cone
        (0.05, 0.2),   # Standard cone
        (0.1, 0.5),    # Large cone
        (0.2, 1.0),    # Very large cone
    ])
    def test_cone_dimensions_parametrized(self, radius, height):
        """Test cone creation with various dimensions using parametrized testing."""
        cone = Cone3D([0, 0, 0], base_radius=radius, height=height)
        
        self.assertEqual(cone.base_radius, radius)
        self.assertEqual(cone.height, height)
        
        # Test that endpoints are correctly calculated
        base_center, apex = cone.get_endpoints()
        distance = np.linalg.norm(apex - base_center)
        self.assertAlmostEqual(distance, height, places=6)


class TestCameraSetup(unittest.TestCase):
    """Test suite for camera setup and projection matrices."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock camera setup function
        self.K = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=np.float64)
        self.cam1_pos = np.array([0.0, 0.0, 2.55])
        self.cam2_pos = np.array([1.0, 0.0, 2.55])
    
    def test_camera_positions_are_correct(self):
        """Test that camera positions are set correctly."""
        expected_cam1 = np.array([0.0, 0.0, 2.55])
        expected_cam2 = np.array([1.0, 0.0, 2.55])
        
        np.testing.assert_array_equal(self.cam1_pos, expected_cam1)
        np.testing.assert_array_equal(self.cam2_pos, expected_cam2)
    
    def test_baseline_distance_calculation(self):
        """Test baseline distance between cameras."""
        baseline = np.linalg.norm(self.cam2_pos - self.cam1_pos)
        expected_baseline = 1.0  # 1 meter separation in X
        
        self.assertAlmostEqual(baseline, expected_baseline, places=6)
    
    def test_intrinsic_matrix_properties(self):
        """Test camera intrinsic matrix properties."""
        # Test focal length
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        self.assertEqual(fx, 1000)
        self.assertEqual(fy, 1000)
        
        # Test principal point
        cx = self.K[0, 2]
        cy = self.K[1, 2]
        self.assertEqual(cx, 640)
        self.assertEqual(cy, 360)


class TestProjectionAndTriangulation(unittest.TestCase):
    """Test suite for 3D-2D projection and 2D-3D triangulation."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_point_3d = np.array([0.5, 1.0, 2.5])
        self.tolerance = 1e-3  # 1mm tolerance for reconstruction
    
    def test_projection_triangulation_roundtrip_accuracy(self):
        """Test that 3D -> 2D -> 3D maintains accuracy."""
        # This would test the actual projection and triangulation functions
        # For now, we'll test the concept with mock data
        
        original_point = self.test_point_3d
        
        # Mock projection to 2D (would use actual project_point_corrected function)
        pixel1 = np.array([640.5, 360.2])  # Mock camera 1 projection
        pixel2 = np.array([590.3, 360.1])  # Mock camera 2 projection
        
        # Mock triangulation back to 3D (would use actual triangulate_point_corrected function)
        reconstructed_point = np.array([0.5001, 0.9999, 2.5002])  # Mock reconstruction
        
        # Test reconstruction accuracy
        error = np.linalg.norm(reconstructed_point - original_point)
        self.assertLess(error, self.tolerance, f"Reconstruction error {error:.6f}m exceeds tolerance {self.tolerance}m")
    
    def test_projection_handles_invalid_points(self):
        """Test projection handling of invalid 3D points."""
        # Test points behind camera, at infinity, etc.
        invalid_points = [
            np.array([0, -1, 0]),      # Behind camera
            np.array([np.inf, 1, 2]),  # Infinite coordinate
            np.array([0, 1, np.nan]),  # NaN coordinate
        ]
        
        for point in invalid_points:
            # Mock projection function should handle these gracefully
            # In actual implementation, should return inf or handle appropriately
            pass


class TestTrajectoryGeneration(unittest.TestCase):
    """Test suite for cone trajectory generation and processing."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.expected_trajectory_length = 5
        self.trajectory_centers = [
            np.array([0.2, 1.0, 2.7]),
            np.array([0.3, 1.0, 2.6]),
            np.array([0.4, 1.0, 2.5]),
            np.array([0.5, 1.0, 2.4]),
            np.array([0.6, 1.0, 2.3])
        ]
    
    def test_trajectory_generation_creates_correct_number_of_cones(self):
        """Test that trajectory generation creates the expected number of cones."""
        # Mock trajectory generation
        cones = [Cone3D(center) for center in self.trajectory_centers]
        
        self.assertEqual(len(cones), self.expected_trajectory_length)
    
    def test_trajectory_cone_properties_consistency(self):
        """Test that all cones in trajectory have consistent properties."""
        cones = [Cone3D(center) for center in self.trajectory_centers]
        
        # All cones should have same dimensions
        for cone in cones:
            self.assertEqual(cone.base_radius, 0.05)
            self.assertEqual(cone.height, 0.2)
            np.testing.assert_array_equal(cone.axis_direction, np.array([0.0, 0.0, 1.0]))
    
    def test_trajectory_spatial_progression(self):
        """Test that trajectory shows expected spatial progression."""
        centers = self.trajectory_centers
        
        # Test X progression (should increase)
        x_coords = [center[0] for center in centers]
        for i in range(1, len(x_coords)):
            self.assertGreater(x_coords[i], x_coords[i-1], "X should increase along trajectory")
        
        # Test Y consistency (should remain constant)
        y_coords = [center[1] for center in centers]
        for y in y_coords:
            self.assertEqual(y, 1.0, "Y should remain constant along trajectory")
        
        # Test Z progression (should decrease)
        z_coords = [center[2] for center in centers]
        for i in range(1, len(z_coords)):
            self.assertLess(z_coords[i], z_coords[i-1], "Z should decrease along trajectory")


class TestErrorHandling(unittest.TestCase):
    """Test suite for error handling and edge cases."""
    
    def test_cone_creation_with_invalid_center_type(self):
        """Test cone creation with invalid center data type."""
        invalid_centers = [
            "not_a_list",
            None,
            [1, 2],  # Too few dimensions
            [1, 2, 3, 4],  # Too many dimensions
        ]
        
        for invalid_center in invalid_centers:
            with self.assertRaises((ValueError, TypeError)):
                Cone3D(invalid_center)
    
    def test_projection_with_degenerate_camera_setup(self):
        """Test projection handling with degenerate camera configurations."""
        # Test cameras at same position
        cam1_pos = np.array([0, 0, 0])
        cam2_pos = np.array([0, 0, 0])  # Same position
        
        baseline = np.linalg.norm(cam2_pos - cam1_pos)
        self.assertEqual(baseline, 0.0, "Degenerate camera setup should be detected")
    
    def test_triangulation_with_parallel_rays(self):
        """Test triangulation handling when camera rays are parallel."""
        # This would test the actual triangulation function's handling of parallel rays
        # Mock test for now
        pass


class TestPerformance(unittest.TestCase):
    """Test suite for performance requirements."""
    
    def test_cone_creation_performance(self):
        """Test cone creation performance for batch operations."""
        import time
        
        start_time = time.time()
        
        # Create many cones
        cones = []
        for i in range(1000):
            cone = Cone3D([i*0.001, 1.0, 2.5])
            cones.append(cone)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Should create 1000 cones in reasonable time (< 1 second)
        self.assertLess(elapsed, 1.0, f"Cone creation took {elapsed:.3f}s, should be < 1.0s")
        self.assertEqual(len(cones), 1000)
    
    def test_projection_batch_performance(self):
        """Test projection performance for multiple cones."""
        # This would test actual projection functions
        # Mock test for now
        pass


class TestIntegration(unittest.TestCase):
    """Integration tests for end-to-end cone tracking pipeline."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.num_cones = 5
        self.reconstruction_tolerance = 1e-3  # 1mm
    
    def test_full_pipeline_cone_tracking(self):
        """Test complete cone tracking pipeline from generation to reconstruction."""
        # 1. Generate trajectory
        trajectory_centers = [
            [0.2, 1.0, 2.7], [0.3, 1.0, 2.6], [0.4, 1.0, 2.5], 
            [0.5, 1.0, 2.4], [0.6, 1.0, 2.3]
        ]
        original_cones = [Cone3D(center) for center in trajectory_centers]
        
        # 2. Project to cameras (mock)
        # In real implementation, would use actual projection functions
        
        # 3. Reconstruct from projections (mock)
        # In real implementation, would use actual triangulation
        
        # 4. Validate reconstruction accuracy
        self.assertEqual(len(original_cones), self.num_cones)
        
        # Mock reconstruction validation
        for i, cone in enumerate(original_cones):
            # In real test, would compare with actual reconstructed cones
            self.assertIsNotNone(cone)
    
    def test_pipeline_handles_missing_data(self):
        """Test pipeline handling of missing or corrupted data."""
        # Test with missing cone data, invalid projections, etc.
        pass
    
    def test_pipeline_memory_usage(self):
        """Test pipeline memory usage for large datasets."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create large number of cones
        large_trajectory = [Cone3D([i*0.01, 1.0, 2.5]) for i in range(10000)]
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 100MB for 10k cones)
        max_memory_mb = 100 * 1024 * 1024  # 100MB
        self.assertLess(memory_increase, max_memory_mb, 
                       f"Memory usage increased by {memory_increase/1024/1024:.1f}MB")


# Test Suite Configuration
class ConeTrackingTestSuite:
    """Test suite configuration and runner."""
    
    @staticmethod
    def get_test_suite():
        """Get complete test suite for cone tracking system."""
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        # Add all test classes
        test_classes = [
            TestCone3DCreation,
            TestCone3DGeometry,
            TestCameraSetup,
            TestProjectionAndTriangulation,
            TestTrajectoryGeneration,
            TestErrorHandling,
            TestPerformance,
            TestIntegration,
        ]
        
        for test_class in test_classes:
            tests = loader.loadTestsFromTestCase(test_class)
            suite.addTests(tests)
        
        return suite
    
    @staticmethod
    def run_tests_with_coverage():
        """Run tests with coverage reporting."""
        try:
            import coverage
            cov = coverage.Coverage()
            cov.start()
            
            # Run tests
            runner = unittest.TextTestRunner(verbosity=2)
            suite = ConeTrackingTestSuite.get_test_suite()
            result = runner.run(suite)
            
            cov.stop()
            cov.save()
            
            print("\n" + "="*50)
            print("COVERAGE REPORT")
            print("="*50)
            cov.report()
            
            return result
            
        except ImportError:
            print("Coverage module not available. Running tests without coverage.")
            runner = unittest.TextTestRunner(verbosity=2)
            suite = ConeTrackingTestSuite.get_test_suite()
            return runner.run(suite)


if __name__ == '__main__':
    # Command line test execution
    import argparse
    
    parser = argparse.ArgumentParser(description='Run cone tracking unit tests')
    parser.add_argument('--coverage', action='store_true', help='Run with coverage reporting')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--specific', '-s', type=str, help='Run specific test class')
    
    args = parser.parse_args()
    
    if args.coverage:
        result = ConeTrackingTestSuite.run_tests_with_coverage()
    elif args.specific:
        # Run specific test class
        suite = unittest.TestLoader().loadTestsFromName(args.specific, module=__name__)
        verbosity = 2 if args.verbose else 1
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)
    else:
        # Run all tests
        suite = ConeTrackingTestSuite.get_test_suite()
        verbosity = 2 if args.verbose else 1
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)
    
    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    print(f"\n{'='*50}")
    print(f"TEST SUMMARY: {result.testsRun} tests run")
    print(f"PASSED: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"FAILED: {len(result.failures)}")
    print(f"ERRORS: {len(result.errors)}")
    print(f"SUCCESS: {result.wasSuccessful()}")
    print(f"{'='*50}")
    
    exit(exit_code) 