# -*- coding: utf-8 -*-
import numpy as np
import cv2
import math

def test_camera_orientation():
    """Test the actual camera orientation based on boresight direction."""
    print("=== TESTING CAMERA ORIENTATION ===")
    
    # Calculate boresight direction as defined in the code
    elevation_angle = -45  # degrees (looking down)
    azimuth_angle = 0      # degrees (forward direction)
    
    elevation_rad = np.radians(elevation_angle)
    azimuth_rad = np.radians(azimuth_angle)
    
    boresight_dir = np.array([
        np.cos(elevation_rad) * np.sin(azimuth_rad),  # X component
        np.cos(elevation_rad) * np.cos(azimuth_rad),  # Y component  
        np.sin(elevation_rad)                         # Z component
    ])
    
    print(f"Elevation angle: {elevation_angle} degrees")
    print(f"Azimuth angle: {azimuth_angle} degrees")
    print(f"Boresight direction vector: {boresight_dir}")
    print(f"X component: {boresight_dir[0]:.6f}")
    print(f"Y component: {boresight_dir[1]:.6f} (positive = forward in Y)")
    print(f"Z component: {boresight_dir[2]:.6f} (negative = downward)")
    
    print(f"\nCameras are looking:")
    print(f"- Primary direction: +Y (forward)")
    print(f"- Secondary direction: -Z (downward at 45 degrees)")
    print(f"- No X component (azimuth = 0 degrees)")
    
    return boresight_dir

def test_coordinate_system_with_correct_orientation():
    """Test coordinate system understanding with correct camera orientation."""
    print("\n=== TESTING COORDINATE SYSTEM (CORRECT ORIENTATION) ===")
    
    # Camera positions
    cam1_center = np.array([0.0, 0.0, 2.5])  # Camera 1 at origin, raised in Z
    cam2_center = np.array([1.0, 0.0, 2.5])  # Camera 2 translated along X axis, raised in Z
    
    print(f"Camera 1 position: {cam1_center}")
    print(f"Camera 2 position: {cam2_center}")
    print("Both cameras look in +Y direction (forward) with -45 degrees elevation")
    
    # Test trajectory points
    trajectory_points = [
        np.array([0.2, 1.0, 5.0]),  # Moving right in X, forward in Y
        np.array([0.3, 1.0, 4.9]), 
        np.array([0.4, 1.0, 4.8]),
        np.array([0.5, 1.0, 4.7]),
        np.array([0.6, 1.0, 4.6])
    ]
    
    print(f"\nTrajectory analysis:")
    print(f"X coordinates: {[pt[0] for pt in trajectory_points]} (moving RIGHT)")
    print(f"Y coordinates: {[pt[1] for pt in trajectory_points]} (constant, in front of cameras)")
    print(f"Z coordinates: {[pt[2] for pt in trajectory_points]} (moving closer to cameras)")
    
    print(f"\nExpected behavior:")
    print(f"- Camera 1 (X=0): Object moves from X=0.2 to X=0.6 -> moves RIGHT -> pixels INCREASE")
    print(f"- Camera 2 (X=1): Object moves from X=0.2 to X=0.6, but camera is at X=1")
    print(f"  Relative to Camera 2: X=0.2-1.0=-0.8 to X=0.6-1.0=-0.4 -> moves RIGHT -> pixels INCREASE")
    print(f"- Both cameras should show INCREASING X pixel coordinates!")

def test_projection_matrix_construction_multiple_methods():
    """Test projection matrix construction with multiple methods."""
    print("\n=== TESTING PROJECTION MATRIX CONSTRUCTION (MULTIPLE METHODS) ===")
    
    # Camera intrinsics
    K = np.array([
        [1000, 0, 640],
        [0, 1000, 360], 
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Camera positions
    cam1_center = np.array([0.0, 0.0, 2.5])
    cam2_center = np.array([1.0, 0.0, 2.5])
    
    print(f"Intrinsic matrix K:\n{K}")
    print(f"Camera 1 center: {cam1_center}")
    print(f"Camera 2 center: {cam2_center}")
    
    # Test Case 1: Identity rotation (looking in +Y direction, not -Z)
    print(f"\n--- Test Case 1: Identity Rotation (Current Setup) ---")
    R1 = np.eye(3, dtype=np.float64)
    R2 = np.eye(3, dtype=np.float64)
    
    # Method A: t = -R * C (standard computer vision, but for -Z looking cameras)
    t1_A = -np.dot(R1, cam1_center.reshape(3, 1))
    t2_A = -np.dot(R2, cam2_center.reshape(3, 1))
    
    # Method B: t = camera position (incorrect but test anyway)
    t1_B = cam1_center.reshape(3, 1)
    t2_B = cam2_center.reshape(3, 1)
    
    print(f"Method A (t = -R*C): t1={t1_A.flatten()}, t2={t2_A.flatten()}")
    print(f"Method B (t = C): t1={t1_B.flatten()}, t2={t2_B.flatten()}")
    
    # Test Case 2: Rotation to look in +Y direction explicitly
    print(f"\n--- Test Case 2: Explicit +Y Direction Rotation ---")
    # Rotation matrix to look in +Y direction (90 degrees rotation around X-axis)
    R_y_looking = np.array([
        [1, 0, 0],
        [0, 0, -1],  # Y becomes -Z
        [0, 1, 0]    # Z becomes Y
    ], dtype=np.float64)
    
    R1_explicit = R_y_looking
    R2_explicit = R_y_looking
    
    t1_explicit = -np.dot(R1_explicit, cam1_center.reshape(3, 1))
    t2_explicit = -np.dot(R2_explicit, cam2_center.reshape(3, 1))
    
    print(f"Explicit +Y rotation matrix:\n{R_y_looking}")
    print(f"Explicit method: t1={t1_explicit.flatten()}, t2={t2_explicit.flatten()}")
    
    # Construct all projection matrices
    P1_A = np.dot(K, np.hstack((R1, t1_A)))
    P2_A = np.dot(K, np.hstack((R2, t2_A)))
    
    P1_B = np.dot(K, np.hstack((R1, t1_B)))
    P2_B = np.dot(K, np.hstack((R2, t2_B)))
    
    P1_explicit = np.dot(K, np.hstack((R1_explicit, t1_explicit)))
    P2_explicit = np.dot(K, np.hstack((R2_explicit, t2_explicit)))
    
    return {
        'Method_A': (P1_A, P2_A),
        'Method_B': (P1_B, P2_B), 
        'Explicit_Y': (P1_explicit, P2_explicit)
    }

def test_point_projection_comprehensive(projection_matrices):
    """Test point projection with multiple methods and analyze results."""
    print("\n=== COMPREHENSIVE POINT PROJECTION TESTING ===")
    
    # Test trajectory
    trajectory_points = [
        np.array([0.2, 1.0, 5.0]),
        np.array([0.3, 1.0, 4.9]), 
        np.array([0.4, 1.0, 4.8]),
        np.array([0.5, 1.0, 4.7]),
        np.array([0.6, 1.0, 4.6])
    ]
    
    for method_name, (P1, P2) in projection_matrices.items():
        print(f"\n--- Testing {method_name} ---")
        
        cam1_pixels = []
        cam2_pixels = []
        
        for i, point_3d in enumerate(trajectory_points):
            # Convert to homogeneous coordinates
            point_3d_h = np.append(point_3d, 1.0)
            
            # Project to both cameras
            proj1 = np.dot(P1, point_3d_h)
            proj2 = np.dot(P2, point_3d_h)
            
            # Convert to pixel coordinates
            if abs(proj1[2]) > 1e-10:
                pixel1 = proj1[:2] / proj1[2]
                cam1_pixels.append(pixel1)
            else:
                pixel1 = np.array([np.inf, np.inf])
                cam1_pixels.append(pixel1)
                
            if abs(proj2[2]) > 1e-10:
                pixel2 = proj2[:2] / proj2[2]
                cam2_pixels.append(pixel2)
            else:
                pixel2 = np.array([np.inf, np.inf])
                cam2_pixels.append(pixel2)
            
            print(f"  Point {i+1} {point_3d}: Cam1=[{pixel1[0]:.1f}, {pixel1[1]:.1f}], Cam2=[{pixel2[0]:.1f}, {pixel2[1]:.1f}]")
        
        # Analyze X coordinate trends
        cam1_x_coords = [p[0] for p in cam1_pixels if np.isfinite(p[0])]
        cam2_x_coords = [p[0] for p in cam2_pixels if np.isfinite(p[0])]
        
        if len(cam1_x_coords) > 1:
            cam1_trend = "INCREASING" if cam1_x_coords[-1] > cam1_x_coords[0] else "DECREASING"
        else:
            cam1_trend = "INSUFFICIENT_DATA"
            
        if len(cam2_x_coords) > 1:
            cam2_trend = "INCREASING" if cam2_x_coords[-1] > cam2_x_coords[0] else "DECREASING"
        else:
            cam2_trend = "INSUFFICIENT_DATA"
        
        print(f"  Camera 1 X trend: {cam1_trend} ({cam1_x_coords[0]:.1f} -> {cam1_x_coords[-1]:.1f})")
        print(f"  Camera 2 X trend: {cam2_trend} ({cam2_x_coords[0]:.1f} -> {cam2_x_coords[-1]:.1f})")
        
        # Check if within image bounds (1280x720)
        cam1_in_bounds = all(0 <= p[0] <= 1280 and 0 <= p[1] <= 720 for p in cam1_pixels if np.isfinite(p[0]))
        cam2_in_bounds = all(0 <= p[0] <= 1280 and 0 <= p[1] <= 720 for p in cam2_pixels if np.isfinite(p[0]))
        
        print(f"  Camera 1 within bounds: {cam1_in_bounds}")
        print(f"  Camera 2 within bounds: {cam2_in_bounds}")

def test_geometric_consistency():
    """Test geometric consistency between different coordinate transformations."""
    print("\n=== TESTING GEOMETRIC CONSISTENCY ===")
    
    # Camera positions
    cam1_pos = np.array([0.0, 0.0, 2.5])
    cam2_pos = np.array([1.0, 0.0, 2.5])
    
    # Test point
    test_point = np.array([0.4, 1.0, 4.8])  # Middle of trajectory
    
    print(f"Test point: {test_point}")
    print(f"Camera 1 position: {cam1_pos}")
    print(f"Camera 2 position: {cam2_pos}")
    
    # Calculate relative positions
    rel_to_cam1 = test_point - cam1_pos
    rel_to_cam2 = test_point - cam2_pos
    
    print(f"Relative to Camera 1: {rel_to_cam1}")
    print(f"Relative to Camera 2: {rel_to_cam2}")
    
    # Calculate angles in XY plane (looking down from above)
    angle_cam1 = math.degrees(math.atan2(rel_to_cam1[0], rel_to_cam1[1]))  # X/Y angle
    angle_cam2 = math.degrees(math.atan2(rel_to_cam2[0], rel_to_cam2[1]))  # X/Y angle
    
    print(f"Angle from Camera 1 (XY plane): {angle_cam1:.1f} degrees (positive = right)")
    print(f"Angle from Camera 2 (XY plane): {angle_cam2:.1f} degrees (positive = right)")
    
    # Expected behavior
    print(f"\nExpected behavior:")
    print(f"- Both angles should be positive (object to the right of camera center)")
    print(f"- Camera 1 angle should be larger (object further right relative to camera)")
    print(f"- Both cameras should show object moving rightward -> increasing X pixels")

def test_opencv_validation():
    """Validate using OpenCV's projection functions."""
    print("\n=== OPENCV VALIDATION ===")
    
    # Camera matrix
    K = np.array([
        [1000, 0, 640],
        [0, 1000, 360], 
        [0, 0, 1]
    ], dtype=np.float32)
    
    dist_coeffs = np.zeros(4, dtype=np.float32)
    
    # Test different rotation setups
    test_cases = [
        {
            'name': 'Identity (Current)',
            'rvec1': np.array([0.0, 0.0, 0.0], dtype=np.float32),
            'tvec1': np.array([0.0, 0.0, -2.5], dtype=np.float32),
            'rvec2': np.array([0.0, 0.0, 0.0], dtype=np.float32), 
            'tvec2': np.array([-1.0, 0.0, -2.5], dtype=np.float32)
        },
        {
            'name': 'Y-Looking (90 deg X rotation)',
            'rvec1': np.array([np.pi/2, 0.0, 0.0], dtype=np.float32),  # 90 deg around X
            'tvec1': np.array([0.0, -2.5, 0.0], dtype=np.float32),
            'rvec2': np.array([np.pi/2, 0.0, 0.0], dtype=np.float32),
            'tvec2': np.array([-1.0, -2.5, 0.0], dtype=np.float32)
        }
    ]
    
    # Test trajectory
    object_points = np.array([
        [0.2, 1.0, 5.0],
        [0.3, 1.0, 4.9],
        [0.4, 1.0, 4.8],
        [0.5, 1.0, 4.7],
        [0.6, 1.0, 4.6]
    ], dtype=np.float32)
    
    for case in test_cases:
        print(f"\n--- OpenCV Test: {case['name']} ---")
        
        # Project points
        points1, _ = cv2.projectPoints(object_points, case['rvec1'], case['tvec1'], K, dist_coeffs)
        points2, _ = cv2.projectPoints(object_points, case['rvec2'], case['tvec2'], K, dist_coeffs)
        
        print("Camera 1 projections:")
        cam1_x = []
        for i, pt in enumerate(points1):
            x, y = pt[0][0], pt[0][1]
            cam1_x.append(x)
            print(f"  Point {i+1}: [{x:.1f}, {y:.1f}]")
            
        print("Camera 2 projections:")
        cam2_x = []
        for i, pt in enumerate(points2):
            x, y = pt[0][0], pt[0][1]
            cam2_x.append(x)
            print(f"  Point {i+1}: [{x:.1f}, {y:.1f}]")
        
        # Analyze trends
        cam1_trend = "INCREASING" if cam1_x[-1] > cam1_x[0] else "DECREASING"
        cam2_trend = "INCREASING" if cam2_x[-1] > cam2_x[0] else "DECREASING"
        
        print(f"Camera 1 X trend: {cam1_trend}")
        print(f"Camera 2 X trend: {cam2_trend}")

def main():
    """Run all comprehensive tests."""
    print("COMPREHENSIVE CAMERA SYSTEM TESTING")
    print("="*50)
    
    # Test 1: Camera orientation
    boresight_dir = test_camera_orientation()
    
    # Test 2: Coordinate system with correct orientation
    test_coordinate_system_with_correct_orientation()
    
    # Test 3: Multiple projection matrix construction methods
    projection_matrices = test_projection_matrix_construction_multiple_methods()
    
    # Test 4: Comprehensive point projection testing
    test_point_projection_comprehensive(projection_matrices)
    
    # Test 5: Geometric consistency
    test_geometric_consistency()
    
    # Test 6: OpenCV validation
    test_opencv_validation()
    
    print("\n" + "="*50)
    print("COMPREHENSIVE TESTING COMPLETE")

if __name__ == "__main__":
    main() 