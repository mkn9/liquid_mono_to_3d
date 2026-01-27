import numpy as np
import cv2

def test_camera_coordinate_system():
    """Test basic camera coordinate system understanding."""
    print("=== TESTING CAMERA COORDINATE SYSTEM ===")
    
    # Camera setup
    cam1_center = np.array([0.0, 0.0, 2.5])  # Camera 1 position
    cam2_center = np.array([1.0, 0.0, 2.5])  # Camera 2 position
    
    # Test points
    test_points = [
        np.array([0.0, 1.0, 3.0]),  # Directly in front of cam1, to the left of cam2
        np.array([1.0, 1.0, 3.0]),  # Directly in front of cam2, to the right of cam1
        np.array([0.5, 1.0, 3.0]),  # Between cameras
    ]
    
    print(f"Camera 1 at: {cam1_center}")
    print(f"Camera 2 at: {cam2_center}")
    print("\nTest points:")
    for i, pt in enumerate(test_points):
        print(f"  Point {i+1}: {pt}")
        # Distance from each camera
        dist1 = np.linalg.norm(pt - cam1_center)
        dist2 = np.linalg.norm(pt - cam2_center)
        print(f"    Distance from cam1: {dist1:.3f}, from cam2: {dist2:.3f}")
        
        # Relative position
        rel1 = pt - cam1_center
        rel2 = pt - cam2_center
        print(f"    Relative to cam1: {rel1}")
        print(f"    Relative to cam2: {rel2}")
        print(f"    Expected in cam1: point to RIGHT (positive X)")
        print(f"    Expected in cam2: point to LEFT (negative X)")
        print()

def test_projection_matrix_construction():
    """Test each step of projection matrix construction."""
    print("=== TESTING PROJECTION MATRIX CONSTRUCTION ===")
    
    # Camera intrinsics
    K = np.array([
        [1000, 0, 640],
        [0, 1000, 360], 
        [0, 0, 1]
    ], dtype=np.float64)
    print(f"Intrinsic matrix K:\n{K}")
    
    # Camera positions
    cam1_center = np.array([0.0, 0.0, 2.5])
    cam2_center = np.array([1.0, 0.0, 2.5])
    
    # Rotation matrices (identity = looking along -Z)
    R1 = np.eye(3, dtype=np.float64)
    R2 = np.eye(3, dtype=np.float64)
    print(f"Rotation matrices (identity):\nR1:\n{R1}\nR2:\n{R2}")
    
    # Test different translation vector formulations
    print("\n--- Testing Translation Vector Formulations ---")
    
    # Method 1: t = -R * C (standard computer vision)
    t1_method1 = -np.dot(R1, cam1_center.reshape(3, 1))
    t2_method1 = -np.dot(R2, cam2_center.reshape(3, 1))
    print(f"Method 1 (t = -R*C):")
    print(f"  t1: {t1_method1.flatten()}")
    print(f"  t2: {t2_method1.flatten()}")
    
    # Method 2: t = camera position (incorrect but let's test)
    t1_method2 = cam1_center.reshape(3, 1)
    t2_method2 = cam2_center.reshape(3, 1)
    print(f"Method 2 (t = camera position):")
    print(f"  t1: {t1_method2.flatten()}")
    print(f"  t2: {t2_method2.flatten()}")
    
    # Construct projection matrices for both methods
    P1_m1 = np.dot(K, np.hstack((R1, t1_method1)))
    P2_m1 = np.dot(K, np.hstack((R2, t2_method1)))
    
    P1_m2 = np.dot(K, np.hstack((R1, t1_method2)))
    P2_m2 = np.dot(K, np.hstack((R2, t2_method2)))
    
    print(f"\nProjection matrices:")
    print(f"Method 1 - P1:\n{P1_m1}")
    print(f"Method 1 - P2:\n{P2_m1}")
    print(f"Method 2 - P1:\n{P1_m2}")
    print(f"Method 2 - P2:\n{P2_m2}")
    
    return (P1_m1, P2_m1), (P1_m2, P2_m2)

def test_point_projection(P_matrices, method_name):
    """Test projection of specific points."""
    print(f"\n=== TESTING POINT PROJECTION - {method_name} ===")
    
    P1, P2 = P_matrices
    
    # Test points - adjusted for proper field of view
    test_points = [
        np.array([0.2, 1.0, 5.0]),  # Further trajectory for visible range
        np.array([0.3, 1.0, 4.9]), 
        np.array([0.4, 1.0, 4.8]),
        np.array([0.5, 1.0, 4.7]),
        np.array([0.6, 1.0, 4.6])
    ]
    
    print("Projecting original trajectory points:")
    for i, point_3d in enumerate(test_points):
        print(f"\nPoint {i+1}: {point_3d}")
        
        # Convert to homogeneous coordinates
        point_3d_h = np.append(point_3d, 1.0)
        
        # Project to both cameras
        proj1 = np.dot(P1, point_3d_h)
        proj2 = np.dot(P2, point_3d_h)
        
        print(f"  Homogeneous projections:")
        print(f"    Camera 1: {proj1}")
        print(f"    Camera 2: {proj2}")
        
        # Convert to pixel coordinates
        if abs(proj1[2]) > 1e-10:
            pixel1 = proj1[:2] / proj1[2]
            print(f"  Camera 1 pixel: [{pixel1[0]:.1f}, {pixel1[1]:.1f}]")
        else:
            print(f"  Camera 1: Invalid depth!")
            
        if abs(proj2[2]) > 1e-10:
            pixel2 = proj2[:2] / proj2[2]
            print(f"  Camera 2 pixel: [{pixel2[0]:.1f}, {pixel2[1]:.1f}]")
        else:
            print(f"  Camera 2: Invalid depth!")

def test_opencv_projection():
    """Test using OpenCV's projection functions for validation."""
    print("\n=== TESTING OPENCV PROJECTION (VALIDATION) ===")
    
    # Camera matrices using OpenCV convention
    camera_matrix = np.array([
        [1000, 0, 640],
        [0, 1000, 360], 
        [0, 0, 1]
    ], dtype=np.float32)
    
    dist_coeffs = np.zeros(4, dtype=np.float32)  # No distortion
    
    # Camera poses
    # Camera 1: at origin
    rvec1 = np.zeros(3, dtype=np.float32)  # No rotation
    tvec1 = np.array([0.0, 0.0, -2.5], dtype=np.float32)  # Translation in camera coords
    
    # Camera 2: translated
    rvec2 = np.zeros(3, dtype=np.float32)  # No rotation  
    tvec2 = np.array([-1.0, 0.0, -2.5], dtype=np.float32)  # Translation in camera coords
    
    # Test points (adjusted coordinates)
    object_points = np.array([
        [0.2, 1.0, 5.0],
        [0.3, 1.0, 4.9],
        [0.4, 1.0, 4.8],
        [0.5, 1.0, 4.7],
        [0.6, 1.0, 4.6]
    ], dtype=np.float32)
    
    print("Using OpenCV projectPoints:")
    
    # Project using OpenCV
    image_points1, _ = cv2.projectPoints(object_points, rvec1, tvec1, camera_matrix, dist_coeffs)
    image_points2, _ = cv2.projectPoints(object_points, rvec2, tvec2, camera_matrix, dist_coeffs)
    
    print("Camera 1 projections (OpenCV):")
    for i, pt in enumerate(image_points1):
        print(f"  Point {i+1}: [{pt[0][0]:.1f}, {pt[0][1]:.1f}]")
        
    print("Camera 2 projections (OpenCV):")
    for i, pt in enumerate(image_points2):
        print(f"  Point {i+1}: [{pt[0][0]:.1f}, {pt[0][1]:.1f}]")

def analyze_x_dimension_inconsistency():
    """Analyze the X dimension inconsistency between 3D and 2D views."""
    print("\n=== ANALYZING X DIMENSION INCONSISTENCY ===")
    
    print("Expected behavior:")
    print("1. Object moves from X=0.2 to X=0.6 (rightward in world coordinates)")
    print("2. Camera 1 at X=0: object moves RIGHT relative to camera → pixels should INCREASE")
    print("3. Camera 2 at X=1: object moves LEFT relative to camera → pixels should DECREASE")
    print("4. In 3D view: trajectory should appear rightmost from Camera 1's perspective")
    print("5. In 2D view: Camera 1 should show increasing X pixels, Camera 2 decreasing X pixels")
    
    print("\nPotential error sources:")
    print("A. Incorrect camera matrix construction (translation vectors)")
    print("B. Wrong coordinate system convention (right-hand vs left-hand)")
    print("C. Incorrect projection matrix multiplication order")
    print("D. Image coordinate system confusion (origin at top-left vs bottom-left)")
    print("E. Camera orientation errors (looking in wrong direction)")

if __name__ == "__main__":
    # Run all tests
    test_camera_coordinate_system()
    
    matrices_m1, matrices_m2 = test_projection_matrix_construction()
    
    test_point_projection(matrices_m1, "METHOD 1 (t = -R*C)")
    test_point_projection(matrices_m2, "METHOD 2 (t = camera position)")
    
    test_opencv_projection()
    
    analyze_x_dimension_inconsistency()
    
    print("\n=== UNIT TESTS COMPLETE ===")
    print("Review the outputs to identify the source of X dimension inconsistency.") 