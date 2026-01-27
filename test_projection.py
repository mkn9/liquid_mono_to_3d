import numpy as np

def set_up_cameras():
    """Set up camera matrices and positions with correct geometric transformation."""
    print("\n=== SETTING UP CAMERAS ===")
    
    # Camera intrinsic matrices (same for both cameras)
    K = np.array([
        [1000, 0, 640],
        [0, 1000, 360], 
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Camera world positions (where cameras are located in 3D space)
    cam1_center = np.array([0.0, 0.0, 2.5])  # Camera 1 at origin, raised in Z
    cam2_center = np.array([1.0, 0.0, 2.5])  # Camera 2 translated along X axis, raised in Z
    
    # Both cameras look in the +Y direction (forward direction)
    # Rotation matrices (identity = cameras look in +Y direction in our coordinate system)
    R1 = np.eye(3, dtype=np.float64)
    R2 = np.eye(3, dtype=np.float64)
    
    # Translation vectors for projection matrices
    # t = -R * C where C is the camera center in world coordinates
    t1 = -np.dot(R1, cam1_center.reshape(3, 1))
    t2 = -np.dot(R2, cam2_center.reshape(3, 1))
    
    print(f"Camera 1 world position: {cam1_center}")
    print(f"Camera 2 world position: {cam2_center}")
    print(f"Camera 1 translation vector: {t1.flatten()}")
    print(f"Camera 2 translation vector: {t2.flatten()}")
    
    # Projection matrices P = K[R|t]
    P1 = np.dot(K, np.hstack((R1, t1)))
    P2 = np.dot(K, np.hstack((R2, t2)))
    
    return P1, P2, cam1_center, cam2_center

def generate_synthetic_tracks():
    """Generate synthetic 2D tracks from 3D trajectory."""
    # Get camera matrices
    P1, P2, _, _ = set_up_cameras()
    
    print("\n=== GENERATING SYNTHETIC TRACKS ===")
    
    # Define a 3D trajectory (moving object)
    # Cameras are at Y=0 looking in +Y direction, so objects must be at positive Y to be visible
    original_3d = [
        np.array([0.2, 1.0, -1.0]),  # In front of cameras
        np.array([0.3, 1.0, -1.1]), 
        np.array([0.4, 1.0, -1.2]),
        np.array([0.5, 1.0, -1.3]),
        np.array([0.6, 1.0, -1.4])
    ]
    
    sensor1_track = []
    sensor2_track = []
    
    for i, point_3d in enumerate(original_3d):
        print(f"Processing point {i}: {point_3d}")
        
        # Convert to homogeneous coordinates
        point_3d_h = np.append(point_3d, 1.0)
        
        # Project to camera 1
        proj1 = np.dot(P1, point_3d_h)
        if abs(proj1[2]) > 1e-10:  # Check for valid depth
            pixel1 = proj1[:2] / proj1[2]
        else:
            print(f"Warning: Invalid depth for camera 1 at point {i}")
            pixel1 = np.array([np.inf, np.inf])
        
        # Project to camera 2  
        proj2 = np.dot(P2, point_3d_h)
        if abs(proj2[2]) > 1e-10:  # Check for valid depth
            pixel2 = proj2[:2] / proj2[2]
        else:
            print(f"Warning: Invalid depth for camera 2 at point {i}")
            pixel2 = np.array([np.inf, np.inf])
        
        sensor1_track.append(pixel1)
        sensor2_track.append(pixel2)
        
        print(f"  Camera 1 pixel: {pixel1}")
        print(f"  Camera 2 pixel: {pixel2}")
    
    # Convert to numpy arrays
    sensor1_track = np.array(sensor1_track)
    sensor2_track = np.array(sensor2_track)
    original_3d = np.array(original_3d)
    
    return sensor1_track, sensor2_track, original_3d

if __name__ == "__main__":
    # Test the projection
    sensor1_track, sensor2_track, original_3d = generate_synthetic_tracks()
    
    print('\n=== RESULTS ===')
    print('3D points:', original_3d)
    print('Camera 1 pixels:', sensor1_track)
    print('Camera 2 pixels:', sensor2_track)
    
    print('\n=== PIXEL RANGES ===')
    print('Camera 1 pixel ranges: X=[{:.1f}, {:.1f}], Y=[{:.1f}, {:.1f}]'.format(
        sensor1_track[:, 0].min(), sensor1_track[:, 0].max(),
        sensor1_track[:, 1].min(), sensor1_track[:, 1].max()))
    print('Camera 2 pixel ranges: X=[{:.1f}, {:.1f}], Y=[{:.1f}, {:.1f}]'.format(
        sensor2_track[:, 0].min(), sensor2_track[:, 0].max(),
        sensor2_track[:, 1].min(), sensor2_track[:, 1].max()))
    
    print('\n=== VISIBILITY CHECK ===')
    print('Image bounds: X=[0, 1280], Y=[0, 720]')
    cam1_visible = np.all((sensor1_track[:, 0] >= 0) & (sensor1_track[:, 0] <= 1280) & 
                         (sensor1_track[:, 1] >= 0) & (sensor1_track[:, 1] <= 720))
    cam2_visible = np.all((sensor2_track[:, 0] >= 0) & (sensor2_track[:, 0] <= 1280) & 
                         (sensor2_track[:, 1] >= 0) & (sensor2_track[:, 1] <= 720))
    
    print(f'Camera 1 trajectory visible: {cam1_visible}')
    print(f'Camera 2 trajectory visible: {cam2_visible}') 