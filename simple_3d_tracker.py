#!/usr/bin/env python3
"""
Simple 3D Tracker

This script demonstrates basic 3D track reconstruction from 2D mono tracks.
All computation is performed on an EC2 instance, with local execution on a MacBook.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_print(message):
    """Print debug messages if debugging is enabled."""
    if DEBUG:
        print("DEBUG: {}".format(message), flush=True)
        sys.stdout.flush()

def set_up_cameras():
    """Set up camera parameters for two cameras."""
    logger.info("Setting up cameras")
    
    # Camera intrinsic matrices
    K1 = np.array([
        [1000, 0, 640],
        [0, 1000, 360],
        [0, 0, 1]
    ])
    
    K2 = np.array([
        [1000, 0, 640],
        [0, 1000, 360],
        [0, 0, 1]
    ])
    
    # Camera extrinsic parameters
    # Camera 1 at origin, but raised in Z
    R1 = np.eye(3)
    t1 = np.array([[0.0], [0.0], [2.5]])  # Camera 1 raised to Z=2.5
    
    # Camera 2 is translated along X axis and raised in Z
    R2 = np.eye(3)
    t2 = np.array([[1.0], [0.0], [2.5]])  # Camera 2 raised to Z=2.5
    
    # Store camera positions
    cam1_pos = t1.flatten()  # Convert to 1D array
    cam2_pos = t2.flatten()  # Convert to 1D array
    
    logger.info(f"Camera 1 position: {cam1_pos}")
    logger.info(f"Camera 2 position: {cam2_pos}")
    
    # Compute projection matrices
    P1 = K1 @ np.hstack((R1, t1))
    P2 = K2 @ np.hstack((R2, t2))
    
    return P1, P2, cam1_pos, cam2_pos

def verify_axis_limits(ax, name="unnamed"):
    """Verify the axis limits after they've been set."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    logger.debug(f"Axis '{name}' limits - X: {xlim}, Y: {ylim}, Z: {zlim}")
    return xlim, ylim, zlim

def plot_camera(ax, camera_pos, color, label, boresight_length=3.0):
    """Plot a camera in 3D space with frustum and boresight line of sight.
    
    Args:
        ax: Matplotlib 3D axis
        camera_pos: Camera position as [x, y, z]
        color: Color for the camera marker and lines
        label: Camera label for the legend
        boresight_length: Length of the boresight line in meters
    """
    logger.info(f"Plotting camera: {label}")
    logger.info(f"Camera position: {camera_pos}")
    
    # Plot camera position with a massive marker
    ax.scatter(camera_pos[0], camera_pos[1], camera_pos[2], 
              color=color, marker='*', s=1000, label=label,
              edgecolor='black', linewidth=3.0)
    
    # Add a vertical stem line from ground to camera
    ax.plot([camera_pos[0], camera_pos[0]], 
            [camera_pos[1], camera_pos[1]], 
            [0, camera_pos[2]], 
            color=color, linewidth=5.0, alpha=0.7)
    
    # Add text label with coordinates
    ax.text(camera_pos[0], camera_pos[1], camera_pos[2] + 0.3,
            f"{label}\n({camera_pos[0]:.1f}, {camera_pos[1]:.1f}, {camera_pos[2]:.1f})",
            color=color, fontsize=12, weight='bold', ha='center')
    
    # Define boresight direction (looking forward and down at 45 degrees)
    elevation_angle = -45  # degrees
    azimuth_angle = 0  # degrees
    
    # Convert angles to direction vector
    elevation_rad = np.radians(elevation_angle)
    azimuth_rad = np.radians(azimuth_angle)
    
    boresight_dir = np.array([
        np.cos(elevation_rad) * np.sin(azimuth_rad),
        np.cos(elevation_rad) * np.cos(azimuth_rad),
        np.sin(elevation_rad)
    ])
    
    # Normalize the direction vector
    boresight_dir = boresight_dir / np.linalg.norm(boresight_dir)
    
    # Calculate boresight end point
    boresight_end = camera_pos + boresight_dir * boresight_length
    
    # Draw boresight line of sight
    ax.plot([camera_pos[0], boresight_end[0]],
            [camera_pos[1], boresight_end[1]],
            [camera_pos[2], boresight_end[2]],
            color=color, linewidth=3.0, linestyle='--',
            label=f"{label} Boresight")
    
    # Add boresight angle text
    midpoint = (camera_pos + boresight_end) / 2
    ax.text(midpoint[0], midpoint[1], midpoint[2],
            f"Elevation: {elevation_angle}°\nAzimuth: {azimuth_angle}°",
            color=color, fontsize=10, ha='left', va='bottom')
    
    # Draw viewing frustum
    fov = 60  # degrees
    frustum_distance = boresight_length * 0.8
    
    # Calculate frustum corners
    fov_rad = np.radians(fov / 2)
    frustum_width = 2 * frustum_distance * np.tan(fov_rad)
    frustum_height = frustum_width * 0.75  # Assuming 4:3 aspect ratio
    
    # Create rotation matrix for boresight direction
    z_axis = boresight_dir
    x_axis = np.array([-z_axis[1], z_axis[0], 0])
    if np.all(x_axis == 0):
        x_axis = np.array([1, 0, 0])
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
    
    # Define frustum corners in camera space
    corners = np.array([
        [ frustum_width/2,  frustum_height/2, frustum_distance],
        [-frustum_width/2,  frustum_height/2, frustum_distance],
        [-frustum_width/2, -frustum_height/2, frustum_distance],
        [ frustum_width/2, -frustum_height/2, frustum_distance]
    ])
    
    # Transform corners to world space
    corners = np.dot(corners, rotation_matrix.T)
    corners = corners + camera_pos
    
    # Draw frustum lines
    for i in range(4):
        # Lines from camera to corners
        ax.plot([camera_pos[0], corners[i,0]],
                [camera_pos[1], corners[i,1]],
                [camera_pos[2], corners[i,2]],
                color=color, linewidth=1.0, alpha=0.3)
        
        # Lines between corners
        next_i = (i + 1) % 4
        ax.plot([corners[i,0], corners[next_i,0]],
                [corners[i,1], corners[next_i,1]],
                [corners[i,2], corners[next_i,2]],
                color=color, linewidth=1.0, alpha=0.3)
    
    logger.info(f"Camera {label} plotted successfully with boresight")

def generate_synthetic_tracks():
    """Generate synthetic 2D tracks from two cameras observing a 3D point."""
    # A 3D point moving in a straight line
    points_3d = [
        np.array([0.2, 0.3, 3.0]),
        np.array([0.3, 0.4, 2.9]),
        np.array([0.4, 0.5, 2.8]),
        np.array([0.5, 0.6, 2.7]),
        np.array([0.6, 0.7, 2.6])
    ]
    
    P1, P2, _, _ = set_up_cameras()
    
    # Project 3D points to 2D for both cameras
    sensor1_track = []
    sensor2_track = []
    
    for point_3d in points_3d:
        # Convert to homogeneous coordinates
        point_3d_h = np.append(point_3d, 1)
        
        # Project to 2D
        point_2d_1 = P1 @ point_3d_h
        point_2d_2 = P2 @ point_3d_h
        
        # Convert from homogeneous to image coordinates
        point_2d_1 = point_2d_1[:2] / point_2d_1[2]
        point_2d_2 = point_2d_2[:2] / point_2d_2[2]
        
        sensor1_track.append(point_2d_1)
        sensor2_track.append(point_2d_2)
    
    # Convert to numpy arrays
    sensor1_track = np.array(sensor1_track)
    sensor2_track = np.array(sensor2_track)
    points_3d = np.array(points_3d)
    
    return sensor1_track, sensor2_track, points_3d

def triangulate_tracks(sensor1_track, sensor2_track, P1, P2):
    """Triangulate 3D points from 2D tracks from two cameras."""
    points_3d = []
    
    for pt1, pt2 in zip(sensor1_track, sensor2_track):
        # Reshape points for OpenCV triangulation
        p1 = np.array(pt1, dtype=np.float32).reshape(2, 1)
        p2 = np.array(pt2, dtype=np.float32).reshape(2, 1)
        
        # Triangulate the point
        point_homog = cv2.triangulatePoints(P1, P2, p1, p2)
        
        # Convert from homogeneous coordinates
        point_3d = (point_homog[:3] / point_homog[3]).flatten()
        points_3d.append(point_3d)
    
    return np.array(points_3d)

def generate_camera_images(sensor1_track, sensor2_track):
    """Generate synthetic camera images showing the tracked points."""
    # Create a directory for the images
    os.makedirs('camera_images', exist_ok=True)
    
    # Create blank images for both cameras
    img_width, img_height = 1280, 720
    
    # Generate images for each frame
    for i, (pt1, pt2) in enumerate(zip(sensor1_track, sensor2_track)):
        # Create images for both cameras
        img1 = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        img2 = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        
        # Check for valid points (not infinity or NaN)
        pt1_valid = np.isfinite(pt1[0]) and np.isfinite(pt1[1])
        pt2_valid = np.isfinite(pt2[0]) and np.isfinite(pt2[1])
        
        # Convert track points to integers for drawing (only if valid)
        if pt1_valid:
            # Clamp coordinates to image bounds
            x1 = max(0, min(img_width - 1, int(np.round(pt1[0]))))
            y1 = max(0, min(img_height - 1, int(np.round(pt1[1]))))
            
            # Draw the track point on camera 1 image
            cv2.circle(img1, (x1, y1), 10, (0, 255, 0), -1)  # Green circle
            
            # Add point coordinates
            cv2.putText(img1, f"Point: ({x1}, {y1})", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            # Mark invalid point
            cv2.putText(img1, "Point: INVALID", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if pt2_valid:
            # Clamp coordinates to image bounds
            x2 = max(0, min(img_width - 1, int(np.round(pt2[0]))))
            y2 = max(0, min(img_height - 1, int(np.round(pt2[1]))))
            
            # Draw the track point on camera 2 image
            cv2.circle(img2, (x2, y2), 10, (0, 255, 0), -1)  # Green circle
            
            # Add point coordinates
            cv2.putText(img2, f"Point: ({x2}, {y2})", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            # Mark invalid point
            cv2.putText(img2, "Point: INVALID", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Add frame number text
        cv2.putText(img1, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img2, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save images
        cv2.imwrite(f'camera_images/camera1_frame{i}.png', img1)
        cv2.imwrite(f'camera_images/camera2_frame{i}.png', img2)
    
    return img_width, img_height

def create_camera_only_plot():
    """Create a plot that ONLY shows cameras to verify visibility."""
    logger.info("\n=== CREATING CAMERA-ONLY SANITY CHECK PLOT ===")
    
    # Get camera positions
    _, _, cam1_pos, cam2_pos = set_up_cameras()
    logger.info(f"CAMERA_ONLY: Camera 1 at {cam1_pos}, Camera 2 at {cam2_pos}")
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set axis limits with plenty of room
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.0)
    ax.set_zlim(-1.0, 4.0)  # Expanded range to make cameras visible
    
    # Print current Z-axis range
    zlim = ax.get_zlim()
    logger.info(f"CAMERA_ONLY: Initial Z-axis range: {zlim}")
    
    # Add a ground plane
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 10), np.linspace(-0.5, 1.0, 10))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')
    
    # Draw Camera 1 (super large red star)
    logger.info(f"CAMERA_ONLY: Drawing Camera 1 at {cam1_pos}")
    cam1_marker = ax.scatter(
        cam1_pos[0], cam1_pos[1], cam1_pos[2],
        color='red', marker='*', s=3000,  # HUGE star marker
        edgecolor='black', linewidth=3
    )
    
    # Add stem for Camera 1
    ax.plot(
        [cam1_pos[0], cam1_pos[0]],
        [cam1_pos[1], cam1_pos[1]],
        [0, cam1_pos[2]],
        'r-', linewidth=10  # Extra thick line
    )
    
    # Add label with arrow for Camera 1
    ax.text(
        cam1_pos[0], cam1_pos[1], cam1_pos[2] + 0.3,
        f"CAMERA 1\n({cam1_pos[0]}, {cam1_pos[1]}, {cam1_pos[2]})",
        color='red', fontsize=16, weight='bold', ha='center'
    )
    
    # Draw Camera 2 (super large blue diamond)
    logger.info(f"CAMERA_ONLY: Drawing Camera 2 at {cam2_pos}")
    cam2_marker = ax.scatter(
        cam2_pos[0], cam2_pos[1], cam2_pos[2],
        color='blue', marker='D', s=3000,  # HUGE diamond marker
        edgecolor='black', linewidth=3
    )
    
    # Add stem for Camera 2
    ax.plot(
        [cam2_pos[0], cam2_pos[0]],
        [cam2_pos[1], cam2_pos[1]],
        [0, cam2_pos[2]],
        'b-', linewidth=10  # Extra thick line
    )
    
    # Add label with arrow for Camera 2
    ax.text(
        cam2_pos[0], cam2_pos[1], cam2_pos[2] + 0.3,
        f"CAMERA 2\n({cam2_pos[0]}, {cam2_pos[1]}, {cam2_pos[2]})",
        color='blue', fontsize=16, weight='bold', ha='center'
    )
    
    # Add explicit text about Z-axis range
    ax.text(
        0.5, 0.01, 0.95,
        f"Z-AXIS RANGE: {zlim[0]} to {zlim[1]}\nCamera 1 Z: {cam1_pos[2]}, Camera 2 Z: {cam2_pos[2]}",
        transform=ax.transAxes,
        fontsize=14, color='black',
        bbox=dict(facecolor='white', alpha=0.7)
    )
    
    # Set labels and title
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    ax.set_zlabel('Z', fontsize=14)
    ax.set_title('CAMERA POSITIONS SANITY CHECK', fontsize=16)
    
    # Adjust view angle
    ax.view_init(elev=20, azim=-40)
    
    # Print final Z-axis range
    zlim = ax.get_zlim()
    logger.info(f"CAMERA_ONLY: Final Z-axis range: {zlim}")
    
    # Save figure - WITHOUT tight_layout
    plt.savefig('camera_only_sanity_check.png', dpi=300)
    logger.info("CAMERA_ONLY: Saved camera-only plot to camera_only_sanity_check.png")
    plt.close()

def create_standalone_3d_view(original_3d, reconstructed_3d, cam1_pos, cam2_pos):
    """Create a large, standalone 3D visualization plot.
    
    This creates a figure with just the 3D view, optimized for clarity and detail.
    The plot can be rotated interactively when displayed in a Python environment.
    """
    logger.info("\n=== CREATING STANDALONE 3D VISUALIZATION ===")
    
    # Create a large figure
    plt.figure(figsize=(16, 12))
    ax = plt.axes(projection='3d')
    
    # Set consistent view limits with more space
    ax.set_xlim(-1.0, 2.0)
    ax.set_ylim(-1.0, 1.5)
    ax.set_zlim(-0.5, 4.0)
    
    # Enable grid for better spatial reference
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Plot ground plane
    xx, yy = np.meshgrid(np.linspace(-1.0, 2.0, 20), np.linspace(-1.0, 1.5, 20))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')
    
    # Plot cameras with increased boresight length for clarity
    plot_camera(ax, cam1_pos, 'red', 'Camera 1', boresight_length=4.0)
    plot_camera(ax, cam2_pos, 'blue', 'Camera 2', boresight_length=4.0)
    
    # Plot all 3D points with increased size and better visibility
    ax.scatter(
        [p[0] for p in original_3d],
        [p[1] for p in original_3d],
        [p[2] for p in original_3d],
        color='green', label='Original Track', s=200, alpha=0.7
    )
    
    ax.scatter(
        reconstructed_3d[:, 0],
        reconstructed_3d[:, 1],
        reconstructed_3d[:, 2],
        color='orange', label='Reconstructed Track', s=200, alpha=0.7
    )
    
    # Add trajectory lines with increased thickness
    ax.plot(
        [p[0] for p in original_3d],
        [p[1] for p in original_3d],
        [p[2] for p in original_3d],
        'g-', linewidth=3, alpha=0.5
    )
    
    ax.plot(
        reconstructed_3d[:, 0],
        reconstructed_3d[:, 1],
        reconstructed_3d[:, 2],
        color='orange', linewidth=3, alpha=0.5
    )
    
    # Add frame numbers next to points
    for i, (orig, recon) in enumerate(zip(original_3d, reconstructed_3d)):
        ax.text(orig[0], orig[1], orig[2], f' {i}', color='green', fontsize=12)
        ax.text(recon[0], recon[1], recon[2], f' {i}', color='orange', fontsize=12)
    
    # Set labels with increased size
    ax.set_xlabel('X (meters)', fontsize=14, labelpad=10)
    ax.set_ylabel('Y (meters)', fontsize=14, labelpad=10)
    ax.set_zlabel('Z (meters)', fontsize=14, labelpad=10)
    
    # Increase tick label size
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add title
    plt.title('3D Track Reconstruction\nRotate plot with mouse when viewing interactively', 
              fontsize=16, pad=20)
    
    # Move legend outside the plot
    ax.legend(bbox_to_anchor=(1.15, 1), loc='upper right', fontsize=12)
    
    # Add text about interactive viewing
    plt.figtext(0.02, 0.02, 
                'Note: This plot supports interactive rotation and zoom when viewed in a Python environment.\n'
                'Use left mouse button to rotate, right button to zoom, middle button to pan.',
                fontsize=10, ha='left', va='bottom')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save with high DPI for better quality
    plt.savefig('3d_visualization.png', dpi=300, bbox_inches='tight')
    logger.info("Saved standalone 3D visualization to '3d_visualization.png'")
    plt.close()

def create_frame_comparison(original_3d, reconstructed_3d, sensor1_track, sensor2_track, img_dims):
    """Create separate visualizations for each frame, saved as individual files."""
    logger.info("\n=== CREATING FRAME COMPARISON VISUALIZATIONS ===")
    
    # Get camera positions
    _, _, cam1_pos, cam2_pos = set_up_cameras()
    logger.info(f"Frame comparison using cameras at:\nCamera 1: {cam1_pos}\nCamera 2: {cam2_pos}")
    
    img_width, img_height = img_dims
    
    # Create a directory for the frame comparisons
    os.makedirs('frame_comparisons', exist_ok=True)
    
    # Process each frame
    for frame in range(len(original_3d)):
        logger.info(f"\n--- Processing Frame {frame} ---")
        
        # Create figure with fixed size and adjusted subplot ratios
        fig = plt.figure(figsize=(20, 8))
        gs = plt.GridSpec(1, 3, width_ratios=[1.5, 1, 1])
        
        # 3D plot
        ax_3d = fig.add_subplot(gs[0], projection='3d')
        
        # Set consistent view limits
        ax_3d.set_xlim(-0.5, 1.5)
        ax_3d.set_ylim(-0.5, 1.0)
        ax_3d.set_zlim(-0.5, 3.5)
        
        # Enable grid for better spatial reference
        ax_3d.grid(True, linestyle='--', alpha=0.6)
        
        # Plot ground plane
        xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 10), np.linspace(-0.5, 1.0, 10))
        zz = np.zeros_like(xx)
        ax_3d.plot_surface(xx, yy, zz, alpha=0.2, color='gray')
        
        # Plot cameras
        plot_camera(ax_3d, cam1_pos, 'red', 'Camera 1')
        plot_camera(ax_3d, cam2_pos, 'blue', 'Camera 2')
        
        # Plot all 3D points with increased size
        ax_3d.scatter(
            [p[0] for p in original_3d],
            [p[1] for p in original_3d],
            [p[2] for p in original_3d],
            color='green', label='Original', s=100, alpha=0.5
        )
        
        ax_3d.scatter(
            reconstructed_3d[:, 0],
            reconstructed_3d[:, 1],
            reconstructed_3d[:, 2],
            color='orange', label='Reconstructed', s=100, alpha=0.5
        )
        
        # Highlight current points
        ax_3d.scatter(
            [original_3d[frame][0]],
            [original_3d[frame][1]],
            [original_3d[frame][2]],
            color='green', s=300, edgecolor='black', label='Current Original'
        )
        
        ax_3d.scatter(
            [reconstructed_3d[frame][0]],
            [reconstructed_3d[frame][1]],
            [reconstructed_3d[frame][2]],
            color='orange', s=300, edgecolor='black', label='Current Reconstructed'
        )
        
        # Add trajectory lines
        ax_3d.plot(
            [p[0] for p in original_3d],
            [p[1] for p in original_3d],
            [p[2] for p in original_3d],
            'g-', alpha=0.5
        )
        
        ax_3d.plot(
            reconstructed_3d[:, 0],
            reconstructed_3d[:, 1],
            reconstructed_3d[:, 2],
            'orange', alpha=0.5
        )
        
        # Set labels and title
        ax_3d.set_xlabel('X (m)', fontsize=12)
        ax_3d.set_ylabel('Y (m)', fontsize=12)
        ax_3d.set_zlabel('Z (m)', fontsize=12)
        ax_3d.legend(loc='upper right', fontsize=10)
        ax_3d.set_title(f'3D Track - Frame {frame}', fontsize=14)
        
        # Set consistent view angle
        ax_3d.view_init(elev=20, azim=-40)
        
        # Camera views
        ax_cam1 = fig.add_subplot(gs[1])
        ax_cam2 = fig.add_subplot(gs[2])
        
        # Load and display camera images
        img1 = cv2.imread(f'camera_images/camera1_frame{frame}.png')
        img2 = cv2.imread(f'camera_images/camera2_frame{frame}.png')
        
        if img1 is None or img2 is None:
            logger.warning(f"Warning: Could not load camera images for frame {frame}")
            continue
        
        # Convert BGR to RGB
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        ax_cam1.imshow(img1)
        ax_cam2.imshow(img2)
        
        ax_cam1.set_title(f'Camera 1 - Frame {frame}', fontsize=14)
        ax_cam2.set_title(f'Camera 2 - Frame {frame}', fontsize=14)
        
        # Remove axis ticks for camera views
        ax_cam1.set_xticks([])
        ax_cam1.set_yticks([])
        ax_cam2.set_xticks([])
        ax_cam2.set_yticks([])
        
        # Add note about interactive viewing
        plt.figtext(0.02, 0.02, 
                   'Note: 3D plot supports interactive rotation when viewed in a Python environment',
                   fontsize=10, ha='left', va='bottom')
        
        # Adjust subplot spacing
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.3)
        
        # Save figure
        plt.savefig(f'frame_comparisons/frame_{frame}_comparison.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved frame {frame} comparison")
        plt.close()
    
    logger.info("Frame comparisons saved in 'frame_comparisons/' directory")

def create_interactive_visualization(original_3d, reconstructed_3d, sensor1_track, sensor2_track, img_dims):
    """Create an interactive 3D visualization using Plotly.
    
    Args:
        original_3d: List of original 3D points
        reconstructed_3d: List of reconstructed 3D points
        sensor1_track: List of 2D points from camera 1
        sensor2_track: List of 2D points from camera 2
        img_dims: Tuple of (width, height) for the image dimensions
    
    Returns:
        plotly.graph_objects.Figure: Interactive 3D figure
    """
    logger.info("Creating interactive visualization...")
    logger.debug(f"Original 3D points: {original_3d}")
    logger.debug(f"Reconstructed 3D points: {reconstructed_3d}")
    logger.debug(f"Sensor 1 track: {sensor1_track}")
    logger.debug(f"Sensor 2 track: {sensor2_track}")
    logger.debug(f"Image dimensions: {img_dims}")

    # Set up cameras
    P1, P2, cam1_pos, cam2_pos = set_up_cameras()
    logger.debug(f"Camera 1 position: {cam1_pos}")
    logger.debug(f"Camera 2 position: {cam2_pos}")
    
    # Convert points to numpy arrays for easier manipulation
    original_3d = np.array(original_3d)
    reconstructed_3d = np.array(reconstructed_3d)
    
    # Create the 3D figure
    fig = go.Figure()
    
    # Plot original trajectory
    fig.add_trace(go.Scatter3d(
        x=original_3d[:, 0],
        y=original_3d[:, 1],
        z=original_3d[:, 2],
        mode='lines+markers',
        name='Original Track',
        line=dict(color='blue', width=4),
        marker=dict(size=8)
    ))
    
    # Plot reconstructed trajectory
    fig.add_trace(go.Scatter3d(
        x=reconstructed_3d[:, 0],
        y=reconstructed_3d[:, 1],
        z=reconstructed_3d[:, 2],
        mode='lines+markers',
        name='Reconstructed Track',
        line=dict(color='red', width=4),
        marker=dict(size=8)
    ))
    
    # Plot camera positions
    fig.add_trace(go.Scatter3d(
        x=[cam1_pos[0]],
        y=[cam1_pos[1]],
        z=[cam1_pos[2]],
        mode='markers+text',
        name='Camera 1',
        marker=dict(size=15, symbol='star', color='green'),
        text=['Camera 1'],
        textposition='top center'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[cam2_pos[0]],
        y=[cam2_pos[1]],
        z=[cam2_pos[2]],
        mode='markers+text',
        name='Camera 2',
        marker=dict(size=15, symbol='star', color='orange'),
        text=['Camera 2'],
        textposition='top center'
    ))
    
    # Add camera stems (vertical lines from ground to camera)
    for cam_pos, color in [(cam1_pos, 'green'), (cam2_pos, 'orange')]:
        fig.add_trace(go.Scatter3d(
            x=[cam_pos[0], cam_pos[0]],
            y=[cam_pos[1], cam_pos[1]],
            z=[0, cam_pos[2]],
            mode='lines',
            line=dict(color=color, width=3),
            showlegend=False
        ))
    
    # Add camera boresight lines
    elevation_angle = -45  # degrees
    azimuth_angle = 0  # degrees
    boresight_length = 3.0
    
    for cam_pos, color in [(cam1_pos, 'green'), (cam2_pos, 'orange')]:
        # Calculate boresight direction
        elevation_rad = np.radians(elevation_angle)
        azimuth_rad = np.radians(azimuth_angle)
        
        boresight_dir = np.array([
            np.cos(elevation_rad) * np.sin(azimuth_rad),
            np.cos(elevation_rad) * np.cos(azimuth_rad),
            np.sin(elevation_rad)
        ])
        
        # Normalize and calculate end point
        boresight_dir = boresight_dir / np.linalg.norm(boresight_dir)
        boresight_end = cam_pos + boresight_dir * boresight_length
        
        # Add boresight line
        fig.add_trace(go.Scatter3d(
            x=[cam_pos[0], boresight_end[0]],
            y=[cam_pos[1], boresight_end[1]],
            z=[cam_pos[2], boresight_end[2]],
            mode='lines',
            line=dict(color=color, width=2, dash='dash'),
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        title='3D Track Reconstruction',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    logger.info("Interactive visualization created successfully")
    return fig

def main():
    """Main execution function."""
    logger.info("\n=== STARTING 3D TRACKER VISUALIZATION ===")
    
    # First, create a sanity check plot that just shows cameras
    create_camera_only_plot()
    
    # Set up camera projection matrices and positions
    P1, P2, cam1_pos, cam2_pos = set_up_cameras()
    logger.info(f"\nCamera Positions:")
    logger.info(f"Camera 1: {cam1_pos}")
    logger.info(f"Camera 2: {cam2_pos}")
    
    # Generate synthetic 2D tracks
    logger.info("\nGenerating synthetic tracks...")
    sensor1_track, sensor2_track, original_3d = generate_synthetic_tracks()
    
    logger.info("\nCamera 1 2D Track:")
    for i, pt in enumerate(sensor1_track):
        logger.info(f"Frame {i}: ({pt[0]:.2f}, {pt[1]:.2f})")
    
    logger.info("\nCamera 2 2D Track:")
    for i, pt in enumerate(sensor2_track):
        logger.info(f"Frame {i}: ({pt[0]:.2f}, {pt[1]:.2f})")
    
    # Triangulate to get 3D track
    logger.info("\nTriangulating 3D points...")
    reconstructed_3d = triangulate_tracks(sensor1_track, sensor2_track, P1, P2)
    
    logger.info("\nOriginal 3D Track:")
    for i, pt in enumerate(original_3d):
        logger.info(f"Frame {i}: ({pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f})")
    
    logger.info("\nReconstructed 3D Track:")
    for i, pt in enumerate(reconstructed_3d):
        logger.info(f"Frame {i}: ({pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f})")
    
    # Calculate reconstruction error
    errors = np.linalg.norm(np.array(original_3d) - reconstructed_3d, axis=1)
    mean_error = np.mean(errors)
    logger.info(f"\nMean Reconstruction Error: {mean_error:.6f} meters")
    
    # Generate camera images for each frame
    logger.info("\nGenerating camera images...")
    img_dims = generate_camera_images(sensor1_track, sensor2_track)
    logger.info("Camera images saved in 'camera_images/' directory")
    
    # Create standalone 3D visualization
    create_standalone_3d_view(original_3d, reconstructed_3d, cam1_pos, cam2_pos)
    
    # Create frame comparison visualizations
    logger.info("\nCreating frame comparison visualizations...")
    create_frame_comparison(original_3d, reconstructed_3d, sensor1_track, sensor2_track, img_dims)
    logger.info("Frame comparisons saved in 'frame_comparisons/' directory")
    
    # Create interactive visualization
    logger.info("\nCreating interactive visualization...")
    interactive_fig = create_interactive_visualization(original_3d, reconstructed_3d, sensor1_track, sensor2_track, img_dims)
    logger.info("Interactive visualization saved to 'interactive_visualization.html'")
    
    logger.info("\n=== 3D TRACKER VISUALIZATION COMPLETE ===")

if __name__ == "__main__":
    main() 