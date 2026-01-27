#!/usr/bin/env python3
"""
3D Error Visualization for Sensor Impact Analysis
This script creates a comprehensive 3D visualization showing the impact of 
sensor errors on 3D tracking accuracy with deliberately large errors for visibility.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

def setup_stereo_cameras(baseline=0.65, height=2.55, focal_length=800):
    """Set up stereo camera system with specified parameters."""
    # Camera intrinsic parameters
    K = np.array([[focal_length, 0, 320],
                  [0, focal_length, 240],
                  [0, 0, 1]])
    
    # Camera positions
    cam1_pos = np.array([-baseline/2, 0, height])
    cam2_pos = np.array([baseline/2, 0, height])
    
    # Simple projection matrices (no rotation)
    R = np.eye(3)
    t1 = -R @ cam1_pos
    t2 = -R @ cam2_pos
    
    P1 = K @ np.hstack([R, t1.reshape(-1, 1)])
    P2 = K @ np.hstack([R, t2.reshape(-1, 1)])
    
    return P1, P2, (cam1_pos, cam2_pos)

def create_rotation_matrix(roll, pitch, yaw):
    """Create 3D rotation matrix from roll, pitch, yaw angles (in radians)."""
    # Individual rotation matrices
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    
    # Combined rotation matrix (order: Rz * Ry * Rx)
    R = Rz @ Ry @ Rx
    
    return R

def setup_stereo_cameras_with_large_error(baseline=0.65, height=2.55, focal_length=800, 
                                         rotation_errors_deg=(0, 0, 0)):
    """Set up stereo cameras with deliberate large rotation errors."""
    # Camera intrinsic parameters
    K = np.array([[focal_length, 0, 320],
                  [0, focal_length, 240],
                  [0, 0, 1]])
    
    # Convert rotation errors to radians
    roll_rad, pitch_rad, yaw_rad = np.radians(rotation_errors_deg)
    
    # Camera positions
    cam1_pos = np.array([-baseline/2, 0, height])
    cam2_pos = np.array([baseline/2, 0, height])
    
    # Rotation matrices with large errors
    R1 = create_rotation_matrix(roll_rad, pitch_rad, yaw_rad)
    R2 = create_rotation_matrix(roll_rad, pitch_rad, yaw_rad)
    
    t1 = -R1 @ cam1_pos
    t2 = -R2 @ cam2_pos
    
    P1 = K @ np.hstack([R1, t1.reshape(-1, 1)])
    P2 = K @ np.hstack([R2, t2.reshape(-1, 1)])
    
    return P1, P2, (cam1_pos, cam2_pos)

def triangulate_point(P1, P2, point1, point2):
    """Triangulate 3D point from stereo correspondences."""
    # Linear triangulation using DLT
    A = np.array([
        point1[0] * P1[2] - P1[0],
        point1[1] * P1[2] - P1[1],
        point2[0] * P2[2] - P2[0],
        point2[1] * P2[2] - P2[1]
    ])
    
    # Solve using SVD
    _, _, Vt = np.linalg.svd(A)
    point_3d_homogeneous = Vt[-1]
    
    # Convert from homogeneous coordinates
    point_3d = point_3d_homogeneous[:3] / point_3d_homogeneous[3]
    
    return point_3d

def project_3d_to_2d(P, point_3d):
    """Project 3D point to 2D using camera projection matrix."""
    point_3d_homogeneous = np.append(point_3d, 1)
    point_2d_homogeneous = P @ point_3d_homogeneous
    
    # Convert from homogeneous coordinates
    point_2d = point_2d_homogeneous[:2] / point_2d_homogeneous[2]
    
    return point_2d

def create_3d_cylinder(center, base_radius, height, num_points=20):
    """Create 3D cylinder mesh for visualization."""
    # Create cylinder points
    theta = np.linspace(0, 2*np.pi, num_points)
    
    # Base circle
    x_base = center[0] + base_radius * np.cos(theta)
    y_base = center[1] + base_radius * np.sin(theta)
    z_base = np.full_like(x_base, center[2] - height/2)
    
    # Top circle
    x_top = center[0] + base_radius * np.cos(theta)
    y_top = center[1] + base_radius * np.sin(theta)
    z_top = np.full_like(x_top, center[2] + height/2)
    
    return (x_base, y_base, z_base), (x_top, y_top, z_top)

def create_camera_frustum(cam_pos, viewing_direction, fov_degrees=60, depth=0.3):
    """Create camera frustum for visualization."""
    # Convert FOV to radians
    fov_rad = np.radians(fov_degrees)
    
    # Calculate frustum corners
    half_width = depth * np.tan(fov_rad / 2)
    half_height = half_width * 0.75  # Assume 4:3 aspect ratio
    
    # Frustum corners in camera coordinate system
    corners = np.array([
        [0, 0, 0],  # Camera center
        [-half_width, -half_height, depth],  # Bottom left
        [half_width, -half_height, depth],   # Bottom right
        [half_width, half_height, depth],    # Top right
        [-half_width, half_height, depth]    # Top left
    ])
    
    # Transform to world coordinates (simple translation for now)
    world_corners = corners + cam_pos
    
    return world_corners

def create_comprehensive_error_visualization():
    """Create comprehensive 3D visualization showing sensor accuracy errors."""
    
    # Set up the plot
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define true 3D object position (cylinder)
    true_center = np.array([0.5, 1.0, 2.5])
    cylinder_radius = 0.05
    cylinder_height = 0.2
    
    # Perfect stereo camera setup
    P1_perfect, P2_perfect, (cam1_pos, cam2_pos) = setup_stereo_cameras()
    
    # Create scenarios with different large errors
    scenarios = [
        {
            'name': 'Perfect Reconstruction',
            'pixel_noise': 0.0,
            'angle_error': 0.0,
            'color': 'green',
            'alpha': 0.8,
            'marker': 'o'
        },
        {
            'name': 'Large Pixel Noise (5.0 pixels)',
            'pixel_noise': 5.0,
            'angle_error': 0.0,
            'color': 'orange',
            'alpha': 0.7,
            'marker': '^'
        },
        {
            'name': 'Large Angle Error (3.0 degrees)',
            'pixel_noise': 0.5,
            'angle_error': 3.0,
            'color': 'red',
            'alpha': 0.7,
            'marker': 's'
        },
        {
            'name': 'Combined Large Errors',
            'pixel_noise': 3.0,
            'angle_error': 2.0,
            'color': 'purple',
            'alpha': 0.7,
            'marker': 'D'
        }
    ]
    
    # Store results for analysis
    results = []
    
    # Process each scenario
    for i, scenario in enumerate(scenarios):
        # Set up cameras with errors
        if scenario['angle_error'] > 0:
            P1_error, P2_error, _ = setup_stereo_cameras_with_large_error(
                rotation_errors_deg=(scenario['angle_error'], scenario['angle_error'], 0)
            )
        else:
            P1_error, P2_error = P1_perfect, P2_perfect
        
        # Project true point to 2D
        true_2d_cam1 = project_3d_to_2d(P1_error, true_center)
        true_2d_cam2 = project_3d_to_2d(P2_error, true_center)
        
        # Add pixel noise
        if scenario['pixel_noise'] > 0:
            noise_cam1 = np.random.normal(0, scenario['pixel_noise'], 2)
            noise_cam2 = np.random.normal(0, scenario['pixel_noise'], 2)
            noisy_2d_cam1 = true_2d_cam1 + noise_cam1
            noisy_2d_cam2 = true_2d_cam2 + noise_cam2
        else:
            noisy_2d_cam1 = true_2d_cam1
            noisy_2d_cam2 = true_2d_cam2
        
        # Triangulate to get reconstructed 3D point
        try:
            reconstructed_3d = triangulate_point(P1_perfect, P2_perfect, noisy_2d_cam1, noisy_2d_cam2)
            
            # Calculate error
            position_error = np.linalg.norm(reconstructed_3d - true_center)
            error_vector = reconstructed_3d - true_center
            
            # Store results
            results.append({
                'scenario': scenario['name'],
                'true_position': true_center,
                'reconstructed_position': reconstructed_3d,
                'position_error': position_error,
                'error_vector': error_vector,
                'color': scenario['color'],
                'marker': scenario['marker']
            })
            
            # Plot reconstructed position
            ax.scatter(reconstructed_3d[0], reconstructed_3d[1], reconstructed_3d[2], 
                      c=scenario['color'], marker=scenario['marker'], s=200, 
                      alpha=scenario['alpha'], edgecolors='black', linewidth=2,
                      label=f"{scenario['name']}\nError: {position_error:.3f}m")
            
            # Draw error vector
            if position_error > 0.01:  # Only draw if error is significant
                ax.plot([true_center[0], reconstructed_3d[0]], 
                       [true_center[1], reconstructed_3d[1]], 
                       [true_center[2], reconstructed_3d[2]], 
                       color=scenario['color'], linewidth=3, alpha=0.7, linestyle='--')
                
                # Add error magnitude annotation
                mid_point = (true_center + reconstructed_3d) / 2
                ax.text(mid_point[0], mid_point[1], mid_point[2], 
                       f'{position_error:.2f}m', fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=scenario['color'], alpha=0.5))
        
        except Exception as e:
            print(f"Error in scenario {scenario['name']}: {e}")
            continue
    
    # Plot true object position (cylinder)
    base_circle, top_circle = create_3d_cylinder(true_center, cylinder_radius, cylinder_height)
    
    # Plot cylinder base and top
    ax.plot(base_circle[0], base_circle[1], base_circle[2], 'b-', linewidth=3, alpha=0.8)
    ax.plot(top_circle[0], top_circle[1], top_circle[2], 'b-', linewidth=3, alpha=0.8)
    
    # Connect base to top to show cylinder sides
    for i in range(0, len(base_circle[0]), 4):
        ax.plot([base_circle[0][i], top_circle[0][i]], 
               [base_circle[1][i], top_circle[1][i]], 
               [base_circle[2][i], top_circle[2][i]], 'b-', linewidth=2, alpha=0.6)
    
    # Mark true center
    ax.scatter(true_center[0], true_center[1], true_center[2], 
              c='blue', marker='*', s=300, edgecolors='black', linewidth=2,
              label='True Object Position', zorder=10)
    
    # Plot cameras
    ax.scatter(cam1_pos[0], cam1_pos[1], cam1_pos[2], 
              c='black', marker='H', s=200, alpha=0.9, edgecolors='white', linewidth=2,
              label='Camera 1', zorder=5)
    ax.scatter(cam2_pos[0], cam2_pos[1], cam2_pos[2], 
              c='black', marker='H', s=200, alpha=0.9, edgecolors='white', linewidth=2,
              label='Camera 2', zorder=5)
    
    # Draw camera boresight lines
    boresight_length = 1.0
    # Camera 1 boresight (pointing toward object)
    cam1_direction = (true_center - cam1_pos) / np.linalg.norm(true_center - cam1_pos)
    cam1_end = cam1_pos + cam1_direction * boresight_length
    ax.plot([cam1_pos[0], cam1_end[0]], 
           [cam1_pos[1], cam1_end[1]], 
           [cam1_pos[2], cam1_end[2]], 'k--', linewidth=2, alpha=0.5)
    
    # Camera 2 boresight
    cam2_direction = (true_center - cam2_pos) / np.linalg.norm(true_center - cam2_pos)
    cam2_end = cam2_pos + cam2_direction * boresight_length
    ax.plot([cam2_pos[0], cam2_end[0]], 
           [cam2_pos[1], cam2_end[1]], 
           [cam2_pos[2], cam2_end[2]], 'k--', linewidth=2, alpha=0.5)
    
    # Add baseline distance line
    ax.plot([cam1_pos[0], cam2_pos[0]], 
           [cam1_pos[1], cam2_pos[1]], 
           [cam1_pos[2], cam2_pos[2]], 'k-', linewidth=4, alpha=0.7)
    
    # Add baseline distance annotation
    baseline_mid = (cam1_pos + cam2_pos) / 2
    ax.text(baseline_mid[0], baseline_mid[1], baseline_mid[2] - 0.1, 
           f'Baseline: {np.linalg.norm(cam2_pos - cam1_pos):.2f}m', 
           fontsize=12, ha='center', 
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
    
    # Customize the plot
    ax.set_xlabel('X (meters)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y (meters)', fontsize=14, fontweight='bold')
    ax.set_zlabel('Z (meters)', fontsize=14, fontweight='bold')
    
    ax.set_title('3D Sensor Accuracy & Error Visualization\n'
                'Impact of Large Sensor Errors on 3D Reconstruction', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Set axis limits for better visualization
    ax.set_xlim([-0.5, 1.5])
    ax.set_ylim([-0.5, 2.0])
    ax.set_zlim([2.0, 3.0])
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend with custom positioning
    legend1 = ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=10)
    legend1.get_frame().set_alpha(0.9)
    
    # Add statistics box
    stats_text = "ERROR ANALYSIS:\n"
    for result in results:
        if result['position_error'] > 0.01:
            stats_text += f"• {result['scenario']}: {result['position_error']:.3f}m\n"
    
    # Add text box with statistics
    ax.text2D(0.02, 0.02, stats_text, 
             transform=ax.transAxes, fontsize=10, 
             verticalalignment='bottom',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    
    # Add coordinate system indicators
    ax.text(0.5, 0.5, 2.0, 'Ground Level', fontsize=10, alpha=0.7)
    
    plt.tight_layout()
    return fig, ax, results

def create_error_comparison_plot(results):
    """Create a 2D comparison plot of different error sources."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract data
    scenarios = [r['scenario'] for r in results if r['position_error'] > 0.01]
    errors = [r['position_error'] for r in results if r['position_error'] > 0.01]
    colors = [r['color'] for r in results if r['position_error'] > 0.01]
    
    # Bar plot of errors
    bars = ax1.bar(range(len(scenarios)), errors, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Error Scenario', fontweight='bold')
    ax1.set_ylabel('Position Error (meters)', fontweight='bold')
    ax1.set_title('Position Error Comparison', fontweight='bold')
    ax1.set_xticks(range(len(scenarios)))
    ax1.set_xticklabels([s.replace(' ', '\n') for s in scenarios], rotation=0, ha='center')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, error in zip(bars, errors):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{error:.3f}m', ha='center', va='bottom', fontweight='bold')
    
    # Error vector components
    error_components = []
    for result in results:
        if result['position_error'] > 0.01:
            error_components.append({
                'scenario': result['scenario'],
                'x_error': abs(result['error_vector'][0]),
                'y_error': abs(result['error_vector'][1]),
                'z_error': abs(result['error_vector'][2])
            })
    
    # Stacked bar plot of error components
    if error_components:
        scenarios_comp = [ec['scenario'] for ec in error_components]
        x_errors = [ec['x_error'] for ec in error_components]
        y_errors = [ec['y_error'] for ec in error_components]
        z_errors = [ec['z_error'] for ec in error_components]
        
        width = 0.6
        x_pos = np.arange(len(scenarios_comp))
        
        ax2.bar(x_pos, x_errors, width, label='X Error', color='red', alpha=0.7)
        ax2.bar(x_pos, y_errors, width, bottom=x_errors, label='Y Error', color='green', alpha=0.7)
        ax2.bar(x_pos, z_errors, width, bottom=np.array(x_errors) + np.array(y_errors), 
               label='Z Error', color='blue', alpha=0.7)
        
        ax2.set_xlabel('Error Scenario', fontweight='bold')
        ax2.set_ylabel('Error Component (meters)', fontweight='bold')
        ax2.set_title('Error Components (X, Y, Z)', fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([s.replace(' ', '\n') for s in scenarios_comp], rotation=0, ha='center')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    """Main function to generate comprehensive 3D error visualization."""
    print("Generating 3D Error Visualization...")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create comprehensive 3D visualization
    fig_3d, ax_3d, results = create_comprehensive_error_visualization()
    
    # Create error comparison plot
    fig_2d = create_error_comparison_plot(results)
    
    # Print results summary
    print("\n3D RECONSTRUCTION ERROR ANALYSIS:")
    print("=" * 60)
    for result in results:
        print(f"Scenario: {result['scenario']}")
        print(f"  Position Error: {result['position_error']:.4f} meters")
        print(f"  Error Vector: [{result['error_vector'][0]:.4f}, {result['error_vector'][1]:.4f}, {result['error_vector'][2]:.4f}]")
        print(f"  Error Components: X={abs(result['error_vector'][0]):.4f}m, Y={abs(result['error_vector'][1]):.4f}m, Z={abs(result['error_vector'][2]):.4f}m")
        print()
    
    # Save plots
    fig_3d.savefig('3d_error_visualization.png', dpi=300, bbox_inches='tight')
    fig_2d.savefig('error_comparison.png', dpi=300, bbox_inches='tight')
    
    print("✅ Visualizations saved:")
    print("  - 3d_error_visualization.png")
    print("  - error_comparison.png")
    
    # Display plots
    plt.show()
    
    return results

if __name__ == "__main__":
    results = main() 