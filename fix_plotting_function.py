#!/usr/bin/env python3
"""
Fixed plotting function that handles any number of data points
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_individual_camera_views_fixed(sensor1_track, sensor2_track, original_3d):
    """
    FIXED version: Create individual detailed camera views with proper color handling.
    """
    print("=== CREATING INDIVIDUAL CAMERA VIEWS ===")
    
    # Create a figure with subplots for both cameras
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Generate enough colors for any number of points
    base_colors = ['darkred', 'red', 'orange', 'yellow', 'lightcoral', 'blue', 'green', 'purple', 'brown', 'pink']
    num_points = len(original_3d)
    
    # Extend colors if needed
    colors = []
    for i in range(num_points):
        colors.append(base_colors[i % len(base_colors)])
    
    print(f"Number of points: {num_points}")
    print(f"Number of colors available: {len(colors)}")
    print(f"sensor1_track length: {len(sensor1_track)}")
    print(f"sensor2_track length: {len(sensor2_track)}")
    print(f"original_3d length: {len(original_3d)}")
    
    # Camera 1 view
    ax1.set_title('Camera 1 View (Detailed)', fontsize=16, weight='bold', pad=20)
    ax1.set_xlabel('X Pixel Coordinate', fontsize=14, weight='bold')
    ax1.set_ylabel('Y Pixel Coordinate', fontsize=14, weight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#f8f9fa')
    
    # Plot points for Camera 1
    for i, (pixel, point_3d) in enumerate(zip(sensor1_track, original_3d)):
        # Check if pixel coordinates are valid
        if not (np.isfinite(pixel[0]) and np.isfinite(pixel[1])):
            print(f"Skipping point {i} due to infinite pixel coordinates: {pixel}")
            continue
            
        color = colors[i] if i < len(colors) else 'gray'
        ax1.scatter(pixel[0], pixel[1], s=300, c=color, alpha=0.8, 
                   edgecolor='black', linewidth=3, zorder=5, label=f'Point {i+1}')
        
        # Enhanced annotation with 3D coordinates
        ax1.annotate(f'Point {i+1}\nPixel: ({pixel[0]:.0f}, {pixel[1]:.0f})\n3D: ({point_3d[0]:.1f}, {point_3d[1]:.1f}, {point_3d[2]:.1f})', 
                    (pixel[0], pixel[1]), xytext=(20, 20), textcoords='offset points',
                    fontsize=11, ha='left', va='bottom', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='red'))
    
    # Set reasonable axis limits for Camera 1
    ax1.set_xlim(-100, 1380)
    ax1.set_ylim(0, 720)
    ax1.invert_yaxis()  # Image coordinates (0,0 at top-left)
    ax1.legend(loc='upper right', fontsize=10)
    
    # Camera 2 view
    ax2.set_title('Camera 2 View (Detailed)', fontsize=16, weight='bold', pad=20)
    ax2.set_xlabel('X Pixel Coordinate', fontsize=14, weight='bold')
    ax2.set_ylabel('Y Pixel Coordinate', fontsize=14, weight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#f8f9fa')
    
    # Plot points for Camera 2
    for i, (pixel, point_3d) in enumerate(zip(sensor2_track, original_3d)):
        # Check if pixel coordinates are valid
        if not (np.isfinite(pixel[0]) and np.isfinite(pixel[1])):
            print(f"Skipping point {i} due to infinite pixel coordinates: {pixel}")
            continue
            
        color = colors[i] if i < len(colors) else 'gray'
        ax2.scatter(pixel[0], pixel[1], s=300, c=color, alpha=0.8, 
                   edgecolor='black', linewidth=3, zorder=5, label=f'Point {i+1}')
        
        # Enhanced annotation with 3D coordinates
        ax2.annotate(f'Point {i+1}\nPixel: ({pixel[0]:.0f}, {pixel[1]:.0f})\n3D: ({point_3d[0]:.1f}, {point_3d[1]:.1f}, {point_3d[2]:.1f})', 
                    (pixel[0], pixel[1]), xytext=(20, 20), textcoords='offset points',
                    fontsize=11, ha='left', va='bottom', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='blue'))
    
    # Set reasonable axis limits for Camera 2
    ax2.set_xlim(-100, 1380)
    ax2.set_ylim(0, 720)
    ax2.invert_yaxis()  # Image coordinates (0,0 at top-left)
    ax2.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    print("Individual camera views created successfully!")

# Test function to check data structure lengths
def debug_data_lengths(sensor1_track, sensor2_track, original_3d):
    """Debug function to check data structure lengths."""
    print("=== DEBUGGING DATA LENGTHS ===")
    print(f"sensor1_track: {len(sensor1_track)} items")
    print(f"sensor2_track: {len(sensor2_track)} items")
    print(f"original_3d: {len(original_3d)} items")
    
    print("\nFirst few sensor1_track items:")
    for i, item in enumerate(sensor1_track[:5]):
        print(f"  [{i}]: {item}")
    
    print("\nFirst few sensor2_track items:")
    for i, item in enumerate(sensor2_track[:5]):
        print(f"  [{i}]: {item}")
    
    print("\nFirst few original_3d items:")
    for i, item in enumerate(original_3d[:5]):
        print(f"  [{i}]: {item}")

if __name__ == '__main__':
    print("Fixed plotting function ready to use!")
    print("Copy the plot_individual_camera_views_fixed function to your notebook.") 