#!/usr/bin/env python3
"""
Generate five sphere trajectories with constant linear velocity.
Saves trajectory data and creates 3D visualizations.

Updated to use timestamped filenames per requirements.md Section 5.4.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from datetime import datetime
from output_utils_shared import save_figure, save_data

class Sphere:
    """Represents a sphere with position, velocity, and radius."""
    
    def __init__(self, position, velocity, radius=0.05):
        """
        Initialize sphere with position, velocity, and radius.
        
        Args:
            position: Initial position [x, y, z] in meters
            velocity: Constant velocity [vx, vy, vz] in m/s
            radius: Sphere radius in meters (default 0.05m)
        """
        self.initial_position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.radius = radius
        
    def get_position(self, time):
        """Get position at given time with constant velocity."""
        return self.initial_position + self.velocity * time
    
    def generate_trajectory(self, duration=5.0, dt=0.1):
        """
        Generate trajectory points over specified duration.
        
        Args:
            duration: Total time duration in seconds
            dt: Time step in seconds
            
        Returns:
            pandas.DataFrame with columns: time, x, y, z
        """
        times = np.arange(0, duration + dt, dt)
        positions = np.array([self.get_position(t) for t in times])
        
        trajectory_data = pd.DataFrame({
            'time': times,
            'x': positions[:, 0],
            'y': positions[:, 1],
            'z': positions[:, 2]
        })
        
        return trajectory_data

def create_sphere_trajectories():
    """Create five different sphere trajectories with constant linear velocity."""
    
    # Define five different sphere trajectories
    trajectories = [
        {
            'name': 'horizontal_forward',
            'sphere': Sphere(
                position=[0.0, 0.0, 2.5],
                velocity=[0.5, 0.0, 0.0],
                radius=0.05
            ),
            'description': 'Horizontal forward motion'
        },
        {
            'name': 'diagonal_ascending',
            'sphere': Sphere(
                position=[-1.0, -1.0, 2.0],
                velocity=[0.3, 0.4, 0.2],
                radius=0.06
            ),
            'description': 'Diagonal ascending motion'
        },
        {
            'name': 'vertical_drop',
            'sphere': Sphere(
                position=[0.5, 1.0, 3.0],
                velocity=[0.0, 0.0, -0.6],
                radius=0.04
            ),
            'description': 'Vertical dropping motion'
        },
        {
            'name': 'curved_path',
            'sphere': Sphere(
                position=[-0.5, 0.0, 2.2],
                velocity=[0.4, 0.3, 0.1],
                radius=0.07
            ),
            'description': 'Curved trajectory motion'
        },
        {
            'name': 'reverse_motion',
            'sphere': Sphere(
                position=[2.0, 1.5, 2.8],
                velocity=[-0.6, -0.2, -0.1],
                radius=0.05
            ),
            'description': 'Reverse direction motion'
        }
    ]
    
    return trajectories

def save_trajectory_data(trajectory_data, filename, output_dir):
    """Save trajectory data to CSV file with timestamp."""
    saved_path = save_data(trajectory_data, output_dir, f"{filename}.csv", index=False)
    return saved_path

def plot_3d_trajectory(trajectory_data, title, filename, output_dir, sphere_radius=0.05):
    """Create 3D plot of trajectory."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory path
    ax.plot(trajectory_data['x'], trajectory_data['y'], trajectory_data['z'], 
            'b-', linewidth=2, label='Trajectory')
    
    # Mark start and end points
    ax.scatter(trajectory_data['x'].iloc[0], trajectory_data['y'].iloc[0], 
              trajectory_data['z'].iloc[0], color='green', s=100, label='Start')
    ax.scatter(trajectory_data['x'].iloc[-1], trajectory_data['y'].iloc[-1], 
              trajectory_data['z'].iloc[-1], color='red', s=100, label='End')
    
    # Add sphere at multiple positions along trajectory
    n_spheres = 8
    indices = np.linspace(0, len(trajectory_data) - 1, n_spheres, dtype=int)
    
    for i, idx in enumerate(indices):
        alpha = 0.3 - i * 0.02  # Fade spheres along trajectory
        ax.scatter(trajectory_data['x'].iloc[idx], 
                  trajectory_data['y'].iloc[idx], 
                  trajectory_data['z'].iloc[idx], 
                  s=200, alpha=alpha, color='orange')
    
    # Set labels and title
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title(f'3D Sphere Trajectory: {title}')
    ax.legend()
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Save plot with timestamp
    saved_path = save_figure(fig, output_dir, f"{filename}_3d_plot.png", dpi=300)
    
    # Show plot
    plt.show()
    return saved_path
    plt.close()

def generate_summary_report(trajectories, trajectory_data_list, output_dir):
    """Generate summary report of all trajectories."""
    report = []
    
    for i, (traj_info, traj_data) in enumerate(zip(trajectories, trajectory_data_list)):
        sphere = traj_info['sphere']
        
        # Calculate trajectory statistics
        total_distance = np.sum(np.sqrt(np.diff(traj_data['x'])**2 + 
                                       np.diff(traj_data['y'])**2 + 
                                       np.diff(traj_data['z'])**2))
        
        speed = np.linalg.norm(sphere.velocity)
        
        report.append({
            'Trajectory': traj_info['name'],
            'Description': traj_info['description'],
            'Start Position': f"({sphere.initial_position[0]:.2f}, {sphere.initial_position[1]:.2f}, {sphere.initial_position[2]:.2f})",
            'Velocity': f"({sphere.velocity[0]:.2f}, {sphere.velocity[1]:.2f}, {sphere.velocity[2]:.2f})",
            'Speed (m/s)': f"{speed:.3f}",
            'Radius (m)': f"{sphere.radius:.3f}",
            'Total Distance (m)': f"{total_distance:.3f}",
            'Duration (s)': f"{traj_data['time'].iloc[-1]:.1f}"
        })
    
    summary_df = pd.DataFrame(report)
    
    # Save summary report with timestamp
    save_data(summary_df, output_dir, "trajectory_summary.csv", index=False)
    
    return summary_df

def main():
    """Main function to generate sphere trajectories."""
    print("Generating sphere trajectories with constant linear velocity...")
    
    # Create output directory
    output_dir = "output/sphere_trajectories"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate trajectory definitions
    trajectories = create_sphere_trajectories()
    trajectory_data_list = []
    
    # Generate each trajectory
    for i, traj_info in enumerate(trajectories, 1):
        print(f"\n--- Generating Trajectory {i}: {traj_info['name']} ---")
        
        # Generate trajectory data
        trajectory_data = traj_info['sphere'].generate_trajectory(duration=5.0, dt=0.1)
        trajectory_data_list.append(trajectory_data)
        
        # Save trajectory data
        save_trajectory_data(trajectory_data, traj_info['name'], output_dir)
        
        # Create 3D plot
        plot_3d_trajectory(trajectory_data, traj_info['description'], 
                          traj_info['name'], output_dir, traj_info['sphere'].radius)
    
    # Generate summary report
    print("\n--- Generating Summary Report ---")
    summary_df = generate_summary_report(trajectories, trajectory_data_list, output_dir)
    
    print("\n=== TRAJECTORY SUMMARY ===")
    print(summary_df.to_string(index=False))
    
    print(f"\nAll results saved to: {output_dir}")
    print("Generated files:")
    for file in os.listdir(output_dir):
        print(f"  - {file}")

if __name__ == "__main__":
    main() 