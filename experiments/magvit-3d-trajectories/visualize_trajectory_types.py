#!/usr/bin/env python3
"""
Visualize the 4 trajectory types clearly.

This creates a proper visualization showing Linear, Circular, Helical, and Parabolic
trajectories as distinct patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

from magvit_3d_generator import (
    generate_linear_trajectory,
    generate_circular_trajectory,
    generate_helical_trajectory,
    generate_parabolic_trajectory
)


def main():
    """Generate visualization showing 4 distinct trajectory types."""
    
    # Generate one example of each trajectory type
    linear = generate_linear_trajectory(16)
    circular = generate_circular_trajectory(16)
    helical = generate_helical_trajectory(16)
    parabolic = generate_parabolic_trajectory(16)
    
    trajectories = [linear, circular, helical, parabolic]
    names = ['Linear', 'Circular', 'Helical', 'Parabolic']
    colors = ['red', 'green', 'blue', 'orange']
    
    # Create figure with 2x2 subplots
    fig = plt.figure(figsize=(14, 12))
    
    for i, (traj, name, color) in enumerate(zip(trajectories, names, colors)):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        
        # Plot trajectory
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                color=color, linewidth=3, marker='o', markersize=4, alpha=0.8)
        
        # Mark start and end points
        ax.scatter(*traj[0], s=200, c='green', marker='o', edgecolors='black', 
                   linewidths=2, label='Start', zorder=5)
        ax.scatter(*traj[-1], s=200, c='red', marker='s', edgecolors='black', 
                   linewidths=2, label='End', zorder=5)
        
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_zlabel('Z (m)', fontsize=10)
        ax.set_title(f'{name} Trajectory', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.view_init(elev=20, azim=45)
        
        # Set consistent axis limits for comparison
        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([-0.5, 0.5])
        ax.set_zlim([-0.3, 0.3])
    
    plt.suptitle('Four Trajectory Types: Linear, Circular, Helical, Parabolic', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save
    output_dir = Path(__file__).parent / 'results'
    output_file = output_dir / 'trajectory_types_clear.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    print(f"   File size: {output_file.stat().st_size / 1024:.1f} KB")
    
    plt.close()
    
    # Also create a single combined view
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for traj, name, color in zip(trajectories, names, colors):
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                color=color, linewidth=3, marker='o', markersize=4, alpha=0.7,
                label=name)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title('All Four Trajectory Types (Combined View)', 
                 fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=20, azim=45)
    
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.3, 0.3])
    
    output_file = output_dir / 'trajectory_types_combined.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    print(f"   File size: {output_file.stat().st_size / 1024:.1f} KB")
    
    plt.close()
    
    print("")
    print("Trajectory characteristics:")
    print(f"  Linear: Straight line from {linear[0]} to {linear[-1]}")
    print(f"  Circular: Radius = {np.linalg.norm(circular[0, :2]):.3f}m, Z constant")
    print(f"  Helical: Circular in XY, Z changes from {helical[0, 2]:.3f} to {helical[-1, 2]:.3f}")
    print(f"  Parabolic: Y = X², shape visible in Y-Z relationship")


if __name__ == "__main__":
    main()

