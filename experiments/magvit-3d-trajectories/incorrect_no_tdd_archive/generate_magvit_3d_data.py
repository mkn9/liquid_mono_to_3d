#!/usr/bin/env python3
"""
Generate MAGVIT 3D trajectory dataset with 50 samples.

This script generates cube, cylinder, and cone 3D trajectories
with multi-camera rendering.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Camera3D:
    """3D camera for multi-view rendering."""
    
    def __init__(self, position, target, fov=60, width=128, height=128):
        self.position = np.array(position)
        self.target = np.array(target)
        self.fov = fov
        self.width = width
        self.height = height
        
        # Compute view matrix
        self.view_matrix = self.compute_view_matrix()
        
        # Compute projection matrix
        self.projection_matrix = self.compute_projection_matrix()
    
    def compute_view_matrix(self):
        """Compute view matrix for camera."""
        # Camera forward direction
        forward = self.target - self.position
        forward = forward / np.linalg.norm(forward)
        
        # World up vector
        up = np.array([0, 0, 1])
        
        # Camera right vector
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        # Camera up vector
        cam_up = np.cross(right, forward)
        
        # View matrix
        view = np.eye(4)
        view[0, :3] = right
        view[1, :3] = cam_up
        view[2, :3] = -forward
        view[:3, 3] = -np.array([
            np.dot(right, self.position),
            np.dot(cam_up, self.position),
            np.dot(-forward, self.position)
        ])
        
        return view
    
    def compute_projection_matrix(self):
        """Compute projection matrix."""
        aspect = self.width / self.height
        fov_rad = np.radians(self.fov)
        f = 1.0 / np.tan(fov_rad / 2.0)
        near = 0.1
        far = 100.0
        
        proj = np.zeros((4, 4))
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = (2 * far * near) / (near - far)
        proj[3, 2] = -1
        
        return proj
    
    def project_point(self, point_3d):
        """Project 3D point to 2D."""
        point_4d = np.append(point_3d, 1)
        
        # View transform
        view_point = self.view_matrix @ point_4d
        
        # Projection transform  
        proj_point = self.projection_matrix @ view_point
        
        # Perspective divide
        if proj_point[3] != 0:
            proj_point /= proj_point[3]
        
        # To screen coordinates
        x = int((proj_point[0] + 1) * 0.5 * self.width)
        y = int((1 - proj_point[1]) * 0.5 * self.height)
        
        return x, y, proj_point[2]


class MAGVIT3DTrajectoryDataGenerator:
    """Generate MAGVIT 3D trajectory dataset."""
    
    def __init__(self, seq_length=16, img_size=128, num_cameras=3):
        self.seq_length = seq_length
        self.img_size = img_size
        self.num_cameras = num_cameras
        self.shapes_3d = ['cube', 'cylinder', 'cone']
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        
        # Setup cameras
        self.cameras = [
            Camera3D(np.array([0.0, 0.0, 2.55]), np.array([0.0, 0.0, 0.0])),
            Camera3D(np.array([0.65, 0.0, 2.55]), np.array([0.0, 0.0, 0.0])),
            Camera3D(np.array([0.325, 0.56, 2.55]), np.array([0.0, 0.0, 0.0]))
        ]
    
    def generate_linear_trajectory(self):
        """Generate linear 3D trajectory."""
        start = np.array([-0.3, -0.3, 0.0])
        end = np.array([0.3, 0.3, 0.2])
        t = np.linspace(0, 1, self.seq_length)
        trajectory = start[None, :] + t[:, None] * (end - start)[None, :]
        return trajectory
    
    def generate_circular_trajectory(self):
        """Generate circular 3D trajectory."""
        t = np.linspace(0, 2 * np.pi, self.seq_length)
        radius = 0.3
        x = radius * np.cos(t)
        y = radius * np.sin(t)
        z = np.zeros(self.seq_length)
        trajectory = np.stack([x, y, z], axis=1)
        return trajectory
    
    def generate_helical_trajectory(self):
        """Generate helical 3D trajectory."""
        t = np.linspace(0, 2 * np.pi, self.seq_length)
        radius = 0.25
        x = radius * np.cos(t)
        y = radius * np.sin(t)
        z = t / (2 * np.pi) * 0.3 - 0.15
        trajectory = np.stack([x, y, z], axis=1)
        return trajectory
    
    def generate_parabolic_trajectory(self):
        """Generate parabolic 3D trajectory."""
        t = np.linspace(-1, 1, self.seq_length)
        x = t * 0.3
        y = t ** 2 * 0.3 - 0.3
        z = -t ** 2 * 0.1
        trajectory = np.stack([x, y, z], axis=1)
        return trajectory
    
    def render_shape_at_point(self, shape, point_3d, color, camera):
        """Render shape at a 3D point from camera view."""
        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        # Project center point
        x, y, depth = camera.project_point(point_3d)
        
        if 0 <= x < self.img_size and 0 <= y < self.img_size and depth > 0:
            # Draw simple representation
            if shape == 'cube':
                cv2.rectangle(img, (x-15, y-15), (x+15, y+15), color, -1)
            elif shape == 'cylinder':
                cv2.circle(img, (x, y), 15, color, -1)
            elif shape == 'cone':
                pts = np.array([[x, y-15], [x-15, y+15], [x+15, y+15]], dtype=np.int32)
                cv2.fillPoly(img, [pts], color)
        
        return img
    
    def render_multi_view_sequence(self, shape, trajectory, color):
        """Render multi-view video sequence."""
        multi_view_video = np.zeros((self.num_cameras, self.seq_length, 
                                     self.img_size, self.img_size, 3), dtype=np.uint8)
        
        for cam_idx, camera in enumerate(self.cameras):
            for frame_idx in range(self.seq_length):
                point_3d = trajectory[frame_idx]
                frame = self.render_shape_at_point(shape, point_3d, color, camera)
                multi_view_video[cam_idx, frame_idx] = frame
        
        return multi_view_video
    
    def generate_dataset(self, num_samples):
        """Generate complete dataset."""
        logger.info(f"Generating {num_samples} 3D trajectory samples...")
        
        trajectories_3d = []
        multi_view_videos = []
        labels = []
        
        trajectory_funcs = [
            self.generate_linear_trajectory,
            self.generate_circular_trajectory,
            self.generate_helical_trajectory,
            self.generate_parabolic_trajectory
        ]
        
        for i in range(num_samples):
            # Choose shape
            shape_idx = i % 3
            shape = self.shapes_3d[shape_idx]
            color = self.colors[shape_idx]
            
            # Choose trajectory pattern
            traj_func = trajectory_funcs[i % len(trajectory_funcs)]
            trajectory = traj_func()
            
            # Add noise
            noise = np.random.normal(0, 0.02, trajectory.shape)
            noisy_trajectory = trajectory + noise
            
            # Render multi-view video
            multi_view_video = self.render_multi_view_sequence(shape, noisy_trajectory, color)
            
            trajectories_3d.append(noisy_trajectory)
            multi_view_videos.append(multi_view_video)
            labels.append(shape_idx)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Generated {i + 1}/{num_samples} samples")
        
        return {
            'trajectories_3d': np.array(trajectories_3d),
            'multi_view_videos': np.array(multi_view_videos),
            'labels': np.array(labels)
        }


def create_visualizations(dataset, output_dir):
    """Create visualization plots."""
    logger.info("Creating visualizations...")
    
    trajectories_3d = dataset['trajectories_3d']
    labels = dataset['labels']
    shape_names = ['Cube', 'Cylinder', 'Cone']
    colors = ['red', 'green', 'blue']
    
    # 3D Trajectory Plot
    fig = plt.figure(figsize=(15, 5))
    
    for i, (shape_name, color) in enumerate(zip(shape_names, colors)):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        
        # Plot trajectories for this shape
        shape_trajs = trajectories_3d[labels == i]
        for traj in shape_trajs[:5]:  # Plot first 5
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, alpha=0.6)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{shape_name} Trajectories')
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'magvit_3d_trajectories.png', dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {output_dir / 'magvit_3d_trajectories.png'}")
    plt.close()
    
    # 2D Error Analysis
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (shape_name, color, ax) in enumerate(zip(shape_names, colors, axes)):
        shape_trajs = trajectories_3d[labels == i]
        
        # Calculate trajectory lengths
        lengths = []
        for traj in shape_trajs:
            diffs = np.diff(traj, axis=0)
            lengths.append(np.sum(np.linalg.norm(diffs, axis=1)))
        
        ax.hist(lengths, bins=20, color=color, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Trajectory Length')
        ax.set_ylabel('Count')
        ax.set_title(f'{shape_name} Trajectory Lengths')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'magvit_3d_errors_2d.png', dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {output_dir / 'magvit_3d_errors_2d.png'}")
    plt.close()
    
    # Camera Position Visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Camera positions
    cam_positions = [
        [0.0, 0.0, 2.55],
        [0.65, 0.0, 2.55],
        [0.325, 0.56, 2.55]
    ]
    
    for i, pos in enumerate(cam_positions):
        ax.scatter(*pos, s=200, c=f'C{i}', marker='^', label=f'Camera {i+1}')
        # Draw viewing direction
        target = [0, 0, 0]
        ax.plot([pos[0], target[0]], [pos[1], target[1]], 
                [pos[2], target[2]], f'C{i}--', alpha=0.5)
    
    # Plot origin
    ax.scatter(0, 0, 0, s=100, c='black', marker='o', label='Target (origin)')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Camera Positions and Viewing Directions')
    ax.legend()
    ax.grid(True)
    
    plt.savefig(output_dir / 'magvit_3d_cameras.png', dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {output_dir / 'magvit_3d_cameras.png'}")
    plt.close()


def main():
    """Generate MAGVIT 3D dataset with 50 samples."""
    logger.info("Starting MAGVIT 3D dataset generation")
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Output directory
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    # Generate dataset
    generator = MAGVIT3DTrajectoryDataGenerator(seq_length=16, img_size=128, num_cameras=3)
    dataset = generator.generate_dataset(num_samples=50)
    
    # Save dataset
    output_file = output_dir / 'magvit_3d_dataset.npz'
    np.savez_compressed(output_file, **dataset)
    logger.info(f"Dataset saved to: {output_file}")
    logger.info(f"  trajectories_3d shape: {dataset['trajectories_3d'].shape}")
    logger.info(f"  multi_view_videos shape: {dataset['multi_view_videos'].shape}")
    logger.info(f"  labels shape: {dataset['labels'].shape}")
    
    # Create visualizations
    create_visualizations(dataset, output_dir)
    
    logger.info("MAGVIT 3D dataset generation complete!")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Files created:")
    for f in sorted(output_dir.glob('*')):
        logger.info(f"  - {f.name}")


if __name__ == "__main__":
    main()

