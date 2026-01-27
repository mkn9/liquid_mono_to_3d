#!/usr/bin/env python3
"""
D-NeRF Data Augmentation Pipeline
Converts sphere trajectory data into D-NeRF compatible format with multi-view camera images.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import imageio
from pathlib import Path

class DNerfDataGenerator:
    """Generate D-NeRF compatible data from sphere trajectories."""
    
    def __init__(self, output_dir="data/sphere_trajectories"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Camera intrinsic parameters
        self.image_width = 800
        self.image_height = 600
        self.focal_length = 800
        self.cx = self.image_width / 2
        self.cy = self.image_height / 2
        
        # Camera setup - multiple viewpoints around the scene
        self.setup_camera_array()
        
    def setup_camera_array(self):
        """Setup multiple camera viewpoints around the scene."""
        # Create 8 cameras in a circle around the scene center
        n_cameras = 8
        radius = 3.0  # Distance from scene center
        height = 2.5  # Camera height
        
        self.camera_positions = []
        self.camera_orientations = []
        
        for i in range(n_cameras):
            angle = 2 * np.pi * i / n_cameras
            
            # Camera position
            pos = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                height
            ])
            
            # Camera looks toward scene center
            look_at = np.array([0, 0, 2.5])  # Scene center
            up = np.array([0, 0, 1])
            
            # Calculate camera orientation
            forward = look_at - pos
            forward = forward / np.linalg.norm(forward)
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)
            
            # Rotation matrix (camera to world)
            R_c2w = np.column_stack([right, up, -forward])
            
            self.camera_positions.append(pos)
            self.camera_orientations.append(R_c2w)
    
    def get_camera_intrinsic_matrix(self):
        """Get camera intrinsic matrix."""
        K = np.array([
            [self.focal_length, 0, self.cx],
            [0, self.focal_length, self.cy],
            [0, 0, 1]
        ])
        return K
    
    def get_camera_extrinsic_matrix(self, cam_idx):
        """Get camera extrinsic matrix (world to camera)."""
        R_c2w = self.camera_orientations[cam_idx]
        t_c2w = self.camera_positions[cam_idx]
        
        # Convert to world-to-camera transformation
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ t_c2w
        
        # Create 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = R_w2c
        T[:3, 3] = t_w2c
        
        return T
    
    def project_sphere_to_image(self, sphere_center, sphere_radius, cam_idx):
        """Project 3D sphere to 2D image coordinates."""
        K = self.get_camera_intrinsic_matrix()
        T = self.get_camera_extrinsic_matrix(cam_idx)
        
        # Transform sphere center to camera coordinates
        center_homo = np.append(sphere_center, 1)
        center_cam = T @ center_homo
        
        # Check if sphere is in front of camera
        if center_cam[2] <= 0:
            return None, None
        
        # Project to image coordinates
        center_2d = K @ center_cam[:3]
        center_2d = center_2d[:2] / center_2d[2]
        
        # Calculate projected radius (approximate)
        # This is a simplified projection - more accurate methods exist
        projected_radius = (sphere_radius * self.focal_length) / center_cam[2]
        
        return center_2d, projected_radius
    
    def render_sphere_scene(self, trajectory_data, time_idx, cam_idx):
        """Render a scene with spheres at specific time and camera viewpoint."""
        # Create blank image
        img = np.ones((self.image_height, self.image_width, 3), dtype=np.uint8) * 255
        
        # Load all trajectory data
        trajectory_files = [
            "horizontal_forward.csv",
            "diagonal_ascending.csv", 
            "vertical_drop.csv",
            "curved_path.csv",
            "reverse_motion.csv"
        ]
        
        sphere_radii = [0.05, 0.06, 0.04, 0.07, 0.05]  # From our original data
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
        ]
        
        for i, (traj_file, radius, color) in enumerate(zip(trajectory_files, sphere_radii, colors)):
            try:
                # Load trajectory data
                traj_path = Path("../output/sphere_trajectories") / traj_file
                if not traj_path.exists():
                    continue
                    
                traj_df = pd.read_csv(traj_path)
                
                # Get sphere position at current time
                if time_idx >= len(traj_df):
                    continue
                    
                sphere_pos = np.array([
                    traj_df.iloc[time_idx]['x'],
                    traj_df.iloc[time_idx]['y'],
                    traj_df.iloc[time_idx]['z']
                ])
                
                # Project to image
                center_2d, proj_radius = self.project_sphere_to_image(
                    sphere_pos, radius, cam_idx
                )
                
                if center_2d is not None:
                    # Check if sphere is visible in image
                    if (0 <= center_2d[0] < self.image_width and 
                        0 <= center_2d[1] < self.image_height and
                        proj_radius > 1):
                        
                        # Draw sphere as filled circle
                        cv2.circle(img, 
                                 (int(center_2d[0]), int(center_2d[1])), 
                                 int(proj_radius), 
                                 color, 
                                 -1)
                        
                        # Add subtle shading
                        cv2.circle(img, 
                                 (int(center_2d[0] - proj_radius/3), 
                                  int(center_2d[1] - proj_radius/3)), 
                                 int(proj_radius/3), 
                                 tuple(int(c * 1.3) for c in color), 
                                 -1)
                                 
            except Exception as e:
                print(f"Error processing trajectory {traj_file}: {e}")
                continue
        
        return img
    
    def generate_dnerf_dataset(self):
        """Generate complete D-NeRF dataset."""
        print("Generating D-NeRF compatible dataset...")
        
        # Create directory structure
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "poses").mkdir(exist_ok=True)
        
        # Load one trajectory to get time information
        sample_traj = pd.read_csv("../output/sphere_trajectories/horizontal_forward.csv")
        n_frames = len(sample_traj)
        n_cameras = len(self.camera_positions)
        
        print(f"Generating {n_frames} frames × {n_cameras} cameras = {n_frames * n_cameras} images")
        
        # Generate images for each time step and camera
        all_images = []
        all_poses = []
        
        for time_idx in range(n_frames):
            print(f"Processing time step {time_idx + 1}/{n_frames}")
            
            for cam_idx in range(n_cameras):
                # Render scene
                img = self.render_sphere_scene(None, time_idx, cam_idx)
                
                # Save image
                img_filename = f"frame_{time_idx:03d}_cam_{cam_idx:02d}.png"
                img_path = self.output_dir / "images" / img_filename
                cv2.imwrite(str(img_path), img)
                
                # Store image info
                all_images.append({
                    "file_name": img_filename,
                    "time": float(sample_traj.iloc[time_idx]['time']),
                    "camera_id": cam_idx
                })
                
                # Store pose info (4x4 transformation matrix)
                pose_matrix = np.eye(4)
                pose_matrix[:3, :3] = self.camera_orientations[cam_idx]
                pose_matrix[:3, 3] = self.camera_positions[cam_idx]
                
                all_poses.append({
                    "file_name": img_filename,
                    "transform_matrix": pose_matrix.tolist(),
                    "camera_id": cam_idx,
                    "time": float(sample_traj.iloc[time_idx]['time'])
                })
        
        # Save D-NeRF compatible transforms.json
        transforms_data = {
            "camera_angle_x": 2 * np.arctan(self.image_width / (2 * self.focal_length)),
            "camera_angle_y": 2 * np.arctan(self.image_height / (2 * self.focal_length)),
            "fl_x": self.focal_length,
            "fl_y": self.focal_length,
            "cx": self.cx,
            "cy": self.cy,
            "w": self.image_width,
            "h": self.image_height,
            "frames": all_poses
        }
        
        with open(self.output_dir / "transforms.json", 'w') as f:
            json.dump(transforms_data, f, indent=2)
        
        print(f"Dataset generated successfully in {self.output_dir}")
        return all_images, all_poses
    
    def create_dnerf_config(self):
        """Create D-NeRF configuration file."""
        config_content = f"""
# D-NeRF Configuration for Sphere Trajectories
expname = sphere_trajectories
basedir = ./logs
datadir = {self.output_dir}
dataset_type = blender

# Network parameters
netdepth = 8
netwidth = 256
netdepth_fine = 8
netwidth_fine = 256

# Training parameters
N_rand = 1024
N_samples = 64
N_importance = 128
N_iters = 200000

# Render parameters
chunk = 1024*32
netchunk = 1024*64

# Learning rates
lrate = 5e-4
lrate_decay = 250

# Batch size
batch_size = 4096

# Test parameters
render_only = False
render_test = False
render_factor = 0

# Time parameters (for dynamic scenes)
use_time = True
time_conditioned = True
"""
        
        config_path = self.output_dir / "config.txt"
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        print(f"D-NeRF config saved to {config_path}")
        return config_path
    
    def analyze_temporal_prediction(self):
        """Analyze how D-NeRF can predict next frames."""
        print("\n=== D-NeRF Temporal Prediction Analysis ===")
        
        # Load trajectory data to analyze motion patterns
        traj_files = [
            "horizontal_forward.csv",
            "diagonal_ascending.csv", 
            "vertical_drop.csv",
            "curved_path.csv",
            "reverse_motion.csv"
        ]
        
        motion_analysis = []
        
        for traj_file in traj_files:
            traj_path = Path("../output/sphere_trajectories") / traj_file
            if not traj_path.exists():
                continue
                
            df = pd.read_csv(traj_path)
            
            # Calculate velocities
            velocities = np.array([
                np.diff(df['x']),
                np.diff(df['y']),
                np.diff(df['z'])
            ]).T
            
            # Calculate accelerations
            accelerations = np.array([
                np.diff(velocities[:, 0]),
                np.diff(velocities[:, 1]),
                np.diff(velocities[:, 2])
            ]).T
            
            motion_analysis.append({
                'trajectory': traj_file,
                'avg_velocity': np.mean(np.linalg.norm(velocities, axis=1)),
                'velocity_std': np.std(np.linalg.norm(velocities, axis=1)),
                'avg_acceleration': np.mean(np.linalg.norm(accelerations, axis=1)),
                'motion_type': 'constant_velocity' if np.std(np.linalg.norm(velocities, axis=1)) < 0.01 else 'variable_velocity'
            })
        
        return motion_analysis

def main():
    """Main function to generate D-NeRF compatible dataset."""
    generator = DNerfDataGenerator()
    
    # Generate dataset
    images, poses = generator.generate_dnerf_dataset()
    
    # Create config file
    config_path = generator.create_dnerf_config()
    
    # Analyze temporal prediction capabilities
    motion_analysis = generator.analyze_temporal_prediction()
    
    print("\n=== MOTION ANALYSIS FOR D-NeRF PREDICTION ===")
    for analysis in motion_analysis:
        print(f"Trajectory: {analysis['trajectory']}")
        print(f"  Average Velocity: {analysis['avg_velocity']:.3f} m/s")
        print(f"  Velocity Std: {analysis['velocity_std']:.3f}")
        print(f"  Average Acceleration: {analysis['avg_acceleration']:.3f} m/s²")
        print(f"  Motion Type: {analysis['motion_type']}")
        print()
    
    print("\n=== D-NeRF DATASET SUMMARY ===")
    print(f"Total images generated: {len(images)}")
    print(f"Camera viewpoints: {len(set(img['camera_id'] for img in images))}")
    print(f"Time steps: {len(set(img['time'] for img in images))}")
    print(f"Dataset location: {generator.output_dir}")
    
    print("\n=== NEXT STEPS FOR D-NeRF PREDICTION ===")
    print("1. Train D-NeRF model on generated dataset")
    print("2. Use temporal interpolation to predict future frames")
    print("3. Evaluate prediction accuracy against ground truth")
    print("4. Fine-tune model parameters for better temporal consistency")
    
    return generator

if __name__ == "__main__":
    generator = main() 