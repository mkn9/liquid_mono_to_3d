#!/usr/bin/env python3
"""
MAGVIT 3D Trajectory Experiment Setup
=====================================

This script sets up the MAGVIT framework for learning and predicting 3D trajectories
of geometric shapes (cubes, cylinders, cones) using the existing mono_to_3d codebase.

Based on: https://github.com/google-research/magvit
Paper: https://arxiv.org/abs/2204.02896
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from pathlib import Path
import json
import subprocess
import sys
from typing import List, Tuple, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MAGVIT3DTrajectorySetup:
    """Setup class for MAGVIT 3D trajectory experiments."""
    
    def __init__(self, experiment_dir: str = "experiments/magvit-3d-trajectories"):
        self.experiment_dir = Path(experiment_dir)
        self.data_dir = self.experiment_dir / "data"
        self.models_dir = self.experiment_dir / "models"
        self.results_dir = self.experiment_dir / "results"
        self.magvit_dir = self.experiment_dir / "magvit"
        
        # Create directories
        for dir_path in [self.data_dir, self.models_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def clone_magvit_repository(self):
        """Clone the MAGVIT repository."""
        logger.info("Cloning MAGVIT repository...")
        if not self.magvit_dir.exists():
            try:
                subprocess.run([
                    "git", "clone", 
                    "https://github.com/google-research/magvit.git",
                    str(self.magvit_dir)
                ], check=True)
                logger.info("Successfully cloned MAGVIT repository")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to clone MAGVIT repository: {e}")
                raise
        else:
            logger.info("MAGVIT repository already exists")
    
    def setup_environment(self):
        """Set up the Python environment for MAGVIT 3D."""
        logger.info("Setting up MAGVIT 3D environment...")
        
        # Install JAX and other dependencies
        requirements = [
            "jax>=0.4.0",
            "jaxlib>=0.4.0",
            "flax>=0.7.0",
            "optax>=0.1.4",
            "orbax-checkpoint>=0.2.0",
            "tensorflow>=2.12.0",
            "tensorflow-datasets>=4.9.0",
            "matplotlib>=3.5.0",
            "opencv-python>=4.5.0",
            "Pillow>=9.0.0",
            "tqdm>=4.64.0",
            "wandb>=0.13.0",
            "open3d>=0.16.0",
            "trimesh>=3.15.0",
            "scipy>=1.9.0"
        ]
        
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install"
            ] + requirements, check=True)
            logger.info("Successfully installed MAGVIT 3D dependencies")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            raise

class Cube3D:
    """3D Cube object for trajectory generation."""
    
    def __init__(self, center: List[float], size: float = 0.1):
        self.center = np.array(center)
        self.size = size
        self.vertices = self.generate_vertices()
    
    def generate_vertices(self) -> np.ndarray:
        """Generate cube vertices."""
        half_size = self.size / 2
        vertices = np.array([
            [-half_size, -half_size, -half_size],
            [half_size, -half_size, -half_size],
            [half_size, half_size, -half_size],
            [-half_size, half_size, -half_size],
            [-half_size, -half_size, half_size],
            [half_size, -half_size, half_size],
            [half_size, half_size, half_size],
            [-half_size, half_size, half_size]
        ]) + self.center
        return vertices
    
    def get_edges(self) -> List[Tuple[int, int]]:
        """Get cube edge connections."""
        return [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
        ]

class Cylinder3D:
    """3D Cylinder object for trajectory generation."""
    
    def __init__(self, center: List[float], radius: float = 0.05, height: float = 0.2):
        self.center = np.array(center)
        self.radius = radius
        self.height = height
        self.base_center = self.center - np.array([0, 0, height/2])
        self.top_center = self.center + np.array([0, 0, height/2])
    
    def generate_circle_points(self, center: np.ndarray, n_points: int = 16) -> np.ndarray:
        """Generate points on a circle."""
        angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        points = np.zeros((n_points, 3))
        points[:, 0] = center[0] + self.radius * np.cos(angles)
        points[:, 1] = center[1] + self.radius * np.sin(angles)
        points[:, 2] = center[2]
        return points
    
    def get_surface_points(self) -> np.ndarray:
        """Get cylinder surface points."""
        base_points = self.generate_circle_points(self.base_center)
        top_points = self.generate_circle_points(self.top_center)
        return np.vstack([base_points, top_points])

class Cone3D:
    """3D Cone object for trajectory generation."""
    
    def __init__(self, center: List[float], base_radius: float = 0.05, height: float = 0.2):
        self.center = np.array(center)
        self.base_radius = base_radius
        self.height = height
        self.base_center = self.center - np.array([0, 0, height/2])
        self.apex = self.center + np.array([0, 0, height/2])
    
    def generate_base_points(self, n_points: int = 16) -> np.ndarray:
        """Generate points on the base circle."""
        angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        points = np.zeros((n_points, 3))
        points[:, 0] = self.base_center[0] + self.base_radius * np.cos(angles)
        points[:, 1] = self.base_center[1] + self.base_radius * np.sin(angles)
        points[:, 2] = self.base_center[2]
        return points
    
    def get_endpoints(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get cone base center and apex."""
        return self.base_center, self.apex

class Camera3D:
    """3D Camera system for rendering object trajectories."""
    
    def __init__(self, position: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0, 0, 1])):
        self.position = position
        self.target = target
        self.up = up
        self.view_matrix = self.look_at_matrix()
        self.projection_matrix = self.perspective_matrix()
    
    def look_at_matrix(self) -> np.ndarray:
        """Compute view matrix."""
        forward = self.target - self.position
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, self.up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        view = np.eye(4)
        view[:3, 0] = right
        view[:3, 1] = up
        view[:3, 2] = -forward
        view[:3, 3] = self.position
        
        return np.linalg.inv(view)
    
    def perspective_matrix(self, fov: float = 45, aspect: float = 1.0, near: float = 0.1, far: float = 100.0) -> np.ndarray:
        """Compute perspective projection matrix."""
        f = 1.0 / np.tan(np.radians(fov) / 2.0)
        proj = np.zeros((4, 4))
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = (2 * far * near) / (near - far)
        proj[3, 2] = -1
        return proj
    
    def project_point(self, point_3d: np.ndarray, image_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
        """Project 3D point to 2D image coordinates."""
        # Transform to camera space
        point_homo = np.append(point_3d, 1.0)
        view_point = self.view_matrix @ point_homo
        
        # Project to screen
        proj_point = self.projection_matrix @ view_point
        
        if proj_point[3] != 0:
            proj_point /= proj_point[3]
        
        # Convert to image coordinates
        x = int((proj_point[0] + 1) * image_size[0] / 2)
        y = int((1 - proj_point[1]) * image_size[1] / 2)
        
        return np.array([x, y])

class Trajectory3DDataGenerator:
    """Generate 3D trajectory data for geometric shapes."""
    
    def __init__(self, image_size: Tuple[int, int] = (128, 128), seq_length: int = 16):
        self.image_size = image_size
        self.seq_length = seq_length
        self.shapes = ['cube', 'cylinder', 'cone']
        
        # Setup multiple cameras for 3D observation
        self.cameras = [
            Camera3D(np.array([2, 0, 1]), np.array([0, 0, 0])),
            Camera3D(np.array([0, 2, 1]), np.array([0, 0, 0])),
            Camera3D(np.array([-2, 0, 1]), np.array([0, 0, 0]))
        ]
    
    def generate_3d_trajectory_patterns(self) -> List[np.ndarray]:
        """Generate different 3D trajectory patterns."""
        patterns = []
        
        # Linear 3D trajectory
        start_pos = np.array([-0.5, -0.5, 0.0])
        end_pos = np.array([0.5, 0.5, 0.2])
        linear_path = np.array([
            start_pos + (end_pos - start_pos) * i / (self.seq_length - 1)
            for i in range(self.seq_length)
        ])
        patterns.append(linear_path)
        
        # Circular 3D trajectory (horizontal circle)
        center = np.array([0.0, 0.0, 0.1])
        radius = 0.3
        circular_path = np.array([
            [center[0] + radius * np.cos(2 * np.pi * i / self.seq_length),
             center[1] + radius * np.sin(2 * np.pi * i / self.seq_length),
             center[2]]
            for i in range(self.seq_length)
        ])
        patterns.append(circular_path)
        
        # Helical 3D trajectory
        center = np.array([0.0, 0.0, 0.0])
        radius = 0.2
        height_change = 0.3
        helical_path = np.array([
            [center[0] + radius * np.cos(4 * np.pi * i / self.seq_length),
             center[1] + radius * np.sin(4 * np.pi * i / self.seq_length),
             center[2] + height_change * i / (self.seq_length - 1)]
            for i in range(self.seq_length)
        ])
        patterns.append(helical_path)
        
        # Parabolic 3D trajectory
        parabolic_path = np.array([
            [0.5 * (i / (self.seq_length - 1)) - 0.25,
             0.3 * (i / (self.seq_length - 1)) - 0.15,
             0.2 * ((i / (self.seq_length - 1)) - 0.5) ** 2]
            for i in range(self.seq_length)
        ])
        patterns.append(parabolic_path)
        
        return patterns
    
    def render_cube_trajectory(self, trajectory_3d: np.ndarray) -> Dict[str, np.ndarray]:
        """Render cube moving along 3D trajectory from multiple camera views."""
        multi_view_frames = {f'camera_{i}': [] for i in range(len(self.cameras))}
        
        for pos_3d in trajectory_3d:
            cube = Cube3D(pos_3d)
            
            for cam_idx, camera in enumerate(self.cameras):
                frame = np.zeros((*self.image_size, 3), dtype=np.uint8)
                
                # Project cube vertices
                projected_vertices = []
                for vertex in cube.vertices:
                    proj_pt = camera.project_point(vertex, self.image_size)
                    projected_vertices.append(proj_pt)
                
                # Draw cube edges
                edges = cube.get_edges()
                for edge in edges:
                    pt1, pt2 = projected_vertices[edge[0]], projected_vertices[edge[1]]
                    if (0 <= pt1[0] < self.image_size[0] and 0 <= pt1[1] < self.image_size[1] and
                        0 <= pt2[0] < self.image_size[0] and 0 <= pt2[1] < self.image_size[1]):
                        cv2.line(frame, tuple(pt1.astype(int)), tuple(pt2.astype(int)), (255, 100, 100), 2)
                
                multi_view_frames[f'camera_{cam_idx}'].append(frame)
        
        return {key: np.array(frames) for key, frames in multi_view_frames.items()}
    
    def render_cylinder_trajectory(self, trajectory_3d: np.ndarray) -> Dict[str, np.ndarray]:
        """Render cylinder moving along 3D trajectory from multiple camera views."""
        multi_view_frames = {f'camera_{i}': [] for i in range(len(self.cameras))}
        
        for pos_3d in trajectory_3d:
            cylinder = Cylinder3D(pos_3d)
            
            for cam_idx, camera in enumerate(self.cameras):
                frame = np.zeros((*self.image_size, 3), dtype=np.uint8)
                
                # Project cylinder surface points
                surface_points = cylinder.get_surface_points()
                projected_points = []
                for point in surface_points:
                    proj_pt = camera.project_point(point, self.image_size)
                    projected_points.append(proj_pt)
                
                # Draw cylinder outline
                for i, pt in enumerate(projected_points):
                    if 0 <= pt[0] < self.image_size[0] and 0 <= pt[1] < self.image_size[1]:
                        cv2.circle(frame, tuple(pt.astype(int)), 2, (100, 255, 100), -1)
                
                multi_view_frames[f'camera_{cam_idx}'].append(frame)
        
        return {key: np.array(frames) for key, frames in multi_view_frames.items()}
    
    def render_cone_trajectory(self, trajectory_3d: np.ndarray) -> Dict[str, np.ndarray]:
        """Render cone moving along 3D trajectory from multiple camera views."""
        multi_view_frames = {f'camera_{i}': [] for i in range(len(self.cameras))}
        
        for pos_3d in trajectory_3d:
            cone = Cone3D(pos_3d)
            
            for cam_idx, camera in enumerate(self.cameras):
                frame = np.zeros((*self.image_size, 3), dtype=np.uint8)
                
                # Project cone points
                base_points = cone.generate_base_points()
                projected_base = []
                for point in base_points:
                    proj_pt = camera.project_point(point, self.image_size)
                    projected_base.append(proj_pt)
                
                apex_proj = camera.project_point(cone.apex, self.image_size)
                
                # Draw cone outline
                for pt in projected_base:
                    if (0 <= pt[0] < self.image_size[0] and 0 <= pt[1] < self.image_size[1] and
                        0 <= apex_proj[0] < self.image_size[0] and 0 <= apex_proj[1] < self.image_size[1]):
                        cv2.line(frame, tuple(pt.astype(int)), tuple(apex_proj.astype(int)), (100, 100, 255), 1)
                
                # Draw base circle
                for i, pt in enumerate(projected_base):
                    next_pt = projected_base[(i + 1) % len(projected_base)]
                    if (0 <= pt[0] < self.image_size[0] and 0 <= pt[1] < self.image_size[1] and
                        0 <= next_pt[0] < self.image_size[0] and 0 <= next_pt[1] < self.image_size[1]):
                        cv2.line(frame, tuple(pt.astype(int)), tuple(next_pt.astype(int)), (100, 100, 255), 1)
                
                multi_view_frames[f'camera_{cam_idx}'].append(frame)
        
        return {key: np.array(frames) for key, frames in multi_view_frames.items()}
    
    def generate_dataset(self, num_samples: int = 1000) -> Dict[str, Any]:
        """Generate complete 3D trajectory dataset."""
        dataset = {
            'multi_view_videos': [],
            'trajectories_3d': [],
            'shape_labels': []
        }
        
        trajectory_patterns = self.generate_3d_trajectory_patterns()
        
        for sample_idx in range(num_samples):
            # Random shape and trajectory
            shape = np.random.choice(self.shapes)
            trajectory = np.random.choice(trajectory_patterns)
            
            # Add noise to trajectory
            noise = np.random.normal(0, 0.02, trajectory.shape)
            noisy_trajectory = trajectory + noise
            
            # Render multi-view video sequence
            if shape == 'cube':
                multi_view_video = self.render_cube_trajectory(noisy_trajectory)
            elif shape == 'cylinder':
                multi_view_video = self.render_cylinder_trajectory(noisy_trajectory)
            elif shape == 'cone':
                multi_view_video = self.render_cone_trajectory(noisy_trajectory)
            
            dataset['multi_view_videos'].append(multi_view_video)
            dataset['trajectories_3d'].append(noisy_trajectory)
            dataset['shape_labels'].append(self.shapes.index(shape))
            
            if sample_idx % 100 == 0:
                logger.info(f"Generated {sample_idx}/{num_samples} 3D samples")
        
        return dataset

class MAGVIT3DTraining:
    """Training configuration for MAGVIT 3D trajectory prediction."""
    
    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.config = self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for MAGVIT 3D training."""
        return {
            'model': {
                'name': 'magvit_3d_trajectories',
                'vocab_size': 2048,
                'hidden_dim': 768,
                'num_layers': 12,
                'num_heads': 12,
                'sequence_length': 16,
                'image_size': 128,
                'patch_size': 8,
                'num_cameras': 3,
            },
            'training': {
                'batch_size': 16,
                'learning_rate': 5e-5,
                'num_epochs': 150,
                'warmup_steps': 2000,
                'weight_decay': 0.02,
                'gradient_clip_norm': 1.0,
            },
            'data': {
                'train_samples': 6000,
                'val_samples': 1000,
                'test_samples': 1000,
                'num_workers': 6,
            },
            'logging': {
                'log_interval': 50,
                'eval_interval': 500,
                'save_interval': 1000,
                'use_wandb': True,
            },
            '3d_specific': {
                'coordinate_system': 'right_handed',
                'world_bounds': [-1.0, 1.0, -1.0, 1.0, -0.5, 0.5],
                'camera_positions': 3,
                'trajectory_prediction_horizon': 8,
            }
        }
    
    def save_config(self):
        """Save configuration to file."""
        config_path = self.experiment_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"Configuration saved to {config_path}")

def main():
    """Main setup function."""
    logger.info("Setting up MAGVIT 3D Trajectory Experiment")
    
    # Initialize setup
    setup = MAGVIT3DTrajectorySetup()
    
    # Clone repository and setup environment
    setup.clone_magvit_repository()
    setup.setup_environment()
    
    # Generate training data
    logger.info("Generating 3D training data...")
    data_generator = Trajectory3DDataGenerator()
    dataset = data_generator.generate_dataset(num_samples=500)  # Smaller for 3D complexity
    
    # Save dataset
    data_path = setup.data_dir / "trajectories_3d_dataset.npz"
    np.savez_compressed(data_path, **dataset)
    logger.info(f"3D Dataset saved to {data_path}")
    
    # Setup training configuration
    training_setup = MAGVIT3DTraining(setup.experiment_dir)
    training_setup.save_config()
    
    # Create training script
    training_script = setup.experiment_dir / "train_magvit_3d.py"
    with open(training_script, 'w') as f:
        f.write("""#!/usr/bin/env python3
# MAGVIT 3D Trajectory Training Script
# This script will be populated with MAGVIT-specific 3D training code

import numpy as np
import json
from pathlib import Path

def load_3d_dataset(data_path):
    \"\"\"Load the 3D trajectory dataset.\"\"\"
    data = np.load(data_path, allow_pickle=True)
    return data['multi_view_videos'], data['trajectories_3d'], data['shape_labels']

def train_magvit_3d_model():
    \"\"\"Train the MAGVIT model on 3D trajectories.\"\"\"
    print("Training MAGVIT 3D trajectory model...")
    
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Load dataset
    multi_view_videos, trajectories_3d, shape_labels = load_3d_dataset('data/trajectories_3d_dataset.npz')
    
    # TODO: Implement MAGVIT 3D training logic
    print(f"Multi-view videos count: {len(multi_view_videos)}")
    print(f"3D trajectories shape: {trajectories_3d.shape}")
    print(f"Shape labels shape: {shape_labels.shape}")
    
    if len(multi_view_videos) > 0:
        print(f"First video cameras: {list(multi_view_videos[0].keys())}")
        for cam_key in multi_view_videos[0].keys():
            print(f"  {cam_key} shape: {multi_view_videos[0][cam_key].shape}")
    
    print("3D Training complete!")

if __name__ == "__main__":
    train_magvit_3d_model()
""")
    
    training_script.chmod(0o755)
    
    logger.info("MAGVIT 3D Trajectory experiment setup complete!")
    logger.info(f"Experiment directory: {setup.experiment_dir}")
    logger.info(f"Run training with: python {training_script}")

if __name__ == "__main__":
    main() 