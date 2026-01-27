#!/usr/bin/env python3
"""
MAGVIT 2D Trajectory Experiment Setup
=====================================

This script sets up the MAGVIT framework for learning and predicting 2D trajectories
of geometric shapes (squares, circles, triangles).

Based on: https://github.com/google-research/magvit
Paper: https://arxiv.org/abs/2204.02896
"""

import os
import numpy as np
import matplotlib.pyplot as plt
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

class MAGVIT2DTrajectorySetup:
    """Setup class for MAGVIT 2D trajectory experiments."""
    
    def __init__(self, experiment_dir: str = "experiments/magvit-2d-trajectories"):
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
        """Set up the Python environment for MAGVIT."""
        logger.info("Setting up MAGVIT environment...")
        
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
            "wandb>=0.13.0"
        ]
        
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install"
            ] + requirements, check=True)
            logger.info("Successfully installed MAGVIT dependencies")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            raise

class TrajectoryDataGenerator:
    """Generate 2D trajectory data for geometric shapes."""
    
    def __init__(self, width: int = 128, height: int = 128, seq_length: int = 16):
        self.width = width
        self.height = height
        self.seq_length = seq_length
        self.shapes = ['square', 'circle', 'triangle']
        
    def generate_square_trajectory(self, center_path: np.ndarray, size: int = 20) -> np.ndarray:
        """Generate a square moving along a trajectory."""
        frames = []
        
        for i, (cx, cy) in enumerate(center_path):
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Draw square
            half_size = size // 2
            top_left = (int(cx - half_size), int(cy - half_size))
            bottom_right = (int(cx + half_size), int(cy + half_size))
            
            cv2.rectangle(frame, top_left, bottom_right, (255, 100, 100), -1)
            frames.append(frame)
            
        return np.array(frames)
    
    def generate_circle_trajectory(self, center_path: np.ndarray, radius: int = 15) -> np.ndarray:
        """Generate a circle moving along a trajectory."""
        frames = []
        
        for i, (cx, cy) in enumerate(center_path):
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Draw circle
            cv2.circle(frame, (int(cx), int(cy)), radius, (100, 255, 100), -1)
            frames.append(frame)
            
        return np.array(frames)
    
    def generate_triangle_trajectory(self, center_path: np.ndarray, size: int = 20) -> np.ndarray:
        """Generate a triangle moving along a trajectory."""
        frames = []
        
        for i, (cx, cy) in enumerate(center_path):
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Draw triangle
            pts = np.array([
                [cx, cy - size],  # Top
                [cx - size, cy + size],  # Bottom left
                [cx + size, cy + size]   # Bottom right
            ], dtype=np.int32)
            
            cv2.fillPoly(frame, [pts], (100, 100, 255))
            frames.append(frame)
            
        return np.array(frames)
    
    def generate_trajectory_patterns(self) -> List[np.ndarray]:
        """Generate different trajectory patterns."""
        patterns = []
        
        # Linear trajectory
        start_x, start_y = 30, 64
        end_x, end_y = 98, 64
        linear_path = np.array([
            [start_x + (end_x - start_x) * i / (self.seq_length - 1),
             start_y + (end_y - start_y) * i / (self.seq_length - 1)]
            for i in range(self.seq_length)
        ])
        patterns.append(linear_path)
        
        # Circular trajectory
        center_x, center_y = 64, 64
        radius = 30
        circular_path = np.array([
            [center_x + radius * np.cos(2 * np.pi * i / self.seq_length),
             center_y + radius * np.sin(2 * np.pi * i / self.seq_length)]
            for i in range(self.seq_length)
        ])
        patterns.append(circular_path)
        
        # Sine wave trajectory
        sine_path = np.array([
            [20 + 88 * i / (self.seq_length - 1),
             64 + 20 * np.sin(4 * np.pi * i / (self.seq_length - 1))]
            for i in range(self.seq_length)
        ])
        patterns.append(sine_path)
        
        # Parabolic trajectory
        parabolic_path = np.array([
            [20 + 88 * i / (self.seq_length - 1),
             64 + 30 * ((i / (self.seq_length - 1)) - 0.5) ** 2]
            for i in range(self.seq_length)
        ])
        patterns.append(parabolic_path)
        
        return patterns
    
    def generate_dataset(self, num_samples: int = 1000) -> Dict[str, np.ndarray]:
        """Generate complete dataset for all shapes and trajectories."""
        dataset = {'videos': [], 'labels': [], 'trajectories': []}
        
        trajectory_patterns = self.generate_trajectory_patterns()
        
        for sample_idx in range(num_samples):
            # Random shape and trajectory
            shape = np.random.choice(self.shapes)
            trajectory = np.random.choice(trajectory_patterns)
            
            # Add noise to trajectory
            noise = np.random.normal(0, 2, trajectory.shape)
            noisy_trajectory = trajectory + noise
            
            # Generate video sequence
            if shape == 'square':
                video = self.generate_square_trajectory(noisy_trajectory)
            elif shape == 'circle':
                video = self.generate_circle_trajectory(noisy_trajectory)
            elif shape == 'triangle':
                video = self.generate_triangle_trajectory(noisy_trajectory)
            
            dataset['videos'].append(video)
            dataset['labels'].append(self.shapes.index(shape))
            dataset['trajectories'].append(noisy_trajectory)
            
            if sample_idx % 100 == 0:
                logger.info(f"Generated {sample_idx}/{num_samples} samples")
        
        return {
            'videos': np.array(dataset['videos']),
            'labels': np.array(dataset['labels']),
            'trajectories': np.array(dataset['trajectories'])
        }

class MAGVITTraining:
    """Training configuration for MAGVIT 2D trajectory prediction."""
    
    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.config = self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for MAGVIT training."""
        return {
            'model': {
                'name': 'magvit_2d_trajectories',
                'vocab_size': 1024,
                'hidden_dim': 512,
                'num_layers': 8,
                'num_heads': 8,
                'sequence_length': 16,
                'image_size': 128,
                'patch_size': 8,
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 1e-4,
                'num_epochs': 100,
                'warmup_steps': 1000,
                'weight_decay': 0.01,
                'gradient_clip_norm': 1.0,
            },
            'data': {
                'train_samples': 8000,
                'val_samples': 1000,
                'test_samples': 1000,
                'num_workers': 4,
            },
            'logging': {
                'log_interval': 100,
                'eval_interval': 1000,
                'save_interval': 2000,
                'use_wandb': True,
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
    logger.info("Setting up MAGVIT 2D Trajectory Experiment")
    
    # Initialize setup
    setup = MAGVIT2DTrajectorySetup()
    
    # Clone repository and setup environment
    setup.clone_magvit_repository()
    setup.setup_environment()
    
    # Generate training data
    logger.info("Generating training data...")
    data_generator = TrajectoryDataGenerator()
    dataset = data_generator.generate_dataset(num_samples=1000)
    
    # Save dataset
    data_path = setup.data_dir / "trajectories_dataset.npz"
    np.savez_compressed(data_path, **dataset)
    logger.info(f"Dataset saved to {data_path}")
    
    # Setup training configuration
    training_setup = MAGVITTraining(setup.experiment_dir)
    training_setup.save_config()
    
    # Create training script
    training_script = setup.experiment_dir / "train_magvit_2d.py"
    with open(training_script, 'w') as f:
        f.write("""#!/usr/bin/env python3
# MAGVIT 2D Trajectory Training Script
# This script will be populated with MAGVIT-specific training code

import numpy as np
import json
from pathlib import Path

def load_dataset(data_path):
    \"\"\"Load the trajectory dataset.\"\"\"
    data = np.load(data_path)
    return data['videos'], data['labels'], data['trajectories']

def train_magvit_model():
    \"\"\"Train the MAGVIT model on 2D trajectories.\"\"\"
    print("Training MAGVIT 2D trajectory model...")
    
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Load dataset
    videos, labels, trajectories = load_dataset('data/trajectories_dataset.npz')
    
    # TODO: Implement MAGVIT training logic
    print(f"Dataset shape: {videos.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Trajectories shape: {trajectories.shape}")
    
    print("Training complete!")

if __name__ == "__main__":
    train_magvit_model()
""")
    
    training_script.chmod(0o755)
    
    logger.info("MAGVIT 2D Trajectory experiment setup complete!")
    logger.info(f"Experiment directory: {setup.experiment_dir}")
    logger.info(f"Run training with: python {training_script}")

if __name__ == "__main__":
    main() 