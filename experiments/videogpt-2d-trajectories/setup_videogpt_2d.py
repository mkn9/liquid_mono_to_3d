#!/usr/bin/env python3
"""
VideoGPT 2D Trajectory Experiment Setup
=======================================

This script sets up the VideoGPT framework for learning and predicting 2D trajectories
of geometric shapes (squares, circles, triangles).

Based on: https://github.com/wilson1yan/VideoGPT
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
import h5py

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoGPT2DTrajectorySetup:
    """Setup class for VideoGPT 2D trajectory experiments."""
    
    def __init__(self, experiment_dir: str = "experiments/videogpt-2d-trajectories"):
        self.experiment_dir = Path(experiment_dir)
        self.data_dir = self.experiment_dir / "data"
        self.models_dir = self.experiment_dir / "models"
        self.results_dir = self.experiment_dir / "results"
        self.videogpt_dir = self.experiment_dir / "videogpt"
        
        # Create directories
        for dir_path in [self.data_dir, self.models_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def clone_videogpt_repository(self):
        """Clone the VideoGPT repository."""
        logger.info("Cloning VideoGPT repository...")
        if not self.videogpt_dir.exists():
            try:
                subprocess.run([
                    "git", "clone", 
                    "https://github.com/wilson1yan/VideoGPT.git",
                    str(self.videogpt_dir)
                ], check=True)
                logger.info("Successfully cloned VideoGPT repository")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to clone VideoGPT repository: {e}")
                raise
        else:
            logger.info("VideoGPT repository already exists")
    
    def setup_environment(self):
        """Set up the Python environment for VideoGPT."""
        logger.info("Setting up VideoGPT environment...")
        
        # Install PyTorch and other dependencies
        requirements = [
            "torch>=1.7.1",
            "torchvision>=0.8.2",
            "torchaudio>=0.7.2",
            "pytorch-lightning>=1.5.0",
            "transformers>=4.12.0",
            "tensorboard>=2.7.0",
            "matplotlib>=3.5.0",
            "opencv-python>=4.5.0",
            "Pillow>=9.0.0",
            "tqdm>=4.64.0",
            "wandb>=0.13.0",
            "h5py>=3.7.0",
            "einops>=0.4.1"
        ]
        
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install"
            ] + requirements, check=True)
            logger.info("Successfully installed VideoGPT dependencies")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            raise

class VideoGPTTrajectoryDataGenerator:
    """Generate 2D trajectory data for geometric shapes in VideoGPT format."""
    
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
    
    def save_hdf5_dataset(self, dataset: Dict[str, np.ndarray], output_path: Path):
        """Save dataset in HDF5 format required by VideoGPT."""
        logger.info(f"Saving dataset to HDF5 format: {output_path}")
        
        with h5py.File(output_path, 'w') as f:
            # Create train and test splits
            split_idx = int(0.8 * len(dataset['videos']))
            
            # Training data
            train_group = f.create_group('train')
            train_group.create_dataset('videos', data=dataset['videos'][:split_idx])
            train_group.create_dataset('labels', data=dataset['labels'][:split_idx])
            train_group.create_dataset('trajectories', data=dataset['trajectories'][:split_idx])
            
            # Test data
            test_group = f.create_group('test')
            test_group.create_dataset('videos', data=dataset['videos'][split_idx:])
            test_group.create_dataset('labels', data=dataset['labels'][split_idx:])
            test_group.create_dataset('trajectories', data=dataset['trajectories'][split_idx:])
            
            # Metadata
            f.attrs['num_train_samples'] = split_idx
            f.attrs['num_test_samples'] = len(dataset['videos']) - split_idx
            f.attrs['sequence_length'] = self.seq_length
            f.attrs['image_width'] = self.width
            f.attrs['image_height'] = self.height
            f.attrs['num_classes'] = len(self.shapes)
            f.attrs['class_names'] = [name.encode('utf-8') for name in self.shapes]
        
        logger.info("HDF5 dataset saved successfully")

class VideoGPTTraining:
    """Training configuration for VideoGPT 2D trajectory prediction."""
    
    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.config = self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for VideoGPT training."""
        return {
            'model': {
                'name': 'videogpt_2d_trajectories',
                'n_codes': 2048,
                'n_hiddens': 240,
                'n_res_layers': 4,
                'downsample': [4, 4, 4],
                'sequence_length': 16,
                'resolution': 128,
                'embedding_dim': 256,
            },
            'training': {
                'batch_size': 16,
                'learning_rate': 3e-4,
                'num_epochs': 100,
                'warmup_steps': 1000,
                'weight_decay': 0.01,
                'gradient_clip_val': 1.0,
                'sync_batchnorm': True,
            },
            'data': {
                'train_samples': 8000,
                'test_samples': 2000,
                'num_workers': 8,
                'data_path': 'data/trajectories_dataset.h5',
            },
            'logging': {
                'log_interval': 100,
                'eval_interval': 1000,
                'save_interval': 2000,
                'use_wandb': True,
            },
            'vqvae': {
                'train_vqvae': True,
                'vqvae_epochs': 50,
                'videogpt_epochs': 50,
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
    logger.info("Setting up VideoGPT 2D Trajectory Experiment")
    
    # Initialize setup
    setup = VideoGPT2DTrajectorySetup()
    
    # Clone repository and setup environment
    setup.clone_videogpt_repository()
    setup.setup_environment()
    
    # Generate training data
    logger.info("Generating training data...")
    data_generator = VideoGPTTrajectoryDataGenerator()
    dataset = data_generator.generate_dataset(num_samples=1000)
    
    # Save dataset in HDF5 format
    hdf5_path = setup.data_dir / "trajectories_dataset.h5"
    data_generator.save_hdf5_dataset(dataset, hdf5_path)
    
    # Save dataset in NumPy format as backup
    npz_path = setup.data_dir / "trajectories_dataset.npz"
    np.savez_compressed(npz_path, **dataset)
    logger.info(f"Dataset saved to {npz_path} and {hdf5_path}")
    
    # Setup training configuration
    training_setup = VideoGPTTraining(setup.experiment_dir)
    training_setup.save_config()
    
    # Create training script
    training_script = setup.experiment_dir / "train_videogpt_2d.py"
    with open(training_script, 'w') as f:
        f.write("""#!/usr/bin/env python3
# VideoGPT 2D Trajectory Training Script
# This script will be populated with VideoGPT-specific training code

import numpy as np
import json
import h5py
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class TrajectoryDataset(Dataset):
    \"\"\"Dataset class for trajectory data.\"\"\"
    
    def __init__(self, data_path, split='train'):
        self.data_path = data_path
        self.split = split
        
        with h5py.File(data_path, 'r') as f:
            self.videos = f[split]['videos'][...]
            self.labels = f[split]['labels'][...]
            self.trajectories = f[split]['trajectories'][...]
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        video = torch.from_numpy(self.videos[idx]).float() / 255.0
        label = torch.from_numpy(np.array(self.labels[idx])).long()
        trajectory = torch.from_numpy(self.trajectories[idx]).float()
        
        # Permute to (T, C, H, W) format
        video = video.permute(0, 3, 1, 2)
        
        return video, label, trajectory

def load_dataset(data_path, split='train'):
    \"\"\"Load the trajectory dataset.\"\"\"
    return TrajectoryDataset(data_path, split)

def train_videogpt_model():
    \"\"\"Train the VideoGPT model on 2D trajectories.\"\"\"
    print("Training VideoGPT 2D trajectory model...")
    
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Load dataset
    train_dataset = load_dataset('data/trajectories_dataset.h5', 'train')
    test_dataset = load_dataset('data/trajectories_dataset.h5', 'test')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True,
        num_workers=config['data']['num_workers']
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    # TODO: Implement VideoGPT training logic
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Example batch
    for batch_idx, (videos, labels, trajectories) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Videos shape: {videos.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Trajectories shape: {trajectories.shape}")
        break
    
    print("Training complete!")

if __name__ == "__main__":
    train_videogpt_model()
""")
    
    training_script.chmod(0o755)
    
    logger.info("VideoGPT 2D Trajectory experiment setup complete!")
    logger.info(f"Experiment directory: {setup.experiment_dir}")
    logger.info(f"Run training with: python {training_script}")

if __name__ == "__main__":
    main() 