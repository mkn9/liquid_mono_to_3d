#!/usr/bin/env python3
"""
OV-NeRF Synthetic Data Generator for Cone and Cylinder Classification
====================================================================

This module implements an OV-NeRF based synthetic data generator specifically
designed for creating training data for cone and cylinder classification in
the MONO_TO_3D tracking system.

Key Features:
- Open-vocabulary 3D scene understanding
- Controllable object generation (cones, cylinders)
- Multi-view consistent rendering
- Semantic annotation generation
- Integration with stereo camera setup

Author: AI Assistant
Date: 2025-01-23
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from tqdm import tqdm
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CameraConfig:
    """Camera configuration for stereo setup."""
    fx: float = 800.0
    fy: float = 800.0
    cx: float = 320.0
    cy: float = 240.0
    baseline: float = 0.65  # 65cm baseline
    height: float = 2.55    # Camera height
    image_width: int = 640
    image_height: int = 480

@dataclass
class ObjectConfig:
    """Configuration for 3D objects."""
    object_type: str  # 'cone' or 'cylinder'
    radius: float
    height: float
    position: np.ndarray
    orientation: np.ndarray  # roll, pitch, yaw
    material_properties: Dict[str, float]

class PositionalEncoding(nn.Module):
    """Positional encoding for coordinate inputs."""
    
    def __init__(self, input_dim: int = 3, num_frequencies: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.output_dim = input_dim * (2 * num_frequencies + 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding to input coordinates."""
        # x shape: (..., input_dim)
        encoded = [x]
        
        for freq in range(self.num_frequencies):
            for func in [torch.sin, torch.cos]:
                encoded.append(func(2.0 ** freq * np.pi * x))
        
        return torch.cat(encoded, dim=-1)

class NeRFMLP(nn.Module):
    """Neural Radiance Field MLP for density and color prediction."""
    
    def __init__(self, 
                 pos_encoding_dim: int = 63,  # 3 * (2 * 10 + 1)
                 dir_encoding_dim: int = 27,  # 3 * (2 * 4 + 1)
                 hidden_dim: int = 256,
                 num_layers: int = 8):
        super().__init__()
        
        # Density network
        self.density_layers = nn.ModuleList()
        self.density_layers.append(nn.Linear(pos_encoding_dim, hidden_dim))
        
        for i in range(num_layers - 1):
            if i == 4:  # Skip connection
                self.density_layers.append(nn.Linear(hidden_dim + pos_encoding_dim, hidden_dim))
            else:
                self.density_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.density_output = nn.Linear(hidden_dim, 1)
        self.feature_output = nn.Linear(hidden_dim, hidden_dim)
        
        # Color network
        self.color_layers = nn.ModuleList([
            nn.Linear(hidden_dim + dir_encoding_dim, hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 3)
        ])
        
    def forward(self, pos_encoded: torch.Tensor, dir_encoded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through NeRF MLP."""
        # Density prediction
        x = pos_encoded
        for i, layer in enumerate(self.density_layers):
            if i == 4:  # Skip connection
                x = torch.cat([x, pos_encoded], dim=-1)
            x = F.relu(layer(x))
        
        density = F.relu(self.density_output(x))
        features = self.feature_output(x)
        
        # Color prediction
        color_input = torch.cat([features, dir_encoded], dim=-1)
        color = color_input
        for layer in self.color_layers[:-1]:
            color = F.relu(layer(color))
        color = torch.sigmoid(self.color_layers[-1](color))
        
        return density, color

class SemanticEncoder(nn.Module):
    """Semantic encoder for open-vocabulary understanding."""
    
    def __init__(self, vocab_size: int = 1000, embedding_dim: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.semantic_mlp = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
    def forward(self, semantic_tokens: torch.Tensor) -> torch.Tensor:
        """Encode semantic tokens to features."""
        embedded = self.embedding(semantic_tokens)
        return self.semantic_mlp(embedded.mean(dim=1))  # Average pooling

class OVNeRF(nn.Module):
    """Open-Vocabulary Neural Radiance Field."""
    
    def __init__(self, 
                 pos_frequencies: int = 10,
                 dir_frequencies: int = 4,
                 vocab_size: int = 1000):
        super().__init__()
        
        self.pos_encoding = PositionalEncoding(3, pos_frequencies)
        self.dir_encoding = PositionalEncoding(3, dir_frequencies)
        self.semantic_encoder = SemanticEncoder(vocab_size)
        
        # Enhanced NeRF with semantic conditioning
        self.nerf = NeRFMLP(
            pos_encoding_dim=self.pos_encoding.output_dim + 64,  # +semantic features
            dir_encoding_dim=self.dir_encoding.output_dim
        )
        
    def forward(self, 
                positions: torch.Tensor, 
                directions: torch.Tensor,
                semantic_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through OV-NeRF."""
        # Encode inputs
        pos_encoded = self.pos_encoding(positions)
        dir_encoded = self.dir_encoding(directions)
        semantic_features = self.semantic_encoder(semantic_tokens)
        
        # Broadcast semantic features to match position batch size
        semantic_features = semantic_features.unsqueeze(1).expand(-1, positions.shape[1], -1)
        semantic_features = semantic_features.reshape(-1, semantic_features.shape[-1])
        
        # Concatenate position encoding with semantic features
        pos_semantic = torch.cat([pos_encoded, semantic_features], dim=-1)
        
        return self.nerf(pos_semantic, dir_encoded)

class SyntheticDataset(Dataset):
    """Dataset for synthetic cone and cylinder scenes."""
    
    def __init__(self, 
                 num_scenes: int = 1000,
                 camera_config: CameraConfig = None,
                 image_size: Tuple[int, int] = (480, 640)):
        self.num_scenes = num_scenes
        self.camera_config = camera_config or CameraConfig()
        self.image_size = image_size
        self.scenes = self._generate_scenes()
        
        # Vocabulary for semantic tokens
        self.vocab = {
            'cone': 0, 'cylinder': 1, 'background': 2,
            'metal': 3, 'plastic': 4, 'wood': 5,
            'red': 6, 'blue': 7, 'green': 8, 'gray': 9
        }
        
    def _generate_scenes(self) -> List[Dict]:
        """Generate random scenes with cones and cylinders."""
        scenes = []
        
        for i in range(self.num_scenes):
            # Random object type
            object_type = random.choice(['cone', 'cylinder'])
            
            # Random object parameters
            if object_type == 'cone':
                radius = random.uniform(0.025, 0.075)  # 2.5-7.5cm
                height = random.uniform(0.15, 0.25)    # 15-25cm
            else:  # cylinder
                radius = random.uniform(0.025, 0.075)  # 2.5-7.5cm
                height = random.uniform(0.10, 0.30)    # 10-30cm
            
            # Random position (within camera view)
            position = np.array([
                random.uniform(0.2, 0.6),  # x: 20-60cm from left camera
                random.uniform(0.8, 1.2),  # y: 80-120cm from cameras
                random.uniform(2.3, 2.7)   # z: table height variation
            ])
            
            # Random orientation
            orientation = np.array([
                random.uniform(-0.2, 0.2),  # roll: ±11.5°
                random.uniform(-0.2, 0.2),  # pitch: ±11.5°
                random.uniform(0, 2*np.pi)  # yaw: full rotation
            ])
            
            # Random material properties
            material = random.choice(['metal', 'plastic', 'wood'])
            color = random.choice(['red', 'blue', 'green', 'gray'])
            
            material_props = {
                'roughness': random.uniform(0.1, 0.9),
                'metallic': 1.0 if material == 'metal' else 0.0,
                'specular': random.uniform(0.1, 0.9)
            }
            
            # Random lighting
            lighting = {
                'ambient': random.uniform(0.2, 0.4),
                'directional': random.uniform(0.4, 0.8),
                'direction': np.random.uniform(-1, 1, 3)
            }
            lighting['direction'] = lighting['direction'] / np.linalg.norm(lighting['direction'])
            
            scene = {
                'scene_id': i,
                'object': ObjectConfig(
                    object_type=object_type,
                    radius=radius,
                    height=height,
                    position=position,
                    orientation=orientation,
                    material_properties=material_props
                ),
                'material': material,
                'color': color,
                'lighting': lighting,
                'semantic_tokens': [object_type, material, color, 'background']
            }
            
            scenes.append(scene)
            
        return scenes
    
    def __len__(self) -> int:
        return self.num_scenes
    
    def __getitem__(self, idx: int) -> Dict:
        scene = self.scenes[idx]
        
        # Convert semantic tokens to indices
        semantic_indices = [self.vocab.get(token, 0) for token in scene['semantic_tokens']]
        
        return {
            'scene_id': scene['scene_id'],
            'object_config': scene['object'],
            'semantic_tokens': torch.tensor(semantic_indices, dtype=torch.long),
            'material': scene['material'],
            'color': scene['color'],
            'lighting': scene['lighting']
        }

class VolumeRenderer:
    """Volume renderer for NeRF."""
    
    def __init__(self, 
                 near: float = 0.5, 
                 far: float = 3.0, 
                 num_samples: int = 64,
                 num_importance_samples: int = 128):
        self.near = near
        self.far = far
        self.num_samples = num_samples
        self.num_importance_samples = num_importance_samples
    
    def sample_points_along_rays(self, 
                                rays_o: torch.Tensor, 
                                rays_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample points along camera rays."""
        # Stratified sampling
        t_vals = torch.linspace(0.0, 1.0, self.num_samples, device=rays_o.device)
        z_vals = self.near * (1.0 - t_vals) + self.far * t_vals
        
        # Add random perturbation
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., :1], mids], dim=-1)
        t_rand = torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * t_rand
        
        # Compute 3D points
        points = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        
        return points, z_vals
    
    def volume_render(self, 
                     densities: torch.Tensor, 
                     colors: torch.Tensor, 
                     z_vals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform volume rendering."""
        # Compute deltas
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)
        
        # Compute alpha values
        alpha = 1.0 - torch.exp(-densities.squeeze(-1) * dists)
        
        # Compute transmittance
        transmittance = torch.cumprod(
            torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha[..., :-1]], dim=-1), 
            dim=-1
        )
        
        # Compute weights
        weights = alpha * transmittance
        
        # Composite colors
        rgb = torch.sum(weights[..., None] * colors, dim=-2)
        
        # Compute depth
        depth = torch.sum(weights * z_vals, dim=-1)
        
        return rgb, depth

class StereoDataGenerator:
    """Generate stereo training data using OV-NeRF."""
    
    def __init__(self, 
                 model: OVNeRF,
                 camera_config: CameraConfig,
                 device: str = 'cuda'):
        self.model = model
        self.camera_config = camera_config
        self.device = device
        self.renderer = VolumeRenderer()
        
        # Stereo camera setup
        self.left_camera_pos = np.array([0.0, 0.0, camera_config.height])
        self.right_camera_pos = np.array([camera_config.baseline, 0.0, camera_config.height])
        
    def generate_camera_rays(self, camera_pos: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate camera rays for given camera position."""
        H, W = self.camera_config.image_height, self.camera_config.image_width
        fx, fy = self.camera_config.fx, self.camera_config.fy
        cx, cy = self.camera_config.cx, self.camera_config.cy
        
        # Pixel coordinates
        i, j = np.meshgrid(
            np.arange(W, dtype=np.float32),
            np.arange(H, dtype=np.float32),
            indexing='xy'
        )
        
        # Camera rays in camera coordinates
        dirs = np.stack([
            (i - cx) / fx,
            -(j - cy) / fy,  # Negative for image coordinate system
            -np.ones_like(i)
        ], axis=-1)
        
        # Transform to world coordinates (assuming camera looks in +Y direction)
        # Camera coordinate system: X=right, Y=forward, Z=up
        rays_d = dirs  # In world coordinates
        rays_o = np.broadcast_to(camera_pos, rays_d.shape)
        
        return torch.from_numpy(rays_o).float(), torch.from_numpy(rays_d).float()
    
    def render_scene(self, scene_data: Dict) -> Dict:
        """Render a complete scene from both camera viewpoints."""
        self.model.eval()
        
        with torch.no_grad():
            # Generate rays for both cameras
            left_rays_o, left_rays_d = self.generate_camera_rays(self.left_camera_pos)
            right_rays_o, right_rays_d = self.generate_camera_rays(self.right_camera_pos)
            
            # Move to device
            left_rays_o = left_rays_o.to(self.device)
            left_rays_d = left_rays_d.to(self.device)
            right_rays_o = right_rays_o.to(self.device)
            right_rays_d = right_rays_d.to(self.device)
            
            semantic_tokens = scene_data['semantic_tokens'].unsqueeze(0).to(self.device)
            
            # Render left view
            left_rgb, left_depth = self._render_view(left_rays_o, left_rays_d, semantic_tokens)
            
            # Render right view
            right_rgb, right_depth = self._render_view(right_rays_o, right_rays_d, semantic_tokens)
            
            return {
                'left_image': left_rgb.cpu().numpy(),
                'right_image': right_rgb.cpu().numpy(),
                'left_depth': left_depth.cpu().numpy(),
                'right_depth': right_depth.cpu().numpy(),
                'object_config': scene_data['object_config'],
                'semantic_tokens': scene_data['semantic_tokens'],
                'scene_id': scene_data['scene_id']
            }
    
    def _render_view(self, 
                    rays_o: torch.Tensor, 
                    rays_d: torch.Tensor, 
                    semantic_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Render a single view."""
        H, W = rays_o.shape[:2]
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        
        # Process in chunks to avoid memory issues
        chunk_size = 1024
        rgb_chunks = []
        depth_chunks = []
        
        for i in range(0, rays_o.shape[0], chunk_size):
            chunk_rays_o = rays_o[i:i+chunk_size]
            chunk_rays_d = rays_d[i:i+chunk_size]
            
            # Sample points along rays
            points, z_vals = self.renderer.sample_points_along_rays(chunk_rays_o, chunk_rays_d)
            
            # Flatten for network input
            points_flat = points.reshape(-1, 3)
            dirs_flat = chunk_rays_d[:, None, :].expand_as(points).reshape(-1, 3)
            
            # Expand semantic tokens
            batch_size = points_flat.shape[0]
            semantic_expanded = semantic_tokens.expand(batch_size, -1)
            
            # Forward pass
            densities, colors = self.model(points_flat, dirs_flat, semantic_expanded)
            
            # Reshape back
            densities = densities.reshape(*points.shape[:-1], 1)
            colors = colors.reshape(*points.shape[:-1], 3)
            
            # Volume rendering
            rgb, depth = self.renderer.volume_render(densities, colors, z_vals)
            
            rgb_chunks.append(rgb)
            depth_chunks.append(depth)
        
        # Concatenate chunks
        rgb = torch.cat(rgb_chunks, dim=0).reshape(H, W, 3)
        depth = torch.cat(depth_chunks, dim=0).reshape(H, W)
        
        return rgb, depth

class OVNeRFTrainer:
    """Trainer for OV-NeRF model."""
    
    def __init__(self, 
                 model: OVNeRF,
                 dataset: SyntheticDataset,
                 device: str = 'cuda',
                 learning_rate: float = 5e-4):
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.data_generator = StereoDataGenerator(model, dataset.camera_config, device)
        
    def train_epoch(self, num_scenes: int = 100) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True)
        
        for i, scene_data in enumerate(tqdm(dataloader, desc="Training")):
            if i >= num_scenes:
                break
                
            self.optimizer.zero_grad()
            
            # Generate target images (simplified geometric rendering)
            target_left, target_right = self._generate_target_images(scene_data)
            
            # Render with NeRF
            rendered = self.data_generator.render_scene({
                'semantic_tokens': scene_data['semantic_tokens'][0],
                'object_config': scene_data['object_config'],
                'scene_id': scene_data['scene_id'][0]
            })
            
            # Compute loss
            left_loss = F.mse_loss(
                torch.from_numpy(rendered['left_image']).to(self.device),
                target_left.to(self.device)
            )
            right_loss = F.mse_loss(
                torch.from_numpy(rendered['right_image']).to(self.device),
                target_right.to(self.device)
            )
            
            loss = left_loss + right_loss
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / min(num_scenes, len(dataloader))
    
    def _generate_target_images(self, scene_data: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate target images using geometric rendering."""
        # Simplified geometric rendering for training targets
        # In practice, this would use a proper graphics pipeline
        
        H, W = self.dataset.camera_config.image_height, self.dataset.camera_config.image_width
        
        # Create simple target images (placeholder)
        left_target = torch.zeros(H, W, 3)
        right_target = torch.zeros(H, W, 3)
        
        # Add object rendering based on object_config
        obj_config = scene_data['object_config']
        
        # Simple object rendering (placeholder - would be replaced with proper rendering)
        center_x, center_y = W // 2, H // 2
        radius_pixels = int(obj_config.radius * 1000)  # Convert to pixels
        
        # Draw simple colored circle/ellipse
        color = torch.tensor([0.7, 0.3, 0.3])  # Red-ish
        
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        mask = ((x - center_x) ** 2 + (y - center_y) ** 2) < radius_pixels ** 2
        
        left_target[mask] = color
        right_target[mask] = color
        
        return left_target, right_target

def main():
    """Main training and data generation pipeline."""
    parser = argparse.ArgumentParser(description='OV-NeRF Synthetic Data Generator')
    parser.add_argument('--num_scenes', type=int, default=100, help='Number of scenes to generate')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--output_dir', type=str, default='./synthetic_data', help='Output directory')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize components
    camera_config = CameraConfig()
    dataset = SyntheticDataset(num_scenes=args.num_scenes, camera_config=camera_config)
    model = OVNeRF()
    
    logger.info(f"Initialized OV-NeRF with {args.num_scenes} scenes")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Generate synthetic dataset metadata
    logger.info("Generating synthetic dataset metadata...")
    
    # Generate training data metadata
    train_data = []
    for i in tqdm(range(args.num_scenes), desc="Generating metadata"):
        scene_data = dataset[i]
        
        metadata = {
            'scene_id': int(scene_data['scene_id']),
            'object_type': scene_data['object_config'].object_type,
            'object_radius': float(scene_data['object_config'].radius),
            'object_height': float(scene_data['object_config'].height),
            'object_position': scene_data['object_config'].position.tolist(),
            'object_orientation': scene_data['object_config'].orientation.tolist(),
            'material': scene_data['material'],
            'color': scene_data['color'],
            'camera_config': {
                'fx': camera_config.fx,
                'fy': camera_config.fy,
                'cx': camera_config.cx,
                'cy': camera_config.cy,
                'baseline': camera_config.baseline,
                'height': camera_config.height
            }
        }
        
        train_data.append(metadata)
    
    # Save dataset summary
    dataset_summary = {
        'num_scenes': args.num_scenes,
        'camera_config': camera_config.__dict__,
        'scenes': train_data
    }
    
    with open(output_dir / "dataset_summary.json", 'w') as f:
        json.dump(dataset_summary, f, indent=2)
    
    logger.info(f"Generated metadata for {args.num_scenes} synthetic scenes in {output_dir}")
    logger.info("Dataset generation complete!")

if __name__ == "__main__":
    main() 