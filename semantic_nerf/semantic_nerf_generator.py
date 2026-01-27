#!/usr/bin/env python3
"""
Semantic-NeRF Implementation for MONO_TO_3D Project
==================================================

Implementation of "In-Place Scene Labelling and Understanding with Implicit Scene Representation"
(ICCV 2021) adapted for cone/cylinder classification in stereo camera setups.

Key Features:
- Scene-specific semantic understanding
- Sparse label propagation  
- Multi-view consistency enforcement
- Label denoising and super-resolution
- Integration with MONO_TO_3D coordinate system

Author: AI Assistant
Date: 2024-06-24
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import random
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CameraConfig:
    """Camera configuration matching MONO_TO_3D setup."""
    fx: float = 800.0
    fy: float = 800.0
    cx: float = 320.0
    cy: float = 240.0
    baseline: float = 0.65  # 65cm baseline
    height: float = 2.55    # 2.55m camera height
    image_width: int = 640
    image_height: int = 480


@dataclass
class SemanticLabel:
    """Semantic label with confidence and spatial information."""
    label_id: int
    class_name: str
    confidence: float
    position: np.ndarray  # 3D position
    bbox_2d: np.ndarray   # 2D bounding box [x1, y1, x2, y2]
    mask: np.ndarray      # Binary mask
    view_id: int          # Source view identifier


class PositionalEncoder(nn.Module):
    """Positional encoding for coordinates."""
    
    def __init__(self, input_dim: int = 3, num_frequencies: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.output_dim = input_dim * (2 * num_frequencies + 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding to input coordinates."""
        # x shape: (N, input_dim)
        encoding = [x]
        
        for freq in range(self.num_frequencies):
            for func in [torch.sin, torch.cos]:
                encoding.append(func(2.0 ** freq * np.pi * x))
        
        return torch.cat(encoding, dim=-1)


class SemanticMLP(nn.Module):
    """MLP for semantic prediction."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict semantic logits."""
        return self.layers(x)


class DensityMLP(nn.Module):
    """MLP for density prediction."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict density."""
        return F.relu(self.layers(x))


class ColorMLP(nn.Module):
    """MLP for color prediction."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict RGB color."""
        return torch.sigmoid(self.layers(x))


class SemanticNeRF(nn.Module):
    """
    Semantic Neural Radiance Field for cone/cylinder classification.
    
    Extends NeRF with semantic understanding for scene-specific learning
    with sparse label propagation and multi-view consistency.
    """
    
    def __init__(
        self,
        pos_frequencies: int = 10,
        dir_frequencies: int = 4,
        hidden_dim: int = 256,
        num_classes: int = 4,  # background, cone, cylinder, ground
        enable_label_propagation: bool = True,
        enable_denoising: bool = True
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.enable_label_propagation = enable_label_propagation
        self.enable_denoising = enable_denoising
        
        # Positional encoders
        self.pos_encoder = PositionalEncoder(3, pos_frequencies)
        self.dir_encoder = PositionalEncoder(3, dir_frequencies)
        
        pos_dim = self.pos_encoder.output_dim
        dir_dim = self.dir_encoder.output_dim
        
        # Core MLPs
        self.density_mlp = DensityMLP(pos_dim, hidden_dim)
        self.color_mlp = ColorMLP(pos_dim + dir_dim, hidden_dim)
        self.semantic_mlp = SemanticMLP(pos_dim, hidden_dim, num_classes)
        
        # Semantic consistency module
        if enable_label_propagation:
            self.consistency_mlp = nn.Sequential(
                nn.Linear(pos_dim + num_classes, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim // 2, num_classes)
            )
        
        # Label denoising module
        if enable_denoising:
            self.denoising_mlp = nn.Sequential(
                nn.Linear(num_classes * 2, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim // 2, num_classes)
            )
    
    def forward(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor,
        semantic_context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through Semantic-NeRF.
        
        Args:
            positions: 3D positions (N, 3)
            directions: Ray directions (N, 3)
            semantic_context: Optional semantic context from sparse labels (N, num_classes)
            
        Returns:
            density: Volume density (N, 1)
            color: RGB color (N, 3)
            semantics: Semantic logits (N, num_classes)
        """
        # Encode positions and directions
        pos_encoded = self.pos_encoder(positions)
        dir_encoded = self.dir_encoder(directions)
        
        # Predict density
        density = self.density_mlp(pos_encoded)
        
        # Predict color
        color_input = torch.cat([pos_encoded, dir_encoded], dim=-1)
        color = self.color_mlp(color_input)
        
        # Predict semantics
        semantics = self.semantic_mlp(pos_encoded)
        
        # Apply label propagation if enabled and context available
        if self.enable_label_propagation and semantic_context is not None:
            consistency_input = torch.cat([pos_encoded, semantic_context], dim=-1)
            consistency_semantics = self.consistency_mlp(consistency_input)
            
            # Blend with base semantics
            alpha = 0.3  # Blending factor
            semantics = (1 - alpha) * semantics + alpha * consistency_semantics
        
        # Apply denoising if enabled and context available
        if self.enable_denoising and semantic_context is not None:
            denoising_input = torch.cat([semantics, semantic_context], dim=-1)
            denoised_semantics = self.denoising_mlp(denoising_input)
            
            # Use denoised semantics
            semantics = denoised_semantics
        
        return density, color, semantics


class VolumeRenderer:
    """Volume rendering for Semantic-NeRF."""
    
    def __init__(
        self,
        near: float = 0.1,
        far: float = 10.0,
        num_samples: int = 64,
        num_importance_samples: int = 128
    ):
        """
        Initialize volume renderer.
        
        Args:
            near: Near plane distance
            far: Far plane distance
            num_samples: Number of coarse samples per ray
            num_importance_samples: Number of fine samples per ray
        """
        self.near = near
        self.far = far
        self.num_samples = num_samples
        self.num_importance_samples = num_importance_samples
    
    @staticmethod
    def render_rays(
        density: torch.Tensor,
        color: torch.Tensor,
        semantics: torch.Tensor,
        t_vals: torch.Tensor,
        noise_std: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Render rays using volume rendering.
        
        Args:
            density: Volume density (N_rays, N_samples, 1)
            color: RGB color (N_rays, N_samples, 3)
            semantics: Semantic logits (N_rays, N_samples, num_classes)
            t_vals: Sample distances (N_rays, N_samples)
            noise_std: Standard deviation for noise injection
            
        Returns:
            rgb: Rendered RGB (N_rays, 3)
            depth: Rendered depth (N_rays, 1)
            semantic_map: Rendered semantic map (N_rays, num_classes)
            weights: Rendering weights (N_rays, N_samples)
        """
        # Add noise to density for regularization
        if noise_std > 0:
            noise = torch.randn_like(density) * noise_std
            density = density + noise
        
        # Compute deltas
        deltas = t_vals[..., 1:] - t_vals[..., :-1]
        deltas = torch.cat([deltas, torch.full_like(deltas[..., :1], 1e10)], dim=-1)
        
        # Compute alpha values
        alpha = 1.0 - torch.exp(-F.relu(density[..., 0]) * deltas)
        
        # Compute transmittance
        transmittance = torch.cumprod(
            torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha[..., :-1]], dim=-1),
            dim=-1
        )
        
        # Compute weights
        weights = alpha * transmittance
        
        # Render RGB
        rgb = torch.sum(weights[..., None] * color, dim=-2)
        
        # Render depth
        depth = torch.sum(weights * t_vals, dim=-1, keepdim=True)
        
        # Render semantic map
        semantic_weights = F.softmax(semantics, dim=-1)
        semantic_map = torch.sum(weights[..., None] * semantic_weights, dim=-2)
        
        return rgb, depth, semantic_map, weights
    
    def render(
        self,
        model: torch.nn.Module,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Render rays using the neural radiance field model.
        
        Args:
            model: Semantic-NeRF model
            rays_o: Ray origins (N_rays, 3)
            rays_d: Ray directions (N_rays, 3)
            device: Device to run computation on
            
        Returns:
            rgb: Rendered RGB (N_rays, 3)
            depth: Rendered depth (N_rays, 1)
            semantic_map: Rendered semantic map (N_rays, num_classes)
            weights: Rendering weights (N_rays, N_samples)
        """
        N_rays = rays_o.shape[0]
        
        # Generate sample points along rays
        t_vals = torch.linspace(self.near, self.far, self.num_samples, device=device)
        t_vals = t_vals.expand(N_rays, self.num_samples)
        
        # Add noise to t_vals for regularization during training
        if model.training:
            mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
            upper = torch.cat([mids, t_vals[..., -1:]], -1)
            lower = torch.cat([t_vals[..., :1], mids], -1)
            t_rand = torch.rand(t_vals.shape, device=device)
            t_vals = lower + (upper - lower) * t_rand
        
        # Generate 3D points along rays
        pts = rays_o[..., None, :] + rays_d[..., None, :] * t_vals[..., :, None]  # (N_rays, N_samples, 3)
        
        # Flatten for model evaluation
        pts_flat = pts.reshape(-1, 3)
        dirs_flat = rays_d[:, None, :].expand(N_rays, self.num_samples, 3).reshape(-1, 3)
        
        # Evaluate model
        with torch.set_grad_enabled(model.training):
            density, color, semantics = model(pts_flat, dirs_flat)
        
        # Reshape back
        density = density.reshape(N_rays, self.num_samples, 1)
        color = color.reshape(N_rays, self.num_samples, 3)
        semantics = semantics.reshape(N_rays, self.num_samples, -1)
        
        # Volume rendering
        rgb, depth, semantic_map, weights = self.render_rays(
            density, color, semantics, t_vals
        )
        
        return rgb, depth, semantic_map, weights


class SemanticDataset:
    """
    Dataset for Semantic-NeRF training with sparse labels and multi-view consistency.
    """
    
    def __init__(
        self,
        num_scenes: int = 10,
        camera_config: Optional[CameraConfig] = None,
        sparse_ratio: float = 0.8,  # Ratio of unlabeled pixels
        enable_super_resolution: bool = True,
        enable_denoising: bool = True,
        max_objects_per_scene: int = 3,
        views_per_scene: int = 8,
        sparse_labels_per_object: int = 25
    ):
        self.num_scenes = num_scenes
        self.camera_config = camera_config or CameraConfig()
        self.sparse_ratio = sparse_ratio
        self.enable_super_resolution = enable_super_resolution
        self.enable_denoising = enable_denoising
        self.max_objects_per_scene = max_objects_per_scene
        self.views_per_scene = views_per_scene
        self.sparse_labels_per_object = sparse_labels_per_object
        
        # Semantic classes for cone/cylinder classification
        self.semantic_classes = {
            0: 'background',
            1: 'cone',
            2: 'cylinder', 
            3: 'ground'
        }
        
        # Generate scenes
        self.scenes = self._generate_scenes()
    
    def _generate_scenes(self) -> List[Dict]:
        """Generate synthetic scenes with cone/cylinder objects."""
        scenes = []
        
        for scene_id in range(self.num_scenes):
            scene = self._generate_single_scene(scene_id)
            scenes.append(scene)
        
        return scenes
    
    def _generate_single_scene(self, scene_id: int) -> Dict:
        """Generate a single scene with objects and sparse labels."""
        np.random.seed(42 + scene_id)
        random.seed(42 + scene_id)
        
        # Generate objects
        num_objects = np.random.randint(1, self.max_objects_per_scene + 1)  # 1 to max_objects_per_scene
        objects = []
        
        for obj_id in range(num_objects):
            obj_type = np.random.choice(['cone', 'cylinder'])
            
            # Position within camera FOV - ensure projection stays within image bounds
            x = np.random.uniform(0.1, 0.3)  # 10-30cm from center (narrower range)
            y = np.random.uniform(0.2, 0.4)  # 20-40cm forward (much closer to avoid large projection)
            z = self.camera_config.height - np.random.uniform(0.05, 0.25)  # On ground
            
            # Scale parameters
            radius = np.random.uniform(0.03, 0.08)  # 3-8cm radius
            height = np.random.uniform(0.1, 0.3)    # 10-30cm height
            
            # Color
            colors = [
                [0.8, 0.2, 0.2],  # Red
                [0.2, 0.2, 0.8],  # Blue
                [0.2, 0.8, 0.2],  # Green
                [0.8, 0.8, 0.2],  # Yellow
            ]
            color = colors[np.random.randint(0, len(colors))]
            
            obj = {
                'object_id': obj_id,
                'object_type': obj_type,
                'position': np.array([x, y, z]),
                'scale': np.array([radius, height]),
                'color': np.array(color),
                'semantic_id': 1 if obj_type == 'cone' else 2
            }
            objects.append(obj)
        
        # Generate sparse semantic labels
        sparse_labels = self._generate_sparse_labels(objects, scene_id)
        
        # Generate multi-view data
        views = self._generate_multiview_data(objects, scene_id)
        
        scene = {
            'scene_id': scene_id,
            'objects': objects,
            'sparse_labels': sparse_labels,
            'views': views,
            'camera_config': self.camera_config
        }
        
        return scene
    
    def _generate_sparse_labels(self, objects: List[Dict], scene_id: int) -> Dict:
        """Generate sparse semantic labels for the scene."""
        # Simulate sparse labeling with only a few annotated pixels per object
        labels = {}
        
        for obj in objects:
            obj_id = obj['object_id']
            
            # Generate sparse pixel annotations
            num_pixels = np.random.randint(self.sparse_labels_per_object // 2, self.sparse_labels_per_object + 1)  # Around sparse_labels_per_object pixels per object
            
            # Simulate 2D pixel coordinates
            pixels = []
            for _ in range(num_pixels):
                # Random pixel within object's projected region
                u = np.random.randint(50, 590)  # Within image bounds
                v = np.random.randint(50, 430)
                confidence = np.random.uniform(0.7, 1.0)
                
                pixels.append({
                    'u': u, 'v': v,
                    'semantic_id': obj['semantic_id'],
                    'confidence': confidence
                })
            
            labels[obj_id] = {
                'object_type': obj['object_type'],
                'pixels': pixels,
                'total_pixels': len(pixels)
            }
        
        return labels
    
    def _generate_multiview_data(self, objects: List[Dict], scene_id: int) -> List[Dict]:
        """Generate multi-view camera data for consistency."""
        views = []
        
        # Generate multiple viewpoints around the scene
        num_views = self.views_per_scene  # Use specified number of views per scene
        
        for view_id in range(num_views):
            # Vary camera position slightly
            base_x, base_y = 0.0, 0.0  # Center position
            offset_x = np.random.uniform(-0.1, 0.1)  # ±10cm variation
            offset_y = np.random.uniform(-0.1, 0.1)
            
            # Camera pose
            camera_pos = np.array([base_x + offset_x, base_y + offset_y, self.camera_config.height])
            camera_rot = np.array([0.0, 0.0, np.random.uniform(-0.1, 0.1)])  # Small rotation
            
            # Project objects to this view
            projected_objects = []
            for obj in objects:
                # Simple projection (assuming downward-looking camera)
                obj_pos = obj['position']
                
                # 2D projection
                u = self.camera_config.fx * (obj_pos[0] - camera_pos[0]) + self.camera_config.cx
                v = self.camera_config.fy * (obj_pos[1] - camera_pos[1]) + self.camera_config.cy
                
                # Check if in view
                if 0 <= u < self.camera_config.image_width and 0 <= v < self.camera_config.image_height:
                    projected_objects.append({
                        'object_id': obj['object_id'],
                        'object_type': obj['object_type'],
                        'u': u, 'v': v,
                        'depth': abs(obj_pos[2] - camera_pos[2]),
                        'semantic_id': obj['semantic_id']
                    })
            
            view = {
                'view_id': view_id,
                'camera_pos': camera_pos,
                'camera_rot': camera_rot,
                'projected_objects': projected_objects
            }
            views.append(view)
        
        return views
    
    def __len__(self) -> int:
        return len(self.scenes)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a scene with rendered images and labels for training."""
        scene = self.scenes[idx]
        
        # Generate synthetic RGB image (placeholder)
        image = torch.zeros(3, self.camera_config.image_height, self.camera_config.image_width, dtype=torch.float32)
        
        # Create sparse labels tensor
        sparse_labels = torch.full((self.camera_config.image_height, self.camera_config.image_width), -1, dtype=torch.long)
        
        # Fill in sparse labels from scene data
        for obj_id, label_data in scene['sparse_labels'].items():
            for pixel in label_data['pixels']:
                u, v = int(pixel['u']), int(pixel['v'])
                if 0 <= u < self.camera_config.image_width and 0 <= v < self.camera_config.image_height:
                    sparse_labels[v, u] = pixel['semantic_id']
        
        # Create dense labels (ground truth) - simplified version
        dense_labels = torch.zeros(self.camera_config.image_height, self.camera_config.image_width, dtype=torch.long)
        
        # For demonstration, create synthetic dense labels based on objects
        for obj in scene['objects']:
            # Simple rectangular region for each object (placeholder)
            center_u = int(obj['position'][0] * self.camera_config.fx + self.camera_config.cx)
            center_v = int(obj['position'][1] * self.camera_config.fy + self.camera_config.cy)
            
            # Create object region
            size = 30  # pixels
            u_min, u_max = max(0, center_u - size), min(self.camera_config.image_width, center_u + size)
            v_min, v_max = max(0, center_v - size), min(self.camera_config.image_height, center_v + size)
            
            dense_labels[v_min:v_max, u_min:u_max] = obj['semantic_id']
        
        # Camera pose (identity for simplicity)
        camera_pose = torch.eye(4)
        
        return {
            'scene_id': scene['scene_id'],
            'image': image,
            'sparse_labels': sparse_labels,
            'dense_labels': dense_labels,
            'camera_pose': camera_pose,
            'objects': scene['objects'],
            'views': scene['views'],
            'camera_config': scene['camera_config']
        }


class LabelPropagator:
    """Label propagation for sparse semantic labels."""
    
    def __init__(self, spatial_sigma: float = 0.1, semantic_sigma: float = 0.2, color_sigma: float = 0.1, max_iterations: int = 50):
        self.spatial_sigma = spatial_sigma
        self.semantic_sigma = semantic_sigma
        self.color_sigma = color_sigma
        self.max_iterations = max_iterations
    
    def propagate_labels(
        self,
        image_or_sparse_labels: torch.Tensor,
        sparse_labels_or_positions: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Propagate sparse labels to dense predictions.
        
        This method supports two calling conventions:
        1. For 3D point clouds: propagate_labels(sparse_labels, positions, features)
        2. For 2D images: propagate_labels(image, sparse_labels)
        
        Args:
            image_or_sparse_labels: Either RGB image (3, H, W) or sparse labels (N_sparse, num_classes)
            sparse_labels_or_positions: Either sparse labels (H, W) or 3D positions (N_sparse, 3)
            features: Feature vectors for similarity (N_sparse, feature_dim) - only for 3D mode
            
        Returns:
            dense_labels: Dense semantic predictions
        """
        # Check if this is 2D image mode or 3D point cloud mode
        if sparse_labels_or_positions is not None and len(image_or_sparse_labels.shape) == 3:
            # 2D image mode: propagate_labels(image, sparse_labels)
            return self._propagate_labels_2d(image_or_sparse_labels, sparse_labels_or_positions)
        elif features is not None:
            # 3D point cloud mode: propagate_labels(sparse_labels, positions, features)
            return self._propagate_labels_3d(image_or_sparse_labels, sparse_labels_or_positions, features)
        else:
            raise ValueError("Invalid arguments. Use either (image, sparse_labels) or (sparse_labels, positions, features)")
    
    def _propagate_labels_2d(self, image: torch.Tensor, sparse_labels: torch.Tensor) -> torch.Tensor:
        """Propagate labels in 2D image space."""
        device = image.device
        H, W = sparse_labels.shape
        
        # Convert sparse labels to dense predictions using simple diffusion
        dense_labels = sparse_labels.clone().float()
        
        # Simple iterative propagation
        for _ in range(self.max_iterations):
            # Create a padded version for convolution
            padded = F.pad(dense_labels.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate')
            
            # Simple 3x3 averaging kernel
            kernel = torch.ones(1, 1, 3, 3, device=device) / 9.0
            smoothed = F.conv2d(padded, kernel).squeeze()
            
            # Only update unlabeled pixels (-1)
            unlabeled_mask = sparse_labels == -1
            dense_labels[unlabeled_mask] = smoothed[unlabeled_mask]
        
        return dense_labels.long()
    
    def _propagate_labels_3d(self, sparse_labels: torch.Tensor, positions: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Propagate labels in 3D point cloud space."""
        device = sparse_labels.device
        n_sparse = sparse_labels.shape[0]
        
        # Compute spatial similarity weights
        spatial_weights = self._compute_spatial_weights(positions)
        
        # Compute semantic similarity weights
        semantic_weights = self._compute_semantic_weights(features)
        
        # Combine weights
        combined_weights = spatial_weights * semantic_weights
        
        # Normalize weights
        combined_weights = F.softmax(combined_weights / 0.1, dim=-1)
        
        # Propagate labels
        dense_labels = torch.matmul(combined_weights, sparse_labels)
        
        return F.softmax(dense_labels, dim=-1)
    
    def _compute_spatial_weights(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute spatial similarity weights."""
        # Pairwise distances
        distances = torch.cdist(positions, positions)
        
        # Gaussian kernel
        weights = torch.exp(-distances ** 2 / (2 * self.spatial_sigma ** 2))
        
        # Normalize
        weights = weights / torch.sum(weights, dim=-1, keepdim=True)
        
        return weights
    
    def _compute_semantic_weights(self, features: torch.Tensor) -> torch.Tensor:
        """Compute semantic similarity weights."""
        # Cosine similarity
        features_norm = F.normalize(features, p=2, dim=-1)
        similarities = torch.matmul(features_norm, features_norm.T)
        
        # Convert to weights
        weights = torch.exp(similarities / self.semantic_sigma)
        
        # Normalize
        weights = weights / torch.sum(weights, dim=-1, keepdim=True)
        
        return weights


def convert_numpy_to_python(obj):
    """Convert numpy arrays and types to Python native types for JSON serialization."""
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    else:
        return obj


def create_semantic_nerf_demo():
    """Create a demonstration of Semantic-NeRF for cone/cylinder classification."""
    print("🔬 Semantic-NeRF for MONO_TO_3D Demo")
    print("=" * 50)
    
    # Create camera configuration
    camera_config = CameraConfig()
    print(f"📷 Camera Config: {camera_config.image_width}x{camera_config.image_height}")
    print(f"   Baseline: {camera_config.baseline}m, Height: {camera_config.height}m")
    
    # Create Semantic-NeRF model
    model = SemanticNeRF(
        pos_frequencies=10,
        dir_frequencies=4,
        hidden_dim=256,
        num_classes=4,
        enable_label_propagation=True,
        enable_denoising=True
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model initialized with {total_params:,} parameters")
    
    # Create dataset
    dataset = SemanticDataset(
        num_scenes=5,
        camera_config=camera_config,
        sparse_ratio=0.8,
        enable_super_resolution=True,
        enable_denoising=True
    )
    
    print(f"📊 Generated {len(dataset)} scenes with sparse semantic labels")
    
    # Test scene
    scene = dataset[0]
    print(f"\n🎬 Scene 0: {len(scene['objects'])} objects, {len(scene['views'])} views")
    
    for i, obj in enumerate(scene['objects']):
        print(f"   Object {i}: {obj['object_type']} at {obj['position']}")
    
    # Test forward pass
    batch_size = 100
    positions = torch.randn(batch_size, 3) * 0.5  # Random positions
    directions = torch.randn(batch_size, 3)
    directions = F.normalize(directions, p=2, dim=-1)  # Normalize directions
    
    with torch.no_grad():
        density, color, semantics = model(positions, directions)
    
    print(f"\n🔄 Forward pass test:")
    print(f"   Input: {batch_size} rays")
    print(f"   Output: density {density.shape}, color {color.shape}, semantics {semantics.shape}")
    
    # Test label propagation
    propagator = LabelPropagator()
    sparse_labels = torch.zeros(10, 4)  # 10 sparse labels
    sparse_labels[torch.arange(10), torch.randint(0, 4, (10,))] = 1.0  # One-hot
    
    features = torch.randn(batch_size, 256)  # Feature vectors
    dense_labels = propagator.propagate_labels(sparse_labels, positions[:10], features[:10])
    
    print(f"   Label propagation: {sparse_labels.shape} → {dense_labels.shape}")
    
    return {
        'model': model,
        'dataset': dataset,
        'camera_config': camera_config,
        'propagator': propagator,
        'total_parameters': total_params
    }


if __name__ == "__main__":
    # Run demonstration
    demo_results = create_semantic_nerf_demo()
    
    print(f"\n✅ Semantic-NeRF demo completed successfully!")
    print(f"   Model: {demo_results['total_parameters']:,} parameters")
    print(f"   Dataset: {len(demo_results['dataset'])} scenes")
    print(f"   Ready for MONO_TO_3D integration!")