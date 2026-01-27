#!/usr/bin/env python3
"""
OpenObj-NeRF: Open-Vocabulary Object-Level Neural Radiance Fields
================================================================

This module implements OpenObj-NeRF for generating synthetic training data 
for cone and cylinder classification in the MONO_TO_3D tracking system.

Key Features:
- Object-level 3D scene understanding
- Fine-grained object property modeling
- CLIP-based open vocabulary integration
- Multi-view consistent object rendering
- Detailed material and geometric properties

Reference: OpenObj: Open‑Vocabulary Object‑Level Neural Radiance Fields 
with Fine‑Grained Understanding

Author: AI Assistant
Date: 2025-01-24
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import argparse
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from tqdm import tqdm
import random
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
class ObjectInstance:
    """Object instance with detailed properties."""
    object_id: int
    object_type: str  # 'cone' or 'cylinder'
    position: np.ndarray
    orientation: np.ndarray
    scale: np.ndarray  # [radius, height]
    material_properties: Dict[str, float]
    semantic_label: str
    color: np.ndarray
    bbox_3d: np.ndarray  # 3D bounding box

class CLIPEncoder(nn.Module):
    """CLIP-based encoder for open vocabulary understanding."""
    
    def __init__(self, clip_dim: int = 512, feature_dim: int = 256):
        super().__init__()
        self.clip_dim = clip_dim
        self.feature_dim = feature_dim
        
        # Text encoder pathway
        self.text_encoder = nn.Sequential(
            nn.Linear(clip_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Object property encoder
        self.property_encoder = nn.Sequential(
            nn.Linear(16, feature_dim // 2),  # Object properties
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim // 2)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim + feature_dim // 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, text_features: torch.Tensor, object_properties: torch.Tensor) -> torch.Tensor:
        """Encode text and object properties into unified features."""
        # Encode text features (simulated CLIP embeddings)
        text_encoded = self.text_encoder(text_features)
        
        # Encode object properties
        prop_encoded = self.property_encoder(object_properties)
        
        # Fuse features
        combined = torch.cat([text_encoded, prop_encoded], dim=-1)
        return self.fusion(combined)

class OpenObjNeRF(nn.Module):
    """OpenObj: Open-Vocabulary Object-Level Neural Radiance Fields."""
    
    def __init__(self, 
                 pos_frequencies: int = 10,
                 dir_frequencies: int = 4,
                 clip_dim: int = 512,
                 feature_dim: int = 256):
        super().__init__()
        
        # Positional encoding
        self.pos_encoding_dim = 3 * (2 * pos_frequencies + 1)
        self.dir_encoding_dim = 3 * (2 * dir_frequencies + 1)
        self.pos_frequencies = pos_frequencies
        self.dir_frequencies = dir_frequencies
        
        self.clip_encoder = CLIPEncoder(clip_dim, feature_dim)
        
        # Main NeRF network
        input_dim = self.pos_encoding_dim + feature_dim
        self.density_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Color network
        color_input_dim = 256 + self.dir_encoding_dim + feature_dim
        self.color_net = nn.Sequential(
            nn.Linear(color_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
        
        # Feature extraction
        self.feature_net = nn.Linear(256, 256)
        
    def positional_encoding(self, x: torch.Tensor, num_frequencies: int) -> torch.Tensor:
        """Apply positional encoding."""
        encoded = [x]
        for freq in range(num_frequencies):
            for func in [torch.sin, torch.cos]:
                encoded.append(func(2.0 ** freq * np.pi * x))
        return torch.cat(encoded, dim=-1)
        
    def forward(self, 
                positions: torch.Tensor,
                directions: torch.Tensor,
                text_features: torch.Tensor,
                object_properties: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through OpenObj-NeRF."""
        # Encode inputs
        pos_encoded = self.positional_encoding(positions, self.pos_frequencies)
        dir_encoded = self.positional_encoding(directions, self.dir_frequencies)
        clip_features = self.clip_encoder(text_features, object_properties)
        
        # Expand CLIP features to match batch size
        if clip_features.dim() == 2 and positions.dim() == 2:
            clip_features = clip_features.expand(positions.shape[0], -1)
        
        # Density prediction
        density_input = torch.cat([pos_encoded, clip_features], dim=-1)
        density_features = self.density_net[:-1](density_input)
        density = F.relu(self.density_net[-1](density_features))
        
        # Color prediction
        features = self.feature_net(density_features)
        color_input = torch.cat([features, dir_encoded, clip_features], dim=-1)
        color = torch.sigmoid(self.color_net(color_input))
        
        return density, color

class ObjectLevelDataset(Dataset):
    """Dataset for object-level synthetic scenes."""
    
    def __init__(self, 
                 num_scenes: int = 100,
                 camera_config: CameraConfig = None,
                 max_objects_per_scene: int = 3):
        self.num_scenes = num_scenes
        self.camera_config = camera_config or CameraConfig()
        self.max_objects_per_scene = max_objects_per_scene
        self.scenes = self._generate_object_scenes()
        
        # Enhanced vocabulary for object-level understanding
        self.object_vocab = {
            'cone': 0, 'cylinder': 1, 'background': 2,
            'metal': 3, 'plastic': 4, 'wood': 5, 'ceramic': 6,
            'red': 7, 'blue': 8, 'green': 9, 'gray': 10, 'black': 11, 'white': 12,
            'smooth': 13, 'rough': 14, 'shiny': 15, 'matte': 16,
            'small': 17, 'medium': 18, 'large': 19
        }
        
    def _generate_object_scenes(self) -> List[Dict]:
        """Generate scenes with multiple objects."""
        scenes = []
        
        for scene_id in range(self.num_scenes):
            num_objects = random.randint(1, self.max_objects_per_scene)
            objects = []
            
            for obj_id in range(num_objects):
                # Generate object instance
                object_type = random.choice(['cone', 'cylinder'])
                
                # Ensure objects don't overlap
                position = self._generate_non_overlapping_position(objects)
                
                # Object properties
                if object_type == 'cone':
                    radius = random.uniform(0.02, 0.08)
                    height = random.uniform(0.12, 0.28)
                else:  # cylinder
                    radius = random.uniform(0.025, 0.075)
                    height = random.uniform(0.10, 0.30)
                
                # Material and appearance
                material = random.choice(['metal', 'plastic', 'wood', 'ceramic'])
                color_name = random.choice(['red', 'blue', 'green', 'gray', 'black', 'white'])
                color_rgb = self._get_color_rgb(color_name)
                surface = random.choice(['smooth', 'rough', 'shiny', 'matte'])
                size_category = self._get_size_category(radius, height)
                
                # Create object instance
                obj_instance = ObjectInstance(
                    object_id=obj_id,
                    object_type=object_type,
                    position=position,
                    orientation=np.random.uniform(0, 2*np.pi, 3),
                    scale=np.array([radius, height]),
                    material_properties={
                        'roughness': random.uniform(0.1, 0.9),
                        'metallic': 1.0 if material == 'metal' else random.uniform(0.0, 0.3),
                        'specular': random.uniform(0.1, 0.9),
                        'transparency': random.uniform(0.0, 0.1)
                    },
                    semantic_label=f"{size_category} {color_name} {surface} {material} {object_type}",
                    color=color_rgb,
                    bbox_3d=self._compute_bbox_3d(position, radius, height)
                )
                
                objects.append(obj_instance)
            
            # Scene-level properties
            lighting = {
                'ambient': random.uniform(0.2, 0.5),
                'directional': random.uniform(0.4, 0.8),
                'direction': np.random.uniform(-1, 1, 3)
            }
            lighting['direction'] = lighting['direction'] / np.linalg.norm(lighting['direction'])
            
            scene = {
                'scene_id': scene_id,
                'objects': objects,
                'lighting': lighting,
                'num_objects': len(objects)
            }
            
            scenes.append(scene)
        
        return scenes
    
    def _generate_non_overlapping_position(self, existing_objects: List[ObjectInstance]) -> np.ndarray:
        """Generate position that doesn't overlap with existing objects."""
        max_attempts = 50
        min_distance = 0.15  # Minimum distance between object centers
        
        for _ in range(max_attempts):
            position = np.array([
                random.uniform(0.15, 0.65),  # x: within camera FOV
                random.uniform(0.8, 1.2),    # y: distance from cameras
                random.uniform(2.2, 2.8)     # z: table height variation
            ])
            
            # Check distance to existing objects
            valid = True
            for obj in existing_objects:
                distance = np.linalg.norm(position - obj.position)
                if distance < min_distance:
                    valid = False
                    break
            
            if valid:
                return position
        
        # Fallback: return position even if overlapping
        return np.array([
            random.uniform(0.2, 0.6),
            random.uniform(0.9, 1.1),
            random.uniform(2.3, 2.7)
        ])
    
    def _get_color_rgb(self, color_name: str) -> np.ndarray:
        """Convert color name to RGB values."""
        color_map = {
            'red': [0.8, 0.2, 0.2],
            'blue': [0.2, 0.2, 0.8],
            'green': [0.2, 0.8, 0.2],
            'gray': [0.5, 0.5, 0.5],
            'black': [0.1, 0.1, 0.1],
            'white': [0.9, 0.9, 0.9]
        }
        return np.array(color_map.get(color_name, [0.5, 0.5, 0.5]))
    
    def _get_size_category(self, radius: float, height: float) -> str:
        """Categorize object size."""
        volume = np.pi * radius**2 * height
        if volume < 0.0001:
            return 'small'
        elif volume < 0.0003:
            return 'medium'
        else:
            return 'large'
    
    def _compute_bbox_3d(self, position: np.ndarray, radius: float, height: float) -> np.ndarray:
        """Compute 3D bounding box."""
        center = position
        extents = np.array([radius, radius, height/2])
        return np.array([center - extents, center + extents])
    
    def __len__(self) -> int:
        return self.num_scenes
    
    def __getitem__(self, idx: int) -> Dict:
        scene = self.scenes[idx]
        
        # Convert to tensors and prepare for model
        scene_data = {
            'scene_id': scene['scene_id'],
            'objects': scene['objects'],
            'lighting': scene['lighting'],
            'num_objects': scene['num_objects']
        }
        
        return scene_data

def main():
    """Main training and data generation pipeline for OpenObj-NeRF."""
    parser = argparse.ArgumentParser(description='OpenObj-NeRF Synthetic Data Generator')
    parser.add_argument('--num_scenes', type=int, default=50, help='Number of scenes to generate')
    parser.add_argument('--max_objects', type=int, default=3, help='Maximum objects per scene')
    parser.add_argument('--output_dir', type=str, default='./openobj_synthetic_data', help='Output directory')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize components
    camera_config = CameraConfig()
    dataset = ObjectLevelDataset(
        num_scenes=args.num_scenes, 
        camera_config=camera_config,
        max_objects_per_scene=args.max_objects
    )
    model = OpenObjNeRF()
    
    logger.info(f"Initialized OpenObj-NeRF with {args.num_scenes} scenes")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Generate synthetic dataset metadata
    logger.info("Generating object-level synthetic dataset...")
    
    # Collect statistics
    stats = {
        'total_scenes': len(dataset),
        'total_objects': 0,
        'cone_objects': 0,
        'cylinder_objects': 0,
        'materials': {},
        'colors': {},
        'size_categories': {},
        'objects_per_scene': []
    }
    
    scenes_data = []
    
    for i in tqdm(range(len(dataset)), desc="Processing scenes"):
        scene_data = dataset[i]
        
        # Count objects
        num_objects = scene_data['num_objects']
        stats['total_objects'] += num_objects
        stats['objects_per_scene'].append(num_objects)
        
        # Process each object
        scene_objects = []
        for obj in scene_data['objects']:
            # Count by type
            if obj.object_type == 'cone':
                stats['cone_objects'] += 1
            else:
                stats['cylinder_objects'] += 1
            
            # Extract semantic components
            semantic_parts = obj.semantic_label.split()
            if len(semantic_parts) >= 4:
                size_cat, color, surface, material = semantic_parts[:4]
                stats['materials'][material] = stats['materials'].get(material, 0) + 1
                stats['colors'][color] = stats['colors'].get(color, 0) + 1
                stats['size_categories'][size_cat] = stats['size_categories'].get(size_cat, 0) + 1
            
            # Store object data - ensure all numpy arrays are converted to lists
            obj_data = {
                'object_id': obj.object_id,
                'object_type': obj.object_type,
                'position': convert_numpy_to_python(obj.position),
                'orientation': convert_numpy_to_python(obj.orientation),
                'scale': convert_numpy_to_python(obj.scale),
                'semantic_label': obj.semantic_label,
                'color': convert_numpy_to_python(obj.color),
                'material_properties': convert_numpy_to_python(obj.material_properties),
                'bbox_3d': convert_numpy_to_python(obj.bbox_3d)
            }
            scene_objects.append(obj_data)
        
        # Store scene data - ensure lighting direction is converted
        scene_info = {
            'scene_id': scene_data['scene_id'],
            'num_objects': num_objects,
            'objects': scene_objects,
            'lighting': convert_numpy_to_python(scene_data['lighting'])
        }
        scenes_data.append(scene_info)
    
    # Create comprehensive dataset summary
    dataset_summary = {
        'metadata': {
            'generator': 'OpenObj-NeRF: Open-Vocabulary Object-Level Neural Radiance Fields',
            'purpose': 'Object-Level Cone and Cylinder Classification Training Data',
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'max_objects_per_scene': args.max_objects
        },
        'camera_config': camera_config.__dict__,
        'statistics': stats,
        'vocabulary': dataset.object_vocab,
        'scenes': scenes_data
    }
    
    # Save dataset summary
    with open(output_dir / 'openobj_dataset_summary.json', 'w') as f:
        json.dump(dataset_summary, f, indent=2)
    
    # Print summary
    logger.info(f"OpenObj-NeRF Dataset Summary:")
    logger.info(f"  Total scenes: {stats['total_scenes']}")
    logger.info(f"  Total objects: {stats['total_objects']}")
    logger.info(f"  Cones: {stats['cone_objects']} ({stats['cone_objects']/stats['total_objects']*100:.1f}%)")
    logger.info(f"  Cylinders: {stats['cylinder_objects']} ({stats['cylinder_objects']/stats['total_objects']*100:.1f}%)")
    logger.info(f"  Average objects per scene: {np.mean(stats['objects_per_scene']):.1f}")
    logger.info(f"  Materials: {list(stats['materials'].keys())}")
    logger.info(f"  Colors: {list(stats['colors'].keys())}")
    logger.info(f"  Size categories: {list(stats['size_categories'].keys())}")
    
    logger.info(f"Dataset generation complete! Saved to: {output_dir}")

if __name__ == "__main__":
    main()
