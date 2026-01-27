#!/usr/bin/env python3
"""
Unit Tests for OpenObj-NeRF: Open-Vocabulary Object-Level Neural Radiance Fields
===============================================================================

Comprehensive test suite covering all major components of the OpenObj-NeRF
implementation with focus on correctness, edge cases, and integration.

Author: AI Assistant
Date: 2025-01-24
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
import json
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from openobj_nerf_generator import (
    OpenObjNeRF, ObjectLevelDataset, CameraConfig, 
    ObjectInstance, CLIPEncoder
)

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

class TestCameraConfig(unittest.TestCase):
    """Test camera configuration functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.camera_config = CameraConfig()
    
    def test_default_camera_config(self):
        """Test default camera configuration values."""
        self.assertEqual(self.camera_config.fx, 800.0)
        self.assertEqual(self.camera_config.fy, 800.0)
        self.assertEqual(self.camera_config.baseline, 0.65)
        self.assertEqual(self.camera_config.height, 2.55)
        self.assertEqual(self.camera_config.image_width, 640)
        self.assertEqual(self.camera_config.image_height, 480)
    
    def test_camera_config_serialization(self):
        """Test camera configuration can be serialized to JSON."""
        config_dict = self.camera_config.__dict__
        json_str = json.dumps(config_dict)
        loaded_config = json.loads(json_str)
        
        self.assertEqual(loaded_config['fx'], 800.0)
        self.assertEqual(loaded_config['baseline'], 0.65)

class TestCLIPEncoder(unittest.TestCase):
    """Test CLIP encoder functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.clip_encoder = CLIPEncoder(clip_dim=512, feature_dim=256)
        self.batch_size = 4
        self.text_features = torch.randn(self.batch_size, 512)
        self.object_properties = torch.randn(self.batch_size, 16)
    
    def test_clip_encoder_forward(self):
        """Test CLIP encoder forward pass."""
        with torch.no_grad():
            output = self.clip_encoder(self.text_features, self.object_properties)
        
        self.assertEqual(output.shape, (self.batch_size, 256))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

class TestOpenObjNeRF(unittest.TestCase):
    """Test main OpenObj-NeRF model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = OpenObjNeRF()
        self.batch_size = 100
        self.positions = torch.randn(self.batch_size, 3)
        self.directions = torch.randn(self.batch_size, 3)
        self.text_features = torch.randn(1, 512)
        self.object_properties = torch.randn(1, 16)
    
    def test_model_forward_pass(self):
        """Test complete forward pass."""
        with torch.no_grad():
            density, color = self.model(
                self.positions, self.directions,
                self.text_features, self.object_properties
            )
        
        self.assertEqual(density.shape, (self.batch_size, 1))
        self.assertEqual(color.shape, (self.batch_size, 3))
        
        # Check output ranges
        self.assertTrue((density >= 0).all())  # ReLU applied
        self.assertTrue((color >= 0).all() and (color <= 1).all())  # Sigmoid applied
    
    def test_model_parameter_count(self):
        """Test model has expected number of parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(total_params, 900000)
        self.assertLess(total_params, 1100000)

class TestObjectLevelDataset(unittest.TestCase):
    """Test object-level dataset generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.camera_config = CameraConfig()
        self.dataset = ObjectLevelDataset(
            num_scenes=5,
            camera_config=self.camera_config,
            max_objects_per_scene=2
        )
    
    def test_dataset_initialization(self):
        """Test dataset initializes correctly."""
        self.assertEqual(len(self.dataset), 5)
        self.assertEqual(self.dataset.max_objects_per_scene, 2)
        self.assertIsNotNone(self.dataset.object_vocab)
    
    def test_scene_generation(self):
        """Test scene generation produces valid scenes."""
        for i in range(len(self.dataset)):
            scene_data = self.dataset[i]
            
            self.assertIn('scene_id', scene_data)
            self.assertIn('objects', scene_data)
            self.assertIn('lighting', scene_data)
            self.assertIn('num_objects', scene_data)
            
            # Check object count is within bounds
            self.assertGreaterEqual(scene_data['num_objects'], 1)
            self.assertLessEqual(scene_data['num_objects'], 2)
    
    def test_object_types(self):
        """Test objects have valid types."""
        scene_data = self.dataset[0]
        
        for obj in scene_data['objects']:
            self.assertIn(obj.object_type, ['cone', 'cylinder'])

class TestJSONSerialization(unittest.TestCase):
    """Test JSON serialization functionality."""
    
    def test_convert_numpy_to_python_basic_types(self):
        """Test conversion of basic numpy types."""
        # Test numpy arrays
        np_array = np.array([1, 2, 3])
        result = convert_numpy_to_python(np_array)
        self.assertEqual(result, [1, 2, 3])
        
        # Test numpy scalars
        np_int = np.int32(42)
        result = convert_numpy_to_python(np_int)
        self.assertEqual(result, 42)
        self.assertIsInstance(result, int)
    
    def test_json_serialization_complete_dataset(self):
        """Test complete dataset can be serialized to JSON."""
        dataset = ObjectLevelDataset(num_scenes=2, max_objects_per_scene=1)
        
        # Create a dataset summary
        dataset_summary = {
            'metadata': {
                'generator': 'Test OpenObj-NeRF',
                'total_scenes': len(dataset)
            },
            'scenes': []
        }
        
        for i in range(len(dataset)):
            scene_data = dataset[i]
            scene_objects = []
            
            for obj in scene_data['objects']:
                obj_data = {
                    'object_id': obj.object_id,
                    'object_type': obj.object_type,
                    'position': convert_numpy_to_python(obj.position),
                    'scale': convert_numpy_to_python(obj.scale),
                    'color': convert_numpy_to_python(obj.color)
                }
                scene_objects.append(obj_data)
            
            scene_info = {
                'scene_id': scene_data['scene_id'],
                'objects': scene_objects
            }
            dataset_summary['scenes'].append(scene_info)
        
        # Test JSON serialization
        json_str = json.dumps(dataset_summary, indent=2)
        loaded_data = json.loads(json_str)
        
        self.assertEqual(loaded_data['metadata']['total_scenes'], 2)
        self.assertEqual(len(loaded_data['scenes']), 2)

class TestIntegration(unittest.TestCase):
    """Integration tests for complete pipeline."""
    
    def test_complete_pipeline(self):
        """Test complete pipeline from dataset to model inference."""
        # Create dataset
        dataset = ObjectLevelDataset(num_scenes=2, max_objects_per_scene=1)
        
        # Create model
        model = OpenObjNeRF()
        
        # Test inference on dataset scenes
        scene_data = dataset[0]
        
        # Simulate ray sampling
        num_rays = 10
        positions = torch.randn(num_rays, 3)
        directions = torch.randn(num_rays, 3)
        text_features = torch.randn(1, 512)
        object_properties = torch.randn(1, 16)
        
        # Test inference
        with torch.no_grad():
            density, color = model(positions, directions, text_features, object_properties)
        
        self.assertEqual(density.shape, (num_rays, 1))
        self.assertEqual(color.shape, (num_rays, 3))

def run_openobj_nerf_tests():
    """Run all OpenObj-NeRF tests with detailed reporting."""
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestCameraConfig,
        TestCLIPEncoder,
        TestOpenObjNeRF,
        TestObjectLevelDataset,
        TestJSONSerialization,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print("OpenObj-NeRF Test Summary")
    print(f"{'='*60}")
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    if result.testsRun > 0:
        success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100)
        print(f"Success Rate: {success_rate:.1f}%")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_openobj_nerf_tests()
    sys.exit(0 if success else 1)
