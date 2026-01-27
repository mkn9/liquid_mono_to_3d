#!/usr/bin/env python3
"""
Comprehensive Test Suite for Semantic-NeRF Implementation
========================================================

Tests for the Semantic-NeRF implementation adapted for MONO_TO_3D project.
Covers all components including model architecture, dataset generation,
label propagation, volume rendering, and MONO_TO_3D integration.

Author: AI Assistant
Date: 2024-06-24
"""

import unittest
import torch
import numpy as np
import json
import tempfile
from pathlib import Path

# Import our Semantic-NeRF implementation
from semantic_nerf_generator import (
    SemanticNeRF, SemanticDataset, CameraConfig, LabelPropagator,
    VolumeRenderer, PositionalEncoder, SemanticMLP, DensityMLP,
    ColorMLP, convert_numpy_to_python, create_semantic_nerf_demo
)


class TestCameraConfig(unittest.TestCase):
    """Test camera configuration for MONO_TO_3D compatibility."""
    
    def test_default_config(self):
        """Test default camera configuration values."""
        config = CameraConfig()
        
        # Check default values
        self.assertEqual(config.fx, 800.0)
        self.assertEqual(config.fy, 800.0)
        self.assertEqual(config.cx, 320.0)
        self.assertEqual(config.cy, 240.0)
        self.assertEqual(config.baseline, 0.65)
        self.assertEqual(config.height, 2.55)
        self.assertEqual(config.image_width, 640)
        self.assertEqual(config.image_height, 480)
    
    def test_custom_config(self):
        """Test custom camera configuration."""
        config = CameraConfig(
            fx=1000.0, fy=1000.0,
            cx=400.0, cy=300.0,
            baseline=0.7, height=3.0,
            image_width=800, image_height=600
        )
        
        self.assertEqual(config.fx, 1000.0)
        self.assertEqual(config.fy, 1000.0)
        self.assertEqual(config.cx, 400.0)
        self.assertEqual(config.cy, 300.0)
        self.assertEqual(config.baseline, 0.7)
        self.assertEqual(config.height, 3.0)
        self.assertEqual(config.image_width, 800)
        self.assertEqual(config.image_height, 600)


class TestPositionalEncoder(unittest.TestCase):
    """Test positional encoding module."""
    
    def test_encoder_dimensions(self):
        """Test positional encoder output dimensions."""
        encoder = PositionalEncoder(input_dim=3, num_frequencies=10)
        
        # Check output dimension calculation
        expected_dim = 3 * (2 * 10 + 1)  # input_dim * (2 * num_frequencies + 1)
        self.assertEqual(encoder.output_dim, expected_dim)
        
        # Test forward pass
        input_tensor = torch.randn(100, 3)
        output = encoder(input_tensor)
        
        self.assertEqual(output.shape, (100, expected_dim))
    
    def test_encoder_different_configs(self):
        """Test encoder with different configurations."""
        configs = [
            (2, 5),   # 2D input, 5 frequencies
            (3, 8),   # 3D input, 8 frequencies  
            (4, 12),  # 4D input, 12 frequencies
        ]
        
        for input_dim, num_freq in configs:
            encoder = PositionalEncoder(input_dim, num_freq)
            expected_dim = input_dim * (2 * num_freq + 1)
            
            self.assertEqual(encoder.output_dim, expected_dim)
            
            # Test forward pass
            input_tensor = torch.randn(50, input_dim)
            output = encoder(input_tensor)
            self.assertEqual(output.shape, (50, expected_dim))


class TestMLPModules(unittest.TestCase):
    """Test MLP modules for semantic, density, and color prediction."""
    
    def test_semantic_mlp(self):
        """Test semantic MLP."""
        mlp = SemanticMLP(input_dim=128, hidden_dim=256, num_classes=4)
        
        # Test forward pass
        input_tensor = torch.randn(100, 128)
        output = mlp(input_tensor)
        
        self.assertEqual(output.shape, (100, 4))
        self.assertEqual(mlp.num_classes, 4)
    
    def test_density_mlp(self):
        """Test density MLP."""
        mlp = DensityMLP(input_dim=128, hidden_dim=256)
        
        # Test forward pass
        input_tensor = torch.randn(100, 128)
        output = mlp(input_tensor)
        
        self.assertEqual(output.shape, (100, 1))
        # Check that output is non-negative (due to ReLU)
        self.assertTrue(torch.all(output >= 0))
    
    def test_color_mlp(self):
        """Test color MLP."""
        mlp = ColorMLP(input_dim=128, hidden_dim=256)
        
        # Test forward pass
        input_tensor = torch.randn(100, 128)
        output = mlp(input_tensor)
        
        self.assertEqual(output.shape, (100, 3))
        # Check that output is in [0, 1] range (due to sigmoid)
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))


class TestSemanticNeRF(unittest.TestCase):
    """Test main Semantic-NeRF model."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = SemanticNeRF(
            pos_frequencies=10,
            dir_frequencies=4,
            hidden_dim=256,
            num_classes=4,
            enable_label_propagation=True,
            enable_denoising=True
        )
        
        self.assertEqual(model.num_classes, 4)
        self.assertTrue(model.enable_label_propagation)
        self.assertTrue(model.enable_denoising)
        
        # Check that model has required components
        self.assertIsNotNone(model.pos_encoder)
        self.assertIsNotNone(model.dir_encoder)
        self.assertIsNotNone(model.density_mlp)
        self.assertIsNotNone(model.color_mlp)
        self.assertIsNotNone(model.semantic_mlp)
        self.assertIsNotNone(model.consistency_mlp)
        self.assertIsNotNone(model.denoising_mlp)
    
    def test_model_forward_pass(self):
        """Test model forward pass."""
        model = SemanticNeRF(num_classes=4)
        
        batch_size = 100
        positions = torch.randn(batch_size, 3)
        directions = torch.randn(batch_size, 3)
        directions = torch.nn.functional.normalize(directions, p=2, dim=-1)
        
        # Forward pass without semantic context
        density, color, semantics = model(positions, directions)
        
        self.assertEqual(density.shape, (batch_size, 1))
        self.assertEqual(color.shape, (batch_size, 3))
        self.assertEqual(semantics.shape, (batch_size, 4))
        
        # Check output ranges
        self.assertTrue(torch.all(density >= 0))
        self.assertTrue(torch.all(color >= 0) and torch.all(color <= 1))
    
    def test_model_with_semantic_context(self):
        """Test model with semantic context."""
        model = SemanticNeRF(
            num_classes=4,
            enable_label_propagation=True,
            enable_denoising=True
        )
        
        batch_size = 100
        positions = torch.randn(batch_size, 3)
        directions = torch.randn(batch_size, 3)
        directions = torch.nn.functional.normalize(directions, p=2, dim=-1)
        
        # Create semantic context
        semantic_context = torch.zeros(batch_size, 4)
        semantic_context[torch.arange(batch_size), torch.randint(0, 4, (batch_size,))] = 1.0
        
        # Forward pass with semantic context
        density, color, semantics = model(positions, directions, semantic_context)
        
        self.assertEqual(density.shape, (batch_size, 1))
        self.assertEqual(color.shape, (batch_size, 3))
        self.assertEqual(semantics.shape, (batch_size, 4))
    
    def test_parameter_count(self):
        """Test parameter count is reasonable."""
        model = SemanticNeRF()
        total_params = sum(p.numel() for p in model.parameters())
        
        # Should be reasonable number of parameters (not too small or too large)
        self.assertGreater(total_params, 100000)  # At least 100K parameters
        self.assertLess(total_params, 10000000)   # Less than 10M parameters


class TestSemanticDataset(unittest.TestCase):
    """Test semantic dataset generation."""
    
    def test_dataset_creation(self):
        """Test dataset creation."""
        dataset = SemanticDataset(num_scenes=5)
        
        self.assertEqual(len(dataset), 5)
        self.assertIsNotNone(dataset.camera_config)
        self.assertEqual(len(dataset.semantic_classes), 4)
        
        # Check semantic classes
        expected_classes = {'background', 'cone', 'cylinder', 'ground'}
        actual_classes = set(dataset.semantic_classes.values())
        self.assertEqual(actual_classes, expected_classes)
    
    def test_dataset_with_parameters(self):
        """Test dataset creation with custom parameters."""
        dataset = SemanticDataset(
            num_scenes=3,
            max_objects_per_scene=2,
            views_per_scene=5,
            sparse_labels_per_object=20
        )
        
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.max_objects_per_scene, 2)
        self.assertEqual(dataset.views_per_scene, 5)
        self.assertEqual(dataset.sparse_labels_per_object, 20)
    
    def test_scene_structure(self):
        """Test scene data structure."""
        dataset = SemanticDataset(num_scenes=3)
        sample = dataset[0]
        
        # Check required keys for __getitem__ output
        required_keys = {'scene_id', 'image', 'sparse_labels', 'dense_labels', 'camera_pose', 'objects', 'views', 'camera_config'}
        self.assertEqual(set(sample.keys()), required_keys)
        
        # Check tensor shapes
        self.assertEqual(sample['image'].shape, (3, 480, 640))  # C, H, W
        self.assertEqual(sample['sparse_labels'].shape, (480, 640))  # H, W
        self.assertEqual(sample['dense_labels'].shape, (480, 640))  # H, W
        self.assertEqual(sample['camera_pose'].shape, (4, 4))  # 4x4 transformation matrix
        
        # Check objects structure
        self.assertIsInstance(sample['objects'], list)
        self.assertGreater(len(sample['objects']), 0)
        
        for obj in sample['objects']:
            obj_keys = {'object_id', 'object_type', 'position', 'scale', 'color', 'semantic_id'}
            self.assertEqual(set(obj.keys()), obj_keys)
            self.assertIn(obj['object_type'], ['cone', 'cylinder'])
            self.assertIn(obj['semantic_id'], [1, 2])  # cone=1, cylinder=2
    
    def test_sparse_labels(self):
        """Test sparse label generation."""
        dataset = SemanticDataset(num_scenes=2, sparse_ratio=0.8)
        sample = dataset[0]
        
        # Check that sparse_labels is now a tensor (from __getitem__)
        sparse_labels = sample['sparse_labels']
        self.assertIsInstance(sparse_labels, torch.Tensor)
        self.assertEqual(sparse_labels.shape, (480, 640))  # H, W
        
        # Check that most pixels are unlabeled (-1) due to sparsity
        unlabeled_count = torch.sum(sparse_labels == -1).item()
        total_pixels = sparse_labels.numel()
        sparsity = unlabeled_count / total_pixels
        self.assertGreater(sparsity, 0.5)  # Should be quite sparse
        
        # Check that some pixels are labeled with valid semantic IDs
        labeled_mask = sparse_labels != -1
        if torch.sum(labeled_mask) > 0:
            labeled_values = sparse_labels[labeled_mask]
            self.assertTrue(torch.all(labeled_values >= 0))
            self.assertTrue(torch.all(labeled_values <= 3))  # 0-3 for our 4 classes
    
    def test_multiview_data(self):
        """Test multi-view data generation."""
        dataset = SemanticDataset(num_scenes=2)
        scene = dataset[0]
        
        views = scene['views']
        self.assertIsInstance(views, list)
        self.assertGreater(len(views), 0)
        
        for view in views:
            view_keys = {'view_id', 'camera_pos', 'camera_rot', 'projected_objects'}
            self.assertEqual(set(view.keys()), view_keys)
            
            # Check camera position
            self.assertEqual(len(view['camera_pos']), 3)
            self.assertEqual(len(view['camera_rot']), 3)


class TestLabelPropagator(unittest.TestCase):
    """Test label propagation functionality."""
    
    def test_propagator_initialization(self):
        """Test propagator initialization."""
        propagator = LabelPropagator(
            spatial_sigma=0.1, 
            semantic_sigma=0.2,
            color_sigma=0.05,
            max_iterations=100
        )
        
        self.assertEqual(propagator.spatial_sigma, 0.1)
        self.assertEqual(propagator.semantic_sigma, 0.2)
        self.assertEqual(propagator.color_sigma, 0.05)
        self.assertEqual(propagator.max_iterations, 100)
    
    def test_label_propagation_3d(self):
        """Test 3D label propagation functionality."""
        propagator = LabelPropagator()
        
        # Create test data
        num_sparse = 10
        num_classes = 4
        feature_dim = 64
        
        sparse_labels = torch.zeros(num_sparse, num_classes)
        sparse_labels[torch.arange(num_sparse), torch.randint(0, num_classes, (num_sparse,))] = 1.0
        
        positions = torch.randn(num_sparse, 3)
        features = torch.randn(num_sparse, feature_dim)
        
        # Propagate labels
        dense_labels = propagator.propagate_labels(sparse_labels, positions, features)
        
        self.assertEqual(dense_labels.shape, (num_sparse, num_classes))
        
        # Check that output is properly normalized (sums to 1)
        label_sums = torch.sum(dense_labels, dim=-1)
        torch.testing.assert_close(label_sums, torch.ones(num_sparse), atol=1e-6, rtol=1e-6)
    
    def test_label_propagation_2d(self):
        """Test 2D image label propagation."""
        propagator = LabelPropagator(max_iterations=5)
        
        # Create test image and sparse labels
        image = torch.randn(3, 32, 32)  # Small test image
        sparse_labels = torch.full((32, 32), -1, dtype=torch.long)  # All unlabeled initially
        
        # Add some sparse labels
        sparse_labels[10:15, 10:15] = 1  # Cone region
        sparse_labels[20:25, 20:25] = 2  # Cylinder region
        
        # Propagate labels
        dense_labels = propagator.propagate_labels(image, sparse_labels)
        
        # Check output shape
        self.assertEqual(dense_labels.shape, (32, 32))
        
        # Check that original labels are preserved
        self.assertTrue(torch.all(dense_labels[10:15, 10:15] == 1))
        self.assertTrue(torch.all(dense_labels[20:25, 20:25] == 2))


class TestVolumeRenderer(unittest.TestCase):
    """Test volume rendering functionality."""
    
    def test_ray_rendering(self):
        """Test ray rendering."""
        renderer = VolumeRenderer()
        
        # Create test data
        num_rays = 50
        num_samples = 32
        num_classes = 4
        
        density = torch.rand(num_rays, num_samples, 1)
        color = torch.rand(num_rays, num_samples, 3)
        semantics = torch.randn(num_rays, num_samples, num_classes)
        t_vals = torch.linspace(0.1, 2.0, num_samples).expand(num_rays, -1)
        
        # Render rays
        rgb, depth, semantic_map, weights = renderer.render_rays(
            density, color, semantics, t_vals
        )
        
        self.assertEqual(rgb.shape, (num_rays, 3))
        self.assertEqual(depth.shape, (num_rays, 1))
        self.assertEqual(semantic_map.shape, (num_rays, num_classes))
        self.assertEqual(weights.shape, (num_rays, num_samples))
        
        # Check output ranges
        self.assertTrue(torch.all(rgb >= 0) and torch.all(rgb <= 1))
        self.assertTrue(torch.all(depth >= 0))
        
        # Check that weights sum approximately to 1 (or less due to transmittance)
        weight_sums = torch.sum(weights, dim=-1)
        self.assertTrue(torch.all(weight_sums <= 1.1))  # Allow small numerical errors


class TestJSONSerialization(unittest.TestCase):
    """Test JSON serialization utilities."""
    
    def test_numpy_conversion(self):
        """Test numpy to Python conversion."""
        # Test numpy arrays
        arr = np.array([1, 2, 3])
        converted = convert_numpy_to_python(arr)
        self.assertEqual(converted, [1, 2, 3])
        
        # Test numpy scalars
        scalar_int = np.int64(42)
        scalar_float = np.float32(3.14)
        self.assertEqual(convert_numpy_to_python(scalar_int), 42)
        self.assertAlmostEqual(convert_numpy_to_python(scalar_float), 3.14, places=6)
        
        # Test nested structures
        nested = {
            'array': np.array([1, 2, 3]),
            'scalar': np.float64(2.71),
            'list': [np.int32(1), np.array([4, 5])],
            'normal': 'string'
        }
        
        converted = convert_numpy_to_python(nested)
        expected = {
            'array': [1, 2, 3],
            'scalar': 2.71,
            'list': [1, [4, 5]],
            'normal': 'string'
        }
        
        self.assertEqual(converted, expected)
    
    def test_json_serialization(self):
        """Test complete JSON serialization."""
        data = {
            'positions': np.array([[1, 2, 3], [4, 5, 6]]),
            'confidence': np.float32(0.95),
            'count': np.int64(100)
        }
        
        converted = convert_numpy_to_python(data)
        
        # Should be JSON serializable
        json_str = json.dumps(converted)
        self.assertIsInstance(json_str, str)
        
        # Should be deserializable
        deserialized = json.loads(json_str)
        expected = {
            'positions': [[1, 2, 3], [4, 5, 6]],
            'count': 100
        }
        
        # Check structure (excluding confidence due to float32 precision variations)
        self.assertEqual(deserialized['positions'], expected['positions'])
        self.assertEqual(deserialized['count'], expected['count'])
        self.assertAlmostEqual(deserialized['confidence'], 0.95, places=2)


class TestMONOTO3DIntegration(unittest.TestCase):
    """Test integration with MONO_TO_3D coordinate system."""
    
    def test_coordinate_system_compatibility(self):
        """Test coordinate system compatibility."""
        dataset = SemanticDataset(num_scenes=3)
        camera_config = dataset.camera_config
        
        # Test coordinate ranges
        for scene in dataset.scenes:
            for obj in scene['objects']:
                pos = obj['position']
                
                # Check X coordinate (stereo baseline constraint)
                self.assertGreaterEqual(pos[0], 0.0)
                self.assertLessEqual(pos[0], 0.5)
                
                # Check Y coordinate (forward distance)
                self.assertGreaterEqual(pos[1], 0.1)
                self.assertLessEqual(pos[1], 0.5)
                
                # Check Z coordinate (height above ground)
                self.assertGreaterEqual(pos[2], 2.0)
                self.assertLessEqual(pos[2], 2.6)
    
    def test_stereo_projection(self):
        """Test stereo camera projection."""
        camera_config = CameraConfig()
        dataset = SemanticDataset(num_scenes=5, camera_config=camera_config)  # Test multiple scenes
        
        valid_projections = 0
        total_objects = 0
        
        for scene_data in dataset.scenes:  # Access raw scene data
            for obj in scene_data['objects']:
                total_objects += 1
                pos = obj['position']
                
                # Left camera projection
                u_left = camera_config.fx * pos[0] + camera_config.cx
                v_left = camera_config.fy * pos[1] + camera_config.cy
                
                # Right camera projection
                x_right = pos[0] - camera_config.baseline
                u_right = camera_config.fx * x_right + camera_config.cx
                v_right = camera_config.fy * pos[1] + camera_config.cy
                
                # Check if projection is within reasonable bounds (allow some tolerance)
                if (0 <= u_left < camera_config.image_width and 
                    0 <= v_left < camera_config.image_height and
                    0 <= v_right < camera_config.image_height):
                    valid_projections += 1
                    
                    # For valid projections, check disparity
                    disparity = u_left - u_right
                    self.assertGreater(disparity, 0)
        
        # Ensure at least some objects project within bounds
        self.assertGreater(total_objects, 0)
        self.assertGreater(valid_projections, 0)
        
        # At least 40% should project within bounds (allow for some objects to be outside FOV)
        projection_rate = valid_projections / total_objects
        self.assertGreaterEqual(projection_rate, 0.4)


class TestEndToEndIntegration(unittest.TestCase):
    """Test end-to-end integration."""
    
    def test_complete_pipeline(self):
        """Test complete Semantic-NeRF pipeline."""
        # Create demo
        demo_results = create_semantic_nerf_demo()
        
        # Check demo results
        self.assertIn('model', demo_results)
        self.assertIn('dataset', demo_results)
        self.assertIn('camera_config', demo_results)
        self.assertIn('propagator', demo_results)
        self.assertIn('total_parameters', demo_results)
        
        # Check model
        model = demo_results['model']
        self.assertIsInstance(model, SemanticNeRF)
        
        # Check dataset
        dataset = demo_results['dataset']
        self.assertIsInstance(dataset, SemanticDataset)
        self.assertGreater(len(dataset), 0)
        
        # Check parameter count
        total_params = demo_results['total_parameters']
        self.assertGreater(total_params, 100000)
    
    def test_training_compatibility(self):
        """Test training compatibility."""
        model = SemanticNeRF(num_classes=4)
        dataset = SemanticDataset(num_scenes=2)
        
        # Test that model can process dataset samples
        scene = dataset[0]
        objects = scene['objects']
        
        # Create sample batch
        batch_size = 10
        positions = torch.randn(batch_size, 3)
        directions = torch.randn(batch_size, 3)
        directions = torch.nn.functional.normalize(directions, p=2, dim=-1)
        
        # Forward pass
        with torch.no_grad():
            density, color, semantics = model(positions, directions)
        
        # Check outputs are valid for training
        self.assertFalse(torch.isnan(density).any())
        self.assertFalse(torch.isnan(color).any())
        self.assertFalse(torch.isnan(semantics).any())
        
        # Check gradients can be computed
        model.train()
        model.zero_grad()  # Clear any existing gradients
        
        density, color, semantics = model(positions, directions)
        
        # Dummy loss with proper tensor operations
        loss = density.sum() + color.sum() + semantics.sum()
        loss.backward()
        
        # Check that gradients exist for at least some parameters
        grad_count = 0
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                grad_count += 1
        
        # Should have gradients for at least some parameters
        self.assertGreater(grad_count, 0)


def run_semantic_nerf_tests():
    """Run all Semantic-NeRF tests."""
    print("🧪 Running Semantic-NeRF Test Suite")
    print("=" * 50)
    
    # Create test suite
    test_classes = [
        TestCameraConfig,
        TestPositionalEncoder, 
        TestMLPModules,
        TestSemanticNeRF,
        TestSemanticDataset,
        TestLabelPropagator,
        TestVolumeRenderer,
        TestJSONSerialization,
        TestMONOTO3DIntegration,
        TestEndToEndIntegration
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"🧪 Test Summary:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   {test}: {traceback}")
    
    if result.errors:
        print(f"\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"   {test}: {traceback}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun
    print(f"\n✅ Success rate: {success_rate * 100:.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_semantic_nerf_tests()
    exit(0 if success else 1)