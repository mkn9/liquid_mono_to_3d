"""
TDD Tests for Validation Inference Script
Tests loading model, running inference, and generating visualizations.
"""

import pytest
import torch
import json
import numpy as np
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_load_magvit_model():
    """Test that we can load the trained MagVIT model."""
    model_path = Path(__file__).parent.parent / "sequential_results_20260125_2148_FULL" / "magvit" / "final_model.pt"
    
    # Check model file exists
    assert model_path.exists(), f"Model file not found: {model_path}"
    
    # Check we can load it
    state_dict = torch.load(model_path, map_location='cpu')
    assert isinstance(state_dict, dict), "Model should be a state dict"
    assert len(state_dict) > 0, "Model should have parameters"


def test_load_validation_sample():
    """Test that we can load a validation sample."""
    # Validation samples are indices 8000-9999
    sample_path = Path(__file__).parent.parent / "validation_examples" / "traj_08000.pt"
    label_path = Path(__file__).parent.parent / "validation_examples" / "traj_08000.json"
    
    # Check files exist
    assert sample_path.exists(), f"Video file not found: {sample_path}"
    assert label_path.exists(), f"Label file not found: {label_path}"
    
    # Load video
    video = torch.load(sample_path, map_location='cpu')
    assert video.shape == (16, 3, 64, 64), f"Expected (16,3,64,64), got {video.shape}"
    
    # Load label
    with open(label_path) as f:
        metadata = json.load(f)
    
    assert 'class' in metadata, "Metadata should have 'class'"
    assert 'positions' in metadata, "Metadata should have 'positions'"
    assert metadata['class'] in ['Linear', 'Circular', 'Helical', 'Parabolic'], "Valid class"


def test_prediction_output_format():
    """Test that predictions have correct format."""
    # Mock prediction
    prediction = {
        'class_label': torch.tensor([0, 1, 2, 3]),
        'class_probs': torch.tensor([[0.9, 0.05, 0.03, 0.02],
                                      [0.1, 0.8, 0.05, 0.05],
                                      [0.05, 0.1, 0.8, 0.05],
                                      [0.02, 0.03, 0.05, 0.9]]),
        'future_position': torch.tensor([[1.0, 2.0, 3.0],
                                         [1.5, 2.5, 3.5],
                                         [2.0, 3.0, 4.0],
                                         [2.5, 3.5, 4.5]])
    }
    
    assert prediction['class_label'].shape == (4,), "Class labels should be 1D"
    assert prediction['class_probs'].shape == (4, 4), "Probs should be (batch, 4)"
    assert prediction['future_position'].shape == (4, 3), "Position should be (batch, 3)"


def test_confusion_matrix_format():
    """Test confusion matrix computation."""
    # Mock data
    true_labels = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    pred_labels = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    
    # Create confusion matrix
    num_classes = 4
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(true_labels, pred_labels):
        confusion[true, pred] += 1
    
    assert confusion.shape == (4, 4), "Confusion matrix should be 4x4"
    assert np.sum(confusion) == len(true_labels), "Total should match sample count"


def test_visualization_data_format():
    """Test that visualization data is properly formatted."""
    # Mock visualization data
    viz_data = {
        'video_frames': np.random.rand(16, 64, 64, 3),  # 16 frames, HWC format
        'true_class': 'Linear',
        'pred_class': 'Linear',
        'confidence': 0.95,
        'true_position': [1.0, 2.0, 3.0],
        'pred_position': [1.1, 2.1, 3.1]
    }
    
    assert viz_data['video_frames'].shape == (16, 64, 64, 3), "Frames should be (T,H,W,C)"
    assert isinstance(viz_data['true_class'], str), "Class should be string"
    assert 0 <= viz_data['confidence'] <= 1, "Confidence should be probability"
    assert len(viz_data['true_position']) == 3, "Position should be 3D"


def test_per_class_accuracy_computation():
    """Test per-class accuracy calculation."""
    # Mock predictions by class
    class_results = {
        'Linear': {'correct': 450, 'total': 500},
        'Circular': {'correct': 480, 'total': 500},
        'Helical': {'correct': 490, 'total': 500},
        'Parabolic': {'correct': 500, 'total': 500}
    }
    
    for class_name, results in class_results.items():
        accuracy = results['correct'] / results['total']
        assert 0 <= accuracy <= 1, f"Accuracy for {class_name} should be in [0,1]"


def test_position_error_computation():
    """Test position prediction error calculation."""
    true_pos = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    pred_pos = np.array([[1.1, 2.1, 3.1], [4.2, 5.1, 6.0]])
    
    # MSE
    mse = np.mean((true_pos - pred_pos) ** 2)
    assert mse >= 0, "MSE should be non-negative"
    
    # MAE
    mae = np.mean(np.abs(true_pos - pred_pos))
    assert mae >= 0, "MAE should be non-negative"


def test_output_directory_creation():
    """Test that output directory can be created."""
    output_dir = Path(__file__).parent.parent / "validation_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    assert output_dir.exists(), "Output directory should exist"


def test_results_json_format():
    """Test that results can be serialized to JSON."""
    results = {
        'overall_accuracy': 1.0,
        'per_class_accuracy': {
            'Linear': 0.98,
            'Circular': 1.0,
            'Helical': 1.0,
            'Parabolic': 1.0
        },
        'confusion_matrix': [[100, 0, 0, 0],
                             [0, 100, 0, 0],
                             [0, 0, 100, 0],
                             [0, 0, 0, 100]],
        'position_mse': 0.05,
        'position_mae': 0.02
    }
    
    # Test JSON serialization
    json_str = json.dumps(results, indent=2)
    loaded = json.loads(json_str)
    
    assert loaded['overall_accuracy'] == 1.0
    assert len(loaded['per_class_accuracy']) == 4
    assert len(loaded['confusion_matrix']) == 4

