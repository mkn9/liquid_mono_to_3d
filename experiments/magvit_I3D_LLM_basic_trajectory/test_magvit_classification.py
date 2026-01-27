#!/usr/bin/env python3
"""
TDD Tests for MAGVIT Trajectory Classification

Tests the classification of trajectory types using encoded MAGVIT representations.
This validates whether the MAGVIT encodings preserve class-discriminative information.

Test Strategy:
1. Dataset loading and encoding
2. Classifier initialization
3. Training loop functionality
4. Validation and metrics
5. Checkpoint saving/loading
6. Progress monitoring
7. Integration test with expected accuracy
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import pytest
import sys
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from train_magvit import create_model as create_magvit_model, load_dataset


class TestClassificationDataset:
    """Test dataset loading and encoding for classification."""
    
    def test_load_dataset_for_classification(self):
        """Test loading dataset and extracting labels."""
        from classify_magvit import load_classification_data
        
        dataset_path = "results/20260125_0304_dataset_200_validated.npz"
        data = load_classification_data(dataset_path)
        
        assert 'videos' in data
        assert 'labels' in data
        assert len(data['videos']) == len(data['labels'])
        assert len(data['videos']) == 200
        assert data['labels'].min() >= 0
        assert data['labels'].max() <= 3  # 4 classes
    
    def test_encode_dataset_with_magvit(self):
        """Test encoding videos into MAGVIT codes."""
        from classify_magvit import encode_dataset_to_codes
        
        dataset_path = "results/20260125_0304_dataset_200_validated.npz"
        model_checkpoint = sorted(list(Path("results/magvit_training").glob("*_best_model.pt")))[-1]
        
        codes, labels = encode_dataset_to_codes(
            dataset_path=dataset_path,
            model_checkpoint=str(model_checkpoint),
            batch_size=4
        )
        
        assert codes is not None
        assert labels is not None
        assert len(codes) == 200
        assert len(labels) == 200
        assert codes.shape[0] == 200
        # Codes should be flattened or pooled to a fixed size
        assert len(codes.shape) == 2  # (N, feature_dim)
    
    def test_train_test_split(self):
        """Test splitting dataset into train/val/test sets."""
        from classify_magvit import split_dataset
        
        # Create dummy data
        codes = torch.randn(200, 128)
        labels = torch.randint(0, 4, (200,))
        
        splits = split_dataset(codes, labels, train_ratio=0.7, val_ratio=0.15)
        
        assert 'train_codes' in splits
        assert 'val_codes' in splits
        assert 'test_codes' in splits
        assert 'train_labels' in splits
        assert 'val_labels' in splits
        assert 'test_labels' in splits
        
        # Check sizes (approximately correct due to stratification rounding)
        assert 135 <= len(splits['train_codes']) <= 145  # ~70%
        assert 25 <= len(splits['val_codes']) <= 35       # ~15%
        assert 25 <= len(splits['test_codes']) <= 35      # ~15%
        
        # Check no data leakage
        total_samples = (len(splits['train_codes']) + 
                        len(splits['val_codes']) + 
                        len(splits['test_codes']))
        assert total_samples == 200


class TestClassifierModel:
    """Test classifier model architecture."""
    
    def test_classifier_initialization(self):
        """Test creating classifier model."""
        from classify_magvit import TrajectoryClassifier
        
        input_dim = 128
        num_classes = 4
        
        classifier = TrajectoryClassifier(
            input_dim=input_dim,
            hidden_dims=[256, 128, 64],
            num_classes=num_classes,
            dropout=0.3
        )
        
        assert classifier is not None
        assert hasattr(classifier, 'forward')
        
        # Test forward pass
        dummy_input = torch.randn(8, input_dim)
        output = classifier(dummy_input)
        assert output.shape == (8, num_classes)
    
    def test_classifier_with_batch_norm(self):
        """Test classifier includes batch normalization."""
        from classify_magvit import TrajectoryClassifier
        
        classifier = TrajectoryClassifier(
            input_dim=128,
            hidden_dims=[256, 128],
            num_classes=4,
            dropout=0.3
        )
        
        # Check for BatchNorm layers
        has_batchnorm = any(isinstance(m, nn.BatchNorm1d) for m in classifier.modules())
        assert has_batchnorm, "Classifier should include batch normalization"
    
    def test_classifier_deterministic(self):
        """Test classifier produces consistent output in eval mode."""
        from classify_magvit import TrajectoryClassifier
        
        torch.manual_seed(42)
        classifier = TrajectoryClassifier(input_dim=128, hidden_dims=[256], num_classes=4, dropout=0.3)
        classifier.eval()
        
        dummy_input = torch.randn(4, 128)
        
        output1 = classifier(dummy_input)
        output2 = classifier(dummy_input)
        
        assert torch.allclose(output1, output2), "Outputs should be deterministic in eval mode"


class TestClassificationTraining:
    """Test classification training functionality."""
    
    def test_train_one_epoch(self):
        """Test single training epoch."""
        from classify_magvit import train_one_epoch, TrajectoryClassifier
        
        # Create tiny dataset
        train_codes = torch.randn(20, 128)
        train_labels = torch.randint(0, 4, (20,))
        
        classifier = TrajectoryClassifier(input_dim=128, hidden_dims=[64], num_classes=4, dropout=0.3)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        loss = train_one_epoch(
            model=classifier,
            train_codes=train_codes,
            train_labels=train_labels,
            optimizer=optimizer,
            criterion=criterion,
            batch_size=4,
            device='cpu'
        )
        
        assert isinstance(loss, float)
        assert loss > 0
        assert not np.isnan(loss)
        assert not np.isinf(loss)
    
    def test_validate_epoch(self):
        """Test validation epoch."""
        from classify_magvit import validate, TrajectoryClassifier
        
        # Create tiny dataset
        val_codes = torch.randn(20, 128)
        val_labels = torch.randint(0, 4, (20,))
        
        classifier = TrajectoryClassifier(input_dim=128, hidden_dims=[64], num_classes=4, dropout=0.3)
        criterion = nn.CrossEntropyLoss()
        
        metrics = validate(
            model=classifier,
            val_codes=val_codes,
            val_labels=val_labels,
            criterion=criterion,
            batch_size=4,
            device='cpu'
        )
        
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert 'per_class_accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        assert metrics['loss'] > 0
    
    def test_loss_decreases_over_epochs(self):
        """Test that loss decreases with training."""
        from classify_magvit import train_one_epoch, TrajectoryClassifier
        
        # Create simple dataset with clear patterns
        torch.manual_seed(42)
        train_codes = torch.randn(40, 128)
        # Make patterns separable
        train_codes[:10, :10] += 5   # Class 0
        train_codes[10:20, 10:20] += 5  # Class 1
        train_codes[20:30, 20:30] += 5  # Class 2
        train_codes[30:40, 30:40] += 5  # Class 3
        train_labels = torch.cat([torch.zeros(10), torch.ones(10), 
                                  torch.ones(10)*2, torch.ones(10)*3]).long()
        
        classifier = TrajectoryClassifier(input_dim=128, hidden_dims=[64], num_classes=4, dropout=0.1)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        losses = []
        for _ in range(10):
            loss = train_one_epoch(classifier, train_codes, train_labels, 
                                  optimizer, criterion, batch_size=8, device='cpu')
            losses.append(loss)
        
        # Loss should decrease
        assert losses[-1] < losses[0], f"Loss should decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


class TestCheckpointAndProgress:
    """Test checkpoint saving and progress monitoring."""
    
    def test_save_checkpoint(self):
        """Test saving classification checkpoint."""
        from classify_magvit import save_checkpoint, TrajectoryClassifier
        
        classifier = TrajectoryClassifier(input_dim=128, hidden_dims=[64], num_classes=4, dropout=0.3)
        optimizer = torch.optim.Adam(classifier.parameters())
        
        checkpoint_path = "results/classification/test_checkpoint.pt"
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        
        save_checkpoint(
            model=classifier,
            optimizer=optimizer,
            epoch=5,
            train_loss=0.5,
            val_loss=0.6,
            val_accuracy=0.85,
            checkpoint_path=checkpoint_path
        )
        
        assert Path(checkpoint_path).exists()
        
        # Verify checkpoint contents
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'epoch' in checkpoint
        assert checkpoint['epoch'] == 5
        assert checkpoint['val_accuracy'] == 0.85
        
        # Cleanup
        Path(checkpoint_path).unlink()
    
    def test_load_checkpoint(self):
        """Test loading classification checkpoint."""
        from classify_magvit import save_checkpoint, load_checkpoint, TrajectoryClassifier
        
        # Create and save checkpoint
        classifier = TrajectoryClassifier(input_dim=128, hidden_dims=[64], num_classes=4, dropout=0.3)
        optimizer = torch.optim.Adam(classifier.parameters())
        
        checkpoint_path = "results/classification/test_checkpoint.pt"
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        
        save_checkpoint(classifier, optimizer, 5, 0.5, 0.6, 0.85, checkpoint_path)
        
        # Load checkpoint
        new_classifier = TrajectoryClassifier(input_dim=128, hidden_dims=[64], num_classes=4, dropout=0.3)
        new_optimizer = torch.optim.Adam(new_classifier.parameters())
        
        loaded_data = load_checkpoint(checkpoint_path, new_classifier, new_optimizer)
        
        assert loaded_data['epoch'] == 5
        assert loaded_data['val_accuracy'] == 0.85
        
        # Verify weights are loaded
        for p1, p2 in zip(classifier.parameters(), new_classifier.parameters()):
            assert torch.allclose(p1, p2)
        
        # Cleanup
        Path(checkpoint_path).unlink()
    
    def test_progress_file_creation(self):
        """Test that progress file is created and updated."""
        from classify_magvit import update_progress
        
        progress_path = "results/classification/test_PROGRESS.txt"
        Path(progress_path).parent.mkdir(parents=True, exist_ok=True)
        
        update_progress(
            progress_path=progress_path,
            epoch=5,
            train_loss=0.5,
            val_loss=0.6,
            val_accuracy=0.85,
            best_accuracy=0.87
        )
        
        assert Path(progress_path).exists()
        
        content = Path(progress_path).read_text()
        assert "Epoch: 5" in content
        assert "0.8500" in content  # accuracy (formatted as .4f)
        
        # Cleanup
        Path(progress_path).unlink()


class TestIntegration:
    """Integration tests for full classification pipeline."""
    
    @pytest.mark.timeout(300)  # 5 minutes max
    def test_full_classification_pipeline(self):
        """Test complete classification pipeline from encoding to training."""
        from classify_magvit import train_classifier
        
        dataset_path = "results/20260125_0304_dataset_200_validated.npz"
        model_checkpoint = sorted(list(Path("results/magvit_training").glob("*_best_model.pt")))[-1]
        output_dir = "results/classification"
        
        results = train_classifier(
            dataset_path=dataset_path,
            magvit_checkpoint=str(model_checkpoint),
            output_dir=output_dir,
            epochs=50,  # Increased for small dataset
            batch_size=8,
            learning_rate=0.0005,  # Lower LR for stability
            hidden_dims=[256, 128],  # Smaller model for regularization
            dropout=0.5,  # Higher dropout
            device='cpu'
        )
        
        # Check results
        assert 'best_accuracy' in results
        assert 'final_accuracy' in results
        assert 'train_history' in results
        
        # Check accuracy is reasonable (random baseline is 25% for 4 classes)
        assert results['best_accuracy'] > 0.35, "Accuracy should be clearly better than random"
        
        # Check files created
        assert Path(f"{output_dir}/PROGRESS.txt").exists()
        assert len(list(Path(output_dir).glob("*_training_history.json"))) > 0
        assert len(list(Path(output_dir).glob("*_best_classifier.pt"))) > 0
    
    @pytest.mark.timeout(600)  # 10 minutes max
    def test_target_accuracy_achievable(self):
        """Test that target accuracy (>70%) is achievable with full training."""
        from classify_magvit import train_classifier
        
        dataset_path = "results/20260125_0304_dataset_200_validated.npz"
        model_checkpoint = sorted(list(Path("results/magvit_training").glob("*_best_model.pt")))[-1]
        output_dir = "results/classification"
        
        results = train_classifier(
            dataset_path=dataset_path,
            magvit_checkpoint=str(model_checkpoint),
            output_dir=output_dir,
            epochs=100,
            batch_size=16,
            learning_rate=0.001,
            device='cpu'
        )
        
        # Target: >70% accuracy (random is 25%)
        assert results['best_accuracy'] > 0.70, \
            f"Target accuracy not reached: {results['best_accuracy']:.2%}"
        
        print(f"\n✅ Classification achieved {results['best_accuracy']:.2%} accuracy")
        print(f"   Final test accuracy: {results['final_accuracy']:.2%}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

