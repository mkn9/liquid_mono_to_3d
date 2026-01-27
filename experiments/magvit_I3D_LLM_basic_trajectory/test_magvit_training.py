#!/usr/bin/env python3
"""
TDD Tests for MAGVIT Training

Per cursorrules: Write tests FIRST, then implementation.

Uses TINY datasets for fast testing (<30 seconds per test).
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import json
import shutil


@pytest.fixture
def tiny_dataset(tmp_path):
    """Create a tiny test dataset (10 samples, 8 frames, 32x32) for fast testing"""
    # Create tiny videos
    videos = np.random.randn(10, 8, 3, 32, 32).astype(np.float32)
    labels = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])
    trajectory_3d = np.random.randn(10, 8, 3).astype(np.float32)
    
    # Save as .npz
    dataset_path = tmp_path / "tiny_dataset.npz"
    np.savez(dataset_path, videos=videos, labels=labels, trajectory_3d=trajectory_3d)
    
    return dataset_path


class TestMAGVITTrainingSetup:
    """Test training environment setup"""
    
    @pytest.mark.timeout(30)  # 30 second timeout
    def test_dataset_loads_correctly(self, tiny_dataset):
        """Test that dataset can be loaded for training"""
        from train_magvit import load_dataset
        
        train_loader, val_loader, dataset_info = load_dataset(
            dataset_path=str(tiny_dataset),
            batch_size=2,
            train_split=0.8
        )
        
        # Check loaders exist
        assert train_loader is not None
        assert val_loader is not None
        
        # Check dataset info
        assert 'num_train' in dataset_info
        assert 'num_val' in dataset_info
        assert dataset_info['num_train'] > 0
        assert dataset_info['num_val'] > 0
        
        # Check batch shape
        batch = next(iter(train_loader))
        videos, labels = batch
        assert videos.shape[0] <= 2  # batch size
        assert videos.shape[1] == 3  # channels
        assert videos.ndim == 5  # (B, C, T, H, W)
    
    @pytest.mark.timeout(30)  # 30 second timeout
    def test_model_initializes(self):
        """Test MAGVIT model initializes correctly (small size for speed)"""
        from train_magvit import create_model
        
        model = create_model(
            image_size=32,  # Smaller for faster testing
            init_dim=32,    # Smaller for faster testing
            use_fsq=True
        )
        
        assert model is not None
        assert hasattr(model, 'encode')
        assert hasattr(model, 'decode')
        
        # Test forward pass with small input
        test_input = torch.randn(1, 3, 8, 32, 32)
        with torch.no_grad():
            codes = model.encode(test_input)
            reconstructed = model.decode(codes)
        
        assert reconstructed.shape == test_input.shape


class TestMAGVITTrainingLoop:
    """Test training loop execution"""
    
    @pytest.mark.timeout(60)  # 60 second timeout
    def test_training_runs_one_epoch(self, tiny_dataset):
        """Test training loop runs for one epoch without errors (using tiny dataset)"""
        from train_magvit import train_one_epoch, load_dataset, create_model
        
        # Tiny dataset for speed
        train_loader, _, _ = load_dataset(
            dataset_path=str(tiny_dataset),
            batch_size=2,
            train_split=0.8
        )
        
        model = create_model(image_size=32, init_dim=32, use_fsq=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        
        # Train one epoch (should be fast with tiny dataset)
        metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device='cpu',
            epoch=0,
            verbose=False  # Disable verbose output in tests
        )
        
        # Check metrics returned
        assert 'loss' in metrics
        assert 'mse' in metrics
        assert metrics['loss'] > 0
        assert metrics['mse'] >= 0
    
    @pytest.mark.timeout(60)  # 60 second timeout
    def test_validation_runs(self, tiny_dataset):
        """Test validation loop runs without errors (using tiny dataset)"""
        from train_magvit import validate, load_dataset, create_model
        
        _, val_loader, _ = load_dataset(
            dataset_path=str(tiny_dataset),
            batch_size=2,
            train_split=0.8
        )
        
        model = create_model(image_size=32, init_dim=32, use_fsq=True)
        
        # Validate
        metrics = validate(
            model=model,
            val_loader=val_loader,
            device='cpu'
        )
        
        # Check metrics
        assert 'val_loss' in metrics
        assert 'val_mse' in metrics
        assert metrics['val_loss'] > 0
        assert metrics['val_mse'] >= 0


class TestMAGVITCheckpoints:
    """Test checkpoint saving and loading"""
    
    @pytest.mark.timeout(60)  # 60 second timeout
    def test_checkpoint_saves(self, tmp_path):
        """Test checkpoint is saved correctly"""
        from train_magvit import save_checkpoint, create_model
        
        model = create_model(image_size=32, init_dim=32, use_fsq=True)
        optimizer = torch.optim.Adam(model.parameters())
        
        checkpoint_path = tmp_path / "checkpoint.pt"
        
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=5,
            loss=0.123,
            checkpoint_path=checkpoint_path
        )
        
        # Check file exists
        assert checkpoint_path.exists()
        
        # Check file size > 0
        assert checkpoint_path.stat().st_size > 1000
    
    @pytest.mark.timeout(60)  # 60 second timeout
    def test_checkpoint_loads(self, tmp_path):
        """Test checkpoint can be loaded correctly"""
        from train_magvit import save_checkpoint, load_checkpoint, create_model
        
        # Save checkpoint
        model = create_model(image_size=32, init_dim=32, use_fsq=True)
        optimizer = torch.optim.Adam(model.parameters())
        
        checkpoint_path = tmp_path / "checkpoint.pt"
        save_checkpoint(model, optimizer, epoch=5, loss=0.123, checkpoint_path=checkpoint_path)
        
        # Load checkpoint
        new_model = create_model(image_size=32, init_dim=32, use_fsq=True)
        new_optimizer = torch.optim.Adam(new_model.parameters())
        
        start_epoch, best_loss = load_checkpoint(
            model=new_model,
            optimizer=new_optimizer,
            checkpoint_path=checkpoint_path
        )
        
        # Check values
        assert start_epoch == 6  # Should resume from next epoch
        assert abs(best_loss - 0.123) < 1e-6


class TestMAGVITProgressMonitoring:
    """Test progress monitoring and periodic saves"""
    
    @pytest.mark.timeout(10)  # 10 second timeout (fast test)
    def test_progress_file_created(self, tmp_path, capsys):
        """Test PROGRESS.txt is created and stdout is printed"""
        from train_magvit import update_progress
        
        progress_path = tmp_path / "PROGRESS.txt"
        
        update_progress(
            progress_path=progress_path,
            epoch=5,
            total_epochs=100,
            train_loss=0.5,
            val_loss=0.6,
            elapsed_time=120.5
        )
        
        # Check file exists
        assert progress_path.exists()
        
        # Check file content
        content = progress_path.read_text()
        assert "5/100" in content
        assert "Train Loss:" in content
        assert "Val Loss:" in content
        
        # Check stdout was printed (keeps SSH alive!)
        captured = capsys.readouterr()
        assert "PROGRESS UPDATE" in captured.out
        assert "5/100" in captured.out
    
    @pytest.mark.timeout(10)  # 10 second timeout (fast test)
    def test_periodic_checkpoint_saving(self, tmp_path):
        """Test checkpoints are saved at intervals"""
        from train_magvit import should_save_checkpoint
        
        # Should save at intervals
        assert should_save_checkpoint(epoch=0, checkpoint_interval=5)
        assert should_save_checkpoint(epoch=5, checkpoint_interval=5)
        assert should_save_checkpoint(epoch=10, checkpoint_interval=5)
        
        # Should NOT save in between
        assert not should_save_checkpoint(epoch=1, checkpoint_interval=5)
        assert not should_save_checkpoint(epoch=3, checkpoint_interval=5)


class TestMAGVITLearning:
    """Test that MAGVIT actually learns"""
    
    @pytest.mark.timeout(120)  # 2 minute timeout
    def test_loss_decreases_over_epochs(self, tiny_dataset):
        """Test that loss decreases over multiple epochs (basic learning)"""
        from train_magvit import train_one_epoch, load_dataset, create_model
        
        train_loader, _, _ = load_dataset(
            dataset_path=str(tiny_dataset),
            batch_size=2,
            train_split=0.8
        )
        
        model = create_model(image_size=32, init_dim=32, use_fsq=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train for 3 epochs (fast with tiny dataset)
        losses = []
        for epoch in range(3):
            metrics = train_one_epoch(model, train_loader, optimizer, 'cpu', epoch, verbose=False)
            losses.append(metrics['loss'])
        
        # Check loss trend (should generally decrease or stay stable)
        # Allow for some noise, but final should be <= initial
        assert losses[-1] <= losses[0] * 1.1, \
            f"Loss should decrease or stay stable: {losses[0]:.4f} -> {losses[-1]:.4f}"
    
    @pytest.mark.timeout(180)  # 3 minute timeout
    def test_reconstruction_quality_improves(self, tiny_dataset):
        """Test reconstruction MSE improves with training"""
        from train_magvit import train_one_epoch, validate, load_dataset, create_model
        
        train_loader, val_loader, _ = load_dataset(
            dataset_path=str(tiny_dataset),
            batch_size=2,
            train_split=0.8
        )
        
        model = create_model(image_size=32, init_dim=32, use_fsq=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Initial validation
        metrics_before = validate(model, val_loader, 'cpu')
        initial_mse = metrics_before['val_mse']
        
        # Train for 5 epochs
        for epoch in range(5):
            train_one_epoch(model, train_loader, optimizer, 'cpu', epoch, verbose=False)
        
        # Final validation
        metrics_after = validate(model, val_loader, 'cpu')
        final_mse = metrics_after['val_mse']
        
        # MSE should improve (decrease)
        assert final_mse < initial_mse * 0.95, \
            f"MSE should improve after training: {initial_mse:.4f} -> {final_mse:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

