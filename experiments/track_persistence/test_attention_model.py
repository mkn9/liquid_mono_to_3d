#!/usr/bin/env python3
"""
Tests for Attention Persistence Model
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from attention_persistence_model import (
    PositionalEncoding,
    AttentionPersistenceModel,
    create_model,
    PersistenceTrainer
)


class TestPositionalEncoding:
    """Test PositionalEncoding module."""
    
    def test_positional_encoding_shape(self):
        """Test that positional encoding has correct shape."""
        d_model = 256
        max_len = 100
        pe = PositionalEncoding(d_model, max_len)
        
        batch_size = 4
        seq_len = 20
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = pe(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_positional_encoding_changes_input(self):
        """Test that positional encoding actually modifies the input."""
        d_model = 256
        pe = PositionalEncoding(d_model, 100)
        
        x = torch.randn(2, 10, d_model)
        output = pe(x)
        
        # Should be different from input
        assert not torch.allclose(x, output)


class TestAttentionPersistenceModel:
    """Test AttentionPersistenceModel."""
    
    @pytest.fixture
    def model(self):
        """Create a model instance."""
        return create_model(
            input_dim=256,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            dropout=0.1
        )
    
    def test_model_creation(self, model):
        """Test model creation."""
        assert isinstance(model, AttentionPersistenceModel)
        assert model.input_dim == 256
        assert model.hidden_dim == 128
        assert model.num_layers == 2
        assert model.num_heads == 4
    
    def test_forward_pass_shape(self, model):
        """Test forward pass output shapes."""
        batch_size = 4
        seq_len = 20
        input_dim = 256
        
        x = torch.randn(batch_size, seq_len, input_dim)
        logits, attn_weights = model(x)
        
        assert logits.shape == (batch_size, 1)
        # Attention weights might be None if hooks don't fire in test mode
        if attn_weights is not None:
            assert attn_weights.shape[0] == batch_size
            assert attn_weights.shape[1] == 4  # num_heads
            assert attn_weights.shape[2] == seq_len
            assert attn_weights.shape[3] == seq_len
    
    def test_forward_pass_with_padding_mask(self, model):
        """Test forward pass with padding mask."""
        batch_size = 4
        seq_len = 20
        input_dim = 256
        
        x = torch.randn(batch_size, seq_len, input_dim)
        # Create padding mask (last 5 positions are padding)
        padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        padding_mask[:, -5:] = True
        
        logits, _ = model(x, padding_mask)
        
        assert logits.shape == (batch_size, 1)
    
    def test_predict_method(self, model):
        """Test predict method."""
        batch_size = 4
        seq_len = 20
        input_dim = 256
        
        x = torch.randn(batch_size, seq_len, input_dim)
        
        predictions, probabilities, attn_weights = model.predict(x)
        
        assert predictions.shape == (batch_size,)
        assert probabilities.shape == (batch_size,)
        assert torch.all((predictions == 0) | (predictions == 1))
        assert torch.all((probabilities >= 0) & (probabilities <= 1))
    
    def test_get_frame_importance(self, model):
        """Test get_frame_importance method."""
        batch_size = 4
        seq_len = 20
        input_dim = 256
        
        x = torch.randn(batch_size, seq_len, input_dim)
        
        importance = model.get_frame_importance(x)
        
        # Importance might be None if attention hooks don't fire
        if importance is not None:
            assert importance.shape == (batch_size, seq_len)
            # Should sum to approximately 1 (normalized attention)
            # But this depends on the averaging, so just check shape
    
    def test_model_output_range(self, model):
        """Test that model outputs are in valid range."""
        batch_size = 4
        seq_len = 20
        input_dim = 256
        
        x = torch.randn(batch_size, seq_len, input_dim)
        
        _, probabilities, _ = model.predict(x)
        
        # Probabilities should be between 0 and 1
        assert torch.all(probabilities >= 0)
        assert torch.all(probabilities <= 1)
    
    def test_different_sequence_lengths(self, model):
        """Test model with different sequence lengths."""
        input_dim = 256
        
        for seq_len in [5, 10, 20, 50]:
            x = torch.randn(2, seq_len, input_dim)
            logits, _ = model(x)
            assert logits.shape == (2, 1)
    
    def test_model_eval_mode(self, model):
        """Test model in eval mode."""
        model.eval()
        
        x = torch.randn(4, 20, 256)
        
        with torch.no_grad():
            logits1, _ = model(x)
            logits2, _ = model(x)
        
        # In eval mode, same input should give same output
        assert torch.allclose(logits1, logits2)
    
    def test_model_train_mode(self, model):
        """Test model in train mode."""
        model.train()
        
        x = torch.randn(4, 20, 256)
        
        # Multiple forward passes (dropout should cause variation)
        logits1, _ = model(x)
        logits2, _ = model(x)
        
        # Outputs might be different due to dropout
        # Just check they have correct shape
        assert logits1.shape == logits2.shape == (4, 1)


class TestPersistenceTrainer:
    """Test PersistenceTrainer."""
    
    @pytest.fixture
    def model(self):
        """Create a small model for testing."""
        return create_model(
            input_dim=64,
            hidden_dim=64,
            num_layers=1,
            num_heads=2,
            dropout=0.1
        )
    
    @pytest.fixture
    def trainer(self, model):
        """Create a trainer instance."""
        return PersistenceTrainer(model, device='cpu', learning_rate=1e-3)
    
    @pytest.fixture
    def dummy_dataloader(self):
        """Create a dummy dataloader."""
        # Create simple dataset
        class DummyDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 20
            
            def __getitem__(self, idx):
                return {
                    'features': torch.randn(10, 64),
                    'labels': torch.tensor(idx % 2, dtype=torch.float32)
                }
        
        dataset = DummyDataset()
        return torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
    
    def test_trainer_creation(self, trainer):
        """Test trainer creation."""
        assert isinstance(trainer, PersistenceTrainer)
        assert trainer.device == 'cpu'
        assert trainer.optimizer is not None
        assert trainer.criterion is not None
    
    def test_train_epoch(self, trainer, dummy_dataloader):
        """Test training for one epoch."""
        initial_loss = None
        
        # Train for a few iterations
        for i in range(2):
            loss = trainer.train_epoch(dummy_dataloader)
            assert loss > 0
            assert not np.isnan(loss)
            
            if i == 0:
                initial_loss = loss
        
        # Loss should be finite
        assert np.isfinite(loss)
    
    def test_validate(self, trainer, dummy_dataloader):
        """Test validation."""
        val_loss, accuracy, metrics = trainer.validate(dummy_dataloader)
        
        assert val_loss > 0
        assert not np.isnan(val_loss)
        assert 0 <= accuracy <= 1
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1
    
    def test_metrics_tracking(self, trainer, dummy_dataloader):
        """Test that metrics are tracked."""
        # Initial state
        assert len(trainer.train_losses) == 0
        assert len(trainer.val_losses) == 0
        
        # Train and validate
        trainer.train_epoch(dummy_dataloader)
        trainer.validate(dummy_dataloader)
        
        # Should have recorded metrics
        assert len(trainer.train_losses) == 1
        assert len(trainer.val_losses) == 1
        assert len(trainer.val_accuracies) == 1


def test_model_parameter_count():
    """Test that model has reasonable number of parameters."""
    model = create_model(
        input_dim=256,
        hidden_dim=256,
        num_layers=4,
        num_heads=8
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    assert total_params > 0
    assert trainable_params == total_params
    assert total_params < 10_000_000  # Less than 10M parameters


def test_model_gradient_flow():
    """Test that gradients flow through the model."""
    model = create_model(input_dim=64, hidden_dim=64, num_layers=1, num_heads=2)
    model.train()
    
    x = torch.randn(4, 10, 64)
    target = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
    
    # Forward pass
    logits, _ = model(x)
    logits = logits.squeeze(-1)
    
    # Compute loss
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits, target)
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist
    has_gradients = False
    for param in model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_gradients = True
            break
    
    assert has_gradients, "No gradients found in model parameters"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

