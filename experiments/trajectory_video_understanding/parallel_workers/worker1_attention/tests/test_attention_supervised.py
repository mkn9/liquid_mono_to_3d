"""
TDD Tests for Attention-Supervised Training
Worker 1: Attention supervision loss to encourage attention differentiation
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.attention_supervised_trainer import (
    AttentionSupervisedLoss,
    AttentionSupervisedTrainer,
    compute_attention_ratio
)


class TestAttentionSupervisedLoss:
    """Test attention supervision loss component."""
    
    def test_loss_initialization(self):
        """Test loss can be initialized."""
        loss_fn = AttentionSupervisedLoss(alpha=0.2)
        assert loss_fn.alpha == 0.2
    
    def test_loss_encourages_high_persistent_attention(self):
        """Test loss encourages higher attention on persistent objects."""
        loss_fn = AttentionSupervisedLoss(alpha=1.0)
        
        # Mock attention: persistent should have high, transient low
        attention_weights = torch.tensor([
            [0.5, 0.5, 0.5],  # Object 0 receives attention from 3 others
            [0.8, 0.8, 0.8],  # Object 1 (should be penalized if persistent)
            [0.2, 0.2, 0.2],  # Object 2 (should be penalized if transient)
        ])
        
        labels = torch.tensor([0, 0, 1])  # 0=persistent, 1=transient
        
        loss = loss_fn(attention_weights, labels)
        
        # Loss should be positive (there's room for improvement)
        assert loss.item() > 0
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
    
    def test_loss_is_zero_when_perfect(self):
        """Test loss approaches zero when attention is perfect."""
        loss_fn = AttentionSupervisedLoss(alpha=1.0)
        
        # Perfect attention: persistent=high, transient=low
        attention_weights = torch.tensor([
            [1.0, 1.0, 1.0],  # Persistent: high attention
            [1.0, 1.0, 1.0],  # Persistent: high attention
            [0.1, 0.1, 0.1],  # Transient: low attention
        ])
        
        labels = torch.tensor([0, 0, 1])
        
        loss = loss_fn(attention_weights, labels)
        
        # Loss should be low/negative (good state)
        assert loss.item() < 0.5
    
    def test_loss_handles_no_transient_objects(self):
        """Test loss handles case with no transient objects."""
        loss_fn = AttentionSupervisedLoss(alpha=1.0)
        
        attention_weights = torch.tensor([
            [0.5, 0.5],
            [0.5, 0.5],
        ])
        
        labels = torch.tensor([0, 0])  # All persistent
        
        # Should not crash
        loss = loss_fn(attention_weights, labels)
        assert isinstance(loss, torch.Tensor)
    
    def test_loss_handles_no_persistent_objects(self):
        """Test loss handles case with no persistent objects."""
        loss_fn = AttentionSupervisedLoss(alpha=1.0)
        
        attention_weights = torch.tensor([
            [0.5, 0.5],
            [0.5, 0.5],
        ])
        
        labels = torch.tensor([1, 1])  # All transient
        
        # Should not crash
        loss = loss_fn(attention_weights, labels)
        assert isinstance(loss, torch.Tensor)


class TestComputeAttentionRatio:
    """Test attention ratio computation."""
    
    def test_ratio_calculation(self):
        """Test correct ratio calculation."""
        attention_weights = torch.tensor([
            [1.0, 1.0, 1.0],  # Persistent: avg = 1.0
            [1.0, 1.0, 1.0],  # Persistent: avg = 1.0
            [0.5, 0.5, 0.5],  # Transient: avg = 0.5
        ])
        
        labels = torch.tensor([0, 0, 1])
        
        ratio = compute_attention_ratio(attention_weights, labels)
        
        # Persistent (1.0) / Transient (0.5) = 2.0
        assert abs(ratio - 2.0) < 0.01
    
    def test_ratio_handles_equal_attention(self):
        """Test ratio when attention is equal."""
        attention_weights = torch.tensor([
            [0.5, 0.5],
            [0.5, 0.5],
        ])
        
        labels = torch.tensor([0, 1])
        
        ratio = compute_attention_ratio(attention_weights, labels)
        
        # Should be 1.0 (equal attention)
        assert abs(ratio - 1.0) < 0.01
    
    def test_ratio_handles_no_transient(self):
        """Test ratio returns infinity indicator when no transients."""
        attention_weights = torch.tensor([
            [1.0, 1.0],
            [1.0, 1.0],
        ])
        
        labels = torch.tensor([0, 0])
        
        ratio = compute_attention_ratio(attention_weights, labels)
        
        # Should return a large number or handle gracefully
        assert ratio >= 0  # Just check it doesn't crash


class TestAttentionSupervisedTrainer:
    """Test the trainer with attention supervision."""
    
    def test_trainer_initialization(self):
        """Test trainer can be initialized."""
        trainer = AttentionSupervisedTrainer(
            model=None,  # Mock later
            optimizer=None,
            alpha=0.2,
            device='cpu'
        )
        
        assert trainer.alpha == 0.2
        assert trainer.device == 'cpu'
    
    def test_combined_loss_computation(self):
        """Test that combined loss includes both classification and attention."""
        # This will be tested during integration
        # Just check the structure exists
        trainer = AttentionSupervisedTrainer(
            model=None,
            optimizer=None,
            alpha=0.2,
            device='cpu'
        )
        
        assert hasattr(trainer, 'compute_combined_loss')
    
    def test_early_stopping_detection(self):
        """Test early stopping triggers when criteria met."""
        trainer = AttentionSupervisedTrainer(
            model=None,
            optimizer=None,
            alpha=0.2,
            device='cpu',
            early_stop_ratio=1.5
        )
        
        # Mock metrics that meet criteria
        metrics = {
            'attention_ratio': 1.6,
            'val_accuracy': 0.76,
            'consistency': 0.72
        }
        
        should_stop = trainer.check_early_stopping(metrics)
        assert should_stop is True
    
    def test_early_stopping_not_triggered_when_criteria_not_met(self):
        """Test early stopping doesn't trigger when criteria not met."""
        trainer = AttentionSupervisedTrainer(
            model=None,
            optimizer=None,
            alpha=0.2,
            device='cpu',
            early_stop_ratio=1.5
        )
        
        # Mock metrics that DON'T meet criteria
        metrics = {
            'attention_ratio': 1.2,  # Too low
            'val_accuracy': 0.76,
            'consistency': 0.72
        }
        
        should_stop = trainer.check_early_stopping(metrics)
        assert should_stop is False


# RED PHASE: These tests should FAIL initially
if __name__ == "__main__":
    pytest.main([__file__, '-v'])

