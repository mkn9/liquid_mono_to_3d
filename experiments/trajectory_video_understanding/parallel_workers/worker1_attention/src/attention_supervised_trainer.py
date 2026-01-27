"""
Attention-Supervised Trainer (Worker 1)
Adds explicit loss term to encourage attention differentiation between persistent and transient objects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class AttentionSupervisedLoss(nn.Module):
    """
    Attention supervision loss that encourages:
    - High attention weights on persistent objects
    - Low attention weights on transient objects
    """
    
    def __init__(self, alpha: float = 0.2):
        """
        Args:
            alpha: Weight for attention loss component (default: 0.2)
        """
        super().__init__()
        self.alpha = alpha
    
    def forward(self, attention_weights: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute attention supervision loss.
        
        Args:
            attention_weights: Attention weights from transformer
                              Shape: (batch_size, num_heads, seq_len, seq_len)
                              or (batch_size, seq_len, seq_len) if averaged
            labels: Object persistence labels (0=persistent, 1=transient)
                   Shape: (batch_size, seq_len) or (seq_len,)
        
        Returns:
            attention_loss: Scalar loss value
        """
        # Handle different attention weight shapes
        if attention_weights.dim() == 4:
            # Average across heads: (batch, heads, seq, seq) -> (batch, seq, seq)
            attention_weights = attention_weights.mean(dim=1)
        
        if attention_weights.dim() == 3:
            # Average across batch: (batch, seq, seq) -> (seq, seq)
            attention_weights = attention_weights.mean(dim=0)
        
        # Average attention received by each object (column-wise)
        # Each object is a query, sum attention it receives from all keys
        avg_attention_per_object = attention_weights.mean(dim=0)  # Shape: (seq_len,)
        
        # Ensure labels is 1D
        if labels.dim() > 1:
            labels = labels.squeeze()
        
        # Move labels to GPU if attention_weights is on GPU
        labels = labels.to(avg_attention_per_object.device)
        
        # Separate persistent and transient
        persistent_mask = (labels == 0)
        transient_mask = (labels == 1)
        
        persistent_attention = avg_attention_per_object[persistent_mask]
        transient_attention = avg_attention_per_object[transient_mask]
        
        # Loss encourages: high persistent attention, low transient attention
        persistent_loss = -persistent_attention.mean() if len(persistent_attention) > 0 else torch.tensor(0.0)
        transient_loss = transient_attention.mean() if len(transient_attention) > 0 else torch.tensor(0.0)
        
        total_attention_loss = persistent_loss + transient_loss
        
        return total_attention_loss


def compute_attention_ratio(attention_weights: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute ratio of attention on persistent vs transient objects.
    
    Args:
        attention_weights: Attention weights, shape varies
        labels: Persistence labels (0=persistent, 1=transient)
    
    Returns:
        ratio: persistent_attention / transient_attention
    """
    # Handle different shapes
    if attention_weights.dim() == 4:
        attention_weights = attention_weights.mean(dim=1)  # Average heads
    if attention_weights.dim() == 3:
        attention_weights = attention_weights.mean(dim=0)  # Average batch
    
    # Average attention per object
    avg_attention_per_object = attention_weights.mean(dim=0)
    
    # Ensure labels is 1D and on same device as attention_weights
    if labels.dim() > 1:
        labels = labels.squeeze()
    
    # Move labels to GPU if attention_weights is on GPU
    labels = labels.to(avg_attention_per_object.device)
    
    # Move labels to GPU if attention_weights is on GPU
    labels = labels.to(avg_attention_per_object.device)
    
    # Separate
    persistent_mask = (labels == 0)
    transient_mask = (labels == 1)
    
    persistent_attn = avg_attention_per_object[persistent_mask].mean().item() if persistent_mask.any() else 0.0
    transient_attn = avg_attention_per_object[transient_mask].mean().item() if transient_mask.any() else 0.0
    
    # Compute ratio (avoid division by zero)
    if transient_attn < 1e-6:
        return 100.0  # Very high ratio
    
    ratio = persistent_attn / transient_attn
    return ratio


class AttentionSupervisedTrainer:
    """
    Trainer that combines classification loss with attention supervision loss.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        alpha: float = 0.2,
        device: str = 'cuda',
        early_stop_ratio: float = 1.5,
        early_stop_accuracy: float = 0.75,
        early_stop_consistency: float = 0.70
    ):
        """
        Args:
            model: Transformer model with attention extraction
            optimizer: Optimizer for training
            alpha: Weight for attention supervision loss
            device: Training device
            early_stop_ratio: Min attention ratio to trigger early stopping
            early_stop_accuracy: Min validation accuracy to trigger early stopping
            early_stop_consistency: Min consistency (% samples passing) for early stopping
        """
        self.model = model
        self.optimizer = optimizer
        self.alpha = alpha
        self.device = device
        
        self.early_stop_ratio = early_stop_ratio
        self.early_stop_accuracy = early_stop_accuracy
        self.early_stop_consistency = early_stop_consistency
        
        self.classification_loss_fn = nn.CrossEntropyLoss()
        self.attention_loss_fn = AttentionSupervisedLoss(alpha=alpha)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'attention_ratio': [],
            'consistency': []
        }
    
    def compute_combined_loss(
        self,
        logits: torch.Tensor,
        attention_weights: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined classification + attention supervision loss.
        
        Args:
            logits: Classification logits, shape (batch, num_classes)
            attention_weights: Attention weights from transformer
            labels: Ground truth labels, shape (batch,)
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        # Classification loss
        class_loss = self.classification_loss_fn(logits, labels)
        
        # Attention supervision loss
        attn_loss = self.attention_loss_fn(attention_weights, labels)
        
        # Combined
        total_loss = class_loss + self.alpha * attn_loss
        
        loss_dict = {
            'classification_loss': class_loss.item(),
            'attention_loss': attn_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict
    
    def check_early_stopping(self, metrics: Dict[str, float]) -> bool:
        """
        Check if early stopping criteria are met.
        
        Args:
            metrics: Dictionary containing:
                - attention_ratio: Current attention ratio
                - val_accuracy: Validation accuracy
                - consistency: Fraction of samples meeting ratio threshold
        
        Returns:
            should_stop: True if all criteria met
        """
        ratio_met = metrics['attention_ratio'] >= self.early_stop_ratio
        accuracy_met = metrics['val_accuracy'] >= self.early_stop_accuracy
        consistency_met = metrics['consistency'] >= self.early_stop_consistency
        
        return ratio_met and accuracy_met and consistency_met
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            # Forward pass
            tokens = batch['tokens'].to(self.device)
            labels = batch['labels'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Get predictions and attention
            logits, attention_weights = self.model(tokens, mask, return_attention=True)
            
            # Compute combined loss
            loss, loss_dict = self.compute_combined_loss(logits, attention_weights, labels)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader):
        """Validate and compute all metrics including attention ratio."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_ratios = []
        
        with torch.no_grad():
            for batch in val_loader:
                tokens = batch['tokens'].to(self.device)
                labels = batch['labels'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                # Forward
                logits, attention_weights = self.model(tokens, mask, return_attention=True)
                
                # Compute loss
                loss, _ = self.compute_combined_loss(logits, attention_weights, labels)
                total_loss += loss.item()
                
                # Compute accuracy
                predictions = logits.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                # Compute attention ratio for this batch
                ratio = compute_attention_ratio(attention_weights, labels)
                all_ratios.append(ratio)
        
        metrics = {
            'val_loss': total_loss / len(val_loader),
            'val_accuracy': correct / total,
            'attention_ratio': np.mean(all_ratios),
            'consistency': np.mean([r >= self.early_stop_ratio * 0.87 for r in all_ratios])  # 87% of threshold
        }
        
        return metrics

