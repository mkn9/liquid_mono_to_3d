#!/usr/bin/env python3
"""
Transformer Attention Model for Track Persistence
=================================================
Uses Transformer with attention mechanism to classify 2D tracks as persistent or transient.

The attention weights reveal which temporal frames are most important for the decision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict
import numpy as np


class PositionalEncoding(nn.Module):
    """Positional encoding for temporal sequences."""
    
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:x.size(1), :]


class AttentionPersistenceModel(nn.Module):
    """
    Transformer-based model for track persistence classification.
    
    Architecture:
        Input: MagVIT features (T, D)
        ↓
        Positional Encoding
        ↓
        Transformer Encoder (4 layers, 8 heads)
        ↓
        Extract attention weights
        ↓
        Global pooling + Classification head
        ↓
        Output: persistence probability + attention weights
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 100
    ):
        """
        Initialize model.
        
        Args:
            input_dim: Dimension of input features (from MagVIT)
            hidden_dim: Hidden dimension for Transformer
            num_layers: Number of Transformer encoder layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Input projection (if needed)
        if input_dim != hidden_dim:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_proj = nn.Identity()
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Store attention weights for visualization
        self.attention_weights = None
        
        # Register hooks to extract attention weights
        self._register_attention_hooks()
    
    def _register_attention_hooks(self):
        """Register hooks to capture attention weights from last layer."""
        def hook_fn(module, input, output):
            # output is (attn_output, attn_output_weights)
            if len(output) == 2 and output[1] is not None:
                self.attention_weights = output[1]  # (batch, num_heads, seq_len, seq_len)
        
        # Register hook on last encoder layer's attention
        last_layer = self.transformer.layers[-1].self_attn
        last_layer.register_forward_hook(hook_fn)
    
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input features of shape (batch, seq_len, input_dim)
            padding_mask: Optional mask for padded positions (batch, seq_len)
            
        Returns:
            logits: Classification logits of shape (batch, 1)
            attention_weights: Attention weights of shape (batch, num_heads, seq_len, seq_len)
        """
        # Input projection
        x = self.input_proj(x)  # (batch, seq_len, hidden_dim)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        encoded = self.transformer(x, src_key_padding_mask=padding_mask)  # (batch, seq_len, hidden_dim)
        
        # Global average pooling over sequence
        if padding_mask is not None:
            # Mask out padded positions before pooling
            mask_expanded = ~padding_mask.unsqueeze(-1)  # (batch, seq_len, 1)
            masked_encoded = encoded * mask_expanded
            pooled = masked_encoded.sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            pooled = encoded.mean(dim=1)  # (batch, hidden_dim)
        
        # Classification
        logits = self.classifier(pooled)  # (batch, 1)
        
        return logits, self.attention_weights
    
    def predict(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make predictions with attention analysis.
        
        Args:
            x: Input features of shape (batch, seq_len, input_dim)
            padding_mask: Optional mask for padded positions
            threshold: Classification threshold
            
        Returns:
            predictions: Binary predictions (batch,)
            probabilities: Predicted probabilities (batch,)
            attention_weights: Averaged attention weights (batch, seq_len, seq_len)
        """
        logits, attn_weights = self.forward(x, padding_mask)
        
        # Convert logits to probabilities
        probabilities = torch.sigmoid(logits).squeeze(-1)  # (batch,)
        
        # Binary predictions
        predictions = (probabilities >= threshold).long()
        
        # Average attention weights across heads
        if attn_weights is not None:
            # (batch, num_heads, seq_len, seq_len) -> (batch, seq_len, seq_len)
            avg_attn = attn_weights.mean(dim=1)
        else:
            avg_attn = None
        
        return predictions, probabilities, avg_attn
    
    def get_frame_importance(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get importance score for each frame in the sequence.
        
        This averages the attention weights to show which frames the model
        focuses on when making its decision.
        
        Args:
            x: Input features of shape (batch, seq_len, input_dim)
            padding_mask: Optional mask for padded positions
            
        Returns:
            importance: Frame importance scores of shape (batch, seq_len)
        """
        _, attn_weights = self.forward(x, padding_mask)
        
        if attn_weights is None:
            return None
        
        # Average over heads and target positions to get source importance
        # (batch, num_heads, seq_len, seq_len) -> (batch, seq_len)
        importance = attn_weights.mean(dim=(1, 2))
        
        return importance


def create_model(
    input_dim: int = 256,
    hidden_dim: int = 256,
    num_layers: int = 4,
    num_heads: int = 8,
    dropout: float = 0.1
) -> AttentionPersistenceModel:
    """
    Factory function to create persistence model.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension
        num_layers: Number of Transformer layers
        num_heads: Number of attention heads
        dropout: Dropout rate
        
    Returns:
        AttentionPersistenceModel instance
    """
    return AttentionPersistenceModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout
    )


class PersistenceTrainer:
    """Trainer for persistence classification model."""
    
    def __init__(
        self,
        model: AttentionPersistenceModel,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 1e-4
    ):
        """
        Initialize trainer.
        
        Args:
            model: AttentionPersistenceModel instance
            device: Device to train on
            learning_rate: Learning rate
        """
        self.model = model.to(device)
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            features = batch['features'].to(self.device)  # (batch, seq_len, input_dim)
            labels = batch['labels'].to(self.device).float()  # (batch,)
            padding_mask = batch.get('padding_mask', None)
            
            if padding_mask is not None:
                padding_mask = padding_mask.to(self.device)
            
            # Forward pass
            logits, _ = self.model(features, padding_mask)
            logits = logits.squeeze(-1)
            
            # Compute loss
            loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    @torch.no_grad()
    def validate(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Validate model.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            avg_loss: Average validation loss
            accuracy: Classification accuracy
            metrics: Dictionary with precision, recall, F1
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_labels = []
        
        for batch in dataloader:
            features = batch['features'].to(self.device)
            labels = batch['labels'].to(self.device).float()
            padding_mask = batch.get('padding_mask', None)
            
            if padding_mask is not None:
                padding_mask = padding_mask.to(self.device)
            
            # Forward pass
            logits, _ = self.model(features, padding_mask)
            logits = logits.squeeze(-1)
            
            # Compute loss
            loss = self.criterion(logits, labels)
            total_loss += loss.item()
            num_batches += 1
            
            # Get predictions
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).long()
            
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
        
        # Compute metrics
        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)
        
        accuracy = (all_predictions == all_labels).mean()
        
        # Precision, recall, F1
        tp = ((all_predictions == 1) & (all_labels == 1)).sum()
        fp = ((all_predictions == 1) & (all_labels == 0)).sum()
        fn = ((all_predictions == 0) & (all_labels == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        
        return avg_loss, accuracy, metrics


if __name__ == "__main__":
    # Test model creation
    model = create_model()
    
    # Test forward pass
    batch_size = 4
    seq_len = 20
    input_dim = 256
    
    x = torch.randn(batch_size, seq_len, input_dim)
    logits, attn_weights = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {logits.shape}")
    if attn_weights is not None:
        print(f"Attention weights shape: {attn_weights.shape}")
    
    # Test frame importance
    importance = model.get_frame_importance(x)
    if importance is not None:
        print(f"Frame importance shape: {importance.shape}")
        print(f"Frame importance example: {importance[0].tolist()}")

