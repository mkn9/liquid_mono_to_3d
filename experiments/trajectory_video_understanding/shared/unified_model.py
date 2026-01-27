"""
Unified Multi-Task Model for Trajectory Video Understanding
============================================================

This model combines:
1. Any feature extractor (I3D, Slow/Fast, MagVIT, Transformer)
2. Two task heads:
   - Classification: Predict trajectory class (Linear, Circular, Helical, Parabolic)
   - Prediction: Predict future 3D position (x, y, z) at t+5

Architecture:
    Video (B, T, C, H, W)
        ↓
    Feature Extractor → Features (B, T, D)
        ↓
    Temporal Pooling → Aggregated Features (B, D)
        ↓
    ┌─────────────┴─────────────┐
    ↓                             ↓
Classification Head         Prediction Head
(Linear, Dropout)           (Linear, Dropout)
    ↓                             ↓
Class Logits (B, 4)         Position (B, 3)

Loss:
    Total Loss = class_weight * CrossEntropy + pred_weight * MSE

TDD Evidence: See artifacts/tdd_unified_model_*.txt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from base_extractor import FeatureExtractor


class UnifiedModel(nn.Module):
    """
    Unified model with pluggable feature extractor and two task heads.
    
    Args:
        extractor (FeatureExtractor): Feature extraction module
        num_classes (int): Number of trajectory classes (default: 4)
        dropout (float): Dropout probability (default: 0.3)
        hidden_dim (int): Hidden dimension for task heads (default: 512)
    
    Example:
        >>> from base_extractor import DummyExtractor
        >>> extractor = DummyExtractor(dim=256)
        >>> model = UnifiedModel(extractor, num_classes=4)
        >>> video = torch.randn(2, 16, 3, 64, 64)
        >>> output = model(video)
        >>> print(output['classification'].shape)  # (2, 4)
        >>> print(output['prediction'].shape)      # (2, 3)
    """
    
    def __init__(
        self,
        extractor: FeatureExtractor,
        num_classes: int = 4,
        dropout: float = 0.3,
        hidden_dim: int = 512
    ):
        """Initialize unified model with feature extractor and task heads."""
        super().__init__()
        
        # Store extractor
        self.extractor = extractor
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Get feature dimension from extractor
        feature_dim = extractor.feature_dim
        
        # Temporal pooling (average over time dimension)
        # Input: (B, T, D) → Output: (B, D)
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # Shared dropout
        self.dropout = nn.Dropout(dropout)
        
        # Task 1: Classification Head
        # Predicts trajectory class (Linear, Circular, Helical, Parabolic)
        self.classification_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Task 2: Prediction Head
        # Predicts future 3D position (x, y, z)
        self.prediction_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)  # x, y, z
        )
    
    def forward(self, video: torch.Tensor) -> dict:
        """
        Forward pass through the model.
        
        Args:
            video (torch.Tensor): Input video
                Shape: (B, T, C, H, W)
                - B: Batch size
                - T: Number of frames
                - C: Number of channels (typically 3 for RGB)
                - H: Height in pixels
                - W: Width in pixels
        
        Returns:
            dict: Dictionary containing:
                - 'classification': Class logits (B, num_classes)
                - 'prediction': Future position (B, 3)
        """
        # Extract features: (B, T, C, H, W) → (B, T, D)
        features = self.extractor.extract(video)
        
        # Temporal pooling: (B, T, D) → (B, D, 1) → (B, D)
        # Transpose for AdaptiveAvgPool1d which expects (B, D, T)
        features_transposed = features.transpose(1, 2)  # (B, D, T)
        pooled = self.temporal_pool(features_transposed)  # (B, D, 1)
        pooled = pooled.squeeze(-1)  # (B, D)
        
        # Apply dropout
        pooled = self.dropout(pooled)
        
        # Task 1: Classification
        class_logits = self.classification_head(pooled)  # (B, num_classes)
        
        # Task 2: Prediction
        future_position = self.prediction_head(pooled)  # (B, 3)
        
        return {
            'classification': class_logits,
            'prediction': future_position
        }
    
    def __repr__(self):
        """String representation."""
        return (
            f"UnifiedModel(\n"
            f"  extractor={self.extractor},\n"
            f"  num_classes={self.num_classes},\n"
            f"  hidden_dim={self.hidden_dim}\n"
            f")"
        )


def compute_loss(
    outputs: dict,
    targets: dict,
    class_weight: float = 1.0,
    pred_weight: float = 1.0
) -> tuple:
    """
    Compute multi-task loss.
    
    Args:
        outputs (dict): Model outputs with keys:
            - 'classification': (B, num_classes) logits
            - 'prediction': (B, 3) predicted positions
        
        targets (dict): Ground truth with keys:
            - 'class_label': (B,) class indices (long tensor)
            - 'future_position': (B, 3) true future positions
        
        class_weight (float): Weight for classification loss (default: 1.0)
        pred_weight (float): Weight for prediction loss (default: 1.0)
    
    Returns:
        tuple: (total_loss, classification_loss, prediction_loss)
            - total_loss: Weighted sum of both losses
            - classification_loss: Cross-entropy loss
            - prediction_loss: MSE loss
    
    Example:
        >>> outputs = {
        ...     'classification': torch.randn(2, 4),
        ...     'prediction': torch.randn(2, 3)
        ... }
        >>> targets = {
        ...     'class_label': torch.tensor([0, 2]),
        ...     'future_position': torch.randn(2, 3)
        ... }
        >>> total_loss, class_loss, pred_loss = compute_loss(outputs, targets)
    """
    # Classification loss (Cross-Entropy)
    class_logits = outputs['classification']
    class_labels = targets['class_label']
    classification_loss = F.cross_entropy(class_logits, class_labels)
    
    # Prediction loss (MSE)
    predictions = outputs['prediction']
    true_positions = targets['future_position']
    prediction_loss = F.mse_loss(predictions, true_positions)
    
    # Total loss (weighted sum)
    total_loss = class_weight * classification_loss + pred_weight * prediction_loss
    
    return total_loss, classification_loss, prediction_loss


def accuracy(outputs: dict, targets: dict) -> float:
    """
    Compute classification accuracy.
    
    Args:
        outputs (dict): Model outputs with 'classification' key
        targets (dict): Ground truth with 'class_label' key
    
    Returns:
        float: Classification accuracy (0.0 to 1.0)
    """
    class_logits = outputs['classification']
    class_labels = targets['class_label']
    
    predictions = torch.argmax(class_logits, dim=1)
    correct = (predictions == class_labels).float()
    
    return correct.mean().item()


def prediction_error(outputs: dict, targets: dict) -> float:
    """
    Compute prediction RMSE (Root Mean Square Error).
    
    Args:
        outputs (dict): Model outputs with 'prediction' key
        targets (dict): Ground truth with 'future_position' key
    
    Returns:
        float: RMSE of position predictions
    """
    predictions = outputs['prediction']
    true_positions = targets['future_position']
    
    mse = F.mse_loss(predictions, true_positions)
    rmse = torch.sqrt(mse)
    
    return rmse.item()


if __name__ == "__main__":
    print("=" * 70)
    print("Unified Multi-Task Model - Demo")
    print("=" * 70)
    
    # Import extractor
    from base_extractor import DummyExtractor
    
    # Create model
    print("\n1. Creating model...")
    extractor = DummyExtractor(dim=256)
    model = UnifiedModel(extractor, num_classes=4, hidden_dim=512)
    print(f"✅ Model created:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Forward pass
    print("\n2. Testing forward pass...")
    video = torch.randn(2, 16, 3, 64, 64)
    print(f"Input shape: {video.shape}")
    
    outputs = model(video)
    print(f"Classification output: {outputs['classification'].shape}")
    print(f"Prediction output: {outputs['prediction'].shape}")
    print("✅ Forward pass successful!")
    
    # Loss computation
    print("\n3. Testing loss computation...")
    targets = {
        'class_label': torch.tensor([0, 2]),
        'future_position': torch.randn(2, 3)
    }
    
    total_loss, class_loss, pred_loss = compute_loss(outputs, targets)
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Classification loss: {class_loss.item():.4f}")
    print(f"Prediction loss: {pred_loss.item():.4f}")
    print("✅ Loss computation successful!")
    
    # Metrics
    print("\n4. Testing metrics...")
    acc = accuracy(outputs, targets)
    rmse = prediction_error(outputs, targets)
    print(f"Classification accuracy: {acc:.2%}")
    print(f"Prediction RMSE: {rmse:.4f}")
    print("✅ Metrics computation successful!")
    
    # Backward pass
    print("\n5. Testing backward pass...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    print("✅ Backward pass successful!")
    
    print("\n" + "=" * 70)
    print("Demo complete. Ready for TDD GREEN phase verification.")
    print("=" * 70)

