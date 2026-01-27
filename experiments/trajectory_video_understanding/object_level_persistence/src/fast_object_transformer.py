"""
Fast object-aware transformer for quick attention validation.

Minimal implementation focused on:
1. Processing object tokens
2. Extracting attention weights
3. Classifying persistent vs transient
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class FastObjectTransformer(nn.Module):
    """Minimal transformer for object-level persistence detection."""
    
    def __init__(
        self,
        feature_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)  # persistent (0) or transient (1)
        )
        
        # Store attention weights for visualization
        self.attention_weights = None
        
        # Register hook to capture attention
        self._register_attention_hook()
    
    def _register_attention_hook(self):
        """Register hook to capture attention weights from first layer."""
        def hook(module, input, output):
            # TransformerEncoderLayer returns (output, attention_weights) when needed
            # We'll capture during forward pass
            pass
        
        # We'll extract attention manually in forward()
    
    def forward(
        self, 
        object_tokens: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through transformer.
        
        Args:
            object_tokens: (batch, seq_len, feature_dim) object token features
            src_key_padding_mask: (batch, seq_len) padding mask (True for padding)
            return_attention: Whether to return attention weights
            
        Returns:
            logits: (batch, seq_len, 2) classification logits per object
            attention_weights: (batch, num_heads, seq_len, seq_len) if return_attention=True
        """
        batch_size, seq_len, _ = object_tokens.shape
        
        if return_attention:
            # Use MultiheadAttention directly to get weights
            attention_weights_list = []
            x = object_tokens
            
            # Process through transformer layers, capturing attention
            for layer in self.transformer.layers:
                # Self-attention with attention weights
                attn_output, attn_weights = layer.self_attn(
                    x, x, x,
                    key_padding_mask=src_key_padding_mask,
                    need_weights=True,
                    average_attn_weights=False  # Get per-head weights
                )
                
                # Store first layer attention
                if len(attention_weights_list) == 0:
                    attention_weights_list.append(attn_weights)
                
                # Complete the layer forward pass
                x = layer.norm1(x + layer.dropout1(attn_output))
                ff_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
                x = layer.norm2(x + layer.dropout2(ff_output))
            
            transformed = x
            attention_weights = attention_weights_list[0]  # (batch, num_heads, seq_len, seq_len)
        else:
            # Standard forward without attention
            transformed = self.transformer(
                object_tokens,
                src_key_padding_mask=src_key_padding_mask
            )
            attention_weights = None
        
        # Classify each object
        logits = self.classifier(transformed)  # (batch, seq_len, 2)
        
        return logits, attention_weights
    
    def predict(
        self,
        object_tokens: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict with attention weights for visualization.
        
        Args:
            object_tokens: (batch, seq_len, feature_dim)
            src_key_padding_mask: (batch, seq_len) padding mask
            
        Returns:
            predictions: (batch, seq_len) predicted class (0=persistent, 1=transient)
            confidence: (batch, seq_len) prediction confidence
            attention: (batch, num_heads, seq_len, seq_len) attention weights
        """
        with torch.no_grad():
            logits, attention = self.forward(
                object_tokens,
                src_key_padding_mask=src_key_padding_mask,
                return_attention=True
            )
            
            # Get predictions
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
            confidence = torch.max(probs, dim=-1)[0]
            
            return predictions, confidence, attention

