"""
Liquid Dual-Modal Fusion Layer
Worker 1 Implementation: Dynamic 2D+3D fusion using Liquid Neural Networks
"""
import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "liquid_models"))
from liquid_cell import LiquidCell


class LiquidDualModalFusion(nn.Module):
    """
    Dynamic fusion of 2D visual features (MagVIT) and 3D trajectory features
    using Liquid Neural Networks for temporal consistency.
    
    Replaces static linear fusion with continuous-time ODE dynamics.
    """
    
    def __init__(self, input_2d_dim=512, input_3d_dim=256, output_llm_dim=4096, dt=0.02):
        super().__init__()
        
        self.output_llm_dim = output_llm_dim
        
        # Adapters to project 2D and 3D features to LLM dimension
        self.adapter_2d = nn.Linear(input_2d_dim, output_llm_dim)
        self.adapter_3d = nn.Linear(input_3d_dim, output_llm_dim)
        
        # Liquid cell for dynamic fusion
        self.liquid_fusion = LiquidCell(
            input_size=output_llm_dim * 2,  # Concatenated 2D + 3D
            hidden_size=output_llm_dim,
            dt=dt
        )
        
        # Hidden state buffer (persistent across forward passes)
        self.register_buffer('h_fusion', None)
    
    def forward(self, features_2d, features_3d, reset_state=False):
        """
        Args:
            features_2d: (B, 512) - MagVIT embeddings
            features_3d: (B, 256) - 3D trajectory features
            reset_state: If True, reset hidden state to zero
            
        Returns:
            fused_embedding: (B, 4096) - Fused LLM-compatible embedding
        """
        # Project to LLM dimension
        emb_2d = self.adapter_2d(features_2d)  # (B, 4096)
        emb_3d = self.adapter_3d(features_3d)  # (B, 4096)
        
        # Concatenate modalities
        combined = torch.cat([emb_2d, emb_3d], dim=-1)  # (B, 8192)
        
        # Initialize or reset hidden state
        if self.h_fusion is None or reset_state:
            self.h_fusion = torch.zeros(
                combined.shape[0], 
                self.output_llm_dim,
                device=combined.device,
                dtype=combined.dtype
            )
        
        # Liquid dynamics for fusion
        self.h_fusion = self.liquid_fusion(combined, self.h_fusion)
        
        return self.h_fusion
    
    def reset_hidden_state(self):
        """Manually reset hidden state."""
        self.h_fusion = None
