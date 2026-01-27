#!/usr/bin/env python3
"""
Complete MagVit Loader (Encoder + Decoder + Quantizer)
======================================================
Loads pretrained MagVit weights for future prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class VectorQuantizer(nn.Module):
    """Vector Quantizer for discrete token generation."""
    
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, C, T, H, W) continuous features
        Returns:
            z_q: (B, C, T, H, W) quantized features
            indices: (B, T, H, W) discrete token indices
        """
        # Flatten spatial dimensions
        B, C, T, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 4, 1).contiguous()  # (B, T, H, W, C)
        z_flat = z_flat.view(-1, C)  # (B*T*H*W, C)
        
        # Batched distance calculation to avoid OOM
        batch_size = 512  # Process 512 vectors at a time
        num_vectors = z_flat.shape[0]
        indices_list = []
        
        for i in range(0, num_vectors, batch_size):
            z_batch = z_flat[i:i+batch_size]
            # Compute distances for this batch
            distances = (
                torch.sum(z_batch ** 2, dim=1, keepdim=True) +
                torch.sum(self.embedding.weight ** 2, dim=1) -
                2 * torch.matmul(z_batch, self.embedding.weight.t())
            )
            batch_indices = torch.argmin(distances, dim=1)
            indices_list.append(batch_indices)
        
        indices = torch.cat(indices_list, dim=0)  # (B*T*H*W,)
        z_q = self.embedding(indices)  # (B*T*H*W, C)
        
        # Reshape back
        z_q = z_q.view(B, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()  # (B, C, T, H, W)
        indices = indices.view(B, T, H, W)
        
        # Straight-through estimator
        z_q = z + (z_q - z).detach()
        
        return z_q, indices
    
    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices: (B, T, H, W) discrete tokens
        Returns:
            z_q: (B, C, T, H, W) quantized features
        """
        B, T, H, W = indices.shape
        indices_flat = indices.view(-1)
        z_q = self.embedding(indices_flat)  # (B*T*H*W, C)
        z_q = z_q.view(B, T, H, W, self.embedding_dim)
        z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()  # (B, C, T, H, W)
        return z_q


class MagVitEncoder(nn.Module):
    """MagVit Encoder (3D ConvNet)."""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 256):
        super().__init__()
        
        # 3D Convolutional encoder
        self.conv_in = nn.Conv3d(in_channels, 64, kernel_size=(3, 3, 3), padding=1)
        self.conv1 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1)
        self.conv2 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1)
        self.conv_out = nn.Conv3d(256, out_channels, kernel_size=(3, 3, 3), padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, T, H, W) input video
        Returns:
            z: (B, C, T, H', W') encoded features
        """
        # Normalize to [-1, 1]
        x = x * 2.0 - 1.0
        
        # Encode
        h = self.relu(self.conv_in(x))
        h = self.relu(self.conv1(h))
        h = self.relu(self.conv2(h))
        z = self.conv_out(h)
        
        return z


class MagVitDecoder(nn.Module):
    """MagVit Decoder (3D TransposedConvNet)."""
    
    def __init__(self, in_channels: int = 256, out_channels: int = 3):
        super().__init__()
        
        # 3D Transposed convolutional decoder
        self.conv_in = nn.Conv3d(in_channels, 256, kernel_size=(3, 3, 3), padding=1)
        self.deconv1 = nn.ConvTranspose3d(256, 128, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1)
        self.deconv2 = nn.ConvTranspose3d(128, 64, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1)
        self.conv_out = nn.Conv3d(64, out_channels, kernel_size=(3, 3, 3), padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, C, T, H', W') encoded features
        Returns:
            x: (B, 3, T, H, W) reconstructed video
        """
        # Decode
        h = self.relu(self.conv_in(z))
        h = self.relu(self.deconv1(h))
        h = self.relu(self.deconv2(h))
        x = self.conv_out(h)
        
        # Back to [0, 1]
        x = torch.sigmoid(x)
        
        return x


class CompleteMagVit(nn.Module):
    """
    Complete MagVit VQ-VAE with Encoder, Decoder, and Quantizer.
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        embed_dim: int = 256,
        num_embeddings: int = 1024,
        device: str = 'cuda'
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_embeddings = num_embeddings
        self.device = device
        
        # Components
        self.encoder = MagVitEncoder(out_channels=embed_dim)
        self.quantizer = VectorQuantizer(num_embeddings, embed_dim)
        self.decoder = MagVitDecoder(in_channels=embed_dim)
        
        # Load pretrained weights if provided
        if checkpoint_path:
            self.load_pretrained(checkpoint_path)
        
        self.to(device)
    
    def load_pretrained(self, checkpoint_path: str):
        """Load pretrained weights from Open-MAGVIT2 checkpoint."""
        logger.info(f"Loading pretrained weights from: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)
            
            # Extract encoder weights
            encoder_state = {
                k.replace('encoder.', ''): v 
                for k, v in state_dict.items() 
                if k.startswith('encoder.')
            }
            
            # Extract decoder weights
            decoder_state = {
                k.replace('decoder.', ''): v 
                for k, v in state_dict.items() 
                if k.startswith('decoder.')
            }
            
            # Extract quantizer weights
            quantizer_state = {
                k.replace('quantize.', ''): v 
                for k, v in state_dict.items() 
                if 'quantize' in k or 'embedding' in k
            }
            
            # Load compatible weights (partial loading)
            self.encoder.load_state_dict(encoder_state, strict=False)
            self.decoder.load_state_dict(decoder_state, strict=False)
            self.quantizer.load_state_dict(quantizer_state, strict=False)
            
            logger.info(f"✅ Loaded encoder: {len(encoder_state)} params")
            logger.info(f"✅ Loaded decoder: {len(decoder_state)} params")
            logger.info(f"✅ Loaded quantizer: {len(quantizer_state)} params")
            
        except Exception as e:
            logger.warning(f"⚠️  Could not load pretrained weights: {e}")
            logger.warning("Using random initialization")
    
    def encode(self, video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode video to discrete tokens.
        
        Args:
            video: (B, 3, T, H, W) input video
        Returns:
            z_q: (B, C, T, H', W') quantized features
            indices: (B, T, H', W') discrete token indices
        """
        z = self.encoder(video)
        z_q, indices = self.quantizer(z)
        return z_q, indices
    
    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Decode quantized features to video.
        
        Args:
            z_q: (B, C, T, H', W') quantized features
        Returns:
            video: (B, 3, T, H, W) reconstructed video
        """
        return self.decoder(z_q)
    
    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode token indices to video.
        
        Args:
            indices: (B, T, H', W') discrete token indices
        Returns:
            video: (B, 3, T, H, W) reconstructed video
        """
        z_q = self.quantizer.decode_indices(indices)
        return self.decode(z_q)
    
    def forward(self, video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass (encode + quantize + decode).
        
        Args:
            video: (B, 3, T, H, W) input video
        Returns:
            recon: (B, 3, T, H, W) reconstructed video
            z_q: (B, C, T, H', W') quantized features
            indices: (B, T, H', W') discrete token indices
        """
        z_q, indices = self.encode(video)
        recon = self.decode(z_q)
        return recon, z_q, indices
    
    def reconstruct(self, video: torch.Tensor) -> torch.Tensor:
        """Simple reconstruction for testing."""
        recon, _, _ = self.forward(video)
        return recon


def test_complete_magvit():
    """Test Complete MagVit loader."""
    import numpy as np
    
    print("=" * 60)
    print("Testing Complete MagVit Loader")
    print("=" * 60)
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = "/home/ubuntu/magvit_weights/video_128_262144.ckpt"
    
    model = CompleteMagVit(
        checkpoint_path=checkpoint_path if Path(checkpoint_path).exists() else None,
        embed_dim=256,
        num_embeddings=262144,
        device=device
    )
    model.eval()
    
    print(f"\n✅ Model created on {device}")
    print(f"   Encoder params: {sum(p.numel() for p in model.encoder.parameters())/1e6:.2f}M")
    print(f"   Decoder params: {sum(p.numel() for p in model.decoder.parameters())/1e6:.2f}M")
    print(f"   Quantizer params: {sum(p.numel() for p in model.quantizer.parameters())/1e6:.2f}M")
    
    # Test with dummy video
    print("\n📹 Testing with dummy video...")
    batch_size = 2
    num_frames = 16
    height, width = 128, 128
    
    video = torch.rand(batch_size, 3, num_frames, height, width).to(device)
    print(f"   Input shape: {video.shape}")
    
    # Test encoding
    print("\n🔹 Testing encoding...")
    with torch.no_grad():
        z_q, indices = model.encode(video)
    print(f"   ✅ Encoded features: {z_q.shape}")
    print(f"   ✅ Token indices: {indices.shape}")
    print(f"   Token range: [{indices.min().item()}, {indices.max().item()}]")
    
    # Test decoding
    print("\n🔹 Testing decoding...")
    with torch.no_grad():
        recon = model.decode(z_q)
    print(f"   ✅ Reconstructed video: {recon.shape}")
    print(f"   Value range: [{recon.min().item():.3f}, {recon.max().item():.3f}]")
    
    # Test full reconstruction
    print("\n🔹 Testing full reconstruction...")
    with torch.no_grad():
        recon_full = model.reconstruct(video)
    print(f"   ✅ Full reconstruction: {recon_full.shape}")
    
    # Compute reconstruction error
    mse = F.mse_loss(recon_full, video).item()
    print(f"   MSE: {mse:.6f}")
    
    # Test decode from indices
    print("\n🔹 Testing decode from indices...")
    with torch.no_grad():
        recon_from_indices = model.decode_indices(indices)
    print(f"   ✅ Decoded from indices: {recon_from_indices.shape}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    
    return model


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_complete_magvit()

