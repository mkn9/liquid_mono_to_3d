#!/usr/bin/env python3
"""
Branch 1: Baseline Future Prediction
====================================
Frozen MagVit + Transformer (no motion model)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys
import json
from datetime import datetime
from typing import Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from complete_magvit_loader import CompleteMagVit
from shared_utilities import (
    setup_logging, save_results, create_video_dataset_for_prediction,
    SimpleTransformer, compute_metrics, run_test_suite
)


class BaselineFuturePrediction(nn.Module):
    """Baseline: Frozen MagVit + Transformer."""
    
    def __init__(
        self,
        magvit_checkpoint: str,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 12,
        device: str = 'cuda'
    ):
        super().__init__()
        self.device = device
        
        # Frozen MagVit
        self.magvit = CompleteMagVit(
            checkpoint_path=magvit_checkpoint,
            embed_dim=embed_dim,
            device=device
        )
        # Freeze MagVit
        for param in self.magvit.parameters():
            param.requires_grad = False
        self.magvit.eval()
        
        # Trainable Transformer
        self.transformer = SimpleTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
        
        self.to(device)
    
    def forward(self, past_frames: torch.Tensor, num_future_frames: int = 25, return_latents: bool = False):
        """
        Predict future frames from past frames.
        
        Args:
            past_frames: (B, 3, T_past, H, W) past video frames
            num_future_frames: number of frames to predict
        Returns:
            pred_frames: (B, 3, T_future, H, W) predicted future frames
        """
        B = past_frames.shape[0]
        
        # Encode past frames with frozen MagVit
        with torch.no_grad():
            z_past, indices_past = self.magvit.encode(past_frames)
        
        # Spatial pooling to reduce dimensions: (B, C, T, H, W) -> (B, C, T, 1, 1)
        B, C, T, H, W = z_past.shape
        spatial_pool = nn.AdaptiveAvgPool3d((T, 1, 1)).to(self.device)
        z_pooled = spatial_pool(z_past)  # (B, C, T, 1, 1)
        
        # Flatten for transformer: (B, C, T, 1, 1) -> (B, T, C)
        z_seq = z_pooled.squeeze(-1).squeeze(-1).permute(0, 2, 1)  # (B, T, C)
        
        # Predict future features with transformer
        z_future_seq = self.transformer(z_seq)  # (B, T, C)
        
        # Take last num_future_frames timesteps
        z_future_seq = z_future_seq[:, -num_future_frames:, :]  # (B, T_future, C)
        
        # Reshape back and broadcast to spatial dimensions: (B, T_future, C) -> (B, C, T_future, H, W)
        z_future = z_future_seq.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # (B, C, T_future, 1, 1)
        z_future = z_future.expand(-1, -1, -1, H, W)  # (B, C, T_future, H, W)
        
        # Return latents for training (has gradients)
        if return_latents:
            return z_future
        
        # Decode with frozen MagVit decoder
        with torch.no_grad():
            pred_frames = self.magvit.decode(z_future)
        
        return pred_frames


def train_baseline(
    model: BaselineFuturePrediction,
    videos: torch.Tensor,
    num_epochs: int,
    batch_size: int,
    lr: float,
    logger
) -> Dict[str, Any]:
    """Train baseline model."""
    
    logger.info("=" * 60)
    logger.info("Training Baseline Model")
    logger.info("=" * 60)
    
    optimizer = optim.Adam(model.transformer.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    num_samples = videos.shape[0]
    past_length = 25
    future_length = 25
    
    results = {
        'epochs': [],
        'losses': [],
        'metrics': []
    }
    
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # Mini-batch training
        for i in range(0, num_samples, batch_size):
            batch_videos = videos[i:i+batch_size]
            
            # Split into past and future
            past_frames = batch_videos[:, :, :past_length, :, :]
            future_frames = batch_videos[:, :, past_length:past_length+future_length, :, :]
            
            # Get target latents
            with torch.no_grad():
                z_target, _ = model.magvit.encode(future_frames)
            
            # Forward pass - compute loss in latent space
            optimizer.zero_grad()
            z_pred = model(past_frames, num_future_frames=future_length, return_latents=True)
            loss = criterion(z_pred, z_target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        results['epochs'].append(epoch)
        results['losses'].append(avg_loss)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                test_batch = videos[:4]
                past_test = test_batch[:, :, :past_length, :, :]
                future_test = test_batch[:, :, past_length:past_length+future_length, :, :]
                pred_test = model(past_test, num_future_frames=future_length)
                metrics = compute_metrics(pred_test, future_test, logger)
                results['metrics'].append(metrics)
            model.train()
        
        # Save checkpoint every 10 epochs and at the end
        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            checkpoint_dir = Path('/home/ubuntu/mono_to_3d/experiments/future-prediction/output/baseline/checkpoints')
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f'baseline_epoch{epoch+1}.pth'
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'results': results,
                'config': {
                    'past_length': past_length,
                    'future_length': future_length,
                    'batch_size': batch_size,
                    'learning_rate': lr
                }
            }
            
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"✅ Saved checkpoint: {checkpoint_path.name}")
            
            # Also save the best model (lowest loss)
            if epoch == 0 or avg_loss < min(results['losses'][:-1] if len(results['losses']) > 1 else [float('inf')]):
                best_path = checkpoint_dir / 'baseline_best.pth'
                torch.save(checkpoint, best_path)
                logger.info(f"✅ Saved best model (loss: {avg_loss:.6f})"  )
    
    logger.info("✅ Training completed")
    return results


def test_magvit_loading(logger):
    """Test 1: MagVit loading."""
    logger.info("Testing MagVit loading...")
    
    checkpoint_path = "/home/ubuntu/magvit_weights/video_128_262144.ckpt"
    if not Path(checkpoint_path).exists():
        logger.warning(f"⚠️  Checkpoint not found: {checkpoint_path}")
        checkpoint_path = None
    
    magvit = CompleteMagVit(checkpoint_path=checkpoint_path, device='cuda')
    
    # Test forward pass
    dummy_video = torch.rand(1, 3, 16, 128, 128, device='cuda')
    with torch.no_grad():
        recon = magvit.reconstruct(dummy_video)
    
    assert recon.shape == dummy_video.shape, "Shape mismatch"
    logger.info(f"✅ MagVit loaded, reconstruction shape: {recon.shape}")
    
    return {'status': 'passed', 'output_shape': str(recon.shape)}


def test_baseline_model_creation(logger):
    """Test 2: Baseline model creation."""
    logger.info("Testing baseline model creation...")
    
    checkpoint_path = "/home/ubuntu/magvit_weights/video_128_262144.ckpt"
    if not Path(checkpoint_path).exists():
        checkpoint_path = None
    
    model = BaselineFuturePrediction(
        magvit_checkpoint=checkpoint_path,
        device='cuda'
    )
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    logger.info(f"✅ Model created")
    logger.info(f"   Trainable params: {trainable_params/1e6:.2f}M")
    logger.info(f"   Frozen params: {frozen_params/1e6:.2f}M")
    
    return {
        'status': 'passed',
        'trainable_params': trainable_params,
        'frozen_params': frozen_params
    }


def test_forward_pass(logger):
    """Test 3: Forward pass."""
    logger.info("Testing forward pass...")
    
    checkpoint_path = "/home/ubuntu/magvit_weights/video_128_262144.ckpt"
    if not Path(checkpoint_path).exists():
        checkpoint_path = None
    
    model = BaselineFuturePrediction(magvit_checkpoint=checkpoint_path, device='cuda')
    
    # Create dummy input
    past_frames = torch.rand(2, 3, 25, 128, 128, device='cuda')
    
    # Forward pass
    with torch.no_grad():
        pred_frames = model(past_frames, num_future_frames=25)
    
    assert pred_frames.shape == (2, 3, 25, 128, 128), f"Shape mismatch: {pred_frames.shape}"
    logger.info(f"✅ Forward pass successful, output shape: {pred_frames.shape}")
    
    return {'status': 'passed', 'output_shape': str(pred_frames.shape)}


def test_training_step(logger):
    """Test 4: Single training step."""
    logger.info("Testing training step...")
    
    checkpoint_path = "/home/ubuntu/magvit_weights/video_128_262144.ckpt"
    if not Path(checkpoint_path).exists():
        checkpoint_path = None
    
    model = BaselineFuturePrediction(magvit_checkpoint=checkpoint_path, device='cuda')
    model.train()  # Ensure model is in training mode
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    criterion = nn.MSELoss()
    
    # Create dummy data
    videos = torch.rand(4, 3, 50, 128, 128, device='cuda')
    past_frames = videos[:, :, :25, :, :]
    future_frames = videos[:, :, 25:50, :, :]
    
    # Get target latents
    with torch.no_grad():
        z_target, _ = model.magvit.encode(future_frames)
    
    # Training step - compute loss in latent space
    optimizer.zero_grad()
    z_pred = model(past_frames, num_future_frames=25, return_latents=True)
    loss = criterion(z_pred, z_target)
    loss.backward()
    optimizer.step()
    
    logger.info(f"✅ Training step successful, loss: {loss.item():.6f}")
    
    return {'status': 'passed', 'loss': loss.item()}


def main():
    """Main execution."""
    # Setup
    output_dir = Path('/home/ubuntu/mono_to_3d/experiments/future-prediction/output/baseline')
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging('baseline', output_dir)
    
    logger.info("=" * 60)
    logger.info("Branch 1: Baseline Future Prediction")
    logger.info("=" * 60)
    logger.info("Configuration:")
    logger.info("  - MagVit: FROZEN (pretrained)")
    logger.info("  - Transformer: TRAIN (12 layers, 8 heads)")
    logger.info("  - Motion model: None")
    logger.info("=" * 60)
    
    # Run test suite
    test_results = run_test_suite([
        test_magvit_loading,
        test_baseline_model_creation,
        test_forward_pass,
        test_training_step
    ], logger)
    
    # If tests passed, run training
    if test_results['failed'] == 0:
        logger.info("\n✅ All tests passed! Starting training...")
        
        # Create dataset
        logger.info("Creating synthetic trajectory dataset...")
        videos = create_video_dataset_for_prediction(num_samples=100, device='cuda')
        logger.info(f"Dataset created: {videos.shape}")
        
        # Create model
        checkpoint_path = "/home/ubuntu/magvit_weights/video_128_262144.ckpt"
        if not Path(checkpoint_path).exists():
            checkpoint_path = None
        
        model = BaselineFuturePrediction(
            magvit_checkpoint=checkpoint_path,
            device='cuda'
        )
        
        # Train
        training_results = train_baseline(
            model=model,
            videos=videos,
            num_epochs=50,
            batch_size=4,
            lr=1e-4,
            logger=logger
        )
        
        # Save results
        final_results = {
            'branch': 'baseline',
            'timestamp': datetime.now().isoformat(),
            'test_results': test_results,
            'training_results': training_results,
            'config': {
                'magvit_frozen': True,
                'motion_model': None,
                'num_epochs': 50,
                'batch_size': 4,
                'learning_rate': 1e-4
            }
        }
        
        save_results(final_results, output_dir, 'baseline')
        logger.info("=" * 60)
        logger.info("✅ BASELINE TRAINING COMPLETE")
        logger.info("=" * 60)
    else:
        logger.error("❌ Tests failed. Aborting training.")
        final_results = {
            'branch': 'baseline',
            'timestamp': datetime.now().isoformat(),
            'test_results': test_results,
            'status': 'failed_tests'
        }
        save_results(final_results, output_dir, 'baseline')


if __name__ == '__main__':
    main()

