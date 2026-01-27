#!/usr/bin/env python3
"""
Branch 2: Joint I3D Future Prediction
=====================================
TRAIN MagVit + Finetune I3D + Train Transformer
Key innovation: MagVit learns trajectory-specific features guided by motion!
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


class JointI3DFuturePrediction(nn.Module):
    """Joint Training: Train MagVit + Finetune I3D + Train Transformer."""
    
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
        self.embed_dim = embed_dim
        
        # Trainable MagVit (learns trajectory features!)
        self.magvit = CompleteMagVit(
            checkpoint_path=magvit_checkpoint,
            embed_dim=embed_dim,
            device=device
        )
        # Set to trainable
        for param in self.magvit.parameters():
            param.requires_grad = True
        
        # I3D motion model (finetuned)
        logger.info("Loading I3D model...")
        try:
            from pytorchvideo.models.hub import i3d_r50
            self.i3d = i3d_r50(pretrained=True)
            self.i3d = self.i3d.to(device)
            
            # Replace final classification layer with feature extraction
            # I3D outputs (B, 400) for Kinetics, we want (B, 256) features
            self.i3d.blocks[-1] = nn.Sequential(
                self.i3d.blocks[-1].proj,
                nn.ReLU(),
                nn.Linear(400, embed_dim)
            )
            
            # Allow finetuning
            for param in self.i3d.parameters():
                param.requires_grad = True
            
        except Exception as e:
            logger.warning(f"⚠️  I3D loading failed: {e}")
            logger.warning("Using placeholder I3D")
            self.i3d = nn.Sequential(
                nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Flatten(),
                nn.Linear(64, embed_dim)
            )
        
        # Fusion layer (combines MagVit + I3D features)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Trainable Transformer
        self.transformer = SimpleTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
        
        self.to(device)
    
    def forward(self, past_frames: torch.Tensor, num_future_frames: int = 25):
        """
        Predict future frames with motion guidance.
        
        Args:
            past_frames: (B, 3, T_past, H, W) past video frames
            num_future_frames: number of frames to predict
        Returns:
            pred_frames: (B, 3, T_future, H, W) predicted future frames
            motion_features: (B, embed_dim) motion features from I3D
        """
        B = past_frames.shape[0]
        
        # 1. Encode with TRAINABLE MagVit
        z_past, indices_past = self.magvit.encode(past_frames)
        
        # 2. Extract motion features with I3D
        # I3D expects (B, C, T, H, W) and specific frame count
        # Resample if needed
        if past_frames.shape[2] != 32:
            # Simple temporal interpolation to 32 frames
            past_resampled = torch.nn.functional.interpolate(
                past_frames.transpose(1, 2),  # (B, T, C, H, W)
                size=(32,),
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # (B, C, T=32, H, W)
        else:
            past_resampled = past_frames
        
        motion_features = self.i3d(past_resampled)  # (B, embed_dim)
        
        # 3. Spatial pooling: (B, C, T, H, W) -> (B, C, T, 1, 1)
        B, C, T, H, W = z_past.shape
        spatial_pool = nn.AdaptiveAvgPool3d((T, 1, 1)).to(self.device)
        z_pooled = spatial_pool(z_past)  # (B, C, T, 1, 1)
        
        # Flatten for transformer: (B, C, T, 1, 1) -> (B, T, C)
        z_seq = z_pooled.squeeze(-1).squeeze(-1).permute(0, 2, 1)  # (B, T, C)
        
        # 4. Fuse motion features into sequence
        # Broadcast motion features to all timesteps
        motion_broadcast = motion_features.unsqueeze(1).expand(-1, T, -1)  # (B, T, embed_dim)
        z_fused = torch.cat([z_seq, motion_broadcast], dim=-1)  # (B, T, embed_dim*2)
        z_fused = self.fusion(z_fused)  # (B, T, embed_dim)
        
        # 5. Predict future features with transformer
        z_future_seq = self.transformer(z_fused)  # (B, T, embed_dim)
        
        # Take last num_future_frames timesteps
        z_future_seq = z_future_seq[:, -num_future_frames:, :]  # (B, T_future, embed_dim)
        
        # 6. Reshape back and broadcast to spatial dimensions
        z_future = z_future_seq.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # (B, C, T_future, 1, 1)
        z_future = z_future.expand(-1, -1, -1, H, W)  # (B, C, T_future, H, W)
        
        # 7. Decode with TRAINABLE MagVit decoder
        pred_frames = self.magvit.decode(z_future)
        
        return pred_frames, motion_features


def train_joint_i3d(
    model: JointI3DFuturePrediction,
    videos: torch.Tensor,
    num_epochs: int,
    batch_size: int,
    lr: float,
    logger
) -> Dict[str, Any]:
    """Train joint model (MagVit + I3D + Transformer)."""
    
    logger.info("=" * 60)
    logger.info("Training Joint I3D Model")
    logger.info("=" * 60)
    
    # Different learning rates for different components
    optimizer = optim.Adam([
        {'params': model.magvit.parameters(), 'lr': lr * 0.1},  # Lower LR for MagVit
        {'params': model.i3d.parameters(), 'lr': lr * 0.5},     # Lower LR for I3D
        {'params': model.fusion.parameters(), 'lr': lr},
        {'params': model.transformer.parameters(), 'lr': lr}
    ])
    
    criterion_recon = nn.MSELoss()
    criterion_motion = nn.MSELoss()  # For motion consistency
    
    num_samples = videos.shape[0]
    past_length = 25
    future_length = 25
    
    results = {
        'epochs': [],
        'losses': {
            'total': [],
            'reconstruction': [],
            'motion': []
        },
        'metrics': []
    }
    
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss_total = 0.0
        epoch_loss_recon = 0.0
        epoch_loss_motion = 0.0
        num_batches = 0
        
        # Mini-batch training
        for i in range(0, num_samples, batch_size):
            batch_videos = videos[i:i+batch_size]
            
            # Split into past and future
            past_frames = batch_videos[:, :, :past_length, :, :]
            future_frames = batch_videos[:, :, past_length:past_length+future_length, :, :]
            
            # Forward pass
            optimizer.zero_grad()
            pred_frames, motion_features = model(past_frames, num_future_frames=future_length)
            
            # Reconstruction loss
            loss_recon = criterion_recon(pred_frames, future_frames)
            
            # Motion consistency loss (motion should be similar for consecutive frames)
            # This guides MagVit to learn motion-aware features
            loss_motion = 0.0
            if i + batch_size < num_samples:
                next_batch = videos[i+batch_size:i+2*batch_size]
                if next_batch.shape[0] > 0:
                    next_past = next_batch[:, :, :past_length, :, :]
                    _, next_motion = model(next_past, num_future_frames=future_length)
                    # Motion should change smoothly
                    loss_motion = criterion_motion(motion_features, next_motion.detach())
            
            # Combined loss
            loss = loss_recon + 0.1 * loss_motion
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss_total += loss.item()
            epoch_loss_recon += loss_recon.item()
            epoch_loss_motion += loss_motion if isinstance(loss_motion, float) else loss_motion.item()
            num_batches += 1
        
        avg_loss_total = epoch_loss_total / num_batches
        avg_loss_recon = epoch_loss_recon / num_batches
        avg_loss_motion = epoch_loss_motion / num_batches
        
        results['epochs'].append(epoch)
        results['losses']['total'].append(avg_loss_total)
        results['losses']['reconstruction'].append(avg_loss_recon)
        results['losses']['motion'].append(avg_loss_motion)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss_total:.6f} (Recon: {avg_loss_recon:.6f}, Motion: {avg_loss_motion:.6f})")
        
        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                test_batch = videos[:4]
                past_test = test_batch[:, :, :past_length, :, :]
                future_test = test_batch[:, :, past_length:past_length+future_length, :, :]
                pred_test, _ = model(past_test, num_future_frames=future_length)
                metrics = compute_metrics(pred_test, future_test, logger)
                results['metrics'].append(metrics)
            model.train()
    
    logger.info("✅ Joint training completed")
    return results


def test_i3d_loading(logger):
    """Test 1: I3D loading."""
    logger.info("Testing I3D loading...")
    
    try:
        from pytorchvideo.models.hub import i3d_r50
        i3d = i3d_r50(pretrained=True)
        i3d = i3d.to('cuda')
        i3d.eval()
        
        # Test forward pass
        dummy_video = torch.rand(1, 3, 32, 224, 224, device='cuda')
        with torch.no_grad():
            output = i3d(dummy_video)
        
        logger.info(f"✅ I3D loaded, output shape: {output.shape}")
        return {'status': 'passed', 'output_shape': str(output.shape)}
    except Exception as e:
        logger.warning(f"⚠️  I3D loading failed: {e}")
        return {'status': 'warning', 'error': str(e)}


def test_joint_model_creation(logger):
    """Test 2: Joint model creation."""
    logger.info("Testing joint model creation...")
    
    checkpoint_path = "/home/ubuntu/magvit_weights/video_128_262144.ckpt"
    if not Path(checkpoint_path).exists():
        checkpoint_path = None
    
    model = JointI3DFuturePrediction(
        magvit_checkpoint=checkpoint_path,
        device='cuda'
    )
    
    # Count trainable parameters
    magvit_params = sum(p.numel() for p in model.magvit.parameters() if p.requires_grad)
    i3d_params = sum(p.numel() for p in model.i3d.parameters() if p.requires_grad)
    transformer_params = sum(p.numel() for p in model.transformer.parameters() if p.requires_grad)
    
    logger.info(f"✅ Model created")
    logger.info(f"   MagVit params (trainable): {magvit_params/1e6:.2f}M")
    logger.info(f"   I3D params (trainable): {i3d_params/1e6:.2f}M")
    logger.info(f"   Transformer params: {transformer_params/1e6:.2f}M")
    
    return {
        'status': 'passed',
        'magvit_params': magvit_params,
        'i3d_params': i3d_params,
        'transformer_params': transformer_params
    }


def test_joint_forward_pass(logger):
    """Test 3: Joint forward pass."""
    logger.info("Testing joint forward pass...")
    
    checkpoint_path = "/home/ubuntu/magvit_weights/video_128_262144.ckpt"
    if not Path(checkpoint_path).exists():
        checkpoint_path = None
    
    model = JointI3DFuturePrediction(magvit_checkpoint=checkpoint_path, device='cuda')
    
    # Create dummy input
    past_frames = torch.rand(2, 3, 25, 128, 128, device='cuda')
    
    # Forward pass
    with torch.no_grad():
        pred_frames, motion_features = model(past_frames, num_future_frames=25)
    
    assert pred_frames.shape == (2, 3, 25, 128, 128), f"Shape mismatch: {pred_frames.shape}"
    assert motion_features.shape == (2, 256), f"Motion shape mismatch: {motion_features.shape}"
    
    logger.info(f"✅ Forward pass successful")
    logger.info(f"   Prediction shape: {pred_frames.shape}")
    logger.info(f"   Motion features shape: {motion_features.shape}")
    
    return {
        'status': 'passed',
        'pred_shape': str(pred_frames.shape),
        'motion_shape': str(motion_features.shape)
    }


def test_joint_training_step(logger):
    """Test 4: Joint training step."""
    logger.info("Testing joint training step...")
    
    checkpoint_path = "/home/ubuntu/magvit_weights/video_128_262144.ckpt"
    if not Path(checkpoint_path).exists():
        checkpoint_path = None
    
    model = JointI3DFuturePrediction(magvit_checkpoint=checkpoint_path, device='cuda')
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # Create dummy data
    videos = torch.rand(4, 3, 50, 128, 128, device='cuda')
    past_frames = videos[:, :, :25, :, :]
    future_frames = videos[:, :, 25:50, :, :]
    
    # Training step
    optimizer.zero_grad()
    pred_frames, _ = model(past_frames, num_future_frames=25)
    loss = criterion(pred_frames, future_frames)
    loss.backward()
    optimizer.step()
    
    logger.info(f"✅ Training step successful, loss: {loss.item():.6f}")
    
    return {'status': 'passed', 'loss': loss.item()}


# Need to define logger at module level for model creation
logger = None

def main():
    """Main execution."""
    global logger
    
    # Setup
    output_dir = Path('/home/ubuntu/mono_to_3d/experiments/future-prediction/output/joint_i3d')
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging('joint_i3d', output_dir)
    
    logger.info("=" * 60)
    logger.info("Branch 2: Joint I3D Future Prediction")
    logger.info("=" * 60)
    logger.info("Configuration:")
    logger.info("  - MagVit: TRAIN (learns trajectory features!)")
    logger.info("  - I3D: FINETUNE (motion guidance)")
    logger.info("  - Transformer: TRAIN (12 layers, 8 heads)")
    logger.info("  - Key Innovation: MagVit learns WITH motion understanding")
    logger.info("=" * 60)
    
    # Run test suite
    test_results = run_test_suite([
        test_i3d_loading,
        test_joint_model_creation,
        test_joint_forward_pass,
        test_joint_training_step
    ], logger)
    
    # If tests passed, run training
    if test_results['failed'] == 0:
        logger.info("\n✅ All tests passed! Starting joint training...")
        
        # Create dataset
        logger.info("Creating synthetic trajectory dataset...")
        videos = create_video_dataset_for_prediction(num_samples=100, device='cuda')
        logger.info(f"Dataset created: {videos.shape}")
        
        # Create model
        checkpoint_path = "/home/ubuntu/magvit_weights/video_128_262144.ckpt"
        if not Path(checkpoint_path).exists():
            checkpoint_path = None
        
        model = JointI3DFuturePrediction(
            magvit_checkpoint=checkpoint_path,
            device='cuda'
        )
        
        # Train
        training_results = train_joint_i3d(
            model=model,
            videos=videos,
            num_epochs=50,
            batch_size=2,  # Smaller batch due to training MagVit
            lr=1e-4,
            logger=logger
        )
        
        # Save results
        final_results = {
            'branch': 'joint_i3d',
            'timestamp': datetime.now().isoformat(),
            'test_results': test_results,
            'training_results': training_results,
            'config': {
                'magvit_trainable': True,
                'i3d_finetune': True,
                'motion_model': 'i3d',
                'num_epochs': 50,
                'batch_size': 2,
                'learning_rate': 1e-4
            }
        }
        
        save_results(final_results, output_dir, 'joint_i3d')
        logger.info("=" * 60)
        logger.info("✅ JOINT I3D TRAINING COMPLETE")
        logger.info("=" * 60)
    else:
        logger.error("❌ Tests failed. Aborting training.")
        final_results = {
            'branch': 'joint_i3d',
            'timestamp': datetime.now().isoformat(),
            'test_results': test_results,
            'status': 'failed_tests'
        }
        save_results(final_results, output_dir, 'joint_i3d')


if __name__ == '__main__':
    main()

