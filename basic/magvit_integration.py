#!/usr/bin/env python3
"""
MagVit Integration Layer
========================
Bridge between classifier and MagVit without modifying MagVit code.

This module provides a PyTorch-friendly interface to the JAX-based MagVit
implementation, allowing trajectory classification and generation to work together.
"""

import sys
import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn.functional as F

# Add MagVit path to sys.path
MAGVIT_PATH = Path(__file__).parent.parent / 'experiments' / 'magvit-3d-trajectories' / 'magvit'
if MAGVIT_PATH.exists():
    sys.path.insert(0, str(MAGVIT_PATH.parent))
else:
    print(f"⚠️  Warning: MagVit path not found at {MAGVIT_PATH}")
    print("   MagVit integration features will be limited.")

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basic.trajectory_to_video import trajectory_to_video, batch_trajectories_to_video


class MagVitIntegration:
    """
    Integration layer for MagVit without modifying original code.
    Provides PyTorch-friendly interface to JAX-based MagVit.
    """
    
    def __init__(
        self,
        magvit_config_path: Optional[str] = None,
        use_jax: bool = False
    ):
        """
        Initialize MagVit integration.
        
        Args:
            magvit_config_path: Path to MagVit config file (optional)
            use_jax: Whether to use actual JAX-based MagVit (requires JAX setup)
        """
        self.config_path = magvit_config_path
        self.use_jax = use_jax
        self.tokenizer = None
        self.detokenizer = None
        self.transformer = None
        self.pytorch_tokenizer = None  # PyTorch VideoTokenizer
        self.pytorch_config = None
        self._initialized = False
        
        if use_jax:
            self._load_magvit()
        else:
            # Try to load PyTorch VideoTokenizer
            self._load_pytorch_magvit()
    
    def _load_magvit(self):
        """Load MagVit models (VQ-VAE tokenizer and Transformer)."""
        if not self.use_jax:
            return
        
        try:
            # Import JAX and MagVit modules
            import jax.numpy as jnp
            from videogvt.interfaces.mm_vq import load_mm_vq_model
            import ml_collections
            
            # Load config
            if self.config_path:
                config = self._load_config(self.config_path)
            else:
                # Use default config
                config = self._get_default_config()
            
            # Load tokenizer
            tokenizer_dict = load_mm_vq_model(config)
            self.tokenizer = tokenizer_dict.get('tokenizer')
            self.detokenizer = tokenizer_dict.get('detokenizer')
            
            self._initialized = True
            print("✅ MagVit models loaded successfully")
            
        except ImportError as e:
            print(f"⚠️  Warning: Could not import MagVit modules: {e}")
            print("   MagVit integration will use PyTorch fallback.")
            self.use_jax = False
        except Exception as e:
            print(f"⚠️  Warning: Error loading MagVit: {e}")
            print("   MagVit integration will use PyTorch fallback.")
            self.use_jax = False
    
    def _load_pytorch_magvit(self):
        """Load PyTorch VideoTokenizer from separate MAGVIT repo."""
        try:
            # Try to import from magvit_options folder (where we transferred it)
            magvit_options_path = Path(__file__).parent.parent / 'magvit_options'
            if (magvit_options_path / 'simple_magvit_model.py').exists():
                sys.path.insert(0, str(magvit_options_path))
                from simple_magvit_model import VideoTokenizer, ModelConfig
                
                # Configure tokenizer for trajectory videos
                self.pytorch_config = ModelConfig(
                    video_height=64,
                    video_width=64,
                    video_frames=50,
                    patch_size=8,
                    vocab_size=1024,
                    hidden_dim=256
                )
                
                # Create tokenizer
                self.pytorch_tokenizer = VideoTokenizer(self.pytorch_config)
                self.pytorch_tokenizer.eval()
                
                # Try to load pre-trained weights if available
                checkpoint_path = magvit_options_path / 'checkpoint_best.pth'
                if checkpoint_path.exists():
                    try:
                        checkpoint = torch.load(checkpoint_path, map_location='cpu')
                        if 'tokenizer_state_dict' in checkpoint:
                            self.pytorch_tokenizer.load_state_dict(checkpoint['tokenizer_state_dict'])
                            print("✅ PyTorch MagVit tokenizer loaded with pre-trained weights")
                        else:
                            print("⚠️  Checkpoint doesn't contain tokenizer_state_dict, using random init")
                    except Exception as e:
                        print(f"⚠️  Could not load checkpoint: {e}, using random init")
                else:
                    print("ℹ️  PyTorch MagVit tokenizer initialized (random weights)")
                    print("   Using VideoTokenizer for real visual token extraction")
                
                self._initialized = True
            else:
                print("ℹ️  PyTorch MagVit tokenizer not found, using simple fallback")
        except ImportError as e:
            print(f"ℹ️  Could not import PyTorch VideoTokenizer: {e}")
            print("   Using simple fallback feature extraction")
        except Exception as e:
            print(f"⚠️  Error loading PyTorch MagVit: {e}")
            print("   Using simple fallback feature extraction")
    
    def _load_config(self, config_path: str):
        """Load MagVit configuration."""
        # This would load the actual config file
        # For now, return a placeholder
        import ml_collections
        config = ml_collections.ConfigDict()
        return config
    
    def _get_default_config(self):
        """Get default MagVit configuration."""
        import ml_collections
        config = ml_collections.ConfigDict()
        # Add default config values here
        return config
    
    def encode_trajectory(
        self,
        trajectory: np.ndarray,
        return_video: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Encode trajectory using MagVit's VQ-VAE tokenizer.
        
        Args:
            trajectory: (N, 3) trajectory array
            return_video: Whether to return the video representation
        
        Returns:
            tokens: Discrete tokens from VQ-VAE (or features if JAX not available)
            video: Optional video representation (T, H, W, 3)
        """
        # Convert trajectory to video (match tokenizer config: 50 frames, 128x128)
        video = trajectory_to_video(trajectory, resolution=(128, 128), num_frames=50)
        
        if return_video:
            video_out = video.copy()
        else:
            video_out = None
        
        if self.use_jax and self.tokenizer is not None:
            try:
                import jax.numpy as jnp
                # Convert to JAX array
                video_jax = jnp.array(video.astype(np.float32) / 255.0)
                
                # Tokenize with MagVit
                tokens = self.tokenizer(video_jax)
                return np.array(tokens), video_out
            except Exception as e:
                print(f"⚠️  Error in JAX tokenization: {e}")
                # Fall through to PyTorch fallback
        
        # PyTorch fallback: return video features as "tokens"
        # This is a placeholder - actual implementation would use a PyTorch VQ-VAE
        tokens = self._pytorch_encode_fallback(video)
        return tokens, video_out
    
    def _pytorch_encode_fallback(self, video: np.ndarray) -> np.ndarray:
        """
        PyTorch fallback for encoding when JAX MagVit is not available.
        Uses real PyTorch VideoTokenizer if available, otherwise simple mean pooling.
        """
        # Use real PyTorch VideoTokenizer if available
        if self.pytorch_tokenizer is not None and self.pytorch_config is not None:
            try:
                # Convert video to tensor
                video_tensor = torch.from_numpy(video).float()  # (T, H, W, 3)
                
                # Resize to match tokenizer (64x64) and ensure correct frame count
                T, H, W, C = video_tensor.shape
                target_frames = self.pytorch_config.video_frames
                
                # Trim or pad frames to match expected count
                if T > target_frames:
                    video_tensor = video_tensor[:target_frames]
                elif T < target_frames:
                    # Pad with last frame
                    last_frame = video_tensor[-1:].repeat(target_frames - T, 1, 1, 1)
                    video_tensor = torch.cat([video_tensor, last_frame], dim=0)
                
                # Resize spatial dimensions if needed
                if H != self.pytorch_config.video_height or W != self.pytorch_config.video_width:
                    video_tensor = video_tensor.permute(0, 3, 1, 2)  # (T, 3, H, W)
                    video_resized = F.interpolate(
                        video_tensor,
                        size=(self.pytorch_config.video_height, self.pytorch_config.video_width),
                        mode='bilinear',
                        align_corners=False
                    )
                    video_resized = video_resized.permute(0, 2, 3, 1)  # (T, 64, 64, 3)
                else:
                    video_resized = video_tensor
                
                # Normalize to [0, 1]
                video_normalized = video_resized / 255.0
                
                # Extract features with tokenizer
                with torch.no_grad():
                    embeddings, token_ids = self.pytorch_tokenizer(video_normalized.unsqueeze(0))
                    # Pool embeddings (mean pooling over sequence)
                    features = embeddings.mean(dim=1).squeeze(0)  # (hidden_dim,)
                
                # Convert back to numpy
                try:
                    return features.cpu().numpy()
                except (RuntimeError, TypeError):
                    return np.array(features.cpu().tolist())
                    
            except Exception as e:
                print(f"⚠️  Error in PyTorch tokenizer: {e}, falling back to simple features")
                # Fall through to simple fallback
        
        # Simple fallback: mean pooling over spatial dimensions
        try:
            video_tensor = torch.from_numpy(video).float() / 255.0
        except (RuntimeError, TypeError) as e:
            if "Numpy is not available" in str(e) or "not available" in str(e):
                video_list = video.tolist()
                video_tensor = torch.tensor(video_list, dtype=torch.float32) / 255.0
            else:
                raise
        
        # (T, H, W, 3) -> (T, 3) by spatial mean pooling
        features = video_tensor.mean(dim=(1, 2))  # Mean over H, W
        
        # Convert back to numpy safely
        try:
            return features.numpy()
        except (RuntimeError, TypeError):
            return np.array(features.tolist())
    
    def decode_tokens(self, tokens: np.ndarray) -> np.ndarray:
        """
        Decode tokens back to video using MagVit's decoder.
        
        Args:
            tokens: Discrete tokens or features
        
        Returns:
            video: Reconstructed video (T, H, W, 3)
        """
        if self.use_jax and self.detokenizer is not None:
            try:
                import jax.numpy as jnp
                tokens_jax = jnp.array(tokens)
                video = self.detokenizer(tokens_jax)
                return np.array(video)
            except Exception as e:
                print(f"⚠️  Error in JAX decoding: {e}")
                # Fall through to PyTorch fallback
        
        # PyTorch fallback: simple reconstruction
        # This is a placeholder
        return self._pytorch_decode_fallback(tokens)
    
    def _pytorch_decode_fallback(self, tokens: np.ndarray) -> np.ndarray:
        """PyTorch fallback for decoding."""
        # Placeholder: would use a trained decoder
        # For now, return a simple visualization
        T = len(tokens)
        video = np.zeros((T, 128, 128, 3), dtype=np.uint8)
        # Simple visualization based on tokens
        for t in range(T):
            if len(tokens[t]) >= 3:
                # Handle both numpy array and list inputs
                if isinstance(tokens[t], np.ndarray):
                    color = (tokens[t][:3] * 255).astype(np.uint8)
                else:
                    # Convert list to numpy if needed
                    token_array = np.array(tokens[t][:3])
                    color = (token_array * 255).astype(np.uint8)
                video[t, :, :] = np.clip(color, 0, 255)
        return video
    
    def extract_features(
        self,
        trajectory: np.ndarray,
        use_magvit: bool = True
    ) -> np.ndarray:
        """
        Extract learned features from MagVit encoder.
        
        Args:
            trajectory: (N, 3) trajectory array
            use_magvit: Whether to use MagVit (if False, uses simple features)
        
        Returns:
            features: Learned feature representation
        """
        if use_magvit and self.use_jax and self.tokenizer is not None:
            tokens, _ = self.encode_trajectory(trajectory)
            return tokens
        else:
            # Fallback: simple trajectory features
            return self._extract_simple_features(trajectory)
    
    def _extract_simple_features(self, trajectory: np.ndarray) -> np.ndarray:
        """Extract simple features from trajectory."""
        # Velocity
        velocity = np.diff(trajectory, axis=0)
        # Acceleration
        acceleration = np.diff(velocity, axis=0)
        # Concatenate
        features = np.concatenate([
            trajectory[1:],  # Position
            velocity,        # Velocity
            np.pad(acceleration, ((0, 1), (0, 0)), mode='edge')  # Acceleration
        ], axis=1)
        return features
    
    def generate_conditioned(
        self,
        class_label: int,
        condition_features: Optional[np.ndarray] = None,
        num_samples: int = 10,
        seed_trajectory: Optional[np.ndarray] = None
    ) -> List[np.ndarray]:
        """
        Generate trajectories conditioned on class label.
        
        Args:
            class_label: Class ID (0-4)
            condition_features: Optional features from classifier
            num_samples: Number of trajectories to generate
            seed_trajectory: Optional seed trajectory to start from
        
        Returns:
            List of generated trajectories
        """
        if self.use_jax and self.transformer is not None:
            # Use actual MagVit generation
            return self._magvit_generate(class_label, condition_features, num_samples, seed_trajectory)
        else:
            # Fallback: simple generation based on class
            return self._simple_generate(class_label, num_samples, seed_trajectory)
    
    def _magvit_generate(
        self,
        class_label: int,
        condition_features: Optional[np.ndarray],
        num_samples: int,
        seed_trajectory: Optional[np.ndarray]
    ) -> List[np.ndarray]:
        """Generate using actual MagVit (requires JAX setup)."""
        # This would use MagVit's masked modeling to generate
        # Implementation depends on MagVit's generation API
        print("⚠️  Full MagVit generation not yet implemented.")
        print("   Falling back to simple generation.")
        return self._simple_generate(class_label, num_samples, seed_trajectory)
    
    def _simple_generate(
        self,
        class_label: int,
        num_samples: int,
        seed_trajectory: Optional[np.ndarray]
    ) -> List[np.ndarray]:
        """Simple trajectory generation based on class characteristics."""
        generated = []
        
        # Class-specific trajectory patterns
        class_patterns = {
            0: {'radius': 12.5, 'speed': 1.0},  # Long radius
            1: {'radius': 6.5, 'speed': 1.0},   # Medium radius
            2: {'radius': 3.0, 'speed': 1.0},   # Short radius
            3: {'radius': 6.5, 'speed': 0.75},  # Deceleration
            4: {'radius': 6.5, 'speed': 1.25}, # Acceleration
        }
        
        pattern = class_patterns.get(class_label, {'radius': 6.5, 'speed': 1.0})
        
        for _ in range(num_samples):
            # Generate a simple trajectory based on class pattern
            trajectory = self._generate_class_trajectory(pattern, seed_trajectory)
            generated.append(trajectory)
        
        return generated
    
    def _generate_class_trajectory(
        self,
        pattern: Dict[str, float],
        seed: Optional[np.ndarray]
    ) -> np.ndarray:
        """Generate a single trajectory based on class pattern."""
        num_points = 100
        t = np.linspace(0, 2 * np.pi, num_points)
        
        # Add some randomness
        radius = pattern['radius'] * (1 + np.random.uniform(-0.2, 0.2))
        speed = pattern['speed'] * (1 + np.random.uniform(-0.1, 0.1))
        
        # Generate trajectory
        x = radius * np.cos(t) + np.random.uniform(-2, 2)
        y = radius * np.sin(t) + np.random.uniform(-2, 2)
        z = speed * t + np.random.uniform(-1, 1)
        
        trajectory = np.array([x, y, z]).T
        
        if seed is not None:
            # Blend with seed trajectory
            alpha = 0.3
            trajectory = alpha * seed + (1 - alpha) * trajectory
        
        return trajectory


if __name__ == '__main__':
    # Test the integration
    print("Testing MagVit integration...")
    
    # Initialize (without JAX for testing)
    magvit = MagVitIntegration(use_jax=False)
    
    # Create a sample trajectory
    t = np.linspace(0, 4 * np.pi, 100)
    trajectory = np.array([
        np.cos(t) * t,
        np.sin(t) * t,
        t * 0.5
    ]).T
    
    # Test encoding
    tokens, video = magvit.encode_trajectory(trajectory, return_video=True)
    print(f"Trajectory shape: {trajectory.shape}")
    print(f"Tokens shape: {tokens.shape}")
    print(f"Video shape: {video.shape}")
    
    # Test feature extraction
    features = magvit.extract_features(trajectory)
    print(f"Features shape: {features.shape}")
    
    # Test generation
    generated = magvit.generate_conditioned(class_label=1, num_samples=3)
    print(f"Generated {len(generated)} trajectories")
    print(f"Generated trajectory shape: {generated[0].shape}")
    
    print("✅ Integration test successful!")

