#!/usr/bin/env python3
"""
Extract Visual Features from 2D Tracks using MagVIT
===================================================
Uses pretrained MagVIT encoder to extract visual features from 2D track sequences.

These features capture:
- Object appearance (what does it look like?)
- Motion patterns (how does it move?)
- Temporal consistency (does appearance persist?)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import json
from tqdm import tqdm
import sys

# Add experiments to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from experiments.magvit_pretrained_models.complete_magvit import CompleteMagVit


class TrackFeatureExtractor:
    """Extracts MagVIT features from 2D track sequences."""
    
    def __init__(
        self,
        magvit_checkpoint: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize feature extractor.
        
        Args:
            magvit_checkpoint: Path to pretrained MagVIT checkpoint
            device: Device to run on
            cache_dir: Directory to cache extracted features
        """
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load pretrained MagVIT
        print(f"Loading MagVIT from {magvit_checkpoint}...")
        self.magvit = CompleteMagVit(
            checkpoint_path=magvit_checkpoint,
            device=device
        )
        
        # Set to eval mode and freeze
        self.magvit.encoder.eval()
        for param in self.magvit.encoder.parameters():
            param.requires_grad = False
        
        print(f"MagVIT loaded on {device}")
    
    @torch.no_grad()
    def extract_features(
        self,
        pixels: np.ndarray,
        track_id: Optional[int] = None
    ) -> np.ndarray:
        """
        Extract MagVIT features from track pixels.
        
        Args:
            pixels: Track pixel sequence of shape (T, H, W, 3) with values 0-255
            track_id: Optional track ID for caching
            
        Returns:
            features: np.ndarray of shape (T, D) where D is feature dimension
        """
        # Check cache first
        if self.cache_dir and track_id is not None:
            cache_file = self.cache_dir / f"track_{track_id}_features.npy"
            if cache_file.exists():
                return np.load(cache_file)
        
        # Prepare input
        T, H, W, C = pixels.shape
        
        # Normalize to 0-1 range
        pixels_norm = pixels.astype(np.float32) / 255.0
        
        # Convert to tensor: (1, C, T, H, W) - batch, channels, time, height, width
        video_tensor = torch.from_numpy(pixels_norm).permute(0, 3, 1, 2).unsqueeze(0)  # (1, T, H, W, 3) -> (1, 3, T, H, W)
        video_tensor = video_tensor.to(self.device)
        
        # Actually we need (B, C, T, H, W) format, let me fix:
        video_tensor = torch.from_numpy(pixels_norm).float()  # (T, H, W, 3)
        video_tensor = video_tensor.permute(3, 0, 1, 2).unsqueeze(0)  # (1, 3, T, H, W)
        video_tensor = video_tensor.to(self.device)
        
        # Extract features using MagVIT encoder
        # Output: z_q of shape (B, C, T', H', W') and indices
        z_q, indices = self.magvit.encode(video_tensor)
        
        # Pool spatial dimensions: (B, C, T', H', W') -> (B, C, T')
        # We average over spatial dimensions to get a feature vector per frame
        features = z_q.mean(dim=(-2, -1))  # (B, C, T')
        
        # Transpose to (B, T', C) then extract batch
        features = features.permute(0, 2, 1).squeeze(0)  # (T', C)
        
        # Convert to numpy
        features = features.cpu().numpy()
        
        # Handle temporal dimension mismatch (if MagVIT downsampled)
        if features.shape[0] != T:
            # Interpolate to match original temporal length
            from scipy.interpolate import interp1d
            old_indices = np.linspace(0, T - 1, features.shape[0])
            new_indices = np.arange(T)
            
            interpolated_features = []
            for dim in range(features.shape[1]):
                f = interp1d(old_indices, features[:, dim], kind='linear', fill_value='extrapolate')
                interpolated_features.append(f(new_indices))
            
            features = np.stack(interpolated_features, axis=1)  # (T, D)
        
        # Cache if requested
        if self.cache_dir and track_id is not None:
            cache_file = self.cache_dir / f"track_{track_id}_features.npy"
            np.save(cache_file, features)
        
        return features
    
    def extract_dataset_features(
        self,
        data_dir: Path,
        output_dir: Path,
        max_tracks: Optional[int] = None
    ):
        """
        Extract features for entire dataset.
        
        Args:
            data_dir: Directory containing track pixels and metadata
            output_dir: Directory to save extracted features
            max_tracks: Maximum number of tracks to process (for testing)
        """
        data_dir = Path(data_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load metadata
        metadata_file = data_dir / "tracks_metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        if max_tracks:
            metadata = metadata[:max_tracks]
        
        print(f"Extracting features for {len(metadata)} tracks...")
        
        # Set cache directory to output directory
        self.cache_dir = output_dir
        
        # Extract features for each track
        for track_meta in tqdm(metadata, desc="Extracting features"):
            track_id = track_meta['track_id']
            
            # Load pixels
            pixel_file = data_dir / f"track_{track_id}_pixels.npy"
            if not pixel_file.exists():
                print(f"Warning: Pixel file not found for track {track_id}")
                continue
            
            pixels = np.load(pixel_file)
            
            # Extract features (will cache automatically)
            features = self.extract_features(pixels, track_id=track_id)
        
        print(f"Feature extraction complete. Features saved to {output_dir}")
        
        # Save feature extraction metadata
        feature_meta = {
            'num_tracks': len(metadata),
            'feature_dim': features.shape[1] if features is not None else None,
            'magvit_checkpoint': str(self.magvit.checkpoint_path) if hasattr(self.magvit, 'checkpoint_path') else 'unknown',
            'device': self.device
        }
        
        with open(output_dir / "feature_extraction_metadata.json", 'w') as f:
            json.dump(feature_meta, f, indent=2)


def main():
    """Main entry point for feature extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract MagVIT features from 2D tracks')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing track pixel data')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save extracted features')
    parser.add_argument('--magvit-checkpoint', type=str, required=True,
                        help='Path to pretrained MagVIT checkpoint')
    parser.add_argument('--max-tracks', type=int, default=None,
                        help='Maximum number of tracks to process (for testing)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = TrackFeatureExtractor(
        magvit_checkpoint=args.magvit_checkpoint,
        device=args.device if torch.cuda.is_available() else 'cpu'
    )
    
    # Extract features for dataset
    extractor.extract_dataset_features(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        max_tracks=args.max_tracks
    )


if __name__ == "__main__":
    main()

