"""
Fast dataset loader for quick attention validation.

Uses existing augmented dataset with ground-truth color-based tracking.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import json
from typing import List, Tuple
import sys

# Add parent dirs to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .object_detector import ObjectDetector
from .object_tokenizer import ObjectTokenizer
from .pseudo_tracker import PseudoTracker


class FastObjectDataset(Dataset):
    """Fast dataset for object-level persistence training."""
    
    def __init__(
        self,
        data_root: str,
        max_samples: int = 1000,
        max_frames_per_video: int = 16,
        split: str = 'train'
    ):
        """
        Initialize dataset.
        
        Args:
            data_root: Path to augmented dataset
            max_samples: Max number of samples to load (for speed)
            max_frames_per_video: Max frames to process per video
            split: 'train' or 'val'
        """
        self.data_root = Path(data_root)
        self.max_frames_per_video = max_frames_per_video
        
        # Initialize components
        self.detector = ObjectDetector(input_size=(64, 64))
        self.tokenizer = ObjectTokenizer(feature_dim=256, max_frames=max_frames_per_video)
        self.tracker = PseudoTracker()
        
        # Load sample list
        self.samples = self._load_samples(max_samples, split)
        
        print(f"Loaded {len(self.samples)} samples for {split}")
    
    def _load_samples(self, max_samples: int, split: str) -> List[Tuple[Path, Path]]:
        """Load list of (video_file, metadata_file) tuples."""
        # Check for PT format dataset (augmented_traj_*.pt)
        pt_files = sorted(list(self.data_root.glob('augmented_traj_*.pt')))
        
        if len(pt_files) > 0:
            # Use PT format
            print(f"Found {len(pt_files)} PT format samples")
            samples = [(pt_file, pt_file.with_suffix('.json')) for pt_file in pt_files]
        else:
            # Try NPY format (sample_*/video.npy)
            augmented_dir = self.data_root / 'augmented_trajectories'
            if not augmented_dir.exists():
                raise ValueError(f"Dataset not found at {augmented_dir} or {self.data_root}")
            
            sample_dirs = sorted(list(augmented_dir.glob('sample_*')))
            samples = [(d / 'video.npy', d / 'metadata.json') for d in sample_dirs]
        
        # Split train/val (80/20)
        split_idx = int(len(samples) * 0.8)
        if split == 'train':
            samples = samples[:split_idx]
        else:
            samples = samples[split_idx:]
        
        # Limit to max_samples for speed
        samples = samples[:max_samples]
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Get a sample using iterative approach (non-recursive).
        
        Returns:
            object_tokens: (seq_len, feature_dim) object token features
            labels: (seq_len,) persistence labels (0=persistent, 1=transient)
            mask: (seq_len,) valid token mask
            metadata: dict with track_ids, confidences, etc.
        """
        # Try up to 10 samples to find one with detections
        max_attempts = 10
        num_samples = len(self.samples)
        
        for attempt in range(max_attempts):
            current_idx = (idx + attempt) % num_samples
            video_file, metadata_file = self.samples[current_idx]
            
            try:
                # Load video
                if video_file.suffix == '.pt':
                    video = torch.load(video_file).numpy()  # (T, C, H, W) from PT format
                    # Convert (T, C, H, W) -> (T, H, W, C)
                    if video.ndim == 4 and video.shape[1] == 3:
                        video = video.transpose(0, 2, 3, 1)  # (T, C, H, W) -> (T, H, W, C)
                else:
                    video = np.load(video_file)  # (T, H, W, 3)
                
                # Debug: Check video format (only for first sample, first attempt)
                if idx == 0 and attempt == 0:
                    print(f"DEBUG: Video shape: {video.shape}, dtype: {video.dtype}, "
                          f"min: {video.min():.3f}, max: {video.max():.3f}")
                
                # Load metadata
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Limit frames
                video = video[:self.max_frames_per_video]
                
                # Reset tracker
                self.tracker.reset()
                
                # Process each frame
                all_tokens = []
                all_labels = []
                all_track_ids = []
                
                for frame_idx, frame in enumerate(video):
                    # Detect objects
                    detections = self.detector.detect(frame)
                    
                    if len(detections) == 0:
                        continue
                    
                    # Assign tracks (color-based)
                    track_ids = self.tracker.assign_tracks(frame, detections)
                    
                    # Create tokens
                    tokens = self.tokenizer.tokenize_frame(
                        frame=frame,
                        detections=detections,
                        track_ids=track_ids,
                        frame_idx=frame_idx
                    )
                    
                    # Assign labels based on track ID
                    # Track 1 = persistent (label 0)
                    # Track 2+ = transient (label 1)
                    labels = [0 if token.track_id == 1 else 1 for token in tokens]
                    
                    all_tokens.extend(tokens)
                    all_labels.extend(labels)
                    all_track_ids.extend(track_ids)
                
                # Check if we found any tokens
                if len(all_tokens) > 0:
                    # Success! Convert to tensors and return
                    break
                else:
                    if attempt == 0:
                        print(f"Sample {video_file.stem} has no detections, trying next...")
                    
            except Exception as e:
                print(f"Error loading sample {video_file.stem}: {e}, trying next...")
                continue
        
        # If we still have no tokens after all attempts, return dummy
        if len(all_tokens) == 0:
            print(f"Warning: Could not find valid sample after {max_attempts} attempts, returning dummy")
            dummy_tokens = torch.randn(1, self.tokenizer.feature_dim)
            dummy_labels = torch.zeros(1, dtype=torch.long)
            dummy_mask = torch.ones(1, dtype=torch.bool)
            dummy_meta = {'sample_id': f'dummy_{idx}', 'num_objects': 1, 'track_ids': [1], 
                         'num_persistent': 1, 'num_transient': 0}
            return dummy_tokens, dummy_labels, dummy_mask, dummy_meta
        
        # Success! Stack features
        object_tokens = torch.stack([token.features for token in all_tokens])
        labels = torch.tensor(all_labels, dtype=torch.long)
        mask = torch.ones(len(all_tokens), dtype=torch.bool)
        
        # Metadata for analysis
        meta = {
            'sample_id': video_file.stem,  # e.g., 'augmented_traj_00000' or 'video'
            'num_objects': len(all_tokens),
            'track_ids': all_track_ids,
            'num_persistent': (labels == 0).sum().item(),
            'num_transient': (labels == 1).sum().item()
        }
        
        return object_tokens, labels, mask, meta


def collate_fn(batch):
    """Collate function to pad sequences and filter empty samples."""
    # Filter out any samples with zero tokens (safety check)
    batch = [(tokens, labels, mask, meta) for tokens, labels, mask, meta in batch 
             if tokens.shape[0] > 0]
    
    if len(batch) == 0:
        # Return dummy batch if all samples were empty
        return (torch.zeros(1, 1, 256), 
                torch.zeros(1, 1, dtype=torch.long),
                torch.zeros(1, 1, dtype=torch.bool),
                [{'sample_id': 'empty', 'num_objects': 0}])
    
    # Find max sequence length
    max_len = max(item[0].shape[0] for item in batch)
    feature_dim = batch[0][0].shape[1]
    batch_size = len(batch)
    
    # Ensure max_len is at least 1
    max_len = max(1, max_len)
    
    # Initialize padded tensors
    padded_tokens = torch.zeros(batch_size, max_len, feature_dim)
    padded_labels = torch.zeros(batch_size, max_len, dtype=torch.long)
    padded_masks = torch.zeros(batch_size, max_len, dtype=torch.bool)
    metadata_list = []
    
    for i, (tokens, labels, mask, meta) in enumerate(batch):
        seq_len = tokens.shape[0]
        if seq_len > 0:  # Extra safety check
            padded_tokens[i, :seq_len] = tokens
            padded_labels[i, :seq_len] = labels
            padded_masks[i, :seq_len] = mask
        metadata_list.append(meta)
    
    return padded_tokens, padded_labels, padded_masks, metadata_list

