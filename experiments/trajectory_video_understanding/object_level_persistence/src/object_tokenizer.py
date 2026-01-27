"""
Object tokenizer for converting detected objects to transformer tokens.

TDD GREEN phase - Implementation to pass tests.
"""

from dataclasses import dataclass
from typing import List, Tuple
import torch
import torch.nn as nn
import numpy as np
from .object_detector import DetectedObject


@dataclass
class ObjectToken:
    """Represents an object as a transformer token."""
    features: torch.Tensor  # (feature_dim,)
    frame_idx: int
    track_id: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    
    def __repr__(self):
        return f"ObjectToken(frame_idx={self.frame_idx}, track_id={self.track_id}, confidence={self.confidence:.2f})"


class ObjectTokenizer:
    """Converts detected objects to transformer tokens."""
    
    def __init__(self, feature_dim=256, max_frames=16, max_objects_per_frame=4):
        """Initialize tokenizer."""
        self.feature_dim = feature_dim
        self.max_frames = max_frames
        self.max_objects_per_frame = max_objects_per_frame
        
        # Simple CNN encoder for patches
        self.patch_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, feature_dim)
        )
        
        # Positional encoding
        self.max_position = 100
        self.position_embedding = nn.Embedding(self.max_position, feature_dim)
    
    def extract_patch(self, frame: np.ndarray, detected_object: DetectedObject) -> np.ndarray:
        """Extract object patch from frame."""
        x, y, w, h = detected_object.bbox
        
        # Ensure bounds are within frame
        frame_h, frame_w = frame.shape[:2]
        x = max(0, min(x, frame_w - 1))
        y = max(0, min(y, frame_h - 1))
        x_end = min(x + w, frame_w)
        y_end = min(y + h, frame_h)
        
        # Extract patch
        patch = frame[y:y_end, x:x_end]
        
        return patch
    
    def encode_patch(self, patch: np.ndarray) -> torch.Tensor:
        """Encode patch to feature vector."""
        # Convert to torch tensor (C, H, W)
        if isinstance(patch, np.ndarray):
            patch_tensor = torch.from_numpy(patch).float()
        else:
            patch_tensor = patch
        
        # Ensure CHW format
        if patch_tensor.ndim == 3 and patch_tensor.shape[2] == 3:
            patch_tensor = patch_tensor.permute(2, 0, 1)  # HWC -> CHW
        elif patch_tensor.ndim == 2:
            patch_tensor = patch_tensor.unsqueeze(0).repeat(3, 1, 1)
        
        # Add batch dimension
        patch_tensor = patch_tensor.unsqueeze(0)
        
        # Encode
        with torch.no_grad():
            features = self.patch_encoder(patch_tensor)
        
        return features.squeeze(0)
    
    def add_positional_encoding(self, features: torch.Tensor, frame_idx: int) -> torch.Tensor:
        """Add positional encoding to features."""
        # Get position embedding
        position = torch.tensor([frame_idx])
        pos_embedding = self.position_embedding(position).squeeze(0)
        
        # Add to features
        encoded = features + pos_embedding
        
        return encoded
    
    def create_token(self, frame: np.ndarray, detected_object: DetectedObject, 
                    frame_idx: int, track_id: int) -> ObjectToken:
        """Create token from detection."""
        # Extract and encode patch
        patch = self.extract_patch(frame, detected_object)
        features = self.encode_patch(patch)
        
        # Add positional encoding
        features = self.add_positional_encoding(features, frame_idx)
        
        # Create token
        token = ObjectToken(
            features=features,
            frame_idx=frame_idx,
            track_id=track_id,
            bbox=detected_object.bbox,
            confidence=detected_object.confidence
        )
        
        return token
    
    def tokenize_frame(self, frame: np.ndarray, detections: List[DetectedObject],
                      track_ids: List[int], frame_idx: int) -> List[ObjectToken]:
        """Tokenize all objects in a frame."""
        tokens = []
        
        for detection, track_id in zip(detections, track_ids):
            token = self.create_token(frame, detection, frame_idx, track_id)
            tokens.append(token)
        
        return tokens
    
    def tokenize_video(self, video: np.ndarray, detections_per_frame: List[List[DetectedObject]],
                      track_ids_per_frame: List[List[int]]) -> List[ObjectToken]:
        """Tokenize all objects in a video sequence."""
        all_tokens = []
        
        for frame_idx, (frame, detections, track_ids) in enumerate(
            zip(video, detections_per_frame, track_ids_per_frame)
        ):
            tokens = self.tokenize_frame(frame, detections, track_ids, frame_idx)
            all_tokens.extend(tokens)
        
        return all_tokens
    
    def tokens_to_tensor(self, tokens: List[ObjectToken]) -> torch.Tensor:
        """Convert token list to tensor."""
        # Truncate if too long
        if len(tokens) > self.max_frames:
            tokens = tokens[:self.max_frames]
        
        # Stack features
        features_list = [token.features for token in tokens]
        sequence_tensor = torch.stack(features_list)
        
        return sequence_tensor
    
    def tokens_to_tensor_padded(self, tokens: List[ObjectToken]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert token list to padded tensor with mask."""
        # Truncate if too long
        if len(tokens) > self.max_frames:
            tokens = tokens[:self.max_frames]
        
        # Create padded tensor
        seq_len = len(tokens)
        sequence_tensor = torch.zeros(self.max_frames, self.feature_dim)
        
        # Fill in actual tokens
        for i, token in enumerate(tokens):
            sequence_tensor[i] = token.features
        
        # Create mask (True for valid tokens, False for padding)
        mask = torch.zeros(self.max_frames, dtype=torch.bool)
        mask[:seq_len] = True
        
        return sequence_tensor, mask
