"""
Pre-trained ResNet Tokenizer (Worker 2)
Uses frozen ResNet-18 features instead of simple CNN for better object representation.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import cv2


@dataclass
class ObjectToken:
    """Represents a single object as a token."""
    features: torch.Tensor  # Shape: (feature_dim,)
    track_id: int
    frame_idx: int
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)


class PretrainedTokenizer(nn.Module):
    """
    Tokenizer using frozen ResNet-18 for feature extraction.
    
    Pipeline:
    1. Extract object patch from frame using bbox
    2. Resize to 224x224 for ResNet
    3. Extract features using frozen ResNet-18 (512-dim)
    4. Project to target feature_dim using trainable layer
    5. Add positional encoding
    """
    
    def __init__(self, feature_dim: int = 256, device: str = 'cuda'):
        """
        Args:
            feature_dim: Target feature dimension after projection
            device: Device for computation
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.device = device
        
        # Load pre-trained ResNet-18 and remove final FC layer
        resnet = models.resnet18(pretrained=True)
        # Remove the final FC layer, keep everything up to avgpool
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze ResNet parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.feature_extractor.eval()
        self.feature_extractor.to(device)
        
        # Trainable projection layer: 512 -> feature_dim
        self.projection = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ReLU(),
            nn.LayerNorm(feature_dim)
        )
        self.projection.to(device)
        
        # ImageNet normalization for ResNet
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def extract_patch(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract and resize object patch from frame.
        
        Args:
            frame: Frame as numpy array, shape (H, W, 3), values in [0, 1]
            bbox: Bounding box (x, y, width, height)
        
        Returns:
            patch: Resized patch, shape (224, 224, 3)
        """
        x, y, w, h = bbox
        
        # Ensure bounds are valid
        x = max(0, min(x, frame.shape[1] - 1))
        y = max(0, min(y, frame.shape[0] - 1))
        w = max(1, min(w, frame.shape[1] - x))
        h = max(1, min(h, frame.shape[0] - y))
        
        # Extract patch
        patch = frame[y:y+h, x:x+w, :]
        
        # Resize to 224x224 for ResNet
        patch_resized = cv2.resize(patch, (224, 224), interpolation=cv2.INTER_LINEAR)
        
        return patch_resized
    
    def encode_patch_to_features(self, patch: np.ndarray) -> torch.Tensor:
        """
        Encode patch using ResNet and project to target dimension.
        
        Args:
            patch: Patch as numpy array, shape (224, 224, 3), values in [0, 1]
        
        Returns:
            features: Feature vector, shape (feature_dim,)
        """
        # Convert to tensor and normalize
        patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float()  # (3, 224, 224)
        patch_tensor = self.normalize(patch_tensor).unsqueeze(0).to(self.device)  # (1, 3, 224, 224)
        
        # Extract ResNet features (frozen)
        with torch.no_grad():
            resnet_features = self.feature_extractor(patch_tensor)  # (1, 512, 1, 1)
        
        # Flatten
        resnet_features = resnet_features.view(1, 512)  # (1, 512)
        
        # Project to target dimension (trainable)
        projected_features = self.projection(resnet_features)  # (1, feature_dim)
        
        return projected_features.squeeze(0)  # (feature_dim,)
    
    def create_token(
        self,
        frame: np.ndarray,
        detection,  # Object with bbox and confidence
        track_id: int,
        frame_idx: int
    ) -> ObjectToken:
        """
        Create a token for a single detected object.
        
        Args:
            frame: Frame as numpy array
            detection: Detected object with bbox and confidence attributes
            track_id: Track ID for this object
            frame_idx: Frame index in sequence
        
        Returns:
            token: ObjectToken containing features and metadata
        """
        # Extract and encode patch
        patch = self.extract_patch(frame, detection.bbox)
        features = self.encode_patch_to_features(patch)
        
        # Create token
        token = ObjectToken(
            features=features,
            track_id=track_id,
            frame_idx=frame_idx,
            confidence=detection.confidence,
            bbox=detection.bbox
        )
        
        return token
    
    def tokenize_frame(
        self,
        frame: np.ndarray,
        detections: List,
        track_ids: List[int],
        frame_idx: int
    ) -> List[ObjectToken]:
        """
        Tokenize all objects in a frame.
        
        Args:
            frame: Frame as numpy array
            detections: List of detected objects
            track_ids: List of track IDs corresponding to detections
            frame_idx: Frame index
        
        Returns:
            tokens: List of ObjectTokens
        """
        tokens = []
        
        for detection, track_id in zip(detections, track_ids):
            token = self.create_token(frame, detection, track_id, frame_idx)
            tokens.append(token)
        
        return tokens
    
    def tokenize_video(
        self,
        video: np.ndarray,
        detections_per_frame: List[List],
        track_ids_per_frame: List[List[int]]
    ) -> List[ObjectToken]:
        """
        Tokenize all objects across all frames in a video.
        
        Args:
            video: Video as numpy array, shape (num_frames, H, W, 3)
            detections_per_frame: List of detection lists for each frame
            track_ids_per_frame: List of track ID lists for each frame
        
        Returns:
            all_tokens: List of all ObjectTokens across the video
        """
        all_tokens = []
        
        for frame_idx, (frame, detections, track_ids) in enumerate(
            zip(video, detections_per_frame, track_ids_per_frame)
        ):
            tokens = self.tokenize_frame(frame, detections, track_ids, frame_idx)
            all_tokens.extend(tokens)
        
        return all_tokens
    
    def tokens_to_tensor(self, tokens: List[ObjectToken]) -> torch.Tensor:
        """
        Convert list of tokens to tensor.
        
        Args:
            tokens: List of ObjectTokens
        
        Returns:
            tensor: Stacked features, shape (num_tokens, feature_dim)
        """
        if len(tokens) == 0:
            return torch.zeros(0, self.feature_dim, device=self.device)
        
        features = torch.stack([token.features for token in tokens])
        return features
    
    def tokens_to_tensor_padded(
        self,
        tokens: List[ObjectToken],
        max_length: int,
        padding_value: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert tokens to padded tensor with mask.
        
        Args:
            tokens: List of ObjectTokens
            max_length: Maximum sequence length
            padding_value: Value to use for padding
        
        Returns:
            padded_tensor: Padded features, shape (max_length, feature_dim)
            mask: Boolean mask, shape (max_length,), True = padding
        """
        features = self.tokens_to_tensor(tokens)
        num_tokens = len(tokens)
        
        if num_tokens == 0:
            # All padding
            padded = torch.full(
                (max_length, self.feature_dim),
                padding_value,
                device=self.device
            )
            mask = torch.ones(max_length, dtype=torch.bool, device=self.device)
            return padded, mask
        
        if num_tokens >= max_length:
            # Truncate
            return features[:max_length], torch.zeros(max_length, dtype=torch.bool, device=self.device)
        
        # Pad
        padding_size = max_length - num_tokens
        padding = torch.full(
            (padding_size, self.feature_dim),
            padding_value,
            device=self.device
        )
        padded = torch.cat([features, padding], dim=0)
        
        # Create mask (True for padding)
        mask = torch.cat([
            torch.zeros(num_tokens, dtype=torch.bool, device=self.device),
            torch.ones(padding_size, dtype=torch.bool, device=self.device)
        ])
        
        return padded, mask


def create_pretrained_tokenizer(feature_dim: int = 256, device: str = 'cuda') -> PretrainedTokenizer:
    """
    Factory function to create a pre-trained tokenizer.
    
    Args:
        feature_dim: Target feature dimension
        device: Device for computation
    
    Returns:
        tokenizer: PretrainedTokenizer instance
    """
    return PretrainedTokenizer(feature_dim=feature_dim, device=device)

