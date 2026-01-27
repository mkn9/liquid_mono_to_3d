"""
Early Persistence Classifier with MagVIT

Classifies tracks as persistent or transient with early stopping capability.
Uses MagVIT for feature extraction with temporal modeling.
"""

import torch
import torch.nn as nn
from typing import Tuple, List
import sys
from pathlib import Path

# Add path to shared modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'shared'))
from base_extractor import FeatureExtractor

# Import MagVIT extractor
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'branch_4_magvit'))
try:
    from feature_extractor import MagVITExtractor
except ImportError:
    # Fallback for testing
    MagVITExtractor = None


class EarlyPersistenceClassifier(nn.Module):
    """
    Early persistence classifier with MagVIT backbone.
    
    Key Features:
    - Processes video frame-by-frame
    - Makes early decisions with confidence thresholds
    - Stops processing when confident about transient detection
    - Full processing for persistent tracks
    """
    
    def __init__(self, feature_extractor: str = 'magvit',
                 early_stop_frame: int = 4,
                 confidence_threshold: float = 0.9,
                 feature_dim: int = 256):
        """
        Initialize early persistence classifier.
        
        Args:
            feature_extractor: Type of feature extractor ('magvit', 'transformer', etc.)
            early_stop_frame: Maximum frame to make early decision
            confidence_threshold: Confidence threshold for early stopping
            feature_dim: Feature dimension from extractor
        """
        super().__init__()
        
        self.feature_extractor_type = feature_extractor
        self.early_stop_frame = early_stop_frame
        self.confidence_threshold = confidence_threshold
        self.feature_dim = feature_dim
        
        # Initialize feature extractor
        if feature_extractor == 'magvit' and MagVITExtractor is not None:
            self.extractor = MagVITExtractor(feature_dim=feature_dim)
        else:
            # Simple fallback for testing
            self.extractor = SimpleFeatureExtractor(feature_dim=feature_dim)
        
        # Temporal processing with LSTM for frame-by-frame analysis
        self.temporal_lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=False  # Unidirectional for online processing
        )
        
        # Early classifier head (makes decisions at each frame)
        self.early_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)  # Binary: persistent vs transient
        )
        
        # Confidence estimator
        self.confidence_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, video: torch.Tensor, max_frames: int = None) -> dict:
        """
        Forward pass with early stopping capability.
        
        Args:
            video: Video tensor (B, T, C, H, W) or (T, C, H, W)
            max_frames: Maximum frames to process (for early stopping)
            
        Returns:
            Dictionary with logits, confidence, and decision info
        """
        # Handle single video input
        if video.dim() == 4:
            video = video.unsqueeze(0)  # Add batch dimension
        
        B, T, C, H, W = video.shape
        
        if max_frames is not None:
            T = min(T, max_frames)
            video = video[:, :T]
        
        # Extract features
        features = self.extractor(video)  # (B, T, feature_dim)
        
        # Process temporally
        lstm_out, (hidden, cell) = self.temporal_lstm(features)  # (B, T, 128)
        
        # Get predictions for each frame
        logits_per_frame = []
        confidences_per_frame = []
        
        for t in range(T):
            frame_hidden = lstm_out[:, t, :]  # (B, 128)
            frame_logits = self.early_classifier(frame_hidden)  # (B, 2)
            frame_confidence = self.confidence_head(frame_hidden)  # (B, 1)
            
            logits_per_frame.append(frame_logits)
            confidences_per_frame.append(frame_confidence)
        
        logits_per_frame = torch.stack(logits_per_frame, dim=1)  # (B, T, 2)
        confidences_per_frame = torch.stack(confidences_per_frame, dim=1)  # (B, T, 1)
        
        # Use last frame for final decision
        final_logits = logits_per_frame[:, -1, :]  # (B, 2)
        final_confidence = confidences_per_frame[:, -1, :]  # (B, 1)
        
        return {
            'logits': final_logits,
            'confidence': final_confidence,
            'logits_per_frame': logits_per_frame,
            'confidences_per_frame': confidences_per_frame,
            'features': features
        }
    
    def predict_batch(self, batch: torch.Tensor) -> Tuple[List[str], List[float], List[int]]:
        """
        Predict on batch of videos with early stopping.
        
        Args:
            batch: Batch of videos (B, T, C, H, W)
            
        Returns:
            Tuple of (decisions, confidences, decision_frames)
        """
        self.eval()
        with torch.no_grad():
            decisions = []
            confidences = []
            decision_frames = []
            
            for video in batch:
                decision, confidence, frame_idx = get_early_decision(self, video)
                decisions.append(decision)
                confidences.append(confidence)
                decision_frames.append(frame_idx)
        
        return decisions, confidences, decision_frames


class SimpleFeatureExtractor(nn.Module):
    """Simple CNN feature extractor for testing."""
    
    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim
        
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc = nn.Linear(64 * 4 * 4, feature_dim)
    
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Extract features from video."""
        B, T, C, H, W = video.shape
        
        # Process each frame
        frames = video.view(B * T, C, H, W)
        features = self.conv(frames)
        features = features.view(B * T, -1)
        features = self.fc(features)
        features = features.view(B, T, self.feature_dim)
        
        return features


def get_early_decision(classifier: EarlyPersistenceClassifier, 
                       video: torch.Tensor) -> Tuple[str, float, int]:
    """
    Get early decision on video with confidence-based stopping.
    
    Args:
        classifier: Trained classifier
        video: Video tensor (T, C, H, W) or (B, T, C, H, W)
        
    Returns:
        Tuple of (decision, confidence, decision_frame)
    """
    classifier.eval()
    
    if video.dim() == 4:
        video = video.unsqueeze(0)  # Add batch dimension
    
    T = video.shape[1]
    
    with torch.no_grad():
        # Process frame by frame up to early_stop_frame
        for frame_idx in range(1, min(T, classifier.early_stop_frame) + 1):
            # Process up to current frame
            output = classifier(video[:, :frame_idx])
            
            logits = output['logits'][0]  # (2,)
            confidence = output['confidence'][0, 0].item()
            
            # Get prediction
            probs = torch.softmax(logits, dim=0)
            predicted_class = torch.argmax(probs).item()
            class_confidence = probs[predicted_class].item()
            
            # Check if confident enough to stop early
            if class_confidence >= classifier.confidence_threshold:
                decision = 'transient' if predicted_class == 0 else 'persistent'
                return decision, class_confidence, frame_idx
        
        # If we didn't stop early, process full sequence
        if T > classifier.early_stop_frame:
            output = classifier(video)
            logits = output['logits'][0]
            probs = torch.softmax(logits, dim=0)
            predicted_class = torch.argmax(probs).item()
            class_confidence = probs[predicted_class].item()
            decision = 'transient' if predicted_class == 0 else 'persistent'
            return decision, class_confidence, T
        else:
            # Already processed all frames
            decision = 'transient' if predicted_class == 0 else 'persistent'
            return decision, class_confidence, frame_idx


def compute_persistence_probability(classifier: EarlyPersistenceClassifier,
                                    video: torch.Tensor,
                                    max_frames: int = 4) -> List[float]:
    """
    Compute persistence probability over time (frame by frame).
    
    Args:
        classifier: Trained classifier
        video: Video tensor (T, C, H, W) or (B, T, C, H, W)
        max_frames: Maximum frames to process
        
    Returns:
        List of persistence probabilities for each frame
    """
    classifier.eval()
    
    if video.dim() == 4:
        video = video.unsqueeze(0)  # Add batch dimension
    
    T = min(video.shape[1], max_frames)
    probabilities = []
    
    with torch.no_grad():
        for frame_idx in range(1, T + 1):
            output = classifier(video[:, :frame_idx])
            logits = output['logits'][0]
            probs = torch.softmax(logits, dim=0)
            persistence_prob = probs[1].item()  # Prob of persistent class
            probabilities.append(persistence_prob)
    
    return probabilities
