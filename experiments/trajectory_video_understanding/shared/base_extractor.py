"""
Abstract Base Class for Video Feature Extractors
=================================================

This module defines the interface that all feature extractors must implement.
Different extractors (I3D, Slow/Fast, MagVIT, Transformer) will inherit from this base class.

Design Pattern: Strategy Pattern
- Each extractor is a strategy for feature extraction
- Extractors are interchangeable through common interface
- Enables parallel development and experimentation

Interface Requirements:
1. extract(video) -> features: Core extraction method
2. feature_dim: Property returning feature dimensionality

Input Format:
    video: torch.Tensor of shape (B, T, C, H, W)
        - B: Batch size
        - T: Number of frames (temporal dimension)
        - C: Number of channels (typically 3 for RGB)
        - H: Height in pixels
        - W: Width in pixels

Output Format:
    features: torch.Tensor of shape (B, T, D)
        - B: Batch size (same as input)
        - T: Number of frames (same as input)
        - D: Feature dimension (extractor-specific)

TDD Evidence: See artifacts/tdd_base_extractor_*.txt
"""

from abc import ABC, abstractmethod
import torch


class FeatureExtractor(ABC):
    """
    Abstract base class for video feature extractors.
    
    All concrete feature extractors (I3D, Slow/Fast, MagVIT, Transformer)
    must inherit from this class and implement:
    1. extract() method
    2. feature_dim property
    
    Example:
        >>> class MyExtractor(FeatureExtractor):
        ...     @property
        ...     def feature_dim(self):
        ...         return 512
        ...
        ...     def extract(self, video):
        ...         # Custom extraction logic
        ...         B, T = video.shape[0], video.shape[1]
        ...         return torch.randn(B, T, self.feature_dim)
        ...
        >>> extractor = MyExtractor()
        >>> video = torch.randn(2, 16, 3, 64, 64)  # 2 videos, 16 frames each
        >>> features = extractor.extract(video)    # Output: (2, 16, 512)
    """
    
    @abstractmethod
    def extract(self, video: torch.Tensor) -> torch.Tensor:
        """
        Extract features from video.
        
        This method must be implemented by all subclasses.
        
        Args:
            video (torch.Tensor): Input video tensor
                Shape: (B, T, C, H, W)
                - B: Batch size
                - T: Number of frames
                - C: Number of channels (typically 3 for RGB)
                - H: Height in pixels
                - W: Width in pixels
        
        Returns:
            torch.Tensor: Extracted features
                Shape: (B, T, D)
                - B: Batch size (same as input)
                - T: Number of frames (same as input)
                - D: Feature dimension (extractor-specific, = self.feature_dim)
        
        Raises:
            NotImplementedError: If subclass doesn't implement this method
        
        Note:
            - Output must preserve batch size and temporal dimension
            - Output must contain finite values (no NaN or Inf)
            - Implementation should be efficient and GPU-friendly
        """
        pass
    
    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """
        Return the dimensionality of extracted features.
        
        This property must be implemented by all subclasses.
        
        Returns:
            int: Feature dimension (D in output shape (B, T, D))
        
        Example:
            - I3D extractor might return 1024
            - Slow/Fast might return 2048
            - Transformer might return 512
            - MagVIT might return 256
        
        Note:
            This dimension must match the last dimension of the tensor
            returned by extract() method.
        """
        pass
    
    def __repr__(self):
        """String representation of the extractor."""
        return f"{self.__class__.__name__}(feature_dim={self.feature_dim})"
    
    def __str__(self):
        """Human-readable string representation."""
        return f"{self.__class__.__name__} [D={self.feature_dim}]"


# Example: Dummy extractor for testing and demonstration
class DummyExtractor(FeatureExtractor):
    """
    Minimal working example of a feature extractor.
    
    This extractor generates random features for testing purposes.
    It demonstrates the minimum requirements for a concrete extractor.
    
    Args:
        dim (int): Feature dimension (default: 256)
    """
    
    def __init__(self, dim: int = 256):
        """Initialize with specified feature dimension."""
        self._feature_dim = dim
    
    @property
    def feature_dim(self) -> int:
        """Return feature dimension."""
        return self._feature_dim
    
    def extract(self, video: torch.Tensor) -> torch.Tensor:
        """
        Extract random features (for testing only).
        
        Args:
            video: Input video (B, T, C, H, W)
        
        Returns:
            Random features (B, T, D)
        """
        B, T = video.shape[0], video.shape[1]
        return torch.randn(B, T, self.feature_dim)


if __name__ == "__main__":
    print("=" * 70)
    print("Feature Extractor Base Class - Demo")
    print("=" * 70)
    
    # Demonstrate that abstract class cannot be instantiated
    try:
        base = FeatureExtractor()
        print("❌ ERROR: Should not be able to instantiate abstract class")
    except TypeError as e:
        print(f"✅ Correct: Cannot instantiate abstract class")
        print(f"   Error: {e}")
    
    print()
    
    # Demonstrate working concrete implementation
    print("Creating DummyExtractor (concrete implementation)...")
    extractor = DummyExtractor(dim=512)
    print(f"✅ Created: {extractor}")
    
    print()
    
    # Test extraction
    print("Testing feature extraction...")
    video = torch.randn(2, 16, 3, 64, 64)
    print(f"Input video shape: {video.shape}")
    
    features = extractor.extract(video)
    print(f"Output features shape: {features.shape}")
    print(f"Expected shape: (2, 16, {extractor.feature_dim})")
    
    if features.shape == (2, 16, extractor.feature_dim):
        print("✅ Shape correct!")
    else:
        print("❌ Shape mismatch!")
    
    if torch.all(torch.isfinite(features)):
        print("✅ All values finite!")
    else:
        print("❌ Contains NaN or Inf!")
    
    print()
    print("=" * 70)
    print("Demo complete. Ready for TDD GREEN phase verification.")
    print("=" * 70)

