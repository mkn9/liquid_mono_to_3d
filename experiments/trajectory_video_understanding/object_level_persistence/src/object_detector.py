"""
Object detector for sphere detection in trajectory videos.

TDD GREEN phase - Implementation to pass tests.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage


@dataclass
class DetectedObject:
    """Represents a detected object in a frame."""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    class_id: int
    
    def center(self) -> Tuple[int, int]:
        """Compute center of bounding box."""
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)
    
    def area(self) -> float:
        """Compute area of bounding box."""
        _, _, w, h = self.bbox
        return w * h
    
    def iou(self, other: 'DetectedObject') -> float:
        """Compute IoU with another detection."""
        x1, y1, w1, h1 = self.bbox
        x2, y2, w2, h2 = other.bbox
        
        # Compute intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Compute union
        area1 = self.area()
        area2 = other.area()
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


class ObjectDetector:
    """Detects spheres in video frames using blob detection."""
    
    def __init__(self, input_size=(64, 64), confidence_threshold=0.5, nms_threshold=0.3):
        """Initialize detector."""
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.device = 'cpu'
        
        # Parameters for blob detection
        self.min_sigma = 1
        self.max_sigma = 5
        self.threshold = 0.1
    
    def detect(self, frame: np.ndarray) -> List[DetectedObject]:
        """
        Detect objects in a single frame using blob detection.
        
        Args:
            frame: (H, W, 3) RGB frame
            
        Returns:
            List of DetectedObject instances
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = np.mean(frame, axis=2)
        else:
            gray = frame
        
        # Find bright regions (spheres are bright)
        # Lower threshold to catch red spheres (grayscale ~0.5)
        threshold = 0.4
        binary = gray > threshold
        
        # Label connected components
        labeled, num_features = ndimage.label(binary)
        
        detections = []
        
        for i in range(1, num_features + 1):
            # Get component mask
            mask = labeled == i
            
            # Get bounding box
            coords = np.where(mask)
            if len(coords[0]) == 0:
                continue
            
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            
            w = x_max - x_min + 1
            h = y_max - y_min + 1
            
            # Skip tiny detections
            if w < 3 or h < 3:
                continue
            
            # Compute confidence based on brightness and size
            region = gray[y_min:y_max+1, x_min:x_max+1]
            avg_brightness = region.mean()
            
            # Compute confidence based on multiple factors
            # Size score: prefer objects of reasonable size (spheres are ~7x7 = 49 pixels)
            size_score = min(w * h / 50.0, 1.0)
            
            # Shape score: prefer square-ish objects (spheres have similar w and h)
            aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0
            shape_score = aspect_ratio
            
            # Brightness score: normalized, but clamped to avoid penalizing dimmer objects
            brightness_score = min(avg_brightness * 2.0, 1.0)
            
            # Combined confidence (prioritize shape and size over brightness)
            confidence = shape_score * 0.4 + size_score * 0.4 + brightness_score * 0.2
            
            # Apply confidence threshold
            if confidence < self.confidence_threshold:
                continue
            
            detections.append(DetectedObject(
                bbox=(int(x_min), int(y_min), int(w), int(h)),
                confidence=float(confidence),
                class_id=0
            ))
        
        # Apply NMS
        detections = self._apply_nms(detections)
        
        return detections
    
    def detect_batch(self, frames: np.ndarray) -> List[List[DetectedObject]]:
        """
        Detect objects in batch of frames.
        
        Args:
            frames: (N, H, W, 3) batch of frames
            
        Returns:
            List of detection lists, one per frame
        """
        batch_detections = []
        for frame in frames:
            detections = self.detect(frame)
            batch_detections.append(detections)
        return batch_detections
    
    def _apply_nms(self, detections: List[DetectedObject]) -> List[DetectedObject]:
        """Apply non-maximum suppression to remove overlapping detections."""
        if len(detections) == 0:
            return detections
        
        # Sort by confidence (descending)
        detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        
        keep = []
        while len(detections) > 0:
            # Keep highest confidence detection
            best = detections[0]
            keep.append(best)
            detections = detections[1:]
            
            # Remove overlapping detections
            filtered = []
            for det in detections:
                iou = best.iou(det)
                if iou < self.nms_threshold:
                    filtered.append(det)
            detections = filtered
        
        return keep
    
    def to(self, device: str):
        """Move detector to device."""
        self.device = device
        return self
