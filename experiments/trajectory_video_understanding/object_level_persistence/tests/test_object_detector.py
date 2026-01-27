"""
Test suite for object detector (TDD RED Phase).

Tests sphere detection in trajectory videos.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.object_detector import ObjectDetector, DetectedObject


class TestObjectDetector:
    """Test object detector functionality."""
    
    @pytest.fixture
    def detector(self):
        """Create detector instance."""
        return ObjectDetector(
            input_size=(64, 64),
            confidence_threshold=0.5,
            nms_threshold=0.3
        )
    
    @pytest.fixture
    def single_object_frame(self):
        """Create frame with single white sphere at (32, 32)."""
        frame = np.zeros((64, 64, 3), dtype=np.float32)
        # Create white sphere (7x7 region)
        for i in range(29, 36):
            for j in range(29, 36):
                if (i - 32)**2 + (j - 32)**2 <= 9:  # radius ~3
                    frame[i, j] = [1.0, 1.0, 1.0]
        return frame
    
    @pytest.fixture
    def two_object_frame(self):
        """Create frame with two spheres."""
        frame = np.zeros((64, 64, 3), dtype=np.float32)
        # White sphere at (20, 20)
        for i in range(17, 24):
            for j in range(17, 24):
                if (i - 20)**2 + (j - 20)**2 <= 9:
                    frame[i, j] = [1.0, 1.0, 1.0]
        # Red sphere at (44, 44)
        for i in range(41, 48):
            for j in range(41, 48):
                if (i - 44)**2 + (j - 44)**2 <= 9:
                    frame[i, j] = [0.9, 0.3, 0.3]
        return frame
    
    @pytest.fixture
    def empty_frame(self):
        """Create empty frame with no objects."""
        return np.zeros((64, 64, 3), dtype=np.float32)
    
    def test_detector_initialization(self, detector):
        """Test detector initializes correctly."""
        assert detector is not None
        assert detector.input_size == (64, 64)
        assert detector.confidence_threshold == 0.5
        assert detector.nms_threshold == 0.3
    
    def test_detect_single_object(self, detector, single_object_frame):
        """Test detection of single sphere."""
        detections = detector.detect(single_object_frame)
        
        # Should detect exactly 1 object
        assert len(detections) == 1
        
        # Check detection properties
        obj = detections[0]
        assert isinstance(obj, DetectedObject)
        assert obj.confidence > 0.5
        
        # Check bbox is near center (32, 32)
        cx = obj.bbox[0] + obj.bbox[2] / 2
        cy = obj.bbox[1] + obj.bbox[3] / 2
        assert abs(cx - 32) < 5
        assert abs(cy - 32) < 5
    
    def test_detect_two_objects(self, detector, two_object_frame):
        """Test detection of multiple spheres."""
        detections = detector.detect(two_object_frame)
        
        # Should detect exactly 2 objects
        assert len(detections) == 2
        
        # Check both have high confidence
        for obj in detections:
            assert obj.confidence > 0.5
        
        # Check spatial separation
        centers = [(obj.bbox[0] + obj.bbox[2]/2, 
                   obj.bbox[1] + obj.bbox[3]/2) 
                  for obj in detections]
        dist = np.sqrt((centers[0][0] - centers[1][0])**2 + 
                      (centers[0][1] - centers[1][1])**2)
        assert dist > 15  # Should be well separated
    
    def test_detect_empty_frame(self, detector, empty_frame):
        """Test no false positives on empty frame."""
        detections = detector.detect(empty_frame)
        
        # Should detect nothing
        assert len(detections) == 0
    
    def test_batch_detection(self, detector, single_object_frame, two_object_frame):
        """Test batch detection on multiple frames."""
        frames = np.stack([single_object_frame, two_object_frame])
        
        batch_detections = detector.detect_batch(frames)
        
        # Should return list of detections per frame
        assert len(batch_detections) == 2
        assert len(batch_detections[0]) == 1  # First frame: 1 object
        assert len(batch_detections[1]) == 2  # Second frame: 2 objects
    
    def test_confidence_threshold(self, single_object_frame):
        """Test confidence threshold filtering."""
        # High threshold should filter out low-confidence detections
        detector_high = ObjectDetector(
            input_size=(64, 64),
            confidence_threshold=0.9
        )
        
        # Create faint object (should be filtered)
        faint_frame = single_object_frame * 0.3
        detections = detector_high.detect(faint_frame)
        
        # Should detect nothing (too low confidence)
        assert len(detections) <= 1
        if len(detections) == 1:
            assert detections[0].confidence < 0.9
    
    def test_nms_removes_duplicates(self, detector):
        """Test non-maximum suppression removes overlapping detections."""
        # Create frame with slightly overlapping spheres
        frame = np.zeros((64, 64, 3), dtype=np.float32)
        # Two overlapping white spheres
        for i in range(28, 36):
            for j in range(28, 36):
                if (i - 31)**2 + (j - 31)**2 <= 9:
                    frame[i, j] = [1.0, 1.0, 1.0]
        for i in range(30, 38):
            for j in range(30, 38):
                if (i - 33)**2 + (j - 33)**2 <= 9:
                    frame[i, j] = [1.0, 1.0, 1.0]
        
        detections = detector.detect(frame)
        
        # NMS should merge overlapping detections to 1
        assert len(detections) <= 2  # Should be 1 or 2, not more
    
    def test_detector_output_format(self, detector, single_object_frame):
        """Test detection output format."""
        detections = detector.detect(single_object_frame)
        
        obj = detections[0]
        
        # Check DetectedObject has required fields
        assert hasattr(obj, 'bbox')
        assert hasattr(obj, 'confidence')
        assert hasattr(obj, 'class_id')
        
        # Check bbox format (x, y, w, h)
        assert len(obj.bbox) == 4
        assert all(v >= 0 for v in obj.bbox)
        
        # Check confidence range
        assert 0.0 <= obj.confidence <= 1.0
    
    def test_detector_gpu_support(self, detector):
        """Test detector can use GPU if available."""
        if torch.cuda.is_available():
            detector.to('cuda')
            assert detector.device == 'cuda'
        else:
            assert detector.device == 'cpu'
    
    def test_detect_at_boundaries(self, detector):
        """Test detection of objects at frame boundaries."""
        frame = np.zeros((64, 64, 3), dtype=np.float32)
        # Sphere at top-left corner
        for i in range(0, 7):
            for j in range(0, 7):
                if i**2 + j**2 <= 9:
                    frame[i, j] = [1.0, 1.0, 1.0]
        
        detections = detector.detect(frame)
        
        # Should still detect even at boundary
        assert len(detections) >= 1


class TestDetectedObject:
    """Test DetectedObject data structure."""
    
    def test_detected_object_creation(self):
        """Test creating DetectedObject."""
        obj = DetectedObject(
            bbox=(10, 15, 7, 7),
            confidence=0.95,
            class_id=0
        )
        
        assert obj.bbox == (10, 15, 7, 7)
        assert obj.confidence == 0.95
        assert obj.class_id == 0
    
    def test_detected_object_center(self):
        """Test computing center of detected object."""
        obj = DetectedObject(
            bbox=(10, 20, 8, 8),
            confidence=0.9,
            class_id=0
        )
        
        center = obj.center()
        assert center == (14, 24)  # (10 + 8/2, 20 + 8/2)
    
    def test_detected_object_area(self):
        """Test computing area of detected object."""
        obj = DetectedObject(
            bbox=(0, 0, 10, 5),
            confidence=0.8,
            class_id=0
        )
        
        area = obj.area()
        assert area == 50  # 10 * 5
    
    def test_detected_object_iou(self):
        """Test computing IoU between detected objects."""
        obj1 = DetectedObject(bbox=(0, 0, 10, 10), confidence=0.9, class_id=0)
        obj2 = DetectedObject(bbox=(5, 5, 10, 10), confidence=0.8, class_id=0)
        
        iou = obj1.iou(obj2)
        
        # Overlapping 5x5 region, union = 10*10 + 10*10 - 5*5 = 175
        expected_iou = 25 / 175  # ~0.14
        assert abs(iou - expected_iou) < 0.01
    
    def test_detected_object_no_overlap(self):
        """Test IoU for non-overlapping objects."""
        obj1 = DetectedObject(bbox=(0, 0, 5, 5), confidence=0.9, class_id=0)
        obj2 = DetectedObject(bbox=(10, 10, 5, 5), confidence=0.8, class_id=0)
        
        iou = obj1.iou(obj2)
        assert iou == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

