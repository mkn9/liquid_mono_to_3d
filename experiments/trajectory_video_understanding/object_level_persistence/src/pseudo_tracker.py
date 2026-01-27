"""
Pseudo-tracker using ground truth color information.
Fast implementation for quick attention validation.

White spheres → Persistent track (Track ID 1)
Red spheres → Transient tracks (Track ID 2+)
"""

import numpy as np
from typing import List, Tuple
from .object_detector import DetectedObject


class PseudoTracker:
    """Assigns tracks based on object color (ground truth)."""
    
    def __init__(self):
        self.next_transient_track_id = 2
        
    def assign_tracks(self, frame: np.ndarray, detections: List[DetectedObject]) -> List[int]:
        """
        Assign track IDs based on object color.
        
        Args:
            frame: (H, W, 3) RGB frame
            detections: List of detected objects
            
        Returns:
            List of track IDs (1 for persistent, 2+ for transient)
        """
        track_ids = []
        
        for detection in detections:
            # Extract object patch
            x, y, w, h = detection.bbox
            patch = frame[y:y+h, x:x+w]
            
            if patch.size == 0:
                track_ids.append(1)  # Default to persistent
                continue
            
            # Check color
            avg_color = patch.mean(axis=(0, 1))
            
            # White sphere: RGB close to [1.0, 1.0, 1.0]
            # Red sphere: RGB close to [0.9, 0.3, 0.3]
            
            if avg_color[0] > 0.7 and avg_color[1] > 0.7 and avg_color[2] > 0.7:
                # White → Persistent
                track_ids.append(1)
            else:
                # Red or other → Transient
                track_id = self.next_transient_track_id
                self.next_transient_track_id += 1
                track_ids.append(track_id)
        
        return track_ids
    
    def reset(self):
        """Reset transient track counter."""
        self.next_transient_track_id = 2

