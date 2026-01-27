#!/usr/bin/env python3
"""
Realistic 2D Track Generator
=============================
Generates realistic 2D track sequences that simulate real object detector output (e.g., YOLO).

Includes:
- Persistent objects (20-50 frames)
- Transient detections (2-5 frames): glare, reflections, shadows
- False positives (1 frame): noise
- Bounding boxes with visual appearance
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from dataclasses import dataclass, asdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


@dataclass
class Track2D:
    """Represents a 2D track from an object detector."""
    track_id: int
    frames: List[int]  # Frame numbers where detections occur
    bboxes: List[Tuple[int, int, int, int]]  # (x, y, w, h) for each frame
    confidences: List[float]  # Detection confidence per frame
    pixels: np.ndarray  # Video sequence: (T, H, W, 3) - crops of bounding boxes
    is_persistent: bool  # Ground truth label
    track_type: str  # 'persistent', 'brief', 'noise'
    duration: int  # Number of frames
    camera_id: int  # Which camera this track came from (0 or 1)
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'track_id': self.track_id,
            'frames': self.frames,
            'bboxes': self.bboxes,
            'confidences': self.confidences,
            'is_persistent': self.is_persistent,
            'track_type': self.track_type,
            'duration': self.duration,
            'camera_id': self.camera_id,
            # Note: pixels not included in JSON (saved separately as .npy)
        }


class Realistic2DTrackGenerator:
    """Generates realistic 2D track sequences for training persistence filter."""
    
    def __init__(
        self,
        img_width: int = 1280,
        img_height: int = 720,
        num_frames: int = 50,
        seed: Optional[int] = None
    ):
        """
        Initialize track generator.
        
        Args:
            img_width: Image width in pixels
            img_height: Image height in pixels
            num_frames: Total number of frames in video sequence
            seed: Random seed for reproducibility
        """
        self.img_width = img_width
        self.img_height = img_height
        self.num_frames = num_frames
        self.rng = np.random.RandomState(seed)
        
    def generate_persistent_track(
        self,
        track_id: int,
        camera_id: int,
        start_frame: int = 0,
        min_duration: int = 20,
        max_duration: int = 50
    ) -> Track2D:
        """
        Generate a persistent object track (long duration, stable appearance).
        
        Args:
            track_id: Unique track identifier
            camera_id: Camera index (0 or 1)
            start_frame: When track starts
            min_duration: Minimum track duration
            max_duration: Maximum track duration
            
        Returns:
            Track2D object
        """
        # Duration: long enough to be considered persistent
        duration = self.rng.randint(min_duration, min(max_duration + 1, self.num_frames - start_frame))
        frames = list(range(start_frame, start_frame + duration))
        
        # Initial position and velocity
        x = self.rng.randint(100, self.img_width - 200)
        y = self.rng.randint(100, self.img_height - 200)
        vx = self.rng.uniform(-3, 3)  # pixels/frame
        vy = self.rng.uniform(-3, 3)
        
        # Bounding box size (stable for persistent objects)
        w = self.rng.randint(40, 100)
        h = self.rng.randint(40, 100)
        
        # Generate bounding boxes with smooth motion
        bboxes = []
        confidences = []
        
        for i in range(duration):
            # Update position
            x += vx + self.rng.normal(0, 0.5)  # Add small noise
            y += vy + self.rng.normal(0, 0.5)
            
            # Keep within bounds
            x = np.clip(x, 0, self.img_width - w)
            y = np.clip(y, 0, self.img_height - h)
            
            # Slight size variation (object rotates, perspective change)
            w_current = max(20, int(w + self.rng.normal(0, 2)))
            h_current = max(20, int(h + self.rng.normal(0, 2)))
            
            bboxes.append((int(x), int(y), w_current, h_current))
            
            # High confidence for persistent objects
            confidences.append(self.rng.uniform(0.85, 0.99))
        
        # Generate visual appearance (simulated object pixels)
        pixels = self._generate_track_pixels(bboxes, track_type='persistent')
        
        return Track2D(
            track_id=track_id,
            frames=frames,
            bboxes=bboxes,
            confidences=confidences,
            pixels=pixels,
            is_persistent=True,
            track_type='persistent',
            duration=duration,
            camera_id=camera_id
        )
    
    def generate_brief_track(
        self,
        track_id: int,
        camera_id: int,
        start_frame: int = 0,
        min_duration: int = 2,
        max_duration: int = 5
    ) -> Track2D:
        """
        Generate a brief/transient detection (reflection, shadow, glare).
        
        Args:
            track_id: Unique track identifier
            camera_id: Camera index (0 or 1)
            start_frame: When track starts
            min_duration: Minimum track duration
            max_duration: Maximum track duration
            
        Returns:
            Track2D object
        """
        # Short duration
        duration = self.rng.randint(min_duration, max_duration + 1)
        frames = list(range(start_frame, start_frame + duration))
        
        # Random position
        x = self.rng.randint(50, self.img_width - 150)
        y = self.rng.randint(50, self.img_height - 150)
        
        # Smaller, less stable bounding boxes
        w = self.rng.randint(20, 60)
        h = self.rng.randint(20, 60)
        
        bboxes = []
        confidences = []
        
        for i in range(duration):
            # More erratic motion for transients
            x += self.rng.normal(0, 5)
            y += self.rng.normal(0, 5)
            
            # Keep within bounds
            x = np.clip(x, 0, self.img_width - w)
            y = np.clip(y, 0, self.img_height - h)
            
            # Size varies more
            w_current = max(15, int(w + self.rng.normal(0, 5)))
            h_current = max(15, int(h + self.rng.normal(0, 5)))
            
            bboxes.append((int(x), int(y), w_current, h_current))
            
            # Lower confidence, possibly declining over time
            conf = self.rng.uniform(0.5, 0.8) * (1.0 - 0.15 * i / duration)
            confidences.append(max(0.3, conf))
        
        # Generate visual appearance (less stable)
        pixels = self._generate_track_pixels(bboxes, track_type='brief')
        
        return Track2D(
            track_id=track_id,
            frames=frames,
            bboxes=bboxes,
            confidences=confidences,
            pixels=pixels,
            is_persistent=False,
            track_type='brief',
            duration=duration,
            camera_id=camera_id
        )
    
    def generate_noise_track(
        self,
        track_id: int,
        camera_id: int,
        start_frame: int = 0
    ) -> Track2D:
        """
        Generate a false positive detection (single frame noise).
        
        Args:
            track_id: Unique track identifier
            camera_id: Camera index (0 or 1)
            start_frame: When detection occurs
            
        Returns:
            Track2D object
        """
        # Single frame only
        duration = 1
        frames = [start_frame]
        
        # Random position
        x = self.rng.randint(0, self.img_width - 50)
        y = self.rng.randint(0, self.img_height - 50)
        
        # Small bounding box
        w = self.rng.randint(10, 40)
        h = self.rng.randint(10, 40)
        
        bboxes = [(int(x), int(y), w, h)]
        
        # Low confidence
        confidences = [self.rng.uniform(0.3, 0.6)]
        
        # Generate visual appearance (random noise)
        pixels = self._generate_track_pixels(bboxes, track_type='noise')
        
        return Track2D(
            track_id=track_id,
            frames=frames,
            bboxes=bboxes,
            confidences=confidences,
            pixels=pixels,
            is_persistent=False,
            track_type='noise',
            duration=duration,
            camera_id=camera_id
        )
    
    def _generate_track_pixels(
        self,
        bboxes: List[Tuple[int, int, int, int]],
        track_type: str,
        target_size: Tuple[int, int] = (64, 64)
    ) -> np.ndarray:
        """
        Generate simulated pixel appearance for track.
        
        This creates synthetic "object" appearances that resemble what a real
        detector would extract from bounding boxes.
        
        Args:
            bboxes: List of bounding boxes
            track_type: 'persistent', 'brief', or 'noise'
            target_size: Resize all crops to this size
            
        Returns:
            np.ndarray of shape (T, H, W, 3) with pixel values 0-255
        """
        T = len(bboxes)
        H, W = target_size
        pixels = np.zeros((T, H, W, 3), dtype=np.uint8)
        
        # Generate base appearance depending on track type
        if track_type == 'persistent':
            # Stable object appearance (e.g., colored circle)
            base_color = self.rng.randint(50, 255, size=3)
            for i in range(T):
                # Create a simple shape
                img = np.ones((H, W, 3), dtype=np.uint8) * 30  # Dark background
                center = (W // 2, H // 2)
                radius = min(W, H) // 3
                
                # Add slight color variation over time
                color = tuple(np.clip(base_color + self.rng.randint(-10, 10, size=3), 0, 255).tolist())
                cv2.circle(img, center, radius, color, -1)
                
                # Add some texture
                noise = self.rng.randint(-10, 10, size=(H, W, 3))
                img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                
                pixels[i] = img
                
        elif track_type == 'brief':
            # Fading appearance (reflection/glare)
            for i in range(T):
                img = np.ones((H, W, 3), dtype=np.uint8) * 30
                
                # Brightness decreases over time
                brightness = int(255 * (1.0 - i / T))
                color = (brightness, brightness, brightness)
                
                # Create blob-like shape
                center = (W // 2 + self.rng.randint(-5, 5), H // 2 + self.rng.randint(-5, 5))
                axes = (W // 4, H // 4)
                cv2.ellipse(img, center, axes, 0, 0, 360, color, -1)
                
                # Add more noise
                noise = self.rng.randint(-30, 30, size=(H, W, 3))
                img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                
                pixels[i] = img
                
        else:  # noise
            # Random appearance (false positive)
            img = self.rng.randint(0, 100, size=(H, W, 3), dtype=np.uint8)
            pixels[0] = img
        
        return pixels
    
    def generate_scene(
        self,
        scene_id: int,
        num_persistent: int = 2,
        num_brief: int = 3,
        num_noise: int = 2,
        camera_id: int = 0
    ) -> List[Track2D]:
        """
        Generate a complete scene with multiple tracks.
        
        Args:
            scene_id: Unique scene identifier
            num_persistent: Number of persistent tracks
            num_brief: Number of brief tracks
            num_noise: Number of noise tracks
            camera_id: Camera index (0 or 1)
            
        Returns:
            List of Track2D objects
        """
        tracks = []
        track_id = scene_id * 1000  # Unique IDs per scene
        
        # Generate persistent tracks
        for i in range(num_persistent):
            start_frame = self.rng.randint(0, 10)  # Start early
            track = self.generate_persistent_track(track_id, camera_id, start_frame)
            tracks.append(track)
            track_id += 1
        
        # Generate brief tracks
        for i in range(num_brief):
            start_frame = self.rng.randint(0, self.num_frames - 10)
            track = self.generate_brief_track(track_id, camera_id, start_frame)
            tracks.append(track)
            track_id += 1
        
        # Generate noise tracks
        for i in range(num_noise):
            start_frame = self.rng.randint(0, self.num_frames)
            track = self.generate_noise_track(track_id, camera_id, start_frame)
            tracks.append(track)
            track_id += 1
        
        return tracks


def generate_dataset(
    output_dir: Path,
    num_scenes: int = 1000,
    persistent_ratio: float = 0.6,
    brief_ratio: float = 0.3,
    noise_ratio: float = 0.1,
    seed: Optional[int] = 42
):
    """
    Generate a complete dataset of realistic 2D tracks.
    
    Args:
        output_dir: Directory to save dataset
        num_scenes: Number of scenes to generate
        persistent_ratio: Fraction of tracks that are persistent
        brief_ratio: Fraction of tracks that are brief
        noise_ratio: Fraction of tracks that are noise
        seed: Random seed
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generator = Realistic2DTrackGenerator(seed=seed)
    
    all_tracks = []
    all_metadata = []
    
    print(f"Generating {num_scenes} scenes...")
    
    for scene_id in range(num_scenes):
        if scene_id % 100 == 0:
            print(f"Generated {scene_id}/{num_scenes} scenes")
        
        # Vary number of tracks per scene
        total_tracks = generator.rng.randint(5, 10)
        num_persistent = int(total_tracks * persistent_ratio)
        num_brief = int(total_tracks * brief_ratio)
        num_noise = total_tracks - num_persistent - num_brief
        
        # Generate for both cameras
        for camera_id in [0, 1]:
            tracks = generator.generate_scene(
                scene_id,
                num_persistent=num_persistent,
                num_brief=num_brief,
                num_noise=num_noise,
                camera_id=camera_id
            )
            
            for track in tracks:
                # Save pixels separately (too large for JSON)
                pixel_file = output_dir / f"track_{track.track_id}_pixels.npy"
                np.save(pixel_file, track.pixels)
                
                # Save metadata
                all_metadata.append(track.to_dict())
                all_tracks.append(track)
    
    # Save metadata JSON
    metadata_file = output_dir / "tracks_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    # Save summary statistics
    summary = {
        'total_scenes': num_scenes,
        'total_tracks': len(all_tracks),
        'num_persistent': sum(1 for t in all_tracks if t.is_persistent),
        'num_brief': sum(1 for t in all_tracks if t.track_type == 'brief'),
        'num_noise': sum(1 for t in all_tracks if t.track_type == 'noise'),
        'avg_persistent_duration': np.mean([t.duration for t in all_tracks if t.is_persistent]),
        'avg_brief_duration': np.mean([t.duration for t in all_tracks if t.track_type == 'brief']),
    }
    
    summary_file = output_dir / "dataset_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nDataset generated successfully!")
    print(f"Total tracks: {len(all_tracks)}")
    print(f"Persistent: {summary['num_persistent']}")
    print(f"Brief: {summary['num_brief']}")
    print(f"Noise: {summary['num_noise']}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    # Generate dataset
    output_dir = Path("experiments/track_persistence/data/realistic_2d_tracks")
    generate_dataset(
        output_dir=output_dir,
        num_scenes=1000,
        seed=42
    )

