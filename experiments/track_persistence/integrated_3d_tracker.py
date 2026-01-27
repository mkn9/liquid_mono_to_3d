#!/usr/bin/env python3
"""
Integrated 3D Tracker with Persistence Filtering
================================================
Integrates the persistence filter into the 3D tracking pipeline.

Pipeline:
1. Generate/receive realistic 2D tracks from both cameras
2. Apply persistence filter to each track pair
3. Only triangulate tracks classified as persistent
4. Output clean 3D point cloud with reduced noise
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch

# Add to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from simple_3d_tracker import set_up_cameras, triangulate_tracks
from experiments.track_persistence.attention_persistence_model import AttentionPersistenceModel
from experiments.track_persistence.extract_track_features import TrackFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PersistenceFilter:
    """Wrapper class for persistence filtering in 3D tracking pipeline."""
    
    def __init__(
        self,
        model_checkpoint: str,
        magvit_checkpoint: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        threshold: float = 0.5
    ):
        """
        Initialize persistence filter.
        
        Args:
            model_checkpoint: Path to trained persistence model
            magvit_checkpoint: Path to pretrained MagVIT checkpoint
            device: Device to run on
            threshold: Classification threshold
        """
        self.device = device
        self.threshold = threshold
        
        # Load persistence model
        logger.info(f"Loading persistence model from {model_checkpoint}...")
        self.model = AttentionPersistenceModel()
        checkpoint = torch.load(model_checkpoint, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # Load feature extractor
        logger.info(f"Loading MagVIT feature extractor...")
        self.feature_extractor = TrackFeatureExtractor(
            magvit_checkpoint=magvit_checkpoint,
            device=device
        )
        
        logger.info("Persistence filter initialized")
    
    @torch.no_grad()
    def predict(
        self,
        track_pixels: np.ndarray
    ) -> Tuple[bool, float, Optional[np.ndarray]]:
        """
        Predict if a track is persistent.
        
        Args:
            track_pixels: Track pixel sequence (T, H, W, 3)
            
        Returns:
            is_persistent: Boolean classification
            confidence: Prediction confidence (probability)
            attention_weights: Frame importance scores (T,)
        """
        # Extract MagVIT features
        features = self.feature_extractor.extract_features(track_pixels)  # (T, D)
        
        # Convert to tensor
        features_tensor = torch.from_numpy(features).float().unsqueeze(0)  # (1, T, D)
        features_tensor = features_tensor.to(self.device)
        
        # Get prediction with attention
        predictions, probabilities, attn_weights = self.model.predict(
            features_tensor,
            threshold=self.threshold
        )
        
        is_persistent = bool(predictions[0].item())
        confidence = float(probabilities[0].item())
        
        # Get frame importance
        importance = self.model.get_frame_importance(features_tensor)
        if importance is not None:
            importance = importance[0].cpu().numpy()  # (T,)
        
        return is_persistent, confidence, importance
    
    def get_explanation(
        self,
        attention_weights: Optional[np.ndarray],
        track_duration: int
    ) -> str:
        """
        Generate human-readable explanation of filter decision.
        
        Args:
            attention_weights: Frame importance scores (T,)
            track_duration: Number of frames in track
            
        Returns:
            Explanation string
        """
        if attention_weights is None:
            return "No attention information available"
        
        # Find most important frames
        top_k = min(3, len(attention_weights))
        top_indices = np.argsort(attention_weights)[-top_k:][::-1]
        
        explanation = f"Track duration: {track_duration} frames. "
        explanation += f"Model focused on frames {top_indices.tolist()}, "
        explanation += f"with attention weights {attention_weights[top_indices].tolist()}"
        
        return explanation


class Integrated3DTracker:
    """
    3D tracker with integrated persistence filtering.
    
    Filters 2D tracks before triangulation to reduce noise in 3D reconstruction.
    """
    
    def __init__(
        self,
        persistence_filter: Optional[PersistenceFilter] = None,
        use_filter: bool = True
    ):
        """
        Initialize tracker.
        
        Args:
            persistence_filter: PersistenceFilter instance (optional)
            use_filter: Whether to apply persistence filter
        """
        self.persistence_filter = persistence_filter
        self.use_filter = use_filter and persistence_filter is not None
        
        # Set up cameras
        self.P1, self.P2, self.cam1_pos, self.cam2_pos = set_up_cameras()
        
        # Statistics
        self.stats = {
            'total_tracks': 0,
            'filtered_out': 0,
            'kept': 0,
            'triangulated_points': 0
        }
    
    def process_scene(
        self,
        camera1_tracks: List[Dict],
        camera2_tracks: List[Dict],
        verbose: bool = True
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        Process a scene with multiple tracks.
        
        Args:
            camera1_tracks: List of tracks from camera 1
                Each track: {'frames': List[int], 'bboxes': List[Tuple], 'pixels': np.ndarray}
            camera2_tracks: List of tracks from camera 2 (corresponding tracks)
            verbose: Whether to log detailed information
            
        Returns:
            reconstructed_3d: List of 3D point arrays
            filter_decisions: Dictionary of filter decisions per track
        """
        reconstructed_3d = []
        filter_decisions = {}
        
        if len(camera1_tracks) != len(camera2_tracks):
            logger.warning(f"Mismatched track counts: {len(camera1_tracks)} vs {len(camera2_tracks)}")
        
        num_tracks = min(len(camera1_tracks), len(camera2_tracks))
        self.stats['total_tracks'] += num_tracks
        
        for i in range(num_tracks):
            track1 = camera1_tracks[i]
            track2 = camera2_tracks[i]
            
            track_id = f"track_{i}"
            
            # Apply persistence filter if enabled
            if self.use_filter:
                # Combine tracks for filtering (use camera 1 pixels for now)
                pixels = track1['pixels']
                
                is_persistent, confidence, attention = self.persistence_filter.predict(pixels)
                
                decision = {
                    'is_persistent': is_persistent,
                    'confidence': confidence,
                    'attention_weights': attention,
                    'explanation': self.persistence_filter.get_explanation(
                        attention,
                        len(track1['frames'])
                    )
                }
                
                filter_decisions[track_id] = decision
                
                if verbose:
                    status = "KEEP" if is_persistent else "FILTER"
                    logger.info(f"{track_id}: {status} (confidence: {confidence:.3f})")
                
                if not is_persistent:
                    self.stats['filtered_out'] += 1
                    continue
                
                self.stats['kept'] += 1
            else:
                # No filtering, keep all tracks
                if verbose:
                    logger.info(f"{track_id}: KEEP (filtering disabled)")
                self.stats['kept'] += 1
            
            # Triangulate this track
            # Extract 2D points from bboxes (use center of bbox)
            sensor1_points = []
            sensor2_points = []
            
            for bbox1, bbox2 in zip(track1['bboxes'], track2['bboxes']):
                # Bbox format: (x, y, w, h)
                x1, y1, w1, h1 = bbox1
                x2, y2, w2, h2 = bbox2
                
                # Use center point
                center1 = (x1 + w1/2, y1 + h1/2)
                center2 = (x2 + w2/2, y2 + h2/2)
                
                sensor1_points.append(center1)
                sensor2_points.append(center2)
            
            # Triangulate
            points_3d = triangulate_tracks(
                np.array(sensor1_points),
                np.array(sensor2_points),
                self.P1,
                self.P2
            )
            
            reconstructed_3d.append(points_3d)
            self.stats['triangulated_points'] += len(points_3d)
        
        return reconstructed_3d, filter_decisions
    
    def get_statistics(self) -> Dict:
        """Get filtering statistics."""
        stats = self.stats.copy()
        if stats['total_tracks'] > 0:
            stats['filter_rate'] = stats['filtered_out'] / stats['total_tracks']
            stats['keep_rate'] = stats['kept'] / stats['total_tracks']
        return stats
    
    def visualize_results(
        self,
        reconstructed_3d: List[np.ndarray],
        output_path: Optional[str] = None
    ):
        """
        Visualize 3D reconstruction results.
        
        Args:
            reconstructed_3d: List of 3D point arrays
            output_path: Optional path to save visualization
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each track
        colors = plt.cm.rainbow(np.linspace(0, 1, len(reconstructed_3d)))
        
        for i, points_3d in enumerate(reconstructed_3d):
            ax.plot(
                points_3d[:, 0],
                points_3d[:, 1],
                points_3d[:, 2],
                'o-',
                color=colors[i],
                label=f'Track {i}',
                linewidth=2,
                markersize=6
            )
        
        # Plot cameras
        ax.scatter(
            [self.cam1_pos[0], self.cam2_pos[0]],
            [self.cam1_pos[1], self.cam2_pos[1]],
            [self.cam1_pos[2], self.cam2_pos[2]],
            c=['red', 'blue'],
            marker='*',
            s=500,
            label='Cameras'
        )
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Track Reconstruction with Persistence Filtering')
        ax.legend()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {output_path}")
        
        plt.close()


def main():
    """Demo integration with persistence filtering."""
    import argparse
    
    parser = argparse.ArgumentParser(description='3D tracking with persistence filtering')
    parser.add_argument('--model-checkpoint', type=str, required=True,
                        help='Path to trained persistence model')
    parser.add_argument('--magvit-checkpoint', type=str, required=True,
                        help='Path to pretrained MagVIT checkpoint')
    parser.add_argument('--use-filter', action='store_true',
                        help='Enable persistence filtering')
    parser.add_argument('--output-dir', type=str, default='output/integrated_3d',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize filter
    persistence_filter = None
    if args.use_filter:
        persistence_filter = PersistenceFilter(
            model_checkpoint=args.model_checkpoint,
            magvit_checkpoint=args.magvit_checkpoint
        )
    
    # Initialize tracker
    tracker = Integrated3DTracker(
        persistence_filter=persistence_filter,
        use_filter=args.use_filter
    )
    
    # Demo: Generate some example tracks (would be from real detector in production)
    # TODO: Replace with actual 2D track generation
    logger.info("Demo mode: Using synthetic tracks")
    
    # Get statistics
    stats = tracker.get_statistics()
    logger.info(f"Statistics: {stats}")


if __name__ == "__main__":
    main()

