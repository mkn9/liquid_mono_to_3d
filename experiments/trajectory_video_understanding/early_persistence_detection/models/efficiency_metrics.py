"""
Efficiency Metrics Tracking

Tracks time-to-decision, compute-per-track, and attention efficiency.
Provides comprehensive analysis of system performance.
"""

import torch
import numpy as np
from typing import Dict, List
import json
from pathlib import Path
import time


class EfficiencyMetrics:
    """
    Tracks efficiency metrics for early persistence detection.
    """
    def __init__(self, num_frames: int):
        """
        Initialize EfficiencyMetrics.
        
        Args:
            num_frames (int): Total number of frames in a full sequence.
        """
        self.num_frames = num_frames
        self.processed_frames = 0
        self.total_flops = 0
        self.start_time = time.time()

    def update_processed_frames(self, count: int = 1):
        """Increments the count of processed frames."""
        self.processed_frames += count

    def update_flops(self, flops: int):
        """Adds to the total FLOPs count."""
        self.total_flops += flops

    def compute_time_to_decision(self, decision_frame: int, fps: float) -> float:
        """
        Computes the time taken to make a decision.
        
        Args:
            decision_frame (int): The frame index at which the decision was made.
            fps (float): Frames per second of the video.
        
        Returns:
            float: Time to decision in seconds.
        """
        if fps <= 0:
            return float('inf')
        return (decision_frame + 1) / fps # +1 because frame_idx is 0-based

    def compute_compute_savings(self, frames_processed: int) -> float:
        """
        Computes the percentage of compute saved compared to processing all frames.
        
        Args:
            frames_processed (int): Number of frames actually processed.
        
        Returns:
            float: Compute savings as a percentage (0.0 to 1.0).
        """
        if self.num_frames <= 0:
            return 0.0
        return (self.num_frames - frames_processed) / self.num_frames

    def get_summary(self, decision_frame: int, fps: float = 10.0) -> dict:
        """
        Returns a summary of all collected efficiency metrics.
        
        Args:
            decision_frame (int): The frame index at which the decision was made.
            fps (float): Frames per second for time calculation.
        
        Returns:
            dict: A dictionary containing various efficiency metrics.
        """
        frames_processed = self.processed_frames
        compute_savings = self.compute_compute_savings(frames_processed)
        time_to_decision_sec = self.compute_time_to_decision(decision_frame, fps)
        
        return {
            "frames_processed": frames_processed,
            "compute_savings": compute_savings,
            "time_to_decision_sec": time_to_decision_sec,
            "total_flops": self.total_flops,
            "elapsed_real_time_sec": time.time() - self.start_time
        }


class EfficiencyTracker:
    """Tracks efficiency metrics across multiple tracks."""
    
    def __init__(self):
        """Initialize efficiency tracker."""
        self.total_tracks = 0
        self.decisions = []
        self.confidences = []
        self.decision_frames = []
        self.compute_used = []
        self.processing_times = []
        self.classes = []
    
    def add_track(self, decision: str, confidence: float, decision_frame: int,
                  compute_used: float, time_ms: float):
        """
        Add track to tracker.
        
        Args:
            decision: Decision ('persistent' or 'transient')
            confidence: Decision confidence
            decision_frame: Frame where decision was made
            compute_used: Compute budget used (0.0 to 1.0)
            time_ms: Processing time in milliseconds
        """
        self.total_tracks += 1
        self.decisions.append(decision)
        self.confidences.append(confidence)
        self.decision_frames.append(decision_frame)
        self.compute_used.append(compute_used)
        self.processing_times.append(time_ms)
        self.classes.append(decision)
    
    def get_summary(self) -> Dict:
        """
        Get summary statistics.
        
        Returns:
            Dictionary with aggregated metrics
        """
        if self.total_tracks == 0:
            return {
                'total_tracks': 0,
                'message': 'No tracks processed yet'
            }
        
        summary = {
            'total_tracks': self.total_tracks,
            'avg_decision_frame': float(np.mean(self.decision_frames)),
            'median_decision_frame': float(np.median(self.decision_frames)),
            'avg_confidence': float(np.mean(self.confidences)),
            'avg_compute_per_track': float(np.mean(self.compute_used)),
            'total_compute_saved': float(
                self.total_tracks * 1.0 - sum(self.compute_used)
            ),
            'avg_time_ms': float(np.mean(self.processing_times)),
            'total_time_ms': float(sum(self.processing_times)),
            'decisions': {
                'persistent': self.decisions.count('persistent'),
                'transient': self.decisions.count('transient')
            }
        }
        
        # Per-class statistics
        persistent_indices = [i for i, d in enumerate(self.decisions) if d == 'persistent']
        transient_indices = [i for i, d in enumerate(self.decisions) if d == 'transient']
        
        if persistent_indices:
            summary['persistent_stats'] = {
                'count': len(persistent_indices),
                'avg_decision_frame': float(np.mean([self.decision_frames[i] for i in persistent_indices])),
                'avg_compute': float(np.mean([self.compute_used[i] for i in persistent_indices])),
                'avg_time_ms': float(np.mean([self.processing_times[i] for i in persistent_indices]))
            }
        
        if transient_indices:
            summary['transient_stats'] = {
                'count': len(transient_indices),
                'avg_decision_frame': float(np.mean([self.decision_frames[i] for i in transient_indices])),
                'avg_compute': float(np.mean([self.compute_used[i] for i in transient_indices])),
                'avg_time_ms': float(np.mean([self.processing_times[i] for i in transient_indices]))
            }
        
        # Early stopping efficiency
        early_stops = sum(1 for f in self.decision_frames if f <= 4)
        summary['early_stop_rate'] = float(early_stops / self.total_tracks)
        
        return summary
    
    def save_to_file(self, output_path: Path):
        """Save tracking data to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'summary': self.get_summary(),
            'tracks': [
                {
                    'decision': self.decisions[i],
                    'confidence': self.confidences[i],
                    'decision_frame': self.decision_frames[i],
                    'compute_used': self.compute_used[i],
                    'time_ms': self.processing_times[i]
                }
                for i in range(self.total_tracks)
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)


def compute_time_to_decision(start_time: float, end_time: float,
                             decision_frame: int) -> Dict:
    """
    Compute time-to-decision metrics.
    
    Args:
        start_time: Start timestamp (seconds)
        end_time: End timestamp (seconds)
        decision_frame: Frame where decision was made
        
    Returns:
        Dictionary with time metrics
    """
    total_time_sec = end_time - start_time
    total_time_ms = total_time_sec * 1000
    
    time_per_frame_ms = total_time_ms / decision_frame if decision_frame > 0 else 0
    
    metrics = {
        'total_time_ms': total_time_ms,
        'total_time_sec': total_time_sec,
        'time_per_frame_ms': time_per_frame_ms,
        'decision_frame': decision_frame
    }
    
    return metrics


def compute_compute_per_track(compute_log: Dict) -> Dict:
    """
    Compute compute-per-track metrics.
    
    Args:
        compute_log: Dictionary with compute usage information
            - frames_processed: Number of frames processed
            - early_stopped: Whether processing stopped early
            - total_flops: Total FLOPs used (optional)
            - decision: Final decision
            
    Returns:
        Dictionary with compute metrics
    """
    frames_processed = compute_log.get('frames_processed', 0)
    early_stopped = compute_log.get('early_stopped', False)
    total_flops = compute_log.get('total_flops', 0)
    decision = compute_log.get('decision', 'unknown')
    
    # Compute efficiency (assuming full processing is 16 frames)
    max_frames = 16
    compute_saved = max_frames - frames_processed if early_stopped else 0
    efficiency_ratio = compute_saved / max_frames if max_frames > 0 else 0
    
    metrics = {
        'frames_processed': frames_processed,
        'early_stopped': early_stopped,
        'compute_saved_frames': compute_saved,
        'efficiency_ratio': efficiency_ratio,
        'decision': decision
    }
    
    if total_flops > 0:
        metrics['total_flops'] = total_flops
        metrics['flops_per_frame'] = total_flops / frames_processed if frames_processed > 0 else 0
    
    return metrics


def compute_attention_efficiency(attention_on_persistent: torch.Tensor,
                                 attention_on_transient: torch.Tensor) -> Dict:
    """
    Compute attention efficiency metrics.
    
    Args:
        attention_on_persistent: Attention weights on persistent frames
        attention_on_transient: Attention weights on transient frames
        
    Returns:
        Dictionary with attention efficiency metrics
    """
    # Convert to numpy
    if isinstance(attention_on_persistent, torch.Tensor):
        attention_on_persistent = attention_on_persistent.cpu().numpy()
    if isinstance(attention_on_transient, torch.Tensor):
        attention_on_transient = attention_on_transient.cpu().numpy()
    
    avg_attention_persistent = float(np.mean(attention_on_persistent))
    avg_attention_transient = float(np.mean(attention_on_transient))
    
    # Efficiency ratio (higher is better)
    if avg_attention_transient > 0:
        efficiency_ratio = avg_attention_persistent / avg_attention_transient
    else:
        efficiency_ratio = float('inf') if avg_attention_persistent > 0 else 1.0
    
    # Attention concentration (std dev - lower means more focused)
    std_persistent = float(np.std(attention_on_persistent))
    std_transient = float(np.std(attention_on_transient))
    
    efficiency = {
        'avg_attention_persistent': avg_attention_persistent,
        'avg_attention_transient': avg_attention_transient,
        'efficiency_ratio': efficiency_ratio,
        'std_attention_persistent': std_persistent,
        'std_attention_transient': std_transient,
        'attention_gap': avg_attention_persistent - avg_attention_transient
    }
    
    return efficiency
