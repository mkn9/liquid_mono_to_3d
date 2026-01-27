"""
TDD Tests for Early Persistence Detection System

Tests all four components:
1. Early persistence classifier with MagVIT
2. Attention visualization
3. Compute gating mechanism
4. Efficiency metrics tracking
"""

import pytest
import torch
import numpy as np
import json
from pathlib import Path
import sys
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.early_persistence_classifier import (
    EarlyPersistenceClassifier,
    get_early_decision,
    compute_persistence_probability
)

from models.attention_visualization import (
    AttentionVisualizer,
    extract_attention_weights,
    plot_attention_heatmap,
    save_attention_analysis
)

from models.compute_gating import (
    ComputeGate,
    should_continue_processing,
    allocate_compute_budget,
    get_gating_decision
)

from models.efficiency_metrics import (
    EfficiencyTracker,
    compute_time_to_decision,
    compute_compute_per_track,
    compute_attention_efficiency
)


class TestEarlyPersistenceClassifier:
    """Test early persistence classifier with MagVIT backbone."""
    
    def test_init(self):
        """Test classifier initialization with MagVIT."""
        classifier = EarlyPersistenceClassifier(
            feature_extractor='magvit',
            early_stop_frame=4,
            confidence_threshold=0.9
        )
        assert classifier.feature_extractor_type == 'magvit'
        assert classifier.early_stop_frame == 4
        assert classifier.confidence_threshold == 0.9
    
    def test_early_decision_on_short_sequence(self):
        """Test early decision on 3-frame sequence."""
        classifier = EarlyPersistenceClassifier(early_stop_frame=4)
        
        # Short sequence (3 frames) - should be classified as transient
        short_video = torch.randn(3, 3, 64, 64)
        decision, confidence, frame_idx = get_early_decision(classifier, short_video)
        
        assert decision in ['persistent', 'transient']
        assert 0.0 <= confidence <= 1.0
        assert frame_idx <= 4  # Should decide by frame 4
    
    def test_early_decision_on_long_sequence(self):
        """Test early decision on 16-frame sequence."""
        classifier = EarlyPersistenceClassifier(early_stop_frame=4)
        
        # Long sequence (16 frames) - should be classified as persistent
        long_video = torch.randn(16, 3, 64, 64)
        decision, confidence, frame_idx = get_early_decision(classifier, long_video)
        
        assert decision in ['persistent', 'transient']
        assert 0.0 <= confidence <= 1.0
        assert frame_idx <= 16
    
    def test_persistence_probability_computation(self):
        """Test computing persistence probability over time."""
        classifier = EarlyPersistenceClassifier()
        video = torch.randn(16, 3, 64, 64)
        
        # Get probabilities for frames 1, 2, 3, 4
        probs = compute_persistence_probability(classifier, video, max_frames=4)
        
        assert len(probs) <= 4
        assert all(0.0 <= p <= 1.0 for p in probs)
    
    def test_batch_inference(self):
        """Test batch inference on multiple videos."""
        classifier = EarlyPersistenceClassifier()
        
        # Batch of 4 videos
        batch = torch.randn(4, 16, 3, 64, 64)
        decisions, confidences, frame_indices = classifier.predict_batch(batch)
        
        assert len(decisions) == 4
        assert len(confidences) == 4
        assert len(frame_indices) == 4


class TestAttentionVisualization:
    """Test attention visualization module."""
    
    def test_visualizer_init(self):
        """Test attention visualizer initialization."""
        viz = AttentionVisualizer(num_heads=4, save_dir='./viz_test')
        assert viz.num_heads == 4
        assert viz.save_dir == Path('./viz_test')
    
    def test_extract_attention_weights(self):
        """Test extracting attention weights from model."""
        # Mock attention weights (num_heads, seq_len, seq_len)
        mock_attention = torch.randn(4, 16, 16).softmax(dim=-1)
        
        weights = extract_attention_weights(mock_attention)
        
        assert weights.shape[0] == 4  # num_heads
        assert weights.shape[1] == 16  # seq_len
    
    def test_plot_attention_heatmap(self, tmp_path):
        """Test plotting attention heatmap."""
        attention_weights = torch.randn(4, 16, 16).softmax(dim=-1)
        transient_frames = [5, 6, 10]
        
        output_path = tmp_path / "attention_test.png"
        plot_attention_heatmap(
            attention_weights,
            transient_frames=transient_frames,
            output_path=output_path
        )
        
        assert output_path.exists()
    
    def test_save_attention_analysis(self, tmp_path):
        """Test saving attention analysis to JSON."""
        analysis = {
            'avg_attention_persistent': 0.85,
            'avg_attention_transient': 0.15,
            'attention_ratio': 5.67,
            'frame_attention': [0.9, 0.8, 0.1, 0.05, 0.85]
        }
        
        output_file = tmp_path / "attention_analysis.json"
        save_attention_analysis(analysis, output_file)
        
        assert output_file.exists()
        with open(output_file, 'r') as f:
            loaded = json.load(f)
        assert loaded['attention_ratio'] == 5.67


class TestComputeGating:
    """Test compute gating mechanism."""
    
    def test_gate_init(self):
        """Test compute gate initialization."""
        gate = ComputeGate(
            confidence_threshold=0.9,
            early_stop_frame=4,
            compute_budget={'persistent': 1.0, 'transient': 0.2}
        )
        assert gate.confidence_threshold == 0.9
        assert gate.early_stop_frame == 4
        assert gate.compute_budget['persistent'] == 1.0
    
    def test_should_continue_processing_high_confidence(self):
        """Test gating decision with high confidence at early frame."""
        gate = ComputeGate(confidence_threshold=0.9)
        
        # High confidence transient at frame 2
        should_continue = should_continue_processing(
            confidence=0.95,
            predicted_class='transient',
            current_frame=2,
            gate=gate
        )
        
        assert should_continue is False  # Should stop early
    
    def test_should_continue_processing_low_confidence(self):
        """Test gating decision with low confidence."""
        gate = ComputeGate(confidence_threshold=0.9)
        
        # Low confidence at frame 2
        should_continue = should_continue_processing(
            confidence=0.6,
            predicted_class='transient',
            current_frame=2,
            gate=gate
        )
        
        assert should_continue is True  # Should continue processing
    
    def test_allocate_compute_budget(self):
        """Test compute budget allocation."""
        gate = ComputeGate(compute_budget={'persistent': 1.0, 'transient': 0.2})
        
        # Persistent track gets full budget
        persistent_budget = allocate_compute_budget('persistent', gate)
        assert persistent_budget == 1.0
        
        # Transient track gets reduced budget
        transient_budget = allocate_compute_budget('transient', gate)
        assert transient_budget == 0.2
    
    def test_gating_decision_with_metadata(self):
        """Test complete gating decision with tracking metadata."""
        gate = ComputeGate(confidence_threshold=0.9, early_stop_frame=4)
        
        decision_info = get_gating_decision(
            confidence=0.92,
            predicted_class='transient',
            current_frame=3,
            gate=gate
        )
        
        assert 'should_continue' in decision_info
        assert 'compute_budget' in decision_info
        assert 'decision_frame' in decision_info
        assert decision_info['decision_frame'] == 3


class TestEfficiencyMetrics:
    """Test efficiency metrics tracking."""
    
    def test_tracker_init(self):
        """Test efficiency tracker initialization."""
        tracker = EfficiencyTracker()
        assert tracker.total_tracks == 0
        assert len(tracker.decisions) == 0
    
    def test_compute_time_to_decision(self):
        """Test time-to-decision metric."""
        start_time = time.time()
        time.sleep(0.01)  # Small delay
        end_time = time.time()
        
        ttd = compute_time_to_decision(start_time, end_time, decision_frame=3)
        
        assert isinstance(ttd, dict)
        assert 'total_time_ms' in ttd
        assert 'time_per_frame_ms' in ttd
        assert ttd['decision_frame'] == 3
        assert ttd['total_time_ms'] > 0
    
    def test_compute_compute_per_track(self):
        """Test compute-per-track metric."""
        # Mock compute usage
        compute_log = {
            'frames_processed': 4,
            'early_stopped': True,
            'total_flops': 1e9,
            'decision': 'transient'
        }
        
        metrics = compute_compute_per_track(compute_log)
        
        assert 'frames_processed' in metrics
        assert 'early_stopped' in metrics
        assert metrics['frames_processed'] == 4
    
    def test_compute_attention_efficiency(self):
        """Test attention efficiency metric."""
        # Mock attention distribution
        attention_on_persistent = torch.tensor([0.9, 0.85, 0.88, 0.92])
        attention_on_transient = torch.tensor([0.1, 0.15, 0.12])
        
        efficiency = compute_attention_efficiency(
            attention_on_persistent,
            attention_on_transient
        )
        
        assert 'avg_attention_persistent' in efficiency
        assert 'avg_attention_transient' in efficiency
        assert 'efficiency_ratio' in efficiency
        assert efficiency['efficiency_ratio'] > 1.0  # More attention on persistent
    
    def test_tracker_accumulation(self):
        """Test accumulating metrics over multiple tracks."""
        tracker = EfficiencyTracker()
        
        # Track 1: Transient, early stop at frame 2
        tracker.add_track(
            decision='transient',
            confidence=0.95,
            decision_frame=2,
            compute_used=0.2,
            time_ms=5.0
        )
        
        # Track 2: Persistent, full processing
        tracker.add_track(
            decision='persistent',
            confidence=0.88,
            decision_frame=16,
            compute_used=1.0,
            time_ms=25.0
        )
        
        assert tracker.total_tracks == 2
        
        summary = tracker.get_summary()
        assert 'total_tracks' in summary
        assert 'avg_decision_frame' in summary
        assert 'avg_compute_per_track' in summary
        assert summary['total_tracks'] == 2


class TestIntegration:
    """Test integration of all four components."""
    
    def test_full_pipeline_transient_track(self):
        """Test complete pipeline on a transient track."""
        # Initialize all components
        classifier = EarlyPersistenceClassifier(
            feature_extractor='magvit',
            early_stop_frame=4,
            confidence_threshold=0.9
        )
        gate = ComputeGate(confidence_threshold=0.9)
        tracker = EfficiencyTracker()
        visualizer = AttentionVisualizer(num_heads=4)
        
        # Short video (3 frames) - simulates transient
        video = torch.randn(3, 3, 64, 64)
        
        # Run pipeline
        start_time = time.time()
        decision, confidence, frame_idx = get_early_decision(classifier, video)
        end_time = time.time()
        
        # Check gating
        should_continue = should_continue_processing(
            confidence, decision, frame_idx, gate
        )
        
        # Track metrics
        ttd = compute_time_to_decision(start_time, end_time, frame_idx)
        
        # Assertions
        assert decision in ['persistent', 'transient']
        assert frame_idx <= 4  # Early decision
        assert isinstance(should_continue, bool)
        assert ttd['decision_frame'] == frame_idx
    
    def test_full_pipeline_persistent_track(self):
        """Test complete pipeline on a persistent track."""
        classifier = EarlyPersistenceClassifier(early_stop_frame=4)
        gate = ComputeGate(confidence_threshold=0.9)
        tracker = EfficiencyTracker()
        
        # Long video (16 frames) - simulates persistent
        video = torch.randn(16, 3, 64, 64)
        
        start_time = time.time()
        decision, confidence, frame_idx = get_early_decision(classifier, video)
        end_time = time.time()
        
        should_continue = should_continue_processing(
            confidence, decision, frame_idx, gate
        )
        
        compute_budget = allocate_compute_budget(decision, gate)
        
        # Assertions
        assert decision in ['persistent', 'transient']
        assert isinstance(compute_budget, float)
        assert 0.0 <= compute_budget <= 1.0

