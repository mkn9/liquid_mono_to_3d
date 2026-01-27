"""
TDD Tests for Evaluation Scripts

Tests for model evaluation, visualization, and analysis scripts.
"""

import pytest
import torch
import json
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

# Add paths for imports
eval_dir = Path(__file__).parent.parent
models_dir = eval_dir.parent / 'models'
sys.path.insert(0, str(eval_dir))
sys.path.insert(0, str(models_dir))

# Now import evaluation modules
import evaluate_model
import visualize_attention
import analyze_efficiency
import generate_report

# Import functions directly
from evaluate_model import (
    load_model_for_evaluation,
    evaluate_on_test_set,
    compute_confusion_matrix,
    save_evaluation_report
)

from visualize_attention import (
    extract_attention_from_model,
    generate_attention_heatmap,
    analyze_attention_distribution,
    save_visualization_batch
)

from analyze_efficiency import (
    load_efficiency_metrics,
    compute_efficiency_statistics,
    generate_efficiency_plots,
    create_efficiency_report
)

from generate_report import (
    collect_all_results,
    generate_markdown_report,
    generate_html_report,
    embed_images_in_html
)


class TestModelEvaluation:
    """Test model evaluation functionality."""
    
    @patch('evaluate_model.EarlyPersistenceClassifier')
    def test_load_model_for_evaluation(self, MockClassifier, tmp_path):
        """Test loading trained model for evaluation."""
        # Create dummy model file with proper structure
        model_path = tmp_path / "test_model.pt"
        dummy_state = {
            'model_state_dict': {},
            'feature_dim': 256,
            'hidden_dim': 128,
            'early_stop_frame': 4,
            'confidence_threshold': 0.9
        }
        torch.save(dummy_state, model_path)
        
        # Mock the model class
        mock_instance = MagicMock()
        mock_instance.eval = MagicMock()
        MockClassifier.return_value = mock_instance
        
        model = load_model_for_evaluation(str(model_path))
        
        assert model is not None
        assert hasattr(model, 'eval')
    
    @patch('evaluate_model.get_early_decision')
    @patch('evaluate_model.EfficiencyMetrics')
    def test_evaluate_on_test_set(self, MockMetrics, mock_get_early_decision):
        """Test evaluation on test dataset."""
        # Mock model
        model = MagicMock()
        model.eval = MagicMock()
        
        # Mock get_early_decision to return predictable results
        mock_get_early_decision.return_value = ("persistent", torch.tensor([0.95]), 3)
        
        # Mock EfficiencyMetrics
        mock_metrics_instance = MagicMock()
        mock_metrics_instance.processed_frames = 4
        mock_metrics_instance.compute_compute_savings = MagicMock(return_value=0.75)
        MockMetrics.return_value = mock_metrics_instance
        
        # Mock test samples
        test_samples = [
            (torch.randn(16, 3, 64, 64), 1),  # Persistent
            (torch.randn(3, 3, 64, 64), 0),   # Transient
        ]
        
        results = evaluate_on_test_set(model, test_samples)
        
        assert 'accuracy' in results
        assert 'early_stop_rate' in results
        assert 'avg_decision_frame' in results
        assert 0.0 <= results['accuracy'] <= 1.0
    
    def test_compute_confusion_matrix(self):
        """Test confusion matrix computation."""
        y_true = [0, 0, 1, 1, 0, 1]
        y_pred = [0, 1, 1, 1, 0, 0]
        
        cm = compute_confusion_matrix(y_true, y_pred)
        
        assert cm.shape == (2, 2)
        assert cm.sum() == len(y_true)
    
    def test_save_evaluation_report(self, tmp_path):
        """Test saving evaluation report."""
        metrics = {
            'accuracy': 0.85,
            'early_stop_rate': 0.65,
            'compute_savings': 0.72
        }
        
        output_file = tmp_path / "eval_report.json"
        save_evaluation_report(metrics, output_file)
        
        assert output_file.exists()
        
        with open(output_file, 'r') as f:
            loaded = json.load(f)
        assert loaded['accuracy'] == 0.85


class TestAttentionVisualization:
    """Test attention visualization functionality."""
    
    def test_extract_attention_from_model(self):
        """Test extracting attention weights from model."""
        # Mock model with feature_extractor
        model = MagicMock()
        model.eval = MagicMock()
        mock_features = torch.randn(1, 16, 256)
        model.feature_extractor = MagicMock(return_value=mock_features)
        
        video = torch.randn(1, 16, 3, 64, 64)
        
        attention_weights = extract_attention_from_model(model, video)
        
        assert attention_weights is not None
        assert isinstance(attention_weights, torch.Tensor)
    
    def test_generate_attention_heatmap(self, tmp_path):
        """Test generating attention heatmap."""
        attention_weights = torch.randn(4, 16, 16).softmax(dim=-1)
        transient_frames = [5, 6, 10]
        
        output_path = tmp_path / "attention_test.png"
        generate_attention_heatmap(attention_weights, transient_frames, output_path)
        
        assert output_path.exists()
    
    def test_analyze_attention_distribution(self):
        """Test analyzing attention distribution."""
        attention_weights = torch.randn(4, 16, 16).softmax(dim=-1)
        transient_frames = [5, 6, 10]
        
        analysis = analyze_attention_distribution(attention_weights, transient_frames)
        
        assert 'avg_attention_persistent' in analysis
        assert 'avg_attention_transient' in analysis
        assert 'attention_ratio' in analysis
    
    def test_save_visualization_batch(self, tmp_path):
        """Test saving batch of visualizations."""
        samples = [
            {
                'video': torch.randn(16, 3, 64, 64),
                'metadata': {'transient_frames': [5, 6]}
            }
        ]
        
        output_dir = tmp_path / "viz"
        save_visualization_batch(samples, output_dir)
        
        assert output_dir.exists()


class TestEfficiencyAnalysis:
    """Test efficiency analysis functionality."""
    
    def test_load_efficiency_metrics(self, tmp_path):
        """Test loading efficiency metrics."""
        metrics = {
            'total_tracks': 100,
            'avg_decision_frame': 3.2,
            'early_stop_rate': 0.68
        }
        
        metrics_file = tmp_path / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f)
        
        loaded = load_efficiency_metrics(metrics_file)
        
        assert loaded['total_tracks'] == 100
        assert loaded['early_stop_rate'] == 0.68
    
    def test_compute_efficiency_statistics(self):
        """Test computing efficiency statistics."""
        metrics = {
            'total_tracks': 100,
            'avg_decision_frame': 3.2,
            'avg_compute_per_track': 0.35,
            'early_stop_rate': 0.68
        }
        
        stats = compute_efficiency_statistics(metrics)
        
        assert 'compute_savings_percent' in stats
        assert 'avg_speedup' in stats
        assert stats['compute_savings_percent'] > 0
    
    def test_generate_efficiency_plots(self, tmp_path):
        """Test generating efficiency plots."""
        metrics = {
            'decision_frames': [2, 3, 3, 4, 2, 16, 16],
            'compute_used': [0.2, 0.3, 0.3, 0.4, 0.2, 1.0, 1.0]
        }
        
        output_dir = tmp_path / "plots"
        generate_efficiency_plots(metrics, output_dir)
        
        assert output_dir.exists()
    
    def test_create_efficiency_report(self, tmp_path):
        """Test creating efficiency report."""
        metrics = {
            'total_tracks': 100,
            'avg_decision_frame': 3.2,
            'early_stop_rate': 0.68,
            'total_compute_saved': 65.5
        }
        
        output_file = tmp_path / "efficiency_report.md"
        create_efficiency_report(metrics, output_file)
        
        assert output_file.exists()
        content = output_file.read_text()
        assert 'Efficiency Report' in content


class TestReportGeneration:
    """Test report generation functionality."""
    
    def test_collect_all_results(self, tmp_path):
        """Test collecting all results from directories."""
        # Create dummy result files
        (tmp_path / "evaluation").mkdir()
        (tmp_path / "visualizations").mkdir()
        (tmp_path / "analysis").mkdir()
        
        (tmp_path / "evaluation" / "metrics.json").write_text('{"accuracy": 0.85}')
        (tmp_path / "visualizations" / "sample_1.png").write_text('dummy')
        (tmp_path / "analysis" / "efficiency.json").write_text('{"savings": 0.72}')
        
        results = collect_all_results(tmp_path)
        
        assert 'evaluation' in results
        assert 'visualizations' in results
        assert 'analysis' in results
    
    def test_generate_markdown_report(self, tmp_path):
        """Test generating markdown report."""
        results = {
            'evaluation': {'evaluation_metrics': {'accuracy': 0.85, 'early_stop_rate': 0.68}},
            'efficiency': {'compute_savings': 0.72},
            'metadata': {
                'collection_time': '2024-01-25T12:00:00',
                'results_dir': str(tmp_path)
            }
        }
        
        output_file = tmp_path / "report.md"
        generate_markdown_report(results, output_file)
        
        assert output_file.exists()
        content = output_file.read_text()
        assert 'Early Persistence Detection' in content
        assert '0.85' in content or '85' in content
    
    def test_generate_html_report(self, tmp_path):
        """Test generating HTML report."""
        results = {
            'evaluation': {'evaluation_metrics': {'accuracy': 0.85, 'early_stop_rate': 0.68}},
            'visualizations': [],
            'metadata': {
                'collection_time': '2024-01-25T12:00:00',
                'results_dir': str(tmp_path)
            }
        }
        
        output_file = tmp_path / "report.html"
        generate_html_report(results, output_file)
        
        assert output_file.exists()
        content = output_file.read_text()
        assert '<html' in content and '</html>' in content
    
    def test_embed_images_in_html(self, tmp_path):
        """Test embedding images in HTML."""
        # Create dummy image
        image_path = tmp_path / "test.png"
        image_path.write_bytes(b'fake_png_data')
        
        html_template = '<html><img src="{{IMAGE_PATH}}"></html>'
        
        result = embed_images_in_html(html_template, str(image_path))
        
        assert 'data:image/png;base64' in result or '{{IMAGE_PATH}}' not in result

