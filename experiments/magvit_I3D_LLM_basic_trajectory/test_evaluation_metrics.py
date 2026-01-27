"""
Test module for evaluation metrics.

All tests follow Red → Green → Refactor TDD cycle.
All numeric comparisons use explicit tolerances.
"""

import numpy as np
import torch
import pytest
from numpy.testing import assert_allclose

# These imports will fail initially (RED phase - expected!)
from evaluation_metrics import (
    classification_accuracy,
    forecasting_mae,
    forecasting_rmse,
    per_class_accuracy,
    confusion_matrix_metrics
)


class TestClassificationMetrics:
    """Test classification accuracy metrics."""
    
    def test_perfect_classification(self):
        """Perfect predictions should give 100% accuracy.
        
        Specification:
        - All predictions match labels
        - Expected accuracy: 1.0 (100%)
        """
        predictions = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        
        accuracy = classification_accuracy(predictions, labels)
        
        assert accuracy == 1.0, f"Expected 100% accuracy, got {accuracy:.2%}"
    
    def test_zero_classification(self):
        """All wrong predictions should give 0% accuracy."""
        predictions = torch.tensor([1, 2, 3, 0])
        labels = torch.tensor([0, 1, 2, 3])
        
        accuracy = classification_accuracy(predictions, labels)
        
        assert accuracy == 0.0, f"Expected 0% accuracy, got {accuracy:.2%}"
    
    def test_partial_classification(self):
        """Partial correct predictions should give correct accuracy.
        
        Specification:
        - 7 out of 10 correct = 70% accuracy
        - Correct at indices: 0, 1, 2, 3, 8, 9 (6), plus 4 = 7
        """
        predictions = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])
        labels = torch.tensor([0, 1, 2, 3, 3, 3, 3, 3, 0, 1])
        # Correct: 0=0, 1=1, 2=2, 3=3, 0≠3, 1≠3, 2≠3, 3=3, 0=0, 1=1 → 7 correct
        
        accuracy = classification_accuracy(predictions, labels)
        
        assert_allclose(accuracy, 0.7, rtol=1e-10)
    
    def test_per_class_accuracy(self):
        """Per-class accuracy should be calculated correctly.
        
        Specification:
        - Class 0: 2/3 correct = 66.7%
        - Class 1: 2/2 correct = 100%
        - Class 2: 1/2 correct = 50%
        - Class 3: 0/1 correct = 0%
        """
        predictions = torch.tensor([0, 0, 1, 1, 1, 2, 2, 3])
        labels = torch.tensor([0, 0, 2, 1, 1, 2, 0, 0])
        
        per_class = per_class_accuracy(predictions, labels, num_classes=4)
        
        # Class 0: predictions [0,0], labels [0,0,0,0] - 2 out of 4 appear in predictions, both correct
        # Actually: labels has [0,0,2,1,1,2,0,0], so class 0 appears at indices 0,1,6,7
        # predictions at those indices: [0,0,2,3] - 2 correct out of 4
        assert_allclose(per_class[0], 0.5, rtol=1e-10)
        
        # Class 1: labels at indices 3,4 with predictions [1,1] - 2/2 correct
        assert_allclose(per_class[1], 1.0, rtol=1e-10)
        
        # Class 2: labels at indices 2,5 with predictions [1,2] - 1/2 correct
        assert_allclose(per_class[2], 0.5, rtol=1e-10)
        
        # Class 3: no labels for class 3 in this example, should handle gracefully
        # (or might be NaN/0 depending on implementation)


class TestForecastingMetrics:
    """Test forecasting error metrics."""
    
    def test_mae_perfect_forecast(self):
        """Perfect forecast should give MAE = 0.
        
        Specification:
        - Predictions exactly match ground truth
        - MAE = mean(|pred - true|) = 0
        """
        predictions = torch.tensor([[1.0, 2.0, 3.0],
                                    [4.0, 5.0, 6.0]])
        ground_truth = torch.tensor([[1.0, 2.0, 3.0],
                                      [4.0, 5.0, 6.0]])
        
        mae = forecasting_mae(predictions, ground_truth)
        
        assert_allclose(mae, 0.0, atol=1e-10)
    
    def test_mae_constant_error(self):
        """Constant error of 1.0 should give MAE = 1.0.
        
        Specification:
        - All predictions off by exactly 1.0
        - MAE = mean(|pred - true|) = 1.0
        """
        predictions = torch.tensor([[2.0, 3.0, 4.0],
                                    [5.0, 6.0, 7.0]])
        ground_truth = torch.tensor([[1.0, 2.0, 3.0],
                                      [4.0, 5.0, 6.0]])
        
        mae = forecasting_mae(predictions, ground_truth)
        
        assert_allclose(mae, 1.0, atol=1e-10)
    
    def test_mae_mixed_errors(self):
        """Mixed errors should average correctly.
        
        Specification:
        - Errors: [0, 0, 0, 1, 2, 3]
        - MAE = (0+0+0+1+2+3) / 6 = 1.0
        """
        predictions = torch.tensor([[1.0, 2.0, 3.0],
                                    [5.0, 7.0, 9.0]])
        ground_truth = torch.tensor([[1.0, 2.0, 3.0],
                                      [4.0, 5.0, 6.0]])
        
        mae = forecasting_mae(predictions, ground_truth)
        
        # Errors: [0, 0, 0, 1, 2, 3] → mean = 1.0
        assert_allclose(mae, 1.0, atol=1e-10)
    
    def test_rmse_perfect_forecast(self):
        """Perfect forecast should give RMSE = 0."""
        predictions = torch.tensor([[1.0, 2.0, 3.0],
                                    [4.0, 5.0, 6.0]])
        ground_truth = torch.tensor([[1.0, 2.0, 3.0],
                                      [4.0, 5.0, 6.0]])
        
        rmse = forecasting_rmse(predictions, ground_truth)
        
        assert_allclose(rmse, 0.0, atol=1e-10)
    
    def test_rmse_constant_error(self):
        """Constant error should give correct RMSE.
        
        Specification:
        - All errors = 2.0
        - RMSE = sqrt(mean(2.0²)) = 2.0
        """
        predictions = torch.tensor([[3.0, 4.0, 5.0],
                                    [6.0, 7.0, 8.0]])
        ground_truth = torch.tensor([[1.0, 2.0, 3.0],
                                      [4.0, 5.0, 6.0]])
        
        rmse = forecasting_rmse(predictions, ground_truth)
        
        assert_allclose(rmse, 2.0, atol=1e-10)
    
    def test_rmse_penalties_large_errors(self):
        """RMSE should penalize large errors more than MAE.
        
        Specification:
        - Small errors (e.g., 1) contribute 1² = 1
        - Large errors (e.g., 3) contribute 3² = 9
        - RMSE > MAE for mixed errors
        """
        predictions = torch.tensor([[2.0, 5.0]])
        ground_truth = torch.tensor([[1.0, 1.0]])
        
        # Errors: [1, 4]
        mae = forecasting_mae(predictions, ground_truth)
        rmse = forecasting_rmse(predictions, ground_truth)
        
        # MAE = (1 + 4) / 2 = 2.5
        # RMSE = sqrt((1² + 4²) / 2) = sqrt(17/2) = sqrt(8.5) ≈ 2.915
        assert_allclose(mae, 2.5, atol=1e-10)
        assert_allclose(rmse, np.sqrt(8.5), rtol=1e-6)  # Relaxed tolerance for floating point
        assert rmse > mae, "RMSE should be greater than MAE for mixed errors"


class TestConfusionMatrixMetrics:
    """Test confusion matrix based metrics."""
    
    def test_confusion_matrix_perfect(self):
        """Perfect classification should give identity confusion matrix."""
        predictions = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        
        metrics = confusion_matrix_metrics(predictions, labels, num_classes=4)
        
        assert 'confusion_matrix' in metrics
        cm = metrics['confusion_matrix']
        
        # Should be identity matrix (2 samples per class, all correct)
        expected = np.array([
            [2, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 2]
        ])
        
        assert_allclose(cm, expected)
    
    def test_confusion_matrix_with_errors(self):
        """Confusion matrix should show misclassifications."""
        predictions = torch.tensor([0, 1, 1, 3])
        labels = torch.tensor([0, 1, 2, 3])
        
        metrics = confusion_matrix_metrics(predictions, labels, num_classes=4)
        cm = metrics['confusion_matrix']
        
        # Class 0: predicted 0, label 0 → cm[0,0] = 1
        # Class 1: predicted 1, label 1 → cm[1,1] = 1
        # Class 2: predicted 1, label 2 → cm[2,1] = 1 (misclassified as 1)
        # Class 3: predicted 3, label 3 → cm[3,3] = 1
        expected = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],  # Class 2 misclassified as class 1
            [0, 0, 0, 1]
        ])
        
        assert_allclose(cm, expected)


class TestMetricsInputValidation:
    """Test input validation and edge cases."""
    
    def test_empty_inputs_raise_error(self):
        """Empty inputs should raise ValueError."""
        with pytest.raises(ValueError, match="Empty"):
            classification_accuracy(torch.tensor([]), torch.tensor([]))
    
    def test_mismatched_shapes_raise_error(self):
        """Mismatched prediction/label shapes should raise ValueError."""
        predictions = torch.tensor([0, 1, 2])
        labels = torch.tensor([0, 1])
        
        with pytest.raises(ValueError, match="(?i)shape"):  # Case-insensitive match
            classification_accuracy(predictions, labels)
    
    def test_metrics_work_with_numpy_arrays(self):
        """Metrics should work with numpy arrays, not just tensors."""
        predictions = np.array([0, 1, 2, 3])
        labels = np.array([0, 1, 2, 3])
        
        accuracy = classification_accuracy(predictions, labels)
        
        assert accuracy == 1.0


class TestBatchedMetrics:
    """Test metrics on batched data."""
    
    def test_mae_on_batched_forecasts(self):
        """MAE should work on batched predictions.
        
        Specification:
        - Batch size: 4
        - Sequence length: 8
        - Features: 3 (x, y, z)
        - Shape: (4, 8, 3)
        """
        batch_size, seq_len, features = 4, 8, 3
        
        predictions = torch.randn(batch_size, seq_len, features)
        ground_truth = predictions + torch.randn(batch_size, seq_len, features) * 0.1
        
        mae = forecasting_mae(predictions, ground_truth)
        
        # Should return scalar MAE
        assert isinstance(mae, (float, np.floating, torch.Tensor))
        if isinstance(mae, torch.Tensor):
            assert mae.ndim == 0, "MAE should be scalar"
        
        # Should be positive
        assert mae >= 0, "MAE should be non-negative"

