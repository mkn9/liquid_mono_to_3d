"""
Evaluation metrics for classification and forecasting tasks.

Provides standard metrics for comparing model performance across branches.

Following TDD: Implementation created after tests (GREEN phase).
"""

import numpy as np
import torch
from typing import Dict, Union


def _to_numpy(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert tensor or array to numpy array."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def classification_accuracy(
    predictions: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray]
) -> float:
    """Calculate classification accuracy.
    
    Args:
        predictions: Predicted class labels, shape (N,)
        labels: Ground truth labels, shape (N,)
    
    Returns:
        float: Accuracy in range [0, 1]
    
    Raises:
        ValueError: If inputs are empty or shapes don't match
    """
    predictions = _to_numpy(predictions)
    labels = _to_numpy(labels)
    
    if predictions.size == 0 or labels.size == 0:
        raise ValueError("Empty inputs not allowed")
    
    if predictions.shape != labels.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} vs labels {labels.shape}"
        )
    
    correct = (predictions == labels).sum()
    total = len(predictions)
    
    return float(correct / total)


def per_class_accuracy(
    predictions: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
    num_classes: int
) -> np.ndarray:
    """Calculate per-class accuracy.
    
    Args:
        predictions: Predicted class labels, shape (N,)
        labels: Ground truth labels, shape (N,)
        num_classes: Total number of classes
    
    Returns:
        np.ndarray: Per-class accuracy, shape (num_classes,)
            Returns 0.0 for classes with no samples
    """
    predictions = _to_numpy(predictions)
    labels = _to_numpy(labels)
    
    accuracies = np.zeros(num_classes)
    
    for class_id in range(num_classes):
        # Find samples belonging to this class
        class_mask = (labels == class_id)
        num_samples = class_mask.sum()
        
        if num_samples == 0:
            accuracies[class_id] = 0.0
            continue
        
        # Count correct predictions for this class
        class_predictions = predictions[class_mask]
        class_labels = labels[class_mask]
        correct = (class_predictions == class_labels).sum()
        
        accuracies[class_id] = correct / num_samples
    
    return accuracies


def forecasting_mae(
    predictions: Union[torch.Tensor, np.ndarray],
    ground_truth: Union[torch.Tensor, np.ndarray]
) -> float:
    """Calculate Mean Absolute Error for forecasting.
    
    Args:
        predictions: Predicted values, shape (N, T, D) or (N, D)
        ground_truth: Ground truth values, same shape as predictions
    
    Returns:
        float: MAE = mean(|predictions - ground_truth|)
    """
    predictions = _to_numpy(predictions)
    ground_truth = _to_numpy(ground_truth)
    
    absolute_errors = np.abs(predictions - ground_truth)
    mae = absolute_errors.mean()
    
    return float(mae)


def forecasting_rmse(
    predictions: Union[torch.Tensor, np.ndarray],
    ground_truth: Union[torch.Tensor, np.ndarray]
) -> float:
    """Calculate Root Mean Squared Error for forecasting.
    
    Args:
        predictions: Predicted values, shape (N, T, D) or (N, D)
        ground_truth: Ground truth values, same shape as predictions
    
    Returns:
        float: RMSE = sqrt(mean((predictions - ground_truth)²))
    """
    predictions = _to_numpy(predictions)
    ground_truth = _to_numpy(ground_truth)
    
    squared_errors = (predictions - ground_truth) ** 2
    mse = squared_errors.mean()
    rmse = np.sqrt(mse)
    
    return float(rmse)


def confusion_matrix_metrics(
    predictions: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
    num_classes: int
) -> Dict[str, np.ndarray]:
    """Calculate confusion matrix and related metrics.
    
    Args:
        predictions: Predicted class labels, shape (N,)
        labels: Ground truth labels, shape (N,)
        num_classes: Total number of classes
    
    Returns:
        Dict containing:
            - confusion_matrix: np.ndarray (num_classes, num_classes)
                  Element [i, j] = number of samples with true label i predicted as j
            - precision: np.ndarray (num_classes,) - per-class precision
            - recall: np.ndarray (num_classes,) - per-class recall
            - f1_score: np.ndarray (num_classes,) - per-class F1 score
    """
    predictions = _to_numpy(predictions)
    labels = _to_numpy(labels)
    
    # Build confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    for true_label, pred_label in zip(labels, predictions):
        cm[int(true_label), int(pred_label)] += 1
    
    # Calculate precision, recall, F1 for each class
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1_score = np.zeros(num_classes)
    
    for class_id in range(num_classes):
        # True Positives: diagonal element
        tp = cm[class_id, class_id]
        
        # False Positives: sum of column (predicted as this class) minus TP
        fp = cm[:, class_id].sum() - tp
        
        # False Negatives: sum of row (actually this class) minus TP
        fn = cm[class_id, :].sum() - tp
        
        # Precision = TP / (TP + FP)
        if tp + fp > 0:
            precision[class_id] = tp / (tp + fp)
        else:
            precision[class_id] = 0.0
        
        # Recall = TP / (TP + FN)
        if tp + fn > 0:
            recall[class_id] = tp / (tp + fn)
        else:
            recall[class_id] = 0.0
        
        # F1 = 2 * (precision * recall) / (precision + recall)
        if precision[class_id] + recall[class_id] > 0:
            f1_score[class_id] = 2 * (precision[class_id] * recall[class_id]) / (
                precision[class_id] + recall[class_id]
            )
        else:
            f1_score[class_id] = 0.0
    
    return {
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

