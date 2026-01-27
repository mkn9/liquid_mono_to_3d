"""
Compute Gating Mechanism

Allocates compute budget based on persistence confidence.
Enables early stopping and efficient resource allocation.
"""

from typing import Dict, Optional


class ComputeGate:
    """
    Gating mechanism for compute allocation.
    
    Decides whether to continue processing based on:
    - Confidence threshold
    - Current frame number
    - Predicted class
    - Compute budget constraints
    """
    
    def __init__(self, confidence_threshold: float = 0.9,
                 early_stop_frame: int = 4,
                 compute_budget: Optional[Dict[str, float]] = None):
        """
        Initialize compute gate.
        
        Args:
            confidence_threshold: Minimum confidence for early stopping
            early_stop_frame: Maximum frame for early decisions
            compute_budget: Compute budget per class {'persistent': 1.0, 'transient': 0.2}
        """
        self.confidence_threshold = confidence_threshold
        self.early_stop_frame = early_stop_frame
        
        # Default compute budgets
        if compute_budget is None:
            compute_budget = {
                'persistent': 1.0,   # Full processing
                'transient': 0.2,    # Minimal processing (20%)
                'uncertain': 0.6     # Medium processing
            }
        self.compute_budget = compute_budget
    
    def get_compute_budget_for_class(self, predicted_class: str) -> float:
        """Get compute budget for a predicted class."""
        return self.compute_budget.get(predicted_class, 0.5)


def should_continue_processing(confidence: float, predicted_class: str,
                               current_frame: int, gate: ComputeGate) -> bool:
    """
    Decide if processing should continue.
    
    Args:
        confidence: Current prediction confidence
        predicted_class: Predicted class ('persistent' or 'transient')
        current_frame: Current frame index
        gate: ComputeGate instance
        
    Returns:
        True if should continue, False if should stop
    """
    # If we're past early stop frame, always continue
    if current_frame > gate.early_stop_frame:
        return True
    
    # If confidence is low, continue processing
    if confidence < gate.confidence_threshold:
        return True
    
    # If confident about transient, stop early (save compute)
    if predicted_class == 'transient' and confidence >= gate.confidence_threshold:
        return False
    
    # If confident about persistent, continue processing (needs full analysis)
    if predicted_class == 'persistent' and confidence >= gate.confidence_threshold:
        return True
    
    # Default: continue
    return True


def allocate_compute_budget(predicted_class: str, gate: ComputeGate) -> float:
    """
    Allocate compute budget based on predicted class.
    
    Args:
        predicted_class: Predicted class
        gate: ComputeGate instance
        
    Returns:
        Compute budget (0.0 to 1.0)
    """
    return gate.get_compute_budget_for_class(predicted_class)


def get_gating_decision(confidence: float, predicted_class: str,
                       current_frame: int, gate: ComputeGate) -> Dict:
    """
    Get complete gating decision with metadata.
    
    Args:
        confidence: Prediction confidence
        predicted_class: Predicted class
        current_frame: Current frame
        gate: ComputeGate instance
        
    Returns:
        Dictionary with gating decision and metadata
    """
    should_continue = should_continue_processing(
        confidence, predicted_class, current_frame, gate
    )
    
    compute_budget = allocate_compute_budget(predicted_class, gate)
    
    decision_info = {
        'should_continue': should_continue,
        'compute_budget': compute_budget,
        'decision_frame': current_frame,
        'confidence': confidence,
        'predicted_class': predicted_class,
        'reason': _get_decision_reason(
            should_continue, confidence, predicted_class, 
            current_frame, gate
        )
    }
    
    return decision_info


def _get_decision_reason(should_continue: bool, confidence: float,
                        predicted_class: str, current_frame: int,
                        gate: ComputeGate) -> str:
    """Get human-readable reason for gating decision."""
    if not should_continue:
        if confidence >= gate.confidence_threshold and predicted_class == 'transient':
            return f"Early stop: Confident transient detection at frame {current_frame}"
        else:
            return "Early stop: Unknown reason"
    else:
        if confidence < gate.confidence_threshold:
            return f"Continue: Low confidence ({confidence:.2f} < {gate.confidence_threshold})"
        elif predicted_class == 'persistent':
            return "Continue: Persistent track needs full analysis"
        elif current_frame > gate.early_stop_frame:
            return f"Continue: Past early stop frame ({current_frame} > {gate.early_stop_frame})"
        else:
            return "Continue: Default behavior"
