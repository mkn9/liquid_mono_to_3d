"""
LLM Prompter for Trajectory Analysis

Generates prompts for different tasks:
- Description: Describe trajectory in natural language
- Explanation: Explain classification decision
- Question Answering: Answer questions about trajectory

Following TDD per requirements.md Section 3.4
"""

import torch
import numpy as np
from typing import Dict, Any, Optional

# Constants
CLASS_NAMES = {0: 'Transient', 1: 'Persistent'}
FEATURE_PRECISION = 4  # Decimal places for feature statistics
CONFIDENCE_PRECISION = 3  # Decimal places for confidence values


class LLMPrompter:
    """
    Generate prompts for LLM from visual features and predictions.
    
    Follows specification-by-example approach (requirements.md Section 3.4.2):
    - Prompts guide LLM without enabling gaming
    - Multiple varied examples prevent hardcoding
    - Includes context without exact answers
    """
    
    def __init__(self):
        """Initialize prompter."""
        pass
    
    def generate_description_prompt(
        self,
        features: Optional[torch.Tensor] = None,
        predictions: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        include_feature_summary: bool = True
    ) -> str:
        """
        Generate prompt for trajectory description task.
        
        Args:
            features: Visual features, shape (T, D) or (B, T, D)
            predictions: Model predictions (optional)
            metadata: Additional context (optional)
            include_feature_summary: Include feature statistics in prompt
        
        Returns:
            prompt: Formatted prompt string
        
        Specification:
            - Includes feature summary statistics if requested
            - Includes prediction if available
            - Includes metadata if available
            - Requests natural language description
        """
        prompt_parts = []
        
        prompt_parts.append("# Trajectory Analysis Task: Description\n")
        prompt_parts.append("Generate a natural language description of this trajectory.\n\n")
        
        # Add feature summary if requested and available
        if include_feature_summary and features is not None:
            feature_summary = self._summarize_features(features)
            prompt_parts.append("## Visual Features\n")
            prompt_parts.append(feature_summary + "\n\n")
        
        # Add predictions if available
        if predictions:
            prompt_parts.append("## Model Prediction\n")
            prompt_parts.append(self._format_predictions(predictions) + "\n\n")
        
        # Add metadata if available
        if metadata:
            prompt_parts.append("## Scene Metadata\n")
            prompt_parts.append(self._format_metadata(metadata) + "\n\n")
        
        prompt_parts.append("## Instructions\n")
        prompt_parts.append(
            "Provide a concise description of the object's motion and behavior. "
            "Include information about persistence, motion patterns, and any notable characteristics."
        )
        
        return "".join(prompt_parts)
    
    def generate_explanation_prompt(
        self,
        features: Optional[torch.Tensor] = None,
        predictions: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        include_feature_summary: bool = True
    ) -> str:
        """
        Generate prompt for classification explanation task.
        
        Args:
            features: Visual features
            predictions: Model predictions
            metadata: Additional context
            include_feature_summary: Include feature statistics in prompt
        
        Returns:
            prompt: Formatted prompt string
        
        Specification:
            - Focuses on explaining WHY model made prediction
            - Includes quantitative evidence
            - Cites specific features/patterns
        """
        prompt_parts = []
        
        prompt_parts.append("# Trajectory Analysis Task: Explain Classification\n")
        prompt_parts.append("Explain why the model made this persistence classification.\n\n")
        
        # Add predictions (required for explanation)
        if predictions:
            prompt_parts.append("## Model Prediction\n")
            prompt_parts.append(self._format_predictions(predictions) + "\n\n")
        
        # Add feature summary if requested and available
        if include_feature_summary and features is not None:
            feature_summary = self._summarize_features(features)
            prompt_parts.append("## Visual Features\n")
            prompt_parts.append(feature_summary + "\n\n")
        
        # Add metadata if available
        if metadata:
            prompt_parts.append("## Scene Metadata\n")
            prompt_parts.append(self._format_metadata(metadata) + "\n\n")
        
        prompt_parts.append("## Instructions\n")
        prompt_parts.append(
            "Explain the model's classification decision. "
            "Cite quantitative evidence from the features and predictions. "
            "Discuss what patterns in the visual features support this classification."
        )
        
        return "".join(prompt_parts)
    
    def generate_qa_prompt(
        self,
        features: Optional[torch.Tensor] = None,
        predictions: Optional[Dict[str, Any]] = None,
        question: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        include_feature_summary: bool = True
    ) -> str:
        """
        Generate prompt for question answering task.
        
        Args:
            features: Visual features
            predictions: Model predictions
            question: User's question
            metadata: Additional context
            include_feature_summary: Include feature statistics in prompt
        
        Returns:
            prompt: Formatted prompt string
        
        Specification:
            - Includes user's question
            - Provides context for answering
            - Requests evidence-based answer
        """
        prompt_parts = []
        
        prompt_parts.append("# Trajectory Analysis Task: Question Answering\n")
        prompt_parts.append(f"Question: {question}\n\n")
        
        # Add feature summary if requested and available
        if include_feature_summary and features is not None:
            feature_summary = self._summarize_features(features)
            prompt_parts.append("## Visual Features\n")
            prompt_parts.append(feature_summary + "\n\n")
        
        # Add predictions if available
        if predictions:
            prompt_parts.append("## Model Prediction\n")
            prompt_parts.append(self._format_predictions(predictions) + "\n\n")
        
        # Add metadata if available
        if metadata:
            prompt_parts.append("## Scene Metadata\n")
            prompt_parts.append(self._format_metadata(metadata) + "\n\n")
        
        prompt_parts.append("## Instructions\n")
        prompt_parts.append(
            "Answer the question based on the video features and predictions. "
            "Provide specific evidence to support your answer."
        )
        
        return "".join(prompt_parts)
    
    def _summarize_features(self, features: torch.Tensor) -> str:
        """
        Generate summary statistics for features.
        
        Args:
            features: Feature tensor, shape (T, D) or (B, T, D)
        
        Returns:
            summary: Formatted summary string
        """
        # Handle batch dimension
        if features.ndim == 3:
            features = features[0]  # Use first item in batch
        
        # Calculate statistics
        mean = features.mean().item()
        std = features.std().item()
        min_val = features.min().item()
        max_val = features.max().item()
        
        # Temporal statistics (variation across time)
        temporal_mean = features.mean(dim=1)  # Mean across feature dimension
        temporal_variation = temporal_mean.std().item()
        
        summary = (
            f"- Frames: {features.shape[0]}\n"
            f"- Feature dimension: {features.shape[1]}\n"
            f"- Feature statistics:\n"
            f"  - Mean: {mean:.4f}\n"
            f"  - Std: {std:.4f}\n"
            f"  - Range: [{min_val:.4f}, {max_val:.4f}]\n"
            f"- Temporal variation: {temporal_variation:.4f}"
        )
        
        return summary
    
    def _format_predictions(self, predictions: Dict[str, Any]) -> str:
        """
        Format predictions dictionary as human-readable string.
        
        Args:
            predictions: Dictionary containing class, confidence, and optionally logits
        
        Returns:
            Formatted string with prediction information
        """
        pred_class = predictions.get('class', -1)
        confidence = predictions.get('confidence', 0.0)
        
        # Convert tensor to scalar if needed
        if isinstance(confidence, torch.Tensor):
            confidence = confidence.item()
        
        # Format main prediction info
        formatted = (
            f"- Predicted class: {CLASS_NAMES.get(pred_class, 'Unknown')}\n"
            f"- Confidence: {confidence:.{CONFIDENCE_PRECISION}f} ({confidence*100:.1f}%)"
        )
        
        # Add logits if available
        if 'logits' in predictions:
            logits = predictions['logits']
            if isinstance(logits, torch.Tensor):
                logits_str = ", ".join([
                    f"{l:.{CONFIDENCE_PRECISION}f}" 
                    for l in logits.flatten()[:2]
                ])
                formatted += f"\n- Logits: [{logits_str}]"
        
        return formatted
    
    def _format_metadata(self, metadata: Dict[str, Any]) -> str:
        """Format metadata dictionary as string."""
        formatted_lines = []
        
        for key, value in metadata.items():
            formatted_lines.append(f"- {key}: {value}")
        
        return "\n".join(formatted_lines) if formatted_lines else "No additional metadata"

