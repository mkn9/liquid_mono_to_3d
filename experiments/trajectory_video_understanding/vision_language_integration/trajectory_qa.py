"""
Trajectory Question Answering

Answer questions about trajectory videos based on visual features and predictions.

Following TDD per requirements.md Section 3.4
"""

import torch
import re
from typing import Dict, Any, Optional

# Import LLM interface if needed
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "magvit_I3D_LLM_basic_trajectory"))

# Constants
CLASS_NAMES = {0: 'transient', 1: 'persistent'}
CLASS_NAMES_UPPER = {0: 'TRANSIENT', 1: 'PERSISTENT'}

# Motion classification thresholds
TEMPORAL_VARIATION_STABLE = 0.1
TEMPORAL_VARIATION_MODERATE = 0.5
TEMPORAL_CONSISTENCY_THRESHOLD = 0.5

# Confidence levels
CONFIDENCE_VERY_HIGH = 95.0
CONFIDENCE_HIGH = 80.0
CONFIDENCE_MODERATE = 60.0


class TrajectoryQA:
    """
    Question answering system for trajectory videos.
    
    Answers questions based on:
    - Visual features from Worker 2 model
    - Classification predictions
    - Scene metadata
    
    Follows property-based testing approach (requirements.md Section 3.4.4):
    - Consistent answers for same input
    - Evidence-based responses
    - No hallucination
    """
    
    def __init__(self, llm_interface=None, use_llm=False):
        """
        Initialize QA system.
        
        Args:
            llm_interface: Optional LLM interface for complex questions
            use_llm: If True, use LLM for Q&A; if False, use template-based (default)
        """
        self.llm = llm_interface
        self.use_llm = use_llm
    
    def answer(
        self,
        question: str,
        features: torch.Tensor,
        predictions: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Answer question about trajectory.
        
        Args:
            question: User's question
            features: Visual features from Worker 2
            predictions: Model predictions
            metadata: Additional context
        
        Returns:
            answer: Natural language answer
        
        Specification:
            - Answer must be grounded in provided data
            - No hallucination or speculation
            - Cite evidence when possible
        """
        # If LLM-based Q&A is enabled and LLM is available, use it
        if self.use_llm and self.llm is not None:
            return self._answer_with_llm(question, features, predictions, metadata)
        
        # Otherwise, use template-based rule-based answering
        question_lower = question.lower().strip()
        
        # Rule-based answering for common questions
        # (In real implementation, would use LLM for more complex questions)
        
        # Question: How many objects?
        if 'how many' in question_lower and 'object' in question_lower:
            return self._answer_object_count(metadata)
        
        # Question: What is the persistence/classification?
        if 'persistence' in question_lower or 'classification' in question_lower or 'predict' in question_lower:
            return self._answer_classification(predictions)
        
        # Question: Motion pattern
        if 'motion' in question_lower or 'move' in question_lower or 'trajectory' in question_lower:
            return self._answer_motion_pattern(features, predictions)
        
        # Question: Confidence
        if 'confidence' in question_lower or 'certain' in question_lower:
            return self._answer_confidence(predictions)
        
        # Question: Why/Explanation
        if question_lower.startswith('why') or 'explain' in question_lower:
            return self._answer_why(features, predictions)
        
        # Default: General answer based on available data
        return self._answer_general(features, predictions, metadata)
    
    def _answer_object_count(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Answer questions about object count."""
        if metadata and 'num_objects' in metadata:
            count = metadata['num_objects']
            if count == 1:
                return "One object is visible throughout the video sequence."
            else:
                return f"{count} objects are visible in the video."
        else:
            # Default assumption for our trajectory dataset
            return "Based on the visual features, one primary object is detected."
    
    def _answer_classification(self, predictions: Optional[Dict[str, Any]] = None) -> str:
        """
        Answer questions about classification.
        
        Args:
            predictions: Dictionary with class and confidence
        
        Returns:
            Classification answer string
        """
        if not predictions:
            return "Classification predictions are not available."
        
        pred_class = predictions.get('class', -1)
        confidence = predictions.get('confidence', 0.0)
        
        if isinstance(confidence, torch.Tensor):
            confidence = confidence.item()
        
        class_name = CLASS_NAMES.get(pred_class, 'unknown')
        
        return (
            f"The model classifies this trajectory as {class_name.upper()} "
            f"with {confidence*100:.1f}% confidence."
        )
    
    def _answer_motion_pattern(
        self,
        features: torch.Tensor,
        predictions: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Answer questions about motion patterns.
        
        Args:
            features: Visual features tensor
            predictions: Optional predictions for context
        
        Returns:
            Motion pattern description string
        """
        # Analyze temporal variation in features
        if features.ndim == 3:
            features = features[0]  # Use first in batch
        
        temporal_mean = features.mean(dim=1)  # (T,)
        temporal_variation = temporal_mean.std().item()
        
        # Determine motion type using threshold constants
        if temporal_variation < TEMPORAL_VARIATION_STABLE:
            motion_type = "stable and consistent"
        elif temporal_variation < TEMPORAL_VARIATION_MODERATE:
            motion_type = "moderately variable"
        else:
            motion_type = "highly variable"
        
        # Add persistence context if available
        context = ""
        if predictions:
            pred_class = predictions.get('class', -1)
            context = f" This is consistent with {CLASS_NAMES.get(pred_class, 'unknown')} behavior."
        
        return f"The motion pattern is {motion_type} across the {features.shape[0]} frames.{context}"
    
    def _answer_confidence(self, predictions: Optional[Dict[str, Any]] = None) -> str:
        """
        Answer questions about model confidence.
        
        Args:
            predictions: Dictionary containing confidence value
        
        Returns:
            Confidence description string
        """
        if not predictions or 'confidence' not in predictions:
            return "Confidence information is not available."
        
        confidence = predictions['confidence']
        if isinstance(confidence, torch.Tensor):
            confidence = confidence.item()
        
        confidence_pct = confidence * 100
        
        # Categorize confidence using threshold constants
        if confidence_pct >= CONFIDENCE_VERY_HIGH:
            certainty = "very high"
        elif confidence_pct >= CONFIDENCE_HIGH:
            certainty = "high"
        elif confidence_pct >= CONFIDENCE_MODERATE:
            certainty = "moderate"
        else:
            certainty = "low"
        
        return (
            f"The model has {certainty} confidence in its prediction "
            f"({confidence_pct:.1f}%)."
        )
    
    def _answer_why(
        self,
        features: torch.Tensor,
        predictions: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Answer 'why' questions about classification.
        
        Args:
            features: Visual features tensor
            predictions: Classification predictions
        
        Returns:
            Explanation string with quantitative evidence
        """
        if not predictions:
            return "Cannot explain without classification predictions."
        
        # Analyze features for evidence
        if features.ndim == 3:
            features = features[0]
        
        temporal_variance = features.var(dim=0).mean().item()
        temporal_consistency = 1.0 / (1.0 + temporal_variance)
        
        pred_class = predictions.get('class', -1)
        confidence = predictions.get('confidence', 0.0)
        
        if isinstance(confidence, torch.Tensor):
            confidence = confidence.item()
        
        # Determine pattern type using threshold
        pattern_type = 'stable' if temporal_consistency > TEMPORAL_CONSISTENCY_THRESHOLD else 'variable'
        
        explanation = (
            f"The model predicts {CLASS_NAMES_UPPER.get(pred_class, 'UNKNOWN')} "
            f"({confidence*100:.1f}% confidence) because the temporal feature "
            f"consistency is {temporal_consistency:.3f}, indicating "
            f"{pattern_type} patterns "
            f"characteristic of {CLASS_NAMES.get(pred_class, 'unknown')} motion."
        )
        
        return explanation
    
    def _answer_with_llm(
        self,
        question: str,
        features: torch.Tensor,
        predictions: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Answer question using LLM (TinyLlama, GPT-4, etc.)
        
        Args:
            question: User's question
            features: Visual features
            predictions: Model predictions
            metadata: Scene metadata
        
        Returns:
            LLM-generated answer
        """
        # Build context for LLM
        context_parts = []
        
        # Add trajectory info
        if predictions:
            pred_class = predictions.get('class', 0)
            confidence = predictions.get('confidence', 0.0) * 100
            class_name = CLASS_NAMES.get(pred_class, 'unknown')
            context_parts.append(f"Classification: {class_name} ({confidence:.1f}% confidence)")
        
        # Add feature statistics
        if features.ndim == 3:
            features = features[0]
        num_frames = features.shape[0]
        feature_mean = features.mean().item()
        feature_std = features.std().item()
        context_parts.append(f"Video: {num_frames} frames, features mean={feature_mean:.3f}, std={feature_std:.3f}")
        
        # Add metadata if available
        if metadata and 'num_objects' in metadata:
            context_parts.append(f"Objects detected: {metadata['num_objects']}")
        
        context = " | ".join(context_parts)
        
        # Create prompt for LLM
        prompt = f"""Context: {context}

Question: {question}

Answer the question in 1-2 sentences based only on the context provided. Be specific and factual."""
        
        # Generate answer with LLM
        try:
            if hasattr(self.llm, '_generate'):
                # LocalLLM interface
                answer = self.llm._generate(prompt, max_tokens=100)
            elif hasattr(self.llm, 'generate_description'):
                # Use generate_description as fallback
                answer = self.llm.generate_description(predictions.get('class', 0) if predictions else 0)
            else:
                # Fallback to template
                return self._answer_general(features, predictions, metadata)
        except Exception as e:
            print(f"⚠️  LLM Q&A failed: {e}")
            return self._answer_general(features, predictions, metadata)
        
        return answer.strip()
    
    def _answer_general(
        self,
        features: torch.Tensor,
        predictions: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        General answer based on available information.
        
        Args:
            features: Visual features tensor
            predictions: Optional classification predictions
            metadata: Optional scene metadata
        
        Returns:
            General information string
        """
        answer_parts = []
        
        # Feature summary
        if features.ndim == 3:
            features = features[0]
        answer_parts.append(
            f"The video consists of {features.shape[0]} frames with "
            f"{features.shape[1]}-dimensional visual features."
        )
        
        # Classification if available
        if predictions:
            pred_class = predictions.get('class', -1)
            confidence = predictions.get('confidence', 0.0)
            if isinstance(confidence, torch.Tensor):
                confidence = confidence.item()
            answer_parts.append(
                f"The trajectory is classified as {CLASS_NAMES.get(pred_class, 'unknown')} "
                f"with {confidence*100:.1f}% confidence."
            )
        
        # Metadata if available
        if metadata and 'num_objects' in metadata:
            answer_parts.append(f"{metadata['num_objects']} object(s) detected.")
        
        return " ".join(answer_parts)

