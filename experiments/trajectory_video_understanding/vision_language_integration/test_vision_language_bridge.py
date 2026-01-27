"""
Test Vision-Language Bridge for Trajectory Analysis

Following TDD per requirements.md Section 3.4:
- Write tests FIRST (RED phase)
- All tests use explicit seeds for determinism
- All numeric comparisons use explicit tolerances

Test coverage:
1. Feature extraction from trained Worker 2 model
2. LLM prompt generation from visual features
3. Description generation
4. Explanation generation
5. Question answering
6. Error handling and edge cases
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import json
import sys

# Module to test (will fail initially - expected for RED phase)
try:
    from vision_language_bridge import VisionLanguageBridge, extract_visual_features, generate_prompt
    from llm_prompter import LLMPrompter
    from trajectory_qa import TrajectoryQA
except ImportError:
    # Expected to fail in RED phase
    pass


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_video():
    """Create mock video tensor for testing."""
    # Deterministic video: 16 frames, 64x64, 3 channels
    rng = np.random.default_rng(42)
    video = torch.from_numpy(rng.standard_normal((16, 64, 64, 3))).float()
    return video


@pytest.fixture
def mock_worker2_model():
    """Create mock Worker 2 model structure."""
    # This would be replaced with actual model loading in implementation
    class MockWorker2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.resnet_features = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 7, 2, 3),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((1, 1))
            )
            self.transformer = torch.nn.TransformerEncoder(
                torch.nn.TransformerEncoderLayer(d_model=512, nhead=8),
                num_layers=2
            )
            self.classifier = torch.nn.Linear(512, 2)
        
        def forward(self, x):
            # x: (B, T, H, W, C)
            B, T, H, W, C = x.shape
            x = x.permute(0, 1, 4, 2, 3)  # (B, T, C, H, W)
            
            # Extract features per frame
            features_list = []
            for t in range(T):
                feat = self.resnet_features(x[:, t])  # (B, 64, 1, 1)
                features_list.append(feat.flatten(1))
            
            features = torch.stack(features_list, dim=1)  # (B, T, 64)
            
            # Pad to 512 dim (mock)
            features = torch.nn.functional.pad(features, (0, 512 - 64))
            
            # Transformer
            transformed = self.transformer(features.permute(1, 0, 2)).permute(1, 0, 2)
            
            # Classification
            pooled = transformed.mean(dim=1)
            logits = self.classifier(pooled)
            
            return {
                'logits': logits,
                'features': features,
                'attention': None  # Placeholder
            }
    
    model = MockWorker2()
    model.eval()
    return model


@pytest.fixture
def sample_predictions():
    """Sample classification predictions."""
    return {
        'class': 1,  # High persistence
        'confidence': 0.987,
        'logits': torch.tensor([[0.1, 3.8]]),
        'features': torch.randn(1, 16, 512)
    }


# ============================================================================
# TEST CLASS 1: Feature Extraction
# ============================================================================

class TestFeatureExtraction:
    """Test extraction of visual features from trained Worker 2 model."""
    
    def test_extract_features_from_video(self, mock_worker2_model, mock_video):
        """Test extracting features from single video.
        
        Specification:
        - Input: Video tensor (16, 64, 64, 3)
        - Output: Features tensor (16, 512)
        - Units: Normalized features [-inf, inf]
        - Deterministic: Yes (model in eval mode)
        - Tolerance: Exact (deterministic)
        """
        features = extract_visual_features(mock_worker2_model, mock_video)
        
        assert features.shape == (16, 512), \
            f"Expected features shape (16, 512), got {features.shape}"
        assert torch.all(torch.isfinite(features)), \
            "Features contain NaN or Inf"
        assert features.dtype == torch.float32, \
            f"Expected float32, got {features.dtype}"
    
    def test_extract_features_batch(self, mock_worker2_model):
        """Test batch feature extraction.
        
        Specification:
        - Input: Batch of videos (B, 16, 64, 64, 3)
        - Output: Batch of features (B, 16, 512)
        - Batch sizes: B ∈ [1, 32]
        """
        for batch_size in [1, 4, 16, 32]:
            rng = np.random.default_rng(42)
            videos = torch.from_numpy(
                rng.standard_normal((batch_size, 16, 64, 64, 3))
            ).float()
            
            features = extract_visual_features(mock_worker2_model, videos)
            
            assert features.shape == (batch_size, 16, 512), \
                f"Batch size {batch_size}: expected ({batch_size}, 16, 512), got {features.shape}"
    
    def test_features_are_deterministic(self, mock_worker2_model, mock_video):
        """Test that feature extraction is deterministic.
        
        Specification:
        - Same input → same output (eval mode, no dropout)
        - Tolerance: rtol=1e-10, atol=1e-12 (numerical precision)
        """
        with torch.no_grad():
            features1 = extract_visual_features(mock_worker2_model, mock_video)
            features2 = extract_visual_features(mock_worker2_model, mock_video)
        
        torch.testing.assert_close(
            features1, features2,
            rtol=1e-10, atol=1e-12,
            msg="Feature extraction is not deterministic"
        )


# ============================================================================
# TEST CLASS 2: LLM Prompt Generation
# ============================================================================

class TestLLMPromptGeneration:
    """Test generation of LLM prompts from visual features."""
    
    def test_generate_description_prompt(self, sample_predictions):
        """Test generating prompt for video description.
        
        Specification:
        - Input: predictions dict with class, confidence, features
        - Output: String prompt for LLM
        - Must include: classification result, confidence
        """
        prompt = generate_prompt(
            task='description',
            predictions=sample_predictions
        )
        
        assert isinstance(prompt, str), "Prompt must be string"
        assert len(prompt) > 0, "Prompt cannot be empty"
        assert 'persistence' in prompt.lower() or 'transient' in prompt.lower(), \
            "Prompt must mention persistence/transient"
        assert '98' in prompt or '0.98' in prompt, \
            "Prompt must include confidence value"
    
    def test_generate_explanation_prompt(self, sample_predictions):
        """Test generating prompt for classification explanation.
        
        Specification:
        - Task: 'explanation'
        - Must request reasoning for classification
        - Should include frame-level information if available
        """
        prompt = generate_prompt(
            task='explanation',
            predictions=sample_predictions
        )
        
        assert isinstance(prompt, str), "Prompt must be string"
        assert 'explain' in prompt.lower() or 'why' in prompt.lower(), \
            "Explanation prompt must request reasoning"
        assert 'classified' in prompt.lower() or 'prediction' in prompt.lower(), \
            "Must mention classification"
    
    def test_generate_qa_prompt(self, sample_predictions):
        """Test generating prompt for question answering.
        
        Specification:
        - Task: 'qa'
        - Must include the user's question
        - Should provide video context
        """
        question = "How many objects are visible in this video?"
        
        prompt = generate_prompt(
            task='qa',
            predictions=sample_predictions,
            question=question
        )
        
        assert isinstance(prompt, str), "Prompt must be string"
        assert question in prompt, "Prompt must include user question"
        assert 'video' in prompt.lower(), "Prompt should mention video context"
    
    def test_prompt_includes_features_summary(self, sample_predictions):
        """Test that prompt includes summary of visual features.
        
        Specification:
        - Features should be summarized (not full 512-dim vector)
        - Include statistics: mean, std, temporal pattern
        """
        prompt = generate_prompt(
            task='description',
            predictions=sample_predictions,
            include_feature_summary=True
        )
        
        # Should have some numerical information
        assert any(char.isdigit() for char in prompt), \
            "Feature summary should include numbers"


# ============================================================================
# TEST CLASS 3: Vision-Language Bridge Integration
# ============================================================================

class TestVisionLanguageBridge:
    """Test complete vision-language bridge functionality."""
    
    def test_bridge_initialization(self):
        """Test initializing VisionLanguageBridge.
        
        Specification:
        - Must specify vision_model_path
        - Must specify llm_provider
        - Optional: api_key for cloud LLMs
        """
        # This will fail in RED phase - expected!
        bridge = VisionLanguageBridge(
            vision_model_path='mock_model.pt',
            llm_provider='gpt4',
            api_key='test_key'
        )
        
        assert bridge.vision_model is not None, \
            "Vision model should be loaded"
        assert bridge.llm is not None, \
            "LLM interface should be initialized"
    
    def test_describe_video(self, mock_video):
        """Test generating natural language description of video.
        
        Specification:
        - Input: Video tensor (16, 64, 64, 3)
        - Output: String description
        - Should mention: objects, persistence/transience, behavior
        """
        bridge = VisionLanguageBridge(
            vision_model_path='mock_model.pt',
            llm_provider='gpt4',
            api_key='test_key'
        )
        
        description = bridge.describe_video(mock_video)
        
        assert isinstance(description, str), "Description must be string"
        assert len(description) > 20, \
            "Description should be substantive (>20 chars)"
        assert description[0].isupper(), \
            "Description should start with capital letter"
    
    def test_explain_classification(self, mock_video):
        """Test generating explanation for classification decision.
        
        Specification:
        - Input: Video tensor
        - Output: String explanation with reasoning
        - Must cite: frames, percentages, thresholds
        """
        bridge = VisionLanguageBridge(
            vision_model_path='mock_model.pt',
            llm_provider='gpt4',
            api_key='test_key'
        )
        
        explanation = bridge.explain_classification(mock_video)
        
        assert isinstance(explanation, str), "Explanation must be string"
        assert len(explanation) > 30, \
            "Explanation should be detailed"
        # Should contain quantitative information
        assert any(char.isdigit() for char in explanation), \
            "Explanation should cite numbers (frames, %)"
    
    def test_answer_question(self, mock_video):
        """Test answering questions about video.
        
        Specification:
        - Input: Video + question string
        - Output: Answer string
        - Answer should be relevant to question
        """
        bridge = VisionLanguageBridge(
            vision_model_path='mock_model.pt',
            llm_provider='gpt4',
            api_key='test_key'
        )
        
        question = "How many frames show transient objects?"
        answer = bridge.answer_question(mock_video, question)
        
        assert isinstance(answer, str), "Answer must be string"
        assert len(answer) > 0, "Answer cannot be empty"
    
    def test_multiple_llm_backends(self, mock_video):
        """Test that different LLM backends work correctly.
        
        Specification:
        - Support: gpt4, mistral, phi2
        - All should produce valid outputs
        - May have different quality but same interface
        """
        for llm_provider in ['gpt4', 'mistral', 'phi2']:
            bridge = VisionLanguageBridge(
                vision_model_path='mock_model.pt',
                llm_provider=llm_provider,
                api_key='test_key' if llm_provider == 'gpt4' else None
            )
            
            description = bridge.describe_video(mock_video)
            
            assert isinstance(description, str), \
                f"{llm_provider}: Description must be string"
            assert len(description) > 0, \
                f"{llm_provider}: Description cannot be empty"


# ============================================================================
# TEST CLASS 4: Error Handling and Edge Cases
# ============================================================================

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_video_shape_raises_error(self):
        """Test that invalid video shapes raise clear errors.
        
        Specification:
        - Valid shape: (T, H, W, C) or (B, T, H, W, C)
        - Invalid shapes should raise ValueError with message
        """
        bridge = VisionLanguageBridge(
            vision_model_path='mock_model.pt',
            llm_provider='gpt4',
            api_key='test_key'
        )
        
        # Wrong number of dimensions
        invalid_video = torch.randn(64, 64, 3)  # Missing time dimension
        
        with pytest.raises(ValueError, match="video.*shape"):
            bridge.describe_video(invalid_video)
    
    def test_missing_api_key_for_gpt4_raises_error(self):
        """Test that GPT-4 without API key raises error.
        
        Specification:
        - GPT-4 requires api_key
        - Should raise ValueError if not provided
        """
        with pytest.raises(ValueError, match="api_key.*required"):
            bridge = VisionLanguageBridge(
                vision_model_path='mock_model.pt',
                llm_provider='gpt4'
                # api_key not provided
            )
    
    def test_nonexistent_model_path_raises_error(self):
        """Test that nonexistent model path raises error.
        
        Specification:
        - Should check if model file exists
        - Raise FileNotFoundError with clear message
        """
        with pytest.raises(FileNotFoundError, match="model.*not found"):
            bridge = VisionLanguageBridge(
                vision_model_path='nonexistent_model.pt',
                llm_provider='gpt4',
                api_key='test_key'
            )
    
    def test_empty_question_raises_error(self):
        """Test that empty question string raises error.
        
        Specification:
        - Question must be non-empty string
        - Raise ValueError if empty
        """
        bridge = VisionLanguageBridge(
            vision_model_path='mock_model.pt',
            llm_provider='gpt4',
            api_key='test_key'
        )
        
        video = torch.randn(16, 64, 64, 3)
        
        with pytest.raises(ValueError, match="question.*empty"):
            bridge.answer_question(video, "")


# ============================================================================
# TEST CLASS 5: Output Quality and Format
# ============================================================================

class TestOutputQuality:
    """Test quality and format of generated outputs."""
    
    def test_description_mentions_key_elements(self, mock_video):
        """Test that descriptions include essential elements.
        
        Specification:
        - Must mention: object count, persistence level, motion
        - Check for keywords: "persistent", "transient", "frames", "objects"
        """
        bridge = VisionLanguageBridge(
            vision_model_path='mock_model.pt',
            llm_provider='gpt4',
            api_key='test_key'
        )
        
        description = bridge.describe_video(mock_video)
        
        # Convert to lowercase for checking
        desc_lower = description.lower()
        
        # Should mention persistence concept
        assert any(word in desc_lower for word in ['persistent', 'transient', 'temporary']), \
            "Description should mention persistence/transience"
        
        # Should mention objects or frames
        assert any(word in desc_lower for word in ['object', 'sphere', 'frame']), \
            "Description should mention objects or frames"
    
    def test_explanation_cites_quantitative_evidence(self, mock_video):
        """Test that explanations cite numerical evidence.
        
        Specification:
        - Must include: frame counts, percentages, or thresholds
        - Format: "X out of Y frames" or "X%" or "threshold of X"
        """
        bridge = VisionLanguageBridge(
            vision_model_path='mock_model.pt',
            llm_provider='gpt4',
            api_key='test_key'
        )
        
        explanation = bridge.explain_classification(mock_video)
        
        # Should contain numbers
        assert any(char.isdigit() for char in explanation), \
            "Explanation must cite numerical evidence"
        
        # Should mention frames or percentage
        exp_lower = explanation.lower()
        assert any(word in exp_lower for word in ['frame', '%', 'percent', 'threshold']), \
            "Explanation should mention frames, percentages, or thresholds"
    
    def test_output_is_reproducible_with_same_input(self, mock_video):
        """Test that outputs are deterministic for same input.
        
        Specification:
        - Same video → same classification explanation
        - May vary for creative descriptions (temperature > 0)
        - Explanation (temperature=0) should be identical
        
        Note: For GPT-4, even with temperature=0, there may be slight variations
        due to API updates. This test focuses on the vision part being deterministic.
        """
        bridge = VisionLanguageBridge(
            vision_model_path='mock_model.pt',
            llm_provider='gpt4',
            api_key='test_key'
        )
        
        # Vision features should be deterministic
        with torch.no_grad():
            features1 = bridge._extract_features(mock_video)
            features2 = bridge._extract_features(mock_video)
        
        torch.testing.assert_close(
            features1, features2,
            rtol=1e-10, atol=1e-12,
            msg="Vision features should be deterministic"
        )


# ============================================================================
# DETERMINISTIC TEST DOCUMENTATION
# ============================================================================

"""
DETERMINISTIC TEST SUMMARY:

All tests in this file follow deterministic testing requirements per
requirements.md Section 3.4:

1. RNG Seeds:
   - np.random.default_rng(42) for all numpy random generation
   - torch.manual_seed(42) would be used if training (not applicable here)

2. Numeric Tolerances:
   - torch.testing.assert_close() with explicit rtol/atol
   - Never use == for float comparisons

3. Model Determinism:
   - All models in eval() mode
   - no_grad() context for inference
   - Frozen weights (Worker 2 model not retrained)

4. LLM Determinism:
   - temperature=0 for explanations (deterministic)
   - temperature=0.3-0.7 for descriptions (controlled randomness)
   - Documented in test docstrings
"""

