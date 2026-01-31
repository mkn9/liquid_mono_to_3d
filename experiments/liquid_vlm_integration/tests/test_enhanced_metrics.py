"""
TDD Tests for Enhanced Evaluation Metrics
RED Phase: Tests written FIRST, will fail until implementation
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestEnhancedMetrics:
    """Test suite for enhanced evaluation metrics (BLEU, ROUGE, semantic similarity)."""
    
    def test_calculate_bleu_score_function_exists(self):
        """Test that calculate_bleu_score function exists."""
        from enhanced_metrics import calculate_bleu_score
        assert callable(calculate_bleu_score)
    
    def test_calculate_bleu_score_returns_float(self):
        """Test that BLEU score returns a float between 0 and 1."""
        from enhanced_metrics import calculate_bleu_score
        
        reference = "A straight line moving in the X direction"
        candidate = "A straight trajectory moving along X axis"
        
        score = calculate_bleu_score(reference, candidate)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    def test_calculate_rouge_l_function_exists(self):
        """Test that calculate_rouge_l function exists."""
        from enhanced_metrics import calculate_rouge_l
        assert callable(calculate_rouge_l)
    
    def test_calculate_rouge_l_returns_float(self):
        """Test that ROUGE-L score returns a float between 0 and 1."""
        from enhanced_metrics import calculate_rouge_l
        
        reference = "A curved trajectory in the XY plane"
        candidate = "The path is curved and lies in the XY plane"
        
        score = calculate_rouge_l(reference, candidate)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    def test_calculate_semantic_similarity_function_exists(self):
        """Test that calculate_semantic_similarity function exists."""
        from enhanced_metrics import calculate_semantic_similarity
        assert callable(calculate_semantic_similarity)
    
    def test_calculate_semantic_similarity_returns_float(self):
        """Test that semantic similarity returns a float between -1 and 1."""
        from enhanced_metrics import calculate_semantic_similarity
        
        text1 = "A circular motion in 3D space"
        text2 = "The object moves in a circle through 3D"
        
        score = calculate_semantic_similarity(text1, text2)
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0
    
    def test_semantic_similarity_high_for_similar_text(self):
        """Test that semantic similarity is high for similar descriptions."""
        from enhanced_metrics import calculate_semantic_similarity
        
        text1 = "A straight line trajectory"
        text2 = "A linear path"
        
        score = calculate_semantic_similarity(text1, text2)
        assert score > 0.2  # Should be reasonably similar (bag-of-words is simple)
    
    def test_evaluate_all_metrics_function_exists(self):
        """Test that evaluate_all_metrics function exists."""
        from enhanced_metrics import evaluate_all_metrics
        assert callable(evaluate_all_metrics)
    
    def test_evaluate_all_metrics_returns_dict(self):
        """Test that evaluate_all_metrics returns a dictionary with all metrics."""
        from enhanced_metrics import evaluate_all_metrics
        
        reference = "A straight line from (0,0,0) to (1,1,1)"
        candidate = "Linear trajectory starting at origin ending at (1,1,1)"
        
        metrics = evaluate_all_metrics(reference, candidate)
        
        assert isinstance(metrics, dict)
        assert "bleu" in metrics
        assert "rouge_l" in metrics
        assert "semantic_similarity" in metrics
    
    def test_perfect_match_scores(self):
        """Test that perfect matches get high scores."""
        from enhanced_metrics import evaluate_all_metrics
        
        text = "A circular trajectory in the XY plane"
        metrics = evaluate_all_metrics(text, text)
        
        # Perfect match should have very high scores
        assert metrics["bleu"] > 0.8
        assert metrics["rouge_l"] > 0.8
        assert metrics["semantic_similarity"] > 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

