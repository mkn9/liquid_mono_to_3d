"""
TDD Tests for Full Embeddings Testing (No Compression).

Purpose: Test if compression (4096-dim → 5 statistics) is causing information loss.

Pipeline tested: MagVIT → Liquid → Full 4096-dim embeddings → GPT-4 
(NOT just mean/std/min/max/L2norm)

Expected: If full embeddings > statistics, then compression is the bottleneck.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFullEmbeddingsEvaluation:
    """Test evaluation with full 4096-dim embeddings (no compression)."""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_full_embeddings_module_exists(self):
        """Test that full_embeddings_test module exists."""
        import full_embeddings_test
        assert full_embeddings_test is not None
    
    def test_encode_full_embeddings_function_exists(self):
        """Test that full embedding encoding function exists."""
        from full_embeddings_test import encode_full_embeddings_for_llm
        assert encode_full_embeddings_for_llm is not None
    
    def test_full_embeddings_not_compressed_to_5_stats(self, device):
        """Critical test: Verify we're NOT using just 5 statistics."""
        from full_embeddings_test import encode_full_embeddings_for_llm
        
        embedding = torch.randn(1, 4096, device=device)
        
        encoded = encode_full_embeddings_for_llm(embedding, strategy='histogram')
        
        # Should be richer than just 5 numbers
        # Histogram should have multiple bins
        assert 'histogram' in encoded
        assert len(encoded['histogram']) >= 10, "Histogram should have at least 10 bins"
    
    def test_full_embeddings_preserve_more_info_than_stats(self, device):
        """Test that full embedding encoding preserves more info than stats."""
        from full_embeddings_test import (
            encode_full_embeddings_for_llm,
            compute_information_preservation
        )
        from true_e2e_visual_evaluation import extract_embedding_statistics
        
        embedding = torch.randn(1, 4096, device=device)
        
        # Get both encodings
        stats_encoding = extract_embedding_statistics(embedding)
        full_encoding = encode_full_embeddings_for_llm(embedding)
        
        # Measure information preservation
        info_stats = compute_information_preservation(embedding, stats_encoding)
        info_full = compute_information_preservation(embedding, full_encoding)
        
        # Full encoding should preserve more information
        assert info_full > info_stats
    
    def test_create_prompt_with_full_embeddings(self, device):
        """Test prompt creation with full embeddings."""
        from full_embeddings_test import create_prompt_with_full_embeddings
        
        embedding = torch.randn(1, 4096, device=device)
        
        prompt = create_prompt_with_full_embeddings(embedding)
        
        # Should mention full embeddings, not just statistics
        assert 'embedding' in prompt.lower()
        assert len(prompt) > 100  # Should be detailed
    
    def test_evaluate_with_full_embeddings(self, device):
        """Test complete evaluation pipeline with full embeddings."""
        from full_embeddings_test import evaluate_with_full_embeddings
        
        embedding = torch.randn(1, 4096, device=device)
        ground_truth = {
            'type': 'straight line',
            'description': 'A straight line trajectory'
        }
        
        result = evaluate_with_full_embeddings(embedding, ground_truth)
        
        assert 'description' in result
        assert 'accuracy' in result
        assert 'compression_used' in result
        assert result['compression_used'] == False  # Using full embeddings
    
    def test_run_compression_ablation(self, device):
        """Test ablation: compressed stats vs full embeddings."""
        from full_embeddings_test import run_compression_ablation_study
        
        # Create 3 mock samples
        samples = []
        for i in range(3):
            samples.append({
                'embedding': torch.randn(1, 4096, device=device),
                'ground_truth': {
                    'type': 'straight line',
                    'description': f'Sample {i}'
                }
            })
        
        results = run_compression_ablation_study(samples)
        
        assert 'stats_only_accuracy' in results
        assert 'full_embeddings_accuracy' in results
        assert 'improvement' in results
        assert 0 <= results['stats_only_accuracy'] <= 100
        assert 0 <= results['full_embeddings_accuracy'] <= 100


class TestEmbeddingEncodingStrategies:
    """Test different strategies for encoding full embeddings."""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_histogram_encoding_exists(self, device):
        """Test histogram-based encoding."""
        from full_embeddings_test import encode_as_histogram
        
        embedding = torch.randn(1, 4096, device=device)
        histogram = encode_as_histogram(embedding, num_bins=20)
        
        assert len(histogram) == 20
        assert all(count >= 0 for count in histogram.values())
    
    def test_pca_encoding_exists(self, device):
        """Test PCA-based encoding."""
        from full_embeddings_test import encode_with_pca
        
        embedding = torch.randn(1, 4096, device=device)
        pca_components = encode_with_pca(embedding, num_components=50)
        
        assert len(pca_components) == 50
    
    def test_quantile_encoding_exists(self, device):
        """Test quantile-based encoding."""
        from full_embeddings_test import encode_as_quantiles
        
        embedding = torch.randn(1, 4096, device=device)
        quantiles = encode_as_quantiles(embedding, num_quantiles=10)
        
        # Returns num_quantiles + 1 values (including both 0% and 100%)
        assert len(quantiles) == 11
        # Should be sorted
        assert all(quantiles[i] <= quantiles[i+1] for i in range(len(quantiles)-1))


class TestComparisonWithPreviousResults:
    """Compare full embeddings results with previous compression results."""
    
    def test_interpret_compression_impact(self):
        """Test interpretation of compression impact."""
        from full_embeddings_test import interpret_compression_results
        
        # Scenario 1: Full > Stats (compression is bottleneck)
        result1 = interpret_compression_results(
            stats_accuracy=52.5,
            full_accuracy=65.0
        )
        assert 'compression' in result1['bottleneck'].lower()
        
        # Scenario 2: Full = Stats (compression not the issue)
        result2 = interpret_compression_results(
            stats_accuracy=52.5,
            full_accuracy=53.0
        )
        assert 'llm' in result2['bottleneck'].lower() or 'decoding' in result2['bottleneck'].lower()

