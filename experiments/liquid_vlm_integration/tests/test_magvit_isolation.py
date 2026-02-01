"""
TDD Tests for MagVIT Isolation Testing.

Purpose: Test MagVIT alone (bypass Liquid fusion) to determine if visual 
features from MagVIT are meaningful or if the problem is downstream.

Expected: If MagVIT features > random, then Liquid/LLM is the problem.
         If MagVIT features = random, then MagVIT or video quality is the problem.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMagVITIsolation:
    """Test MagVIT feature extraction in isolation."""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_magvit_isolation_module_exists(self):
        """Test that magvit_isolation module exists."""
        import magvit_isolation_test
        assert magvit_isolation_test is not None
    
    def test_extract_magvit_features_function_exists(self):
        """Test that MagVIT feature extraction function exists."""
        from magvit_isolation_test import extract_magvit_features_only
        assert extract_magvit_features_only is not None
    
    def test_magvit_features_have_correct_shape(self, device):
        """Test that MagVIT features are 512-dimensional."""
        from magvit_isolation_test import extract_magvit_features_only
        
        # Mock video input (B, T, C, H, W)
        mock_video = torch.randn(1, 32, 3, 128, 128, device=device)
        
        features = extract_magvit_features_only(mock_video)
        
        assert features.shape == (1, 512), f"Expected (1, 512), got {features.shape}"
    
    def test_magvit_bypasses_liquid_fusion(self, device):
        """Critical test: Verify Liquid fusion is NOT used."""
        from magvit_isolation_test import extract_magvit_features_only
        import inspect
        
        # Get source code
        source = inspect.getsource(extract_magvit_features_only)
        
        # Should NOT contain actual Liquid fusion calls (checking imports/calls, not docs)
        assert 'liquid_fusion(' not in source
        assert 'LiquidCell(' not in source
        assert 'LiquidDualModalFusion' not in source
    
    def test_evaluate_magvit_only_pipeline(self, device):
        """Test complete pipeline: MagVIT → stats → LLM."""
        from magvit_isolation_test import evaluate_magvit_only
        
        # Mock video and ground truth
        mock_video = torch.randn(1, 32, 3, 128, 128, device=device)
        ground_truth = {
            'type': 'straight line',
            'description': 'A straight line trajectory'
        }
        
        result = evaluate_magvit_only(mock_video, ground_truth)
        
        assert 'description' in result
        assert 'accuracy' in result
        assert 'magvit_features_used' in result
        assert result['magvit_features_used'] == True
        assert result['liquid_fusion_used'] == False
    
    def test_magvit_features_are_deterministic(self, device):
        """Test that same video produces same features."""
        from magvit_isolation_test import extract_magvit_features_only
        
        # Use fixed seed for reproducibility
        torch.manual_seed(42)
        mock_video = torch.randn(1, 32, 3, 128, 128, device=device)
        
        features1 = extract_magvit_features_only(mock_video)
        features2 = extract_magvit_features_only(mock_video)
        
        # Should be very similar (allowing for numerical precision)
        assert torch.allclose(features1, features2, rtol=1e-4)
    
    def test_different_videos_produce_different_features(self, device):
        """Test that different videos produce different features."""
        from magvit_isolation_test import extract_magvit_features_only
        
        video1 = torch.randn(1, 32, 3, 128, 128, device=device)
        video2 = torch.randn(1, 32, 3, 128, 128, device=device)
        
        features1 = extract_magvit_features_only(video1)
        features2 = extract_magvit_features_only(video2)
        
        # Should be different
        assert not torch.allclose(features1, features2, rtol=0.1)
    
    def test_run_magvit_ablation(self, device):
        """Test ablation: compare MagVIT-only vs Random features."""
        from magvit_isolation_test import run_magvit_ablation_study
        
        # Create 3 mock samples
        samples = []
        for i in range(3):
            samples.append({
                'video': torch.randn(1, 32, 3, 128, 128, device=device),
                'ground_truth': {
                    'type': 'straight line',
                    'description': f'Sample {i}'
                }
            })
        
        results = run_magvit_ablation_study(samples)
        
        assert 'random_accuracy' in results
        assert 'magvit_only_accuracy' in results
        assert 'improvement' in results
        assert 0 <= results['random_accuracy'] <= 100
        assert 0 <= results['magvit_only_accuracy'] <= 100


class TestComparisonWithPreviousResults:
    """Compare MagVIT-only results with previous pipeline results."""
    
    def test_comparison_framework_exists(self):
        """Test that comparison function exists."""
        from magvit_isolation_test import compare_with_previous_results
        assert compare_with_previous_results is not None
    
    def test_interpret_magvit_results(self):
        """Test interpretation logic for MagVIT-only results."""
        from magvit_isolation_test import interpret_magvit_results
        
        # Scenario 1: MagVIT > Random (Liquid is problem)
        result1 = interpret_magvit_results(
            magvit_only_accuracy=65.0,
            random_accuracy=52.5,
            previous_real_accuracy=52.5
        )
        assert 'liquid' in result1['bottleneck'].lower()
        
        # Scenario 2: MagVIT = Random (MagVIT is problem)
        result2 = interpret_magvit_results(
            magvit_only_accuracy=52.5,
            random_accuracy=52.5,
            previous_real_accuracy=52.5
        )
        assert 'magvit' in result2['bottleneck'].lower() or 'video' in result2['bottleneck'].lower()
        
        # Scenario 3: MagVIT slightly better (compression is problem)
        result3 = interpret_magvit_results(
            magvit_only_accuracy=58.0,
            random_accuracy=52.5,
            previous_real_accuracy=52.5
        )
        assert result3 is not None

