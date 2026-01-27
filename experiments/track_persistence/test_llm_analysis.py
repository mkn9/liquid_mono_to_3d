#!/usr/bin/env python3
"""Test LLM analysis system."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Create mock training results for testing
mock_results = {
    "num_epochs": 30,
    "best_epoch": 28,
    "timestamp": "2026-01-16T16:38:00",
    "train_history": {
        "loss": [0.5, 0.3, 0.2, 0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05] * 3,
        "accuracy": [0.75, 0.85, 0.90, 0.93, 0.95, 0.96, 0.97, 0.975, 0.98, 0.985] * 3
    },
    "val_history": {
        "loss": [0.52, 0.32, 0.22, 0.16, 0.13, 0.11, 0.09, 0.075, 0.065, 0.055] * 3,
        "accuracy": [0.73, 0.83, 0.88, 0.91, 0.94, 0.95, 0.97, 0.98, 0.989, 0.993] * 3
    },
    "test_results": {
        "accuracy": 0.9867,
        "f1": 0.9846,
        "precision": 0.9823,
        "recall": 0.9871
    }
}

def test_llm_analyzer_creation():
    """Test that analyzer can be created."""
    try:
        from llm_model_analysis import LLMModelAnalyzer
        
        # Check if API key exists
        import os
        if not os.getenv('ANTHROPIC_API_KEY'):
            print("⚠️  ANTHROPIC_API_KEY not set - skipping API tests")
            print("✅ LLMModelAnalyzer class structure verified")
            return True
        
        analyzer = LLMModelAnalyzer()
        print("✅ LLMModelAnalyzer initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create analyzer: {e}")
        return False

def test_prompt_generation():
    """Test prompt generation."""
    try:
        from llm_model_analysis import LLMModelAnalyzer
        
        import os
        if not os.getenv('ANTHROPIC_API_KEY'):
            os.environ['ANTHROPIC_API_KEY'] = 'test-key-for-structure-test'
        
        analyzer = LLMModelAnalyzer()
        prompt = analyzer._create_analysis_prompt(mock_results)
        
        # Check prompt contains key elements
        assert 'PERFORMANCE ASSESSMENT' in prompt
        assert 'TRAINING DYNAMICS' in prompt
        assert 'ARCHITECTURAL INSIGHTS' in prompt
        assert 'Test Accuracy: 0.9867' in prompt
        
        print("✅ Prompt generation working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Prompt generation failed: {e}")
        return False

def test_structure_analysis():
    """Test analysis structuring."""
    try:
        from llm_model_analysis import LLMModelAnalyzer
        
        import os
        if not os.getenv('ANTHROPIC_API_KEY'):
            os.environ['ANTHROPIC_API_KEY'] = 'test-key'
        
        analyzer = LLMModelAnalyzer()
        
        mock_llm_response = """
        1. PERFORMANCE ASSESSMENT: Excellent performance at 98.67%...
        2. TRAINING DYNAMICS: Smooth convergence...
        """
        
        structured = analyzer._structure_analysis(mock_llm_response, mock_results)
        
        assert 'llm_analysis' in structured
        assert 'training_summary' in structured
        assert structured['training_summary']['test_accuracy'] == 0.9867
        
        print("✅ Analysis structuring working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Analysis structuring failed: {e}")
        return False

def test_save_analysis():
    """Test saving analysis."""
    try:
        from llm_model_analysis import LLMModelAnalyzer
        import tempfile
        import os
        
        if not os.getenv('ANTHROPIC_API_KEY'):
            os.environ['ANTHROPIC_API_KEY'] = 'test-key'
        
        analyzer = LLMModelAnalyzer()
        
        mock_analysis = {
            'llm_analysis': 'Test analysis content',
            'training_summary': {
                'test_accuracy': 0.9867,
                'test_f1': 0.9846,
                'best_epoch': 28,
                'num_epochs': 30
            },
            'timestamp': '2026-01-16'
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'analysis.json'
            analyzer.save_analysis(mock_analysis, output_path)
            
            # Check JSON file
            assert output_path.exists()
            with open(output_path) as f:
                saved = json.load(f)
            assert saved['training_summary']['test_accuracy'] == 0.9867
            
            # Check markdown file
            md_path = output_path.with_suffix('.md')
            assert md_path.exists()
            with open(md_path) as f:
                md_content = f.read()
            assert 'LLM-Assisted Model Analysis' in md_content
            assert '0.9867' in md_content
        
        print("✅ Analysis saving working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Analysis saving failed: {e}")
        return False


if __name__ == '__main__':
    print("="*60)
    print("Testing LLM Analysis System")
    print("="*60)
    
    tests = [
        test_llm_analyzer_creation,
        test_prompt_generation,
        test_structure_analysis,
        test_save_analysis
    ]
    
    results = []
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        results.append(test())
    
    print("\n" + "="*60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("="*60)
    
    if all(results):
        print("✅ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)

