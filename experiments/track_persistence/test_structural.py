#!/usr/bin/env python3
"""
Structural Tests - Can Run Without PyTorch
Tests that modules can be imported and have correct structure.
"""

import pytest
from pathlib import Path
import ast
import sys


def test_realistic_track_generator_imports():
    """Test that realistic_track_generator can be imported."""
    try:
        from realistic_track_generator import (
            Realistic2DTrackGenerator,
            Track2D,
            generate_dataset
        )
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import: {e}")


def test_attention_model_structure():
    """Test that attention model file has correct structure."""
    file_path = Path(__file__).parent / "attention_persistence_model.py"
    
    with open(file_path, 'r') as f:
        tree = ast.parse(f.read())
    
    # Check for required classes
    class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    
    assert 'PositionalEncoding' in class_names
    assert 'AttentionPersistenceModel' in class_names
    assert 'PersistenceTrainer' in class_names
    
    # Check for required functions
    func_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    
    assert 'create_model' in func_names
    assert 'forward' in func_names
    assert 'predict' in func_names


def test_extract_features_structure():
    """Test that extract_track_features has correct structure."""
    file_path = Path(__file__).parent / "extract_track_features.py"
    
    with open(file_path, 'r') as f:
        tree = ast.parse(f.read())
    
    class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    
    assert 'TrackFeatureExtractor' in class_names
    
    func_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    
    assert 'extract_features' in func_names
    assert 'extract_dataset_features' in func_names


def test_integrated_tracker_structure():
    """Test that integrated_tracker has correct structure."""
    file_path = Path(__file__).parent / "integrated_3d_tracker.py"
    
    with open(file_path, 'r') as f:
        tree = ast.parse(f.read())
    
    class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    
    assert 'PersistenceFilter' in class_names
    assert 'Integrated3DTracker' in class_names
    
    func_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    
    assert 'predict' in func_names
    assert 'process_scene' in func_names


def test_test_scenarios_structure():
    """Test that test_3d_scenarios has correct structure."""
    file_path = Path(__file__).parent / "test_3d_scenarios.py"
    
    with open(file_path, 'r') as f:
        tree = ast.parse(f.read())
    
    func_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    
    assert 'create_scenario_1_clean' in func_names
    assert 'create_scenario_2_cluttered' in func_names
    assert 'create_scenario_3_noisy' in func_names
    assert 'run_scenario' in func_names


def test_llm_analyzer_structure():
    """Test that llm_attention_analyzer has correct structure."""
    file_path = Path(__file__).parent / "llm_attention_analyzer.py"
    
    with open(file_path, 'r') as f:
        tree = ast.parse(f.read())
    
    class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    
    assert 'LLMAttentionAnalyzer' in class_names
    
    func_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    
    assert 'analyze_attention_patterns' in func_names
    assert 'analyze_failure_cases' in func_names
    assert 'generate_research_questions' in func_names


def test_all_files_have_docstrings():
    """Test that all modules have docstrings."""
    module_files = [
        "realistic_track_generator.py",
        "attention_persistence_model.py",
        "extract_track_features.py",
        "integrated_3d_tracker.py",
        "test_3d_scenarios.py",
        "llm_attention_analyzer.py"
    ]
    
    for filename in module_files:
        file_path = Path(__file__).parent / filename
        
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())
        
        # Check module has docstring
        assert ast.get_docstring(tree) is not None, f"{filename} missing module docstring"


def test_all_files_have_main_guard():
    """Test that all modules have if __name__ == '__main__' guard."""
    module_files = [
        "realistic_track_generator.py",
        "attention_persistence_model.py",
        "extract_track_features.py",
        "integrated_3d_tracker.py",
        "test_3d_scenarios.py",
        "llm_attention_analyzer.py"
    ]
    
    for filename in module_files:
        file_path = Path(__file__).parent / filename
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        assert 'if __name__ == "__main__":' in content or "if __name__ == '__main__':" in content, \
            f"{filename} missing main guard"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

