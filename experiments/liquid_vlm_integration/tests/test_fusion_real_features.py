"""
Worker 3: Liquid Fusion with Real Features Tests (TDD RED->GREEN)
"""
import torch
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "experiments/trajectory_video_understanding"))

def test_fusion_with_real_2d_and_3d():
    """Test Liquid Fusion with real MagVIT 2D + real triangulated 3D."""
    from test_fusion_integration import test_real_data_fusion
    
    result = test_real_data_fusion()
    
    assert result["llm_embeddings"].shape == (1, 4096)
    assert torch.isfinite(result["llm_embeddings"]).all()
    print(f"\n✅ Real fusion output: {result['llm_embeddings'].shape}")

def test_fusion_gradients_with_real_data():
    """Test that gradients flow through fusion with real data."""
    from test_fusion_integration import test_real_data_fusion
    
    result = test_real_data_fusion()
    loss = result["llm_embeddings"].sum()
    loss.backward()
    
    assert result["features_2d"].grad is not None
    print("\n✅ Gradients flow through real fusion")

def test_compare_fusion_vs_baseline():
    """Compare Liquid Fusion vs static linear fusion."""
    from test_fusion_integration import compare_fusion_methods
    
    comparison = compare_fusion_methods()
    
    assert "liquid_output" in comparison
    assert "static_output" in comparison
    assert not torch.allclose(comparison["liquid_output"], comparison["static_output"], rtol=0.1)
    print("\n✅ Liquid fusion differs from static baseline")
