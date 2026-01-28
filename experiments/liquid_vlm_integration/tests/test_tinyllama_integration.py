"""
Worker 4: TinyLlama Integration Tests (TDD RED Phase)
"""
import torch
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_tinyllama_model_loads():
    """Test that TinyLlama model loads successfully."""
    from tinyllama_vlm import TinyLlamaVLM
    
    vlm = TinyLlamaVLM()
    assert vlm.model is not None
    print("\n✅ TinyLlama loaded")

def test_generate_description_from_embeddings():
    """Test generating trajectory description from Liquid embeddings."""
    from tinyllama_vlm import TinyLlamaVLM
    
    vlm = TinyLlamaVLM()
    
    # Fake embeddings for now
    embeddings = torch.randn(1, 4096)
    
    description = vlm.generate_description(embeddings, prompt="Describe this trajectory:")
    
    assert isinstance(description, str)
    assert len(description) > 10
    print(f"\n✅ Generated description: {description[:100]}...")

def test_batch_generation():
    """Test generating multiple descriptions."""
    from tinyllama_vlm import TinyLlamaVLM
    
    vlm = TinyLlamaVLM()
    embeddings = torch.randn(3, 4096)
    
    descriptions = vlm.generate_batch(embeddings)
    
    assert len(descriptions) == 3
    assert all(isinstance(d, str) for d in descriptions)
    print(f"\n✅ Generated {len(descriptions)} descriptions")
