"""
Tests for Simple Baseline 3D CNN Model

This is an HONEST simple baseline:
- Basic 3D CNN (not I3D, not SlowFast)
- No MAGVIT
- No CLIP
- Template-based outputs (not real LLM)
"""

import pytest
import torch
import numpy as np


def test_simple_cnn_accepts_video_tensors():
    """Baseline model must accept video tensors as input."""
    from simple_3dcnn_baseline import Simple3DCNNClassifier
    
    model = Simple3DCNNClassifier(num_classes=4)
    video = torch.randn(2, 16, 3, 64, 64)  # B, T, C, H, W
    output = model(video)
    
    assert output.shape == (2, 4), f"Expected (2, 4), got {output.shape}"


def test_simple_cnn_classification():
    """Model should classify trajectories into 4 classes."""
    from simple_3dcnn_baseline import Simple3DCNNClassifier
    
    torch.manual_seed(42)
    model = Simple3DCNNClassifier(num_classes=4)
    model.eval()
    
    video = torch.randn(1, 16, 3, 64, 64)
    output = model(video)
    
    # Should output logits for 4 classes
    assert output.shape == (1, 4)
    
    # Should be able to convert to class prediction
    pred_class = output.argmax(dim=1)
    assert 0 <= pred_class.item() < 4


def test_model_is_deterministic():
    """Model should produce same output for same input."""
    from simple_3dcnn_baseline import Simple3DCNNClassifier
    
    torch.manual_seed(42)
    model = Simple3DCNNClassifier(num_classes=4)
    model.eval()
    
    video = torch.randn(1, 16, 3, 64, 64)
    
    output1 = model(video)
    output2 = model(video)
    
    assert torch.allclose(output1, output2, atol=1e-6)


def test_model_has_reasonable_size():
    """Model should not be too large (this is a simple baseline)."""
    from simple_3dcnn_baseline import Simple3DCNNClassifier
    
    model = Simple3DCNNClassifier(num_classes=4)
    num_params = sum(p.numel() for p in model.parameters())
    
    # Simple baseline: should be < 10M parameters
    assert num_params < 10_000_000, f"Too many parameters: {num_params}"
    # But not trivially small
    assert num_params > 10_000, f"Too few parameters: {num_params}"


def test_template_generator_is_honest():
    """Template generator must be labeled as template, not real LLM."""
    from simple_3dcnn_baseline import generate_description_from_template
    
    # Get function docstring
    docstring = generate_description_from_template.__doc__
    
    # Must mention it's a template or placeholder
    assert docstring is not None
    assert "template" in docstring.lower() or "placeholder" in docstring.lower()


def test_template_output_format():
    """Template generator should produce expected format."""
    from simple_3dcnn_baseline import generate_description_from_template
    
    description = generate_description_from_template(
        trajectory_class=0,  # linear
        params={"start": [0, 0, 1], "end": [1, 1, 2]}
    )
    
    assert isinstance(description, str)
    assert len(description) > 10  # Non-trivial output
    assert "linear" in description.lower() or "straight" in description.lower()


def test_no_misleading_imports():
    """Module should not import components it doesn't use."""
    import simple_3dcnn_baseline
    import inspect
    
    source = inspect.getsource(simple_3dcnn_baseline)
    
    # Should NOT import these (we don't have them)
    assert "from magvit" not in source.lower()
    assert "from clip" not in source.lower()
    assert "from openai" not in source.lower()
    assert "import openai" not in source.lower()


def test_batch_processing():
    """Model should handle batches efficiently."""
    from simple_3dcnn_baseline import Simple3DCNNClassifier
    
    torch.manual_seed(42)
    model = Simple3DCNNClassifier(num_classes=4)
    model.eval()
    
    batch_sizes = [1, 4, 8]
    for bs in batch_sizes:
        video = torch.randn(bs, 16, 3, 64, 64)
        output = model(video)
        assert output.shape == (bs, 4)
