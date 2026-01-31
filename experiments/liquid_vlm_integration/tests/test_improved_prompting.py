"""
TDD Tests for Improved TinyLlama Prompting
RED Phase: Tests written FIRST, will fail until implementation
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestImprovedPrompting:
    """Test suite for improved TinyLlama prompting."""
    
    def test_create_structured_prompt_function_exists(self):
        """Test that create_structured_prompt function exists."""
        from tinyllama_vlm import create_structured_prompt
        assert callable(create_structured_prompt)
    
    def test_structured_prompt_has_required_components(self):
        """Test that structured prompt includes all required components."""
        from tinyllama_vlm import create_structured_prompt
        
        prompt = create_structured_prompt()
        
        # Should mention these key aspects
        assert "shape" in prompt.lower() or "path" in prompt.lower()
        assert "direction" in prompt.lower() or "axis" in prompt.lower()
        assert "coordinates" in prompt.lower() or "position" in prompt.lower()
        assert "speed" in prompt.lower() or "motion" in prompt.lower()
    
    def test_structured_prompt_has_constraints(self):
        """Test that structured prompt includes constraints to prevent hallucination."""
        from tinyllama_vlm import create_structured_prompt
        
        prompt = create_structured_prompt()
        
        # Should tell model what NOT to do
        assert "do not" in prompt.lower() or "don't" in prompt.lower()
        assert "video" in prompt.lower() or "url" in prompt.lower() or "tutorial" in prompt.lower()
    
    def test_structured_prompt_returns_string(self):
        """Test that structured prompt returns a string."""
        from tinyllama_vlm import create_structured_prompt
        
        prompt = create_structured_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 50  # Should be detailed
    
    def test_tinyllama_vlm_can_use_custom_prompt(self):
        """Test that TinyLlamaVLM.generate_description accepts custom prompts."""
        from tinyllama_vlm import TinyLlamaVLM, create_structured_prompt
        
        # This test just checks the interface, not actual model loading
        # (which would be too slow for unit tests)
        assert hasattr(TinyLlamaVLM.generate_description, '__call__')
        
        # Check method signature
        import inspect
        sig = inspect.signature(TinyLlamaVLM.generate_description)
        assert 'prompt' in sig.parameters
    
    def test_structured_prompt_is_different_from_default(self):
        """Test that structured prompt differs from generic prompt."""
        from tinyllama_vlm import create_structured_prompt
        
        structured = create_structured_prompt()
        generic = "Describe this trajectory:"
        
        assert structured != generic
        assert len(structured) > len(generic) * 3  # Should be much more detailed
    
    def test_structured_prompt_has_examples_or_instructions(self):
        """Test that structured prompt provides clear instructions."""
        from tinyllama_vlm import create_structured_prompt
        
        prompt = create_structured_prompt()
        
        # Should have numbered points or clear structure
        assert "1." in prompt or "2." in prompt or "-" in prompt or ":" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

