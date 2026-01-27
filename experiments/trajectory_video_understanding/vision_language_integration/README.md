# Vision-Language Model Integration

**Integration of LLM with Worker 2 vision model for interpretable trajectory analysis.**

---

## Overview

Connects the trained Worker 2 model (100% validation accuracy) with Large Language Models to provide:
- Natural language descriptions of trajectory videos
- Explanations of classification decisions
- Question answering about trajectories
- Symbolic representations of motion patterns

## Architecture

```
Video → Worker 2 (ResNet + Transformer) → Features → LLM → Natural Language
```

**Supported LLMs:**
- GPT-4 (best quality, cloud API)
- Mistral-7B (good quality, local)
- Phi-2 (lightweight, local)

## Quick Start

```python
from vision_language_bridge import VisionLanguageBridge

# Initialize with trained model
bridge = VisionLanguageBridge(
    vision_model_path='../parallel_workers/worker2_pretrained/results/final_model.pt',
    llm_provider='gpt4'  # or 'mistral', 'phi2'
)

# Generate description
description = bridge.describe_video(video)

# Get explanation
explanation = bridge.explain_classification(video)

# Answer questions
answer = bridge.answer_question(video, "How many objects are visible?")
```

## Development

**Following TDD (see requirements.md Section 3.4):**
1. Behavioral tests in `test_vision_language_bridge.py`
2. Structural tests in `test_vision_language_bridge_structural.py`
3. Evidence in `artifacts/tdd_*.txt`

**API Key Setup:**
```bash
# Set environment variable (secure)
export OPENAI_API_KEY="your-key"

# Or use local models (no API key needed)
llm_provider='mistral'  # or 'phi2'
```

## Files

- `vision_language_bridge.py` - Main integration class
- `llm_prompter.py` - Prompt generation
- `trajectory_qa.py` - Question answering
- `test_vision_language_bridge.py` - Behavioral tests
- `test_vision_language_bridge_structural.py` - Implementation tests

## Related

- Worker 2 Training: `../parallel_workers/RESULTS_ANALYSIS_20260126.md`
- TDD Methodology: `../../../requirements.md` Section 3.4
- LLM Interface: `../../magvit_I3D_LLM_basic_trajectory/llm_interface.py`

---

**Last Updated:** January 26, 2026
