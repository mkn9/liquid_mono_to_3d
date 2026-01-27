# 4 Parallel Branch Specifications

**Experiment:** magvit_I3D_LLM_basic_trajectory  
**Date:** January 20, 2026  
**Status:** Planning Phase

---

## Branch Comparison Matrix

| Branch | Video Model | Video Enhancement | LLM | Deployment | Strengths |
|--------|-------------|-------------------|-----|------------|-----------|
| **1** | I3D | MAGVIT encoder | GPT-4/5 | Cloud API | Best LLM quality, proven video |
| **2** | SlowFast | MAGVIT encoder | GPT-4/5 | Cloud API | Best motion understanding + best LLM |
| **3** | I3D | CLIP bridge | Mistral-Instruct | Local (EC2) | No API costs, vision-language aligned |
| **4** | SlowFast | None | Phi-2/WizardMath | Local (EC2) | Lightweight, math-specialized |

---

## Branch 1: `magvit-I3D-LLM/i3d-magvit-gpt4`

### Architecture

```
Input: Video frames (B, T, 3, H, W)
    ↓
I3D Encoder (Kinetics-400 pretrained)
    Output: (B, T, 2048) temporal features
    ↓
MAGVIT VQ-VAE Encoder (compress)
    Output: (B, T, 256) compressed latents
    ↓
Temporal Aggregation (Transformer or LSTM)
    Output: (B, 512) sequence embedding
    ↓
├─→ Classification Head
│   Linear(512 → 128 → 4)
│   Output: (B, 4) logits [linear, circular, helical, parabolic]
│
├─→ Forecasting Head
│   Linear(512 → T_future × 256)
│   MAGVIT Decoder → future frames
│   Output: (B, T_future, 3, H, W)
│
└─→ GPT-4 API Interface
    Input: Concatenate [classification_logits, sequence_embedding, sample_points]
    Prompt: "Given trajectory of type X with motion pattern Y, generate:
             1. Symbolic equation
             2. Natural language description"
    Output: {"equation": "y = mx + b", "description": "Linear motion..."}
```

### Components

**1. I3D (Inflated 3D ConvNet)**
- Pre-trained on Kinetics-400
- Input: 16 frames × 224×224 (or 64×64 for efficiency)
- Output: 2048-dim features per frame
- Frozen backbone (only fine-tune classification head)

**2. MAGVIT Encoder**
- VQ-VAE style compression
- Reduces 2048-dim → 256-dim latents
- Trained or use pretrained weights if available
- Enables efficient forecasting in latent space

**3. GPT-4 API**
- Model: gpt-4-turbo or gpt-4o
- Temperature: 0.7 for creative descriptions
- Max tokens: 500
- Cost: ~$0.01 per sample

### Training Config

```yaml
model:
  i3d_backbone: 'kinetics400'
  i3d_freeze: true
  magvit_latent_dim: 256
  temporal_aggregator: 'transformer'
  
training:
  epochs: 50
  batch_size: 16
  learning_rate: 0.001
  optimizer: 'adam'
  scheduler: 'cosine'
  
data:
  num_samples: 1200
  augmentation: true
  image_size: [64, 64]
  frames: 16
  
llm:
  provider: 'openai'
  model: 'gpt-4o'
  api_key: ${OPENAI_API_KEY}
```

### Expected Performance

- **Classification:** 92-97% (I3D excellent for video)
- **Forecasting:** 8-12% MAE (MAGVIT latent helps)
- **Equations:** 95%+ quality (GPT-4 very strong)
- **Descriptions:** 98%+ quality (GPT-4 best-in-class)

---

## Branch 2: `magvit-I3D-LLM/slowfast-magvit-gpt4`

### Architecture

```
Input: Video frames (B, T, 3, H, W)
    ↓
SlowFast Two-Pathway Network
    ├─→ Slow: Low FPS (4 frames), high spatial detail
    └─→ Fast: High FPS (32 frames), low spatial detail
    ↓
Lateral connections (fuse pathways)
    Output: (B, 2304) fused features
    ↓
MAGVIT VQ-VAE Encoder
    Output: (B, 256) compressed latents
    ↓
├─→ Classification Head
│   Output: (B, 4)
│
├─→ Forecasting Head
│   MAGVIT Decoder → future frames
│   Output: (B, T_future, 3, H, W)
│
└─→ GPT-4 API Interface
    Similar to Branch 1
```

### Components

**1. SlowFast Network**
- Slow pathway: T/4 frames at high resolution
- Fast pathway: T frames at low resolution
- Fusion via lateral connections
- Better motion understanding than single-pathway

**2. MAGVIT Encoder**
- Same as Branch 1
- Compresses SlowFast output

**3. GPT-4 API**
- Same configuration as Branch 1

### Training Config

```yaml
model:
  slowfast_backbone: 'slowfast_r50'
  slowfast_freeze: true
  slow_alpha: 4  # Frame rate ratio
  fast_beta: 8   # Channel ratio
  magvit_latent_dim: 256
  
training:
  epochs: 50
  batch_size: 12  # Larger model, smaller batch
  learning_rate: 0.0008
  
data:
  # Same as Branch 1
  
llm:
  # Same as Branch 1
```

### Expected Performance

- **Classification:** 94-98% (SlowFast best for motion)
- **Forecasting:** 7-10% MAE (Best motion modeling)
- **Equations:** 95%+ quality (GPT-4)
- **Descriptions:** 98%+ quality (GPT-4)

---

## Branch 3: `magvit-I3D-LLM/i3d-mistral-clip`

### Architecture

```
Input: Video frames (B, T, 3, H, W)
    ↓
I3D Encoder (Kinetics-400)
    Output: (B, T, 2048)
    ↓
Project each frame to CLIP space
    CLIP Vision Encoder: 2048 → 512 (CLIP embedding)
    Output: (B, T, 512) in CLIP space
    ↓
Temporal Transformer
    Output: (B, 512)
    ↓
├─→ Classification Head
│   Output: (B, 4)
│
├─→ Forecasting Head
│   Predict future CLIP embeddings
│   Output: (B, T_future, 512)
│   Decode: CLIP decoder → future frames
│
└─→ Mistral-Instruct (Local)
    Input: [CLIP embeddings + classification]
    Model: mistralai/Mistral-7B-Instruct-v0.2
    Running on EC2 (no API costs)
    Output: Equations + descriptions
```

### Components

**1. I3D + CLIP Bridge**
- I3D for temporal features
- CLIP to align with language space
- Bridge enables better LLM integration

**2. Mistral-Instruct (7B)**
- Running locally on EC2
- 7B parameters (fits in ~14GB VRAM)
- Strong instruction following
- No API costs

### Training Config

```yaml
model:
  i3d_backbone: 'kinetics400'
  i3d_freeze: true
  clip_model: 'ViT-B/32'
  clip_freeze: true
  temporal_transformer_layers: 4
  
training:
  epochs: 50
  batch_size: 16
  learning_rate: 0.001
  
llm:
  provider: 'local'
  model: 'mistralai/Mistral-7B-Instruct-v0.2'
  device: 'cuda'
  max_tokens: 500
  temperature: 0.7
```

### Expected Performance

- **Classification:** 90-95% (I3D + CLIP solid)
- **Forecasting:** 10-15% MAE (CLIP space prediction)
- **Equations:** 85-90% quality (Mistral good but not GPT-4)
- **Descriptions:** 90-95% quality (Mistral strong)

---

## Branch 4: `magvit-I3D-LLM/slowfast-phi2-wizardmath`

### Architecture

```
Input: Video frames (B, T, 3, H, W)
    ↓
SlowFast Network
    Output: (B, 2304) fused features
    ↓
Temporal Aggregation
    Output: (B, 512)
    ↓
├─→ Classification Head
│   Output: (B, 4)
│
├─→ Forecasting Head
│   Direct regression in pixel space
│   Output: (B, T_future, 3, H, W)
│
└─→ Phi-2 or WizardMath (Local)
    Phi-2: 2.7B general model
    WizardMath: 7B math-specialized model
    Input: [features + classification + sample points]
    Output: Equations + descriptions
```

### Components

**1. SlowFast**
- Same as Branch 2
- No MAGVIT (simpler pipeline)

**2. Phi-2 or WizardMath**
- **Phi-2:** 2.7B parameters, fast inference
- **WizardMath:** 7B parameters, specialized for math
- Choice depends on equation quality in evaluation

### Training Config

```yaml
model:
  slowfast_backbone: 'slowfast_r50'
  slowfast_freeze: true
  
training:
  epochs: 50
  batch_size: 16
  learning_rate: 0.001
  
llm:
  provider: 'local'
  model: 'microsoft/phi-2'  # or 'WizardLM/WizardMath-7B-V1.1'
  device: 'cuda'
  max_tokens: 500
  temperature: 0.5  # Lower for math precision
```

### Expected Performance

- **Classification:** 93-97% (SlowFast excellent)
- **Forecasting:** 9-13% MAE (Good motion modeling)
- **Equations:** 80-90% quality (WizardMath helps, but smaller LLM)
- **Descriptions:** 85-92% quality (Smaller LLM limitation)

---

## Resource Requirements

### GPU Memory (per branch)

| Branch | Video Model | LLM | Total VRAM | Can Share GPU? |
|--------|-------------|-----|------------|----------------|
| 1 | I3D (~4GB) | GPT-4 (API) | ~6GB | ✅ Yes |
| 2 | SlowFast (~6GB) | GPT-4 (API) | ~8GB | ✅ Yes |
| 3 | I3D + CLIP (~5GB) | Mistral-7B (~14GB) | ~20GB | ⚠️ Needs own GPU |
| 4 | SlowFast (~6GB) | Phi-2/WizardMath (~10GB) | ~16GB | ⚠️ Needs own GPU |

### Recommended GPU Allocation

**Option A: 4 GPUs (Ideal)**
- GPU 0: Branch 1
- GPU 1: Branch 2
- GPU 2: Branch 3
- GPU 3: Branch 4

**Option B: 2 GPUs (Acceptable)**
- GPU 0: Branch 1 + Branch 2 (time-sliced, both use API for LLM)
- GPU 1: Branch 3 OR Branch 4 (alternate days)

**Option C: 1 GPU (Slow but feasible)**
- Sequential execution: Branch 1 → 2 → 3 → 4
- Total time: 4× longer (not truly parallel)

### Training Time Estimates

**Per Branch (to completion):**
- Data generation: 30-40 minutes (shared)
- Training classification: 2-4 hours (50 epochs)
- Training forecasting: 3-5 hours (50 epochs)
- LLM integration: 1-2 hours (API or local setup)
- Evaluation: 30-60 minutes
- **Total per branch: ~8-12 hours**

**Parallel (4 GPUs): ~8-12 hours total**
**Sequential (1 GPU): ~32-48 hours total**

---

## Success Metrics (Comparative)

### Expected Ranking

**Classification:**
1. Branch 2 (SlowFast-MAGVIT-GPT4) - 94-98%
2. Branch 4 (SlowFast-Phi2) - 93-97%
3. Branch 1 (I3D-MAGVIT-GPT4) - 92-97%
4. Branch 3 (I3D-Mistral-CLIP) - 90-95%

**Forecasting:**
1. Branch 2 (SlowFast-MAGVIT-GPT4) - 7-10% MAE
2. Branch 1 (I3D-MAGVIT-GPT4) - 8-12% MAE
3. Branch 4 (SlowFast-Phi2) - 9-13% MAE
4. Branch 3 (I3D-Mistral-CLIP) - 10-15% MAE

**Equation Quality:**
1. Branch 1 & 2 (GPT-4) - 95%+
2. Branch 3 (Mistral) - 85-90%
3. Branch 4 (Phi-2/WizardMath) - 80-90%

**NL Description Quality:**
1. Branch 1 & 2 (GPT-4) - 98%+
2. Branch 3 (Mistral) - 90-95%
3. Branch 4 (Phi-2) - 85-92%

**Overall Winner Prediction:** Branch 2 (SlowFast-MAGVIT-GPT4)

---

## Cost Analysis

| Branch | Training Cost | LLM API Cost | Total (per run) |
|--------|---------------|--------------|-----------------|
| 1 | $3-5 (EC2) | $12-15 (GPT-4) | $15-20 |
| 2 | $3-5 (EC2) | $12-15 (GPT-4) | $15-20 |
| 3 | $4-6 (EC2) | $0 (local) | $4-6 |
| 4 | $4-6 (EC2) | $0 (local) | $4-6 |

**Total for all 4 branches: $38-52**

---

## Branch Status Files (Synced to MacBook)

**File:** `status/branch1_i3d-magvit-gpt4_status.json`
```json
{
  "branch": "i3d-magvit-gpt4",
  "status": "training",
  "phase": "classification",
  "epoch": 23,
  "max_epochs": 50,
  "accuracy": 0.89,
  "loss": 0.245,
  "est_time_remaining": "1.2 hours",
  "last_update": "2026-01-27T14:30:00Z",
  "gpu": "cuda:0",
  "results_saved": [
    "20260127_1430_epoch23_classification.png",
    "20260127_1430_epoch23_metrics.json"
  ]
}
```

Similar files for branches 2, 3, 4.

---

**Next Step: Begin Week 1 Foundation (Day 1: Trajectory Renderer TDD)**

