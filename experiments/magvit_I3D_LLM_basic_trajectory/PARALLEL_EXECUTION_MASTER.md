# Parallel Execution Master Plan

**Created:** January 21, 2026  
**Status:** Coordinating 4 Parallel Branches  
**Dataset:** 1,200 samples ready

---

## 4 Branches Overview

| Branch | Video Model | Enhancement | LLM | GPU | Status |
|--------|-------------|-------------|-----|-----|--------|
| **Branch 1** | I3D-like CNN | Feature compression | GPT-4 API | cuda:0 | Planning |
| **Branch 2** | Dual-pathway CNN | Feature compression | GPT-4 API | cuda:0 | Planning |
| **Branch 3** | I3D-like CNN | CLIP alignment | Mistral (local) | cuda:0 | Planning |
| **Branch 4** | Dual-pathway CNN | None | Phi-2 (local) | cuda:0 | Planning |

**Note:** Implementing simplified but functional versions for rapid parallel execution.

---

## Simplified Architecture Strategy

### Why Simplified?
- Full I3D/SlowFast require large pretrained weights (~100MB each)
- Full MAGVIT is complex (VQ-VAE with codebooks)
- Goal: Working system that demonstrates full pipeline
- Can be enhanced with full models later

### Simplified Components

**Video Encoder (I3D-like):**
```python
# 3D CNN that mimics I3D structure
Conv3D → BatchNorm → ReLU → MaxPool3D → 
Conv3D → BatchNorm → ReLU → MaxPool3D →
Conv3D → BatchNorm → ReLU → AdaptiveAvgPool3D →
Flatten → Linear(→ 512 features)
```

**Dual-Pathway (SlowFast-like):**
```python
# Two parallel 3D CNNs
Slow: Process every frame, high spatial resolution
Fast: Process every 4th frame, low spatial resolution
Concatenate features → Linear(→ 512)
```

**Feature Compression (MAGVIT-like):**
```python
# Simple autoencoder
Encoder: Linear(512 → 256 → 128)
Decoder: Linear(128 → 256 → 512)
```

**CLIP Alignment:**
```python
# Project features to CLIP space
Linear(512 → 512) with cosine similarity loss to CLIP text embeddings
```

---

## Implementation Phases (All Branches in Parallel)

### Phase 1: Base Models (TDD) - 4 hours
All branches implement simultaneously:

**Branch 1-4 Tasks:**
1. Write model architecture tests
2. Implement simplified video encoder
3. Implement classification head (4 classes)
4. Implement forecasting head (predict next 4 frames' 3D positions)
5. Test forward pass
6. Capture TDD evidence

**Deliverables per branch:**
- `branch{N}/model.py`
- `branch{N}/test_model.py`
- `branch{N}/artifacts/tdd_model_*.txt`

### Phase 2: Training Loop (TDD) - 2 hours
All branches implement simultaneously:

**Branch 1-4 Tasks:**
1. Write training loop tests
2. Implement data loading
3. Implement combined loss (classification + forecasting)
4. Implement optimizer and scheduler
5. Test one epoch
6. Capture TDD evidence

**Deliverables per branch:**
- `branch{N}/train.py`
- `branch{N}/test_train.py`
- `branch{N}/artifacts/tdd_train_*.txt`

### Phase 3: Parallel Training - 3-4 hours
**All branches train simultaneously on EC2:**

```bash
# Terminal 1 - Branch 1
CUDA_VISIBLE_DEVICES=0 python branch1/train.py &

# Terminal 2 - Branch 2 (time-slice with Branch 1)
CUDA_VISIBLE_DEVICES=0 python branch2/train.py &

# Terminal 3 - Branch 3 (time-slice)
CUDA_VISIBLE_DEVICES=0 python branch3/train.py &

# Terminal 4 - Branch 4 (time-slice)
CUDA_VISIBLE_DEVICES=0 python branch4/train.py &

# Monitor all
python monitor_all_branches.py
```

**Training Config:**
- Epochs: 30 (sufficient for small models)
- Batch size: 16
- Learning rate: 0.001
- Train/val split: 80/20 (960/240)

**Each branch saves:**
- Status JSON every 5 minutes
- Checkpoint every 5 epochs
- Final results with timestamps

### Phase 4: LLM Integration - 1 hour
**All branches integrate LLMs:**

- Branches 1 & 2: GPT-4 API calls
- Branch 3: Mistral-7B local inference
- Branch 4: Phi-2 local inference

Generate equations + descriptions for test set.

### Phase 5: Results Compilation - 1 hour
**Generate comparison report:**

```
PARALLEL_RESULTS_COMPARISON.md:
- Classification accuracy (all 4 branches)
- Forecasting MAE/RMSE (all 4 branches)
- Confusion matrices (all 4)
- Sample equations/descriptions (all 4)
- Training curves (all 4)
- Winner declaration
```

---

## Execution Timeline

**Total estimated time: 10-12 hours**

| Time | Phase | Activity |
|------|-------|----------|
| 0-4h | Phase 1 | Model implementation (TDD, all branches) |
| 4-6h | Phase 2 | Training loops (TDD, all branches) |
| 6-10h | Phase 3 | Parallel training (all branches) |
| 10-11h | Phase 4 | LLM integration (all branches) |
| 11-12h | Phase 5 | Results compilation |

---

## Status Monitoring

### Monitor Script
`monitor_all_branches.py` displays:

```
=== Branch Training Status ===
Updated: 2026-01-21 14:30:00

Branch 1 (I3D+MAGVIT+GPT4):     Epoch 15/30 | Acc: 87.3% | Loss: 0.234 | MAE: 0.12
Branch 2 (SlowFast+MAGVIT+GPT4): Epoch 16/30 | Acc: 89.1% | Loss: 0.198 | MAE: 0.09
Branch 3 (I3D+CLIP+Mistral):     Epoch 14/30 | Acc: 84.2% | Loss: 0.287 | MAE: 0.15
Branch 4 (SlowFast+Phi2):        Epoch 15/30 | Acc: 86.5% | Loss: 0.251 | MAE: 0.13

Est. completion: 2.5 hours
```

### Files Synced to MacBook
```
status/
├── branch1_status.json (updated every 5 min)
├── branch2_status.json
├── branch3_status.json
└── branch4_status.json

results/
├── branch1/YYYYMMDD_HHMM_*.png
├── branch2/YYYYMMDD_HHMM_*.png
├── branch3/YYYYMMDD_HHMM_*.png
└── branch4/YYYYMMDD_HHMM_*.png
```

---

## Expected Final Results

### Classification Accuracy
- Branch 1: 85-90%
- Branch 2: 88-93% (best motion understanding)
- Branch 3: 82-87%
- Branch 4: 84-89%

### Forecasting MAE
- Branch 1: 10-14%
- Branch 2: 8-12% (best)
- Branch 3: 12-16%
- Branch 4: 10-14%

### LLM Quality
- Branches 1 & 2 (GPT-4): 95%+ equation accuracy
- Branch 3 (Mistral): 85-90% equation accuracy
- Branch 4 (Phi-2): 80-85% equation accuracy

---

## Success Criteria

### All Branches Must Achieve:
- ✅ Complete TDD evidence for all components
- ✅ Training completes successfully
- ✅ Results saved with timestamps
- ✅ Status updates every 5 minutes
- ✅ Classification accuracy >80%
- ✅ Forecasting MAE <20%

### Comparison Complete When:
- ✅ All 4 branches finished training
- ✅ All results documented
- ✅ Winner declared with justification
- ✅ All code committed and pushed

---

## Next Step: Begin Phase 1

Creating simplified model implementations for all 4 branches now...

**Status:** READY TO EXECUTE

