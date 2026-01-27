# Branch 1 Implementation Plan: I3D + MAGVIT + GPT-4

**Branch:** `magvit-I3D-LLM/i3d-magvit-gpt4`  
**Status:** Planning → Implementation → Training → Evaluation  
**Created:** January 21, 2026

---

## Architecture Overview

```
Input: Video frames (B, T, 3, H, W)
    ↓
I3D Encoder (Kinetics-400 pretrained, frozen backbone)
    Output: (B, T, 2048) temporal features
    ↓
MAGVIT VQ-VAE Encoder (compress features)
    Output: (B, T, 256) compressed latents
    ↓
Temporal Aggregation (Transformer)
    Output: (B, 512) sequence embedding
    ↓
├─→ Classification Head: Linear(512 → 128 → 4)
│   Loss: CrossEntropyLoss
│   Output: (B, 4) logits
│
├─→ Forecasting Head: Linear(512 → T_future × 256)
│   MAGVIT Decoder → future frames
│   Loss: MSE on 3D positions
│   Output: (B, T_future, 3, H, W) or (B, T_future, 3) positions
│
└─→ GPT-4 API Integration
    Input: [classification_logits, sequence_embedding, sample_points]
    Output: {"equation": str, "description": str}
```

---

## Implementation Tasks (TDD)

### Phase 1: Model Architecture (Days 4-5)

**Task 1.1: I3D Integration (TDD)**
- [ ] Write tests for I3D wrapper (`test_i3d_wrapper.py`)
- [ ] Implement I3D wrapper with pretrained weights
- [ ] Verify output shapes and feature extraction
- [ ] Capture TDD evidence (RED-GREEN-REFACTOR)

**Task 1.2: MAGVIT Encoder Integration (TDD)**
- [ ] Write tests for MAGVIT encoder (`test_magvit_encoder.py`)
- [ ] Implement MAGVIT VQ-VAE encoder wrapper
- [ ] Test compression/decompression
- [ ] Capture TDD evidence

**Task 1.3: Temporal Aggregator (TDD)**
- [ ] Write tests for temporal transformer (`test_temporal_aggregator.py`)
- [ ] Implement transformer-based aggregation
- [ ] Test sequence embedding generation
- [ ] Capture TDD evidence

**Task 1.4: Classification Head (TDD)**
- [ ] Write tests for classification head (`test_classification_head.py`)
- [ ] Implement 2-layer MLP classifier
- [ ] Test forward pass and loss computation
- [ ] Capture TDD evidence

**Task 1.5: Forecasting Head (TDD)**
- [ ] Write tests for forecasting head (`test_forecasting_head.py`)
- [ ] Implement forecasting with MAGVIT decoder
- [ ] Test future frame generation
- [ ] Capture TDD evidence

### Phase 2: Training Loop (Days 6-7)

**Task 2.1: Data Loading (TDD)**
- [ ] Write tests for dataloader (`test_dataloader.py`)
- [ ] Implement PyTorch Dataset wrapper
- [ ] Test batching and augmentation
- [ ] Capture TDD evidence

**Task 2.2: Training Script (TDD)**
- [ ] Write tests for trainer (`test_trainer.py`)
- [ ] Implement training loop with:
  - Combined loss (classification + forecasting)
  - Optimizer (AdamW)
  - Learning rate scheduler (Cosine)
  - Gradient clipping
  - Checkpointing
- [ ] Test one training step
- [ ] Capture TDD evidence

**Task 2.3: Validation and Metrics**
- [ ] Implement validation loop
- [ ] Log metrics every epoch
- [ ] Save checkpoints
- [ ] Generate visualizations

### Phase 3: GPT-4 Integration (Day 8)

**Task 3.1: GPT-4 Interface Testing**
- [ ] Test GPT-4 API connection
- [ ] Test equation generation
- [ ] Test description generation
- [ ] Handle API errors gracefully

**Task 3.2: Post-Processing**
- [ ] Generate equations for all samples
- [ ] Generate descriptions for all samples
- [ ] Save results with timestamps

### Phase 4: Training Execution (Days 9-10)

**Task 4.1: Full Dataset Training**
- [ ] Train on 1,200 samples (960 train, 240 val)
- [ ] 50 epochs (~2-3 hours estimated)
- [ ] Monitor and log:
  - Classification accuracy
  - Forecasting MAE/RMSE
  - Training/validation loss
- [ ] Save best checkpoint

**Task 4.2: Result Generation**
- [ ] Generate final metrics
- [ ] Create confusion matrix
- [ ] Plot forecasting results
- [ ] Run GPT-4 on test samples
- [ ] Save all results with timestamps

### Phase 5: Documentation and Sync (Day 10)

**Task 5.1: Results Documentation**
- [ ] Create `BRANCH1_RESULTS.md`
- [ ] Include all metrics
- [ ] Include visualizations
- [ ] Include sample equations/descriptions
- [ ] Commit to git

**Task 5.2: Sync to MacBook**
- [ ] Push all results to remote
- [ ] Sync result images
- [ ] Update status file every 15 minutes during training

---

## Expected Performance

### Classification
- **Target:** >92% accuracy
- **Per-class:** >88% for each trajectory type
- **Confusion matrix:** High diagonal, low off-diagonal

### Forecasting
- **Target MAE:** <12% (4 frames ahead)
- **Target RMSE:** <15%
- **Best for:** Linear and circular (most predictable)

### LLM Quality
- **Equations:** >95% syntactically correct
- **Descriptions:** >98% coherent and accurate
- **GPT-4 cost:** ~$12-15 for 1,200 samples

---

## File Structure

```
experiments/magvit_I3D_LLM_basic_trajectory/
├── branch1/
│   ├── models/
│   │   ├── i3d_wrapper.py
│   │   ├── magvit_encoder.py
│   │   ├── temporal_aggregator.py
│   │   ├── classification_head.py
│   │   ├── forecasting_head.py
│   │   └── full_model.py
│   ├── tests/
│   │   ├── test_i3d_wrapper.py
│   │   ├── test_magvit_encoder.py
│   │   ├── test_temporal_aggregator.py
│   │   ├── test_classification_head.py
│   │   ├── test_forecasting_head.py
│   │   ├── test_dataloader.py
│   │   └── test_trainer.py
│   ├── artifacts/
│   │   ├── tdd_*.txt (all TDD evidence)
│   │   └── training_logs/
│   ├── results/
│   │   ├── YYYYMMDD_HHMM_*.png (visualizations)
│   │   ├── YYYYMMDD_HHMM_*.json (metrics)
│   │   └── YYYYMMDD_HHMM_best_checkpoint.pth
│   ├── train.py
│   ├── evaluate.py
│   └── config.yaml
├── BRANCH1_IMPLEMENTATION_PLAN.md (this file)
└── BRANCH1_RESULTS.md (created after training)
```

---

## Dependencies

### Additional Packages
```bash
pip install torch torchvision
pip install timm  # For I3D models
pip install einops  # For tensor operations
pip install openai  # For GPT-4 API
pip install wandb  # Optional: for experiment tracking
```

### Pretrained Models
- I3D: From torchvision or timm
- MAGVIT: Either pretrained or simplified encoder

---

## Monitoring and Status Updates

### Status File Format
`status/branch1_status.json`:
```json
{
  "branch": "i3d-magvit-gpt4",
  "status": "training",
  "phase": "classification",
  "epoch": 23,
  "max_epochs": 50,
  "metrics": {
    "train_acc": 0.89,
    "val_acc": 0.86,
    "train_loss": 0.245,
    "val_loss": 0.312,
    "forecast_mae": 0.098
  },
  "est_time_remaining": "1.2 hours",
  "last_update": "2026-01-21T14:30:00Z",
  "gpu": "cuda:0"
}
```

### Update Frequency
- Every 15 minutes during training
- Every epoch completion
- On checkpoint save
- On error/completion

---

## Risk Mitigation

### Potential Issues
1. **I3D pretrained weights:** May need to adapt input size
2. **MAGVIT complexity:** Implement simplified version if needed
3. **GPU memory:** May need gradient checkpointing
4. **GPT-4 costs:** Rate limit API calls, batch if possible
5. **Training time:** May take longer than estimated

### Fallback Plans
1. Use simpler CNN if I3D too complex
2. Skip MAGVIT, use direct features
3. Reduce batch size, use mixed precision
4. Use cached GPT-4 responses for common patterns
5. Continue training in background

---

## Success Criteria

### Minimum Acceptable
- ✅ Classification accuracy >85%
- ✅ Forecasting MAE <20%
- ✅ All TDD evidence captured
- ✅ Results synced to MacBook

### Excellent Performance  
- 🎯 Classification accuracy >92%
- 🎯 Forecasting MAE <12%
- 🎯 GPT-4 equations >95% correct
- 🎯 Training completed in <3 hours

---

## Timeline

**Days 4-5:** Model implementation (TDD)  
**Days 6-7:** Training loop (TDD)  
**Day 8:** GPT-4 integration  
**Days 9-10:** Training execution + results  

**Total:** ~7 days for complete branch

---

**Status:** Ready to begin Phase 1 - Model Architecture

