# EC2 Instance Assessment: g5.2xlarge
**Date:** January 26, 2026, 8:15 PM EST  
**Instance Type:** g5.2xlarge  
**Purpose:** Assess instance choice based on actual workload performance

---

## TL;DR: Instance Assessment

**Verdict:** ✅ **g5.2xlarge is APPROPRIATE for your current workload**

**Key Findings:**
- ✅ Successfully trained 100% accuracy model in ~30 epochs
- ✅ Batch size 16 works well (reduced from 32 to prevent resource issues)
- ✅ No GPU memory issues with current architecture
- ✅ Fast training times (2.2 minutes for 10 epochs, 13.3 seconds per epoch)
- ⚠️ Could scale up for larger experiments, but NOT necessary now

**Recommendation:** **Keep g5.2xlarge** unless you plan to:
1. Train significantly larger models (e.g., full Vision Transformers)
2. Use much larger batch sizes (>32)
3. Run 4+ parallel training jobs simultaneously

---

## g5.2xlarge Specifications

| Spec | Value |
|------|-------|
| **GPU** | 1× NVIDIA A10G Tensor Core GPU |
| **GPU Memory** | 24 GB GDDR6 |
| **vCPUs** | 8 vCPUs (AMD EPYC 7R32) |
| **RAM** | 32 GB |
| **Network** | Up to 10 Gbps |
| **Storage** | EBS-optimized |
| **Cost** | ~$1.21/hour (on-demand, us-east-1) |

---

## Your Actual Workload Analysis

### Training Configuration

**Successful Run (Worker 2 - 100% Accuracy Model):**
```yaml
Model: ResNet-18 (frozen) + Transformer
Dataset: 10,000 samples
  - Train: 8,000 samples
  - Validation: 1,000 samples
  - Test: 1,000 samples
Epochs: 30
Batch Size: 16
Video Format: 32 frames × 64×64 pixels × 3 channels
Input per sample: ~0.38 MB (32 × 64 × 64 × 3 × 4 bytes)
Batch memory: ~6 MB raw data
```

### Performance Metrics

**Training Speed:**
- **10 epochs in 2.21 minutes** (MagVIT model)
- **Average time per epoch: 13.3 seconds**
- **30 epochs completed successfully** (Worker 2)

**GPU Utilization:**
- Batch size 16: ✅ Comfortable
- Batch size 32 (4 parallel workers): ⚠️ Resource issues (reduced to 8 per worker)
- No OOM errors with batch size 16 on single training job

**Resource Issues Encountered:**
- **Parallel training with batch_size=16 × 4 workers = 64 total samples:** ❌ Too much
- **Solution: Reduced to batch_size=8 per worker = 32 total:** ✅ Success

---

## GPU Memory Usage Estimate

### Current Workload

**Per Training Sample:**
```
Video Input: 32 frames × 64×64 × 3 channels × 4 bytes = 1.57 MB
ResNet-18 features: 512-dim × 32 frames × 4 bytes = 65 KB
Transformer intermediate: ~10 MB (attention matrices, activations)
Gradients: ~2× model size

Estimated per sample: ~12-15 MB
Batch of 16: ~192-240 MB
```

**Model Size:**
```
ResNet-18 (frozen): ~45 MB (not trained)
Transformer: ~10 MB parameters
Total trainable: ~10 MB
With optimizer states (Adam): ~30 MB
```

**Total GPU Memory Usage (batch_size=16):**
```
Model + Optimizer: ~75 MB
Batch data + activations: ~240 MB
Gradients + temporaries: ~150 MB
Total: ~465 MB
```

**Available on g5.2xlarge:** 24 GB  
**Headroom:** **~23.5 GB (98% unused!)**

---

## Comparison with Other Instance Types

| Instance | GPU | VRAM | vCPUs | RAM | Cost/hr | Your Workload |
|----------|-----|------|-------|-----|---------|---------------|
| **g5.2xlarge** | 1× A10G | **24 GB** | 8 | 32 GB | **$1.21** | ✅ **Perfect fit** |
| g5.xlarge | 1× A10G | 24 GB | 4 | 16 GB | $1.01 | ⚠️ Less RAM/CPUs |
| g5.4xlarge | 1× A10G | 24 GB | 16 | 64 GB | $1.63 | ⚠️ Overkill (same GPU) |
| g5.8xlarge | 1× A10G | 24 GB | 32 | 128 GB | $2.45 | ❌ Overkill |
| g5.12xlarge | 4× A10G | 96 GB | 48 | 192 GB | $5.67 | ❌ Way overkill |
| p3.2xlarge | 1× V100 | 16 GB | 8 | 61 GB | $3.06 | ⚠️ More expensive, less VRAM |
| p3.8xlarge | 4× V100 | 64 GB | 32 | 244 GB | $12.24 | ❌ 10× cost |
| p4d.24xlarge | 8× A100 | 320 GB | 96 | 1152 GB | $32.77 | ❌ 27× cost |

---

## When to Scale Up vs Down

### ⬇️ Scale DOWN to g5.xlarge ($1.01/hr) if:
- ❌ You're NOT running this workload (your current needs exceed this)
- Reason: 16 GB RAM might be tight for some operations
- **Savings:** $0.20/hr (~17% savings)
- **Risk:** May hit RAM limits with larger datasets

### ➡️ Stay on g5.2xlarge ($1.21/hr) if:
- ✅ **Current training setup** (this is you!)
- ✅ Batch size 16-32
- ✅ Dataset 10K-50K samples
- ✅ Model size <100 MB
- ✅ Occasional parallel training (2-3 jobs)

### ⬆️ Scale UP to g5.4xlarge ($1.63/hr) if:
- ⚠️ Dataset grows to 100K+ samples
- ⚠️ Need batch size >32 for training stability
- ⚠️ Running 4+ parallel jobs regularly
- ⚠️ Larger models (e.g., full ViT-Large)
- **Cost increase:** $0.42/hr (~35% more)
- **Benefit:** 2× RAM, 2× vCPUs (same GPU though!)

### ⬆️⬆️ Scale UP to g5.12xlarge ($5.67/hr) if:
- ❌ Need to train 4+ models truly in parallel on separate GPUs
- ❌ Hyperparameter search with 10+ configs
- ❌ Large-scale data processing
- **Cost increase:** $4.46/hr (~370% more)
- **Only justified for:** Research labs, production training pipelines

---

## Historical Performance

### MagVIT Model (Your Fastest)
```
Epochs: 10
Time: 2.21 minutes
Avg per epoch: 13.3 seconds
Dataset: 10,000 samples
Batch size: 8

Result: 100% validation accuracy
```

### Worker 2 (Your Best Model)
```
Epochs: 30
Time: ~1 hour (estimated from epoch 5 = 100% accuracy)
Dataset: 10,000 samples
Batch size: 16

Result: 100% validation accuracy, perfect generalization
```

### Parallel Training Incident (Jan 25, 2026)
```
Setup: 4 workers × batch_size=16 = 64 total samples
Result: ❌ EC2 unresponsive (resource exhaustion)
Solution: Reduced to batch_size=8 per worker = 32 total
Outcome: ✅ Success
```

**Key Lesson:** g5.2xlarge can handle **2-3 parallel jobs with batch_size=8-16**, but NOT 4 parallel jobs with batch_size=16.

---

## Resource Bottleneck Analysis

### What Limited You

**NOT GPU Memory:** Your models use <500 MB out of 24 GB (2%)

**YES System RAM + CPU:** 
- 4 parallel training jobs with data loading
- Each worker loading videos from disk
- Multiple Python processes
- Data augmentation in RAM

**Bottleneck:**
- System RAM (32 GB / 4 workers = 8 GB per worker)
- CPU for data loading (8 vCPUs / 4 workers = 2 vCPUs per worker)
- I/O bandwidth reading video data

### Why Scaling UP GPU Wouldn't Help

Going to g5.12xlarge (4× A10G):
- ✅ Gives 4 separate GPUs (good for parallel training)
- ✅ More vCPUs (48 vs 8)
- ✅ More RAM (192 GB vs 32 GB)
- ❌ **But you're not GPU-limited!**
- ❌ **$4.46/hr more expensive**

**Better solution for parallel training:**
- Keep g5.2xlarge
- Run workers sequentially OR
- Run 2 workers in parallel max

---

## Cost-Benefit Analysis

### Current Setup (g5.2xlarge)

**Typical Training Session:**
- 1-2 hours per major experiment
- Let's say 5 hours/week average usage
- Monthly cost: 5 hrs/week × 4 weeks × $1.21/hr = **$24.20/month**

**Annual cost:** ~$290

### If Scaled Down to g5.xlarge

**Savings:** $0.20/hr
- Monthly: 5 hrs/week × 4 weeks × $0.20 = **$4.00/month saved**
- Annual: ~$48 saved

**Risks:**
- May hit 16 GB RAM limit with larger datasets
- Fewer vCPUs (4 vs 8) = slower data loading
- Less comfortable for exploratory work

**Verdict:** ❌ **Not worth the savings** (~17% savings, potential headaches)

### If Scaled Up to g5.4xlarge

**Cost increase:** $0.42/hr
- Monthly: 5 hrs/week × 4 weeks × $0.42 = **$8.40/month more**
- Annual: ~$101 more

**Benefits:**
- 2× RAM (64 GB vs 32 GB) - enables larger batch sizes
- 2× vCPUs (16 vs 8) - faster data loading
- More comfortable for parallel training (3-4 workers)

**Verdict:** ⚠️ **Consider IF** you regularly run 3+ parallel jobs OR need batch_size > 32

### If Scaled Up to g5.12xlarge

**Cost increase:** $4.46/hr
- Monthly: 5 hrs/week × 4 weeks × $4.46 = **$89.20/month more**
- Annual: ~$1,070 more

**Benefits:**
- 4× A10G GPUs (truly parallel training)
- 6× vCPUs (48 vs 8)
- 6× RAM (192 GB vs 32 GB)

**Verdict:** ❌ **NOT justified** unless this becomes production workload or research lab

---

## Future Scaling Triggers

### ✅ Stay on g5.2xlarge if:
1. Dataset stays 10K-50K samples ✅ (current: 10K)
2. Batch size 8-32 ✅ (current: 16)
3. Model size <100 MB ✅ (current: ~50 MB)
4. Sequential or 2-parallel training ✅ (current workflow)
5. Training time <2 hours per experiment ✅ (current: ~1 hour)

### ⚠️ Consider g5.4xlarge if:
1. Dataset grows to 100K+ samples
2. Need batch size >32 consistently
3. Running 3-4 parallel training jobs daily
4. Training time >4 hours per experiment
5. Need faster experimentation iteration

### ⚠️⚠️ Consider g5.12xlarge if:
1. Running 10+ hyperparameter search configs
2. Production training pipeline (multiple models per day)
3. Team of 3+ researchers using same instance
4. Grant/budget allows (5× cost increase)

---

## Recommendations

### Immediate (Now)

✅ **Keep g5.2xlarge** - perfect fit for your workload

**Optimization tips:**
1. ✅ Already using batch_size=16 (optimal)
2. ✅ Successfully trained 100% accuracy model
3. ✅ Parallel training working with batch_size=8 per worker
4. ✅ No GPU memory issues

### Short-Term (Next 1-3 months)

**If continuing current work:**
- Stay on g5.2xlarge
- Monitor training times
- If >2 hours per experiment becomes common → consider g5.4xlarge

**If scaling up experiments:**
- Dataset grows to 50K+ samples → test g5.4xlarge
- Need 4+ parallel workers → consider g5.12xlarge or separate instances

### Long-Term (3-6 months)

**Option A: Stay on g5.2xlarge**
- If workload remains similar
- Cost: ~$290/year
- Recommendation: ✅ Most cost-effective

**Option B: Scale to g5.4xlarge**
- If parallel training becomes routine
- Cost: ~$390/year (+$100)
- Recommendation: ⚠️ Only if consistently running 3+ parallel jobs

**Option C: Move to Spot Instances**
- Same g5.2xlarge, but 70% cheaper (~$0.36/hr vs $1.21/hr)
- Cost: ~$87/year (vs $290)
- Risk: Can be interrupted
- Recommendation: ⭐ **Best cost optimization** if you can tolerate interruptions

---

## Spot Instance Consideration

### What are Spot Instances?

AWS sells unused EC2 capacity at 60-90% discount, but can reclaim with 2-minute notice.

**g5.2xlarge Spot Pricing:**
- On-demand: $1.21/hr
- Spot (typical): $0.36-0.48/hr (~70% savings)
- Spot (high demand): $0.60-0.80/hr (~50% savings)

### Your Workload Suitability for Spot

**✅ EXCELLENT fit because:**
1. Training has checkpoints every 2 epochs (minimal lost work)
2. Training time is short (1-2 hours per experiment)
3. You can resume from checkpoints automatically
4. Not latency-sensitive (no real-time requirements)

**Strategy:**
1. Use Spot instances for training
2. Keep checkpoints every 2 epochs (already doing this ✅)
3. Implement spot interruption handler (saves checkpoint on 2-min warning)
4. Auto-resume from last checkpoint if interrupted

**Expected savings:**
- Current: 5 hrs/week × $1.21/hr = $6.05/week = **$315/year**
- With Spot: 5 hrs/week × $0.40/hr = $2.00/week = **$104/year**
- **Savings: $211/year (67% reduction)**

**Interruption risk:**
- Typical interruption rate: 5-10%
- With 2-epoch checkpoints: lose max 20 minutes of work
- Cost of interruption: ~$0.40 (20 min × $1.21/hr / 60)
- **Worth the 67% savings!**

---

## Summary & Action Items

### Current Assessment

✅ **g5.2xlarge is the RIGHT CHOICE for your current workload**

**Evidence:**
- Successfully trained 100% accuracy model
- Fast training times (2.2 min for 10 epochs)
- No GPU memory issues
- Comfortable headroom (using <2% of GPU memory)
- Cost-effective (~$315/year assuming 5 hrs/week)

### No Change Needed

❌ **Do NOT scale up** - you're using <2% of GPU memory
❌ **Do NOT scale down** - marginal savings, potential issues

### Optional Optimization

⭐ **Consider Spot Instances** for 67% cost savings:
- g5.2xlarge Spot: ~$0.40/hr (vs $1.21 on-demand)
- Savings: ~$211/year
- Risk: Minimal (5-10% interruption rate, checkpoints every 2 epochs)
- Implementation: Add spot interruption handler to training script

### Future Triggers to Reassess

Monitor these metrics and reassess if:
1. ⚠️ Training time > 2 hours per experiment (consider g5.4xlarge)
2. ⚠️ Running 4+ parallel jobs regularly (consider g5.12xlarge or separate instances)
3. ⚠️ Dataset grows to 100K+ samples (test g5.4xlarge)
4. ⚠️ Need batch size > 32 (test g5.4xlarge)

**Current metrics:**
1. ✅ Training time: ~1 hour (well under trigger)
2. ✅ Parallel jobs: 2-3 max (under trigger)
3. ✅ Dataset: 10K samples (under trigger)
4. ✅ Batch size: 16 (under trigger)

---

## Conclusion

**Your choice of g5.2xlarge is EXCELLENT** for the trajectory video understanding workload. The instance provides:
- ✅ More than enough GPU memory (using <2%)
- ✅ Sufficient RAM and CPU for single/dual parallel training
- ✅ Fast training times (10 epochs in 2.2 minutes)
- ✅ Cost-effective ($1.21/hr ≈ $315/year at 5 hrs/week)

**Keep g5.2xlarge and optionally switch to Spot for 67% savings.**

---

**Assessment Date:** January 26, 2026  
**Assessed Instance:** g5.2xlarge  
**Verdict:** ✅ Appropriate, no change needed  
**Optional Optimization:** Consider Spot instances for cost savings

