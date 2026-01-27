# Parallel Attention Training - Execution Guide

**Status**: ✅ Ready to Execute  
**Date**: 2026-01-26  
**Implementation**: Option A (Worker 1 + Worker 2 in Parallel)

---

## 📋 **Implementation Summary**

### ✅ **Completed**
1. **Worker Setup Scripts**
   - `scripts/setup_parallel_attention_workers.sh` - Creates git worktrees
   - Git branches: `early-persistence/attention-supervised`, `early-persistence/pretrained-features`

2. **Worker 1: Attention-Supervised Loss (85% success probability)**
   - **Tests**: `experiments/trajectory_video_understanding/parallel_workers/worker1_attention/tests/test_attention_supervised.py`
   - **Implementation**: `experiments/trajectory_video_understanding/parallel_workers/worker1_attention/src/attention_supervised_trainer.py`
   - **Features**:
     - Explicit loss term: `-persistent_attention + transient_attention`
     - Alpha=0.2 weighting
     - Early stopping detection
     - Attention ratio computation

3. **Worker 2: Pre-trained ResNet Features (70% success probability)**
   - **Tests**: `experiments/trajectory_video_understanding/parallel_workers/worker2_pretrained/tests/test_pretrained_tokenizer.py`
   - **Implementation**: `experiments/trajectory_video_understanding/parallel_workers/worker2_pretrained/src/pretrained_tokenizer.py`
   - **Features**:
     - Frozen ResNet-18 backbone (512-dim features)
     - Trainable projection to 256-dim
     - ImageNet normalization
     - Batch tokenization

4. **Monitoring Infrastructure**
   - **Script**: `scripts/monitor_parallel_workers.py`
   - **Features**:
     - Checks metrics every 30s
     - Syncs to MacBook every 2 minutes
     - Automatic early stopping
     - Terminates other workers when winner found
     - Generates PARALLEL_PROGRESS.md

5. **Orchestration Scripts**
   - **Master Script**: `scripts/execute_parallel_attention_training.sh`
   - Handles:
     - TDD RED phase (both workers)
     - TDD GREEN phase (both workers)
     - Evidence capture
     - Training launch
     - Monitoring

---

## 🚀 **Execution Steps**

### **Step 1: Prepare EC2 Environment**

```bash
# SSH into EC2
ssh ubuntu@<EC2_IP>

# Clone/update repo
cd ~/
git clone https://github.com/mkn9/mono_to_3d.git
cd mono_to_3d
git checkout early-persistence/magvit
git pull origin early-persistence/magvit

# Verify Python environment
python --version  # Should be 3.8+
pip install -r requirements.txt
```

### **Step 2: Configure MacBook Connection**

Edit `scripts/monitor_parallel_workers.py` and `scripts/execute_parallel_attention_training.sh`:

```python
# Update these variables:
MACBOOK_USER = "mike"
MACBOOK_HOST = "<YOUR_MACBOOK_IP>"  # Get with: ipconfig getifaddr en0
```

Ensure SSH keys are set up:
```bash
# On EC2, test connection
ssh mike@<MACBOOK_IP> "echo 'Connection OK'"
```

### **Step 3: Run Complete Pipeline**

**Option A: Automated Full Pipeline (Recommended)**

```bash
cd ~/mono_to_3d
bash scripts/execute_parallel_attention_training.sh
```

This will:
1. Setup git worktrees
2. Run TDD RED phase (both workers)
3. Run TDD GREEN phase (both workers)
4. Capture evidence
5. Create training script templates

**Option B: Manual Step-by-Step**

```bash
# 1. Setup worktrees
bash scripts/setup_parallel_attention_workers.sh

# 2. Run TDD (both workers)
python -m pytest experiments/trajectory_video_understanding/parallel_workers/worker1_attention/tests/ -v
python -m pytest experiments/trajectory_video_understanding/parallel_workers/worker2_pretrained/tests/ -v

# 3. Continue to Step 4...
```

### **Step 4: Start Parallel Training**

**Terminal 1 (Worker 1 - Attention-Supervised)**:
```bash
cd ~/worker1_attention

# Note: Full training script needs to be completed
# Current implementation has trainer class ready
# Need to add:
# - Dataset loading from persistence_augmented_dataset
# - Model initialization
# - Training loop
# - Metrics saving to latest_metrics.json

python experiments/trajectory_video_understanding/parallel_workers/worker1_attention/train.py
```

**Terminal 2 (Worker 2 - Pre-trained Features)**:
```bash
cd ~/worker2_pretrained

# Note: Full training script needs to be completed
# Current implementation has tokenizer ready
# Need to add:
# - Dataset loading
# - Model initialization with ResNet tokenizer
# - Training loop
# - Metrics saving to latest_metrics.json

python experiments/trajectory_video_understanding/parallel_workers/worker2_pretrained/train.py
```

**Terminal 3 (Monitoring)**:
```bash
cd ~/mono_to_3d
python scripts/monitor_parallel_workers.py
```

---

## 📊 **Monitoring & Progress**

### **On MacBook**

Watch for automatic syncs every 2 minutes:

```bash
# View live progress
watch -n 10 cat /Users/mike/Dropbox/.../parallel_workers/monitoring/PARALLEL_PROGRESS.md
```

Expected format:
```markdown
# Parallel Training Progress

**Last Updated**: 2026-01-26 10:45:32

## Worker Status

| Worker | Status | Epoch | Ratio | Val Acc | Consistency |
|--------|--------|-------|-------|---------|-------------|
| Attention-Supervised | 🟡 Training | 15/50 | 1.32x | 68.2% | 64.5% |
| Pre-trained Features | 🟡 Training | 12/30 | 1.28x | 71.5% | 62.3% |

## Success Criteria
- Attention Ratio: ≥ 1.5x
- Validation Accuracy: ≥ 75%
- Consistency: ≥ 70%
```

### **Success Indicators**

When a worker achieves all criteria:
```markdown
## 🏆 Winner
**Attention-Supervised** achieved success criteria!

Epoch: 28
Ratio: 1.62x ✅
Val Acc: 76.3% ✅
Consistency: 72.1% ✅
```

---

## 🎯 **Success Criteria (Early Stopping)**

Training stops when **any worker** achieves:

| Metric | Threshold | Description |
|--------|-----------|-------------|
| **Attention Ratio** | ≥ 1.5x | Persistent objects receive 50% more attention |
| **Validation Accuracy** | ≥ 75% | Classification performance |
| **Consistency** | ≥ 70% | % of validation samples passing ratio threshold |

Checked **every 5 epochs**.

---

## 📈 **Expected Timeline**

| Phase | Duration | Description |
|-------|----------|-------------|
| Setup | 15 min | Git worktrees, TDD phases |
| Worker 1 Training | 2-4 hours | Until early stop (likely winner) |
| Worker 2 Training | 2-3 hours | Faster with ResNet features |
| Post-processing | 15 min | Generate visualizations |

**Total**: ~3-5 hours to success indicator

---

## 🔍 **TDD Evidence**

All evidence captured in `artifacts/`:
- `tdd_worker1_red.txt` - Worker 1 failing tests (RED)
- `tdd_worker1_green.txt` - Worker 1 passing tests (GREEN)
- `tdd_worker2_red.txt` - Worker 2 failing tests (RED)
- `tdd_worker2_green.txt` - Worker 2 passing tests (GREEN)

---

## ⚠️ **Known Issues & Notes**

1. **Training Scripts Incomplete**
   - Trainer and tokenizer classes are implemented
   - Need to add complete training loops
   - Need dataset loading integration
   - Template provided in `execute_parallel_attention_training.sh`

2. **Dataset Path**
   - Training should use: `experiments/trajectory_video_understanding/persistence_augmented_dataset/`
   - Contains 10K samples with transient objects augmented

3. **MacBook IP Configuration**
   - Must update scripts with actual MacBook IP
   - Ensure SSH keys are configured
   - Test connection before starting

4. **GPU Requirements**
   - Worker 2 (ResNet) requires more GPU memory
   - Recommend p3.2xlarge or better
   - Can run both on single GPU with batch_size=8

---

## 📝 **Next Steps**

### **Immediate**
1. ✅ Setup complete
2. ✅ TDD complete
3. ✅ Monitoring ready
4. ⏳ **Complete training scripts**
5. ⏳ **Execute on EC2**

### **During Training**
1. Monitor PARALLEL_PROGRESS.md on MacBook
2. Watch for early stopping trigger
3. Winner will be announced automatically

### **After Success**
1. Generate success visualizations:
   ```bash
   python scripts/generate_success_visualizations.py --worker <winner_id>
   ```
2. Review attention patterns
3. Analyze what made the winner successful
4. Consider if other workers had promising approaches

---

## 🏆 **Success Visualization**

When criteria met, auto-generate 4-panel visualization:

1. **Attention Ratio Progression**: Line graph showing ratio climbing above 1.5x
2. **Before/After Comparison**: Side-by-side bar charts
3. **Distribution Separation**: Violin plots showing persistent vs transient
4. **Sample Success Grid**: Per-sample validation results

Saved to: `results/<winner>/success_viz/SUCCESS_VISUALIZATION.png`

---

## 🔗 **Related Documents**

- `PARALLEL_ATTENTION_PLAN.md` - Full strategy and rationale
- `DESIGN_DOCUMENT.md` - Original object-level persistence design
- `IMPLEMENTATION_ROADMAP.md` - Overall project roadmap

---

## 📧 **Support**

If monitoring stops or issues arise:
1. Check EC2 instance status
2. Check training logs in worker directories
3. Verify MacBook connection: `ssh mike@<MACBOOK_IP>`
4. Review artifacts/ for TDD evidence

---

**Status**: Ready to execute! Complete training scripts and launch on EC2.

