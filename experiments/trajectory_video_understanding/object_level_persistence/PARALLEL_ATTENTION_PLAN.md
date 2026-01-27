# Parallel Attention Improvement Plan

**Date**: 2026-01-26  
**Goal**: Achieve clear attention differentiation (Persistent/Transient ratio ≥ 1.5x)  
**Strategy**: Test 4 approaches in parallel using git worktrees + early stopping

---

## 🎯 **Success Criteria (Early Stopping Triggers)**

### Primary Objective
**Attention Ratio ≥ 1.5x** (persistent objects get 50% more attention than transient)

### Validation Metrics (all must pass)
1. **Attention Ratio**: ≥ 1.5x on validation set (20+ samples)
2. **Classification Accuracy**: ≥ 75% on validation set
3. **Separation Quality**: Box plots show clear separation (p < 0.01)
4. **Consistency**: ≥ 70% of samples show ratio > 1.3x

### Checkpoint Evaluation
- Check metrics **every 5 epochs**
- **Stop immediately** if all criteria met
- **Continue to 50 epochs max** if not achieved
- **Report failure** if not achieved by epoch 50

---

## 🌳 **Git Worktree Strategy**

### Branch Structure
```
early-persistence/magvit (baseline - current fast-track)
├── early-persistence/attention-supervised (Worker 1)
├── early-persistence/pretrained-features (Worker 2)
├── early-persistence/contrastive-loss (Worker 3)
└── early-persistence/multitask-learning (Worker 4)
```

### Parallel Execution
- Each worker operates in separate worktree
- Independent training on EC2 (different output directories)
- Periodic sync of results to MacBook
- First to achieve success criteria wins
- Others terminate early

---

## 🔬 **Four Parallel Approaches**

### **Worker 1: Attention Supervision Loss** ⭐ (HIGHEST PRIORITY)
**Branch**: `early-persistence/attention-supervised`  
**Hypothesis**: Explicit loss term will force attention differentiation

**Implementation**:
```python
# Additional loss component
persistent_attn = attention_weights[persistent_mask].mean()
transient_attn = attention_weights[transient_mask].mean()

# Encourage high attention on persistent, low on transient
attention_loss = -persistent_attn + transient_attn

# Combined loss
total_loss = classification_loss + 0.2 * attention_loss
```

**Configuration**:
- Dataset: Full 10K samples (8K train, 1K val, 1K test)
- Epochs: 50 max (early stop if criteria met)
- Batch size: 16
- Architecture: Same 2-layer transformer
- Features: Same simple CNN

**Expected Timeline**: 3-4 hours on EC2
**Success Probability**: 85% - most direct approach

---

### **Worker 2: Pre-trained Visual Features** ⭐⭐
**Branch**: `early-persistence/pretrained-features`  
**Hypothesis**: Better features will enable attention to find persistence patterns

**Implementation**:
```python
# Replace simple CNN with pre-trained ResNet-18
from torchvision.models import resnet18

class PretrainedTokenizer:
    def __init__(self):
        # Use ResNet-18 as feature extractor
        resnet = resnet18(pretrained=True)
        # Remove final FC layer, keep features
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_extractor.eval()  # Freeze
        
        # Project to 256-dim
        self.projection = nn.Linear(512, 256)
```

**Configuration**:
- Dataset: Full 10K samples
- Epochs: 30 max (ResNet features learn faster)
- Batch size: 16
- Architecture: Same transformer
- Features: **Frozen ResNet-18** + learned projection

**Expected Timeline**: 2-3 hours on EC2
**Success Probability**: 70% - good features help, but no explicit attention loss

---

### **Worker 3: Contrastive Loss** ⭐
**Branch**: `early-persistence/contrastive-loss`  
**Hypothesis**: Pushing persistent/transient embeddings apart will improve attention

**Implementation**:
```python
# Contrastive loss on embeddings
persistent_embeds = transformer_output[persistent_mask]
transient_embeds = transformer_output[transient_mask]

# Pull same-class together, push different-class apart
contrastive_loss = contrastive_fn(
    persistent_embeds, 
    transient_embeds,
    margin=0.5
)

total_loss = classification_loss + 0.1 * contrastive_loss
```

**Configuration**:
- Dataset: Full 10K samples
- Epochs: 50 max
- Batch size: 16
- Architecture: Same transformer
- Features: Same simple CNN

**Expected Timeline**: 3-4 hours on EC2
**Success Probability**: 60% - indirect approach, may not directly affect attention

---

### **Worker 4: Multi-Task Learning** ⭐⭐
**Branch**: `early-persistence/multitask-learning`  
**Hypothesis**: Predicting persistence duration provides richer signal

**Implementation**:
```python
# Two prediction heads
class MultiTaskTransformer(nn.Module):
    def forward(self, src, mask):
        features = self.transformer(src, mask)
        
        # Task 1: Binary classification (persistent vs transient)
        class_logits = self.classifier(features)
        
        # Task 2: Duration prediction (how many frames object persists)
        duration_pred = self.duration_head(features)
        
        return class_logits, duration_pred

# Multi-task loss
classification_loss = cross_entropy(class_logits, labels)
duration_loss = mse(duration_pred, actual_durations)

total_loss = classification_loss + 0.3 * duration_loss
```

**Configuration**:
- Dataset: Full 10K samples + duration labels
- Epochs: 50 max
- Batch size: 16
- Architecture: Transformer + 2 heads
- Features: Same simple CNN

**Expected Timeline**: 4-5 hours on EC2 (need duration labels)
**Success Probability**: 65% - richer signal but more complex

---

## 📊 **Post-Processing Visualization (SUCCESS INDICATOR)**

### Visualization 1: Attention Ratio Progression
**Purpose**: Show attention differentiation improving over epochs

```
┌─────────────────────────────────────────────┐
│  Attention Ratio vs Epoch                   │
│                                              │
│  2.0┤                              ●──●──●  │← SUCCESS ZONE
│  1.8┤                        ●──●──          │  (ratio ≥ 1.5x)
│  1.6┤                  ●──●──                │
│  1.5┤- - - - - - - - - - - - - - - - - - - -│← THRESHOLD
│  1.4┤            ●──●──                      │
│  1.2┤      ●──●──                            │
│  1.0┤●──●──                                  │← BASELINE
│     └────────────────────────────────────────│
│      0    10   20   30   40   50  Epoch     │
└─────────────────────────────────────────────┘

✅ SUCCESS: Ratio reached 1.8x at epoch 42
   Early stopping triggered!
```

### Visualization 2: Before/After Comparison
**Purpose**: Show dramatic improvement in attention patterns

```
BEFORE (Epoch 1):                    AFTER (Epoch 42):
┌──────────────────────┐            ┌──────────────────────┐
│ Attention per Object │            │ Attention per Object │
│ 1.2┤                 │            │ 1.8┤ ███            │
│ 1.0┤ ███ ██ ███ ██   │            │ 1.6┤ ███            │
│    │ ███ ██ ███ ██   │            │ 1.4┤ ███ ███ ███    │
│ 0.8┤ ███ ██ ███ ██   │            │ 1.2┤ ███ ███ ███    │
│    └─────────────────│            │ 1.0┤ ███ ███ ███    │
│      P   T   P   T   │            │ 0.8┤                │
│                      │            │ 0.6┤     ██  ██  ██  │
│ ⚠️ Ratio: 1.01x      │            │ 0.4┤     ██  ██  ██  │
│    Nearly uniform    │            │    └─────────────────│
└──────────────────────┘            │      P   P   P   T   │
                                    │                      │
                                    │ ✅ Ratio: 1.8x       │
                                    │    Clear separation! │
                                    └──────────────────────┘
```

### Visualization 3: Statistical Separation
**Purpose**: Show distributions are now clearly separated

```
Attention Distribution by Object Type (Final Model)

  Density
    │
    │     Transient              Persistent
    │        ↓                       ↓
 1.0│        ╱╲                    ╱─╲
    │       ╱  ╲                  ╱   ╲
 0.8│      ╱    ╲                ╱     ╲
    │     ╱      ╲              ╱       ╲
 0.6│    ╱        ╲            ╱         ╲
    │   ╱          ╲          ╱           ╲
 0.4│  ╱            ╲        ╱             ╲
    │ ╱              ╲      ╱               ╲
 0.2│╱                ╲____╱                 ╲___
    └────────────────────────────────────────────
    0.4  0.6  0.8  1.0  1.2  1.4  1.6  1.8  2.0
                 Attention Weight

✅ Distributions clearly separated (p < 0.001)
✅ Minimal overlap
✅ Transient: mean=0.65, std=0.12
✅ Persistent: mean=1.42, std=0.18
✅ Cohen's d = 2.8 (huge effect size)
```

### Visualization 4: Sample-Level Success Grid
**Purpose**: Show consistency across validation samples

```
Validation Samples (20 total)

Sample  Accuracy  Ratio  Status
────────────────────────────────────
08000   82.3%     1.76x  ✅ PASS
08001   78.9%     1.62x  ✅ PASS
08002   91.2%     1.84x  ✅ PASS
08003   75.4%     1.52x  ✅ PASS
08004   88.1%     1.69x  ✅ PASS
08005   79.3%     1.58x  ✅ PASS
08006   82.7%     1.73x  ✅ PASS
08007   76.8%     1.51x  ✅ PASS
...
08018   83.2%     1.66x  ✅ PASS
08019   77.5%     1.54x  ✅ PASS
────────────────────────────────────
Overall: 18/20 pass (90%)
Avg Accuracy: 81.2%
Avg Ratio: 1.67x

✅ SUCCESS CRITERIA MET
```

---

## 🔄 **Execution Pipeline**

### Phase 1: Setup (Parallel) - 10 minutes
```bash
# On EC2, create 4 worktrees
cd ~/mono_to_3d
git fetch origin

# Worker 1: Attention Supervised
git worktree add ~/worker1_attention early-persistence/attention-supervised
cd ~/worker1_attention
# Copy and modify training script

# Worker 2: Pre-trained Features
git worktree add ~/worker2_pretrained early-persistence/pretrained-features
cd ~/worker2_pretrained
# Implement ResNet tokenizer

# Worker 3: Contrastive Loss
git worktree add ~/worker3_contrastive early-persistence/contrastive-loss
cd ~/worker3_contrastive
# Implement contrastive loss

# Worker 4: Multi-task
git worktree add ~/worker4_multitask early-persistence/multitask-learning
cd ~/worker4_multitask
# Implement multi-task heads
```

### Phase 2: Training (Parallel) - 2-5 hours
```bash
# Each worker runs independently
# Output to separate directories
cd ~/worker1_attention && python train_supervised.py \
  --output results/attention_supervised \
  --early-stop-ratio 1.5 \
  --check-every 5 > training.log 2>&1 &

cd ~/worker2_pretrained && python train_pretrained.py \
  --output results/pretrained_features \
  --early-stop-ratio 1.5 \
  --check-every 5 > training.log 2>&1 &

cd ~/worker3_contrastive && python train_contrastive.py \
  --output results/contrastive_loss \
  --early-stop-ratio 1.5 \
  --check-every 5 > training.log 2>&1 &

cd ~/worker4_multitask && python train_multitask.py \
  --output results/multitask \
  --early-stop-ratio 1.5 \
  --check-every 5 > training.log 2>&1 &

# Master monitor script checks all workers
python scripts/monitor_parallel_training.py
```

### Phase 3: Post-Processing (Winner Only) - 15 minutes
```bash
# Automatically triggered when worker achieves success
# Generate 4 key visualizations
python scripts/generate_success_visualizations.py \
  --checkpoint results/${winner}/best_model.pt \
  --output results/${winner}/success_viz/

# Visualizations:
# 1. attention_ratio_progression.png
# 2. before_after_comparison.png
# 3. distribution_separation.png
# 4. sample_level_grid.png
```

---

## 📈 **Monitoring & Heartbeat**

### Real-Time Monitoring Script
```python
# scripts/monitor_parallel_training.py
# Runs on EC2, syncs to MacBook every 2 minutes

workers = [
    {'name': 'attention-supervised', 'dir': 'worker1_attention'},
    {'name': 'pretrained-features', 'dir': 'worker2_pretrained'},
    {'name': 'contrastive-loss', 'dir': 'worker3_contrastive'},
    {'name': 'multitask-learning', 'dir': 'worker4_multitask'},
]

while True:
    for worker in workers:
        # Read latest metrics
        metrics = read_latest_checkpoint(worker)
        
        # Check success criteria
        if (metrics['attention_ratio'] >= 1.5 and
            metrics['val_accuracy'] >= 0.75 and
            metrics['consistency'] >= 0.70):
            
            print(f"🎉 WINNER: {worker['name']} at epoch {metrics['epoch']}")
            print(f"   Attention Ratio: {metrics['attention_ratio']:.2f}x")
            print(f"   Validation Acc: {metrics['val_accuracy']:.1%}")
            
            # Terminate other workers
            terminate_other_workers(worker)
            
            # Trigger post-processing
            generate_success_visualizations(worker)
            
            # Sync to MacBook
            sync_to_macbook(worker)
            
            return worker
    
    # Sync progress to MacBook
    sync_progress_summary()
    time.sleep(120)  # Check every 2 minutes
```

### MacBook Progress Dashboard
File synced to: `results/PARALLEL_PROGRESS.md`

```markdown
# Parallel Training Progress

**Last Updated**: 2026-01-26 10:45:32

## Worker Status

| Worker | Epoch | Ratio | Val Acc | Status |
|--------|-------|-------|---------|--------|
| Attention-Supervised | 15/50 | 1.32x | 68.2% | 🟡 Training |
| Pre-trained Features | 12/30 | 1.28x | 71.5% | 🟡 Training |
| Contrastive Loss | 18/50 | 1.19x | 64.8% | 🟡 Training |
| Multi-task Learning | 8/50 | 1.15x | 62.3% | 🟡 Training |

## Best So Far
**Pre-trained Features** at Epoch 12:
- Ratio: 1.28x (need 1.5x)
- Val Acc: 71.5% (need 75%)
- Trending: 📈 Improving

**ETA to Success**: ~45 minutes (if current trend continues)
```

---

## 🎯 **Success Visualization Script**

### Automatic Generation When Criteria Met
```python
# scripts/generate_success_visualizations.py

def generate_success_visualizations(checkpoint_path, output_dir):
    """Generate 4-panel success visualization."""
    
    # Load model and validation data
    model = load_model(checkpoint_path)
    val_data = load_validation_set()
    
    # Extract all attention patterns
    results = evaluate_with_attention(model, val_data)
    
    # Create 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel 1: Attention ratio progression over epochs
    plot_ratio_progression(axes[0, 0], results['history'])
    
    # Panel 2: Before/After bar chart comparison
    plot_before_after_comparison(axes[0, 1], results)
    
    # Panel 3: Distribution separation (violin/density plots)
    plot_distribution_separation(axes[1, 0], results)
    
    # Panel 4: Sample-level success grid
    plot_sample_grid(axes[1, 1], results)
    
    # Add big SUCCESS banner
    fig.suptitle('🎉 OBJECTIVE ACHIEVED 🎉', 
                 fontsize=24, fontweight='bold', color='green')
    
    # Save
    plt.tight_layout()
    plt.savefig(f'{output_dir}/SUCCESS_VISUALIZATION.png', dpi=150)
    
    print(f"✅ Success visualization saved to {output_dir}")
    return True
```

---

## 📋 **Implementation Checklist**

### TDD Requirements (Per Worker)
- [ ] Write tests for new loss components
- [ ] Run TDD RED phase (capture failures)
- [ ] Implement components
- [ ] Run TDD GREEN phase (capture passes)
- [ ] Commit evidence to artifacts/

### Parallel Execution
- [ ] Create 4 git worktrees on EC2
- [ ] Implement each approach in its branch
- [ ] Start all 4 training jobs
- [ ] Run monitoring script
- [ ] Sync progress to MacBook every 2 min

### Early Stopping
- [ ] Check metrics every 5 epochs
- [ ] Terminate others when winner found
- [ ] Generate success visualizations
- [ ] Commit results and merge winner branch

### Documentation
- [ ] Each worker logs to separate file
- [ ] Heartbeat updates PARALLEL_PROGRESS.md
- [ ] Winner generates SUCCESS_REPORT.md
- [ ] All results synced to MacBook

---

## 🏆 **Expected Outcome**

### Optimistic Scenario (2-3 hours)
- **Worker 1** (attention-supervised) achieves 1.6x ratio at epoch 25
- Other workers terminated
- Success visualizations generated
- Clear proof that objective is met

### Realistic Scenario (3-4 hours)
- **Worker 2** (pre-trained features) achieves 1.5x ratio at epoch 28
- **Worker 1** close behind at 1.48x
- Winner visualizations show clear separation
- Objective achieved with good features

### Pessimistic Scenario (5+ hours)
- All workers reach epoch 50
- Best ratio: 1.42x (close but not quite)
- Need to combine approaches (supervised + pretrained)
- Run Worker 5: hybrid approach

---

## 📊 **Resource Requirements**

### EC2 Instance
- **Type**: p3.2xlarge (1 GPU) or p3.8xlarge (4 GPUs)
- **Duration**: 3-5 hours
- **Cost**: ~$3-15 depending on instance
- **Storage**: 50GB (for all 4 workers + data)

### MacBook
- **Synced Files**: Progress updates (~1KB every 2 min)
- **Final Results**: ~500MB (models + visualizations)
- **No training**: Just monitoring

---

## ✅ **Ready to Execute?**

This plan provides:
1. ✅ **Parallel approach** using git worktrees
2. ✅ **Early stopping** when ratio ≥ 1.5x achieved
3. ✅ **Clear visualizations** showing success
4. ✅ **Post-processing** automatically triggered
5. ✅ **Real-time monitoring** synced to MacBook
6. ✅ **Resource efficient** - stops early workers

**Recommend**: Start with **Worker 1 (attention-supervised)** and **Worker 2 (pre-trained features)** as highest probability approaches. Add Workers 3-4 if needed.

