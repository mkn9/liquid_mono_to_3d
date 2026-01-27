# Parallel TDD Implementation Plan
## Trajectory Video Understanding - 5 Branch Development

**Date**: 2026-01-25  
**Status**: READY TO EXECUTE  
**Approach**: PARALLEL + TDD + MONITORED

---

## 🚨 **CRITICAL REQUIREMENTS** 🚨

### ✅ **TDD Workflow (MANDATORY)**
- Write tests FIRST (RED phase)
- Run tests, confirm failures
- Implement code (GREEN phase)
- Run tests, confirm passes
- Capture evidence: `artifacts/tdd_*.txt`

### ✅ **Parallel Execution (MANDATORY)**
- 5 workers simultaneously, NOT sequential
- Each worker on separate git branch
- Independent progress tracking
- No blocking dependencies

### ✅ **Monitoring (MANDATORY)**
- Use `run_with_keepalive.sh` for all long operations
- Heartbeat every 30 seconds
- Progress visible on MacBook
- Auto-cleanup when complete

### ✅ **Checkpointing (MANDATORY)**
- Training: Save checkpoint every 2 epochs
- Dataset: Save checkpoint every 2K samples
- Resume capability if interrupted
- PROGRESS.txt updated every minute

### ✅ **Validation First (MANDATORY)**
- 10 epochs quick validation per branch
- Verify approach works before full training
- Only proceed to 40 more epochs if validation successful

---

## 🌳 **Parallel Branch Structure**

```
master
│
├── trajectory-video/branch-1-i3d          [Worker 1 - EC2 Terminal 1]
├── trajectory-video/branch-2-slowfast     [Worker 2 - EC2 Terminal 2]
├── trajectory-video/branch-3-transformer  [Worker 3 - EC2 Terminal 3]
├── trajectory-video/branch-4-magvit       [Worker 4 - EC2 Terminal 4]
└── trajectory-video/branch-5-data-10k     [Worker 5 - EC2 Terminal 5]
```

---

## 📋 **Phase 1: Setup (Day 1 - Parallel)**

### **Worker 0: Infrastructure (Local MacBook - YOU)**

#### **Step 1.1: Create Shared Infrastructure** [30 min]

**TDD Workflow:**

**RED Phase - Write Tests First:**
```bash
cd experiments/trajectory_video_understanding
mkdir -p tests/shared

# Create test_base_extractor.py (TESTS FIRST!)
cat > tests/shared/test_base_extractor.py << 'EOF'
"""Tests for abstract base feature extractor."""
import pytest
import torch
from shared.base_extractor import FeatureExtractor

def test_feature_extractor_is_abstract():
    """Cannot instantiate abstract base class."""
    with pytest.raises(TypeError):
        FeatureExtractor()

def test_feature_extractor_interface_requires_extract():
    """Subclass must implement extract method."""
    class BadExtractor(FeatureExtractor):
        @property
        def feature_dim(self):
            return 256
    
    with pytest.raises(TypeError):
        BadExtractor()

def test_feature_extractor_output_shape():
    """Extract method must return (B, T, D) tensor."""
    class DummyExtractor(FeatureExtractor):
        @property
        def feature_dim(self):
            return 256
        
        def extract(self, video):
            B, T = video.shape[0], video.shape[1]
            return torch.randn(B, T, self.feature_dim)
    
    extractor = DummyExtractor()
    video = torch.randn(2, 16, 3, 64, 64)
    features = extractor.extract(video)
    
    assert features.shape == (2, 16, 256)
    assert torch.all(torch.isfinite(features))
EOF

# Run tests - EXPECT FAILURES (RED)
pytest tests/shared/test_base_extractor.py -v
# CAPTURE: Should fail (base_extractor.py doesn't exist yet)
```

**GREEN Phase - Minimal Implementation:**
```bash
# Now create the actual code
mkdir -p shared
cat > shared/base_extractor.py << 'EOF'
"""Abstract base class for feature extractors."""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class FeatureExtractor(ABC):
    """Base class for all video feature extractors."""
    
    @abstractmethod
    def extract(self, video: torch.Tensor) -> torch.Tensor:
        """
        Extract features from video.
        
        Args:
            video: (B, T, C, H, W) tensor
            
        Returns:
            features: (B, T, D) tensor where D = feature_dim
        """
        pass
    
    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Return feature dimension."""
        pass
EOF

# Run tests again - EXPECT PASSES (GREEN)
pytest tests/shared/test_base_extractor.py -v
# CAPTURE EVIDENCE
```

**Evidence Capture:**
```bash
pytest tests/shared/test_base_extractor.py -v 2>&1 | tee artifacts/tdd_base_extractor_green.txt
```

---

#### **Step 1.2: Create Unified Model** [45 min]

**TDD - Write Tests First:**
```python
# tests/shared/test_unified_model.py

def test_unified_model_two_task_heads():
    """Model must have classification and prediction heads."""
    from shared.unified_model import UnifiedModel
    from shared.base_extractor import FeatureExtractor
    
    class DummyExtractor(FeatureExtractor):
        @property
        def feature_dim(self):
            return 256
        
        def extract(self, video):
            return torch.randn(video.shape[0], video.shape[1], 256)
    
    model = UnifiedModel(DummyExtractor())
    video = torch.randn(2, 16, 3, 64, 64)
    
    output = model(video)
    
    # Task 1: Classification (4 classes)
    assert 'classification' in output
    assert output['classification'].shape == (2, 4)
    
    # Task 2: Prediction (x, y, z)
    assert 'prediction' in output
    assert output['prediction'].shape == (2, 3)

def test_unified_model_loss_computation():
    """Test multi-task loss computation."""
    from shared.unified_model import compute_loss
    
    outputs = {
        'classification': torch.randn(2, 4),
        'prediction': torch.randn(2, 3)
    }
    targets = {
        'class_label': torch.tensor([0, 2]),
        'future_position': torch.randn(2, 3)
    }
    
    total_loss, class_loss, pred_loss = compute_loss(outputs, targets)
    
    assert torch.isfinite(total_loss)
    assert torch.isfinite(class_loss)
    assert torch.isfinite(pred_loss)
    assert total_loss > 0
```

**Run TDD cycle, capture evidence.**

---

### **Worker 5: Data Generation (EC2 - Starts Immediately)**

#### **Step 1.3: Generate 10K Dataset** [2-3 hours]

**TDD First:**
```bash
# SSH to EC2
ssh -i /Users/mike/keys/AutoGenKeyPair.pem ubuntu@34.196.155.11

# Create branch
cd ~/mono_to_3d
git checkout -b trajectory-video/branch-5-data-10k

# Write tests FIRST
cd experiments/trajectory_video_understanding/branch_5_data_10k

# Create test_dataset_generation.py
# Tests:
# - test_dataset_shape_correct()
# - test_class_balance()
# - test_trajectory_validity()
# - test_video_quality()

# Run tests (RED - will fail, no data yet)
pytest tests/ -v 2>&1 | tee artifacts/tdd_data_gen_red.txt
```

**Generate with Monitoring:**
```bash
# Use auto-cleanup keep-alive wrapper
./scripts/run_with_keepalive.sh "python generate_10k_dataset.py \
    --output data/10k_trajectories \
    --checkpoint-interval 2000 \
    --workers 4"

# While running, monitor in separate terminal:
tail -f /tmp/ai_keepalive_*.log

# Or use progress tracker:
./scripts/ai_progress_tracker.sh
```

**Dataset Generation Config:**
```yaml
# config_data_generation.yaml
num_samples: 10000
checkpoint_interval: 2000  # Every 2K samples
workers: 4  # Parallel generation
trajectory_classes:
  - linear
  - circular
  - helical
  - parabolic
video_params:
  frames: 32
  height: 64
  width: 64
  fps: 30
auto_camera: true
augmentation:
  noise_level: 0.02
  rotation: true
  scale: true
```

---

## 📋 **Phase 2: Validation Round (Day 2-4 - Parallel)**

### **10 Epochs Per Branch - Quick Validation**

#### **Worker 1: Branch 1 - I3D** [Day 2, EC2 Terminal 1]

**TDD Workflow:**

```bash
# EC2 Terminal 1
ssh -i /Users/mike/keys/AutoGenKeyPair.pem ubuntu@34.196.155.11
cd ~/mono_to_3d
git checkout -b trajectory-video/branch-1-i3d

# RED Phase: Write tests
cd experiments/trajectory_video_understanding/branch_1_i3d

cat > tests/test_i3d_extractor.py << 'EOF'
def test_i3d_extractor_initialization():
    """Test I3D extractor can be created."""
    from feature_extractor import I3DExtractor
    extractor = I3DExtractor(pretrained=True)
    assert extractor.feature_dim == 1024

def test_i3d_extractor_output_shape():
    """Test I3D produces correct output shape."""
    from feature_extractor import I3DExtractor
    extractor = I3DExtractor(pretrained=False)
    video = torch.randn(2, 16, 3, 64, 64)
    features = extractor.extract(video)
    assert features.shape == (2, 16, 1024)
EOF

# Run tests (RED)
pytest tests/ -v 2>&1 | tee artifacts/tdd_i3d_red.txt

# GREEN Phase: Implement
# ... create feature_extractor.py ...

# Run tests (GREEN)
pytest tests/ -v 2>&1 | tee artifacts/tdd_i3d_green.txt

# Train with monitoring (10 epochs validation)
./scripts/run_with_keepalive.sh "python train.py \
    --config config_validation.yaml \
    --epochs 10 \
    --checkpoint-interval 2 \
    --output results/validation"
```

**Training Config (Validation):**
```yaml
# config_validation.yaml
epochs: 10  # Quick validation
batch_size: 32
learning_rate: 0.001
checkpoint_interval: 2  # Save every 2 epochs
early_stopping: true
early_stopping_patience: 5

tasks:
  classification:
    enabled: true
    weight: 1.0
  prediction:
    enabled: true
    weight: 1.0
    predict_future: 5  # t+5
```

---

#### **Worker 2: Branch 2 - Slow/Fast** [Day 2, EC2 Terminal 2]

```bash
# EC2 Terminal 2 (parallel to Worker 1)
ssh -i /Users/mike/keys/AutoGenKeyPair.pem ubuntu@34.196.155.11
cd ~/mono_to_3d
git checkout -b trajectory-video/branch-2-slowfast

# Same TDD workflow:
# 1. Write tests (RED)
# 2. Implement extractor (GREEN)
# 3. Train 10 epochs with keep-alive
# 4. Capture evidence
```

---

#### **Worker 3: Branch 3 - Transformer** [Day 3, EC2 Terminal 3]

```bash
# EC2 Terminal 3 (parallel to Workers 1 & 2)
ssh -i /Users/mike/keys/AutoGenKeyPair.pem ubuntu@34.196.155.11
cd ~/mono_to_3d
git checkout -b trajectory-video/branch-3-transformer

# Same TDD workflow
```

---

#### **Worker 4: Branch 4 - MagVIT** [Day 3, EC2 Terminal 4]

```bash
# EC2 Terminal 4 (parallel to Workers 1, 2, 3)
ssh -i /Users/mike/keys/AutoGenKeyPair.pem ubuntu@34.196.155.11
cd ~/mono_to_3d
git checkout -b trajectory-video/branch-4-magvit

# Leverage existing MagVIT work from Jan 24-25
# Same TDD workflow
```

---

## 📊 **Validation Checkpoint (Day 5)**

### **Evaluate 10-Epoch Results**

After all 4 branches complete 10 epochs, evaluate:

```python
# evaluate_validation.py

branches = ['branch-1-i3d', 'branch-2-slowfast', 
            'branch-3-transformer', 'branch-4-magvit']

results = {}
for branch in branches:
    checkpoint = f"experiments/trajectory_video_understanding/{branch}/results/validation/checkpoint_epoch_10.pt"
    
    # Load model
    model = load_model(checkpoint)
    
    # Evaluate
    results[branch] = {
        'classification_acc': evaluate_classification(model, test_loader),
        'prediction_mse': evaluate_prediction(model, test_loader),
        'training_time': get_training_time(branch),
        'model_size': get_model_size(checkpoint)
    }

# Generate comparison report
generate_report(results, "results/validation_comparison.md")
```

**Decision Criteria:**

| Metric | Threshold | Action if Below |
|--------|-----------|-----------------|
| Classification Acc | >70% | Investigate, may need tuning |
| Prediction MSE | <0.1 | Investigate, may need tuning |
| Training Time | <2 hours | Acceptable |
| Convergence | Improving | Continue to 40 epochs |

**If validation successful → Proceed to Phase 3**  
**If issues found → Debug, retune, revalidate**

---

## 📋 **Phase 3: Full Training (Day 6-10 - Parallel)**

### **40 More Epochs Per Branch**

Only start Phase 3 after validation success!

```bash
# Update config for full training
# config_full_training.yaml
epochs: 40  # Additional 40 epochs (50 total)
batch_size: 32
learning_rate: 0.0005  # Slightly lower for fine-tuning
checkpoint_interval: 5  # Save every 5 epochs
resume_from: "results/validation/checkpoint_epoch_10.pt"
```

#### **All Workers Run in Parallel:**

```bash
# Worker 1 - Terminal 1
./scripts/run_with_keepalive.sh "python train.py \
    --config config_full_training.yaml \
    --resume results/validation/checkpoint_epoch_10.pt"

# Worker 2 - Terminal 2  
./scripts/run_with_keepalive.sh "python train.py \
    --config config_full_training.yaml \
    --resume results/validation/checkpoint_epoch_10.pt"

# Worker 3 - Terminal 3
./scripts/run_with_keepalive.sh "python train.py \
    --config config_full_training.yaml \
    --resume results/validation/checkpoint_epoch_10.pt"

# Worker 4 - Terminal 4
./scripts/run_with_keepalive.sh "python train.py \
    --config config_full_training.yaml \
    --resume results/validation/checkpoint_epoch_10.pt"
```

**Monitor All Workers:**
```bash
# MacBook - Monitor all training
./scripts/ai_progress_tracker.sh

# Check progress files (synced from EC2)
watch -n 30 'cat experiments/trajectory_video_understanding/*/results/PROGRESS.txt'
```

---

## 📊 **Checkpoint Strategy**

### **Training Checkpoints**

Every branch must save:

```python
# In train.py

def save_checkpoint(epoch, model, optimizer, metrics, path):
    """Save training checkpoint with all state."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, path)
    print(f"✅ Checkpoint saved: {path}")

# Save every 2 epochs during validation
# Save every 5 epochs during full training
if (epoch + 1) % checkpoint_interval == 0:
    checkpoint_path = f"results/checkpoint_epoch_{epoch+1}.pt"
    save_checkpoint(epoch, model, optimizer, metrics, checkpoint_path)
```

### **Progress Tracking**

Update PROGRESS.txt every epoch:

```python
# In train.py

def update_progress_file(epoch, total_epochs, metrics, output_dir):
    """Update progress file for MacBook visibility."""
    progress_path = Path(output_dir) / "PROGRESS.txt"
    
    content = f"""Training Progress - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
========================================

Branch: {get_branch_name()}
Epoch: {epoch + 1}/{total_epochs} ({100 * (epoch + 1) / total_epochs:.1f}%)

Metrics:
  Classification Acc: {metrics['class_acc']:.2%}
  Prediction MSE: {metrics['pred_mse']:.4f}
  Total Loss: {metrics['total_loss']:.4f}

Training Time: {metrics['elapsed']:.1f} minutes
ETA: {metrics['eta']:.1f} minutes

Last Update: {datetime.now().isoformat()}
"""
    
    progress_path.write_text(content)
    print(f"📊 Progress updated: {progress_path}")
```

---

## 🔍 **Monitoring Dashboard**

### **Multi-Branch Monitoring Script**

```bash
# scripts/monitor_all_branches.sh

#!/bin/bash

while true; do
    clear
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║     TRAJECTORY VIDEO UNDERSTANDING - ALL BRANCHES          ║"
    echo "║              $(date '+%Y-%m-%d %H:%M:%S')                      ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    
    for branch in branch-1-i3d branch-2-slowfast branch-3-transformer branch-4-magvit; do
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📊 $branch"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        progress_file="experiments/trajectory_video_understanding/$branch/results/PROGRESS.txt"
        
        if [ -f "$progress_file" ]; then
            tail -10 "$progress_file"
        else
            echo "⚪ Not started yet"
        fi
        
        echo ""
    done
    
    echo "════════════════════════════════════════════════════════════"
    echo "Refresh in 30 seconds... (Ctrl+C to stop)"
    sleep 30
done
```

---

## ✅ **TDD Evidence Requirements**

### **Per Branch, Must Capture:**

```
artifacts/
├── tdd_<component>_red.txt      # Tests fail before implementation
├── tdd_<component>_green.txt    # Tests pass after implementation
├── tdd_<component>_refactor.txt # Tests still pass after cleanup
└── test_coverage_report.html    # >80% coverage required
```

### **Training Evidence:**

```
results/
├── PROGRESS.txt                  # Updated every epoch
├── training_curves.png           # Loss/accuracy over time
├── checkpoint_epoch_X.pt         # Every 2 or 5 epochs
├── validation_metrics.json       # 10-epoch validation results
└── final_metrics.json            # 50-epoch final results
```

---

## 📅 **Detailed Timeline with Monitoring**

### **Day 1: Setup (Parallel)**
```
09:00 - Worker 0: Create shared infrastructure (TDD)
        └─ Keep-alive: run_with_keepalive.sh "pytest shared/tests/"
        
09:30 - Worker 5: Start 10K generation (EC2 background)
        └─ Keep-alive: run_on_ec2_with_keepalive.sh "python generate_10k_dataset.py"
        
10:00 - Workers 1-4: Create branch infrastructure
        └─ Each worker: git checkout -b, mkdir, setup tests
        
12:00 - Worker 5: Validate generated dataset
        └─ pytest data/tests/ -v
        
End of Day: 10K dataset ready, all branches set up
```

### **Day 2-4: Validation Training (Parallel)**
```
Day 2:
09:00 - Worker 1: Branch 1 (I3D) - 10 epochs
        └─ run_with_keepalive.sh "python train.py --epochs 10"
        
09:00 - Worker 2: Branch 2 (Slow/Fast) - 10 epochs
        └─ run_with_keepalive.sh "python train.py --epochs 10"
        
Day 3:
09:00 - Worker 3: Branch 3 (Transformer) - 10 epochs
09:00 - Worker 4: Branch 4 (MagVIT) - 10 epochs

All running in parallel with heartbeat monitoring!
```

### **Day 5: Validation Checkpoint**
```
09:00 - Evaluate all 10-epoch results
10:00 - Compare metrics across branches
11:00 - Decision: Proceed to full training?
12:00 - Document validation findings
```

### **Day 6-10: Full Training (Parallel)**
```
All workers resume training for 40 more epochs
Each with keep-alive monitoring
Each saving checkpoints every 5 epochs
MacBook dashboard showing all progress
```

---

## 🎯 **Success Criteria Checklist**

### **Phase 1: Setup**
- [ ] Abstract base class created and tested
- [ ] Unified model created and tested
- [ ] 10K dataset generated and validated
- [ ] All 5 branches created
- [ ] All TDD evidence captured

### **Phase 2: Validation (10 Epochs)**
- [ ] Branch 1 (I3D): >70% classification, <0.1 prediction MSE
- [ ] Branch 2 (Slow/Fast): >70% classification, <0.1 prediction MSE
- [ ] Branch 3 (Transformer): >70% classification, <0.1 prediction MSE
- [ ] Branch 4 (MagVIT): >70% classification, <0.1 prediction MSE
- [ ] All checkpoints saved
- [ ] All progress tracked

### **Phase 3: Full Training (50 Epochs Total)**
- [ ] Branch 1: >90% classification, <0.05 prediction MSE
- [ ] Branch 2: >90% classification, <0.05 prediction MSE
- [ ] Branch 3: >90% classification, <0.05 prediction MSE
- [ ] Branch 4: >90% classification, <0.05 prediction MSE
- [ ] All models saved
- [ ] Comparison report generated

---

## 🚨 **Critical Reminders**

### **1. TDD Always First**
```bash
# WRONG:
./scripts/run_with_keepalive.sh "python implement_feature.py"

# RIGHT:
# 1. Write tests
pytest tests/test_new_feature.py  # RED
# 2. Implement
pytest tests/test_new_feature.py  # GREEN
# 3. Refactor
pytest tests/test_new_feature.py  # Still GREEN
# 4. Capture evidence
pytest tests/ -v 2>&1 | tee artifacts/tdd_evidence.txt
```

### **2. Always Use Keep-Alive for Long Operations**
```bash
# Any operation >3 minutes:
./scripts/run_with_keepalive.sh "<command>"
./scripts/run_on_ec2_with_keepalive.sh "<ec2_command>"
```

### **3. Parallel, Not Sequential**
```bash
# WRONG (sequential):
train branch 1 → finish → train branch 2 → finish → ...

# RIGHT (parallel):
train branch 1 &
train branch 2 &
train branch 3 &
train branch 4 &
# All running simultaneously
```

### **4. Validation Before Full Training**
```bash
# MUST complete 10-epoch validation
# MUST evaluate results
# ONLY THEN proceed to 40 more epochs
```

---

## ✅ **Ready to Execute**

This plan ensures:
- ✅ TDD compliance (tests first, evidence captured)
- ✅ Parallel execution (5 workers, no blocking)
- ✅ Monitoring (keep-alive, progress files)
- ✅ Checkpointing (every 2-5 epochs, resume capable)
- ✅ Validation first (10 epochs before full 50)

**Status**: READY TO START  
**First Action**: Worker 0 creates shared infrastructure with TDD  
**First Command**: See Day 1, 09:00 above

---

**Shall we begin?** 🚀

