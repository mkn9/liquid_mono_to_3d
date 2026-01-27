# Object-Level Persistence Detection - Parallel Implementation Plan

**Date**: 2026-01-26  
**Approach**: 2 Parallel Workers using Git Worktrees  
**All Standard Procedures**: ✅ TDD, ✅ Periodic Saves, ✅ Heartbeat, ✅ EC2 Execution

---

## 🌳 **Git Worktree Structure**

```
~/mono_to_3d/                           # Main repo
├── .git/
└── experiments/...

~/mono_to_3d_worker1/                  # Worktree 1
├── branch: object-level/detection-tracking
└── Task: Object Detection + Tracking

~/mono_to_3d_worker2/                  # Worktree 2
├── branch: object-level/transformer
└── Task: Object-Aware Transformer Architecture
```

---

## 👥 **Worker Division**

### **Worker 1: Detection + Tracking Pipeline**
**Branch**: `object-level/detection-tracking`  
**Duration**: 4-5 days  
**Dependencies**: None (can start immediately)

**Tasks**:
1. Object Detection (TDD)
   - Sphere detector architecture
   - Training on existing dataset
   - Validation: >95% detection rate
   
2. Object Tracking (TDD)
   - IoU-based tracker implementation
   - Track ID assignment
   - Validation: >90% track purity
   
3. Integration
   - Detection → Tracking pipeline
   - Output: Track objects with IDs

**Deliverables**:
- `object_detector.py`
- `object_tracker.py`
- `detection_tracking_pipeline.py`
- Tests: `test_object_detector.py`, `test_object_tracker.py`
- Evidence: `artifacts/tdd_detection_*.txt`, `artifacts/tdd_tracking_*.txt`

---

### **Worker 2: Object-Aware Transformer**
**Branch**: `object-level/transformer`  
**Duration**: 4-5 days  
**Dependencies**: Mock detector/tracker for development

**Tasks**:
1. Object Token Architecture (TDD)
   - Convert objects to token sequence
   - Positional encoding (frame + object index)
   - Track ID embedding
   
2. Transformer Adaptation (TDD)
   - Modify MagVIT to accept object tokens
   - Extract per-object attention weights
   - Per-object classification head
   
3. Training Loop (TDD)
   - Loss function design
   - Training script
   - Validation metrics

**Deliverables**:
- `object_tokenizer.py`
- `object_aware_transformer.py`
- `train_object_transformer.py`
- Tests: `test_object_tokenizer.py`, `test_transformer.py`
- Evidence: `artifacts/tdd_transformer_*.txt`

---

## 🔄 **Integration Phase** (After Workers Complete)

**Branch**: `object-level/integration`  
**Duration**: 2-3 days  

**Tasks**:
1. Merge worker branches
2. Connect detection → tracking → transformer
3. End-to-end testing
4. Generate visualizations (attention heatmaps)
5. Comprehensive evaluation

---

## 📋 **Standard Procedures (Both Workers)**

### ✅ **1. TDD Process** (Mandatory)

**RED Phase**:
```bash
# Write tests first
cd ~/mono_to_3d_worker{1|2}
# Create test file
# Run: bash scripts/tdd_capture.sh red
# Verify: artifacts/tdd_*_red.txt shows FAILURES
```

**GREEN Phase**:
```bash
# Implement code
# Run: bash scripts/tdd_capture.sh green
# Verify: artifacts/tdd_*_green.txt shows PASSES
```

**REFACTOR Phase**:
```bash
# Refactor if needed
# Run: bash scripts/tdd_capture.sh refactor
# Verify: artifacts/tdd_*_refactor.txt shows PASSES
```

---

### ✅ **2. Periodic Saves to MacBook**

**Setup** (in each worker):
```bash
# Use result syncer module
from shared.result_syncer import ResultSyncer

syncer = ResultSyncer(
    local_results_dir='./results',
    heartbeat_interval=60,  # 1 minute
    checkpoint_interval=300  # 5 minutes
)

# During training/processing
syncer.update_heartbeat("Processing frame 100/500")
syncer.save_checkpoint({'epoch': 5, 'loss': 0.23})
```

**Sync Script** (runs every 5 minutes):
```bash
# scripts/sync_worker_results.sh
rsync -avz -e 'ssh -i ~/keys/AutoGenKeyPair.pem' \
  ubuntu@34.196.155.11:~/mono_to_3d_worker1/results/ \
  /local/path/worker1_results/

rsync -avz -e 'ssh -i ~/keys/AutoGenKeyPair.pem' \
  ubuntu@34.196.155.11:~/mono_to_3d_worker2/results/ \
  /local/path/worker2_results/
```

---

### ✅ **3. Heartbeat Monitoring**

**Each worker maintains**:
```
results/HEARTBEAT.txt:
[2026-01-26 10:23:45] Worker 1: TDD RED phase complete
[2026-01-26 10:25:12] Worker 1: Implementing object detector
[2026-01-26 10:30:45] Worker 1: Training detector - epoch 1/10
[2026-01-26 10:35:12] Worker 1: Training detector - epoch 2/10
...
```

**Monitor script** (MacBook):
```bash
# Check worker health
watch -n 30 'cat /local/path/worker*/results/HEARTBEAT.txt | tail -5'
```

---

### ✅ **4. Git Workflow**

**Each worker commits frequently**:
```bash
# After RED phase
git add tests/
git commit -m "test: Add TDD tests for object detector [RED]"

# After GREEN phase  
git add src/
git commit -m "feat: Implement object detector [GREEN]"

# After REFACTOR phase
git add src/
git commit -m "refactor: Optimize detector inference [REFACTOR]"

# Push to remote
git push origin object-level/detection-tracking
```

---

### ✅ **5. EC2 Execution**

**All compute on EC2**:
```bash
# SSH to EC2
ssh -i ~/keys/AutoGenKeyPair.pem ubuntu@34.196.155.11

# Setup worktrees
cd ~/mono_to_3d
git worktree add ~/mono_to_3d_worker1 object-level/detection-tracking
git worktree add ~/mono_to_3d_worker2 object-level/transformer

# Activate venv in each
cd ~/mono_to_3d_worker1 && source venv/bin/activate
cd ~/mono_to_3d_worker2 && source venv/bin/activate

# Run with monitoring
cd ~/mono_to_3d_worker1
nohup bash scripts/run_detection_tracking_pipeline.sh > worker1.log 2>&1 &

cd ~/mono_to_3d_worker2
nohup bash scripts/run_transformer_development.sh > worker2.log 2>&1 &
```

---

## 📊 **Progress Tracking**

### **Worker 1 Checklist**:
- [ ] TDD: Object Detector (RED-GREEN-REFACTOR)
- [ ] Train object detector on dataset
- [ ] Validate detection accuracy (>95%)
- [ ] TDD: Object Tracker (RED-GREEN-REFACTOR)
- [ ] Implement tracking pipeline
- [ ] Validate tracking purity (>90%)
- [ ] Integration: Detection + Tracking
- [ ] Save results + sync to MacBook
- [ ] Evidence files captured

### **Worker 2 Checklist**:
- [ ] TDD: Object Tokenizer (RED-GREEN-REFACTOR)
- [ ] Implement token sequence generation
- [ ] TDD: Transformer Adaptation (RED-GREEN-REFACTOR)
- [ ] Modify MagVIT for object tokens
- [ ] Add attention extraction
- [ ] Create training script
- [ ] Test with mock data
- [ ] Save results + sync to MacBook
- [ ] Evidence files captured

### **Integration Checklist**:
- [ ] Merge worker branches
- [ ] End-to-end pipeline test
- [ ] Generate attention heatmaps
- [ ] Compute efficiency metrics
- [ ] Comprehensive evaluation
- [ ] Final report

---

## 🎯 **Success Criteria**

### **Worker 1 Success**:
- ✅ Object detector: >95% detection rate
- ✅ Object tracker: >90% track purity
- ✅ Pipeline processes full dataset
- ✅ All TDD evidence captured
- ✅ Results synced to MacBook

### **Worker 2 Success**:
- ✅ Object tokens correctly generated
- ✅ Transformer processes object sequences
- ✅ Attention weights extractable
- ✅ All TDD evidence captured
- ✅ Results synced to MacBook

### **Integration Success**:
- ✅ End-to-end accuracy: >90%
- ✅ Attention ratio (persistent/transient): >3x
- ✅ Compute savings: >40%
- ✅ All visualizations generated
- ✅ Complete documentation

---

## 🚀 **Execution Timeline**

| Day | Worker 1 | Worker 2 |
|-----|----------|----------|
| **1** | TDD: Detector (RED) | TDD: Tokenizer (RED) |
| **2** | Detector implementation (GREEN) | Tokenizer implementation (GREEN) |
| **3** | Detector training | Transformer architecture |
| **4** | TDD: Tracker (RED+GREEN) | Transformer adaptation |
| **5** | Tracker implementation | Training script |
| **6-7** | **Integration Phase** | **Integration Phase** |
| **8-9** | **Evaluation & Visualization** | **Evaluation & Visualization** |

**Total Estimated Time**: 9-10 days with parallel execution

---

## 📁 **File Structure**

```
object_level_persistence/
├── DESIGN_DOCUMENT.md
├── IMPLEMENTATION_ROADMAP.md
├── PARALLEL_IMPLEMENTATION_PLAN.md (this file)
├── src/
│   ├── __init__.py
│   ├── object_detector.py          # Worker 1
│   ├── object_tracker.py           # Worker 1
│   ├── object_tokenizer.py         # Worker 2
│   ├── object_aware_transformer.py # Worker 2
│   └── pipeline.py                 # Integration
├── tests/
│   ├── test_object_detector.py     # Worker 1
│   ├── test_object_tracker.py      # Worker 1
│   ├── test_object_tokenizer.py    # Worker 2
│   └── test_transformer.py         # Worker 2
├── scripts/
│   ├── run_detection_tracking.sh   # Worker 1
│   ├── run_transformer_dev.sh      # Worker 2
│   ├── sync_worker_results.sh      # MacBook
│   └── integrate_and_evaluate.sh   # Integration
├── results/
│   ├── worker1/
│   ├── worker2/
│   └── integration/
└── artifacts/
    ├── tdd_detection_red.txt
    ├── tdd_detection_green.txt
    ├── tdd_tracking_red.txt
    ├── tdd_tracking_green.txt
    ├── tdd_transformer_red.txt
    └── tdd_transformer_green.txt
```

---

## ⚠️ **Risk Mitigation**

**Risk**: Workers diverge, integration difficult  
**Mitigation**: 
- Clear interface contracts defined upfront
- Regular sync meetings (daily standup via progress files)
- Mock implementations for testing

**Risk**: EC2 crashes, lose progress  
**Mitigation**:
- Periodic git commits (every hour)
- Result syncing every 5 minutes
- Heartbeat monitoring

**Risk**: One worker blocks the other  
**Mitigation**:
- Worker 2 uses mock detector/tracker during development
- Integration phase has buffer time

---

## 🔔 **Monitoring Commands** (MacBook)

```bash
# Watch worker heartbeats
watch -n 30 'tail -5 /path/to/worker*/results/HEARTBEAT.txt'

# Check worker progress
cat /path/to/worker1/results/PROGRESS.txt
cat /path/to/worker2/results/PROGRESS.txt

# Check logs
tail -f /path/to/worker1/results/training.log
tail -f /path/to/worker2/results/training.log

# Check TDD evidence
ls -la /path/to/artifacts/tdd_*
```

---

**Ready to start parallel implementation!** 🚀

**Next Steps**:
1. Create git branches on EC2
2. Setup git worktrees
3. Initialize both workers
4. Start parallel development

