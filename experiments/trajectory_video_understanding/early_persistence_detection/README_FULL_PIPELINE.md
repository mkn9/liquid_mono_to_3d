# MagVIT Early Persistence Detection - Full EC2 Pipeline

**Complete implementation following all standard procedures**

## 📋 Overview

This document describes the complete pipeline for training and evaluating a MagVIT-based early persistence detection system, following all project standards:

✅ **TDD** - Test-Driven Development with RED-GREEN-REFACTOR  
✅ **Periodic Saving** - Real-time result syncing to MacBook  
✅ **Heartbeat Monitoring** - Progress visibility  
✅ **Git Branch Workflow** - Organized development on EC2  
✅ **Evidence Capture** - Complete audit trail  

---

## 🎯 What Was Built

### 1. Evaluation Scripts (TDD Complete)

**Location**: `experiments/trajectory_video_understanding/early_persistence_detection/evaluation/`

Four comprehensive evaluation modules:

#### a) `evaluate_model.py`
- Loads trained models
- Runs inference on test dataset
- Computes accuracy, early-stop rate, compute savings
- Generates confusion matrix
- **Usage**:
  ```bash
  python evaluation/evaluate_model.py \
      --model results/final_model.pt \
      --test_data ../persistence_augmented_dataset/output \
      --output results/evaluation \
      --device cuda
  ```

#### b) `visualize_attention.py`
- Extracts attention weights from model
- Generates attention heatmaps
- Analyzes attention distribution (persistent vs transient)
- Creates visual reports
- **Usage**:
  ```bash
  python evaluation/visualize_attention.py \
      --model results/final_model.pt \
      --data ../persistence_augmented_dataset/output \
      --output results/visualizations \
      --num_samples 20 \
      --device cuda
  ```

#### c) `analyze_efficiency.py`
- Loads efficiency metrics
- Computes savings statistics
- Generates decision frame histograms
- Creates compute usage charts
- Produces efficiency report
- **Usage**:
  ```bash
  python evaluation/analyze_efficiency.py \
      --metrics results/efficiency_metrics.json \
      --output results/analysis
  ```

#### d) `generate_report.py`
- Collects all results
- Generates markdown report
- Generates HTML report with embedded images
- Creates comprehensive documentation
- **Usage**:
  ```bash
  python evaluation/generate_report.py \
      --results_dir results \
      --output results/FINAL_REPORT
  ```

### 2. TDD Evidence

**Location**: `artifacts/`

- `tdd_evaluation_red.txt` - RED phase (tests fail before implementation) ✅
- `tdd_evaluation_green.txt` - GREEN phase (tests pass after implementation) ✅
- `tdd_evaluation_refactor.txt` - REFACTOR phase (tests still pass after cleanup)

**Verification Script**: `scripts/tdd_evaluation_capture.sh`
- Supports RED, GREEN, REFACTOR phases
- Automatic evidence capture
- Validates TDD compliance

### 3. Full EC2 Pipeline Script

**Location**: `scripts/run_magvit_early_persistence_pipeline_ec2.sh`

**Features**:
- ✅ Git branch creation/checkout on EC2
- ✅ TDD evidence capture and commit
- ✅ Training with heartbeat monitoring
- ✅ Real-time periodic syncing to MacBook (every 60s)
- ✅ Evaluation, visualization, and analysis
- ✅ Final report generation
- ✅ Automatic git commits at each stage
- ✅ MacBook result visibility throughout

---

## 🚀 How to Run

### Option 1: Full Automated Pipeline

**Single command** to run everything:

```bash
cd /Users/mike/Dropbox/Documents/Machine_Learning/LLM_and_Multimodal_Models/repos/mono_to_3d
bash scripts/run_magvit_early_persistence_pipeline_ec2.sh
```

This will:
1. Set up git branch `early-persistence/magvit` on EC2
2. Run TDD verification
3. Train model (20 epochs, batch_size=8)
4. Run evaluation
5. Generate visualizations
6. Analyze efficiency
7. Create final report
8. Sync everything to MacBook
9. Pull git branch locally

**Estimated time**: 1-2 hours (depending on dataset size and GPU)

### Option 2: Manual Step-by-Step

If you want to run steps manually:

#### Step 1: SSH to EC2
```bash
ssh -i ~/keys/AutoGenKeyPair.pem ubuntu@34.196.155.11
cd ~/mono_to_3d
source venv/bin/activate
```

#### Step 2: Create Git Branch
```bash
git checkout -b early-persistence/magvit
git push -u origin early-persistence/magvit
```

#### Step 3: Run TDD Verification
```bash
bash scripts/tdd_evaluation_capture.sh green
git add artifacts/tdd_evaluation_green.txt
git commit -m "TDD: Evaluation scripts GREEN phase"
git push
```

#### Step 4: Train Model
```bash
python experiments/trajectory_video_understanding/early_persistence_detection/training/train_early_persistence.py \
    --data_dir experiments/trajectory_video_understanding/persistence_augmented_dataset/output \
    --output_dir experiments/trajectory_video_understanding/early_persistence_detection/results \
    --epochs 20 \
    --batch_size 8 \
    --device cuda
```

#### Step 5: Run Evaluation
```bash
python experiments/trajectory_video_understanding/early_persistence_detection/evaluation/evaluate_model.py \
    --model experiments/trajectory_video_understanding/early_persistence_detection/results/final_model.pt \
    --test_data experiments/trajectory_video_understanding/persistence_augmented_dataset/output \
    --output experiments/trajectory_video_understanding/early_persistence_detection/results/evaluation \
    --device cuda
```

#### Step 6: Generate Visualizations
```bash
python experiments/trajectory_video_understanding/early_persistence_detection/evaluation/visualize_attention.py \
    --model experiments/trajectory_video_understanding/early_persistence_detection/results/final_model.pt \
    --data experiments/trajectory_video_understanding/persistence_augmented_dataset/output \
    --output experiments/trajectory_video_understanding/early_persistence_detection/results/visualizations \
    --num_samples 20 \
    --device cuda
```

#### Step 7: Analyze Efficiency
```bash
python experiments/trajectory_video_understanding/early_persistence_detection/evaluation/analyze_efficiency.py \
    --metrics experiments/trajectory_video_understanding/early_persistence_detection/results/efficiency_metrics.json \
    --output experiments/trajectory_video_understanding/early_persistence_detection/results/analysis
```

#### Step 8: Generate Report
```bash
python experiments/trajectory_video_understanding/early_persistence_detection/evaluation/generate_report.py \
    --results_dir experiments/trajectory_video_understanding/early_persistence_detection/results \
    --output experiments/trajectory_video_understanding/early_persistence_detection/results/FINAL_REPORT
```

#### Step 9: Commit and Push
```bash
git add experiments/trajectory_video_understanding/early_persistence_detection/results
git commit -m "Complete: MagVIT early persistence detection pipeline"
git push origin early-persistence/magvit
```

#### Step 10: Sync to MacBook
```bash
# On MacBook
cd /Users/mike/Dropbox/Documents/Machine_Learning/LLM_and_Multimodal_Models/repos/mono_to_3d
git fetch origin
git checkout early-persistence/magvit
git pull

# Sync results
rsync -avz -e "ssh -i ~/keys/AutoGenKeyPair.pem" \
    ubuntu@34.196.155.11:~/mono_to_3d/experiments/trajectory_video_understanding/early_persistence_detection/results/ \
    experiments/trajectory_video_understanding/early_persistence_detection/results/
```

---

## 📊 Monitoring Progress

### Real-Time Monitoring (during pipeline execution)

The automated pipeline provides real-time visibility:

#### 1. PROGRESS.txt
**Location**: `experiments/trajectory_video_understanding/early_persistence_detection/results/PROGRESS.txt`

**Updates**: Every major milestone
```
[2026-01-25 12:00:00] Training started
[2026-01-25 12:15:00] Epoch 5/20 complete
[2026-01-25 12:30:00] Training complete
[2026-01-25 12:31:00] Evaluation started
...
```

#### 2. HEARTBEAT.txt
**Location**: `experiments/trajectory_video_understanding/early_persistence_detection/results/HEARTBEAT.txt`

**Updates**: Every 30 seconds
```
[2026-01-25 12:00:00] Heartbeat: Training initialization
[2026-01-25 12:00:30] Heartbeat: Training in progress
[2026-01-25 12:01:00] Heartbeat: Training in progress
...
[2026-01-25 12:30:00] Heartbeat: Training completed
```

#### 3. Training Log
**Location**: `experiments/trajectory_video_understanding/early_persistence_detection/results/training.log`

**Updates**: Real-time during training
```
Epoch 1/20: Loss=0.523, Accuracy=0.756
Epoch 2/20: Loss=0.412, Accuracy=0.823
...
```

### MacBook Visibility

**All files are synced to MacBook every 60 seconds during pipeline execution!**

To monitor in real-time on MacBook:

```bash
# Watch PROGRESS
watch -n 5 cat experiments/trajectory_video_understanding/early_persistence_detection/results/PROGRESS.txt

# Watch HEARTBEAT
watch -n 5 cat experiments/trajectory_video_understanding/early_persistence_detection/results/HEARTBEAT.txt

# Tail training log
tail -f experiments/trajectory_video_understanding/early_persistence_detection/results/training.log
```

---

## 📁 Directory Structure

```
experiments/trajectory_video_understanding/early_persistence_detection/
├── models/                              # Core model implementations
│   ├── early_persistence_classifier.py  # Main classifier
│   ├── attention_visualization.py       # Attention viz module
│   ├── compute_gating.py                # Compute allocation
│   ├── efficiency_metrics.py            # Efficiency tracking
│   └── tests/                           # Model tests
│       └── test_early_persistence_classifier.py
│
├── evaluation/                          # Evaluation scripts (TDD complete)
│   ├── __init__.py
│   ├── evaluate_model.py                # Model evaluation
│   ├── visualize_attention.py           # Attention visualization
│   ├── analyze_efficiency.py            # Efficiency analysis
│   ├── generate_report.py               # Report generation
│   └── tests/                           # Evaluation tests
│       ├── __init__.py
│       └── test_evaluation_scripts.py   # 16 tests (all passing)
│
├── training/                            # Training scripts
│   └── train_early_persistence.py       # Main training script
│
├── results/                             # All results (synced to MacBook)
│   ├── PROGRESS.txt                     # Progress updates
│   ├── HEARTBEAT.txt                    # Heartbeat monitor
│   ├── training.log                     # Training logs
│   ├── final_model.pt                   # Trained model (not synced)
│   ├── efficiency_metrics.json          # Efficiency data
│   ├── evaluation/                      # Evaluation results
│   │   ├── evaluation_metrics.json
│   │   └── confusion_matrix.png
│   ├── visualizations/                  # Attention visualizations
│   │   ├── sample_0000_attention.png
│   │   ├── sample_0001_attention.png
│   │   └── ...
│   ├── analysis/                        # Efficiency analysis
│   │   ├── decision_frame_histogram.png
│   │   ├── compute_usage_chart.png
│   │   └── efficiency_report.md
│   ├── FINAL_REPORT.md                  # Markdown report
│   ├── FINAL_REPORT.html                # HTML report
│   ├── TRAINING_COMPLETE.txt            # Training completion marker
│   └── PIPELINE_COMPLETE.txt            # Pipeline completion marker
│
└── README_FULL_PIPELINE.md              # This file!
```

---

## 🔍 Viewing Results

### Final Report

**HTML Version** (Recommended):
```bash
open experiments/trajectory_video_understanding/early_persistence_detection/results/FINAL_REPORT.html
```

**Markdown Version**:
```bash
cat experiments/trajectory_video_understanding/early_persistence_detection/results/FINAL_REPORT.md
```

### Individual Results

**Evaluation Metrics**:
```bash
cat experiments/trajectory_video_understanding/early_persistence_detection/results/evaluation/evaluation_metrics.json
```

**Efficiency Report**:
```bash
cat experiments/trajectory_video_understanding/early_persistence_detection/results/analysis/efficiency_report.md
```

**Visualizations**:
```bash
open experiments/trajectory_video_understanding/early_persistence_detection/results/visualizations/
```

### Git History

**See all commits on branch**:
```bash
git log early-persistence/magvit --oneline

# Example output:
# a1b2c3d docs: Final comprehensive report
# d4e5f6g analysis: Efficiency analysis complete
# g7h8i9j viz: Attention visualizations (20 samples)
# j0k1l2m eval: MagVIT evaluation metrics
# m3n4o5p training: MagVIT early persistence - 20 epochs complete
# p6q7r8s TDD: Evaluation scripts GREEN phase evidence
```

---

## ✅ Verification Checklist

After pipeline completion, verify:

- [ ] TDD Evidence exists:
  - [ ] `artifacts/tdd_evaluation_red.txt`
  - [ ] `artifacts/tdd_evaluation_green.txt`

- [ ] Training completed successfully:
  - [ ] `results/TRAINING_COMPLETE.txt` exists
  - [ ] `results/final_model.pt` exists on EC2
  - [ ] `results/training.log` shows 20 epochs

- [ ] Evaluation completed:
  - [ ] `results/evaluation/evaluation_metrics.json` exists
  - [ ] Accuracy, early_stop_rate, compute_savings present

- [ ] Visualizations generated:
  - [ ] `results/visualizations/` contains PNG files
  - [ ] At least 20 attention heatmaps

- [ ] Efficiency analysis complete:
  - [ ] `results/analysis/efficiency_report.md` exists
  - [ ] Decision frame histogram exists
  - [ ] Compute usage chart exists

- [ ] Final report generated:
  - [ ] `results/FINAL_REPORT.md` exists
  - [ ] `results/FINAL_REPORT.html` exists

- [ ] Git branch synced:
  - [ ] Branch `early-persistence/magvit` exists
  - [ ] Multiple commits with descriptive messages
  - [ ] All results committed and pushed

- [ ] MacBook sync complete:
  - [ ] Results directory exists locally
  - [ ] Can open HTML report locally
  - [ ] Git branch pulled locally

---

## 🐛 Troubleshooting

### Pipeline Hangs

**Check heartbeat**:
```bash
cat experiments/trajectory_video_understanding/early_persistence_detection/results/HEARTBEAT.txt
```

If no recent updates (> 2 minutes), SSH to EC2 and check processes:
```bash
ssh -i ~/keys/AutoGenKeyPair.pem ubuntu@34.196.155.11
ps aux | grep python
nvidia-smi  # Check GPU usage
```

### Training Fails

**Check training log**:
```bash
tail -100 experiments/trajectory_video_understanding/early_persistence_detection/results/training.log
```

Common issues:
- CUDA out of memory → Reduce batch_size
- Data not found → Verify augmented dataset exists
- Import errors → Check venv activation

### Results Not Syncing

**Check sync process**:
```bash
ps aux | grep rsync
```

**Manual sync**:
```bash
rsync -avz -e "ssh -i ~/keys/AutoGenKeyPair.pem" \
    ubuntu@34.196.155.11:~/mono_to_3d/experiments/trajectory_video_understanding/early_persistence_detection/results/ \
    experiments/trajectory_video_understanding/early_persistence_detection/results/
```

### TDD Tests Fail

**Re-run GREEN phase**:
```bash
# On EC2
bash scripts/tdd_evaluation_capture.sh green

# Check failures
cat artifacts/tdd_evaluation_green.txt
```

---

## 📚 Next Steps

After successful pipeline completion:

1. **Review Results**:
   - Open `FINAL_REPORT.html`
   - Analyze accuracy and efficiency metrics
   - Examine attention visualizations

2. **Compare with Other Extractors**:
   - Run same pipeline for I3D
   - Run same pipeline for Slow/Fast
   - Run same pipeline for Basic Transformer
   - Compare efficiency across all 4

3. **Optimize Hyperparameters**:
   - Adjust `early_stop_frame` (try 3 instead of 4)
   - Tune `confidence_threshold` (try 0.85 or 0.95)
   - Experiment with batch size and learning rate

4. **Integrate with LLM**:
   - Add natural language explanations
   - Provide attention pattern descriptions
   - Generate decision rationales

5. **Production Deployment**:
   - Package model for inference
   - Create API endpoint
   - Deploy to production environment

---

## 🎉 Success Criteria

Pipeline is successful when:

✅ **Accuracy** > 80%  
✅ **Early Stop Rate** > 60%  
✅ **Compute Savings** > 50%  
✅ **Attention Efficiency Ratio** > 2.0 (persistent vs transient)  
✅ **All TDD tests pass**  
✅ **Results visible on MacBook**  
✅ **Git history complete**  

---

## 📞 Support

If you encounter issues:

1. Check this README's Troubleshooting section
2. Review TDD evidence in `artifacts/`
3. Check training logs in `results/`
4. Verify git branch state
5. Contact the development team

---

**Last Updated**: 2026-01-25  
**Version**: 1.0  
**Status**: Ready for Production Execution

