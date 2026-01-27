# Parallel Training Status Report

**Date**: 2026-01-25 19:44  
**Status**: 🏃 4 Workers DEPLOYED & RUNNING

---

## Executive Summary

✅ **All 4 feature extractors deployed to EC2 in parallel training configuration**
- Worker 1: I3D (PID: 40478)
- Worker 2: Slow/Fast (PID: 40483)  
- Worker 3: Transformer (PID: 40488)
- Worker 4: MagVIT (PID: 40493)

✅ **Training on 10,000-sample dataset** (perfectly balanced, 2,500 per class)

⚠️ **Connection Status**: EC2 under heavy load - SSH timeouts expected during training

---

## Training Configuration

### Dataset
- **Path**: `~/mono_to_3d/data/10k_trajectories/`
- **Total Samples**: 10,000
- **Classes**: 4 (Linear, Circular, Helical, Parabolic)
- **Balance**: 2,500 samples per class
- **Video Format**: (4 cameras, 32 frames, 3 channels, 64x64)

### Training Parameters (All Workers)
```yaml
epochs: 10
batch_size: 16
learning_rate: 0.001
checkpoint_interval: 2  # Save every 2 epochs
data_dir: ../../data/10k_trajectories
early_stopping: false
weight_decay: 0.0001
```

### Feature Extractor Specifications
| Worker | Feature Dim | Architecture |
|--------|-------------|--------------|
| I3D | 1024 | 3D Inflated Convolutions |
| Slow/Fast | 1024 | Two-pathway (slow + fast) |
| Transformer | 512 | Self-attention encoder |
| MagVIT | 1024 | Video transformer with attention |

---

## Infrastructure Setup

### Git Worktree Structure
```
~/mono_to_3d/parallel_training/
├── worker_i3d/          (trajectory-video/branch-1-i3d)
├── worker_slowfast/     (trajectory-video/branch-2-slowfast)
├── worker_transformer/  (trajectory-video/branch-3-transformer)
└── worker_magvit/       (trajectory-video/branch-4-magvit)
```

### Shared Resources
- **Dataset**: Symlinked to `~/mono_to_3d/data/`
- **Virtual Env**: Symlinked to `~/mono_to_3d/venv/`
- **Shared Code**: 
  - `base_extractor.py` (abstract interface)
  - `unified_model.py` (multi-task model)

### Logs & Checkpoints
- **Logs**: `~/mono_to_3d/parallel_training/logs/worker_*_20260125_1944.log`
- **PIDs**: `~/mono_to_3d/parallel_training/logs/worker_*.pid`
- **Checkpoints**: Each worker saves to `results/validation/checkpoints/`

---

## Monitoring Commands

### Check if processes are running
```bash
ssh -i ~/.ssh/aws-key-mike.pem ubuntu@34.196.155.11 \
  'cd ~/mono_to_3d/parallel_training/logs && \
   for pid_file in *.pid; do \
     worker=$(basename "$pid_file" .pid); \
     pid=$(cat "$pid_file"); \
     if ps -p "$pid" > /dev/null 2>&1; then \
       echo "✅ $worker (PID: $pid) - RUNNING"; \
     else \
       echo "❌ $worker (PID: $pid) - STOPPED"; \
     fi; \
   done'
```

### View training logs
```bash
# View all logs
ssh -i ~/.ssh/aws-key-mike.pem ubuntu@34.196.155.11 \
  'tail -f ~/mono_to_3d/parallel_training/logs/worker_*_20260125_1944.log'

# View specific worker
ssh -i ~/.ssh/aws-key-mike.pem ubuntu@34.196.155.11 \
  'tail -f ~/mono_to_3d/parallel_training/logs/worker_i3d_20260125_1944.log'
```

### Check training progress
```bash
ssh -i ~/.ssh/aws-key-mike.pem ubuntu@34.196.155.11 \
  'for worker in ~/mono_to_3d/parallel_training/worker_*/experiments/trajectory_video_understanding/*/results/validation; do \
     echo "=== $(basename $(dirname $(dirname $(dirname $worker)))) ==="; \
     ls -lh $worker/checkpoints/ 2>/dev/null | tail -5 || echo "No checkpoints yet"; \
     echo ""; \
   done'
```

### Monitor GPU usage
```bash
ssh -i ~/.ssh/aws-key-mike.pem ubuntu@34.196.155.11 'nvidia-smi'
```

### Check disk space
```bash
ssh -i ~/.ssh/aws-key-mike.pem ubuntu@34.196.155.11 'df -h /'
```

---

## Expected Timeline

### Per Worker (10 epochs, 10K samples)
- **Samples per epoch**: 10,000
- **Batch size**: 16
- **Steps per epoch**: 625
- **Estimated time per epoch**: 2-5 minutes (varies by architecture)
- **Total training time**: 20-50 minutes per worker

### Checkpoints
- Checkpoint saved every 2 epochs
- Total checkpoints per worker: 5 (epochs 2, 4, 6, 8, 10)

---

## Troubleshooting

### If SSH times out
This is **expected** during heavy training. The EC2 instance is fully utilizing GPU/CPU resources. Wait 5-10 minutes and try again.

### If a worker stops
```bash
# Check which workers stopped
ssh -i ~/.ssh/aws-key-mike.pem ubuntu@34.196.155.11 \
  'cd ~/mono_to_3d/parallel_training && \
   grep -l "error\|Error\|ERROR" logs/worker_*_20260125_1944.log'

# View error
ssh -i ~/.ssh/aws-key-mike.pem ubuntu@34.196.155.11 \
  'tail -50 ~/mono_to_3d/parallel_training/logs/worker_XXX_20260125_1944.log'
```

### If training hangs
```bash
# Kill all training processes
ssh -i ~/.ssh/aws-key-mike.pem ubuntu@34.196.155.11 \
  'cd ~/mono_to_3d/parallel_training/logs && \
   for pid_file in *.pid; do kill $(cat "$pid_file") 2>/dev/null; done'

# Restart training
ssh -i ~/.ssh/aws-key-mike.pem ubuntu@34.196.155.11 \
  'cd ~/mono_to_3d/parallel_training && ./start_parallel_training.sh'
```

---

## Results Collection

### After Training Completes

1. **Copy logs to local machine**:
```bash
scp -i ~/.ssh/aws-key-mike.pem \
  ubuntu@34.196.155.11:~/mono_to_3d/parallel_training/logs/worker_*_20260125_1944.log \
  ./experiments/trajectory_video_understanding/training_logs/
```

2. **Copy checkpoints**:
```bash
scp -i ~/.ssh/aws-key-mike.pem -r \
  ubuntu@34.196.155.11:~/mono_to_3d/parallel_training/worker_*/experiments/trajectory_video_understanding/*/results/validation/ \
  ./experiments/trajectory_video_understanding/validation_results/
```

3. **Analyze results**:
```bash
# Compare validation metrics across all 4 extractors
python experiments/trajectory_video_understanding/compare_results.py
```

---

## Next Steps

### After 10-Epoch Validation Completes

1. **Collect and analyze results** from all 4 workers
2. **Compare metrics**:
   - Classification accuracy
   - Prediction error (MSE for next-frame position)
   - Training time
   - Model size
3. **Decide on full training**:
   - Option A: Train best 1-2 extractors for full 50 epochs
   - Option B: Train all 4 for full 50 epochs (if resources allow)
4. **Generate comparison report** with visualizations

### Deferred Tasks
- Ensemble integration (combining multiple extractors)
- LLM reasoning layer (explaining predictions)
- Attention visualization
- 30K dataset generation (if needed)

---

## Key Achievements

✅ EC2 storage extended: 194GB → 388GB  
✅ 10K dataset generated and validated  
✅ 6 git branches created and pushed  
✅ All 4 feature extractors implemented with TDD  
✅ Parallel training infrastructure deployed  
✅ 4 workers training simultaneously on 10K samples  

---

**Status**: 🏃 Training in progress - check back in 20-50 minutes for results

**Estimated Completion**: 2026-01-25 20:05 - 20:35 (depending on GPU utilization)

