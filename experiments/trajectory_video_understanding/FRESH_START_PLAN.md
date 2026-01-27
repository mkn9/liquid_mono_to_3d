# Fresh Start Plan - Proper Parallel Training

**Date**: 2026-01-25 20:45 UTC  
**Status**: 🔄 IN PROGRESS

---

## Objectives

✅ Follow ALL mandatory procedures:
1. **Periodic saving** - Results visible on MacBook during training
2. **Visible heartbeat** - Progress updates every minute
3. **TDD** - Test modifications before deployment
4. **Parallel git tree processing** - All 4 workers simultaneously
5. **Resource safety** - Conservative settings to prevent freeze

---

## Changes from Failed Run

| Aspect | Failed Run | New Approach |
|--------|-----------|--------------|
| Result sync | Pull-only (external script) | **Push from training script** |
| Batch size | 16 per worker (64 total) | **8 per worker (32 total)** |
| Monitoring | None | **Built-in resource monitoring** |
| Heartbeat | None | **Every 60 seconds to MacBook** |
| Testing | Skipped | **TDD + capacity test** |
| Visibility | Zero | **Real-time on MacBook** |

---

## Implementation Plan

### Phase 1: Cleanup (5 min)
- [x] Document failed run
- [ ] Stop frozen EC2 instance (AWS Console)
- [ ] Kill background sync script (PID: 33458)

### Phase 2: Modify Training Scripts with TDD (20 min)
- [ ] Write tests for result pushing functionality
- [ ] Add `push_results_to_macbook()` function
- [ ] Add heartbeat mechanism
- [ ] Update batch_size in configs (16 → 8)
- [ ] Run TDD cycle (Red → Green → Refactor)
- [ ] Capture TDD evidence

### Phase 3: Capacity Test (10 min)
- [ ] Start EC2 (if stopped)
- [ ] Deploy ONE worker with new scripts
- [ ] Run for 2-3 epochs (~5 min)
- [ ] Verify results appear on MacBook
- [ ] Check resource usage (nvidia-smi, htop)
- [ ] Confirm no freeze

### Phase 4: Parallel Deployment (5 min)
- [ ] Deploy all 4 workers simultaneously
- [ ] Verify all heartbeats appearing
- [ ] Monitor resource usage

### Phase 5: Training (20-50 min)
- [ ] Watch results sync to MacBook in real-time
- [ ] Monitor for any issues
- [ ] Collect results automatically

---

## Technical Implementation

### Modified Training Script Structure

```python
# train.py - with result pushing

import subprocess
import json
from datetime import datetime

class MacBookSyncer:
    """Pushes results to MacBook periodically"""
    
    def __init__(self, macbook_host, results_dir):
        self.macbook_host = macbook_host
        self.results_dir = results_dir
        self.local_path = f"experiments/trajectory_video_understanding/training_results_live/{worker_name}/"
        
    def push_heartbeat(self, epoch, step, loss):
        """Push lightweight heartbeat every minute"""
        heartbeat = {
            "timestamp": datetime.utcnow().isoformat(),
            "epoch": epoch,
            "step": step,
            "loss": float(loss)
        }
        
        # Save locally
        with open('heartbeat.json', 'w') as f:
            json.dump(heartbeat, f)
        
        # Push to MacBook
        subprocess.run([
            'scp', '-i', '~/.ssh/key.pem', '-o', 'StrictHostKeyChecking=no',
            'heartbeat.json',
            f'{self.macbook_host}:{self.local_path}/heartbeat.json'
        ], check=False)  # Don't fail training if push fails
    
    def push_checkpoint(self, epoch, checkpoint_path, metrics):
        """Push checkpoint and metrics after each epoch"""
        # Save metrics locally
        with open(f'metrics_epoch_{epoch}.json', 'w') as f:
            json.dump(metrics, f)
        
        # Push both to MacBook
        for file in [checkpoint_path, f'metrics_epoch_{epoch}.json']:
            subprocess.run([
                'scp', '-i', '~/.ssh/key.pem',
                file,
                f'{self.macbook_host}:{self.local_path}/'
            ], check=False)
    
    def push_log_tail(self):
        """Push last 50 lines of log"""
        subprocess.run([
            f'tail -50 train.log | scp -i ~/.ssh/key.pem - {self.macbook_host}:{self.local_path}/latest_log.txt'
        ], shell=True, check=False)

# Training loop with integrated pushing
syncer = MacBookSyncer(
    macbook_host='your_macbook_user@your_macbook_ip',
    results_dir='training_results_live'
)

last_heartbeat = time.time()

for epoch in range(num_epochs):
    for step, batch in enumerate(dataloader):
        # Training step
        loss = train_step(batch)
        
        # Heartbeat every 60 seconds
        if time.time() - last_heartbeat > 60:
            syncer.push_heartbeat(epoch, step, loss)
            last_heartbeat = time.time()
    
    # End of epoch
    metrics = evaluate()
    
    if epoch % checkpoint_interval == 0:
        checkpoint_path = save_checkpoint(epoch)
        syncer.push_checkpoint(epoch, checkpoint_path, metrics)
        syncer.push_log_tail()
```

### Updated Configs

```yaml
# config_validation.yaml - Updated
epochs: 10
batch_size: 8  # ← Changed from 16
learning_rate: 0.001
checkpoint_interval: 2
output_dir: results/validation
data_dir: ../../data/10k_trajectories

# New: MacBook sync settings
macbook_sync:
  enabled: true
  heartbeat_interval: 60  # seconds
  push_checkpoints: true
  push_logs: true
```

---

## Success Criteria

✅ **Must achieve ALL of these:**

1. Results appear on MacBook within 2 minutes of training start
2. Heartbeat updates visible every 60 seconds
3. Checkpoints sync automatically after each save
4. Training logs accessible on MacBook in real-time
5. No EC2 freeze or unresponsiveness
6. All 4 workers complete 10 epochs
7. Full TDD evidence captured

---

## Rollback Plan

If issues arise:
1. Stop training immediately (have the stop commands ready)
2. Check what results were synced to MacBook
3. Analyze logs locally (they're already on MacBook!)
4. Fix issue
5. Restart affected worker(s)

---

## Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| Cleanup | 5 min | 🔄 In progress |
| TDD Modifications | 20 min | ⏳ Pending |
| Capacity Test | 10 min | ⏳ Pending |
| Parallel Deploy | 5 min | ⏳ Pending |
| Training | 20-50 min | ⏳ Pending |
| **Total** | **60-90 min** | |

---

**Next Step:** Modify training scripts with TDD (Red-Green-Refactor)

