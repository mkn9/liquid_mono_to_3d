# Periodic Monitoring Failure Analysis

**Date**: 2026-01-25  
**Issue**: TDD tests timed out - periodic monitoring not working

---

## 🚨 ROOT CAUSE ANALYSIS

### What Happened
1. TDD GREEN phase test **timed out** (hung indefinitely)
2. SSH session hung waiting for output
3. No periodic "keep-alive" signals to prevent timeout

### Why It Failed

#### Problem 1: No Batch-Level Output
```python
# Current implementation in train_one_epoch():
for batch_idx, (videos, labels) in enumerate(train_loader):
    videos = videos.to(device)
    # ... training code ...
    # NO OUTPUT HERE!
    
# Only prints AFTER entire epoch completes
print(f"Epoch {epoch+1}/{epochs}...")
```

**Issue**: If an epoch takes 5+ minutes, NO OUTPUT for 5 minutes → SSH hangs!

#### Problem 2: No Output Buffer Flushing
```python
print(f"Epoch {epoch+1}...")  # Output might be buffered
# Need: sys.stdout.flush() to force immediate output
```

**Issue**: Python buffers stdout - output might not appear until buffer fills or process ends.

#### Problem 3: Tests Run Full Training
```python
# In test_loss_decreases_over_epochs():
for epoch in range(3):  # 3 full epochs!
    train_one_epoch(...)  # Could take 1-2 min per epoch
    # No timeout, no output during training
```

**Issue**: Tests run for 3-6 minutes with NO output → timeout!

#### Problem 4: Progress File Only (No stdout)
```python
def update_progress(...):
    progress_path.write_text(content)  # Writes file
    # But NO print() statement!
```

**Issue**: File updates don't keep SSH alive - need stdout output!

---

## 📚 BEST PRACTICES FOR LONG-RUNNING PROCESSES

### 1. Batch-Level Progress (Most Important!)
```python
for batch_idx, (videos, labels) in enumerate(train_loader):
    # ... training ...
    
    # Print every N batches (e.g., every 5)
    if batch_idx % 5 == 0:
        print(f"  Batch {batch_idx}/{len(train_loader)}", flush=True)
```

**Why**: Ensures output every 30-60 seconds, keeps SSH alive

### 2. Always Flush Output
```python
print("Progress update", flush=True)
sys.stdout.flush()  # Explicit flush for safety
```

**Why**: Ensures output appears immediately, not buffered

### 3. Use Python Unbuffered Mode
```bash
python -u train_magvit.py  # -u flag disables buffering
# OR
PYTHONUNBUFFERED=1 python train_magvit.py
```

**Why**: Forces immediate output, prevents buffering issues

### 4. Add Heartbeat Mechanism
```python
import threading
import time

def heartbeat(interval=30):
    """Print heartbeat every N seconds"""
    while training_active:
        print(f"[HEARTBEAT {datetime.now()}]", flush=True)
        time.sleep(interval)

# Start heartbeat thread
threading.Thread(target=heartbeat, daemon=True).start()
```

**Why**: Guarantees output even if training stalls

### 5. Use tqdm for Progress Bars
```python
from tqdm import tqdm

for batch_idx, (videos, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
    # Training code
    pass
```

**Why**: Automatic progress updates, ETA, keeps terminal active

### 6. Set Test Timeouts
```python
@pytest.mark.timeout(60)  # 60 second timeout
def test_training_runs_one_epoch():
    # Test code
    pass
```

**Why**: Prevents infinite hangs, fails fast if something's wrong

### 7. Mock Long Operations in Tests
```python
# In tests, use tiny dataset or mock training:
@patch('train_magvit.train_one_epoch')
def test_training_loop(mock_train):
    mock_train.return_value = {'loss': 0.5}
    # Test orchestration logic without actual training
```

**Why**: Tests should be fast (<10 seconds), not run real training

---

## 🔧 REQUIRED FIXES

### Fix 1: Add Batch-Level Progress to train_one_epoch()
```python
def train_one_epoch(model, train_loader, optimizer, device, epoch, verbose=True):
    model.train()
    
    total_loss = 0.0
    num_batches = len(train_loader)
    
    for batch_idx, (videos, labels) in enumerate(train_loader):
        videos = videos.to(device)
        
        # ... training code ...
        
        # CRITICAL: Print progress every 5 batches
        if verbose and (batch_idx % 5 == 0 or batch_idx == num_batches - 1):
            print(f"  Batch {batch_idx+1}/{num_batches} - Loss: {loss.item():.4f}", 
                  flush=True)
        
        total_loss += loss.item()
    
    return {'loss': total_loss / num_batches}
```

### Fix 2: Use PYTHONUNBUFFERED in Training
```python
# In main training script:
import sys
import os

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # Unbuffered
```

### Fix 3: Add Timeouts to Tests
```python
import pytest

class TestMAGVITTrainingLoop:
    @pytest.mark.timeout(120)  # 2 minute timeout
    def test_training_runs_one_epoch(self):
        # Test code
        pass
```

### Fix 4: Use Tiny Dataset in Tests
```python
def test_training_runs_one_epoch(tmp_path):
    # Create TINY test dataset (10 samples only!)
    tiny_videos = torch.randn(10, 3, 8, 32, 32)  # Small: 8 frames, 32x32
    tiny_labels = torch.randint(0, 4, (10,))
    
    # Save tiny dataset
    tiny_path = tmp_path / "tiny_dataset.npz"
    np.savez(tiny_path, videos=tiny_videos, labels=tiny_labels)
    
    # Test with tiny dataset (should complete in <30 seconds)
    train_loader, _, _ = load_dataset(tiny_path, batch_size=2)
    # ... rest of test
```

### Fix 5: Print + Flush in update_progress()
```python
def update_progress(progress_path, epoch, total_epochs, train_loss, val_loss, elapsed_time):
    # ... write file ...
    progress_path.write_text(content)
    
    # CRITICAL: Also print to stdout to keep SSH alive
    print(f"[PROGRESS] Epoch {epoch+1}/{total_epochs} - "
          f"Train: {train_loss:.6f}, Val: {val_loss:.6f}", 
          flush=True)
```

### Fix 6: Add Heartbeat Thread
```python
import threading
import time

training_active = False

def heartbeat_thread(interval=30):
    """Print heartbeat every 30 seconds"""
    global training_active
    while training_active:
        print(f"[HEARTBEAT {datetime.now().strftime('%H:%M:%S')}] Training active...", 
              flush=True)
        time.sleep(interval)

def train_magvit(...):
    global training_active
    training_active = True
    
    # Start heartbeat
    heartbeat = threading.Thread(target=heartbeat_thread, daemon=True)
    heartbeat.start()
    
    try:
        # Training loop
        pass
    finally:
        training_active = False
```

---

## 🎯 IMPLEMENTATION PRIORITY

### Immediate (Critical):
1. ✅ Add batch-level progress prints with `flush=True`
2. ✅ Add test timeouts
3. ✅ Use tiny datasets in tests
4. ✅ Add `flush=True` to ALL print statements
5. ✅ Print in `update_progress()` (not just file write)

### Important (Should have):
6. ✅ Use `python -u` or `PYTHONUNBUFFERED=1`
7. ✅ Add heartbeat thread

### Nice to have:
8. ⭕ Use tqdm for progress bars
9. ⭕ Mock training in unit tests

---

## 📊 WHY THIS MATTERS

### SSH Timeout Mechanism
```
SSH Connection
    ↓
No output for 5+ minutes
    ↓
TCP keepalive fails
    ↓
SSH server marks connection as dead
    ↓
Process killed or orphaned
```

### Solution: Regular Output
```
Training process
    ↓
Print every 30-60 seconds
    ↓
SSH sees activity
    ↓
Connection stays alive
    ↓
Process completes successfully
```

---

## 🔍 ROOT CAUSE SUMMARY

**Why monitoring failed**:
1. ❌ No batch-level output (only epoch-level)
2. ❌ No stdout flushing
3. ❌ Tests run real training (takes minutes)
4. ❌ No test timeouts
5. ❌ Progress only writes files (doesn't print)

**Result**: SSH hangs after 5 minutes of no output → timeout

**Fix**: Add frequent prints with `flush=True` at batch level

---

## ✅ VERIFICATION PLAN

After fixes:
1. Run single test with timeout: Should complete or fail fast
2. Watch for batch-level progress: Should see prints every 30-60 sec
3. Monitor SSH connection: Should stay alive during long runs
4. Check PROGRESS.txt: Should update AND print to stdout

Expected output:
```
Epoch 1/100
  Batch 1/20 - Loss: 0.5432
  Batch 5/20 - Loss: 0.5123
  Batch 10/20 - Loss: 0.4987
  ...
[PROGRESS] Epoch 1/100 - Train: 0.5000, Val: 0.5100
Epoch 2/100
  Batch 1/20 - Loss: 0.4876
  ...
```

This ensures continuous output every 30-60 seconds.

