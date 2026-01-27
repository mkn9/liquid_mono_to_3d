# Critical Design Flaw: No Incremental Saves

**Date**: 2026-01-25  
**Issue**: 30K dataset generation runs for 35+ minutes with NO intermediate saves

---

## The Problem

Current implementation:
```python
# Generate all 30K samples in memory
dataset = generate_dataset_parallel(num_samples=30000, ...)

# Save ONLY at the very end
np.savez_compressed(output_path, **dataset)
```

**Issues**:
1. ❌ No progress visible on MacBook during generation
2. ❌ No intermediate checkpoints (35+ min = high risk)
3. ❌ If process crashes, lose ALL work
4. ❌ No way to monitor what's actually happening
5. ❌ Can't use partial results if stopped

---

## Evidence It's Working (But Not Visible)

**Checked 2026-01-25 01:22 UTC (33 min runtime)**:
```bash
lsof -p 18132 | grep torch
# Results:
python 18132 ubuntu 12u REG 0,25 5898240000 53 /dev/shm/torch_18142_*_0
python 18132 ubuntu 14u REG 0,25 5898240000 55 /dev/shm/torch_18143_*_0
```

- **5.9 GB + 5.9 GB = ~12 GB in shared memory** ✓
- Workers at 10-11% CPU (actively computing) ✓
- **Data exists but invisible until save!**

---

## What Should Have Been Done

### ✅ CORRECT Design (Incremental Saves):

```python
def generate_dataset_parallel_with_checkpoints(
    num_samples: int,
    checkpoint_every: int = 1000,  # Save every 1K samples
    output_dir: str = "results"
):
    """Generate dataset with periodic checkpoints."""
    
    for batch_start in range(0, num_samples, checkpoint_every):
        # Generate batch
        batch = generate_batch(batch_start, checkpoint_every)
        
        # SAVE IMMEDIATELY
        checkpoint_path = f"{output_dir}/checkpoint_{batch_start:05d}.npz"
        np.savez_compressed(checkpoint_path, **batch)
        print(f"✓ Saved checkpoint: {batch_start}/{num_samples}")
        
        # Update progress file (visible on MacBook)
        with open(f"{output_dir}/PROGRESS.txt", "w") as f:
            f.write(f"{batch_start}/{num_samples} ({100*batch_start/num_samples:.1f}%)\n")
            f.write(f"Last update: {datetime.now()}\n")
```

**Benefits**:
- ✅ Progress visible on MacBook (sync PROGRESS.txt every 30s)
- ✅ Checkpoints every 1K samples (~1 min intervals)
- ✅ Can resume if crashed
- ✅ Can use partial results
- ✅ Low risk (max 1 min of lost work)

---

## Commitment Going Forward

### Rule: **NEVER create long-running processes without incremental saves**

**All future implementations MUST include**:
1. Progress files updated every 30-60 seconds
2. Checkpoints saved every 1-5 minutes
3. Resume capability from checkpoints
4. Clear status visible on MacBook without SSH

### Test: "Can I see progress by just looking at my MacBook?"
- If NO → **DESIGN IS WRONG**

---

## Current 30K Generation Status

**Start**: 2026-01-25 00:52 UTC  
**Now**: 2026-01-25 01:22 UTC (33 minutes)  
**Data**: 12 GB in RAM (not saved)  
**Workers**: Active (10-11% CPU each)  
**ETA**: Unknown (estimated 15-20 min, now at 33 min)

**Risk**: If crashes, lose 33 minutes of work

---

## Immediate Action

1. ✅ Let current run complete (too far to restart)
2. ✅ Create monitoring script to track memory usage
3. ⏳ After completion, implement checkpoint version
4. ⏳ Add to requirements.MD as mandatory pattern

---

## Lesson Learned

**User feedback**: "we should never ever make a process that doesn't periodically save where results can be seen on the macbook"

**Response**: **ABSOLUTELY CORRECT**. This was a critical oversight.

**Fix**: All future long-running processes will have:
- Incremental saves (every 1-5 min)
- Progress files (updated every 30-60 sec)
- Resume capability
- MacBook-visible status

---

**This pattern will be added to `.cursorrules` as a mandatory requirement.**

