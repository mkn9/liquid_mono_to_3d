# MANDATORY: Incremental Saves for All Long-Running Processes

**Added**: 2026-01-25  
**Priority**: CRITICAL  
**Status**: MANDATORY for all future development

---

## The Rule

**ALL processes that run longer than 5 minutes MUST include incremental saves and progress tracking.**

### ❌ NEVER DO THIS:

```python
# BAD: All work in memory, save only at end
data = generate_large_dataset(num_samples=30000)  # 30+ minutes
save(data)  # If crashes before this, lose everything
```

### ✅ ALWAYS DO THIS:

```python
# GOOD: Incremental checkpoints every 1-5 minutes
for batch in range(0, num_samples, checkpoint_interval):
    data_batch = generate_batch(batch, checkpoint_interval)
    
    # SAVE IMMEDIATELY
    save_checkpoint(data_batch, batch_num)
    
    # UPDATE PROGRESS (visible on MacBook)
    update_progress_file(completed=batch, total=num_samples)
```

---

## Requirements

Every long-running process MUST have:

### 1. **Incremental Checkpoints** (Every 1-5 minutes)
- Save partial results regularly
- Max acceptable loss: 5 minutes of work
- Checkpoints should be mergeable into final result

### 2. **Progress File** (Updated every 30-60 seconds)
- Plain text file showing: completed/total, percent, ETA
- Visible on MacBook without SSH
- Updated in real-time

### 3. **Resume Capability**
- Can continue from last checkpoint if interrupted
- Detect existing checkpoints on startup
- Skip already-completed work

### 4. **MacBook Visibility Test**
**Question**: "Can I see progress by just looking at my MacBook?"
- If NO → **DESIGN IS WRONG, FIX IT**

---

## Implementation Pattern

```python
def long_running_process_with_checkpoints(
    total_work: int,
    checkpoint_interval: int = 1000,  # Adjust based on time
    output_dir: str = "results"
):
    """Template for any long-running process."""
    
    output_dir = Path(output_dir)
    start_time = time.time()
    
    # Check for existing checkpoints (resume capability)
    completed = load_last_checkpoint(output_dir)
    
    for batch_start in range(completed, total_work, checkpoint_interval):
        # Do work
        result = do_work_batch(batch_start, checkpoint_interval)
        
        # SAVE CHECKPOINT IMMEDIATELY
        checkpoint_path = output_dir / f"checkpoint_{batch_start:05d}.npz"
        save(checkpoint_path, result)
        
        # UPDATE PROGRESS FILE (MacBook visible)
        progress_file = output_dir / "PROGRESS.txt"
        with open(progress_file, 'w') as f:
            completed = batch_start + checkpoint_interval
            percent = 100 * completed / total_work
            elapsed = time.time() - start_time
            rate = completed / elapsed
            eta = (total_work - completed) / rate
            
            f.write(f"Progress: {completed}/{total_work} ({percent:.1f}%)\n")
            f.write(f"Elapsed: {elapsed/60:.1f} min\n")
            f.write(f"ETA: {eta/60:.1f} min\n")
            f.write(f"Last update: {datetime.now()}\n")
        
        print(f"✓ Checkpoint {batch_start}: {percent:.1f}% complete")
    
    # Merge checkpoints into final result
    final_result = merge_checkpoints(output_dir)
    return final_result
```

---

## Monitoring Pattern

Create a monitoring script that can run on MacBook:

```bash
#!/bin/bash
# monitor_progress.sh

while true; do
    clear
    echo "Process Monitor"
    echo "==============="
    date
    echo ""
    
    # Show progress from file
    scp -i $KEY user@ec2:~/project/results/PROGRESS.txt /tmp/progress.txt 2>/dev/null
    cat /tmp/progress.txt 2>/dev/null || echo "No progress file yet"
    
    # Check for completion
    if grep -q "COMPLETE" /tmp/progress.txt 2>/dev/null; then
        echo "✅ Process completed!"
        break
    fi
    
    echo ""
    echo "Next update in 30 seconds..."
    sleep 30
done
```

---

## Examples

### Dataset Generation (30K samples, ~30 min)
- **Checkpoint interval**: 2000 samples (~2-3 min each)
- **Progress updates**: Every checkpoint
- **Result**: 15 checkpoints, visible progress every 2-3 min

### Model Training (100 epochs, ~5 hours)
- **Checkpoint interval**: 5 epochs (~15 min each)
- **Progress updates**: Every epoch (~3 min)
- **Result**: 20 checkpoints, visible progress every 3 min

### Video Rendering (1000 videos, ~1 hour)
- **Checkpoint interval**: 100 videos (~6 min each)
- **Progress updates**: Every 10 videos (~36 sec)
- **Result**: 10 checkpoints, visible progress every 36 sec

---

## Why This Matters

### Without Incremental Saves:
- ❌ No visibility into progress
- ❌ High risk (lose all work if crashes)
- ❌ Can't use partial results
- ❌ No way to estimate completion time
- ❌ Debugging is blind

### With Incremental Saves:
- ✅ Real-time progress visible on MacBook
- ✅ Low risk (max 5 min of lost work)
- ✅ Can use partial results if needed
- ✅ Accurate ETA estimates
- ✅ Easy debugging (see where it stopped)

---

## Integration with TDD

Tests should verify:
1. ✅ Checkpoints are created at correct intervals
2. ✅ Progress file is updated correctly
3. ✅ Resume works from any checkpoint
4. ✅ Final merge produces correct result

Example test:
```python
def test_incremental_saves():
    # Generate with checkpoints
    process_with_checkpoints(total=1000, checkpoint_interval=100)
    
    # Verify checkpoints exist
    checkpoints = list(Path("results").glob("checkpoint_*.npz"))
    assert len(checkpoints) == 10  # 1000/100
    
    # Verify progress file
    progress = Path("results/PROGRESS.txt").read_text()
    assert "1000/1000" in progress
    assert "100%" in progress
```

---

## Commitment

**This pattern is now MANDATORY for all long-running processes.**

Any code review or pull request with a long-running process (>5 min) without incremental saves will be **REJECTED**.

**Test**: "Can I see progress on my MacBook without SSH?" If NO → FIX IT.

---

**This requirement will be added to `.cursorrules` immediately.**

