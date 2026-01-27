# Automatic Keep-Alive with Auto-Cleanup

**Purpose**: Run commands with AI keep-alive that **automatically stops** when code finishes

**Created**: 2026-01-25  
**Status**: PRODUCTION READY

---

## 🎯 Problem Solved

**Before**: 
- Start keep-alive manually
- Run your code
- **REMEMBER** to stop keep-alive manually
- Easy to forget → orphaned processes

**Now**:
- One command does everything
- Keep-alive starts automatically
- Code runs
- Keep-alive stops automatically ✨
- **Never forget cleanup!**

---

## 🚀 Quick Start

### **Instead of this:**
```bash
# Old way (3 steps, easy to forget step 3!)
./scripts/ai_keepalive.sh "Task" 30 &
python train.py
pkill -f "ai_keepalive.sh"  # ← Often forgotten!
```

### **Do this:**
```bash
# New way (1 step, auto-cleanup!)
./scripts/run_with_keepalive.sh "python train.py"
```

---

## 📋 Usage Examples

### **Local Python Script**
```bash
./scripts/run_with_keepalive.sh "python train_magvit.py"
```

### **Local Shell Script**
```bash
./scripts/run_with_keepalive.sh "bash scripts/prove.sh"
```

### **Long Command**
```bash
./scripts/run_with_keepalive.sh "python generate_dataset.py --samples 30000"
```

### **EC2 Command**
```bash
./scripts/run_on_ec2_with_keepalive.sh "cd mono_to_3d && python train.py"
```

### **EC2 with Multiple Commands**
```bash
./scripts/run_on_ec2_with_keepalive.sh "cd mono_to_3d && git pull && source venv/bin/activate && python train.py"
```

---

## ✅ What Happens Automatically

### **1. Startup**
```
🚀 Starting AI keep-alive monitor...
   Task: python_train_magvit_py
   Interval: 30s
   Log: /tmp/ai_keepalive_python_train_magvit_py_20260125_120000.log

✅ Keep-alive started (PID: 12345)
📊 Watch progress: tail -f /tmp/ai_keepalive_python_train_magvit_py_20260125_120000.log

▶️  Running: python train_magvit.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Your code output appears here...]
```

### **2. While Running**
- Keep-alive sends heartbeats every 30 seconds
- You can monitor: `tail -f /tmp/ai_keepalive_*.log`
- Your code runs normally

### **3. Automatic Cleanup**
```
🧹 Cleanup: Stopping keep-alive monitor...
✅ Keep-alive stopped (PID: 12345)
📋 Keep-alive log: /tmp/ai_keepalive_python_train_magvit_py_20260125_120000.log
✅ Command completed successfully
```

---

## 🛡️ Safety Features

### **Auto-Cleanup Triggers**

The keep-alive is **automatically stopped** when:

1. ✅ **Code completes successfully** (exit 0)
2. ✅ **Code fails/crashes** (exit non-zero)
3. ✅ **You press Ctrl+C** (manual interrupt)
4. ✅ **Terminal closes unexpectedly**
5. ✅ **System signals** (TERM, INT)

**In ALL cases, cleanup happens!** No orphaned processes.

### **Exit Code Preservation**

Your command's exit code is preserved:
```bash
./scripts/run_with_keepalive.sh "python train.py"
echo $?  # Shows exit code from train.py, not wrapper
```

This means CI/CD, scripts, error handling all work correctly.

---

## 📊 Monitoring While Running

### **Watch Keep-Alive Heartbeats**
Open a new terminal:
```bash
# Find the latest keep-alive log
ls -lt /tmp/ai_keepalive_*.log | head -1

# Watch it live
tail -f /tmp/ai_keepalive_python_train_magvit_py_20260125_120000.log
```

You'll see:
```
⏱️  HEARTBEAT #1 [0m 0s] - python_train_magvit_py - System Active ✅
⏱️  HEARTBEAT #2 [0m 30s] - python_train_magvit_py - System Active ✅
⏱️  HEARTBEAT #3 [1m 0s] - python_train_magvit_py - System Active ✅
...
```

### **Check Process Status**
```bash
ps aux | grep "[a]i_keepalive.sh"
```

---

## 🎯 Real-World Examples

### **Example 1: TDD Workflow on EC2**
```bash
./scripts/run_on_ec2_with_keepalive.sh "cd mono_to_3d && bash scripts/tdd_capture.sh"
```

**What happens:**
1. Local keep-alive starts
2. SSH to EC2
3. Runs TDD workflow
4. When complete (or fail), SSH closes
5. Local keep-alive stops automatically

### **Example 2: Long Training Run**
```bash
./scripts/run_with_keepalive.sh "python experiments/magvit_I3D_LLM_basic_trajectory/train_magvit.py --epochs 50"
```

**Benefits:**
- Keep-alive prevents session timeout
- Auto-stops when training completes
- No manual cleanup needed

### **Example 3: Dataset Generation**
```bash
./scripts/run_with_keepalive.sh "python experiments/magvit_I3D_LLM_basic_trajectory/generate_parallel_30k.py"
```

**Benefits:**
- Keep-alive during long generation
- Auto-stops after 30K samples complete
- Clean exit whether success or failure

### **Example 4: Multiple Commands in Sequence**
```bash
./scripts/run_with_keepalive.sh "python generate_data.py && python train.py && python evaluate.py"
```

**Behavior:**
- Keep-alive runs for entire sequence
- If any command fails, stops immediately
- Cleanup happens after last command

---

## 🔧 Advanced Usage

### **Custom Keep-Alive Interval**

Edit the script to change `KEEPALIVE_INTERVAL`:
```bash
# In run_with_keepalive.sh, line 16:
KEEPALIVE_INTERVAL=30  # Change to 15, 60, etc.
```

Or create a custom wrapper:
```bash
#!/bin/bash
KEEPALIVE_INTERVAL=15 ./scripts/run_with_keepalive.sh "$@"
```

### **Disable Auto-Cleanup (Not Recommended)**

If you want keep-alive to persist:
```bash
# Start manually (old way)
./scripts/ai_keepalive.sh "Task" 30 &

# Run command (keep-alive continues after)
python train.py
```

### **Multiple Parallel Tasks**

If running multiple tasks simultaneously:
```bash
# Terminal 1
./scripts/run_with_keepalive.sh "python task1.py" &

# Terminal 2
./scripts/run_with_keepalive.sh "python task2.py" &

# Each has its own keep-alive
# Each auto-cleans when done
```

---

## 🐛 Troubleshooting

### **Problem: "No command provided" Error**

```bash
❌ ./scripts/run_with_keepalive.sh
Error: No command provided
```

**Solution**: Always provide a command in quotes:
```bash
✅ ./scripts/run_with_keepalive.sh "python train.py"
```

### **Problem: Command Not Found**

```bash
./scripts/run_with_keepalive.sh "train.py"
# Error: train.py: command not found
```

**Solution**: Include full command with interpreter:
```bash
./scripts/run_with_keepalive.sh "python train.py"
```

### **Problem: Keep-Alive Doesn't Stop**

Very rare, but if it happens:
```bash
# Check for stuck processes
ps aux | grep "[a]i_keepalive.sh"

# Force kill all
pkill -9 -f "ai_keepalive.sh"
```

### **Problem: Can't Find Keep-Alive Log**

```bash
# List all recent keep-alive logs
ls -lt /tmp/ai_keepalive_*.log | head -5

# Find by task name
ls /tmp/ai_keepalive_*train*.log
```

---

## 📊 Comparison: Manual vs Auto

| Aspect | Manual Keep-Alive | Auto Keep-Alive |
|--------|------------------|-----------------|
| **Start** | `./scripts/ai_keepalive.sh "Task" 30 &` | `./scripts/run_with_keepalive.sh "cmd"` |
| **Run Code** | `python train.py` | *(included)* |
| **Stop** | `pkill -f "ai_keepalive.sh"` *(often forgotten!)* | *(automatic)* |
| **Steps** | 3 | 1 |
| **Risk of Orphan** | High (if forget step 3) | Zero |
| **Exit Code** | Manual tracking | Preserved |
| **Error Cleanup** | Manual | Automatic |
| **Ctrl+C Cleanup** | Manual | Automatic |

---

## ✅ Best Practices

### **1. Always Use Quotes**
```bash
✅ ./scripts/run_with_keepalive.sh "python train.py --epochs 50"
❌ ./scripts/run_with_keepalive.sh python train.py --epochs 50
```

### **2. Use for Long-Running Tasks (>3 min)**
```bash
# Good uses:
- Training models
- Dataset generation
- Running full test suites
- EC2 long operations

# Don't need for:
- Quick scripts (<1 min)
- Simple file operations
- Short tests
```

### **3. Monitor in Separate Terminal**
```bash
# Terminal 1: Run with keep-alive
./scripts/run_with_keepalive.sh "python train.py"

# Terminal 2: Watch heartbeats
tail -f /tmp/ai_keepalive_python_train_py_*.log
```

### **4. Check Logs After Completion**
```bash
# Keep-alive log shows timing
cat /tmp/ai_keepalive_python_train_py_20260125_120000.log

# See how many heartbeats (= runtime / interval)
grep "HEARTBEAT" /tmp/ai_keepalive_*.log | wc -l
```

---

## 🎓 Migration Guide

### **If You Have Existing Scripts**

**Before:**
```bash
#!/bin/bash
./scripts/ai_keepalive.sh "Training" 30 &
KEEPALIVE_PID=$!
python train.py
kill $KEEPALIVE_PID
```

**After:**
```bash
#!/bin/bash
./scripts/run_with_keepalive.sh "python train.py"
```

**Much simpler!**

### **If Using in CI/CD**

**Before:**
```yaml
- name: Run training
  run: |
    ./scripts/ai_keepalive.sh "CI_Training" 30 &
    python train.py
    pkill -f "ai_keepalive.sh"
```

**After:**
```yaml
- name: Run training
  run: ./scripts/run_with_keepalive.sh "python train.py"
```

---

## 📁 File Reference

### **Scripts Created:**
- `scripts/run_with_keepalive.sh` - Local commands wrapper
- `scripts/run_on_ec2_with_keepalive.sh` - EC2 commands wrapper

### **Dependencies:**
- `scripts/ai_keepalive.sh` - Core keep-alive monitor

### **Log Files:**
- `/tmp/ai_keepalive_<task>_<timestamp>.log` - Keep-alive heartbeats
- Auto-cleaned after 7 days (optional manual cleanup)

---

## ✅ Testing

Try a simple test:
```bash
./scripts/run_with_keepalive.sh "sleep 5 && echo 'Done!'"
```

You should see:
1. Keep-alive starts
2. Command runs for 5 seconds
3. "Done!" appears
4. Keep-alive stops automatically
5. Clean exit

---

## 🎯 Summary

✅ **One command** to run anything with keep-alive  
✅ **Automatic cleanup** on exit (success, failure, Ctrl+C)  
✅ **No orphaned processes** - guaranteed  
✅ **Exit codes preserved** - works in scripts/CI/CD  
✅ **Works locally and on EC2**  
✅ **Simple migration** from manual approach  

**Just add `./scripts/run_with_keepalive.sh` before any command!**

---

**Status**: PRODUCTION READY  
**Tested**: 2026-01-25  
**No known bugs**: Safe to use immediately

