# Parallel Development - Master Execution Guide
**Project:** Liquid NN Integration for Mono-to-3D  
**Strategy:** 3 Workers in Parallel (Option A)  
**Timeline:** 12 days (vs 20 sequential)  
**Date:** January 28, 2026

---

## 🎯 Quick Start (5 Minutes)

```bash
# On EC2
ssh -i /Users/mike/keys/AutoGenKeyPair.pem ubuntu@204.236.245.232
cd ~/liquid_mono_to_3d

# Setup branches and infrastructure
bash scripts/parallel_setup.sh

# Start all 3 workers in tmux
bash scripts/start_parallel_workers.sh

# Attach to Worker 1 and begin
tmux attach -t worker1
bash scripts/worker1_tasks.sh  # Interactive menu
```

**That's it! You're ready to develop in parallel.**

---

## 📐 Architecture Overview

```
3 Workers Running Simultaneously in tmux:

┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Worker 1      │  │   Worker 2      │  │   Worker 3      │
│  (Fusion)       │  │  (3D Recon)     │  │ (Integration)   │
│                 │  │                 │  │                 │
│ Day 1-2: RED    │  │ Day 1-2: RED    │  │ Waits for 1&2   │
│ Day 3-4: GREEN  │  │ Day 3-4: GREEN  │  │ Day 5+: Merge   │
│ Day 5: Ready    │  │ Day 5: Ready    │  │ Day 6+: Eval    │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                   │                     │
         └───────────────────┴─────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   Monitoring    │
                    │  - Heartbeat    │
                    │  - MacBook Sync │
                    └─────────────────┘
```

---

## 📋 Complete Timeline (12 Days)

### **Day 0: Setup (Today - 30 min)**
- [x] Create directory structure
- [ ] Create git branches
- [ ] Start tmux sessions
- [ ] Begin Worker 1 & 2 in parallel

### **Days 1-2: RED Phase (Parallel)**

**Worker 1:**
```bash
tmux attach -t worker1
bash scripts/worker1_tasks.sh
# Choose: 1 (Port liquid_cell.py)
# Choose: 2 (Write tests)
# Choose: 3 (Capture RED)
# Choose: 4 (Commit)
```

**Worker 2:**
```bash
tmux attach -t worker2
bash scripts/worker2_tasks.sh
# Choose: 1 (Copy liquid_cell.py)
# Choose: 2 (Write tests)
# Choose: 3 (Capture RED)
# Choose: 4 (Commit)
```

**Worker 3:**
- Monitors Worker 1 & 2
- Prepares integration test structure

### **Days 3-4: GREEN Phase (Parallel)**

**Worker 1:**
```bash
tmux attach -t worker1
bash scripts/worker1_tasks.sh
# Choose: 5 (Implement LiquidDualModalFusion)
# Choose: 6 (Run tests - expect GREEN)
# Choose: 7 (Capture GREEN)
# Choose: 8 (Commit)
```

**Worker 2:**
```bash
tmux attach -t worker2
bash scripts/worker2_tasks.sh
# Choose: 5 (Implement Liquid3DTrajectoryReconstructor)
# Choose: 6 (Run tests - expect GREEN)
# Choose: 7 (Capture GREEN)
# Choose: 8 (Commit)
```

### **Day 5: Integration**

**Worker 3:**
```bash
tmux attach -t worker3
bash scripts/worker3_tasks.sh
# Choose: 1 (Check Worker 1 & 2 status)
# Choose: 2 (Merge Worker 1)
# Choose: 3 (Merge Worker 2)
# Choose: 5 (Write integration tests)
# Choose: 6 (Run integration tests)
```

### **Days 6-10: Evaluation (Week 2)**

**All Workers:**
- Worker 1: Fusion evaluation (20 samples)
- Worker 2: 3D evaluation (20 samples)
- Worker 3: Combined evaluation + final report

### **Days 11-12: Merge & Release**

**Worker 3:**
```bash
# Run full test suite
pytest tests/ -v

# Create proof bundle
bash scripts/prove.sh

# Merge to main
git checkout liquid-nn-integration
git merge worker/liquid-worker-3-integration
git checkout main
git merge liquid-nn-integration
git tag v1.0-liquid-nn
git push --all && git push --tags
```

---

## 🔧 Standard Processes Integration

### **1. TDD Process (Automated)**

Every worker follows RED → GREEN → REFACTOR:

```bash
# Automatic TDD capture
bash scripts/tdd_capture.sh

# Creates:
# - artifacts/worker1/tdd_red_*.txt
# - artifacts/worker1/tdd_green_*.txt
```

**Built into worker scripts** - just select menu options.

### **2. MacBook Sync (Every 5 Minutes)**

Runs automatically in monitoring session:

```bash
# Manual sync anytime:
bash scripts/sync_to_macbook.sh

# Syncs:
# - results/
# - status/
# - artifacts/
# - monitoring/
```

**View on MacBook:**
```bash
cd ~/Dropbox/Documents/.../liquid_mono_to_3d
git pull
ls results/worker1/  # See latest results
cat monitoring/status.txt  # Worker health
```

### **3. Heartbeat Monitoring (Every 30 Seconds)**

Automatically monitors all workers:

```bash
# View status anytime:
cat monitoring/status.txt
cat monitoring/heartbeat.json  # Detailed JSON

# Example output:
# Worker 1 (Fusion):      running (CPU: 45%, MEM: 12%)
# Worker 2 (3D Recon):    running (CPU: 38%, MEM: 10%)
# Worker 3 (Integration): idle (CPU: 0%, MEM: 0%)
```

### **4. Status Updates (Automatic)**

Each worker maintains status:

```bash
# View any worker status:
cat status/worker1_status.md
cat status/worker2_status.md
cat status/worker3_status.md
```

### **5. Proof Bundles**

Run at end of each phase:

```bash
bash scripts/prove.sh
# Creates: artifacts/proof/<commit-sha>/
```

---

## 🎮 tmux Command Reference

### **Access Workers:**
```bash
tmux attach -t worker1    # Liquid Fusion
tmux attach -t worker2    # Liquid 3D
tmux attach -t worker3    # Integration
tmux attach -t monitoring # Monitoring dashboard
```

### **Detach (leave running):**
```
Ctrl-b then d
```

### **List all sessions:**
```bash
tmux ls
```

### **Kill a session:**
```bash
tmux kill-session -t worker1
```

### **Create split windows:**
```
Ctrl-b then "  # Horizontal split
Ctrl-b then %  # Vertical split
```

### **Switch between panes:**
```
Ctrl-b then arrow keys
```

---

## 📊 Monitoring Dashboard

### **Real-Time Status:**
```bash
# In monitoring session
watch -n 5 cat monitoring/status.txt

# Or manually:
cat monitoring/status.txt
```

### **Worker Logs:**
```bash
tail -f logs/heartbeat.log
tail -f logs/worker1.log  # If created
tail -f logs/worker2.log
tail -f logs/worker3.log
```

### **Git Status (all branches):**
```bash
git log --all --graph --decorate --oneline | head -30
```

---

## 🔄 Daily Workflow

### **Morning (Start of Day):**
```bash
# 1. Connect to EC2
ssh -i /Users/mike/keys/AutoGenKeyPair.pem ubuntu@204.236.245.232

# 2. Check monitoring
cd ~/liquid_mono_to_3d
cat monitoring/status.txt

# 3. Pull latest changes
git fetch --all

# 4. Resume work
tmux attach -t worker1  # Or worker2, worker3
bash scripts/worker1_tasks.sh
```

### **During Day (Every Hour):**
```bash
# Commit progress
git add .
git commit -m "🚧 WIP: [description]"
git push

# Sync results (or wait for automatic sync)
bash scripts/sync_to_macbook.sh
```

### **Evening (End of Day):**
```bash
# Update status
bash scripts/worker1_tasks.sh
# Choose: 15 (Update status)

# Final commit
git add .
git commit -m "📅 EOD: Worker [N] checkpoint"
git push

# View progress
cat status/worker1_status.md
```

---

## 🆘 Troubleshooting

### **Worker not responding:**
```bash
# Check if running
tmux ls

# Reattach
tmux attach -t worker1

# If crashed, restart:
tmux kill-session -t worker1
tmux new -s worker1
cd ~/liquid_mono_to_3d
git checkout worker/liquid-worker-1-fusion
bash scripts/worker1_tasks.sh
```

### **Git conflicts:**
```bash
# View conflicts
git status

# Resolve in editor
vim <conflicted-file>

# Mark resolved
git add <file>
git commit
```

### **Tests failing unexpectedly:**
```bash
# Re-run with verbose output
pytest tests/test_liquid_fusion.py -vvs

# Check imports
python -c "import sys; sys.path.insert(0, '...'); from liquid_cell import LiquidCell"

# Check GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### **Monitoring stopped:**
```bash
# Restart monitoring
tmux attach -t monitoring
# Ctrl-c to stop old process
bash scripts/heartbeat_monitor.sh
```

---

## 📈 Success Metrics (Track These)

### **Process Metrics:**
- [ ] All 3 workers active simultaneously
- [ ] Automatic sync working (check MacBook every hour)
- [ ] Heartbeat monitoring active
- [ ] No branch conflicts during merges
- [ ] TDD evidence captured for all phases

### **Technical Metrics:**
- [ ] Worker 1: All 6 fusion tests passing
- [ ] Worker 2: All 6 3D tests passing
- [ ] Worker 3: All 5 integration tests passing
- [ ] Combined: Hallucination < 18%
- [ ] Combined: Description quality ≥ 8.5/10
- [ ] Combined: 3D smoothness ≥ 8/10

### **Timeline Metrics:**
- [ ] RED phase complete: Day 2
- [ ] GREEN phase complete: Day 4
- [ ] Integration complete: Day 5
- [ ] Evaluation complete: Day 10
- [ ] Merge to main: Day 12

**Target: 12 days total (vs 20 sequential = 40% faster)**

---

## 🎓 Tips for Success

### **1. Stay in Sync**
- Pull from origin frequently
- Commit small, often
- Push at end of each task

### **2. Use tmux Effectively**
- Keep all 4 sessions open
- Detach, don't close
- Use named sessions

### **3. Monitor Health**
- Check monitoring/status.txt hourly
- Review logs for errors
- Watch CPU/Memory usage

### **4. Communicate via Status**
- Update status files frequently
- Use clear commit messages
- Tag commits: 🔴 RED, ✅ GREEN, 🔄 SYNC

### **5. Follow TDD Strictly**
- Always write tests FIRST
- Capture RED evidence
- Capture GREEN evidence
- No exceptions!

---

## 📞 Quick Reference Card

```
┌─────────────────────────────────────────────────┐
│ PARALLEL DEVELOPMENT QUICK REFERENCE            │
├─────────────────────────────────────────────────┤
│ START SYSTEM:                                   │
│   bash scripts/start_parallel_workers.sh        │
│                                                 │
│ ACCESS WORKERS:                                 │
│   tmux attach -t worker1                        │
│   tmux attach -t worker2                        │
│   tmux attach -t worker3                        │
│   tmux attach -t monitoring                     │
│                                                 │
│ WORKER TASKS:                                   │
│   bash scripts/worker1_tasks.sh                 │
│   bash scripts/worker2_tasks.sh                 │
│   bash scripts/worker3_tasks.sh                 │
│                                                 │
│ MONITORING:                                     │
│   cat monitoring/status.txt                     │
│   cat status/worker1_status.md                  │
│                                                 │
│ SYNC:                                           │
│   bash scripts/sync_to_macbook.sh               │
│                                                 │
│ TDD:                                            │
│   bash scripts/tdd_capture.sh                   │
│                                                 │
│ PROOF:                                          │
│   bash scripts/prove.sh                         │
└─────────────────────────────────────────────────┘
```

---

## ✅ Pre-Flight Checklist

Before starting parallel development:

- [ ] EC2 Spot instance running (204.236.245.232)
- [ ] All scripts synced to EC2
- [ ] Scripts are executable (chmod +x)
- [ ] Git configured (user.name, user.email)
- [ ] liquid_ai_2 project accessible on MacBook
- [ ] liquid_cell.py ready to copy
- [ ] tmux installed on EC2
- [ ] pytest installed
- [ ] All directories created

---

**You're ready for parallel development! Start with:**
```bash
bash scripts/start_parallel_workers.sh
```

**All automation is in place. Focus on implementation!** 🚀

