# AI Agent Keep-Alive & Monitoring System

**Purpose**: Prevent Cursor AI from hanging during long operations and provide visible progress

**Created**: 2026-01-25  
**Status**: PRODUCTION READY

---

## 🎯 Problem Statement

During long AI operations (multiple tool calls, complex analysis, long-running tasks), the AI agent can:
- Hang without output
- Time out due to inactivity
- Get stuck in loops
- Provide no visibility into progress

This monitoring system provides:
- ✅ Periodic heartbeat to keep system active
- ✅ Watchdog to detect hangs
- ✅ Progress dashboard for multi-task monitoring
- ✅ Automatic logging and diagnostics

---

## 📦 Components

### 1. **ai_keepalive.sh** - Heartbeat Generator
Provides periodic "I'm alive" signals to prevent hanging.

**Features**:
- Outputs heartbeat every N seconds (default: 30s)
- Creates log file for audit trail
- Updates shared heartbeat file for other processes
- Visible output in terminal

**Usage**:
```bash
# Basic usage (30s interval)
./scripts/ai_keepalive.sh "Dataset_Generation"

# Custom interval (15s)
./scripts/ai_keepalive.sh "Training" 15

# Run in background (recommended)
./scripts/ai_keepalive.sh "Analysis" 30 &
```

### 2. **ai_watchdog.sh** - Hang Detector
Monitors for AI hangs and alerts when no progress detected.

**Features**:
- Checks heartbeat file freshness
- Alerts if no activity for threshold period (default: 5 min)
- System diagnostics (processes, load)
- Automatic recovery suggestions

**Usage**:
```bash
# Basic usage (60s checks, 5min threshold)
./scripts/ai_watchdog.sh

# Custom settings (30s checks, 3min threshold)
./scripts/ai_watchdog.sh 30 180

# Run in background
./scripts/ai_watchdog.sh &
```

### 3. **ai_progress_tracker.sh** - Multi-Task Dashboard
Visual dashboard showing all active AI operations.

**Features**:
- Real-time status of all monitors
- EC2 process status
- System resource monitoring
- Log file tracking
- Auto-refresh every 10s

**Usage**:
```bash
# Start dashboard (refreshes automatically)
./scripts/ai_progress_tracker.sh

# Will display live dashboard until Ctrl+C
```

---

## 🚀 Quick Start

### Scenario 1: Starting a Long AI Task

**Terminal 1** - Start Keep-Alive:
```bash
cd ~/Dropbox/.../mono_to_3d
./scripts/ai_keepalive.sh "Long_Analysis" 30 &
```

**Terminal 2** - Monitor Progress:
```bash
./scripts/ai_progress_tracker.sh
```

**Terminal 3** - Your actual work (Cursor AI commands)
- AI does its work
- Keep-alive provides heartbeats
- Dashboard shows progress

### Scenario 2: EC2 Long-Running Process

**Terminal 1** - Local Keep-Alive:
```bash
./scripts/ai_keepalive.sh "EC2_Training" 60 &
```

**Terminal 2** - EC2 Process:
```bash
ssh -i ~/keys/AutoGenKeyPair.pem ubuntu@34.196.155.11
cd ~/mono_to_3d/experiments/magvit_I3D_LLM_basic_trajectory
python train_magvit.py
```

**Terminal 3** - Dashboard:
```bash
./scripts/ai_progress_tracker.sh
```

### Scenario 3: Multi-Task AI Work

**Terminal 1** - Multiple Keep-Alives:
```bash
./scripts/ai_keepalive.sh "Task_A" 30 &
./scripts/ai_keepalive.sh "Task_B" 45 &
./scripts/ai_keepalive.sh "Task_C" 60 &
```

**Terminal 2** - Watchdog:
```bash
./scripts/ai_watchdog.sh 60 300 &
```

**Terminal 3** - Dashboard:
```bash
./scripts/ai_progress_tracker.sh
```

---

## 📊 Output Examples

### Keep-Alive Output:
```
🔴 AI Keep-Alive Monitor Started
Task: Dataset_Generation
Interval: 30s
Log: /tmp/ai_keepalive_Dataset_Generation_20260125_143000.log
=========================================

⏱️  HEARTBEAT #1 [0m 0s] - 2026-01-25 14:30:00 - Dataset_Generation - System Active ✅
⏱️  HEARTBEAT #2 [0m 30s] - 2026-01-25 14:30:30 - Dataset_Generation - System Active ✅
⏱️  HEARTBEAT #3 [1m 0s] - 2026-01-25 14:31:00 - Dataset_Generation - System Active ✅
```

### Watchdog Alert:
```
🔍 AI Watchdog Monitor Started
Check Interval: 60s
Hang Threshold: 300s
=========================================

✅ [2026-01-25 14:30:15] AI Active - Last heartbeat: 15s ago
✅ [2026-01-25 14:31:15] AI Active - Last heartbeat: 45s ago

🚨 [2026-01-25 14:36:15] ALERT #1: AI May Be Hung!
   Last heartbeat: 330s ago (threshold: 300s)
   Heartbeat file: /tmp/ai_heartbeat.txt

   Diagnostic Actions:
   1. Check terminal output above for last activity
   2. Look for error messages
   3. Consider interrupting (Ctrl+C) and restarting
   4. Check: tail -20 /tmp/ai_heartbeat.txt
```

### Progress Tracker Dashboard:
```
╔════════════════════════════════════════════════════════════════╗
║         AI PROGRESS TRACKER - MULTI-TASK DASHBOARD            ║
║                2026-01-25 14:35:00                            ║
╚════════════════════════════════════════════════════════════════╝

🟢 ACTIVE KEEP-ALIVE MONITORS:
   PIDs: 12345 12346
   └─ 12345: ai_keepalive.sh Dataset_Generation 30
   └─ 12346: ai_keepalive.sh Training 60

💓 HEARTBEAT STATUS:
   Last Heartbeat:
   │  ⏱️  HEARTBEAT #7 [3m 30s] - Dataset_Generation - System Active ✅
   └─ Status: ✅ Active (15s ago)

🖥️  EC2 PROCESSES (via SSH):
   │  ubuntu   18132  0.1  python train_magvit.py
   │  ubuntu   18142  10.2 python -c multiprocessing.spawn

📊 LOCAL SYSTEM STATUS:
   CPU Usage: 25.0%
   Memory: 8.2GB used
   Disk: 45% used

📝 RECENT LOGS:
   Found 2 log file(s):
   │  -rw-r--r-- 1 mike staff 2.1K Jan 25 14:35 ai_keepalive_Training_*.log
   │  -rw-r--r-- 1 mike staff 1.8K Jan 25 14:30 ai_keepalive_Dataset_*.log

════════════════════════════════════════════════════════════════
Press Ctrl+C to stop monitoring | Refresh: 10s
```

---

## 🔧 Integration with Cursor AI Workflow

### When to Use:

1. **Before Starting Long Analysis**
   ```bash
   ./scripts/ai_keepalive.sh "Requirements_Review" 30 &
   # Then ask AI to review requirements
   ```

2. **Before EC2 Long-Running Tasks**
   ```bash
   ./scripts/ai_keepalive.sh "EC2_Training" 60 &
   # Then start EC2 training
   ```

3. **During Multi-File Code Generation**
   ```bash
   ./scripts/ai_keepalive.sh "Code_Gen" 45 &
   # Then ask AI to generate multiple files
   ```

4. **During TDD Workflows**
   ```bash
   ./scripts/ai_keepalive.sh "TDD_Cycle" 30 &
   # Then run full TDD cycle
   ```

### Best Practices:

1. **Always start keep-alive BEFORE the long task**
2. **Use descriptive task names** (helps with log identification)
3. **Choose appropriate intervals**:
   - 15-30s for very long tasks (>10 min)
   - 30-60s for medium tasks (5-10 min)
   - 60-120s for background monitoring
4. **Monitor dashboard in separate terminal** (optional but recommended)
5. **Stop keep-alive when task completes**:
   ```bash
   pkill -f "ai_keepalive.sh"
   ```

---

## 🛠️ Troubleshooting

### Problem: Keep-Alive Not Starting
```bash
# Check if script is executable
ls -l scripts/ai_keepalive.sh

# If not, make executable
chmod +x scripts/ai_keepalive.sh
```

### Problem: No Heartbeat File
```bash
# Check if keep-alive is running
ps aux | grep ai_keepalive

# Check logs
ls -lht /tmp/ai_keepalive_*.log | head -3
tail /tmp/ai_keepalive_*.log
```

### Problem: Dashboard Shows "Unable to connect to EC2"
```bash
# Test EC2 connection
ssh -i /Users/mike/keys/AutoGenKeyPair.pem ubuntu@34.196.155.11 'echo OK'

# If fails, check EC2 status or network
```

### Problem: Watchdog Keeps Alerting
```bash
# Check if keep-alive is actually running
ps aux | grep ai_keepalive

# Check heartbeat file age
ls -lh /tmp/ai_heartbeat.txt
cat /tmp/ai_heartbeat.txt

# If keep-alive died, restart it
./scripts/ai_keepalive.sh "Restart" 30 &
```

---

## 📁 File Locations

### Scripts:
- `scripts/ai_keepalive.sh` - Heartbeat generator
- `scripts/ai_watchdog.sh` - Hang detector
- `scripts/ai_progress_tracker.sh` - Dashboard

### Runtime Files:
- `/tmp/ai_keepalive_*.log` - Keep-alive logs (one per invocation)
- `/tmp/ai_heartbeat.txt` - Latest heartbeat (shared between processes)
- `/tmp/ai_last_activity.txt` - Last activity timestamp

### Cleanup:
```bash
# Remove old logs (older than 7 days)
find /tmp -name "ai_keepalive_*.log" -mtime +7 -delete

# Remove all logs
rm /tmp/ai_keepalive_*.log

# Remove heartbeat file
rm /tmp/ai_heartbeat.txt
```

---

## 🎯 Implementation in Cursorrules

**Add to cursorrules:**

```markdown
🚨 AI AGENT KEEP-ALIVE REQUIREMENT (MANDATORY) 🚨

For ANY task estimated to take >3 minutes, AI MUST:

1. Instruct user to start keep-alive:
   "./scripts/ai_keepalive.sh <task_name> 30 &"

2. Provide periodic status in responses

3. If task involves multiple phases, break into separate responses

Keep-Alive Benefits:
- Prevents AI hanging
- Provides visible progress
- Enables hang detection
- Creates audit trail
```

---

## 🚀 Advanced Usage

### Scenario: Multiple Parallel Tasks

```bash
# Terminal 1 - Keep-alives for multiple tasks
./scripts/ai_keepalive.sh "Task_A_Data_Gen" 30 &
./scripts/ai_keepalive.sh "Task_B_Training" 60 &
./scripts/ai_keepalive.sh "Task_C_Analysis" 45 &

# Terminal 2 - Watchdog with tight monitoring
./scripts/ai_watchdog.sh 30 180 &  # Check every 30s, alert at 3min

# Terminal 3 - Dashboard
./scripts/ai_progress_tracker.sh

# Terminal 4 - Actual work
# <Do your AI-assisted work here>
```

### Scenario: EC2 + Local Work

```bash
# Terminal 1 - Local keep-alive
./scripts/ai_keepalive.sh "Local_Analysis" 30 &

# Terminal 2 - EC2 SSH with keep-alive
ssh -i ~/keys/AutoGenKeyPair.pem ubuntu@34.196.155.11 \
  'cd ~/mono_to_3d && \
   ./scripts/ai_keepalive.sh "EC2_Training" 60 & \
   python train_magvit.py'

# Terminal 3 - Combined dashboard
./scripts/ai_progress_tracker.sh
```

---

## ✅ Checklist

Before starting long AI task:
- [ ] Identify task name and estimated duration
- [ ] Start keep-alive: `./scripts/ai_keepalive.sh <name> <interval> &`
- [ ] Optional: Start watchdog in background
- [ ] Optional: Start dashboard in separate terminal
- [ ] Begin AI task
- [ ] Monitor progress via dashboard or heartbeat output
- [ ] When complete, stop keep-alive: `pkill -f ai_keepalive`

---

**Status**: PRODUCTION READY  
**Tested**: 2026-01-25  
**Maintainer**: Project Team

