# Next Run Recommendations

**Date**: 2026-01-30  
**Current Status**: 80% Complete - VLM Quality is the Only Blocker  
**Last Session**: January 28, 2026 (Workers 2-5 completed)

---

## 🎯 **Top Priority: GPT-4 Baseline Evaluation**

**Status**: ⏰ **CRITICAL PATH** - Everything else depends on this

### Why This Matters
Current state:
- ✅ Liquid NN integration: 100% complete (99% jitter reduction)
- ✅ Real data pipeline: 100% complete
- ✅ Infrastructure: Production-ready
- ❌ TinyLlama VLM: Only 35% accuracy ← **BLOCKER**

**Need to know**: Is this a TinyLlama problem or an architecture problem?

### Action Items

**1. Get OpenAI API Key**
```
Current status: Partial key found
Location: mono_to_3d/experiments/.../demo_full_output.log (line 11)
Key format: sk-proj-Nae9JoShWsxa...
Need: Full key from user
```

**2. Launch EC2 and Set Key**
```bash
# Launch via Auto Scaling Group
AWS Console → Auto Scaling → GPU G5 spot – ASG → Set Desired Capacity: 1

# Connect (after 2-5 min)
ssh -i ~/keys/AutoGenKeyPair.pem ubuntu@<NEW_IP>

# Setup
cd ~/liquid_mono_to_3d
source ~/mono_to_3d_env/bin/activate
git pull origin main
export OPENAI_API_KEY="sk-proj-[FULL_KEY]"
```

**3. Run Evaluation**
```bash
cd ~/liquid_mono_to_3d
python3 experiments/liquid_vlm_integration/evaluate_vlm_accuracy.py
```

**4. Review Results**
```bash
# Check latest evaluation
cat experiments/liquid_vlm_integration/results/$(ls -t experiments/liquid_vlm_integration/results/*evaluation*.json | head -1)

# Sync to MacBook
rsync -avz experiments/liquid_vlm_integration/results/ \
  mike@macbook:~/Dropbox/.../liquid_mono_to_3d/experiments/liquid_vlm_integration/results/
```

### Decision Tree

```
GPT-4 Result?
│
├─ ≥ 80% accuracy ✅
│  └─> TinyLlama is the problem
│     └─> Next: Fine-tune TinyLlama (2-3 days)
│        ├─ Generate 1000+ trajectory descriptions
│        ├─ Fine-tune with LoRA
│        └─ Deploy for $0/request (vs GPT-4 $30/1000)
│
├─ 50-80% accuracy ⚠️
│  └─> Prompting + model size issue
│     └─> Next: Improve prompting OR use GPT-4 for production
│
└─ < 50% accuracy ❌
   └─> Architecture problem (embeddings/fusion)
      └─> Next: Revisit visual grounding approach
```

**Time**: 1-2 hours  
**Cost**: ~$0.30 (10 samples × $0.03/request)

---

## 📊 **Priority 2: Create Liquid NN Visualizations**

**Status**: ⏳ **HIGH VALUE** - Need visual evidence

### Why This Matters
- Have 99% jitter reduction (text evidence only)
- No visualizations showing Liquid NN performance
- Critical for validation and documentation

### What to Create

**Visualization 1: Noisy vs. Smooth Trajectory**
```python
# File: experiments/liquid_vlm_integration/create_trajectory_viz.py
# Output: results/20260131_HHMM_liquid_trajectory_comparison.png

# 3-panel plot:
# - Left: 3D trajectory (noisy in red, smooth in blue)
# - Middle: XY projection (top view)
# - Right: Jerk over time (before/after)
# Annotations: "Jerk: 0.010879 → 0.000112 (99% reduction)"
```

**Visualization 2: Multi-Sample Grid**
```python
# File: experiments/liquid_vlm_integration/create_trajectory_grid.py
# Output: results/20260131_HHMM_liquid_nn_performance_grid.png

# 3×3 grid showing:
# - 9 different trajectory types
# - Each: noisy (gray) + smooth (blue) overlaid
# - Demonstrates consistency across samples
```

**Visualization 3: Jitter Reduction Analysis**
```python
# File: experiments/liquid_vlm_integration/create_jitter_analysis.py
# Output: results/20260131_HHMM_jitter_reduction_analysis.png

# Multi-panel showing:
# - Top: Position over time (x, y, z)
# - Middle: Velocity (1st derivative)
# - Bottom: Acceleration/Jerk (2nd/3rd derivative)
# All panels: noisy vs smooth comparison
```

### Implementation Script

```python
#!/usr/bin/env python3
"""
Create Liquid NN trajectory visualizations
Following TDD and output naming conventions
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from simple_3d_tracker import generate_synthetic_tracks, triangulate_tracks, set_up_cameras

# Import Liquid NN (on EC2)
from experiments.trajectory_video_understanding.vision_language_integration.liquid_3d_reconstructor import (
    Liquid3DTrajectoryReconstructor
)
import torch

def create_trajectory_comparison():
    """Create noisy vs smooth trajectory visualization."""
    # Generate real data
    P1, P2 = set_up_cameras()
    sensor1, sensor2, original_3d = generate_synthetic_tracks()
    reconstructed = triangulate_tracks(sensor1, sensor2, P1, P2)
    
    # Add realistic noise
    noise = np.random.randn(*reconstructed.shape) * 0.01  # 10mm noise
    noisy_3d = reconstructed + noise
    
    # Apply Liquid NN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reconstructor = Liquid3DTrajectoryReconstructor(
        input_dim=3, hidden_dim=64, output_feature_dim=256, dt=0.033
    ).to(device)
    
    noisy_tensor = torch.tensor(noisy_3d, dtype=torch.float32).unsqueeze(0).to(device)
    _, smooth_3d = reconstructor(noisy_tensor)
    smooth_3d = smooth_3d[0].cpu().detach().numpy()
    
    # Calculate jerk
    noisy_jerk = np.abs(np.diff(noisy_3d, n=2, axis=0)).mean()
    smooth_jerk = np.abs(np.diff(smooth_3d, n=2, axis=0)).mean()
    reduction = (noisy_jerk - smooth_jerk) / noisy_jerk * 100
    
    # Create figure
    fig = plt.figure(figsize=(15, 5))
    
    # Panel 1: 3D trajectory
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(noisy_3d[:, 0], noisy_3d[:, 1], noisy_3d[:, 2], 
             'r-', alpha=0.5, linewidth=1, label='Noisy (triangulated)')
    ax1.plot(smooth_3d[:, 0], smooth_3d[:, 1], smooth_3d[:, 2], 
             'b-', linewidth=2, label='Smooth (Liquid NN)')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: XY projection
    ax2 = fig.add_subplot(132)
    ax2.plot(noisy_3d[:, 0], noisy_3d[:, 1], 'r-', alpha=0.5, linewidth=1, label='Noisy')
    ax2.plot(smooth_3d[:, 0], smooth_3d[:, 1], 'b-', linewidth=2, label='Smooth')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top View (XY Projection)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Panel 3: Jerk over time
    ax3 = fig.add_subplot(133)
    noisy_jerk_t = np.abs(np.diff(noisy_3d, n=2, axis=0)).sum(axis=1)
    smooth_jerk_t = np.abs(np.diff(smooth_3d, n=2, axis=0)).sum(axis=1)
    time = np.arange(len(noisy_jerk_t))
    ax3.plot(time, noisy_jerk_t, 'r-', alpha=0.7, label='Noisy')
    ax3.plot(time, smooth_jerk_t, 'b-', linewidth=2, label='Smooth')
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Jerk (m/frame³)')
    ax3.set_title(f'Jerk Over Time\n{reduction:.1f}% Reduction')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add summary text
    fig.suptitle(f'Liquid Neural Network Trajectory Smoothing\n'
                f'Noisy Jerk: {noisy_jerk:.6f} → Smooth Jerk: {smooth_jerk:.6f} '
                f'({reduction:.1f}% reduction)', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save with proper naming convention
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_path = Path('experiments/liquid_vlm_integration/results') / \
                  f'{timestamp}_liquid_trajectory_comparison.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    
    return output_path

if __name__ == "__main__":
    create_trajectory_comparison()
```

**Time**: 2-3 hours  
**Output**: 3-5 PNG files in `experiments/liquid_vlm_integration/results/`

---

## 🔧 **Priority 3: Quick Win - Improve TinyLlama Prompting**

**Status**: ⚡ **QUICK WIN** - 30-60 minutes, may improve 35% → 50-60%

### Current Prompt (Too Generic)
```python
prompt = "Describe this 3D trajectory."
```

### Improved Prompt (Structured)
```python
prompt = """You are analyzing a 3D trajectory from stereo camera tracking.

Describe ONLY what you observe about:
1. Path shape: Is it straight, curved, circular, or spiral?
2. Direction: Which axis shows the most movement (X, Y, or Z)?
3. Start/End: Approximate coordinates where the path begins and ends
4. Motion speed: Is it moving fast, slow, or changing speed?

Be factual and specific. Use only what you see in the data.
Do NOT mention: videos, URLs, tutorials, or make up information.

Trajectory:"""
```

### Implementation

**Edit**: `experiments/liquid_vlm_integration/tinyllama_vlm.py`

```python
# Find line with:
# prompt = "Describe this 3D trajectory."

# Replace with structured prompt above
```

**Test**:
```bash
python3 experiments/liquid_vlm_integration/evaluate_vlm_accuracy.py
```

**Expected**: 35% → 50-60% accuracy (not production-ready, but validates approach)

---

## 📋 **Complete Session Checklist**

### Pre-Session (MacBook)
- [ ] Confirm OpenAI API key available
- [ ] Review `SHUTDOWN_STATUS_20260128.md` for context
- [ ] Check AWS console access

### Launch EC2 (5 minutes)
```bash
# Via AWS Console
1. Go to: https://console.aws.amazon.com/ec2/autoscaling/
2. Region: us-east-1
3. Select: "GPU G5 spot – ASG"
4. Edit → Set Desired Capacity: 1
5. Wait 2-5 min for launch
6. EC2 → Instances → Get new Public IPv4 address
```

### Connect & Setup (5 minutes)
```bash
# From MacBook
ssh -i ~/keys/AutoGenKeyPair.pem ubuntu@<NEW_IP>

# On EC2
cd ~/liquid_mono_to_3d
source ~/mono_to_3d_env/bin/activate
git pull origin main
git status  # Verify clean state

# Set API key (USER PROVIDED)
export OPENAI_API_KEY="sk-proj-[FULL_KEY_HERE]"
```

### Priority 1: GPT-4 Baseline (1-2 hours)
```bash
# Run evaluation
python3 experiments/liquid_vlm_integration/evaluate_vlm_accuracy.py

# Review results
ls -lt experiments/liquid_vlm_integration/results/*evaluation*.json | head -5
cat experiments/liquid_vlm_integration/results/$(ls -t experiments/liquid_vlm_integration/results/*evaluation*.json | head -1)

# Sync to MacBook
rsync -avz experiments/liquid_vlm_integration/results/ \
  mike@macbook:~/Dropbox/.../liquid_mono_to_3d/experiments/liquid_vlm_integration/results/
```

### Priority 2: Create Visualizations (2-3 hours)
```bash
# Create visualization script
vim experiments/liquid_vlm_integration/create_trajectory_viz.py
# [Paste implementation from above]

# Following TDD
cd experiments/liquid_vlm_integration
mkdir -p tests

# Write tests FIRST
vim tests/test_trajectory_viz.py

# Run TDD cycle
bash ~/liquid_mono_to_3d/scripts/tdd_capture.sh  # RED
python3 create_trajectory_viz.py  # Implement
bash ~/liquid_mono_to_3d/scripts/tdd_capture.sh  # GREEN

# Verify outputs
ls -lh results/*liquid_trajectory*.png

# Sync to MacBook
rsync -avz results/ mike@macbook:~/Dropbox/.../liquid_mono_to_3d/experiments/liquid_vlm_integration/results/
```

### Priority 3: Improve Prompting (30-60 min)
```bash
# Edit TinyLlama VLM
vim experiments/liquid_vlm_integration/tinyllama_vlm.py
# [Update prompt as shown above]

# Rerun evaluation
python3 experiments/liquid_vlm_integration/evaluate_vlm_accuracy.py

# Compare results
diff <old_results> <new_results>
```

### Documentation (15 minutes)
```bash
# Create session summary
vim SESSION_SUMMARY_$(date +%Y%m%d).md

# Key sections:
# - GPT-4 baseline result
# - Decision for next phase
# - Visualizations created
# - Prompting improvement results
```

### Commit & Push (5 minutes)
```bash
git add experiments/liquid_vlm_integration/
git add artifacts/
git commit -m "feat: GPT-4 baseline, Liquid NN viz, improved prompting"
git push origin main
```

### Shutdown EC2 (5 minutes)
```bash
# Via AWS Console
1. Go to: https://console.aws.amazon.com/ec2/autoscaling/
2. Select: "GPU G5 spot – ASG"
3. Edit → Set Desired Capacity: 0
4. Set Minimum Capacity: 0
5. Wait 1-2 min for termination confirmation
```

---

## 🎯 **Success Criteria**

| Priority | Goal | Target | Status |
|----------|------|--------|--------|
| **1** | GPT-4 Baseline | Record accuracy (expect 80%+) | ⏳ Pending |
| **2** | Liquid NN Viz | 3-5 PNG files created | ⏳ Pending |
| **3** | TinyLlama Prompting | 35% → 50%+ improvement | ⏳ Pending |
| **Bonus** | Documentation | Session summary created | ⏳ Pending |

---

## 💡 **Critical Decision Point**

**After GPT-4 baseline**, you'll know the path forward:

### Path A: GPT-4 ≥ 80% (Most Likely)
```
Next steps (2-3 days):
1. Generate 1000+ trajectory descriptions (1 day)
2. Fine-tune TinyLlama with LoRA (1 day)
3. Evaluate fine-tuned model (0.5 day)
4. Deploy to production (0.5 day)

Result: Self-hosted VLM with 80%+ accuracy, $0/request
```

### Path B: GPT-4 = 50-80%
```
Next steps (1 day):
1. Try few-shot prompting (0.5 day)
2. If insufficient, use GPT-4 for production (0.5 day)

Result: Good accuracy, $30/1000 requests (acceptable for low volume)
```

### Path C: GPT-4 < 50% (Unlikely)
```
Next steps (1 week):
1. Debug visual grounding (2 days)
2. Test MagVIT embeddings quality (1 day)
3. Verify Liquid fusion vs static (1 day)
4. Revisit architecture if needed (2-3 days)

Result: Fix fundamental issue before production
```

---

## 📊 **Time Estimates**

| Session Component | Time |
|------------------|------|
| EC2 Launch & Setup | 10 min |
| GPT-4 Baseline | 1-2 hours |
| Visualizations | 2-3 hours |
| Prompting Improvement | 30-60 min |
| Documentation | 15 min |
| Commit & Shutdown | 10 min |
| **Total** | **4-6 hours** |

---

## 🔗 **Quick Reference Links**

**Documentation**:
- `LIQUID_NN_INTEGRATION_RESULTS_SUMMARY.md` - Current status
- `SHUTDOWN_STATUS_20260128.md` - Last session state
- `JITTER_METRIC_EXPLAINED.md` - Liquid NN explanation
- `LIQUID_NN_VS_NCP_COMPARISON.md` - Architecture details

**Code**:
- `experiments/liquid_vlm_integration/evaluate_vlm_accuracy.py` - VLM evaluation
- `experiments/liquid_vlm_integration/tinyllama_vlm.py` - TinyLlama wrapper
- `experiments/liquid_vlm_integration/gpt4_vlm.py` - GPT-4 wrapper

**Results**:
- `experiments/liquid_vlm_integration/results/` - All outputs
- `artifacts/` - TDD evidence

---

## 🚀 **Ready to Start?**

**Immediate next action**: Get OpenAI API key, then follow Priority 1 checklist above.

**Remember**:
- ✅ Follow TDD (RED → GREEN → REFACTOR)
- ✅ Use proper file naming (`YYYYMMDD_HHMM_description.ext`)
- ✅ Sync results to MacBook periodically
- ✅ Commit frequently with descriptive messages
- ✅ Set ASG capacity to 0 when done

---

**Created**: 2026-01-30  
**Status**: Ready for next session  
**Blocker**: OpenAI API key needed


