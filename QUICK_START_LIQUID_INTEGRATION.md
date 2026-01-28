# Quick Start: Liquid AI Integration

**TL;DR:** Your `liquid_ai_2` project has production-ready Liquid Neural Network code. Port it to add true "Liquid" dynamics to your trajectory models.

---

## 🎯 The Opportunity

**What you have:**
- ✅ `liquid_mono_to_3d`: MagVIT model (100% accuracy), stereo 3D tracking, VLM integration
- ✅ `liquid_ai_2`: LiquidCell with closed-form adjoint, MagVIT+Liquid patterns, VLM fusion

**What's missing:**
- ❌ No Liquid dynamics in `liquid_mono_to_3d` yet (despite the name!)

**The win:**
Merge both projects → Get continuous-time dynamics for trajectory understanding.

---

## 🚀 One-Week Quickstart

### Day 1: Port LiquidCell

```bash
# On MacBook (file management)
cd ~/Dropbox/Documents/.../liquid_mono_to_3d
mkdir -p experiments/trajectory_video_understanding/liquid_models

# Copy core LNN implementation
cp ~/Dropbox/Code/repos/liquid_ai_2/option1_synthetic/liquid_cell.py \
   experiments/trajectory_video_understanding/liquid_models/

# Commit
git add experiments/trajectory_video_understanding/liquid_models/
git commit -m "Add LiquidCell from liquid_ai_2 project"
git push
```

### Day 2-3: Write Tests (TDD - MANDATORY)

```bash
# On EC2
ssh -i /Users/mike/keys/AutoGenKeyPair.pem ubuntu@34.196.155.11
cd ~/mono_to_3d
git pull

# Following TDD per cursorrules
cd experiments/trajectory_video_understanding

# Write tests FIRST (RED phase)
cat > tests/test_liquid_cell.py << 'EOF'
import torch
import pytest
import sys
sys.path.insert(0, '../liquid_models')
from liquid_cell import LiquidCell

def test_liquid_cell_forward_shape():
    """LiquidCell produces correct output shape."""
    cell = LiquidCell(input_size=512, hidden_size=64, dt=0.02)
    x = torch.randn(4, 512)  # Batch=4, MagVIT embedding dim
    h = torch.zeros(4, 64)
    
    h_new = cell(x, h)
    
    assert h_new.shape == (4, 64), f"Expected (4, 64), got {h_new.shape}"

def test_liquid_cell_gradients_flow():
    """Backpropagation works through LiquidCell."""
    cell = LiquidCell(input_size=512, hidden_size=64, dt=0.02)
    x = torch.randn(4, 512, requires_grad=True)
    h = torch.zeros(4, 64)
    
    h_new = cell(x, h)
    loss = h_new.sum()
    loss.backward()
    
    assert x.grad is not None, "Gradients didn't flow"
    assert torch.isfinite(x.grad).all(), "NaN gradients"

def test_liquid_cell_continuous_dynamics():
    """Small dt produces continuous outputs."""
    cell = LiquidCell(input_size=64, hidden_size=32, dt=0.01)
    x = torch.randn(1, 64)
    h = torch.zeros(1, 32)
    
    h1 = cell(x, h)
    h2 = cell(x, h1)
    
    # Difference should be small (continuous)
    diff = (h2 - h1).abs().max()
    assert diff < 1.0, f"Discontinuous: diff={diff}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
EOF

# Capture RED phase (tests should fail - no implementation yet in test context)
bash ../../scripts/tdd_capture.sh
# Verify artifacts/tdd_red.txt exists
```

### Day 3-4: Verify and Test

```bash
# GREEN phase - tests should now pass
python -m pytest tests/test_liquid_cell.py -v
bash ../../scripts/tdd_capture.sh

# Verify artifacts/tdd_green.txt shows PASSES
cat ../../artifacts/tdd_green.txt
```

### Day 5: Create Proof Bundle

```bash
cd ~/mono_to_3d
bash scripts/prove.sh
echo $?  # Should be 0

# Check proof
git rev-parse HEAD  # Get SHA
ls artifacts/proof/<SHA>/
```

**Week 1 Deliverable:** ✅ LiquidCell working with TDD evidence

---

## 🎯 What to Do Next (Week 2+)

### Option A: Integrate with MagVIT (Recommended)

Connect your 100% accuracy MagVIT to Liquid dynamics:

```python
# experiments/trajectory_video_understanding/liquid_models/liquid_trajectory.py

class LiquidTrajectoryClassifier(nn.Module):
    def __init__(self, magvit_checkpoint):
        self.magvit = load_pretrained(magvit_checkpoint)
        self.magvit.eval()  # Freeze
        
        self.liquid = LiquidCell(512, 64, dt=0.02)
        self.classifier = nn.Linear(64, 2)  # Binary: persistent/transient
    
    def forward(self, video):
        # Your trained MagVIT
        with torch.no_grad():
            features = self.magvit.encode(video)  # [B, 512]
        
        # Add Liquid dynamics
        h = torch.zeros(features.shape[0], 64, device=features.device)
        h = self.liquid(features, h)
        
        # Classification
        logits = self.classifier(h)
        return logits
```

**Goal:** Match or exceed 100% accuracy with continuous-time dynamics.

### Option B: Visual Grounding with Liquid VLM

Port the VLM fusion from liquid_ai_2:

```python
# Copy from liquid_ai_2
cp ~/Dropbox/Code/repos/liquid_ai_2/magvit_integration/option4_liquid_vlm/liquid_vlm_fusion.py \
   experiments/trajectory_video_understanding/vision_language_integration/
```

Then integrate with your TinyLlama VLM.

### Option C: Continuous 3D Trajectory Prediction

Use Liquid dynamics for smooth 3D motion:

```python
class Liquid3DTracker(nn.Module):
    def __init__(self):
        self.liquid = LiquidCell(3, 64, dt=0.02)  # 3D position input
        self.predictor = nn.Linear(64, 3)  # Next position
    
    def forward(self, position_3d, h):
        h_new = self.liquid(position_3d, h)
        next_pos = self.predictor(h_new)
        return next_pos, h_new
```

---

## 📊 Key Files Reference

### Copy from liquid_ai_2:
```
liquid_ai_2/option1_synthetic/liquid_cell.py                    → Core LNN
liquid_ai_2/magvit_integration/option1_*/magvit_liquid_*.py    → Integration pattern
liquid_ai_2/magvit_integration/option4_*/liquid_vlm_fusion.py  → VLM fusion
```

### Add to liquid_mono_to_3d:
```
experiments/trajectory_video_understanding/
├── liquid_models/
│   ├── liquid_cell.py              ← Copy from liquid_ai_2
│   ├── liquid_trajectory.py        ← New architecture
│   └── test_liquid_*.py            ← TDD tests
└── vision_language_integration/
    └── liquid_visual_grounding.py  ← Copy fusion from liquid_ai_2
```

---

## ⚠️ Critical Reminders

### 1. TDD is Mandatory (Memory ID: 13642272)
```
ALWAYS:
1. Write tests FIRST
2. Capture RED phase: bash scripts/tdd_capture.sh
3. Implement code
4. Capture GREEN phase: bash scripts/tdd_capture.sh
5. Create proof bundle: bash scripts/prove.sh
```

### 2. EC2 Execution Only
- All Python execution on EC2
- MacBook for file editing and git only
- SSH: `ssh -i /Users/mike/keys/AutoGenKeyPair.pem ubuntu@34.196.155.11`

### 3. Timestamp Output Files
```python
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
filename = f"results/{timestamp}_liquid_trajectory_comparison.png"
```

---

## 🎓 Learning Resources

### From liquid_ai_2:
- `liquid_neural_nets_info.md` - Theory and math
- `README_LIQUID_NN.md` - Quick start guide
- `chat_history/` - Development notes

### From liquid_mono_to_3d:
- `requirements.md` - Complete TDD methodology
- `ARCHITECTURE_PLANNING_LNN.md` - Your roadmap (already mentions LNNs!)
- `cursorrules` - Development rules

---

## 🤝 Why This Makes Sense

1. **Project name:** "Liquid Mono to 3D" → You planned for LNNs
2. **Code exists:** liquid_ai_2 has production-ready implementation
3. **Benefits:**
   - Better temporal dynamics
   - Smaller models
   - Continuous-time trajectories
   - Aligns with cutting-edge research

4. **Low risk:** Just porting proven code
5. **High value:** Completes the "Liquid" vision

---

## 📝 Decision Matrix

| Integration | Effort | Risk | Value | Priority |
|-------------|--------|------|-------|----------|
| **LiquidCell only** | 1 week | Low | Medium | **Start here** |
| **MagVIT + Liquid** | 2 weeks | Medium | High | Week 2-3 |
| **VLM Fusion** | 2 weeks | Medium | Very High | Week 4-5 |
| **3D Prediction** | 3 weeks | Medium | High | Future |

**Recommended:** Start with LiquidCell (1 week), then decide based on results.

---

## ✅ Ready to Start?

1. Read: `LIQUID_AI_2_INTEGRATION_RECOMMENDATIONS.md` (full details)
2. Decide: Which integration phase? (1, 2, or 3)
3. Execute: Follow Day 1-5 plan above
4. Review: Evaluate results, plan next phase

**Questions?** Refer to the detailed recommendations document.

---

**Created:** January 27, 2026  
**Status:** Ready to implement  
**Next Action:** Start Day 1 porting when ready

