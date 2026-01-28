# EC2 Spot Instance Setup - COMPLETE ✅
**Date:** January 28, 2026  
**Instance IP:** 204.236.245.232  
**Instance Type:** g5.2xlarge (NVIDIA A10G, 23GB VRAM)  
**Cost:** ~$0.40/hr (vs $1.21/hr on-demand)

---

## ✅ Setup Status

### Instance Configuration
- [x] Spot instance launched and running
- [x] SSH connection verified  
- [x] GPU accessible (NVIDIA A10G, 23GB VRAM)
- [x] CUDA 12.6 installed and working
- [x] PyTorch 2.6.0+cu126 with GPU support
- [x] Python 3.12.6 ready

### Project Setup
- [x] liquid_mono_to_3d repository cloned
- [x] Dependencies installed (requirements.txt)
- [x] All Liquid integration documents synced
- [x] Project structure verified

### Liquid Integration Documents Available
1. **START_HERE.md** - Navigation guide
2. **EXECUTIVE_SUMMARY_LIQUID_INTEGRATION.md** - Decision framework
3. **ARCHITECTURE_COMPARISON.md** - Before/after comparison
4. **LIQUID_NN_INTEGRATION_REVISED.md** - Complete implementation guide
5. **SPOT_INSTANCE_QUICKSTART.md** - This instance setup guide
6. **README_LIQUID_INTEGRATION.md** - Document index

---

## 🚀 You Are Here

**Location:** `/home/ubuntu/liquid_mono_to_3d`

**What to do next:**

### Option 1: Review Documentation (Recommended)

```bash
# Read the navigation guide
less START_HERE.md

# Review the executive summary (decision document)
less EXECUTIVE_SUMMARY_LIQUID_INTEGRATION.md

# When ready, review the implementation guide
less LIQUID_NN_INTEGRATION_REVISED.md
```

### Option 2: Start Implementation Immediately

If you've already decided to proceed with Phase 1:

```bash
# Create directory for Liquid models
mkdir -p experiments/trajectory_video_understanding/liquid_models

# From MacBook, copy liquid_cell.py from liquid_ai_2:
# rsync -avz -e "ssh -i /Users/mike/keys/AutoGenKeyPair.pem" \
#   /path/to/liquid_ai_2/option1_synthetic/liquid_cell.py \
#   ubuntu@204.236.245.232:~/liquid_mono_to_3d/experiments/trajectory_video_understanding/liquid_models/

# Start Phase 1, Day 1 tasks (see LIQUID_NN_INTEGRATION_REVISED.md)
```

---

## 📡 Connection Info

**From your MacBook:**
```bash
ssh -i /Users/mike/keys/AutoGenKeyPair.pem ubuntu@204.236.245.232
cd ~/liquid_mono_to_3d
```

**Quick Commands:**
```bash
# Check GPU
nvidia-smi

# Check PyTorch + GPU
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Run tests
pytest tests/ -v

# Start TDD workflow
bash scripts/tdd_capture.sh
```

---

## 💰 Cost Tracking

| Duration | Cost |
|----------|------|
| Hourly | $0.40 |
| Daily (8 hrs) | ~$3.20 |
| Phase 1 (2 weeks, ~30 hrs) | ~$12 |
| Phase 1+2 (4 weeks, ~60 hrs) | ~$24 |

**Remember to stop instance when done:**
```bash
# From MacBook - Get instance ID
aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=liquid-mono-to-3d-spot" \
    --query 'Reservations[0].Instances[0].InstanceId' \
    --output text

# Stop instance
aws ec2 stop-instances --instance-ids <INSTANCE-ID>
```

---

## 📚 Quick Reference

**Navigation Path:**  
START_HERE.md → EXECUTIVE_SUMMARY_LIQUID_INTEGRATION.md → LIQUID_NN_INTEGRATION_REVISED.md

**Current Status:** Ready for review and decision  
**Next Task:** Review documentation, decide on Phase 1  
**If Approved:** Start Phase 1 Day 1 (port liquid_cell.py, write tests)

---

## 🔗 Important Paths

**Project Root:** `/home/ubuntu/liquid_mono_to_3d`  
**Experiments:** `/home/ubuntu/liquid_mono_to_3d/experiments/`  
**Scripts:** `/home/ubuntu/liquid_mono_to_3d/scripts/`  
**Tests:** `/home/ubuntu/liquid_mono_to_3d/tests/`  
**Artifacts:** `/home/ubuntu/liquid_mono_to_3d/artifacts/`

---

## ⚙️ Available Tools

**TDD Capture:**
```bash
bash scripts/tdd_capture.sh
```

**Proof Bundle:**
```bash
bash scripts/prove.sh
```

**Git Workflow:**
```bash
git status
git add <files>
git commit -m "message"
git push
```

---

## 🎯 Phase 1 Overview (If You Proceed)

**Week 1: Implementation**
- Day 1-2: Port liquid_cell.py, write tests (RED)
- Day 3-4: Implement LiquidDualModalFusion (GREEN)
- Day 5: Integration & testing

**Week 2: Evaluation**
- Day 6-7: Compare static vs. liquid fusion
- Day 8: Decision point for Phase 2

**Deliverables:**
- Working Liquid fusion layer
- Test suite passing
- TDD evidence captured
- Performance comparison

---

## 🆘 Quick Help

**Need to reconnect?**
```bash
ssh -i /Users/mike/keys/AutoGenKeyPair.pem ubuntu@204.236.245.232
```

**Lost your place?**
```bash
cd ~/liquid_mono_to_3d
cat START_HERE.md | head -50
```

**Want to test GPU?**
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
nvidia-smi
```

**Check disk space:**
```bash
df -h /home/ubuntu
```

**Check memory:**
```bash
free -h
```

---

**Setup completed:** January 28, 2026, 02:30 UTC  
**Status:** All systems GO ✅  
**Ready for:** Liquid Neural Network integration work

---

## 📝 Notes

- Instance IP may change after stop/start
- EBS volume persists (your data is safe)
- Commit work frequently (git push)
- Use TDD workflow (scripts/tdd_capture.sh)
- Stop instance when done to save costs

**Estimated interruption rate (current time):** ~1.8% per hour (overnight)  
**Next business hours peak:** ~8.7% per hour

---

**All documentation is ready. You can now review and decide on implementation!** 🚀

