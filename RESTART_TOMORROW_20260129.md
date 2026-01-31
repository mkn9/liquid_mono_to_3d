# Restart Guide: January 29, 2026

## 🌅 Quick Start for Tomorrow

### EC2 Instance Info (from today)
- **Instance ID**: `i-0a79e889e073ef0ec`
- **Type**: Spot instance (g5.2xlarge or similar)
- **Last IP**: `204.236.245.232` (will change when restarted)
- **Region**: us-east-1

---

## Step 1: Launch Instance via Auto Scaling Group

### ✅ Use Auto Scaling Group (Your Setup)

**You have an ASG configured: `GPU G5 spot – ASG`**

#### Restart Steps:
1. Go to: https://console.aws.amazon.com/ec2/autoscaling/
2. Region: **us-east-1**
3. Select: `GPU G5 spot – ASG`
4. Click **"Edit"**
5. Set **"Desired capacity"**: `1`
6. Set **"Minimum capacity"**: `0` (or 1 if you want it always running)
7. Click **"Update"**

#### Wait for Launch (2-5 minutes)
- ASG will automatically launch spot instance
- Monitor in "Activity" tab
- Instance will have same configuration (launch template)

### Get New IP Address

#### Option A: AWS Console
1. EC2 → Instances
2. Find instance with ASG tag: `GPU G5 spot – ASG`
3. Copy **Public IPv4 address**

#### Option B: AWS CLI
```bash
aws ec2 describe-instances \
  --filters "Name=tag:aws:autoscaling:groupName,Values=GPU G5 spot – ASG" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text
```

---

## Step 2: Connect and Setup

### SSH Connection
```bash
ssh -i /Users/mike/keys/AutoGenKeyPair.pem ubuntu@<NEW_IP_ADDRESS>
```

### Quick Setup Script
```bash
# Clone repository
cd ~
git clone https://github.com/mkn9/liquid_mono_to_3d.git
cd liquid_mono_to_3d

# Activate Python environment (create if needed)
if [ -d ~/mono_to_3d_env ]; then
  source ~/mono_to_3d_env/bin/activate
else
  python3 -m venv ~/mono_to_3d_env
  source ~/mono_to_3d_env/bin/activate
  pip install -r requirements.txt
fi

# Verify CUDA
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check status
bash scripts/heartbeat_vlm.sh
```

---

## Step 3: Resume Work

### Yesterday's Status (Completed) ✅
1. ✅ Workers 2-5: Complete (12/12 tests passing)
2. ✅ VLM Evaluation: Framework created
3. ✅ Architecture: Documented and clarified
4. ✅ Git: All committed and pushed
5. ✅ Results: 10 visualizations, accuracy metrics

### Today's Priority Tasks

#### Option 1: GPT-4 Evaluation (if API key available)
```bash
# Set OpenAI API key
export OPENAI_API_KEY="sk-proj-..." # Get full key from user

# Run GPT-4 evaluation
cd ~/liquid_mono_to_3d
python3 experiments/liquid_vlm_integration/evaluate_vlm_accuracy.py

# Results will be in:
# experiments/liquid_vlm_integration/results/YYYYMMDD_HHMM_vlm_evaluation.json

# Sync to MacBook
rsync -avz -e "ssh -i /Users/mike/keys/AutoGenKeyPair.pem" \
  experiments/liquid_vlm_integration/results/ \
  mike@<MACBOOK_IP>:~/path/to/results/
```

#### Option 2: Improve TinyLlama (if no API key)
```bash
# Create better prompting
cd ~/liquid_mono_to_3d/experiments/liquid_vlm_integration

# Edit tinyllama_vlm.py to improve prompts
# Then test:
python3 -c "
from tinyllama_vlm import TinyLlamaVLM
import torch

vlm = TinyLlamaVLM()
embeddings = torch.randn(1, 4096)
desc = vlm.generate_description(embeddings, 
    prompt='Describe ONLY the trajectory path you observe: shape, direction, speed.')
print(desc)
"
```

#### Option 3: Generate Training Data for Fine-tuning
```bash
# Generate 1000 trajectory samples with descriptions
python3 -c "
from simple_3d_tracker import generate_synthetic_tracks
import json

dataset = []
for i in range(1000):
    sensor1, sensor2, points_3d = generate_synthetic_tracks()
    # Create ground truth description
    # ... (implement ground truth generation)
    dataset.append({'trajectory': points_3d.tolist(), 'description': gt_desc})

with open('trajectory_training_data.json', 'w') as f:
    json.dump(dataset, f)
"

# Then fine-tune TinyLlama with LoRA
# ... (implement fine-tuning script)
```

---

## Step 4: Save Results Before Shutdown

### Always Do Before Stopping:
```bash
# 1. Commit any new code
cd ~/liquid_mono_to_3d
git add -A
git commit -m "Session YYYYMMDD: [describe work]"
git push origin main

# 2. Sync results to MacBook
rsync -avz -e "ssh -i /Users/mike/keys/AutoGenKeyPair.pem" \
  ~/liquid_mono_to_3d/experiments/liquid_vlm_integration/results/ \
  ubuntu@<YOUR_MACBOOK>:/path/to/results/

# 3. Create shutdown status
cat > SHUTDOWN_STATUS_YYYYMMDD.md << EOF
# Shutdown Status: $(date)
## Work Completed:
- ...
## Outstanding:
- ...
EOF

# 4. Terminate instance
# (via AWS Console or CLI)
```

---

## Quick Reference

### Files to Review on MacBook First
```
experiments/liquid_vlm_integration/
├── results/20260128_0508_vlm_evaluation.json  ← Yesterday's evaluation
├── results/2020260128_0508_sample_*.png       ← 10 visualizations
├── evaluate_vlm_accuracy.py                    ← Evaluation script
├── tinyllama_vlm.py                            ← TinyLlama wrapper
└── gpt4_vlm.py                                 ← GPT-4 wrapper
```

### Key Findings from Yesterday
- **TinyLlama Accuracy**: 35% (±16.6%) ❌
  - Hallucinates YouTube URLs, 3D printing tutorials
  - Doesn't actually describe trajectories
  - Needs fine-tuning or better prompting

- **Liquid NN Integration**: ✅ WORKING
  - `LiquidCell` confirmed inside `LiquidDualModalFusion`
  - Output: `h_fusion` (4096-dim) → TinyLlama/GPT-4
  - All tests passing (12/12)

### Outstanding Questions
1. **OpenAI API Key**: Need full key for GPT-4 evaluation
   - Partial: `sk-proj-Nae9JoShWsxa...`
   - Location: Check `mono_to_3d` project requirements or logs

2. **TinyLlama Improvement Strategy**: Fine-tune vs better prompting?

3. **Additional Metrics**: Should we add BLEU/ROUGE scores?

---

## Troubleshooting

### If Repository Clone Fails
```bash
# Use cached version if available
cd ~/liquid_mono_to_3d
git pull origin main
```

### If CUDA Not Available
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch if needed
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### If Dependencies Missing
```bash
pip install transformers openai accelerate matplotlib pandas numpy opencv-python
```

---

## Cost Reminder

**Spot Instance Costs** (approximate):
- g5.2xlarge: ~$0.30-0.50/hour (spot)
- g4dn.xlarge: ~$0.15-0.25/hour (spot)
- On-demand backup: 3-4x more expensive

**Best Practice**: Always terminate when done, never leave running overnight

---

## Emergency Contact Info

- **GitHub Repo**: https://github.com/mkn9/liquid_mono_to_3d
- **Last Commit**: 1b2c391
- **Chat History**: `CHAT_HISTORY_20260128_WORKERS_2_5_COMPLETE.md`
- **MacBook Backup**: `~/Dropbox/Documents/Machine_Learning/.../liquid_mono_to_3d/`

---

**Created**: 2026-01-28 05:20 UTC  
**For**: January 29, 2026 session  
**Status**: Instance `i-0a79e889e073ef0ec` terminated safely ✅

