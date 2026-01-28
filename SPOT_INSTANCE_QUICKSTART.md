# AWS Spot Instance Quick Connection Guide
**Date:** January 27, 2026  
**Project:** liquid_mono_to_3d  
**Purpose:** How to launch and connect to AWS Spot instance for Liquid NN development

---

## 🚀 Fastest Path (5 Minutes)

### Option 1: Use AWS Console (GUI) - EASIEST

**Step 1: Launch Spot Instance (2 min)**

1. Go to [AWS EC2 Console](https://console.aws.amazon.com/ec2/)
2. Click **"Launch Instance"**
3. Configure:
   - **Name:** `liquid-mono-to-3d-spot`
   - **AMI:** Deep Learning AMI (Ubuntu 22.04) - `ami-0c02fb55b34f5b9f0`
   - **Instance type:** `g5.2xlarge` (A10G GPU, 24GB VRAM)
   - **Key pair:** `AutoGenKeyPair` (your existing key)
   - **Storage:** 100 GB gp3 (default is fine)
   
4. **Under "Advanced Details":**
   - Scroll to **"Purchasing option"**
   - ✅ Check **"Request Spot instances"**
   - Leave max price blank (use market price)
   - Interruption behavior: **"Stop"** (not "Terminate")

5. Click **"Launch instance"**

**Step 2: Get Connection Info (1 min)**

1. Wait for instance to start (~2 min)
2. Go to **EC2 Dashboard → Instances**
3. Select your `liquid-mono-to-3d-spot` instance
4. Copy **Public IPv4 address** (e.g., `54.123.45.67`)

**Step 3: Connect (30 sec)**

```bash
# On your MacBook
ssh -i /Users/mike/keys/AutoGenKeyPair.pem ubuntu@<SPOT-IP>

# Example:
ssh -i /Users/mike/keys/AutoGenKeyPair.pem ubuntu@54.123.45.67
```

**That's it!** You're connected. Cost: ~$0.40/hr (vs. $1.21/hr on-demand)

---

## Option 2: Use AWS CLI (Command Line) - FASTER IF YOU KNOW CLI

### Quick Launch Script

```bash
#!/bin/bash
# Launch Spot instance for liquid_mono_to_3d

# Configuration
REGION="us-east-1"
AMI_ID="ami-0c02fb55b34f5b9f0"  # Deep Learning AMI (Ubuntu 22.04)
INSTANCE_TYPE="g5.2xlarge"
KEY_NAME="AutoGenKeyPair"

# Get your existing security group and subnet
# (Run this once to find your values)
aws ec2 describe-security-groups --query 'SecurityGroups[0].GroupId' --output text
aws ec2 describe-subnets --query 'Subnets[0].SubnetId' --output text

# Set these based on output above
SECURITY_GROUP="sg-XXXXXXXXX"  # Replace with your SG
SUBNET_ID="subnet-XXXXXXXXX"   # Replace with your subnet

# Launch Spot instance
aws ec2 run-instances \
    --region $REGION \
    --image-id $AMI_ID \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-group-ids $SECURITY_GROUP \
    --subnet-id $SUBNET_ID \
    --instance-market-options '{
        "MarketType": "spot",
        "SpotOptions": {
            "SpotInstanceType": "persistent",
            "InstanceInterruptionBehavior": "stop"
        }
    }' \
    --block-device-mappings '[{
        "DeviceName": "/dev/sda1",
        "Ebs": {
            "VolumeSize": 100,
            "VolumeType": "gp3",
            "DeleteOnTermination": false
        }
    }]' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=liquid-mono-to-3d-spot}]' \
    --output json > spot_launch.json

# Get instance ID
SPOT_INSTANCE_ID=$(jq -r '.Instances[0].InstanceId' spot_launch.json)
echo "Spot Instance ID: $SPOT_INSTANCE_ID"
echo "Waiting for instance to start..."

# Wait for instance to start
aws ec2 wait instance-running --instance-ids $SPOT_INSTANCE_ID

# Get public IP
SPOT_IP=$(aws ec2 describe-instances \
    --instance-ids $SPOT_INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "✅ Spot instance running!"
echo "Instance ID: $SPOT_INSTANCE_ID"
echo "Public IP: $SPOT_IP"
echo ""
echo "To connect:"
echo "  ssh -i /Users/mike/keys/$KEY_NAME.pem ubuntu@$SPOT_IP"
```

**Save this as:** `scripts/launch_spot.sh`  
**Run:** `bash scripts/launch_spot.sh`

---

## 🔌 After Connecting: Setup liquid_mono_to_3d

Once you're connected via SSH:

```bash
# On EC2 instance

# 1. Check CUDA/GPU
nvidia-smi  # Should show A10G GPU

# 2. Clone liquid_mono_to_3d
git clone https://github.com/mkn9/liquid_mono_to_3d.git
cd liquid_mono_to_3d

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy liquid_cell.py from liquid_ai_2 (we'll do this as part of Phase 1)
# For now, you're ready to start!

# 5. Test PyTorch + GPU
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
# Should output: CUDA available: True, GPU: NVIDIA A10G
```

---

## 💰 Cost Comparison

| Instance Type | Pricing | Monthly (40 hrs) | Annual (480 hrs) |
|---------------|---------|------------------|------------------|
| **On-Demand g5.2xlarge** | $1.21/hr | $48.40 | $580.80 |
| **Spot g5.2xlarge** | $0.36-0.48/hr | $14-19 | $173-230 |
| **Savings** | **67-70%** | **~$31/mo** | **~$375/yr** |

---

## ⚠️ Important: Spot Interruptions

**What are interruptions?**
- AWS can reclaim your Spot instance if they need capacity
- You get **2-minute warning**
- Happens when demand is high (usually business hours)

**Interruption rates:**
- Overnight (12am-6am): **1.8%** (very rare)
- Business hours (12pm-6pm): **8.7%** (more common)
- Weekends: **2.3%** (rare)

**How to handle:**

1. **Save work frequently** (commit to git often)
2. **Use checkpointing** in training (save every epoch)
3. **Run during off-peak** (overnight = lowest interruption)

**If interrupted:**
- Instance stops (not terminated)
- You can restart it: `aws ec2 start-instances --instance-ids <INSTANCE-ID>`
- EBS volume persists (your data is safe)
- Gets new IP after restart

---

## 🛠️ Recommended: Setup Interruption Handler

Create this script on your Spot instance:

```bash
#!/bin/bash
# ~/liquid_mono_to_3d/scripts/spot_interruption_handler.sh

METADATA_URL="http://169.254.169.254/latest/meta-data/spot/instance-action"

while true; do
    HTTP_CODE=$(curl -o /dev/null -s -w "%{http_code}" $METADATA_URL --max-time 1)
    
    if [ "$HTTP_CODE" -eq 200 ]; then
        echo "⚠️  SPOT INTERRUPTION - Saving work..."
        
        # Save current work
        cd ~/liquid_mono_to_3d
        git add -A
        git stash save "Auto-stash: spot interruption $(date)"
        
        # Kill training jobs gracefully
        pkill -SIGTERM -f "python.*train"
        
        echo "✅ Saved - instance will stop in 2 min"
        exit 0
    fi
    
    sleep 5
done
```

**To run in background:**

```bash
cd ~/liquid_mono_to_3d
chmod +x scripts/spot_interruption_handler.sh
nohup bash scripts/spot_interruption_handler.sh > logs/interruption.log 2>&1 &
```

---

## 📋 Quick Commands Reference

### Check Instance Status

```bash
# On MacBook - check if your Spot instance is running
aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=liquid-mono-to-3d-spot" \
    --query 'Reservations[0].Instances[0].[State.Name,PublicIpAddress]' \
    --output table
```

### Start Stopped Instance

```bash
# Get instance ID
INSTANCE_ID=$(aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=liquid-mono-to-3d-spot" \
    --query 'Reservations[0].Instances[0].InstanceId' \
    --output text)

# Start it
aws ec2 start-instances --instance-ids $INSTANCE_ID

# Wait for it to start
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

# Get new IP (changes after stop/start)
NEW_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "New IP: $NEW_IP"
ssh -i /Users/mike/keys/AutoGenKeyPair.pem ubuntu@$NEW_IP
```

### Stop Instance When Done

```bash
# On MacBook
aws ec2 stop-instances --instance-ids $INSTANCE_ID

# Verify stopped (costs $0/hr when stopped)
aws ec2 describe-instances --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].State.Name' \
    --output text
# Should output: stopped
```

---

## 🎯 Best Practices

### For Development (Short Sessions)

1. **Launch Spot instance** when you start work
2. **Connect and code** (~$0.40/hr)
3. **Commit work frequently** (git push)
4. **Stop instance** when done ($0/hr)
5. **Restart tomorrow** (may get new IP)

**Cost:** ~$0.40/hr × 2-3 hrs/day = $0.80-1.20/day

### For Training (Long Runs)

1. **Launch Spot instance**
2. **Setup checkpointing** (save every epoch)
3. **Run interruption handler** in background
4. **Start training** (with checkpoint resume logic)
5. **If interrupted:** Restart instance, resume from checkpoint
6. **Stop when done**

**Cost:** Still ~60-70% cheaper than on-demand even with interruptions

### For Peace of Mind (Backup Strategy)

1. **Launch Spot instance** (use this 90% of time)
2. **Launch on-demand backup** and stop it ($0/hr when stopped)
3. **If Spot interrupted:** Start backup, continue work (1-2 min downtime)
4. **Stop backup when Spot available again**

**Cost:** ~$0.43/hr average (includes occasional backup usage)

**See full backup setup:** `mono_to_3d/AWS_SPOT_SETUP_LIQUID_MONO_TO_3D.md`

---

## ❓ FAQ

### Q: What if I lose my connection?

**A:** Your work persists! Just SSH back in. The instance keeps running.

### Q: What if instance is interrupted?

**A:** Instance stops (not terminated). Your EBS volume and data persist. Just restart:
```bash
aws ec2 start-instances --instance-ids <INSTANCE-ID>
```

### Q: Can I get my data back after interruption?

**A:** Yes! EBS volume persists. When you restart the instance, all your files are there.

### Q: How often will interruptions happen?

**A:** Historically:
- Overnight: 1.8% chance per hour (very rare)
- Business hours: 8.7% chance per hour (uncommon but possible)
- If you work 2 hrs/day overnight: ~1 interruption per month

### Q: Is Spot worth it?

**A:** Absolutely for development and training:
- ✅ 67-70% cost savings
- ✅ Interruptions are rare during off-peak
- ✅ Data persists (EBS volumes)
- ✅ Can have on-demand backup for peace of mind

### Q: Should I use Spot for the Liquid NN integration work?

**A:** Yes! Perfect use case:
- Development sessions are 2-4 hours (short)
- Code committed frequently (TDD workflow)
- Training runs have checkpoints
- Off-peak hours = very low interruption rate
- **Save ~$375/year**

---

## 🚀 Ready to Start?

**For liquid_mono_to_3d Phase 1 (Liquid Fusion Layer):**

1. **Launch Spot instance** (Option 1 above - AWS Console)
2. **Connect:** `ssh -i /Users/mike/keys/AutoGenKeyPair.pem ubuntu@<SPOT-IP>`
3. **Setup project** (clone, install dependencies)
4. **Start Phase 1** (follow `LIQUID_NN_INTEGRATION_REVISED.md`)

**Estimated cost for Phase 1 (2 weeks):**
- Development: ~20 hours × $0.40/hr = $8
- Training/evaluation: ~10 hours × $0.40/hr = $4
- **Total: ~$12** (vs. $36 on-demand)

**Savings: $24 for Phase 1 alone**

---

## 📚 Full Documentation

For more details, see:
- `mono_to_3d/AWS_SPOT_SETUP_LIQUID_MONO_TO_3D.md` (complete setup with backup)
- `mono_to_3d/AWS_SPOT_INSTANCES_GUIDE.md` (820 lines, comprehensive)
- `mono_to_3d/AWS_SPOT_FAILOVER_SETUP.md` (automatic failover)
- `mono_to_3d/AWS_SPOT_INTERRUPTION_DISTRIBUTION.md` (interruption patterns)

---

**Created:** January 27, 2026  
**Status:** Ready to use  
**Next Action:** Launch Spot instance → Connect → Start Phase 1 implementation

