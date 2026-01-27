# GPU Cloud Provider Cost Comparison
**Date:** January 26, 2026, 8:30 PM EST  
**Purpose:** Compare AWS g5.2xlarge with alternative GPU cloud providers

---

## TL;DR: Cost Comparison

**Your Current Setup (AWS g5.2xlarge):**
- Cost: $1.21/hour on-demand
- Annual: ~$315/year (5 hrs/week average)

**Alternative Providers (Typical Pricing):**

| Provider | GPU Type | Cost/hr | Annual Cost | vs AWS |
|----------|----------|---------|-------------|--------|
| **AWS g5.2xlarge** | A10G (24 GB) | $1.21 | $315 | Baseline |
| AWS g5.2xlarge Spot | A10G (24 GB) | $0.36-0.48 | $94-125 | **-70%** ⭐ |
| Vast.ai (on-demand) | A10G / RTX 3090 | $0.25-0.45 | $65-117 | **-60-80%** ⭐⭐ |
| Vast.ai (interruptible) | A10G / RTX 3090 | $0.15-0.30 | $39-78 | **-75-87%** ⭐⭐⭐ |
| RunPod (secure) | A10G / RTX 4090 | $0.39-0.69 | $101-180 | **-43-68%** ⭐ |
| RunPod (community) | RTX 3090 / 4090 | $0.25-0.49 | $65-127 | **-60-79%** ⭐⭐ |
| Lambda Labs | A10 (24 GB) | $0.60 | $156 | **-50%** ⭐ |
| Paperspace | A4000 (16 GB) | $0.76 | $198 | **-37%** |
| Google Cloud A2 | A100 (40 GB) | $3.67 | $955 | **+203%** ❌ |

**⚠️ Note:** Pricing is approximate and changes frequently. Always verify current pricing before committing.

---

## Detailed Provider Analysis

### 1. AWS EC2 (Current Provider)

#### g5.2xlarge On-Demand
**Specs:**
- GPU: 1× NVIDIA A10G (24 GB GDDR6)
- vCPUs: 8
- RAM: 32 GB
- Storage: EBS-optimized
- Network: Up to 10 Gbps

**Pricing:**
- On-demand: **$1.21/hour**
- Spot (typical): **$0.36-0.48/hour** (70% discount)
- Monthly (5 hrs/week): $26.25 on-demand, $7.85 spot
- Annual (5 hrs/week): **$315 on-demand, $94-125 spot**

**Pros:**
- ✅ Extremely reliable uptime (99.99%)
- ✅ Enterprise-grade infrastructure
- ✅ Integrated with AWS ecosystem
- ✅ Excellent network performance
- ✅ Mature tooling (CloudWatch, IAM, etc.)
- ✅ Predictable performance
- ✅ 24/7 support available

**Cons:**
- ❌ Most expensive option (on-demand)
- ❌ Complex pricing model
- ❌ Overkill for simple training jobs

**Best for:** Production workloads, enterprises, mission-critical training

---

### 2. Vast.ai (Decentralized GPU Marketplace)

#### Typical A10G / RTX 3090 / RTX 4090 Instances

**Specs (comparable):**
- GPU: A10G (24 GB), RTX 3090 (24 GB), or RTX 4090 (24 GB)
- vCPUs: 4-16 (varies by host)
- RAM: 16-64 GB (varies by host)
- Storage: Local SSD (varies)
- Network: 100 Mbps - 10 Gbps (varies)

**Pricing (Typical):**
- **On-demand (reliable hosts):** $0.25-0.45/hour
- **Interruptible (spot-like):** $0.15-0.30/hour
- Monthly (5 hrs/week): $5.40-9.75 (on-demand), $3.25-6.50 (interruptible)
- Annual (5 hrs/week): **$65-117 (on-demand), $39-78 (interruptible)**

**Actual Price Range by GPU:**
- RTX 3090 (24 GB): $0.15-0.35/hr
- RTX 4090 (24 GB): $0.20-0.50/hr
- A10G (24 GB): $0.25-0.45/hr
- A100 (40 GB): $0.50-1.20/hr

**Pros:**
- ✅ **60-87% cheaper than AWS**
- ✅ Wide variety of GPUs available
- ✅ Can filter by specs (VRAM, bandwidth, etc.)
- ✅ Pay-per-second billing
- ✅ No commitment required
- ✅ Community-driven marketplace
- ✅ Can negotiate long-term rates

**Cons:**
- ❌ Variable reliability (community hosts)
- ❌ Instances can be reclaimed by hosts
- ❌ Network speeds vary widely
- ❌ Some hosts have poor connectivity
- ❌ Less mature ecosystem
- ❌ No enterprise support
- ❌ Must carefully vet hosts (check reliability ratings)

**Best for:** Researchers, students, cost-sensitive projects, non-critical training

**Your Workload Fit:** ⭐⭐⭐ **EXCELLENT**
- Short training jobs (1-2 hours) = low interruption risk
- Checkpoints every 2 epochs = minimal lost work
- Not latency-sensitive = network variability OK
- **Potential savings: $237-276/year (75-88% savings)**

---

### 3. RunPod (Managed GPU Cloud)

#### Secure Cloud (RunPod-owned hardware)
**Specs:**
- GPU: A10G, RTX 4090, A40, A100
- vCPUs: 8-16
- RAM: 32-128 GB
- Storage: Network SSD
- Network: 10 Gbps

**Pricing:**
- **A10G:** $0.39-0.49/hour
- **RTX 4090:** $0.49-0.69/hour
- **A40 (48 GB):** $0.79/hour
- Monthly (5 hrs/week): $8.45-14.95
- Annual (5 hrs/week): **$101-180**

#### Community Cloud (Third-party hosts)
**Pricing:**
- **RTX 3090:** $0.25-0.39/hour
- **RTX 4090:** $0.29-0.49/hour
- Monthly (5 hrs/week): $5.40-10.60
- Annual (5 hrs/week): **$65-127**

**Pros:**
- ✅ **43-79% cheaper than AWS**
- ✅ Easy-to-use interface
- ✅ Docker-based deployment
- ✅ Good documentation
- ✅ Serverless GPU option
- ✅ Built-in Jupyter/SSH access
- ✅ Volume persistence across runs

**Cons:**
- ❌ Community Cloud less reliable
- ❌ Limited geographic regions
- ❌ Can have cold start times
- ❌ Less enterprise features than AWS

**Best for:** ML practitioners, startups, iterative training

**Your Workload Fit:** ⭐⭐ **VERY GOOD**
- Secure Cloud = reliable, good for production
- Community Cloud = cheaper, good for dev/experimentation
- **Potential savings: $135-214/year (43-68% savings)**

---

### 4. Lambda Labs (GPU Cloud)

#### A10 Instances (24 GB)

**Specs:**
- GPU: 1× NVIDIA A10 (24 GB) - similar to A10G
- vCPUs: 8-30
- RAM: 32-200 GB
- Storage: 512 GB - 6 TB SSD
- Network: 10 Gbps

**Pricing:**
- **1× A10:** $0.60/hour
- **2× A10:** $1.20/hour
- **4× A10:** $2.40/hour
- Monthly (5 hrs/week): $13.00
- Annual (5 hrs/week): **$156**

**Pros:**
- ✅ **50% cheaper than AWS**
- ✅ Simple, transparent pricing
- ✅ High-quality hardware
- ✅ Good for deep learning (optimized)
- ✅ Excellent network connectivity
- ✅ Pre-installed ML frameworks
- ✅ Good community support

**Cons:**
- ❌ Limited availability (high demand)
- ❌ Waitlist for some GPUs
- ❌ No spot instances
- ❌ Less flexibility than AWS
- ❌ Minimum 1-hour billing increments

**Best for:** ML researchers, teams, consistent workloads

**Your Workload Fit:** ⭐ **GOOD**
- Reliable, but may have availability issues
- **Potential savings: $159/year (50% savings)**

---

### 5. Paperspace (Gradient)

#### A4000 / A5000 Instances

**Specs:**
- GPU: NVIDIA A4000 (16 GB) or A5000 (24 GB)
- vCPUs: 8
- RAM: 32 GB
- Storage: 50 GB SSD
- Network: 1 Gbps

**Pricing:**
- **A4000 (16 GB):** $0.76/hour
- **A5000 (24 GB):** $1.38/hour
- **A6000 (48 GB):** $1.89/hour
- Monthly (5 hrs/week, A4000): $16.50
- Annual (5 hrs/week, A4000): **$198**

**Pros:**
- ✅ **37% cheaper than AWS** (A4000)
- ✅ User-friendly interface
- ✅ Jupyter notebooks built-in
- ✅ Good for prototyping
- ✅ Free tier available

**Cons:**
- ❌ A4000 has only 16 GB VRAM (less than A10G's 24 GB)
- ❌ Higher pricing than community providers
- ❌ Network speeds lower than AWS
- ❌ Limited GPU options

**Best for:** Individual researchers, notebook-based workflows

**Your Workload Fit:** ⚠️ **MARGINAL**
- A4000 (16 GB) might be tight for some experiments
- **Potential savings: $117/year (37% savings) but less VRAM**

---

### 6. Google Cloud Platform (GCP)

#### A2 Machine (A100 GPU)

**Specs:**
- GPU: 1× NVIDIA A100 (40 GB)
- vCPUs: 12
- RAM: 85 GB
- Storage: 100 GB SSD
- Network: 16 Gbps

**Pricing:**
- **a2-highgpu-1g:** $3.67/hour
- Preemptible: $1.10/hour (70% discount)
- Monthly (5 hrs/week): $79.55 on-demand, $23.85 preemptible
- Annual (5 hrs/week): **$955 on-demand, $286 preemptible**

**Pros:**
- ✅ A100 is more powerful than A10G
- ✅ Good for large models
- ✅ Integrated with GCP ecosystem
- ✅ TPU options available

**Cons:**
- ❌ **3× more expensive than AWS** (on-demand)
- ❌ Overkill for your workload
- ❌ Complex pricing
- ❌ A100 unnecessary for 500 MB models

**Best for:** Large-scale training, production ML pipelines

**Your Workload Fit:** ❌ **POOR**
- Way too expensive for your needs
- A100 overkill (you use <2% of A10G's 24 GB)

---

### 7. Tensor Dock (Budget GPU Cloud)

#### RTX 3090 / RTX 4090 Instances

**Specs:**
- GPU: RTX 3090 (24 GB) or RTX 4090 (24 GB)
- vCPUs: 6-12
- RAM: 32-64 GB
- Storage: 100 GB - 1 TB
- Network: 1-10 Gbps

**Pricing:**
- **RTX 3090:** $0.29-0.39/hour
- **RTX 4090:** $0.39-0.59/hour
- Monthly (5 hrs/week, RTX 3090): $6.30-8.45
- Annual (5 hrs/week, RTX 3090): **$75-101**

**Pros:**
- ✅ **68-76% cheaper than AWS**
- ✅ Consumer GPUs (RTX 3090/4090) perform well for training
- ✅ Simple pricing
- ✅ Good value for money

**Cons:**
- ❌ Smaller provider (less established)
- ❌ Limited geographic locations
- ❌ Less enterprise support
- ❌ Variable availability

**Best for:** Budget-conscious developers, student projects

**Your Workload Fit:** ⭐⭐ **VERY GOOD**
- RTX 3090 (24 GB) matches A10G VRAM
- **Potential savings: $214-240/year (68-76% savings)**

---

## Cost Comparison Summary Table

### Annual Cost (5 hours/week usage)

| Provider | Instance Type | VRAM | Cost/hr | Annual | Savings vs AWS | Reliability |
|----------|--------------|------|---------|--------|----------------|-------------|
| **AWS** | g5.2xlarge (on-demand) | 24 GB | $1.21 | **$315** | Baseline | ⭐⭐⭐⭐⭐ |
| **AWS** | g5.2xlarge (spot) | 24 GB | $0.40 | **$104** | **-67%** | ⭐⭐⭐⭐ |
| **Vast.ai** | A10G (on-demand) | 24 GB | $0.35 | **$91** | **-71%** | ⭐⭐⭐ |
| **Vast.ai** | RTX 3090 (interruptible) | 24 GB | $0.25 | **$65** | **-79%** | ⭐⭐ |
| **RunPod** | A10G (secure) | 24 GB | $0.44 | **$114** | **-64%** | ⭐⭐⭐⭐ |
| **RunPod** | RTX 3090 (community) | 24 GB | $0.32 | **$83** | **-74%** | ⭐⭐⭐ |
| **Lambda Labs** | A10 | 24 GB | $0.60 | **$156** | **-50%** | ⭐⭐⭐⭐ |
| **TensorDock** | RTX 3090 | 24 GB | $0.34 | **$88** | **-72%** | ⭐⭐⭐ |
| **Paperspace** | A4000 | 16 GB | $0.76 | **$198** | **-37%** | ⭐⭐⭐⭐ |
| **Google Cloud** | A100 (preemptible) | 40 GB | $1.10 | **$286** | **-9%** | ⭐⭐⭐⭐ |

---

## Recommendations by Use Case

### For Your Current Workload (Trajectory Training)

#### Option 1: AWS Spot Instances ⭐⭐⭐⭐⭐ **RECOMMENDED**
**Why:**
- Same infrastructure you're already using
- 70% cost savings ($315 → $104/year)
- Minimal operational changes
- Your workload is PERFECT for spot (short runs, checkpoints)

**Setup:**
- Use same g5.2xlarge, just request spot instance
- Add spot interruption handler to training script
- Auto-resume from checkpoints on interruption

**Savings:** **$211/year (67% reduction)**

---

#### Option 2: Vast.ai On-Demand (Reliable Hosts) ⭐⭐⭐⭐ **BEST VALUE**
**Why:**
- **79% cost savings** ($315 → $65/year with RTX 3090)
- Comparable performance to A10G
- Filter for high-reliability hosts (5-star ratings)
- Pay-per-second billing

**Setup:**
- Create Vast.ai account
- Filter: RTX 3090 or A10G, 24 GB VRAM, >10 Gbps network, 5-star host
- Use their CLI/API to launch instances
- Same training scripts (Docker container)

**Tradeoffs:**
- Slightly less reliable than AWS (but still good)
- Must vet hosts carefully
- Network speeds can vary

**Savings:** **$250/year (79% reduction)**

---

#### Option 3: RunPod Secure Cloud ⭐⭐⭐ **MIDDLE GROUND**
**Why:**
- 64% cost savings ($315 → $114/year)
- More reliable than Vast.ai (RunPod-owned hardware)
- Easy Docker-based deployment
- Good balance of cost and reliability

**Setup:**
- Create RunPod account
- Choose "Secure Cloud" for reliability
- Deploy with Docker template
- SSH or Jupyter access built-in

**Tradeoffs:**
- More expensive than Vast.ai
- Less geographic coverage than AWS

**Savings:** **$201/year (64% reduction)**

---

#### Option 4: Lambda Labs ⭐⭐ **IF AVAILABLE**
**Why:**
- 50% cost savings ($315 → $156/year)
- High reliability
- Simple pricing
- Good for ML workloads

**Tradeoffs:**
- Often have waitlists (high demand)
- Limited availability
- 1-hour minimum billing

**Savings:** **$159/year (50% reduction)**

---

### Decision Matrix

| Priority | Recommendation | Annual Cost | Savings | Reliability |
|----------|---------------|-------------|---------|-------------|
| **Max Savings** | Vast.ai (interruptible) | $65 | **-79%** | ⭐⭐ |
| **Best Value** | Vast.ai (on-demand, reliable) | $91 | **-71%** | ⭐⭐⭐ |
| **Easy Migration** | AWS Spot | $104 | **-67%** | ⭐⭐⭐⭐ |
| **Balance** | RunPod Secure | $114 | **-64%** | ⭐⭐⭐⭐ |
| **Reliability** | AWS On-Demand | $315 | 0% | ⭐⭐⭐⭐⭐ |

---

## Implementation Guide

### Switching to AWS Spot (Easiest)

```bash
# Launch spot instance (AWS CLI)
aws ec2 request-spot-instances \
  --spot-price "0.60" \
  --instance-count 1 \
  --type "one-time" \
  --launch-specification \
    InstanceType=g5.2xlarge,\
    ImageId=ami-xxxxxxxxx,\
    KeyName=your-key-pair

# Add spot interruption handler to training script
import requests

def check_spot_interruption():
    try:
        r = requests.get(
            'http://169.254.169.254/latest/meta-data/spot/instance-action',
            timeout=1
        )
        if r.status_code == 200:
            print("⚠️ Spot interruption detected, saving checkpoint...")
            save_checkpoint()
            return True
    except:
        pass
    return False

# In training loop
for epoch in range(epochs):
    train_epoch()
    save_checkpoint()  # Already doing this ✅
    if check_spot_interruption():
        break
```

---

### Switching to Vast.ai

```bash
# Install Vast.ai CLI
pip install vastai

# Login
vastai set api-key YOUR_API_KEY

# Search for instances
vastai search offers \
  'gpu_name=RTX_3090 num_gpus=1 gpu_ram>=24 reliability>=0.95 inet_down>=100'

# Launch instance
vastai create instance <offer_id> \
  --image pytorch/pytorch:latest \
  --disk 50

# SSH into instance
vastai ssh instance <instance_id>

# Run training (same script)
cd /workspace
git clone your-repo
python train.py
```

---

### Switching to RunPod

```bash
# Web UI: runpod.io
# 1. Create account
# 2. Select "Secure Cloud" or "Community Cloud"
# 3. Choose GPU (A10G or RTX 4090)
# 4. Deploy with template or custom Docker image
# 5. Connect via SSH or Jupyter

# Or use API
import runpod

runpod.api_key = "YOUR_API_KEY"

pod = runpod.create_pod(
    name="training-job",
    image_name="pytorch/pytorch:latest",
    gpu_type_id="NVIDIA A10",
    cloud_type="SECURE"
)

# SSH into pod
ssh root@<pod-ip>
```

---

## Risk Assessment

### AWS Spot Instances
**Interruption Risk:** 5-10%
**Impact:** Lose max 20 minutes of work (2 epochs)
**Mitigation:** Auto-resume from checkpoints ✅ Already implemented
**Overall Risk:** 🟢 **LOW**

### Vast.ai On-Demand (Reliable Hosts)
**Interruption Risk:** 10-15% (host can reclaim)
**Impact:** Same as spot (checkpoints every 2 epochs)
**Mitigation:** Filter for 5-star hosts, >95% reliability
**Overall Risk:** 🟡 **MEDIUM**

### RunPod Secure Cloud
**Interruption Risk:** <5% (RunPod-owned hardware)
**Impact:** Minimal (very rare)
**Mitigation:** Built-in volume persistence
**Overall Risk:** 🟢 **LOW**

### Vast.ai Interruptible
**Interruption Risk:** 20-30%
**Impact:** More frequent restarts
**Mitigation:** Checkpoints + auto-resume
**Overall Risk:** 🟡 **MEDIUM** (acceptable for max savings)

---

## My Recommendation

### For You Specifically:

**Primary Recommendation:** ⭐⭐⭐⭐⭐ **AWS g5.2xlarge Spot**

**Why:**
1. ✅ **67% cost savings** ($315 → $104/year)
2. ✅ **Minimal operational change** (same infrastructure)
3. ✅ **Your workload is PERFECT for spot** (short runs, checkpoints)
4. ✅ **Low risk** (5-10% interruption, auto-resume ready)
5. ✅ **No new account/billing setup needed**

**Alternative:** ⭐⭐⭐⭐ **Vast.ai (Reliable Hosts)**

**Why:**
1. ✅ **79% cost savings** ($315 → $65/year with RTX 3090)
2. ✅ **Similar performance** to A10G
3. ✅ **Low interruption risk** with reliable hosts
4. ✅ **Pay-per-second billing** (no waste)

**Not Recommended:** Google Cloud A100 (overkill, 3× cost)

---

## Action Items

### Immediate (This Week)

1. **Test AWS Spot:**
   - Launch g5.2xlarge spot instance
   - Run a training job
   - Verify checkpointing works
   - Monitor for interruptions

2. **If Spot works well:**
   - Migrate all training to spot
   - **Save $211/year**

### Optional (Next Month)

1. **Create Vast.ai account** (free)
2. **Run test job** on RTX 3090
3. **Compare:**
   - Performance (training speed)
   - Reliability (interruptions)
   - Ease of use
4. **If Vast.ai works:**
   - Use for dev/experimentation
   - **Additional $39-46/year savings**

---

## Bottom Line

**Your current AWS g5.2xlarge on-demand is costing you $315/year.**

**Easy wins:**
- Switch to **AWS Spot** → **$104/year** (67% savings, minimal risk)
- Switch to **Vast.ai** → **$65/year** (79% savings, slightly more risk)

**Both are excellent choices for your workload. Start with AWS Spot for easiest migration, then explore Vast.ai for maximum savings.**

---

**Cost Comparison Date:** January 26, 2026  
**Note:** Pricing subject to change. Verify current rates before committing.  
**Recommendation:** Test AWS Spot first (lowest risk, high savings)

