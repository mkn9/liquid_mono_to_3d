# EC2 Shutdown Procedure - CORRECTED

**Date**: 2026-01-28 05:30 UTC  
**User Correction**: Use Auto Scaling Group, not manual termination

---

## ✅ Correct Shutdown Procedure

### **Use Auto Scaling Group (ASG) - NOT Manual Termination**

Your setup uses an Auto Scaling Group for spot instances, so you should manage it through ASG:

### Step 1: Go to Auto Scaling Console
1. Navigate to: https://console.aws.amazon.com/ec2/autoscaling/
2. Region: **us-east-1** (or your configured region)

### Step 2: Find Your ASG
- **ASG Name**: `GPU G5 spot – ASG` (or similar)
- Should show your current running instance

### Step 3: Set Desired Capacity to 0
1. Select your ASG: `GPU G5 spot – ASG`
2. Click **"Edit"** or **"Details"** tab
3. Set **"Desired capacity"**: `0`
4. Set **"Minimum capacity"**: `0` (optional, but recommended for full shutdown)
5. Click **"Update"**

### Step 4: Verify Termination
- The ASG will automatically terminate the instance
- Wait 1-2 minutes for termination to complete
- Check "Instances" tab to confirm instance is terminated

---

## 🎯 Why Use ASG Instead of Manual Termination?

### Benefits of ASG Approach:
1. ✅ **Preserves Configuration**: Launch template settings saved
2. ✅ **Easy Restart**: Just set desired capacity back to 1
3. ✅ **Automatic Failover**: ASG can handle spot interruptions
4. ✅ **Cost Optimization**: Can use spot instance best practices
5. ✅ **Consistent Setup**: Same config every time you restart

### Why NOT Manual Termination:
1. ❌ ASG would detect terminated instance as "unhealthy"
2. ❌ ASG might automatically launch replacement
3. ❌ Conflicts with ASG desired capacity
4. ❌ Loses ASG management benefits

---

## 🌅 Restart Tomorrow (ASG Approach)

### Step 1: Set Desired Capacity to 1
1. Go to Auto Scaling Console
2. Select: `GPU G5 spot – ASG`
3. Edit → Set **"Desired capacity"**: `1`
4. Click **"Update"**

### Step 2: Wait for Instance Launch
- ASG will automatically launch a new spot instance
- Takes 2-5 minutes
- Monitor in "Activity" tab

### Step 3: Get New IP Address
```bash
# Via AWS Console
EC2 → Instances → Find your new instance → Copy Public IPv4

# Or via CLI
aws ec2 describe-instances \
  --filters "Name=tag:aws:autoscaling:groupName,Values=GPU G5 spot – ASG" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text
```

### Step 4: Connect
```bash
ssh -i /Users/mike/keys/AutoGenKeyPair.pem ubuntu@<NEW_IP>
```

---

## 📋 Current Instance Info

**Instance ID**: `i-0a79e889e073ef0ec`  
**Current IP**: `204.236.245.232`  
**ASG**: `GPU G5 spot – ASG`

**Action**: Set ASG desired capacity to 0 (NOT manual termination)

---

## ⚙️ ASG Configuration (Recommended Settings)

### For Daily Shutdown/Startup:
```
Desired capacity: 0 (shutdown) or 1 (running)
Minimum capacity: 0
Maximum capacity: 1
```

### For Cost Optimization:
- Use spot instances (70-90% cheaper than on-demand)
- Enable capacity rebalancing (handles interruptions)
- Set maximum price (optional, prevents overcharging)

---

## 🚨 If Spot Instance Gets Interrupted

**ASG Handles This Automatically**:
1. AWS sends 2-minute interruption notice
2. ASG marks instance as unhealthy
3. ASG launches replacement instance
4. Your work is safe (backed up to GitHub)

**What You Need to Do**:
1. Check for new instance in ASG
2. Get new IP address
3. SSH in and continue working
4. No data loss (all code on GitHub)

---

## 💰 Cost Tracking

### Spot Instance Costs (Approximate):
- **g5.2xlarge spot**: ~$0.30-0.50/hour
- **g5.xlarge spot**: ~$0.15-0.25/hour
- **On-demand backup**: 3-4x more expensive

### Daily Usage Pattern:
```
6 hours/day × $0.40/hour = $2.40/day
30 days × $2.40/day = $72/month
```

**With ASG at capacity 0**: $0.00/hour when not running ✅

---

## 📊 ASG Best Practices

### 1. Always Set Capacity to 0 When Done
- Prevents accidental charges
- Easy to restart next day

### 2. Use Launch Template
- Consistent instance configuration
- Pre-configured AMI, security groups, storage

### 3. Enable Detailed Monitoring (Optional)
- Track instance metrics
- Better cost analysis

### 4. Set Up CloudWatch Alarms (Optional)
```
- Alarm if instance running > 8 hours
- Alarm if costs exceed $X/day
- Alarm on spot interruption
```

---

## 🔗 Useful AWS Console Links

- **Auto Scaling Groups**: https://console.aws.amazon.com/ec2/autoscaling/
- **EC2 Instances**: https://console.aws.amazon.com/ec2/
- **Spot Requests**: https://console.aws.amazon.com/ec2sp/
- **Cost Explorer**: https://console.aws.amazon.com/cost-management/

---

## ✅ Updated Shutdown Checklist

### Before Setting Capacity to 0:
- [x] All code committed to git
- [x] All commits pushed to GitHub
- [x] Results synced to MacBook
- [x] Documentation complete
- [x] No active training processes

### Shutdown Steps:
1. [ ] Go to Auto Scaling Console
2. [ ] Select `GPU G5 spot – ASG`
3. [ ] Set Desired capacity: 0
4. [ ] Set Minimum capacity: 0
5. [ ] Click Update
6. [ ] Verify instance termination (1-2 min)

### Tomorrow:
1. [ ] Set Desired capacity: 1
2. [ ] Wait for instance launch (2-5 min)
3. [ ] Get new IP address
4. [ ] SSH in and continue working

---

**Status**: ✅ CORRECTED - Use ASG, not manual termination  
**Action**: Set `GPU G5 spot – ASG` desired capacity to 0

Thank you for the correction! 🙏

