#!/bin/bash
# Diagnose EC2 connectivity and status issues

EC2_IP="34.196.155.11"
SSH_KEY="/Users/mike/keys/AutoGenKeyPair.pem"

echo "======================================"
echo "EC2 Instance Diagnostic"
echo "======================================"
echo "Target: $EC2_IP"
echo "Time: $(date)"
echo ""

echo "=== 1. Network Ping Test ==="
if ping -c 3 -W 2 "$EC2_IP" > /dev/null 2>&1; then
    echo "✅ Instance responds to ping"
else
    echo "❌ Instance does NOT respond to ping (100% packet loss)"
fi
echo ""

echo "=== 2. SSH Port Check ==="
if nc -zv -w5 "$EC2_IP" 22 2>&1 | grep -q "succeeded"; then
    echo "✅ SSH port 22 is open"
else
    echo "❌ SSH port 22 is not responding"
fi
echo ""

echo "=== 3. SSH Connection Test ==="
if timeout 10 ssh -i "$SSH_KEY" -o ConnectTimeout=5 -o BatchMode=yes "ubuntu@$EC2_IP" "echo 'Connected'" 2>/dev/null | grep -q "Connected"; then
    echo "✅ SSH connection successful"
else
    echo "❌ SSH connection failed or timed out"
fi
echo ""

echo "=== 4. Alternative Diagnostic Commands ==="
echo "If you have AWS CLI configured, try:"
echo "  aws ec2 describe-instances --instance-ids <INSTANCE_ID>"
echo "  aws ec2 get-console-output --instance-id <INSTANCE_ID>"
echo ""

echo "=== 5. Possible Issues ==="
echo "❌ Instance crashed due to resource exhaustion"
echo "❌ Instance was stopped (manually or by AWS)"
echo "❌ Instance froze under extreme GPU/CPU load"
echo "❌ Out of memory killed critical processes"
echo "❌ Disk filled up completely (even /tmp)"
echo "❌ Network configuration changed"
echo ""

echo "=== 6. Recovery Options ==="
echo ""
echo "Option A: Reboot instance (if running)"
echo "  aws ec2 reboot-instances --instance-ids <INSTANCE_ID>"
echo ""
echo "Option B: Start instance (if stopped)"
echo "  aws ec2 start-instances --instance-ids <INSTANCE_ID>"
echo ""
echo "Option C: Connect via AWS Systems Manager (if configured)"
echo "  aws ssm start-session --target <INSTANCE_ID>"
echo ""
echo "Option D: Check console logs"
echo "  AWS Console → EC2 → Instance → Actions → Monitor and troubleshoot → Get system log"
echo ""

echo "=== 7. What About Training Results? ==="
echo ""
echo "If instance is stopped/terminated:"
echo "  ❌ All training results are LOST (not on persistent storage)"
echo "  ❌ No results synced to MacBook"
echo "  ❌ Must restart training from scratch"
echo ""
echo "If instance is running but frozen:"
echo "  ⚠️  Results may be on disk but inaccessible"
echo "  ✅ Reboot may allow recovery"
echo ""

echo "======================================"
echo "Check AWS Console for instance status"
echo "======================================"

