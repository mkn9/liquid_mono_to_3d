#!/bin/bash
# Restart 30K generation with checkpoint version for progress visibility

echo "================================================"
echo "RESTART WITH CHECKPOINT VERSION"
echo "================================================"
echo ""

# Kill current run
echo "1. Stopping current generation..."
ssh -i /Users/mike/keys/AutoGenKeyPair.pem ubuntu@34.196.155.11 "pkill -TERM -f generate_parallel_30k && sleep 2 && pkill -9 -f generate_parallel_30k 2>/dev/null; echo 'Stopped.'"

echo ""
echo "2. Starting checkpoint version..."
echo "   (Progress visible every 2-3 minutes)"
echo ""

# Start new run with checkpoints
ssh -i /Users/mike/keys/AutoGenKeyPair.pem ubuntu@34.196.155.11 \
  "cd ~/mono_to_3d/experiments/magvit_I3D_LLM_basic_trajectory && \
   nohup ../../venv/bin/python parallel_dataset_generator_with_checkpoints.py > logs/\$(date +%Y%m%d_%H%M%S)_30k_with_checkpoints.log 2>&1 & \
   echo \"Started PID: \$!\" && \
   sleep 2 && \
   tail -20 logs/*_30k_with_checkpoints.log"

echo ""
echo "================================================"
echo "✅ Checkpoint version started!"
echo "================================================"
echo ""
echo "Monitor progress:"
echo "  ./monitor_checkpoint_progress.sh"
echo ""
echo "Or check manually:"
echo "  ssh ... 'cat ~/mono_to_3d/experiments/magvit_I3D_LLM_basic_trajectory/results/PROGRESS.txt'"

