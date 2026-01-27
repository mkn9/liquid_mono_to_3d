#!/bin/bash
# Launch all 4 branches in parallel

cd /home/ubuntu/mono_to_3d/experiments/magvit_I3D_LLM_basic_trajectory

echo "Launching all 4 branches in parallel..."

# Branch 1
python3 branch1/train.py > branch1/results/training.log 2>&1 &
PID1=$!
echo "Branch 1 (I3D+MAGVIT+GPT4) started: PID $PID1"

# Branch 2
python3 branch2/train.py > branch2/results/training.log 2>&1 &
PID2=$!
echo "Branch 2 (SlowFast+MAGVIT+GPT4) started: PID $PID2"

# Branch 3
python3 branch3/train.py > branch3/results/training.log 2>&1 &
PID3=$!
echo "Branch 3 (I3D+CLIP+Mistral) started: PID $PID3"

# Branch 4
python3 branch4/train.py > branch4/results/training.log 2>&1 &
PID4=$!
echo "Branch 4 (SlowFast+Phi2) started: PID $PID4"

echo ""
echo "All 4 branches launched!"
echo "Monitor with: tail -f branch*/results/training.log"
echo "Or check status: cat branch*/status/status.json"
echo ""
echo "PIDs: $PID1 $PID2 $PID3 $PID4"
echo "Wait for all: wait $PID1 $PID2 $PID3 $PID4"

# Wait for all to complete
wait $PID1 $PID2 $PID3 $PID4

echo ""
echo "============================="
echo "All 4 branches completed!"
echo "============================="

