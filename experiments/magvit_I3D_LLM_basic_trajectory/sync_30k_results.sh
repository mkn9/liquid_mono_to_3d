#!/bin/bash
# Sync 30K dataset results from EC2 to MacBook

echo "Syncing 30K dataset from EC2..."

# Sync the 30K dataset file
scp -i /Users/mike/keys/AutoGenKeyPair.pem \
    ubuntu@34.196.155.11:~/mono_to_3d/experiments/magvit_I3D_LLM_basic_trajectory/results/*30k*.npz \
    /Users/mike/Dropbox/Documents/Machine_Learning/LLM_and_Multimodal_Models/repos/mono_to_3d/experiments/magvit_I3D_LLM_basic_trajectory/results/

# Sync the generation log
scp -i /Users/mike/keys/AutoGenKeyPair.pem \
    ubuntu@34.196.155.11:~/mono_to_3d/experiments/magvit_I3D_LLM_basic_trajectory/logs/20260125_005159_parallel_30k_generation.log \
    /Users/mike/Dropbox/Documents/Machine_Learning/LLM_and_Multimodal_Models/repos/mono_to_3d/experiments/magvit_I3D_LLM_basic_trajectory/logs/

echo "✓ Sync complete!"
echo ""
echo "Files synced:"
ls -lh /Users/mike/Dropbox/Documents/Machine_Learning/LLM_and_Multimodal_Models/repos/mono_to_3d/experiments/magvit_I3D_LLM_basic_trajectory/results/*30k*.npz 2>/dev/null || echo "  (no 30k file found)"

