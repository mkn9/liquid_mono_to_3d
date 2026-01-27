#!/bin/bash
# Launch All Neural Video Generation Experiments
# Usage: ./launch_all_experiments.sh

echo "🚀 Launching all neural video generation experiments..."

# Function to run experiment in background
run_experiment() {
    local exp_name=$1
    local branch=$2
    local script_path=$3
    
    echo "Starting $exp_name on branch $branch..."
    (
        git checkout $branch
        cd $(dirname $script_path)
        python $(basename $script_path) 2>&1 | tee ../logs/${exp_name}_$(date +%Y%m%d_%H%M%S).log
    ) &
    
    echo "Started $exp_name with PID $!"
}

# Create logs directory
mkdir -p logs

# Launch experiments
run_experiment "magvit-2d" "experiment/magvit-2d-trajectories" "experiments/magvit-2d-trajectories/train_magvit_2d.py"
run_experiment "videogpt-2d" "experiment/videogpt-2d-trajectories" "experiments/videogpt-2d-trajectories/train_videogpt_2d.py" 
run_experiment "magvit-3d" "experiment/magvit-3d-trajectories" "experiments/magvit-3d-trajectories/train_magvit_3d.py"

echo "✅ All experiments launched! Check logs/ directory for outputs."
echo "Use 'ps aux | grep python' to see running processes."
