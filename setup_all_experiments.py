#!/usr/bin/env python3
"""
Multi-Experiment Video Generation Setup
=======================================

Master setup script for parallel execution of:
1. MAGVIT 2D Trajectories (squares, circles, triangles)
2. VideoGPT 2D Trajectories (squares, circles, triangles)
3. MAGVIT 3D Trajectories (cubes, cylinders, cones)

This script provides branch management and parallel execution infrastructure.
"""

import os
import subprocess
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiExperimentManager:
    """Manager for multiple neural video generation experiments."""
    
    def __init__(self):
        self.experiments = {
            'magvit-2d': {
                'branch': 'experiment/magvit-2d-trajectories',
                'setup_script': 'experiments/magvit-2d-trajectories/setup_magvit_2d.py',
                'train_script': 'experiments/magvit-2d-trajectories/train_magvit_2d.py',
                'description': 'MAGVIT 2D trajectory prediction (squares, circles, triangles)',
                'dependencies': ['jax', 'flax', 'tensorflow'],
                'gpu_memory': '8GB'
            },
            'videogpt-2d': {
                'branch': 'experiment/videogpt-2d-trajectories',
                'setup_script': 'experiments/videogpt-2d-trajectories/setup_videogpt_2d.py',
                'train_script': 'experiments/videogpt-2d-trajectories/train_videogpt_2d.py',
                'description': 'VideoGPT 2D trajectory prediction (squares, circles, triangles)',
                'dependencies': ['torch', 'pytorch-lightning', 'transformers'],
                'gpu_memory': '10GB'
            },
            'magvit-3d': {
                'branch': 'experiment/magvit-3d-trajectories',
                'setup_script': 'experiments/magvit-3d-trajectories/setup_magvit_3d.py',
                'train_script': 'experiments/magvit-3d-trajectories/train_magvit_3d.py',
                'description': 'MAGVIT 3D trajectory prediction (cubes, cylinders, cones)',
                'dependencies': ['jax', 'flax', 'open3d', 'trimesh'],
                'gpu_memory': '12GB'
            }
        }
        
        self.current_branch = self.get_current_branch()
        self.base_dir = Path.cwd()
    
    def get_current_branch(self) -> str:
        """Get current git branch."""
        try:
            result = subprocess.run(['git', 'branch', '--show-current'], 
                                  capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return "unknown"
    
    def create_experiment_branches(self) -> bool:
        """Create all experiment branches if they don't exist."""
        logger.info("Creating experiment branches...")
        
        # Ensure we're on master
        try:
            subprocess.run(['git', 'checkout', 'master'], check=True)
            logger.info("Switched to master branch")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to switch to master: {e}")
            return False
        
        # Create branches for each experiment
        for exp_name, exp_config in self.experiments.items():
            branch_name = exp_config['branch']
            
            # Check if branch exists
            try:
                result = subprocess.run(['git', 'branch', '--list', branch_name], 
                                      capture_output=True, text=True)
                if branch_name in result.stdout:
                    logger.info(f"Branch {branch_name} already exists")
                else:
                    # Create new branch
                    subprocess.run(['git', 'checkout', '-b', branch_name], check=True)
                    subprocess.run(['git', 'checkout', 'master'], check=True)
                    logger.info(f"Created branch {branch_name}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to create branch {branch_name}: {e}")
                return False
        
        return True
    
    def setup_experiment(self, experiment_name: str) -> bool:
        """Set up a single experiment."""
        exp_config = self.experiments[experiment_name]
        logger.info(f"Setting up {experiment_name}: {exp_config['description']}")
        
        # Switch to experiment branch
        try:
            subprocess.run(['git', 'checkout', exp_config['branch']], check=True)
            
            # Run setup script
            setup_script = Path(exp_config['setup_script'])
            if setup_script.exists():
                subprocess.run([sys.executable, str(setup_script)], check=True)
                logger.info(f"Setup completed for {experiment_name}")
                return True
            else:
                logger.error(f"Setup script not found: {setup_script}")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to setup {experiment_name}: {e}")
            return False
        finally:
            # Return to original branch
            subprocess.run(['git', 'checkout', self.current_branch], check=False)
    
    def setup_all_experiments(self) -> Dict[str, bool]:
        """Set up all experiments in parallel."""
        logger.info("Setting up all experiments...")
        
        # First create branches
        if not self.create_experiment_branches():
            return {exp: False for exp in self.experiments.keys()}
        
        # Setup experiments sequentially to avoid git conflicts
        results = {}
        for exp_name in self.experiments.keys():
            results[exp_name] = self.setup_experiment(exp_name)
        
        return results
    
    def run_experiment(self, experiment_name: str, background: bool = True) -> subprocess.Popen:
        """Run a single experiment."""
        exp_config = self.experiments[experiment_name]
        logger.info(f"Starting {experiment_name} training...")
        
        # Prepare command
        train_script = Path(exp_config['train_script'])
        if not train_script.exists():
            logger.error(f"Training script not found: {train_script}")
            return None
        
        # Switch to experiment branch and run
        commands = [
            f"git checkout {exp_config['branch']}",
            f"cd {train_script.parent}",
            f"python {train_script.name}"
        ]
        
        command = " && ".join(commands)
        
        if background:
            # Run in background
            process = subprocess.Popen(command, shell=True)
            logger.info(f"Started {experiment_name} with PID {process.pid}")
            return process
        else:
            # Run synchronously
            subprocess.run(command, shell=True, check=True)
            return None
    
    def run_all_experiments_parallel(self) -> Dict[str, subprocess.Popen]:
        """Run all experiments in parallel."""
        logger.info("Starting all experiments in parallel...")
        processes = {}
        
        for exp_name in self.experiments.keys():
            process = self.run_experiment(exp_name, background=True)
            if process:
                processes[exp_name] = process
        
        return processes
    
    def monitor_experiments(self, processes: Dict[str, subprocess.Popen]):
        """Monitor running experiments."""
        logger.info("Monitoring experiments...")
        
        while processes:
            time.sleep(30)  # Check every 30 seconds
            
            completed = []
            for exp_name, process in processes.items():
                poll_result = process.poll()
                if poll_result is not None:
                    if poll_result == 0:
                        logger.info(f"✅ {exp_name} completed successfully")
                    else:
                        logger.error(f"❌ {exp_name} failed with code {poll_result}")
                    completed.append(exp_name)
                else:
                    logger.info(f"🔄 {exp_name} running (PID: {process.pid})")
            
            # Remove completed processes
            for exp_name in completed:
                del processes[exp_name]
        
        logger.info("All experiments completed!")
    
    def generate_experiment_summary(self) -> Dict[str, Any]:
        """Generate a summary of all experiments."""
        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_experiments': len(self.experiments),
            'experiments': {}
        }
        
        for exp_name, exp_config in self.experiments.items():
            summary['experiments'][exp_name] = {
                'description': exp_config['description'],
                'branch': exp_config['branch'],
                'dependencies': exp_config['dependencies'],
                'gpu_memory': exp_config['gpu_memory'],
                'status': 'configured'
            }
        
        return summary
    
    def save_experiment_summary(self, summary: Dict[str, Any]):
        """Save experiment summary to file."""
        summary_path = self.base_dir / "experiment_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Experiment summary saved to {summary_path}")

def create_parallel_execution_scripts():
    """Create helper scripts for parallel execution."""
    logger.info("Creating parallel execution scripts...")
    
    # Create launch script
    launch_script = Path("launch_all_experiments.sh")
    with open(launch_script, 'w') as f:
        f.write("""#!/bin/bash
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
""")
    
    launch_script.chmod(0o755)
    
    # Create monitoring script
    monitor_script = Path("monitor_experiments.py")
    with open(monitor_script, 'w') as f:
        f.write("""#!/usr/bin/env python3
import psutil
import time
import subprocess

def monitor_experiments():
    print("🔍 Monitoring experiment processes...")
    
    while True:
        python_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
            try:
                if proc.info['name'] == 'python' and 'train_' in ' '.join(proc.info['cmdline']):
                    python_processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        if python_processes:
            print(f"\\n📊 Active experiments ({len(python_processes)} processes):")
            for proc in python_processes:
                cmdline = ' '.join(proc['cmdline'])
                if 'magvit' in cmdline:
                    exp_type = '🧠 MAGVIT'
                elif 'videogpt' in cmdline:
                    exp_type = '🎬 VideoGPT'
                else:
                    exp_type = '⚡ Unknown'
                
                print(f"  {exp_type} PID:{proc['pid']} CPU:{proc['cpu_percent']:.1f}% MEM:{proc['memory_percent']:.1f}%")
        else:
            print("\\n✅ No experiment processes running")
            break
        
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    monitor_experiments()
""")
    
    monitor_script.chmod(0o755)
    
    logger.info("Parallel execution scripts created!")

def main():
    """Main function."""
    logger.info("🚀 Multi-Experiment Video Generation Setup")
    
    # Initialize manager
    manager = MultiExperimentManager()
    
    # Create execution scripts
    create_parallel_execution_scripts()
    
    # Setup all experiments
    setup_results = manager.setup_all_experiments()
    
    # Print results
    logger.info("📋 Setup Results:")
    for exp_name, success in setup_results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        description = manager.experiments[exp_name]['description']
        logger.info(f"  {status} - {exp_name}: {description}")
    
    # Generate and save summary
    summary = manager.generate_experiment_summary()
    manager.save_experiment_summary(summary)
    
    # Instructions
    logger.info("🎯 Next Steps:")
    logger.info("  1. Run all experiments: ./launch_all_experiments.sh")
    logger.info("  2. Monitor progress: python monitor_experiments.py")
    logger.info("  3. Check individual logs in logs/ directory")
    
    # Ask if user wants to run experiments now
    try:
        run_now = input("\\n🤔 Would you like to start all experiments now? (y/N): ").strip().lower()
        if run_now in ['y', 'yes']:
            logger.info("Starting all experiments...")
            processes = manager.run_all_experiments_parallel()
            if processes:
                manager.monitor_experiments(processes)
    except KeyboardInterrupt:
        logger.info("\\n👋 Setup complete! Use the launch script to start experiments later.")

if __name__ == "__main__":
    main() 