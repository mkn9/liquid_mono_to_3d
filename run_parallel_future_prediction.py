#!/usr/bin/env python3
"""
Run Parallel Future Prediction Experiments
==========================================
Executes 3 branches in parallel on EC2.
"""

import subprocess
import threading
import time
import json
from pathlib import Path
from datetime import datetime
import sys


class ParallelTaskRunner:
    """Run multiple tasks in parallel."""
    
    def __init__(self, tasks: list, ec2_host: str, ec2_key: str):
        self.tasks = tasks
        self.ec2_host = ec2_host
        self.ec2_key = ec2_key
        self.results = {}
        self.threads = []
        
    def run_task(self, task_name: str, script_path: str):
        """Run a single task on EC2."""
        print(f"\n[{task_name}] Starting...")
        
        # SSH command to run on EC2
        cmd = [
            'ssh', '-i', self.ec2_key, self.ec2_host,
            f'cd ~/mono_to_3d && python3 {script_path} > /tmp/{task_name}.log 2>&1 && echo "SUCCESS" || echo "FAILED"'
        ]
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )
            
            duration = time.time() - start_time
            
            self.results[task_name] = {
                'status': 'success' if 'SUCCESS' in result.stdout else 'failed',
                'duration': duration,
                'stdout': result.stdout[-500:],  # Last 500 chars
                'stderr': result.stderr[-500:] if result.stderr else None
            }
            
            print(f"[{task_name}] Completed in {duration:.1f}s")
            
        except subprocess.TimeoutExpired:
            self.results[task_name] = {
                'status': 'timeout',
                'duration': 7200,
                'error': 'Task exceeded 2 hour timeout'
            }
            print(f"[{task_name}] TIMEOUT after 2 hours")
            
        except Exception as e:
            self.results[task_name] = {
                'status': 'error',
                'error': str(e)
            }
            print(f"[{task_name}] ERROR: {e}")
    
    def run_all(self):
        """Run all tasks in parallel."""
        print("=" * 60)
        print("Starting Parallel Execution")
        print("=" * 60)
        print(f"Tasks: {len(self.tasks)}")
        print(f"EC2 Host: {self.ec2_host}")
        print("=" * 60)
        
        # Create threads for each task
        for task in self.tasks:
            thread = threading.Thread(
                target=self.run_task,
                args=(task['name'], task['script'])
            )
            thread.start()
            self.threads.append(thread)
            time.sleep(5)  # Stagger starts by 5 seconds
        
        # Wait for all to complete
        print("\nWaiting for all tasks to complete...")
        for thread in self.threads:
            thread.join()
        
        print("\n" + "=" * 60)
        print("Parallel Execution Complete")
        print("=" * 60)
        
        # Print summary
        for task_name, result in self.results.items():
            status = result.get('status', 'unknown')
            duration = result.get('duration', 0)
            print(f"{task_name}: {status.upper()} ({duration:.1f}s)")
        
        # Save results
        results_file = Path('parallel_execution_results.json')
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'tasks': self.tasks,
                'results': self.results
            }, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        
        return self.results


def main():
    """Main execution."""
    
    # Configuration
    EC2_HOST = "ubuntu@34.196.155.11"
    EC2_KEY = "/Users/mike/keys/AutoGenKeyPair.pem"
    
    # Tasks to run in parallel
    tasks = [
        {
            'name': 'baseline',
            'script': 'experiments/future-prediction/train_baseline.py'
        },
        {
            'name': 'joint_i3d',
            'script': 'experiments/future-prediction/train_joint_i3d.py'
        },
        {
            'name': 'slowfast',
            'script': 'experiments/future-prediction/train_slowfast.py'
        }
    ]
    
    # Run parallel execution
    runner = ParallelTaskRunner(tasks, EC2_HOST, EC2_KEY)
    results = runner.run_all()
    
    # Check if all succeeded
    all_success = all(r.get('status') == 'success' for r in results.values())
    
    if all_success:
        print("\n✅ All tasks completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tasks failed or timed out")
        sys.exit(1)


if __name__ == '__main__':
    main()

