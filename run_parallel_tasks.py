#!/usr/bin/env python3
"""
Parallel Task Execution Script
==============================
Executes multiple tasks in parallel using multiprocessing.
Enhanced version that supports true parallel execution on EC2.
"""

import sys
import os
import yaml
import subprocess
import multiprocessing
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import time
import json

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: str = 'config.yaml') -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_task_on_ec2(task_name: str, task_config: Dict, config: Dict) -> Dict:
    """Run a single task on EC2 and return results."""
    script_path = task_config['script']
    branch_name = task_config['branch']
    ec2_host = config['project']['ec2_host']
    ec2_key = config['project']['ec2_key']
    ec2_path = config['project']['ec2_path']
    python_env = config['execution']['ec2']['python_env']
    
    # Build command
    cmd = f"""
    cd {ec2_path} && \
    source {python_env} && \
    git fetch origin && \
    git checkout {branch_name} && \
    git pull origin {branch_name} 2>/dev/null || true && \
    python3 {script_path} 2>&1
    """
    
    ssh_cmd = [
        'ssh',
        '-i', ec2_key,
        '-o', 'StrictHostKeyChecking=no',
        '-o', 'ConnectTimeout=10',
        ec2_host,
        cmd
    ]
    
    print(f"[{task_name}] Starting execution on EC2 (branch: {branch_name})...")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        elapsed_time = time.time() - start_time
        
        return {
            'task_name': task_name,
            'branch': branch_name,
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'elapsed_time': elapsed_time,
            'timestamp': datetime.now().isoformat()
        }
    except subprocess.TimeoutExpired:
        return {
            'task_name': task_name,
            'branch': branch_name,
            'success': False,
            'error': 'Timeout after 1 hour',
            'elapsed_time': 3600,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'task_name': task_name,
            'branch': branch_name,
            'success': False,
            'error': str(e),
            'elapsed_time': time.time() - start_time,
            'timestamp': datetime.now().isoformat()
        }


def transfer_results_from_ec2(task_name: str, task_config: Dict, config: Dict) -> bool:
    """Transfer results from EC2 to MacBook."""
    ec2_host = config['project']['ec2_host']
    ec2_key = config['project']['ec2_key']
    ec2_path = config['project']['ec2_path']
    output_dir = task_config.get('output_dir', 'experiments')
    
    # Build rsync command
    ec2_output = f"{ec2_path}/{output_dir}"
    local_output = Path(PROJECT_ROOT / output_dir)
    local_output.mkdir(parents=True, exist_ok=True)
    
    rsync_cmd = [
        'rsync',
        '-avz',
        '--progress',
        '-e', f"ssh -i {ec2_key} -o StrictHostKeyChecking=no",
        f"{ec2_host}:{ec2_output}/",
        str(local_output) + '/',
        '--update'
    ]
    
    try:
        result = subprocess.run(rsync_cmd, capture_output=True, text=True, timeout=300)
        return result.returncode == 0
    except Exception as e:
        print(f"  ⚠️  Transfer failed for {task_name}: {e}")
        return False


def monitor_task_progress(task_name: str, task_config: Dict, config: Dict, check_interval: int = 60):
    """Monitor task progress by checking output files."""
    ec2_host = config['project']['ec2_host']
    ec2_key = config['project']['ec2_key']
    ec2_path = config['project']['ec2_path']
    output_dir = task_config.get('output_dir', 'experiments')
    
    check_cmd = f"ls -la {ec2_path}/{output_dir}/*.json 2>/dev/null | tail -1"
    
    ssh_cmd = [
        'ssh',
        '-i', ec2_key,
        '-o', 'StrictHostKeyChecking=no',
        ec2_host,
        check_cmd
    ]
    
    last_file = None
    while True:
        try:
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                current_file = result.stdout.strip().split()[-1] if result.stdout.strip() else None
                if current_file != last_file:
                    print(f"[{task_name}] New output file detected: {current_file}")
                    last_file = current_file
        except:
            pass
        
        time.sleep(check_interval)


def execute_parallel_tasks(task_names: List[str], max_workers: int = 3):
    """Execute multiple tasks in parallel."""
    config = load_config()
    tasks = config['tasks']
    
    # Filter to requested tasks
    selected_tasks = {name: tasks[name] for name in task_names if name in tasks}
    
    if not selected_tasks:
        print("❌ No valid tasks found")
        return
    
    print("=" * 60)
    print("Parallel Task Execution")
    print("=" * 60)
    print(f"Tasks: {', '.join(selected_tasks.keys())}")
    print(f"Max parallel workers: {max_workers}")
    print(f"Execution location: {config['execution']['location']}")
    print()
    
    # Create process pool
    with multiprocessing.Pool(processes=max_workers) as pool:
        # Submit all tasks
        print("Submitting tasks to execution pool...")
        results = []
        
        for task_name, task_config in selected_tasks.items():
            if config['execution']['location'] == 'ec2':
                result = pool.apply_async(
                    run_task_on_ec2,
                    args=(task_name, task_config, config)
                )
                results.append((task_name, result))
            else:
                # Local execution (sequential for now)
                result = run_task_on_ec2(task_name, task_config, config)
                results.append((task_name, result))
        
        print(f"✅ {len(results)} tasks submitted")
        print("\nWaiting for completion...")
        print("-" * 60)
        
        # Collect results
        final_results = {}
        for task_name, result in results:
            if isinstance(result, multiprocessing.pool.AsyncResult):
                task_result = result.get()  # Wait for completion
            else:
                task_result = result
            
            final_results[task_name] = task_result
            
            status = "✅" if task_result.get('success') else "❌"
            elapsed = task_result.get('elapsed_time', 0)
            print(f"{status} [{task_name}] Completed in {elapsed:.1f}s")
            
            if task_result.get('success'):
                # Transfer results
                if config['execution']['options']['transfer_results']:
                    print(f"  Transferring results for {task_name}...")
                    transfer_success = transfer_results_from_ec2(task_name, selected_tasks[task_name], config)
                    if transfer_success:
                        print(f"  ✅ Results transferred")
                    else:
                        print(f"  ⚠️  Transfer had issues")
            else:
                error = task_result.get('error', 'Unknown error')
                print(f"  ❌ Error: {error}")
                if task_result.get('stderr'):
                    print(f"  Stderr: {task_result['stderr'][:200]}")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'tasks_executed': list(selected_tasks.keys()),
        'results': final_results,
        'summary': {
            'total': len(final_results),
            'successful': sum(1 for r in final_results.values() if r.get('success')),
            'failed': sum(1 for r in final_results.values() if not r.get('success'))
        }
    }
    
    summary_path = PROJECT_ROOT / 'parallel_execution_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Execution Summary")
    print("=" * 60)
    print(f"Total tasks: {summary['summary']['total']}")
    print(f"Successful: {summary['summary']['successful']}")
    print(f"Failed: {summary['summary']['failed']}")
    print(f"\n✅ Summary saved to: {summary_path}")
    
    return summary


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Execute tasks in parallel')
    parser.add_argument(
        'tasks',
        nargs='*',
        help='Task names to execute (or all parallel tasks if none specified)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=3,
        help='Maximum number of parallel workers'
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to config.yaml'
    )
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.tasks:
        task_names = args.tasks
    else:
        # Get all parallel tasks
        task_names = [
            name for name, task in config['tasks'].items()
            if task.get('parallel', True)
        ]
    
    if not task_names:
        print("❌ No tasks to execute")
        return
    
    execute_parallel_tasks(task_names, max_workers=args.max_workers)


if __name__ == '__main__':
    main()

