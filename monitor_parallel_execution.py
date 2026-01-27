#!/usr/bin/env python3
"""
Monitor Parallel Task Execution
================================
Monitors the progress of parallel task execution by checking EC2 output files.
"""

import sys
import yaml
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: str = 'config.yaml') -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def check_task_output(task_name: str, task_config: Dict, config: Dict) -> Dict:
    """Check if task has produced output files."""
    ec2_host = config['project']['ec2_host']
    ec2_key = config['project']['ec2_key']
    ec2_path = config['project']['ec2_path']
    output_dir = task_config.get('output_dir', 'experiments')
    
    # Check for output files
    check_cmd = f"""
    cd {ec2_path}/{output_dir} && \
    ls -t *.json 2>/dev/null | head -1 && \
    find . -name "*{task_name}*" -type f -mmin -10 2>/dev/null | head -5
    """
    
    ssh_cmd = [
        'ssh',
        '-i', ec2_key,
        '-o', 'StrictHostKeyChecking=no',
        ec2_host,
        check_cmd
    ]
    
    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
            return {
                'task_name': task_name,
                'has_output': len(files) > 0,
                'output_files': files,
                'status': 'running' if files else 'no_output'
            }
    except Exception as e:
        return {
            'task_name': task_name,
            'has_output': False,
            'error': str(e),
            'status': 'error'
        }
    
    return {
        'task_name': task_name,
        'has_output': False,
        'status': 'no_output'
    }


def check_process_status(task_name: str, task_config: Dict, config: Dict) -> Dict:
    """Check if task process is still running on EC2."""
    ec2_host = config['project']['ec2_host']
    ec2_key = config['project']['ec2_key']
    script_path = task_config['script']
    script_name = Path(script_path).name
    
    check_cmd = f"ps aux | grep '{script_name}' | grep -v grep | wc -l"
    
    ssh_cmd = [
        'ssh',
        '-i', ec2_key,
        '-o', 'StrictHostKeyChecking=no',
        ec2_host,
        check_cmd
    ]
    
    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            count = int(result.stdout.strip())
            return {
                'task_name': task_name,
                'is_running': count > 0,
                'process_count': count
            }
    except:
        pass
    
    return {
        'task_name': task_name,
        'is_running': False,
        'process_count': 0
    }


def monitor_tasks(task_names: List[str], check_interval: int = 60, max_checks: int = 60):
    """Monitor multiple tasks."""
    config = load_config()
    tasks = config['tasks']
    
    selected_tasks = {name: tasks[name] for name in task_names if name in tasks}
    
    if not selected_tasks:
        print("❌ No valid tasks found")
        return
    
    print("=" * 60)
    print("Monitoring Parallel Task Execution")
    print("=" * 60)
    print(f"Tasks: {', '.join(selected_tasks.keys())}")
    print(f"Check interval: {check_interval}s")
    print(f"Max checks: {max_checks}")
    print()
    
    check_count = 0
    
    while check_count < max_checks:
        check_count += 1
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"\n[{timestamp}] Check #{check_count}/{max_checks}")
        print("-" * 60)
        
        all_complete = True
        
        for task_name, task_config in selected_tasks.items():
            # Check process status
            proc_status = check_process_status(task_name, task_config, config)
            # Check output files
            output_status = check_task_output(task_name, task_config, config)
            
            if proc_status['is_running']:
                status_icon = "🔄"
                status_text = "RUNNING"
                all_complete = False
            elif output_status['has_output']:
                status_icon = "✅"
                status_text = "COMPLETE (output found)"
            else:
                status_icon = "⏳"
                status_text = "WAITING/UNKNOWN"
                all_complete = False
            
            print(f"{status_icon} [{task_name}] {status_text}")
            
            if output_status.get('output_files'):
                for f in output_status['output_files'][:3]:  # Show first 3
                    print(f"     📄 {f}")
        
        if all_complete:
            print("\n✅ All tasks appear to be complete!")
            break
        
        if check_count < max_checks:
            print(f"\n⏳ Waiting {check_interval}s before next check...")
            time.sleep(check_interval)
    
    print("\n" + "=" * 60)
    print("Monitoring Complete")
    print("=" * 60)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor parallel task execution')
    parser.add_argument(
        'tasks',
        nargs='*',
        help='Task names to monitor (or all if none specified)'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='Check interval in seconds'
    )
    parser.add_argument(
        '--max-checks',
        type=int,
        default=60,
        help='Maximum number of checks'
    )
    
    args = parser.parse_args()
    
    config = load_config()
    
    if args.tasks:
        task_names = args.tasks
    else:
        # Get all parallel tasks
        task_names = [
            name for name, task in config['tasks'].items()
            if task.get('parallel', True)
        ]
    
    if not task_names:
        print("❌ No tasks to monitor")
        return
    
    monitor_tasks(task_names, check_interval=args.interval, max_checks=args.max_checks)


if __name__ == '__main__':
    main()

