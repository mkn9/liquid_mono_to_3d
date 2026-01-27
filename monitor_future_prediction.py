#!/usr/bin/env python3
"""
Monitor Future Prediction Experiments
=====================================
Provides periodic updates on running experiments.
"""

import subprocess
import time
import json
from pathlib import Path
from datetime import datetime
import sys


class ExperimentMonitor:
    """Monitor running experiments and sync results."""
    
    def __init__(self, ec2_host: str, ec2_key: str, interval: int = 300):
        self.ec2_host = ec2_host
        self.ec2_key = ec2_key
        self.interval = interval
        self.tasks = ['baseline', 'joint_i3d', 'slowfast']
        
    def check_task_status(self, task_name: str) -> dict:
        """Check status of a single task."""
        
        # Check if Python process is running
        cmd_check_process = [
            'ssh', '-i', self.ec2_key, self.ec2_host,
            f'pgrep -f "train_{task_name}.py" > /dev/null && echo "RUNNING" || echo "NOT_RUNNING"'
        ]
        
        try:
            result = subprocess.run(cmd_check_process, capture_output=True, text=True, timeout=30)
            is_running = 'RUNNING' in result.stdout
        except:
            is_running = False
        
        # Check latest results file
        results_path = f"~/mono_to_3d/experiments/future-prediction/output/{task_name}/results"
        cmd_get_results = [
            'ssh', '-i', self.ec2_key, self.ec2_host,
            f'ls -t {results_path}/*_results.json 2>/dev/null | head -1 | xargs cat 2>/dev/null || echo "{{}}"'
        ]
        
        try:
            result = subprocess.run(cmd_get_results, capture_output=True, text=True, timeout=30)
            results = json.loads(result.stdout) if result.stdout.strip() else {}
        except:
            results = {}
        
        # Get latest log tail
        log_path = f"~/mono_to_3d/experiments/future-prediction/output/{task_name}/logs"
        cmd_get_log = [
            'ssh', '-i', self.ec2_key, self.ec2_host,
            f'ls -t {log_path}/*.log 2>/dev/null | head -1 | xargs tail -5 2>/dev/null || echo "No logs"'
        ]
        
        try:
            result = subprocess.run(cmd_get_log, capture_output=True, text=True, timeout=30)
            log_tail = result.stdout.strip()
        except:
            log_tail = "Could not fetch logs"
        
        return {
            'is_running': is_running,
            'results': results,
            'log_tail': log_tail
        }
    
    def sync_results(self):
        """Sync all results from EC2 to local."""
        print("\n📥 Syncing results from EC2...")
        
        local_output = Path('experiments/future-prediction/output')
        local_output.mkdir(parents=True, exist_ok=True)
        
        # rsync results
        cmd = [
            'rsync', '-avz', '-e', f'ssh -i {self.ec2_key}',
            f'{self.ec2_host}:~/mono_to_3d/experiments/future-prediction/output/',
            str(local_output) + '/'
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, timeout=300)
            print("✅ Results synced")
        except Exception as e:
            print(f"⚠️  Sync failed: {e}")
    
    def display_status(self):
        """Display current status of all tasks."""
        print("\n" + "=" * 80)
        print(f"Future Prediction Monitoring - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        for task_name in self.tasks:
            print(f"\n📊 Task: {task_name.upper()}")
            print("-" * 80)
            
            status = self.check_task_status(task_name)
            
            # Running status
            if status['is_running']:
                print("  Status: 🟢 RUNNING")
            else:
                print("  Status: ⚪ NOT RUNNING")
            
            # Results
            results = status['results']
            if results:
                # Training progress
                if 'training_results' in results:
                    tr = results['training_results']
                    if 'epochs' in tr and tr['epochs']:
                        latest_epoch = tr['epochs'][-1]
                        latest_loss = tr['losses'][-1] if 'losses' in tr and tr['losses'] else 'N/A'
                        print(f"  Progress: Epoch {latest_epoch}, Loss: {latest_loss}")
                    
                    if 'metrics' in tr and tr['metrics']:
                        latest_metrics = tr['metrics'][-1]
                        print(f"  Metrics: MSE={latest_metrics.get('mse', 'N/A'):.6f}, "
                              f"PSNR={latest_metrics.get('psnr', 'N/A'):.2f}")
                
                # Test results
                if 'test_results' in results:
                    test_res = results['test_results']
                    passed = test_res.get('passed', 0)
                    total = test_res.get('total_tests', 0)
                    print(f"  Tests: {passed}/{total} passed")
            else:
                print("  Results: No results yet")
            
            # Recent logs
            print(f"  Recent logs:")
            for line in status['log_tail'].split('\n')[-3:]:
                if line.strip():
                    print(f"    {line}")
        
        print("\n" + "=" * 80)
    
    def monitor_continuous(self, duration: int = None):
        """Monitor continuously until completion or timeout."""
        print("🔍 Starting continuous monitoring...")
        print(f"Check interval: {self.interval}s")
        if duration:
            print(f"Max duration: {duration}s")
        print("Press Ctrl+C to stop")
        
        start_time = time.time()
        iteration = 0
        
        try:
            while True:
                iteration += 1
                print(f"\n{'='*80}")
                print(f"Monitoring Iteration #{iteration}")
                print(f"{'='*80}")
                
                # Display status
                self.display_status()
                
                # Sync results every 2 iterations (10 minutes if interval=300s)
                if iteration % 2 == 0:
                    self.sync_results()
                
                # Check if any tasks still running
                any_running = False
                for task_name in self.tasks:
                    status = self.check_task_status(task_name)
                    if status['is_running']:
                        any_running = True
                        break
                
                if not any_running:
                    print("\n✅ All tasks completed!")
                    self.sync_results()  # Final sync
                    break
                
                # Check timeout
                if duration and (time.time() - start_time) > duration:
                    print(f"\n⏱️  Monitoring timeout ({duration}s) reached")
                    self.sync_results()  # Final sync
                    break
                
                # Wait for next check
                print(f"\n⏳ Next check in {self.interval}s...")
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            print("\n\n⛔ Monitoring stopped by user")
            self.sync_results()  # Final sync
        
        print("\n" + "=" * 80)
        print("Monitoring Complete")
        print("=" * 80)


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor future prediction experiments')
    parser.add_argument('--interval', type=int, default=300, help='Check interval in seconds (default: 300)')
    parser.add_argument('--duration', type=int, default=None, help='Max monitoring duration in seconds')
    parser.add_argument('--sync-only', action='store_true', help='Just sync results and exit')
    
    args = parser.parse_args()
    
    # Configuration
    EC2_HOST = "ubuntu@34.196.155.11"
    EC2_KEY = "/Users/mike/keys/AutoGenKeyPair.pem"
    
    # Create monitor
    monitor = ExperimentMonitor(EC2_HOST, EC2_KEY, args.interval)
    
    if args.sync_only:
        print("Syncing results only...")
        monitor.sync_results()
    else:
        # Start monitoring
        monitor.monitor_continuous(duration=args.duration)


if __name__ == '__main__':
    main()

