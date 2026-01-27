"""
Parallel Worker Monitoring Script
Monitors both workers, checks for early stopping, syncs results to MacBook
"""

import time
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import sys

# Configuration
BASE_DIR = Path.home() / "mono_to_3d"
RESULTS_DIR = BASE_DIR / "experiments/trajectory_video_understanding/parallel_workers"
SYNC_INTERVAL = 120  # Sync every 2 minutes
CHECK_INTERVAL = 30  # Check progress every 30 seconds

WORKERS = {
    'worker1': {
        'name': 'Attention-Supervised',
        'dir': Path.home() / 'worker1_attention',
        'results': RESULTS_DIR / 'worker1_attention/results',
        'checkpoint': 'latest_metrics.json',
        'priority': 1
    },
    'worker2': {
        'name': 'Pre-trained Features',
        'dir': Path.home() / 'worker2_pretrained',
        'results': RESULTS_DIR / 'worker2_pretrained/results',
        'checkpoint': 'latest_metrics.json',
        'priority': 2
    }
}

# Success criteria
SUCCESS_CRITERIA = {
    'attention_ratio': 1.5,
    'val_accuracy': 0.75,
    'consistency': 0.70
}

# MacBook sync configuration (update with your details)
MACBOOK_USER = "mike"
MACBOOK_HOST = "192.168.1.100"  # Update with actual IP
MACBOOK_PATH = "/Users/mike/Dropbox/Documents/Machine_Learning/LLM_and_Multimodal_Models/repos/mono_to_3d/experiments/trajectory_video_understanding/parallel_workers"


class WorkerMonitor:
    """Monitor parallel workers and handle early stopping."""
    
    def __init__(self):
        self.workers = WORKERS
        self.last_sync = time.time()
        self.winner = None
        self.start_time = time.time()
        
        # Create results directories
        for worker_info in self.workers.values():
            worker_info['results'].mkdir(parents=True, exist_ok=True)
    
    def read_worker_metrics(self, worker_id: str) -> Optional[Dict]:
        """Read latest metrics from worker checkpoint."""
        worker_info = self.workers[worker_id]
        metrics_file = worker_info['results'] / worker_info['checkpoint']
        
        if not metrics_file.exists():
            return None
        
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            return metrics
        except (json.JSONDecodeError, IOError):
            return None
    
    def check_success_criteria(self, metrics: Dict) -> bool:
        """Check if metrics meet success criteria."""
        if metrics is None:
            return False
        
        ratio_met = metrics.get('attention_ratio', 0) >= SUCCESS_CRITERIA['attention_ratio']
        accuracy_met = metrics.get('val_accuracy', 0) >= SUCCESS_CRITERIA['val_accuracy']
        consistency_met = metrics.get('consistency', 0) >= SUCCESS_CRITERIA['consistency']
        
        return ratio_met and accuracy_met and consistency_met
    
    def terminate_worker(self, worker_id: str):
        """Terminate a running worker."""
        worker_info = self.workers[worker_id]
        worker_dir = worker_info['dir']
        
        print(f"🛑 Terminating {worker_info['name']}...")
        
        # Find and kill the training process
        try:
            # Look for training script process
            result = subprocess.run(
                ['pgrep', '-f', f'train.*{worker_dir}'],
                capture_output=True,
                text=True
            )
            if result.stdout:
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    subprocess.run(['kill', pid])
                print(f"   Killed processes: {', '.join(pids)}")
        except Exception as e:
            print(f"   Warning: Could not terminate {worker_id}: {e}")
    
    def sync_to_macbook(self, force: bool = False):
        """Sync results to MacBook."""
        current_time = time.time()
        
        if not force and (current_time - self.last_sync) < SYNC_INTERVAL:
            return  # Not time yet
        
        print(f"📤 Syncing results to MacBook...")
        
        try:
            # Sync entire parallel_workers directory
            subprocess.run([
                'rsync', '-avz', '--progress',
                str(RESULTS_DIR) + '/',
                f'{MACBOOK_USER}@{MACBOOK_HOST}:{MACBOOK_PATH}/'
            ], timeout=60)
            print("   ✅ Sync complete")
            self.last_sync = current_time
        except Exception as e:
            print(f"   ⚠️  Sync failed: {e}")
    
    def generate_progress_report(self) -> str:
        """Generate markdown progress report."""
        report = [
            "# Parallel Training Progress",
            "",
            f"**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Elapsed Time**: {(time.time() - self.start_time) / 3600:.1f} hours",
            "",
            "## Worker Status",
            "",
            "| Worker | Status | Epoch | Ratio | Val Acc | Consistency |",
            "|--------|--------|-------|-------|---------|-------------|"
        ]
        
        for worker_id, worker_info in self.workers.items():
            metrics = self.read_worker_metrics(worker_id)
            
            if metrics is None:
                status = "🔵 Starting"
                epoch = "0/??"
                ratio = "-"
                acc = "-"
                consistency = "-"
            else:
                epoch_str = f"{metrics.get('epoch', 0)}/{metrics.get('max_epochs', 50)}"
                ratio_val = metrics.get('attention_ratio', 0)
                ratio = f"{ratio_val:.2f}x"
                acc = f"{metrics.get('val_accuracy', 0):.1%}"
                consistency = f"{metrics.get('consistency', 0):.1%}"
                
                # Status emoji
                if self.check_success_criteria(metrics):
                    status = "🎉 SUCCESS"
                elif ratio_val >= SUCCESS_CRITERIA['attention_ratio'] * 0.9:
                    status = "🟢 Close!"
                else:
                    status = "🟡 Training"
            
            report.append(
                f"| {worker_info['name']} | {status} | {epoch} | {ratio} | {acc} | {consistency} |"
            )
        
        report.extend([
            "",
            "## Success Criteria",
            f"- Attention Ratio: ≥ {SUCCESS_CRITERIA['attention_ratio']}x",
            f"- Validation Accuracy: ≥ {SUCCESS_CRITERIA['val_accuracy']:.0%}",
            f"- Consistency: ≥ {SUCCESS_CRITERIA['consistency']:.0%}",
            ""
        ])
        
        if self.winner:
            report.extend([
                "## 🏆 Winner",
                f"**{self.workers[self.winner]['name']}** achieved success criteria!",
                ""
            ])
        
        return "\n".join(report)
    
    def save_progress_report(self):
        """Save progress report to file."""
        report = self.generate_progress_report()
        report_file = RESULTS_DIR / 'monitoring' / 'PARALLEL_PROGRESS.md'
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            f.write(report)
    
    def monitor_loop(self):
        """Main monitoring loop."""
        print("================================================================")
        print("PARALLEL WORKER MONITORING")
        print("================================================================")
        print(f"Workers: {len(self.workers)}")
        for worker_id, worker_info in self.workers.items():
            print(f"  - {worker_info['name']} ({worker_id})")
        print(f"\nChecking every {CHECK_INTERVAL}s, syncing every {SYNC_INTERVAL}s")
        print("================================================================\n")
        
        iteration = 0
        
        while True:
            iteration += 1
            print(f"\n[Check {iteration}] {datetime.now().strftime('%H:%M:%S')}")
            
            # Check each worker
            for worker_id, worker_info in sorted(self.workers.items(), key=lambda x: x[1]['priority']):
                metrics = self.read_worker_metrics(worker_id)
                
                if metrics is None:
                    print(f"  {worker_info['name']}: No metrics yet")
                    continue
                
                # Display current status
                epoch = metrics.get('epoch', 0)
                ratio = metrics.get('attention_ratio', 0)
                acc = metrics.get('val_accuracy', 0)
                consistency = metrics.get('consistency', 0)
                
                print(f"  {worker_info['name']}:")
                print(f"    Epoch: {epoch}")
                print(f"    Ratio: {ratio:.2f}x {'✅' if ratio >= SUCCESS_CRITERIA['attention_ratio'] else '⏳'}")
                print(f"    Val Acc: {acc:.1%} {'✅' if acc >= SUCCESS_CRITERIA['val_accuracy'] else '⏳'}")
                print(f"    Consistency: {consistency:.1%} {'✅' if consistency >= SUCCESS_CRITERIA['consistency'] else '⏳'}")
                
                # Check for success
                if self.check_success_criteria(metrics):
                    print(f"\n{'='*60}")
                    print(f"🎉 SUCCESS! {worker_info['name']} achieved all criteria!")
                    print(f"{'='*60}\n")
                    
                    self.winner = worker_id
                    
                    # Terminate other workers
                    for other_id in self.workers.keys():
                        if other_id != worker_id:
                            self.terminate_worker(other_id)
                    
                    # Final sync
                    self.save_progress_report()
                    self.sync_to_macbook(force=True)
                    
                    print(f"\n✅ Monitoring complete!")
                    print(f"Winner: {worker_info['name']}")
                    print(f"Results synced to MacBook")
                    print(f"\nNext step: Generate success visualizations")
                    print(f"  python scripts/generate_success_visualizations.py --worker {worker_id}")
                    
                    return worker_id
            
            # Save and sync progress
            self.save_progress_report()
            self.sync_to_macbook()
            
            # Wait before next check
            time.sleep(CHECK_INTERVAL)


def main():
    """Main entry point."""
    monitor = WorkerMonitor()
    
    try:
        winner = monitor.monitor_loop()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring interrupted by user")
        monitor.save_progress_report()
        monitor.sync_to_macbook(force=True)
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during monitoring: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

