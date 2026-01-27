"""
MacBook Result Syncer
=====================

Pushes training results to MacBook periodically during training.
Ensures visibility and compliance with incremental save requirements.
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional


class MacBookSyncer:
    """Syncs training results to MacBook in real-time."""
    
    def __init__(
        self,
        worker_name: str,
        results_dir: str = "results/validation",
        macbook_path: str = "",
        ssh_key: str = "/home/ubuntu/.ssh/macbook_key.pem",
        enabled: bool = True
    ):
        """
        Initialize MacBook syncer.
        
        Args:
            worker_name: Name of the worker (e.g., 'i3d', 'slowfast')
            results_dir: Local directory to write results to
            macbook_path: Path on MacBook to sync to (optional, for future push support)
            ssh_key: Path to SSH key for MacBook (optional)
            enabled: Whether syncing is enabled
        """
        self.worker_name = worker_name
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.macbook_path = macbook_path
        self.ssh_key = ssh_key
        self.enabled = enabled
        
    def push_heartbeat(self, epoch: int, step: int, loss: float, timestamp: Optional[str] = None) -> bool:
        """
        Push lightweight heartbeat update.
        
        Args:
            epoch: Current epoch
            step: Current step within epoch
            loss: Current loss value
            timestamp: Optional timestamp (defaults to now)
            
        Returns:
            True if push succeeded, False otherwise
        """
        if not self.enabled:
            return False
            
        heartbeat = {
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
            "worker": self.worker_name,
            "epoch": epoch,
            "step": step,
            "loss": float(loss),
            "status": "training"
        }
        
        # Save locally
        heartbeat_file = Path('heartbeat.json')
        with open(heartbeat_file, 'w') as f:
            json.dump(heartbeat, f, indent=2)
        
        # Push to MacBook
        return self._scp_file(heartbeat_file, 'heartbeat.json')
    
    def push_checkpoint(
        self,
        epoch: int,
        checkpoint_path: Path,
        metrics: Dict[str, Any]
    ) -> bool:
        """
        Push checkpoint and metrics after epoch.
        
        Args:
            epoch: Epoch number
            checkpoint_path: Path to checkpoint file
            metrics: Training metrics dictionary
            
        Returns:
            True if push succeeded, False otherwise
        """
        if not self.enabled:
            return False
        
        # Save metrics locally
        metrics_file = Path(f'metrics_epoch_{epoch}.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Push both files
        success = True
        success &= self._scp_file(metrics_file, f'metrics_epoch_{epoch}.json')
        success &= self._scp_file(checkpoint_path, checkpoint_path.name)
        
        return success
    
    def push_log_tail(self, log_file: Path, num_lines: int = 50) -> bool:
        """
        Push last N lines of log file.
        
        Args:
            log_file: Path to log file
            num_lines: Number of lines to tail
            
        Returns:
            True if push succeeded, False otherwise
        """
        if not self.enabled:
            return False
        
        if not log_file.exists():
            return False
        
        # Read last N lines
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                tail_content = ''.join(lines[-num_lines:])
            
            # Write to temporary file
            tail_file = Path('latest_log.txt')
            with open(tail_file, 'w') as f:
                f.write(tail_content)
            
            # Push to MacBook
            return self._scp_file(tail_file, 'latest_log.txt')
        except Exception:
            return False
    
    def push_completion_status(self, success: bool, final_metrics: Dict[str, Any]) -> bool:
        """
        Push training completion status.
        
        Args:
            success: Whether training completed successfully
            final_metrics: Final metrics dictionary
            
        Returns:
            True if push succeeded, False otherwise
        """
        if not self.enabled:
            return False
        
        status = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "worker": self.worker_name,
            "status": "completed" if success else "failed",
            "final_metrics": final_metrics
        }
        
        status_file = Path('training_status.json')
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
        
        return self._scp_file(status_file, 'training_status.json')
    
    def _scp_file(self, local_file: Path, remote_name: str) -> bool:
        """
        SCP file to MacBook.
        
        Args:
            local_file: Local file to copy
            remote_name: Name to give file on remote
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Construct full remote path
            remote_path = f"{self.macbook_path}/{self.worker_name}/{remote_name}"
            
            # Run SCP
            result = subprocess.run(
                [
                    'scp',
                    '-i', self.ssh_key,
                    '-o', 'StrictHostKeyChecking=no',
                    '-o', 'ConnectTimeout=5',
                    str(local_file),
                    remote_path
                ],
                capture_output=True,
                timeout=10
            )
            
            return result.returncode == 0
        except Exception:
            return False

