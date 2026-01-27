#!/usr/bin/env python3
"""Train all 4 branches sequentially (memory-efficient)"""
import sys
import subprocess
from pathlib import Path

def run_branch(branch_num):
    """Run a single branch training."""
    print(f"\n{'='*60}")
    print(f"Starting Branch {branch_num} Training")
    print(f"{'='*60}\n")
    
    script = Path(__file__).parent / f"branch{branch_num}" / "train.py"
    log_file = Path(__file__).parent / f"branch{branch_num}" / "results" / "training.log"
    
    # Run training
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True
    )
    
    # Save log
    log_file.parent.mkdir(exist_ok=True)
    with open(log_file, 'w') as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n\n=== STDERR ===\n")
            f.write(result.stderr)
    
    print(result.stdout)
    
    if result.returncode == 0:
        print(f"✓ Branch {branch_num} completed successfully!")
    else:
        print(f"✗ Branch {branch_num} failed with code {result.returncode}")
        print(result.stderr)
    
    return result.returncode == 0

if __name__ == "__main__":
    print("Training all 4 branches sequentially...")
    
    results = {}
    for branch in [1, 2, 3]:  # 4 already done
        success = run_branch(branch)
        results[f"branch{branch}"] = "SUCCESS" if success else "FAILED"
    
    print("\n" + "="*60)
    print("Training Summary:")
    print("="*60)
    for branch, status in results.items():
        print(f"{branch}: {status}")

