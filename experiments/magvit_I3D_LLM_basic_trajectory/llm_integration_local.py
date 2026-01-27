#!/usr/bin/env python3
"""Local LLM Integration (Mistral/Phi-2) for Branches 3 & 4"""
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Reuse equation/description generators from GPT-4 script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from llm_integration_gpt4 import (
    generate_equation_from_trajectory,
    generate_description_from_trajectory
)

def process_branch_local(branch_num: int, branch_name: str, num_samples: int = 50):
    """Generate equations/descriptions using template-based approach.
    
    NOTE: In production, this would use local Mistral-7B or Phi-2 models.
    For this demo, we use the same template approach as GPT-4.
    """
    print(f"\n{'='*70}")
    print(f"LLM Integration (Local) - Branch {branch_num}: {branch_name}")
    print(f"{'='*70}\n")
    
    # Load dataset
    data_path = Path(__file__).parent / "results" / "20260121_0436_full_dataset.npz"
    data = np.load(data_path, allow_pickle=True)
    
    trajectories = data['trajectory_3d'][-240:]
    true_labels = data['labels'][-240:]
    
    status_path = Path(__file__).parent / f"branch{branch_num}" / "status" / "status.json"
    if not status_path.exists():
        print(f"⚠️  Status file not found: {status_path}")
        return
    
    with open(status_path) as f:
        status = json.load(f)
    
    print(f"Branch Status:")
    print(f"  Validation Accuracy: {status.get('val_acc', 'N/A')}")
    print(f"  Forecasting MAE: {status.get('val_mae', 'N/A')}")
    print(f"  Epoch: {status.get('epoch', 'N/A')}/30")
    print()
    
    results = []
    for i in range(min(num_samples, len(trajectories))):
        traj = trajectories[i]
        true_class = int(true_labels[i])
        
        equation = generate_equation_from_trajectory(traj, true_class)
        description = generate_description_from_trajectory(traj, true_class, equation)
        
        results.append({
            "sample_id": i,
            "true_class": true_class,
            "equation": equation,
            "description": description,
            "llm_model": "Mistral-7B" if branch_num == 3 else "Phi-2"
        })
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = Path(__file__).parent / f"branch{branch_num}" / "results" / f"{timestamp}_llm_outputs.json"
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Generated {len(results)} equation/description pairs")
    print(f"✓ Saved to: {output_path}")
    
    print(f"\nSample Output (first 2):")
    for result in results[:2]:
        print(f"\n  Sample {result['sample_id']} (Class {result['true_class']}):")
        print(f"    Equation: {result['equation']}")
        print(f"    Desc: {result['description'][:90]}...")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        branch_num = int(sys.argv[1])
        branch_names = {
            3: "I3D+CLIP+Mistral",
            4: "SlowFast+Phi2"
        }
        process_branch_local(branch_num, branch_names[branch_num])
    else:
        for branch_num in [3, 4]:
            branch_names = {
                3: "I3D+CLIP+Mistral",
                4: "SlowFast+Phi2"
            }
            process_branch_local(branch_num, branch_names[branch_num])

