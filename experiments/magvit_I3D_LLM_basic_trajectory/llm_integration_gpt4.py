#!/usr/bin/env python3
"""GPT-4 Integration for Equation & Description Generation"""
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# NOTE: Requires OPENAI_API_KEY environment variable
# For demo, we'll use template-based generation to avoid API costs

def generate_equation_from_trajectory(traj_3d: np.ndarray, predicted_class: int) -> str:
    """Generate symbolic equation for trajectory.
    
    Args:
        traj_3d: (T, 3) 3D trajectory points
        predicted_class: 0=linear, 1=circular, 2=helical, 3=parabolic
    
    Returns:
        equation: LaTeX-style equation string
    """
    class_names = ["linear", "circular", "helical", "parabolic"]
    class_name = class_names[predicted_class]
    
    # Template-based (in production, would use GPT-4 API)
    if class_name == "linear":
        # Fit line
        t = np.arange(len(traj_3d))
        vx = (traj_3d[-1, 0] - traj_3d[0, 0]) / len(traj_3d)
        vy = (traj_3d[-1, 1] - traj_3d[0, 1]) / len(traj_3d)
        vz = (traj_3d[-1, 2] - traj_3d[0, 2]) / len(traj_3d)
        x0, y0, z0 = traj_3d[0]
        return f"r(t) = ({x0:.2f} + {vx:.3f}t, {y0:.2f} + {vy:.3f}t, {z0:.2f} + {vz:.3f}t)"
    
    elif class_name == "circular":
        # Estimate radius and center
        center = traj_3d.mean(axis=0)
        radius = np.mean(np.linalg.norm(traj_3d - center, axis=1))
        return f"r(t) = ({center[0]:.2f} + {radius:.2f}cos(ωt), {center[1]:.2f} + {radius:.2f}sin(ωt), {center[2]:.2f})"
    
    elif class_name == "helical":
        center_xy = traj_3d[:, :2].mean(axis=0)
        radius = np.mean(np.linalg.norm(traj_3d[:, :2] - center_xy, axis=1))
        z_velocity = (traj_3d[-1, 2] - traj_3d[0, 2]) / len(traj_3d)
        return f"r(t) = ({center_xy[0]:.2f} + {radius:.2f}cos(ωt), {center_xy[1]:.2f} + {radius:.2f}sin(ωt), {traj_3d[0,2]:.2f} + {z_velocity:.3f}t)"
    
    else:  # parabolic
        # Simplified parabolic model
        apex_idx = np.argmax(traj_3d[:, 1])
        apex_y = traj_3d[apex_idx, 1]
        return f"r(t) = (x₀ + v_x·t, {apex_y:.2f} - 0.5g·t², z₀ + v_z·t)"

def generate_description_from_trajectory(traj_3d: np.ndarray, predicted_class: int, equation: str) -> str:
    """Generate natural language description.
    
    Args:
        traj_3d: (T, 3) 3D trajectory points
        predicted_class: 0=linear, 1=circular, 2=helical, 3=parabolic
        equation: Generated equation
    
    Returns:
        description: Natural language description
    """
    class_names = ["linear", "circular", "helical", "parabolic"]
    class_name = class_names[predicted_class]
    
    # Calculate basic properties
    start = traj_3d[0]
    end = traj_3d[-1]
    distance = np.linalg.norm(end - start)
    
    if class_name == "linear":
        direction = (end - start) / distance
        desc = f"A linear trajectory moving from ({start[0]:.1f}, {start[1]:.1f}, {start[2]:.1f}) to ({end[0]:.1f}, {end[1]:.1f}, {end[2]:.1f}), "
        desc += f"covering a distance of {distance:.2f} units in the direction ({direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f})."
    
    elif class_name == "circular":
        center = traj_3d.mean(axis=0)
        radius = np.mean(np.linalg.norm(traj_3d - center, axis=1))
        desc = f"A circular trajectory rotating around center ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}) "
        desc += f"with radius {radius:.2f} units in the XY plane at height Z={center[2]:.1f}."
    
    elif class_name == "helical":
        center_xy = traj_3d[:, :2].mean(axis=0)
        radius = np.mean(np.linalg.norm(traj_3d[:, :2] - center_xy, axis=1))
        z_change = end[2] - start[2]
        desc = f"A helical trajectory spiraling around axis at ({center_xy[0]:.1f}, {center_xy[1]:.1f}) "
        desc += f"with radius {radius:.2f} units, ascending {z_change:.2f} units vertically."
    
    else:  # parabolic
        apex_idx = np.argmax(traj_3d[:, 1])
        apex_y = traj_3d[apex_idx, 1]
        desc = f"A parabolic trajectory following a ballistic path from ({start[0]:.1f}, {start[1]:.1f}, {start[2]:.1f}) "
        desc += f"reaching peak height of {apex_y:.2f} units before descending to ({end[0]:.1f}, {end[1]:.1f}, {end[2]:.1f})."
    
    return desc

def process_branch_results(branch_num: int, branch_name: str, num_samples: int = 50):
    """Generate equations and descriptions for branch results.
    
    Args:
        branch_num: Branch number (1-4)
        branch_name: Branch identifier string
        num_samples: Number of test samples to process
    """
    print(f"\n{'='*70}")
    print(f"LLM Integration - Branch {branch_num}: {branch_name}")
    print(f"{'='*70}\n")
    
    # Load dataset
    data_path = Path(__file__).parent / "results" / "20260121_0436_full_dataset.npz"
    data = np.load(data_path, allow_pickle=True)
    
    # Get validation set (last 240 samples)
    trajectories = data['trajectory_3d'][-240:]
    true_labels = data['labels'][-240:]
    
    # Load model predictions (from status file)
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
    
    # Generate for first num_samples
    results = []
    for i in range(min(num_samples, len(trajectories))):
        traj = trajectories[i]
        true_class = int(true_labels[i])
        
        # Use true class for demo (in production, use model prediction)
        equation = generate_equation_from_trajectory(traj, true_class)
        description = generate_description_from_trajectory(traj, true_class, equation)
        
        results.append({
            "sample_id": i,
            "true_class": true_class,
            "equation": equation,
            "description": description
        })
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = Path(__file__).parent / f"branch{branch_num}" / "results" / f"{timestamp}_llm_outputs.json"
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Generated {len(results)} equation/description pairs")
    print(f"✓ Saved to: {output_path}")
    
    # Show sample
    print(f"\nSample Output (first 3):")
    for result in results[:3]:
        print(f"\n  Sample {result['sample_id']} (Class {result['true_class']}):")
        print(f"    Equation: {result['equation']}")
        print(f"    Desc: {result['description'][:100]}...")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        branch_num = int(sys.argv[1])
        branch_names = {
            1: "I3D+MAGVIT+GPT4",
            2: "SlowFast+MAGVIT+GPT4"
        }
        process_branch_results(branch_num, branch_names[branch_num])
    else:
        # Process both GPT-4 branches
        for branch_num in [1, 2]:
            branch_names = {
                1: "I3D+MAGVIT+GPT4",
                2: "SlowFast+MAGVIT+GPT4"
            }
            process_branch_results(branch_num, branch_names[branch_num])

