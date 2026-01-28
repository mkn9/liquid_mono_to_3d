#!/bin/bash
# Worker 3: Integration & Evaluation - Daily Tasks
# Integrates Worker 1 & 2, runs end-to-end evaluation

set -e

PROJECT_DIR="$HOME/liquid_mono_to_3d"
cd "$PROJECT_DIR"

echo "🟡 Worker 3: Integration & Evaluation"
echo "======================================"
echo ""

CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "worker/liquid-worker-3-integration" ]; then
    git checkout worker/liquid-worker-3-integration
fi

cat << 'EOF'
Select task:

SETUP & SYNC
  1) Check Worker 1 & 2 status
  2) Merge Worker 1 (Fusion)
  3) Merge Worker 2 (3D Recon)
  4) Resolve conflicts (if any)

INTEGRATION TESTING
  5) Write end-to-end integration tests
  6) Run integration test suite
  7) Create demo script (liquid vs static)

EVALUATION (Week 2)
  8) Run evaluation: Fusion comparison
  9) Run evaluation: 3D reconstruction comparison
 10) Run evaluation: Combined system
 11) Generate final comparison report
 12) Measure all metrics

ANALYSIS
 13) Analyze hallucination rates
 14) Analyze temporal consistency
 15) Analyze 3D smoothness
 16) Create visualizations

MERGE TO MAIN
 17) Prepare for main merge
 18) Run full test suite
 19) Create proof bundle
 20) Merge to main

STATUS
 21) Update worker status
 22) View current status
 23) Sync to MacBook

 0) Exit

Enter choice:
EOF

read -p "> " choice

case $choice in
    1)
        echo "📊 Checking Worker 1 & 2 status..."
        echo ""
        echo "=== Worker 1 (Fusion) ==="
        git show worker/liquid-worker-1-fusion:status/worker1_status.md 2>/dev/null || echo "Not available yet"
        echo ""
        echo "=== Worker 2 (3D Recon) ==="
        git show worker/liquid-worker-2-3d:status/worker2_status.md 2>/dev/null || echo "Not available yet"
        ;;
        
    2)
        echo "🔄 Merging Worker 1 (Fusion)..."
        git fetch origin
        git merge worker/liquid-worker-1-fusion --no-ff -m "🔗 Worker 3: Merge Worker 1 (Fusion)" || {
            echo "⚠️  Conflicts detected. Resolve manually, then:"
            echo "  git add ."
            echo "  git commit"
            exit 1
        }
        echo "✅ Worker 1 merged"
        ;;
        
    3)
        echo "🔄 Merging Worker 2 (3D)..."
        git fetch origin
        git merge worker/liquid-worker-2-3d --no-ff -m "🔗 Worker 3: Merge Worker 2 (3D)" || {
            echo "⚠️  Conflicts detected. Resolve manually"
            exit 1
        }
        echo "✅ Worker 2 merged"
        ;;
        
    4)
        echo "🔧 Checking for conflicts..."
        git status
        echo ""
        echo "If conflicts exist:"
        echo "  1. Edit conflicted files"
        echo "  2. git add <files>"
        echo "  3. git commit"
        ;;
        
    5)
        echo "✍️  Writing integration tests..."
        
        mkdir -p tests
        cat > tests/test_integration_liquid.py << 'PYTEST'
"""
End-to-End Integration Tests for Liquid System
Tests that Workers 1 & 2 work together
"""
import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments', 'trajectory_video_understanding'))

def test_can_import_all_components():
    """All Liquid components can be imported together."""
    from vision_language_integration.dual_visual_adapter import LiquidDualModalFusion
    from vision_language_integration.liquid_3d_reconstructor import Liquid3DTrajectoryReconstructor
    from liquid_models.liquid_cell import LiquidCell
    assert True

def test_fusion_and_3d_together():
    """Fusion and 3D reconstruction work together."""
    from vision_language_integration.dual_visual_adapter import LiquidDualModalFusion
    from vision_language_integration.liquid_3d_reconstructor import Liquid3DTrajectoryReconstructor
    
    # Create components
    recon_3d = Liquid3DTrajectoryReconstructor()
    fusion = LiquidDualModalFusion()
    
    # Simulate pipeline
    noisy_3d = torch.randn(2, 32, 3)  # Noisy 3D from triangulation
    feat_2d = torch.randn(2, 512)     # 2D features from MagVIT
    
    # 3D reconstruction
    feat_3d, smooth_traj = recon_3d(noisy_3d)  # (2, 256), (2, 32, 3)
    
    # Fusion
    fused = fusion(feat_2d, feat_3d)  # (2, 4096)
    
    assert fused.shape == (2, 4096)
    assert smooth_traj.shape == (2, 32, 3)

def test_end_to_end_gradients():
    """Gradients flow through entire Liquid pipeline."""
    from vision_language_integration.dual_visual_adapter import LiquidDualModalFusion
    from vision_language_integration.liquid_3d_reconstructor import Liquid3DTrajectoryReconstructor
    
    recon_3d = Liquid3DTrajectoryReconstructor()
    fusion = LiquidDualModalFusion()
    
    noisy_3d = torch.randn(1, 16, 3, requires_grad=True)
    feat_2d = torch.randn(1, 512, requires_grad=True)
    
    feat_3d, _ = recon_3d(noisy_3d)
    fused = fusion(feat_2d, feat_3d)
    
    loss = fused.sum()
    loss.backward()
    
    assert noisy_3d.grad is not None
    assert feat_2d.grad is not None

def test_liquid_vs_static_shapes_match():
    """Liquid and static outputs have same shapes."""
    # This ensures drop-in compatibility
    from vision_language_integration.dual_visual_adapter import LiquidDualModalFusion
    
    fusion = LiquidDualModalFusion()
    
    feat_2d = torch.randn(4, 512)
    feat_3d = torch.randn(4, 256)
    
    liquid_out = fusion(feat_2d, feat_3d)
    
    # Static would be: cat + linear
    # Liquid output should match static shape
    assert liquid_out.shape == (4, 4096)

def test_batch_independence():
    """Different samples in batch don't affect each other."""
    from vision_language_integration.dual_visual_adapter import LiquidDualModalFusion
    
    fusion = LiquidDualModalFusion()
    
    feat_2d = torch.randn(4, 512)
    feat_3d = torch.randn(4, 256)
    
    # Process as batch
    batch_out = fusion(feat_2d, feat_3d, reset_state=True)
    
    # Process individually
    individual_outs = []
    for i in range(4):
        out = fusion(feat_2d[i:i+1], feat_3d[i:i+1], reset_state=True)
        individual_outs.append(out)
    individual_outs = torch.cat(individual_outs, dim=0)
    
    # Should match (batch independence)
    assert torch.allclose(batch_out, individual_outs, rtol=1e-3)
PYTEST

        echo "✅ Integration tests created"
        ;;
        
    6)
        echo "🧪 Running integration tests..."
        pytest tests/test_integration_liquid.py -v
        ;;
        
    7)
        echo "📝 Creating demo script..."
        
        cat > experiments/trajectory_video_understanding/vision_language_integration/compare_liquid_vs_static.py << 'PYTHON'
"""
Compare Liquid vs Static Fusion
Evaluation script for Worker 3
"""
import torch
import json
from datetime import datetime
from pathlib import Path

def compare_fusion_methods(num_samples=20):
    """Compare static vs liquid fusion on real data."""
    
    print("🔬 Comparing Liquid vs Static Fusion")
    print(f"Samples: {num_samples}")
    print("")
    
    results = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M"),
        "num_samples": num_samples,
        "static": {},
        "liquid": {},
        "comparison": {}
    }
    
    # TODO: Load real data
    # TODO: Run both methods
    # TODO: Compare outputs
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = Path(f"results/worker3/{timestamp}_fusion_comparison.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved: {output_path}")
    return results

if __name__ == "__main__":
    compare_fusion_methods()
PYTHON

        echo "✅ Demo script created"
        ;;
        
    8)
        echo "📊 Running fusion evaluation..."
        cd experiments/trajectory_video_understanding/vision_language_integration
        python compare_liquid_vs_static.py
        ;;
        
    11)
        echo "📊 Generating final comparison report..."
        
        TIMESTAMP=$(date +"%Y%m%d_%H%M")
        REPORT="results/worker3/${TIMESTAMP}_final_report.md"
        
        cat > "$REPORT" << EOF
# Liquid NN Integration - Final Evaluation Report

**Date:** $(date)
**Branch:** worker/liquid-worker-3-integration

## Summary

This report compares the Liquid Neural Network enhanced system against the baseline static system.

## Components Evaluated

1. **Liquid Fusion Layer** (Worker 1)
   - Replaces static concatenation+linear with Liquid dynamics
   - Adds temporal consistency

2. **Liquid 3D Reconstruction** (Worker 2)
   - Replaces frame-by-frame triangulation
   - Adds temporal smoothing

3. **Combined System** (Worker 3)
   - Both components integrated
   - End-to-end evaluation

## Metrics

### Description Quality
- Baseline: 8.0/10
- Liquid Fusion Only: TBD
- Liquid 3D Only: TBD
- Combined: TBD

### Hallucination Rate
- Baseline: 20%
- Liquid Fusion Only: TBD
- Liquid 3D Only: TBD
- Combined: TBD

### 3D Trajectory Smoothness
- Baseline: 7.0/10
- Liquid 3D: TBD

## Visualizations

(See results/worker3/visualizations/)

## Conclusion

TBD after evaluation

## Next Steps

- [ ] Merge to main if successful
- [ ] Document performance improvements
- [ ] Create proof bundle

---
Generated: $(date)
EOF

        echo "✅ Report created: $REPORT"
        cat "$REPORT"
        ;;
        
    17)
        echo "🎯 Preparing for main merge..."
        echo ""
        echo "Pre-merge checklist:"
        echo "  [ ] Worker 1 complete"
        echo "  [ ] Worker 2 complete"
        echo "  [ ] Both workers merged to Worker 3"
        echo "  [ ] All integration tests passing"
        echo "  [ ] Evaluation complete"
        echo "  [ ] Metrics show improvement"
        echo "  [ ] Documentation updated"
        echo ""
        read -p "Ready to proceed? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Proceeding..."
        else
            echo "Cancelled"
            exit 0
        fi
        ;;
        
    18)
        echo "🧪 Running full test suite..."
        cd "$PROJECT_DIR"
        pytest tests/ -v --tb=short
        ;;
        
    19)
        echo "📦 Creating proof bundle..."
        cd "$PROJECT_DIR"
        bash scripts/prove.sh
        ;;
        
    20)
        echo "🔀 Merging to main..."
        echo ""
        echo "Steps:"
        echo "  1. git checkout liquid-nn-integration"
        echo "  2. git merge --no-ff worker/liquid-worker-3-integration"
        echo "  3. git checkout main"
        echo "  4. git merge --no-ff liquid-nn-integration"
        echo "  5. git tag -a v1.0-liquid-nn -m 'Liquid NN Integration Complete'"
        echo "  6. git push --all && git push --tags"
        echo ""
        read -p "Execute these steps? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            git checkout liquid-nn-integration
            git merge --no-ff worker/liquid-worker-3-integration -m "🎉 Final integration complete"
            git checkout main
            git merge --no-ff liquid-nn-integration -m "🚀 Liquid NN Integration v1.0"
            git tag -a v1.0-liquid-nn -m "Liquid Neural Network Integration Complete"
            git push --all
            git push --tags
            echo "✅ Merged to main and tagged!"
        fi
        ;;
        
    21)
        echo "📝 Updating status..."
        vim status/worker3_status.md
        git add status/worker3_status.md
        git commit -m "📊 Worker 3: Status update"
        git push
        ;;
        
    22)
        cat status/worker3_status.md
        ;;
        
    23)
        bash scripts/sync_to_macbook.sh
        ;;
        
    0)
        exit 0
        ;;
esac

