#!/bin/bash
# Worker 2: Liquid 3D Reconstruction - Daily Tasks
# Implements temporally-smooth 3D trajectory reconstruction

set -e

PROJECT_DIR="$HOME/liquid_mono_to_3d"
cd "$PROJECT_DIR"

echo "🟢 Worker 2: Liquid 3D Reconstruction"
echo "======================================"
echo ""

# Check branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "worker/liquid-worker-2-3d" ]; then
    echo "⚠️  Wrong branch! Switching..."
    git checkout worker/liquid-worker-2-3d
fi

cat << 'EOF'
Select task:

DAY 1-2: Setup & TDD RED Phase
  1) Copy liquid_cell.py (from Worker 1 or liquid_ai_2)
  2) Write 3D reconstruction tests (RED)
  3) Capture RED evidence
  4) Update status & commit

DAY 3-4: Implementation (TDD GREEN Phase)
  5) Implement Liquid3DTrajectoryReconstructor
  6) Run tests (expect GREEN)
  7) Capture GREEN evidence
  8) Update status & commit

EVALUATION
  9) Compare static vs liquid 3D on 20 samples
 10) Measure smoothness, noise, occlusions
 11) Generate comparison report

STATUS & UTILITIES
 12) Update worker status
 13) View current status
 14) Sync to MacBook
 15) Run TDD capture
 16) Create proof bundle

 0) Exit

Enter choice:
EOF

read -p "> " choice

case $choice in
    1)
        echo "📦 Copying liquid_cell.py..."
        mkdir -p experiments/trajectory_video_understanding/liquid_models
        
        # Check if Worker 1 has it
        if [ -f "../worker1-branch/liquid_models/liquid_cell.py" ]; then
            cp ../worker1-branch/liquid_models/liquid_cell.py \
               experiments/trajectory_video_understanding/liquid_models/
            echo "✅ Copied from Worker 1"
        else
            echo "Copy from liquid_ai_2 (run on MacBook):"
            echo "rsync -avz -e \"ssh -i /Users/mike/keys/AutoGenKeyPair.pem\" \\"
            echo "  ~/Dropbox/Code/repos/liquid_ai_2/option1_synthetic/liquid_cell.py \\"
            echo "  ubuntu@204.236.245.232:~/liquid_mono_to_3d/experiments/trajectory_video_understanding/liquid_models/"
        fi
        ;;
        
    2)
        echo "✍️  Writing 3D reconstruction tests (RED)..."
        
        mkdir -p tests
        cat > tests/test_liquid_3d_reconstruction.py << 'PYTEST'
"""
Tests for Liquid 3D Trajectory Reconstruction
TDD RED Phase - Tests written FIRST
"""
import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments', 'trajectory_video_understanding'))

def test_liquid_3d_can_import():
    """Test import of Liquid3DTrajectoryReconstructor."""
    try:
        from vision_language_integration.liquid_3d_reconstructor import Liquid3DTrajectoryReconstructor
        assert True
    except ImportError as e:
        pytest.fail(f"Cannot import: {e}")

def test_liquid_3d_initialization():
    """Liquid 3D reconstructor initializes correctly."""
    from vision_language_integration.liquid_3d_reconstructor import Liquid3DTrajectoryReconstructor
    
    recon = Liquid3DTrajectoryReconstructor()
    
    assert hasattr(recon, 'liquid_dynamics')
    assert hasattr(recon, 'position_predictor')
    assert hasattr(recon, 'feature_encoder')

def test_liquid_3d_forward_shape():
    """Liquid 3D produces correct output shapes."""
    from vision_language_integration.liquid_3d_reconstructor import Liquid3DTrajectoryReconstructor
    
    recon = Liquid3DTrajectoryReconstructor()
    
    # Noisy 3D points from triangulation
    noisy_points = torch.randn(4, 32, 3)  # Batch=4, Time=32, 3D
    
    features, smooth_traj = recon(noisy_points)
    
    assert features.shape == (4, 256), f"Features: expected (4, 256), got {features.shape}"
    assert smooth_traj.shape == (4, 32, 3), f"Trajectory: expected (4, 32, 3), got {smooth_traj.shape}"

def test_liquid_3d_smoothness():
    """Liquid 3D produces smoother trajectories than input."""
    from vision_language_integration.liquid_3d_reconstructor import Liquid3DTrajectoryReconstructor
    
    recon = Liquid3DTrajectoryReconstructor()
    
    # Create noisy trajectory
    t = torch.linspace(0, 1, 32)
    clean = torch.stack([t, t**2, torch.sin(t*6.28)], dim=-1)  # (32, 3)
    noisy = clean + torch.randn_like(clean) * 0.1  # Add noise
    noisy = noisy.unsqueeze(0)  # (1, 32, 3)
    
    _, smooth = recon(noisy)
    
    # Measure smoothness via 2nd derivative
    noise_jerk = torch.diff(noisy[0], n=2, dim=0).abs().mean()
    smooth_jerk = torch.diff(smooth[0], n=2, dim=0).abs().mean()
    
    assert smooth_jerk < noise_jerk, f"Smoothing failed: {smooth_jerk} >= {noise_jerk}"

def test_liquid_3d_gradients_flow():
    """Backpropagation works through Liquid 3D."""
    from vision_language_integration.liquid_3d_reconstructor import Liquid3DTrajectoryReconstructor
    
    recon = Liquid3DTrajectoryReconstructor()
    
    points = torch.randn(2, 16, 3, requires_grad=True)
    features, smooth = recon(points)
    
    loss = features.sum() + smooth.sum()
    loss.backward()
    
    assert points.grad is not None
    assert torch.isfinite(points.grad).all()

def test_liquid_3d_temporal_consistency():
    """Liquid 3D maintains temporal consistency."""
    from vision_language_integration.liquid_3d_reconstructor import Liquid3DTrajectoryReconstructor
    
    recon = Liquid3DTrajectoryReconstructor()
    
    # Same points, run twice
    points = torch.randn(1, 10, 3)
    
    _, traj1 = recon(points)
    _, traj2 = recon(points)
    
    # Should be deterministic for same input
    assert torch.allclose(traj1, traj2, rtol=1e-4)
PYTEST

        echo "✅ Test file created: tests/test_liquid_3d_reconstruction.py"
        echo "Run: pytest tests/test_liquid_3d_reconstruction.py -v"
        ;;
        
    3)
        echo "📸 Capturing RED evidence..."
        pytest tests/test_liquid_3d_reconstruction.py -v > artifacts/worker2/tdd_red_3d.txt 2>&1 || true
        
        cat >> artifacts/worker2/tdd_red_3d.txt << EOF

=== TDD RED Phase Evidence ===
Date: $(date)
Branch: $(git branch --show-current)
Worker: 2 (Liquid 3D Reconstruction)

Tests written FIRST (expected to FAIL)
EOF

        echo "✅ RED evidence: artifacts/worker2/tdd_red_3d.txt"
        ;;
        
    4)
        echo "💾 Committing RED phase..."
        
        cat > status/worker2_status.md << EOF
# Worker 2: Liquid 3D Reconstruction

**Last Update:** $(date)
**Branch:** worker/liquid-worker-2-3d
**Phase:** RED (Tests Written)

## Completed
- [x] liquid_cell.py available
- [x] Test suite written (6 tests)
- [x] RED evidence captured

## Next
- [ ] Implement Liquid3DTrajectoryReconstructor

**ETA:** 2 days to GREEN
EOF

        git add tests/ artifacts/worker2/ status/worker2_status.md
        git commit -m "🔴 Worker 2: RED - 3D reconstruction tests" || echo "No changes"
        git push origin worker/liquid-worker-2-3d
        ;;
        
    5)
        echo "💻 Implementing Liquid3DTrajectoryReconstructor..."
        echo ""
        echo "Create file:"
        echo "  experiments/.../vision_language_integration/liquid_3d_reconstructor.py"
        echo ""
        cat << 'PYTHONCODE'
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '..')
from liquid_models.liquid_cell import LiquidCell

class Liquid3DTrajectoryReconstructor(nn.Module):
    """Temporally-consistent 3D reconstruction using Liquid dynamics."""
    
    def __init__(self, dt=0.033):
        super().__init__()
        # Liquid cell for temporal smoothing
        self.liquid_dynamics = LiquidCell(
            input_size=3,      # 3D position (x, y, z)
            hidden_size=64,    # Internal state
            dt=dt             # 30 FPS video
        )
        self.position_predictor = nn.Linear(64, 3)  # Refined position
        self.feature_encoder = nn.Linear(64, 256)   # Trajectory features
    
    def forward(self, noisy_3d_points):
        """
        Args:
            noisy_3d_points: (B, T, 3) - frame-by-frame triangulated
        Returns:
            features: (B, 256) - temporally-consistent encoding
            smooth_trajectory: (B, T, 3) - smoothed 3D positions
        """
        B, T, _ = noisy_3d_points.shape
        device = noisy_3d_points.device
        h = torch.zeros(B, 64, device=device)
        
        smooth_positions = []
        for t in range(T):
            # Liquid dynamics for temporal consistency
            h = self.liquid_dynamics(noisy_3d_points[:, t], h)
            
            # Predict refined position
            smooth_pos = self.position_predictor(h)
            smooth_positions.append(smooth_pos)
        
        # Final encoding
        features = self.feature_encoder(h)  # (B, 256)
        smooth_trajectory = torch.stack(smooth_positions, dim=1)  # (B, T, 3)
        
        return features, smooth_trajectory
PYTHONCODE
        ;;
        
    6)
        echo "🧪 Running tests (expect GREEN)..."
        pytest tests/test_liquid_3d_reconstruction.py -v
        ;;
        
    7)
        echo "📸 Capturing GREEN evidence..."
        pytest tests/test_liquid_3d_reconstruction.py -v > artifacts/worker2/tdd_green_3d.txt 2>&1
        
        cat >> artifacts/worker2/tdd_green_3d.txt << EOF

=== TDD GREEN Phase Evidence ===
Date: $(date)
Worker: 2 (Liquid 3D)

All tests PASSING
EOF

        echo "✅ GREEN evidence captured"
        ;;
        
    8)
        echo "💾 Committing GREEN phase..."
        
        cat > status/worker2_status.md << EOF
# Worker 2: Liquid 3D Reconstruction

**Last Update:** $(date)
**Branch:** worker/liquid-worker-2-3d
**Phase:** GREEN (Complete)

## Completed
- [x] All tests passing
- [x] Implementation complete
- [x] Ready for Worker 3 integration

**ETA:** Ready now
EOF

        git add .
        git commit -m "✅ Worker 2: GREEN - 3D reconstruction complete"
        git push origin worker/liquid-worker-2-3d
        ;;
        
    12)
        vim status/worker2_status.md
        git add status/worker2_status.md
        git commit -m "📊 Worker 2: Status update"
        git push
        ;;
        
    13)
        cat status/worker2_status.md
        ;;
        
    14)
        bash scripts/sync_to_macbook.sh
        ;;
        
    15)
        bash scripts/tdd_capture.sh
        ;;
        
    16)
        bash scripts/prove.sh
        ;;
        
    0)
        exit 0
        ;;
esac

