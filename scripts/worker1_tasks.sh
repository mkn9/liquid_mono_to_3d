#!/bin/bash
# Worker 1: Liquid Fusion Layer - Daily Tasks
# Implements dual-modal fusion with Liquid dynamics

set -e

PROJECT_DIR="$HOME/liquid_mono_to_3d"
cd "$PROJECT_DIR"

echo "🔵 Worker 1: Liquid Fusion Layer"
echo "=================================="
echo ""

# Check we're on correct branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "worker/liquid-worker-1-fusion" ]; then
    echo "⚠️  Wrong branch! Switching to worker/liquid-worker-1-fusion..."
    git checkout worker/liquid-worker-1-fusion
fi

# Show task menu
cat << 'EOF'
Select task:

DAY 1-2: Setup & TDD RED Phase
  1) Port liquid_cell.py from liquid_ai_2
  2) Write fusion tests (RED phase)
  3) Capture RED evidence
  4) Update status & commit

DAY 3-4: Implementation (TDD GREEN Phase)
  5) Implement LiquidDualModalFusion
  6) Run tests (expect GREEN)
  7) Capture GREEN evidence
  8) Update status & commit

DAY 5: Integration
  9) Add --fusion-type flag to demo
 10) Test both static and liquid fusion
 11) Initial comparison

EVALUATION (Week 2)
 12) Run evaluation on 20 samples
 13) Measure metrics
 14) Generate comparison report

STATUS & UTILITIES
 15) Update worker status
 16) View current status
 17) Sync to MacBook now
 18) Run TDD capture
 19) Create proof bundle

 0) Exit

Enter choice:
EOF

read -p "> " choice

case $choice in
    1)
        echo "📦 Porting liquid_cell.py from liquid_ai_2..."
        echo ""
        echo "Run this from your MacBook:"
        echo ""
        echo "rsync -avz -e \"ssh -i /Users/mike/keys/AutoGenKeyPair.pem\" \\"
        echo "  ~/Dropbox/Code/repos/liquid_ai_2/option1_synthetic/liquid_cell.py \\"
        echo "  ubuntu@204.236.245.232:~/liquid_mono_to_3d/experiments/trajectory_video_understanding/liquid_models/"
        echo ""
        echo "After copying, verify:"
        ls -lh experiments/trajectory_video_understanding/liquid_models/liquid_cell.py 2>/dev/null || echo "❌ Not found yet"
        ;;
        
    2)
        echo "✍️  Writing fusion tests (RED phase)..."
        echo ""
        
        # Create test file
        mkdir -p tests
        cat > tests/test_liquid_fusion.py << 'PYTEST'
"""
Tests for Liquid Dual-Modal Fusion Layer
TDD RED Phase - Tests written FIRST
"""
import torch
import pytest
import sys
import os

# Add project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments', 'trajectory_video_understanding'))

def test_liquid_fusion_can_import():
    """Test that we can import the fusion module."""
    try:
        from vision_language_integration.dual_visual_adapter import LiquidDualModalFusion
        assert True
    except ImportError as e:
        pytest.fail(f"Cannot import LiquidDualModalFusion: {e}")

def test_liquid_fusion_initialization():
    """Liquid fusion initializes with correct parameters."""
    from vision_language_integration.dual_visual_adapter import LiquidDualModalFusion
    
    fusion = LiquidDualModalFusion()
    
    # Check adapters exist
    assert hasattr(fusion, 'adapter_2d')
    assert hasattr(fusion, 'adapter_3d')
    assert hasattr(fusion, 'liquid_fusion')
    
    # Check dimensions
    assert fusion.adapter_2d.in_features == 512
    assert fusion.adapter_2d.out_features == 4096
    assert fusion.adapter_3d.in_features == 256
    assert fusion.adapter_3d.out_features == 4096

def test_liquid_fusion_forward_shape():
    """Liquid fusion produces correct output shape."""
    from vision_language_integration.dual_visual_adapter import LiquidDualModalFusion
    
    fusion = LiquidDualModalFusion()
    
    # Create dummy inputs
    feat_2d = torch.randn(4, 512)  # Batch=4, 2D features
    feat_3d = torch.randn(4, 256)  # Batch=4, 3D features
    
    output = fusion(feat_2d, feat_3d)
    
    assert output.shape == (4, 4096), f"Expected (4, 4096), got {output.shape}"

def test_liquid_fusion_temporal_consistency():
    """Liquid fusion maintains temporal state."""
    from vision_language_integration.dual_visual_adapter import LiquidDualModalFusion
    
    fusion = LiquidDualModalFusion()
    
    feat_2d = torch.randn(1, 512)
    feat_3d = torch.randn(1, 256)
    
    # First call
    out1 = fusion(feat_2d, feat_3d, reset_state=True)
    
    # Second call with same input (should differ due to state)
    out2 = fusion(feat_2d, feat_3d, reset_state=False)
    
    # Outputs should be different (fusion has memory)
    assert not torch.allclose(out1, out2, rtol=1e-3), \
        "Fusion should maintain state between calls"

def test_liquid_fusion_gradients_flow():
    """Backpropagation works through Liquid fusion."""
    from vision_language_integration.dual_visual_adapter import LiquidDualModalFusion
    
    fusion = LiquidDualModalFusion()
    
    feat_2d = torch.randn(2, 512, requires_grad=True)
    feat_3d = torch.randn(2, 256, requires_grad=True)
    
    output = fusion(feat_2d, feat_3d)
    loss = output.sum()
    loss.backward()
    
    # Check gradients exist
    assert feat_2d.grad is not None, "No gradients for 2D features"
    assert feat_3d.grad is not None, "No gradients for 3D features"
    
    # Check gradients are finite
    assert torch.isfinite(feat_2d.grad).all(), "NaN/Inf in 2D gradients"
    assert torch.isfinite(feat_3d.grad).all(), "NaN/Inf in 3D gradients"

def test_liquid_fusion_reset_state():
    """State reset works correctly."""
    from vision_language_integration.dual_visual_adapter import LiquidDualModalFusion
    
    fusion = LiquidDualModalFusion()
    
    feat_2d = torch.randn(1, 512)
    feat_3d = torch.randn(1, 256)
    
    # Run with state
    fusion(feat_2d, feat_3d, reset_state=True)
    fusion(feat_2d, feat_3d, reset_state=False)
    out1 = fusion(feat_2d, feat_3d, reset_state=False)
    
    # Reset and run again
    out2 = fusion(feat_2d, feat_3d, reset_state=True)
    
    # Should be different (different state)
    assert not torch.allclose(out1, out2, rtol=1e-3), \
        "Reset should clear state"
PYTEST

        echo "✅ Test file created: tests/test_liquid_fusion.py"
        echo ""
        echo "Run tests to verify RED phase:"
        echo "  pytest tests/test_liquid_fusion.py -v"
        ;;
        
    3)
        echo "📸 Capturing RED evidence..."
        cd "$PROJECT_DIR"
        
        # Run tests (should fail)
        pytest tests/test_liquid_fusion.py -v > artifacts/worker1/tdd_red_fusion.txt 2>&1 || true
        
        # Capture system state
        cat >> artifacts/worker1/tdd_red_fusion.txt << EOF

=== TDD RED Phase Evidence ===
Date: $(date)
Branch: $(git branch --show-current)
Commit: $(git rev-parse HEAD)
Worker: 1 (Liquid Fusion)

Tests written FIRST (expected to FAIL)
Implementation will come next (GREEN phase)
EOF

        echo "✅ RED evidence captured: artifacts/worker1/tdd_red_fusion.txt"
        cat artifacts/worker1/tdd_red_fusion.txt | tail -20
        ;;
        
    4)
        echo "💾 Updating status and committing..."
        
        # Update status
        cat > status/worker1_status.md << EOF
# Worker 1: Liquid Fusion Layer

**Last Update:** $(date)
**Branch:** worker/liquid-worker-1-fusion
**Phase:** RED (Tests Written)

## Completed
- [x] Directory structure
- [x] liquid_cell.py ported
- [x] Test suite written (6 tests)
- [x] RED evidence captured

## In Progress
- [ ] LiquidDualModalFusion implementation

## Blocked
- None

## Next Steps
1. Implement LiquidDualModalFusion class
2. Run tests (expect GREEN)
3. Capture GREEN evidence

**ETA:** 2 days to GREEN phase
EOF

        # Commit
        git add tests/test_liquid_fusion.py
        git add artifacts/worker1/
        git add status/worker1_status.md
        git commit -m "🔴 Worker 1: RED - Liquid fusion tests written" || echo "Nothing to commit"
        git push origin worker/liquid-worker-1-fusion
        
        echo "✅ Status updated and committed"
        ;;
        
    5)
        echo "💻 Implementing LiquidDualModalFusion..."
        echo ""
        echo "Create/edit this file:"
        echo "  experiments/trajectory_video_understanding/vision_language_integration/dual_visual_adapter.py"
        echo ""
        echo "Add this class:"
        cat << 'PYTHONCODE'

from liquid_models.liquid_cell import LiquidCell

class LiquidDualModalFusion(nn.Module):
    """Dynamic fusion of 2D and 3D features using Liquid dynamics."""
    
    def __init__(self):
        super().__init__()
        # Keep existing adapters
        self.adapter_2d = nn.Linear(512, 4096)
        self.adapter_3d = nn.Linear(256, 4096)
        
        # Replace static fusion with Liquid
        self.liquid_fusion = LiquidCell(
            input_size=8192,   # Concatenated 2D+3D
            hidden_size=4096,  # Output for LLM
            dt=0.02           # 50 Hz
        )
        
        # Hidden state buffer
        self.register_buffer('h_fusion', None)
    
    def forward(self, features_2d, features_3d, reset_state=False):
        """
        Args:
            features_2d: (B, 512) - 2D visual features
            features_3d: (B, 256) - 3D trajectory features
            reset_state: Reset hidden state
        Returns:
            fused: (B, 4096) - Fused embedding
        """
        # Project to common space
        emb_2d = self.adapter_2d(features_2d)     # (B, 4096)
        emb_3d = self.adapter_3d(features_3d)     # (B, 4096)
        combined = torch.cat([emb_2d, emb_3d], dim=-1)  # (B, 8192)
        
        # Initialize or reset hidden state
        if self.h_fusion is None or reset_state:
            self.h_fusion = torch.zeros(combined.shape[0], 4096, 
                                        device=combined.device)
        
        # Liquid dynamics for temporal fusion
        self.h_fusion = self.liquid_fusion(combined, self.h_fusion)
        
        return self.h_fusion
PYTHONCODE

        echo ""
        echo "After implementing, run: pytest tests/test_liquid_fusion.py -v"
        ;;
        
    6)
        echo "🧪 Running tests (expect GREEN)..."
        pytest tests/test_liquid_fusion.py -v
        ;;
        
    7)
        echo "📸 Capturing GREEN evidence..."
        
        pytest tests/test_liquid_fusion.py -v > artifacts/worker1/tdd_green_fusion.txt 2>&1
        
        cat >> artifacts/worker1/tdd_green_fusion.txt << EOF

=== TDD GREEN Phase Evidence ===
Date: $(date)
Branch: $(git branch --show-current)
Commit: $(git rev-parse HEAD)
Worker: 1 (Liquid Fusion)

All tests PASSING
Implementation complete
Ready for integration
EOF

        echo "✅ GREEN evidence captured: artifacts/worker1/tdd_green_fusion.txt"
        cat artifacts/worker1/tdd_green_fusion.txt | tail -20
        ;;
        
    8)
        echo "💾 Updating status (GREEN phase)..."
        
        cat > status/worker1_status.md << EOF
# Worker 1: Liquid Fusion Layer

**Last Update:** $(date)
**Branch:** worker/liquid-worker-1-fusion
**Phase:** GREEN (Implementation Complete)

## Completed
- [x] Directory structure
- [x] liquid_cell.py ported
- [x] Test suite written (6 tests)
- [x] RED evidence captured
- [x] LiquidDualModalFusion implemented
- [x] All tests passing (GREEN)
- [x] GREEN evidence captured

## In Progress
- [ ] Integration testing

## Blocked
- None

## Next Steps
1. Add --fusion-type flag to demo
2. Test static vs liquid
3. Evaluation phase

**ETA:** Ready for Worker 3 integration
EOF

        git add .
        git commit -m "✅ Worker 1: GREEN - Liquid fusion implementation complete" || echo "Nothing new"
        git push origin worker/liquid-worker-1-fusion
        
        echo "✅ GREEN phase complete!"
        ;;
        
    15)
        echo "📝 Updating worker status..."
        vim status/worker1_status.md
        git add status/worker1_status.md
        git commit -m "📊 Worker 1: Status update" || echo "No changes"
        git push
        ;;
        
    16)
        echo "📊 Current Worker 1 Status:"
        cat status/worker1_status.md
        ;;
        
    17)
        echo "📤 Syncing to MacBook..."
        bash scripts/sync_to_macbook.sh
        ;;
        
    18)
        echo "📸 Running TDD capture..."
        bash scripts/tdd_capture.sh
        ;;
        
    19)
        echo "📦 Creating proof bundle..."
        cd "$PROJECT_DIR"
        bash scripts/prove.sh
        ;;
        
    0)
        echo "👋 Exiting Worker 1 tasks"
        exit 0
        ;;
        
    *)
        echo "❌ Invalid choice"
        ;;
esac

