# Three-Layer Multi-Camera Validation System

## TDD Evidence

- **RED Phase**: `artifacts/tdd_validation_red.txt` - 16 tests failed (expected)
- **GREEN Phase**: `artifacts/tdd_validation_GREEN.txt` - 16 tests passed
- **REFACTOR Phase**: `artifacts/tdd_validation_refactor.txt` - 51 tests passed (no regressions)

## System Overview

### Layer 1: Design-Time Validation
- Validates camera/workspace compatibility upfront
- Cameras: 2
- Workspace bounds: X=(-0.25, 0.25), Y=(-0.2, 0.2), Z=(1.6, 2.2)
- Focal length: 40
- All workspace corners visible: True

### Layer 2: Workspace-Constrained Generation
- Trajectories generated within validated bounds
- Safety margin: 5% (prevents edge cases)
- Trajectory types: linear, circular, helical, parabolic

### Layer 3: Runtime Validation
- Safety net for edge cases
- Retries needed: 0 (should be ~0)
- Min visible ratio: 95%

## Dataset Generated

- Total videos: 32
- Video shape: (16, 3, 64, 64) (T, C, H, W)
- Unique labels: 4
- Cameras per trajectory: 2
- **Visibility guarantee: 100% from all cameras**

## Key Features

1. **Proactive Prevention (Layer 1)**: Validates design before generating data
2. **Constrained Generation (Layer 2)**: Trajectories stay within validated workspace
3. **Runtime Safety Net (Layer 3)**: Catches rare edge cases
4. **Extensible**: Easy to add new trajectory types via `register_generator()`
5. **Zero Hidden Failures**: If Layer 1 passes, Layers 2 & 3 should rarely trigger
