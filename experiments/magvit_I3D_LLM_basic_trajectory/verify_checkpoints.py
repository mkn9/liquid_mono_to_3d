#!/usr/bin/env python3
"""
Pragmatic Checkpoint Verification

Verifies that saved checkpoints are actually usable.
Tests:
1. Checkpoints are loadable
2. Model can be restored from checkpoint
3. Restored model can make predictions
4. Checkpoints track progress correctly
"""

import torch
import numpy as np
from pathlib import Path
import json
from train_magvit import create_model

print("="*70)
print("PRAGMATIC CHECKPOINT VERIFICATION")
print("="*70)
print()

results_dir = Path("results/magvit_training")

# Find all checkpoint files
checkpoint_files = sorted(results_dir.glob("*checkpoint_epoch_*.pt"))
best_model_file = results_dir / "20260125_0329_best_model.pt"
final_model_file = list(results_dir.glob("*final_model.pt"))

print(f"Found {len(checkpoint_files)} epoch checkpoints")
print(f"Best model exists: {best_model_file.exists()}")
print(f"Final model exists: {len(final_model_file) > 0}")
print()

# TEST 1: Verify checkpoints are loadable
print("TEST 1: Checkpoint Loadability")
print("-" * 70)

loadable_count = 0
for cp_file in checkpoint_files:
    try:
        checkpoint = torch.load(cp_file, map_location='cpu')
        
        # Check required keys
        required_keys = ['epoch', 'model_state_dict', 'optimizer_state_dict', 'loss']
        missing_keys = [k for k in required_keys if k not in checkpoint]
        
        if missing_keys:
            print(f"❌ {cp_file.name}: Missing keys: {missing_keys}")
        else:
            loadable_count += 1
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            print(f"✅ {cp_file.name}: Epoch {epoch}, Loss {loss:.6f}")
            
    except Exception as e:
        print(f"❌ {cp_file.name}: Load error: {e}")

print(f"\nResult: {loadable_count}/{len(checkpoint_files)} checkpoints loadable")
print()

# TEST 2: Verify model can be restored from checkpoint
print("TEST 2: Model Restoration")
print("-" * 70)

test_checkpoint = checkpoint_files[3] if len(checkpoint_files) > 3 else checkpoint_files[0]
print(f"Testing with: {test_checkpoint.name}")

try:
    # Load checkpoint
    checkpoint = torch.load(test_checkpoint, map_location='cpu')
    print(f"✅ Checkpoint loaded: epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.6f}")
    
    # Create model
    model = create_model(image_size=64, init_dim=64, use_fsq=True)
    print(f"✅ Model created")
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Model state loaded successfully")
    
    # Check model has expected attributes
    assert hasattr(model, 'encode'), "Model missing encode method"
    assert hasattr(model, 'decode'), "Model missing decode method"
    print(f"✅ Model has encode/decode methods")
    
    restoration_success = True
    
except Exception as e:
    print(f"❌ Model restoration failed: {e}")
    import traceback
    traceback.print_exc()
    restoration_success = False

print()

# TEST 3: Verify restored model can make predictions
print("TEST 3: Model Prediction")
print("-" * 70)

if restoration_success:
    try:
        # Create test input (1 video, 3 channels, 16 frames, 64x64)
        test_input = torch.randn(1, 3, 16, 64, 64)
        print(f"✅ Test input created: {test_input.shape}")
        
        # Test encoding
        model.eval()
        with torch.no_grad():
            codes = model.encode(test_input)
        print(f"✅ Encoding works: {codes.shape}")
        
        # Test decoding
        with torch.no_grad():
            reconstructed = model.decode(codes)
        print(f"✅ Decoding works: {reconstructed.shape}")
        
        # Verify reconstruction shape matches input
        assert reconstructed.shape == test_input.shape, \
            f"Shape mismatch: {reconstructed.shape} vs {test_input.shape}"
        print(f"✅ Reconstruction shape correct")
        
        # Calculate reconstruction error
        mse = torch.mean((reconstructed - test_input) ** 2).item()
        print(f"✅ MSE (random input): {mse:.6f}")
        
        prediction_success = True
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        prediction_success = False
else:
    print("⚠️ Skipped (model restoration failed)")
    prediction_success = False

print()

# TEST 4: Verify checkpoints track progress correctly
print("TEST 4: Progress Tracking")
print("-" * 70)

try:
    epochs = []
    losses = []
    
    for cp_file in checkpoint_files:
        checkpoint = torch.load(cp_file, map_location='cpu')
        epochs.append(checkpoint['epoch'])
        losses.append(checkpoint['loss'])
    
    # Check epoch progression
    expected_epochs = sorted(epochs)
    epoch_progression_correct = (epochs == expected_epochs)
    
    print(f"Epochs: {epochs}")
    print(f"✅ Epoch progression correct: {epoch_progression_correct}")
    
    # Check loss trend (should generally decrease)
    loss_decreasing = losses[-1] < losses[0]
    print(f"Losses: {[f'{l:.6f}' for l in losses]}")
    print(f"✅ Final loss < Initial loss: {loss_decreasing} ({losses[-1]:.6f} < {losses[0]:.6f})")
    
    # Check for duplicate epochs
    unique_epochs = len(set(epochs)) == len(epochs)
    print(f"✅ No duplicate epochs: {unique_epochs}")
    
    progress_tracking_success = epoch_progression_correct and loss_decreasing and unique_epochs
    
except Exception as e:
    print(f"❌ Progress tracking check failed: {e}")
    import traceback
    traceback.print_exc()
    progress_tracking_success = False

print()

# TEST 5: Compare adjacent checkpoints (weights changed)
print("TEST 5: Weight Update Verification")
print("-" * 70)

try:
    if len(checkpoint_files) >= 2:
        cp1 = torch.load(checkpoint_files[0], map_location='cpu')
        cp2 = torch.load(checkpoint_files[1], map_location='cpu')
        
        # Get first weight tensor from each
        key = list(cp1['model_state_dict'].keys())[0]
        w1 = cp1['model_state_dict'][key]
        w2 = cp2['model_state_dict'][key]
        
        weights_changed = not torch.equal(w1, w2)
        max_diff = torch.max(torch.abs(w1 - w2)).item()
        
        print(f"Comparing: {checkpoint_files[0].name} vs {checkpoint_files[1].name}")
        print(f"Weight key: {key}")
        print(f"✅ Weights changed: {weights_changed}")
        print(f"✅ Max weight diff: {max_diff:.6e}")
        
        weights_updated = weights_changed and max_diff > 1e-6
        
    else:
        print("⚠️ Not enough checkpoints to compare")
        weights_updated = None
        
except Exception as e:
    print(f"❌ Weight comparison failed: {e}")
    weights_updated = False

print()

# TEST 6: Load training history and verify consistency
print("TEST 6: Training History Consistency")
print("-" * 70)

try:
    history_file = list(results_dir.glob("*training_history.json"))[0]
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    print(f"✅ History file loaded: {len(history)} epochs")
    
    # Verify last checkpoint matches last history entry
    last_checkpoint = checkpoint_files[-1]
    last_cp = torch.load(last_checkpoint, map_location='cpu')
    last_history = history[-1]
    
    epoch_match = last_cp['epoch'] == last_history['epoch']
    loss_match = abs(last_cp['loss'] - last_history['val_loss']) < 1e-6
    
    print(f"Last checkpoint epoch: {last_cp['epoch']}")
    print(f"Last history epoch: {last_history['epoch']}")
    print(f"✅ Epochs match: {epoch_match}")
    
    print(f"Last checkpoint loss: {last_cp['loss']:.6f}")
    print(f"Last history val_loss: {last_history['val_loss']:.6f}")
    print(f"✅ Losses match: {loss_match}")
    
    history_consistent = epoch_match and loss_match
    
except Exception as e:
    print(f"❌ History consistency check failed: {e}")
    history_consistent = False

print()

# FINAL SUMMARY
print("="*70)
print("VERIFICATION SUMMARY")
print("="*70)

tests = {
    "Checkpoint Loadability": loadable_count == len(checkpoint_files),
    "Model Restoration": restoration_success,
    "Model Prediction": prediction_success,
    "Progress Tracking": progress_tracking_success,
    "Weight Updates": weights_updated,
    "History Consistency": history_consistent,
}

passed = sum(1 for v in tests.values() if v is True)
total = len(tests)

print()
for test_name, result in tests.items():
    if result is True:
        status = "✅ PASS"
    elif result is False:
        status = "❌ FAIL"
    else:
        status = "⚠️ SKIP"
    print(f"{status} - {test_name}")

print()
print(f"Result: {passed}/{total} tests passed")
print()

if passed == total:
    print("🎉 ALL CHECKS PASSED - Checkpoints are verified to work!")
    confidence = "HIGH (70-80%)"
elif passed >= total * 0.75:
    print("✅ MOST CHECKS PASSED - Checkpoints appear functional")
    confidence = "MEDIUM-HIGH (60-70%)"
elif passed >= total * 0.5:
    print("⚠️ SOME ISSUES - Checkpoints may have problems")
    confidence = "MEDIUM (40-50%)"
else:
    print("❌ MAJOR ISSUES - Checkpoints may not be usable")
    confidence = "LOW (20-30%)"

print(f"Confidence level: {confidence}")
print()
print("="*70)

# Save verification results
verification_results = {
    "timestamp": "2026-01-25",
    "tests": {k: str(v) for k, v in tests.items()},
    "passed": passed,
    "total": total,
    "confidence": confidence
}

verification_file = results_dir / "checkpoint_verification_results.json"
with open(verification_file, 'w') as f:
    json.dump(verification_results, f, indent=2)

print(f"✅ Results saved to: {verification_file}")

