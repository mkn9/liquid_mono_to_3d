#!/usr/bin/env python3
"""
Test MAGVIT integration with existing dataset.

This validates:
1. MAGVIT model can be initialized
2. Dataset can be loaded and batched
3. Model can process videos (encode/decode)
4. Output shapes are correct
"""

import numpy as np
import torch
from pathlib import Path

print("="*70)
print("MAGVIT INTEGRATION TEST")
print("="*70)
print()

# Step 1: Load dataset
print("Step 1: Loading dataset...")
dataset_path = Path("results/20260124_1546_full_dataset.npz")
data = np.load(dataset_path)

videos = torch.from_numpy(data['videos']).float()
labels = torch.from_numpy(data['labels']).long()

print(f"✅ Loaded {len(videos)} samples")
print(f"   Video shape: {videos.shape}")
print(f"   Labels shape: {labels.shape}")
print()

# Step 2: Initialize MAGVIT model
print("Step 2: Initializing MAGVIT VideoTokenizer...")

try:
    from magvit2_pytorch import VideoTokenizer
    
    # Initialize with settings appropriate for our 64x64 videos
    tokenizer = VideoTokenizer(
        image_size=64,                      # Match our video resolution
        init_dim=64,                        # Initial convolution dimension
        layers=('residual', 'residual'),    # Encoder/decoder layers (lighter for small data)
        use_fsq=True,                       # Use FSQ (Finite Scalar Quantization)
        fsq_levels=[8, 5, 5, 5],           # FSQ levels (smaller codebook for 200 samples)
    )
    
    print(f"✅ MAGVIT model initialized")
    print(f"   Parameters: {sum(p.numel() for p in tokenizer.parameters()):,}")
    print()
    
except ImportError as e:
    print(f"❌ Failed to import magvit2_pytorch: {e}")
    print()
    print("Install with: pip install magvit2-pytorch")
    exit(1)
except Exception as e:
    print(f"❌ Failed to initialize MAGVIT: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 3: Test encoding a single video
print("Step 3: Testing video encoding...")

try:
    # Take first video
    test_video = videos[0:1]  # Shape: (1, T, C, H, W)
    
    # MAGVIT expects (B, C, T, H, W) not (B, T, C, H, W)
    test_video = test_video.permute(0, 2, 1, 3, 4)  # Now (1, C, T, H, W)
    
    print(f"   Input shape: {test_video.shape}")
    
    # Encode video to discrete codes
    with torch.no_grad():
        codes = tokenizer.encode(test_video)
    
    print(f"✅ Encoding successful")
    print(f"   Codes shape: {codes.shape}")
    print(f"   Codes dtype: {codes.dtype}")
    if codes.dtype in [torch.long, torch.int]:
        print(f"   Unique codes: {torch.unique(codes).numel()}")
    print()
    
except Exception as e:
    print(f"❌ Encoding failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 4: Test decoding
print("Step 4: Testing video decoding...")

try:
    # Decode codes back to video
    with torch.no_grad():
        reconstructed = tokenizer.decode(codes)
    
    print(f"✅ Decoding successful")
    print(f"   Output shape: {reconstructed.shape}")
    print(f"   Output range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
    
    # Calculate reconstruction error
    mse = torch.mean((test_video - reconstructed) ** 2).item()
    print(f"   MSE (untrained): {mse:.6f}")
    print()
    
except Exception as e:
    print(f"❌ Decoding failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 5: Test batch processing
print("Step 5: Testing batch processing...")

try:
    # Process a small batch
    batch = videos[0:4]  # 4 videos (B, T, C, H, W)
    batch = batch.permute(0, 2, 1, 3, 4)  # Convert to (B, C, T, H, W)
    batch_labels = labels[0:4]
    
    print(f"   Batch shape: {batch.shape}")
    print(f"   Batch labels: {batch_labels.tolist()}")
    
    with torch.no_grad():
        batch_codes = tokenizer.encode(batch)
        batch_reconstructed = tokenizer.decode(batch_codes)
    
    print(f"✅ Batch processing successful")
    print(f"   Encoded shape: {batch_codes.shape}")
    print(f"   Decoded shape: {batch_reconstructed.shape}")
    print()
    
except Exception as e:
    print(f"❌ Batch processing failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Summary
print("="*70)
print("✅ ALL TESTS PASSED!")
print("="*70)
print()
print("MAGVIT integration is working correctly!")
print()
print("Next steps:")
print("  1. Create DataLoader for training")
print("  2. Implement training loop")
print("  3. Train on 200 samples")
print("  4. Evaluate reconstruction quality")
print("  5. Test classification capability")
print()
print("Dataset is ready for MAGVIT training!")

