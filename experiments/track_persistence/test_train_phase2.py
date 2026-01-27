#!/usr/bin/env python3
"""Test Phase 2 training script structure."""

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_dataset_creation():
    """Test TrackDataset can be created."""
    try:
        from train_phase2_magvit import TrackDataset
        
        # Mock data
        videos = np.random.rand(100, 25, 128, 128, 3).astype(np.float32)
        metadata = [
            {'label': 'persistent'} for _ in range(50)
        ] + [
            {'label': 'brief'} for _ in range(50)
        ]
        
        dataset = TrackDataset(videos, metadata)
        
        assert len(dataset) == 100
        assert dataset[0]['video'].shape == (25, 128, 128, 3)
        assert dataset[0]['label'] in [0, 1]
        
        print("✅ TrackDataset creation working")
        return True
        
    except Exception as e:
        print(f"❌ TrackDataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_load_dataset():
    """Test load_dataset function structure."""
    try:
        from train_phase2_magvit import load_dataset
        
        print("✅ load_dataset function exists")
        return True
        
    except Exception as e:
        print(f"❌ load_dataset import failed: {e}")
        return False


def test_train_functions():
    """Test training function signatures."""
    try:
        from train_phase2_magvit import train_epoch, validate, train_model
        
        print("✅ Training functions exist")
        return True
        
    except Exception as e:
        print(f"❌ Training functions import failed: {e}")
        return False


def test_create_phase2_model_signature():
    """Test create_phase2_model function signature."""
    try:
        from train_phase2_magvit import create_phase2_model
        import inspect
        
        sig = inspect.signature(create_phase2_model)
        params = list(sig.parameters.keys())
        
        assert 'magvit_checkpoint' in params
        assert 'feature_type' in params
        assert 'freeze_encoder' in params
        
        print("✅ create_phase2_model function signature correct")
        return True
        
    except Exception as e:
        print(f"❌ create_phase2_model signature check failed: {e}")
        return False


def test_dataset_labels():
    """Test that labels are correctly assigned."""
    try:
        from train_phase2_magvit import TrackDataset
        
        videos = np.random.rand(10, 25, 128, 128, 3).astype(np.float32)
        metadata = [
            {'label': 'persistent'},  # Should be 1
            {'label': 'mixed'},       # Should be 1
            {'label': 'brief'},       # Should be 0
            {'label': 'noise'},       # Should be 0
            {'label': 'persistent'},  # Should be 1
            {'label': 'brief'},       # Should be 0
            {'label': 'mixed'},       # Should be 1
            {'label': 'noise'},       # Should be 0
            {'label': 'persistent'},  # Should be 1
            {'label': 'brief'},       # Should be 0
        ]
        
        dataset = TrackDataset(videos, metadata)
        
        expected_labels = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]
        
        for i, expected in enumerate(expected_labels):
            actual = dataset[i]['label'].item()
            assert actual == expected, f"Label mismatch at index {i}: expected {expected}, got {actual}"
        
        print("✅ Dataset labels correctly assigned")
        return True
        
    except Exception as e:
        print(f"❌ Dataset labels test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader_compatibility():
    """Test that dataset works with PyTorch DataLoader."""
    try:
        from train_phase2_magvit import TrackDataset
        from torch.utils.data import DataLoader
        
        videos = np.random.rand(50, 25, 128, 128, 3).astype(np.float32)
        metadata = [{'label': 'persistent' if i % 2 == 0 else 'brief'} 
                   for i in range(50)]
        
        dataset = TrackDataset(videos, metadata)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        batch = next(iter(dataloader))
        
        assert batch['video'].shape == (8, 25, 128, 128, 3)
        assert batch['label'].shape == (8,)
        assert len(batch['metadata']) == 8
        
        print("✅ DataLoader compatibility confirmed")
        return True
        
    except Exception as e:
        print(f"❌ DataLoader compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("="*70)
    print("Testing Phase 2 Training Script")
    print("="*70)
    
    tests = [
        test_dataset_creation,
        test_load_dataset,
        test_train_functions,
        test_create_phase2_model_signature,
        test_dataset_labels,
        test_dataloader_compatibility
    ]
    
    results = []
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        results.append(test())
    
    print("\n" + "="*70)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("="*70)
    
    if all(results):
        print("✅ ALL TESTS PASSED")
        print("\n⚠️  Note: Full training can only be tested on EC2 with:")
        print("   - MagVit checkpoint")
        print("   - Generated dataset")
        print("   - GPU support")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)

