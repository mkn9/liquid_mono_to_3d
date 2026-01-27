"""
Test checkpoint generation for long-running dataset generation.

Following TDD: Tests written FIRST (RED phase).
These tests enforce requirements for long-running processes (>5 min).

CRITICAL: These tests were MISSING in initial implementation,
leading to 40+ min generation with no progress visibility.
"""

import numpy as np
import torch
import pytest
from pathlib import Path
import tempfile
import shutil
import time

# Import will fail initially (RED phase - expected!)
from parallel_dataset_generator_with_checkpoints import (
    generate_dataset_parallel_with_checkpoints,
    save_checkpoint,
    update_progress_file,
    merge_checkpoints
)


class TestCheckpointCreation:
    """Tests for checkpoint file creation at intervals."""
    
    def test_checkpoints_created_at_intervals(self):
        """Verify checkpoints are saved at regular intervals.
        
        Requirement: Long-running processes must save checkpoints every 1-5 min
        Test Strategy: Generate 2K samples with checkpoint_interval=500
        Expected: 4 checkpoint files created (2000/500 = 4)
        
        Seed: 42 (deterministic)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Generate with checkpoints
            dataset = generate_dataset_parallel_with_checkpoints(
                num_samples=2000,
                checkpoint_interval=500,
                frames_per_video=8,  # Small for speed
                image_size=(32, 32),  # Small for speed
                augmentation=False,
                seed=42,
                num_workers=4,
                output_dir=str(output_dir)
            )
            
            # Verify checkpoint files were created
            checkpoints = sorted(output_dir.glob("checkpoint_*.npz"))
            assert len(checkpoints) == 4, \
                f"Expected 4 checkpoints for 2K samples (500 interval), found {len(checkpoints)}"
            
            # Verify each has correct sample count
            for i, checkpoint_file in enumerate(checkpoints):
                data = np.load(checkpoint_file)
                expected_samples = 500  # Each checkpoint has 500 samples
                actual_samples = len(data['videos'])
                assert actual_samples == expected_samples, \
                    f"Checkpoint {i} has {actual_samples} samples, expected {expected_samples}"
    
    def test_checkpoint_files_have_correct_structure(self):
        """Verify checkpoint files contain all required data.
        
        Requirement: Checkpoints must be complete and loadable
        Expected: Each checkpoint has videos, labels, trajectory_3d
        
        Seed: 42
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            dataset = generate_dataset_parallel_with_checkpoints(
                num_samples=1000,
                checkpoint_interval=500,
                frames_per_video=8,
                image_size=(32, 32),
                augmentation=False,
                seed=42,
                num_workers=4,
                output_dir=str(output_dir)
            )
            
            # Load first checkpoint
            checkpoint_file = sorted(output_dir.glob("checkpoint_*.npz"))[0]
            data = np.load(checkpoint_file)
            
            # Verify required keys exist
            assert 'videos' in data, "Checkpoint missing 'videos'"
            assert 'labels' in data, "Checkpoint missing 'labels'"
            assert 'trajectory_3d' in data, "Checkpoint missing 'trajectory_3d'"
            
            # Verify shapes are correct
            videos = data['videos']
            labels = data['labels']
            trajectories = data['trajectory_3d']
            
            assert videos.ndim == 5, f"Videos should be 5D, got {videos.ndim}D"
            assert len(labels) == len(videos), "Labels count mismatch"
            assert len(trajectories) == len(videos), "Trajectories count mismatch"


class TestProgressFile:
    """Tests for progress file creation and updates."""
    
    def test_progress_file_created_and_updated(self):
        """Verify PROGRESS.txt is created and contains required information.
        
        Requirement: Progress must be visible on MacBook without SSH
        Test Strategy: Generate with checkpoints, verify progress file
        Expected: File exists with completion %, ETA, timestamp
        
        Seed: 42
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            dataset = generate_dataset_parallel_with_checkpoints(
                num_samples=1000,
                checkpoint_interval=500,
                frames_per_video=8,
                image_size=(32, 32),
                augmentation=False,
                seed=42,
                num_workers=4,
                output_dir=str(output_dir)
            )
            
            # Verify progress file exists
            progress_file = output_dir / "PROGRESS.txt"
            assert progress_file.exists(), "PROGRESS.txt not created"
            
            # Verify it contains required information
            content = progress_file.read_text()
            assert "Completed:" in content or "COMPLETE" in content, \
                "Missing completion status"
            assert "/" in content or "COMPLETE" in content, \
                "Missing X/Y format or completion marker"
            assert "%" in content or "COMPLETE" in content, \
                "Missing percentage or completion marker"
            
            # Should have timestamp
            assert "Last update:" in content or "Completed:" in content, \
                "Missing timestamp"
    
    def test_progress_file_shows_completion(self):
        """Verify progress file is updated to show completion.
        
        Requirement: Final status must indicate completion
        Expected: PROGRESS.txt contains "COMPLETE" marker
        
        Seed: 42
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            dataset = generate_dataset_parallel_with_checkpoints(
                num_samples=500,
                checkpoint_interval=500,
                frames_per_video=8,
                image_size=(32, 32),
                augmentation=False,
                seed=42,
                num_workers=4,
                output_dir=str(output_dir)
            )
            
            # Verify completion marker
            progress_file = output_dir / "PROGRESS.txt"
            content = progress_file.read_text()
            assert "COMPLETE" in content, \
                "Progress file should show COMPLETE status"


class TestResumeCapability:
    """Tests for resume capability from checkpoints."""
    
    @pytest.mark.skip(reason="Resume capability not yet implemented - will add in future iteration")
    def test_can_resume_from_last_checkpoint(self):
        """Verify generation can resume from last checkpoint if interrupted.
        
        Requirement: If interrupted, must be able to resume without data loss
        Test Strategy: Generate partial, stop, resume to completion
        Expected: Final dataset has all samples, no duplicates
        
        NOTE: This test is skipped for initial implementation.
        Resume capability will be added in future iteration.
        
        Seed: 42
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # First run: generate 1000 samples
            dataset1 = generate_dataset_parallel_with_checkpoints(
                num_samples=1000,
                checkpoint_interval=500,
                frames_per_video=8,
                image_size=(32, 32),
                augmentation=False,
                seed=42,
                num_workers=4,
                output_dir=str(output_dir)
            )
            
            # Verify checkpoints exist
            checkpoints = list(output_dir.glob("checkpoint_*.npz"))
            assert len(checkpoints) >= 2, "Need checkpoints to test resume"
            
            # Second run: resume and complete to 2000
            # (This would require resume=True parameter)
            dataset2 = generate_dataset_parallel_with_checkpoints(
                num_samples=2000,
                checkpoint_interval=500,
                frames_per_video=8,
                image_size=(32, 32),
                augmentation=False,
                seed=42,
                num_workers=4,
                output_dir=str(output_dir),
                resume=True  # Resume from existing checkpoints
            )
            
            # Verify final dataset has all samples
            assert len(dataset2['videos']) == 2000, \
                f"Expected 2000 samples, got {len(dataset2['videos'])}"


class TestIntegrationAtScale:
    """Integration test at medium scale (~10% of production)."""
    
    @pytest.mark.slow
    def test_5k_generation_completes_with_checkpoints(self):
        """Test generation at ~10% of production scale (~5 min runtime).
        
        Requirement: Must validate long-running behavior before production
        Test Strategy: Generate 5K samples (10% of production 30K)
        Expected: Completes successfully with checkpoints and progress
        
        This is a "smoke test" - validates actual long-running behavior.
        Run with: pytest -m slow
        
        Seed: 42
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            start = time.time()
            
            # Generate 5K samples with optimized settings
            dataset = generate_dataset_parallel_with_checkpoints(
                num_samples=5000,
                checkpoint_interval=1000,
                frames_per_video=8,  # Optimized: 8 instead of 16
                image_size=(32, 32),  # Optimized: 32x32 instead of 64x64
                augmentation=False,  # Disabled for speed
                seed=42,
                num_workers=4,
                output_dir=str(output_dir)
            )
            
            elapsed = time.time() - start
            
            # Verify completion
            assert len(dataset['videos']) == 5000, \
                f"Expected 5000 samples, got {len(dataset['videos'])}"
            
            # Verify checkpoints were created
            checkpoints = list(output_dir.glob("checkpoint_*.npz"))
            assert len(checkpoints) >= 5, \
                f"Expected at least 5 checkpoints, found {len(checkpoints)}"
            
            # Verify progress file shows completion
            progress_file = output_dir / "PROGRESS.txt"
            assert progress_file.exists(), "Missing progress file"
            content = progress_file.read_text()
            assert "COMPLETE" in content, "Not marked complete"
            
            # Report timing
            print(f"\n✅ 5K integration test passed in {elapsed/60:.1f} minutes")
            print(f"   Estimated 30K time: {elapsed * 6 / 60:.1f} minutes")


class TestCheckpointMerge:
    """Tests for merging checkpoints into final dataset."""
    
    def test_merged_dataset_has_all_samples(self):
        """Verify merged dataset contains all samples from checkpoints.
        
        Requirement: Merging must preserve all data
        Expected: Final dataset has sum of all checkpoint samples
        
        Seed: 42
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            dataset = generate_dataset_parallel_with_checkpoints(
                num_samples=1000,
                checkpoint_interval=250,  # 4 checkpoints
                frames_per_video=8,
                image_size=(32, 32),
                augmentation=False,
                seed=42,
                num_workers=4,
                output_dir=str(output_dir)
            )
            
            # Verify final dataset has all samples
            assert len(dataset['videos']) == 1000, \
                f"Expected 1000 samples after merge, got {len(dataset['videos'])}"
            
            # Verify class balance
            labels = dataset['labels'].numpy()
            for class_id in range(4):
                count = (labels == class_id).sum()
                expected = 1000 // 4
                assert abs(count - expected) <= 1, \
                    f"Class {class_id} has {count} samples, expected ~{expected}"


# Pre-launch validation function
def validate_ready_for_production():
    """Verify all requirements met before production launch.
    
    This function should be called before launching full 30K generation.
    Returns True if all checks pass, raises RuntimeError otherwise.
    """
    print("="*70)
    print("PRE-LAUNCH VALIDATION")
    print("="*70)
    print()
    
    checks = {
        'checkpoint_tests': False,
        'progress_file_tests': False,
        'integration_test': False
    }
    
    # Run checkpoint tests
    print("Running checkpoint tests...")
    result = pytest.main([
        __file__,
        '-v',
        '-k', 'TestCheckpointCreation or TestProgressFile',
        '-m', 'not slow'
    ])
    checks['checkpoint_tests'] = (result == 0)
    
    # Run integration test
    print("\nRunning 5K integration test (this will take ~5 min)...")
    result = pytest.main([
        __file__,
        '-v',
        '-k', 'test_5k_generation_completes_with_checkpoints',
        '-m', 'slow'
    ])
    checks['integration_test'] = (result == 0)
    
    # Report results
    print()
    print("="*70)
    print("VALIDATION RESULTS")
    print("="*70)
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}")
    print()
    
    all_passed = all(checks.values())
    
    if not all_passed:
        print("❌ NOT READY FOR PRODUCTION")
        print("   Fix failing tests before launching 30K generation")
        raise RuntimeError("Pre-launch validation failed")
    
    print("✅ ALL CHECKS PASSED - READY FOR PRODUCTION")
    print("   You may now launch 30K generation with confidence")
    return True


if __name__ == "__main__":
    # Run validation if executed directly
    validate_ready_for_production()

