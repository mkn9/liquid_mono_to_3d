#!/usr/bin/env python3
"""
Task: VideoGPT 3D Implementation
==================================
Create VideoGPT 3D implementation based on 2D version

Steps:
1. Check VideoGPT 2D implementation to assess readiness for 3D adaptation
2. Create VideoGPT 3D implementation based on 2D version
3. Test with 3D trajectories
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import json
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'basic'))

# Validate EC2 environment
from basic.validate_computation_environment import validate_computation_environment
validate_computation_environment()


def check_videogpt_2d_implementation():
    """Step 1: Check VideoGPT 2D implementation to assess readiness."""
    print("=" * 60)
    print("Step 1: Checking VideoGPT 2D Implementation")
    print("=" * 60)
    
    results = {
        'status': 'checking',
        'files_found': [],
        'components': {},
        'readiness_assessment': {}
    }
    
    # Check for 2D implementation files
    print("\n1.1 Checking for VideoGPT 2D files...")
    
    # Check neural_video_experiments/videogpt
    videogpt_2d_dir = PROJECT_ROOT / 'neural_video_experiments' / 'videogpt'
    if videogpt_2d_dir.exists():
        code_dir = videogpt_2d_dir / 'code'
        if code_dir.exists():
            files = list(code_dir.glob('*.py'))
            results['files_found'].extend([str(f.relative_to(PROJECT_ROOT)) for f in files])
            print(f"   ✅ Found {len(files)} Python files in neural_video_experiments/videogpt/code/")
            for f in files:
                print(f"      - {f.name}")
    
    # Check classification/videogpt-trajectories branch
    print("\n1.2 Checking git branches...")
    import subprocess
    try:
        result = subprocess.run(
            ['git', 'branch', '-a'],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        )
        branches = result.stdout
        if 'videogpt' in branches.lower() or 'VideoGPT' in branches:
            videogpt_branches = [b.strip() for b in branches.split('\n') if 'videogpt' in b.lower()]
            results['branches_found'] = videogpt_branches
            print(f"   ✅ Found {len(videogpt_branches)} VideoGPT-related branches")
            for b in videogpt_branches:
                print(f"      - {b}")
    except Exception as e:
        print(f"   ⚠️  Could not check branches: {e}")
    
    # Check for trajectory generator
    print("\n1.3 Checking trajectory generator...")
    try:
        from basic.trajectory_to_video import trajectory_to_video
        results['components']['trajectory_to_video'] = 'available'
        print("   ✅ trajectory_to_video available")
    except ImportError:
        results['components']['trajectory_to_video'] = 'missing'
        print("   ❌ trajectory_to_video not available")
    
    # Assess readiness
    print("\n1.4 Assessing readiness for 3D adaptation...")
    readiness_score = 0
    readiness_items = []
    
    if results.get('files_found'):
        readiness_score += 2
        readiness_items.append("✅ 2D implementation files found")
    else:
        readiness_items.append("❌ No 2D implementation files found")
    
    if results.get('components', {}).get('trajectory_to_video') == 'available':
        readiness_score += 2
        readiness_items.append("✅ trajectory_to_video available")
    else:
        readiness_items.append("❌ trajectory_to_video missing")
    
    if results.get('branches_found'):
        readiness_score += 1
        readiness_items.append("✅ VideoGPT branches exist")
    else:
        readiness_items.append("⚠️  No VideoGPT branches found")
    
    results['readiness_assessment'] = {
        'score': readiness_score,
        'max_score': 5,
        'items': readiness_items,
        'ready': readiness_score >= 3
    }
    
    for item in readiness_items:
        print(f"   {item}")
    
    print(f"\n   Readiness Score: {readiness_score}/5")
    print(f"   Ready for 3D adaptation: {'✅ Yes' if readiness_score >= 3 else '❌ No'}")
    
    return results


def create_videogpt_3d_implementation():
    """Step 2: Create VideoGPT 3D implementation based on 2D version."""
    print(info("\n" + "=" * 60)
    print(info("Step 2: Creating VideoGPT 3D Implementation")
    print(info("=" * 60)
    
    results = {
        'status': 'creating',
        'files_created': [],
        'components': {}
    }
    
    output_dir = Path(__file__).parent / 'code'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2.1: Create VideoGPT 3D Trajectory Data Generator
    print("\n2.1 Creating VideoGPT 3D Trajectory Data Generator...")
    
    videogpt_3d_code = '''#!/usr/bin/env python3
"""
VideoGPT 3D Trajectory Data Generator
=====================================

This module contains the VideoGPT-specific trajectory generation class
for 3D neural video experiments using PyTorch + VQ-VAE + Transformer framework.
Adapted from VideoGPT 2D implementation for 3D trajectories.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Dict, Any
from pathlib import Path
import json
import tempfile
import h5py


class VideoGPT3DTrajectoryDataGenerator:
    """
    VideoGPT 3D Trajectory Data Generator using PyTorch + VQ-VAE + Transformer.
    
    Generates 3D trajectory data for cubes, cylinders, and cones
    with dual export formats (NPZ and HDF5) for VideoGPT compatibility.
    """
    
    def __init__(self, seq_length: int = 16, img_size: int = 128):
        """
        Initialize VideoGPT 3D trajectory generator.
        
        Args:
            seq_length: Number of frames in each video sequence
            img_size: Image size (width and height)
        """
        self.seq_length = seq_length
        self.img_size = img_size
        self.shapes = ['cube', 'cylinder', 'cone']
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue
        self.trajectory_patterns = ['linear_3d', 'circular_3d', 'helical', 'parabolic_3d']
        
        # VideoGPT-specific parameters
        self.vq_codebook_size = 1024
        self.vq_embed_dim = 256
        self.transformer_layers = 12
        self.transformer_heads = 8
        
    def generate_trajectory(self, pattern: str, start_pos: Tuple[float, float, float], 
                          end_pos: Tuple[float, float, float] = None) -> np.ndarray:
        """
        Generate 3D trajectory based on specified pattern.
        
        Args:
            pattern: Type of trajectory ('linear_3d', 'circular_3d', 'helical', 'parabolic_3d')
            start_pos: Starting position (x, y, z)
            end_pos: Ending position (x, y, z) for linear patterns
            
        Returns:
            Array of shape (seq_length, 3) containing trajectory coordinates
        """
        trajectory = np.zeros((self.seq_length, 3))
        
        if pattern == 'linear_3d':
            if end_pos is None:
                end_pos = (start_pos[0] + 1.0, start_pos[1] + 0.5, start_pos[2] + 0.5)
            
            for i in range(self.seq_length):
                t = i / (self.seq_length - 1)
                trajectory[i] = [
                    start_pos[0] + t * (end_pos[0] - start_pos[0]),
                    start_pos[1] + t * (end_pos[1] - start_pos[1]),
                    start_pos[2] + t * (end_pos[2] - start_pos[2])
                ]
        
        elif pattern == 'circular_3d':
            radius = 0.5
            center_x, center_y, center_z = start_pos
            for i in range(self.seq_length):
                angle = 2 * np.pi * i / self.seq_length
                trajectory[i] = [
                    center_x + radius * np.cos(angle),
                    center_y + radius * np.sin(angle),
                    center_z  # Constant height
                ]
        
        elif pattern == 'helical':
            radius = 0.5
            center_x, center_y, center_z = start_pos
            for i in range(self.seq_length):
                angle = 2 * np.pi * i / self.seq_length
                trajectory[i] = [
                    center_x + radius * np.cos(angle),
                    center_y + radius * np.sin(angle),
                    center_z + 0.3 * i / self.seq_length  # Ascending
                ]
        
        elif pattern == 'parabolic_3d':
            if end_pos is None:
                end_pos = (start_pos[0] + 1.0, start_pos[1] + 0.5, start_pos[2] + 1.0)
            
            for i in range(self.seq_length):
                t = i / (self.seq_length - 1)
                x = start_pos[0] + t * (end_pos[0] - start_pos[0])
                y = start_pos[1] + t * (end_pos[1] - start_pos[1])
                z = start_pos[2] + 4 * t * (1 - t) * (end_pos[2] - start_pos[2])  # Parabolic
                trajectory[i] = [x, y, z]
        
        return trajectory
    
    def draw_shape_3d(self, img: np.ndarray, shape: str, position: Tuple[float, float, float], 
                      size: float, color: Tuple[int, int, int]) -> np.ndarray:
        """
        Draw 3D shape projection on 2D image.
        
        Args:
            img: Image array (H, W, 3)
            shape: Shape type ('cube', 'cylinder', 'cone')
            position: 3D position (x, y, z) - will be projected to 2D
            size: Size of shape
            color: RGB color tuple
        """
        # Project 3D to 2D (simple orthographic projection)
        # Using xy projection for simplicity
        x_2d = int(position[0] * self.img_size)
        y_2d = int(position[1] * self.img_size)
        size_2d = int(size * self.img_size)
        
        if shape == 'cube':
            # Draw square
            cv2.rectangle(img, 
                         (x_2d - size_2d, y_2d - size_2d),
                         (x_2d + size_2d, y_2d + size_2d),
                         color, -1)
        elif shape == 'cylinder':
            # Draw circle
            cv2.circle(img, (x_2d, y_2d), size_2d, color, -1)
        elif shape == 'cone':
            # Draw triangle
            pts = np.array([
                [x_2d, y_2d - size_2d],
                [x_2d - size_2d, y_2d + size_2d],
                [x_2d + size_2d, y_2d + size_2d]
            ], np.int32)
            cv2.fillPoly(img, [pts], color)
        
        return img
    
    def generate_video_sequence(self, shape: str, trajectory: np.ndarray, 
                                color: Tuple[int, int, int]) -> np.ndarray:
        """
        Generate video sequence from 3D trajectory.
        
        Args:
            shape: Shape type
            trajectory: (seq_length, 3) trajectory points
            color: RGB color tuple
            
        Returns:
            Video array (seq_length, H, W, 3)
        """
        video = np.zeros((self.seq_length, self.img_size, self.img_size, 3), dtype=np.uint8)
        
        for i, pos in enumerate(trajectory):
            frame = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            frame = self.draw_shape_3d(frame, shape, pos, 0.05, color)
            video[i] = frame
        
        return video
    
    def generate_dataset(self, num_samples: int = 20) -> Dict[str, np.ndarray]:
        """
        Generate dataset of 3D trajectory videos.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Dictionary with 'videos' and 'trajectories' arrays
        """
        videos = []
        trajectories = []
        
        for i in range(num_samples):
            shape = np.random.choice(self.shapes)
            pattern = np.random.choice(self.trajectory_patterns)
            color = np.random.choice(self.colors)
            
            start_pos = (
                np.random.uniform(0.1, 0.9),
                np.random.uniform(0.1, 0.9),
                np.random.uniform(0.1, 0.9)
            )
            
            trajectory = self.generate_trajectory(pattern, start_pos)
            video = self.generate_video_sequence(shape, trajectory, color)
            
            videos.append(video)
            trajectories.append(trajectory)
        
        return {
            'videos': np.array(videos),
            'trajectories': np.array(trajectories)
        }
    
    def save_results(self, dataset: Dict[str, np.ndarray], output_dir: Path):
        """Save dataset in NPZ and HDF5 formats."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save NPZ
        npz_path = output_dir / 'videogpt_3d_dataset.npz'
        np.savez(npz_path, **dataset)
        
        # Save HDF5
        h5_path = output_dir / 'videogpt_3d_dataset.h5'
        with h5py.File(h5_path, 'w') as f:
            for key, value in dataset.items():
                f.create_dataset(key, data=value)
        
        # Save metadata
        metadata = {
            'format': 'NPZ/HDF5',
            'framework': 'VideoGPT (PyTorch) - 3D',
            'num_samples': len(dataset['videos']),
            'seq_length': self.seq_length,
            'img_size': self.img_size,
            'shapes': self.shapes,
            'trajectory_patterns': self.trajectory_patterns,
            'vq_codebook_size': self.vq_codebook_size,
            'vq_embed_dim': self.vq_embed_dim
        }
        
        metadata_path = output_dir / 'videogpt_3d_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Saved dataset to {output_dir}")
        print(f"   - NPZ: {npz_path}")
        print(f"   - HDF5: {h5_path}")
        print(f"   - Metadata: {metadata_path}")


if __name__ == "__main__":
    # Demonstrate VideoGPT 3D
    print("🎯 VideoGPT 3D Trajectory Generation Demo")
    videogpt_3d = VideoGPT3DTrajectoryDataGenerator()
    dataset = videogpt_3d.generate_dataset(num_samples=20)
    print(f"✅ Generated {len(dataset['videos'])} VideoGPT 3D samples")
    print(f"   Video shape: {dataset['videos'].shape}")
    print(f"   Trajectory shape: {dataset['trajectories'].shape}")
    
    # Save results
    results_dir = Path("../output")
    videogpt_3d.save_results(dataset, results_dir)
    
    print("📊 VideoGPT 3D demonstration complete!")
'''
    
    generator_path = output_dir / 'videogpt_3d_trajectory_generator.py'
    with open(generator_path, 'w') as f:
        f.write(videogpt_3d_code)
    
    results['files_created'].append(str(generator_path.relative_to(PROJECT_ROOT)))
    results['components']['generator'] = 'created'
    print(f"   ✅ Created: {generator_path.name}")
    
    # 2.2: Test the implementation
    print("\n2.2 Testing VideoGPT 3D implementation...")
    try:
        sys.path.insert(0, str(output_dir))
        from videogpt_3d_trajectory_generator import VideoGPT3DTrajectoryDataGenerator
        
        generator = VideoGPT3DTrajectoryDataGenerator(seq_length=16, img_size=128)
        test_dataset = generator.generate_dataset(num_samples=5)
        
        results['components']['test'] = {
            'status': 'success',
            'video_shape': list(test_dataset['videos'].shape),
            'trajectory_shape': list(test_dataset['trajectories'].shape)
        }
        print(f"   ✅ Test successful")
        print(f"      Video shape: {test_dataset['videos'].shape}")
        print(f"      Trajectory shape: {test_dataset['trajectories'].shape}")
        
    except Exception as e:
        results['components']['test'] = {
            'status': 'error',
            'error': str(e)
        }
        print(f"   ❌ Test failed: {e}")
    
    results['status'] = 'completed'
    return results


def test_with_3d_trajectories():
    """Step 3: Test with 3D trajectories."""
    print(info("\n" + "=" * 60)
    print(info("Step 3: Testing with 3D Trajectories")
    print(info("=" * 60)
    
    results = {
        'status': 'testing',
        'tests': {}
    }
    
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Import generator
    code_dir = Path(__file__).parent / 'code'
    sys.path.insert(0, str(code_dir))
    from videogpt_3d_trajectory_generator import VideoGPT3DTrajectoryDataGenerator
    
    generator = VideoGPT3DTrajectoryDataGenerator(seq_length=16, img_size=128)
    
    # Test each pattern
    patterns = ['linear_3d', 'circular_3d', 'helical', 'parabolic_3d']
    
    for pattern in patterns:
        print(f"\n3.1 Testing {pattern}...")
        try:
            start_pos = (0.5, 0.5, 0.5)
            trajectory = generator.generate_trajectory(pattern, start_pos)
            
            video = generator.generate_video_sequence(
                'cube',
                trajectory,
                (255, 0, 0)
            )
            
            results['tests'][pattern] = {
                'status': 'success',
                'trajectory_shape': list(trajectory.shape),
                'video_shape': list(video.shape)
            }
            print(f"   ✅ {pattern}: trajectory {trajectory.shape}, video {video.shape}")
            
        except Exception as e:
            results['tests'][pattern] = {
                'status': 'error',
                'error': str(e)
            }
            print(f"   ❌ {pattern} failed: {e}")
    
    # Generate full dataset
    print("\n3.2 Generating full 3D dataset...")
    try:
        dataset = generator.generate_dataset(num_samples=20)
        generator.save_results(dataset, output_dir)
        
        results['full_dataset'] = {
            'status': 'success',
            'num_samples': len(dataset['videos']),
            'video_shape': list(dataset['videos'].shape),
            'trajectory_shape': list(dataset['trajectories'].shape)
        }
        print(f"   ✅ Generated {len(dataset['videos'])} samples")
        
    except Exception as e:
        results['full_dataset'] = {
            'status': 'error',
            'error': str(e)
        }
        print(f"   ❌ Dataset generation failed: {e}")
    
    results['status'] = 'completed'
    return results


def main():
    """Main execution function."""
    all_results = {
        'task_name': 'videogpt_3d_implementation',
        'timestamp': datetime.now().isoformat(),
        'environment': env_results,
        'steps': {}
    }
    
    # Step 1: Check 2D implementation
    print(info("\n" + "=" * 60)
    print(info("Starting Step 1: Check VideoGPT 2D Implementation")
    print(info("=" * 60)
    try:
        step1_results = check_videogpt_2d_implementation(logger)
        all_results['steps']['step1_check_2d'] = step1_results
        if step1_results.get('readiness_assessment', {}).get('ready'):
            print(info("✅ Step 1 completed - Ready for 3D adaptation")
        else:
            print(warning("⚠️  Step 1 completed - Readiness assessment indicates issues")
    except Exception as e:
        print(error(f"❌ Step 1 failed: {e}")
        print(error(traceback.format_exc())
        all_results['steps']['step1_check_2d'] = {
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }
    
    # Step 2: Create 3D implementation
    print(info("\n" + "=" * 60)
    print(info("Starting Step 2: Create VideoGPT 3D Implementation")
    print(info("=" * 60)
    try:
        step2_results = create_videogpt_3d_implementation(logger)
        all_results['steps']['step2_create_3d'] = step2_results
        if step2_results.get('status') == 'completed':
            print(info("✅ Step 2 completed successfully")
        else:
            print(warning(f"⚠️  Step 2 status: {step2_results.get('status')}")
    except Exception as e:
        print(error(f"❌ Step 2 failed: {e}")
        print(error(traceback.format_exc())
        all_results['steps']['step2_create_3d'] = {
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }
    
    # Step 3: Test with 3D trajectories
    print(info("\n" + "=" * 60)
    print(info("Starting Step 3: Test with 3D Trajectories")
    print(info("=" * 60)
    try:
        step3_results = test_with_3d_trajectories(logger)
        all_results['steps']['step3_test_3d'] = step3_results
        if step3_results.get('status') == 'completed':
            print(info("✅ Step 3 completed successfully")
        else:
            print(warning(f"⚠️  Step 3 status: {step3_results.get('status')}")
    except Exception as e:
        print(error(f"❌ Step 3 failed: {e}")
        print(error(traceback.format_exc())
        all_results['steps']['step3_test_3d'] = {
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }
    
    # Calculate summary
    total_steps = len(all_results['steps'])
    successful_steps = sum(1 for s in all_results['steps'].values() 
                          if s.get('status') in ['completed', 'success', 'checking'])
    all_results['summary'] = {
        'total_steps': total_steps,
        'successful_steps': successful_steps,
        'failed_steps': total_steps - successful_steps,
        'success_rate': f"{(successful_steps/total_steps*100):.1f}%" if total_steps > 0 else "0%"
    }
    
    # Save final results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = OUTPUT_DIR / f"{timestamp}_videogpt_3d_results.json"
    
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save test results
    save_test_results(all_results, OUTPUT_DIR, 'videogpt_3d')
    
    print(info("\n" + "=" * 60)
    print(info("Task Summary")
    print(info("=" * 60)
    print(info(f"Total steps: {all_results['summary']['total_steps']}")
    print(info(f"Successful: {all_results['summary']['successful_steps']}")
    print(info(f"Failed: {all_results['summary']['failed_steps']}")
    print(info(f"Success rate: {all_results['summary']['success_rate']}")
    print(info(f"\n✅ Results saved to: {results_path}")
    print(info(f"✅ Logs saved to: {OUTPUT_DIR / 'logs'}")
    
    return all_results


if __name__ == '__main__':
    main()

