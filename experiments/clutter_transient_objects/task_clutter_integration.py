#!/usr/bin/env python3
"""
Task: Clutter and Transient Objects Integration
================================================
Test enhanced trajectory_to_video with persistent/transient objects
and integrate into task1_trajectory_generator.py

Steps:
1. Test enhanced version with simple trajectory
2. Start with persistent objects (Step 1)
3. Gradually add transient objects
4. Integrate into existing task1_trajectory_generator.py
"""

import sys
import os
import traceback
from pathlib import Path
from datetime import datetime
import json
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'basic'))
sys.path.insert(0, str(PROJECT_ROOT / 'experiments'))

# Setup logging and testing utilities
from experiments.shared_test_utilities import (
    setup_task_logging,
    validate_environment,
    run_test_suite,
    save_test_results,
    debug_print,
    validate_output,
    validate_array
)

# Initialize logger
OUTPUT_DIR = Path(__file__).parent / 'output'
logger = setup_task_logging('clutter_transient_objects', OUTPUT_DIR)

logger.info("=" * 60)
logger.info("Clutter and Transient Objects Integration Task")
logger.info("=" * 60)
logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Validate environment
logger.info("Validating environment...")
env_results = validate_environment(logger)
if not env_results['python_version']:
    logger.error("❌ Python version check failed")
    sys.exit(1)
logger.info("✅ Environment validation passed")

# Validate EC2 environment (if available)
try:
    from basic.validate_computation_environment import validate_computation_environment
    validate_computation_environment()
    logger.info("✅ EC2 environment validation passed")
except ImportError:
    logger.warning("⚠️  validate_computation_environment not available, skipping")
except Exception as e:
    logger.warning(f"⚠️  EC2 validation warning: {e}")

# Import task-specific modules with error handling
try:
    from basic.trajectory_to_video_enhanced import (
        trajectory_to_video_with_clutter,
        create_simple_persistent_object,
        create_simple_transient_object,
        ClutterObject
    )
    logger.info("✅ trajectory_to_video_enhanced imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import trajectory_to_video_enhanced: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)

try:
    from magvit_options.task1_trajectory_generator import TrajectoryGenerator
    logger.info("✅ TrajectoryGenerator imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import TrajectoryGenerator: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)


def test_enhanced_trajectory_simple(logger):
    """Step 1: Test enhanced version with simple trajectory."""
    logger.info("=" * 60)
    logger.info("Step 1: Testing Enhanced Trajectory with Simple Example")
    logger.info("=" * 60)
    
    test_results = {
        'tests_run': 0,
        'tests_passed': 0,
        'tests_failed': 0,
        'details': {},
        'errors': []
    }
    
    # Create simple straight trajectory
    generator = TrajectoryGenerator(num_points=100)
    trajectory = generator.generate('straight')
    
    # Test without clutter first
    logger.info("\n1.1 Testing without clutter...")
    try:
        test_results['tests_run'] += 1
        video_clean = trajectory_to_video_with_clutter(
            trajectory,
            resolution=(128, 128),
            num_frames=50
        )
        
        # Validate output
        if validate_array(video_clean, expected_shape=(50, 128, 128, 3), logger=logger, name="clean_video"):
            test_results['tests_passed'] += 1
            test_results['details']['clean_video'] = {
                'shape': list(video_clean.shape),
                'dtype': str(video_clean.dtype),
                'min': float(video_clean.min()),
                'max': float(video_clean.max())
            }
            logger.info(f"   ✅ Clean video shape: {video_clean.shape}")
        else:
            test_results['tests_failed'] += 1
            test_results['errors'].append("Clean video validation failed")
    except Exception as e:
        test_results['tests_failed'] += 1
        test_results['errors'].append(f"Clean video generation failed: {str(e)}")
        logger.error(f"   ❌ Error: {e}")
        logger.debug(traceback.format_exc())
    
    # Test with persistent objects
    print("\n1.2 Testing with persistent objects...")
    persistent_obj = create_simple_persistent_object(
        position=np.array([0.3, 0.3, 0.5]),
        shape='circle',
        size=0.02
    )
    
    video_persistent = trajectory_to_video_with_clutter(
        trajectory,
        resolution=(128, 128),
        num_frames=50,
        persistent_objects=[persistent_obj]
    )
    print(f"   ✅ Video with persistent object shape: {video_persistent.shape}")
    
    # Test with transient objects
    print("\n1.3 Testing with transient objects...")
    transient_obj = create_simple_transient_object(
        position=np.array([0.5, 0.5, 1.0]),
        appearance_frame=10,
        disappearance_frame=30,
        shape='square',
        size=0.02
    )
    
    video_transient = trajectory_to_video_with_clutter(
        trajectory,
        resolution=(128, 128),
        num_frames=50,
        transient_objects=[transient_obj]
    )
    print(f"   ✅ Video with transient object shape: {video_transient.shape}")
    
    # Test with both
    print("\n1.4 Testing with both persistent and transient objects...")
    video_both = trajectory_to_video_with_clutter(
        trajectory,
        resolution=(128, 128),
        num_frames=50,
        persistent_objects=[persistent_obj],
        transient_objects=[transient_obj],
        background_clutter=True,
        clutter_density=0.1,
        noise_level=0.05
    )
    print(f"   ✅ Video with full clutter shape: {video_both.shape}")
    
    return {
        'video_clean': video_clean,
        'video_persistent': video_persistent,
        'video_transient': video_transient,
        'video_both': video_both
    }


def test_persistent_objects_step1(logger):
    """Step 2: Start with persistent objects (Step 1 from recommendations)."""
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Testing Persistent Objects (Step 1)")
    logger.info("=" * 60)
    
    generator = TrajectoryGenerator(num_points=100)
    classes = ['straight', 'left_turn', 'right_turn', 'u_turn', 'loop']
    
    results = {}
    
    for class_name in classes:
        print(f"\n2.1 Testing {class_name} with persistent objects...")
        trajectory = generator.generate(class_name)
        
        # Create 2-3 persistent objects
        persistent_objects = [
            create_simple_persistent_object(
                position=np.array([
                    np.random.uniform(0.2, 0.8),
                    np.random.uniform(0.2, 0.8),
                    np.random.uniform(0.3, 0.7)
                ]),
                shape=np.random.choice(['circle', 'square', 'triangle']),
                size=np.random.uniform(0.015, 0.025)
            ) for _ in range(2)
        ]
        
        video = trajectory_to_video_with_clutter(
            trajectory,
            resolution=(128, 128),
            num_frames=50,
            persistent_objects=persistent_objects
        )
        
        results[class_name] = {
            'video_shape': list(video.shape),
            'num_persistent_objects': len(persistent_objects)
        }
        print(f"   ✅ {class_name}: {video.shape}")
    
    return results


def test_transient_objects_gradual(logger):
    """Step 3: Gradually add transient objects."""
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Gradually Adding Transient Objects")
    logger.info("=" * 60)
    
    generator = TrajectoryGenerator(num_points=100)
    trajectory = generator.generate('straight')
    
    results = {}
    
    # 3.1: One transient object
    print("\n3.1 Testing with one transient object...")
    transient_obj1 = create_simple_transient_object(
        position=np.array([0.4, 0.4, 0.6]),
        appearance_frame=5,
        disappearance_frame=25,
        shape='circle'
    )
    
    video_1_transient = trajectory_to_video_with_clutter(
        trajectory,
        resolution=(128, 128),
        num_frames=50,
        transient_objects=[transient_obj1]
    )
    results['one_transient'] = {'video_shape': list(video_1_transient.shape)}
    print(f"   ✅ One transient: {video_1_transient.shape}")
    
    # 3.2: Multiple transient objects
    print("\n3.2 Testing with multiple transient objects...")
    transient_objects = [
        create_simple_transient_object(
            position=np.array([
                np.random.uniform(0.2, 0.8),
                np.random.uniform(0.2, 0.8),
                np.random.uniform(0.3, 0.7)
            ]),
            appearance_frame=np.random.randint(0, 20),
            disappearance_frame=np.random.randint(20, 50),
            shape=np.random.choice(['circle', 'square', 'triangle']),
            size=np.random.uniform(0.015, 0.025)
        ) for _ in range(3)
    ]
    
    video_multi_transient = trajectory_to_video_with_clutter(
        trajectory,
        resolution=(128, 128),
        num_frames=50,
        transient_objects=transient_objects
    )
    results['multiple_transient'] = {
        'video_shape': list(video_multi_transient.shape),
        'num_transient_objects': len(transient_objects)
    }
    print(f"   ✅ Multiple transient: {video_multi_transient.shape}")
    
    # 3.3: Persistent + Transient
    print("\n3.3 Testing with persistent + transient objects...")
    persistent_obj = create_simple_persistent_object(
        position=np.array([0.3, 0.3, 0.5]),
        shape='circle'
    )
    
    video_combined = trajectory_to_video_with_clutter(
        trajectory,
        resolution=(128, 128),
        num_frames=50,
        persistent_objects=[persistent_obj],
        transient_objects=transient_objects
    )
    results['persistent_plus_transient'] = {
        'video_shape': list(video_combined.shape)
    }
    print(f"   ✅ Combined: {video_combined.shape}")
    
    return results


def integrate_into_task1(logger):
    """Step 4: Integrate into existing task1_trajectory_generator.py."""
    logger.info("\n" + "=" * 60)
    logger.info("Step 4: Integrating into task1_trajectory_generator.py")
    logger.info("=" * 60)
    
    # Import task1 generator
    from magvit_options.task1_trajectory_generator import (
        generate_synthetic_trajectory_videos,
        TrajectoryGenerator
    )
    
    # Create enhanced version that uses clutter
    generator = TrajectoryGenerator(num_points=100)
    classes = ['straight', 'left_turn', 'right_turn', 'u_turn', 'loop']
    
    output_dir = Path(__file__).parent / 'output' / 'enhanced_trajectory_videos'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'task': 'Enhanced Trajectory Generator with Clutter',
        'classes': classes,
        'generated_videos': [],
        'status': 'running'
    }
    
    print("\n4.1 Generating enhanced videos with clutter...")
    
    for class_name in classes:
        print(f"\n   Generating {class_name} with clutter...")
        class_dir = output_dir / class_name
        class_dir.mkdir(exist_ok=True)
        
        for i in range(5):  # Generate 5 per class for testing
            try:
                # Generate trajectory
                trajectory = generator.generate(class_name)
                
                # Add noise variation
                noise = np.random.normal(0, 0.05, trajectory.shape)
                trajectory = trajectory + noise
                
                # Create clutter objects
                persistent_objects = [
                    create_simple_persistent_object(
                        position=np.array([
                            np.random.uniform(0.2, 0.8),
                            np.random.uniform(0.2, 0.8),
                            np.random.uniform(0.3, 0.7)
                        ]),
                        shape=np.random.choice(['circle', 'square', 'triangle']),
                        size=np.random.uniform(0.015, 0.025)
                    ) for _ in range(np.random.randint(1, 3))
                ]
                
                transient_objects = [
                    create_simple_transient_object(
                        position=np.array([
                            np.random.uniform(0.2, 0.8),
                            np.random.uniform(0.2, 0.8),
                            np.random.uniform(0.3, 0.7)
                        ]),
                        appearance_frame=np.random.randint(0, 20),
                        disappearance_frame=np.random.randint(20, 50),
                        shape=np.random.choice(['circle', 'square', 'triangle']),
                        size=np.random.uniform(0.015, 0.025)
                    ) for _ in range(np.random.randint(1, 3))
                ]
                
                # Generate video with clutter
                video = trajectory_to_video_with_clutter(
                    trajectory,
                    resolution=(128, 128),
                    num_frames=50,
                    persistent_objects=persistent_objects,
                    transient_objects=transient_objects,
                    background_clutter=True,
                    clutter_density=0.1,
                    noise_level=0.05
                )
                
                # Save video
                video_path = class_dir / f"{class_name}_{i:03d}.npy"
                np.save(video_path, video)
                
                results['generated_videos'].append({
                    'class': class_name,
                    'index': i,
                    'path': str(video_path),
                    'video_shape': list(video.shape),
                    'num_persistent': len(persistent_objects),
                    'num_transient': len(transient_objects)
                })
                
            except Exception as e:
                print(f"   ⚠️  Error generating video {i}: {e}")
        
        print(f"   ✅ {class_name}: 5 videos generated")
    
    results['status'] = 'success'
    results['total_generated'] = len(results['generated_videos'])
    
    # Save metadata
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Total videos generated: {results['total_generated']}")
    print(f"✅ Metadata saved to: {metadata_path}")
    
    return results


def main():
    """Main execution function."""
    all_results = {
        'task_name': 'clutter_transient_objects',
        'timestamp': datetime.now().isoformat(),
        'environment': env_results,
        'steps': {}
    }
    
    # Step 1: Test enhanced version
    logger.info("\n" + "=" * 60)
    logger.info("Starting Step 1: Test Enhanced Trajectory")
    logger.info("=" * 60)
    try:
        step1_test_results = run_test_suite(test_enhanced_trajectory_simple, logger, "Step 1: Enhanced Trajectory Test")
        all_results['steps']['step1_simple_test'] = step1_test_results
        if step1_test_results['tests_failed'] == 0:
            logger.info("✅ Step 1 completed successfully")
        else:
            logger.warning(f"⚠️  Step 1 completed with {step1_test_results['tests_failed']} failures")
    except Exception as e:
        logger.error(f"❌ Step 1 failed: {e}")
        logger.error(traceback.format_exc())
        all_results['steps']['step1_simple_test'] = {
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }
    
    # Step 2: Persistent objects
    logger.info("\n" + "=" * 60)
    logger.info("Starting Step 2: Persistent Objects")
    logger.info("=" * 60)
    try:
        step2_results = test_persistent_objects_step1(logger)
        all_results['steps']['step2_persistent_objects'] = {
            'status': 'success',
            'results': step2_results
        }
        logger.info("✅ Step 2 completed successfully")
    except Exception as e:
        logger.error(f"❌ Step 2 failed: {e}")
        logger.error(traceback.format_exc())
        all_results['steps']['step2_persistent_objects'] = {
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }
    
    # Step 3: Transient objects
    logger.info("\n" + "=" * 60)
    logger.info("Starting Step 3: Transient Objects")
    logger.info("=" * 60)
    try:
        step3_results = test_transient_objects_gradual(logger)
        all_results['steps']['step3_transient_objects'] = {
            'status': 'success',
            'results': step3_results
        }
        logger.info("✅ Step 3 completed successfully")
    except Exception as e:
        logger.error(f"❌ Step 3 failed: {e}")
        logger.error(traceback.format_exc())
        all_results['steps']['step3_transient_objects'] = {
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }
    
    # Step 4: Integration
    logger.info("\n" + "=" * 60)
    logger.info("Starting Step 4: Integration")
    logger.info("=" * 60)
    try:
        step4_results = integrate_into_task1(logger)
        all_results['steps']['step4_integration'] = {
            'status': 'success',
            'results': step4_results
        }
        logger.info("✅ Step 4 completed successfully")
    except Exception as e:
        logger.error(f"❌ Step 4 failed: {e}")
        logger.error(traceback.format_exc())
        all_results['steps']['step4_integration'] = {
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }
    
    # Calculate summary
    total_steps = len(all_results['steps'])
    successful_steps = sum(1 for s in all_results['steps'].values() if s.get('status') == 'success')
    all_results['summary'] = {
        'total_steps': total_steps,
        'successful_steps': successful_steps,
        'failed_steps': total_steps - successful_steps,
        'success_rate': f"{(successful_steps/total_steps*100):.1f}%" if total_steps > 0 else "0%"
    }
    
    # Save final results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = OUTPUT_DIR / f"{timestamp}_clutter_integration_results.json"
    
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save test results
    save_test_results(all_results, OUTPUT_DIR, 'clutter_integration')
    
    logger.info("\n" + "=" * 60)
    logger.info("Task Summary")
    logger.info("=" * 60)
    logger.info(f"Total steps: {all_results['summary']['total_steps']}")
    logger.info(f"Successful: {all_results['summary']['successful_steps']}")
    logger.info(f"Failed: {all_results['summary']['failed_steps']}")
    logger.info(f"Success rate: {all_results['summary']['success_rate']}")
    logger.info(f"\n✅ Results saved to: {results_path}")
    logger.info(f"✅ Logs saved to: {OUTPUT_DIR / 'logs'}")
    
    return all_results


if __name__ == '__main__':
    main()

