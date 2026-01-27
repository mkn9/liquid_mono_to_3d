#!/usr/bin/env python3
"""
Test Scenarios for Integrated 3D Tracking with Persistence Filtering
===================================================================
Tests the complete system on realistic 3D tracking scenarios.

Scenarios:
1. Clean scene (2 persistent objects)
2. Cluttered scene (5 objects, 10 transients)
3. Noisy detections (3 objects, 20 false positives)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import json
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from experiments.track_persistence.realistic_track_generator import Realistic2DTrackGenerator
from experiments.track_persistence.integrated_3d_tracker import Integrated3DTracker, PersistenceFilter


def create_scenario_1_clean() -> Tuple[List[Dict], List[Dict]]:
    """
    Scenario 1: Clean scene with 2 persistent objects.
    
    Returns:
        camera1_tracks: Tracks from camera 1
        camera2_tracks: Tracks from camera 2
    """
    generator = Realistic2DTrackGenerator(seed=100)
    
    # Generate persistent tracks only
    camera1_tracks = []
    camera2_tracks = []
    
    for i in range(2):
        track1 = generator.generate_persistent_track(track_id=i, camera_id=0)
        track2 = generator.generate_persistent_track(track_id=i + 100, camera_id=1)
        
        camera1_tracks.append({
            'frames': track1.frames,
            'bboxes': track1.bboxes,
            'pixels': track1.pixels
        })
        
        camera2_tracks.append({
            'frames': track2.frames,
            'bboxes': track2.bboxes,
            'pixels': track2.pixels
        })
    
    return camera1_tracks, camera2_tracks


def create_scenario_2_cluttered() -> Tuple[List[Dict], List[Dict]]:
    """
    Scenario 2: Cluttered scene with 5 persistent objects and 10 transients.
    
    Returns:
        camera1_tracks: Tracks from camera 1
        camera2_tracks: Tracks from camera 2
    """
    generator = Realistic2DTrackGenerator(seed=200)
    
    camera1_tracks = []
    camera2_tracks = []
    
    track_id = 0
    
    # 5 persistent objects
    for i in range(5):
        track1 = generator.generate_persistent_track(track_id=track_id, camera_id=0)
        track2 = generator.generate_persistent_track(track_id=track_id + 1000, camera_id=1)
        
        camera1_tracks.append({
            'frames': track1.frames,
            'bboxes': track1.bboxes,
            'pixels': track1.pixels
        })
        
        camera2_tracks.append({
            'frames': track2.frames,
            'bboxes': track2.bboxes,
            'pixels': track2.pixels
        })
        
        track_id += 1
    
    # 10 transient/brief tracks
    for i in range(10):
        track1 = generator.generate_brief_track(track_id=track_id, camera_id=0)
        track2 = generator.generate_brief_track(track_id=track_id + 1000, camera_id=1)
        
        camera1_tracks.append({
            'frames': track1.frames,
            'bboxes': track1.bboxes,
            'pixels': track1.pixels
        })
        
        camera2_tracks.append({
            'frames': track2.frames,
            'bboxes': track2.bboxes,
            'pixels': track2.pixels
        })
        
        track_id += 1
    
    return camera1_tracks, camera2_tracks


def create_scenario_3_noisy() -> Tuple[List[Dict], List[Dict]]:
    """
    Scenario 3: Noisy scene with 3 persistent objects and 20 false positives.
    
    Returns:
        camera1_tracks: Tracks from camera 1
        camera2_tracks: Tracks from camera 2
    """
    generator = Realistic2DTrackGenerator(seed=300)
    
    camera1_tracks = []
    camera2_tracks = []
    
    track_id = 0
    
    # 3 persistent objects
    for i in range(3):
        track1 = generator.generate_persistent_track(track_id=track_id, camera_id=0)
        track2 = generator.generate_persistent_track(track_id=track_id + 1000, camera_id=1)
        
        camera1_tracks.append({
            'frames': track1.frames,
            'bboxes': track1.bboxes,
            'pixels': track1.pixels
        })
        
        camera2_tracks.append({
            'frames': track2.frames,
            'bboxes': track2.bboxes,
            'pixels': track2.pixels
        })
        
        track_id += 1
    
    # 20 noise tracks (false positives)
    for i in range(20):
        track1 = generator.generate_noise_track(track_id=track_id, camera_id=0)
        track2 = generator.generate_noise_track(track_id=track_id + 1000, camera_id=1)
        
        camera1_tracks.append({
            'frames': track1.frames,
            'bboxes': track1.bboxes,
            'pixels': track1.pixels
        })
        
        camera2_tracks.append({
            'frames': track2.frames,
            'bboxes': track2.bboxes,
            'pixels': track2.pixels
        })
        
        track_id += 1
    
    return camera1_tracks, camera2_tracks


def run_scenario(
    scenario_name: str,
    camera1_tracks: List[Dict],
    camera2_tracks: List[Dict],
    tracker_with_filter: Integrated3DTracker,
    tracker_without_filter: Integrated3DTracker,
    output_dir: Path
) -> Dict:
    """
    Run a test scenario with and without filtering.
    
    Args:
        scenario_name: Name of the scenario
        camera1_tracks: Tracks from camera 1
        camera2_tracks: Tracks from camera 2
        tracker_with_filter: Tracker with persistence filter enabled
        tracker_without_filter: Tracker without filtering
        output_dir: Directory to save results
        
    Returns:
        results: Dictionary with metrics and comparisons
    """
    print(f"\n{'='*60}")
    print(f"Running {scenario_name}")
    print(f"{'='*60}")
    
    # Run without filter
    print("\n[WITHOUT FILTER]")
    reconstructed_no_filter, _ = tracker_without_filter.process_scene(
        camera1_tracks,
        camera2_tracks,
        verbose=False
    )
    stats_no_filter = tracker_without_filter.get_statistics()
    
    # Run with filter
    print("\n[WITH FILTER]")
    reconstructed_with_filter, decisions = tracker_with_filter.process_scene(
        camera1_tracks,
        camera2_tracks,
        verbose=True
    )
    stats_with_filter = tracker_with_filter.get_statistics()
    
    # Compute metrics
    results = {
        'scenario': scenario_name,
        'total_tracks': len(camera1_tracks),
        'without_filter': {
            'kept_tracks': stats_no_filter['kept'],
            'triangulated_points': stats_no_filter['triangulated_points']
        },
        'with_filter': {
            'kept_tracks': stats_with_filter['kept'],
            'filtered_out': stats_with_filter['filtered_out'],
            'triangulated_points': stats_with_filter['triangulated_points'],
            'filter_rate': stats_with_filter.get('filter_rate', 0.0)
        },
        'improvement': {
            'tracks_reduced': stats_no_filter['kept'] - stats_with_filter['kept'],
            'points_reduced': stats_no_filter['triangulated_points'] - stats_with_filter['triangulated_points'],
            'reduction_rate': 1.0 - (stats_with_filter['kept'] / stats_no_filter['kept']) if stats_no_filter['kept'] > 0 else 0.0
        }
    }
    
    # Visualize results
    scenario_dir = output_dir / scenario_name.lower().replace(' ', '_')
    scenario_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize without filter
    tracker_without_filter.visualize_results(
        reconstructed_no_filter,
        str(scenario_dir / 'without_filter.png')
    )
    
    # Visualize with filter
    tracker_with_filter.visualize_results(
        reconstructed_with_filter,
        str(scenario_dir / 'with_filter.png')
    )
    
    # Save filter decisions
    with open(scenario_dir / 'filter_decisions.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        decisions_serializable = {}
        for track_id, decision in decisions.items():
            decisions_serializable[track_id] = {
                'is_persistent': decision['is_persistent'],
                'confidence': float(decision['confidence']),
                'explanation': decision['explanation']
            }
        json.dump(decisions_serializable, f, indent=2)
    
    # Save metrics
    with open(scenario_dir / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{scenario_name} Results:")
    print(f"  Total tracks: {results['total_tracks']}")
    print(f"  Without filter: {results['without_filter']['kept_tracks']} tracks, {results['without_filter']['triangulated_points']} 3D points")
    print(f"  With filter: {results['with_filter']['kept_tracks']} tracks, {results['with_filter']['triangulated_points']} 3D points")
    print(f"  Improvement: {results['improvement']['reduction_rate']*100:.1f}% track reduction")
    print(f"  Results saved to: {scenario_dir}")
    
    return results


def main():
    """Main entry point for testing scenarios."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test 3D tracking with persistence filtering')
    parser.add_argument('--model-checkpoint', type=str, required=True,
                        help='Path to trained persistence model')
    parser.add_argument('--magvit-checkpoint', type=str, required=True,
                        help='Path to pretrained MagVIT checkpoint')
    parser.add_argument('--output-dir', type=str, default='experiments/track_persistence/output/scenarios',
                        help='Output directory for results')
    parser.add_argument('--scenarios', nargs='+', default=['all'],
                        help='Scenarios to run: 1, 2, 3, or all')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize persistence filter
    print("Initializing persistence filter...")
    persistence_filter = PersistenceFilter(
        model_checkpoint=args.model_checkpoint,
        magvit_checkpoint=args.magvit_checkpoint
    )
    
    # Run scenarios
    scenarios_to_run = []
    if 'all' in args.scenarios:
        scenarios_to_run = ['1', '2', '3']
    else:
        scenarios_to_run = args.scenarios
    
    all_results = []
    
    for scenario_id in scenarios_to_run:
        if scenario_id == '1':
            camera1_tracks, camera2_tracks = create_scenario_1_clean()
            scenario_name = "Scenario 1: Clean Scene"
        elif scenario_id == '2':
            camera1_tracks, camera2_tracks = create_scenario_2_cluttered()
            scenario_name = "Scenario 2: Cluttered Scene"
        elif scenario_id == '3':
            camera1_tracks, camera2_tracks = create_scenario_3_noisy()
            scenario_name = "Scenario 3: Noisy Scene"
        else:
            print(f"Unknown scenario: {scenario_id}")
            continue
        
        # Initialize trackers (fresh for each scenario)
        tracker_with_filter = Integrated3DTracker(
            persistence_filter=persistence_filter,
            use_filter=True
        )
        
        tracker_without_filter = Integrated3DTracker(
            persistence_filter=None,
            use_filter=False
        )
        
        # Run scenario
        results = run_scenario(
            scenario_name,
            camera1_tracks,
            camera2_tracks,
            tracker_with_filter,
            tracker_without_filter,
            output_dir
        )
        
        all_results.append(results)
    
    # Save combined results
    with open(output_dir / 'all_scenarios_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    for results in all_results:
        print(f"\n{results['scenario']}:")
        print(f"  Tracks reduced: {results['improvement']['tracks_reduced']} ({results['improvement']['reduction_rate']*100:.1f}%)")
        print(f"  3D points reduced: {results['improvement']['points_reduced']}")
    
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()

