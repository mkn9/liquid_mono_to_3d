#!/usr/bin/env python3
"""
Debug script to check data structure lengths and identify the IndexError cause
"""

import numpy as np
import sys
import os
sys.path.append('/home/ubuntu/mono_to_3d')

# Import the corrected functions
exec(open('/home/ubuntu/mono_to_3d/3d_tracker_8.ipynb').read())

def debug_data_structures():
    """Debug the data structures to find the IndexError cause."""
    print("="*60)
    print("DEBUGGING DATA STRUCTURE LENGTHS")
    print("="*60)
    
    try:
        # Set up cameras
        print("1. Setting up cameras...")
        P1, P2, cam1_pos, cam2_pos = set_up_cameras()
        print(f"   ✓ Camera positions: {cam1_pos}, {cam2_pos}")
        
        # Generate tracks
        print("\n2. Generating synthetic tracks...")
        sensor1_track, sensor2_track, original_3d = generate_synthetic_tracks()
        
        print(f"   sensor1_track type: {type(sensor1_track)}")
        print(f"   sensor1_track length: {len(sensor1_track) if hasattr(sensor1_track, '__len__') else 'No length'}")
        print(f"   sensor2_track type: {type(sensor2_track)}")
        print(f"   sensor2_track length: {len(sensor2_track) if hasattr(sensor2_track, '__len__') else 'No length'}")
        print(f"   original_3d type: {type(original_3d)}")
        print(f"   original_3d length: {len(original_3d) if hasattr(original_3d, '__len__') else 'No length'}")
        
        # Check contents
        print("\n3. Checking sensor1_track contents:")
        if hasattr(sensor1_track, '__iter__'):
            for i, item in enumerate(sensor1_track):
                print(f"   [{i}]: {item} (type: {type(item)})")
                if i >= 10:  # Limit output
                    print("   ... (truncated)")
                    break
        
        print("\n4. Checking sensor2_track contents:")
        if hasattr(sensor2_track, '__iter__'):
            for i, item in enumerate(sensor2_track):
                print(f"   [{i}]: {item} (type: {type(item)})")
                if i >= 10:  # Limit output
                    print("   ... (truncated)")
                    break
        
        print("\n5. Checking original_3d contents:")
        if hasattr(original_3d, '__iter__'):
            for i, item in enumerate(original_3d):
                print(f"   [{i}]: {item} (type: {type(item)})")
                if i >= 10:  # Limit output
                    print("   ... (truncated)")
                    break
        
        # Check for infinite/invalid values
        print("\n6. Checking for infinite/invalid values:")
        if hasattr(sensor1_track, '__iter__'):
            finite_count1 = 0
            for i, item in enumerate(sensor1_track):
                if hasattr(item, '__iter__') and len(item) >= 2:
                    if np.isfinite(item[0]) and np.isfinite(item[1]):
                        finite_count1 += 1
                    else:
                        print(f"   sensor1_track[{i}] has infinite values: {item}")
            print(f"   sensor1_track: {finite_count1} finite values out of {len(sensor1_track)}")
        
        if hasattr(sensor2_track, '__iter__'):
            finite_count2 = 0
            for i, item in enumerate(sensor2_track):
                if hasattr(item, '__iter__') and len(item) >= 2:
                    if np.isfinite(item[0]) and np.isfinite(item[1]):
                        finite_count2 += 1
                    else:
                        print(f"   sensor2_track[{i}] has infinite values: {item}")
            print(f"   sensor2_track: {finite_count2} finite values out of {len(sensor2_track)}")
        
        # Test the colors array
        print("\n7. Testing colors array:")
        colors = ['darkred', 'red', 'orange', 'yellow', 'lightcoral']
        print(f"   colors length: {len(colors)}")
        print(f"   colors: {colors}")
        
        # Simulate the problematic loop
        print("\n8. Simulating the problematic loop:")
        try:
            for i, (pixel, point_3d) in enumerate(zip(sensor1_track, original_3d)):
                print(f"   Loop iteration {i}: pixel={pixel}, point_3d={point_3d}")
                print(f"   Trying to access colors[{i}] = {colors[i] if i < len(colors) else 'INDEX OUT OF RANGE!'}")
                if i >= 10:  # Limit output
                    print("   ... (truncated)")
                    break
        except Exception as e:
            print(f"   ERROR in simulation: {e}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    debug_data_structures() 