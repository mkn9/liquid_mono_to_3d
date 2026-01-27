#!/usr/bin/env python3

"""
Quick NeRF Test Script
This script creates a temporary modified version of run_nerf.py for quick testing
"""

import os
import sys
import shutil
import subprocess

def create_quick_test_nerf():
    """Create a modified version of run_nerf.py for quick testing"""
    
    # Read the original run_nerf.py
    with open('run_nerf.py', 'r') as f:
        content = f.read()
    
    # Replace the hardcoded N_iters with a smaller number
    modified_content = content.replace(
        'N_iters = 200000 + 1',
        'N_iters = 10 + 1  # Modified for quick testing'
    )
    
    # Also modify the print intervals for quicker feedback
    modified_content = modified_content.replace(
        'if i%args.i_print==0:',
        'if i%2==0:  # Print every 2 iterations for quick test'
    )
    
    # Write the modified version
    with open('run_nerf_quick.py', 'w') as f:
        f.write(modified_content)
    
    print("✅ Created run_nerf_quick.py for quick testing")

def run_quick_test():
    """Run the quick NeRF test"""
    
    print("🚀 Starting quick NeRF test...")
    
    # Create the quick test script
    create_quick_test_nerf()
    
    # Run the quick test
    cmd = [
        'python', 'run_nerf_quick.py',
        '--config', 'configs/lego.txt'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        print("=== QUICK TEST OUTPUT ===")
        if result.stdout:
            print(result.stdout[-2000:])  # Show last 2000 characters
        
        if result.stderr:
            print("=== ERRORS/WARNINGS ===")
            print(result.stderr[-1000:])  # Show last 1000 characters
        
        if result.returncode == 0:
            print("✅ Quick NeRF test completed successfully!")
        else:
            print(f"❌ Quick NeRF test failed with return code: {result.returncode}")
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ Quick test timed out (5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error running quick test: {e}")
        return False
    finally:
        # Clean up the temporary file
        if os.path.exists('run_nerf_quick.py'):
            os.remove('run_nerf_quick.py')
            print("🧹 Cleaned up temporary files")

if __name__ == "__main__":
    # Change to the NeRF directory
    nerf_dir = '/home/ubuntu/mono_to_3d/neural_radiance_fields/open_source_implementations/nerf-pytorch'
    if os.path.exists(nerf_dir):
        os.chdir(nerf_dir)
        print(f"📁 Changed to directory: {os.getcwd()}")
    else:
        print(f"❌ NeRF directory not found: {nerf_dir}")
        sys.exit(1)
    
    # Run the quick test
    success = run_quick_test()
    sys.exit(0 if success else 1) 