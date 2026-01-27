#!/usr/bin/env python3
import psutil
import time
import subprocess

def monitor_experiments():
    print("🔍 Monitoring experiment processes...")
    
    while True:
        python_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
            try:
                if proc.info['name'] == 'python' and 'train_' in ' '.join(proc.info['cmdline']):
                    python_processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        if python_processes:
            print(f"\n📊 Active experiments ({len(python_processes)} processes):")
            for proc in python_processes:
                cmdline = ' '.join(proc['cmdline'])
                if 'magvit' in cmdline:
                    exp_type = '🧠 MAGVIT'
                elif 'videogpt' in cmdline:
                    exp_type = '🎬 VideoGPT'
                else:
                    exp_type = '⚡ Unknown'
                
                print(f"  {exp_type} PID:{proc['pid']} CPU:{proc['cpu_percent']:.1f}% MEM:{proc['memory_percent']:.1f}%")
        else:
            print("\n✅ No experiment processes running")
            break
        
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    monitor_experiments()
