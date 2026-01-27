#!/usr/bin/env python3
"""
Main MacBook Orchestration Script
==================================
Manages git tree procedures, branch creation, task execution, and result transfer.
Reads configuration from config.yaml.
"""

import sys
import os
import yaml
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


class GitTreeManager:
    """Manages git branches and tree operations."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.project_root = Path(config['project']['root_dir'])
        self.default_branch = config['git']['default_branch']
        self.branch_prefix = config['git']['branch_prefix']
    
    def _detect_default_branch(self) -> str:
        """Detect the default branch (main or master)."""
        # Try main first
        if self.branch_exists('main'):
            return 'main'
        # Try master
        if self.branch_exists('master'):
            return 'master'
        # Try origin/main
        try:
            result = subprocess.run(
                ['git', 'branch', '-r', '--list', 'origin/main'],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            if 'origin/main' in result.stdout:
                return 'main'
        except:
            pass
        # Try origin/master
        try:
            result = subprocess.run(
                ['git', 'branch', '-r', '--list', 'origin/master'],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            if 'origin/master' in result.stdout:
                return 'master'
        except:
            pass
        # Fallback to current branch
        return self.get_current_branch()
        
    def get_current_branch(self) -> str:
        """Get current git branch."""
        try:
            # Try --show-current first (newer git)
            result = subprocess.run(
                ['git', 'branch', '--show-current'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            branch = result.stdout.strip()
            if branch:
                return branch
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Fallback for older git versions
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return "unknown"
    
    def branch_exists(self, branch_name: str) -> bool:
        """Check if branch exists."""
        try:
            result = subprocess.run(
                ['git', 'branch', '--list', branch_name],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            return branch_name in result.stdout
        except subprocess.CalledProcessError:
            return False
    
    def create_branch(self, branch_name: str, from_branch: Optional[str] = None) -> bool:
        """Create a new git branch."""
        if self.branch_exists(branch_name):
            print(f"  Branch {branch_name} already exists")
            return True
        
        # Detect default branch if not specified
        if from_branch is None:
            from_branch = self._detect_default_branch()
        
        try:
            # Switch to base branch first (if it exists)
            if from_branch and from_branch != "unknown":
                try:
                    subprocess.run(
                        ['git', 'checkout', from_branch],
                        cwd=self.project_root,
                        check=True,
                        capture_output=True
                    )
                except subprocess.CalledProcessError:
                    # If default branch doesn't exist, use current branch
                    current = self.get_current_branch()
                    if current != "unknown":
                        from_branch = current
                    else:
                        # Create from HEAD
                        pass
            
            # Create new branch
            subprocess.run(
                ['git', 'checkout', '-b', branch_name],
                cwd=self.project_root,
                check=True
            )
            
            # Return to default if configured
            if self.config['git']['operations']['return_to_default']:
                subprocess.run(
                    ['git', 'checkout', self.default_branch],
                    cwd=self.project_root,
                    check=True
                )
            
            print(f"  ✅ Created branch: {branch_name}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Failed to create branch {branch_name}: {e}")
            return False
    
    def checkout_branch(self, branch_name: str, stash: bool = True) -> bool:
        """Checkout a git branch."""
        try:
            if stash and self.config['workflow']['switch_branch']['stash_changes']:
                subprocess.run(
                    ['git', 'stash'],
                    cwd=self.project_root,
                    check=False
                )
            
            subprocess.run(
                ['git', 'checkout', branch_name],
                cwd=self.project_root,
                check=True
            )
            
            if self.config['workflow']['switch_branch']['pull_latest']:
                subprocess.run(
                    ['git', 'pull'],
                    cwd=self.project_root,
                    check=False
                )
            
            return True
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Failed to checkout branch {branch_name}: {e}")
            return False
    
    def create_all_task_branches(self) -> Dict[str, bool]:
        """Create branches for all tasks."""
        results = {}
        tasks = self.config['tasks']
        
        print("Creating git branches for all tasks...")
        print("-" * 60)
        
        for task_name, task_config in tasks.items():
            branch_name = task_config['branch']
            results[task_name] = self.create_branch(branch_name)
        
        print("-" * 60)
        return results


class TaskExecutor:
    """Executes tasks on EC2 or locally."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.ec2_host = config['project']['ec2_host']
        self.ec2_key = config['project']['ec2_key']
        self.ec2_path = config['project']['ec2_path']
        self.execution_config = config['execution']
        
    def run_on_ec2(self, task_name: str, task_config: Dict) -> bool:
        """Run task on EC2 instance."""
        script_path = task_config['script']
        branch_name = task_config['branch']
        
        # Build command
        cmd = f"""
        cd {self.ec2_path} && \
        source {self.execution_config['ec2']['python_env']} && \
        git checkout {branch_name} && \
        python3 {script_path}
        """
        
        ssh_cmd = [
            'ssh',
            '-i', self.ec2_key,
            self.ec2_host,
            cmd
        ]
        
        print(f"  Running {task_name} on EC2 (branch: {branch_name})...")
        try:
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True
            )
            print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            return result.returncode == 0
        except Exception as e:
            print(f"  ❌ Error running task: {e}")
            return False
    
    def run_locally(self, task_name: str, task_config: Dict) -> bool:
        """Run task locally."""
        script_path = task_config['script']
        branch_name = task_config['branch']
        
        # Checkout branch
        git_manager = GitTreeManager(self.config)
        if not git_manager.checkout_branch(branch_name):
            return False
        
        # Run script
        script_full_path = self.config['project']['root_dir'] / script_path
        
        print(f"  Running {task_name} locally (branch: {branch_name})...")
        try:
            result = subprocess.run(
                ['python3', str(script_full_path)],
                cwd=self.config['project']['root_dir']
            )
            return result.returncode == 0
        except Exception as e:
            print(f"  ❌ Error running task: {e}")
            return False
    
    def run_task(self, task_name: str, task_config: Dict) -> bool:
        """Run a task based on execution location."""
        location = self.execution_config['location']
        
        if location == 'ec2':
            return self.run_on_ec2(task_name, task_config)
        else:
            return self.run_locally(task_name, task_config)


class ResultTransfer:
    """Handles result transfer from EC2 to MacBook."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.ec2_host = config['project']['ec2_host']
        self.ec2_key = config['project']['ec2_key']
        self.ec2_path = config['project']['ec2_path']
        self.output_config = config['output']
        
    def transfer_results(self, task_name: Optional[str] = None) -> bool:
        """Transfer results from EC2 to MacBook."""
        if not self.output_config['transfer_to_macbook']:
            return True
        
        ec2_output_dir = f"{self.ec2_path}/{self.output_config['directory']}"
        local_output_dir = Path(self.output_config['macbook_path'])
        local_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build rsync command
        rsync_cmd = [
            'rsync',
            '-avz',
            '-e', f"ssh -i {self.ec2_key}",
            f"{self.ec2_host}:{ec2_output_dir}/",
            str(local_output_dir) + '/',
            '--update'
        ]
        
        if task_name:
            # Transfer specific task results
            pattern = f"*{task_name}*"
            rsync_cmd.insert(-2, '--include', pattern)
            rsync_cmd.insert(-2, '--exclude', '*')
        
        print(f"  Transferring results from EC2...")
        try:
            result = subprocess.run(rsync_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  ✅ Results transferred successfully")
                return True
            else:
                print(f"  ⚠️  Transfer completed with warnings")
                return True  # rsync warnings are usually OK
        except Exception as e:
            print(f"  ❌ Transfer failed: {e}")
            return False


class TestRunner:
    """Runs tests before task execution."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.test_config = config['testing']
        
    def run_tests(self) -> bool:
        """Run all configured tests."""
        if not self.test_config['run_before_execution']:
            return True
        
        print("Running tests...")
        print("-" * 60)
        
        all_passed = True
        for test_file in self.test_config['test_files']:
            test_path = Path(self.config['project']['root_dir']) / test_file
            if not test_path.exists():
                print(f"  ⚠️  Test file not found: {test_file}")
                continue
            
            print(f"  Running {test_file}...")
            try:
                result = subprocess.run(
                    ['python3', '-m', 'pytest', str(test_path), '-v'],
                    cwd=self.config['project']['root_dir'],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    print(f"  ✅ {test_file} passed")
                else:
                    print(f"  ❌ {test_file} failed")
                    print(result.stdout)
                    all_passed = False
                    if self.test_config['stop_on_failure']:
                        return False
            except Exception as e:
                print(f"  ❌ Error running {test_file}: {e}")
                all_passed = False
        
        print("-" * 60)
        return all_passed


class MainOrchestrator:
    """Main orchestration class."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = self.load_config()
        self.git_manager = GitTreeManager(self.config)
        self.task_executor = TaskExecutor(self.config)
        self.result_transfer = ResultTransfer(self.config)
        self.test_runner = TestRunner(self.config)
        
    def load_config(self) -> Dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def create_branches(self) -> None:
        """Create all task branches."""
        print("=" * 60)
        print("Creating Git Branches")
        print("=" * 60)
        results = self.git_manager.create_all_task_branches()
        
        # Summary
        created = sum(1 for v in results.values() if v)
        total = len(results)
        print(f"\n✅ Created {created}/{total} branches")
    
    def run_tasks(self, task_names: Optional[List[str]] = None, parallel: bool = True) -> None:
        """Run specified tasks or all tasks."""
        tasks = self.config['tasks']
        
        # Filter tasks
        if task_names:
            tasks = {k: v for k, v in tasks.items() if k in task_names}
        
        # Filter by parallel flag
        if parallel:
            tasks = {k: v for k, v in tasks.items() if v.get('parallel', True)}
        
        # Sort by priority
        sorted_tasks = sorted(
            tasks.items(),
            key=lambda x: x[1].get('priority', 999)
        )
        
        print("=" * 60)
        print("Running Tasks")
        print("=" * 60)
        
        # Run tests first
        if not self.test_runner.run_tests():
            print("⚠️  Tests failed, but continuing...")
        
        # Execute tasks
        results = {}
        for task_name, task_config in sorted_tasks:
            print(f"\n[{task_name}] {task_config['description']}")
            print("-" * 60)
            
            success = self.task_executor.run_task(task_name, task_config)
            results[task_name] = success
            
            # Transfer results
            if success and self.config['execution']['options']['transfer_results']:
                self.result_transfer.transfer_results(task_name)
        
        # Summary
        print("\n" + "=" * 60)
        print("Execution Summary")
        print("=" * 60)
        for task_name, success in results.items():
            status = "✅" if success else "❌"
            print(f"{status} {task_name}")
    
    def list_branches(self) -> None:
        """List all task branches."""
        print("=" * 60)
        print("Task Branches")
        print("=" * 60)
        
        current_branch = self.git_manager.get_current_branch()
        print(f"Current branch: {current_branch}\n")
        
        for task_name, task_config in self.config['tasks'].items():
            branch_name = task_config['branch']
            exists = self.git_manager.branch_exists(branch_name)
            status = "✅" if exists else "❌"
            print(f"{status} {branch_name} ({task_name})")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Git Tree and Task Orchestration"
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to config.yaml file'
    )
    parser.add_argument(
        '--create-branches',
        action='store_true',
        help='Create all task branches'
    )
    parser.add_argument(
        '--run-tasks',
        nargs='*',
        help='Run specified tasks (or all if none specified)'
    )
    parser.add_argument(
        '--list-branches',
        action='store_true',
        help='List all task branches'
    )
    parser.add_argument(
        '--transfer-results',
        action='store_true',
        help='Transfer results from EC2'
    )
    
    args = parser.parse_args()
    
    try:
        orchestrator = MainOrchestrator(args.config)
        
        if args.create_branches:
            orchestrator.create_branches()
        
        if args.list_branches:
            orchestrator.list_branches()
        
        if args.run_tasks is not None:
            task_names = args.run_tasks if args.run_tasks else None
            orchestrator.run_tasks(task_names)
        
        if args.transfer_results:
            orchestrator.result_transfer.transfer_results()
        
        if not any([args.create_branches, args.list_branches, args.run_tasks, args.transfer_results]):
            parser.print_help()
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

