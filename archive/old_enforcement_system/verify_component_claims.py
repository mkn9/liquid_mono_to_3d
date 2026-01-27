#!/usr/bin/env python3
"""
Component Claim Verification Script

Ensures that any component claimed in documentation/branch names actually exists
and is properly implemented. This prevents "MAGVIT" claims when no MAGVIT exists.

Usage:
    python scripts/verify_component_claims.py --spec .component_spec.yaml
"""

import sys
import yaml
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Set
from dataclasses import dataclass

@dataclass
class ComponentSpec:
    """Specification for a required component."""
    name: str
    type: str  # 'model', 'library', 'api', 'algorithm'
    required_imports: List[str]
    required_classes: List[str]
    required_functions: List[str]
    forbidden_patterns: List[str]  # Patterns that indicate fake implementation
    min_lines_of_code: int
    
class ComponentVerifier:
    """Verifies that claimed components actually exist."""
    
    def __init__(self, spec_file: Path):
        self.spec_file = spec_file
        self.specs = self._load_specs()
        self.failures = []
        self.warnings = []
        
    def _load_specs(self) -> Dict[str, ComponentSpec]:
        """Load component specifications from YAML file."""
        if not self.spec_file.exists():
            print(f"ERROR: Spec file not found: {self.spec_file}")
            sys.exit(1)
            
        with open(self.spec_file) as f:
            data = yaml.safe_load(f)
        
        specs = {}
        for name, spec_data in data.get('components', {}).items():
            specs[name] = ComponentSpec(
                name=name,
                type=spec_data.get('type', 'model'),
                required_imports=spec_data.get('required_imports', []),
                required_classes=spec_data.get('required_classes', []),
                required_functions=spec_data.get('required_functions', []),
                forbidden_patterns=spec_data.get('forbidden_patterns', []),
                min_lines_of_code=spec_data.get('min_lines_of_code', 0)
            )
        
        return specs
    
    def verify_all(self, search_paths: List[Path]) -> bool:
        """Verify all components in specification."""
        print("="*70)
        print("COMPONENT CLAIM VERIFICATION")
        print("="*70)
        print()
        
        all_passed = True
        
        for name, spec in self.specs.items():
            print(f"Verifying: {name} ({spec.type})")
            passed = self.verify_component(spec, search_paths)
            
            if not passed:
                all_passed = False
                print(f"  ✗ FAILED\n")
            else:
                print(f"  ✓ PASSED\n")
        
        # Print summary
        print("="*70)
        if all_passed:
            print("✓ ALL COMPONENT CLAIMS VERIFIED")
        else:
            print("✗ COMPONENT VERIFICATION FAILED")
            print()
            print("Failures:")
            for failure in self.failures:
                print(f"  - {failure}")
        
        if self.warnings:
            print()
            print("Warnings:")
            for warning in self.warnings:
                print(f"  ⚠ {warning}")
        
        print("="*70)
        
        return all_passed
    
    def verify_component(self, spec: ComponentSpec, search_paths: List[Path]) -> bool:
        """Verify a single component."""
        # Find all Python files in search paths
        py_files = []
        for path in search_paths:
            if path.is_file():
                py_files.append(path)
            else:
                py_files.extend(path.rglob("*.py"))
        
        # Search for required imports
        found_imports = set()
        found_classes = set()
        found_functions = set()
        forbidden_found = []
        total_relevant_lines = 0
        
        for py_file in py_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    
                # Check imports
                for req_import in spec.required_imports:
                    if re.search(rf'\bimport\s+{re.escape(req_import)}\b|\bfrom\s+{re.escape(req_import)}\b', content):
                        found_imports.add(req_import)
                
                # Check classes
                for req_class in spec.required_classes:
                    if re.search(rf'\bclass\s+{re.escape(req_class)}\b', content):
                        found_classes.add(req_class)
                        # Count lines in this file as relevant
                        total_relevant_lines += len(content.split('\n'))
                
                # Check functions
                for req_func in spec.required_functions:
                    if re.search(rf'\bdef\s+{re.escape(req_func)}\b', content):
                        found_functions.add(req_func)
                
                # Check forbidden patterns (indicates fake implementation)
                for forbidden in spec.forbidden_patterns:
                    if re.search(forbidden, content, re.IGNORECASE):
                        forbidden_found.append((py_file, forbidden))
                        
            except Exception as e:
                self.warnings.append(f"Could not read {py_file}: {e}")
        
        # Evaluate results
        passed = True
        
        # Check imports
        missing_imports = set(spec.required_imports) - found_imports
        if missing_imports:
            self.failures.append(
                f"{spec.name}: Missing required imports: {', '.join(missing_imports)}"
            )
            passed = False
        else:
            print(f"    ✓ All required imports found")
        
        # Check classes
        missing_classes = set(spec.required_classes) - found_classes
        if missing_classes:
            self.failures.append(
                f"{spec.name}: Missing required classes: {', '.join(missing_classes)}"
            )
            passed = False
        else:
            print(f"    ✓ All required classes found")
        
        # Check functions
        missing_functions = set(spec.required_functions) - found_functions
        if missing_functions:
            self.failures.append(
                f"{spec.name}: Missing required functions: {', '.join(missing_functions)}"
            )
            passed = False
        else:
            print(f"    ✓ All required functions found")
        
        # Check forbidden patterns
        if forbidden_found:
            for py_file, pattern in forbidden_found:
                self.failures.append(
                    f"{spec.name}: Found forbidden pattern '{pattern}' in {py_file}"
                )
            passed = False
        else:
            print(f"    ✓ No forbidden patterns found")
        
        # Check minimum lines of code
        if total_relevant_lines < spec.min_lines_of_code:
            self.warnings.append(
                f"{spec.name}: Only {total_relevant_lines} lines found, expected at least {spec.min_lines_of_code}"
            )
        else:
            print(f"    ✓ Sufficient implementation ({total_relevant_lines} lines)")
        
        return passed

def verify_git_branch_claims(branch_spec_file: Path) -> bool:
    """Verify that git branch names match their implementations."""
    if not branch_spec_file.exists():
        print(f"⚠ Branch spec file not found: {branch_spec_file}")
        return True  # Not a failure, just not applicable
    
    print("\n" + "="*70)
    print("GIT BRANCH CLAIM VERIFICATION")
    print("="*70)
    print()
    
    with open(branch_spec_file) as f:
        branch_specs = json.load(f)
    
    all_passed = True
    
    for branch_name, spec in branch_specs.items():
        print(f"Branch: {branch_name}")
        
        # Check if branch exists
        result = subprocess.run(
            ['git', 'rev-parse', '--verify', branch_name],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"  ✗ Branch does not exist")
            all_passed = False
            continue
        
        # Get files in branch
        result = subprocess.run(
            ['git', 'ls-tree', '-r', branch_name, '--name-only'],
            capture_output=True,
            text=True
        )
        
        branch_files = result.stdout.strip().split('\n')
        
        # Check for claimed components
        claimed_components = spec.get('components', [])
        for component in claimed_components:
            # Check if component is actually used in branch
            found = False
            component_lower = component.lower()
            
            for file in branch_files:
                if component_lower in file.lower():
                    found = True
                    break
                
                # Also check file contents
                result = subprocess.run(
                    ['git', 'show', f'{branch_name}:{file}'],
                    capture_output=True,
                    text=True
                )
                
                if component_lower in result.stdout.lower():
                    # Check if it's just a comment or actual usage
                    if re.search(rf'\bimport\s+.*{re.escape(component_lower)}', 
                                result.stdout, re.IGNORECASE):
                        found = True
                        break
            
            if found:
                print(f"  ✓ {component}: Found")
            else:
                print(f"  ✗ {component}: CLAIMED BUT NOT FOUND")
                all_passed = False
        
        print()
    
    return all_passed

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify component claims")
    parser.add_argument('--spec', type=Path, default=Path('.component_spec.yaml'),
                       help='Component specification file')
    parser.add_argument('--branch-spec', type=Path, default=Path('.branch_specifications.json'),
                       help='Branch specification file')
    parser.add_argument('--search-path', type=Path, action='append',
                       help='Paths to search for implementations')
    
    args = parser.parse_args()
    
    # Default search paths
    search_paths = args.search_path or [Path('.')]
    
    # Verify components
    verifier = ComponentVerifier(args.spec)
    components_passed = verifier.verify_all(search_paths)
    
    # Verify git branch claims
    branches_passed = verify_git_branch_claims(args.branch_spec)
    
    # Exit with appropriate code
    if components_passed and branches_passed:
        print("\n✓ ALL VERIFICATIONS PASSED")
        sys.exit(0)
    else:
        print("\n✗ VERIFICATION FAILED")
        sys.exit(1)

