#!/usr/bin/env python3
"""
Generic contract runner for proof bundles.

Reads contracts/*.yaml, runs:
- imports (python -c "<import stmt>")
- commands
- tests (optional, though prove.sh already runs pytest)

Keeps output in the proof bundle directory.
"""

from __future__ import annotations
import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

try:
    import yaml  # pyyaml
except ImportError:
    print("Missing dependency: pyyaml. Install with: pip install pyyaml", file=sys.stderr)
    print("Or skip contracts by removing contracts/ directory", file=sys.stderr)
    sys.exit(0)  # Don't fail if no pyyaml (contracts are optional)

def run(cmd: str, log_path: Path) -> None:
    """Run command and log output."""
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n--- cmd: {cmd}\n")
        p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        f.write(p.stdout)
        f.write(f"\n--- exit_code: {p.returncode}\n")
        if p.returncode != 0:
            print(f"ERROR: Command failed: {cmd}", file=sys.stderr)
            print(p.stdout, file=sys.stderr)
            raise SystemExit(p.returncode)

def main() -> None:
    ap = argparse.ArgumentParser(description="Run component contracts")
    ap.add_argument("--contracts", default="contracts", help="Directory containing YAML contracts")
    ap.add_argument("--outdir", required=True, help="Output directory for proof bundle")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "contracts.log"

    # Create proof_outputs directory for contract outputs
    proof_outputs = Path("artifacts/proof_outputs")
    proof_outputs.mkdir(parents=True, exist_ok=True)

    contract_files = sorted(glob.glob(os.path.join(args.contracts, "*.yaml")))
    
    if not contract_files:
        print(f"No contracts found in {args.contracts}/", file=sys.stderr)
        return

    print(f"Running {len(contract_files)} contract(s)...")

    for path in contract_files:
        contract = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        name = contract.get("name", Path(path).stem)
        
        print(f"  → {name}")
        run(f'echo "== CONTRACT: {name} ({path}) =="', log_path)

        # Run import checks
        for stmt in contract.get("imports", []) or []:
            run(f'python3 -c "{stmt}"', log_path)

        # Run command checks
        for item in contract.get("commands", []) or []:
            cmd = item["cmd"] if isinstance(item, dict) else item
            run(cmd, log_path)

        # Run test checks
        for item in contract.get("tests", []) or []:
            cmd = item["cmd"] if isinstance(item, dict) else item
            run(cmd, log_path)

    print(f"✓ All contracts passed. See {log_path}")

if __name__ == "__main__":
    main()

