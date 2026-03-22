#!/usr/bin/env python3
"""
Generate all paper figures.

This script runs all figure generation scripts in tools/ directory.
Equivalent to running each make_figure_*.py script individually.

Usage:
    python tools/make_all_figures.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_script(script_path: Path, description: str):
    """Run a figure generation script."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(ROOT),
        capture_output=False
    )
    
    if result.returncode != 0:
        print(f"✗ FAILED: {script_path.name}")
        return False
    
    print(f"✓ SUCCESS: {script_path.name}")
    return True


def main():
    print("="*70)
    print("Generating all paper figures")
    print("="*70)
    
    scripts = [
        ("tools/make_figure2_rank_sweep_paper.py", "Figure 2: Rank sweep (paper)"),
        ("tools/make_rank_sweep.py", "Rank sweep (general)"),
        ("tools/make_figure_hera_correlation.py", "HERA correlation structure"),
        ("tools/make_figure_hera_rank_pareto.py", "HERA rank Pareto frontier"),
        ("tools/make_figure_svd_fws_tsvd_panels.py", "SVD/FWS/TSVD comparison panels"),
    ]
    
    results = []
    for script_rel, desc in scripts:
        script_path = ROOT / script_rel
        
        if not script_path.exists():
            print(f"\n⚠ SKIP: {script_path.name} (not found)")
            continue
        
        success = run_script(script_path, desc)
        results.append((script_path.name, success))
    
    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {name}")
    
    print("\nOutput directory: tools/outputs/")
    
    # List generated files
    outputs_dir = ROOT / "tools" / "outputs"
    if outputs_dir.exists():
        png_files = list(outputs_dir.glob("*.png"))
        if png_files:
            print(f"\nGenerated {len(png_files)} PNG files:")
            for f in sorted(png_files):
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  - {f.name} ({size_mb:.2f} MB)")
    
    print("="*70)
    
    # Exit code
    all_success = all(s for _, s in results)
    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
Generate all paper figures.

This script runs all figure generation scripts in tools/ directory.
Equivalent to running each make_figure_*.py script individually.

Usage:
    python tools/make_all_figures.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_script(script_path: Path, description: str):
    """Run a figure generation script."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(ROOT),
        capture_output=False
    )
    
    if result.returncode != 0:
        print(f"✗ FAILED: {script_path.name}")
        return False
    
    print(f"✓ SUCCESS: {script_path.name}")
    return True


def main():
    print("="*70)
    print("Generating all paper figures")
    print("="*70)
    
    scripts = [
        ("tools/make_figure2_rank_sweep_paper.py", "Figure 2: Rank sweep (paper)"),
        ("tools/make_rank_sweep.py", "Rank sweep (general)"),
        ("tools/make_figure_hera_correlation.py", "HERA correlation structure"),
        ("tools/make_figure_hera_rank_pareto.py", "HERA rank Pareto frontier"),
        ("tools/make_figure_svd_fws_tsvd_panels.py", "SVD/FWS/TSVD comparison panels"),
    ]
    
    results = []
    for script_rel, desc in scripts:
        script_path = ROOT / script_rel
        
        if not script_path.exists():
            print(f"\n⚠ SKIP: {script_path.name} (not found)")
            continue
        
        success = run_script(script_path, desc)
        results.append((script_path.name, success))
    
    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {name}")
    
    print("\nOutput directory: tools/outputs/")
    
    # List generated files
    outputs_dir = ROOT / "tools" / "outputs"
    if outputs_dir.exists():
        png_files = list(outputs_dir.glob("*.png"))
        if png_files:
            print(f"\nGenerated {len(png_files)} PNG files:")
            for f in sorted(png_files):
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  - {f.name} ({size_mb:.2f} MB)")
    
    print("="*70)
    
    # Exit code
    all_success = all(s for _, s in results)
    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
