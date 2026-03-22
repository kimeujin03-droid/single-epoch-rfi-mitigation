
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def run(cmd):
    print("\n$ " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(r.returncode)

def main():
    print("="*70)
    print("Running all experiments and generating all figures")
    print("="*70)
    
    # Part 1: Experiments (generate CSV data)
    print("\n" + "="*70)
    print("PART 1: Running experiments (generates runs/)")
    print("="*70)
    
    run([sys.executable, "scripts/run_demo.py"])
    run([sys.executable, "scripts/sweep_weights.py"])
    run([sys.executable, "scripts/sweep_rank.py", "--trials", "10"])
    run([sys.executable, "scripts/sweep_mc.py"])
    
    # Part 2: Plot experiment results
    print("\n" + "="*70)
    print("PART 2: Plotting experiment results")
    print("="*70)
    
    run([sys.executable, "scripts/plot_results.py", "--kind", "weights"])
    run([sys.executable, "scripts/plot_results.py", "--kind", "rank"])
    run([sys.executable, "scripts/plot_results.py", "--kind", "mc"])
    
    # Part 3: Generate paper figures
    print("\n" + "="*70)
    print("PART 3: Generating paper figures (generates tools/outputs/)")
    print("="*70)
    
    run([sys.executable, "tools/make_all_figures.py"])
    
    print("\n" + "="*70)
    print("✓ Complete!")
    print("  - Experiment data: runs/")
    print("  - Paper figures: tools/outputs/")
    print("="*70)

if __name__ == "__main__":
    main()
