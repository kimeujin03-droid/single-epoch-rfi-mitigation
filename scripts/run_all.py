
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
    # Demo + sweeps
    run([sys.executable, "scripts/run_demo.py"])
    run([sys.executable, "scripts/sweep_weights.py"])
    run([sys.executable, "scripts/sweep_rank.py"])
    run([sys.executable, "scripts/sweep_mc.py"])
    # Plots (expects sweep outputs)
    run([sys.executable, "scripts/plot_results.py", "--kind", "weights"])
    run([sys.executable, "scripts/plot_results.py", "--kind", "rank"])
    run([sys.executable, "scripts/plot_results.py", "--kind", "mc"])

if __name__ == "__main__":
    main()
