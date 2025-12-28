# Single-Epoch Low-Rank RFI Mitigation: Reproducibility Toolkit

Code + synthetic experiments for the manuscript on **single-epoch** low-rank interference subtraction
and its failure modes.

Included methods:
- SVD (rank-r)
- Frequency-weighted SVD (FWSVD, rank-1 ALS)
- Hard masking (time excision)
- Baselines for peer-review defensibility: NMF, ICA, and RPCA (PCP). These require scikit-learn.

Repository (paper + code): https://github.com/kimeujin03-droid/single-epoch-rfi-mitigation

## Quick start (Windows/macOS/Linux)

From the repository root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# One-command reproduction (generates runs/* and figures)
python scripts/run_all.py
```

Outputs:
- `runs/` : CSV summaries and per-run metadata (including seeds)
- `figs/` : sweep plots used in the paper

## Reproducibility contract

- Every script writes a `manifest.json` with:
  - seed(s), parameter grid, and code version string
  - output file list + checksums (when applicable)
- Seeds are fixed by default (`--seed` option available on sweeps).
- All plots are regenerated from the CSV summaries (no manual edits).

## What to cite / how to reproduce figures

- Demo figure: `python scripts/run_demo.py`
- Weight sweep: `python scripts/sweep_weights.py` → `python scripts/plot_results.py --kind weights`
- Rank sweep: `python scripts/sweep_rank.py` → `python scripts/plot_results.py --kind rank`
- Monte Carlo: `python scripts/sweep_mc.py` → `python scripts/plot_results.py --kind mc`

## Notes

If you see `HardMask` bias statistics reported as NaN, this is expected when
hard masking removes the entire science core. Use `coverage_core` to interpret
how much of the science core remains after excision.
