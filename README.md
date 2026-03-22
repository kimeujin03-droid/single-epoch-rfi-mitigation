# Single-Epoch Low-Rank RFI Mitigation: Reproducibility Toolkit

Code + synthetic experiments for the manuscript on **single-epoch** low-rank interference subtraction and its failure modes in radio astronomy.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Quick Start

### Installation

```bash
git clone https://github.com/kimeujin03-droid/single-epoch-rfi-mitigation.git
cd single-epoch-rfi-mitigation

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -U pip
pip install -r requirements.txt
```

### Generate Everything (One Command)

```bash
python scripts/run_all.py
```

This will:
1. Run all experiments → `runs/` (CSV data)
2. Plot experiment results → `runs/` (PNG plots)
3. Generate all paper figures → `tools/outputs/` (PNG figures)

---

## Usage

### Option 1: Generate Everything

```bash
# All experiments + all figures
python scripts/run_all.py
```

### Option 2: Generate Only Paper Figures

```bash
# Skip experiments, just make figures
python tools/make_all_figures.py
```

Output: `tools/outputs/*.png`

### Option 3: Individual Figures

```bash
# Figure 2 (exact paper reproduction)
python tools/make_figure2_rank_sweep_paper.py

# Rank sweep (general version)
python tools/make_rank_sweep.py

# HERA figures (requires HERA data)
python tools/make_figure_hera_correlation.py
python tools/make_figure_hera_rank_pareto.py
```

### Option 4: Individual Experiments

```bash
# Quick demo
python scripts/run_demo.py

# Weight sweep
python scripts/sweep_weights.py
python scripts/plot_results.py --kind weights

# Rank sweep
python scripts/sweep_rank.py --trials 10
python scripts/plot_results.py --kind rank

# Monte Carlo
python scripts/sweep_mc.py
python scripts/plot_results.py --kind mc
```

---

## Methods Included

- **SVD** (rank-r truncated)
- **FWSVD** (Frequency-weighted SVD, rank-1 ALS)
- **Hard masking** (time excision)
- **Baselines**: NMF, ICA, RPCA (require scikit-learn)

---

## Repository Structure

```
single-epoch-rfi-mitigation/
├── configs/              # Parameters (Table 2 from paper)
│   └── defaults.json
├── scripts/              # Experiments (generate CSV)
│   ├── run_all.py        # Run everything
│   ├── run_demo.py       # Quick demo
│   ├── sweep_weights.py  # FWSVD weight sweep
│   ├── sweep_rank.py     # Rank sweep experiment
│   └── sweep_mc.py       # Monte Carlo trials
├── src/                  # Core algorithms
│   ├── simulate.py       # Synthetic data generation
│   ├── methods.py        # RFI mitigation methods
│   ├── metrics.py        # Evaluation metrics
│   └── weights.py        # FWSVD weight matrices
├── tools/                # Paper figures (generate PNG)
│   ├── make_all_figures.py  # Generate all figures
│   ├── make_figure2_rank_sweep_paper.py  # Figure 2
│   ├── make_rank_sweep.py
│   ├── make_figure_hera_*.py
│   └── outputs/          # Generated figures (PNG)
└── runs/                 # Experiment outputs (CSV)
```

---

## Configuration

All parameters defined in `configs/defaults.json` (Table 2 from paper):
- T=60, F=240 (time × frequency)
- Science: Gaussian line at 6.0 MHz
- RFI: 5 comb lines + Gaussian burst
- Noise: σ=0.001

---

## Reproducibility

✅ **Fixed seeds** (default: 42)
✅ **Manifest files** (metadata + checksums)
✅ **No manual edits** (all plots from CSV)

---

## Notes

- **Figure 2**: Uses empirical scaling constant (600.0) calibrated for paper
- **HERA figures**: Require real observation data (not included)
- **HardMask NaN values**: Expected when masking removes science core

---

## Citation

If you use this code, please cite:

```bibtex
@article{yourpaper2024,
  title={Single-Epoch Low-Rank RFI Mitigation},
  author={Your Name et al.},
  journal={Journal Name},
  year={2024}
}
```

---

## License

MIT License - see [LICENSE](LICENSE)

## Contact

[GitHub Issues](https://github.com/kimeujin03-droid/single-epoch-rfi-mitigation/issues)
