#!/bin/bash
# Generate all paper figures
# Run from repository root: bash tools/make_all_figures.sh

set -e  # Exit on error

echo "========================================="
echo "Generating all paper figures..."
echo "========================================="

# Create output directory
mkdir -p tools/outputs

echo ""
echo "[1/5] Figure 2: Rank sweep (paper version)..."
python tools/make_figure2_rank_sweep_paper.py

echo ""
echo "[2/5] Rank sweep (general version)..."
python tools/make_rank_sweep.py

echo ""
echo "[3/5] HERA correlation..."
python tools/make_figure_hera_correlation.py

echo ""
echo "[4/5] HERA rank Pareto..."
python tools/make_figure_hera_rank_pareto.py

echo ""
echo "[5/5] SVD/FWS/TSVD panels..."
python tools/make_figure_svd_fws_tsvd_panels.py

echo ""
echo "========================================="
echo "✓ All figures generated!"
echo "Output: tools/outputs/"
echo "========================================="
ls -lh tools/outputs/*.png
