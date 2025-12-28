#!/usr/bin/env python
"""Plot summary outputs from sweeps.

Run from the repository root. This script reads the aggregated CSV files produced by:
  - scripts/sweep_weights.py
  - scripts/sweep_rank.py
  - scripts/sweep_mc.py

Examples:
  python scripts/plot_results.py --kind weights
  python scripts/plot_results.py --kind rank
  python scripts/plot_results.py --kind mc
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _load_or_fail(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run the corresponding sweep first.")
    return pd.read_csv(path)


def plot_weights(df: pd.DataFrame, outpath: Path):
    # Expect columns: w_core, median_bias, mean_bias, max_bias, method (optional)
    if "method" not in df.columns:
        df = df.copy()
        df["method"] = "FWSVD"

    plt.figure()
    for method, g in df.groupby("method"):
        x = g["w_core"].astype(float).values
        y = g["median_bias"].astype(float).values
        order = np.argsort(x)
        plt.plot(x[order], y[order], marker="o", label=str(method))

    plt.xscale("log")
    plt.xlabel("w_core")
    plt.ylabel("Median relative bias (%) in science band")
    plt.title("Weight sweep")
    plt.legend()
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)


def plot_rank(df: pd.DataFrame, outpath: Path):
    # Expect columns: rank, median_bias, method (optional)
    if "method" not in df.columns:
        df = df.copy()
        df["method"] = "SVD"

    plt.figure()
    for method, g in df.groupby("method"):
        x = g["rank"].astype(int).values
        y = g["median_bias"].astype(float).values
        order = np.argsort(x)
        plt.plot(x[order], y[order], marker="o", label=str(method))

    plt.xlabel("Rank r")
    plt.ylabel("Median relative bias (%) in science band")
    plt.title("Rank sweep")
    plt.legend()
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)


def plot_mc(df: pd.DataFrame, outpath: Path):
    """Plot Monte-Carlo summary bars for each method.

    The sweep_mc script outputs summary_agg.csv with per-method stats columns.
    We plot the median science-band bias (and where present, core/fit-fail metrics).
    """
    # Columns are method, median, mean, max, n
    if "method" in df.columns:
        methods = df["method"].astype(str).values
        med = df["median"].astype(float).values

        plt.figure()
        plt.bar(methods, med)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Median relative bias (%) in science band")
        plt.title("Monte-Carlo summary")
        plt.tight_layout()
        outpath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath, dpi=200)
        return

    # Back-compat fallback: look for *_median columns
    cols = [c for c in df.columns if c.endswith("_median")]
    if not cols:
        raise ValueError("Unrecognized MC aggregate format. Expected column 'method' or '*_median' columns.")

    methods = [c.replace("_median", "") for c in cols]
    med = [float(df[c].iloc[0]) for c in cols]

    plt.figure()
    plt.bar(methods, med)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Median relative bias (%) in science band")
    plt.title("Monte-Carlo summary")
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["weights", "rank", "mc"], required=True)
    ap.add_argument("--runs", default="runs", help="Runs directory (default: runs)")
    args = ap.parse_args()

    runs = Path(args.runs)
    if args.kind == "weights":
        df = _load_or_fail(runs / "sweep_weights" / "summary_agg.csv")
        plot_weights(df, runs / "sweep_weights" / "fig_weights.png")
        print(f"Wrote {runs / 'sweep_weights' / 'fig_weights.png'}")
    elif args.kind == "rank":
        df = _load_or_fail(runs / "sweep_rank" / "summary_agg.csv")
        plot_rank(df, runs / "sweep_rank" / "fig_rank.png")
        print(f"Wrote {runs / 'sweep_rank' / 'fig_rank.png'}")
    else:
        df = _load_or_fail(runs / "sweep_mc" / "summary_agg.csv")
        plot_mc(df, runs / "sweep_mc" / "fig_mc.png")
        print(f"Wrote {runs / 'sweep_mc' / 'fig_mc.png'}")


if __name__ == "__main__":
    main()
