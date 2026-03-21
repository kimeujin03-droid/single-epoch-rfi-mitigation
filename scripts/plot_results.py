#!/usr/bin/env python
"""Plot summary outputs from sweeps."""

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
    """Plot FWSVD sensitivity to (w_core, w_prot) like the paper figure.

    Expects a grid over (w_core, w_prot) with FWSVD median bias in ``fws_median``.
    """
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))

    # Require the expected aggregate columns
    required = {"w_core", "w_prot", "fws_median"}
    if not required.issubset(df.columns):
        raise ValueError(f"Weight sweep summary is missing columns: {required - set(df.columns)}")

    # One curve per w_core, x-axis is w_prot (log), y-axis is FWSVD median bias
    for w_core, g in df.groupby("w_core"):
        g = g.copy().sort_values("w_prot")
        x = g["w_prot"].astype(float).values
        y = g["fws_median"].astype(float).values
        plt.plot(x, y, marker="o", label=f"w_core={w_core}")

    plt.xscale("log")
    plt.xlabel("w_prot")
    plt.ylabel("Median bias (%) in science band")
    # Use a log scale on the y-axis so tick labels appear in scientific notation
    plt.yscale("log")
    plt.title("FWSVD sensitivity to (w_core, w_prot)")
    plt.legend(title="", fontsize=9)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_rank(df: pd.DataFrame, outpath: Path):
    g = df.copy().sort_values("rank")
    x = g["rank"].astype(int).values
    y_leak = g["rfi_leakage"].astype(float).values
    y_loss = g["science_loss"].astype(float).values

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.axvspan(2, 3, alpha=0.15, label="Operational knee (k≈2–3)")
    plt.plot(x, y_leak, marker="o", label="Rank Sweep — RFI leakage")
    plt.plot(x, y_loss, marker="s", label="Rank Sweep — science loss")
    plt.yscale("log")
    plt.xlabel("Rank k")
    plt.ylabel("Proxy (lower is better; log scale)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_mc(df: pd.DataFrame, outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    if "method" in df.columns:
        methods = df["method"].astype(str).values
        med = df["median"].astype(float).values
    else:
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
    plt.savefig(outpath, dpi=200)
    plt.close()


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
        df = _load_or_fail(runs / "sweep_mc_final" / "summary_agg.csv")
        plot_mc(df, runs / "sweep_mc_final" / "fig_mc.png")
        print(f"Wrote {runs / 'sweep_mc_final' / 'fig_mc.png'}")


if __name__ == "__main__":
    main()
