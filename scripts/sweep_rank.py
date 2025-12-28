from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse, os
import numpy as np
import pandas as pd

import os, sys
# Allow running scripts directly: add repo root to PYTHONPATH
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from src.simulate import SimParams, make_synthetic
from src.weights import make_weight_matrix
from src.methods import svd_subtract_rank_r
from src.metrics import estimate_signal_1d, relative_bias_percent, aggregate_in_band
from src.io_utils import load_json, save_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/sweep_rank")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--config", default="configs/defaults.json")
    ap.add_argument("--grid", default="configs/sweep_rank.json")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = load_json(args.config)
    grid = load_json(args.grid)

    params = SimParams(
        T=cfg["T"], F=cfg["F"],
        freq_min_mhz=cfg["freq_min_mhz"], freq_max_mhz=cfg["freq_max_mhz"],
        science_center_mhz=cfg["science_center_mhz"], science_sigma_mhz=cfg["science_sigma_mhz"], science_amp=cfg["science_amp"],
        noise_sigma=cfg["noise_sigma"],
        comb_centers_mhz=tuple(cfg["comb_centers_mhz"]), comb_sigma_mhz=cfg["comb_sigma_mhz"], comb_amp=cfg["comb_amp"],
        time_burst_center=cfg["time_burst_center"], time_burst_sigma=cfg["time_burst_sigma"],
        science_band=tuple(cfg["science_band"]), protected_band=tuple(cfg["protected_band"]),
    )

    rows=[]
    trial_seeds = np.arange(args.trials) + int(cfg.get("seed", 0))
    for r in grid["rank_grid"]:
        for s in trial_seeds:
            D, S_true, meta = make_synthetic(params, seed=int(s), overlap=True)
            freqs = meta["freqs"]

            D_svd, _ = svd_subtract_rank_r(D, r=int(r))
            S_hat = estimate_signal_1d(D_svd)
            bias = relative_bias_percent(S_hat, S_true, eps=1e-5)
            agg = aggregate_in_band(freqs, bias, params.science_band)

            rows.append({"seed": int(s), "rank": int(r), "median": agg["median"], "mean": agg["mean"], "max": agg["max"]})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "summary.csv"), index=False)

    agg = df.groupby("rank").agg({"median":"median","mean":"mean","max":"max"}).reset_index()
    agg.to_csv(os.path.join(args.out, "summary_agg.csv"), index=False)

    save_json(os.path.join(args.out, "meta.json"), {"defaults": cfg, "grid": grid, "trials": args.trials})
    print("Saved:", os.path.join(args.out, "summary.csv"))

if __name__ == "__main__":
    main()
