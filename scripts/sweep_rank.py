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

    rows = []
trial_seeds = np.arange(args.trials) + int(cfg.get("seed", 0))

for trial in trial_seeds:
    p = SimParams(**{**cfg, "science_amp": 0.1})
    D, S_true, meta = make_synthetic(p, seed=int(trial), overlap=True)
    freqs = meta["freqs"]

    # science band index (기존 그대로 사용)
    sci_band_idx = tuple(cfg.get("science_band_idx", (110, 130)))
    sci_lo, sci_hi = sci_band_idx  # [sci_lo, sci_hi) 라고 가정

    # leakage 측정을 위한 "outside band" 마스크
    F = D.shape[1]
    outside_mask = np.ones(F, dtype=bool)
    outside_mask[sci_lo:sci_hi] = False

    # rank k = 1..max_rank sweep
    for r in range(1, args.max_rank + 1):
        # 저랭크 클리닝
        D_svd, _ = svd_subtract_rank_r(D, r=r)

        # ---------- (A) science distortion proxy (over-cleaning) ----------
        # S_true 가 있는 band에서의 bias(기존 metric)
        S_hat = estimate_signal_1d(D_svd)
        bias = relative_bias_percent(S_hat, S_true, eps=1e-5)
        agg_sci = aggregate_in_band(freqs, bias, sci_band_idx)
        dist_proxy = float(agg_sci["median"])  # science distortion proxy

        # ---------- (B) RFI leakage proxy (under-cleaning) ----------
        # science band 밖(outside band)에서의 residual RMS
        # D_svd 는 "science + noise (+ 남은 RFI)" 라고 생각하면 됨
        resid_out = D_svd[:, outside_mask]
        # "남은 에너지"가 클수록 under-cleaning 심함
        leak_proxy = float(np.sqrt(np.mean(resid_out**2)))

        rows.append({
            "seed": int(trial),
            "k": int(r),
            "leak_proxy": leak_proxy,
            "dist_proxy": dist_proxy,
        })


    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "summary.csv"), index=False)

    agg = df.groupby("rank").agg({"median":"median","mean":"mean","max":"max"}).reset_index()
    agg.to_csv(os.path.join(args.out, "summary_agg.csv"), index=False)

    save_json(os.path.join(args.out, "meta.json"), {"defaults": cfg, "grid": grid, "trials": args.trials})
    print("Saved:", os.path.join(args.out, "summary.csv"))

if __name__ == "__main__":
    main()
