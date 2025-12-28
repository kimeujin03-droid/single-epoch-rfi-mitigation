from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os, sys
# Allow running scripts directly: add repo root to PYTHONPATH
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from src.simulate import SimParams, make_synthetic
from src.weights import make_weight_matrix
from src.methods import svd_subtract_rank_r, fwsvd_subtract_rank1, hard_mask_clean
from src.metrics import estimate_signal_1d, relative_bias_percent, aggregate_in_band
from src.io_utils import load_json, save_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/demo")
    ap.add_argument("--config", default="configs/defaults.json")
    ap.add_argument("--iters", type=int, default=150)
    args = ap.parse_args()

    cfg = load_json(args.config)
    params = SimParams(
        T=cfg["T"], F=cfg["F"],
        freq_min_mhz=cfg["freq_min_mhz"], freq_max_mhz=cfg["freq_max_mhz"],
        science_center_mhz=cfg["science_center_mhz"], science_sigma_mhz=cfg["science_sigma_mhz"], science_amp=cfg["science_amp"],
        noise_sigma=cfg["noise_sigma"],
        comb_centers_mhz=tuple(cfg["comb_centers_mhz"]), comb_sigma_mhz=cfg["comb_sigma_mhz"], comb_amp=cfg["comb_amp"],
        time_burst_center=cfg["time_burst_center"], time_burst_sigma=cfg["time_burst_sigma"],
        science_band=tuple(cfg["science_band"]), protected_band=tuple(cfg["protected_band"]),
    )

    os.makedirs(args.out, exist_ok=True)
    D, S_true, meta = make_synthetic(params, seed=cfg.get("seed", 0), overlap=True)
    freqs = meta["freqs"]

    # baselines
    D_svd, _ = svd_subtract_rank_r(D, r=int(cfg.get("rank_r", 1)))
    W = make_weight_matrix(freqs, params.T, params.science_band, params.protected_band,
                           w_core=float(cfg.get("w_core", 0.1)), w_prot=float(cfg.get("w_prot", 0.3)))
    D_fws, _ = fwsvd_subtract_rank1(D, W, iters=args.iters)
    D_hm, hm_mask = hard_mask_clean(D, freqs, params.science_band)

    S_svd = estimate_signal_1d(D_svd)
    S_fws = estimate_signal_1d(D_fws)
    # for hard mask, science band is NaN -> bias computed only where S_true exists; keep for completeness
    S_hm = estimate_signal_1d(D_hm)

    bias_svd = relative_bias_percent(S_svd, S_true)
    bias_fws = relative_bias_percent(S_fws, S_true)
    bias_hm = relative_bias_percent(S_hm, S_true)

    agg_svd = aggregate_in_band(freqs, bias_svd, params.science_band)
    agg_fws = aggregate_in_band(freqs, bias_fws, params.science_band)
    agg_hm = aggregate_in_band(freqs, bias_hm, params.science_band)

    df = pd.DataFrame([
        {"method":"SVD", **agg_svd},
        {"method":"FWSVD", **agg_fws},
        {"method":"HardMask", **agg_hm},
    ])
    df.to_csv(os.path.join(args.out, "summary_agg.csv"), index=False)

    # Figure: reconstruction
    plt.figure(figsize=(10, 7))
    ax1 = plt.subplot(2,1,1)
    ax1.plot(freqs, S_true, "k--", label="True Signal", linewidth=2)
    ax1.plot(freqs, S_svd, "r", label="SVD Rank-r", alpha=0.8)
    ax1.plot(freqs, S_fws, "g", label="FWSVD (weighted)", linewidth=2)
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Signal Reconstruction")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2,1,2)
    ax2.semilogy(freqs, bias_svd, "r", label="SVD bias")
    ax2.semilogy(freqs, bias_fws, "g", label="FWSVD bias")
    ax2.axhline(1000, linestyle=":", color="k", label="1000% line")
    ax2.set_xlabel("Frequency (MHz)")
    ax2.set_ylabel("Relative Bias (%)")
    ax2.set_title("Relative Bias Spectrum (log)")
    ax2.legend()
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "fig_demo.png"), dpi=200)
    print(df)

    save_json(os.path.join(args.out, "meta.json"), {"config": cfg, "agg": df.to_dict(orient="records")})

if __name__ == "__main__":
    main()
