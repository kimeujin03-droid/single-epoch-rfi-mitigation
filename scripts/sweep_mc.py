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
from src.methods import (
    svd_subtract_rank_r,
    fwsvd_subtract_rank1,
    nmf_subtract_rank1,
    ica_subtract_rank1,
    rpca_subtract_sparse,
)
from src.metrics import estimate_signal_1d, relative_bias_percent, aggregate_in_band
from src.io_utils import load_json, save_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/sweep_mc")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--config", default="configs/defaults.json")
    ap.add_argument("--grid", default="configs/sweep_mc.json")
    ap.add_argument("--iters", type=int, default=150)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = load_json(args.config)
    grid = load_json(args.grid)

    base = load_json(args.config)
    params = SimParams(
        T=base["T"], F=base["F"],
        freq_min_mhz=base["freq_min_mhz"], freq_max_mhz=base["freq_max_mhz"],
        science_center_mhz=base["science_center_mhz"], science_sigma_mhz=base["science_sigma_mhz"], science_amp=base["science_amp"],
        noise_sigma=base["noise_sigma"],
        comb_centers_mhz=tuple(base["comb_centers_mhz"]), comb_sigma_mhz=base["comb_sigma_mhz"], comb_amp=base["comb_amp"],
        time_burst_center=base["time_burst_center"], time_burst_sigma=base["time_burst_sigma"],
        science_band=tuple(base["science_band"]), protected_band=tuple(base["protected_band"]),
    )

    rows=[]
    trial_seeds = np.arange(args.trials) + int(cfg.get("seed", 0))
    for snr in grid["snr_science_grid"]:
        for comb_amp in grid["comb_amp_grid"]:
            for mode in grid["overlap_mode"]:
                overlap = (mode == "overlap")

                for s in trial_seeds:
                    # modify params per-condition
                    params2 = params
                    params2 = SimParams(**{**params.__dict__, "comb_amp": float(comb_amp)})
                    # interpret "snr" as scaling science amplitude relative to noise
                    # science_amp = snr * noise_sigma (simple control)
                    params2 = SimParams(**{**params2.__dict__, "science_amp": float(snr) * float(params2.noise_sigma)})

                    D, S_true, meta = make_synthetic(params2, seed=int(s), overlap=overlap)
                    freqs = meta["freqs"]

                    # Standard SVD
                    D_svd, _ = svd_subtract_rank_r(D, r=int(grid.get("rank_r", 1)))
                    S_svd = estimate_signal_1d(D_svd)
                    bias_svd = relative_bias_percent(S_svd, S_true, eps=1e-5)
                    agg_svd = aggregate_in_band(freqs, bias_svd, params2.science_band)

                    # FWSVD
                    W = make_weight_matrix(freqs, params2.T, params2.science_band, params2.protected_band,
                                           w_core=float(grid.get("w_core", 0.1)),
                                           w_prot=float(grid.get("w_prot", 0.3)))
                    D_fws, _ = fwsvd_subtract_rank1(D, W, iters=args.iters)
                    S_fws = estimate_signal_1d(D_fws)
                    bias_fws = relative_bias_percent(S_fws, S_true, eps=1e-5)
                    agg_fws = aggregate_in_band(freqs, bias_fws, params2.science_band)

                    # NMF (rank-1) baseline
                    try:
                        D_nmf, _ = nmf_subtract_rank1(D, seed=int(s))
                        S_nmf = estimate_signal_1d(D_nmf)
                        bias_nmf = relative_bias_percent(S_nmf, S_true, eps=1e-5)
                        agg_nmf = aggregate_in_band(freqs, bias_nmf, params2.science_band)
                    except Exception:
                        agg_nmf = {"median": float('nan'), "mean": float('nan'), "max": float('nan')}

                    # ICA (1-component) baseline
                    try:
                        D_ica, _ = ica_subtract_rank1(D, seed=int(s))
                        S_ica = estimate_signal_1d(D_ica)
                        bias_ica = relative_bias_percent(S_ica, S_true, eps=1e-5)
                        agg_ica = aggregate_in_band(freqs, bias_ica, params2.science_band)
                    except Exception:
                        agg_ica = {"median": float('nan'), "mean": float('nan'), "max": float('nan')}

                    # RPCA (PCP via IALM) baseline: interpret sparse component as RFI estimate
                    try:
                        D_rpca, _, _ = rpca_subtract_sparse(D, max_iter=400)
                        S_rpca = estimate_signal_1d(D_rpca)
                        bias_rpca = relative_bias_percent(S_rpca, S_true, eps=1e-5)
                        agg_rpca = aggregate_in_band(freqs, bias_rpca, params2.science_band)
                    except Exception:
                        agg_rpca = {"median": float('nan'), "mean": float('nan'), "max": float('nan')}

                    rows.append({
                        "seed": int(s),
                        "snr": float(snr),
                        "comb_amp": float(comb_amp),
                        "overlap": bool(overlap),
                        "svd_median": agg_svd["median"], "svd_mean": agg_svd["mean"], "svd_max": agg_svd["max"],
                        "fws_median": agg_fws["median"], "fws_mean": agg_fws["mean"], "fws_max": agg_fws["max"],
                        "nmf_median": agg_nmf["median"], "nmf_mean": agg_nmf["mean"], "nmf_max": agg_nmf["max"],
                        "ica_median": agg_ica["median"], "ica_mean": agg_ica["mean"], "ica_max": agg_ica["max"],
                        "rpca_median": agg_rpca["median"], "rpca_mean": agg_rpca["mean"], "rpca_max": agg_rpca["max"],
                    })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "summary.csv"), index=False)

    group_cols = ["snr", "comb_amp", "overlap"]
    agg = df.groupby(group_cols).agg({
        "svd_median":"median","svd_mean":"mean","svd_max":"max",
        "fws_median":"median","fws_mean":"mean","fws_max":"max",
        "nmf_median":"median","nmf_mean":"mean","nmf_max":"max",
        "ica_median":"median","ica_mean":"mean","ica_max":"max",
        "rpca_median":"median","rpca_mean":"mean","rpca_max":"max",
    }).reset_index()
    agg.to_csv(os.path.join(args.out, "summary_agg.csv"), index=False)

    save_json(os.path.join(args.out, "meta.json"), {"defaults": cfg, "grid": grid, "trials": args.trials})
    print("Saved:", os.path.join(args.out, "summary.csv"))

if __name__ == "__main__":
    main()
