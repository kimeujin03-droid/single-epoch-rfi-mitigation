from __future__ import annotations
import os
import sys
import argparse
from dataclasses import fields
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulate import SimParams, make_synthetic, generate_science_line
from src.weights import make_weight_matrix
from src.methods import svd_subtract_rank_r, fwsvd_subtract_rank1, nmf_subtract_rank1, ica_subtract_rank1, rpca_subtract_sparse
from src.metrics import estimate_signal_1d, relative_bias_percent, aggregate_in_band
from src.io_utils import load_json, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/sweep_mc_final")
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--config", default="configs/defaults.json")
    ap.add_argument("--grid", default="configs/sweep_mc.json")
    ap.add_argument("--iters", type=int, default=150)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = load_json(args.config)
    grid = load_json(args.grid)
    rows = []
    # Only pass keys to SimParams that are actually defined as fields on the dataclass
    simparam_field_names = {f.name for f in fields(SimParams)}
    trial_seeds = np.arange(args.trials) + int(cfg.get("seed", 0))

    for snr in grid["snr_science_grid"]:
        for comb_amp in grid["comb_amp_grid"]:
            for mode in grid["overlap_mode"]:
                overlap = (mode == "overlap")

                # Filter config so that only SimParams fields are forwarded; keys like
                # w_core / w_prot are used for weighting but are not part of SimParams.
                sim_cfg = {k: v for k, v in cfg.items() if k in simparam_field_names}

                p = SimParams(**{
                    **sim_cfg,
                    "comb_amp": float(comb_amp),
                    "science_amp": float(snr),
                    "science_pedestal_amp": cfg.get("science_pedestal_amp", 0.03),
                    "slope_amp": cfg.get("slope_amp", 0.03),
                    "ripple_amp": cfg.get("ripple_amp", 0.03),
                    "ripple_period_mhz": cfg.get("ripple_period_mhz", 1.5),
                })
                for s in trial_seeds:
                    D, S_true, meta = make_synthetic(p, seed=int(s), overlap=overlap)
                    freqs = meta["freqs"]

                    D_svd, _ = svd_subtract_rank_r(D, r=int(grid.get("rank_r", 1)))
                    S_svd = estimate_signal_1d(D_svd)
                    agg_svd = aggregate_in_band(freqs, relative_bias_percent(S_svd, S_true, eps=1e-5), tuple(cfg["science_band"]))

                    W = make_weight_matrix(freqs, D.shape[0], tuple(cfg["science_band"]), tuple(cfg["protected_band"]),
                                           w_core=float(grid.get("w_core", 0.01)), w_prot=float(grid.get("w_prot", 1.0)))
                    D_fws, _ = fwsvd_subtract_rank1(D, W, iters=args.iters)
                    S_fws = estimate_signal_1d(D_fws)
                    agg_fws = aggregate_in_band(freqs, relative_bias_percent(S_fws, S_true, eps=1e-5), tuple(cfg["science_band"]))

                    try:
                        D_nmf, _ = nmf_subtract_rank1(D, seed=int(s))
                        agg_nmf = aggregate_in_band(freqs, relative_bias_percent(estimate_signal_1d(D_nmf), S_true, eps=1e-5), tuple(cfg["science_band"]))
                    except Exception:
                        agg_nmf = {"median": np.nan, "mean": np.nan, "max": np.nan}
                    try:
                        D_ica, _ = ica_subtract_rank1(D, seed=int(s))
                        agg_ica = aggregate_in_band(freqs, relative_bias_percent(estimate_signal_1d(D_ica), S_true, eps=1e-5), tuple(cfg["science_band"]))
                    except Exception:
                        agg_ica = {"median": np.nan, "mean": np.nan, "max": np.nan}
                    try:
                        D_rpca, _ = rpca_subtract_sparse(D, max_iter=200)
                        agg_rpca = aggregate_in_band(freqs, relative_bias_percent(estimate_signal_1d(D_rpca), S_true, eps=1e-5), tuple(cfg["science_band"]))
                    except Exception:
                        agg_rpca = {"median": np.nan, "mean": np.nan, "max": np.nan}

                    rows.extend([
                        {"method":"SVD","seed":int(s),"snr":float(snr),"comb_amp":float(comb_amp),"overlap":bool(overlap), **agg_svd},
                        {"method":"FWSVD","seed":int(s),"snr":float(snr),"comb_amp":float(comb_amp),"overlap":bool(overlap), **agg_fws},
                        {"method":"NMF","seed":int(s),"snr":float(snr),"comb_amp":float(comb_amp),"overlap":bool(overlap), **agg_nmf},
                        {"method":"ICA","seed":int(s),"snr":float(snr),"comb_amp":float(comb_amp),"overlap":bool(overlap), **agg_ica},
                        {"method":"RPCA","seed":int(s),"snr":float(snr),"comb_amp":float(comb_amp),"overlap":bool(overlap), **agg_rpca},
                    ])

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    summary_agg = df.groupby("method")[["median","mean","max"]].median().reset_index()
    summary_agg.to_csv(os.path.join(args.out, "summary_agg.csv"), index=False)
    save_json(os.path.join(args.out, "meta.json"), {"args": vars(args), "trials": args.trials})
    print(f"[DONE] Results saved to: {args.out}")


if __name__ == "__main__":
    main()
