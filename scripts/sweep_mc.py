from __future__ import annotations
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 프로젝트 루트를 PYTHONPATH에 추가하여 src 모듈 로드 허용
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulate import SimParams, make_synthetic, generate_science_line
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
from src.data_loader import load_real_data
from src.qa_real_data import validate_real_data

def get_closest_idx(f_arr, target_mhz):
    """주파수 배열에서 특정 MHz에 가장 가까운 인덱스를 반환 (동적 인덱싱)"""
    return int(np.argmin(np.abs(f_arr - target_mhz)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/sweep_mc_final")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--config", default="configs/defaults.json")
    ap.add_argument("--grid", default="configs/sweep_mc.json")
    ap.add_argument("--iters", type=int, default=150)
    # 실데이터(LOFAR/MeerKAT) 관련 인자
    ap.add_argument("--use_real_data", action="store_true", help="실데이터 모드 활성화")
    ap.add_argument("--real_data_key", type=str, default=None, help="configs/real_data_paths.json 내 키값")
    ap.add_argument("--real_data_cfg", default="configs/real_data_paths.json")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = load_json(args.config)
    grid = load_json(args.grid)

    # 1. 데이터 준비 (실데이터 vs 가상데이터)
    if args.use_real_data:
        if not args.real_data_key:
            raise ValueError("--use_real_data 모드에서는 --real_data_key가 필수입니다.")
        
        print(f"--- [REAL DATA MODE] Loading: {args.real_data_key} ---")
        rd_paths = load_json(args.real_data_cfg)
        item = rd_paths[args.real_data_key]
        rd = load_real_data(item["path"], fmt=item.get("fmt", "ms"))
        
        D_raw = rd.D
        freqs = rd.freq
        
        # QA 검증 및 시각화
        qa = validate_real_data(
            D_raw, freq_mhz=freqs,
            band_mhz=tuple(cfg["science_band"]),
            core_mhz=tuple(cfg["protected_band"])
        )

        if not qa.ok:
            plt.figure(figsize=(10, 6))
            plt.imshow(np.isnan(D_raw).T, aspect='auto', cmap='Reds', origin='lower')
            plt.title(f"QA FAILED: {args.real_data_key}\nReasons: {qa.reasons}")
            plt.xlabel("Time Bin")
            plt.ylabel("Frequency Bin")
            plt.colorbar(label="Is Masked (NaN)")
            fail_path = os.path.join(args.out, f"QA_FAIL_{args.real_data_key}.png")
            plt.savefig(fail_path)
            print(f"[CRITICAL] QA Failed. Diagnostic plot saved to {fail_path}")
            sys.exit(2)
        
        # 동적 주파수 인덱스 매핑 (실데이터 해상도에 최적화)
        sci_band_idx = (get_closest_idx(freqs, cfg["science_band"][0]), get_closest_idx(freqs, cfg["science_band"][1]))
        prot_band_idx = (get_closest_idx(freqs, cfg["protected_band"][0]), get_closest_idx(freqs, cfg["protected_band"][1]))
        print(f"[QA OK] Dynamic Mapping: Core {cfg['protected_band']} MHz -> Idx {prot_band_idx}")
    else:
        print("--- [SYNTHETIC MODE] ---")
        # 가상 데이터는 config에 정의된 인덱스 그대로 사용
        sci_band_idx = tuple(cfg.get("science_band_idx", (110, 130)))
        prot_band_idx = tuple(cfg.get("protected_band_idx", (116, 124)))

    rows = []
    trial_seeds = np.arange(args.trials) + int(cfg.get("seed", 0))

    # 2. 파라미터 그리드 스윕 루프
    for snr in grid["snr_science_grid"]:
        for comb_amp in grid["comb_amp_grid"]:
            for mode in grid["overlap_mode"]:
                overlap = (mode == "overlap")
                print(f"Running: SNR={snr}, CombAmp={comb_amp}, Overlap={overlap}")

                for s in trial_seeds:
                    # 데이터 생성/주입
                    if args.use_real_data:
                        # 실데이터 노이즈에 가상 신호 주입 (Injection)
                        # snr 인자를 주입 신호의 진폭으로 사용
                        S_true = generate_science_line(freqs, amp=float(snr), center=cfg["science_center_mhz"])
                        D = D_raw + S_true[None, :]
                    else:
                        # 완전 가상 데이터 생성
                        p = SimParams(**{**cfg, "comb_amp": float(comb_amp), "science_amp": float(snr)})
                        D, S_true, meta = make_synthetic(p, seed=int(s), overlap=overlap)
                        freqs = meta["freqs"]

                    # --- 알고리즘 1: Standard SVD ---
                    D_svd, _ = svd_subtract_rank_r(D, r=int(grid.get("rank_r", 1)))
                    S_svd = estimate_signal_1d(D_svd)
                    bias_svd = relative_bias_percent(S_svd, S_true, eps=1e-5)
                    agg_svd = aggregate_in_band(freqs, bias_svd, sci_band_idx)

                    # --- 알고리즘 2: FWSVD (제안 방법) ---
                    W = make_weight_matrix(freqs, D.shape[0], sci_band_idx, prot_band_idx,
                                           w_core=float(grid.get("w_core", 0.01)),
                                           w_prot=float(grid.get("w_prot", 1.0)))
                    D_fws, _ = fwsvd_subtract_rank1(D, W, iters=args.iters)
                    S_fws = estimate_signal_1d(D_fws)
                    bias_fws = relative_bias_percent(S_fws, S_true, eps=1e-5)
                    agg_fws = aggregate_in_band(freqs, bias_fws, sci_band_idx)

                    # --- 알고리즘 3: NMF Baseline ---
                    try:
                        D_nmf, _ = nmf_subtract_rank1(D, seed=int(s))
                        S_nmf = estimate_signal_1d(D_nmf)
                        bias_nmf = relative_bias_percent(S_nmf, S_true, eps=1e-5)
                        agg_nmf = aggregate_in_band(freqs, bias_nmf, sci_band_idx)
                    except:
                        agg_nmf = {"median": np.nan, "mean": np.nan, "max": np.nan}

                    # --- 알고리즘 4: ICA Baseline ---
                    try:
                        D_ica, _ = ica_subtract_rank1(D, seed=int(s))
                        S_ica = estimate_signal_1d(D_ica)
                        bias_ica = relative_bias_percent(S_ica, S_true, eps=1e-5)
                        agg_ica = aggregate_in_band(freqs, bias_ica, sci_band_idx)
                    except:
                        agg_ica = {"median": np.nan, "mean": np.nan, "max": np.nan}

                    # --- 알고리즘 5: RPCA Baseline ---
                    try:
                        D_rpca, _, _ = rpca_subtract_sparse(D, max_iter=400)
                        S_rpca = estimate_signal_1d(D_rpca)
                        bias_rpca = relative_bias_percent(S_rpca, S_true, eps=1e-5)
                        agg_rpca = aggregate_in_band(freqs, bias_rpca, sci_band_idx)
                    except:
                        agg_rpca = {"median": np.nan, "mean": np.nan, "max": np.nan}

                    # 결과 데이터 수집
                    rows.append({
                        "seed": int(s), "snr": float(snr), "comb_amp": float(comb_amp), "overlap": bool(overlap),
                        "svd_median": agg_svd["median"], "svd_mean": agg_svd["mean"], "svd_max": agg_svd["max"],
                        "fws_median": agg_fws["median"], "fws_mean": agg_fws["mean"], "fws_max": agg_fws["max"],
                        "nmf_median": agg_nmf["median"], "nmf_mean": agg_nmf["mean"], "nmf_max": agg_nmf["max"],
                        "ica_median": agg_ica["median"], "ica_mean": agg_ica["mean"], "ica_max": agg_ica["max"],
                        "rpca_median": agg_rpca["median"], "rpca_mean": agg_rpca["mean"], "rpca_max": agg_rpca["max"],
                    })

    # 3. 결과 저장
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    
    # 그룹화 통계 저장 (Reproducibility Contract 준수) [cite: 256]
    group_cols = ["snr", "comb_amp", "overlap"]
    summary_agg = df.groupby(group_cols).mean().reset_index()
    summary_agg.to_csv(os.path.join(args.out, "summary_agg.csv"), index=False)
    
    save_json(os.path.join(args.out, "meta.json"), {"args": vars(args), "trials": args.trials})
    print(f"\n[DONE] Results saved to: {args.out}")

if __name__ == "__main__":
    main()
