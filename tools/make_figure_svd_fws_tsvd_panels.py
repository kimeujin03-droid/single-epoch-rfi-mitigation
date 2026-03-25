

from __future__ import annotations

import os
import sys
from typing import Tuple, Optional, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================================
# Configuration
# ============================================================================
OUTDIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTDIR, exist_ok=True)

OUTHERA = os.path.join(OUTDIR, "figure_svd_fws_tsvd_hera.png")


# ============================================================================
# Core SVD Functions
# ============================================================================

def truncated_svd_Lk(X: np.ndarray, k: int) -> np.ndarray:
    """Standard truncated SVD reconstruction."""
    if k <= 0:
        return np.zeros_like(X)

    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    Uk = U[:, :k]
    sk = s[:k]
    Vtk = Vt[:k, :]

    return (Uk * sk) @ Vtk


def tsvd_Lk(X: np.ndarray, k: int, window: int = 5) -> np.ndarray:
    """Temporally-smoothed SVD (TSVD)."""
    if window <= 1:
        return truncated_svd_Lk(X, k)

    kern = np.ones(window) / float(window)
    # smooth along time (axis 0)
    Xs = np.apply_along_axis(lambda m: np.convolve(m, kern, mode="same"), 0, X)

    return truncated_svd_Lk(Xs, k)


def fwsvd_lowrank(
    D: np.ndarray,
    r: int,
    w_core: float = 0.05,
    core_idx: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Frequency-weighted SVD (FWSVD)."""
    W = np.ones_like(D, dtype=np.float32)

    if core_idx is not None:
        i0, i1 = core_idx
        W[:, i0:i1] = w_core

    Wsqrt = np.sqrt(W, dtype=np.float32)
    Lp = truncated_svd_Lk(Wsqrt * D, r)

    return (Lp / (Wsqrt + 1e-12)).astype(np.float32)


# ============================================================================
# Optimization Helpers
# ============================================================================

def pareto_mask_minimize(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return mask of Pareto-optimal points (minimize both x and y)."""
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    keep = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            # j dominates i if j is better in both dimensions
            if (x[j] <= x[i]) and (y[j] <= y[i]) and ((x[j] < x[i]) or (y[j] < y[i])):
                keep[i] = False
                break

    return keep


def detect_knee(front_x: np.ndarray, front_y: np.ndarray, front_k: np.ndarray) -> int:
    """Detect knee point in Pareto front using curvature."""
    if len(front_x) < 3:
        return int(front_k[len(front_k) // 2])

    lx = np.log(front_x + 1e-16)
    ly = np.log(front_y + 1e-16)
    curv = np.zeros(len(front_x))

    for i in range(1, len(front_x) - 1):
        v1 = np.array([lx[i] - lx[i - 1], ly[i] - ly[i - 1]])
        v2 = np.array([lx[i + 1] - lx[i], ly[i + 1] - ly[i]])

        nv1 = v1 / (np.linalg.norm(v1) + 1e-16)
        nv2 = v2 / (np.linalg.norm(v2) + 1e-16)

        cosang = np.clip(np.dot(nv1, nv2), -1.0, 1.0)
        curv[i] = 1.0 - cosang

    idx = int(np.argmax(curv))
    return int(front_k[idx])


# ============================================================================
# Data Processing
# ============================================================================

def robust_scale_mad(D: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Robust scaling using median absolute deviation."""
    med = np.nanmedian(D)
    mad = np.nanmedian(np.abs(D - med)) + 1e-12
    X = (D - med) / mad

    return X.astype(np.float32), {"median": float(med), "mad": float(mad)}


def gaussian_line(freq_mhz: np.ndarray, amp: float, center_mhz: float, sigma_mhz: float) -> np.ndarray:
    """Generate Gaussian spectral line."""
    x = (freq_mhz - center_mhz) / sigma_mhz
    return (amp * np.exp(-0.5 * x * x)).astype(np.float32)


def make_freq_axis(F: int, fmin: float = 50.0, fmax: float = 225.0) -> np.ndarray:
    """Generate frequency axis in MHz."""
    return np.linspace(fmin, fmax, F, dtype=np.float64)


def eor_window_metrics(
    S_hat: np.ndarray,
    S_true: np.ndarray,
    freq_mhz: np.ndarray,
    eor_start: float = 140.0,
    eor_end: float = 160.0,
) -> dict:
    """Metrics specific to EoR window recovery."""
    eor_mask = (freq_mhz >= eor_start) & (freq_mhz <= eor_end)

    # 1. EoR window bias (L1)
    eor_bias = float(np.mean(np.abs(S_hat[eor_mask] - S_true[eor_mask])))

    # 2. Foreground leakage (relative energy in EoR window)
    num = float(np.mean(np.abs(S_hat[eor_mask])))
    denom = float(np.mean(np.abs(S_true[eor_mask])) + 1e-12)
    fg_leakage = num / denom

    # 3. Simple power spectrum bias proxy
    ps_bias = float(np.abs(np.var(S_hat[eor_mask]) - np.var(S_true[eor_mask])))

    return {
        "eor_bias": eor_bias,
        "fg_leakage": fg_leakage,
        "ps_bias": ps_bias,
    }


# ============================================================================
# Plotting Helper – Top heatmaps + bottom 4 separate spectra
# ============================================================================

def plot_comparison(
    L_s: np.ndarray,
    L_fw: np.ndarray,
    L_t: np.ndarray,
    S_true: np.ndarray,
    freq_axis: np.ndarray,
    k_choice: int,
    best_w: float,
    best_win: int,
    outpath: str,
    eor_window: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Create comparison figure:
    - Top row: 3 heatmaps (SVD, FWSVD, TSVD)
    - Bottom row: 4 *separate* 1D spectra panels
      (S_true / SVD / FWSVD / TSVD), each with its own y-scale.
    """
    plt.rcParams.update({"font.size": 10})
    fig = plt.figure(figsize=(15, 6.0))

    # Top/bottom layout
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.35)

    # -----------------------
    # Top row: heatmaps
    # -----------------------
    gs_top = gs[0].subgridspec(1, 3, wspace=0.3)
    axes_top = [fig.add_subplot(gs_top[0, i]) for i in range(3)]

    # Shared color limits
    clim_low = min(
        np.nanpercentile(L_s, 1),
        np.nanpercentile(L_fw, 1),
        np.nanpercentile(L_t, 1),
    )
    clim_high = max(
        np.nanpercentile(L_s, 99),
        np.nanpercentile(L_fw, 99),
        np.nanpercentile(L_t, 99),
    )

    titles = [
        f"SVD (k={k_choice})",
        f"FWSVD (k={k_choice}, w={best_w:.3g})",
        f"TSVD (k={k_choice}, win={best_win})",
    ]

    last_im = None
    for ax, A, title in zip(axes_top, [L_s, L_fw, L_t], titles):
        im = ax.imshow(
            np.clip(A, clim_low, clim_high),
            aspect="auto",
            origin="lower",
        )
        ax.set_title(title)
        ax.set_xlabel("Freq bin")
        ax.set_ylabel("Time bin")
        last_im = im

    cbar = fig.colorbar(last_im, ax=axes_top, shrink=0.8, aspect=30)
    cbar.set_label("L amplitude")

    # -----------------------
    # Bottom row: 4 separate spectra
    # -----------------------
    gs_bottom = gs[1].subgridspec(1, 4, wspace=0.25)
    S_hat_s = np.nanmean(L_s, axis=0)
    S_hat_fw = np.nanmean(L_fw, axis=0)
    S_hat_t = np.nanmean(L_t, axis=0)
    S_true_1d = S_true if S_true.ndim == 1 else np.nanmean(S_true, axis=0)

    traces = [
        ("S_true", S_true_1d, "b"),
        ("SVD",   S_hat_s,   "C1"),
        (f"FWSVD (w={best_w:.3g})", S_hat_fw,  "C2"),
        ("TSVD",  S_hat_t,   "C3"),
    ]

    for i, (label, data, color) in enumerate(traces):
        ax = fig.add_subplot(gs_bottom[0, i])

        # EoR window highlight (optional, same in all panels)
        if eor_window is not None:
            f0, f1 = eor_window
            ax.axvspan(f0, f1, color="0.9", alpha=0.6, zorder=0)

        ax.plot(freq_axis, data, color=color, linewidth=2.0)
        ax.set_title(label, fontsize=9)
        ax.set_xlim(freq_axis[0], freq_axis[-1])
        ax.grid(True, alpha=0.3, linewidth=0.5)

        if i == 0:
            ax.set_ylabel("Scaled power", fontsize=10)
        else:
            ax.set_yticklabels([])

        ax.set_xlabel("Freq (MHz)", fontsize=10)

    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"WROTE: {outpath}")


# ============================================================================
# HERA Data with EoR-like injection
# ============================================================================

def run_hera(
    outpath: str = OUTHERA,
    sample_idx: int = 0,
    rank: int = 5,
) -> None:
    """Run analysis on real HERA data with realistic EoR-like injection."""
    # 1) Load pickled HERA data
    pkl_path = os.path.join(os.path.dirname(__file__), "..", "HERA_04-03-2022_all.pkl")

    try:
        obj = pd.read_pickle(pkl_path)
    except FileNotFoundError:
        print(f"ERROR: Could not find {pkl_path}", file=sys.stderr)
        print("Skipping HERA analysis.", file=sys.stderr)
        return
    except Exception as e:
        print(f"ERROR loading pickle: {e}", file=sys.stderr)
        return

    arrays_4d = [x for x in obj if isinstance(x, np.ndarray) and x.ndim == 4]
    if not arrays_4d:
        print("ERROR: No 4D arrays found in pickle file", file=sys.stderr)
        return

    X = max(arrays_4d, key=lambda a: a.size)

    if sample_idx >= X.shape[0]:
        print(
            f"ERROR: sample_idx={sample_idx} out of range (max={X.shape[0]-1})",
            file=sys.stderr,
        )
        return

    D_real = np.squeeze(X[sample_idx, :, :, 0]).astype(np.float32)
    T, F = D_real.shape
    freq_mhz = make_freq_axis(F)  # 50–225 MHz

    # 2) Robust scaling
    D_norm, sc = robust_scale_mad(D_real)

    # 3) RFI 위치 추정 (NaN flag 기준)
    rfi_flags = np.isnan(D_real)
    rfi_channels = np.where(np.any(rfi_flags, axis=0))[0]

    EOR_START, EOR_END = 140.0, 160.0
    EOR_CENTER = 150.0

    if rfi_channels.size > 0:
        rfi_freqs = freq_mhz[rfi_channels]
        idx_rfi = int(np.argmin(np.abs(rfi_freqs - EOR_CENTER)))
        base_chan = int(rfi_channels[idx_rfi])

        df = float(np.median(np.diff(freq_mhz)))
        offset_bins = max(1, int(round(2.0 / max(df, 1e-3))))  # ~2 MHz 옆
        inj_chan = int(np.clip(base_chan + offset_bins, 0, F - 1))
        inj_center = float(freq_mhz[inj_chan])

        # window 밖으로 나가면 그냥 센터로
        if not (EOR_START < inj_center < EOR_END):
            inj_center = EOR_CENTER
    else:
        inj_center = EOR_CENTER

    # 4) EoR-like injection (weak line)
    noise_mad = 1.0  # robust_scale_mad 기준
    eor_snr = 0.1    # 10% of MAD
    INJ_AMP = eor_snr * noise_mad
    INJ_SIGMA = 0.5  # MHz

    S_true = gaussian_line(freq_mhz, INJ_AMP, inj_center, INJ_SIGMA)
    D = (D_norm + S_true[None, :]).astype(np.float32)

    # core region: injection ±1 MHz
    core_left = inj_center - 1.0
    core_right = inj_center + 1.0
    core_idx = (
        int(np.searchsorted(freq_mhz, core_left, side="left")),
        int(np.searchsorted(freq_mhz, core_right, side="right")),
    )

    # 5) Methods
    # Baseline SVD
    L_s = truncated_svd_Lk(D, rank)

    # FWSVD (optimize w_core via EoR bias)
    w_grid = [0.01, 0.02, 0.05, 0.1]
    best_w = w_grid[0]
    best_fw = None
    best_fw_bias = np.inf

    for w in w_grid:
        L_fw = fwsvd_lowrank(D, rank, w_core=w, core_idx=core_idx)
        S_hat_fw = np.nanmean(L_fw, axis=0)
        metrics_fw = eor_window_metrics(S_hat_fw, S_true, freq_mhz, EOR_START, EOR_END)
        if metrics_fw["eor_bias"] < best_fw_bias:
            best_fw_bias = metrics_fw["eor_bias"]
            best_w = w
            best_fw = L_fw

    if best_fw is None:
        best_fw = fwsvd_lowrank(D, rank, w_core=best_w, core_idx=core_idx)

    # TSVD (optimize window via EoR bias)
    window_grid = [3, 5, 7, 9, 11]
    best_win = window_grid[0]
    best_t = None
    best_t_bias = np.inf

    for win in window_grid:
        L_t = tsvd_Lk(D, rank, window=win)
        S_hat_t = np.nanmean(L_t, axis=0)
        metrics_t = eor_window_metrics(S_hat_t, S_true, freq_mhz, EOR_START, EOR_END)
        if metrics_t["eor_bias"] < best_t_bias:
            best_t_bias = metrics_t["eor_bias"]
            best_win = win
            best_t = L_t

    if best_t is None:
        best_t = tsvd_Lk(D, rank, window=best_win)

    # 6) Metrics 출력
    S_hat_s = np.nanmean(L_s, axis=0)
    S_hat_fw = np.nanmean(best_fw, axis=0)
    S_hat_t = np.nanmean(best_t, axis=0)

    metrics_s = eor_window_metrics(S_hat_s, S_true, freq_mhz, EOR_START, EOR_END)
    metrics_fw = eor_window_metrics(S_hat_fw, S_true, freq_mhz, EOR_START, EOR_END)
    metrics_t = eor_window_metrics(S_hat_t, S_true, freq_mhz, EOR_START, EOR_END)

    print(f"Injection @ {inj_center:.2f} MHz, amp={INJ_AMP:.3f}")
    print("  SVD   :", metrics_s)
    print("  FWSVD :", metrics_fw, f"(w={best_w})")
    print("  TSVD  :", metrics_t, f"(win={best_win})")

    # 7) Plot (EoR window shaded)
    plot_comparison(
        L_s,
        best_fw,
        best_t,
        S_true,
        freq_mhz,
        k_choice=rank,
        best_w=round(float(best_w), 3),
        best_win=best_win,
        outpath=outpath,
        eor_window=(EOR_START, EOR_END),
    )


def run_hera_eor_injection(outpath: str = OUTHERA,
                           sample_idx: int = 0,
                           rank: int = 3) -> None:
    """HERA snapshot + EoR-like Gaussian injection, k ~ knee (=3)."""
    pkl_path = os.path.join(os.path.dirname(__file__), '..', 'HERA_04-03-2022_all.pkl')

    try:
        obj = pd.read_pickle(pkl_path)
    except Exception as e:
        print(f"[run_hera_eor_injection] ERROR loading {pkl_path}: {e}", file=sys.stderr)
        return

    arrays_4d = [x for x in obj if isinstance(x, np.ndarray) and x.ndim == 4]
    if not arrays_4d:
        print("[run_hera_eor_injection] No 4D arrays in pickle.", file=sys.stderr)
        return

    X = max(arrays_4d, key=lambda a: a.size)
    if sample_idx >= X.shape[0]:
        print(f"[run_hera_eor_injection] sample_idx={sample_idx} out of range.", file=sys.stderr)
        return

    # Raw dynamic spectrum (time x freq) for one baseline/pol
    D_real = np.squeeze(X[sample_idx, :, :, 0]).astype(np.float32)  # (T,F)
    T, F = D_real.shape
    freq_mhz = make_freq_axis(F)  # 50–225 MHz

    # Global robust scaling
    D_norm, _ = robust_scale_mad(D_real)

    # --- EoR window rescaling: window RMS = 1 ---
    EOR_START, EOR_END = 140.0, 160.0
    eor_mask = (freq_mhz >= EOR_START) & (freq_mhz <= EOR_END)
    rms_eor = np.sqrt(np.nanmean(D_norm[:, eor_mask] ** 2))
    if rms_eor <= 0:
        rms_eor = 1.0
    D_norm = (D_norm / rms_eor).astype(np.float32)

    # --- Injection design: EoR window center, SNR ~ 0.3-0.5 ---
    inj_center = 150.0
    inj_sigma = 0.5
    inj_amp = 0.4

    S_true_1d = gaussian_line(freq_mhz, inj_amp, inj_center, inj_sigma)
    S_true = np.broadcast_to(S_true_1d, D_norm.shape)

    D = (D_norm + S_true).astype(np.float32)

    # core index for optimization
    core_half_width = 2.0
    core_idx = (
        int(np.searchsorted(freq_mhz, inj_center - core_half_width, side='left')),
        int(np.searchsorted(freq_mhz, inj_center + core_half_width, side='right')),
    )

    def rmse_core(S_hat: np.ndarray) -> float:
        i0, i1 = core_idx
        return float(np.sqrt(np.mean((S_hat[i0:i1] - S_true_1d[i0:i1]) ** 2)))

    # Low-rank models
    L_s = truncated_svd_Lk(D, rank)

    # FWSVD weight optimization: try scipy.optimize (log-space bounded) first,
    # fall back to a grid search if scipy is not available or optimization fails.
    best_w, best_L_fw, best_rmse_fw = None, None, np.inf

    # bounds for w (positive); search over log10(w) to cover orders of magnitude
    w_min, w_max = 1e-4, 0.5
    try:
        from scipy.optimize import minimize_scalar

        def rmse_for_logw(logw: float) -> float:
            w = 10.0 ** logw
            L_fw_local = fwsvd_lowrank(D, rank, w_core=w, core_idx=core_idx)
            return rmse_core(np.nanmean(L_fw_local, axis=0))

        res = minimize_scalar(
            rmse_for_logw,
            bounds=(np.log10(w_min), np.log10(w_max)),
            method="bounded",
            options={"xatol": 1e-3},
        )

        if res.success:
            best_w = float(10.0 ** res.x)
            best_L_fw = fwsvd_lowrank(D, rank, w_core=best_w, core_idx=core_idx)
            best_rmse_fw = rmse_core(np.nanmean(best_L_fw, axis=0))
        else:
            raise RuntimeError("scipy.optimize failed to converge")
    except Exception:
        # fallback: grid search over a reasonable set of candidates
        w_grid = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
        for w in w_grid:
            L_fw = fwsvd_lowrank(D, rank, w_core=w, core_idx=core_idx)
            rmse = rmse_core(np.nanmean(L_fw, axis=0))
            if rmse < best_rmse_fw:
                best_rmse_fw, best_w, best_L_fw = rmse, w, L_fw
        if best_L_fw is None:
            # corner case: pick small default
            best_w = w_grid[0]
            best_L_fw = fwsvd_lowrank(D, rank, w_core=best_w, core_idx=core_idx)
            best_rmse_fw = rmse_core(np.nanmean(best_L_fw, axis=0))

    # TSVD window optimization (window sizes widened)
    # include larger windows to allow broader low-rank temporal averaging
    win_grid = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    best_win, best_L_t, best_rmse_t = win_grid[0], None, np.inf
    for win in win_grid:
        L_t = tsvd_Lk(D, rank, window=win)
        rmse = rmse_core(np.nanmean(L_t, axis=0))
        if rmse < best_rmse_t:
            best_rmse_t, best_win, best_L_t = rmse, win, L_t
    if best_L_t is None:
        best_L_t = tsvd_Lk(D, rank, window=best_win)

    # Residuals
    R_s = D - L_s
    R_fw = D - best_L_fw
    R_t = D - best_L_t

    print(f"Injection @ {inj_center:.1f} MHz, amp={inj_amp:.3f}")
    sv_rmse = rmse_core(np.nanmean(L_s, axis=0))
    print(f"  SVD   core RMSE = {sv_rmse:.4f}")
    print(f"  FWSVD core RMSE = {best_rmse_fw:.4f} (w={best_w})")
    print(f"  TSVD  core RMSE = {best_rmse_t:.4f} (win={best_win})")

    # prepare numeric metrics for plotting (pass numbers, format in plot)
    metrics = {
        'SVD': float(sv_rmse),
        'FWSVD': float(best_rmse_fw),
        'TSVD': float(best_rmse_t),
    }

    plot_hera_eor_figure(
        R_s, R_fw, R_t,
        S_true_1d,
        freq_mhz,
        rank,
        best_w,
        best_win,
        eor_window=(EOR_START, EOR_END),
        outpath=outpath,
        metrics=metrics,
    )


def plot_hera_eor_figure(
    R_s: np.ndarray,
    R_fw: np.ndarray,
    R_t: np.ndarray,
    S_true_1d: np.ndarray,
    freq_mhz: np.ndarray,
    rank: int,
    best_w: float,
    best_win: int,
    eor_window: Tuple[float, float],
    outpath: str,
    metrics: Optional[Dict[str, str]] = None,
) -> None:
    """HERA EoR figure: 3 residual heatmaps + 4 small spectra panels."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    plt.rcParams.update({"font.size": 10})
    # Use a 2x4 Grid: three main columns for heatmaps/spectra and a narrow
    # right column reserved for the colorbar (top) and metrics box (bottom).
    fig = plt.figure(figsize=(14, 6.0))
    gs = GridSpec(
        2,
        4,
        figure=fig,
        height_ratios=[3.0, 1.7],
        width_ratios=[1.0, 1.0, 1.0, 0.28],
        hspace=0.28,
        wspace=0.18,
    )

    # Top row: residual heatmaps
    mats = [R_s, R_fw, R_t]
    titles = [
        f"SVD (k={rank})",
        f"FWSVD (k={rank}, w={best_w:.3g})",
        f"TSVD (k={rank}, win={best_win})",
    ]

    all_vals = np.concatenate([m.ravel() for m in mats])
    vmin = np.nanpercentile(all_vals, 1)
    vmax = np.nanpercentile(all_vals, 99)

    top_axes = []
    for i, (M, title) in enumerate(zip(mats, titles)):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(
            np.clip(M, vmin, vmax),
            aspect="auto",
            origin="lower",
            extent=[freq_mhz[0], freq_mhz[-1], 0, M.shape[0]],
        )
        ax.set_title(title)
        # keep axis labels only on the leftmost heatmap to reduce clutter
        if i == 0:
            ax.set_xlabel("Freq (MHz)")
            ax.set_ylabel("Time bin")
        else:
            # hide axis label text for middle/right heatmaps but keep tick numbers
            ax.set_xlabel("")
            ax.set_ylabel("")
        top_axes.append(ax)

    # Colorbar axis in the reserved right column (top cell)
    cax = fig.add_subplot(gs[0, 3])
    # draw colorbar vertically and remove axis decorations for a clean look
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Residual amplitude")
    cax.tick_params(labelsize=8)
    # metrics axis (empty ax) in the right column bottom cell
    metrics_ax = fig.add_subplot(gs[1, 3])
    metrics_ax.axis("off")

    # Bottom row: single combined spectra panel (all traces overlaid)
    fmin_plot, fmax_plot = freq_mhz[0], freq_mhz[-1]
    xmask = (freq_mhz >= fmin_plot) & (freq_mhz <= fmax_plot)
    x = freq_mhz[xmask]

    spec_s = np.nanmean(R_s, axis=0)[xmask]
    spec_fw = np.nanmean(R_fw, axis=0)[xmask]
    spec_t = np.nanmean(R_t, axis=0)[xmask]
    s_true = S_true_1d[xmask]

    # Bottom spectra occupies the left three columns (reserve right column for metrics)
    ax = fig.add_subplot(gs[1, 0:3])

    # EoR shading and injection marker
    EOR_START, EOR_END = eor_window
    ax.axvspan(EOR_START, EOR_END, color="0.9", alpha=0.8, zorder=0)
    inj_center = freq_mhz[np.argmax(S_true_1d)]
    ax.axvline(inj_center, color="k", linestyle="--", linewidth=1.0, alpha=0.8, zorder=1)

    # Compute combined y-limits from trace percentiles to avoid extreme outliers
    all_data = np.concatenate([spec_s, spec_fw, spec_t, s_true])
    all_data = all_data[np.isfinite(all_data)]
    if all_data.size == 0:
        ymin, ymax = -1.0, 1.0
    else:
        p1, p99 = np.nanpercentile(all_data, [1, 99])
        margin = 0.12 * (p99 - p1 if p99 > p1 else (abs(p99) + 1e-3))
        ymin, ymax = p1 - margin, p99 + margin
        ymin = min(0.0, ymin)

    # Plot traces with consistent styles
    ax.plot(x, s_true, label="S_true (injected)", color="C0", linewidth=2.0)
    ax.plot(x, spec_s, label="SVD residual", color="C1", linewidth=1.5, alpha=0.9)
    ax.plot(x, spec_fw, label="FWSVD residual", color="C2", linewidth=1.5, alpha=0.9)
    ax.plot(x, spec_t, label="TSVD residual", color="C3", linewidth=1.5, alpha=0.9)

    ax.set_xlim(fmin_plot, fmax_plot)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Freq (MHz)", fontsize=9)
    ax.set_ylabel("Scaled power", fontsize=9)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    # Do not draw the legend on the spectra axis (it can overlap); we'll
    # place the legend into the reserved right-column metrics axis so it
    # doesn't cover the plot area.
    ax.set_title("Spectra (overlaid)", fontsize=10)

    # Draw legend and metrics textbox in the reserved right-column if provided
    # (keeps overlays off the main plots)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # clear metrics_ax and ensure it's empty
        metrics_ax.clear()
        metrics_ax.axis('off')
        # Build combined labels: one line per trace containing the name and
        # the core RMSE (or injection amplitude for S_true). This creates a
        # single "capsule" per trace (color swatch + text) in the reserved box.
        amp = float(np.nanmax(S_true_1d)) if S_true_1d.size > 0 else 0.0
        # two-line labels: first line is the trace name, second line is the numeric
        # metric; this keeps each legend entry compact and aligned in two rows
        labels_with_metrics = [
            f"S_true (injected)\namp = {amp:.3f}",
            f"SVD residual\ncore RMSE = {metrics.get('SVD', float('nan')):.4f}",
            f"FWSVD residual\ncore RMSE = {metrics.get('FWSVD', float('nan')):.4f}\nw = {best_w:.3g}",
            f"TSVD residual\ncore RMSE = {metrics.get('TSVD', float('nan')):.4f}",
        ]

        # put legend at the top of the metrics_ax (inside the reserved box)
        metrics_ax.legend(
            handles,
            labels_with_metrics,
            loc='upper center',
            bbox_to_anchor=(0.5, 0.95),
            frameon=True,
            fontsize=8,
            ncol=1,
            handlelength=1.0,
            handletextpad=0.6,
            labelspacing=0.7,
        )

    # metrics (already shown in the combined legend) — no separate text block

    # bottom panel drawn above; no separate subpanels needed

    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"WROTE HERA EoR figure -> {outpath}")
# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Running HERA analysis (EoR-like injection, k=2)...")
    run_hera_eor_injection(outpath=OUTHERA, rank=2)

    print("\nDone!")
