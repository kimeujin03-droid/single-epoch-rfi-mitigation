#!/usr/bin/env python3
"""
Synthetic rank sweep for single-epoch low-rank cleaning.

Generates a controlled synthetic dynamic spectrum with:
- a smooth science component
- comb-like narrowband RFI
- weak ripple structure
- Gaussian noise

Then performs a rank sweep with truncated SVD and plots:
1) residual contamination proxy
2) science distortion proxy

Output:
    outputs/figure_rank_sweep_synthetic.png
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def truncated_svd_Lk(X: np.ndarray, k: int) -> np.ndarray:
    """Return rank-k truncated SVD reconstruction."""
    if k <= 0:
        return np.zeros_like(X)

    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    Uk = U[:, :k]
    sk = s[:k]
    Vtk = Vt[:k, :]
    return (Uk * sk) @ Vtk


def make_synthetic(T: int = 200, F: int = 128, seed: int = 11):
    """
    Build a controlled synthetic dynamic spectrum.

    Returns
    -------
    X : (T, F) observed matrix
    S_true : (T, F) true science component
    R_true : (T, F) true RFI component
    """
    rng = np.random.default_rng(seed)

    t = np.linspace(0.0, 1.0, T)
    f = np.linspace(0.0, 1.0, F)

    # ------------------------------------------------------------------
    # Science component: broad smooth envelope + narrow protected feature
    # ------------------------------------------------------------------
    core_center = 0.43
    core_width = 0.06
    band_center = 0.50

    science_core = np.exp(-0.5 * ((f - core_center) / core_width) ** 2)
    science_env = 1.0 / (1.0 + 5.0 * (f - band_center) ** 2)
    S_nu = 0.07 * science_env + 0.03 * science_core
    S_true = np.outer(np.ones(T), S_nu)

    # ------------------------------------------------------------------
    # RFI component: intermittent comb lines + weak ripple
    # ------------------------------------------------------------------
    comb_freqs = np.array([0.12, 0.27, 0.43, 0.62, 0.80])
    comb_idx = np.array([int(np.argmin(np.abs(f - cf))) for cf in comb_freqs])

    comb = np.zeros((T, F), dtype=float)
    for i, idx in enumerate(comb_idx):
        duty = 0.12 + 0.05 * (i % 3)
        on = (rng.random(T) < duty).astype(float)
        amp = 0.60 if i % 2 == 0 else 0.35
        comb[:, idx] += on * amp

    # weak structured ripple
    comb += 0.03 * np.sin(2.0 * np.pi * 2.0 * np.outer(t, np.linspace(0.0, 1.0, F)))

    R_true = comb

    # ------------------------------------------------------------------
    # Noise
    # ------------------------------------------------------------------
    noise = 0.008 * rng.normal(size=(T, F))

    # Observed data
    X = S_true + R_true + noise
    return X, S_true, R_true


def compute_rank_sweep(
    X: np.ndarray,
    S_true: np.ndarray,
    R_true: np.ndarray,
    k_max: int = 15,
):
    """
    Compute rank-sweep proxies.

    residual_contamination:
        ||R_true - L_k|| / ||R_true||
        smaller is better

    science_distortion:
        ||L_k|| / ||S_true||
        interpreted as how much low-rank subtraction intrudes into the
        science-bearing structure; smaller is better

    Returns
    -------
    ks, contamination_vals, distortion_vals
    """
    ks = np.arange(1, k_max + 1)
    contamination_vals = []
    distortion_vals = []

    R_norm = np.linalg.norm(R_true, ord="fro") + 1e-16
    S_norm = np.linalg.norm(S_true, ord="fro") + 1e-16

    for k in ks:
        Lk = truncated_svd_Lk(X, k)

        # Proxy 1: residual contamination
        # If Lk were a perfect contaminant model, this would be small.
        contamination = np.linalg.norm(R_true - Lk, ord="fro") / R_norm

        # Proxy 2: science distortion
        # How much of the low-rank removed structure overlaps the science scale.
        distortion = np.linalg.norm(Lk, ord="fro") / S_norm

        contamination_vals.append(contamination)
        distortion_vals.append(distortion)

    return ks, np.array(contamination_vals), np.array(distortion_vals)


def plot_rank_sweep(
    ks: np.ndarray,
    contamination_vals: np.ndarray,
    distortion_vals: np.ndarray,
    outpath: str,
):
    """Make the rank-sweep plot."""
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 8,
        }
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        ks,
        contamination_vals,
        marker="o",
        lw=1.8,
        ms=4,
        label="Rank Sweep — residual contamination",
    )
    ax.plot(
        ks,
        distortion_vals,
        marker="s",
        lw=1.8,
        ms=4,
        label="Rank Sweep — science distortion",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Rank k")
    ax.set_ylabel("Proxy (lower is better; log scale)")
    ax.set_title("Synthetic rank sweep: contamination vs distortion")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="best")

    # Example shaded operational region
    ax.axvspan(2, 3, color="gray", alpha=0.12)
    ymax = max(np.nanmax(contamination_vals), np.nanmax(distortion_vals))
    ax.text(
        2.5,
        ymax * 0.55,
        "Operational knee (k≈2–3)",
        ha="center",
        va="center",
        fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    outdir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "figure_rank_sweep_synthetic.png")

    X, S_true, R_true = make_synthetic(T=200, F=128, seed=11)
    ks, contamination_vals, distortion_vals = compute_rank_sweep(
        X, S_true, R_true, k_max=15
    )
    plot_rank_sweep(ks, contamination_vals, distortion_vals, outpath)

    print(f"WROTE: {outpath}")


if __name__ == "__main__":
    main()
