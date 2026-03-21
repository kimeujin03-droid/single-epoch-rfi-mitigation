#!/usr/bin/env python3
"""
Reproduce Figure: Mixed singular modes in single-epoch SVD
Shows that science + RFI appear together in each singular mode.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------
# Output
# -----------------------------
outdir = "outputs"
os.makedirs(outdir, exist_ok=True)
outpath = os.path.join(outdir, "figure4_rank_sweep_synthetic.png")

# -----------------------------
# Synthetic data parameters
# -----------------------------
T = 256   # time samples
F = 128   # frequency channels
rng = np.random.default_rng(2)

t = np.linspace(0, 1, T)
f = np.linspace(0, 1, F)

# -----------------------------
# Science-like smooth component
# -----------------------------
time_fg = 1.0 + 0.7 * np.cos(2 * np.pi * 0.4 * t)
freq_fg = 1.0 / (1.0 + 4.0 * (f - 0.5)**2)
science = np.outer(time_fg, freq_fg)

# -----------------------------
# Narrowband RFI comb
# -----------------------------
rfi = np.zeros((T, F))
comb_idx = np.arange(15, F, 14)

for idx in comb_idx:
    burst = (rng.random(T) > 0.88).astype(float)
    rfi[:, idx] += burst * (0.8 + 0.4 * rng.random(T))

# slight spectral ripple (realistic coupling)
rfi += 0.05 * np.sin(2*np.pi*(3*t[:,None] + 5*f[None,:]))

# -----------------------------
# Noise
# -----------------------------
noise = 0.02 * rng.normal(size=(T, F))

# -----------------------------
# Final single-epoch data
# -----------------------------
X = science + rfi + noise

# -----------------------------
# SVD
# -----------------------------
U, S, Vt = np.linalg.svd(X, full_matrices=False)

# -----------------------------
# Plot first 3 singular modes
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
cmap = "RdBu_r"

for i in range(3):
    mode = np.outer(U[:, i] * S[i], Vt[i, :])

    ax = axes[i]
    im = ax.imshow(
        mode,
        aspect="auto",
        origin="lower",
        cmap=cmap,
    )

    ax.set_title(f"Singular Mode {i+1}\n(Science + RFI Mixed)", fontsize=10)
    ax.set_xlabel("Frequency channel")
    if i == 0:
        ax.set_ylabel("Time sample")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle(
    "Mixed singular modes in a single-epoch SVD",
    fontsize=13,
    y=1.05
)

fig.tight_layout()
fig.savefig(outpath, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"[OK] wrote: {outpath}")
