# hera_proxy_validation.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from hera_svd_pipeline import (
    robust_scale_mad, make_freq_axis, mhz_to_index,
    gaussian_line, svd_lowrank, fwsvd_lowrank,
    estimate_S, bias_metrics
)

# ─────────────────────────
# 1. proxy metrics
# ─────────────────────────
def smoothness_metric(spectrum: np.ndarray, idx_range):
    i0, i1 = idx_range
    x = spectrum[i0:i1]
    d2 = x[:-2] - 2*x[1:-1] + x[2:]
    return float(np.sqrt(np.mean(d2**2) + 1e-12))

def residual_energy_metric(D: np.ndarray, L: np.ndarray, idx_range):
    i0, i1 = idx_range
    R = D - L
    R_core = R[:, i0:i1]
    return float(np.sqrt(np.mean(R_core**2) + 1e-12))

# ─────────────────────────
# 2. config
# ─────────────────────────
FILE_PATH = "/content/HERA_04-03-2022_all.pkl"  # 필요하면 수정
SAMPLE_IDX = 0

RANKS = [1, 2, 3, 4, 5, 8, 10, 15]
BAND = (145.0, 147.0)
CORE = (145.6, 146.4)
INJ_AMP = 1.0
INJ_SIGMA = 0.2
INJ_CENTER = 146.0
W_CORE = 0.05

# ─────────────────────────
# 3. load data & build D
# ─────────────────────────
obj = pd.read_pickle(FILE_PATH)
X = max([x for x in obj if isinstance(x, np.ndarray) and x.ndim == 4],
        key=lambda a: a.size)
D_real = np.squeeze(X[SAMPLE_IDX, :, :, 0]).astype(np.float32)

D_norm, sc = robust_scale_mad(D_real)
T, F = D_norm.shape
freq_mhz = make_freq_axis(F)

S_true = gaussian_line(freq_mhz, INJ_AMP, INJ_CENTER, INJ_SIGMA)
D = (D_norm + S_true[None, :]).astype(np.float32)

core_idx = mhz_to_index(freq_mhz, CORE[0], CORE[1])
band_idx = mhz_to_index(freq_mhz, BAND[0], BAND[1])

# ─────────────────────────
# 4. rank sweep
# ─────────────────────────
records = []

for r in RANKS:
    L_svd = svd_lowrank(D, r)
    L_fw  = fwsvd_lowrank(D, r, w_core=W_CORE, core_idx=core_idx)

    S_hat_svd = estimate_S(L_svd)
    S_hat_fw  = estimate_S(L_fw)

    m_svd = bias_metrics(S_hat_svd, S_true, freq_mhz, CORE, BAND)
    m_fw  = bias_metrics(S_hat_fw,  S_true, freq_mhz, CORE, BAND)

    sm_svd_core = smoothness_metric(S_hat_svd, core_idx)
    sm_fw_core  = smoothness_metric(S_hat_fw,  core_idx)

    re_svd_core = residual_energy_metric(D, L_svd, core_idx)
    re_fw_core  = residual_energy_metric(D, L_fw,  core_idx)

    records.append({
        "rank": r,
        "method": "svd",
        **m_svd,
        "smooth_core": sm_svd_core,
        "resid_core": re_svd_core,
    })
    records.append({
        "rank": r,
        "method": "fwsvd",
        **m_fw,
        "smooth_core": sm_fw_core,
        "resid_core": re_fw_core,
    })

df = pd.DataFrame(records)
df.to_csv("hera_proxy_validation_rank_sweep.csv", index=False)
print(df)

# (옵션) quick scatter plot: proxy vs true bias
plt.figure(figsize=(6,5))
mask_svd = (df["method"]=="svd")
plt.scatter(df.loc[mask_svd,"smooth_core"],
            df.loc[mask_svd,"rmse_core"],
            label="SVD", alpha=0.7)
mask_fw = (df["method"]=="fwsvd")
plt.scatter(df.loc[mask_fw,"smooth_core"],
            df.loc[mask_fw,"rmse_core"],
            label="FWSVD", alpha=0.7)
plt.xlabel("smoothness_core (proxy)")
plt.ylabel("rmse_core (true bias)")
plt.legend()
plt.tight_layout()
plt.savefig("hera_proxy_vs_bias.png", dpi=180)
plt.close()
