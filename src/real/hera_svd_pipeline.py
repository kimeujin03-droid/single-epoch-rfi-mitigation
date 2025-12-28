
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# For standalone execution, load D_real here
obj = pd.read_pickle("/content/HERA_04-03-2022_all.pkl")
X = max([x for x in obj if isinstance(x, np.ndarray) and x.ndim == 4], key=lambda a: a.size)
sample_idx = 0
D_real = np.squeeze(X[sample_idx, :, :, 0]).astype(np.float32)


# -------- robust scale (MAD)
def robust_scale_mad(D):
    med = np.nanmedian(D)
    mad = np.nanmedian(np.abs(D - med)) + 1e-12
    X = (D - med) / mad
    return X.astype(np.float32), {"median": float(med), "mad": float(mad)}

# -------- freq axis (임시: 50~225 MHz 선형)
def make_freq_axis(F, fmin=50.0, fmax=225.0):
    return np.linspace(fmin, fmax, F, dtype=np.float64)

def mhz_to_index(freq_mhz, lo, hi):
    i0 = int(np.searchsorted(freq_mhz, lo, side="left"))
    i1 = int(np.searchsorted(freq_mhz, hi, side="right"))
    return i0, i1

def gaussian_line(freq_mhz, amp, center_mhz, sigma_mhz):
    x = (freq_mhz - center_mhz) / sigma_mhz
    return (amp * np.exp(-0.5 * x * x)).astype(np.float32)

# -------- SVD
def svd_lowrank(D, r):
    U, s, Vt = np.linalg.svd(D, full_matrices=False)
    L = (U[:, :r] * s[:r]) @ Vt[:r, :]
    return L.astype(np.float32)

def fwsvd_lowrank(D, r, w_core=0.05, core_idx=None):
    W = np.ones_like(D, dtype=np.float32)
    if core_idx is not None:
        i0, i1 = core_idx
        W[:, i0:i1] = w_core
    Wsqrt = np.sqrt(W, dtype=np.float32)
    Lp = svd_lowrank(Wsqrt * D, r)
    return (Lp / (Wsqrt + 1e-12)).astype(np.float32)

def estimate_S(L):
    return np.nanmean(L, axis=0).astype(np.float32)

def bias_metrics(S_hat, S_true, freq_mhz, core, band):
    c0, c1 = mhz_to_index(freq_mhz, core[0], core[1])
    b0, b1 = mhz_to_index(freq_mhz, band[0], band[1])

    def rmse(a, b): return float(np.sqrt(np.mean((a-b)**2)))
    def rel_l2(a, b):
        num = float(np.linalg.norm(a-b))
        den = float(np.linalg.norm(b) + 1e-12)
        return num/den

    return {
        "rmse_core": rmse(S_hat[c0:c1], S_true[c0:c1]),
        "relL2_core": rel_l2(S_hat[c0:c1], S_true[c0:c1]),
        "rmse_band": rmse(S_hat[b0:b1], S_true[b0:b1]),
        "relL2_band": rel_l2(S_hat[b0:b1], S_true[b0:b1]),
        "mean_bias_core": float(np.mean(S_hat[c0:c1] - S_true[c0:c1])),
        "max_abs_bias_core": float(np.max(np.abs(S_hat[c0:c1] - S_true[c0:c1]))),
    }

def plot_heatmap(A, title, fname, clip_p=(1,99)):
    lo = np.nanpercentile(A, clip_p[0])
    hi = np.nanpercentile(A, clip_p[1])
    plt.figure(figsize=(10,4))
    plt.imshow(np.clip(A, lo, hi), aspect="auto", origin="lower")
    plt.title(title)
    plt.xlabel("freq bin")
    plt.ylabel("time bin")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(fname, dpi=180)
    plt.close()

# ====== CONFIG
RANK = 1
W_CORE = 0.05
BAND = (145.0, 147.0)
CORE = (145.6, 146.4)
INJ_AMP = 1.0
INJ_SIGMA = 0.2
INJ_CENTER = 146.0

# ====== PIPELINE
D_norm, sc = robust_scale_mad(D_real)
T, F = D_norm.shape
freq_mhz = make_freq_axis(F)

S_true = gaussian_line(freq_mhz, INJ_AMP, INJ_CENTER, INJ_SIGMA)
D = (D_norm + S_true[None, :]).astype(np.float32)

core_idx = mhz_to_index(freq_mhz, CORE[0], CORE[1])

L_svd = svd_lowrank(D, RANK)
L_fw  = fwsvd_lowrank(D, RANK, w_core=W_CORE, core_idx=core_idx)

S_hat_svd = estimate_S(L_svd)
S_hat_fw  = estimate_S(L_fw)

m_svd = bias_metrics(S_hat_svd, S_true, freq_mhz, CORE, BAND)
m_fw  = bias_metrics(S_hat_fw,  S_true, freq_mhz, CORE, BAND)

print("[SCALE]", sc)
print("[SVD]", m_svd)
print("[FWSVD]", m_fw)

# ====== SAVE FIGS
plot_heatmap(D,     "HERA Real+Injection D(t,f) (scaled)", f"hera_D_heatmap_s{sample_idx}.png")
plot_heatmap(L_svd, f"SVD low-rank L (r={RANK})",          f"hera_L_svd_s{sample_idx}.png")
plot_heatmap(L_fw,  f"FWSVD low-rank L (r={RANK}, w={W_CORE})", f"hera_L_fwsvd_s{sample_idx}.png")

plt.figure(figsize=(10,4))
plt.plot(freq_mhz, S_true, label="S_true (injected)")
plt.plot(freq_mhz, S_hat_svd, label="S_hat (SVD)")
plt.plot(freq_mhz, S_hat_fw, label="S_hat (FWSVD)")
plt.axvspan(CORE[0], CORE[1], alpha=0.2, label="science core")
plt.axvspan(BAND[0], BAND[1], alpha=0.08, label="science band")
plt.xlim(BAND[0]-2, BAND[1]+2)
plt.xlabel("Frequency (MHz)")
plt.ylabel("Scaled power (MAD units)")
plt.title(f"Spectrum Recovery (sample={sample_idx})")
plt.legend()
plt.tight_layout()
plt.savefig(f"hera_spectrum_compare_s{sample_idx}.png", dpi=180)
plt.close()

print("Saved figures:",
      f"hera_D_heatmap_s{sample_idx}.png, hera_L_svd_s{sample_idx}.png, hera_L_fwsvd_s{sample_idx}.png, hera_spectrum_compare_s{sample_idx}.png")
