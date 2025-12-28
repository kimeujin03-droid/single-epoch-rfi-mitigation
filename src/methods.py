from __future__ import annotations
import numpy as np

# Optional baselines (only used by scripts when scikit-learn is installed).
# The repository's default pipelines (SVD/FWSVD/HardMask) do not depend on sklearn.
try:
    from sklearn.decomposition import NMF, FastICA
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False

def svd_subtract_rank_r(D: np.ndarray, r: int = 1):
    """Standard truncated SVD subtraction. Returns (D_clean, I_hat)."""
    U, s, Vt = np.linalg.svd(D, full_matrices=False)
    r = int(max(1, min(r, min(D.shape))))
    I_hat = (U[:, :r] * s[:r]) @ Vt[:r, :]
    D_clean = D - I_hat
    return D_clean, I_hat

def weighted_rank1_als(D: np.ndarray, W: np.ndarray, iters: int = 100, eps: float = 1e-10):
    """Weighted rank-1 approximation using ALS / power-like updates.
    Minimizes || W ⊙ (D - sigma u v^T) ||_F^2 with ||u||=||v||=1.
    Returns (u, v, sigma).
    """
    m, n = D.shape
    rng = np.random.default_rng(0)
    u = rng.normal(size=m)
    v = rng.normal(size=n)
    u /= (np.linalg.norm(u) + eps)
    v /= (np.linalg.norm(v) + eps)

    for _ in range(iters):
        # update v
        u_col = u[:, None]
        num_v = (D * W * u_col).sum(axis=0)
        den_v = (W * (u_col ** 2)).sum(axis=0)
        v = num_v / (den_v + eps)
        v /= (np.linalg.norm(v) + eps)

        # update u
        v_row = v[None, :]
        num_u = (D * W * v_row).sum(axis=1)
        den_u = (W * (v_row ** 2)).sum(axis=1)
        u = num_u / (den_u + eps)
        u /= (np.linalg.norm(u) + eps)

    uv = u[:, None] @ v[None, :]
    sigma = (D * uv * W).sum() / ((uv ** 2) * W).sum().clip(min=eps)
    return u, v, float(sigma)

def fwsvd_subtract_rank1(D: np.ndarray, W: np.ndarray, iters: int = 100):
    """FWSVD rank-1 subtraction using weighted ALS."""
    u, v, sigma = weighted_rank1_als(D, W, iters=iters)
    I_hat = sigma * (u[:, None] @ v[None, :])
    D_clean = D - I_hat
    return D_clean, I_hat

def hard_mask_clean(D: np.ndarray, freqs: np.ndarray, science_band: tuple[float, float], z: float = 3.5):
    """Conservative baseline: excise the satellite-contaminated TIME window.

    We detect a bursty/intermittent RFI interval in time by thresholding the per-time
    energy OUTSIDE the protected science band, then mask those time samples across
    all frequencies (set to NaN). This produces *no reconstruction* inside masked
    times; the downstream 1D spectrum is obtained by time-averaging remaining samples.

    Returns
    -------
    Dm : (T, F) array
        Data with contaminated time samples set to NaN.
    time_mask : (T,) bool
        Boolean mask indicating excised time samples.
    """
    lo, hi = science_band
    sel = (freqs >= lo) & (freqs <= hi)
    outside = ~sel
    if np.sum(outside) < 4:
        outside = np.ones_like(sel, dtype=bool)  # fallback if band covers almost all freqs

    # Robust burst detection in time
    E = np.nanmean(np.square(D[:, outside]), axis=1)
    med = np.nanmedian(E)
    mad = np.nanmedian(np.abs(E - med)) + 1e-12
    thr = med + z * 1.4826 * mad
    time_mask = E >= thr

    # Ensure at least one unmasked sample
    if np.all(time_mask):
        k = max(1, int(0.2 * len(E)))
        keep = np.argsort(E)[:k]
        time_mask = np.ones_like(E, dtype=bool)
        time_mask[keep] = False

    Dm = D.copy().astype(float)
    Dm[time_mask, :] = np.nan
    return Dm, time_mask


# =========================
# Extra baselines (NMF / ICA / RPCA)
# =========================

def nmf_subtract_rank1(D: np.ndarray, seed: int = 0, max_iter: int = 10000,
                       tol: float = 1e-4, solver: str = "mu", verbose: int = 0):
    """개정된 Rank-1 NMF baseline (스케일링 및 솔버 최적화)"""
    if not _HAVE_SK:
        raise ImportError("scikit-learn is required.")

    # 1. 비음수화 및 스케일링 (중요!)
    D_pos = np.clip(D, 0.0, None)
    max_val = np.max(D_pos) + 1e-12
    D_norm = D_pos / max_val  # 0~1 사이로 정규화

    # 2. NMF 모델 설정
    # solver를 'mu'로 변경하고, init을 'random'으로 시도해보는 것도 방법입니다.
    model = NMF(
        n_components=1,
        init="nndsvda", 
        random_state=seed,
        max_iter=max_iter,
        tol=tol,        # tol을 1e-6에서 1e-4로 완화
        solver=solver,  # "cd" 대신 "mu" 사용 고려
        verbose=verbose,
    )

    # 3. 학습 및 역스케일링
    W = model.fit_transform(D_norm)
    H = model.components_
    
    # 정규화했던 값을 다시 원래 크기로 복원
    D_hat = (W @ H) * max_val 
    
    D_clean = D - D_hat
    return D_clean, D_hat


def ica_subtract_rank1(D: np.ndarray, seed: int = 0, max_iter: int = 2000):
    """Rank-1 ICA baseline using FastICA.

    We treat each time sample as an observation and each frequency channel as a feature.
    The mean is *not* subtracted from the cleaned output (we only subtract the 1-component
    reconstruction around the mean), to avoid removing the DC/science baseline.
    """
    if not _HAVE_SK:
        raise ImportError("scikit-learn is required for ICA baseline. Install scikit-learn.")

    X = D.astype(float)
    ica = FastICA(
        n_components=1,
        random_state=seed,
        max_iter=max_iter,
        whiten="unit-variance",
        tol=1e-4,
    )
    S = ica.fit_transform(X)          # (T, 1)
    A = ica.mixing_                   # (F, 1)
    X_comp = S @ A.T                  # (T, F)
    D_clean = X - X_comp
    return D_clean, X_comp


def rpca_pcp_ialm(D: np.ndarray, lam: float | None = None, mu: float | None = None,
                  tol: float = 1e-7, max_iter: int = 500):
    """Robust PCA via Principal Component Pursuit (inexact ALM).

    Solves:  min ||L||_* + lam||S||_1  s.t.  D = L + S

    Returns
    -------
    L : low-rank
    S : sparse

    Reference implementation is adapted from standard PCP/IALM formulations.
    This is intended as a *baseline*; it is not optimized.
    """
    X = D.astype(float)
    m, n = X.shape
    norm_X = np.linalg.norm(X, ord='fro') + 1e-12
    if lam is None:
        lam = 1.0 / np.sqrt(max(m, n))
    if mu is None:
        mu = (m * n) / (4.0 * np.sum(np.abs(X)) + 1e-12)
        mu = max(mu, 1e-3)

    # Initialize
    L = np.zeros_like(X)
    S = np.zeros_like(X)
    Y = np.zeros_like(X)

    def _soft_thresh(A, tau):
        return np.sign(A) * np.maximum(np.abs(A) - tau, 0.0)

    for _ in range(max_iter):
        # (1) Update L via SVT
        U, s, Vt = np.linalg.svd(X - S + (1.0 / mu) * Y, full_matrices=False)
        s_thresh = np.maximum(s - 1.0 / mu, 0.0)
        r = np.sum(s_thresh > 0)
        if r == 0:
            L = np.zeros_like(X)
        else:
            L = (U[:, :r] * s_thresh[:r]) @ Vt[:r, :]

        # (2) Update S via soft threshold
        S = _soft_thresh(X - L + (1.0 / mu) * Y, lam / mu)

        # (3) Dual update
        Z = X - L - S
        Y = Y + mu * Z

        # (4) Check convergence
        err = np.linalg.norm(Z, ord='fro') / norm_X
        if err < tol:
            break

    return L, S


def rpca_subtract_sparse(D: np.ndarray, seed: int = 0, max_iter: int = 300):
    """RPCA baseline: treat the sparse term as interference and subtract it."""
    # seed is accepted for interface consistency (algorithm is deterministic).
    L, S = rpca_pcp_ialm(D, max_iter=max_iter)
    D_clean = D - S
    return D_clean, S


