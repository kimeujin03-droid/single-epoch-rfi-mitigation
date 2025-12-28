# src/dataset_real_injection.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Tuple
from .data_loader import load_real_data, RealData

@dataclass
class InjectConfig:
    amp: float           # injection amplitude (same unit as D after scaling)
    center_mhz: float
    sigma_mhz: float

def make_freq_axis(F: int, f0_mhz: float, df_mhz: float) -> np.ndarray:
    return f0_mhz + df_mhz * np.arange(F, dtype=np.float32)

def gaussian_line(freq_mhz: np.ndarray, amp: float, center_mhz: float, sigma_mhz: float) -> np.ndarray:
    x = (freq_mhz - center_mhz) / sigma_mhz
    return (amp * np.exp(-0.5 * x * x)).astype(np.float32)

def robust_scale(D: np.ndarray, mode: str = "mad") -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Real vis-derived power scale varies wildly across obs.
    We normalize to make injection amplitude interpretable across datasets.
    """
    X = D.copy()
    med = np.nanmedian(X)
    if mode == "mad":
        mad = np.nanmedian(np.abs(X - med)) + 1e-12
        X = (X - med) / mad
        return X.astype(np.float32), {"median": float(med), "mad": float(mad)}
    # fallback
    std = np.nanstd(X) + 1e-12
    X = (X - med) / std
    return X.astype(np.float32), {"median": float(med), "std": float(std)}

def make_dataset_real_injection(
    path: str,
    fmt: str,
    f0_mhz: float,
    df_mhz: float,
    inj: InjectConfig,
    loader_kwargs: Dict[str, Any] | None = None,
    scale_mode: str = "mad",
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Returns:
      D: (T,F) = scaled(real) + S_inj[None,:]
      S_inj: (F,)
      meta: dict
    """
    if loader_kwargs is None:
        loader_kwargs = {}

    rd: RealData = load_real_data(path, fmt=fmt, **loader_kwargs)
    D0 = rd.D  # (T,F)

    # Normalize (critical so injection amp has meaning)
    Dn, scale_meta = robust_scale(D0, mode=scale_mode)

    T, F = Dn.shape
    freq_mhz = rd.freq if rd.freq is not None else make_freq_axis(F, f0_mhz=f0_mhz, df_mhz=df_mhz)

    # Deterministic injection
    rng = np.random.default_rng(seed)
    # (optional) tiny jitter to avoid overfitting to exact center
    center = inj.center_mhz + float(rng.normal(0.0, 0.0))  # keep 0 for now
    S_inj = gaussian_line(freq_mhz, amp=inj.amp, center_mhz=center, sigma_mhz=inj.sigma_mhz)

    D = (Dn + S_inj[None, :]).astype(np.float32)

    meta = {
        "data_source": "real+injection",
        "path": path,
        "fmt": fmt,
        "loader_meta": rd.meta,
        "scale": {"mode": scale_mode, **scale_meta},
        "freq": {"f0_mhz": f0_mhz, "df_mhz": df_mhz, "F": F},
        "injection": {"amp": inj.amp, "center_mhz": center, "sigma_mhz": inj.sigma_mhz},
        "seed": seed,
    }
    return D, S_inj, meta
