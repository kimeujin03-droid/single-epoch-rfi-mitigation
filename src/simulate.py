from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class SimParams:
    T: int = 60
    F: int = 240
    freq_min_mhz: float = 0.0
    freq_max_mhz: float = 12.0

    science_center_mhz: float = 6.0
    science_sigma_mhz: float = 0.20
    science_amp: float = 0.10

    noise_sigma: float = 0.001

    comb_centers_mhz: tuple[float, ...] = (5.6, 5.8, 6.0, 6.2, 6.4)
    comb_sigma_mhz: float = 0.02
    comb_amp: float = 15.0

    time_burst_center: float = 5.0
    time_burst_sigma: float = 1.0

    # bands
    science_band: tuple[float, float] = (5.5, 6.5)
    protected_band: tuple[float, float] = (5.8, 6.2)

def _gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def make_synthetic(params: SimParams, seed: int = 0, overlap: bool = True):
    """Return (D, S_true_1d, meta) where D is (T,F)."""
    rng = np.random.default_rng(seed)

    times = np.linspace(0.0, 10.0, params.T)
    freqs = np.linspace(params.freq_min_mhz, params.freq_max_mhz, params.F)

    # Science signal (time-invariant)
    S_true_1d = params.science_amp * _gaussian(freqs, params.science_center_mhz, params.science_sigma_mhz)
    D_S = np.tile(S_true_1d, (params.T, 1))

    # Comb interference pattern
    I_pattern = np.zeros_like(freqs)
    centers = list(params.comb_centers_mhz)

    if not overlap:
        # Move comb away from science center (simple non-overlap control)
        shift = 2.0
        centers = [c + shift for c in centers]

    for f0 in centers:
        I_pattern += _gaussian(freqs, f0, params.comb_sigma_mhz)

    time_env = _gaussian(times, params.time_burst_center, params.time_burst_sigma)[:, None]
    D_I = params.comb_amp * np.tile(I_pattern, (params.T, 1)) * time_env

    # Additive noise
    N = rng.normal(0.0, params.noise_sigma, size=D_S.shape)

    D = D_S + D_I + N

    meta = {
        "times": times,
        "freqs": freqs,
        "overlap": overlap,
        "comb_centers_mhz": centers
    }
    return D, S_true_1d, meta
