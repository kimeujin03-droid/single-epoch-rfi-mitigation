from __future__ import annotations
import numpy as np

def estimate_signal_1d(D_clean: np.ndarray) -> np.ndarray:
    """Default estimator: time-average across t (ignores NaNs)."""
    return np.nanmean(D_clean, axis=0)

def relative_bias_percent(S_hat: np.ndarray, S_true: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Relative bias in percent with a robust epsilon floor.

    Note: In our toy setup, S_true can be extremely small outside the signal core.
    The epsilon floor prevents numerical blow-ups (division by ~0) while retaining
    the 'catastrophic bias' behavior when S_true is small-but-nonzero.
    """
    # IMPORTANT: eps is an *explicit* absolute floor (same unit as S_true).
    # We use max(|S_true|, eps) to avoid denominator-driven blow-ups.
    denom = np.maximum(np.abs(S_true), float(eps))
    return 100.0 * np.abs(S_hat - S_true) / denom

def aggregate_in_band(freqs: np.ndarray, bias: np.ndarray, band: tuple[float, float]):
    """Aggregate bias stats in a frequency band + report finite coverage."""
    lo, hi = band
    sel = (freqs >= lo) & (freqs <= hi)
    b = bias[sel]
    finite = np.isfinite(b)
    if np.any(finite):
        med = float(np.nanmedian(b))
        mean = float(np.nanmean(b))
        mx = float(np.nanmax(b))
    else:
        med = mean = mx = float("nan")
    return {
        "median": med,
        "mean": mean,
        "max": mx,
        "coverage": float(np.mean(finite)) if b.size else 0.0,
        "n": int(np.sum(sel))
    }
