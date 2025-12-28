# src/qa_real_data.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np

@dataclass
class QAResult:
    ok: bool
    reasons: list[str]
    stats: Dict[str, Any]

def _finite_frac(x: np.ndarray) -> float:
    return float(np.mean(np.isfinite(x)))

def _nan_frac(x: np.ndarray) -> float:
    return float(np.mean(~np.isfinite(x)))

def _is_monotonic_increasing(x: np.ndarray) -> bool:
    dx = np.diff(x)
    return bool(np.all(dx > 0))

def _safe_percentile(x: np.ndarray, q: float) -> float:
    y = x[np.isfinite(x)]
    if y.size == 0:
        return float("nan")
    return float(np.percentile(y, q))

def validate_real_data(
    D: np.ndarray,
    freq_mhz: Optional[np.ndarray] = None,
    time_s: Optional[np.ndarray] = None,
    *,
    min_T: int = 20,
    min_F: int = 64,
    max_nan_frac: float = 0.5,
    require_monotonic_freq: bool = True,
    band_mhz: Optional[Tuple[float, float]] = None,
    core_mhz: Optional[Tuple[float, float]] = None,
) -> QAResult:
    """
    Validates raw/normalized D(t,f) before running decomposition.

    band_mhz/core_mhz (optional):
      if provided along with freq_mhz, ensure they lie inside freq range.
    """
    reasons: list[str] = []
    stats: Dict[str, Any] = {}

    if not isinstance(D, np.ndarray):
        return QAResult(False, ["D is not a numpy array"], {})

    if D.ndim != 2:
        return QAResult(False, [f"D must be 2D (T,F), got shape={D.shape}"], {})

    T, F = D.shape
    stats["shape"] = {"T": T, "F": F}

    if T < min_T:
        reasons.append(f"T too small: T={T} < {min_T}")
    if F < min_F:
        reasons.append(f"F too small: F={F} < {min_F}")

    nan_frac = _nan_frac(D)
    fin_frac = 1.0 - nan_frac
    stats["finite"] = {"finite_frac": fin_frac, "nan_frac": nan_frac}

    if nan_frac > max_nan_frac:
        reasons.append(f"Too many NaNs/flags: nan_frac={nan_frac:.3f} > {max_nan_frac}")

    # Robust dynamic range summary (helps choose plotting / clipping)
    stats["value_summary"] = {
        "p1": _safe_percentile(D, 1),
        "p50": _safe_percentile(D, 50),
        "p99": _safe_percentile(D, 99),
    }

    # Frequency axis checks
    if freq_mhz is not None:
        if not isinstance(freq_mhz, np.ndarray):
            reasons.append("freq_mhz is not a numpy array")
        else:
            freq_mhz = freq_mhz.astype(np.float64, copy=False)
            stats["freq"] = {
                "F_freq": int(freq_mhz.size),
                "min_mhz": float(np.nanmin(freq_mhz)),
                "max_mhz": float(np.nanmax(freq_mhz)),
            }
            if freq_mhz.size != F:
                reasons.append(f"freq_mhz length mismatch: len(freq_mhz)={freq_mhz.size} != F={F}")
            if require_monotonic_freq and freq_mhz.size == F:
                if not _is_monotonic_increasing(freq_mhz):
                    reasons.append("freq_mhz is not strictly increasing (needs sorting or correct extraction)")

            # Band/core in-range checks
            def _check_range(name: str, r: Tuple[float, float]):
                lo, hi = r
                fmin, fmax = stats["freq"]["min_mhz"], stats["freq"]["max_mhz"]
                if not (fmin <= lo < hi <= fmax):
                    reasons.append(f"{name} outside freq range: {r} not within [{fmin:.3f},{fmax:.3f}]")

            if band_mhz is not None:
                _check_range("band_mhz", band_mhz)
            if core_mhz is not None:
                _check_range("core_mhz", core_mhz)

    # Time axis checks (optional)
    if time_s is not None:
        if not isinstance(time_s, np.ndarray):
            reasons.append("time_s is not a numpy array")
        else:
            stats["time"] = {"T_time": int(time_s.size)}
            if time_s.size != T:
                reasons.append(f"time_s length mismatch: len(time_s)={time_s.size} != T={T}")

    ok = (len(reasons) == 0)
    return QAResult(ok=ok, reasons=reasons, stats=stats)


def mhz_to_index(freq_mhz: np.ndarray, lo_mhz: float, hi_mhz: float) -> Tuple[int, int]:
    """
    Returns [i_lo, i_hi) indices for a MHz range, assuming freq_mhz is increasing.
    """
    i_lo = int(np.searchsorted(freq_mhz, lo_mhz, side="left"))
    i_hi = int(np.searchsorted(freq_mhz, hi_mhz, side="right"))
    return i_lo, i_hi
