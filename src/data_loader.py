# src/data_loader.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np
import json
import os

@dataclass
class RealData:
    D: np.ndarray              # shape (T, F)
    time: Optional[np.ndarray] # shape (T,)
    freq: Optional[np.ndarray] # shape (F,)
    meta: Dict[str, Any]

def _ensure_2d(D: np.ndarray) -> np.ndarray:
    if D.ndim != 2:
        raise ValueError(f"D must be 2D (T,F). Got shape={D.shape}")
    return D

def _nan_guard(D: np.ndarray, fill: float = np.nan) -> np.ndarray:
    # keep NaN by default; your pipeline already supports NaN-safe mean for masking (if implemented)
    return D.astype(np.float32, copy=False)

# -------------------------
# FITS dynamic spectrum loader
# -------------------------
def load_dynamic_spectrum_fits(path: str,
                              data_hdu: int = 0,
                              transpose_if_needed: bool = True,
                              time_axis: int = 0,
                              freq_axis: int = 1) -> RealData:
    """
    For pre-exported dynamic spectrum FITS:
      - data is typically 2D [time, freq] or [freq, time]
    """
    from astropy.io import fits

    with fits.open(path, memmap=True) as hdul:
        arr = hdul[data_hdu].data

        if arr is None:
            raise ValueError(f"No FITS data in HDU={data_hdu}: {path}")

        D = np.array(arr, dtype=np.float32, copy=False)

        # Make 2D if the FITS is stored with extra singleton dims
        D = np.squeeze(D)
        if D.ndim != 2:
            raise ValueError(f"Expected 2D dynamic spectrum in FITS, got {D.ndim}D shape={D.shape}")

        # Ensure (T,F) ordering
        if transpose_if_needed:
            # If user says time_axis=0,freq_axis=1 but file is reversed, swap
            # heuristic: assume longer axis is freq? (optional)
            pass

        # The simplest: treat current as (T,F)
        D = _ensure_2d(D)

        # Optional: parse time/freq from headers if present (often not standardized)
        meta = {"source_format": "FITS_DYNAMIC_SPECTRUM", "path": path}
        return RealData(D=_nan_guard(D), time=None, freq=None, meta=meta)

# -------------------------
# MS (Measurement Set) loader (LOFAR/MeerKAT common)
# -------------------------
def load_ms_to_tf(path: str,
                  field: Optional[str] = None,
                  spw: Optional[int] = None,
                  pol: str = "I",
                  avg_over_baselines: bool = True,
                  amp_mode: str = "abs2") -> RealData:
    """
    Convert MS visibility to a time-frequency power-like matrix.
    Requires python-casacore (preferred) or casatools.
    Output D(t,f) after selecting spw/channel set.

    amp_mode:
      - "abs"  : |V|
      - "abs2" : |V|^2  (often more stable)
    """
    try:
        from casacore.tables import table
    except Exception as e:
        raise ImportError(
            "python-casacore is required for MS loading. "
            "Install: pip install python-casacore"
        ) from e

    meta = {"source_format": "MS", "path": path, "pol": pol, "amp_mode": amp_mode}

    # Open main table
    tb = table(path, readonly=True, ack=False)
    try:
        # Columns commonly needed: DATA, TIME, ANTENNA1/2, FLAG, WEIGHT_SPECTRUM, etc.
        data = tb.getcol("DATA")  # shape: (nrow, nchan, npol) or (nrow, npol, nchan) depending on MS
        time = tb.getcol("TIME")  # shape: (nrow,)
        flag = tb.getcol("FLAG")  # shape matches DATA boolean

        # Basic normalization: map rows -> time bins, channels -> freq bins
        # This is intentionally minimal; you can add binning/grouping later.
        # We assume each row corresponds to a time sample for some baseline.
        # If avg_over_baselines=True, average across baselines per time.

        # Harmonize shapes to (nrow, nchan, npol)
        if data.ndim != 3:
            raise ValueError(f"Unexpected DATA shape: {data.shape}")

        nrow, nchan, npol = data.shape

        # Choose polarization index (simple)
        # For true Stokes I you’d combine XX/YY; here keep minimal and explicit.
        pol_idx = 0
        vis = data[:, :, pol_idx]

        # Apply flag
        vis = np.where(flag[:, :, pol_idx], np.nan, vis)

        if amp_mode == "abs":
            p = np.abs(vis)
        elif amp_mode == "abs2":
            p = (vis.real**2 + vis.imag**2)
        else:
            raise ValueError("amp_mode must be 'abs' or 'abs2'")

        # Aggregate rows into unique time stamps
        # (MS TIME repeats across baselines. Group by TIME.)
        uniq_t = np.unique(time)
        T = len(uniq_t)
        F = nchan

        D = np.full((T, F), np.nan, dtype=np.float32)
        for i, t in enumerate(uniq_t):
            rows = (time == t)
            # average across baselines (rows)
            D[i] = np.nanmean(p[rows, :], axis=0).astype(np.float32)

        return RealData(D=_nan_guard(D), time=uniq_t.astype(np.float64), freq=None, meta=meta)

    finally:
        tb.close()

# -------------------------
# Unified entrypoint
# -------------------------
def load_real_data(path: str, fmt: Optional[str] = None, **kwargs) -> RealData:
    """
    fmt: "fits_dynspec" | "ms" | None(auto by suffix)
    """
    if fmt is None:
        low = path.lower()
        if low.endswith(".fits") or low.endswith(".fit") or low.endswith(".fz"):
            fmt = "fits_dynspec"
        else:
            # MS often is a directory
            fmt = "ms"

    if fmt == "fits_dynspec":
        return load_dynamic_spectrum_fits(path, **kwargs)
    if fmt == "ms":
        return load_ms_to_tf(path, **kwargs)

    raise ValueError(f"Unknown fmt={fmt}")
