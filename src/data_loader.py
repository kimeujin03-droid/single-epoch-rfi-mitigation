# src/data_loader.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

@dataclass
class RealData:
    D: np.ndarray              # (T,F)
    time: Optional[np.ndarray] # (T,)
    freq: Optional[np.ndarray] # (F,)
    meta: Dict[str, Any]

def load_dynamic_spectrum_fits(path: str, data_hdu: int = 0) -> RealData:
    from astropy.io import fits
    with fits.open(path, memmap=True) as hdul:
        arr = np.squeeze(hdul[data_hdu].data)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D dynspec FITS, got shape={arr.shape}")
    D = arr.astype(np.float32, copy=False)
    return RealData(D=D, time=None, freq=None, meta={"format":"fits_dynspec","path":path})

def load_ms_to_tf(path: str, pol_idx: int = 0, amp_mode: str = "abs2") -> RealData:
    try:
        from casacore.tables import table
    except Exception as e:
        raise ImportError("Install python-casacore to read MS: pip install python-casacore") from e

    tb = table(path, readonly=True, ack=False)
    try:
        data = tb.getcol("DATA")  # (nrow, nchan, npol)
        time = tb.getcol("TIME")  # (nrow,)
        flag = tb.getcol("FLAG")  # (nrow, nchan, npol)
        vis = np.where(flag[:, :, pol_idx], np.nan, data[:, :, pol_idx])

        if amp_mode == "abs2":
            p = (vis.real**2 + vis.imag**2)
        elif amp_mode == "abs":
            p = np.abs(vis)
        else:
            raise ValueError("amp_mode must be 'abs' or 'abs2'")

        uniq_t = np.unique(time)
        T, F = len(uniq_t), p.shape[1]
        D = np.full((T, F), np.nan, dtype=np.float32)
        for i, t in enumerate(uniq_t):
            rows = (time == t)
            D[i] = np.nanmean(p[rows, :], axis=0).astype(np.float32)

        return RealData(D=D, time=uniq_t.astype(np.float64), freq=None,
                        meta={"format":"ms","path":path,"pol_idx":pol_idx,"amp_mode":amp_mode})
    finally:
        tb.close()

def load_real_data(path: str, fmt: str, **kwargs) -> RealData:
    if fmt == "ms":
        return load_ms_to_tf(path, **kwargs)
    if fmt == "fits_dynspec":
        return load_dynamic_spectrum_fits(path, **kwargs)
    raise ValueError(f"Unknown fmt={fmt}")
