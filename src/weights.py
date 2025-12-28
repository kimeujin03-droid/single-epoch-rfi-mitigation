from __future__ import annotations
import numpy as np

def make_weight_matrix(freqs: np.ndarray,
                       T: int,
                       science_band: tuple[float, float],
                       protected_band: tuple[float, float],
                       w_core: float,
                       w_prot: float) -> np.ndarray:
    """Piecewise constant frequency weights, broadcast to (T,F).
    - outside science_band: 1.0
    - within science_band but outside protected_band: w_prot
    - within protected_band (core): w_core
    Constraints: 0 < w_core <= w_prot <= 1.
    """
    lo_s, hi_s = science_band
    lo_p, hi_p = protected_band

    Wf = np.ones_like(freqs, dtype=float)
    in_science = (freqs >= lo_s) & (freqs <= hi_s)
    in_prot = (freqs >= lo_p) & (freqs <= hi_p)

    Wf[in_science] = float(w_prot)
    Wf[in_prot] = float(w_core)

    return np.tile(Wf[None, :], (T, 1))
