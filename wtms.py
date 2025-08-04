

"""
wtms.py
=======

Wavelet Transform Modulus **Sum** and inter‑scale ratio utilities.

Algorithm
---------
For a given candidate pixel *p* and WTM map of scale *s*:

1. Run `candidate.edge_search` to obtain a short path of pixels that follow the
   locally strongest gradient direction within Ωₛ(p).
2. Sum the *original* WTM values at those pixels – **WTMS**.
3. Repeat at scale *s+1*; compute the inter‑scale ratio

        R = WTMS_{s+1} / (WTMS_s + eps)

   High R indicates a defect (energy ratio jump).

Public API
----------
wtms_single_scale(wtm, coord, window_hw=3, max_steps=5) -> float
compute_wtms_array(wtm, coords, window_hw=3, max_steps=5) -> NDArray[np.float32]
interscale_ratio(w1, w2, eps=1e-9) -> NDArray[np.float32] | float
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from candidate import edge_search
from io_utils import timer

__all__ = ["wtms_single_scale", "compute_wtms_array", "interscale_ratio"]

Coord = Tuple[int, int]


# --------------------------------------------------------------------------- #
# WTMS helpers
# --------------------------------------------------------------------------- #
@timer
def wtms_single_scale(
    wtm: np.ndarray,
    coord: Coord,
    window_hw: int = 3,
    max_steps: int = 5,
) -> float:
    """
    Compute WTMS for a single coordinate on a single WTM scale.

    Parameters
    ----------
    wtm : np.ndarray
        2-D WTM map of current scale.
    coord : tuple[int, int]
        Seed pixel (row, col).
    window_hw : int, default 3
        Half window size for `edge_search`.
    max_steps : int, default 5
        Maximum steps for `edge_search`.

    Returns
    -------
    float
        Sum of WTM values along path (>=1 pixel).
    """
    path: List[Coord] = edge_search(
        wtm=wtm,
        p=coord,
        window_hw=window_hw,
        max_steps=max_steps,
    )
    return float(np.sum([wtm[y, x] for y, x in path]))


@timer
def compute_wtms_array(
    wtm: np.ndarray,
    coords: Sequence[Coord] | np.ndarray,
    window_hw: int = 3,
    max_steps: int = 5,
) -> np.ndarray:
    """
    Vectorised helper: compute WTMS for a list/array of coordinates.

    Parameters
    ----------
    wtm : np.ndarray
        2-D WTM map.
    coords : sequence of (row, col)
        Candidate pixel positions.
    window_hw, max_steps : int
        Passed through to `wtms_single_scale`.

    Returns
    -------
    np.ndarray
        1‑D array of WTMS values (float32) with same length as *coords*.
    """
    wtms_vals = np.empty(len(coords), dtype=np.float32)
    for i, c in enumerate(coords):
        wtms_vals[i] = wtms_single_scale(
            wtm=wtm,
            coord=tuple(c),
            window_hw=window_hw,
            max_steps=max_steps,
        )
    return wtms_vals


# --------------------------------------------------------------------------- #
# Inter‑scale ratio
# --------------------------------------------------------------------------- #
def interscale_ratio(
    wtms_s: np.ndarray | float,
    wtms_s1: np.ndarray | float,
    eps: float = 1e-9,
) -> np.ndarray | float:
    """
    Compute inter‑scale ratio R between two WTMS arrays (or scalars).

    R = wtms_{s+1} / (wtms_s + eps)

    High R → candidate pixel more likely a defect.

    Parameters
    ----------
    wtms_s, wtms_s1 : float or np.ndarray
        WTMS at scale s and s+1. Must be broadcastable.
    eps : float, default 1e-9
        Small constant to avoid division by zero.

    Returns
    -------
    same type as inputs
        Ratio array / scalar.
    """
    return np.asarray(wtms_s1) / (np.asarray(wtms_s) + eps)