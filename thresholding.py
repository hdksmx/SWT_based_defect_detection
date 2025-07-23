

"""
thresholding.py – Robust σ estimation and coefficient thresholding helpers
for the SWT denoise / scratch‑detection pipeline.

This module is totally *stateless*; it only manipulates the coefficient
structures produced by `swt_utils.swt2_decompose`.

Public API
----------
estimate_sigma(coeffs, diag=True, clip=None)
    Return per‑level noise σ̂ using the MAD/0.6745 rule.
apply_threshold(coeffs, sigmas, k=3.0, mode='soft')
    Hard/soft threshold the *detail* bands in‑place and return a new list.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pywt
from scipy.stats import median_abs_deviation as mad

from swt_utils import CoeffTuple

__all__ = [
    "estimate_sigma",
    "apply_threshold",
]


# ---------------------------------------------------------------------------
# σ estimation – Donoho & Johnstone rule
# ---------------------------------------------------------------------------


def estimate_sigma(
    coeffs: Sequence[CoeffTuple],
    *,
    diag: bool = True,
    clip: float | None = None,
) -> np.ndarray:
    """
    Estimate Gaussian noise σ for each SWT level via MAD.

    Parameters
    ----------
    coeffs : list
        Output of `swt_utils.swt2_decompose`.
    diag : bool, default True
        Whether to include diagonal detail |cD| in the magnitude map. If False,
        only |cH| + |cV| is used (useful if cD is often dominated by texture).
    clip : float or None
        If given, clip the resulting σ̂ to at most this value (helps prevent
        huge thresholds on some pathological tiles).

    Returns
    -------
    np.ndarray, shape = (n_levels,)
        Robust σ̂ for each decomposition level (float32).
    """
    sigmas = []
    for lvl, (cA, (cH, cV, cD)) in enumerate(coeffs, start=1):
        if diag:
            mag = np.abs(cH) + np.abs(cV) + np.abs(cD)
        else:
            mag = np.abs(cH) + np.abs(cV)

        sigma = mad(mag, scale="normal")  # equals MAD/0.6745
        if clip is not None:
            sigma = np.minimum(sigma, clip)
        sigmas.append(np.float32(sigma))
    return np.asarray(sigmas, dtype=np.float32)


# ---------------------------------------------------------------------------
# Thresholding helpers
# ---------------------------------------------------------------------------


def _thr(x: np.ndarray, t: float, mode: str) -> np.ndarray:
    """Core threshold operator (vectorised)."""
    if mode == "hard":
        return np.where(np.abs(x) > t, x, 0.0).astype(np.float32, copy=False)
    elif mode == "soft":
        return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)
    else:
        raise ValueError("`mode` must be 'hard' or 'soft'")


def apply_threshold(
    coeffs: Sequence[CoeffTuple],
    sigmas: Sequence[float],
    *,
    k: float = 3.0,
    mode: str = "soft",
) -> List[CoeffTuple]:
    """
    Apply level‑dependent thresholding to detail coefficients.

    Parameters
    ----------
    coeffs : list
        Coefficient list from `swt_utils.swt2_decompose`.
    sigmas : sequence[float]
        Per‑level σ̂ from `estimate_sigma`.
    k : float, default 3.0
        Threshold scale factor Tℓ = k * σℓ.
    mode : {'hard', 'soft'}, default 'soft'
        Thresholding mode:
        'hard' – set |x|<=T to 0
        'soft' – shrink by T (denoising artefacts ↓, slight bias ↑)

    Returns
    -------
    list
        New coefficient list (same structure) with thresholded detail bands;
        approximation coefficients (`cA`) are unchanged.
    """
    if len(coeffs) != len(sigmas):
        raise ValueError("`sigmas` length must match `coeffs` levels")

    out: List[CoeffTuple] = []
    for (cA, (cH, cV, cD)), sigma in zip(coeffs, sigmas):
        T = k * sigma
        cH_t = _thr(cH, T, mode)
        cV_t = _thr(cV, T, mode)
        cD_t = _thr(cD, T, mode)
        out.append((cA, (cH_t, cV_t, cD_t)))
    return out