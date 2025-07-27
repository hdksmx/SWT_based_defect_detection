

"""
swt_utils.py – Thin wrapper utilities around PyWavelets 2‑D Stationary
Wavelet Transform (SWT) for the wafer‑scratch denoise pipeline.

The goal is to keep this module *stateless* and light‑weight:
it simply provides   (1) forward SWT decomposition,
                     (2) inverse reconstruction, and
                     (3) a helper for magnitude maps.

"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pywt


# ---------------------------------------------------------------------------
# Public API – exported names
# ---------------------------------------------------------------------------

__all__ = [
    "swt2_decompose",
    "swt2_reconstruct",
    "coeff_magnitude",
]

CoeffTuple = Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]


# ---------------------------------------------------------------------------
# Core wrappers
# ---------------------------------------------------------------------------

def _to_float32(img: np.ndarray) -> np.ndarray:
    """Ensure the image is float32 to avoid uint8 overflow in wavelet math."""
    return img.astype(np.float32, copy=False) if img.dtype != np.float32 else img


def swt2_decompose(
    img: np.ndarray,
    wavelet: str = "haar",
    level: int = 2,
) -> List[CoeffTuple]:
    """
    Perform a 2‑D Stationary Wavelet Transform (undecimated).

    Parameters
    ----------
    img : np.ndarray
        2‑D grayscale image (uint8, float32, …).
    wavelet : str, default 'haar'
        Any PyWavelets wavelet name. Symlets give good balance of symmetry
        and orthogonality for scratch‑like edges.
    level : int, default 2
        Number of decomposition scales.

    Returns
    -------
    list[ (cA_l, (cH_l, cV_l, cD_l)) ]   where l = 1 … level
        • Level 1 is the finest (highest spatial frequency).
        • All coefficient arrays are the *same size* as `img`.
    """
    if level < 1:
        raise ValueError("`level` must be >= 1")
    img_f = _to_float32(img)
    # axes default=(-2,-1) ensures last two dims are transformed
    coeffs = pywt.swt2(img_f, wavelet=wavelet, level=level)
    return coeffs


def swt2_reconstruct(
    coeffs: List[CoeffTuple],
    wavelet: str = "haar",
) -> np.ndarray:
    """
    Inverse 2‑D SWT (ISWT) reconstruction.

    Parameters
    ----------
    coeffs : list of tuples
        Output of `swt2_decompose`.
    wavelet : str, default 'haar'
        Wavelet family used during decomposition.

    Returns
    -------
    np.ndarray
        Reconstructed image (float32).
    """
    if not coeffs:
        raise ValueError("`coeffs` must be a non‑empty list")
    rec = pywt.iswt2(coeffs, wavelet=wavelet)
    return rec.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Helper: magnitude map(s)
# ---------------------------------------------------------------------------

def coeff_magnitude(
    coeffs: List[CoeffTuple],
    p: int | float = 1,
    include_diagonal: bool = True,
) -> List[np.ndarray]:
    """
    Compute per‑level magnitude maps from SWT detail coefficients.

    Parameters
    ----------
    coeffs : list
        Output from `swt2_decompose`.
    p : {1, 2, np.inf}, default 1
        Norm type:
        - 1 : |cH| + |cV| (+ |cD| if include_diagonal)
        - 2 : sqrt(cH^2 + cV^2 (+ cD^2))
        - np.inf : max(|cH|, |cV|, |cD|)
    include_diagonal : bool, default True
        If False, ignore `cD` (useful when diagonal scratch is rare/noisy).

    Returns
    -------
    list[np.ndarray]
        Each element is a 2‑D array (same shape as input image) containing
        the aggregated magnitude for one scale.
    """
    mags = []
    for cA, (cH, cV, cD) in coeffs:
        if include_diagonal:
            components = (cH, cV, cD)
        else:
            components = (cH, cV)

        if p == 1:
            mag = sum(np.abs(c) for c in components)
        elif p == 2:
            mag = np.sqrt(sum(c ** 2 for c in components))
        elif p == np.inf:
            mag = np.maximum.reduce([np.abs(c) for c in components])
        else:
            raise ValueError("`p` must be 1, 2, or np.inf")
        mags.append(mag)
    return mags