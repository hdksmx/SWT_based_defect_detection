"""
wavelet.py
==========

Wavelet decomposition & Wavelet Transform Modulus (WTM) utilities for the
**wafer_wtms** pipeline.

Public API
----------
swt2(img: NDArray[np.uint8],
     wavelet: str = "coif6",
     level: int = 2) -> list[CoeffTuple]

wtm(coeffs: CoeffTuple) -> NDArray[np.float32]

Where
------
CoeffTuple = tuple[np.ndarray, np.ndarray, np.ndarray]  # (cH, cV, cD)

Design Notes
------------
* PyWavelets (`pywt`) is used as the backend.
* Input image is expected to be **uint8** (output of prefilter).
* The module does NOT save any intermediate files — visualisation is handled
  by the caller via the `io_utils.debug_path` helper.
* Stationary Wavelet Transform is used instead of Discrete Wavelet Transform.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
import pywt

from io_utils import timer

__all__ = ["CoeffTuple", "swt2", "wtm"]

CoeffTuple = Tuple[np.ndarray, np.ndarray, np.ndarray]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _validate_input(img: np.ndarray) -> None:
    if img.dtype != np.float32:
        raise ValueError("swt2 expects input dtype float32 (prefilter output).")
    if img.ndim != 2:
        raise ValueError("Input image must be 2‑D gray scale.")


# --------------------------------------------------------------------------- #
# Public functions
# --------------------------------------------------------------------------- #
@timer
def swt2(
    img: np.ndarray,
    wavelet: str = "coif6",
    level: int = 2,
) -> List[CoeffTuple]:
    """
    Perform 2‑D Stationary Wavelet Transform.

    Parameters
    ----------
    img : np.ndarray
        2‑D float32 array (H×W).
    wavelet : str, default 'coif6'
        Wavelet family / filter name recognised by PyWavelets.
    level : int, default 2
        Decomposition level. Must be ≥1.

    Returns
    -------
    list[CoeffTuple]
        List of detail coefficient tuples per level.
        Index 0 -> level‑1 (finest), 1 -> level‑2, ...
        Each tuple: (cH, cV, cD) with dtype=float32.
    """
    _validate_input(img)

    if level < 1:
        raise ValueError("level must be ≥ 1")

    coeffs_all = pywt.swt2(
        data=img.astype(np.float32), wavelet=wavelet, level=level, start_level=0, axes=(-2, -1)
    )

    # coeffs_all[0] = (cA1,(cH1,cV1,cD1)) at level-1 (finest), coeffs_all[1] at level-2, etc.
    detail_coeffs_fine_first: List[CoeffTuple] = [
        tuple(map(np.asarray, t[1])) for t in coeffs_all
    ]

    return detail_coeffs_fine_first


def wtm(coeffs: CoeffTuple) -> np.ndarray:
    """
    Compute the Wavelet Transform Modulus (WTM) from a set of detail coefficients.

    Parameters
    ----------
    coeffs : CoeffTuple
        (cH, cV, cD) arrays for the same decomposition level.

    Returns
    -------
    np.ndarray
        float32 array of the same shape, holding the local wavelet energy:
            sqrt(cH^2 + cV^2 + cD^2)

    Notes
    -----
    * No scaling / normalisation is applied here; paper's algorithm sums
      absolute energies later at WTMS stage.
    """
    cH, cV, cD = coeffs
    if not (cH.shape == cV.shape == cD.shape):
        raise ValueError("Coefficient arrays must share the same shape.")

    # Cast to float32 if necessary
    cHf = cH.astype(np.float32, copy=False)
    cVf = cV.astype(np.float32, copy=False)
    cDf = cD.astype(np.float32, copy=False)

    # Compute energy modulus
    return np.sqrt(cHf**2 + cVf**2 + cDf**2, dtype=np.float32)


def dwt_quad(img: np.ndarray, wavelet: str = "haar") -> np.ndarray:
    """
    1-level 2-D Stationary Wavelet Transform → 4-분할 uint8 시각화 이미지를 반환.
    입력은 uint8/uint16 Gray, 출력 크기는 입력의 1/2×1/2.
    """
    cA, (cH, cV, cD) = pywt.swt2(img.astype(np.float32), wavelet, level=1, start_level=0, axes=(-2, -1))[0]

    def norm8_percentile(arr, p_low=1, p_high=99):
        vmin, vmax = np.percentile(arr, [p_low, p_high])
        arr_clipped = np.clip(arr, vmin, vmax)
        out = np.zeros_like(arr_clipped, dtype=np.uint8)
        cv2.normalize(arr_clipped, out, 0, 255, cv2.NORM_MINMAX)
        return out
    
    def norm8_robust(arr):
        """
        Robust normalization for wavelet detail coefficients.
        Uses absolute value percentile to preserve edge information.
        """
        abs_arr = np.abs(arr)
        vmax = np.percentile(abs_arr, 95)  # Use 95th percentile of absolute values
        
        if vmax == 0 or vmax < 1e-8:  # Handle zero or near-zero coefficients
            return np.zeros_like(arr, dtype=np.uint8)
        
        # Map [-vmax, vmax] to [0, 255] with center at 127.5
        arr_norm = np.clip(arr / vmax, -1, 1)
        return ((arr_norm + 1) * 127.5).astype(np.uint8)
    
    def norm8_approximation(arr):
        """
        Normalization specifically for approximation coefficients.
        Uses full range normalization for smooth approximation data.
        """
        vmin, vmax = arr.min(), arr.max()
        
        if vmax == vmin:  # Handle constant arrays
            return np.full_like(arr, 127, dtype=np.uint8)
        
        # Linear mapping from [vmin, vmax] to [0, 255]
        arr_norm = (arr - vmin) / (vmax - vmin)
        return (arr_norm * 255).astype(np.uint8)

    # Use different normalization strategies for different coefficient types
    # cA (approximation): smooth image, use full range normalization
    # cH, cV, cD (details): sparse edge info, use robust normalization
    top    = np.hstack([norm8_approximation(cA), norm8_robust(cH)])
    bottom = np.hstack([norm8_robust(cV), norm8_robust(cD)])



    for name, b in zip(("cA", "cH", "cV", "cD"), (cA, cH, cV, cD)):
        print(name, b.min(), b.max(), b.std())



    return np.vstack([top, bottom])