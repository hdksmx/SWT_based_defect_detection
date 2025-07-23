

"""
reconstruct.py – Post‑processing helpers that *combine* threshold‑processed
SWT coefficient bands and optionally rebuild an image or scratch mask.

The module stays light and purely functional:
    • combine_bands(...) – zero‑out selected sub‑bands
    • reconstruct_image(...) – wrapper around swt_utils.swt2_reconstruct
    • make_mask(...) – OR‑fuse per‑level magnitude maps → binary mask

All functions are independent of global state, returning new objects.
"""

from __future__ import annotations

from typing import List, Literal

import numpy as np
from scipy.stats import median_abs_deviation as mad
from skimage.morphology import remove_small_objects, skeletonize

from swt_utils import (
    CoeffTuple,
    swt2_reconstruct,
    coeff_magnitude,
)

__all__ = [
    "combine_bands",
    "reconstruct_image",
    "make_mask",
]


# ---------------------------------------------------------------------------
# 1. Band‑combination helper
# ---------------------------------------------------------------------------


def combine_bands(
    coeffs: List[CoeffTuple],
    strategy: Literal["keep_all", "keep_hv", "approx_only"] = "keep_all",
) -> List[CoeffTuple]:
    """
    Return a *new* coefficient list where certain detail bands are nulled out.

    Parameters
    ----------
    coeffs : list
        SWT coefficients after thresholding.
    strategy : {'keep_all', 'keep_hv', 'approx_only'}
        keep_all     – leave all detail bands intact.
        keep_hv      – zero the diagonal detail (cD) at every level.
        approx_only  – keep only approximation cA; all detail coeffs to 0.

    Returns
    -------
    list[CoeffTuple]
        Modified coefficients (deep‑copied arrays).
    """
    out: List[CoeffTuple] = []
    for cA, (cH, cV, cD) in coeffs:
        if strategy == "keep_all":
            out.append((cA.copy(), (cH.copy(), cV.copy(), cD.copy())))
        elif strategy == "keep_hv":
            out.append((cA.copy(), (cH.copy(), cV.copy(), np.zeros_like(cD))))
        elif strategy == "approx_only":
            z = np.zeros_like(cH)
            out.append((cA.copy(), (z, z, z)))
        else:
            raise ValueError("Unknown strategy: %s" % strategy)
    return out


# ---------------------------------------------------------------------------
# 2. Image reconstruction wrapper
# ---------------------------------------------------------------------------


def reconstruct_image(
    coeffs: List[CoeffTuple],
    wavelet: str = "sym4",
) -> np.ndarray:
    """
    Wrapper around `swt_utils.swt2_reconstruct` that returns float32 image.

    Typically called after `combine_bands` with desired strategy.

    Returns
    -------
    np.ndarray
        Reconstructed image (float32, same H×W as input).
    """
    return swt2_reconstruct(coeffs, wavelet=wavelet)


# ---------------------------------------------------------------------------
# 3. Scratch / edge binary mask builder
# ---------------------------------------------------------------------------


def make_mask(
    coeffs: List[CoeffTuple],
    *,
    k: float = 2.0,
    norm: int | float = 1,
    include_diagonal: bool = True,
    min_size: int = 50,
    skeleton: bool = False,
) -> np.ndarray:
    """
    Produce a binary OR‑fused mask of 'strong' detail‑band magnitudes.

    Parameters
    ----------
    coeffs : list
        SWT coefficients (thresholded or raw).
    k : float, default 2.0
        Threshold factor T_ℓ = k * MAD(|coeff|).
    norm : {1, 2, np.inf}, default 1
        Aggregation norm passed to `coeff_magnitude`.
    include_diagonal : bool, default True
        Whether to include cD in magnitude calculation.
    min_size : int, default 50
        Remove connected components smaller than this (#pixels).
    skeleton : bool, default False
        If True, skeletonise the final mask (thin to 1‑px lines).

    Returns
    -------
    np.ndarray, dtype=bool
        Final mask the same shape as input image.
    """
    mags = coeff_magnitude(coeffs, p=norm, include_diagonal=include_diagonal)

    level_masks = []
    for mag in mags:
        T = k * mad(mag)
        level_masks.append(mag > T)

    mask = np.logical_or.reduce(level_masks)

    if min_size > 0:
        mask = remove_small_objects(mask, min_size=min_size)

    if skeleton:
        mask = skeletonize(mask).astype(bool)

    return mask