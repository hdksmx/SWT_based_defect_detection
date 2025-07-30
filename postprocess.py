

"""
postprocess.py
==============

Connected‑component clustering and defect‑type classification.

After the WTMS + GS filtering stage, we obtain a **binary mask** whose True
pixels represent defect candidates.  This module groups those pixels into
regions, computes basic morphology metrics, and tags each region as either
*scratch* (elongated) or *particle/contamination* (compact).

Public API
----------
cluster_labels(mask: NDArray[bool_],
               min_area: int = 5) -> list["DefectRegion"]

classify_defect(region: "skimage.measure._regionprops.RegionProperties",
                ecc_thr: float = 0.9,
                len_thr: float = 20) -> str

Return value – list of *DefectRegion* dataclass objects:
    label           : connected‑component label id
    defect_type     : "scratch" | "particle"
    area            : pixel count
    centroid        : (row, col) tuple
    bbox            : (min_row, min_col, max_row, max_col)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from skimage.measure import label, regionprops

from io_utils import timer

__all__ = ["DefectRegion", "cluster_labels", "classify_defect"]


@dataclass(slots=True)
class DefectRegion:
    """Lightweight container for region summary statistics."""
    label: int
    defect_type: str
    area: int
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)


# --------------------------------------------------------------------------- #
# Classification helper
# --------------------------------------------------------------------------- #
def classify_defect(
    region,  # skimage RegionProperties
    ecc_thr: float = 0.9,
    len_thr: float = 20,
) -> str:
    """
    Heuristic defect classifier.

    Parameters
    ----------
    region : skimage.measure._regionprops.RegionProperties
    ecc_thr : float, default 0.9
        Minimum eccentricity to be considered a scratch.
    len_thr : float, default 20
        Minimum major‑axis length for scratch.

    Returns
    -------
    str
        "scratch" if elongated, else "particle".
    """
    if region.eccentricity >= ecc_thr and region.major_axis_length >= len_thr:
        return "scratch"
    return "particle"


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
@timer
def cluster_labels(mask: np.ndarray, min_area: int = 5) -> List[DefectRegion]:
    """
    Connected‑component labelling + region analysis.

    Parameters
    ----------
    mask : np.ndarray(bool)
        Binary mask where True marks candidate defect pixels.
    min_area : int, default 5
        Regions smaller than this are discarded as noise.

    Returns
    -------
    list[DefectRegion]
        One summary object per surviving region.
    """
    if mask.dtype != bool:
        raise ValueError("mask must be boolean array.")
    if mask.ndim != 2:
        raise ValueError("mask must be 2‑D.")

    lbl_img = label(mask, connectivity=2)  # 8‑connected
    regions = regionprops(lbl_img)

    results: List[DefectRegion] = []
    for reg in regions:
        if reg.area < min_area:
            continue

        defect_type = classify_defect(reg)
        results.append(
            DefectRegion(
                label=reg.label,
                defect_type=defect_type,
                area=int(reg.area),
                centroid=tuple(reg.centroid),
                bbox=tuple(reg.bbox),
            )
        )

    return results