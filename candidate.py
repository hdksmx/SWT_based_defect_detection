

"""
candidate.py
============

Candidate pixel sampling and direction-aware edge‑search for WTMS calculation.

Public API
----------
sample_candidates(wtm_lvl1: NDArray[np.float32],
                  std_factor: float = 3.0) -> NDArray[np.int32]

edge_search(wtm: NDArray[np.float32],
            p: tuple[int, int],
            window_hw: int = 3,
            max_steps: int = 5) -> list[tuple[int, int]]

Notes
-----
* All coordinates follow (row, col) order (i, j).
* The window Ω_s(p) from the paper is implemented as a clipped square ROI
  centred at **p** with half‑width *window_hw*.
* The edge‑search heuristic follows the *locally strongest neighbour* path,
  collecting at most `max_steps + 1` pixels (including the seed point).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from io_utils import timer

__all__ = ["sample_candidates", "edge_search"]

Coord = Tuple[int, int]


# --------------------------------------------------------------------------- #
# Candidate sampling
# --------------------------------------------------------------------------- #
@timer
def sample_candidates(
    wtm_lvl1: np.ndarray,
    std_factor: float = 3.0,
) -> np.ndarray:
    """
    μ + kσ threshold to preselect defect candidates on finest‑scale WTM map.

    Parameters
    ----------
    wtm_lvl1 : np.ndarray
        Level‑1 WTM map (float32).
    std_factor : float, default 3.0
        Multiplier *k* in μ + kσ.

    Returns
    -------
    np.ndarray
        (N, 2) int32 array of pixel coordinates (row, col).
    """
    if wtm_lvl1.dtype not in (np.float32, np.float64):
        raise ValueError("wtm_lvl1 must be float array.")
    if wtm_lvl1.ndim != 2:
        raise ValueError("wtm_lvl1 must be 2‑D.")

    mu = float(wtm_lvl1.mean())
    sigma = float(wtm_lvl1.std())
    thr = mu + std_factor * sigma

    mask = wtm_lvl1 > thr
    coords = np.argwhere(mask).astype(np.int32)

    return coords


# --------------------------------------------------------------------------- #
# Edge‑search
# --------------------------------------------------------------------------- #
def _valid_coord(shape: Tuple[int, int], y: int, x: int) -> bool:
    h, w = shape
    return 0 <= y < h and 0 <= x < w


# 8‑direction offsets (N, NE, E, SE, S, SW, W, NW)
_DIRS: List[Coord] = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


@timer
def edge_search(
    wtm: np.ndarray,
    p: Coord,
    window_hw: int = 3,
    max_steps: int = 5,
) -> List[Coord]:
    """
    Direction‑aware energy tracing used in WTMS.

    Starting at pixel *p*, iteratively moves to the neighbour (among 8) with the
    highest WTM value **within the square window Ω** until either:

    * maximum steps reached, or
    * no neighbour has strictly larger WTM energy.

    Parameters
    ----------
    wtm : np.ndarray
        WTM map of current scale.
    p : tuple[int, int]
        Seed coordinate (row, col).
    window_hw : int, default 3
        Half‑width of the square search window Ω.
    max_steps : int, default 5
        Maximum iterations (paper uses 5).

    Returns
    -------
    list[tuple[int, int]]
        Ordered list of coordinates visited (including the seed).
    """
    if wtm.ndim != 2:
        raise ValueError("wtm must be 2‑D.")
    if not _valid_coord(wtm.shape, *p):
        raise ValueError(f"Seed {p} is out of bounds.")

    path: List[Coord] = [p]
    y, x = p

    # Pre‑compute window bounds
    y0, y1 = max(0, y - window_hw), min(wtm.shape[0] - 1, y + window_hw)
    x0, x1 = max(0, x - window_hw), min(wtm.shape[1] - 1, x + window_hw)

    for _ in range(max_steps):
        curr_val = wtm[y, x]
        best_val = curr_val
        best_coord: Coord | None = None

        # Search 8 neighbours
        for dy, dx in _DIRS:
            ny, nx = y + dy, x + dx
            if not (y0 <= ny <= y1 and x0 <= nx <= x1):
                continue
            val = wtm[ny, nx]
            if val > best_val:
                best_val = val
                best_coord = (ny, nx)

        if best_coord is None:
            break  # no higher neighbour → local ridge end

        path.append(best_coord)
        y, x = best_coord

    return path