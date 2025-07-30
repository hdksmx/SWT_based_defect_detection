

"""
golden_set.py
=============

Utility functions for building and using the *Golden Set* (GS) – a collection
of "always-good" pixel coordinates extracted from a defect‑free reference tile.

The GS is stored on disk as a CSV containing two integer columns: **row,col**.

Public API
----------
build_gs(mask: NDArray[np.bool_], csv_path: Path, compress=True) -> set[Coord]
load_gs(csv_path: Path) -> set[Coord]
filter_by_gs(coords: NDArray[int_], gs: set[Coord]) -> NDArray[int_]

* mask – boolean array where True marks pixels **without** defects.
* coords – candidate coordinates (row, col) from current inspection frame.

The GS is kept in memory as a Python *set* of (row, col) tuples for O(1) lookup.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Set, Tuple

import numpy as np

from io_utils import timer

__all__ = ["Coord", "build_gs", "load_gs", "filter_by_gs"]

Coord = Tuple[int, int]


# --------------------------------------------------------------------------- #
# CSV helpers
# --------------------------------------------------------------------------- #
def _write_csv(coords: Iterable[Coord], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open(mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("row", "col"))
        writer.writerows(coords)


def _read_csv(csv_path: Path) -> Set[Coord]:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    with csv_path.open(mode="r", newline="") as f:
        reader = csv.DictReader(f)
        coords = {(int(row["row"]), int(row["col"])) for row in reader}
    return coords


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
@timer
def build_gs(mask: np.ndarray, csv_path: Path) -> Set[Coord]:
    """
    Build Golden Set from a boolean "good‑pixel" mask and save to CSV.

    Parameters
    ----------
    mask : np.ndarray(bool_)
        True where pixel is *not* a defect (reference image).
    csv_path : Path
        Destination CSV filename, e.g. `PROJECT_ROOT/gs/ref_tile.csv`.

    Returns
    -------
    set[Coord]
        Golden set coordinates.
    """
    if mask.dtype != bool:
        raise ValueError("mask must be boolean array.")
    if mask.ndim != 2:
        raise ValueError("mask must be 2‑D.")

    coords: Set[Coord] = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    _write_csv(coords, csv_path)
    return coords


def load_gs(csv_path: Path) -> Set[Coord]:
    """
    Load Golden Set CSV as a *set* of (row, col) tuples.
    """
    return _read_csv(csv_path)


@timer
def filter_by_gs(coords: np.ndarray, gs: Set[Coord]) -> np.ndarray:
    """
    Remove candidate pixels that are listed in the Golden Set.

    Parameters
    ----------
    coords : np.ndarray
        (N, 2) int32 array (row, col) of candidate pixels.
    gs : set[Coord]
        Pre‑loaded Golden Set.

    Returns
    -------
    np.ndarray
        Filtered coordinates (may be zero-length).
    """
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2).")

    keep_mask = [tuple(coord) not in gs for coord in coords]
    return coords[np.asarray(keep_mask, dtype=bool)]