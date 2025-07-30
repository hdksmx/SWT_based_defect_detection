"""
io_utils.py
===========

I/O utilities + lightweight timing decorator for the **wafer_wtms** project.

The module centralises:

1. **Path management**
   * PROJECT_ROOT  – repository root (`io_utils.py` two levels up).
   * INPUT_DIR     – `<root>/input_img`
   * RESULTS_DIR   – `<root>/results/<timestamp>`
   * DEBUG_DIR     – `<results>/debug_img`

2. **Image helpers**
   * read_image  – returns uint8/uint16 numpy array (gray‑scale).
   * save_image  – writes PNG/TIFF, auto‑creates parent dirs.

3. **Debug file naming**
   * debug_path(step_idx:int, tag:str) → Path where intermediary PNG is saved.

4. **@timer decorator**
   * Measures wall‑clock (time.perf_counter) and logs at the module logger.

The design goal is *zero side‑effects at import time* – directories are created
lazily when first used.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Callable, TypeVar

import cv2
import numpy as np

__all__ = [
    "PROJECT_ROOT",
    "INPUT_DIR",
    "get_input_path",
    "RESULTS_DIR",
    "DEBUG_DIR",
    "ensure_dir",
    "read_image",
    "save_image",
    "debug_path",
    "timer",
]

# --------------------------------------------------------------------------- #
# Path management
# --------------------------------------------------------------------------- #

# repository root = directory containing this file (wafer_wtms)
PROJECT_ROOT: Path = Path(__file__).resolve().parent

INPUT_DIR: Path = PROJECT_ROOT / "input_img"
INPUT_DIR.mkdir(parents=True, exist_ok=True)

# Timestamped results directory (e.g. results/20250729_143015)
_RESULTS_STAMP: str = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR: Path = PROJECT_ROOT / "results" / _RESULTS_STAMP

# sub‑directory for intermediary visualisations
DEBUG_DIR: Path = RESULTS_DIR / "debug_img"

# timing tracker for log file (run.txt)
TIMINGS: dict[str, float] = {}

def ensure_dir(p: Path) -> Path:
    """Create directory *p* (and parents) if it does not exist. Return *p*."""
    p.mkdir(parents=True, exist_ok=True)
    return p


# create base dirs lazily
for _d in (RESULTS_DIR, DEBUG_DIR):
    ensure_dir(_d)

# --------------------------------------------------------------------------- #
# Image helpers
# --------------------------------------------------------------------------- #


def get_input_path(filename: str | Path) -> Path:
    """Return absolute path to *filename* inside INPUT_DIR."""
    return INPUT_DIR / Path(filename).name


def read_image(path: str | Path, as_gray: bool = True) -> np.ndarray:
    """
    Load image from *path*.

    * If *as_gray* is True, forces 1‑channel read even for RGB images.
    * Returns uint8 or uint16 numpy ndarray (HxW or HxWxC).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    flag = cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_UNCHANGED
    img = cv2.imread(str(p), flag)
    if img is None:
        raise IOError(f"cv2 failed to read image: {p}")

    return img


def save_image(img: np.ndarray, path: str | Path) -> None:
    """
    Save *img* to *path* (PNG/TIFF determined by extension).
    Creates target directory hierarchy if necessary.
    
    Handles float32 images by converting [0,1] range to uint8 [0,255].
    """
    p = Path(path)
    ensure_dir(p.parent)
    
    # Handle float32 images by converting to uint8
    if img.dtype == np.float32:
        # Assume float32 data is normalized to [0,1] range
        img_to_save = np.clip(img * 255, 0, 255).astype(np.uint8)
    else:
        img_to_save = img
    
    success = cv2.imwrite(str(p), img_to_save)
    if not success:
        raise IOError(f"cv2 failed to write image: {p}")


# --------------------------------------------------------------------------- #
# Debug utilities
# --------------------------------------------------------------------------- #


def debug_path(step_idx: int, tag: str, ext: str = ".png") -> Path:
    """
    Build filepath like `DEBUG_DIR/02_wtm_lvl1.png`.

    Parameters
    ----------
    step_idx : int
        1‑based pipeline step number.
    tag : str
        Short descriptor (snake_case).
    ext : str
        File extension with dot; default '.png'.
    """
    filename = f"{step_idx:02d}_{tag}{ext}"
    return DEBUG_DIR / filename


# --------------------------------------------------------------------------- #
# Timing decorator
# --------------------------------------------------------------------------- #

_F = TypeVar("_F", bound=Callable[..., "T"])
_T = TypeVar("T")

logger = logging.getLogger("io_utils")
logger.setLevel(logging.INFO)


def timer(fn: _F) -> _F:  # type: ignore[misc]
    """
    Decorator that logs wall‑clock time for *fn* at INFO level.

    Usage
    -----
    >>> @timer
    ... def heavy_func(...):
    ...     ...
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):  # type: ignore[override]
        start = time.perf_counter()
        res = fn(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1e3
        logger.info(f"{fn.__name__} finished in {elapsed_ms:.2f} ms")

        TIMINGS[fn.__name__] = TIMINGS.get(fn.__name__, 0.0) + elapsed_ms

        return res

    return wrapper  # type: ignore[return-value]