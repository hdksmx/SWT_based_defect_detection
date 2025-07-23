"""
utils_io.py – I/O & timing helpers for the SWT‑denoise pipeline.

* All input wafer images are assumed to live in `PROJECT_ROOT/input_img`.
* All outputs are written under `PROJECT_ROOT/results`; debug images go to
  `PROJECT_ROOT/results/debug_img`.

Functions
---------
load_gray(fname, normalize=False)
    Read a grayscale image (8‑bit) and optionally min–max normalise to [0,1].
save_image(img, fname, rescale=True)
    Save `img` to the results folder; if `img` is float, rescale to 0‑255.
save_debug(name, img, rescale=True)
    Convenience wrapper that drops files into the debug folder with `.png`.
step_timer(label)
    Context‑manager to measure & log elapsed time for a code block.
summary()
    Print a nicely formatted timing table to stdout.
reset_timings()
    Clear stored timing data (mainly for unit tests).
"""

from __future__ import annotations

import os
import time
import datetime
from contextlib import contextmanager
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import pywt

# ---------------------------------------------------------------------------
# Project‑level paths
# ---------------------------------------------------------------------------

 # Absolute path to the *package* root (wavelet_code)
PROJECT_ROOT: Path = Path(__file__).resolve().parent
INPUT_DIR: Path = PROJECT_ROOT / "input_img"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
RUN_DIR: Path = RESULTS_DIR / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
DEBUG_DIR: Path = RUN_DIR / "debug_img"

# Ensure output directories exist (create parents first)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RUN_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _resolve_input(fname: str | Path) -> Path:
    """Return absolute path for an input image."""
    p = Path(fname)
    return p if p.is_absolute() else INPUT_DIR / p


def load_gray(fname: str | Path, *, normalize: bool = False) -> np.ndarray:
    """
    Load an image as an 8‑bit grayscale numpy array.

    Parameters
    ----------
    fname : str or Path
        File name (relative to input_img) or absolute path.
    normalize : bool, default False
        If True, convert to float32 and linearly rescale into [0,1].

    Returns
    -------
    np.ndarray
        Image array (uint8 or float32, H×W).
    """
    path = _resolve_input(fname)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")

    if normalize:
        img = img.astype(np.float32)
        img = (img - img.min()) / max(img.ptp(), 1e-8)
    return img


def save_image(img: np.ndarray, fname: str | Path, *, rescale: bool = True) -> Path:
    """
    Save an image under the results directory.

    Parameters
    ----------
    img : np.ndarray
        Image to save. Float arrays are auto‑converted to uint8 if `rescale`.
    fname : str or Path
        File name relative to `results/` (sub‑dirs will be created).
    rescale : bool, default True
        If img is float and <=1, multiply by 255 and clip to 0‑255.

    Returns
    -------
    Path
        Absolute path to the written file.
    """
    out_path = RUN_DIR / fname if not Path(fname).is_absolute() else Path(fname)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out = img
    if rescale and issubclass(img.dtype.type, np.floating):
        out = np.clip(img * (255 if img.max() <= 1.0 else 1.0), 0, 255).astype(
            np.uint8
        )

    cv2.imwrite(str(out_path), out)
    return out_path


def save_debug(name: str, img: np.ndarray, *, rescale: bool = True) -> Path:
    """
    Save an image into the debug folder with automatic `.png` extension.

    Examples
    --------
    >>> save_debug("01_swt_level1_mag", mag_map)
    """
    return save_image(img, DEBUG_DIR / f"{name}.png", rescale=rescale)


def dwt_visualize(img: np.ndarray, wavelet: str = "haar") -> np.ndarray:
    """Return a single 2×2 tile image: [[LL, LH], [HL, HH]] after 1-level DWT."""
    LL, (LH, HL, HH) = pywt.dwt2(img.astype(np.float32), wavelet)
    tiles = [LL, LH, HL, HH]
    tiles = [cv2.normalize(t, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
             for t in tiles]
    top = np.hstack([tiles[0], tiles[1]])
    bot = np.hstack([tiles[2], tiles[3]])
    return np.vstack([top, bot])

# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------

_TIMINGS: Dict[str, float] = {}


@contextmanager
def step_timer(label: str):
    """
    Context‑manager to measure the wall‑time of a processing step.

    Examples
    --------
    >>> with step_timer("swt_decompose"):
    ...     coeffs = swt2_decompose(img, 'sym4', 2)
    """
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    _TIMINGS[label] = _TIMINGS.get(label, 0.0) + dt
    print(f"[TIMING] {label:<20}: {dt*1000:8.2f} ms")


def summary() -> None:
    """Pretty‑print timing results collected so far."""
    if not _TIMINGS:
        print("No timing data recorded.")
        return

    print("\n=== Timing summary ===")
    total = 0.0
    for k, v in _TIMINGS.items():
        total += v
        print(f"{k:<25}: {v:8.4f} s")
    print(f"{'-'*25}\nTotal{'':<20}: {total:8.4f} s\n")


def reset_timings() -> None:
    """Erase all stored timing information (useful for tests)."""
    _TIMINGS.clear()


def write_log(param_lines: list[str] | None = None) -> Path:
    """
    Write a text log summarising run parameters + timing to RUN_DIR/run_log.txt.

    Parameters
    ----------
    param_lines : list[str] or None
        Pre‑formatted strings (e.g. ["wavelet: sym4", "level: 2"]).  Each will be
        written on its own line before the timing summary.

    Returns
    -------
    Path
        Absolute path to the written log file.
    """
    log_path = RUN_DIR / "run_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Run timestamp : {RUN_DIR.name}\n")
        if param_lines:
            f.write("\n# Parameters\n")
            for ln in param_lines:
                f.write(ln + "\n")
        if _TIMINGS:
            f.write("\n# Timings (s)\n")
            total = 0.0
            for k, v in _TIMINGS.items():
                total += v
                f.write(f"{k:<25}: {v:.4f}\n")
            f.write(f"{'-'*25}\nTotal{'':<20}: {total:.4f}\n")
    return log_path


__all__ = [
    "load_gray",
    "save_image",
    "save_debug",
    "step_timer",
    "summary",
    "reset_timings",
    "RUN_DIR",
    "write_log",
    "dwt_visualize",
]