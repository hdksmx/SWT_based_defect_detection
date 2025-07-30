"""
pipeline.py
===========

High‑level orchestration for the Wavelet‑WTMS scratch‑detection flow.

The `run()` function wires together all previously implemented modules:

1. Read 8‑bit gray wafer tile.
2. Prefilter (median → Sobel) → float32 normalized gradient image.
3. 2‑level SWT → WTM maps (level‑1, level‑2).
4. μ+3σ candidate sampling on level‑1 WTM.
5. WTMS computation on level‑1 & level‑2 for every candidate.
6. Interscale ratio test (R < 0) → raw defect mask.
7. Golden‑Set filtering (optional) → cleaned mask.
8. Post‑processing: connected‑component clustering + defect type.
9. Save debug images to `results/<timestamp>/debug_img`.

Public API
----------
run(img_path: Path | str,
    cfg: "PipelineConfig" | None = None) -> "PipelineResult"

`PipelineConfig` holds all tunable parameters.  Sensible defaults are provided,
so a user can invoke `run("input_img/wafer.bmp")` without passing a config.

The module depends only on other in‑project packages plus NumPy, OpenCV,
PyWavelets and scikit‑image.
"""

from __future__ import annotations

import cv2
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from candidate import sample_candidates
from golden_set import filter_by_gs, load_gs
from io_utils import (
    DEBUG_DIR,
    RESULTS_DIR,
    debug_path,
    read_image,
    save_image,
    timer,
)
from io_utils import TIMINGS
from postprocess import DefectRegion, cluster_labels
from prefilter import apply_filter_chain, median_then_sobel
from wavelet import swt2, wtm, dwt_quad  # already adds dwt2, wtm; extend to include quad
from wtms import compute_wtms_array, interscale_ratio
from datetime import datetime

logger = logging.getLogger("pipeline")
logger.setLevel(logging.INFO)


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class PipelineConfig:
    # Prefilter chain
    prefilter_chain: List[str] = field(default_factory=lambda: ['median', 'sobel'])
    prefilter_params: Dict[str, Dict] = field(default_factory=lambda: {
        'clahe': {'clip_limit': 2.0, 'tile_grid_size': (8, 8)},
        'median': {'ksize': 3},
        'gaussian': {'ksize': 3, 'sigma': 1.0},
        'sobel': {'ksize': 3, 'normalize': True},
        'laplacian': {'ksize': 3, 'normalize': True},
        'blob_removal': {
            's_med': 3.0 / 255.0,
            's_avg': 20.0 / 255.0,
            'gauss_sigma': 1.0,
            'median_width': 5,
            'lr_width': 3
        }
    })
    
    # Legacy prefilter parameters (for backward compatibility)
    median_kernel: int = 3
    sobel_kernel: int = 3

    # Wavelet
    wavelet_name: str = "coif6"
    dwt_level: int = 2

    # Candidate sampling
    std_factor: float = 3.0

    # Edge search / WTMS
    window_hw: int = 3
    max_steps: int = 5

    # GS filtering
    gs_csv: Optional[Path] = None  # path to golden‑set CSV; None → skip filter

    # Post‑process
    min_region_area: int = 5
    ecc_thr: float = 0.9
    len_thr: float = 20


@dataclass(slots=True)
class PipelineResult:
    raw_mask: np.ndarray  # boolean mask before GS filter
    cleaned_mask: np.ndarray  # after GS filter (may be same as raw)
    regions: List[DefectRegion]
    debug_files: Dict[str, Path] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Run‑log helper
# --------------------------------------------------------------------------- #
def _write_run_log(img_path: Path, cfg: PipelineConfig,
                   result: "PipelineResult", t_start: datetime) -> None:
    """Write run.txt capturing runtime parameters & timings."""
    log_path = RESULTS_DIR / "run.txt"
    with log_path.open("w") as f:
        # --- Run meta information ---
        f.write(f"run_start        : {t_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"input_img        : {img_path}\n")
        f.write(f"wavelet_name     : {cfg.wavelet_name}\n")

        # --- Variable parameters used this run ---
        f.write("\n# Pipeline parameters\n")
        f.write(f"median_kernel    : {cfg.median_kernel}\n")
        f.write(f"sobel_kernel     : {cfg.sobel_kernel}\n")
        f.write(f"std_factor       : {cfg.std_factor}\n")
        f.write(f"window_halfwidth : {cfg.window_hw}\n")
        f.write(f"max_steps        : {cfg.max_steps}\n")
        f.write(f"min_region_area  : {cfg.min_region_area}\n")
        f.write(f"ecc_thr          : {cfg.ecc_thr}\n")
        f.write(f"len_thr          : {cfg.len_thr}\n")
        if cfg.gs_csv:
            f.write(f"gs_csv           : {cfg.gs_csv}\n")

        # --- Timing table ---
        f.write("\n# Stage timings (ms)\n")
        total = 0.0
        for name, ms in sorted(TIMINGS.items(), key=lambda x: x[0]):
            f.write(f"{name:<16}: {ms:8.2f}\n")
            total += ms
        f.write("-" * 32 + "\n")
        f.write(f"{'Total':<16}: {total:8.2f}\n")


# --------------------------------------------------------------------------- #
# Helper: visualise float WTM map as 8‑bit heatmap
# --------------------------------------------------------------------------- #
def _save_wtm_png(wtm_map: np.ndarray, filename: Path) -> None:
    vis = np.empty_like(wtm_map, dtype=np.uint8)
    cv2.normalize(
        wtm_map,
        vis,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
    )
    save_image(vis, filename)


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
@timer
def run(img_path: str | Path, cfg: PipelineConfig | None = None) -> PipelineResult:
    """Execute full WTMS pipeline on *img_path*."""
    t_start = datetime.now()
    cfg = cfg or PipelineConfig()

    debug_files: Dict[str, Path] = {}

    # 1. Load image
    img_u8 = read_image(img_path, as_gray=True)

    # 2. Prefilter - temporarily use legacy function for debugging
    grad_f32 = median_then_sobel(
        img_u8, ksize=cfg.median_kernel, sobel_ksize=cfg.sobel_kernel
    )

    

    # 2‑B. Raw float32 gradient
    save_path = debug_path(1, "prefilter")
    save_image(grad_f32, save_path)
    debug_files["prefilter"] = save_path

    # 2‑C. Quad‑view of 1‑level DWT for quick inspection
    grad_float = grad_f32  # Already normalized to [0,1] range
    quad_img = dwt_quad(grad_float, wavelet=cfg.wavelet_name)
    quad_path = debug_path(0, "dwt_quad")
    save_image(quad_img, quad_path)
    debug_files["dwt_quad"] = quad_path

    # 3. SWT
    detail_coeffs = swt2(
        grad_float, wavelet=cfg.wavelet_name, level=cfg.dwt_level
    )
    # detail_coeffs[0] = level-1 (finest), detail_coeffs[1] = level-2 (coarser)
    wtm_lvl1 = wtm(detail_coeffs[0])  # Level-1 (finest scale)
    wtm_lvl2 = wtm(detail_coeffs[1])  # Level-2 (coarser scale)

    _save_wtm_png(wtm_lvl1, debug_path(2, "wtm_lvl1"))
    _save_wtm_png(wtm_lvl2, debug_path(2, "wtm_lvl2"))

    # 4. Candidate sampling
    cand_coords = sample_candidates(wtm_lvl1, std_factor=cfg.std_factor)
    logger.info(f"Candidates after μ+{cfg.std_factor}σ: {len(cand_coords)}")

    # 5. WTMS
    wtms1 = compute_wtms_array(
        wtm_lvl1,
        cand_coords,
        window_hw=cfg.window_hw,
        max_steps=cfg.max_steps,
    )
    wtms2 = compute_wtms_array(
        wtm_lvl2,
        cand_coords,
        window_hw=cfg.window_hw,   # same resolution in SWT
        max_steps=cfg.max_steps,
    )

    # 6. Interscale ratio test
    R = interscale_ratio(wtms1, wtms2)
    keep_idx = R < 0.0
    coords_after_ratio = cand_coords[keep_idx]
    logger.info(f"Candidates after interscale test: {len(coords_after_ratio)}")

    raw_mask = np.zeros_like(img_u8, dtype=bool)
    raw_mask[coords_after_ratio[:, 0], coords_after_ratio[:, 1]] = True
    save_image(raw_mask.astype(np.uint8) * 255, debug_path(3, "raw_detect_mask"))

    # 7. Golden‑Set filter
    if cfg.gs_csv is not None and cfg.gs_csv.exists():
        gs_set = load_gs(cfg.gs_csv)
        coords_clean = filter_by_gs(coords_after_ratio, gs_set)
        logger.info(
            f"After GS filter: {len(coords_clean)} (filtered {len(coords_after_ratio) - len(coords_clean)})"
        )
    else:
        coords_clean = coords_after_ratio

    cleaned_mask = np.zeros_like(img_u8, dtype=bool)
    cleaned_mask[coords_clean[:, 0], coords_clean[:, 1]] = True
    save_image(
        cleaned_mask.astype(np.uint8) * 255, debug_path(4, "cleaned_detect_mask")
    )

    # 8. Post‑processing
    regions = cluster_labels(
        cleaned_mask,
        min_area=cfg.min_region_area,
    )
    logger.info(f"Found {len(regions)} defect regions.")

    # 9. Overlay visualisation with color-coded defect types
    overlay = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    
    # Create separate masks for each defect type
    scratch_mask = np.zeros_like(cleaned_mask, dtype=bool)
    particle_mask = np.zeros_like(cleaned_mask, dtype=bool)
    
    # Assign pixels to appropriate masks based on region classification
    from skimage.measure import label as sk_label
    labeled_mask = sk_label(cleaned_mask)
    
    for region in regions:
        region_pixels = labeled_mask == region.label
        if region.defect_type == "scratch":
            scratch_mask |= region_pixels
        else:  # particle
            particle_mask |= region_pixels
    
    # Apply colors: scratch=red, particle=blue
    overlay[scratch_mask] = (0, 0, 255)    # Red for scratches (BGR format)
    overlay[particle_mask] = (255, 0, 0)   # Blue for particles (BGR format)
    
    overlay_out = RESULTS_DIR / "defect_overlay.png"
    save_image(overlay, overlay_out)
    debug_files["overlay"] = overlay_out

    result = PipelineResult(
        raw_mask=raw_mask,
        cleaned_mask=cleaned_mask,
        regions=regions,
        debug_files=debug_files,
    )

    # 10. Write run.txt log
    _write_run_log(Path(img_path), cfg, result, t_start)

    return result