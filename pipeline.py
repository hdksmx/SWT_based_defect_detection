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
    wavelet_name: str = "sym8"
    dwt_level: int = 2

    # Candidate sampling
    std_factor: float = 3.0

    # Edge search / WTMS
    window_hw: int = 3
    max_steps: int = 5

    # GS filtering
    gs_csv: Optional[Path] = None  # path to golden‑set CSV; None → skip filter

    # Interscale ratio test
    interscale_threshold: float = 2.0

    # Post‑process
    min_region_area: int = 5
    ecc_thr: float = 0.9
    len_thr: float = 20
    
    # GLCM parameters
    use_glcm_optimization: bool = False
    glcm_window_size: int = 11
    glcm_levels: int = 32
    glcm_distances: List[int] = field(default_factory=lambda: [1, 2])
    glcm_angles: List[int] = field(default_factory=lambda: [0, 45, 90, 135])
    glcm_features: List[str] = field(default_factory=lambda: ['homogeneity', 'contrast', 'energy', 'correlation'])
    glcm_combination_strategy: str = 'scratch_optimized'
    glcm_smoothing_sigma: float = 1.5
    glcm_blend_range: List[float] = field(default_factory=lambda: [0.3, 0.8])
    glcm_multiscale_scales: List[int] = field(default_factory=lambda: [7, 11, 15])
    glcm_multiscale_fusion: str = 'weighted_average'
    
    # Debugging options
    enable_debug_output: bool = False
    
    def build_prefilter_params(self) -> Dict[str, Dict]:
        """
        Build complete prefilter parameters including GLCM parameters.
        
        Dynamically constructs filter parameters from PipelineConfig fields,
        ensuring all GLCM settings are properly passed to filter functions.
        
        Returns
        -------
        Dict[str, Dict]
            Complete parameter dictionary for all filters in the chain.
        """
        # Start with base prefilter params
        params = self.prefilter_params.copy()
        
        # Add GLCM filter parameters
        params.update({
            'glcm_texture': {
                'window_size': self.glcm_window_size,
                'levels': self.glcm_levels,
                'smoothing_sigma': self.glcm_smoothing_sigma,
                'distance': self.glcm_distances[0] if self.glcm_distances else 1,
                'angle': self.glcm_angles[0] if self.glcm_angles else 0,
                'save_debug_images': self.enable_debug_output
            },
            'glcm_multi_feature': {
                'window_size': self.glcm_window_size,
                'distances': self.glcm_distances,
                'angles': self.glcm_angles,
                'levels': self.glcm_levels,
                'features': self.glcm_features,
                'combination_strategy': self.glcm_combination_strategy,
                'smoothing_sigma': self.glcm_smoothing_sigma,
                'blend_range': tuple(self.glcm_blend_range),
                'use_optimization': self.use_glcm_optimization,
                'save_debug_images': self.enable_debug_output
            },
            'glcm_multiscale': {
                'scales': self.glcm_multiscale_scales,
                'features': self.glcm_features,
                'fusion_strategy': self.glcm_multiscale_fusion,
                'distances': self.glcm_distances,
                'angles': self.glcm_angles,
                'levels': self.glcm_levels,
                'combination_strategy': self.glcm_combination_strategy,
                'smoothing_sigma': self.glcm_smoothing_sigma,
                'use_optimization': self.use_glcm_optimization,
                'save_debug_images': self.enable_debug_output
            },
            'glcm_blob_removal': {
                'preserve_scratches': True,
                'scratch_threshold': 0.4,
                'window_size': self.glcm_window_size,
                'smoothing_sigma': self.glcm_smoothing_sigma,
                'use_optimization': self.use_glcm_optimization,
                'save_debug_images': self.enable_debug_output
            }
        })
        
        # Update blob_removal with GLCM integration settings
        if 'blob_removal' in params:
            params['blob_removal'].update({
                'use_glcm_texture': any('glcm' in filter_name for filter_name in self.prefilter_chain),
                'glcm_params': {
                    'window_size': self.glcm_window_size,
                    'smoothing_sigma': self.glcm_smoothing_sigma,
                    'preserve_scratches': True,
                    'scratch_threshold': 0.4
                }
            })
        
        return params


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

    # 2. Prefilter - use filter chain system with GLCM support
    grad_f32 = apply_filter_chain(
        img_u8,
        filter_chain=cfg.prefilter_chain,
        filter_params=cfg.build_prefilter_params()
    )

    

    # 2‑B. Raw float32 gradient
    save_path = debug_path(1, "prefilter")
    save_image(grad_f32, save_path)
    debug_files["prefilter"] = save_path
    
    # 2‑C. Add GLCM debug files if debugging enabled and GLCM filters used
    if cfg.enable_debug_output and any('glcm' in f for f in cfg.prefilter_chain):
        glcm_features = cfg.glcm_features
        debug_counter = 1
        
        # Individual feature maps
        for feature_name in glcm_features:
            debug_files[f"glcm_{feature_name}"] = debug_path(1, f"{debug_counter}_glcm_{feature_name}")
            debug_counter += 1
        
        # Combined texture score and other intermediate results
        debug_files["glcm_combined_score"] = debug_path(1, "5_glcm_combined_score")
        debug_files["glcm_smoothed"] = debug_path(1, "6_glcm_smoothed")
        debug_files["glcm_alpha_blend"] = debug_path(1, "7_glcm_alpha_blend")
        debug_files["glcm_final_result"] = debug_path(1, "8_glcm_final_result")
        
        # Multi-scale specific debug files
        if 'glcm_multiscale' in cfg.prefilter_chain:
            scales = cfg.glcm_multiscale_scales
            for i, scale in enumerate(scales):
                debug_files[f"glcm_scale_{scale}x{scale}"] = debug_path(1, f"2{i+1}_glcm_scale_{scale}x{scale}")
            debug_files["glcm_multiscale_fused"] = debug_path(1, "25_glcm_multiscale_fused")

    # 2‑D. Quad‑view of 1‑level DWT for quick inspection
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
    keep_idx = R >= cfg.interscale_threshold
    coords_after_ratio = cand_coords[keep_idx]
    logger.info(f"Candidates after interscale test (R ≥ {cfg.interscale_threshold}): {len(coords_after_ratio)}")

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