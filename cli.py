#!/usr/bin/env python
"""
cli.py
======

Command‑line interface for the **wafer_wtms** project.

복사‑붙여넣기로 바로 실행할 수 있는 예시
------------------------------------
# 1) 기본 파라미터로 결함 검사
python -m wafer_wtms.cli inspect -i input_img/wafer_tile.bmp

# 2) Golden Set CSV를 사용하여 검사
python -m wafer_wtms.cli inspect -i input_img/wafer_tile.bmp --gs gs/ref_tile.csv

# 3) 결함 없는 mask(흰색=good, 검정=defect)로 Golden Set 생성
python -m wafer_wtms.cli build-gs -m input_img/ref_good_mask.png -o gs/ref_tile.csv

# 4) Haar wavelet + 낮은 std-factor + edge search 확장
python cli.py inspect -i input_img/wafer_tile.bmp --wavelet_name haar --std_factor 2.5 --window_hw 5 --max_steps 10

# 5) 노이즈 제거 강화 + 스크래치 연결성 개선
python cli.py inspect -i input_img/wafer_tile.bmp --std_factor 2.5 --window_hw 5 --max_steps 12 --min_region_area 15 --ecc_thr 0.75

# 6) CLAHE + Median + Sobel 체인
python cli.py inspect -i input_img/wafer_tile.bmp --prefilter_chain clahe median sobel --clahe_clip_limit 3.0 --median_ksize 5

# 7) CLAHE만 적용 후 Sobel
python cli.py inspect -i input_img/wafer_tile.bmp --prefilter_chain clahe sobel --clahe_clip_limit 4.0 --clahe_tile_size 12

# 8) GLCM multi-feature texture filtering
python cli.py inspect -i input_img/wafer_tile.bmp --prefilter_chain glcm_multi_feature sobel --glcm_combination_strategy weighted_adaptive

# 9) GLCM multi-scale analysis
python cli.py inspect -i input_img/wafer_tile.bmp --prefilter_chain glcm_multiscale sobel --glcm_multiscale_scales 7 11 15 --glcm_multiscale_fusion adaptive_fusion

# 10) Blob removal with GLCM texture filtering
python cli.py inspect -i input_img/wafer_tile.bmp --prefilter_chain blob_removal sobel --blob_use_glcm_texture --glcm_window_size 9

# 11) PCA-based GLCM feature combination
python cli.py inspect -i input_img/wafer_tile.bmp --prefilter_chain glcm_multi_feature sobel --glcm_combination_strategy pca_based --glcm_features homogeneity contrast energy correlation entropy

# 12) GLCM with performance optimizations enabled
python cli.py inspect -i input_img/wafer_tile.bmp --prefilter_chain glcm_multi_feature sobel --glcm_optimize --glcm_features homogeneity contrast energy correlation

# 13) Multi-scale GLCM with optimizations for large images
python cli.py inspect -i input_img/wafer_tile.bmp --prefilter_chain glcm_multiscale sobel --glcm_optimize --glcm_multiscale_scales 9 13 17
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NoReturn

import cv2

from golden_set import build_gs
from io_utils import PROJECT_ROOT, save_image
from pipeline import PipelineConfig, PipelineResult, run as run_pipeline


def _cmd_inspect(args: argparse.Namespace) -> None:
    # Build prefilter parameters from CLI arguments
    prefilter_params = {
        'clahe': {
            'clip_limit': args.clahe_clip_limit,
            'tile_grid_size': (args.clahe_tile_size, args.clahe_tile_size)
        },
        'median': {'ksize': args.median_ksize},
        'gaussian': {
            'ksize': args.gaussian_ksize,
            'sigma': args.gaussian_sigma
        },
        'sobel': {'ksize': args.sobel_ksize, 'normalize': True},
        'laplacian': {'ksize': args.laplacian_ksize, 'normalize': True},
        'blob_removal': {
            's_med': args.blob_s_med,
            's_avg': args.blob_s_avg,
            'gauss_sigma': args.blob_gauss_sigma,
            'median_width': args.blob_median_width,
            'lr_width': args.blob_lr_width,
            'use_glcm_texture': args.blob_use_glcm_texture
        },
        # GLCM filter parameters
        'glcm_texture': {
            'window_size': args.glcm_window_size,
            'levels': args.glcm_levels,
            'smoothing_sigma': args.glcm_smoothing_sigma,
            'distance': args.glcm_distances[0] if args.glcm_distances else 1,
            'angle': args.glcm_angles[0] if args.glcm_angles else 0
        },
        'glcm_multi_feature': {
            'window_size': args.glcm_window_size,
            'distances': args.glcm_distances,
            'angles': args.glcm_angles,
            'levels': args.glcm_levels,
            'features': args.glcm_features,
            'combination_strategy': args.glcm_combination_strategy,
            'smoothing_sigma': args.glcm_smoothing_sigma,
            'use_optimization': args.glcm_optimize
        },
        'glcm_multiscale': {
            'scales': args.glcm_multiscale_scales,
            'features': args.glcm_features,
            'fusion_strategy': args.glcm_multiscale_fusion,
            'distances': args.glcm_distances,
            'angles': args.glcm_angles,
            'levels': args.glcm_levels,
            'combination_strategy': args.glcm_combination_strategy,
            'smoothing_sigma': args.glcm_smoothing_sigma,
            'use_optimization': args.glcm_optimize
        },
        'glcm_blob_removal': {
            'preserve_scratches': True,
            'scratch_threshold': 0.4,
            'window_size': args.glcm_window_size,
            'smoothing_sigma': args.glcm_smoothing_sigma
        }
    }
    
    cfg = PipelineConfig(
        prefilter_chain=args.prefilter_chain,
        prefilter_params=prefilter_params,
        # Legacy parameters (for backward compatibility)
        median_kernel=args.median_ksize,
        sobel_kernel=args.sobel_ksize,
        # Other parameters
        std_factor=args.std_factor,
        wavelet_name=args.wavelet_name,
        window_hw=args.window_hw,
        max_steps=args.max_steps,
        min_region_area=args.min_region_area,
        ecc_thr=args.ecc_thr,
        len_thr=args.len_thr,
        gs_csv=Path(args.gs) if args.gs else None,
    )
    result: PipelineResult = run_pipeline(args.img, cfg)

    # 간단 요약 출력
    scratch_cnt = sum(r.defect_type == "scratch" for r in result.regions)
    particle_cnt = len(result.regions) - scratch_cnt
    print(
        f"\nSummary: total {len(result.regions)} regions "
        f"(scratch={scratch_cnt}, particle={particle_cnt})"
    )
    print("Debug images:")
    for tag, p in result.debug_files.items():
        rel = p.relative_to(PROJECT_ROOT)
        print(f"  {tag:<12}: {rel}")


def _cmd_build_gs(args: argparse.Namespace) -> None:
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise IOError(f"Failed to read mask: {args.mask}")

    # white pixels (255) are good → True
    good_mask = mask > 0
    out_csv = Path(args.output)
    build_gs(good_mask, out_csv)
    print(f"Golden Set CSV saved to {out_csv}")


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wafer_wtms.cli",
        description="Wavelet‑WTMS wafer scratch detection pipeline CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # inspect command
    p_ins = subparsers.add_parser("inspect", help="Run defect inspection.")
    p_ins.add_argument(
        "-i",
        "--img",
        required=True,
        help="Path to 8‑bit grayscale wafer tile image.",
    )
    p_ins.add_argument(
        "--gs",
        help="Golden‑Set CSV file (optional).",
    )
    # Prefilter chain selection
    p_ins.add_argument(
        "--prefilter_chain",
        nargs='+',
        default=['median', 'sobel'],
        choices=['clahe', 'median', 'gaussian', 'sobel', 'laplacian', 'blob_removal', 
                'glcm_texture', 'glcm_multi_feature', 'glcm_multiscale', 'glcm_blob_removal'],
        help="Prefilter chain to apply in order (default: median sobel). "
             "GLCM options: glcm_texture, glcm_multi_feature, glcm_multiscale, glcm_blob_removal."
    )
    
    # CLAHE parameters
    p_ins.add_argument(
        "--clahe_clip_limit",
        type=float,
        default=2.0,
        help="CLAHE clip limit for contrast limiting (default: 2.0)."
    )
    p_ins.add_argument(
        "--clahe_tile_size",
        type=int,
        default=8,
        help="CLAHE tile grid size (default: 8 → 8x8 grid)."
    )
    
    # Filter kernel sizes
    p_ins.add_argument("--median_ksize", type=int, default=3, help="Median kernel size.")
    p_ins.add_argument("--gaussian_ksize", type=int, default=3, help="Gaussian kernel size.")
    p_ins.add_argument("--sobel_ksize", type=int, default=3, help="Sobel kernel size.")
    p_ins.add_argument("--laplacian_ksize", type=int, default=3, help="Laplacian kernel size.")
    
    # Gaussian-specific parameters
    p_ins.add_argument(
        "--gaussian_sigma",
        type=float,
        default=1.0,
        help="Gaussian kernel standard deviation (default: 1.0)."
    )
    
    # Blob removal parameters
    p_ins.add_argument(
        "--blob_s_med",
        type=float,
        default=3.0 / 255.0,
        help="Valley condition threshold for blob removal (default: 0.0118)."
    )
    p_ins.add_argument(
        "--blob_s_avg",
        type=float,
        default=20.0 / 255.0,
        help="Symmetry condition threshold for blob removal (default: 0.0784)."
    )
    p_ins.add_argument(
        "--blob_gauss_sigma",
        type=float,
        default=1.0,
        help="Gaussian sigma for blob removal valley condition (default: 1.0)."
    )
    p_ins.add_argument(
        "--blob_median_width",
        type=int,
        default=5,
        help="Horizontal median filter width for blob removal (default: 5)."
    )
    p_ins.add_argument(
        "--blob_lr_width",
        type=int,
        default=3,
        help="Left/right mean window width for blob removal (default: 3)."
    )
    p_ins.add_argument(
        "--blob_use_glcm_texture",
        action="store_true",
        help="Use GLCM texture filtering instead of Gaussian blur in blob removal."
    )
    
    # GLCM parameters
    p_ins.add_argument(
        "--glcm_window_size",
        type=int,
        default=11,
        help="GLCM sliding window size (default: 11)."
    )
    p_ins.add_argument(
        "--glcm_levels",
        type=int,
        default=32,
        help="Number of gray levels for GLCM quantization (default: 32)."
    )
    p_ins.add_argument(
        "--glcm_features",
        nargs='+',
        default=['homogeneity', 'contrast', 'energy', 'correlation'],
        choices=['homogeneity', 'contrast', 'energy', 'correlation', 'entropy', 'dissimilarity'],
        help="GLCM features to compute (default: homogeneity contrast energy correlation)."
    )
    p_ins.add_argument(
        "--glcm_combination_strategy",
        choices=['scratch_optimized', 'weighted_adaptive', 'pca_based'],
        default='scratch_optimized',
        help="Strategy for combining GLCM features (default: scratch_optimized)."
    )
    p_ins.add_argument(
        "--glcm_distances",
        nargs='+',
        type=int,
        default=[1, 2],
        help="GLCM pixel distances (default: 1 2)."
    )
    p_ins.add_argument(
        "--glcm_angles",
        nargs='+',
        type=int,
        default=[0, 45, 90, 135],
        help="GLCM angles in degrees (default: 0 45 90 135)."
    )
    p_ins.add_argument(
        "--glcm_smoothing_sigma", 
        type=float,
        default=1.5,
        help="Gaussian smoothing sigma for GLCM filters (default: 1.5)."
    )
    p_ins.add_argument(
        "--glcm_multiscale_scales",
        nargs='+',
        type=int,
        default=[7, 11, 15],
        help="Window sizes for multi-scale GLCM analysis (default: 7 11 15)."
    )
    p_ins.add_argument(
        "--glcm_multiscale_fusion",
        choices=['weighted_average', 'adaptive_fusion'],
        default='weighted_average',
        help="Multi-scale fusion strategy (default: weighted_average)."
    )
    p_ins.add_argument(
        "--glcm_optimize",
        action="store_true",
        help="Enable GLCM performance optimizations (LUT caching and vectorization)."
    )
    
    p_ins.add_argument(
        "--std_factor",
        type=float,
        default=3.0,
        help="Std‑factor for candidate threshold (μ+kσ).",
    )
    p_ins.add_argument(
        "--wavelet_name",
        default="coif6",
        help="PyWavelets wavelet name (e.g. haar, db2, coif6).",
    )
    
    # Edge search / WTMS parameters
    p_ins.add_argument(
        "--window_hw",
        type=int,
        default=3,
        help="Half window size for edge search (default: 3 → 7×7 window).",
    )
    p_ins.add_argument(
        "--max_steps",
        type=int,
        default=5,
        help="Maximum steps for edge search path (default: 5).",
    )
    
    # Post-processing parameters
    p_ins.add_argument(
        "--min_region_area",
        type=int,
        default=5,
        help="Minimum region area to keep (pixels, default: 5).",
    )
    p_ins.add_argument(
        "--ecc_thr",
        type=float,
        default=0.9,
        help="Eccentricity threshold for scratch/particle classification (default: 0.9).",
    )
    p_ins.add_argument(
        "--len_thr",
        type=float,
        default=20,
        help="Length threshold for scratch classification (default: 20).",
    )
    
    p_ins.set_defaults(func=_cmd_inspect)

    # build-gs command
    p_gs = subparsers.add_parser("build-gs", help="Build Golden Set CSV.")
    p_gs.add_argument(
        "-m",
        "--mask",
        required=True,
        help="Binary mask PNG (white=good pixels).",
    )
    p_gs.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output CSV filename.",
    )
    p_gs.set_defaults(func=_cmd_build_gs)

    return parser


def main(argv: list[str] | None = None) -> NoReturn:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)  # type: ignore[attr-defined]
    sys.exit(0)


if __name__ == "__main__":  # pragma: no cover
    main()