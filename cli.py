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
        'laplacian': {'ksize': args.laplacian_ksize, 'normalize': True}
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
        choices=['clahe', 'median', 'gaussian', 'sobel', 'laplacian'],
        help="Prefilter chain to apply in order (default: median sobel)."
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