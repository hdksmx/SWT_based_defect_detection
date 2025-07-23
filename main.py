

#!/usr/bin/env python3
"""
main.py – End‑to‑end driver for the SWT denoise / scratch‑mask pipeline.

Usage
-----
$ python main.py wafer001.png --level 2 --k 3.0 --combine keep_all

The script expects the input image to live in `input_img/` *unless* an
absolute/relative path is supplied.  All outputs go to `results/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import utils_io as io
import swt_utils as swt
import thresholding as th
import reconstruct as rc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="2‑Level SWT Denoiser")
    p.add_argument(
        "img",
        help="Input image file name or path (grayscale or colour).",
    )
    p.add_argument("--wavelet", default="haar", help="Wavelet family (PyWavelets name).")
    p.add_argument("--level", type=int, default=2, help="Decomposition levels (>=1).")
    p.add_argument(
        "-k", "--k", "--thr_factor",
        dest="thr_factor",
        type=float,
        default=3.0,
        help="Threshold factor T = k × σ̂. (aliases: -k, --k, --thr_factor)",
    )
    p.add_argument(
        "--thr_mode",
        choices=("soft", "hard"),
        default="soft",
        help="Thresholding mode.",
    )
    p.add_argument(
        "--combine",
        choices=("keep_all", "keep_hv", "approx_only"),
        default="keep_all",
        help="Detail‑band combination strategy before reconstruction.",
    )
    p.add_argument(
        "--save_mask",
        action="store_true",
        help="Also generate a scratch mask (OR‑fused) and save.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Save intermediate debug images to results/debug_img/.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load image
    with io.step_timer("load_image"):
        img = io.load_gray(args.img)
        if args.debug:
            io.save_debug("00_input", img)
            # DWT 4‑tile visualisation (LL/LH/HL/HH)
            try:
                quad = io.dwt_visualize(img, wavelet=args.wavelet)
                io.save_debug("00_dwt_quad", quad)
            except Exception as e:
                print(f"[WARN] dwt_visualize failed: {e}")

    # 2. SWT decompose
    with io.step_timer("swt_decompose"):
        coeffs = swt.swt2_decompose(img, wavelet=args.wavelet, level=args.level)

    # 3. σ̂ estimation
    with io.step_timer("sigma_est"):
        sigmas = th.estimate_sigma(coeffs)

    # 4. Threshold
    with io.step_timer("threshold"):
        coeffs_thr = th.apply_threshold(
            coeffs, sigmas, k=args.thr_factor, mode=args.thr_mode
        )

    # Optional: debug magnitude maps
    if args.debug:
        mags = swt.coeff_magnitude(coeffs_thr)
        for idx, mag in enumerate(mags, start=1):
            io.save_debug(f"01_mag_lvl{idx}", mag / mag.max())

    # 5. Combine bands
    with io.step_timer("combine"):
        coeffs_comb = rc.combine_bands(coeffs_thr, strategy=args.combine)

    # 6. Reconstruct image
    with io.step_timer("reconstruct"):
        denoised = rc.reconstruct_image(coeffs_comb, wavelet=args.wavelet)
        out_path = io.save_image(denoised, "denoised.png")
        print(f"Saved denoised image: {out_path.relative_to(io.RESULTS_DIR.parent)}")

    # 7. (Optional) scratch mask
    if args.save_mask:
        with io.step_timer("make_mask"):
            mask = rc.make_mask(coeffs_thr, k=2.0, skeleton=True)
            mask_path = io.save_image(mask.astype("uint8") * 255, "scratch_mask.png")
            print(f"Saved mask: {mask_path.relative_to(io.RESULTS_DIR.parent)}")

    # 8. Timing summary
    io.summary()

    # 9. Write run log
    param_lines = [f"{k}: {v}" for k, v in vars(args).items()]
    log_path = io.write_log(param_lines)
    print(f"Run log saved to: {log_path.relative_to(io.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()