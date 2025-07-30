"""
prefilter.py
============

Median → Sobel edge‑enhancement step for the **wafer_wtms** pipeline.

Input
-----
8‑bit gray‑scale wafer tile (numpy.ndarray, dtype=uint8).

Output
------
32‑bit float gradient‑magnitude image (dtype=float32).  The dynamic range is
normalized to [0, 1] so that subsequent modules (wavelet→WTM)
receive sufficient numeric resolution.

Public API
----------
median_then_sobel(img: NDArray[np.uint8],
                  ksize: int = 3,
                  sobel_ksize: int = 3) -> NDArray[np.float32]

* ksize must be odd and ≥3.
* sobel_ksize is usually 3; OpenCV supports 1,3,5,7.

Notes
-----
The function is deliberately pure (no in‑place mutation) and wrapped with the
`@timer` decorator from *io_utils* so that its runtime appears in the pipeline
logs.
"""

from __future__ import annotations

import cv2
import numpy as np

from io_utils import timer

__all__ = [
    "apply_clahe",
    "apply_median", 
    "apply_gaussian",
    "apply_sobel",
    "apply_laplacian",
    "apply_filter_chain",
    "median_then_sobel",  # backward compatibility
]


# --------------------------------------------------------------------------- #
# Input validation helper
# --------------------------------------------------------------------------- #
def _validate_input(img: np.ndarray, allow_float: bool = False) -> None:
    """Common input validation for all filter functions."""
    if allow_float:
        if img.dtype not in (np.uint8, np.float32):
            raise ValueError("Input image must be dtype uint8 or float32.")
    else:
        if img.dtype != np.uint8:
            raise ValueError("Input image must be dtype uint8 (8-bit gray).")
    if img.ndim != 2:
        raise ValueError("Input image must be single-channel gray (H×W).")


# --------------------------------------------------------------------------- #
# Individual filter functions
# --------------------------------------------------------------------------- #
@timer
def apply_clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8),
) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Parameters
    ----------
    img : np.ndarray
        8-bit grayscale input image (H×W).
    clip_limit : float, default 2.0
        Threshold for contrast limiting.
    tile_grid_size : tuple, default (8, 8)
        Size of grid for local histogram equalization.
        
    Returns
    -------
    np.ndarray
        8-bit CLAHE enhanced image.
    """
    _validate_input(img)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)


@timer
def apply_median(
    img: np.ndarray,
    ksize: int = 3,
) -> np.ndarray:
    """
    Apply median filtering for noise suppression.
    
    Parameters
    ----------
    img : np.ndarray
        8-bit grayscale input image (H×W).
    ksize : int, default 3
        Median filter kernel size. Must be odd.
        
    Returns
    -------
    np.ndarray
        8-bit median filtered image.
    """
    _validate_input(img)
    
    if ksize < 3 or ksize % 2 == 0:
        raise ValueError("ksize must be an odd integer ≥ 3.")
        
    return cv2.medianBlur(img, ksize)


@timer
def apply_gaussian(
    img: np.ndarray,
    ksize: int = 3,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Apply Gaussian blur for noise suppression.
    
    Parameters
    ----------
    img : np.ndarray
        8-bit grayscale input image (H×W).
    ksize : int, default 3
        Gaussian kernel size. Must be odd.
    sigma : float, default 1.0
        Gaussian kernel standard deviation.
        
    Returns
    -------
    np.ndarray
        8-bit Gaussian blurred image.
    """
    _validate_input(img)
    
    if ksize < 3 or ksize % 2 == 0:
        raise ValueError("ksize must be an odd integer ≥ 3.")
        
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


@timer
def apply_sobel(
    img: np.ndarray,
    ksize: int = 3,
    normalize: bool = True,
) -> np.ndarray:
    """
    Apply Sobel edge detection.
    
    Parameters
    ----------
    img : np.ndarray
        8-bit grayscale input image (H×W).
    ksize : int, default 3
        Sobel operator aperture size (1, 3, 5, or 7).
    normalize : bool, default True
        Whether to normalize output to [0, 1] range.
        
    Returns
    -------
    np.ndarray
        32-bit float gradient magnitude image.
    """
    _validate_input(img)
    
    if ksize not in (1, 3, 5, 7):
        raise ValueError("ksize must be one of {1,3,5,7}.")
        
    # Sobel gradients (CV_32F for floating-point precision)
    grad_x = cv2.Sobel(
        img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=ksize, borderType=cv2.BORDER_REFLECT
    )
    grad_y = cv2.Sobel(
        img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=ksize, borderType=cv2.BORDER_REFLECT
    )
    
    # Compute magnitude in float32
    magnitude = cv2.magnitude(grad_x, grad_y)
    
    # Normalize to [0, 1] range if requested
    if normalize and magnitude.max() > 0:
        magnitude = magnitude / magnitude.max()
        
    return magnitude.astype(np.float32)


@timer
def apply_laplacian(
    img: np.ndarray,
    ksize: int = 3,
    normalize: bool = True,
) -> np.ndarray:
    """
    Apply Laplacian edge detection.
    
    Parameters
    ----------
    img : np.ndarray
        8-bit grayscale input image (H×W).
    ksize : int, default 3
        Laplacian kernel size. Must be odd.
    normalize : bool, default True
        Whether to normalize output to [0, 1] range.
        
    Returns
    -------
    np.ndarray
        32-bit float Laplacian response image.
    """
    _validate_input(img)
    
    if ksize < 1 or ksize % 2 == 0:
        raise ValueError("ksize must be an odd integer ≥ 1.")
        
    # Laplacian (CV_32F for floating-point precision)
    laplacian = cv2.Laplacian(img, ddepth=cv2.CV_32F, ksize=ksize)
    
    # Take absolute value and normalize if requested
    magnitude = np.abs(laplacian)
    if normalize and magnitude.max() > 0:
        magnitude = magnitude / magnitude.max()
        
    return magnitude.astype(np.float32)


# --------------------------------------------------------------------------- #
# Filter chain system
# --------------------------------------------------------------------------- #
AVAILABLE_FILTERS = {
    'clahe': apply_clahe,
    'median': apply_median,
    'gaussian': apply_gaussian, 
    'sobel': apply_sobel,
    'laplacian': apply_laplacian,
}


@timer
def apply_filter_chain(
    img: np.ndarray,
    filter_chain: list,
    filter_params: dict | None = None,
) -> np.ndarray:
    """
    Apply a sequence of filters to the input image.
    
    Parameters
    ----------
    img : np.ndarray
        8-bit grayscale input image (H×W).
    filter_chain : list[str]
        List of filter names to apply in order, e.g. ['clahe', 'median', 'sobel'].
    filter_params : dict, optional
        Parameters for each filter, e.g. {'clahe': {'clip_limit': 2.0}, 'median': {'ksize': 5}}.
        
    Returns
    -------
    np.ndarray
        Final processed image (uint8 for preprocessing filters, float32 for edge filters).
        
    Raises
    ------
    ValueError
        If filter name is not recognized or parameters are invalid.
    """
    if not filter_chain:
        raise ValueError("filter_chain cannot be empty.")
        
    filter_params = filter_params or {}
    current_img = img.copy()
    
    for filter_name in filter_chain:
        if filter_name not in AVAILABLE_FILTERS:
            raise ValueError(f"Unknown filter: {filter_name}. Available: {list(AVAILABLE_FILTERS.keys())}")
            
        filter_func = AVAILABLE_FILTERS[filter_name]
        params = filter_params.get(filter_name, {})
        
        try:
            current_img = filter_func(current_img, **params)
        except Exception as e:
            raise ValueError(f"Error applying filter '{filter_name}' with params {params}: {e}")
            
    return current_img


# --------------------------------------------------------------------------- #
# Legacy function for backward compatibility
# --------------------------------------------------------------------------- #
@timer
def median_then_sobel(
    img: np.ndarray,
    ksize: int = 3,
    sobel_ksize: int = 3,
) -> np.ndarray:
    """
    Legacy function: Apply median filtering followed by Sobel gradient magnitude.
    
    This function is maintained for backward compatibility.
    For new code, use apply_filter_chain(['median', 'sobel'], {...}) instead.

    Parameters
    ----------
    img : np.ndarray
        8‑bit gray‑scale input image (H×W).
    ksize : int, default 3
        Median filter kernel size. Must be odd.
    sobel_ksize : int, default 3
        Sobel operator aperture size (1, 3, 5, or 7).

    Returns
    -------
    np.ndarray
        32-bit float gradient magnitude image, normalized to range [0, 1].
    """
    return apply_filter_chain(
        img,
        filter_chain=['median', 'sobel'],
        filter_params={
            'median': {'ksize': ksize},
            'sobel': {'ksize': sobel_ksize, 'normalize': True}
        }
    )