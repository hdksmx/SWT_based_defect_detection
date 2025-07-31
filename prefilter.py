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
from scipy.ndimage import median_filter, uniform_filter1d

from io_utils import timer

__all__ = [
    "apply_clahe",
    "apply_median", 
    "apply_gaussian",
    "apply_sobel",
    "apply_laplacian",
    "apply_blob_removal",
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


def _get_samples_in_radius(
    img: np.ndarray, 
    y: int, 
    x: int, 
    radius: int, 
    blob_mask: np.ndarray
) -> np.ndarray:
    """
    Get non-blob pixel samples within a given radius from point (y, x).
    
    Parameters
    ----------
    img : np.ndarray
        Input image (float32).
    y, x : int
        Center coordinates.
    radius : int
        Search radius.
    blob_mask : np.ndarray
        Boolean mask where True indicates blob pixels.
        
    Returns
    -------
    np.ndarray
        Array of non-blob pixel values within radius.
    """
    h, w = img.shape
    samples = []
    
    y_min = max(0, y - radius)
    y_max = min(h, y + radius + 1)
    x_min = max(0, x - radius)
    x_max = min(w, x + radius + 1)
    
    for ny in range(y_min, y_max):
        for nx in range(x_min, x_max):
            if not blob_mask[ny, nx]:  # non-blob pixel
                distance = np.sqrt((ny - y)**2 + (nx - x)**2)
                if distance <= radius:
                    samples.append(img[ny, nx])
    
    return np.array(samples)


def _fallback_background(
    img: np.ndarray, 
    y: int, 
    x: int, 
    blob_mask: np.ndarray
) -> float:
    """
    Fallback strategy for background estimation when insufficient samples.
    
    Parameters
    ----------
    img : np.ndarray
        Input image (float32).
    y, x : int
        Pixel coordinates.
    blob_mask : np.ndarray
        Boolean mask where True indicates blob pixels.
        
    Returns
    -------
    float
        Estimated background value.
    """
    # Strategy 1: Expand search radius progressively
    for radius in [10, 20, 30]:
        samples = _get_samples_in_radius(img, y, x, radius, blob_mask)
        if len(samples) >= 3:
            return np.mean(samples)
    
    # Strategy 2: Use median of all non-blob pixels
    non_blob_pixels = img[~blob_mask]
    if len(non_blob_pixels) > 0:
        return np.median(non_blob_pixels)
    
    # Strategy 3: Keep original pixel value (last resort)
    return img[y, x]


def _estimate_local_background(
    img: np.ndarray, 
    blob_mask: np.ndarray, 
    window_size: int = 7,
    min_samples: int = 3
) -> np.ndarray:
    """
    Estimate background for blob pixels using local non-blob pixel statistics.
    
    For each blob pixel, compute the mean of non-blob pixels within a local window.
    If insufficient samples are found, use fallback strategy with expanded search.
    
    Parameters
    ----------
    img : np.ndarray
        Input image (float32, range [0,1]).
    blob_mask : np.ndarray
        Boolean mask where True indicates blob pixels to be replaced.
    window_size : int, default=7
        Size of local window for background estimation (must be odd).
    min_samples : int, default=3
        Minimum number of non-blob pixels needed for valid estimation.
        
    Returns
    -------
    np.ndarray
        Background-estimated image (same shape and dtype as input).
    """
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    
    background = img.copy()
    h, w = img.shape
    half_win = window_size // 2
    
    # Find blob pixel coordinates
    blob_coords = np.where(blob_mask)
    
    for y, x in zip(blob_coords[0], blob_coords[1]):
        # Define window boundaries (clipped to image bounds)
        y_min = max(0, y - half_win)
        y_max = min(h, y + half_win + 1)
        x_min = max(0, x - half_win)
        x_max = min(w, x + half_win + 1)
        
        # Extract window pixels
        window_img = img[y_min:y_max, x_min:x_max]
        window_mask = blob_mask[y_min:y_max, x_min:x_max]
        
        # Get non-blob pixels in window
        valid_pixels = window_img[~window_mask]
        
        if len(valid_pixels) >= min_samples:
            # Sufficient samples: use mean
            background[y, x] = np.mean(valid_pixels)
        else:
            # Insufficient samples: use fallback strategy
            background[y, x] = _fallback_background(img, y, x, blob_mask)
    
    return background


def _left_right_means(img: np.ndarray, width: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    Mean of *width* pixels immediately to the left / right of each pixel.

    For example, for width=3 and pixel (x,y) the *left* mean is
    the average of {x-3, x-2, x-1} while the *right* mean is the
    average of {x+1, x+2, x+3}.

    Implementation uses a centred uniform filter followed by an integer
    roll (shift) so we never rely on ``origin`` values that exceed the
    SciPy limit.

    Parameters
    ----------
    img : np.ndarray, float32 in [0,1]
    width : int, default=3
        Number of neighbouring pixels to average (must be ≥1).

    Returns
    -------
    left_mean, right_mean : np.ndarray – same shape as *img*
    """
    if width < 1:
        raise ValueError("width must be ≥ 1")

    # Centred moving‑average
    centred = uniform_filter1d(img, size=width, axis=1, mode="nearest")

    half = (width + 1) // 2  # integer shift
    # left: shift kernel centre to the RIGHT  → looks backwards
    left_mean = np.roll(centred, +half, axis=1)
    # right: shift centre to the LEFT → looks forwards
    right_mean = np.roll(centred, -half, axis=1)

    return left_mean, right_mean


@timer
def apply_blob_removal(
    img: np.ndarray,
    s_med: float = 3.0 / 255.0,
    s_avg: float = 20.0 / 255.0,
    gauss_sigma: float = 1.0,
    median_width: int = 5,
    lr_width: int = 3,
    bg_window_size: int = 7,
    min_bg_samples: int = 3,
) -> np.ndarray:
    """
    Apply blob removal using valley & symmetry conditions.
    
    Removes thick blob-like artifacts while preserving scratch-like structures.
    Uses two conditions:
    - Valley condition: |I_g - I_m| >= s_med (Gaussian blur vs horizontal median)
    - Symmetry condition: |L3 - R3| <= s_avg (left vs right pixel means)
    
    Pixels satisfying both conditions are preserved. Blob pixels (not satisfying
    both conditions) are replaced with estimated background values instead of
    being set to black, preventing artificial edge creation.
    
    Parameters
    ----------
    img : np.ndarray
        8-bit grayscale input image (H×W).
    s_med : float, default=3/255
        Threshold for valley condition |I_g - I_m|.
    s_avg : float, default=20/255
        Threshold for symmetry condition |L3 - R3|.
    gauss_sigma : float, default=1.0
        Standard deviation for Gaussian blur.
    median_width : int, default=5
        Width of horizontal median filter (must be odd).
    lr_width : int, default=3
        Width for left/right mean windows.
    bg_window_size : int, default=7
        Window size for local background estimation (must be odd).
    min_bg_samples : int, default=3
        Minimum non-blob pixels needed for valid background estimation.
    
    Returns
    -------
    np.ndarray
        8-bit image with blobs replaced by estimated background (uint8).
    """
    _validate_input(img)
    
    if median_width % 2 == 0:
        raise ValueError("median_width must be odd")
    if lr_width < 1:
        raise ValueError("lr_width must be ≥ 1")
    if bg_window_size % 2 == 0:
        raise ValueError("bg_window_size must be odd")
    if min_bg_samples < 1:
        raise ValueError("min_bg_samples must be ≥ 1")
    
    # Convert to float32 for processing
    img_f32 = img.astype(np.float32) / 255.0
    
    # I_g: Gaussian blur
    I_g = cv2.GaussianBlur(img_f32, ksize=(0, 0), sigmaX=gauss_sigma, borderType=cv2.BORDER_REPLICATE)
    
    # I_m: horizontal median
    I_m = median_filter(img_f32, size=(1, median_width), mode="nearest")
    
    # Left/right means
    L3, R3 = _left_right_means(img_f32, width=lr_width)
    
    # Valley & symmetry conditions
    c1 = np.abs(I_g - I_m) >= s_med  # Valley condition
    c2 = np.abs(L3 - R3) <= s_avg    # Symmetry condition
    
    # Keep pixels that satisfy both conditions (likely scratches)
    # Replace pixels that don't satisfy both conditions (likely blobs)
    mask = np.logical_and(c1, c2)
    blob_mask = ~mask
    
    # Estimate background for blob pixels
    background_img = _estimate_local_background(
        img_f32, blob_mask, bg_window_size, min_bg_samples
    )
    
    # Apply background estimation to blob pixels
    result_f32 = img_f32.copy()
    result_f32[blob_mask] = background_img[blob_mask]
    
    # Convert back to uint8
    result = (result_f32 * 255).astype(np.uint8)
    
    return result


# --------------------------------------------------------------------------- #
# Filter chain system
# --------------------------------------------------------------------------- #
AVAILABLE_FILTERS = {
    'clahe': apply_clahe,
    'median': apply_median,
    'gaussian': apply_gaussian, 
    'sobel': apply_sobel,
    'laplacian': apply_laplacian,
    'blob_removal': apply_blob_removal,
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