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
        raise ValueError("Input image must be single-channel gray (HxW).")


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
        8-bit grayscale input image (HxW).
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
        8-bit grayscale input image (HxW).
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
        8-bit grayscale input image (HxW).
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
        8-bit grayscale input image (HxW).
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
        8-bit grayscale input image (HxW).
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


# DEPRECATED: GLCM functions moved to glcm.py module
# The following functions are kept for reference but should not be used.
# Use glcm.py module instead.

def _compute_glcm_homogeneity(
    window: np.ndarray,
    distance: int = 1,
    angle: int = 0,
    levels: int = 32
) -> float:
    """
    Compute GLCM homogeneity for a single window.
    
    Homogeneity measures the uniformity of texture by computing:
    Σ P(i,j) / (1 + |i-j|)
    
    where P(i,j) is the normalized GLCM matrix.
    Higher values indicate more homogeneous (uniform) texture.
    
    Parameters
    ----------
    window : np.ndarray
        Input window (typically 11x11) with values in [0,255].
    distance : int, default=1
        Pixel distance for co-occurrence calculation.
    angle : int, default=0
        Angle in degrees (0, 45, 90, 135). 0 = horizontal.
    levels : int, default=32
        Number of gray levels for quantization (reduced for speed).
        
    Returns
    -------
    float
        Homogeneity value in [0,1]. Higher = more homogeneous.
    """
    h, w = window.shape
    
    # Quantize to reduce gray levels for computational efficiency
    quantized = (window * (levels - 1) / 255.0).astype(np.int32)
    quantized = np.clip(quantized, 0, levels - 1)
    
    # Initialize GLCM matrix
    glcm = np.zeros((levels, levels), dtype=np.int32)
    
    # Compute co-occurrence pairs based on angle
    if angle == 0:  # Horizontal (0°)
        for r in range(h):
            for c in range(w - distance):
                i, j = quantized[r, c], quantized[r, c + distance]
                glcm[i, j] += 1
                glcm[j, i] += 1  # Symmetric
                
    elif angle == 45:  # Diagonal (45°)
        for r in range(h - distance):
            for c in range(w - distance):
                i, j = quantized[r, c], quantized[r + distance, c + distance]
                glcm[i, j] += 1
                glcm[j, i] += 1
                
    elif angle == 90:  # Vertical (90°)
        for r in range(h - distance):
            for c in range(w):
                i, j = quantized[r, c], quantized[r + distance, c]
                glcm[i, j] += 1
                glcm[j, i] += 1
                
    elif angle == 135:  # Anti-diagonal (135°)
        for r in range(h - distance):
            for c in range(distance, w):
                i, j = quantized[r, c], quantized[r + distance, c - distance]
                glcm[i, j] += 1
                glcm[j, i] += 1
    else:
        raise ValueError("angle must be one of {0, 45, 90, 135}")
    
    # Normalize GLCM
    total_pairs = glcm.sum()
    if total_pairs == 0:
        return 0.0
    
    glcm_normalized = glcm.astype(np.float32) / total_pairs
    
    # Compute homogeneity
    homogeneity = 0.0
    for i in range(levels):
        for j in range(levels):
            if glcm_normalized[i, j] > 0:
                homogeneity += glcm_normalized[i, j] / (1.0 + abs(i - j))
    
    return homogeneity


def _fast_glcm_homogeneity_map(
    img: np.ndarray,
    window_size: int = 11,
    distance: int = 1,
    angle: int = 0,
    levels: int = 32
) -> np.ndarray:
    """
    Compute GLCM homogeneity map for entire image using sliding window.
    
    For each pixel, computes the homogeneity of the surrounding window.
    Border pixels use reflected padding.
    
    Parameters
    ----------
    img : np.ndarray
        Input image (uint8, single channel).
    window_size : int, default=11
        Size of sliding window (must be odd).
    distance : int, default=1
        Distance for GLCM computation.
    angle : int, default=0
        Angle for GLCM computation (0, 45, 90, 135).
    levels : int, default=32
        Number of gray levels for quantization.
        
    Returns
    -------
    np.ndarray
        Homogeneity map (float32) with same shape as input.
        Values in [0,1] where higher = more homogeneous.
    """
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    
    h, w = img.shape
    half_win = window_size // 2
    homogeneity_map = np.zeros((h, w), dtype=np.float32)
    
    # Use reflection padding to handle borders
    padded_img = np.pad(
        img, 
        pad_width=half_win, 
        mode='reflect'
    )
    
    # Compute homogeneity for each position
    for y in range(h):
        for x in range(w):
            # Extract window from padded image
            window = padded_img[
                y:y + window_size,
                x:x + window_size
            ]
            
            # Compute homogeneity for this window
            homogeneity_map[y, x] = _compute_glcm_homogeneity(
                window, distance, angle, levels
            )
    
    return homogeneity_map


@timer
def apply_glcm_texture_filter(
    img: np.ndarray,
    homogeneity_threshold: float = 0.6,
    smoothing_sigma: float = 1.5,
    window_size: int = 11,
    distance: int = 1,
    angle: int = 0,
    levels: int = 32,
    blend_range: tuple[float, float] = (0.3, 0.8)
) -> np.ndarray:
    """
    Apply texture-aware filtering based on GLCM homogeneity.
    
    Regions with high homogeneity (uniform texture/background) are smoothed
    to reduce noise, while regions with low homogeneity (potential defects)
    are preserved to maintain detail.
    
    Parameters
    ----------
    img : np.ndarray
        8-bit grayscale input image (HxW).
    homogeneity_threshold : float, default=0.6
        Threshold for determining homogeneous regions.
        Higher values = more selective smoothing.
    smoothing_sigma : float, default=1.5
        Standard deviation for Gaussian smoothing.
    window_size : int, default=11
        Size of window for GLCM computation (must be odd).
    distance : int, default=1
        Distance for GLCM co-occurrence.
    angle : int, default=0
        Angle for GLCM (0=horizontal, useful for scratch detection).
    levels : int, default=32
        Gray levels for GLCM quantization.
    blend_range : tuple[float, float], default=(0.3, 0.8)
        Range for soft blending (min_homogeneity, max_homogeneity).
        
    Returns
    -------
    np.ndarray
        8-bit filtered image with texture-aware smoothing applied.
    """
    _validate_input(img)
    
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    if not (0.0 <= homogeneity_threshold <= 1.0):
        raise ValueError("homogeneity_threshold must be in [0,1]")
    if blend_range[0] >= blend_range[1]:
        raise ValueError("blend_range[0] must be < blend_range[1]")
    
    # 1. Compute homogeneity map
    homogeneity_map = _fast_glcm_homogeneity_map(
        img, window_size, distance, angle, levels
    )
    
    # 2. Apply Gaussian smoothing to entire image
    smoothed_img = cv2.GaussianBlur(
        img,
        ksize=(0, 0),
        sigmaX=smoothing_sigma,
        borderType=cv2.BORDER_REFLECT
    )
    
    # 3. Create soft blending weights based on homogeneity
    # Linear interpolation between blend_range
    min_hom, max_hom = blend_range
    alpha = np.clip(
        (homogeneity_map - min_hom) / (max_hom - min_hom),
        0.0, 1.0
    )
    
    # 4. Blend original and smoothed images
    # alpha=0 → keep original (low homogeneity, potential defects)
    # alpha=1 → use smoothed (high homogeneity, background)
    result = (
        alpha[..., np.newaxis] * smoothed_img[..., np.newaxis] +
        (1 - alpha[..., np.newaxis]) * img[..., np.newaxis]
    ).squeeze().astype(np.uint8)
    
    return result


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
    use_glcm_texture: bool = False,
    glcm_params: dict = None,
) -> np.ndarray:
    """
    Apply blob removal using valley & symmetry conditions with optional GLCM integration.
    
    Removes thick blob-like artifacts while preserving scratch-like structures.
    Uses two conditions:
    - Valley condition: |I_g - I_m| >= s_med (texture-aware blur vs horizontal median)  
    - Symmetry condition: |L3 - R3| <= s_avg (left vs right pixel means)
    
    When use_glcm_texture=True, replaces global Gaussian blur with GLCM-based
    texture-aware filtering that preserves scratch details while smoothing background.
    
    Parameters
    ----------
    img : np.ndarray
        8-bit grayscale input image (HxW).
    s_med : float, default=3/255
        Threshold for valley condition |I_g - I_m|.
    s_avg : float, default=20/255
        Threshold for symmetry condition |L3 - R3|.
    gauss_sigma : float, default=1.0
        Standard deviation for Gaussian blur (used when use_glcm_texture=False).
    median_width : int, default=5
        Width of horizontal median filter (must be odd).
    lr_width : int, default=3
        Width for left/right mean windows.
    bg_window_size : int, default=7
        Window size for local background estimation (must be odd).
    min_bg_samples : int, default=3
        Minimum non-blob pixels needed for valid background estimation.
    use_glcm_texture : bool, default=False
        If True, use GLCM-based texture-aware filtering instead of Gaussian blur.
    glcm_params : dict, optional
        Parameters for GLCM texture filtering. If None, uses defaults.
    
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
    
    # I_g: texture-aware blur (GLCM-based) or traditional Gaussian blur
    if use_glcm_texture:
        # Import GLCM module only when needed to avoid circular imports
        try:
            from glcm import apply_glcm_for_blob_removal
            
            # Prepare GLCM parameters with blob removal optimization
            if glcm_params is None:
                glcm_params = {
                    'preserve_scratches': True,
                    'scratch_threshold': 0.4,  # Preserve potential scratches
                    'window_size': 9,
                    'smoothing_sigma': 1.0
                }
            
            # Apply GLCM-based texture filtering optimized for blob removal
            I_g_uint8 = apply_glcm_for_blob_removal(img, **glcm_params)
            I_g = I_g_uint8.astype(np.float32) / 255.0
            
        except ImportError:
            # Fallback to Gaussian blur if GLCM module not available
            I_g = cv2.GaussianBlur(img_f32, ksize=(0, 0), sigmaX=gauss_sigma, borderType=cv2.BORDER_REPLICATE)
    else:
        # Traditional Gaussian blur
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


def auto_tune_glcm_params(
    img: np.ndarray,
    base_config: dict | None = None
) -> dict:
    """
    Auto-tune GLCM texture filter parameters based on image characteristics.
    
    Analyzes image statistics to adjust parameters for optimal performance:
    - High noise → larger window, stronger smoothing
    - Low contrast → lower threshold, more aggressive smoothing
    - High texture density → smaller window, more selective smoothing
    
    Parameters
    ----------
    img : np.ndarray
        Input image for analysis (uint8).
    base_config : dict, optional
        Base configuration to modify. If None, uses defaults.
        
    Returns
    -------
    dict
        Optimized parameters for apply_glcm_texture_filter.
    """
    # Default parameters
    params = {
        'homogeneity_threshold': 0.6,
        'smoothing_sigma': 1.5,
        'window_size': 11,
        'distance': 1,
        'angle': 0,
        'levels': 32,
        'blend_range': (0.3, 0.8)
    }
    
    # Override with base config if provided
    if base_config:
        params.update(base_config)
    
    # Estimate image characteristics
    mean_intensity = np.mean(img)
    std_intensity = np.std(img)
    
    # Estimate noise level using Laplacian method
    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
    noise_level = laplacian_var / (255.0 ** 2)  # Normalize
    
    # Estimate local contrast variation
    local_std = cv2.GaussianBlur(
        (img.astype(np.float32) - mean_intensity) ** 2,
        ksize=(5, 5),
        sigmaX=1.0
    )
    contrast_variation = np.sqrt(local_std).mean() / 255.0
    
    # Adaptive parameter adjustment
    
    # 1. Noise-based adjustments
    if noise_level > 0.01:  # High noise
        params['window_size'] = 15  # Larger window for better statistics
        params['smoothing_sigma'] = 2.0  # Stronger smoothing
        params['levels'] = 16  # Fewer levels for noise robustness
        
    elif noise_level < 0.001:  # Very low noise
        params['window_size'] = 9  # Smaller window for better locality
        params['smoothing_sigma'] = 1.0  # Gentler smoothing
    
    # 2. Contrast-based adjustments
    if contrast_variation < 0.05:  # Low contrast
        params['homogeneity_threshold'] = 0.5  # More aggressive smoothing
        params['blend_range'] = (0.2, 0.7)  # Wider blending range
        
    elif contrast_variation > 0.15:  # High contrast
        params['homogeneity_threshold'] = 0.7  # More selective smoothing
        params['blend_range'] = (0.4, 0.9)  # Narrower blending range
    
    # 3. Intensity-based adjustments
    if mean_intensity < 50:  # Dark images
        params['levels'] = 16  # Fewer levels for dark regions
        
    elif mean_intensity > 200:  # Bright images
        params['levels'] = 48  # More levels for bright regions
    
    # 4. Ensure parameter validity
    params['window_size'] = max(7, params['window_size'])
    if params['window_size'] % 2 == 0:
        params['window_size'] += 1  # Ensure odd
        
    params['homogeneity_threshold'] = np.clip(
        params['homogeneity_threshold'], 0.1, 0.9
    )
    params['smoothing_sigma'] = np.clip(
        params['smoothing_sigma'], 0.5, 3.0
    )
    params['levels'] = np.clip(params['levels'], 8, 64)
    
    # Ensure blend_range is valid
    min_blend, max_blend = params['blend_range']
    if min_blend >= max_blend:
        params['blend_range'] = (0.3, 0.8)  # Reset to default
    
    return params



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

# Dynamic registration of GLCM filters
def _register_glcm_filters():
    """Dynamically register GLCM filters from glcm module."""
    try:
        from glcm import get_glcm_filters
        AVAILABLE_FILTERS.update(get_glcm_filters())
    except ImportError:
        # GLCM module not available, skip registration
        pass

# Register GLCM filters on module import
_register_glcm_filters()


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
        8-bit grayscale input image (HxW).
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
        8‑bit gray‑scale input image (HxW).
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