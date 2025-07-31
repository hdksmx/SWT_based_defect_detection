"""
glcm.py
=======

GLCM-based texture analysis and filtering for the **wafer_wtms** pipeline.

This module provides advanced texture analysis using Gray-Level Co-occurrence Matrix
(GLCM) features including homogeneity, contrast, energy, correlation, and entropy.
Supports multi-feature analysis, directional robustness, and adaptive parameter tuning.

Public API
----------
apply_glcm_texture_filter(img, **kwargs) -> np.ndarray
    Basic GLCM texture filtering (homogeneity-based)

apply_multi_feature_glcm_filter(img, **kwargs) -> np.ndarray  
    Advanced multi-feature GLCM filtering

apply_multiscale_glcm_filter(img, **kwargs) -> np.ndarray
    Multi-scale GLCM analysis

get_glcm_filters() -> dict[str, callable]
    Export interface for filter chain integration

auto_tune_glcm_params(img, **kwargs) -> dict
    Automatic parameter tuning based on image characteristics

Notes
-----
All GLCM computations use reduced gray levels (default 32) for computational efficiency.
Multi-angle analysis (0°, 45°, 90°, 135°) provides directional robustness.
Smart feature combination optimizes for scratch detection in wafer inspection.
"""

from __future__ import annotations

import math
from typing import Literal

import cv2
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from io_utils import timer, debug_path, save_image

# --------------------------------------------------------------------------- #
# Performance optimization: Global LUT cache
# --------------------------------------------------------------------------- #
_GLCM_FEATURE_CACHE = {}

def _get_homogeneity_lut(levels: int) -> np.ndarray:
    """Get cached homogeneity lookup table."""
    if levels not in _GLCM_FEATURE_CACHE:
        lut = np.zeros((levels, levels), dtype=np.float32)
        for i in range(levels):
            for j in range(levels):
                lut[i, j] = 1.0 / (1.0 + abs(i - j))
        _GLCM_FEATURE_CACHE[levels] = {'homogeneity': lut}
    return _GLCM_FEATURE_CACHE[levels]['homogeneity']

def _get_contrast_lut(levels: int) -> np.ndarray:
    """Get cached contrast lookup table."""
    cache_key = f'contrast_{levels}'
    if cache_key not in _GLCM_FEATURE_CACHE:
        lut = np.zeros((levels, levels), dtype=np.float32)
        for i in range(levels):
            for j in range(levels):
                lut[i, j] = (i - j) ** 2
        _GLCM_FEATURE_CACHE[cache_key] = lut
    return _GLCM_FEATURE_CACHE[cache_key]

def _get_dissimilarity_lut(levels: int) -> np.ndarray:
    """Get cached dissimilarity lookup table."""
    cache_key = f'dissimilarity_{levels}'
    if cache_key not in _GLCM_FEATURE_CACHE:
        lut = np.zeros((levels, levels), dtype=np.float32)
        for i in range(levels):
            for j in range(levels):
                lut[i, j] = abs(i - j)
        _GLCM_FEATURE_CACHE[cache_key] = lut
    return _GLCM_FEATURE_CACHE[cache_key]

__all__ = [
    "apply_glcm_texture_filter",
    "apply_multi_feature_glcm_filter", 
    "apply_multiscale_glcm_filter",
    "get_glcm_filters",
    "auto_tune_glcm_params",
]


# --------------------------------------------------------------------------- #
# Input validation helper
# --------------------------------------------------------------------------- #
def _validate_input(img: np.ndarray, allow_float: bool = False) -> None:
    """Common input validation for all GLCM functions."""
    if allow_float:
        if img.dtype not in (np.uint8, np.float32):
            raise ValueError("Input image must be dtype uint8 or float32.")
    else:
        if img.dtype != np.uint8:
            raise ValueError("Input image must be dtype uint8 (8-bit gray).")
    if img.ndim != 2:
        raise ValueError("Input image must be single-channel gray (H×W).")


def _validate_glcm_params(
    window_size: int,
    distances: list[int],
    angles: list[int], 
    levels: int,
    features: list[str]
) -> None:
    """Validate GLCM computation parameters."""
    if window_size % 2 == 0 or window_size < 3:
        raise ValueError("window_size must be odd and ≥ 3")
    
    if not all(d > 0 for d in distances):
        raise ValueError("All distances must be > 0")
        
    valid_angles = {0, 45, 90, 135}
    if not all(a in valid_angles for a in angles):
        raise ValueError(f"All angles must be in {valid_angles}")
        
    if levels < 4 or levels > 256:
        raise ValueError("levels must be in range [4, 256]")
        
    valid_features = {'homogeneity', 'contrast', 'energy', 'correlation', 'entropy', 'dissimilarity'}
    if not all(f in valid_features for f in features):
        raise ValueError(f"All features must be in {valid_features}")


# --------------------------------------------------------------------------- #
# Core GLCM computation functions
# --------------------------------------------------------------------------- #
def _compute_glcm_matrix(
    window: np.ndarray,
    distance: int = 1,
    angle: int = 0,
    levels: int = 32
) -> np.ndarray:
    """
    Compute GLCM matrix for a single window.
    
    Parameters
    ----------
    window : np.ndarray
        Input window with values in [0, 255].
    distance : int, default=1
        Pixel distance for co-occurrence calculation.
    angle : int, default=0
        Angle in degrees (0, 45, 90, 135).
    levels : int, default=32
        Number of gray levels for quantization.
        
    Returns
    -------
    np.ndarray
        Normalized GLCM matrix (levels × levels).
    """
    h, w = window.shape
    
    # Quantize to reduce gray levels
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
        return np.zeros((levels, levels), dtype=np.float32)
    
    return glcm.astype(np.float32) / total_pairs


def _compute_single_feature(
    glcm: np.ndarray,
    feature: str
) -> float:
    """
    Compute single GLCM feature from normalized GLCM matrix.
    
    Parameters
    ----------
    glcm : np.ndarray
        Normalized GLCM matrix.
    feature : str
        Feature name: 'homogeneity', 'contrast', 'energy', 'correlation', 'entropy', 'dissimilarity'.
        
    Returns
    -------
    float
        Feature value.
    """
    levels = glcm.shape[0]
    
    if feature == 'homogeneity':
        # Homogeneity = Σ P(i,j) / (1 + |i-j|)
        homogeneity = 0.0
        for i in range(levels):
            for j in range(levels):
                if glcm[i, j] > 0:
                    homogeneity += glcm[i, j] / (1.0 + abs(i - j))
        return homogeneity
        
    elif feature == 'contrast':
        # Contrast = Σ P(i,j) * (i-j)²
        contrast = 0.0
        for i in range(levels):
            for j in range(levels):
                contrast += glcm[i, j] * (i - j) ** 2
        return contrast
        
    elif feature == 'energy':
        # Energy (ASM) = Σ P(i,j)²
        energy = 0.0
        for i in range(levels):
            for j in range(levels):
                energy += glcm[i, j] ** 2
        return energy
        
    elif feature == 'correlation':
        # Correlation = Σ ((i-μᵢ)(j-μⱼ)P(i,j)) / (σᵢσⱼ)
        # Compute marginal means
        mu_i = sum(i * sum(glcm[i, :]) for i in range(levels))
        mu_j = sum(j * sum(glcm[:, j]) for j in range(levels))
        
        # Compute marginal standard deviations
        sigma_i = math.sqrt(
            sum((i - mu_i) ** 2 * sum(glcm[i, :]) for i in range(levels))
        )
        sigma_j = math.sqrt(
            sum((j - mu_j) ** 2 * sum(glcm[:, j]) for j in range(levels))
        )
        
        if sigma_i * sigma_j > 1e-10:
            correlation = 0.0
            for i in range(levels):
                for j in range(levels):
                    correlation += (i - mu_i) * (j - mu_j) * glcm[i, j] / (sigma_i * sigma_j)
            return correlation
        else:
            return 0.0
            
    elif feature == 'entropy':
        # Entropy = -Σ P(i,j) * log(P(i,j))
        entropy = 0.0
        for i in range(levels):
            for j in range(levels):
                if glcm[i, j] > 1e-10:  # Avoid log(0)
                    entropy -= glcm[i, j] * math.log(glcm[i, j])
        return entropy
        
    elif feature == 'dissimilarity':
        # Dissimilarity = Σ P(i,j) * |i-j|
        dissimilarity = 0.0
        for i in range(levels):
            for j in range(levels):
                dissimilarity += glcm[i, j] * abs(i - j)
        return dissimilarity
        
    else:
        raise ValueError(f"Unknown feature: {feature}")


def _compute_single_feature_optimized(
    glcm: np.ndarray,
    feature: str
) -> float:
    """
    Compute single GLCM feature using optimized LUT/vectorization approach.
    
    This is the performance-optimized version of _compute_single_feature().
    Uses lookup tables and vectorized operations for significant speed improvement.
    
    Parameters
    ----------
    glcm : np.ndarray
        Normalized GLCM matrix.
    feature : str
        Feature name: 'homogeneity', 'contrast', 'energy', 'correlation', 'entropy', 'dissimilarity'.
        
    Returns
    -------
    float
        Feature value (identical to _compute_single_feature but faster).
    """
    levels = glcm.shape[0]
    
    if feature == 'homogeneity':
        # Optimized with LUT: Homogeneity = Σ P(i,j) / (1 + |i-j|)
        lut = _get_homogeneity_lut(levels)
        return np.sum(glcm * lut)
        
    elif feature == 'contrast':
        # Optimized with LUT: Contrast = Σ P(i,j) * (i-j)²
        lut = _get_contrast_lut(levels)
        return np.sum(glcm * lut)
        
    elif feature == 'energy':
        # Vectorized: Energy (ASM) = Σ P(i,j)²
        return np.sum(glcm ** 2)
        
    elif feature == 'correlation':
        # Vectorized correlation computation
        i_indices, j_indices = np.meshgrid(range(levels), range(levels), indexing='ij')
        
        # Marginal probabilities
        p_i = np.sum(glcm, axis=1)  # Sum over columns
        p_j = np.sum(glcm, axis=0)  # Sum over rows
        
        # Marginal means
        mu_i = np.sum(i_indices[:, 0] * p_i)
        mu_j = np.sum(j_indices[0, :] * p_j)
        
        # Marginal variances
        var_i = np.sum(((i_indices[:, 0] - mu_i) ** 2) * p_i)
        var_j = np.sum(((j_indices[0, :] - mu_j) ** 2) * p_j)
        
        sigma_i = np.sqrt(var_i)
        sigma_j = np.sqrt(var_j)
        
        if sigma_i * sigma_j > 1e-10:
            correlation_matrix = ((i_indices - mu_i) * (j_indices - mu_j)) / (sigma_i * sigma_j)
            return np.sum(glcm * correlation_matrix)
        else:
            return 0.0
            
    elif feature == 'entropy':
        # Vectorized entropy with safe log
        valid_mask = glcm > 1e-10
        entropy_terms = np.zeros_like(glcm)
        entropy_terms[valid_mask] = glcm[valid_mask] * np.log(glcm[valid_mask])
        return -np.sum(entropy_terms)
        
    elif feature == 'dissimilarity':
        # Optimized with LUT: Dissimilarity = Σ P(i,j) * |i-j|
        lut = _get_dissimilarity_lut(levels)
        return np.sum(glcm * lut)
        
    else:
        raise ValueError(f"Unknown feature: {feature}")


def _compute_all_features(
    window: np.ndarray,
    distances: list[int] = [1, 2],
    angles: list[int] = [0, 45, 90, 135],
    levels: int = 32,
    features: list[str] = ['homogeneity', 'contrast', 'energy', 'correlation', 'entropy'],
    use_optimization: bool = False
) -> dict[str, float]:
    """
    Compute all GLCM features for a window across multiple distances and angles.
    
    Parameters
    ----------
    window : np.ndarray
        Input window (typically 11×11).
    distances : list[int], default=[1, 2]
        List of distances for co-occurrence.
    angles : list[int], default=[0, 45, 90, 135]
        List of angles in degrees.
    levels : int, default=32
        Number of gray levels.
    features : list[str]
        List of features to compute.
    use_optimization : bool, default=False
        If True, use optimized LUT/vectorization approach.
        
    Returns
    -------
    dict[str, float]
        Aggregated features with mean, std, max, min across directions.
    """
    all_feature_values = {feature: [] for feature in features}
    
    # Compute features for each distance-angle combination
    for distance in distances:
        for angle in angles:
            glcm = _compute_glcm_matrix(window, distance, angle, levels)
            
            for feature in features:
                if use_optimization:
                    value = _compute_single_feature_optimized(glcm, feature)
                else:
                    value = _compute_single_feature(glcm, feature)
                all_feature_values[feature].append(value)
    
    # Aggregate across directions
    aggregated = {}
    for feature in features:
        values = np.array(all_feature_values[feature])
        aggregated[f'{feature}_mean'] = np.mean(values)
        aggregated[f'{feature}_std'] = np.std(values)
        aggregated[f'{feature}_max'] = np.max(values)
        aggregated[f'{feature}_min'] = np.min(values)
    
    return aggregated


# --------------------------------------------------------------------------- #
# Legacy GLCM functions (moved from prefilter.py)
# --------------------------------------------------------------------------- #
def _compute_glcm_homogeneity(
    window: np.ndarray,
    distance: int = 1,
    angle: int = 0,
    levels: int = 32
) -> float:
    """
    Compute GLCM homogeneity for a single window (legacy function).
    
    This function is kept for backward compatibility.
    New code should use _compute_all_features() instead.
    """
    glcm = _compute_glcm_matrix(window, distance, angle, levels)
    return _compute_single_feature(glcm, 'homogeneity')


def _fast_glcm_homogeneity_map(
    img: np.ndarray,
    window_size: int = 11,
    distance: int = 1,
    angle: int = 0,
    levels: int = 32
) -> np.ndarray:
    """
    Compute GLCM homogeneity map for entire image using sliding window (legacy).
    
    This function is kept for backward compatibility.
    New code should use _compute_multi_feature_map() instead.
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
    blend_range: tuple[float, float] = (0.3, 0.8),
    save_debug_images: bool = False
) -> np.ndarray:
    """
    Apply texture-aware filtering based on GLCM homogeneity (legacy function).
    
    This is the original single-feature GLCM filter, kept for backward compatibility.
    For advanced multi-feature analysis, use apply_multi_feature_glcm_filter() instead.
    
    Parameters
    ----------
    img : np.ndarray
        8-bit grayscale input image (H×W).
    homogeneity_threshold : float, default=0.6
        Threshold for determining homogeneous regions.
    smoothing_sigma : float, default=1.5
        Standard deviation for Gaussian smoothing.
    window_size : int, default=11
        Size of window for GLCM computation (must be odd).
    distance : int, default=1
        Distance for GLCM co-occurrence.
    angle : int, default=0
        Angle for GLCM (0=horizontal).
    levels : int, default=32
        Gray levels for GLCM quantization.
    blend_range : tuple[float, float], default=(0.3, 0.8)
        Range for soft blending.
        
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
    
    # Debug: Save homogeneity map
    if save_debug_images:
        hom_normalized = (homogeneity_map * 255).astype(np.uint8)
        hom_debug = debug_path(1, "1_glcm_homogeneity")
        save_image(hom_normalized, hom_debug)
    
    # 2. Apply Gaussian smoothing to entire image
    smoothed_img = cv2.GaussianBlur(
        img,
        ksize=(0, 0),
        sigmaX=smoothing_sigma,
        borderType=cv2.BORDER_REFLECT
    )
    
    # 3. Create soft blending weights based on homogeneity
    min_hom, max_hom = blend_range
    alpha = np.clip(
        (homogeneity_map - min_hom) / (max_hom - min_hom),
        0.0, 1.0
    )
    
    # 4. Blend original and smoothed images
    result = (
        alpha[..., np.newaxis] * smoothed_img[..., np.newaxis] +
        (1 - alpha[..., np.newaxis]) * img[..., np.newaxis]
    ).squeeze().astype(np.uint8)
    
    # Debug: Save final result
    if save_debug_images:
        result_debug = debug_path(1, "8_glcm_texture_result")
        save_image(result, result_debug)
    
    return result


# --------------------------------------------------------------------------- #
# Multi-feature GLCM analysis
# --------------------------------------------------------------------------- #
def _compute_multi_feature_map(
    img: np.ndarray,
    window_size: int = 11,
    distances: list[int] = [1, 2],
    angles: list[int] = [0, 45, 90, 135],
    levels: int = 32,
    features: list[str] = ['homogeneity', 'contrast', 'energy', 'correlation'],
    use_optimization: bool = False
) -> dict[str, np.ndarray]:
    """
    Compute multi-feature GLCM maps for entire image.
    
    Parameters
    ----------
    img : np.ndarray
        Input image (uint8).
    window_size : int, default=11
        Size of sliding window (must be odd).
    distances : list[int], default=[1, 2]
        List of distances for GLCM computation.
    angles : list[int], default=[0, 45, 90, 135]
        List of angles in degrees.
    levels : int, default=32
        Number of gray levels.
    features : list[str]
        List of features to compute.
    use_optimization : bool, default=False
        If True, use optimized LUT/vectorization approach.
        
    Returns
    -------
    dict[str, np.ndarray]
        Dictionary of feature maps (mean aggregated across directions).
    """
    _validate_glcm_params(window_size, distances, angles, levels, features)
    
    h, w = img.shape
    half_win = window_size // 2
    
    # Initialize feature maps
    feature_maps = {}
    for feature in features:
        feature_maps[f'{feature}_mean'] = np.zeros((h, w), dtype=np.float32)
        feature_maps[f'{feature}_std'] = np.zeros((h, w), dtype=np.float32)
    
    # Use reflection padding
    padded_img = np.pad(img, pad_width=half_win, mode='reflect')
    
    # Compute features for each position
    for y in range(h):
        for x in range(w):
            window = padded_img[y:y + window_size, x:x + window_size]
            
            # Compute all features for this window
            window_features = _compute_all_features(
                window, distances, angles, levels, features, use_optimization
            )
            
            # Store aggregated features
            for key, value in window_features.items():
                if key.endswith('_mean') or key.endswith('_std'):
                    feature_maps[key][y, x] = value
    
    return feature_maps


def _adaptive_feature_weights(
    img: np.ndarray,
    base_weights: dict[str, float] = None
) -> dict[str, float]:
    """
    Compute adaptive feature weights based on image characteristics.
    
    Parameters
    ----------
    img : np.ndarray
        Input image for analysis.
    base_weights : dict[str, float], optional
        Base weights to adjust. If None, uses defaults.
        
    Returns
    -------
    dict[str, float]
        Adaptive feature weights.
    """
    if base_weights is None:
        base_weights = {
            'homogeneity_mean': 0.30,
            'contrast_mean': 0.25,
            'energy_mean': 0.20,
            'correlation_mean': 0.15,
            'entropy_mean': 0.10
        }
    
    weights = base_weights.copy()
    
    # Analyze image characteristics
    mean_intensity = np.mean(img)
    std_intensity = np.std(img)
    global_contrast = std_intensity / 255.0
    
    # Estimate noise level using Laplacian method
    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
    noise_level = laplacian_var / (255.0 ** 2)
    
    # Adaptive adjustments
    if noise_level > 0.01:  # High noise
        weights['homogeneity_mean'] += 0.1  # Emphasize uniformity
        weights['entropy_mean'] -= 0.05     # De-emphasize randomness
        
    if global_contrast < 0.1:  # Low contrast
        weights['contrast_mean'] += 0.1     # Emphasize contrast detection
        weights['energy_mean'] -= 0.05
        
    if global_contrast > 0.3:  # High contrast
        weights['correlation_mean'] += 0.1  # Emphasize directional patterns
        weights['homogeneity_mean'] -= 0.05
    
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v / total_weight for k, v in weights.items()}
    
    return weights


def _combine_features_scratch_optimized(
    feature_maps: dict[str, np.ndarray],
    weights: dict[str, float] = None
) -> np.ndarray:
    """
    Combine GLCM features using scratch-optimized strategy.
    
    This combination is specifically tuned for scratch detection:
    - High homogeneity + low contrast + high energy → uniform background
    - Low homogeneity + high contrast + low energy → potential scratch
    
    Parameters
    ----------
    feature_maps : dict[str, np.ndarray]
        Dictionary of computed feature maps.
    weights : dict[str, float], optional
        Feature weights. If None, uses adaptive weights.
        
    Returns
    -------
    np.ndarray
        Combined texture score map [0,1]. Higher = more uniform background.
    """
    if weights is None:
        # Use dummy image for adaptive weights (simplified)
        sample_img = np.zeros((100, 100), dtype=np.uint8)
        weights = _adaptive_feature_weights(sample_img)
    
    # Initialize combined score
    h, w = next(iter(feature_maps.values())).shape
    combined_score = np.zeros((h, w), dtype=np.float32)
    
    # Combine features with weights
    for feature_key, weight in weights.items():
        if feature_key in feature_maps:
            feature_map = feature_maps[feature_key]
            
            # Normalize feature to [0,1] range
            if feature_key.startswith('homogeneity') or feature_key.startswith('energy'):
                # Higher values indicate uniformity
                normalized = feature_map
            elif feature_key.startswith('contrast') or feature_key.startswith('entropy'):
                # Lower values indicate uniformity (invert)
                max_val = np.max(feature_map)
                if max_val > 0:
                    normalized = 1.0 - (feature_map / max_val)
                else:
                    normalized = np.ones_like(feature_map)
            elif feature_key.startswith('correlation'):
                # Absolute correlation, normalized
                normalized = np.abs(feature_map)
                max_val = np.max(normalized)
                if max_val > 0:
                    normalized = normalized / max_val
            else:
                normalized = feature_map
            
            combined_score += weight * normalized
    
    # Ensure output is in [0,1] range
    combined_score = np.clip(combined_score, 0.0, 1.0)
    
    return combined_score


def _combine_features_weighted_adaptive(
    feature_maps: dict[str, np.ndarray],
    original_img: np.ndarray,
    base_weights: dict[str, float] = None
) -> np.ndarray:
    """
    Combine GLCM features using adaptive weighting based on local image properties.
    
    Adapts feature weights spatially based on:
    - Local noise level: Emphasizes homogeneity in noisy regions
    - Local contrast: Emphasizes contrast features in high-contrast regions
    - Local texture density: Balances features based on texture complexity
    
    Parameters
    ----------
    feature_maps : dict[str, np.ndarray]
        Dictionary of computed feature maps.
    original_img : np.ndarray
        Original image for local property analysis.
    base_weights : dict[str, float], optional
        Base feature weights. If None, uses balanced weights.
        
    Returns
    -------
    np.ndarray
        Combined texture score map [0,1] with spatially adaptive weighting.
    """
    if base_weights is None:
        base_weights = {
            'homogeneity_mean': 0.25,
            'contrast_mean': 0.25,
            'energy_mean': 0.20,
            'correlation_mean': 0.20,
            'entropy_mean': 0.10
        }
    
    h, w = next(iter(feature_maps.values())).shape
    combined_score = np.zeros((h, w), dtype=np.float32)
    
    # Compute local image properties for adaptive weighting
    local_noise = _estimate_local_noise(original_img, window_size=7)
    local_contrast = _compute_local_contrast(original_img, window_size=7)
    local_texture_density = _estimate_local_texture_density(original_img, window_size=7)
    
    # Normalize local properties to [0,1]
    local_noise = _normalize_to_01(local_noise)
    local_contrast = _normalize_to_01(local_contrast)
    local_texture_density = _normalize_to_01(local_texture_density)
    
    # Spatially adaptive combination
    for y in range(h):
        for x in range(w):
            noise_level = local_noise[y, x]
            contrast_level = local_contrast[y, x]
            texture_level = local_texture_density[y, x]
            
            # Compute adaptive weights for this pixel
            adaptive_weights = {}
            for feature_key, base_weight in base_weights.items():
                if feature_key.startswith('homogeneity'):
                    # Emphasize homogeneity in noisy regions
                    adaptive_weights[feature_key] = base_weight * (1.0 + 0.8 * noise_level)
                elif feature_key.startswith('contrast'):
                    # Emphasize contrast in high-contrast regions
                    adaptive_weights[feature_key] = base_weight * (1.0 + 0.6 * contrast_level)
                elif feature_key.startswith('energy'):
                    # Emphasize energy in uniform regions (low texture)
                    adaptive_weights[feature_key] = base_weight * (1.0 + 0.5 * (1.0 - texture_level))
                elif feature_key.startswith('correlation'):
                    # Emphasize correlation in high-texture regions
                    adaptive_weights[feature_key] = base_weight * (1.0 + 0.4 * texture_level)
                elif feature_key.startswith('entropy'):
                    # Emphasize entropy moderately in textured regions
                    adaptive_weights[feature_key] = base_weight * (1.0 + 0.3 * texture_level)
                else:
                    adaptive_weights[feature_key] = base_weight
            
            # Normalize adaptive weights
            total_weight = sum(adaptive_weights.values())
            if total_weight > 0:
                adaptive_weights = {k: v / total_weight for k, v in adaptive_weights.items()}
            
            # Combine features for this pixel
            pixel_score = 0.0
            for feature_key, weight in adaptive_weights.items():
                if feature_key in feature_maps:
                    feature_value = feature_maps[feature_key][y, x]
                    
                    # Normalize feature value
                    if feature_key.startswith('homogeneity') or feature_key.startswith('energy'):
                        normalized_value = feature_value
                    elif feature_key.startswith('contrast') or feature_key.startswith('entropy'):
                        # Invert for uniformity scoring
                        max_val = np.max(feature_maps[feature_key])
                        normalized_value = 1.0 - (feature_value / max_val) if max_val > 0 else 1.0
                    elif feature_key.startswith('correlation'):
                        max_val = np.max(np.abs(feature_maps[feature_key]))
                        normalized_value = abs(feature_value) / max_val if max_val > 0 else 0.0
                    else:
                        normalized_value = feature_value
                    
                    pixel_score += weight * normalized_value
            
            combined_score[y, x] = pixel_score
    
    # Ensure output is in [0,1] range
    combined_score = np.clip(combined_score, 0.0, 1.0)
    
    return combined_score


def _estimate_local_noise(img: np.ndarray, window_size: int = 7) -> np.ndarray:
    """Estimate local noise level using Laplacian variance."""
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    laplacian_abs = np.abs(laplacian)
    
    # Compute local variance of Laplacian
    kernel = np.ones((window_size, window_size), dtype=np.float32) / (window_size ** 2)
    noise_map = cv2.filter2D(laplacian_abs.astype(np.float32), -1, kernel)
    
    return noise_map


def _estimate_local_texture_density(img: np.ndarray, window_size: int = 7) -> np.ndarray:
    """Estimate local texture density using gradient magnitude variance."""
    # Compute gradients
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Local variance of gradient magnitude indicates texture density
    mean_grad = cv2.blur(grad_mag.astype(np.float32), (window_size, window_size))
    mean_grad_sq = cv2.blur((grad_mag**2).astype(np.float32), (window_size, window_size))
    texture_density = mean_grad_sq - mean_grad**2
    
    return np.maximum(texture_density, 0)


def _normalize_to_01(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0,1] range."""
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val > min_val:
        return (arr - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(arr)


def _combine_features_pca_based(
    feature_maps: dict[str, np.ndarray],
    n_components: int = 2,
    explained_variance_threshold: float = 0.85
) -> np.ndarray:
    """
    Combine GLCM features using PCA-based dimensionality reduction.
    
    Uses Principal Component Analysis to:
    1. Reduce dimensionality of multi-feature GLCM data
    2. Focus on components with highest variance (most informative)
    3. Create texture score from principal components
    
    Parameters
    ----------
    feature_maps : dict[str, np.ndarray]
        Dictionary of computed feature maps.
    n_components : int, default=2
        Number of principal components to retain.
    explained_variance_threshold : float, default=0.85
        Minimum explained variance ratio to retain.
        
    Returns
    -------
    np.ndarray
        Combined texture score map [0,1] based on PCA components.
    """
    # Get shape from first feature map
    h, w = next(iter(feature_maps.values())).shape
    
    # Stack all feature maps into a matrix (n_pixels × n_features)
    feature_stack = []
    feature_names = []
    
    for key, feature_map in feature_maps.items():
        if key.endswith('_mean'):  # Use only mean features for PCA
            feature_stack.append(feature_map.flatten())
            feature_names.append(key)
    
    if len(feature_stack) == 0:
        # Fallback: if no mean features, use all features
        for key, feature_map in feature_maps.items():
            feature_stack.append(feature_map.flatten())
            feature_names.append(key)
    
    # Create feature matrix (n_pixels × n_features)
    X = np.column_stack(feature_stack)
    
    # Handle edge case of insufficient features
    n_features = X.shape[1]
    if n_features < 2:
        # If only one feature, return normalized version
        return _normalize_to_01(X[:, 0].reshape(h, w))
    
    # Adjust n_components if needed
    n_components = min(n_components, n_features)
    
    # Standardize features (important for PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Check explained variance and adjust if needed
    explained_variance_ratio = np.sum(pca.explained_variance_ratio_)
    if explained_variance_ratio < explained_variance_threshold and n_components < n_features:
        # Increase components to meet threshold
        while (explained_variance_ratio < explained_variance_threshold and 
               n_components < n_features):
            n_components += 1
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_scaled)
            explained_variance_ratio = np.sum(pca.explained_variance_ratio_)
    
    # Combine principal components into final score
    # Weight by explained variance ratio
    component_weights = pca.explained_variance_ratio_
    component_weights = component_weights / np.sum(component_weights)  # Normalize
    
    # Create texture score from weighted PCA components
    texture_score_flat = np.zeros(X_pca.shape[0])
    for i in range(n_components):
        # Normalize each component to [0,1]
        component = X_pca[:, i]
        component_normalized = _normalize_to_01(component)
        
        # Add weighted contribution
        texture_score_flat += component_weights[i] * component_normalized
    
    # Reshape back to image dimensions
    texture_score = texture_score_flat.reshape(h, w)
    
    # Final normalization to [0,1]
    texture_score = _normalize_to_01(texture_score)
    
    return texture_score


@timer 
def apply_multi_feature_glcm_filter(
    img: np.ndarray,
    window_size: int = 11,
    distances: list[int] = [1, 2],
    angles: list[int] = [0, 45, 90, 135],
    levels: int = 32,
    features: list[str] = ['homogeneity', 'contrast', 'energy', 'correlation'],
    combination_strategy: Literal['scratch_optimized', 'weighted_adaptive', 'pca_based'] = 'scratch_optimized',
    feature_weights: dict[str, float] = None,
    smoothing_sigma: float = 1.5,
    blend_range: tuple[float, float] = (0.3, 0.8),
    use_optimization: bool = False,
    save_debug_images: bool = False
) -> np.ndarray:
    """
    Apply advanced multi-feature GLCM texture filtering.
    
    Uses multiple GLCM features (homogeneity, contrast, energy, correlation, entropy)
    combined across multiple directions and distances for robust texture analysis.
    
    Parameters
    ----------
    img : np.ndarray
        8-bit grayscale input image (H×W).
    window_size : int, default=11
        Size of sliding window for GLCM computation.
    distances : list[int], default=[1, 2]
        List of pixel distances for co-occurrence.
    angles : list[int], default=[0, 45, 90, 135]
        List of angles in degrees for directional analysis.
    levels : int, default=32
        Number of gray levels for GLCM quantization.
    features : list[str]
        List of GLCM features to compute and combine.
    combination_strategy : str, default='scratch_optimized'
        Strategy for combining features:
        - 'scratch_optimized': Optimized for scratch detection
        - 'weighted_adaptive': Spatially adaptive feature weighting
        - 'pca_based': PCA dimensionality reduction
    feature_weights : dict[str, float], optional
        Custom feature weights. If None, uses adaptive weighting.
    smoothing_sigma : float, default=1.5
        Standard deviation for Gaussian smoothing.
    blend_range : tuple[float, float], default=(0.3, 0.8)
        Range for soft blending between original and smoothed.
        
    Returns
    -------
    np.ndarray
        8-bit filtered image with multi-feature texture-aware smoothing.
    """
    _validate_input(img)
    _validate_glcm_params(window_size, distances, angles, levels, features)
    
    # 1. Compute multi-feature maps
    feature_maps = _compute_multi_feature_map(
        img, window_size, distances, angles, levels, features, use_optimization
    )
    
    # Debug: Save individual feature maps
    if save_debug_images:
        debug_counter = 1
        for feature_name in features:
            feature_key = f'{feature_name}_mean'
            if feature_key in feature_maps:
                feature_map = feature_maps[feature_key]
                # Normalize to [0,255] for visualization
                normalized = (feature_map * 255).astype(np.uint8)
                debug_file = debug_path(1, f"{debug_counter}_glcm_{feature_name}")
                save_image(normalized, debug_file)
                debug_counter += 1
    
    # 2. Combine features intelligently
    if combination_strategy == 'scratch_optimized':
        if feature_weights is None:
            feature_weights = _adaptive_feature_weights(img)
        texture_score = _combine_features_scratch_optimized(feature_maps, feature_weights)
        
    elif combination_strategy == 'weighted_adaptive':
        if feature_weights is None:
            feature_weights = {
                'homogeneity_mean': 0.25,
                'contrast_mean': 0.25,
                'energy_mean': 0.20,
                'correlation_mean': 0.20,
                'entropy_mean': 0.10
            }
        texture_score = _combine_features_weighted_adaptive(feature_maps, img, feature_weights)
        
    elif combination_strategy == 'pca_based':
        # PCA-based combination doesn't use feature_weights in the same way
        texture_score = _combine_features_pca_based(feature_maps)
        
    else:
        raise ValueError(f"Unknown combination_strategy: {combination_strategy}")
    
    # Debug: Save combined texture score
    if save_debug_images:
        score_normalized = (texture_score * 255).astype(np.uint8)
        score_debug = debug_path(1, "5_glcm_combined_score")
        save_image(score_normalized, score_debug)
    
    # 3. Apply Gaussian smoothing
    smoothed_img = cv2.GaussianBlur(
        img,
        ksize=(0, 0),
        sigmaX=smoothing_sigma,
        borderType=cv2.BORDER_REFLECT
    )
    
    # Debug: Save smoothed image
    if save_debug_images:
        smoothed_debug = debug_path(1, "6_glcm_smoothed")
        save_image(smoothed_img, smoothed_debug)
    
    # 4. Create blending weights
    min_score, max_score = blend_range
    alpha = np.clip(
        (texture_score - min_score) / (max_score - min_score),
        0.0, 1.0
    )
    
    # Debug: Save alpha blending weights
    if save_debug_images:
        alpha_normalized = (alpha * 255).astype(np.uint8)
        alpha_debug = debug_path(1, "7_glcm_alpha_blend")
        save_image(alpha_normalized, alpha_debug)
    
    # 5. Blend original and smoothed images
    result = (
        alpha[..., np.newaxis] * smoothed_img[..., np.newaxis] +
        (1 - alpha[..., np.newaxis]) * img[..., np.newaxis]
    ).squeeze().astype(np.uint8)
    
    # Debug: Save final result
    if save_debug_images:
        result_debug = debug_path(1, "8_glcm_final_result")
        save_image(result, result_debug)
    
    return result


@timer
def apply_multiscale_glcm_filter(
    img: np.ndarray,
    scales: list[int] = [7, 11, 15],
    features: list[str] = ['homogeneity', 'contrast', 'energy', 'correlation'],
    fusion_strategy: Literal['weighted_average', 'adaptive_fusion'] = 'weighted_average',
    scale_weights: list[float] = None,
    use_optimization: bool = False,
    save_debug_images: bool = False,
    **kwargs
) -> np.ndarray:
    """
    Apply GLCM texture filtering at multiple scales and fuse results.
    
    Multi-scale analysis provides robustness by capturing texture information
    at different levels of detail:
    - Small scales (7×7): Fine texture details, local defects
    - Medium scales (11×11): Balanced local/global texture
    - Large scales (15×15): Global texture patterns, background characterization
    
    Parameters
    ----------
    img : np.ndarray
        8-bit grayscale input image (H×W).
    scales : list[int], default=[7, 11, 15]
        List of window sizes for multi-scale analysis.
    features : list[str]
        GLCM features to compute at each scale.
    fusion_strategy : str, default='weighted_average'
        Strategy for combining multi-scale results:
        - 'weighted_average': Simple weighted combination
        - 'adaptive_fusion': Adaptive fusion based on local statistics
    scale_weights : list[float], optional
        Weights for each scale. If None, uses equal weights.
    **kwargs
        Additional parameters passed to apply_multi_feature_glcm_filter.
        
    Returns
    -------
    np.ndarray
        8-bit filtered image with multi-scale texture analysis applied.
    """
    _validate_input(img)
    
    if len(scales) < 2:
        raise ValueError("At least 2 scales required for multi-scale analysis")
    
    if scale_weights is None:
        scale_weights = [1.0 / len(scales)] * len(scales)
    elif len(scale_weights) != len(scales):
        raise ValueError("scale_weights length must match scales length")
    
    # Normalize weights
    total_weight = sum(scale_weights)
    if total_weight <= 0:
        raise ValueError("scale_weights must sum to positive value")
    scale_weights = [w / total_weight for w in scale_weights]
    
    # Process at each scale
    scale_results = []
    for i, scale in enumerate(scales):
        # Configure parameters for this scale
        scale_params = {
            'window_size': scale,
            'features': features,
            'combination_strategy': 'scratch_optimized',
            'use_optimization': use_optimization,
            'save_debug_images': False,  # Disable debug for individual scales to avoid clutter
            **kwargs
        }
        
        # Apply GLCM filtering at this scale
        scale_result = apply_multi_feature_glcm_filter(img, **scale_params)
        scale_results.append(scale_result)
        
        # Debug: Save individual scale results if debug enabled
        if save_debug_images:
            scale_debug = debug_path(1, f"2{i+1}_glcm_scale_{scale}x{scale}")
            save_image(scale_result, scale_debug)
    
    # Fuse multi-scale results
    if fusion_strategy == 'weighted_average':
        fused_result = _fuse_multiscale_weighted_average(scale_results, scale_weights)
        
    elif fusion_strategy == 'adaptive_fusion':
        fused_result = _fuse_multiscale_adaptive(img, scale_results, scales, scale_weights)
        
    else:
        raise ValueError(f"Unknown fusion_strategy: {fusion_strategy}")
    
    # Debug: Save final multiscale result
    if save_debug_images:
        multiscale_debug = debug_path(1, "25_glcm_multiscale_fused")
        save_image(fused_result, multiscale_debug)
    
    return fused_result


def _fuse_multiscale_weighted_average(
    scale_results: list[np.ndarray],
    weights: list[float]
) -> np.ndarray:
    """
    Fuse multi-scale results using weighted average.
    
    Parameters
    ----------
    scale_results : list[np.ndarray]
        Results from different scales.
    weights : list[float]
        Normalized weights for each scale.
        
    Returns
    -------
    np.ndarray
        Fused result as uint8 image.
    """
    # Convert to float for fusion
    float_results = [result.astype(np.float32) for result in scale_results]
    
    # Weighted combination
    fused_float = np.zeros_like(float_results[0])
    for result, weight in zip(float_results, weights):
        fused_float += weight * result
    
    # Convert back to uint8
    return np.clip(fused_float, 0, 255).astype(np.uint8)


def _fuse_multiscale_adaptive(
    original_img: np.ndarray,
    scale_results: list[np.ndarray],
    scales: list[int],
    base_weights: list[float]
) -> np.ndarray:
    """
    Fuse multi-scale results using adaptive weights based on local image properties.
    
    Adapts fusion weights based on:
    - Local contrast: High contrast regions favor smaller scales
    - Local variance: High variance regions favor medium scales
    - Edge density: Edge-rich regions favor smaller scales
    
    Parameters
    ----------
    original_img : np.ndarray
        Original input image for local analysis.
    scale_results : list[np.ndarray]
        Results from different scales.
    scales : list[int]
        Window sizes used for each scale.
    base_weights : list[float]
        Base weights for each scale.
        
    Returns
    -------
    np.ndarray
        Adaptively fused result as uint8 image.
    """
    h, w = original_img.shape
    fused_result = np.zeros((h, w), dtype=np.float32)
    
    # Compute local image properties
    local_contrast = _compute_local_contrast(original_img, window_size=5)
    local_variance = _compute_local_variance(original_img, window_size=5)
    
    # Adaptive fusion for each pixel
    for y in range(h):
        for x in range(w):
            contrast = local_contrast[y, x]
            variance = local_variance[y, x]
            
            # Compute adaptive weights
            adaptive_weights = []
            for i, (scale, base_weight) in enumerate(zip(scales, base_weights)):
                if scale <= 9:  # Small scale
                    # Favor small scales for high contrast/variance regions
                    weight = base_weight * (1.0 + 0.5 * contrast + 0.3 * variance)
                elif scale >= 13:  # Large scale
                    # Favor large scales for low contrast/variance regions  
                    weight = base_weight * (1.0 + 0.5 * (1.0 - contrast) + 0.3 * (1.0 - variance))
                else:  # Medium scale
                    # Balanced weighting
                    weight = base_weight
                
                adaptive_weights.append(weight)
            
            # Normalize adaptive weights
            total_weight = sum(adaptive_weights)
            if total_weight > 0:
                adaptive_weights = [w / total_weight for w in adaptive_weights]
            else:
                adaptive_weights = base_weights
            
            # Fuse pixel values
            pixel_value = 0.0
            for result, weight in zip(scale_results, adaptive_weights):
                pixel_value += weight * result[y, x].astype(np.float32)
            
            fused_result[y, x] = pixel_value
    
    return np.clip(fused_result, 0, 255).astype(np.uint8)


def _compute_local_contrast(img: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Compute local contrast using standard deviation in local windows."""
    img_float = img.astype(np.float32) / 255.0
    mean_img = cv2.blur(img_float, (window_size, window_size))
    variance_img = cv2.blur(img_float ** 2, (window_size, window_size)) - mean_img ** 2
    contrast_img = np.sqrt(np.maximum(variance_img, 0))
    return contrast_img


def _compute_local_variance(img: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Compute local variance in sliding windows."""
    img_float = img.astype(np.float32) / 255.0
    mean_img = cv2.blur(img_float, (window_size, window_size))
    variance_img = cv2.blur(img_float ** 2, (window_size, window_size)) - mean_img ** 2
    return np.maximum(variance_img, 0)


@timer
def apply_glcm_for_blob_removal(
    img: np.ndarray,
    preserve_scratches: bool = True,
    scratch_threshold: float = 0.4,
    **kwargs
) -> np.ndarray:
    """
    Apply GLCM texture filtering specifically optimized for blob removal preprocessing.
    
    This function is designed to be used as I_g in the blob removal valley condition,
    replacing traditional Gaussian blur with texture-aware processing that:
    - Smooths uniform background regions
    - Preserves linear scratch-like structures
    - Maintains edge information for valley detection
    
    Parameters
    ----------
    img : np.ndarray
        8-bit grayscale input image.
    preserve_scratches : bool, default=True
        If True, uses scratch-optimized texture analysis.
    scratch_threshold : float, default=0.4
        Threshold for scratch preservation (lower = more preservation).
    **kwargs
        Additional parameters passed to apply_multi_feature_glcm_filter.
        
    Returns
    -------
    np.ndarray
        8-bit filtered image optimized for blob removal.
    """
    _validate_input(img)
    
    if preserve_scratches:
        # Scratch-preserving configuration
        scratch_params = {
            'window_size': 9,  # Smaller window for better locality
            'distances': [1],  # Short distance for fine details
            'angles': [0, 90], # Focus on horizontal/vertical (common scratch orientations)
            'levels': 16,      # Fewer levels for speed
            'features': ['homogeneity', 'contrast', 'energy'],  # Key features for scratch detection
            'combination_strategy': 'scratch_optimized',
            'smoothing_sigma': 1.0,  # Gentler smoothing
            'blend_range': (scratch_threshold, 0.9)  # More selective smoothing
        }
        
        # Override with user parameters
        scratch_params.update(kwargs)
        
        return apply_multi_feature_glcm_filter(img, **scratch_params)
    else:
        # Standard texture filtering
        return apply_multi_feature_glcm_filter(img, **kwargs)


# --------------------------------------------------------------------------- #
# Auto-tuning and utilities
# --------------------------------------------------------------------------- #
def auto_tune_glcm_params(
    img: np.ndarray,
    base_config: dict | None = None
) -> dict:
    """
    Auto-tune GLCM parameters based on image characteristics.
    
    Parameters
    ----------
    img : np.ndarray
        Input image for analysis (uint8).
    base_config : dict, optional
        Base configuration to modify. If None, uses defaults.
        
    Returns
    -------
    dict
        Optimized parameters for GLCM filtering functions.
    """
    # Default parameters
    params = {
        'window_size': 11,
        'distances': [1, 2],
        'angles': [0, 45, 90, 135],
        'levels': 32,
        'features': ['homogeneity', 'contrast', 'energy', 'correlation'],
        'smoothing_sigma': 1.5,
        'blend_range': (0.3, 0.8)
    }
    
    # Override with base config
    if base_config:
        params.update(base_config)
    
    # Analyze image characteristics
    mean_intensity = np.mean(img)
    std_intensity = np.std(img)
    
    # Estimate noise level
    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
    noise_level = laplacian_var / (255.0 ** 2)
    
    # Estimate contrast variation
    local_std = cv2.GaussianBlur(
        (img.astype(np.float32) - mean_intensity) ** 2,
        ksize=(5, 5),
        sigmaX=1.0
    )
    contrast_variation = np.sqrt(local_std).mean() / 255.0
    
    # Adaptive adjustments
    if noise_level > 0.01:  # High noise
        params['window_size'] = 15
        params['smoothing_sigma'] = 2.0
        params['levels'] = 16
        
    elif noise_level < 0.001:  # Very low noise
        params['window_size'] = 9
        params['smoothing_sigma'] = 1.0
    
    if contrast_variation < 0.05:  # Low contrast
        params['blend_range'] = (0.2, 0.7)
        
    elif contrast_variation > 0.15:  # High contrast
        params['blend_range'] = (0.4, 0.9)
    
    if mean_intensity < 50:  # Dark images
        params['levels'] = 16
    elif mean_intensity > 200:  # Bright images
        params['levels'] = 48
    
    # Ensure validity
    params['window_size'] = max(7, params['window_size'])
    if params['window_size'] % 2 == 0:
        params['window_size'] += 1
        
    params['smoothing_sigma'] = np.clip(params['smoothing_sigma'], 0.5, 3.0)
    params['levels'] = np.clip(params['levels'], 8, 64)
    
    return params


# --------------------------------------------------------------------------- #
# Export interface for filter chain integration
# --------------------------------------------------------------------------- #
def get_glcm_filters() -> dict[str, callable]:
    """
    Export GLCM filters for registration in main filter chain.
    
    Returns
    -------
    dict[str, callable]
        Dictionary mapping filter names to filter functions.
    """
    return {
        'glcm_texture': apply_glcm_texture_filter,          # Legacy single-feature
        'glcm_multi_feature': apply_multi_feature_glcm_filter,  # Advanced multi-feature
        'glcm_multiscale': apply_multiscale_glcm_filter,    # Multi-scale analysis
        'glcm_blob_removal': apply_glcm_for_blob_removal,   # Blob removal optimized
    }


# Default GLCM configuration
GLCM_DEFAULT_CONFIG = {
    'window_size': 11,
    'distances': [1, 2],
    'angles': [0, 45, 90, 135],
    'levels': 32,
    'features': ['homogeneity', 'contrast', 'energy', 'correlation', 'entropy'],
    'combination_strategy': 'scratch_optimized',
    'smoothing_sigma': 1.5,
    'blend_range': (0.3, 0.8)
}