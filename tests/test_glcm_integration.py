"""
test_glcm_integration.py
========================

HIGH priority integration tests for GLCM system.
Tests critical components that must work together for the system to function.

Test Priority:
- HIGH: CLI→Pipeline parameter passing 
- HIGH: Basic filter chain execution
- MEDIUM: Debugging system (not included in this module)
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import Mock, patch

import numpy as np
import pytest

try:
    # Import main modules
    from pipeline import PipelineConfig
    from prefilter import apply_filter_chain
    
    # Test if GLCM is available
    try:
        from glcm import apply_multi_feature_glcm_filter
        GLCM_AVAILABLE = True
    except ImportError:
        GLCM_AVAILABLE = False
        
except ImportError as e:
    pytest.skip(f"Could not import required modules: {e}", allow_module_level=True)


# --------------------------------------------------------------------------- #
# Test fixtures and utilities
# --------------------------------------------------------------------------- #

def create_synthetic_test_image(height: int = 64, width: int = 64) -> np.ndarray:
    """Create synthetic test image with known patterns."""
    np.random.seed(42)  # Reproducible results
    
    # Base image with gradient
    img = np.zeros((height, width), dtype=np.uint8)
    
    # Add gradient background
    for i in range(height):
        for j in range(width):
            img[i, j] = int(128 + 50 * np.sin(i / 10) + 30 * np.cos(j / 8))
    
    # Add some texture patterns
    img[10:20, 10:50] = 200  # Bright horizontal stripe
    img[30:35, 10:50] = 50   # Dark horizontal stripe
    
    # Add noise
    noise = np.random.normal(0, 10, (height, width)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img


class MockArgs:
    """Mock argparse.Namespace for testing CLI argument parsing."""
    
    def __init__(self, **kwargs):
        # Default values based on CLI defaults
        defaults = {
            'debug': False,
            'glcm_optimize': False,
            'glcm_window_size': 11,
            'glcm_levels': 32,
            'glcm_distances': [1, 2],
            'glcm_angles': [0, 45, 90, 135],
            'glcm_features': ['homogeneity', 'contrast', 'energy', 'correlation'],
            'glcm_combination_strategy': 'scratch_optimized',
            'glcm_smoothing_sigma': 1.5,
            'glcm_multiscale_scales': [7, 11, 15],
            'glcm_multiscale_fusion': 'weighted_average',
            'prefilter_chain': ['median', 'sobel'],
            'median_ksize': 3,
            'sobel_ksize': 3,
            'std_factor': 3.0,
            'wavelet_name': 'coif6',
            'window_hw': 3,
            'max_steps': 5,
            'min_region_area': 5,
            'ecc_thr': 0.9,
            'len_thr': 20,
            'gs': None,
            'clahe_clip_limit': 2.0,
            'clahe_tile_size': 8,
            'gaussian_ksize': 3,
            'gaussian_sigma': 1.0,
            'laplacian_ksize': 3,
            'blob_s_med': 3.0 / 255.0,
            'blob_s_avg': 20.0 / 255.0,
            'blob_gauss_sigma': 1.0,
            'blob_median_width': 5,
            'blob_lr_width': 3,
            'blob_use_glcm_texture': False,
        }
        
        # Update with provided kwargs
        defaults.update(kwargs)
        
        # Set attributes
        for key, value in defaults.items():
            setattr(self, key, value)


def build_pipeline_config_from_mock_args(args: MockArgs) -> PipelineConfig:
    """Build PipelineConfig from mock args (simulates cli.py logic)."""
    # Build base prefilter parameters (non-GLCM filters)
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
        # GLCM parameters - passed directly to PipelineConfig fields
        use_glcm_optimization=args.glcm_optimize,
        glcm_window_size=args.glcm_window_size,
        glcm_levels=args.glcm_levels,
        glcm_distances=args.glcm_distances,
        glcm_angles=args.glcm_angles,
        glcm_features=args.glcm_features,
        glcm_combination_strategy=args.glcm_combination_strategy,
        glcm_smoothing_sigma=args.glcm_smoothing_sigma,
        glcm_multiscale_scales=args.glcm_multiscale_scales,
        glcm_multiscale_fusion=args.glcm_multiscale_fusion,
        # Debugging options
        enable_debug_output=args.debug,
    )
    
    return cfg


# --------------------------------------------------------------------------- #
# HIGH Priority Tests
# --------------------------------------------------------------------------- #

class TestCLIPipelineParameterPassing:
    """Test CLI arguments are correctly passed to PipelineConfig."""
    
    def test_basic_glcm_parameters(self):
        """Test basic GLCM parameters are passed correctly."""
        # Create mock CLI arguments
        mock_args = MockArgs(
            glcm_optimize=True,
            glcm_window_size=9,
            glcm_levels=16,
            glcm_features=['homogeneity', 'contrast'],
            glcm_combination_strategy='weighted_adaptive'
        )
        
        # Build PipelineConfig
        cfg = build_pipeline_config_from_mock_args(mock_args)
        
        # Verify GLCM parameters
        assert cfg.use_glcm_optimization == True
        assert cfg.glcm_window_size == 9
        assert cfg.glcm_levels == 16
        assert cfg.glcm_features == ['homogeneity', 'contrast']
        assert cfg.glcm_combination_strategy == 'weighted_adaptive'
    
    def test_debug_flag_integration(self):
        """Test --debug flag is properly integrated."""
        mock_args = MockArgs(debug=True)
        cfg = build_pipeline_config_from_mock_args(mock_args)
        
        assert cfg.enable_debug_output == True
        
        # Test that debug flag is passed to filter parameters
        params = cfg.build_prefilter_params()
        if 'glcm_multi_feature' in params:
            assert params['glcm_multi_feature']['save_debug_images'] == True
    
    def test_prefilter_chain_glcm_integration(self):
        """Test GLCM filters in prefilter chain."""
        mock_args = MockArgs(
            prefilter_chain=['glcm_multi_feature', 'sobel'],
            glcm_features=['homogeneity', 'energy']
        )
        cfg = build_pipeline_config_from_mock_args(mock_args)
        
        assert 'glcm_multi_feature' in cfg.prefilter_chain
        assert 'sobel' in cfg.prefilter_chain
        assert cfg.glcm_features == ['homogeneity', 'energy']
    
    def test_multiscale_parameters(self):
        """Test multiscale GLCM parameters."""
        mock_args = MockArgs(
            prefilter_chain=['glcm_multiscale', 'sobel'],
            glcm_multiscale_scales=[5, 9, 13],
            glcm_multiscale_fusion='adaptive_fusion'
        )
        cfg = build_pipeline_config_from_mock_args(mock_args)
        
        assert cfg.glcm_multiscale_scales == [5, 9, 13]
        assert cfg.glcm_multiscale_fusion == 'adaptive_fusion'
        
        # Check parameters are built correctly
        params = cfg.build_prefilter_params()
        if 'glcm_multiscale' in params:
            assert params['glcm_multiscale']['scales'] == [5, 9, 13]
            assert params['glcm_multiscale']['fusion_strategy'] == 'adaptive_fusion'


class TestFilterChainExecution:
    """Test GLCM filters execute correctly in filter chains."""
    
    @pytest.mark.skipif(not GLCM_AVAILABLE, reason="GLCM module not available")
    def test_basic_filter_chain_with_glcm(self):
        """Test basic filter chain execution with GLCM."""
        # Create test image
        test_img = create_synthetic_test_image(32, 32)
        
        # Configure simple GLCM pipeline
        cfg = PipelineConfig(
            prefilter_chain=['glcm_multi_feature', 'sobel'],
            glcm_features=['homogeneity', 'contrast'],
            glcm_window_size=7,  # Smaller for faster testing
            enable_debug_output=False
        )
        
        # Execute filter chain
        try:
            result = apply_filter_chain(
                test_img, 
                cfg.prefilter_chain, 
                cfg.build_prefilter_params()
            )
            
            # Verify basic properties
            assert result is not None
            assert isinstance(result, np.ndarray)
            assert result.shape == test_img.shape
            assert result.dtype == np.float32  # Sobel outputs float32
            
        except Exception as e:
            pytest.fail(f"Filter chain execution failed: {e}")
    
    @pytest.mark.skipif(not GLCM_AVAILABLE, reason="GLCM module not available")
    def test_glcm_optimization_flag(self):
        """Test optimization flag is passed through correctly."""
        test_img = create_synthetic_test_image(32, 32)
        
        # Test without optimization
        cfg_normal = PipelineConfig(
            prefilter_chain=['glcm_multi_feature'],
            glcm_features=['homogeneity'],
            glcm_window_size=7,
            use_glcm_optimization=False,
            enable_debug_output=False
        )
        
        # Test with optimization
        cfg_optimized = PipelineConfig(
            prefilter_chain=['glcm_multi_feature'],
            glcm_features=['homogeneity'],
            glcm_window_size=7,
            use_glcm_optimization=True,
            enable_debug_output=False
        )
        
        try:
            result_normal = apply_filter_chain(
                test_img, 
                cfg_normal.prefilter_chain, 
                cfg_normal.build_prefilter_params()
            )
            
            result_optimized = apply_filter_chain(
                test_img, 
                cfg_optimized.prefilter_chain, 
                cfg_optimized.build_prefilter_params()
            )
            
            # Both should produce valid results
            assert result_normal is not None
            assert result_optimized is not None
            assert result_normal.shape == result_optimized.shape
            
            # Results should be similar (allowing for small numerical differences)
            assert np.allclose(result_normal, result_optimized, rtol=1e-3, atol=1e-3)
            
        except Exception as e:
            pytest.fail(f"Optimization flag test failed: {e}")
    
    def test_legacy_filter_chain_still_works(self):
        """Test that legacy median+sobel chain still works."""
        test_img = create_synthetic_test_image(32, 32)
        
        # Legacy configuration
        cfg = PipelineConfig(
            prefilter_chain=['median', 'sobel'],
            median_kernel=3,
            sobel_kernel=3
        )
        
        try:
            result = apply_filter_chain(
                test_img, 
                cfg.prefilter_chain, 
                cfg.build_prefilter_params()
            )
            
            assert result is not None
            assert result.shape == test_img.shape
            assert result.dtype == np.float32
            
        except Exception as e:
            pytest.fail(f"Legacy filter chain failed: {e}")
    
    @pytest.mark.skipif(not GLCM_AVAILABLE, reason="GLCM module not available")
    def test_parameter_validation_in_chain(self):
        """Test that invalid parameters are caught in filter chain."""
        test_img = create_synthetic_test_image(32, 32)
        
        # Invalid window size (even number)
        cfg_invalid = PipelineConfig(
            prefilter_chain=['glcm_multi_feature'],
            glcm_window_size=8,  # Even number should fail
            enable_debug_output=False
        )
        
        with pytest.raises((ValueError, Exception)):
            apply_filter_chain(
                test_img, 
                cfg_invalid.prefilter_chain, 
                cfg_invalid.build_prefilter_params()
            )
    
    @pytest.mark.skipif(not GLCM_AVAILABLE, reason="GLCM module not available")
    def test_multiscale_filter_execution(self):
        """Test multiscale GLCM filter execution."""
        test_img = create_synthetic_test_image(48, 48)  # Larger for multiscale
        
        cfg = PipelineConfig(
            prefilter_chain=['glcm_multiscale'],
            glcm_multiscale_scales=[5, 9],  # Small scales for speed
            glcm_features=['homogeneity'],
            enable_debug_output=False
        )
        
        try:
            result = apply_filter_chain(
                test_img,
                cfg.prefilter_chain,
                cfg.build_prefilter_params()
            )
            
            assert result is not None
            assert result.shape == test_img.shape
            assert result.dtype == np.uint8  # Multiscale returns uint8
            
        except Exception as e:
            pytest.fail(f"Multiscale filter execution failed: {e}")


class TestParameterBuildingLogic:
    """Test build_prefilter_params() method logic."""
    
    def test_glcm_parameters_built_correctly(self):
        """Test GLCM parameters are built correctly."""
        cfg = PipelineConfig(
            glcm_window_size=13,
            glcm_levels=24,
            glcm_distances=[1, 3],
            glcm_angles=[0, 90],
            glcm_features=['homogeneity', 'energy'],
            glcm_combination_strategy='pca_based',
            use_glcm_optimization=True,
            enable_debug_output=True
        )
        
        params = cfg.build_prefilter_params()
        
        # Check glcm_multi_feature parameters
        glcm_params = params.get('glcm_multi_feature', {})
        assert glcm_params['window_size'] == 13
        assert glcm_params['levels'] == 24
        assert glcm_params['distances'] == [1, 3]
        assert glcm_params['angles'] == [0, 90]
        assert glcm_params['features'] == ['homogeneity', 'energy']
        assert glcm_params['combination_strategy'] == 'pca_based'
        assert glcm_params['use_optimization'] == True
        assert glcm_params['save_debug_images'] == True
    
    def test_multiscale_parameters_built_correctly(self):
        """Test multiscale parameters are built correctly."""
        cfg = PipelineConfig(
            glcm_multiscale_scales=[7, 13, 19],
            glcm_multiscale_fusion='adaptive_fusion',
            use_glcm_optimization=False,
            enable_debug_output=False
        )
        
        params = cfg.build_prefilter_params()
        
        multiscale_params = params.get('glcm_multiscale', {})
        assert multiscale_params['scales'] == [7, 13, 19]
        assert multiscale_params['fusion_strategy'] == 'adaptive_fusion'
        assert multiscale_params['use_optimization'] == False
        assert multiscale_params['save_debug_images'] == False
    
    def test_blob_removal_glcm_integration(self):
        """Test blob removal GLCM integration parameters."""
        cfg = PipelineConfig(
            prefilter_chain=['glcm_multi_feature', 'blob_removal'],
            glcm_window_size=11,
            enable_debug_output=True
        )
        
        params = cfg.build_prefilter_params()
        
        # Check blob_removal gets GLCM integration settings
        blob_params = params.get('blob_removal', {})
        assert 'use_glcm_texture' in blob_params
        assert 'glcm_params' in blob_params
        
        glcm_params = blob_params['glcm_params']
        assert glcm_params['window_size'] == 11
        assert glcm_params['preserve_scratches'] == True


# --------------------------------------------------------------------------- #
# Test runner for manual execution
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    """Run tests manually for debugging."""
    
    print("🧪 Running HIGH Priority GLCM Integration Tests")
    print("=" * 60)
    
    # Test 1: CLI Parameter Passing
    print("\n1. Testing CLI → Pipeline parameter passing...")
    try:
        test_class = TestCLIPipelineParameterPassing()
        test_class.test_basic_glcm_parameters()
        test_class.test_debug_flag_integration()
        test_class.test_prefilter_chain_glcm_integration()
        test_class.test_multiscale_parameters()
        print("✅ CLI parameter passing tests PASSED")
    except Exception as e:
        print(f"❌ CLI parameter passing tests FAILED: {e}")
    
    # Test 2: Filter Chain Execution
    print("\n2. Testing filter chain execution...")
    try:
        test_class = TestFilterChainExecution()
        if GLCM_AVAILABLE:
            test_class.test_basic_filter_chain_with_glcm()
            test_class.test_glcm_optimization_flag()
            test_class.test_multiscale_filter_execution()
            print("✅ GLCM filter chain tests PASSED")
        else:
            print("⚠️  GLCM not available, skipping GLCM-specific tests")
            
        test_class.test_legacy_filter_chain_still_works()
        print("✅ Legacy filter chain tests PASSED")
    except Exception as e:
        print(f"❌ Filter chain execution tests FAILED: {e}")
    
    # Test 3: Parameter Building Logic
    print("\n3. Testing parameter building logic...")
    try:
        test_class = TestParameterBuildingLogic()
        test_class.test_glcm_parameters_built_correctly()
        test_class.test_multiscale_parameters_built_correctly()
        test_class.test_blob_removal_glcm_integration()
        print("✅ Parameter building tests PASSED")
    except Exception as e:
        print(f"❌ Parameter building tests FAILED: {e}")
    
    print("\n" + "=" * 60)
    print("🏁 HIGH Priority Integration Tests Complete!")