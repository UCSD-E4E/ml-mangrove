"""
Basic tests for the normalization pipeline framework.
"""

import numpy as np
from normalization_pipeline import (
    ImageNetNormalizer, 
    MinMaxNormalizer, 
    NormalizationPipeline,
    create_default_pipeline,
    normalize_for_training
)


def test_imagenet_normalizer():
    """Test ImageNet normalization."""
    normalizer = ImageNetNormalizer()
    
    # Create test data
    test_data = np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8)
    
    # Normalize
    normalized = normalizer.normalize(test_data)
    
    # Check normalization worked
    assert normalized.shape == test_data.shape
    assert normalized.dtype == np.float64
    
    # Test denormalization
    denormalized = normalizer.denormalize(normalized)
    assert denormalized.shape == test_data.shape
    assert denormalized.dtype == np.uint8


def test_minmax_normalizer():
    """Test min-max normalization."""
    normalizer = MinMaxNormalizer()
    
    # Create test data
    test_data = np.random.rand(3, 224, 224) * 100
    
    # Normalize
    normalized = normalizer.normalize(test_data)
    
    # Check normalization worked
    assert normalized.shape == test_data.shape
    assert np.allclose(normalized.min(), 0.0, atol=1e-6)
    assert np.allclose(normalized.max(), 1.0, atol=1e-6)


def test_pipeline():
    """Test the normalization pipeline."""
    pipeline = NormalizationPipeline()
    
    # Add normalizers
    pipeline.add_normalizer("rgb", ImageNetNormalizer())
    pipeline.add_normalizer("spectral", MinMaxNormalizer())
    
    # Test single source normalization
    test_data = np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8)
    normalized = pipeline.normalize_data("rgb", test_data)
    assert normalized.shape == test_data.shape
    
    # Test batch normalization
    batch_data = {
        "rgb": np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8),
        "spectral": np.random.rand(3, 224, 224) * 100
    }
    
    normalized_batch = pipeline.normalize_batch(batch_data)
    assert len(normalized_batch) == 2
    assert "rgb" in normalized_batch
    assert "spectral" in normalized_batch


def test_convenience_function():
    """Test the convenience function."""
    test_data = np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8)
    normalized = normalize_for_training(test_data)
    
    assert normalized.shape == test_data.shape
    assert normalized.dtype == np.float64


if __name__ == "__main__":
    # Run basic tests
    test_imagenet_normalizer()
    test_minmax_normalizer()
    test_pipeline()
    test_convenience_function()
    print("All tests passed!")
