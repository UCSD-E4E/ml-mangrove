"""
Basic tests for the simplified Normalizer API.
"""

import numpy as np
from normalization_pipeline import Normalizer


def test_imagenet_normalizer():
    test_data = np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8)
    normalized = Normalizer.normalize_imagenet(test_data)
    assert normalized.shape == test_data.shape
    assert normalized.dtype == np.float64
    denormalized = Normalizer.denormalize_imagenet(normalized)
    assert denormalized.shape == test_data.shape
    assert denormalized.dtype == np.uint8


def test_minmax_normalizer():
    test_data = np.random.rand(3, 224, 224) * 100
    normalized, lo, hi = Normalizer.normalize_minmax(test_data)
    assert normalized.shape == test_data.shape
    assert np.allclose(normalized.min(), 0.0, atol=1e-6)
    assert np.allclose(normalized.max(), 1.0, atol=1e-6)
    restored = Normalizer.denormalize_minmax(normalized, lo, hi)
    assert restored.shape == test_data.shape


def test_idempotence_and_shapes():
    test_data = np.random.randint(0, 255, (3, 128, 128), dtype=np.uint8)
    norm = Normalizer.normalize_imagenet(test_data)
    denorm = Normalizer.denormalize_imagenet(norm)
    assert denorm.shape == test_data.shape


def test_dtype_behavior():
    test_data = np.random.rand(3, 64, 64).astype(np.float32)
    out = Normalizer.normalize_imagenet(test_data)
    assert out.dtype == np.float64


if __name__ == "__main__":
    test_imagenet_normalizer()
    test_minmax_normalizer()
    test_idempotence_and_shapes()
    test_dtype_behavior()
    print("All tests passed!")
