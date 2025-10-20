"""
Basic Normalization Pipeline for Human Infrastructure Project

This module provides a foundational framework for normalizing different
data sources for human vs natural feature classification.
"""

import numpy as np
from typing import Dict, List, Optional, Union
from abc import ABC, abstractmethod


class DataNormalizer(ABC):
    """Abstract base class for data normalization strategies."""
    
    @abstractmethod
    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize input data."""
        pass
    
    @abstractmethod
    def denormalize(self, normalized_data: np.ndarray) -> np.ndarray:
        """Reverse normalization."""
        pass


class ImageNetNormalizer(DataNormalizer):
    """Standard ImageNet normalization for RGB images."""
    
    def __init__(self):
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
    
    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Apply ImageNet normalization."""
        if data.max() > 1.0:
            data = data / 255.0
        
        # Reshape mean and std for broadcasting
        mean = self.mean.reshape(3, 1, 1)
        std = self.std.reshape(3, 1, 1)
        
        normalized = (data - mean) / std
        return normalized
    
    def denormalize(self, normalized_data: np.ndarray) -> np.ndarray:
        """Reverse ImageNet normalization."""
        # Reshape mean and std for broadcasting
        mean = self.mean.reshape(3, 1, 1)
        std = self.std.reshape(3, 1, 1)
        
        denormalized = normalized_data * std + mean
        return np.clip(denormalized * 255.0, 0, 255).astype(np.uint8)


class MinMaxNormalizer(DataNormalizer):
    """Min-max normalization to [0, 1] range."""
    
    def __init__(self, min_val: Optional[float] = None, max_val: Optional[float] = None):
        self.min_val = min_val
        self.max_val = max_val
    
    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Apply min-max normalization."""
        if self.min_val is None:
            self.min_val = data.min()
        if self.max_val is None:
            self.max_val = data.max()
        
        normalized = (data - self.min_val) / (self.max_val - self.min_val)
        return normalized
    
    def denormalize(self, normalized_data: np.ndarray) -> np.ndarray:
        """Reverse min-max normalization."""
        return normalized_data * (self.max_val - self.min_val) + self.min_val


class NormalizationPipeline:
    """Main pipeline for handling multiple data source normalization."""
    
    def __init__(self):
        self.normalizers: Dict[str, DataNormalizer] = {}
        self.data_sources: List[str] = []
    
    def add_normalizer(self, source_name: str, normalizer: DataNormalizer):
        """Add a normalizer for a specific data source."""
        self.normalizers[source_name] = normalizer
        self.data_sources.append(source_name)
    
    def normalize_data(self, source_name: str, data: np.ndarray) -> np.ndarray:
        """Normalize data from a specific source."""
        if source_name not in self.normalizers:
            raise ValueError(f"No normalizer found for source: {source_name}")
        
        return self.normalizers[source_name].normalize(data)
    
    def normalize_batch(self, data_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Normalize a batch of data from multiple sources."""
        normalized_batch = {}
        
        for source_name, data in data_dict.items():
            normalized_batch[source_name] = self.normalize_data(source_name, data)
        
        return normalized_batch
    
    def get_available_sources(self) -> List[str]:
        """Get list of available data sources."""
        return self.data_sources.copy()


# Example usage and factory functions
def create_default_pipeline() -> NormalizationPipeline:
    """Create a pipeline with default normalizers."""
    pipeline = NormalizationPipeline()
    
    # Add ImageNet normalizer for RGB images
    pipeline.add_normalizer("rgb_images", ImageNetNormalizer())
    
    # Add min-max normalizer for other data types
    pipeline.add_normalizer("multispectral", MinMaxNormalizer())
    
    return pipeline


def normalize_for_training(data: np.ndarray, source_type: str = "rgb_images") -> np.ndarray:
    """Convenience function for training data normalization."""
    pipeline = create_default_pipeline()
    return pipeline.normalize_data(source_type, data)
