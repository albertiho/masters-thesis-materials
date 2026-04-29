"""Base Feature Extractor - Common interface for all feature extractors.

This module provides a common interface and data structures for feature extraction,
avoiding code duplication between different feature extractors (price, scrape, etc.).

Usage:
    from src.features.base import FeatureExtractor, FeatureVector
    
    class PriceFeatureExtractor(FeatureExtractor):
        @property
        def feature_names(self) -> list[str]:
            return ["price", "price_change_pct", ...]
        
        def extract(self, **kwargs) -> FeatureVector:
            # Extract features
            return FeatureVector(...)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FeatureVector:
    """Output from a feature extractor.
    
    Attributes:
        features: Dictionary mapping feature names to values.
        product_id: Product identifier (FK to products table).
        competitor_id: Competitor identifier.
        raw_record_id: FK to raw_scrape_records for lineage.
        scrape_run_id: FK to scrape_runs.
        is_valid: Whether the feature vector is valid for model input.
        missing_features: List of feature names that couldn't be computed.
        metadata: Additional metadata about the extraction.
    """
    
    features: dict[str, float | None]
    product_id: int
    competitor_id: str
    raw_record_id: int | None = None
    scrape_run_id: str | None = None
    is_valid: bool = True
    missing_features: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def feature_names(self) -> list[str]:
        """Get list of feature names."""
        return list(self.features.keys())
    
    def to_array(self, fill_missing: float = 0.0) -> np.ndarray:
        """Convert features to numpy array.
        
        Args:
            fill_missing: Value to use for missing/None features.
            
        Returns:
            Numpy array of feature values.
        """
        values = []
        for name in sorted(self.features.keys()):
            val = self.features[name]
            if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                values.append(fill_missing)
            else:
                values.append(float(val))
        return np.array(values, dtype=np.float64)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "features": self.features,
            "product_id": self.product_id,
            "competitor_id": self.competitor_id,
            "raw_record_id": self.raw_record_id,
            "scrape_run_id": self.scrape_run_id,
            "is_valid": self.is_valid,
            "missing_features": self.missing_features,
            "metadata": self.metadata,
        }


class FeatureExtractor(ABC):
    """Abstract base class for feature extractors.
    
    All feature extractors (price, scrape, etc.) should inherit from this
    to ensure a consistent interface.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this feature extractor."""
        pass
    
    @property
    @abstractmethod
    def layer(self) -> str:
        """Which detection layer this extractor is for (scrape_bug, price_anomaly)."""
        pass
    
    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """List of feature names this extractor produces."""
        pass
    
    @abstractmethod
    def extract(
        self,
        product_id: int,
        competitor_id: str,
        raw_record_id: int | None = None,
        scrape_run_id: str | None = None,
        **kwargs: Any,
    ) -> FeatureVector:
        """Extract features for a single product.
        
        Args:
            product_id: Product identifier.
            competitor_id: Competitor identifier.
            raw_record_id: Optional FK to raw_scrape_records.
            scrape_run_id: Optional FK to scrape_runs.
            **kwargs: Additional data needed for extraction.
            
        Returns:
            FeatureVector with extracted features.
        """
        pass
    
    def extract_batch(
        self,
        products: list[dict[str, Any]],
    ) -> list[FeatureVector]:
        """Extract features for multiple products.
        
        Default implementation calls extract() for each product.
        Subclasses can override for batch optimization.
        
        Args:
            products: List of dicts with product_id, competitor_id, and other kwargs.
            
        Returns:
            List of FeatureVector objects.
        """
        return [self.extract(**p) for p in products]
    
    def get_info(self) -> dict[str, Any]:
        """Get information about this extractor."""
        return {
            "name": self.name,
            "layer": self.layer,
            "feature_names": self.feature_names,
            "feature_count": len(self.feature_names),
        }


def safe_divide(numerator: float | None, denominator: float | None, default: float = 0.0) -> float:
    """Safely divide two numbers, handling None and zero.
    
    Args:
        numerator: The numerator.
        denominator: The denominator.
        default: Value to return if division is not possible.
        
    Returns:
        The division result or default.
    """
    if numerator is None or denominator is None:
        return default
    if denominator == 0:
        return default
    return numerator / denominator


def safe_pct_change(current: float | None, previous: float | None, default: float = 0.0) -> float:
    """Calculate percentage change safely.
    
    Args:
        current: Current value.
        previous: Previous value.
        default: Value to return if calculation not possible.
        
    Returns:
        Percentage change as decimal (0.1 = 10% increase).
    """
    if current is None or previous is None:
        return default
    if previous == 0:
        return default
    return (current - previous) / previous


def safe_ratio(value: float | None, baseline: float | None, default: float = 1.0) -> float:
    """Calculate ratio safely.
    
    Args:
        value: The value.
        baseline: The baseline to compare against.
        default: Value to return if calculation not possible.
        
    Returns:
        Ratio of value to baseline.
    """
    if value is None or baseline is None:
        return default
    if baseline == 0:
        return default
    return value / baseline
