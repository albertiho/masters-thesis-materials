"""Concrete CombinedDetector Variants.

Implements specific detector combinations optimized for different use cases.
These variants are informed by the analysis from analyze_detector_combinations.py.

Variants:
    - ProductionCombinedDetector: Optimal 3-layer (Sanity -> IForest -> Z-score) [PRODUCTION]
    - DefaultCombinedDetector: Full pipeline with Sanity -> Statistical -> ML
    - StatisticalOnlyCombinedDetector: Statistical methods only (no ML dependencies)
    - MinimalCombinedDetector: Z-score + Sanity only (lowest latency)

Usage:
    from src.anomaly.combined_variants import (
        ProductionCombinedDetector,
        DefaultCombinedDetector,
        StatisticalOnlyCombinedDetector,
    )
    
    # Create production detector (requires IsolationForest)
    detector = ProductionCombinedDetector.create(iforest_detector)
    
    # Create default with ML
    detector = DefaultCombinedDetector.create()
    
    # Or with custom thresholds
    config = CombinedDetectorConfig(
        name="custom",
        min_history_cold=3,
        min_history_warm=5,
    )
    detector = StatisticalOnlyCombinedDetector(config)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.anomaly.combined import (
    BaseCombinedDetector,
    CombinedDetectorConfig,
    DetectorLayer,
)
from src.anomaly.statistical import (
    IQRDetector,
    SanityCheckDetector,
    ThresholdDetector,
    ZScoreDetector,
)

if TYPE_CHECKING:
    from src.anomaly.ml.autoencoder import AutoencoderDetector
    from src.anomaly.ml.isolation_forest import IsolationForestDetector

logger = logging.getLogger(__name__)


# =============================================================================
# Default Configuration Values
# =============================================================================

# History thresholds (based on analysis)
DEFAULT_MIN_HISTORY_COLD = 3  # Products with < 3 obs get cold-start treatment
DEFAULT_MIN_HISTORY_WARM = 5  # Products with >= 5 obs get full statistical
DEFAULT_MIN_HISTORY_ML = 5    # ML models require at least 5 observations

# Persistence acceptance
DEFAULT_PERSISTENCE_RUNS = 2  # Accept anomaly after 2 consecutive runs


# =============================================================================
# Default Combined Detector
# =============================================================================


# =============================================================================
# Production Combined Detector (Winning Configuration)
# =============================================================================


class ProductionCombinedDetector(BaseCombinedDetector):
    """Production detector: Sanity (gate) -> Isolation Forest -> Z-score.
    
    This is the optimal 3-layer configuration identified through analysis in
    analyze_detector_combinations.py. It achieves the best F1 score by:
    
    Layer Architecture:
        1. Sanity (Gate): Business rule violations -> immediate flag, short-circuit
           - Zero history requirement (works on cold-start)
           - Catches impossible values (negative prices, 99% drops)
        
        2. Isolation Forest: ML-based anomaly detection
           - Zero history requirement (works on cold-start via feature imputation)
           - Catches complex multi-feature anomaly patterns
           - Per-competitor models trained on historical data
        
        3. Z-score: Statistical detection backup
           - Requires 3+ observations (skipped on cold-start)
           - Catches deviations from product's own price history
           - Provides additional recall for warm products
    
    This ordering ensures:
        - Sanity catches obvious errors first (cheap, fast)
        - IForest handles ML detection for all products
        - Z-score adds statistical backup for products with history
    
    Note:
        Requires a trained IsolationForestDetector. Unlike DefaultCombinedDetector,
        the ML detector is required, not optional.
    
    Attributes:
        iforest_detector: Required IsolationForestDetector instance.
    """
    
    def __init__(
        self,
        config: CombinedDetectorConfig,
        iforest_detector: "IsolationForestDetector",
    ) -> None:
        """Initialize with configuration and required Isolation Forest detector.
        
        Args:
            config: CombinedDetectorConfig with detection settings.
            iforest_detector: Trained IsolationForestDetector instance. Must be fitted.
        
        Raises:
            ValueError: If iforest_detector is None.
        """
        super().__init__(config)
        if iforest_detector is None:
            raise ValueError(
                "ProductionCombinedDetector requires an IsolationForestDetector. "
                "Use StatisticalOnlyCombinedDetector if no ML model is available."
            )
        self.iforest_detector = iforest_detector
    
    @classmethod
    def create(
        cls,
        iforest_detector: "IsolationForestDetector",
        name: str = "production",
        min_history_cold: int = DEFAULT_MIN_HISTORY_COLD,
        min_history_warm: int = DEFAULT_MIN_HISTORY_WARM,
    ) -> "ProductionCombinedDetector":
        """Factory method for production configuration.
        
        Args:
            iforest_detector: Trained IsolationForestDetector instance.
            name: Detector name (default: "production").
            min_history_cold: Minimum observations for cold-start treatment.
            min_history_warm: Minimum observations for full statistical.
        
        Returns:
            Configured ProductionCombinedDetector instance.
        
        Raises:
            ValueError: If iforest_detector is None.
        
        Example:
            # Load or create IsolationForest
            iforest = IsolationForestDetector.load("/path/to/local/model")
            
            # Create production detector
            detector = ProductionCombinedDetector.create(iforest)
            
            # Use for detection
            result = detector.detect(context)
            results = detector.detect_batch(contexts)
        """
        config = CombinedDetectorConfig(
            name=name,
            min_history_cold=min_history_cold,
            min_history_warm=min_history_warm,
            persistence_runs=DEFAULT_PERSISTENCE_RUNS,
            enable_short_circuit=True,
        )
        return cls(config, iforest_detector)
    
    def get_layers(self) -> list[DetectorLayer]:
        """Return production detection layers.
        
        Order rationale (from analysis):
            1. Sanity first - cheap gate, catches obvious errors, short-circuits
            2. IForest second - ML detection for all products (cold and warm)
            3. Z-score last - statistical backup for products with 3+ observations
        
        Returns:
            List of DetectorLayer instances.
        """
        return [
            # Layer 1: Sanity checks (Gate - short-circuits on anomaly)
            DetectorLayer(
                name="sanity",
                detectors=[SanityCheckDetector()],
                is_gate=True,
                required_history=0,  # Always runs (cold-start safe)
                layer_type="sanity",
            ),
            # Layer 2: Isolation Forest (ML layer - cold-start safe)
            DetectorLayer(
                name="iforest",
                detectors=[self.iforest_detector],
                is_gate=False,
                required_history=0,  # IForest handles cold-start via feature imputation
                layer_type="ml",
            ),
            # Layer 3: Z-score (Statistical backup for warm products)
            DetectorLayer(
                name="zscore",
                detectors=[ZScoreDetector()],
                is_gate=False,
                required_history=self.config.min_history_cold,  # Requires history
                layer_type="statistical",
            ),
        ]


# =============================================================================
# Default Combined Detector
# =============================================================================


class DefaultCombinedDetector(BaseCombinedDetector):
    """Full-featured combined detector with all detection layers.
    
    Layer Architecture (based on analysis findings):
        1. Sanity (Gate): Business rule violations -> immediate flag
        2. Statistical: Z-score, IQR, Threshold ensemble
        3. ML (if available): Autoencoder or Isolation Forest for subtle patterns
    
    The analysis showed:
        - Sanity has LOW overlap with statistical/ML (catches different errors)
        - Z-score and IQR have HIGH overlap (redundant but complementary recall)
        - ML adds ~3-5% recall over statistical (catches subtle patterns)
        - Cold-start products (obs < 3) benefit from sanity checks only
    
    Attributes:
        ml_detector: Optional ML detector (Autoencoder or IsolationForest).
    """
    
    def __init__(
        self,
        config: CombinedDetectorConfig,
        ml_detector: Any | None = None,
    ) -> None:
        """Initialize with configuration and optional ML detector.
        
        Args:
            config: CombinedDetectorConfig with detection settings.
            ml_detector: Optional ML detector instance. If None, ML layer is skipped.
        """
        super().__init__(config)
        self.ml_detector = ml_detector
    
    @classmethod
    def create(
        cls,
        name: str = "default",
        ml_detector: Any | None = None,
        min_history_cold: int = DEFAULT_MIN_HISTORY_COLD,
        min_history_warm: int = DEFAULT_MIN_HISTORY_WARM,
    ) -> DefaultCombinedDetector:
        """Factory method for common configuration.
        
        Args:
            name: Detector name.
            ml_detector: Optional ML detector instance.
            min_history_cold: Minimum observations for cold-start treatment.
            min_history_warm: Minimum observations for full statistical.
        
        Returns:
            Configured DefaultCombinedDetector instance.
        """
        config = CombinedDetectorConfig(
            name=name,
            min_history_cold=min_history_cold,
            min_history_warm=min_history_warm,
            persistence_runs=DEFAULT_PERSISTENCE_RUNS,
            enable_short_circuit=True,
        )
        return cls(config, ml_detector)
    
    def get_layers(self) -> list[DetectorLayer]:
        """Return detection layers in recommended order.
        
        Order rationale (from analysis):
        1. Sanity first - cheap, catches obvious errors, can short-circuit
        2. Statistical ensemble - core detection capability
        3. ML last - expensive, adds marginal recall
        
        Returns:
            List of DetectorLayer instances.
        """
        layers = []
        
        # Layer 1: Sanity checks (Gate layer - short-circuits on violation)
        layers.append(DetectorLayer(
            name="sanity",
            detectors=[SanityCheckDetector()],
            is_gate=True,
            required_history=0,  # Always runs
            layer_type="sanity",
        ))
        
        # Layer 2: Statistical ensemble
        # Z-score is primary, IQR adds recall, Threshold catches sudden changes
        layers.append(DetectorLayer(
            name="statistical",
            detectors=[
                ZScoreDetector(),
                IQRDetector(),
                ThresholdDetector(),
            ],
            is_gate=False,
            required_history=self.config.min_history_cold,
            layer_type="statistical",
        ))
        
        # Layer 3: ML detector (if available)
        if self.ml_detector is not None:
            layers.append(DetectorLayer(
                name="ml",
                detectors=[self.ml_detector],
                is_gate=False,
                required_history=self.config.min_history_warm,
                layer_type="ml",
            ))
        
        return layers


# =============================================================================
# Statistical-Only Combined Detector
# =============================================================================


class StatisticalOnlyCombinedDetector(BaseCombinedDetector):
    """Statistical-only detector (no ML dependencies).
    
    Useful when:
        - ML models aren't trained yet
        - Low-latency detection is required
        - Cold-start products with limited history
    
    Layer Architecture:
        1. Sanity (Gate): Business rule violations
        2. Statistical: Z-score, IQR, Threshold
    
    No ML layer - uses only statistical methods for detection.
    """
    
    @classmethod
    def create(
        cls,
        name: str = "statistical_only",
        min_history_cold: int = DEFAULT_MIN_HISTORY_COLD,
        min_history_warm: int = DEFAULT_MIN_HISTORY_WARM,
    ) -> StatisticalOnlyCombinedDetector:
        """Factory method for common configuration.
        
        Args:
            name: Detector name.
            min_history_cold: Minimum observations for cold-start treatment.
            min_history_warm: Minimum observations for full statistical.
        
        Returns:
            Configured StatisticalOnlyCombinedDetector instance.
        """
        config = CombinedDetectorConfig(
            name=name,
            min_history_cold=min_history_cold,
            min_history_warm=min_history_warm,
            persistence_runs=DEFAULT_PERSISTENCE_RUNS,
            enable_short_circuit=True,
        )
        return cls(config)
    
    def get_layers(self) -> list[DetectorLayer]:
        """Return statistical-only detection layers.
        
        Returns:
            List of DetectorLayer instances.
        """
        return [
            # Layer 1: Sanity checks (Gate)
            DetectorLayer(
                name="sanity",
                detectors=[SanityCheckDetector()],
                is_gate=True,
                required_history=0,
                layer_type="sanity",
            ),
            # Layer 2: Statistical ensemble
            DetectorLayer(
                name="statistical",
                detectors=[
                    ZScoreDetector(),
                    IQRDetector(),
                    ThresholdDetector(),
                ],
                is_gate=False,
                required_history=self.config.min_history_cold,
                layer_type="statistical",
            ),
        ]


# =============================================================================
# Minimal Combined Detector
# =============================================================================


class MinimalCombinedDetector(BaseCombinedDetector):
    """Minimal detector for lowest latency.
    
    Uses only:
        - Sanity checks (always)
        - Z-score (primary statistical method)
    
    Best for:
        - High-throughput scenarios
        - Real-time detection with strict latency requirements
        - Initial triage before more expensive detection
    """
    
    @classmethod
    def create(
        cls,
        name: str = "minimal",
        min_history_cold: int = DEFAULT_MIN_HISTORY_COLD,
    ) -> MinimalCombinedDetector:
        """Factory method for common configuration.
        
        Args:
            name: Detector name.
            min_history_cold: Minimum observations for Z-score detection.
        
        Returns:
            Configured MinimalCombinedDetector instance.
        """
        config = CombinedDetectorConfig(
            name=name,
            min_history_cold=min_history_cold,
            min_history_warm=min_history_cold,  # Same threshold for minimal
            persistence_runs=DEFAULT_PERSISTENCE_RUNS,
            enable_short_circuit=True,
        )
        return cls(config)
    
    def get_layers(self) -> list[DetectorLayer]:
        """Return minimal detection layers.
        
        Returns:
            List of DetectorLayer instances (Sanity + Z-score only).
        """
        return [
            # Layer 1: Sanity checks (Gate)
            DetectorLayer(
                name="sanity",
                detectors=[SanityCheckDetector()],
                is_gate=True,
                required_history=0,
                layer_type="sanity",
            ),
            # Layer 2: Z-score only (fastest statistical method)
            DetectorLayer(
                name="zscore",
                detectors=[ZScoreDetector()],
                is_gate=False,
                required_history=self.config.min_history_cold,
                layer_type="statistical",
            ),
        ]


# =============================================================================
# Factory Function
# =============================================================================


def create_combined_detector(
    variant: str = "default",
    ml_detector: Any | None = None,
    iforest_detector: "IsolationForestDetector | None" = None,
    **kwargs: Any,
) -> BaseCombinedDetector:
    """Factory function to create combined detector by variant name.
    
    Args:
        variant: Variant name - "production", "default", "statistical_only", or "minimal".
        ml_detector: Optional ML detector for "default" variant.
        iforest_detector: Required IsolationForestDetector for "production" variant.
        **kwargs: Additional arguments passed to create() method.
    
    Returns:
        Configured CombinedDetector instance.
    
    Raises:
        ValueError: If variant is unknown or required detector is missing.
    
    Example:
        # Create production with IsolationForest (recommended)
        detector = create_combined_detector("production", iforest_detector=iforest)
        
        # Create default with ML
        detector = create_combined_detector("default", ml_detector=autoencoder)
        
        # Create statistical-only
        detector = create_combined_detector("statistical_only")
        
        # Create minimal with custom threshold
        detector = create_combined_detector("minimal", min_history_cold=2)
    """
    variant_lower = variant.lower()
    
    if variant_lower == "production":
        if iforest_detector is None:
            raise ValueError(
                "ProductionCombinedDetector requires iforest_detector argument. "
                "Use variant='statistical_only' if no ML model is available."
            )
        return ProductionCombinedDetector.create(iforest_detector=iforest_detector, **kwargs)
    elif variant_lower == "default":
        return DefaultCombinedDetector.create(ml_detector=ml_detector, **kwargs)
    elif variant_lower in ("statistical_only", "statistical"):
        return StatisticalOnlyCombinedDetector.create(**kwargs)
    elif variant_lower == "minimal":
        return MinimalCombinedDetector.create(**kwargs)
    else:
        raise ValueError(
            f"Unknown variant '{variant}'. "
            f"Choose from: production, default, statistical_only, minimal"
        )
