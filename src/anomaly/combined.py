"""Combined Detector Framework - Composable anomaly detection pipelines.

Provides a flexible framework for combining multiple anomaly detectors
in configurable pipelines with history-based routing.

Key Components:
- BaseCombinedDetector: Abstract base class for detector pipelines
- CombinedDetector: Concrete class for ad-hoc layer composition
- DetectorLayer: Pluggable detection layer with short-circuit support
- DetectionContext: Shared state passed through detection layers
- CombinedDetectorConfig: Configuration for detector variants

Architecture:
    1. History router determines path (cold start vs warm vs full)
    2. Layers execute in order, each producing votes
    3. Gate layers can short-circuit on failure
    4. ConfidenceAggregator combines all votes

Usage:
    # Create a combined detector variant
    detector = DefaultCombinedDetector(config)
    
    # Create detection context
    context = DetectionContext.from_features(
        numeric_features=numeric_feats,
        temporal_features=temporal_feats,
        price_history=prices,
        observation_count=10,
    )
    
    # Run detection pipeline
    result = detector.detect(context)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from src.anomaly.change_tracker import PersistenceInfo
from src.anomaly.confidence import DetectorVote
from src.anomaly.statistical import AnomalyResult, AnomalySeverity, AnomalyType
from src.features.numeric import NumericFeatures
from src.features.temporal import TemporalFeatures

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class CombinedDetectorConfig:
    """Configuration for a CombinedDetector variant.

    Attributes:
        name: Human-readable name for this detector variant.
        min_history_cold: Below this observation count = cold start path.
        min_history_warm: Below this = limited statistical detection.
        persistence_runs: Consecutive runs required for persistence acceptance.
        enable_short_circuit: If True, gate failures skip remaining layers.
        detector_weights: Optional weights for weighted voting by detector name.
        layer_order: Optional explicit layer order override.
    """

    name: str
    min_history_cold: int = 3
    min_history_warm: int = 5
    persistence_runs: int = 2
    enable_short_circuit: bool = True
    detector_weights: dict[str, float] = field(default_factory=dict)
    layer_order: list[str] | None = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.min_history_cold < 1:
            raise ValueError(f"min_history_cold must be >= 1, got {self.min_history_cold}")
        if self.min_history_warm < self.min_history_cold:
            raise ValueError(
                f"min_history_warm ({self.min_history_warm}) must be >= "
                f"min_history_cold ({self.min_history_cold})"
            )
        if self.persistence_runs < 1:
            raise ValueError(f"persistence_runs must be >= 1, got {self.persistence_runs}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "name": self.name,
            "min_history_cold": self.min_history_cold,
            "min_history_warm": self.min_history_warm,
            "persistence_runs": self.persistence_runs,
            "enable_short_circuit": self.enable_short_circuit,
            "detector_weights": self.detector_weights,
            "layer_order": self.layer_order,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CombinedDetectorConfig:
        """Create from dictionary."""
        return cls(
            name=data.get("name", "unnamed"),
            min_history_cold=data.get("min_history_cold", 3),
            min_history_warm=data.get("min_history_warm", 5),
            persistence_runs=data.get("persistence_runs", 2),
            enable_short_circuit=data.get("enable_short_circuit", True),
            detector_weights=data.get("detector_weights", {}),
            layer_order=data.get("layer_order"),
        )


# =============================================================================
# Detection Context
# =============================================================================


@dataclass
class DetectionContext:
    """Shared context passed through detection layers.

    Carries all input features plus accumulated state from prior layers.
    Immutable input features + mutable detection state.

    Attributes:
        numeric_features: Price and numeric features for the record.
        temporal_features: Rolling statistics and temporal features.
        price_history: Optional list of historical prices for history-dependent detectors.
        observation_count: Number of price observations for this product.
        persistence_info: Optional persistence tracking info.
        votes: Accumulated detector votes from all layers.
        short_circuited: Whether pipeline was short-circuited.
        short_circuit_reason: Reason for short-circuit (if applicable).
        short_circuit_result: The result that caused short-circuit.
        layer_results: Results from each completed layer.
    """

    # Input features (immutable)
    numeric_features: NumericFeatures
    temporal_features: TemporalFeatures
    price_history: list[float] | None = None
    observation_count: int = 0
    persistence_info: PersistenceInfo | None = None

    # Detection state (mutable)
    votes: list[DetectorVote] = field(default_factory=list)
    short_circuited: bool = False
    short_circuit_reason: str | None = None
    short_circuit_result: AnomalyResult | None = None
    layer_results: dict[str, list[AnomalyResult]] = field(default_factory=dict)

    # Routing info
    route_path: str = "unknown"

    @classmethod
    def from_features(
        cls,
        numeric_features: NumericFeatures,
        temporal_features: TemporalFeatures,
        price_history: list[float] | None = None,
        observation_count: int | None = None,
        persistence_info: PersistenceInfo | None = None,
    ) -> DetectionContext:
        """Create context from extracted features.

        Factory method for convenient context creation.

        Args:
            numeric_features: Extracted numeric features.
            temporal_features: Extracted temporal features.
            price_history: Optional price history for history-dependent detectors.
            observation_count: Override observation count (defaults to temporal's).
            persistence_info: Optional persistence tracking info.

        Returns:
            DetectionContext ready for pipeline execution.
        """
        obs_count = (
            observation_count
            if observation_count is not None
            else temporal_features.observation_count
        )

        return cls(
            numeric_features=numeric_features,
            temporal_features=temporal_features,
            price_history=price_history,
            observation_count=obs_count,
            persistence_info=persistence_info,
        )

    def add_vote(
        self,
        detector_name: str,
        layer: str,
        is_flagged: bool,
        score: float | None = None,
    ) -> None:
        """Add a detector vote to the context.

        Args:
            detector_name: Name of the detector.
            layer: Detection layer name.
            is_flagged: Whether detector flagged as anomaly.
            score: Optional detector score (0-1).
        """
        self.votes.append(
            DetectorVote(
                detector_name=detector_name,
                layer=layer,
                is_flagged=is_flagged,
                score=score,
            )
        )

    def add_layer_result(self, layer_name: str, result: AnomalyResult) -> None:
        """Add a result from a specific layer.

        Args:
            layer_name: Name of the layer that produced the result.
            result: AnomalyResult from the layer.
        """
        if layer_name not in self.layer_results:
            self.layer_results[layer_name] = []
        self.layer_results[layer_name].append(result)

    def trigger_short_circuit(self, reason: str, result: AnomalyResult) -> None:
        """Mark the context as short-circuited.

        Args:
            reason: Human-readable reason for short-circuit.
            result: The result that triggered short-circuit.
        """
        self.short_circuited = True
        self.short_circuit_reason = reason
        self.short_circuit_result = result

        logger.debug(
            "detection_short_circuited",
            extra={
                "reason": reason,
                "detector": result.detector,
                "competitor_product_id": result.competitor_product_id,
                "competitor": result.competitor,
            },
        )

    def get_all_results(self) -> list[AnomalyResult]:
        """Get all results from all layers.

        Returns:
            Flat list of all AnomalyResults.
        """
        results = []
        for layer_results in self.layer_results.values():
            results.extend(layer_results)
        return results

    def get_flagged_detectors(self) -> list[str]:
        """Get names of detectors that flagged anomalies.

        Returns:
            List of detector names that voted is_flagged=True.
        """
        return [v.detector_name for v in self.votes if v.is_flagged]

    @property
    def confidence(self) -> float:
        """Compute simple confidence as flagged/total votes.

        Returns:
            Confidence score (0-1), 0 if no votes.
        """
        if not self.votes:
            return 0.0
        flagged = sum(1 for v in self.votes if v.is_flagged)
        return flagged / len(self.votes)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            "competitor_product_id": self.numeric_features.competitor_product_id,
            "competitor": self.numeric_features.competitor,
            "observation_count": self.observation_count,
            "route_path": self.route_path,
            "total_votes": len(self.votes),
            "flagged_votes": sum(1 for v in self.votes if v.is_flagged),
            "confidence": self.confidence,
            "short_circuited": self.short_circuited,
            "short_circuit_reason": self.short_circuit_reason,
            "layers_completed": list(self.layer_results.keys()),
            "flagged_detectors": self.get_flagged_detectors(),
        }


# =============================================================================
# Detector Protocol
# =============================================================================


@runtime_checkable
class Detector(Protocol):
    """Protocol for detectors that can be used in DetectorLayer.

    Any detector implementing this protocol can be added to a layer.
    """

    name: str

    def detect(self, *args: Any, **kwargs: Any) -> AnomalyResult:
        """Run detection and return result."""
        ...


# =============================================================================
# Detector Layer
# =============================================================================


@dataclass
class DetectorLayer:
    """A single detection layer in the pipeline.

    Layers group related detectors that run together. Gate layers can
    short-circuit the pipeline on failure.

    Attributes:
        name: Human-readable layer name.
        detectors: List of detector instances to run.
        is_gate: If True, anomaly detection short-circuits remaining layers.
        required_history: Minimum observations needed to run this layer.
        maximum_history: Maximum observations allowed to run this layer (None = no limit).
        layer_type: Category of detection (sanity, statistical, ml, persistence).
    """

    name: str
    detectors: list[Any]  # Detector instances
    is_gate: bool = False
    required_history: int = 0
    maximum_history: int | None = None
    layer_type: str = "detection"

    def can_run(self, context: DetectionContext) -> bool:
        """Check if this layer can run given the context.

        Args:
            context: Current detection context.

        Returns:
            True if layer requirements are met.
        """
        # Check minimum history requirement
        if context.observation_count < self.required_history:
            logger.debug(
                "layer_skipped_insufficient_history",
                extra={
                    "layer": self.name,
                    "required": self.required_history,
                    "available": context.observation_count,
                    "competitor_product_id": context.numeric_features.competitor_product_id,
                },
            )
            return False

        # Check maximum history requirement
        if self.maximum_history is not None and context.observation_count > self.maximum_history:
            logger.debug(
                "layer_skipped_excessive_history",
                extra={
                    "layer": self.name,
                    "maximum": self.maximum_history,
                    "available": context.observation_count,
                    "competitor_product_id": context.numeric_features.competitor_product_id,
                },
            )
            return False

        return True

    def detect(self, context: DetectionContext) -> list[AnomalyResult]:
        """Run all detectors in this layer.

        Runs each detector, collects results, and adds votes to context.
        If this is a gate layer and any detector flags an anomaly,
        marks context for short-circuit.

        Args:
            context: Detection context with features and accumulated state.

        Returns:
            List of AnomalyResults from all detectors in this layer.
        """
        if not self.can_run(context):
            return []

        results: list[AnomalyResult] = []
        gate_triggered = False
        gate_result: AnomalyResult | None = None

        for detector in self.detectors:
            try:
                result = self._run_detector(detector, context)
                results.append(result)

                # Add vote to context
                context.add_vote(
                    detector_name=result.detector,
                    layer=self.name,
                    is_flagged=result.is_anomaly,
                    score=result.anomaly_score,
                )

                # Track gate triggers
                if self.is_gate and result.is_anomaly and not gate_triggered:
                    gate_triggered = True
                    gate_result = result

            except Exception as e:
                logger.warning(
                    "detector_failed",
                    extra={
                        "layer": self.name,
                        "detector": getattr(detector, "name", str(detector)),
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "competitor_product_id": context.numeric_features.competitor_product_id,
                    },
                )

        # Store results in context
        context.layer_results[self.name] = results

        # Trigger short-circuit if gate layer detected anomaly
        if gate_triggered and gate_result is not None:
            context.trigger_short_circuit(
                reason=f"Gate layer '{self.name}' detected anomaly",
                result=gate_result,
            )

        return results

    def supports_batch(self) -> bool:
        """Check if any detector in this layer supports batch processing.

        Returns:
            True if at least one detector has a detect_batch() method.
        """
        for detector in self.detectors:
            if hasattr(detector, "detect_batch"):
                return True
        return False

    def detect_batch(self, contexts: list[DetectionContext]) -> None:
        """Run all detectors on a batch of contexts.

        Modifies contexts in place by adding votes and layer_results.
        For detectors with detect_batch(), extracts features and calls it.
        For others, falls back to sequential detect().

        Gate layers trigger short-circuit after the entire batch is processed
        for this layer.

        Args:
            contexts: List of DetectionContext instances to process.
        """
        # Filter to contexts that meet layer requirements
        runnable = [ctx for ctx in contexts if self.can_run(ctx)]
        if not runnable:
            return

        for detector in self.detectors:
            detector_name = getattr(detector, "name", str(type(detector).__name__))

            if hasattr(detector, "detect_batch"):
                # Use batch detection for ML detectors
                try:
                    numeric_list = [ctx.numeric_features for ctx in runnable]
                    temporal_list = [ctx.temporal_features for ctx in runnable]
                    results = detector.detect_batch(
                        numeric_features_list=numeric_list,
                        temporal_features_list=temporal_list,
                    )

                    # Store results in contexts
                    for ctx, result in zip(runnable, results, strict=True):
                        ctx.add_vote(
                            detector_name=result.detector,
                            layer=self.name,
                            is_flagged=result.is_anomaly,
                            score=result.anomaly_score,
                        )
                        ctx.add_layer_result(self.name, result)

                except Exception as e:
                    logger.warning(
                        "detector_batch_failed",
                        extra={
                            "layer": self.name,
                            "detector": detector_name,
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "batch_size": len(runnable),
                        },
                    )
            else:
                # Fall back to sequential detection for non-batch detectors
                for ctx in runnable:
                    try:
                        result = self._run_detector(detector, ctx)
                        ctx.add_vote(
                            detector_name=result.detector,
                            layer=self.name,
                            is_flagged=result.is_anomaly,
                            score=result.anomaly_score,
                        )
                        ctx.add_layer_result(self.name, result)
                    except Exception as e:
                        logger.warning(
                            "detector_failed",
                            extra={
                                "layer": self.name,
                                "detector": detector_name,
                                "error": str(e),
                                "error_type": type(e).__name__,
                                "competitor_product_id": ctx.numeric_features.competitor_product_id,
                            },
                        )

        # Handle gate short-circuit after batch
        if self.is_gate:
            for ctx in runnable:
                layer_results = ctx.layer_results.get(self.name, [])
                for result in layer_results:
                    if result.is_anomaly:
                        ctx.trigger_short_circuit(
                            reason=f"Gate layer '{self.name}' detected anomaly",
                            result=result,
                        )
                        break  # Only need one anomaly to trigger short-circuit

    def _run_detector(
        self,
        detector: Any,
        context: DetectionContext,
    ) -> AnomalyResult:
        """Run a single detector with appropriate arguments.

        Handles different detector signatures by inspecting the detector
        and passing appropriate arguments.

        Args:
            detector: Detector instance.
            context: Detection context.

        Returns:
            AnomalyResult from the detector.
        """
        detector_name = getattr(detector, "name", str(type(detector).__name__))

        # SanityCheckDetector only needs numeric features
        if detector_name == "sanity":
            return detector.detect(context.numeric_features)

        # InvariantDetector needs InvariantContext (not handled here)
        if detector_name == "invariant":
            # InvariantDetector requires InvariantContext, which isn't in DetectionContext
            # This would need to be set up separately or passed via context
            raise NotImplementedError(
                "InvariantDetector requires InvariantContext - use a custom layer"
            )

        # History-dependent statistical detectors need raw prior price history.
        if getattr(detector, "requires_price_history", False):
            return detector.detect(
                context.numeric_features,
                context.temporal_features,
                context.price_history,
            )

        # ZScore and Threshold detectors need numeric + temporal
        if detector_name in ("zscore", "threshold"):
            return detector.detect(context.numeric_features, context.temporal_features)

        # ML detectors (autoencoder, iforest) have different signatures
        # They typically take feature arrays - this is a simplified fallback
        if hasattr(detector, "detect"):
            # Try common signatures
            try:
                return detector.detect(context.numeric_features, context.temporal_features)
            except TypeError:
                pass

            try:
                return detector.detect(context.numeric_features)
            except TypeError:
                pass

        raise ValueError(f"Unknown detector type: {detector_name}")


# =============================================================================
# Base Combined Detector
# =============================================================================


class BaseCombinedDetector(ABC):
    """Abstract base class for combined anomaly detectors.

    Subclasses define detector compositions by implementing get_layers().
    The base class handles routing, execution, and result aggregation.

    Design Principles:
    1. Detector instances are reusable - create once, detect many times
    2. Layers are stateless - all state lives in DetectionContext
    3. Short-circuit on gate failure - sanity failures skip expensive ML
    4. History routing is configurable - different thresholds per variant
    5. Weights are per-variant - each variant can weight detectors differently
    """

    def __init__(self, config: CombinedDetectorConfig) -> None:
        """Initialize with configuration.

        Args:
            config: CombinedDetectorConfig with variant settings.
        """
        self.config = config
        self._layers: list[DetectorLayer] | None = None

    @property
    def name(self) -> str:
        """Get detector name from config."""
        return self.config.name

    @property
    def layers(self) -> list[DetectorLayer]:
        """Get or create layers (lazy initialization)."""
        if self._layers is None:
            self._layers = self.get_layers()
        return self._layers

    @abstractmethod
    def get_layers(self) -> list[DetectorLayer]:
        """Return ordered list of detection layers.

        Subclasses implement this to define their detector composition.
        Layers execute in order; gate layers can short-circuit.

        Returns:
            List of DetectorLayer instances in execution order.
        """
        pass

    def route_by_history(self, observation_count: int) -> str:
        """Determine detection path based on observation count.

        Routes products to appropriate detection strategies based on
        how much historical data is available.

        Args:
            observation_count: Number of price observations for the product.

        Returns:
            Path name: "cold_start", "limited", or "full".
        """
        if observation_count < self.config.min_history_cold:
            return "cold_start"
        elif observation_count < self.config.min_history_warm:
            return "limited"
        else:
            return "full"

    def detect(self, context: DetectionContext) -> AnomalyResult:
        """Run the full detection pipeline.

        1. Determines route based on observation count
        2. Executes layers in order
        3. Short-circuits on gate failures (if enabled)
        4. Aggregates results into final AnomalyResult

        Args:
            context: DetectionContext with features and initial state.

        Returns:
            Aggregated AnomalyResult from all detectors.
        """
        # Determine routing path
        context.route_path = self.route_by_history(context.observation_count)

        logger.debug(
            "combined_detection_start",
            extra={
                "detector": self.name,
                "route_path": context.route_path,
                "observation_count": context.observation_count,
                "competitor_product_id": context.numeric_features.competitor_product_id,
                "competitor": context.numeric_features.competitor,
            },
        )

        # Execute layers
        for layer in self.layers:
            if context.short_circuited and self.config.enable_short_circuit:
                logger.debug(
                    "layer_skipped_short_circuit",
                    extra={
                        "layer": layer.name,
                        "reason": context.short_circuit_reason,
                        "competitor_product_id": context.numeric_features.competitor_product_id,
                    },
                )
                continue

            layer.detect(context)

        # Aggregate results
        result = self._aggregate_results(context)

        logger.debug(
            "combined_detection_complete",
            extra={
                "detector": self.name,
                "is_anomaly": result.is_anomaly,
                "anomaly_score": result.anomaly_score,
                "confidence": context.confidence,
                "route_path": context.route_path,
                "short_circuited": context.short_circuited,
                "flagged_detectors": context.get_flagged_detectors(),
                "competitor_product_id": result.competitor_product_id,
                "competitor": result.competitor,
            },
        )

        return result

    def detect_batch(self, contexts: list[DetectionContext]) -> list[AnomalyResult]:
        """Run detection pipeline on a batch of contexts.

        Layers execute sequentially, but within each layer uses batch detection
        if available. This provides ~10-50x speedup for ML-heavy pipelines.

        Short-circuit handling: Gate layers apply short-circuit after the entire
        batch completes for that layer. Subsequent layers skip short-circuited
        contexts.

        Args:
            contexts: List of DetectionContext instances with features populated.

        Returns:
            List of aggregated AnomalyResults, one per context, in same order.
        """
        if not contexts:
            return []

        # Route all contexts based on observation count
        for ctx in contexts:
            ctx.route_path = self.route_by_history(ctx.observation_count)

        logger.debug(
            "combined_batch_detection_start",
            extra={
                "detector": self.name,
                "batch_size": len(contexts),
                "route_paths": {ctx.route_path for ctx in contexts},
            },
        )

        # Execute layers sequentially
        for layer in self.layers:
            # Filter to non-short-circuited contexts
            if self.config.enable_short_circuit:
                active = [ctx for ctx in contexts if not ctx.short_circuited]
            else:
                active = contexts

            if not active:
                logger.debug(
                    "batch_all_short_circuited",
                    extra={
                        "detector": self.name,
                        "layer": layer.name,
                    },
                )
                break

            # Use batch detection (handles fallback internally)
            layer.detect_batch(active)

        # Aggregate results for each context
        results = [self._aggregate_results(ctx) for ctx in contexts]

        # Log summary
        anomaly_count = sum(1 for r in results if r.is_anomaly)
        logger.info(
            "combined_batch_detection_complete",
            extra={
                "detector": self.name,
                "batch_size": len(contexts),
                "anomalies_detected": anomaly_count,
                "short_circuited_count": sum(1 for ctx in contexts if ctx.short_circuited),
            },
        )

        return results

    def _aggregate_results(self, context: DetectionContext) -> AnomalyResult:
        """Aggregate all layer results into a final result.

        Combines anomaly types, computes max score, and determines severity.

        Args:
            context: DetectionContext with completed layer results.

        Returns:
            Aggregated AnomalyResult.
        """
        all_results = context.get_all_results()

        if not all_results:
            # No results - return clean result
            return AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                anomaly_types=[],
                severity=None,
                details={"no_results": True, "route_path": context.route_path},
                detector=self.name,
                competitor_product_id=context.numeric_features.competitor_product_id,
                competitor=context.numeric_features.competitor,
            )

        # Collect unique anomaly types
        all_types: set[AnomalyType] = set()
        for result in all_results:
            all_types.update(result.anomaly_types)

        # Aggregate score (max of all)
        max_score = max(r.anomaly_score for r in all_results)

        # Determine highest severity
        severity_order = [
            AnomalySeverity.CRITICAL,
            AnomalySeverity.HIGH,
            AnomalySeverity.MEDIUM,
            AnomalySeverity.LOW,
        ]
        severities = [r.severity for r in all_results if r.severity is not None]
        max_severity = None
        for sev in severity_order:
            if sev in severities:
                max_severity = sev
                break

        # Build combined details
        details: dict[str, Any] = {
            "route_path": context.route_path,
            "total_detectors": len(context.votes),
            "flagged_detectors": context.get_flagged_detectors(),
            "confidence": context.confidence,
            "short_circuited": context.short_circuited,
        }

        if context.short_circuited:
            details["short_circuit_reason"] = context.short_circuit_reason

        # Add per-layer details
        for layer_name, layer_results in context.layer_results.items():
            details[layer_name] = [r.details for r in layer_results]

        is_anomaly = len(all_types) > 0

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=max_score,
            anomaly_types=list(all_types),
            severity=max_severity,
            details=details,
            detector=self.name,
            competitor_product_id=context.numeric_features.competitor_product_id,
            competitor=context.numeric_features.competitor,
        )

    def get_layer_names(self) -> list[str]:
        """Get names of all layers in order.

        Returns:
            List of layer names.
        """
        return [layer.name for layer in self.layers]

    def get_detector_names(self) -> list[str]:
        """Get names of all detectors across all layers.

        Returns:
            List of detector names.
        """
        names = []
        for layer in self.layers:
            for detector in layer.detectors:
                names.append(getattr(detector, "name", str(type(detector).__name__)))
        return names


# =============================================================================
# Concrete Combined Detector
# =============================================================================


class CombinedDetector(BaseCombinedDetector):
    """Concrete combined detector for ad-hoc layer composition.

    Accepts layers directly in constructor rather than requiring subclassing.
    Use for testing, experimentation, or dynamic detector composition.
    For production variants with fixed layer configurations, prefer named
    subclasses in combined_variants.py.

    Example:
        detector = CombinedDetector(
            config=CombinedDetectorConfig(name="custom"),
            layers=[
                DetectorLayer("sanity", [SanityCheckDetector()], is_gate=True),
                DetectorLayer("statistical", [ZScoreDetector(), IQRDetector()]),
            ],
        )
    """

    def __init__(
        self,
        config: CombinedDetectorConfig,
        layers: list[DetectorLayer],
    ) -> None:
        """Initialize with configuration and layers.

        Args:
            config: CombinedDetectorConfig with variant settings.
            layers: List of DetectorLayer instances in execution order.
        """
        super().__init__(config)
        self._provided_layers = layers

    def get_layers(self) -> list[DetectorLayer]:
        """Return the provided layers.

        Returns:
            List of DetectorLayer instances passed to constructor.
        """
        return self._provided_layers
