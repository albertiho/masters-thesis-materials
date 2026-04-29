"""Statistical Anomaly Detection - Baseline Methods.

Implements classical statistical methods for anomaly detection:
- Z-score: Detect outliers based on standard deviations from mean
- IQR: Interquartile range based outlier detection
- Threshold: Simple percentage change thresholds
- Sanity checks: Business rule violations (e.g., sale > list price)

These serve as baselines for comparison with ML methods.
All methods produce comparable anomaly scores and flags.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

import numpy as np

from src.anomaly.base import BaseDetector
from src.anomaly.z_score_methods import mad_scale, sn_scale
from src.constants import COUNTRY_CURRENCY_MAP
from src.features.numeric import NumericFeatures
from src.features.temporal import MIN_OBSERVATIONS, TemporalFeatures

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of anomalies detected."""

    PRICE_ZSCORE = "price_zscore"  # Price far from historical mean
    PRICE_IQR = "price_iqr"  # Price outside IQR bounds
    PRICE_CHANGE = "price_change"  # Large percentage change
    PRICE_SANITY = "price_sanity"  # Business rule violation
    DATA_QUALITY = "data_quality"  # Missing/invalid data
    # Tier 0 invariants (deterministic, high precision)
    TITLE_COLLAPSE = "title_collapse"  # Title shortened significantly + price changed
    PLACEHOLDER_IMAGE = "placeholder_image"  # Known placeholder image detected
    CURRENCY_MISMATCH = "currency_mismatch"  # Currency doesn't match expected for country
    EXTREME_PRICE = "extreme_price"  # Price far outside expected range
    CONTENT_ANOMALY = "content_anomaly"  # General content degradation


class AnomalySeverity(Enum):
    """Severity levels for detected anomalies."""

    LOW = "low"  # Worth noting, likely not actionable
    MEDIUM = "medium"  # Should be reviewed
    HIGH = "high"  # Likely requires action
    CRITICAL = "critical"  # Definitely requires immediate action


@dataclass
class AnomalyResult:
    """Result of anomaly detection for a single record.

    Attributes:
        is_anomaly: Whether any anomaly was detected
        anomaly_score: Aggregate anomaly score (0-1, higher = more anomalous)
        anomaly_types: List of anomaly types detected
        severity: Highest severity of detected anomalies
        details: Dictionary with details for each anomaly type
        detector: Name of the detector that produced this result
    """

    is_anomaly: bool
    anomaly_score: float
    anomaly_types: list[AnomalyType]
    severity: AnomalySeverity | None
    details: dict[str, Any]
    detector: str

    # Identifiers for joining back
    competitor_product_id: str
    competitor: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            "competitor_product_id": self.competitor_product_id,
            "competitor": self.competitor,
            "is_anomaly": self.is_anomaly,
            "anomaly_score": self.anomaly_score,
            "anomaly_types": [t.value for t in self.anomaly_types],
            "severity": self.severity.value if self.severity else None,
            "detector": self.detector,
            **self.details,
        }


# Default thresholds
DEFAULT_ZSCORE_THRESHOLD = 3.0  # Standard deviations
DEFAULT_MODIFIED_ZSCORE_THRESHOLD = 2.0  # Robust Z-score variants
DEFAULT_IQR_MULTIPLIER = 1.5  # IQR multiplier for outlier bounds
DEFAULT_PRICE_CHANGE_THRESHOLD = 0.20  # 20% change


def _validate_batch_inputs(
    numeric_features_list: Sequence[NumericFeatures],
    temporal_features_list: Sequence[TemporalFeatures],
    price_history_list: Sequence[list[float] | None] | None = None,
) -> list[list[float] | None]:
    """Validate aligned batch inputs and normalize optional histories."""
    if len(numeric_features_list) != len(temporal_features_list):
        raise ValueError("Feature lists must have same length")
    if price_history_list is None:
        return [None] * len(numeric_features_list)
    if len(numeric_features_list) != len(price_history_list):
        raise ValueError("Price history list must have same length as feature lists")
    return list(price_history_list)


def _zscore_severity(
    score: float,
    threshold: float,
) -> AnomalySeverity | None:
    """Map a Z-score-like deviation to severity bands."""
    if score > threshold * 2:
        return AnomalySeverity.CRITICAL
    if score > threshold * 1.5:
        return AnomalySeverity.HIGH
    if score > threshold:
        return AnomalySeverity.MEDIUM
    return None


class _HistoryBasedZScoreDetector(BaseDetector):
    """Base class for history-only Z-score variant wrappers."""

    requires_price_history = True
    min_history = MIN_OBSERVATIONS
    score_detail_key = "variant_score"

    def __init__(self, threshold: float = DEFAULT_MODIFIED_ZSCORE_THRESHOLD):
        self.threshold = threshold

    def detect(
        self,
        numeric_features: NumericFeatures,
        temporal_features: TemporalFeatures,
        price_history: list[float] | None = None,
    ) -> AnomalyResult:
        return self._detect_single(
            numeric_features=numeric_features,
            temporal_features=temporal_features,
            price_history=price_history,
        )

    def detect_batch(
        self,
        numeric_features_list: list[NumericFeatures],
        temporal_features_list: list[TemporalFeatures],
        price_history_list: list[list[float] | None] | None = None,
    ) -> list[AnomalyResult]:
        histories = _validate_batch_inputs(
            numeric_features_list,
            temporal_features_list,
            price_history_list,
        )
        return [
            self._detect_single(
                numeric_features=numeric_features,
                temporal_features=temporal_features,
                price_history=price_history,
            )
            for numeric_features, temporal_features, price_history in zip(
                numeric_features_list,
                temporal_features_list,
                histories,
                strict=True,
            )
        ]

    def _detect_single(
        self,
        numeric_features: NumericFeatures,
        temporal_features: TemporalFeatures,
        price_history: list[float] | None = None,
    ) -> AnomalyResult:
        del temporal_features  # Kept for detector contract compatibility.

        details: dict[str, Any] = {
            "threshold": self.threshold,
            "observation_count": len(price_history) if price_history else 0,
        }

        if price_history is None or len(price_history) < self.min_history:
            details["insufficient_history"] = True
            details["minimum_history_required"] = self.min_history
            return self._build_result(
                numeric_features=numeric_features,
                details=details,
            )

        history = np.asarray(price_history, dtype=np.float64)

        try:
            variant_score, variant_details = self._compute_variant_score(
                current_price=float(numeric_features.price),
                price_history=history,
            )
        except ValueError as exc:
            details["degenerate_scale"] = True
            details["degenerate_reason"] = str(exc)
            return self._build_result(
                numeric_features=numeric_features,
                details=details,
            )

        details.update(variant_details)
        details[self.score_detail_key] = variant_score

        anomaly_score = self.normalize_score(variant_score, self.threshold)
        anomaly_types: list[AnomalyType] = []
        if variant_score > self.threshold:
            anomaly_types.append(AnomalyType.PRICE_ZSCORE)
            details["deviation_score"] = variant_score

        severity = _zscore_severity(variant_score, self.threshold) if anomaly_types else None
        is_anomaly = len(anomaly_types) > 0

        if is_anomaly:
            logger.debug(
                "anomaly_detected",
                extra={
                    "detector": self.name,
                    "competitor_product_id": numeric_features.competitor_product_id,
                    "competitor": numeric_features.competitor,
                    "country": numeric_features.country,
                    "price": numeric_features.price,
                    "anomaly_types": [t.value for t in anomaly_types],
                    "severity": severity.value if severity else None,
                    "anomaly_score": anomaly_score,
                    **details,
                },
            )

        return self._build_result(
            numeric_features=numeric_features,
            details=details,
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            anomaly_types=anomaly_types,
            severity=severity,
        )

    def _build_result(
        self,
        numeric_features: NumericFeatures,
        details: dict[str, Any],
        is_anomaly: bool = False,
        anomaly_score: float = 0.0,
        anomaly_types: list[AnomalyType] | None = None,
        severity: AnomalySeverity | None = None,
    ) -> AnomalyResult:
        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            anomaly_types=anomaly_types or [],
            severity=severity,
            details=details,
            detector=self.name,
            competitor_product_id=numeric_features.competitor_product_id,
            competitor=numeric_features.competitor,
        )

    def _compute_variant_score(
        self,
        current_price: float,
        price_history: np.ndarray,
    ) -> tuple[float, dict[str, Any]]:
        raise NotImplementedError

    def _validate_scale(self, scale_name: str, scale_value: float) -> float:
        if not np.isfinite(scale_value) or scale_value <= 0:
            raise ValueError(f"{scale_name} is zero or undefined")
        return float(scale_value)

    def _baseline_median(self, price_history: np.ndarray) -> float:
        return float(np.median(price_history))


class ModifiedMADDetector(_HistoryBasedZScoreDetector):
    """History-based modified Z-score using MAD scale."""

    score_detail_key = "modified_mad_score"

    def __init__(self, threshold: float = DEFAULT_MODIFIED_ZSCORE_THRESHOLD):
        super().__init__(threshold=threshold)
        self.name = "modified_mad"

    def _compute_variant_score(
        self,
        current_price: float,
        price_history: np.ndarray,
    ) -> tuple[float, dict[str, Any]]:
        baseline_median = self._baseline_median(price_history)
        mad_sigma = self._validate_scale("mad_scale", mad_scale(price_history))
        score = abs(current_price - baseline_median) / mad_sigma
        return score, {
            "baseline_median": baseline_median,
            "mad_scale": mad_sigma,
        }


class ModifiedSNDetector(_HistoryBasedZScoreDetector):
    """History-based modified Z-score using Sn scale."""

    score_detail_key = "modified_sn_score"

    def __init__(self, threshold: float = DEFAULT_MODIFIED_ZSCORE_THRESHOLD):
        super().__init__(threshold=threshold)
        self.name = "modified_sn"

    def _compute_variant_score(
        self,
        current_price: float,
        price_history: np.ndarray,
    ) -> tuple[float, dict[str, Any]]:
        baseline_median = self._baseline_median(price_history)
        sn_sigma = self._validate_scale("sn_scale", sn_scale(price_history))
        score = abs(current_price - baseline_median) / sn_sigma
        return score, {
            "baseline_median": baseline_median,
            "sn_scale": sn_sigma,
        }


class HybridWeightedZScoreDetector(_HistoryBasedZScoreDetector):
    """History-based weighted hybrid Z-score using MAD and Sn scores."""

    score_detail_key = "hybrid_weighted_score"

    def __init__(
        self,
        threshold: float = DEFAULT_MODIFIED_ZSCORE_THRESHOLD,
        w: float = 0.5,
    ):
        if not 0.0 <= w <= 1.0:
            raise ValueError(f"Weight w must be in [0, 1], got {w}.")
        super().__init__(threshold=threshold)
        self.w = w
        self.name = "hybrid_weighted_zscore"

    def _compute_variant_score(
        self,
        current_price: float,
        price_history: np.ndarray,
    ) -> tuple[float, dict[str, Any]]:
        baseline_median = self._baseline_median(price_history)
        mad_sigma = self._validate_scale("mad_scale", mad_scale(price_history))
        sn_sigma = self._validate_scale("sn_scale", sn_scale(price_history))
        mad_score = abs(current_price - baseline_median) / mad_sigma
        sn_score = abs(current_price - baseline_median) / sn_sigma
        score = self.w * mad_score + (1.0 - self.w) * sn_score
        return score, {
            "baseline_median": baseline_median,
            "mad_scale": mad_sigma,
            "sn_scale": sn_sigma,
            "mad_score": mad_score,
            "sn_score": sn_score,
            "weight": self.w,
        }


class HybridMaxZScoreDetector(_HistoryBasedZScoreDetector):
    """History-based maximum hybrid Z-score using MAD and Sn scores."""

    score_detail_key = "hybrid_max_score"

    def __init__(self, threshold: float = DEFAULT_MODIFIED_ZSCORE_THRESHOLD):
        super().__init__(threshold=threshold)
        self.name = "hybrid_max_zscore"

    def _compute_variant_score(
        self,
        current_price: float,
        price_history: np.ndarray,
    ) -> tuple[float, dict[str, Any]]:
        baseline_median = self._baseline_median(price_history)
        mad_sigma = self._validate_scale("mad_scale", mad_scale(price_history))
        sn_sigma = self._validate_scale("sn_scale", sn_scale(price_history))
        mad_score = abs(current_price - baseline_median) / mad_sigma
        sn_score = abs(current_price - baseline_median) / sn_sigma
        score = max(mad_score, sn_score)
        return score, {
            "baseline_median": baseline_median,
            "mad_scale": mad_sigma,
            "sn_scale": sn_sigma,
            "mad_score": mad_score,
            "sn_score": sn_score,
        }


class HybridAvgZScoreDetector(_HistoryBasedZScoreDetector):
    """History-based average hybrid Z-score using MAD and Sn scores."""

    score_detail_key = "hybrid_avg_score"

    def __init__(self, threshold: float = DEFAULT_MODIFIED_ZSCORE_THRESHOLD):
        super().__init__(threshold=threshold)
        self.name = "hybrid_avg_zscore"

    def _compute_variant_score(
        self,
        current_price: float,
        price_history: np.ndarray,
    ) -> tuple[float, dict[str, Any]]:
        baseline_median = self._baseline_median(price_history)
        mad_sigma = self._validate_scale("mad_scale", mad_scale(price_history))
        sn_sigma = self._validate_scale("sn_scale", sn_scale(price_history))
        mad_score = abs(current_price - baseline_median) / mad_sigma
        sn_score = abs(current_price - baseline_median) / sn_sigma
        score = 0.5 * (mad_score + sn_score)
        return score, {
            "baseline_median": baseline_median,
            "mad_scale": mad_sigma,
            "sn_scale": sn_sigma,
            "mad_score": mad_score,
            "sn_score": sn_score,
        }


class ZScoreDetector(BaseDetector):
    """Detect anomalies based on Z-score (standard deviations from mean).

    A price is flagged if:
        |price - rolling_mean| / rolling_std > threshold

    This is the most common statistical anomaly detection method.
    """

    def __init__(self, threshold: float = DEFAULT_ZSCORE_THRESHOLD):
        """Initialize Z-score detector.

        Args:
            threshold: Number of standard deviations to flag as anomaly.
        """
        self.threshold = threshold
        self.name = "zscore"

    def detect(
        self,
        numeric_features: NumericFeatures,
        temporal_features: TemporalFeatures,
    ) -> AnomalyResult:
        return self._detect_single(
            numeric_features=numeric_features,
            temporal_features=temporal_features,
        )

    def detect_batch(
        self,
        numeric_features_list: list[NumericFeatures],
        temporal_features_list: list[TemporalFeatures],
    ) -> list[AnomalyResult]:
        _validate_batch_inputs(numeric_features_list, temporal_features_list)
        return [
            self._detect_single(
                numeric_features=numeric_features,
                temporal_features=temporal_features,
            )
            for numeric_features, temporal_features in zip(
                numeric_features_list,
                temporal_features_list,
                strict=True,
            )
        ]

    def _detect_single(
        self,
        numeric_features: NumericFeatures,
        temporal_features: TemporalFeatures,
    ) -> AnomalyResult:
        """Detect anomalies using Z-score method.

        Args:
            numeric_features: Numeric features from the record.
            temporal_features: Temporal features with rolling statistics.

        Returns:
            AnomalyResult with detection results.
        """
        anomaly_types: list[AnomalyType] = []
        details: dict[str, Any] = {}
        anomaly_score = 0.0

        # Check if we have sufficient history
        if not temporal_features.has_sufficient_history:
            details["insufficient_history"] = True
            details["observation_count"] = temporal_features.observation_count
        elif temporal_features.price_zscore is not None:
            zscore = abs(temporal_features.price_zscore)
            details["zscore"] = temporal_features.price_zscore
            details["threshold"] = self.threshold
            details["rolling_mean"] = temporal_features.rolling_mean
            details["rolling_std"] = temporal_features.rolling_std

            # Normalize score to 0-1 range using base class method
            anomaly_score = self.normalize_score(zscore, self.threshold)

            if zscore > self.threshold:
                anomaly_types.append(AnomalyType.PRICE_ZSCORE)
                details["deviation_std"] = zscore

        # Determine severity
        severity = None
        if anomaly_types:
            zscore = abs(temporal_features.price_zscore or 0)
            if zscore > self.threshold * 2:
                severity = AnomalySeverity.CRITICAL
            elif zscore > self.threshold * 1.5:
                severity = AnomalySeverity.HIGH
            elif zscore > self.threshold:
                severity = AnomalySeverity.MEDIUM

        is_anomaly = len(anomaly_types) > 0

        # Log for audit trail
        if is_anomaly:
            logger.debug(
                "anomaly_detected",
                extra={
                    "detector": self.name,
                    "competitor_product_id": numeric_features.competitor_product_id,
                    "competitor": numeric_features.competitor,
                    "country": numeric_features.country,
                    "price": numeric_features.price,
                    "anomaly_types": [t.value for t in anomaly_types],
                    "severity": severity.value if severity else None,
                    "anomaly_score": anomaly_score,
                    **details,
                },
            )

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            anomaly_types=anomaly_types,
            severity=severity,
            details=details,
            detector=self.name,
            competitor_product_id=numeric_features.competitor_product_id,
            competitor=numeric_features.competitor,
        )


class IQRDetector(BaseDetector):
    """Detect anomalies based on Interquartile Range (IQR).

    A price is flagged if outside:
        [Q1 - multiplier * IQR, Q3 + multiplier * IQR]

    More robust to outliers than Z-score (doesn't assume normal distribution).
    """

    requires_price_history = True

    def __init__(self, multiplier: float = DEFAULT_IQR_MULTIPLIER):
        """Initialize IQR detector.

        Args:
            multiplier: IQR multiplier for bounds (typically 1.5 or 3.0).
        """
        self.multiplier = multiplier
        self.name = "iqr"

    def detect(
        self,
        numeric_features: NumericFeatures,
        temporal_features: TemporalFeatures,
        price_history: list[float] | None = None,
    ) -> AnomalyResult:
        return self._detect_single(
            numeric_features=numeric_features,
            temporal_features=temporal_features,
            price_history=price_history,
        )

    def detect_batch(
        self,
        numeric_features_list: list[NumericFeatures],
        temporal_features_list: list[TemporalFeatures],
        price_history_list: list[list[float] | None] | None = None,
    ) -> list[AnomalyResult]:
        histories = _validate_batch_inputs(
            numeric_features_list,
            temporal_features_list,
            price_history_list,
        )
        return [
            self._detect_single(
                numeric_features=numeric_features,
                temporal_features=temporal_features,
                price_history=price_history,
            )
            for numeric_features, temporal_features, price_history in zip(
                numeric_features_list,
                temporal_features_list,
                histories,
                strict=True,
            )
        ]

    def _detect_single(
        self,
        numeric_features: NumericFeatures,
        temporal_features: TemporalFeatures,
        price_history: list[float] | None = None,
    ) -> AnomalyResult:
        """Detect anomalies using IQR method.

        Args:
            numeric_features: Numeric features from the record.
            temporal_features: Temporal features (for context).
            price_history: Optional list of historical prices for IQR calculation.

        Returns:
            AnomalyResult with detection results.
        """
        anomaly_types: list[AnomalyType] = []
        details: dict[str, Any] = {}
        anomaly_score = 0.0

        # Need price history for IQR
        if price_history is None or len(price_history) < 4:
            details["insufficient_history"] = True
            details["observation_count"] = len(price_history) if price_history else 0
            return AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                anomaly_types=[],
                severity=None,
                details=details,
                detector=self.name,
                competitor_product_id=numeric_features.competitor_product_id,
                competitor=numeric_features.competitor,
            )

        # Calculate quartiles
        sorted_prices = sorted(price_history)
        n = len(sorted_prices)
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        q1 = sorted_prices[q1_idx]
        q3 = sorted_prices[q3_idx]
        iqr = q3 - q1

        # Calculate bounds
        lower_bound = q1 - self.multiplier * iqr
        upper_bound = q3 + self.multiplier * iqr

        current_price = numeric_features.price
        details["q1"] = q1
        details["q3"] = q3
        details["iqr"] = iqr
        details["lower_bound"] = lower_bound
        details["upper_bound"] = upper_bound
        details["multiplier"] = self.multiplier

        # Check if outside bounds
        if current_price < lower_bound:
            anomaly_types.append(AnomalyType.PRICE_IQR)
            distance = lower_bound - current_price
            details["below_lower_bound"] = True
            details["distance_from_bound"] = distance
            anomaly_score = self.normalize_score(distance, iqr) if iqr > 0 else 0.5
        elif current_price > upper_bound:
            anomaly_types.append(AnomalyType.PRICE_IQR)
            distance = current_price - upper_bound
            details["above_upper_bound"] = True
            details["distance_from_bound"] = distance
            anomaly_score = self.normalize_score(distance, iqr) if iqr > 0 else 0.5

        # Determine severity
        severity = None
        if anomaly_types:
            distance = details.get("distance_from_bound", 0)
            if iqr > 0:
                distance_ratio = distance / iqr
                if distance_ratio > 3:
                    severity = AnomalySeverity.CRITICAL
                elif distance_ratio > 2:
                    severity = AnomalySeverity.HIGH
                else:
                    severity = AnomalySeverity.MEDIUM

        is_anomaly = len(anomaly_types) > 0

        if is_anomaly:
            logger.debug(
                "anomaly_detected",
                extra={
                    "detector": self.name,
                    "competitor_product_id": numeric_features.competitor_product_id,
                    "competitor": numeric_features.competitor,
                    "country": numeric_features.country,
                    "price": numeric_features.price,
                    "anomaly_types": [t.value for t in anomaly_types],
                    "severity": severity.value if severity else None,
                    "anomaly_score": anomaly_score,
                    **details,
                },
            )

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            anomaly_types=anomaly_types,
            severity=severity,
            details=details,
            detector=self.name,
            competitor_product_id=numeric_features.competitor_product_id,
            competitor=numeric_features.competitor,
        )


class ThresholdDetector(BaseDetector):
    """Detect anomalies based on simple percentage change thresholds.

    Flags records where:
        |price_change_pct| > threshold

    Simple but effective for catching sudden large changes.
    """

    def __init__(self, threshold: float = DEFAULT_PRICE_CHANGE_THRESHOLD):
        """Initialize threshold detector.

        Args:
            threshold: Percentage change threshold (e.g., 0.20 for 20%).
        """
        self.threshold = threshold
        self.name = "threshold"

    def detect(
        self,
        numeric_features: NumericFeatures,
        temporal_features: TemporalFeatures,
    ) -> AnomalyResult:
        return self._detect_single(
            numeric_features=numeric_features,
            temporal_features=temporal_features,
        )

    def detect_batch(
        self,
        numeric_features_list: list[NumericFeatures],
        temporal_features_list: list[TemporalFeatures],
    ) -> list[AnomalyResult]:
        _validate_batch_inputs(numeric_features_list, temporal_features_list)
        valid_change_mask = np.asarray(
            [temporal_features.price_change_pct is not None for temporal_features in temporal_features_list],
            dtype=bool,
        )
        price_changes = np.asarray(
            [
                float(temporal_features.price_change_pct)
                if temporal_features.price_change_pct is not None
                else 0.0
                for temporal_features in temporal_features_list
            ],
            dtype=np.float64,
        )
        abs_changes = np.abs(price_changes)

        if self.threshold <= 0:
            anomaly_scores = np.where(abs_changes > 0.0, 0.5, 0.0)
        else:
            anomaly_scores = np.minimum(abs_changes / (self.threshold * 2.0), 1.0)
        anomaly_scores = np.where(valid_change_mask, anomaly_scores, 0.0)

        is_anomaly_mask = valid_change_mask & (abs_changes > self.threshold)
        critical_mask = is_anomaly_mask & (abs_changes > (self.threshold * 3.0))
        high_mask = is_anomaly_mask & ~critical_mask & (abs_changes > (self.threshold * 2.0))
        medium_mask = is_anomaly_mask & ~critical_mask & ~high_mask

        results: list[AnomalyResult] = []
        for index, (numeric_features, temporal_features) in enumerate(
            zip(numeric_features_list, temporal_features_list, strict=True)
        ):
            details: dict[str, Any] = {"threshold": self.threshold}
            anomaly_types: list[AnomalyType] = []
            severity = None

            if not valid_change_mask[index]:
                details["no_previous_price"] = True
            else:
                price_change_pct = temporal_features.price_change_pct
                assert price_change_pct is not None
                details["price_change_pct"] = price_change_pct
                details["abs_change"] = float(abs_changes[index])

                if is_anomaly_mask[index]:
                    anomaly_types.append(AnomalyType.PRICE_CHANGE)
                    details["direction"] = "increase" if price_change_pct > 0 else "decrease"
                    if critical_mask[index]:
                        severity = AnomalySeverity.CRITICAL
                    elif high_mask[index]:
                        severity = AnomalySeverity.HIGH
                    elif medium_mask[index]:
                        severity = AnomalySeverity.MEDIUM

            anomaly_score = float(anomaly_scores[index])
            is_anomaly = bool(is_anomaly_mask[index])

            if is_anomaly:
                logger.debug(
                    "anomaly_detected",
                    extra={
                        "detector": self.name,
                        "competitor_product_id": numeric_features.competitor_product_id,
                        "competitor": numeric_features.competitor,
                        "country": numeric_features.country,
                        "price": numeric_features.price,
                        "anomaly_types": [t.value for t in anomaly_types],
                        "severity": severity.value if severity else None,
                        "anomaly_score": anomaly_score,
                        **details,
                    },
                )

            results.append(
                AnomalyResult(
                    is_anomaly=is_anomaly,
                    anomaly_score=anomaly_score,
                    anomaly_types=anomaly_types,
                    severity=severity,
                    details=details,
                    detector=self.name,
                    competitor_product_id=numeric_features.competitor_product_id,
                    competitor=numeric_features.competitor,
                )
            )

        return results

    def _detect_single(
        self,
        numeric_features: NumericFeatures,
        temporal_features: TemporalFeatures,
    ) -> AnomalyResult:
        """Detect anomalies using percentage change threshold.

        Args:
            numeric_features: Numeric features from the record.
            temporal_features: Temporal features with price change percentage.

        Returns:
            AnomalyResult with detection results.
        """
        anomaly_types: list[AnomalyType] = []
        details: dict[str, Any] = {}
        anomaly_score = 0.0

        price_change_pct = temporal_features.price_change_pct
        details["threshold"] = self.threshold

        if price_change_pct is None:
            details["no_previous_price"] = True
        else:
            abs_change = abs(price_change_pct)
            details["price_change_pct"] = price_change_pct
            details["abs_change"] = abs_change

            # Normalize score using base class method
            anomaly_score = self.normalize_score(abs_change, self.threshold)

            if abs_change > self.threshold:
                anomaly_types.append(AnomalyType.PRICE_CHANGE)
                details["direction"] = "increase" if price_change_pct > 0 else "decrease"

        # Determine severity based on magnitude
        severity = None
        if anomaly_types and price_change_pct is not None:
            abs_change = abs(price_change_pct)
            if abs_change > self.threshold * 3:
                severity = AnomalySeverity.CRITICAL
            elif abs_change > self.threshold * 2:
                severity = AnomalySeverity.HIGH
            else:
                severity = AnomalySeverity.MEDIUM

        is_anomaly = len(anomaly_types) > 0

        if is_anomaly:
            logger.debug(
                "anomaly_detected",
                extra={
                    "detector": self.name,
                    "competitor_product_id": numeric_features.competitor_product_id,
                    "competitor": numeric_features.competitor,
                    "country": numeric_features.country,
                    "price": numeric_features.price,
                    "anomaly_types": [t.value for t in anomaly_types],
                    "severity": severity.value if severity else None,
                    "anomaly_score": anomaly_score,
                    **details,
                },
            )

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            anomaly_types=anomaly_types,
            severity=severity,
            details=details,
            detector=self.name,
            competitor_product_id=numeric_features.competitor_product_id,
            competitor=numeric_features.competitor,
        )


class SanityCheckDetector(BaseDetector):
    """Detect business rule violations and data quality issues.

    Checks:
        - Sale price > list price (impossible without data error)
        - Negative prices
        - Zero prices (may be valid but unusual)
        - Missing required fields

    Uses fixed scores based on violation severity - does not use threshold-based
    normalization since these are deterministic rule violations.
    """

    def __init__(self):
        """Initialize sanity check detector."""
        self.name = "sanity"

    @staticmethod
    def _expected_currency(country: str | None) -> str | None:
        if not country:
            return None

        normalized = country.strip().upper()
        thesis_map = {
            "COUNTRY_1": "CURRENCY_1",
            "COUNTRY_2": "CURRENCY_2",
            "COUNTRY_3": "CURRENCY_3",
            "COUNTRY_4": "CURRENCY_4",
        }
        if normalized in thesis_map:
            return thesis_map[normalized]
        return COUNTRY_CURRENCY_MAP.get(normalized)

    def normalize_score(
        self,
        value: float,
        threshold: float = 0,
        cap_multiple: float = 2.0,
    ) -> float:
        """Return value as-is for sanity detectors.

        Sanity detectors use fixed scores based on violation type,
        not threshold-based normalization.

        Args:
            value: The fixed anomaly score (0-1).
            threshold: Ignored for sanity detectors.
            cap_multiple: Ignored for sanity detectors.

        Returns:
            The value unchanged, clamped to [0, 1].
        """
        return max(0.0, min(1.0, value))

    def detect(
        self,
        numeric_features: NumericFeatures,
    ) -> AnomalyResult:
        """Detect business rule violations and data quality issues.

        Args:
            numeric_features: Numeric features from the record.

        Returns:
            AnomalyResult with detection results.
        """
        anomaly_types: list[AnomalyType] = []
        details: dict[str, Any] = {}
        anomaly_score = 0.0
        severity = None

        # Check validation errors from numeric feature extraction
        if numeric_features.validation_errors:
            for error in numeric_features.validation_errors:
                if "negative" in error or "missing" in error:
                    anomaly_types.append(AnomalyType.DATA_QUALITY)
                elif "exceeds_list" in error:
                    anomaly_types.append(AnomalyType.PRICE_SANITY)
                elif "extreme_discount" in error:
                    anomaly_types.append(AnomalyType.PRICE_SANITY)

            details["validation_errors"] = numeric_features.validation_errors
            anomaly_score = 0.8  # High score for data quality issues

        # Check price ratio sanity
        if numeric_features.has_list_price and numeric_features.price_ratio > 1.0:
            if AnomalyType.PRICE_SANITY not in anomaly_types:
                anomaly_types.append(AnomalyType.PRICE_SANITY)
            details["price_ratio"] = numeric_features.price_ratio
            details["sale_exceeds_list"] = True
            anomaly_score = max(anomaly_score, 0.9)
            severity = AnomalySeverity.HIGH

        expected_currency = self._expected_currency(numeric_features.country)
        actual_currency = (
            numeric_features.currency.strip().upper()
            if isinstance(numeric_features.currency, str)
            else None
        )
        if expected_currency and actual_currency and actual_currency != expected_currency:
            if AnomalyType.CURRENCY_MISMATCH not in anomaly_types:
                anomaly_types.append(AnomalyType.CURRENCY_MISMATCH)
            details["currency_mismatch"] = {
                "country": numeric_features.country,
                "currency": actual_currency,
                "expected_currency": expected_currency,
            }
            anomaly_score = max(anomaly_score, 0.95)
            severity = AnomalySeverity.HIGH

        # Check for data validity
        if not numeric_features.is_valid:
            if AnomalyType.DATA_QUALITY not in anomaly_types:
                anomaly_types.append(AnomalyType.DATA_QUALITY)
            details["is_valid"] = False
            anomaly_score = 1.0
            severity = AnomalySeverity.CRITICAL

        is_anomaly = len(anomaly_types) > 0

        # Set severity if not already set
        if is_anomaly and severity is None:
            if AnomalyType.DATA_QUALITY in anomaly_types:
                severity = AnomalySeverity.HIGH
            else:
                severity = AnomalySeverity.MEDIUM

        if is_anomaly:
            logger.debug(
                "anomaly_detected",
                extra={
                    "detector": self.name,
                    "competitor_product_id": numeric_features.competitor_product_id,
                    "competitor": numeric_features.competitor,
                    "country": numeric_features.country,
                    "currency": numeric_features.currency,
                    "price": numeric_features.price,
                    "list_price": numeric_features.list_price,
                    "anomaly_types": [t.value for t in anomaly_types],
                    "severity": severity.value if severity else None,
                    "anomaly_score": anomaly_score,
                    **details,
                },
            )

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            anomaly_types=anomaly_types,
            severity=severity,
            details=details,
            detector=self.name,
            competitor_product_id=numeric_features.competitor_product_id,
            competitor=numeric_features.competitor,
        )


@dataclass
class InvariantContext:
    """Context for invariant checking.

    Provides additional information needed for Tier 0 invariant checks
    that isn't available in NumericFeatures alone.
    """

    # Current record info
    competitor_product_id: str
    competitor: str
    country: str | None
    currency: str | None

    # Title info
    current_title: str | None = None
    previous_title: str | None = None
    title_length_ratio: float | None = None  # current_len / previous_len

    # Image info
    image_urls: list[str] | None = None

    # Price context
    current_price: float | None = None
    previous_price: float | None = None
    price_changed: bool = False
    category_median_price: float | None = None

    @classmethod
    def from_product_record(
        cls,
        record: "ProductRecord",  # type: ignore
        previous_title: str | None = None,
        previous_price: float | None = None,
        category_median: float | None = None,
    ) -> "InvariantContext":
        """Create context from a ProductRecord.

        Args:
            record: Current product record.
            previous_title: Title from previous observation.
            previous_price: Price from previous observation.
            category_median: Median price for this category.

        Returns:
            InvariantContext instance.
        """
        # Calculate title length ratio
        title_ratio = None
        if record.product_name and previous_title:
            prev_len = len(previous_title)
            curr_len = len(record.product_name)
            if prev_len > 0:
                title_ratio = curr_len / prev_len

        # Check if price changed
        price_changed = False
        if previous_price is not None and record.price is not None:
            price_changed = abs(record.price - previous_price) > 0.01

        # Extract image URLs from raw_data
        images = None
        if record.raw_data:
            images = record.raw_data.get("images", [])
            if not images:
                img_url = record.raw_data.get("image_url")
                if img_url:
                    images = [img_url]

        return cls(
            competitor_product_id=record.competitor_product_id,
            competitor=record.competitor,
            country=record.country,
            currency=record.currency,
            current_title=record.product_name,
            previous_title=previous_title,
            title_length_ratio=title_ratio,
            image_urls=images,
            current_price=record.price,
            previous_price=previous_price,
            price_changed=price_changed,
            category_median_price=category_median,
        )


# Known placeholder image URL patterns
PLACEHOLDER_IMAGE_PATTERNS = [
    "placeholder",
    "no-image",
    "noimage",
    "default-product",
    "coming-soon",
    "not-available",
    "missing-image",
    "product-image-placeholder",
    "/static/img/placeholder",
    "/assets/placeholder",
    "data:image/",  # Data URIs often indicate placeholder
]


class InvariantDetector(BaseDetector):
    """Detect deterministic invariant violations (Tier 0).

    High-precision checks that are always indicative of data issues:
    - Title collapse: Title shortened >80% AND price changed
    - Placeholder images: Known placeholder URL patterns
    - Currency mismatch: Currency doesn't match expected for country
    - Extreme prices: Price >10x or <0.1x category median

    These checks are cheap and extremely actionable.

    Uses fixed scores based on violation type - does not use threshold-based
    normalization since these are deterministic invariant violations.
    """

    def __init__(
        self,
        title_collapse_threshold: float = 0.2,  # Alert if title < 20% of previous
        extreme_price_factor: float = 10.0,  # Alert if price > 10x or < 0.1x median
    ):
        """Initialize invariant detector.

        Args:
            title_collapse_threshold: Ratio below which title is considered collapsed.
            extreme_price_factor: Factor for extreme price detection.
        """
        self.title_collapse_threshold = title_collapse_threshold
        self.extreme_price_factor = extreme_price_factor
        self.name = "invariant"

    def normalize_score(
        self,
        value: float,
        threshold: float = 0,
        cap_multiple: float = 2.0,
    ) -> float:
        """Return value as-is for invariant detectors.

        Invariant detectors use fixed scores based on violation type,
        not threshold-based normalization.

        Args:
            value: The fixed anomaly score (0-1).
            threshold: Ignored for invariant detectors.
            cap_multiple: Ignored for invariant detectors.

        Returns:
            The value unchanged, clamped to [0, 1].
        """
        return max(0.0, min(1.0, value))

    def detect(self, context: InvariantContext) -> AnomalyResult:
        """Detect invariant violations.

        Args:
            context: InvariantContext with necessary information.

        Returns:
            AnomalyResult with detection results.
        """
        anomaly_types: list[AnomalyType] = []
        details: dict[str, Any] = {}
        anomaly_score = 0.0
        severity = None

        # Check 1: Title collapse
        title_result = self._check_title_collapse(context)
        if title_result:
            anomaly_types.append(AnomalyType.TITLE_COLLAPSE)
            details["title_collapse"] = title_result
            anomaly_score = max(anomaly_score, 0.9)
            severity = AnomalySeverity.CRITICAL

        # Check 2: Placeholder images
        placeholder_result = self._check_placeholder_images(context)
        if placeholder_result:
            anomaly_types.append(AnomalyType.PLACEHOLDER_IMAGE)
            details["placeholder_image"] = placeholder_result
            anomaly_score = max(anomaly_score, 0.6)
            if severity is None:
                severity = AnomalySeverity.MEDIUM

        # Check 3: Currency mismatch
        currency_result = self._check_currency_mismatch(context)
        if currency_result:
            anomaly_types.append(AnomalyType.CURRENCY_MISMATCH)
            details["currency_mismatch"] = currency_result
            anomaly_score = max(anomaly_score, 0.95)
            severity = AnomalySeverity.CRITICAL

        # Check 4: Extreme price
        extreme_result = self._check_extreme_price(context)
        if extreme_result:
            anomaly_types.append(AnomalyType.EXTREME_PRICE)
            details["extreme_price"] = extreme_result
            anomaly_score = max(anomaly_score, 0.85)
            if severity in (None, AnomalySeverity.MEDIUM, AnomalySeverity.LOW):
                severity = AnomalySeverity.HIGH

        is_anomaly = len(anomaly_types) > 0

        if is_anomaly:
            logger.info(
                "invariant_violation_detected",
                extra={
                    "detector": self.name,
                    "competitor_product_id": context.competitor_product_id,
                    "competitor": context.competitor,
                    "country": context.country,
                    "anomaly_types": [t.value for t in anomaly_types],
                    "severity": severity.value if severity else None,
                    "anomaly_score": anomaly_score,
                    "details": details,
                },
            )

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            anomaly_types=anomaly_types,
            severity=severity,
            details={"invariant": details},
            detector=self.name,
            competitor_product_id=context.competitor_product_id,
            competitor=context.competitor,
        )

    def _check_title_collapse(self, context: InvariantContext) -> dict[str, Any] | None:
        """Check for title collapse (title shortened significantly + price changed).

        This is a strong indicator of scraper failure - when CSS selectors break,
        titles often get truncated AND prices change (usually to 0 or garbage).
        """
        if context.title_length_ratio is None:
            return None

        if context.title_length_ratio < self.title_collapse_threshold and context.price_changed:
            return {
                "title_length_ratio": context.title_length_ratio,
                "previous_title_length": len(context.previous_title or ""),
                "current_title_length": len(context.current_title or ""),
                "price_changed": True,
                "reason": "Title collapsed and price changed - likely scraper failure",
            }

        return None

    def _check_placeholder_images(self, context: InvariantContext) -> dict[str, Any] | None:
        """Check for placeholder images.

        Known patterns indicate missing real product images.
        """
        if not context.image_urls:
            return None

        placeholder_urls = []
        for url in context.image_urls:
            if not url:
                continue
            url_lower = url.lower()
            for pattern in PLACEHOLDER_IMAGE_PATTERNS:
                if pattern in url_lower:
                    placeholder_urls.append(url)
                    break

        if placeholder_urls:
            return {
                "placeholder_urls": placeholder_urls[:5],  # Limit to first 5
                "total_images": len(context.image_urls),
                "placeholder_count": len(placeholder_urls),
                "reason": "Placeholder image URLs detected",
            }

        return None

    def _check_currency_mismatch(self, context: InvariantContext) -> dict[str, Any] | None:
        """Check for currency/country mismatch.

        Each country should have a specific currency. Mismatches indicate data errors.
        """
        if not context.country or not context.currency:
            return None

        expected_currency = COUNTRY_CURRENCY_MAP.get(context.country)
        if expected_currency and context.currency != expected_currency:
            return {
                "country": context.country,
                "currency": context.currency,
                "expected_currency": expected_currency,
                "reason": f"Currency {context.currency} doesn't match expected {expected_currency} for {context.country}",
            }

        return None

    def _check_extreme_price(self, context: InvariantContext) -> dict[str, Any] | None:
        """Check for extreme prices relative to category median.

        Prices >10x or <0.1x the category median are almost always errors.
        """
        if context.current_price is None or context.category_median_price is None:
            return None

        if context.category_median_price <= 0:
            return None

        ratio = context.current_price / context.category_median_price

        if ratio > self.extreme_price_factor:
            return {
                "price": context.current_price,
                "category_median": context.category_median_price,
                "ratio": ratio,
                "direction": "above",
                "reason": f"Price is {ratio:.1f}x the category median (>{self.extreme_price_factor}x)",
            }
        elif ratio < (1 / self.extreme_price_factor):
            return {
                "price": context.current_price,
                "category_median": context.category_median_price,
                "ratio": ratio,
                "direction": "below",
                "reason": f"Price is {ratio:.3f}x the category median (<{1/self.extreme_price_factor:.2f}x)",
            }

        return None


class StatisticalEnsemble:
    """Ensemble of all statistical detectors.

    Combines results from Z-score, IQR, Threshold, Sanity, and Invariant detectors
    into a unified anomaly assessment.
    """

    def __init__(
        self,
        zscore_threshold: float = DEFAULT_ZSCORE_THRESHOLD,
        iqr_multiplier: float = DEFAULT_IQR_MULTIPLIER,
        change_threshold: float = DEFAULT_PRICE_CHANGE_THRESHOLD,
        title_collapse_threshold: float = 0.2,
        extreme_price_factor: float = 10.0,
    ):
        """Initialize the ensemble with all detectors.

        Args:
            zscore_threshold: Threshold for Z-score detector.
            iqr_multiplier: Multiplier for IQR detector.
            change_threshold: Threshold for percentage change detector.
            title_collapse_threshold: Threshold for title collapse detection.
            extreme_price_factor: Factor for extreme price detection.
        """
        self.zscore_detector = ZScoreDetector(threshold=zscore_threshold)
        self.iqr_detector = IQRDetector(multiplier=iqr_multiplier)
        self.threshold_detector = ThresholdDetector(threshold=change_threshold)
        self.sanity_detector = SanityCheckDetector()
        self.invariant_detector = InvariantDetector(
            title_collapse_threshold=title_collapse_threshold,
            extreme_price_factor=extreme_price_factor,
        )
        self.name = "statistical_ensemble"

    @classmethod
    def from_config(
        cls,
        config: "StatisticalConfig",  # type: ignore[name-defined]
        title_collapse_threshold: float = 0.2,
        extreme_price_factor: float = 10.0,
    ) -> "StatisticalEnsemble":
        """Create ensemble from a StatisticalConfig.

        Factory method for creating an ensemble with persisted config thresholds.
        Invariant detector thresholds are not part of StatisticalConfig since they
        are deterministic checks that don't benefit from tuning.

        Args:
            config: StatisticalConfig with tuned thresholds.
            title_collapse_threshold: Threshold for title collapse detection.
            extreme_price_factor: Factor for extreme price detection.

        Returns:
            StatisticalEnsemble configured with the provided thresholds.
        """
        return cls(
            zscore_threshold=config.zscore_threshold,
            iqr_multiplier=config.iqr_multiplier,
            change_threshold=config.price_change_threshold,
            title_collapse_threshold=title_collapse_threshold,
            extreme_price_factor=extreme_price_factor,
        )

    def detect(
        self,
        numeric_features: NumericFeatures,
        temporal_features: TemporalFeatures,
        price_history: list[float] | None = None,
        invariant_context: InvariantContext | None = None,
    ) -> AnomalyResult:
        """Run all detectors and combine results.

        Args:
            numeric_features: Numeric features from the record.
            temporal_features: Temporal features with rolling statistics.
            price_history: Optional price history for IQR detector.
            invariant_context: Optional context for invariant checks.

        Returns:
            Combined AnomalyResult from all detectors.
        """
        # Run all detectors
        zscore_result = self.zscore_detector.detect(numeric_features, temporal_features)
        iqr_result = self.iqr_detector.detect(numeric_features, temporal_features, price_history)
        threshold_result = self.threshold_detector.detect(numeric_features, temporal_features)
        sanity_result = self.sanity_detector.detect(numeric_features)

        # Run invariant detector if context provided
        invariant_result = None
        if invariant_context:
            invariant_result = self.invariant_detector.detect(invariant_context)

        # Collect all results
        all_results = [zscore_result, iqr_result, threshold_result, sanity_result]
        if invariant_result:
            all_results.append(invariant_result)

        # Combine anomaly types (unique)
        all_types: set[AnomalyType] = set()
        for result in all_results:
            all_types.update(result.anomaly_types)

        # Aggregate score (max of all detectors)
        max_score = max(r.anomaly_score for r in all_results)

        # Determine highest severity
        severities = [r.severity for r in all_results if r.severity is not None]
        severity_order = [
            AnomalySeverity.CRITICAL,
            AnomalySeverity.HIGH,
            AnomalySeverity.MEDIUM,
            AnomalySeverity.LOW,
        ]
        max_severity = None
        for sev in severity_order:
            if sev in severities:
                max_severity = sev
                break

        # Combine details
        details = {
            "zscore": zscore_result.details,
            "iqr": iqr_result.details,
            "threshold": threshold_result.details,
            "sanity": sanity_result.details,
        }
        if invariant_result:
            details["invariant"] = invariant_result.details

        # Track which detectors triggered
        triggered = []
        for name, result in [
            ("zscore", zscore_result),
            ("iqr", iqr_result),
            ("threshold", threshold_result),
            ("sanity", sanity_result),
        ]:
            if result.is_anomaly:
                triggered.append(name)
        if invariant_result and invariant_result.is_anomaly:
            triggered.append("invariant")

        details["detectors_triggered"] = triggered

        is_anomaly = len(all_types) > 0

        if is_anomaly:
            logger.debug(
                "ensemble_anomaly_detected",
                extra={
                    "detector": self.name,
                    "competitor_product_id": numeric_features.competitor_product_id,
                    "competitor": numeric_features.competitor,
                    "country": numeric_features.country,
                    "price": numeric_features.price,
                    "anomaly_types": [t.value for t in all_types],
                    "severity": max_severity.value if max_severity else None,
                    "anomaly_score": max_score,
                    "detectors_triggered": triggered,
                },
            )

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=max_score,
            anomaly_types=list(all_types),
            severity=max_severity,
            details=details,
            detector=self.name,
            competitor_product_id=numeric_features.competitor_product_id,
            competitor=numeric_features.competitor,
        )
