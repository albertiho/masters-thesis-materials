"""Base class for anomaly detectors.

Provides a standardized interface and score normalization for all detector types.

Key Components:
- BaseDetector: Abstract base class with normalize_score() method
- normalize_score(): Converts raw values to 0-1 anomaly scores

Score Normalization:
    All detectors should produce scores in the range [0, 1] where:
    - 0.0 = definitely normal
    - 0.5 = at threshold (borderline)
    - 1.0 = definitely anomalous

    Default formula: score = min(value / (threshold * cap_multiple), 1.0)
    Override normalize_score() for custom scoring logic (e.g., Isolation Forest).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.anomaly.statistical import AnomalyResult


class BaseDetector(ABC):
    """Abstract base class for anomaly detectors.

    All detectors should inherit from this class to ensure consistent
    score normalization and interface.

    Attributes:
        name: Unique identifier for the detector.

    Usage:
        class MyDetector(BaseDetector):
            def __init__(self):
                self.name = "my_detector"

            def detect(self, features) -> AnomalyResult:
                raw_value = compute_raw_value(features)
                score = self.normalize_score(raw_value, threshold=3.0)
                # ... build and return AnomalyResult
    """

    name: str

    def normalize_score(
        self,
        value: float,
        threshold: float,
        cap_multiple: float = 2.0,
    ) -> float:
        """Normalize raw value to 0-1 anomaly score.

        Default implementation: Linear scaling where threshold maps to 0.5
        and cap_multiple * threshold maps to 1.0.

        Override in subclasses for custom scoring logic (e.g., Isolation Forest
        uses sklearn's decision function which requires different normalization).

        Args:
            value: Raw metric value (e.g., z-score, reconstruction error).
            threshold: Threshold at which anomaly detection triggers.
            cap_multiple: Multiplier for maximum score. Default 2.0 means
                values at 2x threshold get score 1.0.

        Returns:
            Normalized score in range [0, 1].
            - 0.0: Normal
            - 0.5: At threshold
            - 1.0: Severely anomalous (at cap_multiple * threshold or beyond)

        Examples:
            # Z-score detector with threshold=3.0, cap_multiple=2.0:
            normalize_score(3.0, 3.0)  # Returns 0.5 (at threshold)
            normalize_score(6.0, 3.0)  # Returns 1.0 (at 2x threshold)
            normalize_score(1.5, 3.0)  # Returns 0.25 (half threshold)
        """
        if threshold <= 0:
            return 0.5 if value > 0 else 0.0
        return min(value / (threshold * cap_multiple), 1.0)

    @abstractmethod
    def detect(self, *args: Any, **kwargs: Any) -> AnomalyResult:
        """Run detection and return result.

        Subclasses must implement this method with their specific detection logic.
        The signature varies by detector type.

        Returns:
            AnomalyResult with detection outcome.
        """
        ...
