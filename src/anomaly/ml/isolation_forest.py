"""Isolation Forest anomaly detection built on the shared tree detector base."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.anomaly.ml.tree_base import BaseTreeDetector, TreeScoreResult
from src.anomaly.ml.tree_features import TreeFeatureVector

logger = logging.getLogger(__name__)

_sklearn_available = None


def _check_sklearn() -> bool:
    """Check if scikit-learn is available."""
    global _sklearn_available
    if _sklearn_available is None:
        try:
            from sklearn.ensemble import IsolationForest  # noqa: F401

            _sklearn_available = True
        except ImportError:
            _sklearn_available = False
    return bool(_sklearn_available)


@dataclass
class IsolationForestConfig:
    """Configuration for the Isolation Forest detector."""

    n_estimators: int = 100
    max_samples: str | int = "auto"
    contamination: str | float = "auto"
    max_features: float = 1.0
    random_state: int = 42
    anomaly_threshold: float = 0.6


FeatureVector = TreeFeatureVector


class IsolationForestDetector(BaseTreeDetector[IsolationForestConfig]):
    """Isolation Forest based anomaly detector."""

    config_cls = IsolationForestConfig

    def __init__(self, config: IsolationForestConfig | None = None):
        if not _check_sklearn():
            raise ImportError(
                "scikit-learn is required for IsolationForestDetector. "
                "Install with: pip install scikit-learn"
            )

        super().__init__(config or IsolationForestConfig())
        self.name = "isolation_forest"

    def _create_model(self):
        from sklearn.ensemble import IsolationForest

        return IsolationForest(
            n_estimators=self.config.n_estimators,
            max_samples=self.config.max_samples,
            contamination=self.config.contamination,
            max_features=self.config.max_features,
            random_state=self.config.random_state,
            n_jobs=-1,
        )

    def _fit_matrix(self, X: np.ndarray) -> np.ndarray:
        self._model = self._create_model()
        self._model.fit(X)
        return np.asarray(self._model.decision_function(X), dtype=np.float64)

    def _score_matrix(self, X: np.ndarray) -> list[TreeScoreResult]:
        scores = np.asarray(self._model.decision_function(X), dtype=np.float64)
        predictions = np.asarray(self._model.predict(X), dtype=np.int32)
        return [
            TreeScoreResult(
                decision_score=float(score),
                details={
                    "sklearn_score": float(score),
                    "sklearn_prediction": int(prediction),
                },
            )
            for score, prediction in zip(scores, predictions, strict=True)
        ]

    def get_model_info(self) -> dict[str, object]:
        info = super().get_model_info()
        if not self._is_fitted:
            return info

        info.update(
            {
                "n_estimators": self.config.n_estimators,
                "contamination": self.config.contamination,
                "n_features": len(self._feature_names),
                "feature_names": list(self._feature_names),
            }
        )
        return info
