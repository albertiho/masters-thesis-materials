"""Shared base implementation for tree-based anomaly detectors."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Generic, TypeVar

import numpy as np

from src.anomaly.base import BaseDetector
from src.anomaly.statistical import AnomalyResult, AnomalySeverity, AnomalyType
from src.features.numeric import NumericFeatures
from src.features.temporal import TemporalFeatures

from .tree_features import (
    TREE_FEATURE_NAMES,
    TreeFeatureVector,
    prepare_tree_feature_vector,
)

logger = logging.getLogger(__name__)

ConfigT = TypeVar("ConfigT")


@dataclass
class TreeScoreResult:
    """Detector-specific score output used by the shared tree base."""

    decision_score: float
    details: dict[str, Any] = field(default_factory=dict)


class BaseTreeDetector(BaseDetector, ABC, Generic[ConfigT]):
    """Shared fit/detect/save/load logic for tree detectors."""

    config_cls: type[ConfigT]
    minimum_fit_samples: int = 10
    legacy_score_offset: float = 1.0
    legacy_score_scale: float = 2.0

    def __init__(self, config: ConfigT):
        self.config = config
        self._model: Any = None
        self._is_fitted = False
        self._feature_names: list[str] = []
        self._score_offset: float = 0.0
        self._score_scale: float = 1.0

    def prepare_features(
        self,
        numeric_features: NumericFeatures,
        temporal_features: TemporalFeatures,
    ) -> TreeFeatureVector:
        """Prepare the shared tree feature vector for inference."""
        return prepare_tree_feature_vector(numeric_features, temporal_features)

    def fit(
        self,
        feature_vectors: list[TreeFeatureVector],
    ) -> "BaseTreeDetector[ConfigT]":
        """Fit the detector from precomputed feature vectors."""
        if not feature_vectors:
            raise ValueError("Cannot fit on empty feature vectors")

        valid_vectors = [feature_vector for feature_vector in feature_vectors if feature_vector.is_valid]
        if len(valid_vectors) < self.minimum_fit_samples:
            raise ValueError(
                f"Need at least {self.minimum_fit_samples} valid feature vectors, "
                f"got {len(valid_vectors)}"
            )

        feature_names = valid_vectors[0].feature_names
        X = np.vstack([feature_vector.features for feature_vector in valid_vectors])
        return self.fit_from_matrix(X, feature_names=feature_names)

    def fit_from_matrix(
        self,
        X: np.ndarray,
        *,
        feature_names: list[str] | None = None,
    ) -> "BaseTreeDetector[ConfigT]":
        """Fit the detector directly from a feature matrix."""
        if X.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix, got shape {X.shape}")
        if len(X) < self.minimum_fit_samples:
            raise ValueError(
                f"Need at least {self.minimum_fit_samples} training samples, got {len(X)}"
            )

        train_scores = np.asarray(self._fit_matrix(X), dtype=np.float64)
        if train_scores.shape != (len(X),):
            raise ValueError(
                f"{self.name} returned invalid training scores shape {train_scores.shape}"
            )

        self._feature_names = list(feature_names or TREE_FEATURE_NAMES)
        self._is_fitted = True
        self._score_offset = float(train_scores.max())
        score_range = float(train_scores.max() - train_scores.min())
        self._score_scale = score_range if score_range > 0 else 1.0

        logger.info(
            "%s_fitted",
            self.name,
            extra={
                "detector": self.name,
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "score_offset": self._score_offset,
                "score_scale": self._score_scale,
            },
        )
        return self

    def normalize_score(
        self,
        value: float,
        threshold: float = 0,
        cap_multiple: float = 2.0,
    ) -> float:
        """Min-max normalize detector decision values using training statistics."""
        del threshold, cap_multiple
        if self._score_scale > 0:
            normalized = (self._score_offset - value) / self._score_scale
        else:
            normalized = 0.5
        return max(0.0, min(1.0, float(normalized)))

    def detect(
        self,
        numeric_features: NumericFeatures,
        temporal_features: TemporalFeatures,
    ) -> AnomalyResult:
        """Score one record and return the shared anomaly result shape."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before detection. Call fit() first.")

        feature_vector = self.prepare_features(numeric_features, temporal_features)
        self._validate_feature_schema(feature_vector.feature_names)

        score_result = self._score_matrix(feature_vector.features.reshape(1, -1))[0]
        return self._build_result(
            numeric_features=numeric_features,
            temporal_features=temporal_features,
            feature_vector=feature_vector,
            score_result=score_result,
        )

    def detect_batch(
        self,
        numeric_features_list: list[NumericFeatures],
        temporal_features_list: list[TemporalFeatures],
    ) -> list[AnomalyResult]:
        """Score a batch of records."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before detection. Call fit() first.")
        if len(numeric_features_list) != len(temporal_features_list):
            raise ValueError("Feature lists must have same length")

        feature_vectors = [
            self.prepare_features(numeric_features, temporal_features)
            for numeric_features, temporal_features in zip(
                numeric_features_list,
                temporal_features_list,
                strict=True,
            )
        ]
        if not feature_vectors:
            return []
        if feature_vectors:
            self._validate_feature_schema(feature_vectors[0].feature_names)

        X = np.vstack([feature_vector.features for feature_vector in feature_vectors])
        score_results = self._score_matrix(X)

        return [
            self._build_result(
                numeric_features=numeric_features,
                temporal_features=temporal_features,
                feature_vector=feature_vector,
                score_result=score_result,
            )
            for numeric_features, temporal_features, feature_vector, score_result in zip(
                numeric_features_list,
                temporal_features_list,
                feature_vectors,
                score_results,
                strict=True,
            )
        ]

    def get_model_info(self) -> dict[str, Any]:
        """Return a shared model-info structure."""
        if not self._is_fitted:
            return {"is_fitted": False}

        return {
            "is_fitted": True,
            "feature_names": list(self._feature_names),
            "anomaly_threshold": getattr(self.config, "anomaly_threshold", None),
            "score_offset": self._score_offset,
            "score_scale": self._score_scale,
            "config": self._config_to_dict(),
        }

    def save(self, path: str) -> str:
        """Persist the trained detector to a local joblib file."""
        from joblib import dump as joblib_dump

        if not self._is_fitted:
            raise ValueError("Cannot save unfitted model. Call fit() first.")

        payload = {
            "model": self._model,
            "feature_names": self._feature_names,
            "config": self._config_to_dict(),
            "score_offset": self._score_offset,
            "score_scale": self._score_scale,
            **self._serialize_extra_state(),
        }
        joblib_dump(payload, path)
        logger.info(
            "%s_saved_local",
            self.name,
            extra={"detector": self.name, "path": path, "n_features": len(self._feature_names)},
        )
        return path

    @classmethod
    def load(cls, path: str) -> "BaseTreeDetector[ConfigT]":
        """Load a trained detector from a local joblib file."""
        from joblib import load as joblib_load

        payload = joblib_load(path)
        config = cls.config_cls(**payload["config"])
        detector = cls(config)
        detector._model = payload["model"]
        detector._feature_names = list(payload["feature_names"])
        detector._is_fitted = True
        detector._score_offset = payload.get("score_offset", cls.legacy_score_offset)
        detector._score_scale = payload.get("score_scale", cls.legacy_score_scale)
        detector._restore_extra_state(payload)

        logger.info(
            "%s_loaded_local",
            detector.name,
            extra={
                "detector": detector.name,
                "path": path,
                "n_features": len(detector._feature_names),
                "score_offset": detector._score_offset,
                "score_scale": detector._score_scale,
            },
        )
        return detector

    def _build_result(
        self,
        *,
        numeric_features: NumericFeatures,
        temporal_features: TemporalFeatures,
        feature_vector: TreeFeatureVector,
        score_result: TreeScoreResult,
    ) -> AnomalyResult:
        anomaly_score = self.normalize_score(score_result.decision_score)
        is_anomaly = self._decide_is_anomaly(
            anomaly_score=anomaly_score,
            numeric_features=numeric_features,
            temporal_features=temporal_features,
            score_result=score_result,
        )
        anomaly_types = self._infer_anomaly_types(temporal_features) if is_anomaly else []
        severity = self._infer_severity(anomaly_score) if is_anomaly else None

        details: dict[str, Any] = {
            "feature_vector_valid": feature_vector.is_valid,
            "missing_features": list(feature_vector.missing_features),
            "anomaly_score": anomaly_score,
            "threshold": getattr(self.config, "anomaly_threshold", None),
            **score_result.details,
        }
        if not feature_vector.is_valid:
            details["warning"] = "Feature vector contains missing values"

        self._augment_details(
            details=details,
            numeric_features=numeric_features,
            temporal_features=temporal_features,
            feature_vector=feature_vector,
            score_result=score_result,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
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

    def _validate_feature_schema(self, feature_names: list[str]) -> None:
        if not self._feature_names:
            return
        from src.anomaly.ml import validate_feature_schema

        validate_feature_schema(feature_names, self._feature_names, type(self).__name__)

    def _config_to_dict(self) -> dict[str, Any]:
        if is_dataclass(self.config):
            return asdict(self.config)
        if isinstance(self.config, dict):
            return dict(self.config)
        raise TypeError(f"Unsupported detector config type for {self.name}: {type(self.config)!r}")

    def _infer_anomaly_types(self, temporal_features: TemporalFeatures) -> list[AnomalyType]:
        if (
            temporal_features.price_zscore is not None
            and abs(temporal_features.price_zscore) > 2
        ):
            return [AnomalyType.PRICE_ZSCORE]
        if (
            temporal_features.price_change_pct is not None
            and abs(temporal_features.price_change_pct) > 0.2
        ):
            return [AnomalyType.PRICE_CHANGE]
        return [AnomalyType.PRICE_ZSCORE]

    def _infer_severity(self, anomaly_score: float) -> AnomalySeverity:
        if anomaly_score >= 0.9:
            return AnomalySeverity.CRITICAL
        if anomaly_score >= 0.8:
            return AnomalySeverity.HIGH
        if anomaly_score >= 0.7:
            return AnomalySeverity.MEDIUM
        return AnomalySeverity.LOW

    def _decide_is_anomaly(
        self,
        *,
        anomaly_score: float,
        numeric_features: NumericFeatures,
        temporal_features: TemporalFeatures,
        score_result: TreeScoreResult,
    ) -> bool:
        del numeric_features, temporal_features, score_result
        return anomaly_score >= getattr(self.config, "anomaly_threshold", 0.5)

    def _augment_details(
        self,
        *,
        details: dict[str, Any],
        numeric_features: NumericFeatures,
        temporal_features: TemporalFeatures,
        feature_vector: TreeFeatureVector,
        score_result: TreeScoreResult,
        anomaly_score: float,
        is_anomaly: bool,
    ) -> None:
        del details, numeric_features, temporal_features, feature_vector, score_result, anomaly_score, is_anomaly

    def _serialize_extra_state(self) -> dict[str, Any]:
        return {}

    def _restore_extra_state(self, payload: dict[str, Any]) -> None:
        del payload

    @abstractmethod
    def _fit_matrix(self, X: np.ndarray) -> np.ndarray:
        """Fit the detector on X and return training decision scores."""

    @abstractmethod
    def _score_matrix(self, X: np.ndarray) -> list[TreeScoreResult]:
        """Score X and return one TreeScoreResult per row."""
