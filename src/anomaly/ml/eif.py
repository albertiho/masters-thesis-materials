"""Extended Isolation Forest implemented locally in-repo."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from src.anomaly.ml.tree_base import BaseTreeDetector, TreeScoreResult
from src.anomaly.ml.tree_features import TreeFeatureVector


def _average_path_length(n_samples: int) -> float:
    """Return the official EIF c(n) approximation used for path normalization."""
    if n_samples <= 1:
        return 0.0
    return 2.0 * (math.log(n_samples - 1) + np.euler_gamma) - (2.0 * (n_samples - 1) / n_samples)


def _resolve_sample_size(n_samples: int, max_samples: str | int) -> int:
    if isinstance(max_samples, str):
        if max_samples != "auto":
            raise ValueError(f"Unsupported max_samples value: {max_samples!r}")
        return min(256, n_samples)
    return min(n_samples, max(1, int(max_samples)))


def _resolve_feature_count(n_features: int, max_features: float | int) -> int:
    if isinstance(max_features, float):
        if not (0.0 < max_features <= 1.0):
            raise ValueError(f"max_features must be in (0, 1], got {max_features}")
        return min(n_features, max(1, int(math.ceil(n_features * max_features))))
    return min(n_features, max(1, int(max_features)))


def _resolve_extension_level(n_features: int, max_features: float | int) -> tuple[int, int]:
    """Map the repo's max_features surface to EIF's extension level."""
    feature_count = _resolve_feature_count(n_features, max_features)
    return feature_count - 1, feature_count


@dataclass
class ExtendedIsolationTreeNode:
    """Node in an extended isolation tree."""

    size: int
    depth: int
    normal_vector: np.ndarray | None = None
    intercept_point: np.ndarray | None = None
    left: "ExtendedIsolationTreeNode | None" = None
    right: "ExtendedIsolationTreeNode | None" = None
    node_type: str = "exNode"
    # Legacy fields are kept so older serialized models can still be scored.
    feature_indices: np.ndarray | None = None
    cut_value: float | None = None

    @property
    def is_leaf(self) -> bool:
        return self.node_type == "exNode" or self.left is None or self.right is None


class ExtendedIsolationTree:
    """Single extended isolation tree using official EIF hyperplane splits."""

    def __init__(
        self,
        *,
        max_depth: int,
        extension_level: int,
        rng: np.random.RandomState,
    ) -> None:
        self.max_depth = max_depth
        self.extension_level = extension_level
        self._rng = rng
        self.root: ExtendedIsolationTreeNode | None = None

    def fit(self, X: np.ndarray) -> "ExtendedIsolationTree":
        self.root = self._build_tree(np.asarray(X, dtype=np.float64), depth=0)
        return self

    def _build_tree(self, X: np.ndarray, depth: int) -> ExtendedIsolationTreeNode:
        node = ExtendedIsolationTreeNode(size=len(X), depth=depth)
        if depth >= self.max_depth or len(X) <= 1:
            return node

        dim = X.shape[1]
        mins = X.min(axis=0)
        maxs = X.max(axis=0)

        zero_count = dim - self.extension_level - 1
        zero_indices: np.ndarray | None = None
        if zero_count > 0:
            zero_indices = np.asarray(self._rng.choice(dim, size=zero_count, replace=False))

        normal_vector = np.asarray(self._rng.normal(size=dim), dtype=np.float64)
        if zero_indices is not None:
            normal_vector[zero_indices] = 0.0
        intercept_point = np.asarray(self._rng.uniform(mins, maxs), dtype=np.float64)

        left_mask = ((X - intercept_point) @ normal_vector) < 0.0
        node.normal_vector = normal_vector
        node.intercept_point = intercept_point
        node.left = self._build_tree(X[left_mask], depth + 1)
        node.right = self._build_tree(X[~left_mask], depth + 1)
        node.node_type = "inNode"
        return node

    def path_length(self, point: np.ndarray) -> float:
        if self.root is None:
            raise RuntimeError("Tree must be fitted before scoring")
        return self._path_length(np.asarray(point, dtype=np.float64), self.root)

    def _path_length(self, point: np.ndarray, node: ExtendedIsolationTreeNode) -> float:
        if node.is_leaf:
            if node.size <= 1:
                return float(node.depth)
            return float(node.depth + _average_path_length(node.size))

        if node.normal_vector is not None and node.intercept_point is not None:
            branch_value = float((point - node.intercept_point) @ node.normal_vector)
            if branch_value < 0.0:
                return self._path_length(point, node.left)
            return self._path_length(point, node.right)

        if node.feature_indices is not None and node.normal_vector is not None and node.cut_value is not None:
            projection = float(point[node.feature_indices] @ node.normal_vector)
            if projection <= node.cut_value:
                return self._path_length(point, node.left)
            return self._path_length(point, node.right)

        if node.size <= 1:
            return float(node.depth)
        return float(node.depth + _average_path_length(node.size))


class ExtendedIsolationForestModel:
    """Minimal extended isolation forest model with official EIF split semantics."""

    def __init__(
        self,
        *,
        n_estimators: int,
        max_samples: str | int,
        max_features: float | int,
        random_state: int,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state
        self.sample_size_: int = 0
        self.feature_count_: int = 0
        self.extension_level_: int = 0
        self.trees: list[ExtendedIsolationTree] = []

    def fit(self, X: np.ndarray) -> "ExtendedIsolationForestModel":
        X = np.asarray(X, dtype=np.float64)
        self.sample_size_ = _resolve_sample_size(len(X), self.max_samples)
        self.extension_level_, self.feature_count_ = _resolve_extension_level(
            X.shape[1],
            self.max_features,
        )
        max_depth = int(math.ceil(math.log2(self.sample_size_))) if self.sample_size_ > 1 else 0
        py_rng = random.Random(self.random_state)
        np_rng = np.random.RandomState(self.random_state)

        self.trees = []
        for _ in range(self.n_estimators):
            if self.sample_size_ < len(X):
                sample_indices = py_rng.sample(range(len(X)), self.sample_size_)
                X_sample = X[np.asarray(sample_indices, dtype=np.int32)]
            else:
                X_sample = X

            tree = ExtendedIsolationTree(
                max_depth=max_depth,
                extension_level=self.extension_level_,
                rng=np_rng,
            )
            self.trees.append(tree.fit(X_sample))
        return self

    def decision_function(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.trees:
            raise RuntimeError("Model must be fitted before scoring")

        X = np.asarray(X, dtype=np.float64)
        path_lengths = np.zeros((len(self.trees), len(X)), dtype=np.float64)
        for tree_index, tree in enumerate(self.trees):
            path_lengths[tree_index] = np.array(
                [tree.path_length(point) for point in X],
                dtype=np.float64,
            )

        avg_path_lengths = path_lengths.mean(axis=0)
        normalizer = _average_path_length(self.sample_size_) or 1.0
        anomaly_metric = np.power(2.0, -avg_path_lengths / normalizer)
        # The shared tree base expects lower decision values for more anomalous rows.
        decision_scores = 1.0 - anomaly_metric
        return (
            decision_scores.astype(np.float64),
            avg_path_lengths.astype(np.float64),
            anomaly_metric.astype(np.float64),
        )


@dataclass
class EIFConfig:
    """Configuration for the local EIF detector."""

    n_estimators: int = 100
    max_samples: str | int = "auto"
    max_features: float | int = 1.0
    random_state: int = 42
    anomaly_threshold: float = 0.6


FeatureVector = TreeFeatureVector


class EIFDetector(BaseTreeDetector[EIFConfig]):
    """Extended Isolation Forest with IF-compatible external behavior."""

    config_cls = EIFConfig

    def __init__(self, config: EIFConfig | None = None):
        super().__init__(config or EIFConfig())
        self.name = "eif"

    def _create_model(self) -> ExtendedIsolationForestModel:
        return ExtendedIsolationForestModel(
            n_estimators=self.config.n_estimators,
            max_samples=self.config.max_samples,
            max_features=self.config.max_features,
            random_state=self.config.random_state,
        )

    def _fit_matrix(self, X: np.ndarray) -> np.ndarray:
        self._model = self._create_model().fit(X)
        decision_scores, _, _ = self._model.decision_function(X)
        return decision_scores

    def _score_matrix(self, X: np.ndarray) -> list[TreeScoreResult]:
        decision_scores, average_path_lengths, anomaly_metrics = self._model.decision_function(X)
        return [
            TreeScoreResult(
                decision_score=float(decision_score),
                details={
                    "eif_score": float(anomaly_metric),
                    "average_path_length": float(path_length),
                    "path_anomaly_metric": float(anomaly_metric),
                },
            )
            for decision_score, path_length, anomaly_metric in zip(
                decision_scores,
                average_path_lengths,
                anomaly_metrics,
                strict=True,
            )
        ]

    def get_model_info(self) -> dict[str, object]:
        info = super().get_model_info()
        if not self._is_fitted:
            return info

        info.update(
            {
                "n_estimators": self.config.n_estimators,
                "max_samples": self.config.max_samples,
                "max_features": self.config.max_features,
                "extension_level": getattr(self._model, "extension_level_", None),
                "n_features": len(self._feature_names),
            }
        )
        return info
