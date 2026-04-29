from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from src.anomaly.ml.eif import EIFConfig, EIFDetector


def _reference_average_path_length(n_samples: int) -> float:
    if n_samples <= 1:
        return 0.0
    return 2.0 * (math.log(n_samples - 1) + np.euler_gamma) - (2.0 * (n_samples - 1) / n_samples)


def _reference_resolve_sample_size(n_samples: int, max_samples: str | int) -> int:
    if isinstance(max_samples, str):
        if max_samples != "auto":
            raise ValueError(f"Unsupported max_samples value: {max_samples!r}")
        return min(256, n_samples)
    return min(n_samples, max(1, int(max_samples)))


def _reference_resolve_feature_count(n_features: int, max_features: float | int) -> int:
    if isinstance(max_features, float):
        if not (0.0 < max_features <= 1.0):
            raise ValueError(f"max_features must be in (0, 1], got {max_features}")
        return min(n_features, max(1, int(math.ceil(n_features * max_features))))
    return min(n_features, max(1, int(max_features)))


@dataclass
class _ReferenceNode:
    size: int
    depth: int
    normal_vector: np.ndarray | None = None
    intercept_point: np.ndarray | None = None
    left: "_ReferenceNode | None" = None
    right: "_ReferenceNode | None" = None
    node_type: str = "exNode"


def _reference_fit_and_score(
    X: np.ndarray,
    *,
    n_estimators: int,
    max_samples: str | int,
    max_features: float | int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Small reference EIF derived from the official sahandha/eif Python code."""

    sample_size = _reference_resolve_sample_size(len(X), max_samples)
    feature_count = _reference_resolve_feature_count(X.shape[1], max_features)
    extension_level = feature_count - 1
    depth_limit = int(math.ceil(math.log2(sample_size))) if sample_size > 1 else 0
    py_rng = random.Random(random_state)
    np_rng = np.random.RandomState(random_state)

    def build_tree(X_subset: np.ndarray, depth: int) -> _ReferenceNode:
        node = _ReferenceNode(size=len(X_subset), depth=depth)
        if depth >= depth_limit or len(X_subset) <= 1:
            return node

        dim = X_subset.shape[1]
        mins = X_subset.min(axis=0)
        maxs = X_subset.max(axis=0)

        zero_count = dim - extension_level - 1
        zero_indices: np.ndarray | None = None
        if zero_count > 0:
            zero_indices = np.asarray(np_rng.choice(dim, size=zero_count, replace=False))

        normal_vector = np.asarray(np_rng.normal(size=dim), dtype=np.float64)
        if zero_indices is not None:
            normal_vector[zero_indices] = 0.0
        intercept_point = np.asarray(np_rng.uniform(mins, maxs), dtype=np.float64)

        left_mask = ((X_subset - intercept_point) @ normal_vector) < 0.0
        node.normal_vector = normal_vector
        node.intercept_point = intercept_point
        node.left = build_tree(X_subset[left_mask], depth + 1)
        node.right = build_tree(X_subset[~left_mask], depth + 1)
        node.node_type = "inNode"
        return node

    def path_length(point: np.ndarray, node: _ReferenceNode) -> float:
        if node.node_type == "exNode":
            if node.size <= 1:
                return float(node.depth)
            return float(node.depth + _reference_average_path_length(node.size))

        branch_value = float((point - node.intercept_point) @ node.normal_vector)
        if branch_value < 0.0:
            return path_length(point, node.left)
        return path_length(point, node.right)

    trees: list[_ReferenceNode] = []
    for _ in range(n_estimators):
        if sample_size < len(X):
            sample_indices = py_rng.sample(range(len(X)), sample_size)
            X_subset = X[np.asarray(sample_indices, dtype=np.int32)]
        else:
            X_subset = X
        trees.append(build_tree(X_subset, depth=0))

    path_lengths = np.zeros((n_estimators, len(X)), dtype=np.float64)
    for tree_index, tree in enumerate(trees):
        path_lengths[tree_index] = np.array(
            [path_length(point, tree) for point in X],
            dtype=np.float64,
        )

    avg_path_lengths = path_lengths.mean(axis=0)
    anomaly_scores = np.power(
        2.0,
        -avg_path_lengths / (_reference_average_path_length(sample_size) or 1.0),
    )
    return avg_path_lengths, anomaly_scores


def test_eif_matches_official_reference_scores_on_seeded_matrix() -> None:
    X = np.random.default_rng(42).normal(size=(96, 12))
    config = EIFConfig(
        n_estimators=12,
        max_samples=32,
        max_features=0.75,
        random_state=42,
        anomaly_threshold=0.4,
    )

    detector = EIFDetector(config)
    detector.fit_from_matrix(X)
    _, actual_path_lengths, actual_scores = detector._model.decision_function(X)

    expected_path_lengths, expected_scores = _reference_fit_and_score(
        X,
        n_estimators=config.n_estimators,
        max_samples=config.max_samples,
        max_features=config.max_features,
        random_state=config.random_state,
    )

    np.testing.assert_allclose(actual_path_lengths, expected_path_lengths, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(actual_scores, expected_scores, rtol=0.0, atol=1e-12)


def test_eif_score_details_follow_official_anomaly_direction() -> None:
    X = np.random.default_rng(7).normal(size=(64, 10))
    detector = EIFDetector(
        EIFConfig(
            n_estimators=8,
            max_samples=32,
            max_features=0.5,
            random_state=7,
        )
    )
    detector.fit_from_matrix(X)

    _, _, raw_scores = detector._model.decision_function(X[:5])
    score_results = detector._score_matrix(X[:5])

    np.testing.assert_allclose(
        [result.details["eif_score"] for result in score_results],
        raw_scores,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        [result.details["path_anomaly_metric"] for result in score_results],
        raw_scores,
        rtol=0.0,
        atol=1e-12,
    )
