"""Robust Random Cut Forest core plus a DetectorEvaluator-compatible wrapper."""

from __future__ import annotations

from collections import deque
import logging
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.anomaly.ml.tree_base import BaseTreeDetector, TreeScoreResult
from src.anomaly.ml.tree_features import TreeFeatureVector
from src.features.numeric import NumericFeatures
from src.features.temporal import TemporalFeatures

logger = logging.getLogger(__name__)


@dataclass
class RRCFResult:
    """Result of low-level RRCF anomaly detection."""

    is_anomaly: bool
    anomaly_score: float
    raw_score: float
    num_trees: int
    tree_depth_avg: float
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_anomaly": self.is_anomaly,
            "anomaly_score": self.anomaly_score,
            "raw_score": self.raw_score,
            "num_trees": self.num_trees,
            "tree_depth_avg": self.tree_depth_avg,
            **self.details,
        }


@dataclass
class RCTreeNode:
    """Node in a random cut tree."""

    point: np.ndarray | None = None
    index: int | None = None
    parent: "RCTreeNode | None" = None
    left: "RCTreeNode | None" = None
    right: "RCTreeNode | None" = None
    cut_dimension: int | None = None
    cut_value: float | None = None
    n: int = 1
    bbox_min: np.ndarray | None = None
    bbox_max: np.ndarray | None = None

    @property
    def is_leaf(self) -> bool:
        return self.point is not None


class RCTree:
    """Random cut tree for anomaly scoring."""

    def __init__(self, random_state: int | None = None):
        self.root: RCTreeNode | None = None
        self.points: dict[int, np.ndarray] = {}
        self.leaves: dict[int, RCTreeNode] = {}
        self._point_counter = 0
        self._rng = random.Random(random_state)

    def insert(self, point: np.ndarray, index: int | None = None) -> int:
        if index is None:
            index = self._point_counter
            self._point_counter += 1

        point = np.asarray(point, dtype=np.float64)
        self.points[index] = point

        if self.root is None:
            leaf = RCTreeNode(
                point=point.copy(),
                index=index,
                n=1,
                bbox_min=point.copy(),
                bbox_max=point.copy(),
            )
            self.root = leaf
            self.leaves[index] = leaf
            return index

        self.root, inserted_leaf = self._insert_recursive(self.root, point, index)
        self.root.parent = None
        self.leaves[index] = inserted_leaf
        return index

    def rebuild_structure(self) -> None:
        """Rebuild parent pointers and leaf indexes after deserialization."""
        self.points = {}
        self.leaves = {}

        if self.root is None:
            self._point_counter = 0
            return

        def walk(node: RCTreeNode, parent: RCTreeNode | None) -> None:
            node.parent = parent
            if node.is_leaf:
                assert node.point is not None
                assert node.index is not None
                node.n = 1
                self._update_bbox(node)
                self.points[node.index] = np.asarray(node.point, dtype=np.float64)
                self.leaves[node.index] = node
                return

            assert node.left is not None
            assert node.right is not None
            walk(node.left, node)
            walk(node.right, node)
            self._refresh_node(node)

        walk(self.root, None)
        self._point_counter = max(self.points, default=-1) + 1

    def _insert_recursive(
        self,
        node: RCTreeNode,
        point: np.ndarray,
        index: int,
    ) -> tuple[RCTreeNode, RCTreeNode]:
        if node.is_leaf:
            assert node.point is not None
            assert node.index is not None

            existing_index = node.index
            existing_point = node.point.copy()

            existing_leaf = RCTreeNode(
                point=existing_point,
                index=existing_index,
                n=1,
                bbox_min=existing_point.copy(),
                bbox_max=existing_point.copy(),
            )
            new_leaf = RCTreeNode(
                point=point.copy(),
                index=index,
                n=1,
                bbox_min=point.copy(),
                bbox_max=point.copy(),
            )
            self.leaves[existing_index] = existing_leaf

            cut_dimension, cut_value = self._random_cut(existing_point, point)
            if point[cut_dimension] <= cut_value:
                left_child = new_leaf
                right_child = existing_leaf
            else:
                left_child = existing_leaf
                right_child = new_leaf

            internal = RCTreeNode(
                point=None,
                index=None,
                left=left_child,
                right=right_child,
                cut_dimension=cut_dimension,
                cut_value=cut_value,
                n=2,
            )
            left_child.parent = internal
            right_child.parent = internal
            self._update_bbox(internal)
            return internal, new_leaf

        assert node.left is not None
        assert node.right is not None
        assert node.cut_dimension is not None
        assert node.cut_value is not None

        if point[node.cut_dimension] <= node.cut_value:
            node.left, inserted_leaf = self._insert_recursive(node.left, point, index)
            node.left.parent = node
        else:
            node.right, inserted_leaf = self._insert_recursive(node.right, point, index)
            node.right.parent = node

        self._refresh_node(node)
        return node, inserted_leaf

    def _random_cut(self, point1: np.ndarray, point2: np.ndarray) -> tuple[int, float]:
        diff = np.abs(point1 - point2)
        total_diff = float(np.sum(diff))

        if total_diff < 1e-10:
            cut_dimension = self._rng.randint(0, len(point1) - 1)
            cut_value = float(point1[cut_dimension])
            return cut_dimension, cut_value

        probabilities = diff / total_diff
        cut_dimension = self._rng.choices(range(len(point1)), weights=probabilities)[0]
        min_value = min(point1[cut_dimension], point2[cut_dimension])
        max_value = max(point1[cut_dimension], point2[cut_dimension])
        cut_value = self._rng.uniform(float(min_value), float(max_value))
        return cut_dimension, cut_value

    def _update_bbox(self, node: RCTreeNode) -> None:
        if node.is_leaf:
            assert node.point is not None
            node.bbox_min = node.point.copy()
            node.bbox_max = node.point.copy()
            return

        assert node.left is not None
        assert node.right is not None
        if node.left.bbox_min is None or node.right.bbox_min is None:
            return

        node.bbox_min = np.minimum(node.left.bbox_min, node.right.bbox_min)
        node.bbox_max = np.maximum(node.left.bbox_max, node.right.bbox_max)

    def _refresh_node(self, node: RCTreeNode) -> None:
        if node.is_leaf:
            node.n = 1
            self._update_bbox(node)
            return

        assert node.left is not None
        assert node.right is not None
        node.n = node.left.n + node.right.n
        self._update_bbox(node)

    def _refresh_upwards(self, node: RCTreeNode | None) -> None:
        current = node
        while current is not None:
            self._refresh_node(current)
            current = current.parent

    def delete(self, index: int) -> bool:
        leaf = self.leaves.pop(index, None)
        if leaf is None:
            return False

        self.points.pop(index, None)
        if leaf is self.root:
            self.root = None
            return True

        parent = leaf.parent
        if parent is None:
            self.root = None
            return True

        sibling = parent.right if parent.left is leaf else parent.left
        assert sibling is not None

        grandparent = parent.parent
        sibling.parent = grandparent

        if grandparent is None:
            self.root = sibling
        elif grandparent.left is parent:
            grandparent.left = sibling
        elif grandparent.right is parent:
            grandparent.right = sibling
        else:
            raise RuntimeError("Tree structure corrupted during deletion")

        parent.left = None
        parent.right = None
        parent.parent = None
        leaf.parent = None

        if self.root is not None:
            self.root.parent = None
        self._refresh_upwards(grandparent)
        return True

    def codisp_with_depth(self, index: int) -> tuple[float, int]:
        leaf = self.leaves.get(index)
        if leaf is None or self.root is None:
            return 0.0, 0

        depth = 1
        current = leaf
        while current.parent is not None:
            depth += 1
            current = current.parent

        total = 0.0
        current = leaf
        ancestor_depth = depth - 1
        while current.parent is not None:
            parent = current.parent
            sibling = parent.right if parent.left is current else parent.left
            if sibling is not None and ancestor_depth > 0:
                total += sibling.n / ancestor_depth
            current = parent
            ancestor_depth -= 1

        return float(total), depth

    def codisp(self, index: int) -> float:
        codisp, _ = self.codisp_with_depth(index)
        return codisp

    def depth(self, index: int) -> int:
        _, depth = self.codisp_with_depth(index)
        return depth


class RRCF:
    """Low-level Robust Random Cut Forest for streaming anomaly detection."""

    def __init__(
        self,
        num_trees: int = 40,
        tree_size: int = 256,
        anomaly_threshold: float = 0.8,
        random_state: int | None = None,
    ) -> None:
        self.num_trees = num_trees
        self.tree_size = tree_size
        self.anomaly_threshold = anomaly_threshold
        self.name = "rrcf"

        self._rng = random.Random(random_state)
        self.trees = [RCTree(random_state=self._rng.randint(0, 2**31 - 1)) for _ in range(num_trees)]
        self._next_index = 0
        self._index_to_points: dict[int, np.ndarray] = {}
        self._oldest_indices: deque[int] = deque()

    def clear(self) -> None:
        self._next_index = 0
        self._index_to_points = {}
        self._oldest_indices = deque()
        self.trees = [RCTree(random_state=self._rng.randint(0, 2**31 - 1)) for _ in range(self.num_trees)]

    def rebuild_runtime_state(self) -> None:
        """Restore runtime-only structures after loading an older saved model."""
        for tree in self.trees:
            tree.rebuild_structure()

        if not isinstance(self._oldest_indices, deque):
            self._oldest_indices = deque(self._oldest_indices)

        rebuilt_points: dict[int, np.ndarray] = {}
        for tree in self.trees:
            rebuilt_points.update(tree.points)

        if rebuilt_points:
            self._index_to_points = rebuilt_points
            active_indices = set(rebuilt_points)
            self._oldest_indices = deque(
                index for index in self._oldest_indices if index in active_indices
            )
            if not self._oldest_indices:
                self._oldest_indices = deque(sorted(active_indices))

        self._next_index = max(self._index_to_points, default=-1) + 1

    def insert(self, point: np.ndarray | list[float]) -> int:
        point_array = np.asarray(point, dtype=np.float64)
        index = self._next_index
        self._next_index += 1

        self._index_to_points[index] = point_array
        self._oldest_indices.append(index)

        for tree in self.trees:
            tree.insert(point_array, index)

        while len(self._oldest_indices) > self.tree_size:
            oldest_index = self._oldest_indices.popleft()
            for tree in self.trees:
                tree.delete(oldest_index)
            self._index_to_points.pop(oldest_index, None)

        return index

    def _score_index(self, index: int, *, include_depth: bool = False) -> tuple[float, float]:
        if index not in self._index_to_points:
            return 0.0, 0.0

        total_codisp = 0.0
        total_depth = 0.0
        active_depths = 0

        for tree in self.trees:
            codisp, depth = tree.codisp_with_depth(index)
            total_codisp += codisp
            if include_depth and depth > 0:
                total_depth += depth
                active_depths += 1

        raw_score = total_codisp / max(len(self.trees), 1)
        average_depth = total_depth / active_depths if include_depth and active_depths else 0.0
        return float(raw_score), float(average_depth)

    def insert_and_score(
        self,
        point: np.ndarray | list[float],
        *,
        include_depth: bool = False,
    ) -> tuple[int, float, float, float]:
        index = self.insert(point)
        raw_score, average_depth = self._score_index(index, include_depth=include_depth)
        normalized_score = self.normalize_raw_score(raw_score)
        return index, raw_score, normalized_score, average_depth

    def score(self, index: int) -> float:
        raw_score, _ = self._score_index(index, include_depth=False)
        return raw_score

    def normalize_raw_score(self, raw_score: float) -> float:
        normalized = 1.0 - np.exp(-raw_score / max(self.tree_size / 4.0, 1.0))
        return float(min(max(normalized, 0.0), 1.0))

    def is_anomaly(self, index: int, threshold: float | None = None) -> bool:
        score = self.normalize_raw_score(self.score(index))
        return score > (threshold if threshold is not None else self.anomaly_threshold)

    def detect(self, point: np.ndarray | list[float]) -> RRCFResult:
        index, raw_score, normalized_score, average_depth = self.insert_and_score(
            point,
            include_depth=True,
        )
        details = {
            "index": index,
            "tree_size_current": len(self._oldest_indices),
            "tree_size_max": self.tree_size,
            "threshold": self.anomaly_threshold,
        }
        return RRCFResult(
            is_anomaly=normalized_score > self.anomaly_threshold,
            anomaly_score=normalized_score,
            raw_score=float(raw_score),
            num_trees=self.num_trees,
            tree_depth_avg=average_depth,
            details=details,
        )

    def fit(self, data: np.ndarray | list[list[float]]) -> "RRCF":
        self.clear()
        for point in np.asarray(data, dtype=np.float64):
            self.insert(point)
        return self

    def predict(self, data: np.ndarray | list[list[float]]) -> np.ndarray:
        results = [self.detect(point).anomaly_score for point in np.asarray(data, dtype=np.float64)]
        return np.asarray(results, dtype=np.float64)


@dataclass
class RRCFDetectorConfig:
    """Configuration for the research RRCF detector wrapper."""

    num_trees: int = 40
    tree_size: int = 256
    anomaly_threshold: float = 0.8
    random_state: int = 42
    warmup_samples: int = 32
    min_history_required: int = 3


@dataclass
class RRCFStreamState:
    """Mutable persisted state for the RRCF detector wrapper."""

    forest: RRCF
    observations_seen: int = 0


FeatureVector = TreeFeatureVector


class RRCFDetector(BaseTreeDetector[RRCFDetectorConfig]):
    """RRCF wrapper with the shared tree detector contract."""

    config_cls = RRCFDetectorConfig
    minimum_fit_samples = 5

    def __init__(self, config: RRCFDetectorConfig | None = None):
        super().__init__(config or RRCFDetectorConfig())
        self.name = "rrcf"

    def _create_state(self) -> RRCFStreamState:
        return RRCFStreamState(
            forest=RRCF(
                num_trees=self.config.num_trees,
                tree_size=self.config.tree_size,
                anomaly_threshold=self.config.anomaly_threshold,
                random_state=self.config.random_state,
            ),
            observations_seen=0,
        )

    def _fit_matrix(self, X: np.ndarray) -> np.ndarray:
        state = self._create_state()
        training_scores: list[float] = []
        for point in X:
            _, raw_score, _, _ = state.forest.insert_and_score(point, include_depth=False)
            state.observations_seen += 1
            training_scores.append(-raw_score)
        self._model = state
        return np.asarray(training_scores, dtype=np.float64)

    def _score_matrix(self, X: np.ndarray) -> list[TreeScoreResult]:
        if self._model is None:
            raise RuntimeError("Model must be fitted before scoring")

        state: RRCFStreamState = self._model
        results: list[TreeScoreResult] = []
        for point in X:
            _, raw_score, normalized_score, average_depth = state.forest.insert_and_score(
                point,
                include_depth=True,
            )
            state.observations_seen += 1
            warmup_complete = state.observations_seen >= self.config.warmup_samples
            results.append(
                TreeScoreResult(
                    decision_score=-float(raw_score),
                    details={
                        "raw_score": float(raw_score),
                        "rrcf_stream_score": float(normalized_score),
                        "num_trees": self.config.num_trees,
                        "tree_depth_avg": float(average_depth),
                        "forest_observations_seen": state.observations_seen,
                        "warmup_samples": self.config.warmup_samples,
                        "warmup_complete": warmup_complete,
                    },
                )
            )
        return results

    def _decide_is_anomaly(
        self,
        *,
        anomaly_score: float,
        numeric_features: NumericFeatures,
        temporal_features: TemporalFeatures,
        score_result: TreeScoreResult,
    ) -> bool:
        del numeric_features
        warmup_complete = bool(score_result.details.get("warmup_complete", False))
        temporal_history_ready = temporal_features.observation_count >= self.config.min_history_required
        return warmup_complete and temporal_history_ready and anomaly_score >= self.config.anomaly_threshold

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
        del numeric_features, feature_vector, score_result, anomaly_score, is_anomaly
        warmup_complete = bool(details.get("warmup_complete", False))
        temporal_history_ready = temporal_features.observation_count >= self.config.min_history_required
        details["temporal_history_ready"] = temporal_history_ready
        details["trusted_detection"] = warmup_complete and temporal_history_ready

    def _restore_extra_state(self, payload: dict[str, Any]) -> None:
        del payload
        if self._model is not None:
            self._model.forest.rebuild_runtime_state()

    def get_model_info(self) -> dict[str, object]:
        info = super().get_model_info()
        if not self._is_fitted:
            return info

        state: RRCFStreamState = self._model
        info.update(
            {
                "num_trees": self.config.num_trees,
                "tree_size": self.config.tree_size,
                "warmup_samples": self.config.warmup_samples,
                "observations_seen": state.observations_seen,
                "n_features": len(self._feature_names),
            }
        )
        return info
