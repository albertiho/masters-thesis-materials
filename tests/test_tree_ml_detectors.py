from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.training.scripts.compare_detectors import create_evaluators, find_parquet_files as find_comparison_parquet_files
from research.training.scripts.train_isolation_forest import extract_features_vectorized, train_from_matrix
from src.anomaly.ml import EIFConfig, EIFDetector, RRCF, RRCFDetector, RRCFDetectorConfig
from src.anomaly.persistence import ModelPersistence
from src.anomaly.statistical import AnomalyResult
from src.features.numeric import NumericFeatures
from src.features.temporal import TemporalFeatures
from src.research.evaluation.detector_evaluator import DetectorEvaluator
from src.research.evaluation.test_orchestrator import TestOrchestrator as EvaluationTestOrchestrator


def _numeric_features(
    price: float,
    *,
    product_id: str = "product-1",
    competitor_id: str = "competitor-1",
) -> NumericFeatures:
    return NumericFeatures(
        price=price,
        list_price=price * 1.2,
        price_ratio=price / (price * 1.2),
        has_list_price=True,
        price_log=float(np.log(price + 1.0)),
        is_valid=True,
        validation_errors=[],
        competitor_product_id=product_id,
        competitor=competitor_id,
        country="FI",
    )


def _temporal_features(
    price: float,
    *,
    mean: float = 100.0,
    std: float = 2.0,
    minimum: float = 98.0,
    maximum: float = 102.0,
    observation_count: int = 8,
    has_history: bool = True,
    product_id: str = "product-1",
    competitor_id: str = "competitor-1",
) -> TemporalFeatures:
    if not has_history:
        return TemporalFeatures(
            rolling_mean=None,
            rolling_std=None,
            rolling_min=None,
            rolling_max=None,
            price_zscore=None,
            price_change_pct=None,
            days_since_change=None,
            observation_count=0,
            has_sufficient_history=False,
            competitor_product_id=product_id,
            competitor=competitor_id,
        )

    return TemporalFeatures(
        rolling_mean=mean,
        rolling_std=std,
        rolling_min=minimum,
        rolling_max=maximum,
        price_zscore=(price - mean) / std if std > 0 else 0.0,
        price_change_pct=(price - mean) / mean if mean > 0 else 0.0,
        days_since_change=1.0,
        observation_count=observation_count,
        has_sufficient_history=True,
        competitor_product_id=product_id,
        competitor=competitor_id,
    )


def _history_df() -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=6, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "product_id": ["product-1"] * len(timestamps),
            "competitor_id": ["competitor-1"] * len(timestamps),
            "competitor_product_id": ["product-1"] * len(timestamps),
            "price": [100.0, 101.0, 99.5, 100.5, 100.0, 100.5],
            "list_price": [120.0] * len(timestamps),
            "first_seen_at": timestamps,
        }
    )


def _test_df(prices: list[float]) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-10", periods=len(prices), freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "product_id": ["product-1"] * len(prices),
            "competitor_id": ["competitor-1"] * len(prices),
            "competitor_product_id": ["product-1"] * len(prices),
            "price": prices,
            "list_price": [120.0] * len(prices),
            "first_seen_at": timestamps,
        }
    )


def _price_frame(
    *,
    n_products: int = 10,
    observations_per_product: int = 8,
    start: str = "2026-01-01",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for product_idx in range(n_products):
        base_price = 100.0 + product_idx
        for observation_idx in range(observations_per_product):
            rows.append(
                {
                    "product_id": f"product-{product_idx:03d}",
                    "competitor_id": "competitor-1",
                    "competitor_product_id": f"product-{product_idx:03d}",
                    "price": base_price + (observation_idx % 4) * 0.5,
                    "list_price": base_price * 1.2,
                    "first_seen_at": pd.Timestamp(start, tz="UTC") + pd.Timedelta(days=observation_idx),
                }
            )
    return pd.DataFrame(rows)


def test_eif_fit_detect_and_detect_batch_return_valid_results() -> None:
    rng = np.random.default_rng(42)
    detector = EIFDetector(EIFConfig(n_estimators=12, max_samples=32, max_features=0.75))
    detector.fit_from_matrix(rng.normal(size=(96, 12)))

    single = detector.detect(_numeric_features(104.0), _temporal_features(104.0))
    batch = detector.detect_batch(
        [_numeric_features(104.0), _numeric_features(140.0)],
        [_temporal_features(104.0), _temporal_features(140.0)],
    )

    assert isinstance(single, AnomalyResult)
    assert single.detector == "eif"
    assert np.isfinite(single.anomaly_score)
    assert "eif_score" in single.details
    assert len(batch) == 2
    assert all(isinstance(result, AnomalyResult) for result in batch)
    assert all(np.isfinite(result.anomaly_score) for result in batch)


def test_eif_invalid_and_degenerate_input_paths_are_explicit() -> None:
    detector = EIFDetector()

    with pytest.raises(ValueError, match="Cannot fit on empty feature vectors"):
        detector.fit([])

    with pytest.raises(ValueError, match="Need at least 10 training samples"):
        detector.fit_from_matrix(np.random.default_rng(0).normal(size=(8, 12)))


def test_rrcf_scores_spikes_higher_than_nominal_and_gates_untrusted_history() -> None:
    rng = np.random.default_rng(123)
    training_matrix = rng.normal(size=(96, 12))
    config = RRCFDetectorConfig(num_trees=12, tree_size=64, anomaly_threshold=0.4, warmup_samples=8)

    nominal_detector = RRCFDetector(config)
    nominal_detector.fit_from_matrix(training_matrix)
    nominal = nominal_detector.detect(_numeric_features(101.0), _temporal_features(101.0))

    spike_detector = RRCFDetector(config)
    spike_detector.fit_from_matrix(training_matrix)
    spike = spike_detector.detect(_numeric_features(180.0), _temporal_features(180.0))

    cold_detector = RRCFDetector(
        RRCFDetectorConfig(
            num_trees=12,
            tree_size=64,
            anomaly_threshold=0.4,
            warmup_samples=200,
        )
    )
    cold_detector.fit_from_matrix(training_matrix)
    cold_result = cold_detector.detect(_numeric_features(180.0), _temporal_features(180.0))
    no_history_result = spike_detector.detect(_numeric_features(180.0), _temporal_features(180.0, has_history=False))

    assert spike.anomaly_score > nominal.anomaly_score
    assert cold_result.is_anomaly is False
    assert cold_result.details["trusted_detection"] is False
    assert no_history_result.is_anomaly is False
    assert no_history_result.details["temporal_history_ready"] is False


def test_low_level_rrcf_sliding_window_keeps_recent_points() -> None:
    forest = RRCF(num_trees=4, tree_size=8, anomaly_threshold=0.8, random_state=42)
    latest_result = None

    for offset in range(20):
        latest_result = forest.detect([100.0 + offset, float(offset % 3), float(offset % 5)])

    assert latest_result is not None
    assert np.isfinite(latest_result.raw_score)
    assert np.isfinite(latest_result.anomaly_score)

    expected_indices = set(range(12, 20))
    assert set(forest._index_to_points) == expected_indices
    assert list(forest._oldest_indices) == list(range(12, 20))
    assert forest.score(19) >= 0.0

    for tree in forest.trees:
        assert tree.root is not None
        assert tree.root.parent is None
        assert set(tree.points) == expected_indices
        assert set(tree.leaves) == expected_indices


def test_rrcf_load_restores_runtime_links_for_legacy_saved_models(tmp_path: Path) -> None:
    detector = RRCFDetector(
        RRCFDetectorConfig(num_trees=8, tree_size=32, anomaly_threshold=0.4, warmup_samples=8)
    )
    detector.fit_from_matrix(np.random.default_rng(7).normal(size=(96, 12)))

    def clear_parent_links(node: object) -> None:
        if node is None:
            return
        if hasattr(node, "parent"):
            node.parent = None
        if getattr(node, "left", None) is not None:
            clear_parent_links(node.left)
        if getattr(node, "right", None) is not None:
            clear_parent_links(node.right)

    state = detector._model
    for tree in state.forest.trees:
        clear_parent_links(tree.root)
    state.forest._oldest_indices = list(state.forest._oldest_indices)

    persistence = ModelPersistence(model_root=tmp_path / "models")
    persistence.save_rrcf(detector, "LEGACY_TEST_MODEL", 96)

    loaded = persistence.load_rrcf("LEGACY_TEST_MODEL")
    restored_state = loaded._model
    assert isinstance(restored_state.forest._oldest_indices, deque)

    result = loaded.detect(_numeric_features(140.0), _temporal_features(140.0))

    assert np.isfinite(result.anomaly_score)
    assert restored_state.forest.trees[0].root is not None
    assert restored_state.forest.trees[0].root.parent is None


@pytest.mark.parametrize(
    "detector",
    [
        EIFDetector(EIFConfig(n_estimators=10, max_samples=32, max_features=0.75)),
        RRCFDetector(RRCFDetectorConfig(num_trees=10, tree_size=64, warmup_samples=8, anomaly_threshold=0.4)),
    ],
)
def test_tree_detectors_run_through_detector_evaluator(detector: object) -> None:
    training_matrix = np.random.default_rng(42).normal(size=(96, 12))
    detector.fit_from_matrix(training_matrix)  # type: ignore[attr-defined]

    evaluator = DetectorEvaluator(
        detector,
        enable_persistence_acceptance=False,
    )
    evaluator.populate_cache(_history_df())

    test_df = _test_df([102.0, 140.0])
    col_map = {column: idx for idx, column in enumerate(test_df.columns)}
    rows = list(test_df.itertuples(index=False))

    single = evaluator.process_row(rows[0], col_map, country="FI")
    batch = evaluator.process_batch(rows, col_map, country="FI")

    assert isinstance(single, AnomalyResult)
    assert len(batch) == 2
    assert all(isinstance(result, AnomalyResult) for result in batch)


def test_compare_detectors_research_surface_loads_if_eif_and_rrcf(tmp_path: Path) -> None:
    train_df = _price_frame(n_products=12, observations_per_product=8, start="2026-01-01")
    test_df = _price_frame(n_products=12, observations_per_product=2, start="2026-02-01")
    labels = np.zeros(len(test_df), dtype=bool)
    X_train = extract_features_vectorized(train_df)

    persistence = ModelPersistence(model_root=tmp_path / "models")
    if_detector, _ = train_from_matrix(
        X_train,
        anomaly_threshold=0.4,
        n_estimators=20,
        max_samples=32,
        max_features=0.75,
        random_state=42,
    )
    eif_detector = EIFDetector(EIFConfig(n_estimators=20, max_samples=32, max_features=0.75, anomaly_threshold=0.4))
    eif_detector.fit_from_matrix(X_train)
    rrcf_detector = RRCFDetector(
        RRCFDetectorConfig(num_trees=20, tree_size=32, anomaly_threshold=0.4, warmup_samples=16)
    )
    rrcf_detector.fit_from_matrix(X_train)

    persistence.save_isolation_forest(if_detector, "TEST_MODEL", len(X_train))
    persistence.save_eif(eif_detector, "TEST_MODEL", len(X_train))
    persistence.save_rrcf(rrcf_detector, "TEST_MODEL", len(X_train))

    evaluators = create_evaluators(persistence, "TEST_MODEL", skip_ml=False)
    evaluator_names = {evaluator.name for evaluator in evaluators}

    assert {"Isolation Forest", "EIF", "RRCF"}.issubset(evaluator_names)

    orchestrator = EvaluationTestOrchestrator(evaluators, max_workers=1)
    comparison = orchestrator.run_comparison_with_details(
        train_df=train_df,
        test_df=test_df,
        labels=labels,
        country="FI",
    )

    assert {"Isolation Forest", "EIF", "RRCF"}.issubset(set(comparison.metrics))


def test_compare_detectors_file_discovery_supports_global(tmp_path: Path) -> None:
    data_path = tmp_path / "data" / "training" / "derived"
    global_dir = data_path / "global"
    global_dir.mkdir(parents=True)

    global_test = global_dir / "GLOBAL_2026-02-08_test_new_prices_mh5.parquet"
    _price_frame(n_products=12, observations_per_product=2, start="2026-02-01").to_parquet(
        global_test,
        index=False,
    )

    discovered = find_comparison_parquet_files(
        str(data_path),
        "global",
        "_test_new_prices_mh5",
    )

    assert discovered == [str(global_test)]
