from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from research.training.scripts import tuning_utils
from research.training.scripts.train_isolation_forest import (
    FEATURE_NAMES,
    extract_features_vectorized,
    find_parquet_files as find_train_parquet_files,
    train_from_matrix,
)
from research.training.scripts.tune_isolation_forest import update_model_threshold
from research.training.scripts.tuning_utils import (
    find_parquet_files as find_tuning_parquet_files,
    get_anomaly_scores,
    run_single_trial,
    run_tuning_trials,
)
from src.anomaly.statistical import ZScoreDetector
from src.anomaly.persistence import ModelPersistence
from src.research.evaluation.detector_evaluator import DetectorEvaluator


def _price_frame(
    *,
    n_products: int = 20,
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


def test_train_from_matrix_trains_real_model_and_sets_normalization() -> None:
    detector, threshold = train_from_matrix(
        np.random.default_rng(42).normal(size=(128, 12)),
        contamination="auto",
        anomaly_threshold=0.4,
        n_estimators=16,
        max_samples=64,
        max_features=0.75,
        random_state=42,
    )

    assert detector._is_fitted is True
    assert detector._feature_names == FEATURE_NAMES
    assert detector._score_scale > 0
    assert detector._score_offset != 0 or detector._score_scale != 1.0
    assert threshold == pytest.approx(0.4)


def test_train_from_matrix_rejects_insufficient_valid_rows() -> None:
    X = np.zeros((64, 12), dtype=np.float64)

    with pytest.raises(ValueError, match="Need at least 50 valid samples"):
        train_from_matrix(X)


def test_threshold_tuning_flow_updates_saved_model(tmp_path) -> None:
    train_df = _price_frame(n_products=20, observations_per_product=8, start="2026-01-01")
    test_df = _price_frame(n_products=20, observations_per_product=5, start="2026-02-01")
    X_train = extract_features_vectorized(train_df)

    detector, _ = train_from_matrix(
        X_train,
        contamination="auto",
        anomaly_threshold=0.4,
        n_estimators=20,
        max_samples=64,
        max_features=0.75,
        random_state=42,
    )

    persistence = ModelPersistence(model_root=tmp_path / "models")
    persistence.save_isolation_forest(detector, "TEST_MODEL", len(X_train))
    loaded = persistence.load_isolation_forest("TEST_MODEL")

    thresholds = np.linspace(0.2, 0.8, 4)
    result = run_tuning_trials(
        detector=loaded,
        detector_name="TEST_MODEL",
        test_df=test_df,
        train_df=train_df,
        thresholds=thresholds,
        current_threshold=loaded.config.anomaly_threshold,
        n_trials=1,
        injection_rate=0.1,
        max_workers=1,
        target_metric="f1",
        min_precision=0.0,
        drop_range=(0.1, 0.5),
        min_successful_trials=1,
    )

    assert result is not None
    assert any(result.best_threshold == pytest.approx(threshold) for threshold in thresholds)

    update_model_threshold(persistence, "TEST_MODEL", float(result.best_threshold))
    updated = persistence.load_isolation_forest("TEST_MODEL")

    assert updated.config.anomaly_threshold == pytest.approx(float(result.best_threshold))


def test_get_anomaly_scores_prefers_batch_processing_for_batch_capable_evaluators() -> None:
    df = pd.DataFrame(
        [
            {
                "product_id": "product-001",
                "competitor_id": "competitor-1",
                "price": 100.0,
                "first_seen_at": pd.Timestamp("2026-02-02", tz="UTC"),
                "is_anomaly": 0,
            },
            {
                "product_id": "product-002",
                "competitor_id": "competitor-1",
                "price": 120.0,
                "first_seen_at": pd.Timestamp("2026-02-01", tz="UTC"),
                "is_anomaly": 1,
            },
        ]
    )

    class FakeBatchEvaluator:
        def __init__(self) -> None:
            self.batch_calls = 0
            self.row_calls = 0

        def supports_batch(self) -> bool:
            return True

        def process_batch(self, rows, col_map, country=None):
            self.batch_calls += 1
            assert len(rows) == 2
            return [
                SimpleNamespace(anomaly_score=0.25),
                SimpleNamespace(anomaly_score=0.75),
            ]

        def process_row(self, row, col_map, country=None):
            self.row_calls += 1
            raise AssertionError("Sequential path should not be used for batch-capable evaluators")

    evaluator = FakeBatchEvaluator()
    scores, labels = get_anomaly_scores(evaluator, df, country="FI", log_progress=False)

    assert evaluator.batch_calls == 1
    assert evaluator.row_calls == 0
    assert np.allclose(scores, np.array([0.25, 0.75]))
    assert np.array_equal(labels, np.array([1, 0]))


def test_run_single_trial_uses_cache_snapshot_without_repopulating(tmp_path) -> None:
    train_df = _price_frame(n_products=20, observations_per_product=8, start="2026-01-01")
    test_df = _price_frame(n_products=20, observations_per_product=5, start="2026-02-01")
    X_train = extract_features_vectorized(train_df)
    detector, _ = train_from_matrix(
        X_train,
        contamination="auto",
        anomaly_threshold=0.4,
        n_estimators=16,
        max_samples=64,
        max_features=0.75,
        random_state=42,
    )

    cache_builder = DetectorEvaluator(ZScoreDetector(), name="cache_builder")
    cache_builder.populate_cache(train_df)
    snapshot_path = tmp_path / "template_cache.joblib"
    cache_builder.temporal_cache.save_to_file(str(snapshot_path))

    def _fail_if_repopulated(self, historical_df):
        raise AssertionError("populate_cache should not be called when cache_snapshot_path is provided")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(DetectorEvaluator, "populate_cache", _fail_if_repopulated)
    try:
        result = run_single_trial(
            detector=detector,
            detector_name="TEST_MODEL",
            test_df=test_df,
            train_df=train_df,
            cache_snapshot_path=str(snapshot_path),
            thresholds=np.array([0.3, 0.4, 0.5]),
            current_threshold=0.4,
            injection_rate=0.1,
            seed=1000,
            country=None,
            drop_range=(0.1, 0.5),
        )
    finally:
        monkeypatch.undo()

    assert result["threshold_results"]
    assert result["current_result"] is not None


def test_run_tuning_trials_runs_single_trial_inline_without_executor(monkeypatch) -> None:
    thresholds = np.array([0.4, 0.5], dtype=np.float64)
    calls: list[int] = []

    def _fake_metrics(threshold: float, f1: float) -> dict[str, float]:
        return {
            "threshold": threshold,
            "accuracy": 0.9,
            "precision": 0.8,
            "recall": 0.7,
            "tnr": 0.95,
            "fpr": 0.05,
            "fnr": 0.3,
            "f1": f1,
            "g_mean": 0.8,
            "true_positives": 7,
            "false_positives": 2,
            "false_negatives": 3,
            "true_negatives": 38,
            "n_rows": 50,
            "n_injected": 10,
            "n_predicted": 9,
        }

    def _fake_run_single_trial(
        detector,
        detector_name,
        test_df,
        train_df,
        cache_snapshot_path,
        thresholds,
        current_threshold,
        injection_rate,
        seed,
        country,
        spike_range,
        drop_range,
    ):
        calls.append(seed)
        return {
            "threshold_results": [_fake_metrics(float(thresholds[0]), 0.2), _fake_metrics(float(thresholds[1]), 0.4)],
            "current_result": _fake_metrics(current_threshold, 0.2),
        }

    def _fail_executor(*args, **kwargs):
        raise AssertionError("Executors should not be created when n_trials=1")

    monkeypatch.setattr(tuning_utils, "run_single_trial", _fake_run_single_trial)
    monkeypatch.setattr(tuning_utils, "ProcessPoolExecutor", _fail_executor)
    monkeypatch.setattr(tuning_utils, "ThreadPoolExecutor", _fail_executor)

    result = run_tuning_trials(
        detector=object(),
        detector_name="TEST_MODEL",
        test_df=pd.DataFrame(),
        train_df=None,
        thresholds=thresholds,
        current_threshold=0.4,
        n_trials=1,
        injection_rate=0.1,
        max_workers=6,
        target_metric="f1",
        min_precision=0.0,
        min_successful_trials=1,
    )

    assert calls == [1000]
    assert result is not None
    assert result.best_threshold == pytest.approx(0.5)
    assert result.best_f1 == pytest.approx(0.4)


def test_global_granularity_file_discovery_supports_training_and_tuning(tmp_path) -> None:
    data_path = tmp_path / "data" / "training" / "derived"
    global_dir = data_path / "global"
    global_dir.mkdir(parents=True)

    train_file = global_dir / "GLOBAL_2026-02-08_train_mh5.parquet"
    test_file = global_dir / "GLOBAL_2026-02-08_test_new_prices_mh5.parquet"
    _price_frame(n_products=8, observations_per_product=8).to_parquet(train_file, index=False)
    _price_frame(n_products=8, observations_per_product=2, start="2026-02-01").to_parquet(test_file, index=False)

    discovered_train = find_train_parquet_files(str(data_path), "global", "_train_mh5")
    discovered_test = find_tuning_parquet_files(str(data_path), "global", "_test_new_prices_mh5")

    assert discovered_train == [str(train_file)]
    assert discovered_test == [str(test_file)]
