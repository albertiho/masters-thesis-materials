from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.anomaly.combined import DetectionContext, DetectorLayer
from src.anomaly.statistical import (
    AnomalyResult,
    HybridAvgZScoreDetector,
    HybridMaxZScoreDetector,
    HybridWeightedZScoreDetector,
    IQRDetector,
    ModifiedMADDetector,
    ModifiedSNDetector,
    ThresholdDetector,
    ZScoreDetector,
)
from src.features.numeric import NumericFeatures
from src.features.temporal import TemporalCacheManager, TemporalFeatures
from src.research.evaluation import (
    DetectorEvaluator,
    TestOrchestrator as EvaluationTestOrchestrator,
    create_expanded_statistical_evaluators,
)


BASELINE_HISTORY = [10.0, 11.0, 12.0, 13.0, 14.0]
CACHE_BEHAVIOR_HISTORY = [98.0, 99.0, 100.0, 101.0, 102.0]
WHOLE_HISTORY_WITH_LAST_SPIKE = [90.0, 95.0, 100.0, 105.0, 160.0]


def _history_df(prices: list[float] | None = None) -> pd.DataFrame:
    history = prices if prices is not None else BASELINE_HISTORY
    timestamps = pd.date_range("2026-01-01", periods=len(history), freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "product_id": ["product-1"] * len(history),
            "competitor_id": ["competitor-1"] * len(history),
            "competitor_product_id": ["product-1"] * len(history),
            "price": history,
            "first_seen_at": timestamps,
        }
    )


def _test_df(prices: list[float]) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-06", periods=len(prices), freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "product_id": ["product-1"] * len(prices),
            "competitor_id": ["competitor-1"] * len(prices),
            "competitor_product_id": ["product-1"] * len(prices),
            "price": prices,
            "first_seen_at": timestamps,
        }
    )


def _numeric_features(price: float) -> NumericFeatures:
    return NumericFeatures(
        price=price,
        list_price=None,
        price_ratio=1.0,
        has_list_price=False,
        price_log=0.0,
        is_valid=True,
        validation_errors=[],
        competitor_product_id="product-1",
        competitor="competitor-1",
        country="FI",
    )


def _context(current_price: float) -> DetectionContext:
    cache = TemporalCacheManager()
    cache._create_entry("product-1", "competitor-1", BASELINE_HISTORY)
    cache_entry = cache.get("product-1", "competitor-1")
    assert cache_entry is not None

    current_timestamp = pd.Timestamp("2026-01-06T00:00:00Z").to_pydatetime()
    temporal_features = TemporalFeatures.from_cache(cache_entry, current_price, current_timestamp)

    return DetectionContext.from_features(
        numeric_features=_numeric_features(current_price),
        temporal_features=temporal_features,
        price_history=list(BASELINE_HISTORY),
        observation_count=temporal_features.observation_count,
    )


def _assert_results_equivalent(left: AnomalyResult, right: AnomalyResult) -> None:
    assert left.is_anomaly == right.is_anomaly
    assert left.anomaly_score == pytest.approx(right.anomaly_score)
    assert left.anomaly_types == right.anomaly_types
    assert left.severity == right.severity
    assert left.detector == right.detector
    assert left.competitor_product_id == right.competitor_product_id
    assert left.competitor == right.competitor
    assert set(left.details) == set(right.details)

    for key in left.details:
        left_value = left.details[key]
        right_value = right.details[key]
        if (
            isinstance(left_value, (int, float, np.floating))
            and isinstance(right_value, (int, float, np.floating))
            and not isinstance(left_value, bool)
            and not isinstance(right_value, bool)
        ):
            if np.isnan(left_value) and np.isnan(right_value):
                continue
            assert left_value == pytest.approx(right_value)
        else:
            assert left_value == right_value


class RecordingHistoryDetector:
    name = "recording_history"
    requires_price_history = True

    def __init__(self) -> None:
        self.seen_price_history: list[float] | None = None

    def detect(
        self,
        numeric_features: NumericFeatures,
        temporal_features: TemporalFeatures,
        price_history: list[float] | None = None,
    ) -> AnomalyResult:
        del temporal_features
        self.seen_price_history = price_history
        return AnomalyResult(
            is_anomaly=False,
            anomaly_score=0.0,
            anomaly_types=[],
            severity=None,
            details={"observation_count": len(price_history or [])},
            detector=self.name,
            competitor_product_id=numeric_features.competitor_product_id,
            competitor=numeric_features.competitor,
        )


@pytest.mark.parametrize(
    "detector",
    [
        ModifiedMADDetector(),
        ModifiedSNDetector(),
        HybridWeightedZScoreDetector(),
        HybridMaxZScoreDetector(),
        HybridAvgZScoreDetector(),
    ],
)
def test_new_history_detectors_run_through_detector_evaluator(detector: object) -> None:
    evaluator = DetectorEvaluator(detector, enable_persistence_acceptance=False)
    evaluator.populate_cache(_history_df())

    test_df = _test_df([20.0])
    col_map = {column: idx for idx, column in enumerate(test_df.columns)}
    row = next(test_df.itertuples(index=False))

    result = evaluator.process_row(row, col_map, country="FI")

    assert isinstance(result, AnomalyResult)
    assert result.detector == detector.name  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "detector",
    [
        ModifiedMADDetector(),
        ModifiedSNDetector(),
        HybridWeightedZScoreDetector(),
        HybridMaxZScoreDetector(),
        HybridAvgZScoreDetector(),
        IQRDetector(),
    ],
)
def test_history_dependent_detectors_run_inside_detector_layer(detector: object) -> None:
    context = _context(20.0)
    layer = DetectorLayer(name="statistical", detectors=[detector])

    results = layer.detect(context)

    assert len(results) == 1
    assert isinstance(results[0], AnomalyResult)
    assert results[0].detector == detector.name  # type: ignore[attr-defined]


def test_detector_evaluator_passes_price_history_to_history_dependent_detectors() -> None:
    detector = RecordingHistoryDetector()
    evaluator = DetectorEvaluator(detector, enable_persistence_acceptance=False)
    evaluator.populate_cache(_history_df())

    test_df = _test_df([20.0])
    col_map = {column: idx for idx, column in enumerate(test_df.columns)}
    row = next(test_df.itertuples(index=False))

    evaluator.process_row(row, col_map, country="FI")

    assert detector.seen_price_history == BASELINE_HISTORY


def test_detector_layer_passes_price_history_to_history_dependent_detectors() -> None:
    detector = RecordingHistoryDetector()
    layer = DetectorLayer(name="statistical", detectors=[detector])

    layer.detect(_context(20.0))

    assert detector.seen_price_history == BASELINE_HISTORY


def test_existing_statistical_detectors_still_run_through_evaluator() -> None:
    history_df = _history_df()
    test_df = _test_df([20.0])
    col_map = {column: idx for idx, column in enumerate(test_df.columns)}
    row = next(test_df.itertuples(index=False))

    zscore_eval = DetectorEvaluator(ZScoreDetector(), enable_persistence_acceptance=False)
    zscore_eval.populate_cache(history_df)
    zscore_result = zscore_eval.process_row(row, col_map, country="FI")

    iqr_eval = DetectorEvaluator(IQRDetector(), enable_persistence_acceptance=False)
    iqr_eval.populate_cache(history_df)
    iqr_result = iqr_eval.process_row(row, col_map, country="FI")

    threshold_eval = DetectorEvaluator(ThresholdDetector(), enable_persistence_acceptance=False)
    threshold_eval.populate_cache(history_df)
    threshold_result = threshold_eval.process_row(row, col_map, country="FI")

    assert zscore_result.details["rolling_mean"] == pytest.approx(np.mean(BASELINE_HISTORY))
    assert "q1" in iqr_result.details
    assert threshold_result.details["price_change_pct"] == pytest.approx((20.0 - 14.0) / 14.0)


def test_zscore_detector_uses_full_cached_history_not_just_previous_price() -> None:
    history = [100.0, 100.0, 100.0, 100.0, 160.0]
    history_df = _history_df(history)
    test_df = _test_df([100.0])
    col_map = {column: idx for idx, column in enumerate(test_df.columns)}
    row = next(test_df.itertuples(index=False))

    evaluator = DetectorEvaluator(ZScoreDetector(threshold=1.0), enable_persistence_acceptance=False)
    evaluator.populate_cache(history_df)

    result = evaluator.process_row(row, col_map, country="FI")

    expected_mean = float(np.mean(history))
    expected_std = float(np.std(history, ddof=1))
    expected_zscore = (100.0 - expected_mean) / expected_std

    # The current price returns to the long-run baseline even though t-1 was a spike.
    assert history[-1] == pytest.approx(160.0)
    assert result.details["rolling_mean"] == pytest.approx(expected_mean)
    assert result.details["rolling_std"] == pytest.approx(expected_std)
    assert result.details["zscore"] == pytest.approx(expected_zscore)
    assert result.details["rolling_mean"] != pytest.approx(history[-1])
    assert result.is_anomaly is False


@pytest.mark.parametrize(
    "detector_factory",
    [
        ModifiedMADDetector,
        ModifiedSNDetector,
        HybridWeightedZScoreDetector,
        HybridMaxZScoreDetector,
        HybridAvgZScoreDetector,
        IQRDetector,
        ThresholdDetector,
    ],
)
def test_tuned_statistical_detectors_only_cache_non_anomalous_values(detector_factory) -> None:
    history_df = _history_df(CACHE_BEHAVIOR_HISTORY)
    test_df = _test_df([150.0, 102.4])
    col_map = {column: idx for idx, column in enumerate(test_df.columns)}
    rows = list(test_df.itertuples(index=False))

    evaluator = DetectorEvaluator(detector_factory(), enable_persistence_acceptance=False)
    evaluator.populate_cache(history_df)
    results = evaluator.process_batch(rows, col_map, country="FI")

    cache_entry = evaluator.temporal_cache.get("product-1", "competitor-1")

    assert results[0].is_anomaly is True
    assert results[1].is_anomaly is False
    assert cache_entry is not None
    assert cache_entry.observation_count == len(CACHE_BEHAVIOR_HISTORY) + 1
    assert cache_entry.price_history.tolist() == pytest.approx(CACHE_BEHAVIOR_HISTORY + [102.4])


@pytest.mark.parametrize(
    ("detector", "detail_key", "expected_value"),
    [
        (ModifiedMADDetector(), "baseline_median", 100.0),
        (ModifiedSNDetector(), "baseline_median", 100.0),
        (HybridWeightedZScoreDetector(), "baseline_median", 100.0),
        (HybridMaxZScoreDetector(), "baseline_median", 100.0),
        (HybridAvgZScoreDetector(), "baseline_median", 100.0),
        (IQRDetector(), "q3", 105.0),
    ],
)
def test_history_based_tuned_detectors_use_whole_cache_not_just_latest_price(
    detector: object,
    detail_key: str,
    expected_value: float,
) -> None:
    history_df = _history_df(WHOLE_HISTORY_WITH_LAST_SPIKE)
    test_df = _test_df([100.0])
    col_map = {column: idx for idx, column in enumerate(test_df.columns)}
    row = next(test_df.itertuples(index=False))

    evaluator = DetectorEvaluator(detector, enable_persistence_acceptance=False)
    evaluator.populate_cache(history_df)
    result = evaluator.process_row(row, col_map, country="FI")

    assert result.is_anomaly is False
    assert result.details[detail_key] == pytest.approx(expected_value)
    assert result.details[detail_key] != pytest.approx(WHOLE_HISTORY_WITH_LAST_SPIKE[-1])


def test_threshold_detector_uses_last_accepted_price_change_not_whole_history() -> None:
    history_df = _history_df(WHOLE_HISTORY_WITH_LAST_SPIKE)
    test_df = _test_df([100.0])
    col_map = {column: idx for idx, column in enumerate(test_df.columns)}
    row = next(test_df.itertuples(index=False))

    evaluator = DetectorEvaluator(ThresholdDetector(), enable_persistence_acceptance=False)
    evaluator.populate_cache(history_df)
    result = evaluator.process_row(row, col_map, country="FI")

    expected_change = (100.0 - WHOLE_HISTORY_WITH_LAST_SPIKE[-1]) / WHOLE_HISTORY_WITH_LAST_SPIKE[-1]

    assert result.is_anomaly is True
    assert result.details["price_change_pct"] == pytest.approx(expected_change)


@pytest.mark.parametrize(
    "detector_factory",
    [
        ZScoreDetector,
        ModifiedMADDetector,
        ModifiedSNDetector,
        HybridWeightedZScoreDetector,
        HybridMaxZScoreDetector,
        HybridAvgZScoreDetector,
        IQRDetector,
        ThresholdDetector,
    ],
)
def test_statistical_detectors_match_between_sequential_and_batch_paths(
    detector_factory,
) -> None:
    history_df = _history_df()
    test_df = _test_df([20.0, 12.5])
    col_map = {column: idx for idx, column in enumerate(test_df.columns)}
    rows = list(test_df.itertuples(index=False))

    sequential_eval = DetectorEvaluator(detector_factory(), enable_persistence_acceptance=False)
    sequential_eval.populate_cache(history_df)
    sequential_results = [
        sequential_eval.process_row(row, col_map, country="FI")
        for row in rows
    ]

    batch_eval = DetectorEvaluator(detector_factory(), enable_persistence_acceptance=False)
    batch_eval.populate_cache(history_df)
    assert batch_eval.supports_batch()
    batch_results = batch_eval.process_batch(rows, col_map, country="FI")

    assert len(sequential_results) == len(batch_results)
    for sequential_result, batch_result in zip(sequential_results, batch_results, strict=True):
        _assert_results_equivalent(sequential_result, batch_result)


def test_expanded_statistical_roster_smoke_runs_through_existing_evaluation_path() -> None:
    evaluators = create_expanded_statistical_evaluators()
    orchestrator = EvaluationTestOrchestrator(evaluators, max_workers=1)

    comparison = orchestrator.run_comparison_with_details(
        train_df=_history_df(),
        test_df=_test_df([20.0, 12.5]),
        labels=np.array([True, False], dtype=bool),
        country="FI",
    )

    expected_names = {
        "Z-score",
        "ModifiedMAD",
        "ModifiedSN",
        "HybridWeighted",
        "HybridMax",
        "HybridAvg",
        "IQR",
        "Threshold",
    }

    assert set(comparison.metrics) == expected_names
    assert set(comparison.raw_results) == expected_names
    assert len(comparison.observation_counts) == 2

    for results in comparison.raw_results.values():
        assert len(results) == 2
        assert all(isinstance(result, AnomalyResult) for result in results)

    for evaluator in evaluators:
        assert evaluator.get_cache_stats()["total_products"] >= 1
