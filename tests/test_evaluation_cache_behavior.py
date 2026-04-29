from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from src.anomaly.statistical import AnomalyResult
from src.research.evaluation.detector_evaluator import DetectorEvaluator


BASELINE_HISTORY = [100.0, 100.0, 100.0]


def _history_df() -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=len(BASELINE_HISTORY), freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "product_id": ["product-1"] * len(BASELINE_HISTORY),
            "competitor_id": ["competitor-1"] * len(BASELINE_HISTORY),
            "competitor_product_id": ["product-1"] * len(BASELINE_HISTORY),
            "price": BASELINE_HISTORY,
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
            "first_seen_at": timestamps,
        }
    )


class RecordingSequentialDetector:
    name = "zscore"

    def __init__(self, anomaly_flags: list[bool]) -> None:
        self._anomaly_flags = anomaly_flags
        self.seen_observation_counts: list[int] = []
        self.seen_means: list[float | None] = []

    def detect(self, numeric_features, temporal_features) -> AnomalyResult:
        index = len(self.seen_observation_counts)
        is_anomaly = self._anomaly_flags[index]
        self.seen_observation_counts.append(temporal_features.observation_count)
        self.seen_means.append(temporal_features.rolling_mean)
        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=1.0 if is_anomaly else 0.0,
            anomaly_types=[],
            severity=None,
            details={"observation_count": temporal_features.observation_count},
            detector=self.name,
            competitor_product_id=numeric_features.competitor_product_id,
            competitor=numeric_features.competitor,
        )


class RecordingBatchDetector:
    name = "autoencoder"

    def __init__(self, anomaly_flags_by_round: list[bool]) -> None:
        self._anomaly_flags_by_round = anomaly_flags_by_round
        self.round_observation_counts: list[list[int]] = []
        self.round_means: list[list[float | None]] = []
        self._round_index = 0

    def detect_batch(self, numeric_features_list, temporal_features_list) -> list[AnomalyResult]:
        is_anomaly = self._anomaly_flags_by_round[self._round_index]
        self._round_index += 1

        self.round_observation_counts.append(
            [features.observation_count for features in temporal_features_list]
        )
        self.round_means.append([features.rolling_mean for features in temporal_features_list])

        results = []
        for numeric_features in numeric_features_list:
            results.append(
                AnomalyResult(
                    is_anomaly=is_anomaly,
                    anomaly_score=1.0 if is_anomaly else 0.0,
                    anomaly_types=[],
                    severity=None,
                    details={},
                    detector=self.name,
                    competitor_product_id=numeric_features.competitor_product_id,
                    competitor=numeric_features.competitor,
                )
            )
        return results


def test_sequential_cache_updates_affect_the_next_datapoint_when_non_anomalous() -> None:
    detector = RecordingSequentialDetector([False, False])
    evaluator = DetectorEvaluator(detector, enable_persistence_acceptance=False)
    evaluator.populate_cache(_history_df())

    test_df = _test_df([101.0, 102.0])
    col_map = {column: idx for idx, column in enumerate(test_df.columns)}

    for row in test_df.itertuples(index=False):
        evaluator.process_row(row, col_map, country="FI")

    assert detector.seen_observation_counts == [3, 4]
    assert detector.seen_means[0] == pytest.approx(100.0)
    assert detector.seen_means[1] == pytest.approx(100.25)


def test_sequential_anomalous_value_does_not_pollute_the_next_datapoint_cache() -> None:
    detector = RecordingSequentialDetector([True, False])
    evaluator = DetectorEvaluator(detector, enable_persistence_acceptance=False)
    evaluator.populate_cache(_history_df())

    test_df = _test_df([200.0, 101.0])
    col_map = {column: idx for idx, column in enumerate(test_df.columns)}

    for row in test_df.itertuples(index=False):
        evaluator.process_row(row, col_map, country="FI")

    assert detector.seen_observation_counts == [3, 3]
    assert detector.seen_means == [pytest.approx(100.0), pytest.approx(100.0)]


def test_batch_cache_updates_affect_the_next_round_for_the_same_product() -> None:
    detector = RecordingBatchDetector([False, False])
    evaluator = DetectorEvaluator(detector, enable_persistence_acceptance=False)
    evaluator.populate_cache(_history_df())

    test_df = _test_df([101.0, 102.0])
    col_map = {column: idx for idx, column in enumerate(test_df.columns)}
    rows = list(test_df.itertuples(index=False))

    evaluator.process_batch(rows, col_map, country="FI")

    assert detector.round_observation_counts == [[3], [4]]
    assert detector.round_means[0][0] == pytest.approx(100.0)
    assert detector.round_means[1][0] == pytest.approx(100.25)


def test_batch_anomalous_value_does_not_update_cache_for_the_next_round() -> None:
    detector = RecordingBatchDetector([True, False])
    evaluator = DetectorEvaluator(detector, enable_persistence_acceptance=False)
    evaluator.populate_cache(_history_df())

    test_df = _test_df([200.0, 101.0])
    col_map = {column: idx for idx, column in enumerate(test_df.columns)}
    rows = list(test_df.itertuples(index=False))

    evaluator.process_batch(rows, col_map, country="FI")

    assert detector.round_observation_counts == [[3], [3]]
    assert detector.round_means[0][0] == pytest.approx(100.0)
    assert detector.round_means[1][0] == pytest.approx(100.0)
