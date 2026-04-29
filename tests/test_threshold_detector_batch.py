from __future__ import annotations

import math

import pytest

from src.anomaly.statistical import ThresholdDetector
from src.features.numeric import NumericFeatures
from src.features.temporal import TemporalFeatures


def _numeric_features(price: float, *, product_id: str) -> NumericFeatures:
    return NumericFeatures(
        price=price,
        list_price=None,
        price_ratio=1.0,
        has_list_price=False,
        price_log=math.log(price + 1.0) if price >= 0 else 0.0,
        is_valid=True,
        validation_errors=[],
        competitor_product_id=product_id,
        competitor="competitor-1",
        country="FI",
    )


def _temporal_features(price_change_pct: float | None, *, product_id: str) -> TemporalFeatures:
    return TemporalFeatures(
        rolling_mean=100.0,
        rolling_std=5.0,
        rolling_min=90.0,
        rolling_max=110.0,
        price_zscore=0.0,
        price_change_pct=price_change_pct,
        days_since_change=1.0,
        observation_count=5,
        has_sufficient_history=True,
        competitor_product_id=product_id,
        competitor="competitor-1",
    )


def _assert_results_equivalent(left, right) -> None:
    assert left.is_anomaly == right.is_anomaly
    assert left.anomaly_score == pytest.approx(right.anomaly_score)
    assert left.anomaly_types == right.anomaly_types
    assert left.severity == right.severity
    assert left.detector == right.detector
    assert left.competitor_product_id == right.competitor_product_id
    assert left.competitor == right.competitor
    assert left.details.keys() == right.details.keys()

    for key in left.details:
        left_value = left.details[key]
        right_value = right.details[key]
        if isinstance(left_value, float) and isinstance(right_value, float):
            assert left_value == pytest.approx(right_value)
        else:
            assert left_value == right_value


def test_threshold_detector_batch_matches_single_detection_semantics() -> None:
    detector = ThresholdDetector(threshold=0.20)
    numeric_features_list = [
        _numeric_features(100.0, product_id="product-none"),
        _numeric_features(115.0, product_id="product-below"),
        _numeric_features(120.0, product_id="product-edge"),
        _numeric_features(125.0, product_id="product-medium"),
        _numeric_features(145.0, product_id="product-high"),
        _numeric_features(170.0, product_id="product-critical"),
        _numeric_features(70.0, product_id="product-decrease"),
    ]
    temporal_features_list = [
        _temporal_features(None, product_id="product-none"),
        _temporal_features(0.15, product_id="product-below"),
        _temporal_features(0.20, product_id="product-edge"),
        _temporal_features(0.25, product_id="product-medium"),
        _temporal_features(0.45, product_id="product-high"),
        _temporal_features(0.70, product_id="product-critical"),
        _temporal_features(-0.30, product_id="product-decrease"),
    ]

    batch_results = detector.detect_batch(numeric_features_list, temporal_features_list)
    sequential_results = [
        detector.detect(numeric_features, temporal_features)
        for numeric_features, temporal_features in zip(
            numeric_features_list,
            temporal_features_list,
            strict=True,
        )
    ]

    assert len(batch_results) == len(sequential_results)
    for batch_result, sequential_result in zip(batch_results, sequential_results, strict=True):
        _assert_results_equivalent(batch_result, sequential_result)

    assert batch_results[0].details["no_previous_price"] is True
    assert batch_results[2].is_anomaly is False
    assert batch_results[3].details["direction"] == "increase"
    assert batch_results[4].severity.value == "high"
    assert batch_results[5].severity.value == "critical"
    assert batch_results[6].details["direction"] == "decrease"
