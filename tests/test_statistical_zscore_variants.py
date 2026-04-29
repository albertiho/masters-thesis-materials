from __future__ import annotations

import numpy as np
import pytest

from src.anomaly.statistical import (
    HybridAvgZScoreDetector,
    HybridMaxZScoreDetector,
    HybridWeightedZScoreDetector,
    IQRDetector,
    ModifiedMADDetector,
    ModifiedSNDetector,
)
from src.anomaly.z_score_methods import mad_scale, sn_scale
from src.features.numeric import NumericFeatures
from src.features.temporal import TemporalFeatures


BASELINE_HISTORY = [10.0, 11.0, 12.0, 13.0, 14.0]


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


def _temporal_features(observation_count: int) -> TemporalFeatures:
    return TemporalFeatures(
        rolling_mean=12.0,
        rolling_std=1.0,
        rolling_min=10.0,
        rolling_max=14.0,
        price_zscore=3.0,
        price_change_pct=0.1,
        days_since_change=1.0,
        observation_count=observation_count,
        has_sufficient_history=observation_count >= 3,
        competitor_product_id="product-1",
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
        if isinstance(left_value, (int, float, np.floating)) and isinstance(
            right_value, (int, float, np.floating)
        ) and not isinstance(left_value, bool) and not isinstance(right_value, bool):
            assert left_value == pytest.approx(right_value)
        else:
            assert left_value == right_value


def test_modified_mad_detector_matches_expected_formula() -> None:
    detector = ModifiedMADDetector()
    result = detector.detect(
        _numeric_features(15.0),
        _temporal_features(len(BASELINE_HISTORY)),
        BASELINE_HISTORY,
    )

    expected = abs(15.0 - np.median(BASELINE_HISTORY)) / mad_scale(
        np.asarray(BASELINE_HISTORY, dtype=np.float64)
    )

    assert result.details["baseline_median"] == pytest.approx(12.0)
    assert result.details["modified_mad_score"] == pytest.approx(expected)
    assert result.is_anomaly is True


def test_modified_sn_detector_matches_expected_formula() -> None:
    detector = ModifiedSNDetector()
    result = detector.detect(
        _numeric_features(15.0),
        _temporal_features(len(BASELINE_HISTORY)),
        BASELINE_HISTORY,
    )

    expected = abs(15.0 - np.median(BASELINE_HISTORY)) / sn_scale(
        np.asarray(BASELINE_HISTORY, dtype=np.float64)
    )

    assert result.details["baseline_median"] == pytest.approx(12.0)
    assert result.details["modified_sn_score"] == pytest.approx(expected)
    assert result.is_anomaly is True


def test_weighted_hybrid_detector_respects_constructor_weight() -> None:
    history = np.asarray(BASELINE_HISTORY, dtype=np.float64)
    baseline_median = float(np.median(history))
    mad_score = abs(15.0 - baseline_median) / mad_scale(history)
    sn_score = abs(15.0 - baseline_median) / sn_scale(history)

    low_weight = HybridWeightedZScoreDetector(w=0.2)
    high_weight = HybridWeightedZScoreDetector(w=0.8)

    low_result = low_weight.detect(
        _numeric_features(15.0),
        _temporal_features(len(BASELINE_HISTORY)),
        BASELINE_HISTORY,
    )
    high_result = high_weight.detect(
        _numeric_features(15.0),
        _temporal_features(len(BASELINE_HISTORY)),
        BASELINE_HISTORY,
    )

    assert low_result.details["weight"] == pytest.approx(0.2)
    assert high_result.details["weight"] == pytest.approx(0.8)
    assert low_result.details["hybrid_weighted_score"] == pytest.approx(
        0.2 * mad_score + 0.8 * sn_score
    )
    assert high_result.details["hybrid_weighted_score"] == pytest.approx(
        0.8 * mad_score + 0.2 * sn_score
    )
    assert low_result.details["hybrid_weighted_score"] != pytest.approx(
        high_result.details["hybrid_weighted_score"]
    )


@pytest.mark.parametrize(
    "detector",
    [
        HybridWeightedZScoreDetector(),
        HybridMaxZScoreDetector(),
        HybridAvgZScoreDetector(),
    ],
)
def test_hybrid_detectors_flag_large_deviations(detector: object) -> None:
    result = detector.detect(  # type: ignore[attr-defined]
        _numeric_features(15.0),
        _temporal_features(len(BASELINE_HISTORY)),
        BASELINE_HISTORY,
    )

    assert result.is_anomaly is True
    assert result.anomaly_score > 0.5
    assert result.details["threshold"] == pytest.approx(2.0)


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
def test_history_based_variants_report_insufficient_history(detector: object) -> None:
    result = detector.detect(  # type: ignore[attr-defined]
        _numeric_features(15.0),
        _temporal_features(2),
        [10.0, 11.0],
    )

    assert result.is_anomaly is False
    assert result.details["insufficient_history"] is True
    assert result.details["observation_count"] == 2


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
def test_history_based_variants_report_degenerate_scale(detector: object) -> None:
    result = detector.detect(  # type: ignore[attr-defined]
        _numeric_features(20.0),
        _temporal_features(3),
        [10.0, 10.0, 10.0],
    )

    assert result.is_anomaly is False
    assert result.details["degenerate_scale"] is True
    assert result.details["observation_count"] == 3


def test_history_based_detector_names_are_stable_and_unique() -> None:
    detectors = [
        ModifiedMADDetector(),
        ModifiedSNDetector(),
        HybridWeightedZScoreDetector(),
        HybridMaxZScoreDetector(),
        HybridAvgZScoreDetector(),
    ]

    names = [detector.name for detector in detectors]

    assert names == [
        "modified_mad",
        "modified_sn",
        "hybrid_weighted_zscore",
        "hybrid_max_zscore",
        "hybrid_avg_zscore",
    ]
    assert len(set(names)) == len(names)
    assert all(detector.threshold == pytest.approx(2.0) for detector in detectors)


@pytest.mark.parametrize(
    "detector",
    [
        ModifiedMADDetector(),
        ModifiedSNDetector(),
        HybridWeightedZScoreDetector(w=0.8),
        HybridMaxZScoreDetector(),
        HybridAvgZScoreDetector(),
    ],
)
def test_history_based_variants_detect_batch_matches_single_detection(detector: object) -> None:
    numeric_features_list = [
        _numeric_features(15.0),
        _numeric_features(12.5),
        _numeric_features(20.0),
    ]
    temporal_features_list = [
        _temporal_features(len(BASELINE_HISTORY)),
        _temporal_features(2),
        _temporal_features(3),
    ]
    price_history_list = [
        BASELINE_HISTORY,
        [10.0, 11.0],
        [10.0, 10.0, 10.0],
    ]

    batch_results = detector.detect_batch(  # type: ignore[attr-defined]
        numeric_features_list,
        temporal_features_list,
        price_history_list,
    )
    sequential_results = [
        detector.detect(numeric_features, temporal_features, price_history)  # type: ignore[attr-defined]
        for numeric_features, temporal_features, price_history in zip(
            numeric_features_list,
            temporal_features_list,
            price_history_list,
            strict=True,
        )
    ]

    assert len(batch_results) == len(sequential_results)
    for batch_result, sequential_result in zip(batch_results, sequential_results, strict=True):
        _assert_results_equivalent(batch_result, sequential_result)


def test_iqr_detect_batch_matches_single_detection() -> None:
    detector = IQRDetector(multiplier=1.5)
    numeric_features_list = [
        _numeric_features(20.0),
        _numeric_features(12.0),
        _numeric_features(15.0),
    ]
    temporal_features_list = [
        _temporal_features(len(BASELINE_HISTORY)),
        _temporal_features(3),
        _temporal_features(4),
    ]
    price_history_list = [
        BASELINE_HISTORY,
        [10.0, 11.0, 12.0],
        [10.0, 10.0, 10.0, 10.0],
    ]

    batch_results = detector.detect_batch(
        numeric_features_list,
        temporal_features_list,
        price_history_list,
    )
    sequential_results = [
        detector.detect(numeric_features, temporal_features, price_history)
        for numeric_features, temporal_features, price_history in zip(
            numeric_features_list,
            temporal_features_list,
            price_history_list,
            strict=True,
        )
    ]

    assert len(batch_results) == len(sequential_results)
    for batch_result, sequential_result in zip(batch_results, sequential_results, strict=True):
        _assert_results_equivalent(batch_result, sequential_result)
