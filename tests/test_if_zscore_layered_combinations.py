from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

import research.training.scripts.analyze_if_zscore_layered_combinations as layered_combinations
from research.training.scripts.analyze_if_zscore_layered_combinations import (
    ANOMALY_TYPES,
    COUNTRY_NAME,
    DEFAULT_COMBINATIONS,
    EVALUATION_SPLITS,
    IF_ONLY_NAME,
    IF_ZSCORE_5050_NAME,
    ScopeSpec,
    SANITY_IF_NAME,
    SANITY_ONLY_NAME,
    SANITY_ZSCORE_GT10_IF_NAME,
    SANITY_ZSCORE_GT5_IF_NAME,
    SANITY_ZSCORE_IF_5050_NAME,
    SANITY_ZSCORE_NAME,
    ZSCORE_ONLY_NAME,
    WeightedScoreCombinedDetector,
    build_dataset_paths,
    build_scope_specs,
    create_evaluators,
    create_if_zscore_5050_detector,
    create_sanity_if_detector,
    create_sanity_only_detector,
    create_sanity_zscore_detector,
    create_sanity_zscore_if_detector,
    create_sanity_zscore_if_5050_detector,
)
from src.anomaly.combined import CombinedDetectorConfig, DetectionContext, DetectorLayer
from src.anomaly.statistical import AnomalyResult, AnomalySeverity, AnomalyType, SanityCheckDetector
from src.research.evaluation.synthetic import SyntheticAnomalyType
from src.features.numeric import NumericFeatures
from src.features.temporal import TemporalFeatures
from src.research.evaluation.detector_evaluator import DetectorEvaluator


def _numeric_features(
    price: float = 100.0,
    *,
    country: str = "FI",
    currency: str | None = None,
) -> NumericFeatures:
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
        country=country,
        currency=currency,
    )


def _temporal_features(observation_count: int = 10) -> TemporalFeatures:
    return TemporalFeatures(
        rolling_mean=100.0,
        rolling_std=10.0,
        rolling_min=90.0,
        rolling_max=110.0,
        price_zscore=0.0,
        price_change_pct=0.0,
        days_since_change=1.0,
        observation_count=observation_count,
        has_sufficient_history=observation_count >= 3,
        competitor_product_id="product-1",
        competitor="competitor-1",
    )


def _context(observation_count: int = 10) -> DetectionContext:
    return DetectionContext.from_features(
        numeric_features=_numeric_features(),
        temporal_features=_temporal_features(observation_count),
        price_history=[95.0, 100.0, 105.0][: max(0, min(observation_count, 3))],
        observation_count=observation_count,
    )


class StaticDetector:
    def __init__(
        self,
        *,
        name: str,
        score: float,
        is_anomaly: bool,
        details: dict[str, object] | None = None,
    ) -> None:
        self.name = name
        self.score = score
        self.is_anomaly = is_anomaly
        self.details = details or {}

    def detect(self, numeric_features: NumericFeatures, temporal_features: TemporalFeatures) -> AnomalyResult:
        del temporal_features
        return AnomalyResult(
            is_anomaly=self.is_anomaly,
            anomaly_score=self.score,
            anomaly_types=[AnomalyType.PRICE_ZSCORE] if self.is_anomaly else [],
            severity=AnomalySeverity.LOW if self.is_anomaly else None,
            details=dict(self.details),
            detector=self.name,
            competitor_product_id=numeric_features.competitor_product_id,
            competitor=numeric_features.competitor,
        )


class StubPersistence:
    def __init__(self, detector: object) -> None:
        self.detector = detector
        self.loaded_model_names: list[str] = []

    def load_isolation_forest(self, model_name: str) -> object:
        self.loaded_model_names.append(model_name)
        return self.detector


class MissingModelPersistence(StubPersistence):
    def load_isolation_forest(self, model_name: str) -> object:
        self.loaded_model_names.append(model_name)
        raise FileNotFoundError(model_name)


def test_weighted_score_combined_detector_uses_equal_average_threshold() -> None:
    detector = WeightedScoreCombinedDetector(
        config=CombinedDetectorConfig(
            name="weighted",
            detector_weights={"if_stub": 0.5, "z_stub": 0.5},
        ),
        layers=[
            DetectorLayer(
                name="ensemble",
                detectors=[
                    StaticDetector(name="if_stub", score=0.8, is_anomaly=True),
                    StaticDetector(name="z_stub", score=0.2, is_anomaly=False),
                ],
            )
        ],
        decision_threshold=0.5,
    )

    result = detector.detect(_context())

    assert result.is_anomaly is True
    assert result.anomaly_score == pytest.approx(0.5)
    assert result.details["weighted_score"] == pytest.approx(0.5)
    assert result.details["active_detector_weights"] == {"if_stub": 0.5, "z_stub": 0.5}


def test_weighted_score_combined_detector_skips_insufficient_history_results() -> None:
    detector = WeightedScoreCombinedDetector(
        config=CombinedDetectorConfig(
            name="weighted",
            detector_weights={"if_stub": 0.5, "z_stub": 0.5},
        ),
        layers=[
            DetectorLayer(
                name="ensemble",
                detectors=[
                    StaticDetector(name="if_stub", score=0.6, is_anomaly=True),
                    StaticDetector(
                        name="z_stub",
                        score=0.0,
                        is_anomaly=False,
                        details={"insufficient_history": True},
                    ),
                ],
            )
        ],
        decision_threshold=0.5,
    )

    result = detector.detect(_context(observation_count=2))

    assert result.is_anomaly is True
    assert result.anomaly_score == pytest.approx(0.6)
    assert result.details["active_detectors"] == ["if_stub"]
    assert result.details["skipped_results"] == [
        {"detector": "z_stub", "reason": "insufficient_history"}
    ]


def test_sanity_only_detector_uses_requested_single_gate_layer() -> None:
    detector = create_sanity_only_detector(name=SANITY_ONLY_NAME)

    layers = detector.layers

    assert len(layers) == 1
    assert layers[0].name == "sanity"
    assert layers[0].is_gate is True


def test_sanity_if_detector_uses_requested_layer_order() -> None:
    detector = create_sanity_if_detector(
        StaticDetector(name="isolation_forest", score=0.6, is_anomaly=True),
        name=SANITY_IF_NAME,
    )

    layers = detector.layers

    assert len(layers) == 2
    assert [layer.name for layer in layers] == ["sanity", "iforest"]
    assert layers[0].is_gate is True
    assert layers[1].required_history == 0


def test_sanity_zscore_detector_uses_requested_layer_order() -> None:
    detector = create_sanity_zscore_detector(name=SANITY_ZSCORE_NAME)

    layers = detector.layers

    assert [layer.name for layer in layers] == ["sanity", "zscore"]
    assert layers[0].is_gate is True
    assert layers[1].required_history == 0


def test_if_zscore_5050_detector_uses_single_weighted_ensemble_layer() -> None:
    detector = create_if_zscore_5050_detector(
        StaticDetector(name="isolation_forest", score=0.6, is_anomaly=True),
        name=IF_ZSCORE_5050_NAME,
    )

    layers = detector.layers

    assert [layer.name for layer in layers] == ["if_zscore_5050"]
    assert layers[0].layer_type == "ensemble"


def test_sanity_zscore_if_5050_detector_uses_requested_layers_and_weights() -> None:
    detector = create_sanity_zscore_if_5050_detector(
        StaticDetector(name="isolation_forest", score=0.6, is_anomaly=True),
        name=SANITY_ZSCORE_IF_5050_NAME,
    )

    layers = detector.layers

    assert [layer.name for layer in layers] == ["sanity", "zscore", "iforest"]
    assert layers[0].is_gate is True
    assert detector.config.detector_weights == {
        "sanity": 0.0,
        "isolation_forest": 0.5,
        "zscore": 0.5,
    }


def test_sanity_zscore_gt5_if_detector_uses_requested_history_gate() -> None:
    detector = create_sanity_zscore_if_detector(
        StaticDetector(name="isolation_forest", score=0.6, is_anomaly=True),
        zscore_required_history=5,
        name=SANITY_ZSCORE_GT5_IF_NAME,
    )

    layers = detector.layers

    assert [layer.name for layer in layers] == ["sanity", "zscore", "iforest"]
    assert layers[1].required_history == 5


def test_sanity_zscore_gt10_if_detector_uses_requested_history_gate() -> None:
    detector = create_sanity_zscore_if_detector(
        StaticDetector(name="isolation_forest", score=0.6, is_anomaly=True),
        zscore_required_history=10,
        name=SANITY_ZSCORE_GT10_IF_NAME,
    )

    layers = detector.layers

    assert [layer.name for layer in layers] == ["sanity", "zscore", "iforest"]
    assert layers[1].required_history == 10


def test_sanity_detector_flags_anonymized_country_currency_mismatch() -> None:
    detector = SanityCheckDetector()

    result = detector.detect(
        _numeric_features(
            country="COUNTRY_1",
            currency="CURRENCY_5",
        )
    )

    assert result.is_anomaly is True
    assert AnomalyType.CURRENCY_MISMATCH in result.anomaly_types
    assert result.details["currency_mismatch"] == {
        "country": "COUNTRY_1",
        "currency": "CURRENCY_5",
        "expected_currency": "CURRENCY_1",
    }


def test_detector_evaluator_passes_currency_to_sanity_detector() -> None:
    evaluator = DetectorEvaluator(
        SanityCheckDetector(),
        name="sanity",
        enable_persistence_acceptance=False,
    )
    frame = pd.DataFrame(
        [
            {
                "product_id": "product-1",
                "competitor_id": "competitor-1",
                "competitor_product_id": "product-1",
                "price": 100.0,
                "list_price": None,
                "currency": "CURRENCY_5",
                "first_seen_at": pd.Timestamp("2026-02-08T00:00:00Z"),
            }
        ]
    )
    row = next(frame.itertuples(index=False))
    col_map = {column: index for index, column in enumerate(frame.columns)}

    result = evaluator.process_row(row, col_map, country="COUNTRY_1")

    assert result.is_anomaly is True
    assert AnomalyType.CURRENCY_MISMATCH in result.anomaly_types


def test_create_evaluators_allows_zscore_only_without_persistence() -> None:
    evaluators = create_evaluators(
        persistence=None,
        model_name="COUNTRY_1_B2C_mh5",
        combinations=[ZSCORE_ONLY_NAME],
    )

    assert [evaluator.name for evaluator in evaluators] == [ZSCORE_ONLY_NAME]


def test_create_evaluators_requires_persistence_for_if_variants() -> None:
    with pytest.raises(ValueError, match="require model persistence"):
        create_evaluators(
            persistence=None,
            model_name="COUNTRY_1_B2C_mh5",
            combinations=[IF_ONLY_NAME],
        )


def test_create_evaluators_loads_iforest_once_for_requested_if_variants() -> None:
    persistence = StubPersistence(
        StaticDetector(name="isolation_forest", score=0.6, is_anomaly=True)
    )

    evaluators = create_evaluators(
        persistence=persistence,  # type: ignore[arg-type]
        model_name="COUNTRY_1_B2C_mh5",
        combinations=[IF_ONLY_NAME, ZSCORE_ONLY_NAME],
    )

    assert [evaluator.name for evaluator in evaluators] == [IF_ONLY_NAME, ZSCORE_ONLY_NAME]
    assert persistence.loaded_model_names == ["COUNTRY_1_B2C_mh5"]


def test_create_evaluators_uses_preloaded_iforest_without_persistence() -> None:
    preloaded_iforest = StaticDetector(name="isolation_forest", score=0.6, is_anomaly=True)

    evaluators = create_evaluators(
        persistence=None,
        model_name="COUNTRY_1_mh5",
        iforest_detector=preloaded_iforest,
        combinations=[IF_ONLY_NAME, ZSCORE_ONLY_NAME],
    )

    assert [evaluator.name for evaluator in evaluators] == [IF_ONLY_NAME, ZSCORE_ONLY_NAME]


def test_build_scope_specs_covers_country_and_country1_competitors() -> None:
    scope_specs = build_scope_specs()

    assert [scope.scope_id for scope in scope_specs] == [
        "COUNTRY_1",
        "COMPETITOR_1_COUNTRY_1",
        "COMPETITOR_2_COUNTRY_1",
        "COMPETITOR_3_COUNTRY_1",
    ]
    assert [scope.dataset_granularity for scope in scope_specs] == [
        "country",
        "competitor",
        "competitor",
        "competitor",
    ]
    assert [scope.iforest_model_name for scope in scope_specs] == [
        "COUNTRY_1_mh5",
        "COMPETITOR_1_COUNTRY_1_mh5",
        "COMPETITOR_2_COUNTRY_1_mh5",
        "COMPETITOR_3_COUNTRY_1_mh5",
    ]


def test_build_dataset_paths_targets_country1_country_level_files() -> None:
    country_scope = build_scope_specs()[0]
    dataset_paths = build_dataset_paths(country_scope)

    expected_root = TEST_ROOT / "data-subsets" / "mh5" / "by_country" / "COUNTRY_1"
    assert dataset_paths["train"] == expected_root / "COUNTRY_1_2026-02-08_train.parquet"
    assert dataset_paths["test_new_prices"] == (
        expected_root / "COUNTRY_1_2026-02-08_test_new_prices.parquet"
    )
    assert dataset_paths["test_new_products"] == (
        expected_root / "COUNTRY_1_2026-02-08_test_new_products.parquet"
    )


def test_build_dataset_paths_targets_competitor_level_files() -> None:
    competitor_scope = build_scope_specs()[1]
    dataset_paths = build_dataset_paths(competitor_scope)

    expected_root = TEST_ROOT / "data-subsets" / "mh5" / "by_competitor" / "COUNTRY_1" / "B2B"
    assert dataset_paths["train"] == expected_root / "COMPETITOR_1_COUNTRY_1_2026-02-08_train.parquet"
    assert dataset_paths["test_new_prices"] == (
        expected_root / "COMPETITOR_1_COUNTRY_1_2026-02-08_test_new_prices.parquet"
    )
    assert dataset_paths["test_new_products"] == (
        expected_root / "COMPETITOR_1_COUNTRY_1_2026-02-08_test_new_products.parquet"
    )


def test_hardcoded_run_configuration_uses_country1_scopes_and_both_splits() -> None:
    assert COUNTRY_NAME == "COUNTRY_1"
    assert EVALUATION_SPLITS == ("test_new_prices", "test_new_products")


def test_load_or_train_iforest_reuses_existing_model_without_training(monkeypatch: pytest.MonkeyPatch) -> None:
    persistence = StubPersistence(
        StaticDetector(name="isolation_forest", score=0.6, is_anomaly=True)
    )
    train_called = {"value": False}

    def fake_extract_features_vectorized(train_df: pd.DataFrame) -> pd.DataFrame:
        train_called["value"] = True
        return train_df

    def fake_train_from_matrix(features: pd.DataFrame) -> tuple[object, object]:
        train_called["value"] = True
        return object(), object()

    monkeypatch.setattr(layered_combinations, "extract_features_vectorized", fake_extract_features_vectorized)
    monkeypatch.setattr(layered_combinations, "train_from_matrix", fake_train_from_matrix)

    detector = layered_combinations.load_or_train_iforest(
        pd.DataFrame({"price": [100.0]}),
        persistence=persistence,  # type: ignore[arg-type]
        model_name="COMPETITOR_1_COUNTRY_1_mh5",
    )

    assert detector is persistence.detector
    assert persistence.loaded_model_names == ["COMPETITOR_1_COUNTRY_1_mh5"]
    assert train_called["value"] is False


def test_default_combinations_match_requested_next_testset() -> None:
    assert DEFAULT_COMBINATIONS == [
        IF_ONLY_NAME,
        ZSCORE_ONLY_NAME,
        SANITY_ONLY_NAME,
        SANITY_IF_NAME,
        SANITY_ZSCORE_NAME,
        IF_ZSCORE_5050_NAME,
        SANITY_ZSCORE_IF_5050_NAME,
        SANITY_ZSCORE_GT5_IF_NAME,
        SANITY_ZSCORE_GT10_IF_NAME,
    ]


def test_anomaly_types_match_trimmed_country1_run_set() -> None:
    assert ANOMALY_TYPES == [
        SyntheticAnomalyType.PRICE_SPIKE,
        SyntheticAnomalyType.PRICE_DROP,
        SyntheticAnomalyType.PRICE_NOISE,
        SyntheticAnomalyType.LIST_PRICE_VIOLATION,
        SyntheticAnomalyType.ZERO_PRICE,
        SyntheticAnomalyType.NEGATIVE_PRICE,
        SyntheticAnomalyType.EXTREME_OUTLIER,
        SyntheticAnomalyType.DECIMAL_SHIFT,
        SyntheticAnomalyType.CURRENCY_SWAP,
    ]


def test_inject_split_frame_passes_explicit_anomaly_list(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_inject_anomalies_to_dataframe(
        frame: pd.DataFrame,
        **kwargs: object,
    ) -> tuple[pd.DataFrame, np.ndarray, list[dict[str, object]]]:
        captured["frame"] = frame.copy()
        captured["kwargs"] = kwargs
        return frame.copy(), np.zeros(len(frame), dtype=bool), []

    monkeypatch.setattr(
        layered_combinations,
        "inject_anomalies_to_dataframe",
        fake_inject_anomalies_to_dataframe,
    )

    frame = pd.DataFrame({"price": [100.0, 110.0]})
    injected_frame, labels, injection_details = layered_combinations.inject_split_frame(
        frame,
        split_index=1,
    )

    assert injected_frame.equals(frame)
    assert labels.tolist() == [False, False]
    assert injection_details == []
    assert captured["kwargs"] == {
        "injection_rate": 0.1,
        "seed": 43,
        "spike_range": (2.0, 5.0),
        "drop_range": (0.1, 0.5),
        "anomaly_types": ANOMALY_TYPES,
    }
