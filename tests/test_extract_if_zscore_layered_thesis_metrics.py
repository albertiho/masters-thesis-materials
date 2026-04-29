from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from research.training.scripts.extract_if_zscore_layered_thesis_metrics import (
    build_anomaly_case_metrics_table,
    build_case_frames,
    build_detector_metrics_table,
    build_metric_tables,
)


def _split_tables(
    split_name: str,
    *,
    anomaly_types: list[str | None],
    detector_predictions: dict[str, list[bool]],
    candidate_id: str = "candidate-1",
    scope_id: str | None = None,
    dataset_name: str = "COUNTRY_1",
    dataset_granularity: str = "country",
    scope_kind: str = "country",
    scope_market: str = "",
    mh_level: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    injected_rows = pd.DataFrame(
        {
            "run_id": ["run-1"] * len(anomaly_types),
            "candidate_id": [candidate_id] * len(anomaly_types),
            "dataset_split": [split_name] * len(anomaly_types),
            "evaluation_row_id": list(range(len(anomaly_types))),
            "ground_truth_label": [value is not None for value in anomaly_types],
            "is_injected": [value is not None for value in anomaly_types],
            "anomaly_type": anomaly_types,
            "dataset_name": [dataset_name] * len(anomaly_types),
            "dataset_granularity": [dataset_granularity] * len(anomaly_types),
            "mh_level": [mh_level] * len(anomaly_types),
            "scope_id": [scope_id or candidate_id] * len(anomaly_types),
            "scope_kind": [scope_kind] * len(anomaly_types),
            "scope_market": [scope_market] * len(anomaly_types),
        }
    )

    prediction_rows: list[dict[str, object]] = []
    for detector_name, predictions in detector_predictions.items():
        for evaluation_row_id, predicted_is_anomaly in enumerate(predictions):
            prediction_rows.append(
                {
                    "run_id": "run-1",
                    "candidate_id": candidate_id,
                    "dataset_split": split_name,
                    "evaluation_row_id": evaluation_row_id,
                    "detector_name": detector_name,
                    "predicted_is_anomaly": predicted_is_anomaly,
                }
            )

    predictions = pd.DataFrame(prediction_rows)
    return injected_rows, predictions


def test_build_detector_metrics_table_includes_combined_case() -> None:
    split_tables = {
        "new_prices": _split_tables(
            "new_prices",
            anomaly_types=[None, "price_spike"],
            detector_predictions={
                "Detector A": [False, True],
                "Detector B": [True, False],
            },
        ),
        "new_products": _split_tables(
            "new_products",
            anomaly_types=["zero_price", None],
            detector_predictions={
                "Detector A": [True, False],
                "Detector B": [False, False],
            },
        ),
    }

    case_frames = build_case_frames(split_tables)
    table = build_detector_metrics_table(case_frames)

    combined_a = table[
        (table["test_case_name"] == "combined")
        & (table["detector_combination"] == "Detector A")
    ].iloc[0]
    combined_b = table[
        (table["test_case_name"] == "combined")
        & (table["detector_combination"] == "Detector B")
    ].iloc[0]

    assert combined_a["precision"] == 1.0
    assert combined_a["recall"] == 1.0
    assert combined_a["f1"] == 1.0
    assert combined_a["true_negative_rate"] == 1.0
    assert combined_a["false_positive_rate"] == 0.0
    assert combined_a["false_negative_rate"] == 0.0
    assert combined_a["g_mean"] == 1.0

    assert combined_b["tp"] == 0
    assert combined_b["fp"] == 1
    assert combined_b["fn"] == 2
    assert combined_b["tn"] == 1
    assert combined_b["precision"] == 0.0
    assert combined_b["recall"] == 0.0
    assert combined_b["true_negative_rate"] == 0.5
    assert combined_b["false_positive_rate"] == 0.5
    assert combined_b["false_negative_rate"] == 1.0


def test_build_anomaly_case_metrics_table_uses_one_vs_rest_definition() -> None:
    split_tables = {
        "new_prices": _split_tables(
            "new_prices",
            anomaly_types=[None, "price_spike"],
            detector_predictions={"Detector A": [False, True]},
        ),
        "new_products": _split_tables(
            "new_products",
            anomaly_types=["zero_price", None],
            detector_predictions={"Detector A": [True, False]},
        ),
    }

    case_frames = build_case_frames(split_tables)
    table = build_anomaly_case_metrics_table(case_frames)

    price_spike_combined = table[
        (table["test_case_name"] == "combined")
        & (table["detector_combination"] == "Detector A")
        & (table["anomaly_case"] == "price_spike")
    ].iloc[0]

    assert price_spike_combined["tp"] == 1
    assert price_spike_combined["fp"] == 1
    assert price_spike_combined["fn"] == 0
    assert price_spike_combined["tn"] == 2
    assert price_spike_combined["precision"] == 0.5
    assert price_spike_combined["recall"] == 1.0
    assert price_spike_combined["true_negative_rate"] == 2 / 3
    assert price_spike_combined["false_positive_rate"] == 1 / 3
    assert price_spike_combined["false_negative_rate"] == 0.0


def test_build_case_frames_combines_split_frames_without_losing_rows() -> None:
    split_tables = {
        "new_prices": _split_tables(
            "new_prices",
            anomaly_types=[None, "price_spike"],
            detector_predictions={"Detector A": [False, True]},
        ),
        "new_products": _split_tables(
            "new_products",
            anomaly_types=["zero_price", None, None],
            detector_predictions={"Detector A": [True, False, False]},
        ),
    }

    case_frames = build_case_frames(split_tables)

    assert len(case_frames["new_prices"]) == 2
    assert len(case_frames["new_products"]) == 3
    assert len(case_frames["combined"]) == 5


def test_build_detector_metrics_table_weights_combined_f1_and_g_mean_by_split() -> None:
    split_tables = {
        "new_prices": _split_tables(
            "new_prices",
            anomaly_types=["price_spike", None],
            detector_predictions={"Detector A": [True, False]},
        ),
        "new_products": _split_tables(
            "new_products",
            anomaly_types=["zero_price", None],
            detector_predictions={"Detector A": [False, False]},
        ),
    }

    case_frames = build_case_frames(split_tables)
    table = build_detector_metrics_table(case_frames)

    new_prices = table[
        (table["test_case_name"] == "new_prices")
        & (table["detector_combination"] == "Detector A")
    ].iloc[0]
    new_products = table[
        (table["test_case_name"] == "new_products")
        & (table["detector_combination"] == "Detector A")
    ].iloc[0]
    combined = table[
        (table["test_case_name"] == "combined")
        & (table["detector_combination"] == "Detector A")
    ].iloc[0]

    assert new_prices["f1"] == 1.0
    assert new_prices["g_mean"] == 1.0
    assert new_products["f1"] == 0.0
    assert new_products["g_mean"] == 0.0
    assert combined["f1"] == 0.7
    assert combined["g_mean"] == 0.7


def test_build_detector_metrics_table_keeps_combined_metrics_separate_by_scope() -> None:
    split_tables = {
        "new_prices": (
            pd.concat(
                [
                    _split_tables(
                        "new_prices",
                        anomaly_types=[None, "price_spike"],
                        detector_predictions={"Detector A": [False, True]},
                        candidate_id="COUNTRY_1",
                        dataset_name="COUNTRY_1",
                        dataset_granularity="country",
                        scope_kind="country",
                    )[0],
                    _split_tables(
                        "new_prices",
                        anomaly_types=[None, "price_spike"],
                        detector_predictions={"Detector A": [False, False]},
                        candidate_id="COMPETITOR_1_COUNTRY_1",
                        dataset_name="COMPETITOR_1_COUNTRY_1",
                        dataset_granularity="competitor",
                        scope_kind="competitor",
                        scope_market="B2B",
                    )[0],
                ],
                ignore_index=True,
            ),
            pd.concat(
                [
                    _split_tables(
                        "new_prices",
                        anomaly_types=[None, "price_spike"],
                        detector_predictions={"Detector A": [False, True]},
                        candidate_id="COUNTRY_1",
                        dataset_name="COUNTRY_1",
                        dataset_granularity="country",
                        scope_kind="country",
                    )[1],
                    _split_tables(
                        "new_prices",
                        anomaly_types=[None, "price_spike"],
                        detector_predictions={"Detector A": [False, False]},
                        candidate_id="COMPETITOR_1_COUNTRY_1",
                        dataset_name="COMPETITOR_1_COUNTRY_1",
                        dataset_granularity="competitor",
                        scope_kind="competitor",
                        scope_market="B2B",
                    )[1],
                ],
                ignore_index=True,
            ),
        ),
        "new_products": (
            pd.concat(
                [
                    _split_tables(
                        "new_products",
                        anomaly_types=["zero_price", None],
                        detector_predictions={"Detector A": [True, False]},
                        candidate_id="COUNTRY_1",
                        dataset_name="COUNTRY_1",
                        dataset_granularity="country",
                        scope_kind="country",
                    )[0],
                    _split_tables(
                        "new_products",
                        anomaly_types=["zero_price", None],
                        detector_predictions={"Detector A": [False, False]},
                        candidate_id="COMPETITOR_1_COUNTRY_1",
                        dataset_name="COMPETITOR_1_COUNTRY_1",
                        dataset_granularity="competitor",
                        scope_kind="competitor",
                        scope_market="B2B",
                    )[0],
                ],
                ignore_index=True,
            ),
            pd.concat(
                [
                    _split_tables(
                        "new_products",
                        anomaly_types=["zero_price", None],
                        detector_predictions={"Detector A": [True, False]},
                        candidate_id="COUNTRY_1",
                        dataset_name="COUNTRY_1",
                        dataset_granularity="country",
                        scope_kind="country",
                    )[1],
                    _split_tables(
                        "new_products",
                        anomaly_types=["zero_price", None],
                        detector_predictions={"Detector A": [False, False]},
                        candidate_id="COMPETITOR_1_COUNTRY_1",
                        dataset_name="COMPETITOR_1_COUNTRY_1",
                        dataset_granularity="competitor",
                        scope_kind="competitor",
                        scope_market="B2B",
                    )[1],
                ],
                ignore_index=True,
            ),
        ),
    }

    case_frames = build_case_frames(split_tables)
    table = build_detector_metrics_table(case_frames)

    country_row = table[
        (table["scope_id"] == "COUNTRY_1")
        & (table["test_case_name"] == "combined")
        & (table["detector_combination"] == "Detector A")
    ].iloc[0]
    competitor_row = table[
        (table["scope_id"] == "COMPETITOR_1_COUNTRY_1")
        & (table["test_case_name"] == "combined")
        & (table["detector_combination"] == "Detector A")
    ].iloc[0]

    assert country_row["dataset_granularity"] == "country"
    assert country_row["precision"] == 1.0
    assert country_row["recall"] == 1.0
    assert competitor_row["dataset_granularity"] == "competitor"
    assert competitor_row["precision"] == 0.0
    assert competitor_row["recall"] == 0.0


def test_build_metric_tables_reads_completed_scope_roots(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    country_root = run_root / "scopes" / "COUNTRY_1"
    competitor_root = run_root / "scopes" / "COMPETITOR_1_COUNTRY_1"

    country_splits = {
        "new_prices": _split_tables(
            "new_prices",
            anomaly_types=[None, "price_spike"],
            detector_predictions={"Detector A": [False, True]},
            candidate_id="COUNTRY_1",
            dataset_name="COUNTRY_1",
            dataset_granularity="country",
            scope_kind="country",
        ),
        "new_products": _split_tables(
            "new_products",
            anomaly_types=["zero_price", None],
            detector_predictions={"Detector A": [True, False]},
            candidate_id="COUNTRY_1",
            dataset_name="COUNTRY_1",
            dataset_granularity="country",
            scope_kind="country",
        ),
    }
    competitor_splits = {
        "new_prices": _split_tables(
            "new_prices",
            anomaly_types=[None, "price_spike"],
            detector_predictions={"Detector A": [False, False]},
            candidate_id="COMPETITOR_1_COUNTRY_1",
            dataset_name="COMPETITOR_1_COUNTRY_1",
            dataset_granularity="competitor",
            scope_kind="competitor",
            scope_market="B2B",
        ),
        "new_products": _split_tables(
            "new_products",
            anomaly_types=["zero_price", None],
            detector_predictions={"Detector A": [False, False]},
            candidate_id="COMPETITOR_1_COUNTRY_1",
            dataset_name="COMPETITOR_1_COUNTRY_1",
            dataset_granularity="competitor",
            scope_kind="competitor",
            scope_market="B2B",
        ),
    }

    for scope_root, split_tables in (
        (country_root, country_splits),
        (competitor_root, competitor_splits),
    ):
        for split_name, (injected_rows, predictions) in split_tables.items():
            split_root = scope_root / "splits" / split_name
            split_root.mkdir(parents=True, exist_ok=True)
            injected_rows.to_parquet(split_root / "injected_rows.parquet", index=False)
            predictions.to_parquet(split_root / "predictions.parquet", index=False)

    detector_metrics, anomaly_case_metrics = build_metric_tables(run_root)

    assert set(detector_metrics["scope_id"]) == {"COUNTRY_1", "COMPETITOR_1_COUNTRY_1"}
    assert set(anomaly_case_metrics["scope_id"]) == {"COUNTRY_1", "COMPETITOR_1_COUNTRY_1"}

    country_combined = detector_metrics[
        (detector_metrics["scope_id"] == "COUNTRY_1")
        & (detector_metrics["test_case_name"] == "combined")
    ].iloc[0]
    competitor_combined = detector_metrics[
        (detector_metrics["scope_id"] == "COMPETITOR_1_COUNTRY_1")
        & (detector_metrics["test_case_name"] == "combined")
    ].iloc[0]

    assert country_combined["f1"] == 1.0
    assert competitor_combined["f1"] == 0.0


def test_build_detector_metrics_table_keeps_combined_metrics_separate_by_mh_level() -> None:
    split_tables = {
        "new_prices": (
            pd.concat(
                [
                    _split_tables(
                        "new_prices",
                        anomaly_types=[None, "price_spike"],
                        detector_predictions={"Detector A": [False, True]},
                        candidate_id="COMPETITOR_1_COUNTRY_1__mh5",
                        scope_id="COMPETITOR_1_COUNTRY_1",
                        dataset_name="COMPETITOR_1_COUNTRY_1",
                        dataset_granularity="competitor",
                        scope_kind="competitor",
                        scope_market="B2B",
                        mh_level="mh5",
                    )[0],
                    _split_tables(
                        "new_prices",
                        anomaly_types=[None, "price_spike"],
                        detector_predictions={"Detector A": [False, False]},
                        candidate_id="COMPETITOR_1_COUNTRY_1__mh10",
                        scope_id="COMPETITOR_1_COUNTRY_1",
                        dataset_name="COMPETITOR_1_COUNTRY_1",
                        dataset_granularity="competitor",
                        scope_kind="competitor",
                        scope_market="B2B",
                        mh_level="mh10",
                    )[0],
                ],
                ignore_index=True,
            ),
            pd.concat(
                [
                    _split_tables(
                        "new_prices",
                        anomaly_types=[None, "price_spike"],
                        detector_predictions={"Detector A": [False, True]},
                        candidate_id="COMPETITOR_1_COUNTRY_1__mh5",
                        scope_id="COMPETITOR_1_COUNTRY_1",
                        dataset_name="COMPETITOR_1_COUNTRY_1",
                        dataset_granularity="competitor",
                        scope_kind="competitor",
                        scope_market="B2B",
                        mh_level="mh5",
                    )[1],
                    _split_tables(
                        "new_prices",
                        anomaly_types=[None, "price_spike"],
                        detector_predictions={"Detector A": [False, False]},
                        candidate_id="COMPETITOR_1_COUNTRY_1__mh10",
                        scope_id="COMPETITOR_1_COUNTRY_1",
                        dataset_name="COMPETITOR_1_COUNTRY_1",
                        dataset_granularity="competitor",
                        scope_kind="competitor",
                        scope_market="B2B",
                        mh_level="mh10",
                    )[1],
                ],
                ignore_index=True,
            ),
        ),
        "new_products": (
            pd.concat(
                [
                    _split_tables(
                        "new_products",
                        anomaly_types=["zero_price", None],
                        detector_predictions={"Detector A": [True, False]},
                        candidate_id="COMPETITOR_1_COUNTRY_1__mh5",
                        scope_id="COMPETITOR_1_COUNTRY_1",
                        dataset_name="COMPETITOR_1_COUNTRY_1",
                        dataset_granularity="competitor",
                        scope_kind="competitor",
                        scope_market="B2B",
                        mh_level="mh5",
                    )[0],
                    _split_tables(
                        "new_products",
                        anomaly_types=["zero_price", None],
                        detector_predictions={"Detector A": [False, False]},
                        candidate_id="COMPETITOR_1_COUNTRY_1__mh10",
                        scope_id="COMPETITOR_1_COUNTRY_1",
                        dataset_name="COMPETITOR_1_COUNTRY_1",
                        dataset_granularity="competitor",
                        scope_kind="competitor",
                        scope_market="B2B",
                        mh_level="mh10",
                    )[0],
                ],
                ignore_index=True,
            ),
            pd.concat(
                [
                    _split_tables(
                        "new_products",
                        anomaly_types=["zero_price", None],
                        detector_predictions={"Detector A": [True, False]},
                        candidate_id="COMPETITOR_1_COUNTRY_1__mh5",
                        scope_id="COMPETITOR_1_COUNTRY_1",
                        dataset_name="COMPETITOR_1_COUNTRY_1",
                        dataset_granularity="competitor",
                        scope_kind="competitor",
                        scope_market="B2B",
                        mh_level="mh5",
                    )[1],
                    _split_tables(
                        "new_products",
                        anomaly_types=["zero_price", None],
                        detector_predictions={"Detector A": [False, False]},
                        candidate_id="COMPETITOR_1_COUNTRY_1__mh10",
                        scope_id="COMPETITOR_1_COUNTRY_1",
                        dataset_name="COMPETITOR_1_COUNTRY_1",
                        dataset_granularity="competitor",
                        scope_kind="competitor",
                        scope_market="B2B",
                        mh_level="mh10",
                    )[1],
                ],
                ignore_index=True,
            ),
        ),
    }

    case_frames = build_case_frames(split_tables)
    table = build_detector_metrics_table(case_frames)

    mh5_row = table[
        (table["scope_id"] == "COMPETITOR_1_COUNTRY_1")
        & (table["mh_level"] == "mh5")
        & (table["test_case_name"] == "combined")
    ].iloc[0]
    mh10_row = table[
        (table["scope_id"] == "COMPETITOR_1_COUNTRY_1")
        & (table["mh_level"] == "mh10")
        & (table["test_case_name"] == "combined")
    ].iloc[0]

    assert mh5_row["f1"] == 1.0
    assert mh10_row["f1"] == 0.0


def test_build_case_frames_allows_empty_scope_artifacts() -> None:
    split_tables = {
        "new_prices": (
            pd.DataFrame(
                columns=[
                    "run_id",
                    "candidate_id",
                    "dataset_split",
                    "evaluation_row_id",
                    "ground_truth_label",
                    "is_injected",
                    "anomaly_type",
                    "dataset_name",
                    "dataset_granularity",
                    "mh_level",
                    "scope_id",
                    "scope_kind",
                    "scope_market",
                ]
            ),
            pd.DataFrame(
                columns=[
                    "mh_level",
                    "scope_id",
                    "scope_kind",
                    "scope_market",
                    "country_if_model_name",
                    "country_if_config_source",
                ]
            ),
        ),
        "new_products": (
            pd.DataFrame(
                columns=[
                    "run_id",
                    "candidate_id",
                    "dataset_split",
                    "evaluation_row_id",
                    "ground_truth_label",
                    "is_injected",
                    "anomaly_type",
                    "dataset_name",
                    "dataset_granularity",
                    "mh_level",
                    "scope_id",
                    "scope_kind",
                    "scope_market",
                ]
            ),
            pd.DataFrame(
                columns=[
                    "mh_level",
                    "scope_id",
                    "scope_kind",
                    "scope_market",
                    "country_if_model_name",
                    "country_if_config_source",
                ]
            ),
        ),
    }

    case_frames = build_case_frames(split_tables)
    detector_table = build_detector_metrics_table(case_frames)
    anomaly_table = build_anomaly_case_metrics_table(case_frames)

    assert len(case_frames["combined"]) == 0
    assert detector_table.empty
    assert anomaly_table.empty
