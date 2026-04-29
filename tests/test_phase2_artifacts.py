from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from research.training.scripts.analyze_detector_combinations import (
    AnalysisRunResult,
    aggregate_from_canonical_tables,
    build_canonical_run_artifacts,
)
from research.training.scripts.compare_detectors import (
    ModelComparison,
    build_comparison_run_metadata,
    build_canonical_split_artifacts,
)
from research.training.scripts.compare_granularity_models import (
    DatasetFiles,
    EvaluationTask,
    TaskEvaluationResult,
    _build_run_artifacts,
)
from research.training.scripts.grid_search_autoencoder import (
    CandidateEvaluationResult as AECandidateEvaluationResult,
    GridRow as AEGridRow,
    build_candidate_metrics_frame as build_ae_candidate_metrics_frame,
    persist_candidate_run_artifacts as persist_ae_candidate_run_artifacts,
    write_candidate_run_artifacts as write_ae_candidate_run_artifacts,
)
from research.training.scripts.grid_search_isolation_forest import (
    CandidateEvaluationResult as IFCandidateEvaluationResult,
    GridRow as IFGridRow,
    build_candidate_metrics_frame as build_if_candidate_metrics_frame,
    persist_candidate_run_artifacts as persist_if_candidate_run_artifacts,
    write_candidate_run_artifacts as write_if_candidate_run_artifacts,
)
from research.training.scripts.validate_anomaly_detection import build_detection_matrix_frame
from src.anomaly.statistical import AnomalyResult, AnomalySeverity, AnomalyType
from src.research.artifacts import (
    build_tuning_summary,
    comparison_result_to_tables,
    write_evaluation_run,
    write_tuning_sweep,
)
from src.research.evaluation.test_orchestrator import ComparisonResult, DetectorMetrics


def _make_comparison_result(
    detector_predictions: dict[str, list[bool]],
    *,
    detector_scores: dict[str, list[float]] | None = None,
) -> ComparisonResult:
    labels = np.array([False, True, True], dtype=bool)
    frame = pd.DataFrame(
        {
            "first_seen_at": pd.to_datetime(
                ["2026-03-02T00:00:00Z", "2026-03-03T00:00:00Z", "2026-03-04T00:00:00Z"]
            ),
            "product_id": ["product_1", "product_2", "product_3"],
            "competitor_id": ["competitor_a", "competitor_a", "competitor_a"],
            "competitor_product_id": ["comp_1", "comp_2", "comp_3"],
            "price": [100.0, 150.0, 0.0],
            "source_row_index": [2, 0, 1],
            "ground_truth_label": labels,
            "is_injected": labels,
            "anomaly_type": [None, "price_spike", "zero_price"],
            "injection_strategy": [
                "synthetic_dataframe_injection",
                "synthetic_dataframe_injection",
                "synthetic_dataframe_injection",
            ],
            "injection_phase": pd.Series([pd.NA, 1, 2], dtype="Int64"),
            "injection_seed": pd.Series([42, 42, 42], dtype="Int64"),
            "injection_params_json": ['{}', '{"kind":"spike"}', '{"kind":"zero"}'],
            "original_price": [100.0, 100.0, 75.0],
        }
    )

    raw_results: dict[str, list[AnomalyResult]] = {}
    metrics: dict[str, DetectorMetrics] = {}
    score_map = detector_scores or {}

    for detector_name, predictions in detector_predictions.items():
        scores = score_map.get(detector_name, [0.1, 0.9, 0.2])
        detector_results: list[AnomalyResult] = []
        for idx, predicted in enumerate(predictions):
            details = {
                "feature_valid": True,
                "accepted_via_persistence": False,
                "reconstruction_error": float(scores[idx]),
                "threshold": 0.5,
            }
            detector_results.append(
                AnomalyResult(
                    is_anomaly=bool(predicted),
                    anomaly_score=float(scores[idx]),
                    anomaly_types=[AnomalyType.PRICE_ZSCORE] if predicted else [],
                    severity=AnomalySeverity.LOW if predicted else None,
                    details=details,
                    detector=detector_name,
                    competitor_product_id=f"comp_{idx + 1}",
                    competitor="competitor_a",
                )
            )

        raw_results[detector_name] = detector_results
        predicted = np.asarray(predictions, dtype=bool)
        tp = int((predicted & labels).sum())
        fp = int((predicted & ~labels).sum())
        fn = int((~predicted & labels).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        metrics[detector_name] = DetectorMetrics(
            detector_name=detector_name,
            precision=precision,
            recall=recall,
            f1=f1,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            n_samples=len(labels),
            predictions=predicted,
            scores=np.asarray(scores, dtype=float),
        )

    return ComparisonResult(
        metrics=metrics,
        raw_results=raw_results,
        observation_counts=np.array([1, 2, 3], dtype=np.int32),
        labels=labels,
        df_sorted=frame,
    )


def _if_candidate(run_id: str, *, new_prices: list[bool], new_products: list[bool]) -> IFCandidateEvaluationResult:
    prices_comparison = _make_comparison_result({"iforest": new_prices})
    products_comparison = _make_comparison_result({"iforest": new_products})
    prices_metrics = prices_comparison.metrics["iforest"]
    products_metrics = products_comparison.metrics["iforest"]
    combined_precision = (prices_metrics.precision + products_metrics.precision) / 2
    combined_recall = (prices_metrics.recall + products_metrics.recall) / 2
    combined_f1 = (prices_metrics.f1 + products_metrics.f1) / 2

    return IFCandidateEvaluationResult(
        row=IFGridRow(
            run_id=run_id,
            n_estimators=100,
            max_samples="auto",
            max_features=1.0,
            anomaly_threshold=0.4,
            contamination="auto",
            precision=combined_precision,
            recall=combined_recall,
            f1_score=combined_f1,
            precision_new_prices=prices_metrics.precision,
            recall_new_prices=prices_metrics.recall,
            f1_new_prices=prices_metrics.f1,
            precision_new_products=products_metrics.precision,
            recall_new_products=products_metrics.recall,
            f1_new_products=products_metrics.f1,
            training_time_sec=1.5,
            dataset_name="COUNTRY_7_B2C_mh5",
            n_train=12,
            n_eval_prices=3,
            n_eval_products=3,
        ),
        comparisons={"new_prices": prices_comparison, "new_products": products_comparison},
        config={
            "n_estimators": 100,
            "max_samples": "auto",
            "max_features": 1.0,
            "anomaly_threshold": 0.4,
            "contamination": "auto",
        },
    )


def _ae_candidate(run_id: str, *, new_prices: list[bool], new_products: list[bool]) -> AECandidateEvaluationResult:
    prices_comparison = _make_comparison_result({"Autoencoder": new_prices})
    products_comparison = _make_comparison_result({"Autoencoder": new_products})
    prices_metrics = prices_comparison.metrics["Autoencoder"]
    products_metrics = products_comparison.metrics["Autoencoder"]

    return AECandidateEvaluationResult(
        row=AEGridRow(
            run_id=run_id,
            phase=2,
            hidden_dims=json.dumps([64, 32]),
            latent_dim=8,
            dropout=0.1,
            learning_rate=0.001,
            epochs=10,
            threshold_percentile=95.0,
            threshold_value=0.5,
            precision_new_prices=prices_metrics.precision,
            recall_new_prices=prices_metrics.recall,
            f1_new_prices=prices_metrics.f1,
            precision_new_products=products_metrics.precision,
            recall_new_products=products_metrics.recall,
            f1_new_products=products_metrics.f1,
            f1_combined=(prices_metrics.f1 + products_metrics.f1) / 2,
            training_time_sec=2.0,
            dataset_name="COUNTRY_7_B2C_mh4",
            n_train=12,
            n_eval_prices=3,
            n_eval_products=3,
            mean_reconstruction_error=0.12,
        ),
        comparisons={"new_prices": prices_comparison, "new_products": products_comparison},
        config={
            "phase": 2,
            "hidden_dims": [64, 32],
            "latent_dim": 8,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "epochs": 10,
            "threshold_percentile": 95.0,
            "threshold_value": 0.5,
        },
    )


def _failed_if_candidate(run_id: str) -> IFCandidateEvaluationResult:
    return IFCandidateEvaluationResult(
        row=IFGridRow(
            run_id=run_id,
            n_estimators=100,
            max_samples="auto",
            max_features=1.0,
            anomaly_threshold=0.4,
            contamination="auto",
            precision=float("nan"),
            recall=float("nan"),
            f1_score=float("nan"),
            precision_new_prices=float("nan"),
            recall_new_prices=float("nan"),
            f1_new_prices=float("nan"),
            precision_new_products=float("nan"),
            recall_new_products=float("nan"),
            f1_new_products=float("nan"),
            training_time_sec=0.5,
            dataset_name="COUNTRY_7_B2C_mh5",
            n_train=12,
            n_eval_prices=3,
            n_eval_products=3,
            error="training failed",
        ),
        comparisons={},
        config={
            "n_estimators": 100,
            "max_samples": "auto",
            "max_features": 1.0,
            "anomaly_threshold": 0.4,
            "contamination": "auto",
        },
    )


def _failed_ae_candidate(run_id: str) -> AECandidateEvaluationResult:
    return AECandidateEvaluationResult(
        row=AEGridRow(
            run_id=run_id,
            phase=2,
            hidden_dims=json.dumps([64, 32]),
            latent_dim=8,
            dropout=0.1,
            learning_rate=0.001,
            epochs=10,
            threshold_percentile=95.0,
            threshold_value=0.5,
            precision_new_prices=float("nan"),
            recall_new_prices=float("nan"),
            f1_new_prices=float("nan"),
            precision_new_products=float("nan"),
            recall_new_products=float("nan"),
            f1_new_products=float("nan"),
            f1_combined=float("nan"),
            training_time_sec=0.5,
            dataset_name="COUNTRY_7_B2C_mh4",
            n_train=12,
            n_eval_prices=3,
            n_eval_products=3,
            mean_reconstruction_error=0.0,
            error="training failed",
        ),
        comparisons={},
        config={
            "phase": 2,
            "hidden_dims": [64, 32],
            "latent_dim": 8,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "epochs": 10,
            "threshold_percentile": 95.0,
            "threshold_value": 0.5,
        },
    )


def test_write_evaluation_run_derives_metrics_from_canonical_tables(tmp_path: Path) -> None:
    comparison = _make_comparison_result({"iforest": [False, True, False]})
    injected_rows, predictions = comparison_result_to_tables(
        comparison,
        run_id="comparison_run_001",
        candidate_id="",
        experiment_family="comparison",
        dataset_name="COUNTRY_7_B2C_mh5",
        dataset_granularity="country_segment",
        dataset_split="test_new_prices_mh5",
    )

    assert injected_rows["evaluation_row_id"].tolist() == [0, 1, 2]
    merged = predictions.merge(
        injected_rows[["evaluation_row_id", "ground_truth_label"]],
        on="evaluation_row_id",
        how="inner",
    )
    assert merged["ground_truth_label"].sum() == 2
    assert merged["predicted_is_anomaly"].sum() == 1

    run_root = tmp_path / "results" / "comparison" / "comparison_run_001"
    write_evaluation_run(
        run_root=run_root,
        run_metadata={
            "schema_version": "phase2.v1",
            "experiment_family": "comparison",
            "run_id": "comparison_run_001",
            "source_dataset_paths": ["data/train.parquet", "data/test.parquet"],
            "dataset_names": ["COUNTRY_7_B2C_mh5"],
            "dataset_granularity": "country_segment",
            "dataset_splits": ["new_prices_mh5"],
        },
        split_artifacts={"new_prices_mh5": (injected_rows, predictions)},
    )

    detector_metrics = pd.read_csv(run_root / "metrics" / "detector_metrics.csv")
    anomaly_type_metrics = pd.read_csv(run_root / "metrics" / "anomaly_type_metrics.csv")
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))

    assert detector_metrics.loc[0, "tp"] == 1
    assert detector_metrics.loc[0, "fn"] == 1
    assert detector_metrics.loc[0, "accuracy"] == pytest.approx(2 / 3)
    assert detector_metrics.loc[0, "tnr"] == pytest.approx(1.0)
    assert detector_metrics.loc[0, "g_mean"] == pytest.approx(np.sqrt(0.5 * 1.0))
    assert anomaly_type_metrics["anomaly_type"].tolist() == ["price_spike", "zero_price"]
    assert summary["split_summaries"]["new_prices_mh5"]["best_detector"] == "iforest"
    assert (run_root / "splits" / "new_prices_mh5" / "injected_rows.parquet").exists()
    assert (run_root / "splits" / "new_prices_mh5" / "predictions.parquet").exists()


def test_write_evaluation_run_rejects_split_directory_metadata_mismatch(tmp_path: Path) -> None:
    comparison = _make_comparison_result({"iforest": [False, True, False]})
    injected_rows, predictions = comparison_result_to_tables(
        comparison,
        run_id="comparison_run_mismatch",
        candidate_id="",
        experiment_family="comparison",
        dataset_name="COUNTRY_7_B2C_mh5",
        dataset_granularity="country_segment",
        dataset_split="test_new_prices_mh5",
    )

    with pytest.raises(ValueError, match="row metadata uses"):
        write_evaluation_run(
            run_root=tmp_path / "results" / "comparison" / "comparison_run_mismatch",
            run_metadata={
                "schema_version": "phase2.v1",
                "experiment_family": "comparison",
                "run_id": "comparison_run_mismatch",
                "dataset_splits": ["new_prices"],
            },
            split_artifacts={"new_prices": (injected_rows, predictions)},
        )


def test_write_evaluation_run_summary_counts_unique_rows_per_split(tmp_path: Path) -> None:
    comparison = _make_comparison_result(
        {"detector_a": [False, True, False], "detector_b": [False, True, True]}
    )
    injected_rows, predictions = comparison_result_to_tables(
        comparison,
        run_id="comparison_run_002",
        candidate_id="",
        experiment_family="comparison",
        dataset_name="COUNTRY_7_B2C_mh5",
        dataset_granularity="country_segment",
        dataset_split="new_prices",
    )

    run_root = tmp_path / "results" / "comparison" / "comparison_run_002"
    write_evaluation_run(
        run_root=run_root,
        run_metadata={
            "schema_version": "phase2.v1",
            "experiment_family": "comparison",
            "run_id": "comparison_run_002",
            "dataset_splits": ["new_prices"],
        },
        split_artifacts={"new_prices": (injected_rows, predictions)},
    )

    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    split_summary = summary["split_summaries"]["new_prices"]

    assert split_summary["total_rows"] == 3
    assert split_summary["total_injected"] == 2


def test_script_helpers_build_phase2_tables(tmp_path: Path) -> None:
    comparison_one = _make_comparison_result({"zscore": [False, True, False]})
    comparison_two = _make_comparison_result({"zscore": [False, True, True]})

    split_artifacts = build_canonical_split_artifacts(
        [
            ModelComparison(
                model_name="COUNTRY_7_B2C_mh4",
                granularity="country_segment",
                n_samples=3,
                n_products=2,
                comparison_result=comparison_one,
            ),
            ModelComparison(
                model_name="COUNTRY_10_B2B_mh4",
                granularity="country_segment",
                n_samples=3,
                n_products=2,
                comparison_result=comparison_two,
            ),
        ],
        run_id="comparison_helpers_001",
        dataset_split="test_new_prices_mh4",
    )
    split_name = next(iter(split_artifacts))
    injected_rows, predictions = split_artifacts[split_name]
    assert split_name == "new_prices_mh4"
    assert len(injected_rows) == 6
    assert len(predictions) == 6

    dataset = DatasetFiles(
        granularity="competitor",
        name="COMPETITOR_1_COUNTRY_7_mh4",
        train_path="train.parquet",
        test_paths={"new_prices": "test.parquet"},
        country_segment="COUNTRY_7_B2C",
        competitor_id="COMPETITOR_1_COUNTRY_7",
    )
    task = EvaluationTask(
        model_granularity="country_segment",
        model_name="COUNTRY_7_B2C_mh4",
        dataset=dataset,
        test_split="new_prices",
        scenario="known_products",
    )
    run_artifacts = _build_run_artifacts(
        [TaskEvaluationResult(task=task, comparison=comparison_one, n_samples=3, n_products=2)],
        run_id="granularity_run_001",
    )
    granularity_rows = run_artifacts["new_prices"][0]
    assert "scenario" in granularity_rows.columns
    assert granularity_rows["scenario"].iloc[0] == "known_products"

    combo_artifacts = build_canonical_run_artifacts(
        {
            "test_new_products_mh4": [
                AnalysisRunResult(
                    model_name="COUNTRY_7_B2C_mh4",
                    dataset_granularity="country_segment",
                    dataset_split="new_products",
                    comparison_result=_make_comparison_result(
                        {"Sanity": [False, True, False], "Z-score": [False, False, True]}
                    ),
                )
            ]
        },
        run_id="combination_run_001",
    )
    prediction_map, labels, obs_counts = aggregate_from_canonical_tables(combo_artifacts)
    assert set(prediction_map) == {"Sanity", "Z-score"}
    assert labels.tolist() == [False, True, True]
    assert obs_counts.tolist() == [1, 2, 3]

    detection_matrix = build_detection_matrix_frame(
        {"iforest": {"price_spike": True, "zero_price": False}, "zscore": {"price_spike": True}}
    )
    assert sorted(detection_matrix.columns.tolist()) == ["anomaly_type", "detected", "detector_name"]
    assert len(detection_matrix) == 3


def test_comparison_helpers_reindex_multi_dataset_rows_for_canonical_writes(tmp_path: Path) -> None:
    split_artifacts = build_canonical_split_artifacts(
        [
            ModelComparison(
                model_name="COUNTRY_7_B2C_mh4",
                granularity="country_segment",
                n_samples=3,
                n_products=2,
                comparison_result=_make_comparison_result({"zscore": [False, True, False]}),
            ),
            ModelComparison(
                model_name="COUNTRY_10_B2B_mh4",
                granularity="country_segment",
                n_samples=3,
                n_products=2,
                comparison_result=_make_comparison_result({"zscore": [False, True, True]}),
            ),
        ],
        run_id="comparison_helpers_002",
        dataset_split="test_new_prices_mh4",
    )

    injected_rows, predictions = split_artifacts["new_prices_mh4"]
    key_columns = ["run_id", "candidate_id", "dataset_split", "evaluation_row_id"]

    assert not injected_rows.duplicated(key_columns).any()
    assert injected_rows["evaluation_row_id"].tolist() == [0, 1, 2, 3, 4, 5]

    run_root = tmp_path / "results" / "comparison" / "comparison_helpers_002"
    write_evaluation_run(
        run_root=run_root,
        run_metadata={
            "schema_version": "phase2.v1",
            "experiment_family": "comparison",
            "run_id": "comparison_helpers_002",
            "dataset_splits": ["new_prices_mh4"],
        },
        split_artifacts=split_artifacts,
    )

    detector_metrics = pd.read_csv(run_root / "metrics" / "detector_metrics.csv")
    assert detector_metrics.loc[0, "n_rows"] == 6
    assert detector_metrics.loc[0, "n_injected"] == 4


def test_detector_combination_artifacts_reindex_multi_dataset_rows() -> None:
    combo_artifacts = build_canonical_run_artifacts(
        {
            "test_new_products_mh4": [
                AnalysisRunResult(
                    model_name="COUNTRY_7_B2C_mh4",
                    dataset_granularity="country_segment",
                    dataset_split="new_products",
                    comparison_result=_make_comparison_result(
                        {"Sanity": [False, True, False], "Z-score": [False, False, True]}
                    ),
                ),
                AnalysisRunResult(
                    model_name="COUNTRY_10_B2B_mh4",
                    dataset_granularity="country_segment",
                    dataset_split="new_products",
                    comparison_result=_make_comparison_result(
                        {"Sanity": [False, True, False], "Z-score": [False, False, True]}
                    ),
                ),
            ]
        },
        run_id="combination_run_002",
    )

    injected_rows, _ = combo_artifacts["test_new_products_mh4"]
    key_columns = ["run_id", "candidate_id", "dataset_split", "evaluation_row_id"]
    assert not injected_rows.duplicated(key_columns).any()

    prediction_map, labels, obs_counts = aggregate_from_canonical_tables(combo_artifacts)
    assert set(prediction_map) == {"Sanity", "Z-score"}
    assert len(labels) == 6
    assert labels.sum() == 4
    assert prediction_map["Sanity"].sum() == 2
    assert prediction_map["Z-score"].sum() == 2
    assert obs_counts.tolist() == [1, 2, 3, 1, 2, 3]


def test_isolation_forest_sweep_writes_candidate_artifacts_and_summary(tmp_path: Path) -> None:
    sweep_root = tmp_path / "results" / "tuning" / "isolation_forest" / "if_sweep_001"
    candidates = [
        _if_candidate("candidate_a", new_prices=[False, True, False], new_products=[False, True, False]),
        _if_candidate("candidate_b", new_prices=[False, True, True], new_products=[False, True, True]),
    ]

    for candidate in candidates:
        write_if_candidate_run_artifacts(
            sweep_root=sweep_root,
            candidate=candidate,
            dataset_name="COUNTRY_7_B2C_mh5",
            dataset_granularity="country_segment",
            train_file="data/train.parquet",
            test_file_prices="data/test_new_prices.parquet",
            test_file_products="data/test_new_products.parquet",
            sweep_id="if_sweep_001",
        )

    write_tuning_sweep(
        sweep_root=sweep_root,
        sweep_metadata={
            "schema_version": "phase2.v1",
            "experiment_family": "tuning",
            "detector_family": "isolation_forest",
            "sweep_id": "if_sweep_001",
        },
        candidate_metrics=build_if_candidate_metrics_frame(
            [candidate.row for candidate in candidates],
            sweep_id="if_sweep_001",
            dataset_name="COUNTRY_7_B2C_mh5",
            dataset_granularity="country_segment",
        ),
    )

    candidate_metrics = pd.read_csv(sweep_root / "candidate_metrics.csv")
    summary = json.loads((sweep_root / "summary.json").read_text(encoding="utf-8"))

    assert candidate_metrics["candidate_id"].is_unique
    assert set(candidate_metrics["candidate_id"]) == {"candidate_a", "candidate_b"}
    assert summary["best_candidate"]["candidate_id"] == "candidate_b"
    assert (sweep_root / "candidates" / "candidate_a" / "metrics" / "detector_metrics.csv").exists()
    assert (sweep_root / "candidates" / "candidate_b" / "splits" / "new_products" / "predictions.parquet").exists()


def test_isolation_forest_sweep_writes_failed_candidate_artifacts(tmp_path: Path) -> None:
    sweep_root = tmp_path / "results" / "tuning" / "isolation_forest" / "if_sweep_002"
    candidate = _failed_if_candidate("candidate_error")

    write_if_candidate_run_artifacts(
        sweep_root=sweep_root,
        candidate=candidate,
        dataset_name="COUNTRY_7_B2C_mh5",
        dataset_granularity="country_segment",
        train_file="data/train.parquet",
        test_file_prices="data/test_new_prices.parquet",
        test_file_products="data/test_new_products.parquet",
        sweep_id="if_sweep_002",
    )

    write_tuning_sweep(
        sweep_root=sweep_root,
        sweep_metadata={
            "schema_version": "phase2.v1",
            "experiment_family": "tuning",
            "detector_family": "isolation_forest",
            "sweep_id": "if_sweep_002",
        },
        candidate_metrics=build_if_candidate_metrics_frame(
            [candidate.row],
            sweep_id="if_sweep_002",
            dataset_name="COUNTRY_7_B2C_mh5",
            dataset_granularity="country_segment",
        ),
    )

    candidate_root = sweep_root / "candidates" / "candidate_error"
    detector_metrics = pd.read_csv(candidate_root / "metrics" / "detector_metrics.csv")
    candidate_metrics = pd.read_csv(sweep_root / "candidate_metrics.csv")

    assert (candidate_root / "run_metadata.json").exists()
    assert (candidate_root / "splits" / "new_prices" / "injected_rows.parquet").exists()
    assert (candidate_root / "splits" / "new_products" / "predictions.parquet").exists()
    assert detector_metrics.empty
    assert candidate_metrics.loc[0, "status"] == "error"


def test_autoencoder_sweep_writes_candidate_artifacts_and_summary(tmp_path: Path) -> None:
    sweep_root = tmp_path / "results" / "tuning" / "autoencoder" / "ae_sweep_001"
    candidates = [
        _ae_candidate("candidate_a", new_prices=[False, True, False], new_products=[False, True, False]),
        _ae_candidate("candidate_b", new_prices=[False, True, True], new_products=[False, True, True]),
    ]

    for candidate in candidates:
        write_ae_candidate_run_artifacts(
            sweep_root=sweep_root,
            candidate=candidate,
            dataset_name="COUNTRY_7_B2C_mh4",
            dataset_granularity="country_segment",
            train_file="data/train.parquet",
            test_file_prices="data/test_new_prices.parquet",
            test_file_products="data/test_new_products.parquet",
            sweep_id="ae_sweep_001",
        )

    write_tuning_sweep(
        sweep_root=sweep_root,
        sweep_metadata={
            "schema_version": "phase2.v1",
            "experiment_family": "tuning",
            "detector_family": "autoencoder",
            "sweep_id": "ae_sweep_001",
        },
        candidate_metrics=build_ae_candidate_metrics_frame(
            [candidate.row for candidate in candidates],
            sweep_id="ae_sweep_001",
            dataset_name="COUNTRY_7_B2C_mh4",
            dataset_granularity="country_segment",
        ),
    )

    candidate_metrics = pd.read_csv(sweep_root / "candidate_metrics.csv")
    summary = json.loads((sweep_root / "summary.json").read_text(encoding="utf-8"))

    assert candidate_metrics["candidate_id"].is_unique
    assert set(candidate_metrics["candidate_id"]) == {"candidate_a", "candidate_b"}
    assert summary["best_candidate"]["candidate_id"] == "candidate_b"
    assert (sweep_root / "candidates" / "candidate_a" / "metrics" / "detector_metrics.csv").exists()
    assert (sweep_root / "candidates" / "candidate_b" / "splits" / "new_prices" / "injected_rows.parquet").exists()


def test_autoencoder_sweep_writes_failed_candidate_artifacts(tmp_path: Path) -> None:
    sweep_root = tmp_path / "results" / "tuning" / "autoencoder" / "ae_sweep_002"
    candidate = _failed_ae_candidate("candidate_error")

    write_ae_candidate_run_artifacts(
        sweep_root=sweep_root,
        candidate=candidate,
        dataset_name="COUNTRY_7_B2C_mh4",
        dataset_granularity="country_segment",
        train_file="data/train.parquet",
        test_file_prices="data/test_new_prices.parquet",
        test_file_products="data/test_new_products.parquet",
        sweep_id="ae_sweep_002",
    )

    write_tuning_sweep(
        sweep_root=sweep_root,
        sweep_metadata={
            "schema_version": "phase2.v1",
            "experiment_family": "tuning",
            "detector_family": "autoencoder",
            "sweep_id": "ae_sweep_002",
        },
        candidate_metrics=build_ae_candidate_metrics_frame(
            [candidate.row],
            sweep_id="ae_sweep_002",
            dataset_name="COUNTRY_7_B2C_mh4",
            dataset_granularity="country_segment",
        ),
    )

    candidate_root = sweep_root / "candidates" / "candidate_error"
    detector_metrics = pd.read_csv(candidate_root / "metrics" / "detector_metrics.csv")
    candidate_metrics = pd.read_csv(sweep_root / "candidate_metrics.csv")

    assert (candidate_root / "run_metadata.json").exists()
    assert (candidate_root / "splits" / "new_prices" / "predictions.parquet").exists()
    assert (candidate_root / "splits" / "new_products" / "injected_rows.parquet").exists()
    assert detector_metrics.empty
    assert candidate_metrics.loc[0, "status"] == "error"


def test_write_tuning_sweep_prefers_rank_score_for_best_candidate(tmp_path: Path) -> None:
    sweep_root = tmp_path / "results" / "tuning" / "statistical" / "stat_sweep_001"
    candidate_metrics = pd.DataFrame(
        [
            {
                "schema_version": "phase2.v1",
                "sweep_id": "stat_sweep_001",
                "run_id": "candidate_a",
                "candidate_id": "candidate_a",
                "experiment_family": "tuning",
                "detector_family": "standard_zscore",
                "dataset_name": "GLOBAL",
                "dataset_granularity": "global",
                "status": "ok",
                "rank_score": 0.71,
                "combined_f1": 0.82,
            },
            {
                "schema_version": "phase2.v1",
                "sweep_id": "stat_sweep_001",
                "run_id": "candidate_b",
                "candidate_id": "candidate_b",
                "experiment_family": "tuning",
                "detector_family": "standard_zscore",
                "dataset_name": "GLOBAL",
                "dataset_granularity": "global",
                "status": "ok",
                "rank_score": 0.79,
                "combined_f1": 0.80,
            },
        ]
    )

    write_tuning_sweep(
        sweep_root=sweep_root,
        sweep_metadata={
            "schema_version": "phase2.v1",
            "experiment_family": "tuning",
            "detector_family": "standard_zscore",
            "sweep_id": "stat_sweep_001",
        },
        candidate_metrics=candidate_metrics,
    )

    summary = json.loads((sweep_root / "summary.json").read_text(encoding="utf-8"))

    assert summary["rank_column"] == "rank_score"
    assert summary["best_candidate"]["candidate_id"] == "candidate_b"


def test_comparison_run_metadata_records_evaluated_source_paths() -> None:
    comparison = _make_comparison_result({"zscore": [False, True, False]})
    metadata = build_comparison_run_metadata(
        run_id="comparison_run_003",
        source_dataset_paths=[
            "data/training/by_country_segment/model_b_test_new_prices_mh4.parquet",
            "data/training/by_country_segment/model_a_test_new_prices_mh4.parquet",
            "data/training/by_country_segment/model_a_test_new_prices_mh4.parquet",
        ],
        comparisons=[
            ModelComparison(
                model_name="COUNTRY_10_B2B_mh4",
                granularity="country_segment",
                n_samples=3,
                n_products=2,
                results=comparison.metrics,
                comparison_result=comparison,
            ),
            ModelComparison(
                model_name="COUNTRY_7_B2C_mh4",
                granularity="country_segment",
                n_samples=3,
                n_products=2,
                results=comparison.metrics,
                comparison_result=comparison,
            ),
        ],
        dataset_granularity="country_segment",
        dataset_split="_test_new_prices_mh4",
        injection_rate=0.1,
        skip_ml=False,
        workers=2,
        model_filter="COUNTRY_7",
    )

    assert metadata["source_dataset_paths"] == [
        "data/training/by_country_segment/model_a_test_new_prices_mh4.parquet",
        "data/training/by_country_segment/model_b_test_new_prices_mh4.parquet",
    ]
    assert metadata["dataset_names"] == ["COUNTRY_10_B2B_mh4", "COUNTRY_7_B2C_mh4"]
    assert metadata["dataset_splits"] == ["new_prices_mh4"]


def test_isolation_forest_persist_candidate_artifacts_writes_failed_candidates(tmp_path: Path) -> None:
    sweep_root = tmp_path / "results" / "tuning" / "isolation_forest" / "if_sweep_003"
    candidate = _failed_if_candidate("candidate_error")
    written_candidates: set[str] = set()

    written = persist_if_candidate_run_artifacts(
        sweep_root=sweep_root,
        candidate=candidate,
        dataset_name="COUNTRY_7_B2C_mh5",
        dataset_granularity="country_segment",
        train_file="data/train.parquet",
        test_file_prices="data/test_new_prices.parquet",
        test_file_products="data/test_new_products.parquet",
        sweep_id="if_sweep_003",
        written_candidates=written_candidates,
    )

    candidate_root = sweep_root / "candidates" / "candidate_error"
    detector_metrics = pd.read_csv(candidate_root / "metrics" / "detector_metrics.csv")

    assert written is True
    assert written_candidates == {"candidate_error"}
    assert (candidate_root / "run_metadata.json").exists()
    assert (candidate_root / "splits" / "new_prices" / "injected_rows.parquet").exists()
    assert (candidate_root / "splits" / "new_products" / "predictions.parquet").exists()
    assert detector_metrics.empty


def test_autoencoder_persist_candidate_artifacts_writes_failed_candidates(tmp_path: Path) -> None:
    sweep_root = tmp_path / "results" / "tuning" / "autoencoder" / "ae_sweep_003"
    candidate = _failed_ae_candidate("candidate_error")
    written_candidates: set[str] = set()

    written = persist_ae_candidate_run_artifacts(
        sweep_root=sweep_root,
        candidate=candidate,
        dataset_name="COUNTRY_7_B2C_mh4",
        dataset_granularity="country_segment",
        train_file="data/train.parquet",
        test_file_prices="data/test_new_prices.parquet",
        test_file_products="data/test_new_products.parquet",
        sweep_id="ae_sweep_003",
        written_candidates=written_candidates,
    )

    candidate_root = sweep_root / "candidates" / "candidate_error"
    detector_metrics = pd.read_csv(candidate_root / "metrics" / "detector_metrics.csv")

    assert written is True
    assert written_candidates == {"candidate_error"}
    assert (candidate_root / "run_metadata.json").exists()
    assert (candidate_root / "splits" / "new_prices" / "predictions.parquet").exists()
    assert (candidate_root / "splits" / "new_products" / "injected_rows.parquet").exists()
    assert detector_metrics.empty


def test_tuning_summary_ignores_failed_candidates_for_best_candidate() -> None:
    candidate = _failed_ae_candidate("candidate_error")
    candidate_metrics = build_ae_candidate_metrics_frame(
        [candidate.row],
        sweep_id="ae_sweep_004",
        dataset_name="COUNTRY_7_B2C_mh4",
        dataset_granularity="country_segment",
    )

    summary = build_tuning_summary(
        {
            "schema_version": "phase2.v1",
            "experiment_family": "tuning",
            "detector_family": "autoencoder",
            "sweep_id": "ae_sweep_004",
        },
        candidate_metrics,
    )

    assert summary["candidate_count"] == 1
    assert summary["best_candidate"] is None
