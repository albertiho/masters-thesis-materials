from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.anomaly.statistical import AnomalyResult
from src.research.evaluation.initial_zscore_test_run import (
    CANDIDATE_ID,
    DATASET_ID,
    RUN_ID,
    run,
)
from src.research.evaluation.test_orchestrator import ComparisonResult, DetectorMetrics
from src.research.datasets import ResolvedDataset

DETECTOR_NAMES = {
    "Z-score",
    "ModifiedMAD",
    "ModifiedSN",
    "HybridWeighted",
    "HybridMax",
    "HybridAvg",
    "IQR",
    "Threshold",
}


def _write_parquet(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "product_id": ["p1", "p2"],
            "competitor_id": ["c1", "c1"],
            "competitor_product_id": ["cp1", "cp2"],
            "price": [100.0, 110.0],
            "list_price": [120.0, 130.0],
            "first_seen_at": pd.to_datetime(["2026-02-01", "2026-02-02"], utc=True),
        }
    ).to_parquet(path, index=False)


def _build_comparison_result(test_df: pd.DataFrame, labels: np.ndarray) -> ComparisonResult:
    predictions = np.array([True, False], dtype=bool)
    raw_results = {
        detector_name: [
            AnomalyResult(
                is_anomaly=bool(prediction),
                anomaly_score=0.9 if prediction else 0.1,
                anomaly_types=[],
                severity=None,
                details={"accepted_via_persistence": False, "is_valid_input": True},
                detector=detector_name,
                competitor_product_id=f"cp{index + 1}",
                competitor="c1",
            )
            for index, prediction in enumerate(predictions)
        ]
        for detector_name in DETECTOR_NAMES
    }
    metrics = {
        detector_name: DetectorMetrics(
            detector_name=detector_name,
            precision=1.0,
            recall=1.0,
            f1=1.0,
            true_positives=1,
            false_positives=0,
            false_negatives=0,
            n_samples=len(predictions),
            predictions=predictions,
            scores=predictions.astype(float),
        )
        for detector_name in DETECTOR_NAMES
    }
    return ComparisonResult(
        metrics=metrics,
        raw_results=raw_results,
        observation_counts=np.array([5, 5], dtype=np.int32),
        labels=labels,
        df_sorted=test_df.copy(),
    )


def test_initial_zscore_test_run_writes_canonical_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "data" / "training" / "derived" / "train.parquet"
    prices_path = tmp_path / "data" / "training" / "derived" / "new_prices.parquet"
    products_path = tmp_path / "data" / "training" / "derived" / "new_products.parquet"
    for path in (train_path, prices_path, products_path):
        _write_parquet(path)

    dataset = ResolvedDataset(
        dataset_id=DATASET_ID,
        scope="competitor",
        min_history=5,
        train_split="train",
        evaluation_splits=("test_new_prices", "test_new_products"),
        dataset_name=f"{DATASET_ID}_mh5",
        component_dataset_ids=(DATASET_ID,),
        countries=("COUNTRY_4",),
        segments=("B2C",),
        train_files=(train_path,),
        evaluation_files={
            "test_new_prices": (prices_path,),
            "test_new_products": (products_path,),
        },
        source_dataset_paths=(
            "data/training/derived/train.parquet",
            "data/training/derived/new_prices.parquet",
            "data/training/derived/new_products.parquet",
        ),
    )

    monkeypatch.setattr(
        "src.research.evaluation.initial_zscore_test_run.resolve_dataset_by_id",
        lambda **kwargs: dataset,
    )
    monkeypatch.setattr(
        "src.research.evaluation.initial_zscore_test_run.project_root",
        lambda: tmp_path,
    )

    def fake_injection(frame: pd.DataFrame, **kwargs):
        modified = frame.copy()
        modified["__original_price__"] = modified["price"]
        labels = np.array([True, False], dtype=bool)
        details = [
            {
                "index": 0,
                "anomaly_type": "price_spike",
                "original_price": 100.0,
                "new_price": 300.0,
                "injection_phase": 2,
                "multiplier": 3.0,
            }
        ]
        modified.loc[0, "price"] = 300.0
        return modified, labels, details

    monkeypatch.setattr(
        "src.research.evaluation.initial_zscore_test_run.inject_anomalies_to_dataframe",
        fake_injection,
    )

    class FakeOrchestrator:
        def __init__(self, evaluators, max_workers):
            self.evaluators = evaluators

        def run_comparison_with_details(self, **kwargs):
            return _build_comparison_result(kwargs["test_df"], kwargs["labels"])

    monkeypatch.setattr(
        "src.research.evaluation.initial_zscore_test_run.TestOrchestrator",
        FakeOrchestrator,
    )

    output_root = run()
    detector_metrics = pd.read_csv(output_root / "metrics" / "detector_metrics.csv")
    anomaly_type_metrics = pd.read_csv(output_root / "metrics" / "anomaly_type_metrics.csv")
    run_metadata = json.loads((output_root / "run_metadata.json").read_text(encoding="utf-8"))

    assert output_root == tmp_path / "results" / "comparison" / RUN_ID
    assert set(detector_metrics["dataset_split"]) == {"new_prices", "new_products"}
    assert set(detector_metrics["detector_name"]) == DETECTOR_NAMES
    assert set(anomaly_type_metrics["anomaly_type"]) == {"price_spike"}
    assert run_metadata["candidate_id"] == CANDIDATE_ID
    assert run_metadata["dataset_id"] == DATASET_ID
