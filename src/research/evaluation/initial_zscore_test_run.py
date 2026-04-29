from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.anomaly.statistical import ZScoreDetector
from src.research.artifacts import (
    comparison_result_to_tables,
    initialize_evaluation_tracking_columns,
    json_dumps,
    resolve_git_commit,
    write_evaluation_run,
)
from src.research.evaluation.detector_evaluator import DetectorEvaluator
from src.research.evaluation.synthetic import inject_anomalies_to_dataframe
from src.research.evaluation.test_orchestrator import (
    ComparisonResult,
    TestOrchestrator,
    create_expanded_statistical_evaluators,
)
from src.research.datasets import ResolvedDataset, project_root, resolve_dataset_by_id

DATASET_ID = "COUNTRY_4__B2C__COMPETITOR_3_COUNTRY_4"
DATA_ROOT = "data/training"
MIN_HISTORY = 5
TRAIN_SPLIT = "train"
EVALUATION_SPLITS = ("test_new_prices", "test_new_products")

INJECTION_RATE = 0.1
INJECTION_SEED = 42
SPIKE_RANGE = (2.0, 5.0)
DROP_RANGE = (0.1, 0.5)
INJECTION_STRATEGY = "synthetic_dataframe_injection"

CANDIDATE_ID = "expanded_statistical_baseline"
RUN_ID = "initial_country4_competitor3_expanded_statistical_baseline"


def _load_frames(paths: tuple[Path, ...]) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in paths]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _extract_country_token(dataset_name: str | None) -> str | None:
    if not dataset_name:
        return None
    if dataset_name.startswith("COUNTRY_"):
        parts = dataset_name.split("_")
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
    return None


def _normalize_split_name(split_name: str) -> str:
    mapping = {
        "test_new_prices": "new_prices",
        "test_new_products": "new_products",
    }
    return mapping.get(split_name, split_name)


def _annotate_injected_frame(
    frame: pd.DataFrame,
    *,
    labels: np.ndarray,
    injection_details: list[dict],
    injection_seed: int,
) -> pd.DataFrame:
    annotated = initialize_evaluation_tracking_columns(
        frame,
        injection_seed=injection_seed,
        injection_strategy=INJECTION_STRATEGY,
    )
    annotated["ground_truth_label"] = np.asarray(labels).astype(bool)
    annotated["is_injected"] = annotated["ground_truth_label"]
    annotated["anomaly_type"] = pd.Series([None] * len(annotated), dtype="object")
    annotated["injection_phase"] = pd.Series([pd.NA] * len(annotated), dtype="Int64")
    annotated["injection_params_json"] = "{}"

    if "__original_price__" in annotated.columns:
        annotated["original_price"] = pd.to_numeric(
            annotated["__original_price__"],
            errors="coerce",
        )

    for detail in injection_details:
        row_index = int(detail["index"])
        if row_index not in annotated.index:
            continue

        params = {
            key: value
            for key, value in detail.items()
            if key not in {"index", "anomaly_type", "original_price", "new_price"}
        }
        annotated.at[row_index, "anomaly_type"] = detail.get("anomaly_type")
        annotated.at[row_index, "injection_phase"] = detail.get("injection_phase", pd.NA)
        annotated.at[row_index, "injection_params_json"] = json_dumps(params)
        if "original_price" in detail:
            annotated.at[row_index, "original_price"] = detail["original_price"]

    return annotated


def _build_run_metadata(dataset: ResolvedDataset) -> dict[str, object]:
    evaluators = create_expanded_statistical_evaluators()
    return {
        "schema_version": "phase2.v1",
        "experiment_family": "comparison",
        "run_id": RUN_ID,
        "candidate_id": CANDIDATE_ID,
        "dataset_id": dataset.dataset_id,
        "dataset_scope": dataset.scope,
        "component_dataset_ids": list(dataset.component_dataset_ids),
        "source_dataset_paths": list(dataset.source_dataset_paths),
        "dataset_names": [dataset.dataset_name],
        "dataset_granularity": dataset.scope,
        "dataset_splits": [_normalize_split_name(split) for split in dataset.evaluation_files],
        "random_seeds": {
            "training_seed": INJECTION_SEED,
            "injection_seed": INJECTION_SEED,
        },
        "injection_config": {
            "injection_rate": INJECTION_RATE,
            "spike_range": list(SPIKE_RANGE),
            "drop_range": list(DROP_RANGE),
            "strategy": INJECTION_STRATEGY,
        },
        "detector_identifiers": [e.name for e in evaluators],
        "config_values": {
            "data_root": DATA_ROOT,
            "train_split": TRAIN_SPLIT,
            "evaluation_splits": list(EVALUATION_SPLITS),
            "min_history": MIN_HISTORY,
            "zscore_threshold": 3.0,
            "robust_threshold": 2.0,
            "hybrid_weight": 0.5,
            "iqr_multiplier": 1.5,
            "change_threshold": 0.20,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": resolve_git_commit(project_root()),
    }


def _build_output_root(base_root: Path | None = None) -> Path:
    if base_root is not None:
        return base_root
    return project_root() / "results" / "comparison" / RUN_ID


def run(*, output_root: Path | None = None) -> Path:
    dataset = resolve_dataset_by_id(
        data_root=DATA_ROOT,
        dataset_id=DATASET_ID,
        min_history=MIN_HISTORY,
        train_split=TRAIN_SPLIT,
        evaluation_splits=EVALUATION_SPLITS,
    )
    resolved_output_root = _build_output_root(output_root)

    train_df = _load_frames(dataset.train_files)
    country = _extract_country_token(dataset.dataset_name)
    evaluators = create_expanded_statistical_evaluators()
    detector_family_map = {evaluator.name: evaluator.detector.name for evaluator in evaluators}
    orchestrator = TestOrchestrator(evaluators, max_workers=len(evaluators))

    split_artifacts: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    comparisons: dict[str, ComparisonResult] = {}

    print(f"Dataset: {dataset.dataset_name}")
    print(f"Train rows: {len(train_df):,}")

    for split_index, (split_name, paths) in enumerate(dataset.evaluation_files.items()):
        frame = _load_frames(paths)
        injection_seed = INJECTION_SEED + split_index
        injected_frame, labels, injection_details = inject_anomalies_to_dataframe(
            frame,
            injection_rate=INJECTION_RATE,
            seed=injection_seed,
            spike_range=SPIKE_RANGE,
            drop_range=DROP_RANGE,
        )
        injected_frame = _annotate_injected_frame(
            injected_frame,
            labels=labels,
            injection_details=injection_details,
            injection_seed=injection_seed,
        )

        comparison = orchestrator.run_comparison_with_details(
            train_df=train_df,
            test_df=injected_frame,
            labels=labels,
            country=country,
            injection_details=injection_details,
        )
        comparisons[split_name] = comparison
        split_artifacts[_normalize_split_name(split_name)] = comparison_result_to_tables(
            comparison,
            run_id=RUN_ID,
            candidate_id=CANDIDATE_ID,
            experiment_family="comparison",
            dataset_name=dataset.dataset_name,
            dataset_granularity=dataset.scope,
            dataset_split=split_name,
            detector_family_map=detector_family_map,
        )

    write_evaluation_run(
        run_root=resolved_output_root,
        run_metadata=_build_run_metadata(dataset),
        split_artifacts=split_artifacts,
    )

    print(f"Output root: {resolved_output_root}")
    for split_name, comparison in comparisons.items():
        print(_normalize_split_name(split_name))
        for detector_name, metrics in sorted(
            comparison.metrics.items(),
            key=lambda item: item[1].f1,
            reverse=True,
        ):
            print(
                f"  {detector_name}: "
                f"P={metrics.precision:.4f}, "
                f"R={metrics.recall:.4f}, "
                f"F1={metrics.f1:.4f}, "
                f"TP={metrics.true_positives}, "
                f"FP={metrics.false_positives}, "
                f"FN={metrics.false_negatives}"
            )

    return resolved_output_root


def main() -> None:
    run()


if __name__ == "__main__":
    main()
