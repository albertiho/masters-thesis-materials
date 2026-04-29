#!/usr/bin/env python3
"""Compare Isolation Forest model granularity using production combined detector.

This script evaluates country+segment vs competitor Isolation Forest models
with the ProductionCombinedDetector across both new-price and new-product
test splits, then writes a reviewable CSV for thesis analysis.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Add project root to path (4 levels up from research/training/scripts/)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_script_dir)))
sys.path.insert(0, _project_root)
sys.path.insert(0, _script_dir)  # For importing sibling scripts

from src.anomaly.combined_variants import ProductionCombinedDetector
from src.anomaly.persistence import ModelPersistence
from src.research.artifacts import (
    comparison_result_to_tables,
    create_run_id,
    resolve_git_commit,
    slugify,
    write_evaluation_run,
)
from src.research.evaluation.detector_evaluator import DetectorEvaluator
from src.research.evaluation.synthetic import SyntheticAnomalyType, inject_anomalies_to_dataframe
from src.research.evaluation.test_orchestrator import ComparisonResult, TestOrchestrator
from src.tuning_config import get_min_history
from train_isolation_forest import (
    extract_model_name,
    find_matching_test_file,
    find_parquet_files,
    select_latest_per_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_SEED = 42
ALLOWED_TEST_SPLITS = {"new_prices", "new_products"}
ANOMALY_TYPES = [
    SyntheticAnomalyType.PRICE_SPIKE,
    SyntheticAnomalyType.PRICE_DROP,
    SyntheticAnomalyType.ZERO_PRICE,
    SyntheticAnomalyType.NEGATIVE_PRICE,
    SyntheticAnomalyType.EXTREME_OUTLIER,
    SyntheticAnomalyType.DECIMAL_SHIFT,
]
ANOMALY_TYPE_NAMES = [atype.value for atype in ANOMALY_TYPES]


@dataclass(frozen=True)
class DatasetFiles:
    """Dataset pairing for a single model granularity."""

    granularity: str
    name: str
    train_path: str
    test_paths: dict[str, str]
    country_segment: str
    competitor_id: str | None


@dataclass(frozen=True)
class EvaluationTask:
    """Single evaluation scenario."""

    model_granularity: str
    model_name: str
    dataset: DatasetFiles
    test_split: str
    scenario: str


@dataclass(frozen=True)
class TaskEvaluationResult:
    """Detailed output for one evaluated granularity-comparison task."""

    task: EvaluationTask
    comparison: ComparisonResult
    n_samples: int
    n_products: int


def _strip_min_history_suffix(model_name: str) -> str:
    return re.sub(r"_mh\d+$", "", model_name)


def _country_segment_from_model_name(model_name: str) -> str:
    base = _strip_min_history_suffix(model_name)
    match = re.match(r"^([A-Z]{2})_([A-Z0-9]+)", base)
    if not match:
        raise ValueError(f"Could not extract country_segment from model name: {model_name}")
    return f"{match.group(1)}_{match.group(2)}"


def _country_segment_from_competitor_path(path: str) -> str:
    parts = os.path.normpath(path).split(os.sep)
    if "by_competitor" not in parts:
        raise ValueError(f"Missing by_competitor segment in path: {path}")
    idx = parts.index("by_competitor")
    if idx + 2 >= len(parts):
        raise ValueError(f"Invalid competitor path layout: {path}")
    return f"{parts[idx + 1]}_{parts[idx + 2]}"


def _extract_country(country_segment: str) -> str | None:
    if "_" not in country_segment:
        return None
    return country_segment.split("_", 1)[0]


def _test_suffix(split: str, min_history: int) -> str:
    if split not in ALLOWED_TEST_SPLITS:
        raise ValueError(f"Unknown test split: {split}")
    return f"_test_{split}_mh{min_history}"


def _build_segment_datasets(
    data_path: str,
    test_splits: list[str],
    min_history: int,
    model_filter: str | None,
) -> list[DatasetFiles]:
    train_suffix = f"_train_mh{min_history}"
    candidate_files = find_parquet_files(data_path, "country_segment", train_suffix)
    if not candidate_files:
        raise FileNotFoundError(
            f"No country_segment train files found in {data_path} with suffix {train_suffix}"
        )
    train_files = select_latest_per_model(candidate_files, "country_segment")

    datasets: list[DatasetFiles] = []
    for train_path in train_files:
        model_name = extract_model_name(train_path)
        if model_filter and model_filter not in model_name:
            continue

        test_paths: dict[str, str] = {}
        for split in test_splits:
            test_suffix = _test_suffix(split, min_history)
            test_path = find_matching_test_file(train_path, test_suffix, data_path)
            if not test_path:
                raise FileNotFoundError(
                    f"Missing test file for {model_name}: expected suffix {test_suffix}"
                )
            test_paths[split] = test_path

        datasets.append(
            DatasetFiles(
                granularity="country_segment",
                name=model_name,
                train_path=train_path,
                test_paths=test_paths,
                country_segment=_country_segment_from_model_name(model_name),
                competitor_id=None,
            )
        )
    return datasets


def _build_competitor_datasets(
    data_path: str,
    test_splits: list[str],
    min_history: int,
    required_segments: set[str],
    model_filter: str | None,
) -> list[DatasetFiles]:
    train_suffix = f"_train_mh{min_history}"
    candidate_files = find_parquet_files(data_path, "competitor", train_suffix)
    if not candidate_files:
        raise FileNotFoundError(
            f"No competitor train files found in {data_path} with suffix {train_suffix}"
        )
    train_files = select_latest_per_model(candidate_files, "competitor")

    datasets: list[DatasetFiles] = []
    for train_path in train_files:
        model_name = extract_model_name(train_path)
        country_segment = _country_segment_from_competitor_path(train_path)
        include_for_segment = bool(required_segments) and country_segment in required_segments
        include_for_model = model_filter is None or model_filter in model_name
        if not include_for_segment and not include_for_model:
            continue

        test_paths: dict[str, str] = {}
        for split in test_splits:
            test_suffix = _test_suffix(split, min_history)
            test_path = find_matching_test_file(train_path, test_suffix, data_path)
            if not test_path:
                raise FileNotFoundError(
                    f"Missing test file for {model_name}: expected suffix {test_suffix}"
                )
            test_paths[split] = test_path

        datasets.append(
            DatasetFiles(
                granularity="competitor",
                name=model_name,
                train_path=train_path,
                test_paths=test_paths,
                country_segment=country_segment,
                competitor_id=_strip_min_history_suffix(model_name),
            )
        )
    return datasets


def _build_tasks(
    segment_datasets: list[DatasetFiles],
    competitor_datasets: list[DatasetFiles],
    test_splits: list[str],
    model_filter: str | None,
) -> list[EvaluationTask]:
    tasks: list[EvaluationTask] = []
    competitor_by_segment: dict[str, list[DatasetFiles]] = {}
    for dataset in competitor_datasets:
        competitor_by_segment.setdefault(dataset.country_segment, []).append(dataset)

    for segment in segment_datasets:
        if model_filter and model_filter not in segment.name:
            continue
        for split in test_splits:
            tasks.append(
                EvaluationTask(
                    model_granularity="country_segment",
                    model_name=segment.name,
                    dataset=segment,
                    test_split=split,
                    scenario="segment_on_segment",
                )
            )
            for competitor in competitor_by_segment.get(segment.country_segment, []):
                tasks.append(
                    EvaluationTask(
                        model_granularity="country_segment",
                        model_name=segment.name,
                        dataset=competitor,
                        test_split=split,
                        scenario="segment_on_competitor",
                    )
                )

    for competitor in competitor_datasets:
        if model_filter and model_filter not in competitor.name:
            continue
        for split in test_splits:
            tasks.append(
                EvaluationTask(
                    model_granularity="competitor",
                    model_name=competitor.name,
                    dataset=competitor,
                    test_split=split,
                    scenario="competitor_on_competitor",
                )
            )

    return tasks


def _task_key(task: EvaluationTask) -> tuple[str, str, str, str, str]:
    return (
        task.model_granularity,
        task.model_name,
        task.dataset.granularity,
        task.dataset.name,
        task.test_split,
    )


def _task_candidate_id(task: EvaluationTask) -> str:
    """Build a stable candidate identifier for one evaluation task."""
    return slugify(
        f"{task.scenario}_{task.model_granularity}_{task.model_name}_{task.dataset.granularity}_{task.dataset.name}"
    )


def _build_run_artifacts(
    results: list[TaskEvaluationResult],
    run_id: str,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Build canonical split artifacts for the full granularity run."""
    by_split: dict[str, list[tuple[pd.DataFrame, pd.DataFrame]]] = {}

    for result in results:
        task = result.task
        candidate_id = _task_candidate_id(task)
        injected_rows, predictions = comparison_result_to_tables(
            result.comparison,
            run_id=run_id,
            candidate_id=candidate_id,
            experiment_family="granularity",
            dataset_name=task.dataset.name,
            dataset_granularity=task.dataset.granularity,
            dataset_split=task.test_split,
            injected_row_extras={
                "scenario": task.scenario,
                "model_granularity": task.model_granularity,
                "model_name": task.model_name,
                "country_segment": task.dataset.country_segment,
                "competitor_id": task.dataset.competitor_id,
                "n_products": result.n_products,
            },
            prediction_extras={
                "scenario": task.scenario,
                "model_granularity": task.model_granularity,
                "model_name": task.model_name,
                "country_segment": task.dataset.country_segment,
                "competitor_id": task.dataset.competitor_id,
            },
        )
        by_split.setdefault(task.test_split, []).append((injected_rows, predictions))

    return {
        split: (
            pd.concat([artifact[0] for artifact in artifacts], ignore_index=True),
            pd.concat([artifact[1] for artifact in artifacts], ignore_index=True),
        )
        for split, artifacts in by_split.items()
    }


def _load_completed_keys(output_path: str) -> set[tuple[str, str, str, str, str]]:
    if not os.path.exists(output_path):
        return set()
    df = pd.read_csv(output_path)
    required = {
        "model_granularity",
        "model_name",
        "dataset_granularity",
        "dataset_name",
        "test_split",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Resume requires columns {sorted(required)} in {output_path}; missing {sorted(missing)}"
        )
    return {
        (
            str(row["model_granularity"]),
            str(row["model_name"]),
            str(row["dataset_granularity"]),
            str(row["dataset_name"]),
            str(row["test_split"]),
        )
        for _, row in df.iterrows()
    }


def _ensure_output_schema(output_path: str, columns: list[str]) -> None:
    if not os.path.exists(output_path):
        return
    with open(output_path, "r", encoding="utf-8") as handle:
        header = handle.readline().strip()
    if not header:
        return
    existing = header.split(",")
    if existing != columns:
        raise ValueError(
            f"Output schema mismatch for {output_path}. Expected {columns}, got {existing}"
        )


def _append_row(output_path: str, columns: list[str], row: dict[str, object]) -> None:
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    file_exists = os.path.exists(output_path)
    needs_header = not file_exists or os.path.getsize(output_path) == 0
    with open(output_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if needs_header:
            writer.writeheader()
        writer.writerow({col: row.get(col) for col in columns})


def _evaluate_task(
    task: EvaluationTask,
    model_cache: dict[str, object],
    injection_rate: float,
    seed: int,
) -> TaskEvaluationResult:
    start_time = time.time()
    model = model_cache[task.model_name]
    combined = ProductionCombinedDetector.create(model)
    evaluator = DetectorEvaluator(combined, name="production_combined")
    orchestrator = TestOrchestrator([evaluator], max_workers=1)

    train_df = pd.read_parquet(task.dataset.train_path)
    test_path = task.dataset.test_paths[task.test_split]
    test_df = pd.read_parquet(test_path)
    if test_df.empty:
        raise ValueError(f"Test dataset is empty: {test_path}")

    df_injected, labels, injection_details = inject_anomalies_to_dataframe(
        test_df,
        injection_rate=injection_rate,
        seed=seed,
        spike_range=(2.0, 5.0),
        drop_range=(0.1, 0.5),
        anomaly_types=ANOMALY_TYPES,
    )

    country = _extract_country(task.dataset.country_segment)
    comparison = orchestrator.run_comparison_with_details(
        train_df,
        df_injected,
        labels,
        country,
        injection_details=injection_details,
    )

    metrics = comparison.metrics[evaluator.name]
    elapsed = time.time() - start_time
    logger.info(
        "granularity_comparison_complete",
        extra={
            "model": task.model_name,
            "dataset": task.dataset.name,
            "split": task.test_split,
            "scenario": task.scenario,
            "elapsed_sec": f"{elapsed:.1f}",
            "f1": metrics.f1,
        },
    )
    return TaskEvaluationResult(
        task=task,
        comparison=comparison,
        n_samples=len(df_injected),
        n_products=int(df_injected["product_id"].nunique()),
    )


def _metrics_to_row(
    task: EvaluationTask,
    metrics: DetectorMetrics,
    n_samples: int,
    n_products: int,
) -> dict[str, object]:
    row: dict[str, object] = {
        "scenario": task.scenario,
        "model_granularity": task.model_granularity,
        "model_name": task.model_name,
        "dataset_granularity": task.dataset.granularity,
        "dataset_name": task.dataset.name,
        "country_segment": task.dataset.country_segment,
        "competitor_id": task.dataset.competitor_id,
        "test_split": task.test_split,
        "detector": metrics.detector_name,
        "n_samples": n_samples,
        "n_products": n_products,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
    }

    for anomaly_type in ANOMALY_TYPE_NAMES:
        column = f"{anomaly_type}_rate"
        if anomaly_type in metrics.detection_by_type:
            row[column] = metrics.detection_by_type[anomaly_type].rate
        else:
            row[column] = None
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare country_segment vs competitor Isolation Forest models "
        "using ProductionCombinedDetector across new-price and new-product splits."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/training",
        help="Base directory for training/test parquet files",
    )
    parser.add_argument(
        "--test-splits",
        nargs="+",
        default=["new_prices", "new_products"],
        help="Test splits to evaluate (default: new_prices new_products)",
    )
    parser.add_argument(
        "--injection-rate",
        type=float,
        default=0.1,
        help="Fraction of rows to inject anomalies into",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="results/granularity",
        help="Canonical results root (default: results/granularity)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id for the canonical output directory",
    )
    parser.add_argument(
        "--model-filter",
        type=str,
        default=None,
        help="Only evaluate models matching this substring",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list discovered datasets and planned evaluations",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers for dataset evaluations (default: 1)",
    )
    args = parser.parse_args()

    test_splits = [split.strip() for split in args.test_splits]
    for split in test_splits:
        if split not in ALLOWED_TEST_SPLITS:
            raise ValueError(f"Unsupported test split: {split}")

    load_dotenv()

    min_history = get_min_history("isolation_forest")
    logger.info("Using min_history=%s for file pairing", min_history)

    segment_datasets = _build_segment_datasets(
        data_path=args.data_path,
        test_splits=test_splits,
        min_history=min_history,
        model_filter=args.model_filter,
    )
    required_segments = {dataset.country_segment for dataset in segment_datasets}
    competitor_datasets = _build_competitor_datasets(
        data_path=args.data_path,
        test_splits=test_splits,
        min_history=min_history,
        required_segments=required_segments,
        model_filter=args.model_filter,
    )

    tasks = _build_tasks(segment_datasets, competitor_datasets, test_splits, args.model_filter)
    scenario_counts = Counter(task.scenario for task in tasks)

    print("=" * 70)
    print("Granularity Comparison - ProductionCombinedDetector")
    print("=" * 70)
    print(f"Data path: {args.data_path}")
    print(f"Min history: {min_history}")
    print(f"Test splits: {', '.join(test_splits)}")
    print(f"Injection rate: {args.injection_rate:.1%}")
    print(f"Segment models: {len(segment_datasets)}")
    print(f"Competitor datasets: {len(competitor_datasets)}")
    print(f"Tasks: {len(tasks)}")
    for scenario, count in sorted(scenario_counts.items()):
        print(f"  - {scenario}: {count}")
    if args.model_filter:
        print(f"Model filter: {args.model_filter}")
    print(f"Results root: {args.results_root}")
    if args.run_id:
        print(f"Run ID: {args.run_id}")
    print("=" * 70)
    print()

    if args.dry_run:
        print("Dry run: listing planned evaluations")
        for task in tasks:
            print(
                f"{task.scenario}: {task.model_name} -> {task.dataset.name} "
                f"({task.test_split})"
            )
        return

    if not tasks:
        raise ValueError("No evaluation tasks found (check filters and data path).")

    model_names = sorted({task.model_name for task in tasks})
    persistence = ModelPersistence()
    model_cache = {}
    for model_name in model_names:
        model_cache[model_name] = persistence.load_isolation_forest(model_name)

    task_results: list[TaskEvaluationResult] = []

    if args.workers <= 1:
        for task in tasks:
            task_result = _evaluate_task(task, model_cache, args.injection_rate, DEFAULT_SEED)
            task_results.append(task_result)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            future_map = {
                pool.submit(
                    _evaluate_task,
                    task,
                    model_cache,
                    args.injection_rate,
                    DEFAULT_SEED,
                ): task
                for task in tasks
            }
            for future in as_completed(future_map):
                task_results.append(future.result())

    run_id = args.run_id or create_run_id("granularity")
    run_root = Path(args.results_root) / run_id
    split_artifacts = _build_run_artifacts(task_results, run_id)
    run_metadata = {
        "schema_version": "phase2.v1",
        "experiment_family": "granularity",
        "run_id": run_id,
        "source_dataset_paths": sorted(
            {
                task.dataset.train_path
                for task in tasks
            }
            | {
                path
                for task in tasks
                for path in task.dataset.test_paths.values()
            }
        ),
        "dataset_names": sorted({task.dataset.name for task in tasks}),
        "dataset_granularity": "mixed",
        "dataset_splits": sorted(test_splits),
        "random_seeds": {"injection_seed": DEFAULT_SEED},
        "injection_config": {
            "injection_rate": args.injection_rate,
            "anomaly_types": [atype.value for atype in ANOMALY_TYPES],
        },
        "detector_identifiers": ["production_combined"],
        "config_values": {
            "workers": args.workers,
            "model_filter": args.model_filter,
            "min_history": min_history,
            "scenario_counts": dict(scenario_counts),
        },
        "generated_at": datetime.now().astimezone().isoformat(),
        "git_commit": resolve_git_commit(Path(_project_root)),
    }
    write_evaluation_run(
        run_root=run_root,
        run_metadata=run_metadata,
        split_artifacts=split_artifacts,
    )

    print(f"Saved canonical results to {run_root}")


if __name__ == "__main__":
    main()
