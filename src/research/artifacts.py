"""Canonical Phase 2 artifact helpers for research evaluation outputs."""

from __future__ import annotations

import json
import math
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.research.evaluation.test_orchestrator import ComparisonResult

SCHEMA_VERSION = "phase2.v1"

CANONICAL_INJECTION_COLUMNS = (
    "source_row_index",
    "ground_truth_label",
    "is_injected",
    "anomaly_type",
    "injection_strategy",
    "injection_phase",
    "injection_seed",
    "injection_params_json",
    "original_price",
)

CANONICAL_INJECTED_ROW_COLUMNS = [
    "schema_version",
    "run_id",
    "candidate_id",
    "experiment_family",
    "dataset_name",
    "dataset_granularity",
    "dataset_split",
    "evaluation_row_id",
    "source_row_index",
    "timestamp",
    "product_id",
    "competitor_id",
    "competitor_product_id",
    "original_price",
    "evaluated_price",
    "ground_truth_label",
    "is_injected",
    "anomaly_type",
    "injection_strategy",
    "injection_phase",
    "injection_seed",
    "injection_params_json",
]

CANONICAL_PREDICTION_COLUMNS = [
    "schema_version",
    "run_id",
    "candidate_id",
    "experiment_family",
    "dataset_split",
    "evaluation_row_id",
    "detector_name",
    "detector_family",
    "predicted_is_anomaly",
    "anomaly_score",
    "is_valid_input",
    "accepted_via_persistence",
    "detected_anomaly_types_json",
    "details_json",
]

DETECTOR_METRIC_COLUMNS = [
    "run_id",
    "candidate_id",
    "dataset_split",
    "detector_name",
    "accuracy",
    "precision",
    "recall",
    "tnr",
    "f1",
    "g_mean",
    "tp",
    "fp",
    "fn",
    "tn",
    "n_rows",
    "n_injected",
    "n_predicted",
]

ANOMALY_TYPE_METRIC_COLUMNS = [
    "run_id",
    "candidate_id",
    "dataset_split",
    "detector_name",
    "anomaly_type",
    "injected_count",
    "detected_count",
    "detection_rate",
]


def create_run_id(prefix: str | None = None, timestamp: datetime | None = None) -> str:
    """Create a stable UTC run identifier."""
    current = timestamp or datetime.now(timezone.utc)
    run_id = current.strftime("%Y%m%dT%H%M%SZ")
    if prefix:
        return f"{slugify(prefix)}_{run_id}"
    return run_id


def slugify(value: str) -> str:
    """Create a filesystem-safe slug."""
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    compact = "_".join(part for part in cleaned.split("_") if part)
    return compact or "artifact"


def normalize_dataset_split_name(value: str | None) -> str:
    """Normalize a dataset split name for metadata and directory names."""
    if not value:
        return "default"

    split = value.replace(".parquet", "").strip()
    while split.startswith("_"):
        split = split[1:]
    if split.startswith("test_"):
        split = split[len("test_") :]
    return slugify(split) or "default"


def empty_injected_rows_table(*, extra_columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Build an empty injected-row table with the canonical column order."""
    columns = list(CANONICAL_INJECTED_ROW_COLUMNS)
    if extra_columns:
        columns.extend(column for column in extra_columns if column not in columns)
    return pd.DataFrame(columns=columns)


def empty_predictions_table(*, extra_columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Build an empty predictions table with the canonical column order."""
    columns = list(CANONICAL_PREDICTION_COLUMNS)
    if extra_columns:
        columns.extend(column for column in extra_columns if column not in columns)
    return pd.DataFrame(columns=columns)


def reindex_split_artifacts(
    split_artifacts: Sequence[tuple[pd.DataFrame, pd.DataFrame]],
    *,
    starting_row_id: int = 0,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Rewrite evaluation_row_id values so concatenated split artifacts stay unique."""
    next_row_id = starting_row_id
    reindexed: list[tuple[pd.DataFrame, pd.DataFrame]] = []

    for injected_rows, predictions in split_artifacts:
        updated_injected = injected_rows.copy()
        updated_predictions = predictions.copy()

        if not updated_injected.empty:
            offset = next_row_id
            updated_injected["evaluation_row_id"] = (
                pd.to_numeric(updated_injected["evaluation_row_id"], errors="raise").astype(np.int64) + offset
            )
            if not updated_predictions.empty:
                updated_predictions["evaluation_row_id"] = (
                    pd.to_numeric(updated_predictions["evaluation_row_id"], errors="raise").astype(np.int64)
                    + offset
                )
            next_row_id += len(updated_injected)

        reindexed.append((updated_injected, updated_predictions))

    return reindexed


def resolve_git_commit(cwd: Path | None = None) -> str | None:
    """Resolve the current git commit hash if the repository is available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def initialize_evaluation_tracking_columns(
    df: pd.DataFrame,
    *,
    injection_seed: int | None = None,
    injection_strategy: str = "synthetic_dataframe_injection",
) -> pd.DataFrame:
    """Ensure the canonical row-alignment columns exist before orchestration."""
    frame = df.copy()
    n_rows = len(frame)

    if "source_row_index" not in frame.columns:
        frame["source_row_index"] = np.arange(n_rows, dtype=np.int64)
    if "ground_truth_label" not in frame.columns:
        frame["ground_truth_label"] = False
    if "is_injected" not in frame.columns:
        frame["is_injected"] = False
    if "anomaly_type" not in frame.columns:
        frame["anomaly_type"] = None
    if "injection_strategy" not in frame.columns:
        frame["injection_strategy"] = injection_strategy
    if "injection_phase" not in frame.columns:
        frame["injection_phase"] = pd.Series([pd.NA] * n_rows, dtype="Int64")
    else:
        frame["injection_phase"] = frame["injection_phase"].astype("Int64")
    if "injection_seed" not in frame.columns:
        seed_values = [injection_seed] * n_rows
        frame["injection_seed"] = pd.Series(seed_values, dtype="Int64")
    else:
        frame["injection_seed"] = frame["injection_seed"].astype("Int64")
    if "injection_params_json" not in frame.columns:
        frame["injection_params_json"] = "{}"
    if "original_price" not in frame.columns:
        if "price" in frame.columns:
            frame["original_price"] = frame["price"]
        else:
            frame["original_price"] = np.nan

    frame["ground_truth_label"] = frame["ground_truth_label"].fillna(False).astype(bool)
    frame["is_injected"] = frame["is_injected"].fillna(False).astype(bool)
    frame["injection_strategy"] = frame["injection_strategy"].fillna(injection_strategy)
    frame["injection_params_json"] = frame["injection_params_json"].fillna("{}").astype(str)
    return frame


def comparison_result_to_tables(
    result: ComparisonResult,
    *,
    run_id: str,
    candidate_id: str | None,
    experiment_family: str,
    dataset_name: str,
    dataset_granularity: str,
    dataset_split: str,
    detector_family_map: Mapping[str, str] | None = None,
    injected_row_extras: Mapping[str, Any] | None = None,
    prediction_extras: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert a comparison result into canonical injected-row and prediction tables."""
    if result.df_sorted is None:
        raise ValueError("ComparisonResult.df_sorted is required for canonical artifact writing")

    injected_rows = build_injected_rows_table(
        result.df_sorted,
        labels=result.labels,
        run_id=run_id,
        candidate_id=candidate_id,
        experiment_family=experiment_family,
        dataset_name=dataset_name,
        dataset_granularity=dataset_granularity,
        dataset_split=dataset_split,
        extra_columns=injected_row_extras,
    )
    predictions = build_predictions_table(
        result.raw_results,
        injected_rows,
        run_id=run_id,
        candidate_id=candidate_id,
        experiment_family=experiment_family,
        dataset_split=dataset_split,
        detector_family_map=detector_family_map,
        extra_columns=prediction_extras,
    )
    return injected_rows, predictions


def build_injected_rows_table(
    df: pd.DataFrame,
    *,
    labels: Sequence[bool] | np.ndarray | None,
    run_id: str,
    candidate_id: str | None,
    experiment_family: str,
    dataset_name: str,
    dataset_granularity: str,
    dataset_split: str,
    extra_columns: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Build the canonical injected-row table for one evaluated split."""
    frame = initialize_evaluation_tracking_columns(df)
    if labels is not None:
        frame["ground_truth_label"] = np.asarray(labels).astype(bool)

    frame["evaluation_row_id"] = np.arange(len(frame), dtype=np.int64)

    timestamp_col = _detect_timestamp_column(frame)
    timestamp_series = frame[timestamp_col] if timestamp_col else pd.Series([pd.NaT] * len(frame))

    canonical = pd.DataFrame(
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "candidate_id": candidate_id or "",
            "experiment_family": experiment_family,
            "dataset_name": dataset_name,
            "dataset_granularity": dataset_granularity,
            "dataset_split": normalize_dataset_split_name(dataset_split),
            "evaluation_row_id": frame["evaluation_row_id"].astype(np.int64),
            "source_row_index": frame["source_row_index"].astype(np.int64),
            "timestamp": pd.to_datetime(timestamp_series, errors="coerce"),
            "product_id": _get_series_or_default(frame, "product_id"),
            "competitor_id": _get_series_or_default(frame, "competitor_id"),
            "competitor_product_id": _get_series_or_default(frame, "competitor_product_id"),
            "original_price": pd.to_numeric(frame["original_price"], errors="coerce"),
            "evaluated_price": pd.to_numeric(_get_series_or_default(frame, "price"), errors="coerce"),
            "ground_truth_label": frame["ground_truth_label"].fillna(False).astype(bool),
            "is_injected": frame["is_injected"].fillna(False).astype(bool),
            "anomaly_type": frame["anomaly_type"],
            "injection_strategy": frame["injection_strategy"],
            "injection_phase": frame["injection_phase"].astype("Int64"),
            "injection_seed": frame["injection_seed"].astype("Int64"),
            "injection_params_json": frame["injection_params_json"].fillna("{}").astype(str),
        }
    )

    canonical = _apply_extra_columns(canonical, extra_columns)
    return _reorder_columns(canonical, CANONICAL_INJECTED_ROW_COLUMNS)


def build_predictions_table(
    raw_results: Mapping[str, list[Any]],
    injected_rows: pd.DataFrame,
    *,
    run_id: str,
    candidate_id: str | None,
    experiment_family: str,
    dataset_split: str,
    detector_family_map: Mapping[str, str] | None = None,
    extra_columns: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Build the canonical predictions table for one evaluated split."""
    rows: list[dict[str, Any]] = []
    split_name = normalize_dataset_split_name(dataset_split)
    valid_input = _infer_valid_input(injected_rows)

    for detector_name, results in raw_results.items():
        if len(results) != len(injected_rows):
            raise ValueError(
                f"Detector {detector_name!r} produced {len(results)} results for "
                f"{len(injected_rows)} evaluation rows"
            )

        detector_family = (
            detector_family_map.get(detector_name)
            if detector_family_map is not None and detector_name in detector_family_map
            else infer_detector_family(detector_name)
        )

        for idx, result in enumerate(results):
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": run_id,
                    "candidate_id": candidate_id or "",
                    "experiment_family": experiment_family,
                    "dataset_split": split_name,
                    "evaluation_row_id": int(injected_rows.iloc[idx]["evaluation_row_id"]),
                    "detector_name": detector_name,
                    "detector_family": detector_family,
                    "predicted_is_anomaly": bool(getattr(result, "is_anomaly", False)),
                    "anomaly_score": _coerce_float(getattr(result, "anomaly_score", np.nan)),
                    "is_valid_input": bool(_extract_is_valid_input(result, valid_input[idx])),
                    "accepted_via_persistence": bool(
                        _extract_detail_value(result, "accepted_via_persistence", False)
                    ),
                    "detected_anomaly_types_json": json_dumps(
                        [_normalize_anomaly_type(value) for value in getattr(result, "anomaly_types", [])]
                    ),
                    "details_json": json_dumps(getattr(result, "details", {}) or {}),
                }
            )

    predictions = pd.DataFrame(rows)
    predictions = _apply_extra_columns(predictions, extra_columns)
    return _reorder_columns(predictions, CANONICAL_PREDICTION_COLUMNS)


def compute_detector_metrics(
    injected_rows: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Compute canonical detector metrics from row-level artifacts."""
    merged = predictions.merge(
        injected_rows[
            [
                "run_id",
                "candidate_id",
                "dataset_split",
                "evaluation_row_id",
                "ground_truth_label",
                "is_injected",
            ]
        ],
        on=["run_id", "candidate_id", "dataset_split", "evaluation_row_id"],
        how="left",
        validate="many_to_one",
    )

    rows: list[dict[str, Any]] = []
    grouped = merged.groupby(
        ["run_id", "candidate_id", "dataset_split", "detector_name"],
        dropna=False,
        sort=True,
    )
    for (run_id, candidate_id, dataset_split, detector_name), group in grouped:
        predicted = group["predicted_is_anomaly"].fillna(False).astype(bool)
        truth = group["ground_truth_label"].fillna(False).astype(bool)

        tp = int((predicted & truth).sum())
        fp = int((predicted & ~truth).sum())
        fn = int((~predicted & truth).sum())
        tn = int((~predicted & ~truth).sum())
        accuracy = (tp + tn) / len(group) if len(group) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = _f1(precision, recall)
        tnr = tn / (tn + fp) if (tn + fp) else 0.0
        g_mean = math.sqrt(recall * tnr)

        rows.append(
            {
                "run_id": run_id,
                "candidate_id": candidate_id,
                "dataset_split": dataset_split,
                "detector_name": detector_name,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "tnr": tnr,
                "f1": f1,
                "g_mean": g_mean,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "n_rows": int(len(group)),
                "n_injected": int(group["is_injected"].fillna(False).astype(bool).sum()),
                "n_predicted": int(predicted.sum()),
            }
        )

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        metrics = pd.DataFrame(columns=DETECTOR_METRIC_COLUMNS)
    return _reorder_columns(metrics, DETECTOR_METRIC_COLUMNS)


def compute_anomaly_type_metrics(
    injected_rows: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-anomaly-type detection rates from canonical tables."""
    injected_only = injected_rows[injected_rows["is_injected"].fillna(False).astype(bool)].copy()
    if injected_only.empty:
        return pd.DataFrame(columns=ANOMALY_TYPE_METRIC_COLUMNS)

    merged = predictions.merge(
        injected_only[
            [
                "run_id",
                "candidate_id",
                "dataset_split",
                "evaluation_row_id",
                "anomaly_type",
            ]
        ],
        on=["run_id", "candidate_id", "dataset_split", "evaluation_row_id"],
        how="inner",
        validate="many_to_one",
    )

    rows: list[dict[str, Any]] = []
    grouped = merged.groupby(
        ["run_id", "candidate_id", "dataset_split", "detector_name", "anomaly_type"],
        dropna=False,
        sort=True,
    )
    for (run_id, candidate_id, dataset_split, detector_name, anomaly_type), group in grouped:
        injected_count = int(len(group))
        detected_count = int(group["predicted_is_anomaly"].fillna(False).astype(bool).sum())
        rows.append(
            {
                "run_id": run_id,
                "candidate_id": candidate_id,
                "dataset_split": dataset_split,
                "detector_name": detector_name,
                "anomaly_type": anomaly_type,
                "injected_count": injected_count,
                "detected_count": detected_count,
                "detection_rate": detected_count / injected_count if injected_count else 0.0,
            }
        )

    metrics = pd.DataFrame(rows)
    return _reorder_columns(metrics, ANOMALY_TYPE_METRIC_COLUMNS)


def write_evaluation_run(
    *,
    run_root: Path,
    run_metadata: Mapping[str, Any],
    split_artifacts: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]],
    analysis_artifacts: Mapping[str, Any] | None = None,
) -> None:
    """Write the canonical artifact layout for an evaluation-style run."""
    run_root.mkdir(parents=True, exist_ok=True)

    injected_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []

    for split_name, (injected_rows, predictions) in split_artifacts.items():
        normalized_split = _resolve_split_directory_name(split_name, injected_rows, predictions)
        split_dir = run_root / "splits" / normalized_split
        split_dir.mkdir(parents=True, exist_ok=True)

        injected_rows.to_parquet(split_dir / "injected_rows.parquet", index=False)
        predictions.to_parquet(split_dir / "predictions.parquet", index=False)

        injected_frames.append(injected_rows)
        prediction_frames.append(predictions)

    all_injected = (
        pd.concat(injected_frames, ignore_index=True)
        if injected_frames
        else pd.DataFrame(columns=CANONICAL_INJECTED_ROW_COLUMNS)
    )
    all_predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame(columns=CANONICAL_PREDICTION_COLUMNS)
    )

    detector_metrics = compute_detector_metrics(all_injected, all_predictions)
    anomaly_type_metrics = compute_anomaly_type_metrics(all_injected, all_predictions)

    metrics_dir = run_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    detector_metrics.to_csv(metrics_dir / "detector_metrics.csv", index=False)
    anomaly_type_metrics.to_csv(metrics_dir / "anomaly_type_metrics.csv", index=False)

    metadata = dict(run_metadata)
    metadata.setdefault("schema_version", SCHEMA_VERSION)
    metadata["output_root"] = str(run_root.resolve())
    with (run_root / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(_to_serializable(metadata), handle, indent=2)
        handle.write("\n")

    summary = build_evaluation_summary(metadata, detector_metrics, anomaly_type_metrics)
    with (run_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    (run_root / "summary.md").write_text(
        render_evaluation_summary_markdown(summary, detector_metrics),
        encoding="utf-8",
    )

    if analysis_artifacts:
        analysis_root = run_root / "analysis"
        analysis_root.mkdir(parents=True, exist_ok=True)
        for relative_path, artifact in analysis_artifacts.items():
            artifact_path = analysis_root / relative_path
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            _write_analysis_artifact(artifact_path, artifact)


def write_tuning_sweep(
    *,
    sweep_root: Path,
    sweep_metadata: Mapping[str, Any],
    candidate_metrics: pd.DataFrame,
) -> None:
    """Write the canonical artifact layout for a tuning sweep."""
    sweep_root.mkdir(parents=True, exist_ok=True)
    normalized = normalize_candidate_metrics(candidate_metrics)
    normalized.to_csv(sweep_root / "candidate_metrics.csv", index=False)

    metadata = dict(sweep_metadata)
    metadata.setdefault("schema_version", SCHEMA_VERSION)
    metadata["output_root"] = str(sweep_root.resolve())
    with (sweep_root / "sweep_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(_to_serializable(metadata), handle, indent=2)
        handle.write("\n")

    summary = build_tuning_summary(metadata, normalized)
    with (sweep_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    (sweep_root / "summary.md").write_text(
        render_tuning_summary_markdown(summary, normalized),
        encoding="utf-8",
    )


def normalize_candidate_metrics(candidate_metrics: pd.DataFrame) -> pd.DataFrame:
    """Normalize candidate metrics into the canonical sweep table order."""
    frame = candidate_metrics.copy()
    if "schema_version" not in frame.columns:
        frame.insert(0, "schema_version", SCHEMA_VERSION)
    preferred = [
        "schema_version",
        "sweep_id",
        "run_id",
        "candidate_id",
        "experiment_family",
        "detector_family",
        "dataset_name",
        "dataset_granularity",
        "status",
        "error",
        "training_time_sec",
        "n_train",
        "n_eval_prices",
        "n_eval_products",
        "combined_precision",
        "combined_recall",
        "combined_f1",
    ]
    return _reorder_columns(frame, preferred)


def build_evaluation_summary(
    metadata: Mapping[str, Any],
    detector_metrics: pd.DataFrame,
    anomaly_type_metrics: pd.DataFrame,
) -> dict[str, Any]:
    """Build a JSON-renderable summary derived from canonical evaluation tables."""
    metadata_splits = [
        normalize_dataset_split_name(str(split)) for split in metadata.get("dataset_splits", []) if split
    ]
    summary: dict[str, Any] = {
        "schema_version": metadata.get("schema_version", SCHEMA_VERSION),
        "run_id": metadata.get("run_id", ""),
        "experiment_family": metadata.get("experiment_family", ""),
        "dataset_splits": sorted(detector_metrics["dataset_split"].unique().tolist())
        if not detector_metrics.empty
        else sorted(set(metadata_splits)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "detector_count": int(detector_metrics["detector_name"].nunique()) if not detector_metrics.empty else 0,
        "candidate_count": int(detector_metrics["candidate_id"].nunique()) if not detector_metrics.empty else 0,
        "split_summaries": {},
    }

    for split_name, group in detector_metrics.groupby("dataset_split", dropna=False, sort=True):
        best_row = group.sort_values("f1", ascending=False).iloc[0]
        unique_row_scopes = group.drop_duplicates(subset=["run_id", "candidate_id", "dataset_split"])
        summary["split_summaries"][str(split_name)] = {
            "n_metric_rows": int(len(group)),
            "best_detector": best_row["detector_name"],
            "best_candidate_id": best_row["candidate_id"],
            "best_f1": float(best_row["f1"]),
            "total_rows": int(unique_row_scopes["n_rows"].sum()),
            "total_injected": int(unique_row_scopes["n_injected"].sum()),
        }

    if not anomaly_type_metrics.empty:
        summary["anomaly_types"] = sorted(
            anomaly_type_metrics["anomaly_type"].dropna().astype(str).unique().tolist()
        )
    else:
        summary["anomaly_types"] = []
    return summary


def render_evaluation_summary_markdown(
    summary: Mapping[str, Any],
    detector_metrics: pd.DataFrame,
) -> str:
    """Render a Markdown summary derived from detector metrics."""
    lines = [
        f"# {summary.get('experiment_family', 'evaluation').title()} Summary",
        "",
        f"- Run ID: `{summary.get('run_id', '')}`",
        f"- Schema version: `{summary.get('schema_version', SCHEMA_VERSION)}`",
        f"- Splits: {', '.join(summary.get('dataset_splits', [])) or 'none'}",
        f"- Detector metric rows: {len(detector_metrics)}",
        "",
    ]

    if detector_metrics.empty:
        lines.append("No detector metrics were produced.")
        return "\n".join(lines) + "\n"

    for split_name, group in detector_metrics.groupby("dataset_split", dropna=False, sort=True):
        lines.extend(
            [
                f"## {split_name}",
                "",
                "| Candidate | Detector | Precision | Recall | F1 | TP | FP | FN | TN |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in group.sort_values(["f1", "detector_name"], ascending=[False, True]).iterrows():
            candidate = row["candidate_id"] or "-"
            lines.append(
                "| "
                f"{candidate} | {row['detector_name']} | "
                f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | "
                f"{int(row['tp'])} | {int(row['fp'])} | {int(row['fn'])} | {int(row['tn'])} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def build_tuning_summary(
    metadata: Mapping[str, Any],
    candidate_metrics: pd.DataFrame,
) -> dict[str, Any]:
    """Build a JSON-renderable tuning summary derived from candidate metrics."""
    rank_column = select_candidate_rank_column(candidate_metrics)
    summary: dict[str, Any] = {
        "schema_version": metadata.get("schema_version", SCHEMA_VERSION),
        "sweep_id": metadata.get("sweep_id", ""),
        "experiment_family": metadata.get("experiment_family", ""),
        "detector_family": metadata.get("detector_family", ""),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_count": int(len(candidate_metrics)),
        "rank_column": rank_column,
        "best_candidate": None,
    }

    if not candidate_metrics.empty and rank_column:
        valid = _select_rankable_candidates(candidate_metrics, rank_column)
        if not valid.empty:
            best = valid.sort_values(rank_column, ascending=False).iloc[0]
            summary["best_candidate"] = _to_serializable(best.to_dict())

    return summary


def render_tuning_summary_markdown(
    summary: Mapping[str, Any],
    candidate_metrics: pd.DataFrame,
) -> str:
    """Render a Markdown sweep summary derived from candidate metrics."""
    lines = [
        f"# {summary.get('detector_family', 'tuning').title()} Sweep Summary",
        "",
        f"- Sweep ID: `{summary.get('sweep_id', '')}`",
        f"- Schema version: `{summary.get('schema_version', SCHEMA_VERSION)}`",
        f"- Candidates: {summary.get('candidate_count', 0)}",
        f"- Rank column: `{summary.get('rank_column', '')}`",
        "",
    ]

    if candidate_metrics.empty:
        lines.append("No candidate metrics were produced.")
        return "\n".join(lines) + "\n"

    rank_column = summary.get("rank_column")
    display = candidate_metrics.copy()
    if rank_column and rank_column in display.columns:
        display = display.sort_values(rank_column, ascending=False)

    lines.extend(
        [
            "| Candidate | Status | Combined F1 | Training Time (s) | Error |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for _, row in display.iterrows():
        combined_f1 = row.get("combined_f1", row.get("f1_combined", np.nan))
        lines.append(
            "| "
            f"{row.get('candidate_id', '')} | {row.get('status', '')} | "
            f"{_format_metric(combined_f1)} | {_format_metric(row.get('training_time_sec', np.nan))} | "
            f"{row.get('error', '') or '-'} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def select_candidate_rank_column(candidate_metrics: pd.DataFrame) -> str | None:
    """Select the preferred ranking column from a candidate-metrics table."""
    for column in ("rank_score", "combined_f1", "f1_combined", "best_f1", "f1"):
        if column in candidate_metrics.columns:
            return column
    return None


def _select_rankable_candidates(
    candidate_metrics: pd.DataFrame,
    rank_column: str,
) -> pd.DataFrame:
    """Filter a candidate-metrics table down to rows eligible for best-candidate ranking."""
    valid = candidate_metrics[candidate_metrics[rank_column].notna()].copy()
    if valid.empty:
        return valid

    if "status" in valid.columns:
        status = valid["status"].fillna("").astype(str).str.strip().str.lower()
        return valid[status.isin({"", "ok", "success"})].copy()

    if "error" in valid.columns:
        error = valid["error"].fillna("").astype(str).str.strip()
        return valid[error == ""].copy()

    return valid


def infer_detector_family(detector_name: str) -> str:
    """Infer a stable detector family label from a detector name."""
    name = detector_name.lower().replace(" ", "_")
    if "isolation" in name or "iforest" in name or "forest" in name:
        return "isolation_forest"
    if "autoencoder" in name or name.endswith("_ae") or "+ae" in detector_name.lower():
        return "autoencoder"
    if "zscore" in name or "z-score" in detector_name.lower():
        return "zscore"
    if "iqr" in name:
        return "iqr"
    if "threshold" in name:
        return "threshold"
    if "sanity" in name:
        return "sanity"
    if "ensemble" in name:
        return "statistical_ensemble"
    if "+" in detector_name or "combined" in name:
        return "combined"
    return slugify(detector_name)


def _apply_extra_columns(frame: pd.DataFrame, extra_columns: Mapping[str, Any] | None) -> pd.DataFrame:
    """Apply scalar or row-aligned extra columns to a DataFrame."""
    if not extra_columns:
        return frame

    updated = frame.copy()
    for column, value in extra_columns.items():
        if isinstance(value, (pd.Series, np.ndarray, list, tuple)):
            if len(value) != len(updated):
                raise ValueError(
                    f"Extra column {column!r} expected length {len(updated)}, got {len(value)}"
                )
            updated[column] = value
        else:
            updated[column] = value
    return updated


def _coerce_float(value: Any) -> float:
    """Coerce a scalar into a float, preserving NaN for missing values."""
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _detect_timestamp_column(df: pd.DataFrame) -> str | None:
    """Detect the canonical timestamp column for evaluation rows."""
    for column in ("first_seen_at", "scraped_at", "timestamp", "observed_at", "created_at"):
        if column in df.columns:
            return column
    return None


def _extract_detail_value(result: Any, key: str, default: Any) -> Any:
    """Extract a value from an anomaly result details dictionary."""
    details = getattr(result, "details", {}) or {}
    if not isinstance(details, Mapping):
        return default
    return details.get(key, default)


def _extract_is_valid_input(result: Any, default: bool) -> bool:
    """Extract is_valid_input from a result when present."""
    details = getattr(result, "details", {}) or {}
    if isinstance(details, Mapping) and "is_valid_input" in details:
        return bool(details["is_valid_input"])
    if isinstance(details, Mapping) and "feature_valid" in details:
        return bool(details["feature_valid"])
    return default


def _format_metric(value: Any) -> str:
    """Format a metric for Markdown rendering."""
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return "-"
    if math.isnan(metric):
        return "-"
    return f"{metric:.4f}"


def _f1(precision: float, recall: float) -> float:
    """Compute the harmonic mean of precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _get_series_or_default(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a column or a default NA series aligned to the DataFrame."""
    if column in df.columns:
        return df[column]
    return pd.Series([pd.NA] * len(df), index=df.index)


def _infer_valid_input(injected_rows: pd.DataFrame) -> np.ndarray:
    """Infer whether each evaluated row was a valid detector input."""
    prices = pd.to_numeric(injected_rows["evaluated_price"], errors="coerce")
    return prices.notna() & (prices > 0)


def _normalize_anomaly_type(value: Any) -> Any:
    """Normalize anomaly-type enums into JSON-safe strings."""
    if hasattr(value, "value"):
        return value.value
    return value


def _reorder_columns(frame: pd.DataFrame, preferred_columns: list[str] | tuple[str, ...]) -> pd.DataFrame:
    """Reorder a DataFrame so preferred columns come first."""
    existing_preferred = [column for column in preferred_columns if column in frame.columns]
    remainder = [column for column in frame.columns if column not in existing_preferred]
    return frame.loc[:, existing_preferred + remainder]


def _to_serializable(value: Any) -> Any:
    """Convert values into JSON-serializable Python types."""
    if isinstance(value, Mapping):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return _to_serializable(value.value)
    return value


def _write_analysis_artifact(path: Path, artifact: Any) -> None:
    """Write an analysis artifact under the canonical analysis/ directory."""
    suffix = path.suffix.lower()
    if isinstance(artifact, pd.DataFrame):
        if suffix == ".parquet":
            artifact.to_parquet(path, index=False)
            return
        artifact.to_csv(path, index=False)
        return
    if suffix == ".json":
        with path.open("w", encoding="utf-8") as handle:
            json.dump(_to_serializable(artifact), handle, indent=2)
            handle.write("\n")
        return
    path.write_text(str(artifact), encoding="utf-8")


def _resolve_split_directory_name(
    split_name: str,
    injected_rows: pd.DataFrame,
    predictions: pd.DataFrame,
) -> str:
    """Resolve and validate the canonical split directory name."""
    normalized_split = normalize_dataset_split_name(split_name)
    observed_splits: set[str] = set()

    for frame_name, frame in (("injected_rows", injected_rows), ("predictions", predictions)):
        if frame.empty or "dataset_split" not in frame.columns:
            continue

        split_values = frame["dataset_split"].dropna().astype(str).unique().tolist()
        if not split_values:
            continue
        if len(split_values) != 1:
            raise ValueError(
                f"{frame_name} for split {split_name!r} contains multiple dataset_split values: "
                f"{sorted(split_values)!r}"
            )

        observed_splits.add(normalize_dataset_split_name(split_values[0]))

    if observed_splits and observed_splits != {normalized_split}:
        raise ValueError(
            f"Split artifact key {split_name!r} resolves to {normalized_split!r}, "
            f"but row metadata uses {sorted(observed_splits)!r}"
        )

    return normalized_split


def json_dumps(value: Any) -> str:
    """Dump a JSON value with canonical formatting."""
    return json.dumps(_to_serializable(value), sort_keys=True)
