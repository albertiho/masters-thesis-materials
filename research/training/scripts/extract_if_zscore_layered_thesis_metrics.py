#!/usr/bin/env python3
"""Extract thesis-ready metric tables from the fixed layered-detector run.

The output includes:

- one detector-level table for `new_prices`, `new_products`, and `combined`
- one anomaly-case table with one-vs-rest metrics per synthetic anomaly type

Anomaly-case precision/recall/TNR/FPR/FNR are computed against the injected
synthetic label in `injected_rows.parquet`. Because the detectors only predict
"anomaly vs non-anomaly", these are one-vs-rest detection metrics, not
synthetic-type attribution metrics.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


RUN_ID = "country1_all_scopes_if_zscore_layered_combinations"
RUN_ROOT = Path("results") / "detector_combinations" / RUN_ID
OUTPUT_DIRNAME = "thesis_metrics"
SPLITS = ("new_prices", "new_products")
COMBINED_SPLIT_WEIGHTS = {
    "new_prices": 0.7,
    "new_products": 0.3,
}

SCOPE_COLUMNS = [
    "mh_level",
    "scope_id",
    "dataset_name",
    "dataset_granularity",
    "scope_kind",
    "scope_market",
]

DETECTOR_TABLE_COLUMNS = [
    *SCOPE_COLUMNS,
    "detector_combination",
    "test_case_name",
    "precision",
    "recall",
    "f1",
    "true_negative_rate",
    "false_positive_rate",
    "false_negative_rate",
    "g_mean",
    "tp",
    "fp",
    "fn",
    "tn",
    "positive_count",
    "negative_count",
    "predicted_positive_count",
]

ANOMALY_CASE_TABLE_COLUMNS = [
    *SCOPE_COLUMNS,
    "detector_combination",
    "test_case_name",
    "anomaly_case",
    "precision",
    "recall",
    "f1",
    "true_negative_rate",
    "false_positive_rate",
    "false_negative_rate",
    "g_mean",
    "tp",
    "fp",
    "fn",
    "tn",
    "positive_count",
    "negative_count",
    "predicted_positive_count",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_run_root(run_root: Path | None = None) -> Path:
    target = run_root or RUN_ROOT
    if target.is_absolute():
        return target
    return (_repo_root() / target).resolve()


def _load_split_tables(run_root: Path) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for split_name in SPLITS:
        split_root = run_root / "splits" / split_name
        injected_rows = pd.read_parquet(split_root / "injected_rows.parquet")
        predictions = pd.read_parquet(split_root / "predictions.parquet")
        tables[split_name] = (injected_rows, predictions)
    return tables


def _has_complete_split_tables(run_root: Path) -> bool:
    for split_name in SPLITS:
        split_root = run_root / "splits" / split_name
        if not (split_root / "injected_rows.parquet").exists():
            return False
        if not (split_root / "predictions.parquet").exists():
            return False
    return True


def _evaluation_roots(run_root: Path) -> list[Path]:
    scopes_root = run_root / "scopes"
    if not scopes_root.exists():
        return [run_root]

    roots = [
        child
        for child in sorted(scopes_root.iterdir(), key=lambda path: path.name)
        if child.is_dir() and _has_complete_split_tables(child)
    ]
    if roots:
        return roots
    if _has_complete_split_tables(run_root):
        return [run_root]
    raise FileNotFoundError(
        f"No complete evaluation roots found under {run_root}. "
        "Expected complete per-scope split checkpoints under scopes/*/splits/."
    )


def _merge_predictions_with_truth(
    injected_rows: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    if injected_rows.empty and predictions.empty:
        empty = pd.DataFrame(
            {
                "run_id": pd.Series(dtype="object"),
                "candidate_id": pd.Series(dtype="object"),
                "dataset_split": pd.Series(dtype="object"),
                "evaluation_row_id": pd.Series(dtype="int64"),
                "ground_truth_label": pd.Series(dtype="bool"),
                "is_injected": pd.Series(dtype="bool"),
                "anomaly_type": pd.Series(dtype="object"),
                "dataset_name": pd.Series(dtype="object"),
                "dataset_granularity": pd.Series(dtype="object"),
                "predicted_is_anomaly": pd.Series(dtype="bool"),
                "detector_name": pd.Series(dtype="object"),
                "mh_level": pd.Series(dtype="object"),
                "scope_id": pd.Series(dtype="object"),
                "scope_kind": pd.Series(dtype="object"),
                "scope_market": pd.Series(dtype="object"),
            }
        )
        for column in ("mh_level", "scope_id", "scope_kind", "scope_market"):
            if column in injected_rows.columns and column in empty.columns:
                value = injected_rows[column].dropna().astype(str)
                if not value.empty:
                    empty[column] = value.iloc[0]
            elif column in predictions.columns and column in empty.columns:
                value = predictions[column].dropna().astype(str)
                if not value.empty:
                    empty[column] = value.iloc[0]
        return _ensure_scope_columns(empty)

    base_columns = [
        "run_id",
        "candidate_id",
        "dataset_split",
        "evaluation_row_id",
        "ground_truth_label",
        "is_injected",
        "anomaly_type",
        "dataset_name",
        "dataset_granularity",
    ]
    optional_columns = [
        column
        for column in ("mh_level", "scope_id", "scope_kind", "scope_market")
        if column in injected_rows.columns and column not in predictions.columns
    ]
    merged = predictions.merge(
        injected_rows[base_columns + optional_columns],
        on=["run_id", "candidate_id", "dataset_split", "evaluation_row_id"],
        how="left",
        validate="many_to_one",
    )
    return _ensure_scope_columns(merged)


def _ensure_scope_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if "mh_level" not in normalized.columns:
        normalized["mh_level"] = ""
    if "scope_id" not in normalized.columns:
        normalized["scope_id"] = normalized["candidate_id"].fillna("").astype(str)
    else:
        normalized["scope_id"] = normalized["scope_id"].fillna(
            normalized["candidate_id"].fillna("").astype(str)
        )

    if "dataset_name" not in normalized.columns:
        normalized["dataset_name"] = normalized["scope_id"]
    if "dataset_granularity" not in normalized.columns:
        normalized["dataset_granularity"] = ""
    if "scope_kind" not in normalized.columns:
        normalized["scope_kind"] = normalized["dataset_granularity"].fillna("").astype(str)
    if "scope_market" not in normalized.columns:
        normalized["scope_market"] = ""

    for column in SCOPE_COLUMNS:
        normalized[column] = normalized[column].fillna("").astype(str)

    return normalized


def build_case_frames(
    split_tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> dict[str, pd.DataFrame]:
    """Build merged case frames for split-level and combined metrics."""
    case_frames: dict[str, pd.DataFrame] = {}
    combined_frames: list[pd.DataFrame] = []

    for split_name in SPLITS:
        injected_rows, predictions = split_tables[split_name]
        merged = _merge_predictions_with_truth(injected_rows, predictions)
        case_frames[split_name] = merged
        combined_frames.append(merged)

    case_frames["combined"] = pd.concat(combined_frames, ignore_index=True)
    return case_frames


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _compute_binary_metrics(
    predicted: pd.Series,
    truth: pd.Series,
) -> dict[str, float | int]:
    predicted_bool = predicted.fillna(False).astype(bool)
    truth_bool = truth.fillna(False).astype(bool)

    tp = int((predicted_bool & truth_bool).sum())
    fp = int((predicted_bool & ~truth_bool).sum())
    fn = int((~predicted_bool & truth_bool).sum())
    tn = int((~predicted_bool & ~truth_bool).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    true_negative_rate = tn / (tn + fp) if (tn + fp) else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) else 0.0
    g_mean = math.sqrt(recall * true_negative_rate)

    return {
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "true_negative_rate": true_negative_rate,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "g_mean": g_mean,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "positive_count": int(truth_bool.sum()),
        "negative_count": int((~truth_bool).sum()),
        "predicted_positive_count": int(predicted_bool.sum()),
    }


def build_detector_metrics_table(case_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for test_case_name, frame in case_frames.items():
        grouped = frame.groupby(
            [*SCOPE_COLUMNS, "detector_name"],
            dropna=False,
            sort=True,
        )
        for group_key, group in grouped:
            *scope_values, detector_name = group_key
            row = {
                **dict(zip(SCOPE_COLUMNS, scope_values, strict=True)),
                "detector_combination": detector_name,
                "test_case_name": test_case_name,
            }
            row.update(
                _compute_binary_metrics(
                    group["predicted_is_anomaly"],
                    group["ground_truth_label"],
                )
            )
            rows.append(row)

    table = pd.DataFrame(rows)
    table = table.reindex(columns=DETECTOR_TABLE_COLUMNS)
    table = _apply_weighted_combined_scores(
        table,
        group_columns=[*SCOPE_COLUMNS, "detector_combination"],
    )
    return table.sort_values(
        ["scope_id", "test_case_name", "detector_combination"],
        kind="stable",
    ).reset_index(drop=True)


def build_anomaly_case_metrics_table(case_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for test_case_name, frame in case_frames.items():
        anomaly_cases = sorted(frame["anomaly_type"].dropna().astype(str).unique())
        detector_groups = frame.groupby(
            [*SCOPE_COLUMNS, "detector_name"],
            dropna=False,
            sort=True,
        )

        for group_key, detector_frame in detector_groups:
            *scope_values, detector_name = group_key
            for anomaly_case in anomaly_cases:
                truth = detector_frame["anomaly_type"].fillna("").astype(str).eq(anomaly_case)
                row = {
                    **dict(zip(SCOPE_COLUMNS, scope_values, strict=True)),
                    "detector_combination": detector_name,
                    "test_case_name": test_case_name,
                    "anomaly_case": anomaly_case,
                }
                row.update(
                    _compute_binary_metrics(
                        detector_frame["predicted_is_anomaly"],
                        truth,
                    )
                )
                rows.append(row)

    table = pd.DataFrame(rows)
    table = table.reindex(columns=ANOMALY_CASE_TABLE_COLUMNS)
    table = _apply_weighted_combined_scores(
        table,
        group_columns=[*SCOPE_COLUMNS, "detector_combination", "anomaly_case"],
    )
    return table.sort_values(
        ["scope_id", "test_case_name", "detector_combination", "anomaly_case"],
        kind="stable",
    ).reset_index(drop=True)


def _output_root(run_root: Path) -> Path:
    return run_root / "analysis" / OUTPUT_DIRNAME


def _apply_weighted_combined_scores(
    table: pd.DataFrame,
    *,
    group_columns: list[str],
) -> pd.DataFrame:
    if table.empty:
        return table

    weighted = table.copy()
    split_tables = {
        split_name: weighted[weighted["test_case_name"] == split_name].set_index(group_columns)
        for split_name in SPLITS
    }
    combined_mask = weighted["test_case_name"] == "combined"
    combined_rows = weighted.loc[combined_mask]

    for row_index, row in combined_rows.iterrows():
        group_key = tuple(row[column] for column in group_columns)
        weighted_f1 = 0.0
        weighted_g_mean = 0.0

        for split_name, split_weight in COMBINED_SPLIT_WEIGHTS.items():
            split_table = split_tables[split_name]
            if group_key not in split_table.index:
                continue
            split_row = split_table.loc[group_key]
            if isinstance(split_row, pd.DataFrame):
                split_row = split_row.iloc[0]
            weighted_f1 += split_weight * float(split_row["f1"])
            weighted_g_mean += split_weight * float(split_row["g_mean"])

        weighted.at[row_index, "f1"] = weighted_f1
        weighted.at[row_index, "g_mean"] = weighted_g_mean

    return weighted


def build_metric_tables(run_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    detector_tables: list[pd.DataFrame] = []
    anomaly_tables: list[pd.DataFrame] = []

    for evaluation_root in _evaluation_roots(run_root):
        split_tables = _load_split_tables(evaluation_root)
        case_frames = build_case_frames(split_tables)
        detector_tables.append(build_detector_metrics_table(case_frames))
        anomaly_tables.append(build_anomaly_case_metrics_table(case_frames))

    detector_metrics = pd.concat(detector_tables, ignore_index=True) if detector_tables else pd.DataFrame()
    anomaly_case_metrics = pd.concat(anomaly_tables, ignore_index=True) if anomaly_tables else pd.DataFrame()

    if not detector_metrics.empty:
        detector_metrics = detector_metrics.reindex(columns=DETECTOR_TABLE_COLUMNS).sort_values(
            ["scope_id", "test_case_name", "detector_combination"],
            kind="stable",
        ).reset_index(drop=True)
    else:
        detector_metrics = pd.DataFrame(columns=DETECTOR_TABLE_COLUMNS)

    if not anomaly_case_metrics.empty:
        anomaly_case_metrics = anomaly_case_metrics.reindex(columns=ANOMALY_CASE_TABLE_COLUMNS).sort_values(
            ["scope_id", "test_case_name", "detector_combination", "anomaly_case"],
            kind="stable",
        ).reset_index(drop=True)
    else:
        anomaly_case_metrics = pd.DataFrame(columns=ANOMALY_CASE_TABLE_COLUMNS)

    return detector_metrics, anomaly_case_metrics


def run(*, run_root: Path | None = None) -> Path:
    resolved_run_root = _resolve_run_root(run_root)
    detector_metrics, anomaly_case_metrics = build_metric_tables(resolved_run_root)

    output_root = _output_root(resolved_run_root)
    output_root.mkdir(parents=True, exist_ok=True)

    detector_path = output_root / "layered_detector_metrics.csv"
    anomaly_case_path = output_root / "layered_detector_anomaly_case_metrics.csv"
    detector_metrics.to_csv(detector_path, index=False)
    anomaly_case_metrics.to_csv(anomaly_case_path, index=False)

    print(f"Run root: {resolved_run_root}")
    print(f"Wrote detector metrics: {detector_path}")
    print(f"Wrote anomaly-case metrics: {anomaly_case_path}")

    for test_case_name in ("new_prices", "new_products", "combined"):
        case_table = detector_metrics[detector_metrics["test_case_name"] == test_case_name]
        if case_table.empty:
            continue
        for scope_id, scope_frame in case_table.groupby("scope_id", sort=True):
            best_row = scope_frame.sort_values("f1", ascending=False, kind="stable").iloc[0]
            print(
                f"{scope_id} {test_case_name}: best F1={best_row['f1']:.4f} "
                f"({best_row['detector_combination']})"
            )

    return output_root


def main() -> None:
    run()


if __name__ == "__main__":
    main()
