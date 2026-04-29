#!/usr/bin/env python3
"""Analyze statistical tuning outputs and compare subset-vs-full guidance.

This script reads the outputs produced by ``tune_statistical.py`` and generates:

1. A normalized table of best configurations for every discovered test case.
2. A per-case comparison of guidance from the full ``mh`` set versus a subset
   such as ``mh5,mh10,mh15,mh20,mh25,mh30``.
3. An aggregate detector-level comparison showing whether the subset would have
   led to the same recommended configuration as the full set.

Example:
    python research/training/scripts/analyze_statistical_guidance.py `
        --results-root results/tuning/statistical/by_competitor_single_attempt_batch `
        --subset-mh mh5,mh10,mh15,mh20,mh25,mh30
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger("analyze_statistical_guidance")
MH_PATTERN = re.compile(r"mh(\d+)$", re.IGNORECASE)

COMMON_CANDIDATE_COLUMNS = {
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
    "stage",
    "training_time_sec",
    "n_train",
    "n_eval_prices",
    "n_eval_products",
    "attempt_count",
    "combined_precision",
    "combined_recall",
    "combined_f1",
    "weighted_f1_mean",
    "rank_score",
    "default_distance",
    "new_prices_accuracy_mean",
    "new_prices_accuracy_std",
    "new_prices_precision_mean",
    "new_prices_precision_std",
    "new_prices_recall_mean",
    "new_prices_recall_std",
    "new_prices_f1_mean",
    "new_prices_f1_std",
    "new_prices_g_mean_mean",
    "new_prices_g_mean_std",
    "new_products_accuracy_mean",
    "new_products_accuracy_std",
    "new_products_precision_mean",
    "new_products_precision_std",
    "new_products_recall_mean",
    "new_products_recall_std",
    "new_products_f1_mean",
    "new_products_f1_std",
    "new_products_g_mean_mean",
    "new_products_g_mean_std",
    "mh_level",
    "granularity",
    "scope_id",
    "family_output_dir",
    "candidate_metrics_path",
    "best_configuration_path",
}

NUMERIC_METRICS = (
    "rank_score",
    "combined_f1",
    "weighted_f1_mean",
    "default_distance",
    "new_prices_f1_mean",
    "new_prices_g_mean_mean",
    "new_products_f1_mean",
    "new_products_g_mean_mean",
)


@dataclass(frozen=True)
class FamilyArtifact:
    """Resolved files for one scope/family best-configuration output."""

    best_configuration_path: Path
    candidate_metrics_path: Path
    mh_level: str
    granularity: str
    scope_id: str
    dataset_name: str
    detector_family: str
    path_mh_level: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze statistical tuning outputs and compare subset-vs-full guidance.",
    )
    parser.add_argument(
        "--results-root",
        required=True,
        type=Path,
        help="Root directory produced by tune_statistical.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the generated CSV/JSON/Markdown outputs. Defaults to <results-root>/analysis.",
    )
    parser.add_argument(
        "--subset-mh",
        default="mh5,mh10,mh15,mh20,mh25,mh30",
        help="Comma-separated mh subset to compare against the full available set.",
    )
    parser.add_argument(
        "--detectors",
        default="",
        help="Optional comma-separated detector_family filter.",
    )
    parser.add_argument(
        "--granularities",
        default="",
        help="Optional comma-separated granularity filter.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def mh_sort_key(value: str) -> tuple[int, str]:
    match = MH_PATTERN.fullmatch(str(value).strip())
    if match:
        return int(match.group(1)), str(value)
    return math.inf, str(value)


def stringify_list(values: list[str]) -> str:
    ordered = sorted({str(value) for value in values if str(value)}, key=mh_sort_key)
    return ",".join(ordered)


def to_serializable(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover - defensive
            return value
    return value


def config_json_from_row(row: pd.Series, param_columns: list[str]) -> str:
    payload = {
        column: to_serializable(row[column])
        for column in param_columns
        if column in row.index and not pd.isna(row[column])
    }
    return json.dumps(payload, sort_keys=True)


def select_best_candidate(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None
    valid = frame.copy()
    valid = valid[valid["mean_rank_score"].notna()]
    valid = valid[valid["candidate_id"].fillna("").astype(str) != ""]
    if valid.empty:
        return None
    ordered = valid.sort_values(
        ["mean_rank_score", "mean_combined_f1", "mean_default_distance", "candidate_id"],
        ascending=[False, False, True, True],
    )
    return ordered.iloc[0]


def select_best_candidate_equal_support(frame: pd.DataFrame) -> pd.Series | None:
    """Select the best candidate among the highest-support candidates only."""
    if frame.empty or "support_count" not in frame.columns:
        return None
    valid = frame.copy()
    valid = valid[valid["support_count"].notna()]
    if valid.empty:
        return None
    max_support = valid["support_count"].max()
    return select_best_candidate(valid[valid["support_count"] == max_support])


def coerce_numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    converted = frame.copy()
    for column in converted.columns:
        if converted[column].dtype == object:
            numeric = pd.to_numeric(converted[column], errors="coerce")
            if numeric.notna().any() and numeric.notna().sum() == converted[column].notna().sum():
                converted[column] = numeric
    return converted


def discover_family_artifacts(results_root: Path) -> tuple[list[FamilyArtifact], list[dict[str, Any]], list[str]]:
    artifacts: list[FamilyArtifact] = []
    path_warnings: list[dict[str, Any]] = []
    empty_mh_dirs: list[str] = []

    payload_paths = sorted(results_root.rglob("best_configuration.json"))
    if not payload_paths:
        raise FileNotFoundError(f"No best_configuration.json files found under {results_root}")

    discovered_top_level_mh: set[str] = set()
    for payload_path in payload_paths:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        best_candidate = payload.get("best_candidate")
        if not isinstance(best_candidate, dict):
            continue

        mh_level = str(payload.get("mh_level", "")).strip()
        granularity = str(payload.get("granularity", "")).strip()
        scope_id = str(payload.get("scope_id", "")).strip()
        dataset_name = str(payload.get("dataset_name", "")).strip()
        detector_family = str(payload.get("detector_family", "")).strip()
        if not mh_level or not granularity or not scope_id or not detector_family:
            LOGGER.debug("Skipping incomplete payload at %s", payload_path)
            continue

        relative_parts = payload_path.relative_to(results_root).parts
        mh_parts_in_path = [part for part in relative_parts if MH_PATTERN.fullmatch(part)]
        path_mh_level = mh_parts_in_path[0] if mh_parts_in_path else ""
        if path_mh_level:
            discovered_top_level_mh.add(path_mh_level)
        if path_mh_level and path_mh_level != mh_level:
            path_warnings.append(
                {
                    "warning_type": "mh_path_mismatch",
                    "reported_mh_level": mh_level,
                    "path_mh_level": path_mh_level,
                    "best_configuration_path": str(payload_path),
                }
            )

        artifacts.append(
            FamilyArtifact(
                best_configuration_path=payload_path,
                candidate_metrics_path=payload_path.parent / "candidate_metrics.csv",
                mh_level=mh_level,
                granularity=granularity,
                scope_id=scope_id,
                dataset_name=dataset_name,
                detector_family=detector_family,
                path_mh_level=path_mh_level,
            )
        )

    for top_level_dir in sorted(
        [path for path in results_root.iterdir() if path.is_dir() and MH_PATTERN.fullmatch(path.name)],
        key=lambda path: mh_sort_key(path.name),
    ):
        if top_level_dir.name not in discovered_top_level_mh:
            empty_mh_dirs.append(top_level_dir.name)

    return artifacts, path_warnings, empty_mh_dirs


def build_best_configurations_table(artifacts: list[FamilyArtifact]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    for artifact in artifacts:
        payload = json.loads(artifact.best_configuration_path.read_text(encoding="utf-8"))
        best_candidate = payload.get("best_candidate")
        if not isinstance(best_candidate, dict):
            warnings.append(
                {
                    "warning_type": "missing_best_candidate",
                    "best_configuration_path": str(artifact.best_configuration_path),
                }
            )
            continue

        row: dict[str, Any] = {
            "mh_level": artifact.mh_level,
            "granularity": artifact.granularity,
            "scope_id": artifact.scope_id,
            "dataset_name": artifact.dataset_name,
            "detector_family": artifact.detector_family,
            "family_output_dir": str(artifact.best_configuration_path.parent),
            "best_configuration_path": str(artifact.best_configuration_path),
            "candidate_metrics_path": str(artifact.candidate_metrics_path),
            "configuration_json": json.dumps(payload.get("configuration", {}), sort_keys=True),
            "path_mh_level": artifact.path_mh_level,
        }
        row.update(best_candidate)
        configuration = payload.get("configuration", {})
        if isinstance(configuration, dict):
            for key, value in configuration.items():
                row[f"config__{key}"] = value
        rows.append(row)

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = coerce_numeric_columns(frame)
        frame = frame.sort_values(
            ["detector_family", "granularity", "dataset_name", "scope_id", "mh_level"],
            key=lambda series: series.map(mh_sort_key) if series.name == "mh_level" else series,
        )
    return frame, warnings


def build_candidate_metrics_table(artifacts: list[FamilyArtifact]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    warnings: list[dict[str, Any]] = []

    for artifact in artifacts:
        if not artifact.candidate_metrics_path.exists():
            warnings.append(
                {
                    "warning_type": "missing_candidate_metrics",
                    "candidate_metrics_path": str(artifact.candidate_metrics_path),
                    "best_configuration_path": str(artifact.best_configuration_path),
                }
            )
            continue

        frame = pd.read_csv(artifact.candidate_metrics_path)
        if frame.empty:
            warnings.append(
                {
                    "warning_type": "empty_candidate_metrics",
                    "candidate_metrics_path": str(artifact.candidate_metrics_path),
                }
            )
            continue

        frame = coerce_numeric_columns(frame)
        frame["mh_level"] = artifact.mh_level
        frame["granularity"] = artifact.granularity
        frame["scope_id"] = artifact.scope_id
        frame["dataset_name"] = artifact.dataset_name
        frame["detector_family"] = artifact.detector_family
        frame["family_output_dir"] = str(artifact.best_configuration_path.parent)
        frame["candidate_metrics_path"] = str(artifact.candidate_metrics_path)
        frame["best_configuration_path"] = str(artifact.best_configuration_path)
        frames.append(frame)

    if not frames:
        return pd.DataFrame(), warnings
    return pd.concat(frames, ignore_index=True), warnings


def infer_param_columns(frame: pd.DataFrame) -> list[str]:
    excluded = set(COMMON_CANDIDATE_COLUMNS)
    return sorted(column for column in frame.columns if column not in excluded)


def aggregate_candidate_metrics(
    frame: pd.DataFrame,
    group_keys: list[str],
    param_columns: list[str],
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    working = frame.copy()
    if "status" in working.columns:
        status_mask = working["status"].fillna("").astype(str).str.lower().isin({"ok", "success", ""})
        working = working[status_mask]
    working = working[working["rank_score"].notna()]
    if working.empty:
        return pd.DataFrame()

    aggregation: dict[str, Any] = {
        "mh_level": lambda values: stringify_list(list(values)),
        "rank_score": "mean",
        "combined_f1": "mean",
        "weighted_f1_mean": "mean",
        "default_distance": "mean",
        "new_prices_f1_mean": "mean",
        "new_prices_g_mean_mean": "mean",
        "new_products_f1_mean": "mean",
        "new_products_g_mean_mean": "mean",
    }
    for column in param_columns:
        if column in working.columns:
            aggregation[column] = "first"

    grouped = (
        working.groupby(group_keys + ["candidate_id"], dropna=False, sort=False)
        .agg(aggregation)
        .reset_index()
        .rename(
            columns={
                "mh_level": "mh_values",
                "rank_score": "mean_rank_score",
                "combined_f1": "mean_combined_f1",
                "weighted_f1_mean": "mean_weighted_f1",
                "default_distance": "mean_default_distance",
                "new_prices_f1_mean": "mean_new_prices_f1",
                "new_prices_g_mean_mean": "mean_new_prices_g_mean",
                "new_products_f1_mean": "mean_new_products_f1",
                "new_products_g_mean_mean": "mean_new_products_g_mean",
            }
        )
    )
    support_counts = (
        working.groupby(group_keys + ["candidate_id"], dropna=False, sort=False)
        .size()
        .reset_index(name="support_count")
    )
    grouped = grouped.merge(support_counts, on=group_keys + ["candidate_id"], how="left")
    grouped["mh_count"] = grouped["mh_values"].fillna("").astype(str).apply(
        lambda raw: len([item for item in raw.split(",") if item])
    )
    grouped["configuration_json"] = grouped.apply(
        lambda row: config_json_from_row(row, param_columns),
        axis=1,
    )
    return grouped


def compare_guidance_sets(
    *,
    candidate_metrics: pd.DataFrame,
    group_keys: list[str],
    param_columns: list[str],
    subset_mh: list[str],
) -> pd.DataFrame:
    if candidate_metrics.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for key_values, case_frame in candidate_metrics.groupby(group_keys, dropna=False, sort=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        case_metadata = dict(zip(group_keys, key_values))
        available_mh = sorted(case_frame["mh_level"].dropna().astype(str).unique().tolist(), key=mh_sort_key)
        subset_available = [mh for mh in subset_mh if mh in set(available_mh)]
        if not available_mh or not subset_available:
            continue

        full_agg = aggregate_candidate_metrics(case_frame, group_keys, param_columns)
        subset_agg = aggregate_candidate_metrics(case_frame[case_frame["mh_level"].isin(subset_available)], group_keys, param_columns)
        if full_agg.empty or subset_agg.empty:
            continue

        full_best = select_best_candidate(full_agg)
        subset_best = select_best_candidate(subset_agg)
        full_best_equal_support = select_best_candidate_equal_support(full_agg)
        subset_best_equal_support = select_best_candidate_equal_support(subset_agg)
        if (
            full_best is None
            or subset_best is None
            or full_best_equal_support is None
            or subset_best_equal_support is None
        ):
            continue

        full_subset_best = full_agg[full_agg["candidate_id"] == subset_best["candidate_id"]]
        if full_subset_best.empty:
            continue
        full_subset_best_row = full_subset_best.iloc[0]

        full_subset_best_equal_support = full_agg[full_agg["candidate_id"] == subset_best_equal_support["candidate_id"]]
        if full_subset_best_equal_support.empty:
            continue
        full_subset_best_equal_support_row = full_subset_best_equal_support.iloc[0]

        ranking_merge = full_agg[["candidate_id", "mean_rank_score"]].merge(
            subset_agg[["candidate_id", "mean_rank_score"]],
            on="candidate_id",
            how="inner",
            suffixes=("_full", "_subset"),
        )
        spearman = (
            ranking_merge["mean_rank_score_full"].corr(ranking_merge["mean_rank_score_subset"], method="spearman")
            if len(ranking_merge) >= 2
            else math.nan
        )

        row = {
            **case_metadata,
            "full_mh_values": stringify_list(available_mh),
            "full_mh_count": len(available_mh),
            "subset_target_mh_values": stringify_list(subset_mh),
            "subset_available_mh_values": stringify_list(subset_available),
            "subset_available_mh_count": len(subset_available),
            "candidate_count_full": int(len(full_agg)),
            "candidate_count_subset": int(len(subset_agg)),
            "max_support_count_full": int(full_agg["support_count"].max()),
            "max_support_count_subset": int(subset_agg["support_count"].max()),
            "full_best_candidate_id": str(full_best["candidate_id"]),
            "full_best_configuration_json": str(full_best["configuration_json"]),
            "full_best_support_count": int(full_best["support_count"]),
            "subset_best_candidate_id": str(subset_best["candidate_id"]),
            "subset_best_configuration_json": str(subset_best["configuration_json"]),
            "subset_best_support_count": int(subset_best["support_count"]),
            "config_match": bool(str(full_best["candidate_id"]) == str(subset_best["candidate_id"])),
            "equal_support_full_best_candidate_id": str(full_best_equal_support["candidate_id"]),
            "equal_support_full_best_configuration_json": str(full_best_equal_support["configuration_json"]),
            "equal_support_full_best_support_count": int(full_best_equal_support["support_count"]),
            "equal_support_subset_best_candidate_id": str(subset_best_equal_support["candidate_id"]),
            "equal_support_subset_best_configuration_json": str(subset_best_equal_support["configuration_json"]),
            "equal_support_subset_best_support_count": int(subset_best_equal_support["support_count"]),
            "equal_support_config_match": bool(
                str(full_best_equal_support["candidate_id"]) == str(subset_best_equal_support["candidate_id"])
            ),
            "rank_score_spearman": spearman,
            "full_best_mean_rank_score": float(full_best["mean_rank_score"]),
            "subset_choice_mean_rank_score_on_full": float(full_subset_best_row["mean_rank_score"]),
            "rank_score_regret": float(full_best["mean_rank_score"] - full_subset_best_row["mean_rank_score"]),
            "full_best_mean_combined_f1": float(full_best["mean_combined_f1"]),
            "subset_choice_mean_combined_f1_on_full": float(full_subset_best_row["mean_combined_f1"]),
            "combined_f1_regret": float(full_best["mean_combined_f1"] - full_subset_best_row["mean_combined_f1"]),
            "full_best_mean_new_prices_f1": float(full_best["mean_new_prices_f1"]),
            "subset_choice_mean_new_prices_f1_on_full": float(full_subset_best_row["mean_new_prices_f1"]),
            "new_prices_f1_regret": float(full_best["mean_new_prices_f1"] - full_subset_best_row["mean_new_prices_f1"]),
            "full_best_mean_new_prices_g_mean": float(full_best["mean_new_prices_g_mean"]),
            "subset_choice_mean_new_prices_g_mean_on_full": float(full_subset_best_row["mean_new_prices_g_mean"]),
            "new_prices_g_mean_regret": float(
                full_best["mean_new_prices_g_mean"] - full_subset_best_row["mean_new_prices_g_mean"]
            ),
            "full_best_mean_new_products_f1": float(full_best["mean_new_products_f1"]),
            "subset_choice_mean_new_products_f1_on_full": float(full_subset_best_row["mean_new_products_f1"]),
            "new_products_f1_regret": float(
                full_best["mean_new_products_f1"] - full_subset_best_row["mean_new_products_f1"]
            ),
            "full_best_mean_new_products_g_mean": float(full_best["mean_new_products_g_mean"]),
            "subset_choice_mean_new_products_g_mean_on_full": float(
                full_subset_best_row["mean_new_products_g_mean"]
            ),
            "new_products_g_mean_regret": float(
                full_best["mean_new_products_g_mean"] - full_subset_best_row["mean_new_products_g_mean"]
            ),
            "equal_support_full_best_mean_rank_score": float(full_best_equal_support["mean_rank_score"]),
            "equal_support_subset_choice_mean_rank_score_on_full": float(
                full_subset_best_equal_support_row["mean_rank_score"]
            ),
            "equal_support_rank_score_regret": float(
                full_best_equal_support["mean_rank_score"] - full_subset_best_equal_support_row["mean_rank_score"]
            ),
            "equal_support_full_best_mean_combined_f1": float(full_best_equal_support["mean_combined_f1"]),
            "equal_support_subset_choice_mean_combined_f1_on_full": float(
                full_subset_best_equal_support_row["mean_combined_f1"]
            ),
            "equal_support_combined_f1_regret": float(
                full_best_equal_support["mean_combined_f1"] - full_subset_best_equal_support_row["mean_combined_f1"]
            ),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(group_keys)


def summarize_best_configuration_distribution(best_configurations: pd.DataFrame) -> pd.DataFrame:
    if best_configurations.empty:
        return pd.DataFrame()
    return (
        best_configurations.groupby(["detector_family", "mh_level", "candidate_id", "configuration_json"], dropna=False)
        .size()
        .reset_index(name="case_count")
        .sort_values(
            ["detector_family", "mh_level", "case_count", "candidate_id"],
            ascending=[True, True, False, True],
            key=lambda series: series.map(mh_sort_key) if series.name == "mh_level" else series,
        )
    )


def filter_artifacts(
    artifacts: list[FamilyArtifact],
    detector_filter: set[str],
    granularity_filter: set[str],
) -> list[FamilyArtifact]:
    filtered = []
    for artifact in artifacts:
        if detector_filter and artifact.detector_family not in detector_filter:
            continue
        if granularity_filter and artifact.granularity not in granularity_filter:
            continue
        filtered.append(artifact)
    return filtered


def write_summary(
    *,
    output_dir: Path,
    results_root: Path,
    subset_mh: list[str],
    best_configurations: pd.DataFrame,
    guidance_by_case: pd.DataFrame,
    guidance_by_detector: pd.DataFrame,
    path_warnings: list[dict[str, Any]],
    empty_mh_dirs: list[str],
    additional_warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = {
        "results_root": str(results_root),
        "subset_mh_values": subset_mh,
        "best_configuration_count": int(len(best_configurations)),
        "guidance_case_count": int(len(guidance_by_case)),
        "guidance_detector_count": int(len(guidance_by_detector)),
        "path_warning_count": int(len(path_warnings)),
        "empty_top_level_mh_count": int(len(empty_mh_dirs)),
        "additional_warning_count": int(len(additional_warnings)),
    }

    if not guidance_by_case.empty:
        summary.update(
            {
                "config_match_rate": float(guidance_by_case["config_match"].mean()),
                "median_combined_f1_regret": float(guidance_by_case["combined_f1_regret"].median()),
                "median_rank_score_regret": float(guidance_by_case["rank_score_regret"].median()),
                "max_combined_f1_regret": float(guidance_by_case["combined_f1_regret"].max()),
            }
        )

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Statistical Guidance Analysis",
        "",
        f"- Results root: `{results_root}`",
        f"- Subset mh values: `{','.join(subset_mh)}`",
        f"- Best-configuration rows: {len(best_configurations)}",
        f"- Guidance comparison rows: {len(guidance_by_case)}",
        f"- Path warnings: {len(path_warnings)}",
        f"- Empty top-level mh directories: {', '.join(empty_mh_dirs) if empty_mh_dirs else '(none)'}",
        "",
    ]

    if not guidance_by_case.empty:
        match_rate = guidance_by_case["config_match"].mean() * 100.0
        lines.extend(
            [
                "## Subset-vs-Full Guidance",
                "",
                f"- Configuration match rate: {match_rate:.1f}%",
                f"- Median combined F1 regret: {guidance_by_case['combined_f1_regret'].median():.6f}",
                f"- Median rank-score regret: {guidance_by_case['rank_score_regret'].median():.6f}",
                f"- Max combined F1 regret: {guidance_by_case['combined_f1_regret'].max():.6f}",
                "",
            ]
        )

    if not guidance_by_detector.empty:
        lines.extend(
            [
                "## Detector-Level Summary",
                "",
                "| Detector | Best | Support | Equal-Support Best | Equal-Support Match |",
                "| --- | --- | ---: | --- | ---: |",
            ]
        )
        for _, row in guidance_by_detector.iterrows():
            lines.append(
                "| "
                f"{row['detector_family']} | "
                f"{row['full_best_candidate_id']} | "
                f"{int(row['full_best_support_count'])}/{int(row['max_support_count_full'])} | "
                f"{row['equal_support_full_best_candidate_id']} | "
                f"{'yes' if row['equal_support_config_match'] else 'no'} |"
            )
        lines.append("")

        lines.extend(
            [
                "## Detector-Level Regret",
                "",
                "| Detector | Match | Combined F1 Regret | Equal-Support Combined F1 Regret |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for _, row in guidance_by_detector.iterrows():
            lines.append(
                "| "
                f"{row['detector_family']} | "
                f"{'yes' if row['config_match'] else 'no'} | "
                f"{row['combined_f1_regret']:.6f} | "
                f"{row['equal_support_combined_f1_regret']:.6f} |"
            )
        lines.append("")

    if not guidance_by_case.empty:
        worst = guidance_by_case.sort_values("combined_f1_regret", ascending=False).head(10)
        lines.extend(
            [
                "## Worst Case Regrets",
                "",
                "| Detector | Scope | Full Best | Subset Best | Combined F1 Regret |",
                "| --- | --- | --- | --- | ---: |",
            ]
        )
        for _, row in worst.iterrows():
            lines.append(
                "| "
                f"{row['detector_family']} | "
                f"{row['scope_id']} | "
                f"{row['full_best_candidate_id']} | "
                f"{row['subset_best_candidate_id']} | "
                f"{row['combined_f1_regret']:.6f} |"
            )
        lines.append("")

    if path_warnings or additional_warnings:
        lines.extend(["## Warnings", ""])
        for warning in path_warnings + additional_warnings:
            warning_type = warning.get("warning_type", "warning")
            details = ", ".join(f"{key}={value}" for key, value in warning.items() if key != "warning_type")
            lines.append(f"- `{warning_type}`: {details}")
        lines.append("")

    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    results_root = args.results_root.resolve()
    if not results_root.exists():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")

    output_dir = args.output_dir.resolve() if args.output_dir else (results_root / "analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    subset_mh = parse_csv_list(args.subset_mh)
    detector_filter = set(parse_csv_list(args.detectors))
    granularity_filter = set(parse_csv_list(args.granularities))

    artifacts, path_warnings, empty_mh_dirs = discover_family_artifacts(results_root)
    artifacts = filter_artifacts(artifacts, detector_filter=detector_filter, granularity_filter=granularity_filter)
    if not artifacts:
        raise FileNotFoundError("No matching family outputs found after applying filters.")

    best_configurations, best_warnings = build_best_configurations_table(artifacts)
    candidate_metrics, candidate_warnings = build_candidate_metrics_table(artifacts)
    if candidate_metrics.empty:
        raise FileNotFoundError("No candidate_metrics.csv files could be loaded.")

    param_columns = infer_param_columns(candidate_metrics)
    guidance_by_case = compare_guidance_sets(
        candidate_metrics=candidate_metrics,
        group_keys=["detector_family", "granularity", "dataset_name", "scope_id"],
        param_columns=param_columns,
        subset_mh=subset_mh,
    )
    guidance_by_detector = compare_guidance_sets(
        candidate_metrics=candidate_metrics,
        group_keys=["detector_family"],
        param_columns=param_columns,
        subset_mh=subset_mh,
    )
    best_distribution = summarize_best_configuration_distribution(best_configurations)

    best_configurations.to_csv(output_dir / "best_configurations_all_cases.csv", index=False)
    candidate_metrics.to_csv(output_dir / "candidate_metrics_all_cases.csv", index=False)
    guidance_by_case.to_csv(output_dir / "subset_guidance_by_case.csv", index=False)
    guidance_by_detector.to_csv(output_dir / "subset_guidance_by_detector.csv", index=False)
    best_distribution.to_csv(output_dir / "best_configuration_distribution.csv", index=False)

    warning_rows = path_warnings + best_warnings + candidate_warnings
    pd.DataFrame(warning_rows).to_csv(output_dir / "warnings.csv", index=False)

    summary = write_summary(
        output_dir=output_dir,
        results_root=results_root,
        subset_mh=subset_mh,
        best_configurations=best_configurations,
        guidance_by_case=guidance_by_case,
        guidance_by_detector=guidance_by_detector,
        path_warnings=path_warnings,
        empty_mh_dirs=empty_mh_dirs,
        additional_warnings=best_warnings + candidate_warnings,
    )

    LOGGER.info("Wrote analysis outputs to %s", output_dir)
    LOGGER.info("Configuration match rate: %s", summary.get("config_match_rate"))
    LOGGER.info("Median combined F1 regret: %s", summary.get("median_combined_f1_regret"))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - CLI error path
        LOGGER.error("Analysis failed: %s", exc)
        raise
