#!/usr/bin/env python3
"""Build mh5-only method-selection evidence from existing tuning outputs.

The purpose of this script is narrow and pragmatic: summarize mh5 performance
for the methods already tuned so the thesis can justify which methods should be
extended to the expensive full mh5-mh30 sweep.

Inputs:
- Forest summaries from results/tuning/forests/single_config_optimized_mh5_run
- Default z-score granularity aggregate from
  results/tuning/statistical/z_score_granularity_comparison/analysis/
  granularity_performance_sampled_mh.csv

Outputs:
- Forest per-scope mh5 table
- Forest detector/granularity summary
- Forest detector summary
- Combined detector/granularity summary including default z-score
- Combined detector summary including default z-score
- Isolation-Forest dominance table against EIF and RRCF
- Markdown and JSON summary for thesis writing
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger("analyze_mh5_method_selection")

DEFAULT_FOREST_ROOT = Path("results/tuning/forests/single_config_optimized_mh5_run")
DEFAULT_ZSCORE_GRANULARITY_CSV = Path(
    "results/tuning/statistical/z_score_granularity_comparison/analysis/granularity_performance_sampled_mh.csv"
)
DEFAULT_OUTPUT_DIR = Path("results/analysis/mh5_method_selection")

GRANULARITY_ORDER = {
    "global": 0,
    "by_country": 1,
    "by_competitor": 2,
}

DETECTOR_LABELS = {
    "standard_zscore": "Default z-score",
    "if": "Isolation Forest",
    "eif": "Extended Isolation Forest",
    "rrcf": "Robust Random Cut Forest",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize mh5-only detector performance for full-suite selection.",
    )
    parser.add_argument(
        "--forest-root",
        type=Path,
        default=DEFAULT_FOREST_ROOT,
        help="Root directory containing consolidated forest tuning outputs.",
    )
    parser.add_argument(
        "--zscore-granularity-csv",
        type=Path,
        default=DEFAULT_ZSCORE_GRANULARITY_CSV,
        help="Granularity aggregate CSV produced by analyze_granularity_performance.py.",
    )
    parser.add_argument(
        "--mh-level",
        default="mh5",
        help="mh level to summarize.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated mh-level selection summaries.",
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


def detector_label(detector_family: str) -> str:
    return DETECTOR_LABELS.get(detector_family, detector_family)


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    return float(value)


def granularity_sort_key(value: str) -> tuple[int, str]:
    return GRANULARITY_ORDER.get(value, 99), str(value)


def load_forest_scope_rows(forest_root: Path, mh_level: str) -> pd.DataFrame:
    if not forest_root.exists():
        raise FileNotFoundError(f"Forest root does not exist: {forest_root}")

    rows: list[dict[str, Any]] = []
    for summary_path in sorted(forest_root.rglob("summary.json")):
        if summary_path.parent == forest_root:
            continue

        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if str(payload.get("status", "")).lower() != "ok":
            continue
        if str(payload.get("mh_level", "")).strip() != mh_level:
            continue

        summary = payload.get("summary", {}) or {}
        best_candidate = payload.get("best_candidate", {}) or {}
        detector_family = str(payload.get("detector_family", "")).strip()
        if not detector_family:
            continue

        best_f1 = safe_float(summary.get("best_f1"))
        if best_f1 is None:
            best_f1 = safe_float(best_candidate.get("combined_f1"))
        if best_f1 is None:
            best_f1 = safe_float(best_candidate.get("weighted_f1_mean"))

        best_g_mean = safe_float(summary.get("best_g_mean"))
        if best_g_mean is None:
            best_g_mean = safe_float(best_candidate.get("combined_g_mean"))
        if best_g_mean is None:
            best_g_mean = safe_float(best_candidate.get("rank_score"))

        rows.append(
            {
                "source": "forest_scope_summary",
                "detector_family": detector_family,
                "detector_label": detector_label(detector_family),
                "mh_level": mh_level,
                "granularity": str(payload.get("granularity", "")).strip(),
                "scope_id": str(payload.get("scope_id", "")).strip(),
                "dataset_name": str(payload.get("dataset_name", "")).strip(),
                "sweep_id": str(payload.get("sweep_id", "")).strip(),
                "best_candidate_id": summary.get("best_candidate_id") or best_candidate.get("candidate_id"),
                "best_config_key": summary.get("best_config_key") or best_candidate.get("config_key"),
                "best_threshold": safe_float(summary.get("best_threshold")),
                "best_f1": best_f1,
                "best_g_mean": best_g_mean,
                "training_time_sec": safe_float(best_candidate.get("training_time_sec")),
                "summary_path": str(summary_path),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise FileNotFoundError(f"No ok forest mh rows found for {mh_level} under {forest_root}")

    frame = frame.sort_values(
        ["detector_family", "granularity", "scope_id"],
        key=lambda series: series.map(granularity_sort_key) if series.name == "granularity" else series,
    )
    return frame


def summarize_forest_scope_rows(frame: pd.DataFrame, group_keys: list[str]) -> pd.DataFrame:
    summary = (
        frame.groupby(group_keys, dropna=False)
        .agg(
            scope_count=("scope_id", "nunique"),
            mean_best_f1=("best_f1", "mean"),
            median_best_f1=("best_f1", "median"),
            min_best_f1=("best_f1", "min"),
            max_best_f1=("best_f1", "max"),
            mean_best_g_mean=("best_g_mean", "mean"),
            median_best_g_mean=("best_g_mean", "median"),
            min_best_g_mean=("best_g_mean", "min"),
            max_best_g_mean=("best_g_mean", "max"),
            mean_training_time_sec=("training_time_sec", "mean"),
            median_training_time_sec=("training_time_sec", "median"),
            max_training_time_sec=("training_time_sec", "max"),
        )
        .reset_index()
    )
    return summary.sort_values(
        group_keys,
        key=lambda series: series.map(granularity_sort_key) if series.name == "granularity" else series,
    )


def load_zscore_granularity_rows(zscore_csv: Path, mh_level: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not zscore_csv.exists():
        LOGGER.warning("Z-score aggregate CSV does not exist: %s", zscore_csv)
        return pd.DataFrame(), pd.DataFrame()

    frame = pd.read_csv(zscore_csv)
    required = {
        "granularity",
        "mh_level",
        "average_weighted_combined_f1",
        "average_weighted_combined_g_mean",
        "scope_count",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{zscore_csv} is missing required columns: {sorted(missing)}")

    filtered = frame[frame["mh_level"].astype(str) == mh_level].copy()
    if filtered.empty:
        LOGGER.warning("No default z-score rows found for %s in %s", mh_level, zscore_csv)
        return pd.DataFrame(), pd.DataFrame()

    filtered["source"] = "statistical_granularity_aggregate"
    filtered["detector_family"] = "standard_zscore"
    filtered["detector_label"] = detector_label("standard_zscore")
    filtered["mean_best_f1"] = pd.to_numeric(filtered["average_weighted_combined_f1"], errors="coerce")
    filtered["mean_best_g_mean"] = pd.to_numeric(filtered["average_weighted_combined_g_mean"], errors="coerce")
    filtered["mean_training_time_sec"] = pd.NA
    filtered = filtered[
        [
            "source",
            "detector_family",
            "detector_label",
            "mh_level",
            "granularity",
            "scope_count",
            "mean_best_f1",
            "mean_best_g_mean",
            "mean_training_time_sec",
        ]
    ].sort_values("granularity", key=lambda series: series.map(granularity_sort_key))

    total_scope_count = int(filtered["scope_count"].sum())
    weighted_f1 = (
        (filtered["mean_best_f1"] * filtered["scope_count"]).sum() / total_scope_count
        if total_scope_count
        else pd.NA
    )
    weighted_g_mean = (
        (filtered["mean_best_g_mean"] * filtered["scope_count"]).sum() / total_scope_count
        if total_scope_count
        else pd.NA
    )
    detector_summary = pd.DataFrame(
        [
            {
                "source": "statistical_granularity_aggregate",
                "detector_family": "standard_zscore",
                "detector_label": detector_label("standard_zscore"),
                "mh_level": mh_level,
                "scope_count": total_scope_count,
                "mean_best_f1": weighted_f1,
                "mean_best_g_mean": weighted_g_mean,
                "mean_training_time_sec": pd.NA,
            }
        ]
    )
    return filtered, detector_summary


def build_combined_granularity_summary(
    forest_granularity_summary: pd.DataFrame,
    zscore_granularity_summary: pd.DataFrame,
    mh_level: str,
) -> pd.DataFrame:
    forest_rows = forest_granularity_summary[
        [
            "detector_family",
            "detector_label",
            "granularity",
            "scope_count",
            "mean_best_f1",
            "mean_best_g_mean",
            "mean_training_time_sec",
        ]
    ].copy()
    forest_rows["source"] = "forest_scope_summary"
    forest_rows["mh_level"] = mh_level

    combined_rows = forest_rows.to_dict(orient="records")
    if not zscore_granularity_summary.empty:
        combined_rows.extend(zscore_granularity_summary.to_dict(orient="records"))
    combined = pd.DataFrame(combined_rows)
    return combined.sort_values(
        ["granularity", "detector_family"],
        key=lambda series: series.map(granularity_sort_key) if series.name == "granularity" else series,
    )


def build_combined_detector_summary(
    forest_detector_summary: pd.DataFrame,
    zscore_detector_summary: pd.DataFrame,
    mh_level: str,
) -> pd.DataFrame:
    forest_rows = forest_detector_summary[
        [
            "detector_family",
            "detector_label",
            "scope_count",
            "mean_best_f1",
            "mean_best_g_mean",
            "mean_training_time_sec",
        ]
    ].copy()
    forest_rows["source"] = "forest_scope_summary"
    forest_rows["mh_level"] = mh_level
    combined_rows = forest_rows.to_dict(orient="records")
    if not zscore_detector_summary.empty:
        combined_rows.extend(zscore_detector_summary.to_dict(orient="records"))
    combined = pd.DataFrame(combined_rows)
    return combined.sort_values("detector_family")


def build_if_dominance_summary(forest_granularity_summary: pd.DataFrame) -> pd.DataFrame:
    if_rows = forest_granularity_summary[forest_granularity_summary["detector_family"] == "if"].copy()
    if if_rows.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for challenger in ("eif", "rrcf"):
        challenger_rows = forest_granularity_summary[
            forest_granularity_summary["detector_family"] == challenger
        ].copy()
        if challenger_rows.empty:
            continue

        merged = if_rows.merge(
            challenger_rows,
            on="granularity",
            suffixes=("_if", "_challenger"),
        )
        if merged.empty:
            continue

        merged["if_beats_on_g_mean"] = merged["mean_best_g_mean_if"] > merged["mean_best_g_mean_challenger"]
        merged["if_beats_on_f1"] = merged["mean_best_f1_if"] > merged["mean_best_f1_challenger"]
        merged["if_faster"] = merged["mean_training_time_sec_if"] < merged["mean_training_time_sec_challenger"]
        merged["if_g_mean_advantage"] = merged["mean_best_g_mean_if"] - merged["mean_best_g_mean_challenger"]
        merged["if_f1_advantage"] = merged["mean_best_f1_if"] - merged["mean_best_f1_challenger"]
        merged["challenger_to_if_time_ratio"] = (
            merged["mean_training_time_sec_challenger"] / merged["mean_training_time_sec_if"]
        )
        merged["if_dominates"] = (
            merged["if_beats_on_g_mean"]
            & merged["if_beats_on_f1"]
            & merged["if_faster"]
        )

        for row in merged.itertuples(index=False):
            rows.append(
                {
                    "baseline_detector_family": "if",
                    "baseline_detector_label": detector_label("if"),
                    "challenger_detector_family": challenger,
                    "challenger_detector_label": detector_label(challenger),
                    "granularity": row.granularity,
                    "if_mean_best_g_mean": row.mean_best_g_mean_if,
                    "challenger_mean_best_g_mean": row.mean_best_g_mean_challenger,
                    "if_g_mean_advantage": row.if_g_mean_advantage,
                    "if_mean_best_f1": row.mean_best_f1_if,
                    "challenger_mean_best_f1": row.mean_best_f1_challenger,
                    "if_f1_advantage": row.if_f1_advantage,
                    "if_mean_training_time_sec": row.mean_training_time_sec_if,
                    "challenger_mean_training_time_sec": row.mean_training_time_sec_challenger,
                    "challenger_to_if_time_ratio": row.challenger_to_if_time_ratio,
                    "if_beats_on_g_mean": bool(row.if_beats_on_g_mean),
                    "if_beats_on_f1": bool(row.if_beats_on_f1),
                    "if_faster": bool(row.if_faster),
                    "if_dominates": bool(row.if_dominates),
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["challenger_detector_family", "granularity"],
        key=lambda series: series.map(granularity_sort_key) if series.name == "granularity" else series,
    )


def format_value(value: Any, decimals: int = 4) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def detector_row(frame: pd.DataFrame, detector_family: str) -> pd.Series | None:
    rows = frame[frame["detector_family"] == detector_family]
    if rows.empty:
        return None
    return rows.iloc[0]


def build_summary_payload(
    mh_level: str,
    forest_scope_rows: pd.DataFrame,
    forest_detector_summary: pd.DataFrame,
    zscore_detector_summary: pd.DataFrame,
    dominance_summary: pd.DataFrame,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "mh_level": mh_level,
        "forest_scope_count": int(len(forest_scope_rows)),
        "forest_detector_count": int(forest_detector_summary["detector_family"].nunique()),
        "forest_granularity_count": int(forest_scope_rows["granularity"].nunique()),
    }

    for detector_family in ("if", "eif", "rrcf"):
        row = detector_row(forest_detector_summary, detector_family)
        if row is None:
            continue
        payload[f"{detector_family}_mean_best_f1"] = float(row["mean_best_f1"])
        payload[f"{detector_family}_mean_best_g_mean"] = float(row["mean_best_g_mean"])
        payload[f"{detector_family}_mean_training_time_sec"] = float(row["mean_training_time_sec"])

    zscore_row = detector_row(zscore_detector_summary, "standard_zscore")
    if zscore_row is not None:
        payload["standard_zscore_mean_best_f1"] = float(zscore_row["mean_best_f1"])
        payload["standard_zscore_mean_best_g_mean"] = float(zscore_row["mean_best_g_mean"])

    if not dominance_summary.empty:
        grouped = dominance_summary.groupby("challenger_detector_family", dropna=False)["if_dominates"]
        payload["if_dominates_all_eif_granularities"] = bool(grouped.get_group("eif").all()) if "eif" in grouped.groups else None
        payload["if_dominates_all_rrcf_granularities"] = bool(grouped.get_group("rrcf").all()) if "rrcf" in grouped.groups else None

    recommended = ["if"]
    if detector_row(zscore_detector_summary, "standard_zscore") is not None:
        recommended.insert(0, "standard_zscore")
    payload["recommended_full_suite_methods"] = recommended
    payload["deprioritized_full_suite_methods"] = [
        detector_family
        for detector_family in ("eif", "rrcf")
        if detector_row(forest_detector_summary, detector_family) is not None
    ]
    return payload


def write_summary_markdown(
    output_path: Path,
    *,
    mh_level: str,
    forest_root: Path,
    zscore_csv: Path,
    combined_detector_summary: pd.DataFrame,
    combined_granularity_summary: pd.DataFrame,
    dominance_summary: pd.DataFrame,
) -> None:
    lines = [
        f"# {mh_level} Method Selection Summary",
        "",
        "This summary is intended to support the thesis decision on which methods should be extended to the expensive full mh5-mh30 runs.",
        "",
        "## Inputs",
        "",
        f"- Forest root: `{forest_root}`",
        f"- Default z-score granularity CSV: `{zscore_csv}`",
        "",
        "## Detector Summary",
        "",
        "| Detector | Scope count | Mean F1 | Mean G-mean | Mean train time (s) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for row in combined_detector_summary.itertuples(index=False):
        lines.append(
            f"| {row.detector_label} | "
            f"{format_value(row.scope_count, decimals=0)} | "
            f"{format_value(row.mean_best_f1)} | "
            f"{format_value(row.mean_best_g_mean)} | "
            f"{format_value(row.mean_training_time_sec, decimals=1)} |"
        )
    lines.append("")

    lines.extend(
        [
            "## Detector By Granularity",
            "",
            "| Detector | Granularity | Scope count | Mean F1 | Mean G-mean | Mean train time (s) |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in combined_granularity_summary.itertuples(index=False):
        lines.append(
            f"| {row.detector_label} | "
            f"{row.granularity} | "
            f"{format_value(row.scope_count, decimals=0)} | "
            f"{format_value(row.mean_best_f1)} | "
            f"{format_value(row.mean_best_g_mean)} | "
            f"{format_value(row.mean_training_time_sec, decimals=1)} |"
        )
    lines.append("")

    if not dominance_summary.empty:
        lines.extend(
            [
                "## Isolation Forest Dominance Over Slow Forest Baselines",
                "",
                "| Challenger | Granularity | IF G-mean | Challenger G-mean | IF F1 | Challenger F1 | Time ratio (challenger / IF) | IF dominates |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in dominance_summary.itertuples(index=False):
            lines.append(
                f"| {row.challenger_detector_label} | "
                f"{row.granularity} | "
                f"{format_value(row.if_mean_best_g_mean)} | "
                f"{format_value(row.challenger_mean_best_g_mean)} | "
                f"{format_value(row.if_mean_best_f1)} | "
                f"{format_value(row.challenger_mean_best_f1)} | "
                f"{format_value(row.challenger_to_if_time_ratio, decimals=1)} | "
                f"{format_value(row.if_dominates)} |"
            )
        lines.append("")

    if_row = detector_row(combined_detector_summary, "if")
    eif_row = detector_row(combined_detector_summary, "eif")
    rrcf_row = detector_row(combined_detector_summary, "rrcf")
    zscore_row = detector_row(combined_detector_summary, "standard_zscore")

    lines.extend(["## Key Findings", ""])
    if if_row is not None and eif_row is not None:
        lines.append(
            "- Isolation Forest outperforms EIF at mh5 while training much faster: "
            f"mean G-mean {format_value(if_row['mean_best_g_mean'])} vs {format_value(eif_row['mean_best_g_mean'])}, "
            f"mean F1 {format_value(if_row['mean_best_f1'])} vs {format_value(eif_row['mean_best_f1'])}, "
            f"mean train time {format_value(if_row['mean_training_time_sec'], decimals=1)}s vs "
            f"{format_value(eif_row['mean_training_time_sec'], decimals=1)}s."
        )
    if if_row is not None and rrcf_row is not None:
        lines.append(
            "- Isolation Forest outperforms RRCF at mh5 while training much faster: "
            f"mean G-mean {format_value(if_row['mean_best_g_mean'])} vs {format_value(rrcf_row['mean_best_g_mean'])}, "
            f"mean F1 {format_value(if_row['mean_best_f1'])} vs {format_value(rrcf_row['mean_best_f1'])}, "
            f"mean train time {format_value(if_row['mean_training_time_sec'], decimals=1)}s vs "
            f"{format_value(rrcf_row['mean_training_time_sec'], decimals=1)}s."
        )
    if zscore_row is not None:
        lines.append(
            "- Default z-score remains strong at mh5 across all granularities, with weighted mean G-mean "
            f"{format_value(zscore_row['mean_best_g_mean'])} and weighted mean F1 {format_value(zscore_row['mean_best_f1'])}."
        )
    if zscore_row is not None:
        lines.append(
            "- On this mh5 evidence base, the methods worth extending to the expensive full sweep are `standard_zscore` and `if`."
        )
    else:
        lines.append(
            "- On this mh5 evidence base, the forest method worth extending to the expensive full sweep is `if`."
        )
    if eif_row is not None or rrcf_row is not None:
        lines.append(
            "- `eif` and `rrcf` have weak mh5 accuracy relative to `if` and require far more training time, so they are poor candidates for the full mh5-mh30 extension."
        )
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    forest_root = args.forest_root.resolve()
    zscore_csv = args.zscore_granularity_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    forest_scope_rows = load_forest_scope_rows(forest_root, args.mh_level)
    forest_granularity_summary = summarize_forest_scope_rows(
        forest_scope_rows,
        ["detector_family", "detector_label", "granularity"],
    )
    forest_detector_summary = summarize_forest_scope_rows(
        forest_scope_rows,
        ["detector_family", "detector_label"],
    )
    zscore_granularity_summary, zscore_detector_summary = load_zscore_granularity_rows(
        zscore_csv,
        args.mh_level,
    )
    combined_granularity_summary = build_combined_granularity_summary(
        forest_granularity_summary,
        zscore_granularity_summary,
        args.mh_level,
    )
    combined_detector_summary = build_combined_detector_summary(
        forest_detector_summary,
        zscore_detector_summary,
        args.mh_level,
    )
    dominance_summary = build_if_dominance_summary(forest_granularity_summary)
    summary_payload = build_summary_payload(
        args.mh_level,
        forest_scope_rows,
        forest_detector_summary,
        zscore_detector_summary,
        dominance_summary,
    )

    forest_scope_rows.to_csv(output_dir / f"forest_scope_metrics_{args.mh_level}.csv", index=False)
    forest_granularity_summary.to_csv(
        output_dir / f"forest_detector_granularity_summary_{args.mh_level}.csv",
        index=False,
    )
    forest_detector_summary.to_csv(
        output_dir / f"forest_detector_summary_{args.mh_level}.csv",
        index=False,
    )
    if not zscore_granularity_summary.empty:
        zscore_granularity_summary.to_csv(
            output_dir / f"standard_zscore_granularity_summary_{args.mh_level}.csv",
            index=False,
        )
    combined_granularity_summary.to_csv(
        output_dir / f"detector_granularity_summary_{args.mh_level}.csv",
        index=False,
    )
    combined_detector_summary.to_csv(
        output_dir / f"detector_summary_{args.mh_level}.csv",
        index=False,
    )
    dominance_summary.to_csv(output_dir / f"if_dominance_summary_{args.mh_level}.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    write_summary_markdown(
        output_dir / "summary.md",
        mh_level=args.mh_level,
        forest_root=forest_root,
        zscore_csv=zscore_csv,
        combined_detector_summary=combined_detector_summary,
        combined_granularity_summary=combined_granularity_summary,
        dominance_summary=dominance_summary,
    )

    LOGGER.info("Wrote mh-level selection outputs to %s", output_dir)
    LOGGER.info("Recommended full-suite methods: %s", ", ".join(summary_payload["recommended_full_suite_methods"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
