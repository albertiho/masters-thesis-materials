#!/usr/bin/env python3
"""Aggregate Isolation Forest cross-horizon scores and render a granularity comparison figure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from research.training.scripts.thesis_plot_style import (
    COMBINED_F1_COLOR,
    COMBINED_GMEAN_COLOR,
    GRANULARITY_LEGEND_TITLE,
    GRANULARITY_LABELS,
    GRID_X_COLOR,
    GRID_Y_COLOR,
    METRIC_LEGEND_TITLE,
    PLOT_LABEL_FONTSIZE,
    PLOT_LEGEND_FONTSIZE,
    PLOT_LEGEND_TITLE_FONTSIZE,
    PLOT_TICK_FONTSIZE,
    SCORE_YMAX,
    SCORE_YMIN,
    SCORE_YTICKS,
    WEIGHTED_SCORE_METRIC_KEYS,
    X_AXIS_LABEL,
    granularity_style,
    granularity_legend_handles,
    metric_legend_handles,
    save_thesis_media,
)

SAMPLED_MH_VALUES = (5, 10, 15, 20, 25, 30)
GRANULARITIES = ("global", "by_country", "by_competitor")
DETECTOR_FAMILY = "if"

FOREST_ROOT = _PROJECT_ROOT / "results" / "tuning" / "forests" / "single_config_optimized_mh5_run"
OUTPUT_DIR = _PROJECT_ROOT / "results" / "analysis" / "isolation_forest_granularity_comparison"
OUTPUT_BASENAME = "if_granularity_score_grid"

GRANULARITY_ORDER = {
    "global": 0,
    "by_country": 1,
    "by_competitor": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate Isolation Forest results across mh5,mh10,mh15,mh20,mh25,mh30 "
            "and render a thesis-facing granularity comparison figure."
        ),
    )
    parser.add_argument(
        "--forest-root",
        type=Path,
        default=FOREST_ROOT,
        help="Root directory containing the consolidated forest tuning outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for the aggregated CSV and plot outputs.",
    )
    return parser.parse_args()


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    return float(value)


def load_scope_frames(forest_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    status_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []

    for mh_value in SAMPLED_MH_VALUES:
        for granularity in GRANULARITIES:
            granularity_dir = forest_root / f"mh{mh_value}" / granularity
            if not granularity_dir.exists():
                raise FileNotFoundError(f"Missing granularity directory: {granularity_dir}")

            for scope_dir in sorted(path for path in granularity_dir.iterdir() if path.is_dir()):
                summary_path = scope_dir / DETECTOR_FAMILY / "summary.json"
                if not summary_path.exists():
                    continue

                payload = json.loads(summary_path.read_text(encoding="utf-8"))
                status = str(payload.get("status", "")).strip().lower()
                summary = payload.get("summary", {}) or {}
                best_candidate = payload.get("best_candidate") or {}
                scope_id = str(payload.get("scope_id", "")).strip()

                status_rows.append(
                    {
                        "mh_value": mh_value,
                        "mh_level": str(payload.get("mh_level", f"mh{mh_value}")).strip(),
                        "granularity": str(payload.get("granularity", granularity)).strip(),
                        "granularity_label": GRANULARITY_LABELS.get(granularity, granularity),
                        "scope_id": scope_id,
                        "dataset_name": str(payload.get("dataset_name", "")).strip(),
                        "detector_family": str(payload.get("detector_family", DETECTOR_FAMILY)).strip(),
                        "detector_label": str(payload.get("detector_label", "Isolation Forest")).strip(),
                        "status": status,
                        "error": str(payload.get("error", "")).strip(),
                        "summary_path": str(summary_path),
                    }
                )

                if status != "ok" or not isinstance(best_candidate, dict) or not best_candidate:
                    continue

                combined_f1 = safe_float(summary.get("best_f1"))
                if combined_f1 is None:
                    combined_f1 = safe_float(best_candidate.get("combined_f1"))
                if combined_f1 is None:
                    combined_f1 = safe_float(best_candidate.get("weighted_f1_mean"))

                combined_g_mean = safe_float(summary.get("best_g_mean"))
                if combined_g_mean is None:
                    combined_g_mean = safe_float(best_candidate.get("combined_g_mean"))
                if combined_g_mean is None:
                    combined_g_mean = safe_float(best_candidate.get("rank_score"))

                metric_rows.append(
                    {
                        "mh_value": mh_value,
                        "mh_level": str(payload.get("mh_level", f"mh{mh_value}")).strip(),
                        "granularity": str(payload.get("granularity", granularity)).strip(),
                        "granularity_label": GRANULARITY_LABELS.get(granularity, granularity),
                        "scope_id": scope_id,
                        "dataset_name": str(payload.get("dataset_name", "")).strip(),
                        "combined_f1": combined_f1,
                        "combined_g_mean": combined_g_mean,
                        "training_time_sec": safe_float(best_candidate.get("training_time_sec")),
                        "best_threshold": safe_float(summary.get("best_threshold"))
                        if safe_float(summary.get("best_threshold")) is not None
                        else safe_float(best_candidate.get("threshold")),
                        "best_config_key": summary.get("best_config_key") or best_candidate.get("config_key"),
                        "n_estimators": safe_float(best_candidate.get("n_estimators")),
                        "max_samples": safe_float(best_candidate.get("max_samples")),
                        "max_features": safe_float(best_candidate.get("max_features")),
                        "contamination": best_candidate.get("contamination"),
                        "summary_path": str(summary_path),
                    }
                )

    status_frame = pd.DataFrame(status_rows).sort_values(["granularity", "mh_value", "scope_id"])
    metric_frame = pd.DataFrame(metric_rows)
    if metric_frame.empty:
        raise FileNotFoundError(f"No successful Isolation Forest summary rows found under {forest_root}")

    metric_frame = metric_frame.sort_values(["granularity", "mh_value", "scope_id"]).reset_index(drop=True)
    return status_frame.reset_index(drop=True), metric_frame


def build_granularity_mh_summary(metric_frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        metric_frame.groupby(
            ["granularity", "granularity_label", "mh_value", "mh_level"],
            as_index=False,
        )
        .agg(
            scope_count=("scope_id", "nunique"),
            average_combined_f1=("combined_f1", "mean"),
            average_combined_g_mean=("combined_g_mean", "mean"),
            average_training_time_sec=("training_time_sec", "mean"),
        )
        .sort_values(
            ["granularity", "mh_value"],
            key=lambda series: (
                series.map(GRANULARITY_ORDER) if series.name == "granularity" else series
            ),
        )
        .reset_index(drop=True)
    )
    return summary


def build_granularity_mean_summary(granularity_mh_summary: pd.DataFrame) -> pd.DataFrame:
    summary = (
        granularity_mh_summary.groupby(["granularity", "granularity_label"], as_index=False)
        .agg(
            sampled_mh_count=("mh_value", "nunique"),
            mean_combined_f1=("average_combined_f1", "mean"),
            mean_combined_g_mean=("average_combined_g_mean", "mean"),
            mean_training_time_sec=("average_training_time_sec", "mean"),
            mean_scope_count=("scope_count", "mean"),
            min_scope_count=("scope_count", "min"),
            max_scope_count=("scope_count", "max"),
        )
        .sort_values(
            "granularity",
            key=lambda series: series.map(GRANULARITY_ORDER),
        )
        .reset_index(drop=True)
    )
    return summary


def build_granularity_best_mh_summary(granularity_mh_summary: pd.DataFrame) -> pd.DataFrame:
    f1_best = (
        granularity_mh_summary.sort_values(["granularity", "average_combined_f1", "mh_value"], ascending=[True, False, True])
        .groupby("granularity", as_index=False)
        .first()
        .rename(
            columns={
                "mh_level": "best_f1_mh_level",
                "mh_value": "best_f1_mh_value",
                "average_combined_f1": "best_combined_f1",
                "average_combined_g_mean": "g_mean_at_best_f1",
                "average_training_time_sec": "training_time_at_best_f1_sec",
                "scope_count": "scope_count_at_best_f1",
            }
        )
    )[
        [
            "granularity",
            "best_f1_mh_level",
            "best_f1_mh_value",
            "best_combined_f1",
            "g_mean_at_best_f1",
            "training_time_at_best_f1_sec",
            "scope_count_at_best_f1",
        ]
    ]
    g_best = (
        granularity_mh_summary.sort_values(["granularity", "average_combined_g_mean", "mh_value"], ascending=[True, False, True])
        .groupby("granularity", as_index=False)
        .first()
        .rename(
            columns={
                "mh_level": "best_g_mh_level",
                "mh_value": "best_g_mh_value",
                "average_combined_g_mean": "best_combined_g_mean",
                "average_combined_f1": "f1_at_best_g_mean",
                "average_training_time_sec": "training_time_at_best_g_mean_sec",
                "scope_count": "scope_count_at_best_g_mean",
            }
        )
    )[
        [
            "granularity",
            "best_g_mh_level",
            "best_g_mh_value",
            "best_combined_g_mean",
            "f1_at_best_g_mean",
            "training_time_at_best_g_mean_sec",
            "scope_count_at_best_g_mean",
        ]
    ]
    mean_summary = build_granularity_mean_summary(granularity_mh_summary)[
        ["granularity", "granularity_label", "mean_combined_f1", "mean_combined_g_mean", "mean_training_time_sec", "mean_scope_count", "min_scope_count", "max_scope_count"]
    ]
    merged = mean_summary.merge(f1_best, on="granularity").merge(g_best, on="granularity")
    return merged.sort_values(
        "granularity",
        key=lambda series: series.map(GRANULARITY_ORDER),
    ).reset_index(drop=True)


def build_summary_payload(status_frame: pd.DataFrame, best_mh_summary: pd.DataFrame) -> dict[str, object]:
    ok_rows = status_frame[status_frame["status"] == "ok"]
    error_rows = status_frame[status_frame["status"] != "ok"]
    payload: dict[str, object] = {
        "successful_scope_rows": int(len(ok_rows)),
        "error_scope_rows": int(len(error_rows)),
        "unique_scope_count": int(status_frame["scope_id"].nunique()),
        "granularity_count": int(status_frame["granularity"].nunique()),
        "mh_level_count": len(SAMPLED_MH_VALUES),
    }

    for row in best_mh_summary.itertuples(index=False):
        prefix = str(row.granularity)
        payload[f"{prefix}_mean_combined_f1"] = float(row.mean_combined_f1)
        payload[f"{prefix}_mean_combined_g_mean"] = float(row.mean_combined_g_mean)
        payload[f"{prefix}_best_f1_mh_level"] = str(row.best_f1_mh_level)
        payload[f"{prefix}_best_combined_f1"] = float(row.best_combined_f1)
        payload[f"{prefix}_best_g_mh_level"] = str(row.best_g_mh_level)
        payload[f"{prefix}_best_combined_g_mean"] = float(row.best_combined_g_mean)

    if not best_mh_summary.empty:
        best_f1_row = best_mh_summary.sort_values("best_combined_f1", ascending=False).iloc[0]
        best_g_row = best_mh_summary.sort_values("best_combined_g_mean", ascending=False).iloc[0]
        payload["highest_granularity_best_combined_f1"] = {
            "granularity": str(best_f1_row["granularity"]),
            "mh_level": str(best_f1_row["best_f1_mh_level"]),
            "score": float(best_f1_row["best_combined_f1"]),
        }
        payload["highest_granularity_best_combined_g_mean"] = {
            "granularity": str(best_g_row["granularity"]),
            "mh_level": str(best_g_row["best_g_mh_level"]),
            "score": float(best_g_row["best_combined_g_mean"]),
        }
    return payload


def write_summary_markdown(output_path: Path, *, forest_root: Path, best_mh_summary: pd.DataFrame) -> None:
    lines = [
        "# Isolation Forest Granularity Comparison",
        "",
        "This summary aggregates the retained Isolation Forest detector across the sampled mh grid and the three aggregation granularities.",
        "",
        "## Input",
        "",
        f"- Forest root: `{forest_root}`",
        "",
        "## Granularity Summary",
        "",
        "| Granularity | Mean combined F1 | Mean combined G-mean | Mean training time (s) | Best F1 mh | Best combined F1 | Best G-mean mh | Best combined G-mean | Mean scope count |",
        "| --- | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: |",
    ]
    for row in best_mh_summary.itertuples(index=False):
        lines.append(
            f"| {row.granularity_label} | "
            f"{row.mean_combined_f1:.4f} | "
            f"{row.mean_combined_g_mean:.4f} | "
            f"{row.mean_training_time_sec:.2f} | "
            f"{row.best_f1_mh_level} | "
            f"{row.best_combined_f1:.4f} | "
            f"{row.best_g_mh_level} | "
            f"{row.best_combined_g_mean:.4f} | "
            f"{row.mean_scope_count:.2f} |"
        )
    lines.append("")

    best_f1_row = best_mh_summary.sort_values("best_combined_f1", ascending=False).iloc[0]
    best_g_row = best_mh_summary.sort_values("best_combined_g_mean", ascending=False).iloc[0]
    lines.extend(
        [
            "## Key Findings",
            "",
            f"- The highest observed mean combined F1 was achieved under {best_f1_row.granularity_label.lower()} aggregation at {best_f1_row.best_f1_mh_level}, with a score of {best_f1_row.best_combined_f1:.4f}.",
            f"- The highest observed mean combined G-mean was achieved under {best_g_row.granularity_label.lower()} aggregation at {best_g_row.best_g_mh_level}, with a score of {best_g_row.best_combined_g_mean:.4f}.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_granularity_grid(granularity_mh_summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8.8, 7.2), constrained_layout=True)
    outer_grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 0.34],
        hspace=0.16,
    )
    score_axis = fig.add_subplot(outer_grid[0, :])
    metric_legend_axis = fig.add_subplot(outer_grid[1, 0])
    granularity_legend_axis = fig.add_subplot(outer_grid[1, 1])
    metric_legend_axis.axis("off")
    granularity_legend_axis.axis("off")

    for granularity in GRANULARITIES:
        granularity_frame = granularity_mh_summary[
            granularity_mh_summary["granularity"] == granularity
        ].sort_values("mh_value")
        if granularity_frame.empty:
            continue
        score_axis.plot(
            granularity_frame["mh_value"],
            granularity_frame["average_combined_f1"],
            **granularity_style(granularity, color=COMBINED_F1_COLOR),
        )
        score_axis.plot(
            granularity_frame["mh_value"],
            granularity_frame["average_combined_g_mean"],
            **granularity_style(granularity, color=COMBINED_GMEAN_COLOR),
        )

    score_axis.set_xticks(list(SAMPLED_MH_VALUES))
    score_axis.set_ylim(SCORE_YMIN, SCORE_YMAX)
    score_axis.set_yticks(SCORE_YTICKS)
    score_axis.grid(True, axis="y", color=GRID_Y_COLOR, linewidth=0.8)
    score_axis.grid(True, axis="x", color=GRID_X_COLOR, linewidth=0.6)
    score_axis.set_xlabel(X_AXIS_LABEL, fontsize=PLOT_LABEL_FONTSIZE, labelpad=8)
    score_axis.tick_params(axis="x", pad=3)
    score_axis.tick_params(axis="both", labelsize=PLOT_TICK_FONTSIZE)
    score_axis.set_ylabel("Weighted combined score", fontsize=PLOT_LABEL_FONTSIZE, labelpad=8)

    metric_legend_axis.legend(
        handles=metric_legend_handles(
            keys=WEIGHTED_SCORE_METRIC_KEYS,
            markers=None,
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        frameon=False,
        fontsize=PLOT_LEGEND_FONTSIZE,
        title_fontsize=PLOT_LEGEND_TITLE_FONTSIZE,
        ncol=1,
        title=METRIC_LEGEND_TITLE,
        handlelength=2.2,
        labelspacing=0.9,
        borderaxespad=0.0,
    )
    granularity_legend_axis.legend(
        handles=granularity_legend_handles(
            GRANULARITIES,
            include_linestyle=True,
            linewidth=2.2,
            markersize=6.0,
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        frameon=False,
        fontsize=PLOT_LEGEND_FONTSIZE,
        title_fontsize=PLOT_LEGEND_TITLE_FONTSIZE,
        ncol=1,
        title=GRANULARITY_LEGEND_TITLE,
        handlelength=2.2,
        labelspacing=0.9,
        borderaxespad=0.0,
    )

    save_thesis_media(fig, _PROJECT_ROOT, OUTPUT_BASENAME, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    forest_root = args.forest_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    status_frame, metric_frame = load_scope_frames(forest_root)
    granularity_mh_summary = build_granularity_mh_summary(metric_frame)
    granularity_mean_summary = build_granularity_mean_summary(granularity_mh_summary)
    best_mh_summary = build_granularity_best_mh_summary(granularity_mh_summary)
    summary_payload = build_summary_payload(status_frame, best_mh_summary)

    status_frame.to_csv(output_dir / "if_scope_status.csv", index=False)
    metric_frame.to_csv(output_dir / "if_scope_metrics.csv", index=False)
    granularity_mh_summary.to_csv(output_dir / "if_granularity_mh_summary.csv", index=False)
    granularity_mean_summary.to_csv(output_dir / "if_granularity_mean_summary.csv", index=False)
    best_mh_summary.to_csv(output_dir / "if_granularity_best_mh_summary.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    write_summary_markdown(output_dir / "summary.md", forest_root=forest_root, best_mh_summary=best_mh_summary)
    plot_granularity_grid(granularity_mh_summary, output_dir)
    print(f"Wrote Isolation Forest aggregate outputs to {output_dir}")


if __name__ == "__main__":
    main()
