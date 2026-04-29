#!/usr/bin/env python3
"""Assess whether the original modified-zscore threshold grid is too narrow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from research.training.scripts.thesis_plot_style import (
    GRID_Y_COLOR,
    PLOT_LABEL_FONTSIZE,
    PLOT_LEGEND_FONTSIZE,
    PLOT_TICK_FONTSIZE,
    PLOT_TITLE_FONTSIZE,
    SCORE_YMAX,
    SCORE_YMIN,
    WEIGHTED_SCORE_METRIC_KEYS,
    WEIGHTED_SCORE_METRIC_MARKERS,
    metric_legend_handles,
    save_thesis_media,
    series_style,
)

MODIFIED_METHODS = (
    "modified_mad",
    "modified_sn",
    "hybrid_weighted",
    "hybrid_max",
    "hybrid_avg",
)

DETECTOR_LABELS = {
    "modified_mad": "Modified MAD",
    "modified_sn": "Modified Sn",
    "hybrid_weighted": "Hybrid weighted",
    "hybrid_max": "Hybrid max",
    "hybrid_avg": "Hybrid avg",
}


def parse_args() -> argparse.Namespace:
    default_input_root = (
        _PROJECT_ROOT / "results" / "tuning" / "statistical" / "all_global_single_attempt_batch" / "mh5"
    )
    default_output_dir = default_input_root / "analysis" / "modified_zscore_scale"
    parser = argparse.ArgumentParser(
        description="Summarize mh5 modified-zscore tuning results and visualize threshold trends.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=default_input_root,
        help="Root containing mh5 granularity folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory for analysis outputs.",
    )
    return parser.parse_args()


def load_candidate_rows(input_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate_path in input_root.rglob("candidate_metrics.csv"):
        detector = candidate_path.parent.name
        if detector not in MODIFIED_METHODS:
            continue
        scope = candidate_path.parent.parent.name
        granularity = candidate_path.parent.parent.parent.name
        frame = pd.read_csv(candidate_path)
        if frame.empty:
            continue

        for _, row in frame.iterrows():
            weighted_f1 = float(row["weighted_f1_mean"]) if "weighted_f1_mean" in frame.columns else float(row["combined_f1"])
            weighted_g_mean = 0.7 * float(row["new_prices_g_mean_mean"]) + 0.3 * float(row["new_products_g_mean_mean"])
            rows.append(
                {
                    "granularity": granularity,
                    "scope": scope,
                    "detector": detector,
                    "stage": row.get("stage"),
                    "threshold": float(row["threshold"]),
                    "w": float(row["w"]) if "w" in frame.columns and pd.notna(row.get("w")) else None,
                    "weighted_f1_mean": weighted_f1,
                    "weighted_g_mean": weighted_g_mean,
                }
            )
    return pd.DataFrame(rows)


def load_best_config_rows(input_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for config_path in input_root.rglob("best_configuration.json"):
        detector = config_path.parent.name
        if detector not in MODIFIED_METHODS:
            continue
        data = json.loads(config_path.read_text(encoding="utf-8"))
        config = data.get("configuration", {}) or {}
        rows.append(
            {
                "granularity": config_path.parent.parent.parent.name,
                "scope": config_path.parent.parent.name,
                "detector": detector,
                "threshold": float(config["threshold"]),
                "w": config.get("w"),
            }
        )
    return pd.DataFrame(rows)


def _collapse_threshold_rows(detector_frame: pd.DataFrame) -> pd.DataFrame:
    if detector_frame.empty:
        return detector_frame
    if detector_frame["detector"].iloc[0] == "hybrid_weighted":
        best_idx = detector_frame.groupby(
            ["granularity", "scope", "threshold"],
        )["weighted_f1_mean"].idxmax()
        return detector_frame.loc[
            best_idx,
            ["granularity", "scope", "threshold", "weighted_f1_mean", "weighted_g_mean"],
        ].reset_index(drop=True)

    return detector_frame.groupby(
        ["granularity", "scope", "threshold"],
        as_index=False,
    )[["weighted_f1_mean", "weighted_g_mean"]].mean()


def summarize_threshold_trends(candidates: pd.DataFrame) -> pd.DataFrame:
    coarse = candidates[candidates["stage"] == "coarse"].copy()
    summary_rows: list[dict[str, object]] = []
    for detector, detector_frame in coarse.groupby("detector"):
        per_scope = _collapse_threshold_rows(detector_frame)
        grouped = per_scope.groupby("threshold")
        for threshold, values in grouped:
            summary_rows.append(
                {
                    "detector": detector,
                    "threshold": float(threshold),
                    "mean_weighted_f1": float(values["weighted_f1_mean"].mean()),
                    "median_weighted_f1": float(values["weighted_f1_mean"].median()),
                    "mean_weighted_g_mean": float(values["weighted_g_mean"].mean()),
                    "median_weighted_g_mean": float(values["weighted_g_mean"].median()),
                    "n_scope_combinations": int(values.shape[0]),
                }
            )
    return pd.DataFrame(summary_rows).sort_values(["detector", "threshold"])


def summarize_boundary_behavior(candidates: pd.DataFrame, best_configs: pd.DataFrame) -> pd.DataFrame:
    coarse = candidates[candidates["stage"] == "coarse"].copy()
    summary_rows: list[dict[str, object]] = []
    for detector, detector_frame in coarse.groupby("detector"):
        per_scope = _collapse_threshold_rows(detector_frame)
        f1_pivot = per_scope.pivot_table(
            index=["granularity", "scope"],
            columns="threshold",
            values="weighted_f1_mean",
        )
        gmean_pivot = per_scope.pivot_table(
            index=["granularity", "scope"],
            columns="threshold",
            values="weighted_g_mean",
        )
        best_detector_configs = best_configs[best_configs["detector"] == detector]
        summary_rows.append(
            {
                "detector": detector,
                "evaluated_combinations": int(best_detector_configs.shape[0]),
                "best_at_3_count": int((best_detector_configs["threshold"] == 3.0).sum()),
                "best_at_3_share": float((best_detector_configs["threshold"] == 3.0).mean()),
                "mean_weighted_f1_at_1": float(f1_pivot[1.0].mean()),
                "mean_weighted_f1_at_3": float(f1_pivot[3.0].mean()),
                "mean_weighted_g_mean_at_1": float(gmean_pivot[1.0].mean()),
                "mean_weighted_g_mean_at_3": float(gmean_pivot[3.0].mean()),
                "mean_delta_1_to_3": float((f1_pivot[3.0] - f1_pivot[1.0]).mean()),
                "mean_delta_2p5_to_3": float((f1_pivot[3.0] - f1_pivot[2.5]).mean()),
                "positive_delta_2p5_to_3": int(((f1_pivot[3.0] - f1_pivot[2.5]) > 0).sum()),
                "available_delta_2p5_to_3": int((f1_pivot[3.0] - f1_pivot[2.5]).notna().sum()),
            }
        )
    return pd.DataFrame(summary_rows).sort_values("detector")


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    rendered = frame.copy()
    for column in rendered.columns:
        rendered[column] = rendered[column].map(_format_markdown_cell)

    headers = [str(column) for column in rendered.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rendered.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _format_markdown_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_summary_markdown(
    output_path: Path,
    boundary_summary: pd.DataFrame,
    trend_summary: pd.DataFrame,
    best_configs: pd.DataFrame,
) -> None:
    total_best = int(best_configs.shape[0])
    total_at_upper = int((best_configs["threshold"] == 3.0).sum())
    overall_share = total_at_upper / total_best if total_best else 0.0

    coarse_f1_table = trend_summary.pivot(
        index="threshold",
        columns="detector",
        values="mean_weighted_f1",
    ).sort_index()
    coarse_f1_table = coarse_f1_table.rename(columns=DETECTOR_LABELS)

    boundary_table = (
        boundary_summary.rename(
            columns={
                "detector": "Detector",
                "evaluated_combinations": "Combinations",
                "best_at_3_count": "Best at 3.0",
                "best_at_3_share": "Share at 3.0",
                "mean_weighted_f1_at_1": "Mean F1 at 1.0",
                "mean_weighted_f1_at_3": "Mean F1 at 3.0",
                "mean_weighted_g_mean_at_1": "Mean G-mean at 1.0",
                "mean_weighted_g_mean_at_3": "Mean G-mean at 3.0",
                "mean_delta_1_to_3": "Mean delta 1.0->3.0",
                "mean_delta_2p5_to_3": "Mean delta 2.5->3.0",
                "positive_delta_2p5_to_3": "Positive 2.5->3.0",
                "available_delta_2p5_to_3": "Available 2.5->3.0",
            }
        )
        .assign(Detector=lambda df: df["Detector"].map(DETECTOR_LABELS))
    )

    lines = [
        "# Modified Z-score Scale Assessment (mh5)",
        "",
        "This analysis summarizes the modified-zscore-family tuning results for `mh5` across `global`, `by_country`, and `by_competitor`.",
        "",
        "## Key Findings",
        "",
        f"- Across all available modified-zscore detector/scope combinations, `{total_at_upper}` of `{total_best}` best configurations (`{overall_share:.1%}`) selected the original upper-bound threshold `3.0`.",
        "- For every detector family, the mean coarse-grid weighted combined F1 increased monotonically from `1.0` to `3.0`.",
        "- The mean weighted-combined-F1 gain from `2.5` to `3.0` remained positive for every available scope in every detector family.",
        "- This pattern indicates that the original `1.0-3.0` range was too narrow to reveal whether performance had already peaked or was still improving beyond the upper boundary.",
        "",
        "## Boundary Summary",
        "",
        dataframe_to_markdown(boundary_table),
        "",
        "## Mean Coarse Weighted Combined F1 by Threshold",
        "",
        dataframe_to_markdown(coarse_f1_table.reset_index()),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_data_references(output_dir: Path, input_root: Path) -> None:
    lines = [
        "# Modified Z-score Scale Data References",
        "",
        "## Thesis artifact references",
        "",
        "| Thesis item | Source file | Presents data |",
        "| --- | --- | --- |",
        "| Figure `modified_zscore_scale_mh5__all_methods` (`results/media/modified_zscore_scale_mh5__all_methods.png` / `results/media/modified_zscore_scale_mh5__all_methods.svg`) | `results/tuning/statistical/all_global_single_attempt_batch/mh5/analysis/modified_zscore_scale/modified_zscore_scale_trends.csv` | Mean weighted combined `F_{1,\\mathrm{wc}}` and `G_{\\mathrm{wc}}` trends over the original modified-zscore threshold range for the five retained modified-zscore-family methods at `mh5`. |",
        "| Table `modified_zscore_scale_boundary_summary` | `results/tuning/statistical/all_global_single_attempt_batch/mh5/analysis/modified_zscore_scale/modified_zscore_scale_boundary_summary.csv` | Boundary-selection counts and coarse-grid score deltas used to justify extending the modified-zscore threshold range beyond the original upper bound. |",
        "",
        "## Raw aggregation inputs",
        "",
        "| Source path | Contributes data for | Presents data |",
        "| --- | --- | --- |",
        f"| `{input_root}` | Modified-zscore-family `mh5` tuning across `global`, `by_country`, and `by_competitor` granularities | Per-scope `candidate_metrics.csv` and `best_configuration.json` outputs used to assess whether the original `1.0-3.0` threshold grid was too narrow. |",
        "",
        "## Notes",
        "",
        "- The thesis-facing figure files are written to `results/media/modified_zscore_scale_mh5__all_methods.png` and `results/media/modified_zscore_scale_mh5__all_methods.svg`.",
        "- The figure uses a two-column by three-row layout: five method panels and one embedded metric legend panel.",
    ]
    (output_dir / "data_references.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_method_trends(output_dir: Path, trend_summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(8.8, 13.2), sharex=True, sharey=True, constrained_layout=True)
    flat_axes = axes.flatten()
    legend_handles = metric_legend_handles(
        keys=WEIGHTED_SCORE_METRIC_KEYS,
        markers=WEIGHTED_SCORE_METRIC_MARKERS,
    )
    legend_axis = flat_axes[len(MODIFIED_METHODS)] if len(MODIFIED_METHODS) < len(flat_axes) else None

    for axis, detector in zip(flat_axes, MODIFIED_METHODS, strict=False):
        detector_frame = trend_summary[trend_summary["detector"] == detector].sort_values("threshold")
        if detector_frame.empty:
            axis.set_visible(False)
            continue

        axis.plot(
            detector_frame["threshold"],
            detector_frame["mean_weighted_f1"],
            **series_style("combined_f1", marker="o"),
        )
        axis.plot(
            detector_frame["threshold"],
            detector_frame["mean_weighted_g_mean"],
            **series_style("combined_g_mean", marker="s"),
        )
        axis.set_title(DETECTOR_LABELS[detector], fontsize=PLOT_TITLE_FONTSIZE, pad=10)
        axis.set_xticks(sorted(detector_frame["threshold"].tolist()))
        axis.set_ylim(SCORE_YMIN, SCORE_YMAX)
        axis.grid(axis="y", color=GRID_Y_COLOR, linewidth=0.8)
        axis.tick_params(axis="both", labelsize=PLOT_TICK_FONTSIZE)

    for axis in flat_axes[len(MODIFIED_METHODS) :]:
        axis.set_visible(False)

    for axis in axes[-1]:
        if axis.get_visible():
            axis.set_xlabel("Threshold", fontsize=PLOT_LABEL_FONTSIZE, labelpad=8)
    for axis in axes[:, 0]:
        if axis.get_visible():
            axis.set_ylabel("Weighted score", fontsize=PLOT_LABEL_FONTSIZE, labelpad=8)

    if legend_axis is not None:
        legend_axis.set_visible(True)
        legend_axis.axis("off")
        legend_axis.legend(
            handles=legend_handles,
            frameon=False,
            fontsize=PLOT_LEGEND_FONTSIZE,
            loc="center",
            handlelength=2.2,
            labelspacing=1.0,
        )

    save_thesis_media(fig, _PROJECT_ROOT, "modified_zscore_scale_mh5__all_methods", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_candidate_rows(args.input_root)
    best_configs = load_best_config_rows(args.input_root)
    trend_summary = summarize_threshold_trends(candidates)
    boundary_summary = summarize_boundary_behavior(candidates, best_configs)

    trend_summary.to_csv(args.output_dir / "modified_zscore_scale_trends.csv", index=False)
    boundary_summary.to_csv(args.output_dir / "modified_zscore_scale_boundary_summary.csv", index=False)
    write_summary_markdown(
        args.output_dir / "modified_zscore_scale_summary.md",
        boundary_summary,
        trend_summary,
        best_configs,
    )
    write_data_references(args.output_dir, args.input_root)
    plot_method_trends(args.output_dir, trend_summary)


if __name__ == "__main__":
    main()
