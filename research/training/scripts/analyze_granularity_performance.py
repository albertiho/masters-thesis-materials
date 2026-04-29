#!/usr/bin/env python3
"""Compare sampled-mh z-score performance across tuning granularities."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from thesis_plot_style import (
    COMBINED_F1_COLOR,
    COMBINED_GMEAN_COLOR,
    GRID_X_COLOR,
    GRID_Y_COLOR,
    GRANULARITY_LABELS,
    GUIDE_COLOR,
    SCORE_YMAX,
    SCORE_YMIN,
    SCORE_YTICKS,
    X_AXIS_LABEL,
    granularity_style,
)

LOGGER = logging.getLogger("analyze_granularity_performance")
MH_PATTERN = re.compile(r"mh(\d+)$", re.IGNORECASE)
DEFAULT_SAMPLED_MH = ("mh5", "mh10", "mh15", "mh20", "mh25", "mh30")
DEFAULT_OUTPUT_DIR = Path("results/tuning/statistical/z_score_granularity_comparison/analysis")
DEFAULT_SOURCE_CSVS = {
    "by_competitor": Path(
        "results/tuning/statistical/z_score_by_competitor_single_attempt_batch/analysis/best_configurations_all_cases.csv"
    ),
    "by_country": Path(
        "results/tuning/statistical/z_score_by_country_single_attempt_batch/analysis/best_configurations_all_cases.csv"
    ),
    "global": Path(
        "results/tuning/statistical/z_score_global_single_attempt_batch/analysis/best_configurations_all_cases.csv"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate sampled-mh z-score performance across global, by-country, and by-competitor sweeps.",
    )
    parser.add_argument(
        "--by-competitor-csv",
        type=Path,
        default=DEFAULT_SOURCE_CSVS["by_competitor"],
        help="best_configurations_all_cases.csv for the by_competitor z-score sweep.",
    )
    parser.add_argument(
        "--by-country-csv",
        type=Path,
        default=DEFAULT_SOURCE_CSVS["by_country"],
        help="best_configurations_all_cases.csv for the by_country z-score sweep.",
    )
    parser.add_argument(
        "--global-csv",
        type=Path,
        default=DEFAULT_SOURCE_CSVS["global"],
        help="best_configurations_all_cases.csv for the global z-score sweep.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the generated summary table and plots.",
    )
    parser.add_argument(
        "--detector-family",
        default="standard_zscore",
        help="Detector family to analyze.",
    )
    parser.add_argument(
        "--sampled-mh",
        default=",".join(DEFAULT_SAMPLED_MH),
        help="Comma-separated sampled mh values to include.",
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


def parse_mh_value(raw: str) -> int:
    match = MH_PATTERN.fullmatch(str(raw).strip())
    if not match:
        raise ValueError(f"Invalid mh value: {raw}")
    return int(match.group(1))


def resolve_combined_f1_column(frame: pd.DataFrame) -> str:
    for candidate in ("combined_f1", "weighted_f1_mean"):
        if candidate in frame.columns:
            return candidate
    raise ValueError("Input CSV is missing a combined F1 column.")


def load_best_configuration_frame(
    input_csv: Path,
    *,
    expected_granularity: str,
    detector_family: str,
    sampled_mh_levels: set[str],
) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_csv}")

    frame = pd.read_csv(input_csv)
    f1_column = resolve_combined_f1_column(frame)
    required = {
        "mh_level",
        "granularity",
        "scope_id",
        "dataset_name",
        "detector_family",
        f1_column,
        "new_prices_g_mean_mean",
        "new_products_g_mean_mean",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{input_csv} is missing required columns: {sorted(missing)}")

    filtered = frame[
        (frame["granularity"] == expected_granularity)
        & (frame["detector_family"] == detector_family)
        & (frame["mh_level"].astype(str).isin(sampled_mh_levels))
    ].copy()
    if "status" in filtered.columns:
        filtered = filtered[filtered["status"].fillna("").astype(str).str.lower().isin({"ok", "success", ""})]
    if filtered.empty:
        raise ValueError(
            f"No rows found in {input_csv} for granularity={expected_granularity!r}, "
            f"detector_family={detector_family!r}, sampled mh={sorted(sampled_mh_levels)}"
        )

    filtered["mh_value"] = filtered["mh_level"].astype(str).map(parse_mh_value)
    filtered["weighted_combined_f1"] = pd.to_numeric(filtered[f1_column], errors="coerce")
    filtered["weighted_combined_g_mean"] = (
        0.7 * pd.to_numeric(filtered["new_prices_g_mean_mean"], errors="coerce")
        + 0.3 * pd.to_numeric(filtered["new_products_g_mean_mean"], errors="coerce")
    )
    filtered["granularity_label"] = GRANULARITY_LABELS.get(expected_granularity, expected_granularity)
    filtered["source_csv"] = str(input_csv.resolve())
    filtered = filtered.dropna(subset=["weighted_combined_f1", "weighted_combined_g_mean"])
    return filtered.sort_values(["mh_value", "scope_id"])


def build_aggregate_table(frames: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True)
    aggregate = (
        combined.groupby(["granularity", "granularity_label", "mh_level", "mh_value"], dropna=False)
        .agg(
            average_weighted_combined_f1=("weighted_combined_f1", "mean"),
            average_weighted_combined_g_mean=("weighted_combined_g_mean", "mean"),
            scope_count=("scope_id", "nunique"),
        )
        .reset_index()
        .sort_values(["mh_value", "granularity"])
    )
    return aggregate


def write_summary_markdown(aggregate: pd.DataFrame, output_path: Path) -> None:
    summary = aggregate.copy()
    summary["average_weighted_combined_f1"] = summary["average_weighted_combined_f1"].map(lambda value: f"{value:.4f}")
    summary["average_weighted_combined_g_mean"] = summary["average_weighted_combined_g_mean"].map(
        lambda value: f"{value:.4f}"
    )
    summary["scope_count"] = summary["scope_count"].astype(int)

    lines = [
        "# Granularity Comparison",
        "",
        "| Granularity | mh | Avg weighted combined F1 | Avg weighted combined G-mean | Scope count |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.granularity_label} | {row.mh_level} | {row.average_weighted_combined_f1} | "
            f"{row.average_weighted_combined_g_mean} | {row.scope_count} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def style_axis(ax: plt.Axes, *, mh_values: list[int], ylabel: str, show_xlabel: bool) -> None:
    for mh_value in mh_values:
        ax.axvline(mh_value, color=GUIDE_COLOR, linestyle="--", linewidth=1.0, zorder=0)
    ax.set_xticks(mh_values)
    ax.set_xlim(min(mh_values) - 0.5, max(mh_values) + 0.5)
    ax.set_ylim(SCORE_YMIN, SCORE_YMAX)
    ax.set_yticks(SCORE_YTICKS)
    ax.grid(True, axis="y", color=GRID_Y_COLOR, linewidth=0.9)
    ax.grid(True, axis="x", color=GRID_X_COLOR, linewidth=0.6, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel(X_AXIS_LABEL if show_xlabel else "", fontsize=11)


def plot_granularity_comparison(aggregate: pd.DataFrame, output_prefix: Path) -> None:
    mh_values = sorted(aggregate["mh_value"].dropna().unique().tolist())
    fig, ax = plt.subplots(figsize=(12, 7.5), constrained_layout=True)

    for granularity in ("global", "by_country", "by_competitor"):
        granularity_frame = aggregate[aggregate["granularity"] == granularity].sort_values("mh_value")
        if granularity_frame.empty:
            continue
        ax.plot(
            granularity_frame["mh_value"],
            granularity_frame["average_weighted_combined_f1"],
            **granularity_style(granularity, color=COMBINED_F1_COLOR),
        )
        ax.plot(
            granularity_frame["mh_value"],
            granularity_frame["average_weighted_combined_g_mean"],
            **granularity_style(granularity, color=COMBINED_GMEAN_COLOR),
        )

    style_axis(
        ax,
        mh_values=mh_values,
        ylabel="Average weighted combined score",
        show_xlabel=True,
    )
    ax.set_title("Average Weighted Combined F1 and G-mean by Sampled Minimum History", fontsize=16, pad=12)
    metric_handles = [
        Line2D([0], [0], color=COMBINED_F1_COLOR, linewidth=2.2, label="Average weighted combined F1"),
        Line2D([0], [0], color=COMBINED_GMEAN_COLOR, linewidth=2.2, label="Average weighted combined G-mean"),
    ]
    granularity_handles = []
    for granularity in ("global", "by_country", "by_competitor"):
        style = granularity_style(granularity, color="#4d4d4d")
        granularity_handles.append(
            Line2D(
                [0],
                [0],
                color="#4d4d4d",
                linewidth=2.2,
                marker=style["marker"],
                linestyle=style["linestyle"],
                markersize=5.5,
                label=GRANULARITY_LABELS.get(granularity, granularity),
            )
        )
    metric_legend = ax.legend(
        handles=metric_handles,
        loc="lower left",
        frameon=False,
        fontsize=10,
        title="Metric",
    )
    ax.add_artist(metric_legend)
    ax.legend(
        handles=granularity_handles,
        loc="lower right",
        frameon=False,
        fontsize=10,
        title="Granularity",
    )
    fig.suptitle("Z-score Performance Across Granularities", fontsize=18)

    png_path = output_prefix.with_suffix(".png")
    svg_path = output_prefix.with_suffix(".svg")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s", png_path)
    LOGGER.info("Wrote %s", svg_path)


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    sampled_mh_levels = set(parse_csv_list(args.sampled_mh))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    input_frames = [
        load_best_configuration_frame(
            args.global_csv.resolve(),
            expected_granularity="global",
            detector_family=args.detector_family,
            sampled_mh_levels=sampled_mh_levels,
        ),
        load_best_configuration_frame(
            args.by_country_csv.resolve(),
            expected_granularity="by_country",
            detector_family=args.detector_family,
            sampled_mh_levels=sampled_mh_levels,
        ),
        load_best_configuration_frame(
            args.by_competitor_csv.resolve(),
            expected_granularity="by_competitor",
            detector_family=args.detector_family,
            sampled_mh_levels=sampled_mh_levels,
        ),
    ]

    aggregate = build_aggregate_table(input_frames)
    aggregate_csv_path = output_dir / "granularity_performance_sampled_mh.csv"
    aggregate.to_csv(aggregate_csv_path, index=False)
    LOGGER.info("Wrote %s", aggregate_csv_path)

    summary_md_path = output_dir / "granularity_performance_sampled_mh.md"
    write_summary_markdown(aggregate, summary_md_path)
    LOGGER.info("Wrote %s", summary_md_path)

    plot_granularity_comparison(
        aggregate,
        output_dir / "granularity_performance_sampled_mh",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
