#!/usr/bin/env python3
"""Plot optimal-threshold trajectories from best_configurations_all_cases.csv."""

from __future__ import annotations

import argparse
import logging
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from thesis_plot_style import (
    GRID_X_COLOR,
    GRID_Y_COLOR,
    GUIDE_COLOR,
    NEUTRAL_LEGEND_COLOR,
    SCORE_YMAX,
    SCORE_YMIN,
    SCORE_YTICKS,
    SCOPE_MARKERS,
    THRESHOLD_YMAX,
    THRESHOLD_YMIN,
    THRESHOLD_YTICKS,
    X_AXIS_LABEL,
    legend_handle,
    series_style,
)

LOGGER = logging.getLogger("plot_optimal_thresholds_by_mh")
MH_PATTERN = re.compile(r"mh(\d+)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create threshold and score plots over mh values.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="CSV produced by analyze_statistical_guidance.py, typically best_configurations_all_cases.csv.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="Base output path without extension. The script appends __by_country, __by_competitor, and __average.",
    )
    parser.add_argument(
        "--detector-family",
        default="standard_zscore",
        help="Detector family to plot.",
    )
    parser.add_argument(
        "--granularity",
        default="by_competitor",
        help="Granularity to plot.",
    )
    parser.add_argument(
        "--highlight-mh",
        default="mh5,mh10,mh15,mh20,mh25,mh30",
        help="Comma-separated mh values to highlight with vertical guides.",
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


def parse_mh_value(raw: str) -> int:
    match = MH_PATTERN.fullmatch(str(raw).strip())
    if not match:
        raise ValueError(f"Invalid mh value: {raw}")
    return int(match.group(1))


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def split_scope_id(scope_id: str) -> tuple[str, str, str]:
    parts = scope_id.split("/")
    country = parts[0] if len(parts) >= 1 else ""
    segment = parts[1] if len(parts) >= 2 else ""
    competitor = parts[2] if len(parts) >= 3 else scope_id
    return country, segment, competitor


def resolve_f1_column(df: pd.DataFrame) -> str:
    for candidate in ("combined_f1", "weighted_f1_mean"):
        if candidate in df.columns:
            return candidate
    raise ValueError("Input CSV is missing a combined F1 column.")


def load_plot_frame(input_csv: Path, detector_family: str, granularity: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    threshold_column = "config__threshold" if "config__threshold" in df.columns else "threshold"
    f1_column = resolve_f1_column(df)
    required = {"mh_level", "scope_id", "dataset_name", "detector_family", "granularity", threshold_column, f1_column}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    filtered = df[
        (df["detector_family"] == detector_family)
        & (df["granularity"] == granularity)
    ].copy()
    if filtered.empty:
        raise ValueError(
            f"No rows found for detector_family={detector_family!r}, granularity={granularity!r}"
        )

    filtered["mh_value"] = filtered["mh_level"].astype(str).map(parse_mh_value)
    filtered["threshold_value"] = pd.to_numeric(filtered[threshold_column], errors="coerce")
    filtered["combined_f1_value"] = pd.to_numeric(filtered[f1_column], errors="coerce")
    if {"new_prices_g_mean_mean", "new_products_g_mean_mean"}.issubset(filtered.columns):
        filtered["combined_g_mean_value"] = (
            0.7 * pd.to_numeric(filtered["new_prices_g_mean_mean"], errors="coerce")
            + 0.3 * pd.to_numeric(filtered["new_products_g_mean_mean"], errors="coerce")
        )
    elif "rank_score" in filtered.columns:
        filtered["combined_g_mean_value"] = pd.to_numeric(filtered["rank_score"], errors="coerce")
    else:
        raise ValueError("Input CSV is missing combined G-mean columns.")
    filtered[["country", "segment", "competitor_key"]] = filtered["scope_id"].astype(str).apply(
        lambda scope_id: pd.Series(split_scope_id(scope_id))
    )
    filtered["country_label"] = filtered["country"].astype(str)
    filtered["scope_label"] = filtered.apply(
        lambda row: f"{row['segment']} | {row['dataset_name']}",
        axis=1,
    )
    filtered["competitor_label"] = filtered.apply(
        lambda row: f"{row['country']} | {row['segment']} | {row['dataset_name']}",
        axis=1,
    )
    return filtered.sort_values(["country", "dataset_name", "mh_value"])


def save_figure(fig: plt.Figure, output_prefix: Path) -> None:
    png_path = output_prefix.with_suffix(".png")
    svg_path = output_prefix.with_suffix(".svg")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s", png_path)
    LOGGER.info("Wrote %s", svg_path)


def style_axis(
    ax: plt.Axes,
    *,
    mh_values: list[int],
    highlight_mh_values: list[int],
    xlabel: str,
    ylabel: str,
    y_limits: tuple[float, float],
    y_ticks: list[float],
    show_xlabel: bool = True,
) -> None:
    for mh_value in highlight_mh_values:
        ax.axvline(mh_value, color=GUIDE_COLOR, linestyle="--", linewidth=1.0, zorder=0)

    ax.set_xticks(mh_values)
    ax.set_xlim(min(mh_values) - 0.5, max(mh_values) + 0.5)
    ax.set_ylim(*y_limits)
    ax.set_yticks(y_ticks)
    ax.grid(True, axis="y", color=GRID_Y_COLOR, linewidth=0.9)
    ax.grid(True, axis="x", color=GRID_X_COLOR, linewidth=0.6, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(xlabel if show_xlabel else "", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)


def subplot_grid(count: int, max_cols: int) -> tuple[int, int]:
    cols = min(max_cols, count)
    rows = math.ceil(count / cols)
    return rows, cols


def _scope_marker(index: int) -> str:
    return SCOPE_MARKERS[index % len(SCOPE_MARKERS)]


def _series_style(key: str, *, marker: str) -> dict[str, object]:
    return series_style(key, marker=marker)


def _legend_handle(key: str, *, marker: str = "o") -> Line2D:
    return legend_handle(key, marker=marker)


def plot_by_country(
    frame: pd.DataFrame,
    *,
    output_prefix: Path,
    highlight_mh_values: list[int],
    mh_values: list[int],
) -> None:
    countries = sorted(frame["country_label"].dropna().unique().tolist())
    rows, cols = subplot_grid(len(countries), max_cols=2)
    fig = plt.figure(figsize=(18, 8.5 * rows), constrained_layout=True)
    outer_grid = fig.add_gridspec(rows, cols)
    threshold_handles = [
        _legend_handle("threshold"),
        _legend_handle("sampled_threshold"),
    ]
    metric_handles = [
        _legend_handle("combined_f1"),
        _legend_handle("sampled_combined_f1"),
        _legend_handle("combined_g_mean"),
        _legend_handle("sampled_combined_g_mean"),
    ]

    for index, country in enumerate(countries):
        inner_grid = outer_grid[index // cols, index % cols].subgridspec(2, 1, height_ratios=[3.0, 2.0], hspace=0.08)
        threshold_ax = fig.add_subplot(inner_grid[0])
        score_ax = fig.add_subplot(inner_grid[1], sharex=threshold_ax)
        country_frame = frame[frame["country_label"] == country].copy()
        scope_handles: list[Line2D] = []
        for color_index, (scope_label, scope_frame) in enumerate(country_frame.groupby("scope_label", sort=True)):
            scope_frame = scope_frame.sort_values("mh_value")
            sampled_scope_frame = scope_frame[scope_frame["mh_value"].isin(highlight_mh_values)].copy()
            marker = _scope_marker(color_index)
            threshold_ax.plot(
                scope_frame["mh_value"],
                scope_frame["threshold_value"],
                **_series_style("threshold", marker=marker),
            )
            threshold_ax.plot(
                sampled_scope_frame["mh_value"],
                sampled_scope_frame["threshold_value"],
                **_series_style("sampled_threshold", marker=marker),
            )
            score_ax.plot(
                scope_frame["mh_value"],
                scope_frame["combined_f1_value"],
                **_series_style("combined_f1", marker=marker),
            )
            score_ax.plot(
                sampled_scope_frame["mh_value"],
                sampled_scope_frame["combined_f1_value"],
                **_series_style("sampled_combined_f1", marker=marker),
            )
            score_ax.plot(
                scope_frame["mh_value"],
                scope_frame["combined_g_mean_value"],
                **_series_style("combined_g_mean", marker=marker),
            )
            score_ax.plot(
                sampled_scope_frame["mh_value"],
                sampled_scope_frame["combined_g_mean_value"],
                **_series_style("sampled_combined_g_mean", marker=marker),
            )
            scope_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=NEUTRAL_LEGEND_COLOR,
                    marker=marker,
                    markersize=5,
                    linewidth=0.0,
                    linestyle="None",
                    label=scope_label,
                )
            )
        style_axis(
            threshold_ax,
            mh_values=mh_values,
            highlight_mh_values=highlight_mh_values,
            xlabel=X_AXIS_LABEL,
            ylabel="Optimal threshold",
            y_limits=(THRESHOLD_YMIN, THRESHOLD_YMAX),
            y_ticks=THRESHOLD_YTICKS,
            show_xlabel=False,
        )
        style_axis(
            score_ax,
            mh_values=mh_values,
            highlight_mh_values=highlight_mh_values,
            xlabel=X_AXIS_LABEL,
            ylabel="Score",
            y_limits=(SCORE_YMIN, SCORE_YMAX),
            y_ticks=SCORE_YTICKS,
        )
        threshold_ax.tick_params(labelbottom=False)
        threshold_ax.set_title(country, fontsize=14, pad=10)
        threshold_legend = threshold_ax.legend(
            handles=threshold_handles,
            loc="upper left",
            frameon=False,
            fontsize=8.5,
        )
        threshold_ax.add_artist(threshold_legend)
        threshold_ax.legend(
            handles=scope_handles,
            loc="upper right",
            frameon=False,
            fontsize=8.0,
            title="Scopes",
            title_fontsize=8.5,
        )
        score_ax.legend(handles=metric_handles, loc="lower left", frameon=False, fontsize=8.5)

    for index in range(len(countries), rows * cols):
        empty_ax = fig.add_subplot(outer_grid[index // cols, index % cols])
        empty_ax.axis("off")

    fig.suptitle("Optimal Threshold and Scores by mh: One Subplot per Country", fontsize=18)
    save_figure(fig, output_prefix.with_name(f"{output_prefix.name}__by_country"))


def plot_by_competitor(
    frame: pd.DataFrame,
    *,
    output_prefix: Path,
    highlight_mh_values: list[int],
    mh_values: list[int],
) -> None:
    competitor_rows = (
        frame[["scope_id", "competitor_label"]]
        .drop_duplicates()
        .sort_values(["competitor_label", "scope_id"])
        .to_dict("records")
    )
    rows, cols = subplot_grid(len(competitor_rows), max_cols=3)
    fig = plt.figure(figsize=(18, 6.7 * rows), constrained_layout=True)
    outer_grid = fig.add_gridspec(rows, cols)

    for index, item in enumerate(competitor_rows):
        inner_grid = outer_grid[index // cols, index % cols].subgridspec(2, 1, height_ratios=[3.0, 2.0], hspace=0.08)
        threshold_ax = fig.add_subplot(inner_grid[0])
        score_ax = fig.add_subplot(inner_grid[1], sharex=threshold_ax)
        scope_frame = frame[frame["scope_id"] == item["scope_id"]].sort_values("mh_value")
        sampled_scope_frame = scope_frame[scope_frame["mh_value"].isin(highlight_mh_values)].copy()
        marker = _scope_marker(index)
        threshold_ax.plot(
            scope_frame["mh_value"],
            scope_frame["threshold_value"],
            **_series_style("threshold", marker=marker),
        )
        threshold_ax.plot(
            sampled_scope_frame["mh_value"],
            sampled_scope_frame["threshold_value"],
            **_series_style("sampled_threshold", marker=marker),
        )
        score_ax.plot(
            scope_frame["mh_value"],
            scope_frame["combined_f1_value"],
            **_series_style("combined_f1", marker=marker),
        )
        score_ax.plot(
            sampled_scope_frame["mh_value"],
            sampled_scope_frame["combined_f1_value"],
            **_series_style("sampled_combined_f1", marker=marker),
        )
        score_ax.plot(
            scope_frame["mh_value"],
            scope_frame["combined_g_mean_value"],
            **_series_style("combined_g_mean", marker=marker),
        )
        score_ax.plot(
            sampled_scope_frame["mh_value"],
            sampled_scope_frame["combined_g_mean_value"],
            **_series_style("sampled_combined_g_mean", marker=marker),
        )
        style_axis(
            threshold_ax,
            mh_values=mh_values,
            highlight_mh_values=highlight_mh_values,
            xlabel=X_AXIS_LABEL,
            ylabel="Optimal threshold",
            y_limits=(THRESHOLD_YMIN, THRESHOLD_YMAX),
            y_ticks=THRESHOLD_YTICKS,
            show_xlabel=False,
        )
        style_axis(
            score_ax,
            mh_values=mh_values,
            highlight_mh_values=highlight_mh_values,
            xlabel=X_AXIS_LABEL,
            ylabel="Score",
            y_limits=(SCORE_YMIN, SCORE_YMAX),
            y_ticks=SCORE_YTICKS,
        )
        threshold_ax.tick_params(labelbottom=False)
        threshold_ax.set_title(item["competitor_label"], fontsize=11, pad=8)
        threshold_ax.legend(loc="upper left", frameon=False, fontsize=8)
        score_ax.legend(loc="lower left", frameon=False, fontsize=8)

    for index in range(len(competitor_rows), rows * cols):
        empty_ax = fig.add_subplot(outer_grid[index // cols, index % cols])
        empty_ax.axis("off")

    fig.suptitle("Optimal Threshold and Scores by mh: One Subplot per Scope", fontsize=18)
    save_figure(fig, output_prefix.with_name(f"{output_prefix.name}__by_competitor"))


def plot_average_threshold(
    frame: pd.DataFrame,
    *,
    output_prefix: Path,
    highlight_mh_values: list[int],
    mh_values: list[int],
) -> None:
    average_frame = (
        frame.groupby("mh_value", dropna=False)["threshold_value"]
        .mean()
        .reset_index(name="average_threshold")
        .sort_values("mh_value")
    )
    sampled_average_frame = average_frame[average_frame["mh_value"].isin(highlight_mh_values)].copy()

    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
    ax.plot(
        average_frame["mh_value"],
        average_frame["average_threshold"],
        **_series_style("threshold", marker="o"),
    )
    ax.plot(
        sampled_average_frame["mh_value"],
        sampled_average_frame["average_threshold"],
        **_series_style("sampled_threshold", marker="o"),
    )
    style_axis(
        ax,
        mh_values=mh_values,
        highlight_mh_values=highlight_mh_values,
        xlabel=X_AXIS_LABEL,
        ylabel="Average optimal threshold",
        y_limits=(THRESHOLD_YMIN, THRESHOLD_YMAX),
        y_ticks=THRESHOLD_YTICKS,
    )
    ax.set_title("Average Optimal Threshold by mh", fontsize=16, pad=12)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    save_figure(fig, output_prefix.with_name(f"{output_prefix.name}__average"))


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    input_csv = args.input_csv.resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_csv}")

    output_prefix = args.output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    frame = load_plot_frame(
        input_csv=input_csv,
        detector_family=args.detector_family,
        granularity=args.granularity,
    )
    highlight_mh_values = sorted({parse_mh_value(item) for item in parse_csv_list(args.highlight_mh)})
    mh_values = sorted(frame["mh_value"].dropna().unique().tolist())

    plot_by_country(
        frame,
        output_prefix=output_prefix,
        highlight_mh_values=highlight_mh_values,
        mh_values=mh_values,
    )
    plot_by_competitor(
        frame,
        output_prefix=output_prefix,
        highlight_mh_values=highlight_mh_values,
        mh_values=mh_values,
    )
    plot_average_threshold(
        frame,
        output_prefix=output_prefix,
        highlight_mh_values=highlight_mh_values,
        mh_values=mh_values,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
