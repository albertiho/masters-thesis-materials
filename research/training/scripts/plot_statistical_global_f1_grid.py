#!/usr/bin/env python3
"""Aggregate sampled statistical scores and render a detector grid with an embedded legend row."""

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
    GRANULARITY_LEGEND_TITLE,
    GRID_X_COLOR,
    GRID_Y_COLOR,
    GRANULARITY_LABELS,
    GRANULARITY_MARKERS,
    METRIC_LEGEND_TITLE,
    PLOT_LABEL_FONTSIZE,
    PLOT_LEGEND_FONTSIZE,
    PLOT_LEGEND_TITLE_FONTSIZE,
    PLOT_TICK_FONTSIZE,
    PLOT_TITLE_FONTSIZE,
    SCORE_YMAX,
    SCORE_YMIN,
    SCORE_YTICKS,
    WEIGHTED_SCORE_METRIC_KEYS,
    WEIGHTED_SCORE_METRIC_MARKERS,
    X_AXIS_LABEL,
    granularity_legend_handles,
    metric_legend_handles,
    save_thesis_media,
    series_style,
)

SAMPLED_MH_VALUES = (5, 10, 15, 20, 25, 30)
GRANULARITIES = ("global", "by_country", "by_competitor")

COARSE_ROOT = _PROJECT_ROOT / "results" / "tuning" / "statistical" / "1-6coarse_grid_single_attempt_batch"
HYBRID_WEIGHTED_ROOT = (
    _PROJECT_ROOT / "results" / "tuning" / "statistical" / "no_avg_hybrid_weighted_single_attempt_batch"
)
ZSCORE_BY_COMPETITOR_ROOT = (
    _PROJECT_ROOT / "results" / "tuning" / "statistical" / "z_score_by_competitor_single_attempt_batch"
)
ZSCORE_BY_COUNTRY_ROOT = (
    _PROJECT_ROOT / "results" / "tuning" / "statistical" / "z_score_by_country_single_attempt_batch"
)
ZSCORE_GLOBAL_ROOT = _PROJECT_ROOT / "results" / "tuning" / "statistical" / "z_score_global_single_attempt_batch"

OUTPUT_DIR = _PROJECT_ROOT / "results" / "analysis" / "statistical_detector_score_grid"
OUTPUT_BASENAME = "weighted_score_grid"
DETECTOR_SPECS = (
    {
        "detector_family": "standard_zscore",
        "detector_label": "Z-score",
        "source_roots": {
            "global": ZSCORE_GLOBAL_ROOT,
            "by_country": ZSCORE_BY_COUNTRY_ROOT,
            "by_competitor": ZSCORE_BY_COMPETITOR_ROOT,
        },
    },
    {
        "detector_family": "modified_mad",
        "detector_label": "Modified MAD",
        "source_roots": {granularity: COARSE_ROOT for granularity in GRANULARITIES},
    },
    {
        "detector_family": "modified_sn",
        "detector_label": "Modified Sn",
        "source_roots": {granularity: COARSE_ROOT for granularity in GRANULARITIES},
    },
    {
        "detector_family": "hybrid_avg",
        "detector_label": "Hybrid average",
        "source_roots": {granularity: COARSE_ROOT for granularity in GRANULARITIES},
    },
    {
        "detector_family": "hybrid_max",
        "detector_label": "Hybrid maximum",
        "source_roots": {granularity: COARSE_ROOT for granularity in GRANULARITIES},
    },
    {
        "detector_family": "hybrid_weighted",
        "detector_label": "Hybrid weighted",
        "source_roots": {granularity: HYBRID_WEIGHTED_ROOT for granularity in GRANULARITIES},
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate sampled weighted F1 and G-mean scores from the statistical tuning runs "
            "and render a six-panel detector grid."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for the aggregated CSV and plot outputs.",
    )
    return parser.parse_args()


def find_scope_dirs(source_root: Path, mh_value: int, granularity: str) -> list[Path]:
    granularity_dir = source_root / f"mh{mh_value}" / granularity
    if not granularity_dir.exists():
        raise FileNotFoundError(f"Missing granularity directory: {granularity_dir}")

    return sorted(path for path in granularity_dir.iterdir() if path.is_dir())


def load_scope_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in DETECTOR_SPECS:
        for granularity in GRANULARITIES:
            source_root = Path(spec["source_roots"][granularity])
            for mh_value in SAMPLED_MH_VALUES:
                for scope_dir in find_scope_dirs(source_root, mh_value, granularity):
                    config_path = scope_dir / str(spec["detector_family"]) / "best_configuration.json"
                    if not config_path.exists():
                        continue

                    payload = json.loads(config_path.read_text(encoding="utf-8"))
                    best_candidate = payload.get("best_candidate")
                    configuration = payload.get("configuration")
                    if not isinstance(best_candidate, dict) or not isinstance(configuration, dict):
                        continue
                    if "threshold" not in configuration:
                        continue
                    rows.append(
                        {
                            "detector_family": spec["detector_family"],
                            "detector_label": spec["detector_label"],
                            "mh_value": mh_value,
                            "mh_level": f"mh{mh_value}",
                            "granularity": payload.get("granularity", granularity),
                            "granularity_label": GRANULARITY_LABELS.get(granularity, granularity),
                            "scope_id": payload.get("scope_id"),
                            "threshold": float(configuration["threshold"]),
                            "weighted_f1_mean": float(best_candidate["weighted_f1_mean"]),
                            "weighted_g_mean": (
                                0.7 * float(best_candidate["new_prices_g_mean_mean"])
                                + 0.3 * float(best_candidate["new_products_g_mean_mean"])
                            ),
                            "combined_f1": float(best_candidate["combined_f1"]),
                            "rank_score": float(best_candidate["rank_score"]),
                            "source_root": str(source_root),
                            "best_configuration_path": str(config_path),
                        }
                    )

    frame = pd.DataFrame(rows)
    return frame.sort_values(["detector_label", "granularity", "mh_value", "scope_id"]).reset_index(drop=True)


def build_aggregate_frame(scope_frame: pd.DataFrame) -> pd.DataFrame:
    aggregate = (
        scope_frame.groupby(
            ["detector_family", "detector_label", "granularity", "granularity_label", "mh_value", "mh_level"],
            as_index=False,
        )
        .agg(
            average_weighted_f1=("weighted_f1_mean", "mean"),
            average_weighted_g_mean=("weighted_g_mean", "mean"),
            scope_count=("scope_id", "nunique"),
        )
        .sort_values(["detector_label", "granularity", "mh_value"])
        .reset_index(drop=True)
    )
    return aggregate


def plot_detector_grid(frame: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8.8, 14.6), constrained_layout=True)
    outer_grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[3.0, 0.46],
        hspace=0.16,
    )
    panel_grid = outer_grid[0].subgridspec(3, 2)
    legend_grid = outer_grid[1].subgridspec(1, 2)
    axes = [
        fig.add_subplot(panel_grid[row, col])
        for row in range(3)
        for col in range(2)
    ]
    metric_legend_ax = fig.add_subplot(legend_grid[0, 0])
    granularity_legend_ax = fig.add_subplot(legend_grid[0, 1])
    metric_legend_ax.axis("off")
    granularity_legend_ax.axis("off")

    for index, (axis, spec) in enumerate(zip(axes, DETECTOR_SPECS, strict=True)):
        detector_frame = frame[frame["detector_family"] == spec["detector_family"]].copy()
        for granularity in GRANULARITIES:
            granularity_frame = detector_frame[detector_frame["granularity"] == granularity].sort_values("mh_value")
            if granularity_frame.empty:
                continue
            marker = str(GRANULARITY_MARKERS.get(granularity, "o"))
            axis.plot(
                granularity_frame["mh_value"],
                granularity_frame["average_weighted_f1"],
                **series_style("combined_f1", marker=marker),
            )
            axis.plot(
                granularity_frame["mh_value"],
                granularity_frame["average_weighted_g_mean"],
                **series_style("combined_g_mean", marker=marker),
            )
        axis.set_title(str(spec["detector_label"]), fontsize=PLOT_TITLE_FONTSIZE, pad=10)
        axis.set_xticks(list(SAMPLED_MH_VALUES))
        axis.set_ylim(SCORE_YMIN, SCORE_YMAX)
        axis.set_yticks(SCORE_YTICKS)
        axis.grid(True, axis="y", color=GRID_Y_COLOR, linewidth=0.8)
        axis.grid(True, axis="x", color=GRID_X_COLOR, linewidth=0.6)
        axis.tick_params(axis="both", labelsize=PLOT_TICK_FONTSIZE)
        axis.tick_params(axis="x", pad=3)
        if index % 2 == 0:
            axis.set_ylabel("Weighted combined score", fontsize=PLOT_LABEL_FONTSIZE, labelpad=8)
        if index < 4:
            axis.tick_params(axis="x", labelbottom=False)
        else:
            axis.set_xlabel(X_AXIS_LABEL, fontsize=PLOT_LABEL_FONTSIZE, labelpad=8)

    metric_handles = metric_legend_handles(
        keys=WEIGHTED_SCORE_METRIC_KEYS,
        markers=WEIGHTED_SCORE_METRIC_MARKERS,
    )
    granularity_handles = granularity_legend_handles(GRANULARITIES)
    metric_legend_ax.legend(
        handles=metric_handles,
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
    granularity_legend_ax.legend(
        handles=granularity_handles,
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
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scope_frame = load_scope_rows()
    aggregate_frame = build_aggregate_frame(scope_frame)
    aggregate_frame.to_csv(output_dir / f"{OUTPUT_BASENAME}.csv", index=False)
    plot_detector_grid(aggregate_frame, output_dir)
    print(f"Wrote aggregated data and plots to {output_dir}")


if __name__ == "__main__":
    main()
