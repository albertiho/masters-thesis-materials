#!/usr/bin/env python3
"""Visualize minimum-history subsampling adequacy from z-score guidance outputs.

This script combines the ``subset_guidance_by_case.csv`` files produced by
``analyze_statistical_guidance.py`` for the standard z-score detector across
``by_competitor``, ``by_country``, and ``global`` tuning. It then writes:

1. A combined scope-level CSV.
2. An aggregate summary table by granularity and overall.
3. A two-panel thesis figure showing:
   - Spearman rank correlation between the full and sampled candidate rankings.
   - Equal-support combined-F1 regret under the sampled minimum-history grid.

The purpose of the script is narrow: provide a direct visual companion for the
methodology subsection that justifies replacing the full ``mh5``-``mh30`` grid
with the sampled levels ``mh5,mh10,mh15,mh20,mh25,mh30``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from thesis_plot_style import (
    GRANULARITY_LABELS,
    GRANULARITY_MARKERS,
    GRID_X_COLOR,
    GRID_Y_COLOR,
    GUIDE_COLOR,
)

LOGGER = logging.getLogger("analyze_mh_subsampling_adequacy")

DEFAULT_SOURCE_CSVS = {
    "by_competitor": Path(
        "results/tuning/statistical/z_score_by_competitor_single_attempt_batch/analysis/subset_guidance_by_case.csv"
    ),
    "by_country": Path(
        "results/tuning/statistical/z_score_by_country_single_attempt_batch/analysis/subset_guidance_by_case.csv"
    ),
    "global": Path(
        "results/tuning/statistical/z_score_global_single_attempt_batch/analysis/subset_guidance_by_case.csv"
    ),
}
DEFAULT_OUTPUT_DIR = Path("results/tuning/statistical/z_score_mh_subsampling_adequacy/analysis")
GRANULARITY_ORDER = ("by_competitor", "by_country", "global")
GRANULARITY_COLORS = {
    "by_competitor": "#009E73",
    "by_country": "#E69F00",
    "global": "#0072B2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate z-score subset-guidance outputs and visualize whether the "
            "sampled minimum-history grid preserves the full-grid tuning signal."
        ),
    )
    parser.add_argument(
        "--by-competitor-csv",
        type=Path,
        default=DEFAULT_SOURCE_CSVS["by_competitor"],
        help="subset_guidance_by_case.csv for the by_competitor z-score sweep.",
    )
    parser.add_argument(
        "--by-country-csv",
        type=Path,
        default=DEFAULT_SOURCE_CSVS["by_country"],
        help="subset_guidance_by_case.csv for the by_country z-score sweep.",
    )
    parser.add_argument(
        "--global-csv",
        type=Path,
        default=DEFAULT_SOURCE_CSVS["global"],
        help="subset_guidance_by_case.csv for the global z-score sweep.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the generated CSV, Markdown summary, and figures.",
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


def load_guidance_frame(input_csv: Path, *, expected_granularity: str) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_csv}")

    frame = pd.read_csv(input_csv)
    required = {
        "granularity",
        "dataset_name",
        "scope_id",
        "rank_score_spearman",
        "equal_support_combined_f1_regret",
        "equal_support_config_match",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{input_csv} is missing required columns: {sorted(missing)}")

    filtered = frame[frame["granularity"] == expected_granularity].copy()
    if filtered.empty:
        raise ValueError(
            f"No rows found in {input_csv} for granularity={expected_granularity!r}"
        )

    filtered["rank_score_spearman"] = pd.to_numeric(filtered["rank_score_spearman"], errors="coerce")
    filtered["equal_support_combined_f1_regret"] = pd.to_numeric(
        filtered["equal_support_combined_f1_regret"],
        errors="coerce",
    )
    filtered["granularity_label"] = GRANULARITY_LABELS.get(expected_granularity, expected_granularity)
    filtered["source_csv"] = str(input_csv.resolve())
    filtered = filtered.dropna(subset=["rank_score_spearman", "equal_support_combined_f1_regret"])
    return filtered.sort_values(["dataset_name", "scope_id"])


def build_scope_table(frames: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True)
    combined["granularity_order"] = combined["granularity"].map(
        {granularity: index for index, granularity in enumerate(GRANULARITY_ORDER)}
    )
    combined = combined.sort_values(["granularity_order", "dataset_name", "scope_id"]).drop(
        columns=["granularity_order"]
    )
    return combined


def build_summary_table(scope_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for granularity in GRANULARITY_ORDER:
        subset = scope_table[scope_table["granularity"] == granularity].copy()
        if subset.empty:
            continue
        rows.append(
            {
                "granularity": granularity,
                "granularity_label": GRANULARITY_LABELS.get(granularity, granularity),
                "scope_count": int(len(subset)),
                "mean_spearman": float(subset["rank_score_spearman"].mean()),
                "median_spearman": float(subset["rank_score_spearman"].median()),
                "positive_equal_support_f1_loss_count": int(
                    (subset["equal_support_combined_f1_regret"] > 0).sum()
                ),
                "max_equal_support_f1_loss": float(subset["equal_support_combined_f1_regret"].max()),
            }
        )

    rows.append(
        {
            "granularity": "all",
            "granularity_label": "All",
            "scope_count": int(len(scope_table)),
            "mean_spearman": float(scope_table["rank_score_spearman"].mean()),
            "median_spearman": float(scope_table["rank_score_spearman"].median()),
            "positive_equal_support_f1_loss_count": int(
                (scope_table["equal_support_combined_f1_regret"] > 0).sum()
            ),
            "max_equal_support_f1_loss": float(scope_table["equal_support_combined_f1_regret"].max()),
        }
    )

    return pd.DataFrame(rows)


def write_summary_markdown(summary_table: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Minimum-History Subsampling Adequacy",
        "",
        "| Granularity | Scopes | Mean rho | Median rho | Pos. loss | Max F1 loss |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_table.itertuples(index=False):
        lines.append(
            f"| {row.granularity_label} | {int(row.scope_count)} | "
            f"{row.mean_spearman:.3f} | {row.median_spearman:.3f} | "
            f"{int(row.positive_equal_support_f1_loss_count)} | {row.max_equal_support_f1_loss:.4f} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _deterministic_offsets(count: int, *, half_width: float = 0.18) -> list[float]:
    if count <= 1:
        return [0.0]
    step = (2.0 * half_width) / (count - 1)
    return [(-half_width + (index * step)) for index in range(count)]


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", color=GRID_Y_COLOR, linewidth=0.9)
    ax.grid(True, axis="x", color=GRID_X_COLOR, linewidth=0.6, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_subsampling_adequacy(scope_table: pd.DataFrame, output_prefix: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), constrained_layout=True)

    panel_specs = (
        (
            axes[0],
            "rank_score_spearman",
            "Spearman correlation",
            "Scope-level rank stability",
            1.0,
        ),
        (
            axes[1],
            "equal_support_combined_f1_regret",
            "Equal-support combined F1 loss",
            "Equal-support performance loss",
            0.0,
        ),
    )

    x_positions = {granularity: index + 1 for index, granularity in enumerate(GRANULARITY_ORDER)}

    for ax, column, ylabel, title, reference_line in panel_specs:
        for granularity in GRANULARITY_ORDER:
            subset = scope_table[scope_table["granularity"] == granularity].copy()
            if subset.empty:
                continue
            subset = subset.sort_values(column)
            offsets = _deterministic_offsets(len(subset))
            x_base = x_positions[granularity]
            color = GRANULARITY_COLORS[granularity]
            marker = GRANULARITY_MARKERS.get(granularity, "o")

            ax.scatter(
                [x_base + offset for offset in offsets],
                subset[column],
                color=color,
                marker=marker,
                s=46,
                alpha=0.92,
                linewidths=0.0,
                zorder=3,
            )

            median_value = float(subset[column].median())
            ax.hlines(
                median_value,
                x_base - 0.24,
                x_base + 0.24,
                color="#222222",
                linewidth=2.0,
                zorder=4,
            )

        ax.axhline(reference_line, color=GUIDE_COLOR, linestyle="--", linewidth=1.2, zorder=1)
        _style_axis(ax)
        ax.set_xticks(list(x_positions.values()))
        ax.set_xticklabels(
            [GRANULARITY_LABELS.get(granularity, granularity) for granularity in GRANULARITY_ORDER],
            rotation=0,
        )
        ax.set_xlim(0.5, len(GRANULARITY_ORDER) + 0.5)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, pad=10)

    spearman_min = float(scope_table["rank_score_spearman"].min())
    loss_min = float(scope_table["equal_support_combined_f1_regret"].min())
    loss_max = float(scope_table["equal_support_combined_f1_regret"].max())
    axes[0].set_ylim(min(0.25, spearman_min - 0.03), 1.02)
    axes[1].set_ylim(min(-0.05, loss_min - 0.01), max(0.06, loss_max + 0.01))

    fig.suptitle("Adequacy of the Sampled Minimum-History Grid for the Standard Z-score Detector", fontsize=15)

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

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = [
        load_guidance_frame(args.by_competitor_csv.resolve(), expected_granularity="by_competitor"),
        load_guidance_frame(args.by_country_csv.resolve(), expected_granularity="by_country"),
        load_guidance_frame(args.global_csv.resolve(), expected_granularity="global"),
    ]

    scope_table = build_scope_table(frames)
    summary_table = build_summary_table(scope_table)

    scope_csv_path = output_dir / "mh_subsampling_guidance_by_case.csv"
    summary_csv_path = output_dir / "mh_subsampling_adequacy_summary.csv"
    summary_md_path = output_dir / "mh_subsampling_adequacy_summary.md"
    summary_json_path = output_dir / "mh_subsampling_adequacy_summary.json"

    scope_table.to_csv(scope_csv_path, index=False)
    summary_table.to_csv(summary_csv_path, index=False)
    write_summary_markdown(summary_table, summary_md_path)
    summary_json_path.write_text(
        json.dumps(summary_table.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )

    plot_subsampling_adequacy(
        scope_table,
        output_dir / "mh_subsampling_adequacy",
    )

    LOGGER.info("Wrote %s", scope_csv_path)
    LOGGER.info("Wrote %s", summary_csv_path)
    LOGGER.info("Wrote %s", summary_md_path)
    LOGGER.info("Wrote %s", summary_json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
