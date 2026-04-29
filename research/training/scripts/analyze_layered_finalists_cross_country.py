#!/usr/bin/env python3
"""Aggregate cross-country layered-finalist results into thesis-facing summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from research.training.scripts.thesis_plot_style import (
    COMBINED_F1_COLOR,
    COMBINED_GMEAN_COLOR,
    GRID_X_COLOR,
    GRID_Y_COLOR,
    LAYER_CONFIGURATION_LEGEND_TITLE,
    METRIC_LEGEND_TITLE,
    NEUTRAL_LEGEND_COLOR,
    PLOT_LABEL_FONTSIZE,
    PLOT_LEGEND_FONTSIZE,
    PLOT_LEGEND_TITLE_FONTSIZE,
    PLOT_TICK_FONTSIZE,
    SCORE_YMAX,
    SCORE_YMIN,
    SCORE_YTICKS,
    THRESHOLD_COLOR,
    WEIGHTED_SCORE_METRIC_KEYS,
    X_AXIS_LABEL,
    metric_legend_handles,
    save_thesis_media,
)


RUN_ID = "all_competitors_all_countries_country_level_layered_finalists"
THESIS_METRICS_DIR = (
    _PROJECT_ROOT
    / "results"
    / "detector_combinations"
    / RUN_ID
    / "analysis"
    / "thesis_metrics"
)

METRICS_CSV = THESIS_METRICS_DIR / "layered_detector_metrics.csv"
ANOMALY_CASE_CSV = THESIS_METRICS_DIR / "layered_detector_anomaly_case_metrics.csv"
OUTPUT_DIR = THESIS_METRICS_DIR

COMBO_ORDER = [
    "Sanity -> Z-score",
    "Sanity -> IF",
    "Sanity -> Z-score (>=10) -> IF",
    "Sanity -> Z-score (>=5) -> IF",
]
BALANCED_TOP_TWO_COMBOS = ["Sanity -> Z-score", "Sanity -> IF"]
HIGH_RECALL_TOP_TWO_COMBOS = ["Sanity -> IF", "Sanity -> Z-score (>=5) -> IF"]
COMBO_STYLES = {
    "Sanity -> Z-score": {"marker": "o", "linestyle": "-", "label": "Sanity -> Z-score"},
    "Sanity -> IF": {"marker": "s", "linestyle": "--", "label": "Sanity -> IF"},
    "Sanity -> Z-score (>=10) -> IF": {
        "marker": "^",
        "linestyle": "-.",
        "label": "Sanity -> Z-score (>=10) -> IF",
    },
    "Sanity -> Z-score (>=5) -> IF": {
        "marker": "D",
        "linestyle": ":",
        "label": "Sanity -> Z-score (>=5) -> IF",
    },
}
BALANCED_OUTPUT_BASENAME = "layered_finalist_mh_trends"
RECALL_OUTPUT_BASENAME = "layered_finalist_recall_mh_trends"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate the cross-country layered-finalist run into thesis-facing "
            "summary tables and an mh-trend figure."
        ),
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=METRICS_CSV,
        help="Path to layered_detector_metrics.csv.",
    )
    parser.add_argument(
        "--anomaly-case-csv",
        type=Path,
        default=ANOMALY_CASE_CSV,
        help="Path to layered_detector_anomaly_case_metrics.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for the aggregated Chapter 6 outputs.",
    )
    return parser.parse_args()


def _load_combined_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    combined = frame[frame["test_case_name"] == "combined"].copy()
    if combined.empty:
        raise ValueError(f"No combined rows found in {path}")
    combined["mh_value"] = combined["mh_level"].astype(str).str.replace("mh", "", regex=False).astype(int)
    return combined


def build_overall_summary(metrics_frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        metrics_frame.groupby("detector_combination", as_index=False)
        .agg(
            valid_scope_mh_evaluations=("scope_id", "count"),
            unique_scopes=("scope_id", "nunique"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1_wc=("f1", "mean"),
            mean_g_wc=("g_mean", "mean"),
        )
        .reset_index(drop=True)
    )
    summary["detector_combination"] = pd.Categorical(
        summary["detector_combination"],
        categories=COMBO_ORDER,
        ordered=True,
    )
    summary = summary.sort_values("detector_combination").reset_index(drop=True)
    return summary


def build_mh_summary(metrics_frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        metrics_frame.groupby(["mh_level", "mh_value", "detector_combination"], as_index=False)
        .agg(
            valid_scope_evaluations=("scope_id", "count"),
            unique_scopes=("scope_id", "nunique"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1_wc=("f1", "mean"),
            mean_g_wc=("g_mean", "mean"),
        )
        .reset_index(drop=True)
    )
    summary["detector_combination"] = pd.Categorical(
        summary["detector_combination"],
        categories=COMBO_ORDER,
        ordered=True,
    )
    summary = summary.sort_values(["mh_value", "detector_combination"]).reset_index(drop=True)
    return summary


def build_winner_counts(metrics_frame: pd.DataFrame) -> pd.DataFrame:
    winner_rows: list[dict[str, object]] = []
    for mh_level, mh_frame in metrics_frame.groupby("mh_level", sort=True):
        grouped = mh_frame.groupby("scope_id", sort=True)
        valid_scope_count = grouped.ngroups

        for metric_name, metric_column in (("f1_wc", "f1"), ("g_wc", "g_mean")):
            counts: dict[str, int] = {}
            for _, scope_frame in grouped:
                winning_combo = (
                    scope_frame.sort_values(
                        [metric_column, "detector_combination"],
                        ascending=[False, True],
                        kind="stable",
                    )
                    .iloc[0]["detector_combination"]
                )
                counts[str(winning_combo)] = counts.get(str(winning_combo), 0) + 1
            for combo in COMBO_ORDER:
                winner_rows.append(
                    {
                        "mh_level": mh_level,
                        "mh_value": int(str(mh_level).replace("mh", "")),
                        "metric": metric_name,
                        "detector_combination": combo,
                        "scope_wins": int(counts.get(combo, 0)),
                        "valid_scope_count": valid_scope_count,
                    }
                )

    winners = pd.DataFrame(winner_rows)
    winners["detector_combination"] = pd.Categorical(
        winners["detector_combination"],
        categories=COMBO_ORDER,
        ordered=True,
    )
    winners = winners.sort_values(["mh_value", "metric", "detector_combination"]).reset_index(drop=True)
    return winners


def build_top_two_anomaly_summary(anomaly_frame: pd.DataFrame) -> pd.DataFrame:
    filtered = anomaly_frame[anomaly_frame["detector_combination"].isin(BALANCED_TOP_TWO_COMBOS)].copy()
    summary = (
        filtered.groupby(["anomaly_case", "detector_combination"], as_index=False)
        .agg(
            valid_scope_mh_evaluations=("scope_id", "count"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1", "mean"),
            mean_g=("g_mean", "mean"),
        )
        .reset_index(drop=True)
    )
    summary["detector_combination"] = pd.Categorical(
        summary["detector_combination"],
        categories=BALANCED_TOP_TWO_COMBOS,
        ordered=True,
    )
    summary = summary.sort_values(["anomaly_case", "detector_combination"]).reset_index(drop=True)
    return summary


def build_high_recall_anomaly_summary(anomaly_frame: pd.DataFrame) -> pd.DataFrame:
    filtered = anomaly_frame[anomaly_frame["detector_combination"].isin(HIGH_RECALL_TOP_TWO_COMBOS)].copy()
    summary = (
        filtered.groupby(["anomaly_case", "detector_combination"], as_index=False)
        .agg(
            valid_scope_mh_evaluations=("scope_id", "count"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1", "mean"),
            mean_g=("g_mean", "mean"),
        )
        .reset_index(drop=True)
    )
    summary["detector_combination"] = pd.Categorical(
        summary["detector_combination"],
        categories=HIGH_RECALL_TOP_TWO_COMBOS,
        ordered=True,
    )
    summary = summary.sort_values(["anomaly_case", "detector_combination"]).reset_index(drop=True)
    return summary


def build_recall_stability_summary(metrics_frame: pd.DataFrame) -> pd.DataFrame:
    overall = (
        metrics_frame.groupby("detector_combination", as_index=False)
        .agg(
            valid_scope_mh_evaluations=("scope_id", "count"),
            mean_recall=("recall", "mean"),
            std_recall=("recall", "std"),
            min_recall=("recall", "min"),
            max_recall=("recall", "max"),
        )
        .reset_index(drop=True)
    )
    scope_means = (
        metrics_frame.groupby(["detector_combination", "scope_id"], as_index=False)
        .agg(scope_mean_recall=("recall", "mean"))
        .reset_index(drop=True)
    )
    scope_stability = (
        scope_means.groupby("detector_combination", as_index=False)
        .agg(scope_mean_recall_std=("scope_mean_recall", "std"))
        .reset_index(drop=True)
    )
    summary = overall.merge(scope_stability, on="detector_combination", how="left")
    summary["detector_combination"] = pd.Categorical(
        summary["detector_combination"],
        categories=COMBO_ORDER,
        ordered=True,
    )
    summary = summary.sort_values("detector_combination").reset_index(drop=True)
    return summary


def build_recall_winner_counts(metrics_frame: pd.DataFrame) -> pd.DataFrame:
    winners = (
        metrics_frame.sort_values(
            ["scope_id", "mh_value", "recall", "precision", "detector_combination"],
            ascending=[True, True, False, False, True],
            kind="stable",
        )
        .groupby(["scope_id", "mh_level", "mh_value"], as_index=False, sort=True)
        .head(1)
    )
    counts = (
        winners.groupby("detector_combination", as_index=False)
        .agg(recall_wins=("scope_id", "count"))
        .reset_index(drop=True)
    )
    counts["detector_combination"] = pd.Categorical(
        counts["detector_combination"],
        categories=COMBO_ORDER,
        ordered=True,
    )
    counts = counts.sort_values("detector_combination").reset_index(drop=True)
    counts["total_valid_scope_mh_pairs"] = int(len(winners))
    return counts


def plot_mh_trends(
    mh_summary: pd.DataFrame,
    output_dir: Path,
    y_columns: tuple[str, str],
    y_labels: tuple[str, str],
    y_colors: tuple[str, str],
    output_basename: str,
    ylabel: str,
) -> tuple[Path, Path]:
    fig = plt.figure(figsize=(8.8, 7.2), constrained_layout=True)
    outer_grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 0.34],
        hspace=0.16,
    )
    ax = fig.add_subplot(outer_grid[0, :])
    metric_legend_ax = fig.add_subplot(outer_grid[1, 0])
    layer_legend_ax = fig.add_subplot(outer_grid[1, 1])
    metric_legend_ax.axis("off")
    layer_legend_ax.axis("off")

    mh_values = sorted(mh_summary["mh_value"].unique())
    for combo in COMBO_ORDER:
        combo_frame = mh_summary[mh_summary["detector_combination"] == combo].sort_values("mh_value")
        if combo_frame.empty:
            continue
        style = COMBO_STYLES[combo]

        ax.plot(
            combo_frame["mh_value"],
            combo_frame[y_columns[0]],
            color=y_colors[0],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=2.1,
            markersize=5.2,
            alpha=0.95,
        )
        ax.plot(
            combo_frame["mh_value"],
            combo_frame[y_columns[1]],
            color=y_colors[1],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=2.1,
            markersize=5.2,
            alpha=0.95,
        )

    ax.set_xlim(min(mh_values) - 0.7, max(mh_values) + 0.7)
    ax.set_ylim(SCORE_YMIN, SCORE_YMAX)
    ax.set_xticks(mh_values)
    ax.set_yticks(SCORE_YTICKS)
    ax.set_xlabel(X_AXIS_LABEL, fontsize=PLOT_LABEL_FONTSIZE, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=PLOT_LABEL_FONTSIZE, labelpad=8)
    ax.tick_params(axis="both", labelsize=PLOT_TICK_FONTSIZE)
    ax.grid(axis="y", color=GRID_Y_COLOR, linewidth=0.9)
    ax.grid(axis="x", color=GRID_X_COLOR, linewidth=0.7)
    ax.set_axisbelow(True)

    metric_handles = metric_legend_handles(
        colors=y_colors,
        markers=None,
        labels=y_labels,
    )
    if y_columns == ("mean_f1_wc", "mean_g_wc"):
        metric_handles = metric_legend_handles(
            keys=WEIGHTED_SCORE_METRIC_KEYS,
            markers=None,
        )
    layer_handles = [
        Line2D(
            [0],
            [0],
            color=NEUTRAL_LEGEND_COLOR,
            linestyle=COMBO_STYLES[combo]["linestyle"],
            marker=COMBO_STYLES[combo]["marker"],
            linewidth=2.2,
            markersize=6.0,
            label=COMBO_STYLES[combo]["label"],
        )
        for combo in COMBO_ORDER
    ]
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
    layer_legend_ax.legend(
        handles=layer_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        frameon=False,
        fontsize=PLOT_LEGEND_FONTSIZE,
        title_fontsize=PLOT_LEGEND_TITLE_FONTSIZE,
        ncol=1,
        title=LAYER_CONFIGURATION_LEGEND_TITLE,
        handlelength=2.2,
        labelspacing=0.9,
        borderaxespad=0.0,
    )

    png_path, svg_path = save_thesis_media(fig, _PROJECT_ROOT, output_basename, dpi=220)
    plt.close(fig)
    return png_path, svg_path


def write_summary_markdown(
    output_dir: Path,
    overall_summary: pd.DataFrame,
    mh_winners: pd.DataFrame,
    recall_stability: pd.DataFrame,
    recall_winners: pd.DataFrame,
) -> Path:
    top_rows = overall_summary.sort_values(["mean_f1_wc", "mean_g_wc"], ascending=[False, False])
    best = top_rows.iloc[0]
    second = top_rows.iloc[1]
    recall_best = recall_stability.sort_values(["mean_recall", "std_recall"], ascending=[False, True]).iloc[0]

    f1_wins = (
        mh_winners[mh_winners["metric"] == "f1_wc"]
        .groupby("detector_combination", observed=False)["scope_wins"]
        .sum()
        .to_dict()
    )
    g_wins = (
        mh_winners[mh_winners["metric"] == "g_wc"]
        .groupby("detector_combination", observed=False)["scope_wins"]
        .sum()
        .to_dict()
    )
    recall_win_counts = (
        recall_winners.set_index("detector_combination")["recall_wins"].to_dict()
    )

    lines = [
        "# Cross-Country Layered Finalist Summary",
        "",
        f"- Best overall finalist by mean $F_{{1,\\mathrm{{wc}}}}$: `{best['detector_combination']}` "
        f"({best['mean_f1_wc']:.4f}, $G_{{\\mathrm{{wc}}}}={best['mean_g_wc']:.4f}$)",
        f"- Second-best finalist: `{second['detector_combination']}` "
        f"({second['mean_f1_wc']:.4f}, $G_{{\\mathrm{{wc}}}}={second['mean_g_wc']:.4f}$)",
        f"- Highest-recall finalist: `{recall_best['detector_combination']}` "
        f"(mean recall {recall_best['mean_recall']:.4f}, std {recall_best['std_recall']:.4f})",
        f"- Scope-mh $F_{{1,\\mathrm{{wc}}}}$ wins: `{json.dumps(f1_wins, sort_keys=True)}`",
        f"- Scope-mh $G_{{\\mathrm{{wc}}}}$ wins: `{json.dumps(g_wins, sort_keys=True)}`",
        f"- Scope-mh recall wins: `{json.dumps(recall_win_counts, sort_keys=True)}`",
        "",
        "The aggregate comparison therefore still favors `Sanity -> Z-score` on the balanced criteria, "
        "whereas the recall-first comparison favors `Sanity -> Z-score (>=5) -> IF`.",
    ]
    summary_path = output_dir / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def write_data_references(output_dir: Path) -> Path:
    content = """# Data References

- `tab:layered-finalist-overall` got data from `layered_finalist_overall_summary.csv`, presents the overall cross-country comparison of the four layered finalists on the combined test case.
- `fig:layered-finalist-mh-trends` (`results/media/layered_finalist_mh_trends.png` / `results/media/layered_finalist_mh_trends.svg`) got data from `layered_finalist_mh_summary.csv`, presents the mean `F_{1,\\mathrm{wc}}` and `G_{\\mathrm{wc}}` trends of the layered finalists across the sampled minimum-history settings, with numeric minimum-history tick labels and separate metric and layer-configuration legends.
- `fig:layered-finalist-recall-trends` (`results/media/layered_finalist_recall_mh_trends.png` / `results/media/layered_finalist_recall_mh_trends.svg`) got data from `layered_finalist_mh_summary.csv`, presents the mean recall and mean `F_{1,\\mathrm{wc}}` trends of the layered finalists across the sampled minimum-history settings.
- Chapter 6 minimum-history winner-count discussion got data from `layered_finalist_mh_winner_counts.csv`, presents per-`mh` scope-win counts for `F_{1,\\mathrm{wc}}` and `G_{\\mathrm{wc}}`.
- `tab:layered-finalist-recall-stability` got data from `layered_finalist_recall_stability_summary.csv`, presents overall recall stability of the layered finalists across the valid scope-`mh` surface.
- Chapter 6 recall-winner discussion got data from `layered_finalist_recall_winner_counts.csv`, presents recall winner counts across the full valid scope-`mh` surface.
- `tab:layered-finalist-anomaly-types` got data from `layered_finalist_high_recall_anomaly_case_summary.csv`, presents the anomaly-type recall comparison for `Sanity -> IF` and `Sanity -> Z-score (>=5) -> IF`.
- `layered_finalist_overall_summary.csv`, `layered_finalist_mh_summary.csv`, and `layered_finalist_mh_winner_counts.csv` were aggregated from `layered_detector_metrics.csv` using rows where `test_case_name == combined`.
- `layered_finalist_recall_stability_summary.csv` and `layered_finalist_recall_winner_counts.csv` were aggregated from `layered_detector_metrics.csv` using rows where `test_case_name == combined`.
- `layered_finalist_top2_anomaly_case_summary.csv` and `layered_finalist_high_recall_anomaly_case_summary.csv` were aggregated from `layered_detector_anomaly_case_metrics.csv` using rows where `test_case_name == combined`.
"""
    path = output_dir / "data_references.md"
    path.write_text(content, encoding="utf-8")
    return path


def run(metrics_csv: Path, anomaly_case_csv: Path, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_frame = _load_combined_rows(metrics_csv)
    anomaly_frame = _load_combined_rows(anomaly_case_csv)

    overall_summary = build_overall_summary(metrics_frame)
    mh_summary = build_mh_summary(metrics_frame)
    mh_winner_counts = build_winner_counts(metrics_frame)
    top_two_anomaly_summary = build_top_two_anomaly_summary(anomaly_frame)
    high_recall_anomaly_summary = build_high_recall_anomaly_summary(anomaly_frame)
    recall_stability_summary = build_recall_stability_summary(metrics_frame)
    recall_winner_counts = build_recall_winner_counts(metrics_frame)

    overall_path = output_dir / "layered_finalist_overall_summary.csv"
    mh_summary_path = output_dir / "layered_finalist_mh_summary.csv"
    mh_winner_path = output_dir / "layered_finalist_mh_winner_counts.csv"
    anomaly_summary_path = output_dir / "layered_finalist_top2_anomaly_case_summary.csv"
    high_recall_anomaly_path = output_dir / "layered_finalist_high_recall_anomaly_case_summary.csv"
    recall_stability_path = output_dir / "layered_finalist_recall_stability_summary.csv"
    recall_winner_path = output_dir / "layered_finalist_recall_winner_counts.csv"

    overall_summary.to_csv(overall_path, index=False)
    mh_summary.to_csv(mh_summary_path, index=False)
    mh_winner_counts.to_csv(mh_winner_path, index=False)
    top_two_anomaly_summary.to_csv(anomaly_summary_path, index=False)
    high_recall_anomaly_summary.to_csv(high_recall_anomaly_path, index=False)
    recall_stability_summary.to_csv(recall_stability_path, index=False)
    recall_winner_counts.to_csv(recall_winner_path, index=False)

    png_path, svg_path = plot_mh_trends(
        mh_summary,
        output_dir,
        y_columns=("mean_f1_wc", "mean_g_wc"),
        y_labels=(r"$F_{1,\mathrm{wc}}$", r"$G_{\mathrm{wc}}$"),
        y_colors=(COMBINED_F1_COLOR, COMBINED_GMEAN_COLOR),
        output_basename=BALANCED_OUTPUT_BASENAME,
        ylabel="Weighted score",
    )
    recall_png_path, recall_svg_path = plot_mh_trends(
        mh_summary,
        output_dir,
        y_columns=("mean_recall", "mean_f1_wc"),
        y_labels=("Recall", r"$F_{1,\mathrm{wc}}$"),
        y_colors=(THRESHOLD_COLOR, COMBINED_F1_COLOR),
        output_basename=RECALL_OUTPUT_BASENAME,
        ylabel="Score",
    )
    summary_path = write_summary_markdown(
        output_dir,
        overall_summary,
        mh_winner_counts,
        recall_stability_summary,
        recall_winner_counts,
    )
    references_path = write_data_references(output_dir)

    return {
        "overall_summary": overall_path,
        "mh_summary": mh_summary_path,
        "mh_winner_counts": mh_winner_path,
        "anomaly_summary": anomaly_summary_path,
        "high_recall_anomaly_summary": high_recall_anomaly_path,
        "recall_stability_summary": recall_stability_path,
        "recall_winner_counts": recall_winner_path,
        "figure_png": png_path,
        "figure_svg": svg_path,
        "recall_figure_png": recall_png_path,
        "recall_figure_svg": recall_svg_path,
        "summary_md": summary_path,
        "data_references": references_path,
    }


def main() -> None:
    args = parse_args()
    outputs = run(args.metrics_csv, args.anomaly_case_csv, args.output_dir)
    print("Generated Chapter 6 aggregate artifacts:")
    for label, path in outputs.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
