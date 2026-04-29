"""Shared plotting style constants for thesis analysis figures."""

from __future__ import annotations

from pathlib import Path

from matplotlib.lines import Line2D

THRESHOLD_YMIN = 2.0
THRESHOLD_YMAX = 5.0
THRESHOLD_YTICKS = [2.0 + (0.5 * index) for index in range(7)]
SCORE_YMIN = 0.0
SCORE_YMAX = 1.0
SCORE_YTICKS = [0.2 * index for index in range(6)]
X_AXIS_LABEL = "Minimum history of products in set"

PLOT_TITLE_FONTSIZE = 15
PLOT_LABEL_FONTSIZE = 13
PLOT_TICK_FONTSIZE = 11
PLOT_LEGEND_FONTSIZE = 13
PLOT_LEGEND_TITLE_FONTSIZE = 13
METRIC_LEGEND_TITLE = "Metric"
GRANULARITY_LEGEND_TITLE = "Granularity"
LAYER_CONFIGURATION_LEGEND_TITLE = "Layer configuration"
WEIGHTED_SCORE_METRIC_KEYS = ("combined_g_mean", "combined_f1")
WEIGHTED_SCORE_METRIC_MARKERS = ("s", "o")
MEDIA_OUTPUT_RELATIVE_DIR = Path("results") / "media"

GUIDE_COLOR = "#d9d9d9"
GRID_Y_COLOR = "#e6e6e6"
GRID_X_COLOR = "#f2f2f2"
NEUTRAL_LEGEND_COLOR = "#4d4d4d"

THRESHOLD_COLOR = "#0072B2"
SAMPLED_THRESHOLD_COLOR = "#E69F00"
COMBINED_F1_COLOR = "#009E73"
COMBINED_GMEAN_COLOR = "#D55E00"
SAMPLED_COMBINED_F1_COLOR = "#CC79A7"
SAMPLED_COMBINED_GMEAN_COLOR = "#56B4E9"

SCOPE_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*"]
GRANULARITY_MARKERS = {
    "global": "o",
    "by_country": "s",
    "by_competitor": "^",
}
GRANULARITY_LINESTYLES = {
    "global": "-",
    "by_country": "--",
    "by_competitor": ":",
}
GRANULARITY_LABELS = {
    "global": "Global",
    "by_country": "By country",
    "by_competitor": "By competitor",
}

SERIES_STYLES: dict[str, dict[str, object]] = {
    "threshold": {
        "color": THRESHOLD_COLOR,
        "linestyle": "-",
        "linewidth": 2.0,
        "alpha": 0.95,
        "markersize": 4.5,
        "label": "Z-score threshold",
    },
    "sampled_threshold": {
        "color": SAMPLED_THRESHOLD_COLOR,
        "linestyle": "--",
        "linewidth": 2.0,
        "alpha": 0.95,
        "markersize": 4.5,
        "label": "Sampled mh threshold",
    },
    "combined_f1": {
        "color": COMBINED_F1_COLOR,
        "linestyle": "-",
        "linewidth": 2.0,
        "alpha": 0.95,
        "markersize": 4.5,
        "label": r"$F_{1,\mathrm{wc}}$",
    },
    "sampled_combined_f1": {
        "color": SAMPLED_COMBINED_F1_COLOR,
        "linestyle": "--",
        "linewidth": 2.0,
        "alpha": 0.95,
        "markersize": 4.5,
        "label": "Sampled weighted combined F1",
    },
    "combined_g_mean": {
        "color": COMBINED_GMEAN_COLOR,
        "linestyle": "-",
        "linewidth": 2.0,
        "alpha": 0.95,
        "markersize": 4.5,
        "label": r"$G_{\mathrm{wc}}$",
    },
    "sampled_combined_g_mean": {
        "color": SAMPLED_COMBINED_GMEAN_COLOR,
        "linestyle": "--",
        "linewidth": 2.0,
        "alpha": 0.95,
        "markersize": 4.5,
        "label": "Sampled weighted combined G-mean",
    },
}


def series_style(key: str, *, marker: str) -> dict[str, object]:
    """Return a canonical series style with only the marker overridden."""
    return {
        "marker": marker,
        **SERIES_STYLES[key],
    }


def legend_handle(key: str, *, marker: str = "o") -> Line2D:
    """Return a legend handle for one canonical series."""
    return Line2D(
        [0],
        [0],
        marker=marker,
        **SERIES_STYLES[key],
    )


def save_thesis_media(fig: object, project_root: Path, output_basename: str, *, dpi: int = 220) -> tuple[Path, Path]:
    """Save a thesis-facing figure under the central results/media directory."""
    output_dir = project_root / MEDIA_OUTPUT_RELATIVE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{output_basename}.png"
    svg_path = output_dir / f"{output_basename}.svg"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    return png_path, svg_path


def metric_legend_handles(
    keys: tuple[str, ...] = ("combined_f1", "combined_g_mean"),
    *,
    colors: tuple[str, ...] | None = None,
    markers: tuple[str, ...] | None = ("o", "s"),
    labels: tuple[str, ...] | None = None,
) -> list[Line2D]:
    """Return shared handles for score-metric legends."""
    handles: list[Line2D] = []
    for index, key in enumerate(keys):
        style = SERIES_STYLES[key]
        marker = markers[index] if markers is not None and index < len(markers) else None
        handles.append(
            Line2D(
                [0],
                [0],
                color=colors[index] if colors is not None and index < len(colors) else style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                alpha=style["alpha"],
                marker=marker,
                markersize=style["markersize"] if marker else 0,
                label=labels[index] if labels is not None and index < len(labels) else style["label"],
            )
        )
    return handles


def granularity_legend_handles(
    granularities: tuple[str, ...],
    *,
    include_linestyle: bool = False,
    linewidth: float = 2.0,
    markersize: float = 6.0,
) -> list[Line2D]:
    """Return shared handles for granularity legends."""
    return [
        Line2D(
            [0],
            [0],
            color=NEUTRAL_LEGEND_COLOR,
            linestyle=GRANULARITY_LINESTYLES.get(granularity, "-") if include_linestyle else "-",
            linewidth=linewidth,
            marker=GRANULARITY_MARKERS.get(granularity, "o"),
            markersize=markersize,
            label=GRANULARITY_LABELS.get(granularity, granularity),
        )
        for granularity in granularities
    ]


def granularity_style(
    granularity: str,
    *,
    color: str,
    label: str | None = None,
    linewidth: float = 2.2,
    alpha: float = 0.95,
    markersize: float = 5.5,
) -> dict[str, object]:
    """Return a canonical style for a granularity-comparison series."""
    return {
        "color": color,
        "linestyle": GRANULARITY_LINESTYLES.get(granularity, "-"),
        "linewidth": linewidth,
        "alpha": alpha,
        "marker": GRANULARITY_MARKERS.get(granularity, "o"),
        "markersize": markersize,
        "label": label or GRANULARITY_LABELS.get(granularity, granularity),
    }
