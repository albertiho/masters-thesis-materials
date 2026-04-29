# Thesis Anomaly Detection Research Repository

This repository accompanies a master's thesis on anomaly detection for competitor-pricing data. It contains the detector implementations, evaluation utilities, training scripts, analysis scripts, and figure-generation code used to reproduce the reported experimental results.

The repository is organized around script-based replication. The main entrypoints are under `research/training/scripts/`, publication-ready figures are written under `results/media/`, and manual revalidation guidance is in `REVALIDATE.md`.

## Repository Layout

- `src/`: importable Python package with anomaly detectors, feature extraction, evaluation utilities, and shared research helpers.
- `research/training/scripts/`: training, tuning, aggregation, replay, and plotting scripts used for the thesis experiments.
- `data/training/`: bundled-data preparation inputs and generated derived split files.
- `data-subsets/`: generated minimum-history dataset variants used by the thesis-scale tuning and evaluation scripts.
- `results/`: generated tuning outputs, analysis summaries, provenance notes, and thesis figures.
- `results/media/`: PNG/SVG figures intended for direct thesis use.
- `configs/tuning_config.json`: default minimum-history and split settings used by the bundled-data preparation path.
- `REVALIDATE.md`: manual guide for tracing and regenerating thesis-facing figures and tables.

## Setup

Create a virtual environment and install the development dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

Run commands from this repository root unless a script documents otherwise.

## Dataset Subsets And Splits

The thesis-scale data preparation is handled by one script:

```powershell
python research/training/scripts/split_training_data.py
```

The script expects cleaned source parquet files under:

```text
../cleaned-data/training/by_competitor/
```

It performs four preparation steps:

1. Creates an `mh5` copy-through dataset under `data-subsets/`.
2. Creates filtered `mh10`, `mh15`, `mh20`, `mh25`, and `mh30` dataset variants under `data-subsets/`.
3. Builds `by_country`, `by_country_market`, and `global` aggregate datasets for each processed root.
4. Writes train/test split files beside each processed source parquet:

```text
*_train.parquet
*_test_new_prices.parquet
*_test_new_products.parquet
```

The generated subset and split files can be large because they duplicate the source data across several minimum-history settings and aggregation levels. They are excluded from Git by `.gitignore`.

For the smaller bundled-data path, the preparation command is:

```powershell
python -m src.research.prepare_data --data-root data/training
```

That command writes deterministic derived files under `data/training/derived/`.

## Generating Thesis Figures

After tuning or evaluation scripts have produced result directories under `results/`, the figure scripts aggregate those outputs and write final media to `results/media/`.

```powershell
python research/training/scripts/plot_statistical_global_f1_grid.py
python research/training/scripts/analyze_modified_zscore_scale_mh5.py
python research/training/scripts/plot_isolation_forest_granularity_grid.py
python research/training/scripts/analyze_layered_finalists_cross_country.py
```

These commands produce:

- `results/media/weighted_score_grid.png`
- `results/media/modified_zscore_scale_mh5__all_methods.png`
- `results/media/if_granularity_score_grid.png`
- `results/media/layered_finalist_mh_trends.png`
- `results/media/layered_finalist_recall_mh_trends.png`

SVG versions are written beside the PNG files.

Additional scripts for method selection, minimum-history sampling checks, and replay analysis are documented in `research/training/scripts/README.md`.

## Manual Revalidation

Revalidation is no longer a separate programmatic manifest workflow. To
revalidate a thesis figure or table, follow the provenance chain from the
artifact in `results/media/` or `results/analysis/` back to the CSV/JSON inputs
listed in the relevant `data_references.md` file, then rerun the corresponding
script under `research/training/scripts/`.

See `REVALIDATE.md` for the figure-to-input map, recommended checks, and the
folder/file names that matter for manual review.

## Provenance

The analysis outputs include local `data_references.md` files that map each thesis figure or table back to the CSV/JSON inputs used to produce it. The main provenance entrypoints are:

- `research/training/scripts/README.md`
- `results/analysis/statistical_detector_score_grid/data_references.md`
- `results/analysis/isolation_forest_granularity_comparison/data_references.md`
- `results/tuning/statistical/all_global_single_attempt_batch/mh5/analysis/modified_zscore_scale/data_references.md`
- `results/detector_combinations/all_competitors_all_countries_country_level_layered_finalists/analysis/thesis_metrics/data_references.md`

Use those files to trace generated figures and tables back to their source result files.
