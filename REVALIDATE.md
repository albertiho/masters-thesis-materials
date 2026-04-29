# Manual Revalidation Guide

Revalidation in this repository is provenance-guided and script-based. There is
no separate revalidation CLI: use the committed source data, generated result
trees, script README, and `data_references.md` files to confirm or regenerate
the thesis-facing artifacts.

## Scope

The thesis-facing figures live in `results/media/`:

- `weighted_score_grid.png` / `.svg`
- `modified_zscore_scale_mh5__all_methods.png` / `.svg`
- `if_granularity_score_grid.png` / `.svg`
- `layered_finalist_mh_trends.png` / `.svg`
- `layered_finalist_recall_mh_trends.png` / `.svg`

The most useful provenance files are:

- `research/training/scripts/README.md`
- `results/analysis/statistical_detector_score_grid/data_references.md`
- `results/analysis/isolation_forest_granularity_comparison/data_references.md`
- `results/tuning/statistical/all_global_single_attempt_batch/mh5/analysis/modified_zscore_scale/data_references.md`
- `results/detector_combinations/all_competitors_all_countries_country_level_layered_finalists/analysis/thesis_metrics/data_references.md`

Large row-level artifacts such as `predictions.parquet`,
`injected_rows.parquet`, and `template_cache.joblib` are intentionally excluded
from the GitHub release. See the README files under `results/`,
`results/tuning/`, and `results/detector_combinations/` for the omitted file
patterns, approximate sizes, and generator scripts.

## Setup Check

Create the environment and run the normal test suite:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pytest
```

For a narrower confidence check around thesis artifacts:

```powershell
pytest tests/test_phase1_prepare_data.py tests/test_split_training_data.py tests/test_history_subsets.py
pytest tests/test_phase2_artifacts.py tests/test_synthetic_injection.py tests/test_evaluation_cache_behavior.py
pytest tests/test_tune_statistical_sweep.py tests/test_tune_forests_sweep.py
pytest tests/test_all_competitors_country_level_layered_finalists.py tests/test_extract_if_zscore_layered_thesis_metrics.py
```

## Data Preparation

For the bundled-data path, verify the source contract and regenerate derived
splits:

```powershell
python -m src.research.prepare_data --data-root data/training
```

Relevant files and folders:

- `data/training/source/dataset_manifest.json`
- `data/training/source/SHA256SUMS`
- `data/training/source/by_competitor/`
- `data/training/derived/split_manifest.json`

For thesis-scale runs, rebuild the minimum-history subsets and split files:

```powershell
python research/training/scripts/split_training_data.py
```

That script expects cleaned source parquet files under
`../cleaned-data/training/by_competitor/` and writes generated datasets under
`data-subsets/`.

## Rebuild Thesis Figures

After the relevant result trees exist, regenerate the publication-facing media:

```powershell
python research/training/scripts/plot_statistical_global_f1_grid.py
python research/training/scripts/analyze_modified_zscore_scale_mh5.py
python research/training/scripts/plot_isolation_forest_granularity_grid.py
python research/training/scripts/analyze_layered_finalists_cross_country.py
```

These scripts write PNG/SVG files to `results/media/` and write or refresh
supporting CSV/JSON/Markdown summaries under `results/analysis/`,
`results/tuning/`, and `results/detector_combinations/`.

## Figure-To-Input Map

| Figure | Script | Primary aggregate | Raw result roots |
| --- | --- | --- | --- |
| `weighted_score_grid` | `plot_statistical_global_f1_grid.py` | `results/analysis/statistical_detector_score_grid/weighted_score_grid.csv` | `results/tuning/statistical/1-6coarse_grid_single_attempt_batch/`, `results/tuning/statistical/no_avg_hybrid_weighted_single_attempt_batch/`, `results/tuning/statistical/z_score_global_single_attempt_batch/`, `results/tuning/statistical/z_score_by_country_single_attempt_batch/`, `results/tuning/statistical/z_score_by_competitor_single_attempt_batch/` |
| `modified_zscore_scale_mh5__all_methods` | `analyze_modified_zscore_scale_mh5.py` | `results/tuning/statistical/all_global_single_attempt_batch/mh5/analysis/modified_zscore_scale/modified_zscore_scale_trends.csv` | `results/tuning/statistical/all_global_single_attempt_batch/mh5/` |
| `if_granularity_score_grid` | `plot_isolation_forest_granularity_grid.py` | `results/analysis/isolation_forest_granularity_comparison/if_granularity_mh_summary.csv` | `results/tuning/forests/single_config_optimized_mh5_run/` |
| `layered_finalist_mh_trends` | `analyze_layered_finalists_cross_country.py` | `results/detector_combinations/all_competitors_all_countries_country_level_layered_finalists/analysis/thesis_metrics/layered_finalist_mh_summary.csv` | `results/detector_combinations/all_competitors_all_countries_country_level_layered_finalists/` |
| `layered_finalist_recall_mh_trends` | `analyze_layered_finalists_cross_country.py` | `results/detector_combinations/all_competitors_all_countries_country_level_layered_finalists/analysis/thesis_metrics/layered_finalist_mh_summary.csv` | `results/detector_combinations/all_competitors_all_countries_country_level_layered_finalists/` |

## Validation Routine

1. Read the relevant `data_references.md` file for the figure or table.
2. Confirm the listed raw result roots exist and contain the expected
   `summary.json`, `best_configuration.json`, `candidate_metrics.csv`, or
   thesis-metrics CSV files.
3. Rerun the corresponding aggregation or plotting script.
4. Compare regenerated CSV summaries and media files with the current working
   tree using `git diff` or a file comparison tool.
5. If a difference is expected, update the related `data_references.md` or
   `summary.md` so the new provenance is explicit.

If the validation task needs row-level predictions rather than aggregate
metrics, first regenerate the omitted parquet artifacts with the run-specific
script named in the relevant `results/**/README.md` file.

Full reruns of tuning or detector-combination experiments can be expensive.
When a result tree is missing, start from the script-specific documentation in
`research/training/scripts/README.md` rather than trying to infer command
arguments from downstream plotting scripts.
