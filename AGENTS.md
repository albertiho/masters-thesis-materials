# Repository Guide

This repository is a standalone reproduction package for a master's thesis on
anomaly detection in competitor-pricing data. It contains detector
implementations, data preparation utilities, training and tuning scripts,
evaluation utilities, and generated result artifacts used for thesis
figures and tables.

Run commands from the repository root unless a script says otherwise.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pytest
```

The project targets Python 3.11+ and pins dependencies for reproducibility.
`requirements-dev.txt` includes the base, ML, lint, type-check, and test
dependencies.

Useful checks:

```powershell
pytest
ruff check src tests research/training/scripts
black --check src tests research/training/scripts
```

## Folder Map

- `src/`: importable Python package. Prefer changing reusable logic here before
  adding behavior directly to scripts.
- `src/anomaly/`: detector interfaces and implementations.
  - `base.py`: base detector contract and score-normalization convention.
  - `statistical.py`: z-score, IQR, threshold, and sanity-check detectors.
  - `z_score_methods.py`: standard, MAD, Sn, and hybrid z-score methods from
    the thesis literature review.
  - `combined.py` and `combined_variants.py`: layered detector pipelines,
    gates, weighted voting, and history-based routing.
  - `ml/`: Isolation Forest, Extended Isolation Forest, RRCF, autoencoder, and
    shared tree-feature extraction.
- `src/features/`: feature extraction and temporal state. `temporal.py` owns
  the rolling cache behavior used by evaluators.
- `src/research/`: reusable research workflow helpers.
  - `datasets.py`: split-manifest dataset resolver used by local evaluation
    helpers.
  - `prepare_data.py`: deterministic bundled-data preparation.
  - `history_subsets.py` and `mh_sampling.py`: minimum-history dataset support.
  - `artifacts.py`: canonical evaluation artifact schemas and writers.
  - `evaluation/`: synthetic anomaly injection, evaluator orchestration, and
    detector evaluation logic.
- `src/ingestion/`, `src/quality/`, `src/utils/`: parsing, health checks, and
  small shared utilities.
- `research/training/scripts/`: script entrypoints for thesis-scale work. This
  is where most training, tuning, replay, aggregation, and plotting commands
  live. See `research/training/scripts/README.md` for script-specific purpose,
  inputs, outputs, and examples.
- `configs/`: tuning defaults. `configs/tuning_config.json` is read by
  `src/tuning_config.py`.
- `tests/`: pytest regression suite covering detectors, data preparation,
  synthetic injection, artifact schemas, and tuning sweeps.
- `data/training/source/`: bundled source parquet data plus manifest/checksums.
- `data/training/derived/`: generated deterministic derived files for the
  bundled-data path. This path is ignored by Git.
- `data-subsets/`: generated thesis-scale minimum-history datasets. Usually
  large and ignored by Git.
- `results/`: generated experiment outputs, summaries, CSV/JSON metrics, and
  analysis products. `results/media/` contains thesis-facing PNG/SVG figures.
- `artifacts/`: model artifacts and feature/cache outputs. Usually generated;
  inspect only when a task is about cached models or expensive sweep reuse.
- `REVALIDATE.md`: manual guide for tracing and regenerating thesis-facing
  figures and tables.

## Main Workflows

### Bundled Data Preparation

Use this for the smaller committed-data reproduction path:

```powershell
python -m src.research.prepare_data --data-root data/training
```

It validates `data/training/source/`, writes derived parquet files under
`data/training/derived/`, and emits a split manifest.

### Thesis-Scale Data Preparation

Use this for local full-scale thesis experiments:

```powershell
python research/training/scripts/split_training_data.py
```

The script expects cleaned source files at
`../cleaned-data/training/by_competitor/`. It creates `mh5`, `mh10`, `mh15`,
`mh20`, `mh25`, and `mh30` variants under `data-subsets/`, plus global,
country, and competitor aggregations with train/test split parquet files.

### Manual Revalidation

Revalidation is provenance-guided, not a separate CLI workflow. Use
`REVALIDATE.md` to trace each thesis-facing figure/table from `results/media/`
or `results/analysis/` to its aggregate CSV/JSON files and raw result roots,
then rerun the corresponding script in `research/training/scripts/`.

### Training, Tuning, and Analysis Scripts

Most thesis scripts are under `research/training/scripts/` and are intended to
be run directly with Python. Common families:

- `train_*.py`: train specific detector artifacts.
- `tune_*.py` and `grid_search_*.py`: parameter sweeps.
- `validate_*_mh5_smoke.py`: smoke validation for tree detectors.
- `analyze_*.py`, `compare_*.py`, `plot_*.py`: aggregate existing results and
  generate thesis tables/figures.
- `run_all_competitors_country_level_layered_finalists.py`: final layered
  detector comparison across competitor scopes.

The scripts generally read existing parquet/result trees and write under
`results/`; many do not retrain unless their name or README section says so.

## Implementation Notes

- Detector scores should follow the `BaseDetector.normalize_score` convention:
  0 is normal, 0.5 is at threshold, and 1 is strongly anomalous.
- Single-row detector outputs use `AnomalyResult` from `src/anomaly/statistical.py`.
  Keep `competitor_product_id`, `competitor`, detector name, score, anomaly
  types, and details populated because artifact writers depend on them.
- `DetectorEvaluator` gives each detector an isolated temporal cache and only
  updates that cache with non-anomalous prices. Preserve this no-look-ahead
  behavior when changing evaluators or batch paths.
- Batch-capable detectors should keep single-row and batch behavior equivalent;
  tests assert this for history-dependent statistical detectors.
- Canonical prediction and injected-row schemas live in `src/research/artifacts.py`.
  Update schema-aware tests when adding artifact columns.
- Use absolute imports from `src...` in project code. `tests/conftest.py` adds
  the repository root to `sys.path`.
- Formatting is Black/Ruff with 100-character lines. Ruff is configured in
  `pyproject.toml`; it selects docstring, annotation, pyupgrade, bugbear, and
  simplification rules.

## Testing Pointers

Run the full suite before broad changes:

```powershell
pytest
```

Targeted suites by area:

```powershell
pytest tests/test_z_score_methods.py tests/test_statistical_zscore_variants.py
pytest tests/test_tree_ml_detectors.py tests/test_isolation_forest_training_and_tuning.py
pytest tests/test_phase1_prepare_data.py tests/test_split_training_data.py tests/test_history_subsets.py
pytest tests/test_phase2_artifacts.py tests/test_synthetic_injection.py tests/test_evaluation_cache_behavior.py
pytest tests/test_tune_statistical_sweep.py tests/test_tune_forests_sweep.py
```

## Practical Cautions

- `results/`, `data-subsets/`, `data/training/derived/`, and `artifacts/cache/`
  can be very large. Search them only when a task specifically needs generated
  outputs.
- Do not delete or regenerate large result trees casually; many analysis scripts
  consume existing CSV/JSON summaries rather than rerunning expensive sweeps.
- Generated outputs often include provenance files named `data_references.md`.
  Use them to trace thesis figures and tables back to source CSV/JSON inputs.
- There is no manifest-driven revalidation command in the current workflow;
  `REVALIDATE.md` is the source of truth for manual review steps.
- Some source paths contain anonymized identifiers such as `COUNTRY_1`,
  `COMPETITOR_2`, and minimum-history labels like `mh5`; keep those naming
  conventions stable because scripts parse them.
