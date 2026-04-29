# Results Directory

This directory contains the generated outputs used by the thesis: aggregate
metrics, provenance notes, analysis summaries, and publication media.

The full local experiment workspace also produced large row-level artifacts.
They are part of the reproducibility trail, but they are not included in the
normal GitHub checkout because they are too large for ordinary Git storage:

- `results/**/*.parquet`: approximately 50 GiB of row-level split artifacts.
- `results/**/*.joblib`: approximately 2.2 GiB of model and cache payloads.
- `results/**/*.log`: run logs.
- `results/**/_cache_snapshots/`: generated cache snapshots.

The most important omitted parquet files are named `predictions.parquet` and
`injected_rows.parquet`. They store per-row detector outputs and injected-label
metadata. Some audit and metric-regeneration workflows read them directly, while
the thesis figures in this repository are based on the compact CSV/JSON
summaries derived from them.

Files intended to stay in the Git checkout include:

- `results/media/`
- `results/analysis/**`
- `results/**/summary.md`
- `results/**/summary.json`
- `results/**/data_references.md`
- aggregate CSV files such as detector metrics, candidate metrics, and thesis
  metric summaries.

To regenerate omitted row-level files, use the script named in the relevant
subdirectory README or in `REVALIDATE.md`. Full statistical sweeps and layered
detector runs can take tens of hours.
