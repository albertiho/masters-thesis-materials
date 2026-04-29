# Results Directory

This directory contains generated experiment outputs, thesis-facing summaries,
and publication media.

Large row-level artifacts were intentionally excluded from the GitHub release:

- `results/**/*.parquet`: approximately 50 GiB at the time of cleanup.
- `results/**/*.joblib`: approximately 2.2 GiB at the time of cleanup.
- `results/**/*.log`: operational logs.
- `results/**/_cache_snapshots/`: generated cache snapshots.

These files are not irrelevant. In particular, `predictions.parquet` and
`injected_rows.parquet` are row-level evaluation artifacts used to audit and,
for some workflows, rebuild aggregate metrics. They are omitted because normal
Git is a poor fit for tens of GiB of compressed binary artifacts.

The compact thesis-facing layer remains suitable for Git:

- `results/media/`
- `results/analysis/**`
- `results/**/summary.md`
- `results/**/summary.json`
- `results/**/data_references.md`
- aggregate CSV files such as detector metrics, candidate metrics, and thesis
  metric summaries.

To regenerate omitted row-level files, rerun the script named in the relevant
subdirectory README or in `REVALIDATE.md`. Full statistical sweeps and layered
detector runs can take tens of hours.
