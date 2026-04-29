# Tuning Results

The tuning result tree keeps compact summaries in Git and excludes large
generated binary artifacts.

Omitted artifact families:

- Statistical detector row-level split artifacts under
  `results/tuning/statistical/**/best_candidate/splits/**/`.
- Forest cache snapshots under
  `results/tuning/forests/**/_cache_snapshots/`.
- Any generated `.joblib` model/cache payloads.

Approximate omitted size at cleanup time:

- `results/tuning/statistical`: 40.45 GiB of parquet row-level artifacts.
- `results/tuning/forests`: 2.17 GiB of joblib cache artifacts.

The retained CSV/JSON/Markdown summaries are the source for thesis plots and
tables. Regenerating the omitted files requires rerunning the relevant tuning
scripts and can take many hours.
