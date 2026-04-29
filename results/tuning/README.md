# Tuning Results

This tree contains the statistical and forest tuning outputs used in the
thesis. The Git checkout includes compact CSV/JSON/Markdown summaries and
omits the largest binary artifacts from the full local experiment workspace.

Omitted artifact families:

- Statistical detector row-level split artifacts under
  `results/tuning/statistical/**/best_candidate/splits/**/`.
- Forest cache snapshots under
  `results/tuning/forests/**/_cache_snapshots/`.
- Generated `.joblib` model and cache payloads.

Approximate omitted size:

- `results/tuning/statistical`: 40.45 GiB of parquet row-level artifacts.
- `results/tuning/forests`: 2.17 GiB of joblib cache artifacts.

The included summaries are the inputs for the thesis plots and tables. Regenerate
the omitted files only when row-level audit data or cache payloads are needed.
The corresponding tuning scripts can take many hours.
