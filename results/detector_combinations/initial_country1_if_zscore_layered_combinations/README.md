# Initial Country 1 Layered Combination Run

This is an earlier COUNTRY_1 layered IF/Z-score/sanity evaluation. The full
local run produced about 0.97 GiB of row-level parquet artifacts:

```text
splits/*/injected_rows.parquet
splits/*/predictions.parquet
```

Those files are omitted from the normal GitHub checkout by `.gitignore`. The
current comparable runner is:

```powershell
python research/training/scripts/analyze_if_zscore_layered_combinations.py
```

The included `metrics/`, `summary.md`, `summary.json`, and `run_metadata.json`
files keep the compact result record. Regenerating row-level artifacts may take
many hours.
