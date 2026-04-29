# Initial Country 1 Layered Combination Run

This historical run previously included about 0.97 GiB of row-level parquet
artifacts:

```text
splits/*/injected_rows.parquet
splits/*/predictions.parquet
```

They are excluded from the GitHub release by `.gitignore`.

This was an earlier COUNTRY_1 layered IF/Z-score/sanity evaluation. The current
comparable runner is:

```powershell
python research/training/scripts/analyze_if_zscore_layered_combinations.py
```

The committed `metrics/`, `summary.md`, `summary.json`, and `run_metadata.json`
files keep the compact result record. Regenerating row-level artifacts may take
many hours.
