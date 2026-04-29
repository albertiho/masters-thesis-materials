# Forest Tuning Results

This folder contains compact outputs from the forest tuning runs. The full local
workspace also produced about 2.17 GiB of generated `.joblib` cache snapshots,
mostly files named:

```text
_cache_snapshots/**/template_cache.joblib
```

These files are omitted from the normal GitHub checkout by `.gitignore`.

The snapshots are generated while running forest tuning:

```powershell
python research/training/scripts/tune_forests.py
```

The cache snapshots speed up repeated evaluations and can be rebuilt from the
dataset splits. The thesis-facing Isolation Forest granularity figure uses
compact summaries under this tree, especially nested `if/summary.json` files
consumed by:

```powershell
python research/training/scripts/plot_isolation_forest_granularity_grid.py
```

Regenerating the omitted forest caches and tuning outputs can take many hours.
