# Forest Tuning Results

This folder previously contained about 2.17 GiB of generated `.joblib` cache
snapshots, mostly files named:

```text
_cache_snapshots/**/template_cache.joblib
```

They are excluded from the GitHub release by `.gitignore`.

The snapshots are generated while running forest tuning:

```powershell
python research/training/scripts/tune_forests.py
```

The cache snapshots speed up repeated evaluations, but they are derived from
the dataset splits and can be rebuilt. The thesis-facing Isolation Forest
granularity figure uses compact summaries under this tree, especially nested
`if/summary.json` files consumed by:

```powershell
python research/training/scripts/plot_isolation_forest_granularity_grid.py
```

Regenerating the omitted forest caches and tuning outputs can take many hours.
