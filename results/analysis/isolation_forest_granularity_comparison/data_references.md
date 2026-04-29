# Isolation Forest Granularity Comparison Data References

## Thesis artifact references

| Thesis item | Source file | Presents data |
| --- | --- | --- |
| Figure `if_granularity_score_grid` (`results/media/if_granularity_score_grid.png` / `results/media/if_granularity_score_grid.svg`) | `results/analysis/isolation_forest_granularity_comparison/if_granularity_mh_summary.csv` | Granularity-level mean \(F_{1,\mathrm{wc}}\) and mean \(G_{\mathrm{wc}}\) of the retained \(\mathrm{IF}\) detector across \(mh5\), \(mh10\), \(mh15\), \(mh20\), \(mh25\), and \(mh30\). |
| Table `if_granularity_summary` | `results/analysis/isolation_forest_granularity_comparison/if_granularity_best_mh_summary.csv` | Granularity-level mean \(F_{1,\mathrm{wc}}\), mean \(G_{\mathrm{wc}}\), best sampled \(mh\) for each metric, best observed values, and mean training times for the retained \(\mathrm{IF}\) detector. |

## Raw aggregation inputs

| Source path | Contributes data for | Presents data |
| --- | --- | --- |
| `results/tuning/forests/single_config_optimized_mh5_run/` | Retained \(\mathrm{IF}\) detector across all sampled `mh` levels and granularities | Nested per-scope `if/summary.json` outputs used to aggregate the retained-model comparison after the forest-family pruning step. |

## Notes

- The direct source for the second forest-subsection figure is `if_granularity_mh_summary.csv`.
- The thesis-facing figure uses one full-width score panel with the metric and granularity legends placed below the plot.
- The direct source for the granularity-selection table is `if_granularity_best_mh_summary.csv`.
- The by-competitor series is based on `12` successful scopes at `mh5` and on `11` successful scopes at `mh10` to `mh30`; the missing scope is recorded in `if_scope_status.csv`.
