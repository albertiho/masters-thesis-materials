# Statistical Detector Score Grid Data References

## Thesis artifact references

| Thesis item | Source file | Presents data |
| --- | --- | --- |
| Figure `statistical_detector_score_grid` (`results/media/weighted_score_grid.png` / `results/media/weighted_score_grid.svg`) | `results/analysis/statistical_detector_score_grid/weighted_score_grid.csv` | Mean weighted combined `F_{1,\mathrm{wc}}` and `G_{\mathrm{wc}}` for each retained statistical detector under `global`, `by_country`, and `by_competitor` aggregation at `mh5`, `mh10`, `mh15`, `mh20`, `mh25`, and `mh30`. |
| Table `statistical_performance_comparison_summary` | `results/analysis/statistical_detector_score_grid/weighted_score_grid.csv` | Granularity-level summary in which the mean z-score performance across the tested `mh` range is compared with the strongest non-z-score alternative, together with the corresponding outperformance deltas. |

## Raw aggregation inputs

| Source path | Contributes data for | Presents data |
| --- | --- | --- |
| `results/tuning/statistical/1-6coarse_grid_single_attempt_batch/` | `modified_mad`, `modified_sn`, `hybrid_avg`, `hybrid_max` | Per-scope `best_configuration.json` outputs for the coarse sampled `mh` sweep across the retained statistical detector families. |
| `results/tuning/statistical/no_avg_hybrid_weighted_single_attempt_batch/` | `hybrid_weighted` | Per-scope `best_configuration.json` outputs for the weighted hybrid detector across the sampled `mh` sweep. |
| `results/tuning/statistical/z_score_global_single_attempt_batch/` | `standard_zscore` under `global` aggregation | Per-scope `best_configuration.json` outputs for the standard z-score detector at global granularity. |
| `results/tuning/statistical/z_score_by_country_single_attempt_batch/` | `standard_zscore` under `by_country` aggregation | Per-scope `best_configuration.json` outputs for the standard z-score detector at country-level granularity. |
| `results/tuning/statistical/z_score_by_competitor_single_attempt_batch/` | `standard_zscore` under `by_competitor` aggregation | Per-scope `best_configuration.json` outputs for the standard z-score detector at competitor-level granularity. |

## Notes

- The direct source for thesis tables and figures in this subsection is `weighted_score_grid.csv`; the plotting script aggregates that file from the raw per-scope tuning outputs listed above.
- The thesis-facing figure uses six detector panels in a two-column layout, first-column y-axis labels, and a final legend row split into metric and granularity legends.
- The by-competitor rows in `weighted_score_grid.csv` are based on `12` scopes at `mh5` and `mh10`, and on `11` scopes at `mh15` to `mh30`, because one scope does not complete at the longer retained-history settings.
