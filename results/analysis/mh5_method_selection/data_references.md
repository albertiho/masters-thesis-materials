# mh5 Forest Method Selection Data References

## Thesis artifact references

| Thesis item | Source file | Presents data |
| --- | --- | --- |
| Table `forest_method_comparison_mh5` | `results/analysis/mh5_method_selection/detector_granularity_summary_mh5.csv` | Granularity-level mean \(F_{1,\mathrm{wc}}\), mean \(G_{\mathrm{wc}}\), scope counts, and mean training times for \(\mathrm{IF}\), \(\mathrm{EIF}\), and \(\mathrm{RRCF}\) at \(mh5\). |
| Text claim `if_dominance_over_forest_baselines_mh5` | `results/analysis/mh5_method_selection/if_dominance_summary_mh5.csv` | Direct granularity-level dominance check showing that \(\mathrm{IF}\) exceeds \(\mathrm{EIF}\) and \(\mathrm{RRCF}\) on both \(F_{1,\mathrm{wc}}\) and \(G_{\mathrm{wc}}\) while remaining faster. |

## Raw aggregation inputs

| Source path | Contributes data for | Presents data |
| --- | --- | --- |
| `results/tuning/forests/single_config_optimized_mh5_run/` | \(\mathrm{IF}\), \(\mathrm{EIF}\), \(\mathrm{RRCF}\) at \(mh5\) | Nested per-scope `summary.json` outputs used to aggregate the full-scope forest method comparison. |
| `results/tuning/statistical/z_score_granularity_comparison/analysis/granularity_performance_sampled_mh.csv` | `standard_zscore` mh5 comparator in the selection analysis | Default z-score mh5 aggregate used by the method-selection helper when the broader mh5 comparison includes the statistical baseline. |

## Notes

- The direct source for the thesis table in the first forest subsection is `detector_granularity_summary_mh5.csv`.
- The direct source for the dominance claims in the surrounding prose is `if_dominance_summary_mh5.csv`.
- The generated markdown summary `summary.md` is a readable synthesis of the same mh5 selection outputs, but the CSV files above are the authoritative thesis-table sources.
