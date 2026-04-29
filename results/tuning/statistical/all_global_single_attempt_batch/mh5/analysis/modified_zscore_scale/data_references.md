# Modified Z-score Scale Data References

## Thesis artifact references

| Thesis item | Source file | Presents data |
| --- | --- | --- |
| Figure `modified_zscore_scale_mh5__all_methods` (`results/media/modified_zscore_scale_mh5__all_methods.png` / `results/media/modified_zscore_scale_mh5__all_methods.svg`) | `results/tuning/statistical/all_global_single_attempt_batch/mh5/analysis/modified_zscore_scale/modified_zscore_scale_trends.csv` | Mean weighted combined `F_{1,\mathrm{wc}}` and `G_{\mathrm{wc}}` trends over the original modified-zscore threshold range for the five retained modified-zscore-family methods at `mh5`. |
| Table `modified_zscore_scale_boundary_summary` | `results/tuning/statistical/all_global_single_attempt_batch/mh5/analysis/modified_zscore_scale/modified_zscore_scale_boundary_summary.csv` | Boundary-selection counts and coarse-grid score deltas used to justify extending the modified-zscore threshold range beyond the original upper bound. |

## Raw aggregation inputs

| Source path | Contributes data for | Presents data |
| --- | --- | --- |
| `C:\Users\Administrator\Desktop\dippa\src\results\tuning\statistical\all_global_single_attempt_batch\mh5` | Modified-zscore-family `mh5` tuning across `global`, `by_country`, and `by_competitor` granularities | Per-scope `candidate_metrics.csv` and `best_configuration.json` outputs used to assess whether the original `1.0-3.0` threshold grid was too narrow. |

## Notes

- The thesis-facing figure files are written to `results/media/modified_zscore_scale_mh5__all_methods.png` and `results/media/modified_zscore_scale_mh5__all_methods.svg`.
- The figure uses a two-column by three-row layout: five method panels and one embedded metric legend panel.
