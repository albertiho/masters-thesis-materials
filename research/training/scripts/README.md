# Training Scripts README

## `analyze_layered_finalists_cross_country.py`

Purpose:

- Aggregate the full cross-country layered-finalist run into thesis-facing Chapter 6 artifacts.
- Compare the four retained country-level layered finalists across all competitor scopes, sampled minimum-history settings, and the combined test case.
- Produce the compact overall comparison table, the `mh` trend figure, the per-`mh` winner-count support table, and the top-two anomaly-type comparison needed for the layered-results chapter.

Default inputs:

- `results/detector_combinations/all_competitors_all_countries_country_level_layered_finalists/analysis/thesis_metrics/layered_detector_metrics.csv`
- `results/detector_combinations/all_competitors_all_countries_country_level_layered_finalists/analysis/thesis_metrics/layered_detector_anomaly_case_metrics.csv`

What the script produces:

- `layered_finalist_overall_summary.csv`
  One row per finalist combination with valid scope-`mh` evaluation count and mean precision, recall, `F_{1,\mathrm{wc}}`, and `G_{\mathrm{wc}}`.
- `layered_finalist_mh_summary.csv`
  One row per `mh` setting and finalist combination with the mean precision, recall, `F_{1,\mathrm{wc}}`, and `G_{\mathrm{wc}}` values used in the Chapter 6 trend figures.
- `layered_finalist_mh_winner_counts.csv`
  Per-`mh` scope-win counts for `F_{1,\mathrm{wc}}` and `G_{\mathrm{wc}}`, used to support the minimum-history interpretation in Chapter 6.
- `layered_finalist_recall_stability_summary.csv`
  One row per finalist combination with overall recall mean, recall standard deviation, recall min/max, and cross-scope mean-recall stability.
- `layered_finalist_recall_winner_counts.csv`
  Total scope-`mh` recall wins for each finalist after tie-breaking by precision.
- `layered_finalist_top2_anomaly_case_summary.csv`
  One row per anomaly case for `Sanity -> Z-score` and `Sanity -> IF`, retained for the balanced-performance reading of the finalists.
- `layered_finalist_high_recall_anomaly_case_summary.csv`
  One row per anomaly case for `Sanity -> IF` and `Sanity -> Z-score (>=5) -> IF`, used in the recall-first Chapter 6 comparison.
- `layered_finalist_mh_trends.png`
  Thesis-facing raster figure written to `results/media/`, showing the sampled minimum-history trends of all four finalists for `F_{1,\mathrm{wc}}` and `G_{\mathrm{wc}}`, with separate metric and layer-configuration legends below the plot.
- `layered_finalist_mh_trends.svg`
  Vector version of the same figure, also written to `results/media/`.
- `layered_finalist_recall_mh_trends.png`
  Thesis-facing raster figure written to `results/media/`, showing the `mh` trends of all four finalists for recall and `F_{1,\mathrm{wc}}`.
- `layered_finalist_recall_mh_trends.svg`
  Vector version of the same recall-focused figure, also written to `results/media/`.
- `summary.md`
  Concise markdown summary of both the balanced-metric and recall-first layered-finalist rankings.
- `data_references.md`
  Provenance note mapping the Chapter 6 tables and figure to the generated aggregates and the underlying thesis-metrics CSVs.

When to use it:

- Use this script when the thesis needs the final cross-country layered-detector comparison after detector selection has already retained the sanity detector, the standard z-score, and the country-level Isolation Forest configuration.
- It is the right script for producing both the balanced-performance evidence and the recall-focused evidence behind the final Chapter 6 decision between `Sanity -> Z-score`, `Sanity -> IF`, and the two deeper three-stage cascades.

Example:

```powershell
python research/training/scripts/analyze_layered_finalists_cross_country.py
```

Optional custom output directory:

```powershell
python research/training/scripts/analyze_layered_finalists_cross_country.py `
  --output-dir results/detector_combinations/all_competitors_all_countries_country_level_layered_finalists/analysis/thesis_metrics
```

Notes:

- The script reads only the existing thesis-metrics CSVs and does not rerun the layered comparison.
- It uses `test_case_name == combined` as the thesis comparison surface for the Chapter 6 aggregates.
- It now emits two anomaly-type aggregates: one for the balanced top pair (`Sanity -> Z-score`, `Sanity -> IF`) and one for the recall-oriented high-sensitivity pair (`Sanity -> IF`, `Sanity -> Z-score (>=5) -> IF`).

## `replay_country_if_on_country4_competitor3_failed_cases.py`

Purpose:

- Replay the retained country-level Isolation Forest configuration for `COUNTRY_4` on the failed competitor-level case `COUNTRY_4 / B2C / COMPETITOR_3_COUNTRY_4_2026-02-08`.
- Evaluate both `new_prices` and `new_products` splits for the failed `mh10,mh15,mh20,mh25,mh30` cases with the same synthetic-injection metric path used by the forest sweep.
- Build thesis-facing evidence for the claim that a country-level IF configuration remains usable when a competitor-specific IF model cannot be trained because too few valid samples are available.

Default inputs:

- `results/tuning/forests/single_config_optimized_mh5_run/`
- `data-subsets/`
- `artifacts/models/` when a persisted country-level IF artifact exists locally

What the script produces:

- `split_metrics.csv`
  Per-split replay metrics for each failed `mh` case, including the original competitor-level failure reason, the country-level model source, matched competitor-history rows, and averaged `F_1` / `G` metrics from the repeated replay trials.
- `mh_summary.csv`
  One row per failed `mh` case with split-level metrics and weighted combined `F_1` / `G` when both splits are evaluable.
- `summary.md`
  Thesis-facing markdown summary of the replay outcome, including the concrete `mh10` result and the later empty-split cases.
- `summary.json`
  Machine-readable version of the replay summary.
- `data_references.md`
  Provenance note that maps the replay summary table to the generated CSV and the underlying sweep / dataset roots.

When to use it:

- Use this script when the thesis needs direct evidence about the failed `COUNTRY_4 / B2C / COMPETITOR_3` competitor case rather than the aggregate granularity comparison.
- It is the right script for the operational-selection argument that a retained country-level IF model can still be applied to a sparse competitor stream even when a competitor-level IF model cannot be trained for that `mh` setting.

Example:

```powershell
python research/training/scripts/replay_country_if_on_country4_competitor3_failed_cases.py
```

Notes:

- The script is intentionally a one-off helper. It has no CLI; edit the module-level constants if the target case ever needs to change.
- The script first tries to load the persisted country-level IF model from `artifacts/models/`. If it is not present locally, the script refits the retained country-level configuration from the saved forest sweep metadata before replaying it on the failed competitor case.
- In the current repository state, only `mh10` has non-empty competitor-level evaluation splits for the target case; `mh15` to `mh30` are still reported, but they end in `empty_test_split` / `no_eval_rows` because the corresponding competitor-level test subsets are empty after the minimum-history filter.

## `plot_isolation_forest_granularity_grid.py`

Purpose:

- Aggregate the retained Isolation Forest detector across the sampled `mh5,mh10,mh15,mh20,mh25,mh30` grid and the three thesis granularities.
- Read only the authoritative nested `if/summary.json` payloads under the consolidated forest root and ignore the stale root-level inventory files.
- Produce one thesis-facing figure for the subsection that compares Isolation Forest performance across granularity and retained history length after the forest-family pruning step.

Default inputs:

- `results/tuning/forests/single_config_optimized_mh5_run/`

What the script produces:

- `if_scope_status.csv`
  One row per discovered Isolation Forest summary, including the `ok` and `error` status cases.
- `if_scope_metrics.csv`
  One row per successful Isolation Forest scope summary with the retained threshold, combined scores, training time, and selected hyperparameters.
- `if_granularity_mh_summary.csv`
  Aggregate table of mean combined `F_1`, mean combined `G`, mean training time, and scope count for each `mh` level and granularity.
- `if_granularity_mean_summary.csv`
  Granularity-level mean summary across the sampled `mh` grid.
- `if_granularity_best_mh_summary.csv`
  Granularity-level best-`mh` summary for combined `F_1` and combined `G`, together with the corresponding mean scores and scope counts.
- `if_granularity_score_grid.png`
  Thesis-facing raster figure written to `results/media/`, with one full-width combined-score panel and a bottom legend row split into metric and granularity legends.
- `if_granularity_score_grid.svg`
  Vector version of the same figure, also written to `results/media/`.
- `data_references.md`
  Thesis-facing provenance note that maps the subsection figure and granularity-summary table to the generated aggregate CSV files and the underlying forest result root.
- `summary.md`
  Concise markdown summary of the aggregate results and the strongest observed granularity-by-`mh` operating points.
- `summary.json`
  Machine-readable version of the same high-level summary.

When to use it:

- Use this script when the thesis needs the second forest subsection, that is, the analysis of how the retained Isolation Forest detector behaves across `global`, `by_country`, and `by_competitor` aggregation as the sampled minimum-history level changes.
- It is the right script after the mh5 forest-family comparison has already justified retaining `if` and deprioritizing `eif` and `rrcf`.

Example:

```powershell
python research/training/scripts/plot_isolation_forest_granularity_grid.py
```

Optional custom output directory:

```powershell
python research/training/scripts/plot_isolation_forest_granularity_grid.py `
  --output-dir results/analysis/isolation_forest_granularity_comparison
```

Notes:

- The script reads only existing nested `if/summary.json` outputs; it does not rerun tuning.
- The consolidated forest root mixes several payloads, so the nested per-scope summaries are treated as authoritative and the root-level aggregate files are ignored.
- The known `COUNTRY_4/B2C/COMPETITOR_3_COUNTRY_4_2026-02-08` error cases at `mh10`-`mh30` are preserved in `if_scope_status.csv` and excluded from the score aggregates.

## `plot_statistical_global_f1_grid.py`

Purpose:

- Aggregate the sampled `mh5,mh10,mh15,mh20,mh25,mh30` statistical tuning results into one thesis-facing detector comparison figure.
- Combine the six retained statistical detectors: `standard_zscore`, `modified_mad`, `modified_sn`, `hybrid_avg`, `hybrid_max`, and `hybrid_weighted`.
- Plot average weighted combined `F_1` and weighted combined `G` across `global`, `by_country`, and `by_competitor` in a fixed two-column thesis layout whose final row is split into metric and granularity legends.

Default inputs:

- `results/tuning/statistical/1-6coarse_grid_single_attempt_batch/`
- `results/tuning/statistical/no_avg_hybrid_weighted_single_attempt_batch/`
- `results/tuning/statistical/z_score_global_single_attempt_batch/`
- `results/tuning/statistical/z_score_by_country_single_attempt_batch/`
- `results/tuning/statistical/z_score_by_competitor_single_attempt_batch/`

What the script produces:

- `weighted_score_grid.csv`
  Aggregate detector-by-granularity-by-`mh` table with average weighted combined `F_1`, average weighted combined `G`, and scope counts.
- `weighted_score_grid.png`
  Thesis-facing raster figure written to `results/media/`, with six detector panels, first-column y-axis labels, and a bottom legend row.
- `weighted_score_grid.svg`
  Vector version of the same figure, also written to `results/media/`.
- `data_references.md`
  Thesis-facing provenance note that maps the subsection figure and summary table to `weighted_score_grid.csv` and the underlying raw result roots.

When to use it:

- Use this script when the thesis needs one compact figure for subsection-level comparison of the retained z-score-based statistical methods across all three granularities.
- It is the right script for the chapter material that compares how the retained statistical detectors behave as the sampled minimum-history level changes.

Example:

```powershell
python research/training/scripts/plot_statistical_global_f1_grid.py
```

Optional custom output directory:

```powershell
python research/training/scripts/plot_statistical_global_f1_grid.py `
  --output-dir results/analysis/statistical_detector_score_grid
```

Notes:

- The script reads only existing `best_configuration.json` outputs and does not rerun tuning.
- Weighted combined `G` is calculated as `0.7 * new_prices_g_mean_mean + 0.3 * new_products_g_mean_mean`, matching the other statistical analysis scripts.
- The filename is historical; the current figure is no longer global-only and now aggregates `global`, `by_country`, and `by_competitor`.

## `analyze_modified_zscore_scale_mh5.py`

Purpose:

- Assess whether the original `1.0-3.0` modified-zscore threshold grid was too narrow at `mh5`.
- Aggregate the retained modified-zscore-family detectors across `global`, `by_country`, and `by_competitor` granularities.
- Produce one thesis-facing threshold-scale figure for the subsection that justifies extending the modified-zscore search range.

Default inputs:

- `results/tuning/statistical/all_global_single_attempt_batch/mh5/`

What the script produces:

- `modified_zscore_scale_trends.csv`
  Mean weighted combined `F_1` and weighted combined `G` by detector and threshold.
- `modified_zscore_scale_boundary_summary.csv`
  Boundary-selection counts and score deltas used to support the threshold-range interpretation.
- `modified_zscore_scale_mh5__all_methods.png`
  Thesis-facing raster figure written to `results/media/`, with five method panels and one embedded metric legend panel.
- `modified_zscore_scale_mh5__all_methods.svg`
  Vector version of the same figure, also written to `results/media/`.
- `modified_zscore_scale_summary.md`
  Concise markdown summary of the threshold-boundary evidence.
- `data_references.md`
  Thesis-facing provenance note that maps the subsection figure and boundary table to the generated aggregates and the underlying `mh5` tuning outputs.

When to use it:

- Use this script when the thesis needs direct evidence that the first modified-zscore-family tuning pass did not cover a sufficiently wide threshold range.

Example:

```powershell
python research/training/scripts/analyze_modified_zscore_scale_mh5.py
```

## `analyze_mh_subsampling_adequacy.py`

Purpose:

- Build a direct visual justification for replacing the full `mh5`-`mh30` grid with the sampled levels `mh5,mh10,mh15,mh20,mh25,mh30`.
- Aggregate the standard z-score `subset_guidance_by_case.csv` outputs across competitor-level, country-level, and global tuning.
- Summarize the two adequacy quantities cited in the methodology chapter: scope-level Spearman rank correlation and equal-support combined-F1 loss.

Default inputs:

- `results/tuning/statistical/z_score_by_competitor_single_attempt_batch/analysis/subset_guidance_by_case.csv`
- `results/tuning/statistical/z_score_by_country_single_attempt_batch/analysis/subset_guidance_by_case.csv`
- `results/tuning/statistical/z_score_global_single_attempt_batch/analysis/subset_guidance_by_case.csv`

What the script produces:

- `mh_subsampling_guidance_by_case.csv`
  Combined scope-level adequacy rows across all three granularities.
- `mh_subsampling_adequacy_summary.csv`
  Table-ready aggregate with scope count, mean Spearman correlation, median Spearman correlation, positive equal-support loss count, and maximum equal-support combined-F1 loss.
- `mh_subsampling_adequacy_summary.md`
  Markdown rendering of the same aggregate table.
- `mh_subsampling_adequacy_summary.json`
  Machine-readable version of the aggregate summary.
- `mh_subsampling_adequacy.png`
  Two-panel figure showing scope-level rank stability and equal-support combined-F1 loss by granularity.
- `mh_subsampling_adequacy.svg`
  Vector version of the same figure.

When to use it:

- Use this script when the question is not "how do z-score results vary over sampled `mh`?" but rather "does the sampled `mh` grid preserve the model-selection signal of the full grid well enough to justify subsampling?"
- It is the right script for the methodology subsection that reports the scope-level Spearman correlations and equal-support combined-F1 losses behind the minimum-history subsampling decision.

Example:

```powershell
python research/training/scripts/analyze_mh_subsampling_adequacy.py
```

Optional custom output directory:

```powershell
python research/training/scripts/analyze_mh_subsampling_adequacy.py `
  --output-dir results/tuning/statistical/z_score_mh_subsampling_adequacy/analysis
```

Notes:

- The script depends on the outputs of `analyze_statistical_guidance.py`; it does not rerun tuning.
- The default inputs are specific to the standard z-score detector because that detector was used to justify the sampled minimum-history grid in the methodology chapter.

## `analyze_mh5_method_selection.py`

Purpose:

- Build mh5-only evidence for method selection before running the expensive full `mh5`-`mh30` sweep.
- Quantify whether `eif` and `rrcf` are worth extending beyond `mh5`.
- Combine forest mh5 results with the existing default z-score granularity analysis so the thesis can justify selecting `standard_zscore` and `if` for the full run.

Default inputs:

- Forest results: `results/tuning/forests/single_config_optimized_mh5_run`
- Default z-score aggregate: `results/tuning/statistical/z_score_granularity_comparison/analysis/granularity_performance_sampled_mh.csv`

What the script produces:

- `forest_scope_metrics_mh5.csv`
  One row per successful forest scope summary at `mh5`.
- `forest_detector_granularity_summary_mh5.csv`
  Forest detector averages grouped by detector and granularity.
- `forest_detector_summary_mh5.csv`
  Forest detector averages grouped only by detector.
- `standard_zscore_granularity_summary_mh5.csv`
  Default z-score mh5 granularity rows copied into the same normalized shape.
- `detector_granularity_summary_mh5.csv`
  Combined mh5 summary across default z-score, Isolation Forest, EIF, and RRCF.
- `detector_summary_mh5.csv`
  Combined mh5 detector summary across methods.
- `if_dominance_summary_mh5.csv`
  Direct mh5 comparison showing whether Isolation Forest dominates EIF and RRCF on each granularity.
- `summary.md`
  Thesis-ready text summary of the mh5 selection evidence.
- `summary.json`
  Machine-readable summary, including the recommended full-suite methods.

When to use it:

- Use this script when the question is not "what is the best hyperparameter within one detector family?" but instead "which methods are good enough at `mh5` to justify the full expensive sweep?"
- It is the right script for arguing that `eif` and `rrcf` should not be extended to `mh10`-`mh30` if their mh5 accuracy is weak and their training cost is much higher than Isolation Forest.

Example:

```powershell
python research/training/scripts/analyze_mh5_method_selection.py
```

Optional custom output directory:

```powershell
python research/training/scripts/analyze_mh5_method_selection.py `
  --output-dir results/analysis/mh5_method_selection
```

Notes:

- The script depends on existing tuning outputs. It does not retrain models.
- The default z-score input is an aggregate produced by `analyze_granularity_performance.py`.
- The forest input is expected to come from the consolidated `single_config_optimized_mh5_run` directory, where the nested per-scope `summary.json` files are the authoritative source.
