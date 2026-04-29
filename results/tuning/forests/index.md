# Forest Tuning Runs

This directory contains the forest-based tuning outputs produced between 2026-03-28 and 2026-04-01. The folders are not equivalent snapshots of one pipeline stage. Some are exhaustive pilot sweeps, some are screened follow-up runs, and `single_config_optimized_mh5_run` is now a consolidated container that mixes artifacts from several later additions.

## How to read the folders

- `summary.json` / `summary.md`: per-scope detector summary with the winning candidate and its best threshold.
- `best_configuration.json`: selected winning configuration for one scope and one detector.
- `candidate_metrics.csv`: evaluated candidate grid and threshold sweep for that scope-detector pair.
- `best_configurations.csv`: top-level aggregate of winners, when the run wrote one.
- `scope_status.csv`: top-level bookkeeping for completed scopes.
- `_cache_snapshots/`: cached feature/history state used to speed up reruns; these are infrastructure artifacts rather than analysis outputs.

## Run Inventory

- [`mh5_full_hypergrid_20260328/`](./mh5_full_hypergrid_20260328): exhaustive pilot sweep created on 2026-03-28. Coverage is limited to `mh5`, `by_competitor`, and two pilot scopes: `COUNTRY_1/B2B/COMPETITOR_1_COUNTRY_1_2026-02-08` and `COUNTRY_1/B2C/COMPETITOR_2_COUNTRY_1_2026-02-08`. Detector coverage is intentionally uneven: EIF appears on one scope with 12 configuration families and 132 thresholded candidates, IF on two scopes with 24 configuration families and 264 candidates per scope, and RRCF on one scope with 18 configuration families and 198 candidates. Use this run when the full mh5 pilot search space is needed for provenance, appendix material, or manual inspection of candidate trade-offs. Do not use it as the main benchmark set, because its scope coverage is deliberately narrow and there is no directory-level aggregate summary.

- [`optimized_mh5_run/`](./optimized_mh5_run): screened follow-up pilot run created later on 2026-03-28. It covers the same two `mh5` `by_competitor` pilot scopes as the full hypergrid, but only after configuration screening and promotion. The retained search space is compact: EIF evaluates 1 promoted configuration over 11 thresholds after screening 12 families, IF evaluates 3 promoted configurations over 33 candidates after screening 12 families, and RRCF evaluates 1 promoted configuration over 11 thresholds after screening 9 families. Use this run when the goal is to inspect the promoted mh5 pilot winners without carrying the full hypergrid. It is the compact counterpart to `mh5_full_hypergrid_20260328`, not a replacement for broader benchmarking.

- [`single_config_optimized_mh5_run/`](./single_config_optimized_mh5_run): consolidated evaluation folder. It now contains three distinct payloads:
  1. The original `rrcf` `mh5` run across all 17 scopes.
  2. The full-scope fixed-profile EIF replay that was originally written under sweep id `eif_mh5_granularity_fixed_gmean_winner`.
  3. The IF cross-horizon reuse runs across `mh5`, `mh10`, `mh15`, `mh20`, `mh25`, and `mh30`.

  Successful nested artifacts cover 17 scopes overall: 12 `by_competitor`, 4 `by_country`, and 1 `global`. Within those nested summaries, EIF now has complete `mh5` coverage across all 17 scopes with one fixed profile, `eif__max_features_0p5__max_samples_256__n_estimators_100` at threshold `0.2`. RRCF has complete `mh5` coverage across all 17 scopes with one reused configuration, `rrcf__num_trees_80__tree_size_128__warmup_samples_32`. IF is the only detector family that extends beyond `mh5`; it contributes 102 nested summaries across six history lengths, of which 97 are successful and 5 fail for `COUNTRY_4/B2C/COMPETITOR_3_COUNTRY_4_2026-02-08` because too few valid samples are available at `mh10`-`mh30`.

  Use this folder when the thesis needs the consolidated forest evidence base: the full-scope EIF `mh5` benchmark, the full-scope RRCF `mh5` benchmark, and the IF cross-horizon results. Do not rely on the root-level `summary.json`, `summary.md`, `best_configurations.csv`, or `scope_status.csv` as the authoritative inventory. Those root files still describe only the original `rrcf` `mh5` slice, with 17 completed tasks, and do not enumerate the moved EIF artifacts or the later IF additions. The nested per-scope `summary.json` files are the authoritative source.

- [`eif_mh5_competitor_1_country_1/`](./eif_mh5_competitor_1_country_1): one-scope EIF calibration run created on 2026-03-30 and completed on 2026-03-31. Coverage is restricted to `mh5`, `by_competitor`, and `COUNTRY_1/B2B/COMPETITOR_1_COUNTRY_1_2026-02-08`. The run uses a `two_pass_refinement` search stage, evaluates 12 configuration families and 36 thresholded candidates, and selects `eif__max_features_0p5__max_samples_256__n_estimators_100` at threshold `0.2` as the best candidate. Use this run when the thesis needs the calibrated one-scope EIF search that motivated the later fixed-profile replay now stored inside `single_config_optimized_mh5_run`.

## Which Run Should Be Used For What

- Use [`mh5_full_hypergrid_20260328/`](./mh5_full_hypergrid_20260328) when the thesis needs the raw exhaustive mh5 pilot search space.
- Use [`optimized_mh5_run/`](./optimized_mh5_run) when the thesis needs the promoted mh5 pilot winners without the full candidate grid.
- Use [`single_config_optimized_mh5_run/`](./single_config_optimized_mh5_run) when the thesis needs the main forest comparison base: full-scope EIF `mh5`, full-scope RRCF `mh5`, and IF results across `mh5`-`mh30`.
- Use [`eif_mh5_competitor_1_country_1/`](./eif_mh5_competitor_1_country_1) when the thesis needs the focused EIF calibration run that identifies the fixed profile later replayed across all scopes.

## Caveats

- `single_config_optimized_mh5_run` is now a consolidation folder rather than a single self-describing run.
- The moved EIF summaries inside `single_config_optimized_mh5_run` still retain their original sweep id, `eif_mh5_granularity_fixed_gmean_winner`, inside the nested JSON payloads.
- The root aggregate files in `single_config_optimized_mh5_run` are stale with respect to the current contents of the folder and should not be treated as the canonical inventory.
- `mh5_full_hypergrid_20260328` and `optimized_mh5_run` do not provide a directory-level aggregate summary, so any cross-scope aggregation has to be rebuilt from the nested detector summaries.
- IF is the only detector family in this directory with successful artifacts across multiple history lengths beyond `mh5`.
