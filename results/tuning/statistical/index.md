# Statistical Tuning Runs

This directory contains the statistical-detector tuning batches plus one post-hoc comparison folder. In the completed sweeps, the authoritative per-scope result is the local `best_configuration.json`; the summaries below are based on those stored best configurations and on the top-level `summary.md` / `scope_status.csv` files where they exist.

## Folder Guide

- `summary.md` and `summary.json`: top-level batch summary, when the run wrote one.
- `scope_status.csv`: scope-level completion log with complete / error / skipped status.
- `<mh>/<granularity>/<scope>/<detector>/candidate_metrics.csv`: full candidate grid for one detector family.
- `<mh>/<granularity>/<scope>/<detector>/best_configuration.json`: selected configuration and its combined metrics.
- `<mh>/<granularity>/<scope>/<detector>/best_candidate/`: split-level metrics, predictions, injected rows, and run metadata for the selected configuration.
- `analysis/`: post-hoc plots or comparison tables rather than raw sweep output.

## Run Map

### `all_global_single_attempt_batch`

Folder: [all_global_single_attempt_batch](./all_global_single_attempt_batch/)

- Coverage is partial: `mh5` covers `global`, `by_country`, and `by_competitor`, whereas `mh10` only covers eight `by_competitor` scopes.
- Stored detector families are `threshold`, `iqr`, `modified_mad`, `modified_sn`, `hybrid_max`, `hybrid_avg`, and `hybrid_weighted`, although `mh5/global` does not contain `hybrid_weighted`.
- For thesis-facing interpretation, ignore `threshold` and `iqr` and treat this run as evidence only about the modified / hybrid families.
- In direct matched-scope comparisons against the corresponding `standard_zscore` sweeps at the same `mh`, granularity, and scope, the modified / hybrid families are uniformly weak: `modified_mad`, `modified_sn`, `hybrid_max`, `hybrid_avg`, and `hybrid_weighted` lose every shared comparison that exists in the stored best configurations.
- On those shared scope pairs, the modified / hybrid families reach mean best weighted F1 values of only `0.372` to `0.382`, whereas the matched `standard_zscore` runs are at `0.607` to `0.609`; the average deficit is `0.227` to `0.241` weighted-F1 points, and the best-run training times are usually `1.15x` to `1.66x` higher.
- Use this run for early triage of the modified / hybrid families only.
- Use this run in the thesis as negative evidence for pruning the modified / hybrid families from any further exhaustive grid search; the stored results do support the claim that spending more tuning budget on those families is difficult to justify relative to `standard_zscore`.
- Ignore `threshold` and `iqr` in the thesis dataset plan unless a narrow implementation note explicitly requires mentioning them.
- Do not use this run as the main benchmark table for the thesis, because the coverage is incomplete and uneven across `mh` levels and scopes.

### `1-6coarse_grid_single_attempt_batch`

Folder: [summary](./1-6coarse_grid_single_attempt_batch/summary.md)

- Coverage spans `mh5`, `mh10`, `mh15`, `mh20`, `mh25`, and `mh30` across `global`, `by_country`, and `by_competitor`.
- The batch summary reports 98 complete scopes, 4 error scopes, and 0 skipped scopes.
- The intended detector families, according to the batch summary, are `modified_mad`, `modified_sn`, `hybrid_max`, and `hybrid_avg`.
- The stored artifacts also contain 45 `hybrid_weighted` best configurations; these do not match the top-level run definition and should be treated as extra carry-over output rather than the canonical contents of the sweep.
- Across the intended four families, the mean best weighted F1 values are effectively tied; `modified_mad` is the fastest of the tied group, while `hybrid_avg` is marginally highest on average.
- Use this run for coarse screening of the robust / hybrid detector families over a sampled set of `mh` levels.
- Do not use it as the final word on `hybrid_weighted`; there is a dedicated rerun for that detector.

### `no_avg_hybrid_weighted_single_attempt_batch`

Folder: [summary](./no_avg_hybrid_weighted_single_attempt_batch/summary.md)

- Coverage matches the sampled matrix used in `1-6coarse_grid_single_attempt_batch`: `mh5`, `mh10`, `mh15`, `mh20`, `mh25`, and `mh30` across all three granularities.
- The batch summary reports 98 complete scopes, 4 error scopes, and 0 skipped scopes.
- The run contains only `hybrid_weighted`, which makes it the clean standalone reference for that detector family.
- In the stored best configurations, the strongest average `mh` is `mh30`.
- Use this run when the question is specifically how `hybrid_weighted` behaves on its own, without mixing it into the four-family coarse-grid batch.
- Do not use it for broad detector-family comparison, because it answers only one detector-family question.

### `z_score_by_competitor_single_attempt_batch`

Folder: [summary](./z_score_by_competitor_single_attempt_batch/summary.md)

- Coverage spans `mh5` through `mh30` at `by_competitor` granularity.
- The batch summary reports 292 complete scopes, 20 error scopes, and 0 skipped scopes.
- This is the most local `standard_zscore` sweep and the option favored by the sampled granularity comparison at most tested `mh` values.
- The strongest average `mh` in the stored best configurations is `mh8`.
- Use this run for final `standard_zscore` tuning when competitor-specific sensitivity matters more than having a perfectly complete sweep matrix.
- Caveat: every recorded error belongs to `COUNTRY_4/B2C/COMPETITOR_3_COUNTRY_4_2026-02-08`, failing from `mh11` onward. That looks like a scope-specific pathology rather than a general failure of `standard_zscore`.

### `z_score_by_country_single_attempt_batch`

Folder: [summary](./z_score_by_country_single_attempt_batch/summary.md)

- Coverage spans `mh5` through `mh30` at `by_country` granularity.
- The batch summary reports 104 complete scopes, 0 errors, and 0 skipped scopes.
- This is the cleanest fully completed `standard_zscore` sweep.
- The strongest average `mh` in the stored best configurations is `mh11`.
- Use this run when the goal is a stable and fully complete `standard_zscore` benchmark with country-level locality.
- Relative to the competitor-level run, this is the safer coverage-first choice; across the full stored best configurations, it also has the slightly highest average weighted F1 of the three z-score sweeps.

### `z_score_global_single_attempt_batch`

Folder: [summary](./z_score_global_single_attempt_batch/summary.md)

- Coverage spans `mh5` through `mh30` at `global` granularity.
- The batch summary reports 26 complete scopes, 0 errors, and 0 skipped scopes.
- This is the simplest `standard_zscore` baseline because it searches one global population per `mh`.
- The strongest average `mh` in the stored best configurations is `mh8`.
- Use this run when one global configuration is preferable to country- or competitor-specific settings, or when a simple baseline is needed for comparison.
- Do not use it as the default final choice if local sensitivity matters; the more local z-score runs generally perform better on weighted F1.

### `z_score_granularity_comparison`

Folder: [analysis](./z_score_granularity_comparison/analysis/granularity_performance_sampled_mh.md)

- This is an analysis folder, not a tuning sweep.
- It compares the three `standard_zscore` runs at sampled `mh` values: `mh5`, `mh10`, `mh15`, `mh20`, `mh25`, and `mh30`.
- The main takeaway is that `by_competitor` is the best default when weighted F1 is the priority, `by_country` is the clean fully completed alternative, and `global` is primarily the simplicity baseline.
- Use this folder first when choosing z-score granularity; then open the corresponding underlying sweep for the actual z-score cutoff and `mh` selection.

## Recommended Use

- Use `all_global_single_attempt_batch` for early modified / hybrid family triage only; ignore its `threshold` and `iqr` outputs.
- Use `1-6coarse_grid_single_attempt_batch` to compare the non-z-score robust / hybrid families on a sampled `mh` grid.
- Use `no_avg_hybrid_weighted_single_attempt_batch` when you specifically need the standalone `hybrid_weighted` reference.
- Use `z_score_granularity_comparison` to choose the z-score granularity before reading any one z-score sweep in detail.
- Use `z_score_by_competitor_single_attempt_batch` for the strongest competitor-local z-score search, `z_score_by_country_single_attempt_batch` for the safest complete z-score matrix, and `z_score_global_single_attempt_batch` for the simplest single-population baseline.

## Thesis Use Sketch

- Family-pruning argument: use `all_global_single_attempt_batch` to show that the modified / hybrid families are materially worse than `standard_zscore` on matched scopes, so they belong in the thesis mainly as pruned alternatives or negative controls rather than as serious final-model candidates; ignore `threshold` and `iqr` when presenting that argument.
- Main detector-selection path: use `z_score_granularity_comparison` first to justify the granularity choice, then use the corresponding z-score sweep for the actual z-score cutoff and `mh` selection.
- Primary benchmark candidates: use `z_score_by_competitor_single_attempt_batch` when weighted F1 is the main priority, and use `z_score_by_country_single_attempt_batch` as the clean fully completed robustness benchmark.
- Baseline and appendix roles: use `z_score_global_single_attempt_batch` as the simple global baseline, `1-6coarse_grid_single_attempt_batch` as the sampled non-z-score comparison set, and `no_avg_hybrid_weighted_single_attempt_batch` only if `hybrid_weighted` needs standalone appendix treatment.

## Common Caveats

- `all_global_single_attempt_batch` is not a finished apples-to-apples benchmark run.
- Several competitor-scoped runs repeatedly fail on `COUNTRY_4/B2C/COMPETITOR_3_COUNTRY_4_2026-02-08`; this should be treated as a scope-specific problem that may need separate debugging.
- Cross-run averages are useful for orientation, but they are not perfectly comparable when the scope coverage differs.
