# Country-Level IF Replay On COUNTRY_4 / COMPETITOR_3 Failed Cases

## Purpose

- Evaluate the retained country-level IF configuration for `COUNTRY_4` on the failed competitor-level case `COUNTRY_4/B2C/COMPETITOR_3_COUNTRY_4_2026-02-08` across the mh levels that could not support a competitor-level IF model.
- Use the same synthetic-injection evaluation path as the forest sweep, with `10` repeated trial(s) per split at injection rate `10%`.
- Prefer loading the persisted country-level IF model; when it is absent locally, refit the retained country-level configuration from the saved sweep metadata before scoring.

## Key Findings

- No persisted country-level IF artifact was available locally for the targeted COUNTRY_4 mh levels; all evaluated cases were therefore refit from the retained country-level sweep configuration.
- The only failed mh case with non-empty competitor-level evaluation splits was `mh10`, where the country-level IF replay reached weighted combined `F1=0.7233` and `G=0.8685`.
- The later failed cases `mh15, mh20, mh25, mh30` could not be replayed because both competitor-level test splits are empty after the minimum-history filter.
- At `mh10`, the `new_products` split had `0` matched competitor-history row(s) for the evaluated test products, which indicates that the replay includes an extremely sparse local-history case.

## mh Summary

| mh | Replay status | Original competitor failure | New prices F1 | New prices G | New products F1 | New products G | Combined F1 | Combined G |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mh10 | ok | Need at least 50 valid samples, got 20 | 0.7333 | 0.9407 | 0.7000 | 0.7000 | 0.7233 | 0.8685 |
| mh15 | no_eval_rows | Need at least 50 valid samples, got 0 | NA | NA | NA | NA | NA | NA |
| mh20 | no_eval_rows | Need at least 50 valid samples, got 0 | NA | NA | NA | NA | NA | NA |
| mh25 | no_eval_rows | Need at least 50 valid samples, got 0 | NA | NA | NA | NA | NA | NA |
| mh30 | no_eval_rows | Need at least 50 valid samples, got 0 | NA | NA | NA | NA | NA | NA |

## Files

- `split_metrics.csv`: per-split replay metrics for each failed mh case.
- `mh_summary.csv`: one row per failed mh case, including weighted combined scores when both splits are evaluable.
- `summary.json`: machine-readable summary of the replay output.
- `data_references.md`: provenance note for thesis-facing reuse.
