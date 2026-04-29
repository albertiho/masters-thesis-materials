# Country-Level IF Replay Data References

## Thesis artifact references

| Thesis item | Source file | Presents data |
| --- | --- | --- |
| Table `country_if_country4_competitor3_replay` | `results/analysis/country_level_if_country4_competitor3_replay/mh_summary.csv` | Weighted combined and split-level replay metrics for the retained country-level \(\mathrm{IF}\) configuration on the failed `COUNTRY_4 / B2C / COMPETITOR_3_COUNTRY_4_2026-02-08` competitor case across `mh10`, `mh15`, `mh20`, `mh25`, and `mh30`. |

## Raw aggregation inputs

| Source path | Contributes data for | Presents data |
| --- | --- | --- |
| `results/tuning/forests/single_config_optimized_mh5_run/` | Retained country-level \(\mathrm{IF}\) configurations and original competitor-level failure reasons | Nested `by_country/.../if/best_configuration.json` payloads define the replayed country-level model configuration for each `mh`, while nested `by_competitor/.../if/summary.json` payloads record the original competitor-level failure reason. |
| `data-subsets/` | Replay datasets | The competitor-level `train.parquet`, `test_new_prices.parquet`, and `test_new_products.parquet` files under each failed `mh` level provide the evaluation target for the replay. |

## Notes

- The replay prefers persisted country-level IF artifacts under `artifacts/models/` when they exist locally.
- When the persisted artifact is absent, the replay refits the retained country-level configuration from the saved sweep metadata before evaluating it on the failed competitor case.
