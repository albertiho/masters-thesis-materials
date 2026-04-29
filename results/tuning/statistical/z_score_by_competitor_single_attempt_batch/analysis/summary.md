# Statistical Guidance Analysis

- Results root: `C:\Users\Administrator\Desktop\dippa\src\results\tuning\statistical\by_competitor_single_attempt_batch`
- Subset mh values: `mh5,mh10,mh15,mh20,mh25,mh30`
- Best-configuration rows: 292
- Guidance comparison rows: 12
- Path warnings: 0
- Empty top-level mh directories: (none)

## Subset-vs-Full Guidance

- Configuration match rate: 75.0%
- Median combined F1 regret: 0.000000
- Median rank-score regret: 0.000000
- Max combined F1 regret: 0.141541

## Detector-Level Summary

| Detector | Best | Support | Equal-Support Best | Equal-Support Match |
| --- | --- | ---: | --- | ---: |
| standard_zscore | standard_zscore__threshold_3p750 | 18/292 | standard_zscore__threshold_3p000 | yes |

## Detector-Level Regret

| Detector | Match | Combined F1 Regret | Equal-Support Combined F1 Regret |
| --- | ---: | ---: | ---: |
| standard_zscore | yes | 0.000000 | 0.000000 |

## Worst Case Regrets

| Detector | Scope | Full Best | Subset Best | Combined F1 Regret |
| --- | --- | --- | --- | ---: |
| standard_zscore | COUNTRY_1/B2C/COMPETITOR_3_COUNTRY_1_2026-02-08 | standard_zscore__threshold_3p250 | standard_zscore__threshold_2p250 | 0.141541 |
| standard_zscore | COUNTRY_4/B2B/COMPETITOR_1_COUNTRY_4_2026-02-08 | standard_zscore__threshold_3p750 | standard_zscore__threshold_3p250 | 0.015991 |
| standard_zscore | COUNTRY_1/B2B/COMPETITOR_1_COUNTRY_1_2026-02-08 | standard_zscore__threshold_3p250 | standard_zscore__threshold_3p750 | 0.008794 |
| standard_zscore | COUNTRY_2/B2B/COMPETITOR_1_COUNTRY_2_2026-02-08 | standard_zscore__threshold_3p250 | standard_zscore__threshold_3p250 | 0.000000 |
| standard_zscore | COUNTRY_3/B2B/COMPETITOR_1_COUNTRY_3_2026-02-08 | standard_zscore__threshold_2p750 | standard_zscore__threshold_2p750 | 0.000000 |
| standard_zscore | COUNTRY_1/B2C/COMPETITOR_2_COUNTRY_1_2026-02-08 | standard_zscore__threshold_3p000 | standard_zscore__threshold_3p000 | 0.000000 |
| standard_zscore | COUNTRY_3/B2C/COMPETITOR_2_COUNTRY_3_2026-02-08 | standard_zscore__threshold_2p750 | standard_zscore__threshold_2p750 | 0.000000 |
| standard_zscore | COUNTRY_2/B2C/COMPETITOR_2_COUNTRY_2_2026-02-08 | standard_zscore__threshold_3p250 | standard_zscore__threshold_3p250 | 0.000000 |
| standard_zscore | COUNTRY_4/B2C/COMPETITOR_2_COUNTRY_4_2026-02-08 | standard_zscore__threshold_3p000 | standard_zscore__threshold_3p000 | 0.000000 |
| standard_zscore | COUNTRY_2/B2C/COMPETITOR_3_COUNTRY_2_2026-02-08 | standard_zscore__threshold_3p750 | standard_zscore__threshold_3p750 | 0.000000 |
