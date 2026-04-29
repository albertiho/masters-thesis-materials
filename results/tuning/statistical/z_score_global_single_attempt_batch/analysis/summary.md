# Statistical Guidance Analysis

- Results root: `C:\Users\Administrator\Desktop\dippa\src\results\tuning\statistical\global_single_attempt_batch`
- Subset mh values: `mh5,mh10,mh15,mh20,mh25,mh30`
- Best-configuration rows: 26
- Guidance comparison rows: 1
- Path warnings: 0
- Empty top-level mh directories: (none)

## Subset-vs-Full Guidance

- Configuration match rate: 100.0%
- Median combined F1 regret: 0.000000
- Median rank-score regret: 0.000000
- Max combined F1 regret: 0.000000

## Detector-Level Summary

| Detector | Best | Support | Equal-Support Best | Equal-Support Match |
| --- | --- | ---: | --- | ---: |
| standard_zscore | standard_zscore__threshold_3p250 | 16/26 | standard_zscore__threshold_2p750 | yes |

## Detector-Level Regret

| Detector | Match | Combined F1 Regret | Equal-Support Combined F1 Regret |
| --- | ---: | ---: | ---: |
| standard_zscore | yes | 0.000000 | 0.000000 |

## Worst Case Regrets

| Detector | Scope | Full Best | Subset Best | Combined F1 Regret |
| --- | --- | --- | --- | ---: |
| standard_zscore | GLOBAL_2026-02-08 | standard_zscore__threshold_3p250 | standard_zscore__threshold_3p250 | 0.000000 |
