# mh5 Method Selection Summary

This summary is intended to support the thesis decision on which methods should be extended to the expensive full mh5-mh30 runs.

## Inputs

- Forest root: `C:\Users\Administrator\Desktop\dippa\src\results\tuning\forests\single_config_optimized_mh5_run`
- Default z-score granularity CSV: `C:\Users\Administrator\Desktop\dippa\src\results\tuning\statistical\z_score_granularity_comparison\analysis\granularity_performance_sampled_mh.csv`

## Detector Summary

| Detector | Scope count | Mean F1 | Mean G-mean | Mean train time (s) |
| --- | ---: | ---: | ---: | ---: |
| Extended Isolation Forest | 17 | 0.2588 | 0.5693 | 2320.5 |
| Isolation Forest | 17 | 0.6193 | 0.8800 | 10.0 |
| Robust Random Cut Forest | 17 | 0.2275 | 0.5752 | 6647.9 |
| Default z-score | 17 | 0.6062 | 0.8280 |  |

## Detector By Granularity

| Detector | Granularity | Scope count | Mean F1 | Mean G-mean | Mean train time (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| Extended Isolation Forest | global | 1 | 0.2189 | 0.5038 | 10912.5 |
| Isolation Forest | global | 1 | 0.5711 | 0.8486 | 48.3 |
| Robust Random Cut Forest | global | 1 | 0.2006 | 0.5460 | 23626.4 |
| Default z-score | global | 1 | 0.5968 | 0.8023 |  |
| Extended Isolation Forest | by_country | 4 | 0.2489 | 0.5453 | 3530.1 |
| Isolation Forest | by_country | 4 | 0.5783 | 0.8622 | 15.7 |
| Robust Random Cut Forest | by_country | 4 | 0.2056 | 0.5481 | 11149.3 |
| Default z-score | by_country | 4 | 0.5930 | 0.8122 |  |
| Extended Isolation Forest | by_competitor | 12 | 0.2654 | 0.5827 | 1201.3 |
| Isolation Forest | by_competitor | 12 | 0.6370 | 0.8886 | 4.9 |
| Robust Random Cut Forest | by_competitor | 12 | 0.2370 | 0.5866 | 3732.6 |
| Default z-score | by_competitor | 12 | 0.6113 | 0.8354 |  |

## Isolation Forest Dominance Over Slow Forest Baselines

| Challenger | Granularity | IF G-mean | Challenger G-mean | IF F1 | Challenger F1 | Time ratio (challenger / IF) | IF dominates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Extended Isolation Forest | global | 0.8486 | 0.5038 | 0.5711 | 0.2189 | 226.0 | yes |
| Extended Isolation Forest | by_country | 0.8622 | 0.5453 | 0.5783 | 0.2489 | 225.3 | yes |
| Extended Isolation Forest | by_competitor | 0.8886 | 0.5827 | 0.6370 | 0.2654 | 246.5 | yes |
| Robust Random Cut Forest | global | 0.8486 | 0.5460 | 0.5711 | 0.2006 | 489.3 | yes |
| Robust Random Cut Forest | by_country | 0.8622 | 0.5481 | 0.5783 | 0.2056 | 711.4 | yes |
| Robust Random Cut Forest | by_competitor | 0.8886 | 0.5866 | 0.6370 | 0.2370 | 766.1 | yes |

## Key Findings

- Isolation Forest outperforms EIF at mh5 while training much faster: mean G-mean 0.8800 vs 0.5693, mean F1 0.6193 vs 0.2588, mean train time 10.0s vs 2320.5s.
- Isolation Forest outperforms RRCF at mh5 while training much faster: mean G-mean 0.8800 vs 0.5752, mean F1 0.6193 vs 0.2275, mean train time 10.0s vs 6647.9s.
- Default z-score remains strong at mh5 across all granularities, with weighted mean G-mean 0.8280 and weighted mean F1 0.6062.
- On this mh5 evidence base, the methods worth extending to the expensive full sweep are `standard_zscore` and `if`.
- `eif` and `rrcf` have weak mh5 accuracy relative to `if` and require far more training time, so they are poor candidates for the full mh5-mh30 extension.
