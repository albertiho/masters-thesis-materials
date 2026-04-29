# Modified Z-score Scale Assessment (mh5)

This analysis summarizes the modified-zscore-family tuning results for `mh5` across `global`, `by_country`, and `by_competitor`.

## Key Findings

- Across all available modified-zscore detector/scope combinations, `73` of `84` best configurations (`86.9%`) selected the original upper-bound threshold `3.0`.
- For every detector family, the mean coarse-grid weighted combined F1 increased monotonically from `1.0` to `3.0`.
- The mean weighted-combined-F1 gain from `2.5` to `3.0` remained positive for every available scope in every detector family.
- This pattern indicates that the original `1.0-3.0` range was too narrow to reveal whether performance had already peaked or was still improving beyond the upper boundary.

## Boundary Summary

| Detector | Combinations | Best at 3.0 | Share at 3.0 | Mean F1 at 1.0 | Mean F1 at 3.0 | Mean G-mean at 1.0 | Mean G-mean at 3.0 | Mean delta 1.0->3.0 | Mean delta 2.5->3.0 | Positive 2.5->3.0 | Available 2.5->3.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hybrid avg | 17 | 15 | 0.8824 | 0.2617 | 0.3782 | 0.6234 | 0.7084 | 0.1165 | 0.0212 | 17 | 17 |
| Hybrid max | 17 | 16 | 0.9412 | 0.2555 | 0.3702 | 0.6156 | 0.7062 | 0.1147 | 0.0219 | 17 | 17 |
| Hybrid weighted | 16 | 14 | 0.8750 | 0.2659 | 0.3835 | 0.6281 | 0.7115 | 0.1177 | 0.0201 | 16 | 16 |
| Modified MAD | 17 | 12 | 0.7059 | 0.2672 | 0.3837 | 0.6298 | 0.7085 | 0.1164 | 0.0190 | 17 | 17 |
| Modified Sn | 17 | 16 | 0.9412 | 0.2570 | 0.3732 | 0.6173 | 0.7076 | 0.1162 | 0.0224 | 17 | 17 |

## Mean Coarse Weighted Combined F1 by Threshold

| threshold | Hybrid avg | Hybrid max | Hybrid weighted | Modified MAD | Modified Sn |
| --- | --- | --- | --- | --- | --- |
| 1.0000 | 0.2617 | 0.2555 | 0.2659 | 0.2672 | 0.2570 |
| 1.5000 | 0.3017 | 0.2943 | 0.3105 | 0.3123 | 0.2961 |
| 2.0000 | 0.3316 | 0.3231 | 0.3362 | 0.3356 | 0.3287 |
| 2.5000 | 0.3570 | 0.3483 | 0.3634 | 0.3646 | 0.3508 |
| 3.0000 | 0.3782 | 0.3702 | 0.3835 | 0.3837 | 0.3732 |
