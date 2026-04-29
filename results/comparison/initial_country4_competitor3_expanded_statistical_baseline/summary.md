# Comparison Summary

- Run ID: `initial_country4_competitor3_expanded_statistical_baseline`
- Schema version: `phase2.v1`
- Splits: new_prices, new_products
- Detector metric rows: 16

## new_prices

| Candidate | Detector | Precision | Recall | F1 | TP | FP | FN | TN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| expanded_statistical_baseline | Threshold | 0.8232 | 0.9310 | 0.8738 | 135 | 29 | 10 | 1280 |
| expanded_statistical_baseline | Z-score | 0.7423 | 0.9931 | 0.8496 | 144 | 50 | 1 | 1259 |
| expanded_statistical_baseline | IQR | 0.3897 | 0.8897 | 0.5420 | 129 | 202 | 16 | 1107 |
| expanded_statistical_baseline | ModifiedSN | 0.2854 | 0.8759 | 0.4305 | 127 | 318 | 18 | 991 |
| expanded_statistical_baseline | ModifiedMAD | 0.2810 | 0.8759 | 0.4255 | 127 | 325 | 18 | 984 |
| expanded_statistical_baseline | HybridAvg | 0.2797 | 0.8759 | 0.4240 | 127 | 327 | 18 | 982 |
| expanded_statistical_baseline | HybridWeighted | 0.2797 | 0.8759 | 0.4240 | 127 | 327 | 18 | 982 |
| expanded_statistical_baseline | HybridMax | 0.2679 | 0.8759 | 0.4103 | 127 | 347 | 18 | 962 |

## new_products

| Candidate | Detector | Precision | Recall | F1 | TP | FP | FN | TN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| expanded_statistical_baseline | Threshold | 0.6846 | 0.7542 | 0.7177 | 89 | 41 | 29 | 1025 |
| expanded_statistical_baseline | Z-score | 0.7042 | 0.4237 | 0.5291 | 50 | 21 | 68 | 1045 |
| expanded_statistical_baseline | IQR | 0.6129 | 0.3220 | 0.4222 | 38 | 24 | 80 | 1042 |
| expanded_statistical_baseline | HybridAvg | 0.3669 | 0.4322 | 0.3969 | 51 | 88 | 67 | 978 |
| expanded_statistical_baseline | HybridWeighted | 0.3669 | 0.4322 | 0.3969 | 51 | 88 | 67 | 978 |
| expanded_statistical_baseline | ModifiedMAD | 0.3669 | 0.4322 | 0.3969 | 51 | 88 | 67 | 978 |
| expanded_statistical_baseline | ModifiedSN | 0.3617 | 0.4322 | 0.3938 | 51 | 90 | 67 | 976 |
| expanded_statistical_baseline | HybridMax | 0.3542 | 0.4322 | 0.3893 | 51 | 93 | 67 | 973 |

