# Detector_Combinations Summary

- Run ID: `initial_country1_if_zscore_layered_combinations`
- Schema version: `phase2.v1`
- Splits: new_prices, new_products
- Detector metric rows: 18

## new_prices

| Candidate | Detector | Precision | Recall | F1 | TP | FP | FN | TN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| country1_if_zscore_layered_combinations | Sanity -> IF | 0.6321 | 0.8715 | 0.7327 | 38636 | 22488 | 5699 | 376529 |
| country1_if_zscore_layered_combinations | IF | 0.6102 | 0.7940 | 0.6901 | 35201 | 22486 | 9134 | 376531 |
| country1_if_zscore_layered_combinations | Sanity -> Z-score | 0.5728 | 0.7781 | 0.6599 | 34495 | 25722 | 9840 | 373295 |
| country1_if_zscore_layered_combinations | Sanity -> Z-score (>=10) -> IF | 0.5116 | 0.8963 | 0.6514 | 39739 | 37934 | 4596 | 361083 |
| country1_if_zscore_layered_combinations | Sanity -> Z-score (>=5) -> IF | 0.4958 | 0.8985 | 0.6390 | 39836 | 40503 | 4499 | 358514 |
| country1_if_zscore_layered_combinations | IF -> Z-score (50/50) | 0.6657 | 0.6065 | 0.6347 | 26891 | 13505 | 17444 | 385512 |
| country1_if_zscore_layered_combinations | Sanity | 1.0000 | 0.4471 | 0.6179 | 19821 | 0 | 24514 | 399017 |
| country1_if_zscore_layered_combinations | Z-score | 0.5150 | 0.6161 | 0.5610 | 27315 | 25725 | 17020 | 373292 |
| country1_if_zscore_layered_combinations | Sanity -> Z-score -> IF (50/50) | 0.5193 | 0.3226 | 0.3980 | 14303 | 13240 | 30032 | 385777 |

## new_products

| Candidate | Detector | Precision | Recall | F1 | TP | FP | FN | TN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| country1_if_zscore_layered_combinations | Sanity | 1.0000 | 0.4448 | 0.6157 | 41870 | 0 | 52271 | 847277 |
| country1_if_zscore_layered_combinations | Sanity -> Z-score | 0.5544 | 0.6441 | 0.5959 | 60639 | 48743 | 33502 | 798534 |
| country1_if_zscore_layered_combinations | Sanity -> IF | 0.4630 | 0.7207 | 0.5638 | 67847 | 78676 | 26294 | 768601 |
| country1_if_zscore_layered_combinations | Sanity -> Z-score (>=10) -> IF | 0.4371 | 0.7249 | 0.5453 | 68241 | 87894 | 25900 | 759383 |
| country1_if_zscore_layered_combinations | Sanity -> Z-score (>=5) -> IF | 0.4076 | 0.7292 | 0.5229 | 68646 | 99765 | 25495 | 747512 |
| country1_if_zscore_layered_combinations | IF -> Z-score (50/50) | 0.5144 | 0.3781 | 0.4358 | 35593 | 33606 | 58548 | 813671 |
| country1_if_zscore_layered_combinations | IF | 0.3226 | 0.5350 | 0.4025 | 50368 | 105753 | 43773 | 741524 |
| country1_if_zscore_layered_combinations | Z-score | 0.4390 | 0.3614 | 0.3964 | 34024 | 43487 | 60117 | 803790 |
| country1_if_zscore_layered_combinations | Sanity -> Z-score -> IF (50/50) | 0.4500 | 0.1919 | 0.2691 | 18065 | 22077 | 76076 | 825200 |

