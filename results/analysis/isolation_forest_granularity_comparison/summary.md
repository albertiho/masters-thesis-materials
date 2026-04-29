# Isolation Forest Granularity Comparison

This summary aggregates the retained Isolation Forest detector across the sampled mh grid and the three aggregation granularities.

## Input

- Forest root: `C:\Users\Administrator\Desktop\dippa\src\results\tuning\forests\single_config_optimized_mh5_run`

## Granularity Summary

| Granularity | Mean combined F1 | Mean combined G-mean | Mean training time (s) | Best F1 mh | Best combined F1 | Best G-mean mh | Best combined G-mean | Mean scope count |
| --- | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: |
| Global | 0.5934 | 0.8764 | 35.02 | mh30 | 0.6326 | mh30 | 0.9105 | 1.00 |
| By country | 0.6264 | 0.8901 | 9.84 | mh30 | 0.6629 | mh30 | 0.9150 | 4.00 |
| By competitor | 0.6690 | 0.9073 | 3.33 | mh30 | 0.7029 | mh30 | 0.9225 | 11.17 |

## Key Findings

- The highest observed mean combined F1 was achieved under by competitor aggregation at mh30, with a score of 0.7029.
- The highest observed mean combined G-mean was achieved under by competitor aggregation at mh30, with a score of 0.9225.
