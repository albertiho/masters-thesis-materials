# Data References

- `tab:layered-finalist-overall` got data from `layered_finalist_overall_summary.csv`, presents the overall cross-country comparison of the four layered finalists on the combined test case.
- `fig:layered-finalist-mh-trends` (`results/media/layered_finalist_mh_trends.png` / `results/media/layered_finalist_mh_trends.svg`) got data from `layered_finalist_mh_summary.csv`, presents the mean `F_{1,\mathrm{wc}}` and `G_{\mathrm{wc}}` trends of the layered finalists across the sampled minimum-history settings, with numeric minimum-history tick labels and separate metric and layer-configuration legends.
- `fig:layered-finalist-recall-trends` (`results/media/layered_finalist_recall_mh_trends.png` / `results/media/layered_finalist_recall_mh_trends.svg`) got data from `layered_finalist_mh_summary.csv`, presents the mean recall and mean `F_{1,\mathrm{wc}}` trends of the layered finalists across the sampled minimum-history settings.
- Chapter 6 minimum-history winner-count discussion got data from `layered_finalist_mh_winner_counts.csv`, presents per-`mh` scope-win counts for `F_{1,\mathrm{wc}}` and `G_{\mathrm{wc}}`.
- `tab:layered-finalist-recall-stability` got data from `layered_finalist_recall_stability_summary.csv`, presents overall recall stability of the layered finalists across the valid scope-`mh` surface.
- Chapter 6 recall-winner discussion got data from `layered_finalist_recall_winner_counts.csv`, presents recall winner counts across the full valid scope-`mh` surface.
- `tab:layered-finalist-anomaly-types` got data from `layered_finalist_high_recall_anomaly_case_summary.csv`, presents the anomaly-type recall comparison for `Sanity -> IF` and `Sanity -> Z-score (>=5) -> IF`.
- `layered_finalist_overall_summary.csv`, `layered_finalist_mh_summary.csv`, and `layered_finalist_mh_winner_counts.csv` were aggregated from `layered_detector_metrics.csv` using rows where `test_case_name == combined`.
- `layered_finalist_recall_stability_summary.csv` and `layered_finalist_recall_winner_counts.csv` were aggregated from `layered_detector_metrics.csv` using rows where `test_case_name == combined`.
- `layered_finalist_top2_anomaly_case_summary.csv` and `layered_finalist_high_recall_anomaly_case_summary.csv` were aggregated from `layered_detector_anomaly_case_metrics.csv` using rows where `test_case_name == combined`.
