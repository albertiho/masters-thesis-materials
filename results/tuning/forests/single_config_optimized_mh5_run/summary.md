{
  "schema_version": "phase2.v1",
  "sweep_id": "single_config_optimized_mh5_run",
  "experiment_family": "tuning_forests",
  "complete_task_count": 17,
  "error_task_count": 0,
  "best_configuration_count": 17,
  "config": {
    "data_subsets_root": "C:\\Users\\Administrator\\Desktop\\dippa\\src\\data-subsets",
    "model_root": "C:\\Users\\Administrator\\Desktop\\dippa\\src\\artifacts\\models",
    "detectors": [
      "rrcf"
    ],
    "mh_values": [
      "mh5"
    ],
    "granularities": [
      "global",
      "by_country",
      "by_competitor"
    ],
    "splits": [
      "new_prices",
      "new_products"
    ],
    "attempts": 1,
    "max_workers": 2,
    "trial_workers": 2,
    "target_metric": "gmean",
    "min_precision": 0.3,
    "injection_rate": 0.1,
    "steps": 11,
    "max_products": null,
    "max_history_per_product": null,
    "min_test_rows": null,
    "max_test_rows": null,
    "dry_run": false,
    "resume": true
  }
}
