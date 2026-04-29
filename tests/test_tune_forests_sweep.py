from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.training.scripts import tune_forests as forest_tuner
from research.training.scripts.train_isolation_forest import extract_features_vectorized, train_from_matrix
from src.anomaly.ml.eif import EIFConfig, EIFDetector
from src.anomaly.ml.rrcf import RRCFDetector, RRCFDetectorConfig
from src.anomaly.persistence import ModelPersistence


def _price_frame(
    *,
    competitor_id: str = "COMPETITOR_1_COUNTRY_1",
    country: str = "FI",
    n_products: int = 12,
    observations_per_product: int = 8,
    start: str = "2026-01-01",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for product_idx in range(n_products):
        base_price = 100.0 + product_idx
        for observation_idx in range(observations_per_product):
            rows.append(
                {
                    "product_id": f"product-{product_idx:03d}",
                    "competitor_id": competitor_id,
                    "competitor_product_id": f"product-{product_idx:03d}",
                    "price": base_price + (observation_idx % 4) * 0.5,
                    "list_price": base_price * 1.2,
                    "first_seen_at": pd.Timestamp(start, tz="UTC") + pd.Timedelta(days=observation_idx),
                    "country": country,
                }
            )
    return pd.DataFrame(rows)


def _write_scope(
    *,
    data_subsets_root: Path,
    mh_level: str,
    granularity: str = "by_competitor",
    relative_dir: str = "COUNTRY_1/B2B",
    stem: str = "COMPETITOR_1_COUNTRY_1_2026-02-08",
) -> tuple[Path, Path, Path]:
    scope_dir = data_subsets_root / mh_level / granularity / relative_dir
    scope_dir.mkdir(parents=True, exist_ok=True)

    train_df = _price_frame(start="2026-01-01", observations_per_product=8)
    prices_df = _price_frame(start="2026-02-01", observations_per_product=2)
    products_df = _price_frame(start="2026-03-01", observations_per_product=2)

    train_path = scope_dir / f"{stem}_train.parquet"
    prices_path = scope_dir / f"{stem}_test_new_prices.parquet"
    products_path = scope_dir / f"{stem}_test_new_products.parquet"

    train_df.to_parquet(train_path, index=False)
    prices_df.to_parquet(prices_path, index=False)
    products_df.to_parquet(products_path, index=False)
    return train_path, prices_path, products_path


def _save_forest_models(
    *,
    model_root: Path,
    train_path: Path,
    mh_level: str,
    include_detectors: tuple[str, ...],
) -> None:
    train_df = pd.read_parquet(train_path)
    X_train = extract_features_vectorized(train_df)
    persistence = ModelPersistence(model_root=model_root)
    model_name = f"COMPETITOR_1_COUNTRY_1_{mh_level}"

    if "if" in include_detectors:
        if_detector, _ = train_from_matrix(
            X_train,
            contamination="auto",
            anomaly_threshold=0.4,
            n_estimators=16,
            max_samples=64,
            max_features=0.75,
            random_state=42,
        )
        persistence.save_isolation_forest(if_detector, model_name, len(X_train))

    if "eif" in include_detectors:
        eif_detector = EIFDetector(
            EIFConfig(
                n_estimators=16,
                max_samples=64,
                max_features=0.75,
                random_state=42,
                anomaly_threshold=0.4,
            )
        )
        eif_detector.fit_from_matrix(X_train)
        persistence.save_eif(eif_detector, model_name, len(X_train))

    if "rrcf" in include_detectors:
        rrcf_detector = RRCFDetector(
            RRCFDetectorConfig(
                num_trees=8,
                tree_size=32,
                anomaly_threshold=0.8,
                random_state=42,
                warmup_samples=8,
            )
        )
        rrcf_detector.fit_from_matrix(X_train)
        persistence.save_rrcf(rrcf_detector, model_name, len(X_train))


def test_build_parser_defaults_to_sampled_mh_values() -> None:
    args = forest_tuner.build_parser().parse_args([])
    assert args.mh_values == list(forest_tuner.DEFAULT_SAMPLED_MH_VALUES)
    assert args.granularities == list(forest_tuner.DEFAULT_FOREST_GRANULARITIES)
    assert args.target_metric == "gmean"
    assert args.scope_filter is None
    assert args.eif_search_strategy == "two_pass"


def test_run_sweep_defaults_to_sampled_mh_values_only(tmp_path: Path) -> None:
    data_root = tmp_path / "data-subsets"
    model_root = tmp_path / "models"
    output_root = tmp_path / "results" / "tuning" / "forests" / "default_sampled_mh"

    for mh_level in ("mh5", "mh6", "mh10"):
        train_path, _, _ = _write_scope(data_subsets_root=data_root, mh_level=mh_level)
        _save_forest_models(model_root=model_root, train_path=train_path, mh_level=mh_level, include_detectors=("if",))

    args = forest_tuner.build_parser().parse_args(
        [
            "--data-subsets-root",
            str(data_root),
            "--model-root",
            str(model_root),
            "--output-root",
            str(output_root),
            "--detectors",
            "if",
            "--granularities",
            "by_competitor",
            "--attempts",
            "1",
            "--max-workers",
            "1",
            "--trial-workers",
            "1",
            "--steps",
            "5",
            "--max-products",
            "4",
            "--max-history-per-product",
            "5",
            "--min-test-rows",
            "8",
            "--max-test-rows",
            "12",
            "--dry-run",
        ]
    )

    assert forest_tuner.run_sweep(args) == 0

    best_configurations = pd.read_csv(output_root / "best_configurations.csv")
    scope_status = pd.read_csv(output_root / "scope_status.csv")

    assert set(best_configurations["mh_level"].astype(str)) == {"mh5", "mh10"}
    assert set(scope_status["mh_level"].astype(str)) == {"mh5", "mh10"}
    assert "mh6" not in set(scope_status["mh_level"].astype(str))


def test_run_sweep_scope_filter_limits_execution_to_matching_scope(tmp_path: Path) -> None:
    data_root = tmp_path / "data-subsets"
    model_root = tmp_path / "models"
    output_root = tmp_path / "results" / "tuning" / "forests" / "scope_filter"

    train_one, _, _ = _write_scope(
        data_subsets_root=data_root,
        mh_level="mh5",
        relative_dir="COUNTRY_1/B2B",
        stem="COMPETITOR_1_COUNTRY_1_2026-02-08",
    )
    train_two, _, _ = _write_scope(
        data_subsets_root=data_root,
        mh_level="mh5",
        relative_dir="COUNTRY_1/B2C",
        stem="COMPETITOR_2_COUNTRY_1_2026-02-08",
    )
    _save_forest_models(model_root=model_root, train_path=train_one, mh_level="mh5", include_detectors=("if",))
    _save_forest_models(model_root=model_root, train_path=train_two, mh_level="mh5", include_detectors=("if",))

    args = forest_tuner.build_parser().parse_args(
        [
            "--data-subsets-root",
            str(data_root),
            "--model-root",
            str(model_root),
            "--output-root",
            str(output_root),
            "--scope-filter",
            "COMPETITOR_1_COUNTRY_1",
            "--detectors",
            "if",
            "--mh-values",
            "mh5",
            "--granularities",
            "by_competitor",
            "--attempts",
            "1",
            "--max-workers",
            "1",
            "--trial-workers",
            "1",
            "--steps",
            "5",
            "--max-products",
            "4",
            "--max-history-per-product",
            "5",
            "--min-test-rows",
            "8",
            "--max-test-rows",
            "12",
            "--dry-run",
        ]
    )

    assert forest_tuner.run_sweep(args) == 0

    best_configurations = pd.read_csv(output_root / "best_configurations.csv")
    scope_status = pd.read_csv(output_root / "scope_status.csv")
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))

    assert set(best_configurations["dataset_name"].astype(str)) == {"COMPETITOR_1_COUNTRY_1"}
    assert set(scope_status["dataset_name"].astype(str)) == {"COMPETITOR_1_COUNTRY_1"}
    assert summary["config"]["scope_filter"] == "COMPETITOR_1_COUNTRY_1"


def test_run_sweep_can_freeze_eif_gmean_winner_for_scope(tmp_path: Path) -> None:
    data_root = tmp_path / "data-subsets"
    model_root = tmp_path / "models"
    output_root = tmp_path / "results" / "tuning" / "forests" / "fixed_eif_gmean_winner"

    train_path, _, _ = _write_scope(data_subsets_root=data_root, mh_level="mh5")
    _save_forest_models(model_root=model_root, train_path=train_path, mh_level="mh5", include_detectors=("eif",))

    args = forest_tuner.build_parser().parse_args(
        [
            "--data-subsets-root",
            str(data_root),
            "--model-root",
            str(model_root),
            "--output-root",
            str(output_root),
            "--detectors",
            "eif",
            "--eif-search-strategy",
            "fixed_gmean_winner",
            "--mh-values",
            "mh5",
            "--granularities",
            "by_competitor",
            "--attempts",
            "1",
            "--max-workers",
            "1",
            "--trial-workers",
            "1",
            "--max-products",
            "4",
            "--max-history-per-product",
            "5",
            "--min-test-rows",
            "8",
            "--max-test-rows",
            "12",
            "--dry-run",
        ]
    )

    assert forest_tuner.run_sweep(args) == 0

    best_configurations = pd.read_csv(output_root / "best_configurations.csv")
    scope_status = pd.read_csv(output_root / "scope_status.csv")
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))

    assert set(best_configurations["detector_family"].astype(str)) == {"eif"}
    assert set(scope_status["status"].astype(str)) == {"ok"}
    assert summary["config"]["eif_search_strategy"] == "fixed_gmean_winner"

    detector_dir = Path(scope_status.loc[scope_status["detector_family"] == "eif", "output_dir"].iloc[0])
    candidate_metrics = pd.read_csv(detector_dir / "candidate_metrics.csv")
    best_configuration = json.loads((detector_dir / "best_configuration.json").read_text(encoding="utf-8"))

    assert len(candidate_metrics) == 1
    assert set(candidate_metrics["stage"].astype(str)) == {"fixed_profile"}
    assert set(candidate_metrics["threshold"].astype(float)) == {0.2}
    assert set(candidate_metrics["n_estimators"].astype(int)) == {100}
    assert set(candidate_metrics["max_samples"].astype(int)) == {256}
    assert set(candidate_metrics["max_features"].astype(float)) == {0.5}
    assert candidate_metrics["is_baseline_config"].all()
    assert set(candidate_metrics["promoted_config_count"].astype(int)) == {1}
    assert set(candidate_metrics["screened_config_count"].astype(int)) == {1}
    assert candidate_metrics["screening_threshold"].isna().all()

    summary_payload = best_configuration["summary"]
    assert summary_payload["search_strategy"] == "fixed_eif_gmean_winner"
    assert summary_payload["fixed_profile_name"] == "fixed_gmean_winner"
    assert summary_payload["fixed_threshold"] == 0.2
    assert summary_payload["fixed_hyperparameters"] == {
        "n_estimators": 100,
        "max_samples": 256,
        "max_features": 0.5,
    }
    assert summary_payload["current_threshold"] == 0.2
    assert summary_payload["best_threshold"] == 0.2
    assert summary_payload["current_hyperparameters"] == summary_payload["best_hyperparameters"]


def test_run_sweep_writes_all_forest_detector_results_for_one_scope(tmp_path: Path) -> None:
    data_root = tmp_path / "data-subsets"
    model_root = tmp_path / "models"
    output_root = tmp_path / "results" / "tuning" / "forests" / "single_scope_all_forests"

    train_path, _, _ = _write_scope(data_subsets_root=data_root, mh_level="mh5")
    _save_forest_models(
        model_root=model_root,
        train_path=train_path,
        mh_level="mh5",
        include_detectors=("if", "eif", "rrcf"),
    )

    args = forest_tuner.build_parser().parse_args(
        [
            "--data-subsets-root",
            str(data_root),
            "--model-root",
            str(model_root),
            "--output-root",
            str(output_root),
            "--detectors",
            "if",
            "eif",
            "rrcf",
            "--mh-values",
            "mh5",
            "--granularities",
            "by_competitor",
            "--attempts",
            "1",
            "--max-workers",
            "1",
            "--trial-workers",
            "1",
            "--steps",
            "5",
            "--max-products",
            "4",
            "--max-history-per-product",
            "5",
            "--min-test-rows",
            "8",
            "--max-test-rows",
            "12",
            "--dry-run",
        ]
    )

    assert forest_tuner.run_sweep(args) == 0

    best_configurations = pd.read_csv(output_root / "best_configurations.csv")
    scope_status = pd.read_csv(output_root / "scope_status.csv")
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))

    assert set(best_configurations["detector_family"].astype(str)) == {"if", "eif", "rrcf"}
    assert set(scope_status["detector_family"].astype(str)) == {"if", "eif", "rrcf"}
    assert set(scope_status["status"].astype(str)) == {"ok"}
    assert summary["complete_task_count"] == 3

    expected_rrcf_configs = {
        forest_tuner._config_key(
            "rrcf",
            {"num_trees": 80, "tree_size": 128, "warmup_samples": forest_tuner.RRCF_OPTIMAL_CONFIG.warmup_samples},
        )
    }

    for detector_family in ("if", "eif", "rrcf"):
        detector_dir = Path(
            scope_status.loc[scope_status["detector_family"] == detector_family, "output_dir"].iloc[0]
        )
        best_row = best_configurations[best_configurations["detector_family"] == detector_family].iloc[0]
        candidate_metrics = pd.read_csv(detector_dir / "candidate_metrics.csv")
        assert (detector_dir / "candidate_metrics.csv").exists()
        assert (detector_dir / "best_configuration.json").exists()
        assert (detector_dir / "split_results.json").exists()
        assert pd.notna(best_row["combined_g_mean"])
        assert pd.notna(best_row["combined_fpr"])
        assert pd.notna(best_row["combined_fnr"])
        assert pd.notna(best_row["new_prices_g_mean_mean"])
        assert pd.notna(best_row["new_products_fpr_mean"])

        split_results = json.loads((detector_dir / "split_results.json").read_text(encoding="utf-8"))
        assert set(split_results) == {"new_prices", "new_products"}
        for split_payload in split_results.values():
            assert "current_metrics" in split_payload
            assert "best_metrics" in split_payload
            assert "g_mean" in split_payload["current_metrics"]
            assert "fpr" in split_payload["current_metrics"]
            assert "fnr" in split_payload["current_metrics"]

        expected_rows = {"if": 3 * 5, "eif": 12 * 5, "rrcf": 1 * 5}[detector_family]
        assert len(candidate_metrics) == expected_rows
        assert "config_key" in candidate_metrics.columns
        assert "is_baseline_config" in candidate_metrics.columns
        assert "screening_threshold" in candidate_metrics.columns
        assert "screened_config_count" in candidate_metrics.columns
        assert "promoted_config_count" in candidate_metrics.columns
        if detector_family == "if":
            assert "contamination" in candidate_metrics.columns
            assert set(candidate_metrics["contamination"].astype(str)) == {"auto"}
            assert set(candidate_metrics["promoted_config_count"].astype(int)) == {3}
            assert set(candidate_metrics["screened_config_count"].astype(int)) == {12}
            assert set(candidate_metrics["screening_threshold"].astype(float)) == {0.48}
        if detector_family == "eif":
            assert candidate_metrics["stage"].astype(str).nunique() == 1
            assert set(candidate_metrics["stage"].astype(str)) == {"two_pass_refinement"}
            assert candidate_metrics["config_key"].astype(str).nunique() == 12
            assert set(candidate_metrics["promoted_config_count"].astype(int)) == {12}
            assert set(candidate_metrics["screened_config_count"].astype(int)) == {6}
            assert candidate_metrics["screening_threshold"].isna().all()
            best_configuration = json.loads((detector_dir / "best_configuration.json").read_text(encoding="utf-8"))
            summary_payload = best_configuration["summary"]
            assert summary_payload["search_strategy"] == "two_pass_eif_calibration"
            assert summary_payload["pass1_config_count"] == 6
            assert summary_payload["pass1_threshold_count"] == 15
            assert len(summary_payload["pass1_selected_max_features"]) == 2
            assert summary_payload["pass2_config_count"] == 12
            assert summary_payload["pass2_threshold_count"] == 5
        if detector_family == "rrcf":
            assert "warmup_samples" in candidate_metrics.columns
            assert set(candidate_metrics["config_key"].astype(str)) == expected_rrcf_configs
            assert set(candidate_metrics["warmup_samples"].astype(int)) == {forest_tuner.RRCF_OPTIMAL_CONFIG.warmup_samples}
            assert set(candidate_metrics["promoted_config_count"].astype(int)) == {1}
            assert set(candidate_metrics["screened_config_count"].astype(int)) == {1}
            assert candidate_metrics["screening_threshold"].isna().all()
