from __future__ import annotations

import sys
from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from research.training.scripts.run_all_competitors_country_level_layered_finalists import (
    COUNTRIES,
    MH_LEVELS,
    DATASET_SNAPSHOT,
    FINAL_COMBINATIONS,
    SANITY_IF_NAME,
    SANITY_ZSCORE_GT10_IF_NAME,
    SANITY_ZSCORE_GT5_IF_NAME,
    SANITY_ZSCORE_NAME,
    build_dataset_paths,
    build_scope_specs,
    _load_country_if_config,
)


def test_final_combinations_match_requested_country_level_finalists() -> None:
    assert FINAL_COMBINATIONS == [
        SANITY_IF_NAME,
        SANITY_ZSCORE_NAME,
        SANITY_ZSCORE_GT10_IF_NAME,
        SANITY_ZSCORE_GT5_IF_NAME,
    ]


def test_build_scope_specs_discovers_all_competitor_scopes_with_country_models() -> None:
    scope_specs = build_scope_specs()

    assert len(scope_specs) == 72
    assert {scope.country for scope in scope_specs} == set(COUNTRIES)
    assert {scope.mh_level for scope in scope_specs} == set(MH_LEVELS)
    assert {scope.dataset_granularity for scope in scope_specs} == {"competitor"}
    assert {scope.scope_market for scope in scope_specs} == {"B2B", "B2C"}
    assert {scope.country_if_model_name for scope in scope_specs} == {
        f"{country}_{mh_level}" for country in COUNTRIES for mh_level in MH_LEVELS
    }


def test_build_scope_specs_keep_country_level_if_mapping_per_scope() -> None:
    scope_specs = build_scope_specs()
    by_scope_key = {(scope.scope_id, scope.mh_level): scope for scope in scope_specs}

    assert by_scope_key[("COMPETITOR_1_COUNTRY_1", "mh5")].country_if_model_name == "COUNTRY_1_mh5"
    assert by_scope_key[("COMPETITOR_2_COUNTRY_3", "mh20")].country_if_model_name == "COUNTRY_3_mh20"
    assert by_scope_key[("COMPETITOR_3_COUNTRY_4", "mh30")].country_if_model_name == "COUNTRY_4_mh30"
    assert by_scope_key[("COMPETITOR_3_COUNTRY_4", "mh30")].candidate_id == "COMPETITOR_3_COUNTRY_4__mh30"


def test_build_dataset_paths_targets_competitor_split_files() -> None:
    competitor_scope = next(
        scope
        for scope in build_scope_specs()
        if scope.scope_id == "COMPETITOR_1_COUNTRY_2" and scope.mh_level == "mh10"
    )
    dataset_paths = build_dataset_paths(competitor_scope)

    expected_root = TEST_ROOT / "data-subsets" / "mh10" / "by_competitor" / "COUNTRY_2" / "B2B"
    assert dataset_paths["train"] == expected_root / f"COMPETITOR_1_COUNTRY_2_{DATASET_SNAPSHOT}_train.parquet"
    assert dataset_paths["test_new_prices"] == (
        expected_root / f"COMPETITOR_1_COUNTRY_2_{DATASET_SNAPSHOT}_test_new_prices.parquet"
    )
    assert dataset_paths["test_new_products"] == (
        expected_root / f"COMPETITOR_1_COUNTRY_2_{DATASET_SNAPSHOT}_test_new_products.parquet"
    )


def test_load_country_if_config_reads_retained_by_country_mh5_configuration() -> None:
    config = _load_country_if_config("COUNTRY_1", "mh5")

    assert config.country == "COUNTRY_1"
    assert config.mh_level == "mh5"
    assert config.model_name == "COUNTRY_1_mh5"
    assert config.n_estimators == 100
    assert config.max_samples == 512
    assert config.max_features == 0.5
    assert config.threshold == 0.3
    assert config.source_json == (
        TEST_ROOT
        / "results"
        / "tuning"
        / "forests"
        / "single_config_optimized_mh5_run"
        / "mh5"
        / "by_country"
        / "COUNTRY_1_COUNTRY_1_2026-02-08"
        / "if"
        / "best_configuration.json"
    )
