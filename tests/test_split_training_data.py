from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.training.scripts import split_training_data


def _write_source_parquet(path: Path, product_prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for product_idx in range(10):
        for observation_idx in range(6):
            rows.append(
                {
                    "product_id": f"{product_prefix}_{product_idx:03d}",
                    "competitor_id": "COMPETITOR_1_COUNTRY_1",
                    "competitor_product_id": f"{product_prefix}_{product_idx:03d}",
                    "price": 100.0 + product_idx + observation_idx,
                    "list_price": 120.0 + product_idx,
                    "first_seen_at": pd.Timestamp("2026-02-01") + pd.Timedelta(days=observation_idx),
                    "currency": "EUR",
                    "country": "COUNTRY_1",
                }
            )

    pd.DataFrame(rows).to_parquet(path, index=False, compression="snappy")


def _assert_split_outputs(
    source_file: Path,
    *,
    train_rows: int = 32,
    test_new_products_rows: int = 12,
    test_new_prices_rows: int = 16,
    train_products: int = 8,
    test_new_products_count: int = 2,
    test_new_prices_products: int = 8,
) -> None:
    train_path = source_file.with_name(f"{source_file.stem}_train.parquet")
    test_new_products_path = source_file.with_name(f"{source_file.stem}_test_new_products.parquet")
    test_new_prices_path = source_file.with_name(f"{source_file.stem}_test_new_prices.parquet")

    assert train_path.exists()
    assert test_new_products_path.exists()
    assert test_new_prices_path.exists()
    assert "_mh" not in train_path.name
    assert "_mh" not in test_new_products_path.name
    assert "_mh" not in test_new_prices_path.name

    train_df = pd.read_parquet(train_path)
    test_new_products_df = pd.read_parquet(test_new_products_path)
    test_new_prices_df = pd.read_parquet(test_new_prices_path)

    assert len(train_df) == train_rows
    assert len(test_new_products_df) == test_new_products_rows
    assert len(test_new_prices_df) == test_new_prices_rows
    assert train_df["product_id"].nunique() == train_products
    assert test_new_products_df["product_id"].nunique() == test_new_products_count
    assert test_new_prices_df["product_id"].nunique() == test_new_prices_products


def test_default_history_values_use_sampled_research_set(tmp_path: Path) -> None:
    cleaned_root = tmp_path / "cleaned-data" / "training"
    cleaned_root.mkdir(parents=True)
    data_subsets_root = tmp_path / "src" / "data-subsets"
    for mh_level in ("mh5", "mh6", "mh10", "mh15", "mh20", "mh25", "mh30"):
        (data_subsets_root / mh_level).mkdir(parents=True)

    assert split_training_data.default_history_values() == [5, 10, 15, 20, 25, 30]
    assert split_training_data.resolved_filtered_history_values() == [10, 15, 20, 25, 30]

    discovered = split_training_data.discover_dataset_roots(
        cleaned_data_root=cleaned_root,
        data_subsets_root=data_subsets_root,
    )

    assert cleaned_root in discovered
    assert data_subsets_root / "mh5" in discovered
    assert data_subsets_root / "mh10" in discovered
    assert data_subsets_root / "mh30" in discovered
    assert data_subsets_root / "mh6" not in discovered


def test_run_local_data_preparation_creates_mh_subsets_and_suffixless_splits(
    tmp_path: Path,
) -> None:
    cleaned_root = tmp_path / "cleaned-data" / "training"
    data_subsets_root = tmp_path / "src" / "data-subsets"
    cleaned_b2b_file = (
        cleaned_root
        / "by_competitor"
        / "COUNTRY_1"
        / "B2B"
        / "COMPETITOR_1_COUNTRY_1_2026-02-08.parquet"
    )
    cleaned_b2c_file = (
        cleaned_root
        / "by_competitor"
        / "COUNTRY_1"
        / "B2C"
        / "COMPETITOR_2_COUNTRY_1_2026-02-08.parquet"
    )
    _write_source_parquet(cleaned_b2b_file, "cleaned_b2b_product")
    _write_source_parquet(cleaned_b2c_file, "cleaned_b2c_product")

    result = split_training_data.run_local_data_preparation(
        cleaned_data_root=cleaned_root,
        data_subsets_root=data_subsets_root,
        history_values=[6],
    )
    copied_mh5_b2b_file = (
        data_subsets_root
        / "mh5"
        / "by_competitor"
        / "COUNTRY_1"
        / "B2B"
        / "COMPETITOR_1_COUNTRY_1_2026-02-08.parquet"
    )
    copied_mh5_b2c_file = (
        data_subsets_root
        / "mh5"
        / "by_competitor"
        / "COUNTRY_1"
        / "B2C"
        / "COMPETITOR_2_COUNTRY_1_2026-02-08.parquet"
    )
    cleaned_country_file = (
        cleaned_root
        / "by_country"
        / "COUNTRY_1"
        / "COUNTRY_1_2026-02-08.parquet"
    )
    cleaned_country_market_b2b_file = (
        cleaned_root
        / "by_country_market"
        / "COUNTRY_1"
        / "B2B"
        / "COUNTRY_1_B2B_2026-02-08.parquet"
    )
    cleaned_global_file = cleaned_root / "global" / "GLOBAL_2026-02-08.parquet"
    copied_mh5_country_file = (
        data_subsets_root
        / "mh5"
        / "by_country"
        / "COUNTRY_1"
        / "COUNTRY_1_2026-02-08.parquet"
    )
    copied_mh5_country_market_b2b_file = (
        data_subsets_root
        / "mh5"
        / "by_country_market"
        / "COUNTRY_1"
        / "B2B"
        / "COUNTRY_1_B2B_2026-02-08.parquet"
    )
    copied_mh5_global_file = (
        data_subsets_root
        / "mh5"
        / "global"
        / "GLOBAL_2026-02-08.parquet"
    )
    mh6_b2b_file = (
        data_subsets_root
        / "mh6"
        / "by_competitor"
        / "COUNTRY_1"
        / "B2B"
        / "COMPETITOR_1_COUNTRY_1_2026-02-08.parquet"
    )
    mh6_country_file = (
        data_subsets_root
        / "mh6"
        / "by_country"
        / "COUNTRY_1"
        / "COUNTRY_1_2026-02-08.parquet"
    )
    mh6_country_market_b2b_file = (
        data_subsets_root
        / "mh6"
        / "by_country_market"
        / "COUNTRY_1"
        / "B2B"
        / "COUNTRY_1_B2B_2026-02-08.parquet"
    )
    mh6_global_file = data_subsets_root / "mh6" / "global" / "GLOBAL_2026-02-08.parquet"

    assert result["mh5_copy_result"]["files_processed"] == 2
    assert result["mh5_copy_result"]["copied_files"] == 2
    assert result["mh5_copy_result"]["skipped_files"] == 0
    assert result["history_subset_result"]["files_processed"] == 2
    assert result["history_subset_result"]["generated_subset_files"] == 2
    assert result["history_subset_result"]["skipped_subset_files"] == 0
    assert result["dataset_roots_processed"] == 3
    assert result["created_derived_files"] == 12
    assert result["skipped_derived_files"] == 0
    assert result["source_files"] == 18
    assert result["created_source_files"] == 18
    assert result["skipped_source_files"] == 0
    assert result["created_split_files"] == 54

    assert copied_mh5_b2b_file.exists()
    assert copied_mh5_b2c_file.exists()
    assert cleaned_country_file.exists()
    assert cleaned_country_market_b2b_file.exists()
    assert cleaned_global_file.exists()
    assert copied_mh5_country_file.exists()
    assert copied_mh5_country_market_b2b_file.exists()
    assert copied_mh5_global_file.exists()
    assert mh6_b2b_file.exists()
    assert mh6_country_file.exists()
    assert mh6_country_market_b2b_file.exists()
    assert mh6_global_file.exists()

    _assert_split_outputs(cleaned_b2b_file)
    _assert_split_outputs(cleaned_b2c_file)
    _assert_split_outputs(
        cleaned_country_file,
        train_rows=64,
        test_new_products_rows=24,
        test_new_prices_rows=32,
        train_products=16,
        test_new_products_count=4,
        test_new_prices_products=16,
    )
    _assert_split_outputs(cleaned_country_market_b2b_file)
    _assert_split_outputs(
        cleaned_global_file,
        train_rows=64,
        test_new_products_rows=24,
        test_new_prices_rows=32,
        train_products=16,
        test_new_products_count=4,
        test_new_prices_products=16,
    )
    _assert_split_outputs(copied_mh5_b2b_file)
    _assert_split_outputs(copied_mh5_b2c_file)
    _assert_split_outputs(
        copied_mh5_country_file,
        train_rows=64,
        test_new_products_rows=24,
        test_new_prices_rows=32,
        train_products=16,
        test_new_products_count=4,
        test_new_prices_products=16,
    )
    _assert_split_outputs(copied_mh5_country_market_b2b_file)
    _assert_split_outputs(
        copied_mh5_global_file,
        train_rows=64,
        test_new_products_rows=24,
        test_new_prices_rows=32,
        train_products=16,
        test_new_products_count=4,
        test_new_prices_products=16,
    )
    _assert_split_outputs(mh6_b2b_file)
    _assert_split_outputs(
        mh6_country_file,
        train_rows=64,
        test_new_products_rows=24,
        test_new_prices_rows=32,
        train_products=16,
        test_new_products_count=4,
        test_new_prices_products=16,
    )
    _assert_split_outputs(mh6_country_market_b2b_file)
    _assert_split_outputs(
        mh6_global_file,
        train_rows=64,
        test_new_products_rows=24,
        test_new_prices_rows=32,
        train_products=16,
        test_new_products_count=4,
        test_new_prices_products=16,
    )


def test_run_local_data_preparation_skips_existing_subset_and_split_files(tmp_path: Path) -> None:
    cleaned_root = tmp_path / "cleaned-data" / "training"
    cleaned_file = (
        cleaned_root
        / "by_competitor"
        / "COUNTRY_1"
        / "B2B"
        / "COMPETITOR_1_COUNTRY_1_2026-02-08.parquet"
    )
    _write_source_parquet(cleaned_file, "cleaned_product")

    first_result = split_training_data.run_local_data_preparation(
        cleaned_data_root=cleaned_root,
        data_subsets_root=tmp_path / "src" / "data-subsets",
        history_values=[6],
    )
    second_result = split_training_data.run_local_data_preparation(
        cleaned_data_root=cleaned_root,
        data_subsets_root=tmp_path / "src" / "data-subsets",
        history_values=[6],
    )

    assert first_result["mh5_copy_result"]["copied_files"] == 1
    assert first_result["mh5_copy_result"]["skipped_files"] == 0
    assert first_result["created_derived_files"] == 9
    assert first_result["skipped_derived_files"] == 0
    assert first_result["history_subset_result"]["generated_subset_files"] == 1
    assert first_result["history_subset_result"]["skipped_subset_files"] == 0
    assert first_result["created_source_files"] == 12
    assert first_result["skipped_source_files"] == 0

    assert second_result["mh5_copy_result"]["copied_files"] == 0
    assert second_result["mh5_copy_result"]["skipped_files"] == 1
    assert second_result["created_derived_files"] == 0
    assert second_result["skipped_derived_files"] == 9
    assert second_result["history_subset_result"]["generated_subset_files"] == 0
    assert second_result["history_subset_result"]["skipped_subset_files"] == 1
    assert second_result["created_source_files"] == 0
    assert second_result["skipped_source_files"] == 12
    assert second_result["created_split_files"] == 0
