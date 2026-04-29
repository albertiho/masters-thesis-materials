from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.research.history_subsets import create_history_subsets


def _write_source_parquet(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "product_id": (
                ["p5"] * 5
                + ["p6"] * 6
                + ["p8"] * 8
            ),
            "competitor_id": ["c1"] * 19,
            "competitor_product_id": [f"cp{index}" for index in range(19)],
            "price": [100.0 + index for index in range(19)],
            "list_price": [120.0 + index for index in range(19)],
            "first_seen_at": pd.date_range("2026-02-01", periods=19, freq="D"),
            "currency": ["EUR"] * 19,
            "country": ["COUNTRY_1"] * 19,
        }
    )
    df.to_parquet(path, index=False)


def test_create_history_subsets_filters_products_by_threshold(tmp_path: Path) -> None:
    source_root = tmp_path / "data" / "training" / "source" / "by_competitor"
    source_file = (
        source_root
        / "COUNTRY_1"
        / "B2B"
        / "COMPETITOR_1_COUNTRY_1_2026-02-08.parquet"
    )
    _write_source_parquet(source_file)

    output_root = tmp_path / "data-subsets"
    result = create_history_subsets(
        source_root=source_root,
        output_root=output_root,
        history_values=[6, 8],
    )

    assert result["files_processed"] == 1
    assert result["generated_subset_files"] == 2

    mh6 = pd.read_parquet(
        output_root
        / "mh6"
        / "by_competitor"
        / "COUNTRY_1"
        / "B2B"
        / "COMPETITOR_1_COUNTRY_1_2026-02-08.parquet"
    )
    mh8 = pd.read_parquet(
        output_root
        / "mh8"
        / "by_competitor"
        / "COUNTRY_1"
        / "B2B"
        / "COMPETITOR_1_COUNTRY_1_2026-02-08.parquet"
    )

    assert set(mh6["product_id"].unique()) == {"p6", "p8"}
    assert set(mh8["product_id"].unique()) == {"p8"}


def test_create_history_subsets_dry_run_does_not_write_files(tmp_path: Path) -> None:
    source_root = tmp_path / "data" / "training" / "source" / "by_competitor"
    source_file = (
        source_root
        / "COUNTRY_1"
        / "B2C"
        / "COMPETITOR_2_COUNTRY_1_2026-02-08.parquet"
    )
    _write_source_parquet(source_file)

    output_root = tmp_path / "data-subsets"
    result = create_history_subsets(
        source_root=source_root,
        output_root=output_root,
        history_values=[6],
        dry_run=True,
    )

    assert result["files_processed"] == 1
    assert result["generated_subset_files"] == 0
    assert not output_root.exists()


def test_create_history_subsets_skip_existing_counts_skipped_files(tmp_path: Path) -> None:
    source_root = tmp_path / "data" / "training" / "source" / "by_competitor"
    source_file = (
        source_root
        / "COUNTRY_1"
        / "B2B"
        / "COMPETITOR_1_COUNTRY_1_2026-02-08.parquet"
    )
    _write_source_parquet(source_file)

    output_root = tmp_path / "data-subsets"
    create_history_subsets(
        source_root=source_root,
        output_root=output_root,
        history_values=[6],
    )

    result = create_history_subsets(
        source_root=source_root,
        output_root=output_root,
        history_values=[6],
        skip_existing=True,
    )

    assert result["files_processed"] == 1
    assert result["generated_subset_files"] == 0
    assert result["skipped_subset_files"] == 1
