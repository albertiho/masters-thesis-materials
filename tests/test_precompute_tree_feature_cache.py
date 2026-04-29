from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.training.scripts import precompute_tree_feature_cache as cache_script


def _write_train_parquet(path: Path, *, n_products: int = 4, n_history: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    base_time = pd.Timestamp("2026-02-01")

    for product_index in range(n_products):
        product_id = f"prod_{product_index}"
        base_price = 100.0 + (product_index * 10.0)
        for history_index in range(n_history):
            rows.append(
                {
                    "product_id": product_id,
                    "competitor_id": "COMPETITOR_1_COUNTRY_1",
                    "competitor_product_id": f"{product_id}_{history_index}",
                    "price": base_price + history_index,
                    "list_price": base_price + 20.0,
                    "first_seen_at": base_time + pd.Timedelta(days=history_index),
                }
            )

    pd.DataFrame(rows).to_parquet(path, index=False, compression="snappy")


def test_discover_cache_jobs_defaults_to_research_sampled_mh_levels(tmp_path: Path) -> None:
    data_path = tmp_path / "data" / "training" / "derived"

    _write_train_parquet(
        data_path / "by_country_segment" / "COUNTRY_1_B2C_2026-02-08_train_mh5.parquet"
    )
    _write_train_parquet(
        data_path / "by_country_segment" / "COUNTRY_1_B2C_2026-02-08_train_mh6.parquet"
    )
    _write_train_parquet(
        data_path / "by_country_segment" / "COUNTRY_1_B2C_2026-02-08_train_mh10.parquet"
    )
    _write_train_parquet(
        data_path
        / "by_competitor"
        / "COUNTRY_1"
        / "B2C"
        / "COMPETITOR_1_COUNTRY_1_2026-02-08_train_mh5.parquet"
    )
    _write_train_parquet(
        data_path
        / "by_competitor"
        / "COUNTRY_1"
        / "B2C"
        / "COMPETITOR_1_COUNTRY_1_2026-02-08_train_mh10.parquet"
    )
    _write_train_parquet(
        data_path / "global" / "GLOBAL_2026-02-08_train_mh5.parquet"
    )
    _write_train_parquet(
        data_path / "global" / "GLOBAL_2026-02-08_train_mh10.parquet"
    )

    jobs = cache_script.discover_cache_jobs(data_path=str(data_path))

    assert {(job.mh_level, job.granularity) for job in jobs} == {
        ("mh5", "country_segment"),
        ("mh10", "country_segment"),
        ("mh5", "competitor"),
        ("mh10", "competitor"),
        ("mh5", "global"),
        ("mh10", "global"),
    }
    assert all("mh6" not in job.filepath for job in jobs)


def test_run_precompute_builds_and_reuses_tree_feature_caches(tmp_path: Path) -> None:
    data_path = tmp_path / "data" / "training" / "derived"
    train_file = (
        data_path / "by_country_segment" / "COUNTRY_1_B2C_2026-02-08_train_mh5.parquet"
    )
    _write_train_parquet(train_file)

    first = cache_script.run_precompute(
        data_path=str(data_path),
        mh_values=["mh5"],
        granularity="country_segment",
    )
    second = cache_script.run_precompute(
        data_path=str(data_path),
        mh_values=["mh5"],
        granularity="country_segment",
    )

    cache_path = Path(first["results"][0]["cache_path"])
    assert cache_path.exists()
    assert first["built"] == 1
    assert first["reused"] == 0
    assert second["built"] == 0
    assert second["reused"] == 1
