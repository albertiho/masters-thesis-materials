from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.training.scripts.train_isolation_forest import extract_features_vectorized
from research.training.scripts.validate_rrcf_mh5_smoke import run_smoke_validation
from src.anomaly.ml.rrcf import RRCFDetector, RRCFDetectorConfig
from src.anomaly.persistence import ModelPersistence


def _price_frame(
    *,
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
                    "competitor_id": "COMPETITOR_1_COUNTRY_1",
                    "competitor_product_id": f"product-{product_idx:03d}",
                    "price": base_price + (observation_idx % 4) * 0.5,
                    "list_price": base_price * 1.2,
                    "first_seen_at": pd.Timestamp(start, tz="UTC") + pd.Timedelta(days=observation_idx),
                    "country": "FI",
                }
            )
    return pd.DataFrame(rows)


def test_run_smoke_validation_loads_rrcf_model_and_runs_fast_tuning(tmp_path: Path) -> None:
    data_path = tmp_path / "data" / "training" / "derived"
    train_path = (
        data_path
        / "by_competitor"
        / "COUNTRY_1"
        / "B2C"
        / "COMPETITOR_1_COUNTRY_1_2026-02-08_train_mh5.parquet"
    )
    test_path = train_path.with_name("COMPETITOR_1_COUNTRY_1_2026-02-08_test_new_prices_mh5.parquet")
    train_path.parent.mkdir(parents=True, exist_ok=True)

    train_df = _price_frame(n_products=12, observations_per_product=8, start="2026-01-01")
    test_df = _price_frame(n_products=12, observations_per_product=2, start="2026-02-01")
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    X_train = extract_features_vectorized(train_df)
    detector = RRCFDetector(
        RRCFDetectorConfig(
            num_trees=16,
            tree_size=64,
            anomaly_threshold=0.8,
            random_state=42,
            warmup_samples=8,
        )
    )
    detector.fit_from_matrix(X_train)

    persistence = ModelPersistence(model_root=tmp_path / "models")
    persistence.save_rrcf(detector, "COMPETITOR_1_COUNTRY_1_mh5", len(X_train))

    output_csv = tmp_path / "results" / "rrcf_smoke.csv"
    results = run_smoke_validation(
        data_path=str(data_path),
        model_root=str(tmp_path / "models"),
        granularity="competitor",
        mh_level="mh5",
        split="new_prices",
        model_limit=1,
        max_products=6,
        max_history_per_product=6,
        min_test_rows=12,
        max_test_rows=20,
        min_threshold=0.2,
        max_threshold=0.9,
        steps=3,
        n_trials=1,
        max_workers=1,
        output_csv=str(output_csv),
    )

    assert len(results) == 1
    assert results[0].status == "ok"
    assert results[0].best_threshold is not None
    assert output_csv.exists()
