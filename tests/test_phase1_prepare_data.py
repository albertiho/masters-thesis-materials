from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import IsolationForest

from src.anomaly.ml.isolation_forest import IsolationForestConfig, IsolationForestDetector
from src.anomaly.persistence import ModelPersistence
from src.research.prepare_data import prepare_dataset, write_dataset_contract_files


def _write_tuning_config(config_path: Path) -> None:
    payload = {
        "minimum_history": {
            "autoencoder": 4,
            "isolation_forest": 5,
            "statistical": 3,
        },
        "data_splitting": {
            "test_size": 0.2,
            "test_split_amount_of_prices": 2,
            "random_state": 42,
        },
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sample_frame(product_prefix: str, competitor_id: str, country: str, rows_per_product: int = 6) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for product_idx in range(10):
        for observation_idx in range(rows_per_product):
            rows.append(
                {
                    "product_id": f"{product_prefix}_{product_idx:03d}",
                    "competitor_id": competitor_id,
                    "competitor_product_id": f"{competitor_id}_{product_idx:03d}",
                    "price": 100.0 + product_idx + observation_idx,
                    "list_price": 120.0 + product_idx,
                    "first_seen_at": pd.Timestamp("2026-02-01") + pd.Timedelta(days=observation_idx),
                    "currency": "EUR",
                    "country": country,
                }
            )
    return pd.DataFrame(rows)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="snappy")


def _build_fixture_tree(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "data" / "training"
    source_root = data_root / "source"
    config_path = tmp_path / "configs" / "tuning_config.json"
    _write_tuning_config(config_path)

    _write_parquet(
        source_root / "by_competitor" / "COUNTRY_7" / "B2C" / "COMPETITOR_1_COUNTRY_7_2026-02-08.parquet",
        _sample_frame("country7_comp1", "COMPETITOR_1_COUNTRY_7", "COUNTRY_7"),
    )
    _write_parquet(
        source_root / "by_competitor" / "COUNTRY_7" / "B2C" / "COMPETITOR_2_COUNTRY_7_2026-02-08.parquet",
        _sample_frame("country7_comp2", "COMPETITOR_2_COUNTRY_7", "COUNTRY_7"),
    )
    _write_parquet(
        source_root / "by_competitor" / "COUNTRY_10" / "B2B" / "COMPETITOR_3_COUNTRY_10_2026-02-08.parquet",
        _sample_frame("country10_comp3", "COMPETITOR_3_COUNTRY_10", "COUNTRY_10"),
    )

    write_dataset_contract_files(source_root, config_path=str(config_path))
    return data_root, config_path


def _derived_inventory(derived_root: Path) -> dict[str, int]:
    return {
        path.relative_to(derived_root).as_posix(): pq.ParquetFile(path).metadata.num_rows
        for path in sorted(derived_root.rglob("*.parquet"))
    }


def test_prepare_dataset_generates_identifier_neutral_inventory(tmp_path: Path) -> None:
    data_root, config_path = _build_fixture_tree(tmp_path)

    result = prepare_dataset(data_root=data_root, config_path=str(config_path))

    derived_root = data_root / "derived"
    assert result["source_files"] == 3
    assert result["country_segment_files"] == 2
    assert result["split_files"] == 30

    expected_combined = derived_root / "by_country_segment" / "COUNTRY_7_B2C_2026-02-08.parquet"
    assert expected_combined.exists()
    assert pq.ParquetFile(expected_combined).metadata.num_rows == 120

    manifest = json.loads((derived_root / "split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["split_variants"] == [4, 5]
    assert "by_country_segment/COUNTRY_7_B2C_2026-02-08.parquet" in manifest["generated_files"]["by_country_segment"]

    serialized_manifest = json.dumps(manifest, sort_keys=True)
    assert "DK_" not in serialized_manifest
    assert "NO_" not in serialized_manifest
    assert "SE_" not in serialized_manifest
    assert "FI_" not in serialized_manifest


def test_prepare_dataset_is_deterministic_for_inventory_and_split_manifest(tmp_path: Path) -> None:
    data_root, config_path = _build_fixture_tree(tmp_path)

    prepare_dataset(data_root=data_root, config_path=str(config_path))
    derived_root = data_root / "derived"
    first_manifest = (derived_root / "split_manifest.json").read_text(encoding="utf-8")
    first_inventory = _derived_inventory(derived_root)

    prepare_dataset(data_root=data_root, config_path=str(config_path))
    second_manifest = (derived_root / "split_manifest.json").read_text(encoding="utf-8")
    second_inventory = _derived_inventory(derived_root)

    assert first_manifest == second_manifest
    assert first_inventory == second_inventory


def test_model_persistence_defaults_to_local_round_trip(tmp_path: Path) -> None:
    persistence = ModelPersistence(model_root=tmp_path / "models")

    X = np.random.default_rng(42).normal(size=(128, 12))
    model = IsolationForest(
        n_estimators=25,
        max_samples=64,
        contamination="auto",
        max_features=1.0,
        random_state=42,
        n_jobs=1,
    )
    model.fit(X)

    detector = IsolationForestDetector(
        IsolationForestConfig(
            n_estimators=25,
            max_samples=64,
            contamination="auto",
            max_features=1.0,
            random_state=42,
            anomaly_threshold=0.4,
        )
    )
    detector._model = model
    detector._feature_names = [f"feature_{idx}" for idx in range(12)]
    detector._is_fitted = True
    train_scores = model.decision_function(X)
    detector._score_offset = float(train_scores.max())
    detector._score_scale = float(np.ptp(train_scores)) or 1.0

    storage_uri = persistence.save_isolation_forest(detector, "COUNTRY_7_B2C_mh5", n_samples=len(X))

    assert persistence.model_exists("COUNTRY_7_B2C_mh5", "isolation_forest")
    assert str(tmp_path / "models") in storage_uri

    loaded = persistence.load_isolation_forest("COUNTRY_7_B2C_mh5")
    assert loaded._is_fitted is True
    assert loaded._feature_names == detector._feature_names
    assert loaded.config.anomaly_threshold == detector.config.anomaly_threshold
