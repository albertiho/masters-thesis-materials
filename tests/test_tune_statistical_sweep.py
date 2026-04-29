from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pyarrow import parquet as pq

from research.training.scripts import tune_statistical as stat_tuner
from src.anomaly.statistical import (
    HybridAvgZScoreDetector,
    HybridMaxZScoreDetector,
    HybridWeightedZScoreDetector,
    IQRDetector,
    ModifiedMADDetector,
    ModifiedSNDetector,
    ThresholdDetector,
    ZScoreDetector,
)
from src.features.temporal import TemporalCacheManager


def _make_train_frame(*, n_products: int = 4, n_history: int = 6, competitor_id: str = "COMP_A") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    base_time = pd.Timestamp("2026-02-01")
    for product_index in range(n_products):
        product_id = f"prod_{product_index}"
        base_price = 100.0 + (25.0 * product_index)
        for history_index in range(n_history):
            rows.append(
                {
                    "product_id": product_id,
                    "competitor_id": competitor_id,
                    "price": base_price + ((history_index % 3) - 1) * 2.5 + (history_index * 0.4),
                    "first_seen_at": base_time + pd.Timedelta(days=history_index),
                }
            )
    return pd.DataFrame(rows)


def _make_test_frame(train_df: pd.DataFrame, *, n_future: int = 3) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    max_time = pd.to_datetime(train_df["first_seen_at"]).max()
    grouped = train_df.groupby(["product_id", "competitor_id"], sort=True)
    for product_id, competitor_id in grouped.groups:
        history = grouped.get_group((product_id, competitor_id)).sort_values("first_seen_at")
        last_price = float(history["price"].iloc[-1])
        for future_index in range(n_future):
            rows.append(
                {
                    "product_id": product_id,
                    "competitor_id": competitor_id,
                    "price": last_price + ((future_index % 2) * 1.75) - 0.5,
                    "first_seen_at": max_time + pd.Timedelta(days=future_index + 1),
                }
            )
    return pd.DataFrame(rows)


def _write_scope(
    *,
    data_subsets_root: Path,
    mh_level: str,
    granularity: str,
    relative_dir: str,
    stem: str,
    train_df: pd.DataFrame | None = None,
    test_prices_df: pd.DataFrame | None = None,
    test_products_df: pd.DataFrame | None = None,
) -> None:
    scope_dir = data_subsets_root / mh_level / granularity / relative_dir
    scope_dir.mkdir(parents=True, exist_ok=True)
    effective_train = train_df if train_df is not None else _make_train_frame()
    effective_prices = test_prices_df if test_prices_df is not None else _make_test_frame(effective_train)
    effective_products = test_products_df if test_products_df is not None else _make_test_frame(effective_train)

    effective_train.to_parquet(scope_dir / f"{stem}_train.parquet", index=False)
    if test_prices_df is not False:
        effective_prices.to_parquet(scope_dir / f"{stem}_test_new_prices.parquet", index=False)
    if test_products_df is not False:
        effective_products.to_parquet(scope_dir / f"{stem}_test_new_products.parquet", index=False)


def _build_scope_descriptor(tmp_path: Path) -> stat_tuner.ScopeDescriptor:
    data_root = tmp_path / "data-subsets"
    cache_root = tmp_path / "cache"
    _write_scope(
        data_subsets_root=data_root,
        mh_level="mh5",
        granularity="global",
        relative_dir=".",
        stem="GLOBAL_2026-02-08",
    )
    scopes, skipped = stat_tuner.discover_scopes(
        data_subsets_root=data_root,
        cache_root=cache_root,
    )
    assert not skipped
    assert len(scopes) == 1
    return scopes[0]


def _dummy_context() -> stat_tuner.CachedScopeContext:
    empty_frame = pd.DataFrame(
        {
            "product_id": [],
            "competitor_id": [],
            "price": [],
            "first_seen_at": [],
        }
    )
    scope = stat_tuner.ScopeDescriptor(
        mh_level="mh5",
        granularity="global",
        scope_id="GLOBAL_2026-02-08",
        dataset_name="GLOBAL",
        train_path=Path("train.parquet"),
        test_new_prices_path=Path("prices.parquet"),
        test_new_products_path=Path("products.parquet"),
        cache_snapshot_path=Path("template_cache.joblib"),
    )
    attempts_prices = [
        stat_tuner.SplitAttempt("new_prices", index, 41 + index, empty_frame, np.zeros(0, dtype=bool), [])
        for index in range(1, 6)
    ]
    attempts_products = [
        stat_tuner.SplitAttempt("new_products", index, 51 + index, empty_frame, np.zeros(0, dtype=bool), [])
        for index in range(1, 6)
    ]
    return stat_tuner.CachedScopeContext(
        scope=scope,
        train_df=pd.DataFrame({"x": [1, 2, 3]}),
        template_cache=TemporalCacheManager(),
        new_prices_attempts=attempts_prices,
        new_products_attempts=attempts_products,
        country=None,
    )


def _find_small_real_scope() -> tuple[Path, Path, Path] | None:
    repo_root = Path(__file__).resolve().parents[1]
    mh_root = repo_root / "data-subsets" / "mh5"
    if not mh_root.exists():
        return None

    smallest: tuple[int, Path] | None = None
    for train_path in mh_root.rglob("*_train.parquet"):
        test_prices = train_path.with_name(train_path.name.replace("_train.parquet", "_test_new_prices.parquet"))
        test_products = train_path.with_name(train_path.name.replace("_train.parquet", "_test_new_products.parquet"))
        if not test_prices.exists() or not test_products.exists():
            continue
        row_count = pq.ParquetFile(train_path).metadata.num_rows
        if smallest is None or row_count < smallest[0]:
            smallest = (row_count, train_path)

    if smallest is None:
        return None

    train_path = smallest[1]
    return (
        train_path,
        train_path.with_name(train_path.name.replace("_train.parquet", "_test_new_prices.parquet")),
        train_path.with_name(train_path.name.replace("_train.parquet", "_test_new_products.parquet")),
    )


def test_discover_scopes_pairs_all_granularities_and_skips_missing_pairs(tmp_path: Path) -> None:
    data_root = tmp_path / "data-subsets"
    cache_root = tmp_path / "cache"

    _write_scope(
        data_subsets_root=data_root,
        mh_level="mh5",
        granularity="global",
        relative_dir=".",
        stem="GLOBAL_2026-02-08",
    )
    _write_scope(
        data_subsets_root=data_root,
        mh_level="mh5",
        granularity="by_country",
        relative_dir="COUNTRY_1",
        stem="COUNTRY_1_2026-02-08",
    )
    _write_scope(
        data_subsets_root=data_root,
        mh_level="mh5",
        granularity="by_country_market",
        relative_dir="COUNTRY_1/B2C",
        stem="COUNTRY_1_B2C_2026-02-08",
    )
    _write_scope(
        data_subsets_root=data_root,
        mh_level="mh5",
        granularity="by_competitor",
        relative_dir="COUNTRY_1/B2C",
        stem="COMPETITOR_1_COUNTRY_1_2026-02-08",
    )
    _write_scope(
        data_subsets_root=data_root,
        mh_level="mh5",
        granularity="by_country",
        relative_dir="COUNTRY_2",
        stem="COUNTRY_2_2026-02-08",
        test_products_df=False,
    )

    scopes, skipped = stat_tuner.discover_scopes(
        data_subsets_root=data_root,
        cache_root=cache_root,
    )

    discovered = {(scope.granularity, scope.dataset_name) for scope in scopes}

    assert discovered == {
        ("global", "GLOBAL"),
        ("by_country", "COUNTRY_1"),
        ("by_country_market", "COUNTRY_1_B2C"),
        ("by_competitor", "COMPETITOR_1_COUNTRY_1"),
    }
    assert skipped[0]["granularity"] == "by_country"
    assert skipped[0]["dataset_name"] == "COUNTRY_2"
    assert skipped[0]["reason"] == "missing_test_pair"


def test_build_parser_defaults_to_sampled_mh_values() -> None:
    args = stat_tuner.build_parser().parse_args([])

    assert args.mh_values == list(stat_tuner.DEFAULT_SAMPLED_MH_VALUES)


def test_run_sweep_defaults_to_sampled_mh_values_only(tmp_path: Path) -> None:
    data_root = tmp_path / "data-subsets"
    cache_root = tmp_path / "cache"
    output_root = tmp_path / "results" / "tuning" / "statistical" / "default_sampled_mh"

    for mh_level in ("mh5", "mh6", "mh10"):
        _write_scope(
            data_subsets_root=data_root,
            mh_level=mh_level,
            granularity="global",
            relative_dir=".",
            stem="GLOBAL_2026-02-08",
        )

    args = stat_tuner.build_parser().parse_args(
        [
            "--data-subsets-root",
            str(data_root),
            "--cache-root",
            str(cache_root),
            "--output-root",
            str(output_root),
            "--attempts",
            "1",
            "--detectors",
            "standard_zscore",
            "--max-workers",
            "1",
        ]
    )

    assert stat_tuner.run_sweep(args) == 0

    best_configurations = pd.read_csv(output_root / "best_configurations.csv")
    scope_status = pd.read_csv(output_root / "scope_status.csv")

    assert set(best_configurations["mh_level"].astype(str)) == {"mh5", "mh10"}
    assert set(scope_status["mh_level"].astype(str)) == {"mh5", "mh10"}
    assert "mh6" not in {path.parent.parent.parent.name for path in output_root.rglob("best_configuration.json")}


def test_cache_snapshot_round_trip_and_copy_from_clone(tmp_path: Path) -> None:
    scope = _build_scope_descriptor(tmp_path)
    result = stat_tuner.ensure_cache_snapshot(scope=scope)
    metadata = json.loads(scope.cache_metadata_path.read_text(encoding="utf-8"))

    assert result["status"] == "built"
    assert metadata["row_count"] > 0
    assert metadata["product_count"] > 0

    expected_train = pd.read_parquet(scope.train_path)
    expected_cache = stat_tuner._build_template_cache(expected_train)
    loaded_cache = TemporalCacheManager()
    loaded_cache.load_from_file(str(scope.cache_snapshot_path))

    assert loaded_cache.get_stats() == expected_cache.get_stats()

    entry = loaded_cache.get("prod_0", "COMP_A")
    assert entry is not None
    assert len(entry.price_history) == 6

    cloned_cache = TemporalCacheManager()
    cloned_cache.copy_from(loaded_cache)
    cloned_cache.update_if_changed(
        product_id="prod_0",
        competitor_id="COMP_A",
        price=999.0,
        scraped_at=pd.Timestamp("2026-03-01").to_pydatetime(),
    )

    assert loaded_cache.get("prod_0", "COMP_A").last_price != cloned_cache.get("prod_0", "COMP_A").last_price


def test_cache_snapshot_metadata_mismatch_triggers_rebuild(tmp_path: Path) -> None:
    scope = _build_scope_descriptor(tmp_path)
    stat_tuner.ensure_cache_snapshot(scope=scope)
    metadata_before = json.loads(scope.cache_metadata_path.read_text(encoding="utf-8"))

    train_df = pd.read_parquet(scope.train_path)
    mutated = pd.concat(
        [
            train_df,
            pd.DataFrame(
                [
                    {
                        "product_id": "prod_extra",
                        "competitor_id": "COMP_A",
                        "price": 123.45,
                        "first_seen_at": pd.Timestamp("2026-03-10"),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    mutated.to_parquet(scope.train_path, index=False)

    result = stat_tuner.ensure_cache_snapshot(scope=scope)
    metadata_after = json.loads(scope.cache_metadata_path.read_text(encoding="utf-8"))

    assert result["status"] == "rebuilt"
    assert metadata_after["row_count"] == metadata_before["row_count"] + 1
    assert metadata_after["mtime_ns"] != metadata_before["mtime_ns"]


def test_detector_grids_match_expected_coarse_and_refine_candidates() -> None:
    coarse = stat_tuner.build_coarse_candidates("standard_zscore")
    assert [candidate.params["threshold"] for candidate in coarse] == [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

    modified_mad_coarse = stat_tuner.build_coarse_candidates("modified_mad")
    assert [candidate.params["threshold"] for candidate in modified_mad_coarse] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    refine = stat_tuner.build_refined_candidates("modified_mad", {"threshold": 2.0})
    assert [candidate.params["threshold"] for candidate in refine] == [1.5, 1.75, 2.0, 2.25, 2.5]

    clipped_refine = stat_tuner.build_refined_candidates("modified_mad", {"threshold": 6.0})
    assert [candidate.params["threshold"] for candidate in clipped_refine] == [5.5, 5.75, 6.0]

    hybrid_weighted_coarse = stat_tuner.build_coarse_candidates("hybrid_weighted")
    assert sorted({candidate.params["w"] for candidate in hybrid_weighted_coarse}) == [0.25, 0.75]


@pytest.mark.parametrize(
    ("detector_family", "params", "expected_type", "expected_fields"),
    [
        ("standard_zscore", {"threshold": 3.0}, ZScoreDetector, {"threshold": 3.0}),
        ("modified_mad", {"threshold": 2.0}, ModifiedMADDetector, {"threshold": 2.0}),
        ("modified_sn", {"threshold": 2.0}, ModifiedSNDetector, {"threshold": 2.0}),
        (
            "hybrid_weighted",
            {"threshold": 2.25, "w": 0.65},
            HybridWeightedZScoreDetector,
            {"threshold": 2.25, "w": 0.65},
        ),
        ("hybrid_max", {"threshold": 2.0}, HybridMaxZScoreDetector, {"threshold": 2.0}),
        ("hybrid_avg", {"threshold": 2.0}, HybridAvgZScoreDetector, {"threshold": 2.0}),
        ("iqr", {"multiplier": 1.5}, IQRDetector, {"multiplier": 1.5}),
        (
            "threshold",
            {"price_change_threshold": 0.2},
            ThresholdDetector,
            {"threshold": 0.2},
        ),
    ],
)
def test_build_detector_returns_expected_detector_instance_and_params(
    detector_family: str,
    params: dict[str, float],
    expected_type: type,
    expected_fields: dict[str, float],
) -> None:
    detector = stat_tuner._build_detector(detector_family, params)

    assert isinstance(detector, expected_type)
    assert hasattr(detector, "detect_batch")
    for field_name, expected_value in expected_fields.items():
        assert getattr(detector, field_name) == pytest.approx(expected_value)


def test_hybrid_weighted_refinement_collapses_to_best_coarse_pair() -> None:
    refine = stat_tuner.build_refined_candidates("hybrid_weighted", {"threshold": 2.0, "w": 0.75})

    thresholds = sorted({candidate.params["threshold"] for candidate in refine})
    weights = sorted({candidate.params["w"] for candidate in refine})

    assert thresholds == [2.0]
    assert weights == [0.75]
    assert len(refine) == 1


def test_candidate_aggregation_uses_attempt_averages_and_rank_score() -> None:
    context = _dummy_context()
    candidate = stat_tuner.CandidateSpec(
        detector_family="standard_zscore",
        candidate_id="standard_zscore__threshold_3p000",
        stage="coarse",
        params={"threshold": 3.0},
    )
    detector_metrics = pd.DataFrame(
        [
            {
                "dataset_split": split_name,
                "attempt_index": attempt_index,
                "accuracy": 0.8,
                "precision": 0.5 if split_name == "new_prices" else 0.4,
                "recall": 0.6 if split_name == "new_prices" else 0.3,
                "tnr": 0.9,
                "f1": 0.55 if split_name == "new_prices" else 0.34,
                "g_mean": value,
            }
            for split_name, values in {
                "new_prices": [0.70, 0.75, 0.80, 0.85, 0.90],
                "new_products": [0.40, 0.45, 0.50, 0.55, 0.60],
            }.items()
            for attempt_index, value in enumerate(values, start=1)
        ]
    )

    row = stat_tuner._build_candidate_row(
        sweep_id="statistical_test",
        context=context,
        candidate=candidate,
        detector_metrics=detector_metrics,
        elapsed=1.25,
    )

    assert row["attempt_count"] == 5
    assert row["new_prices_g_mean_mean"] == pytest.approx(np.mean([0.70, 0.75, 0.80, 0.85, 0.90]))
    assert row["new_prices_g_mean_std"] == pytest.approx(np.std([0.70, 0.75, 0.80, 0.85, 0.90], ddof=0))
    assert row["new_products_g_mean_mean"] == pytest.approx(np.mean([0.40, 0.45, 0.50, 0.55, 0.60]))
    assert row["rank_score"] == pytest.approx(
        0.7 * np.mean([0.70, 0.75, 0.80, 0.85, 0.90]) + 0.3 * np.mean([0.40, 0.45, 0.50, 0.55, 0.60])
    )


def test_candidate_tie_break_order_is_deterministic() -> None:
    rows = pd.DataFrame(
        [
            {
                "candidate_id": "candidate_b",
                "rank_score": 0.8,
                "weighted_f1_mean": 0.7,
                "default_distance": 0.2,
                "status": "ok",
            },
            {
                "candidate_id": "candidate_a",
                "rank_score": 0.8,
                "weighted_f1_mean": 0.7,
                "default_distance": 0.2,
                "status": "ok",
            },
            {
                "candidate_id": "candidate_closer",
                "rank_score": 0.8,
                "weighted_f1_mean": 0.7,
                "default_distance": 0.1,
                "status": "ok",
            },
        ]
    )

    best = stat_tuner.select_best_candidate(rows)

    assert best is not None
    assert best["candidate_id"] == "candidate_closer"


def test_promotion_rule_keeps_top_two_and_anything_within_five_percent() -> None:
    rows = pd.DataFrame(
        [
            {
                "candidate_id": "candidate_a",
                "rank_score": 0.80,
                "weighted_f1_mean": 0.70,
                "default_distance": 0.20,
                "status": "ok",
            },
            {
                "candidate_id": "candidate_b",
                "rank_score": 0.79,
                "weighted_f1_mean": 0.69,
                "default_distance": 0.20,
                "status": "ok",
            },
            {
                "candidate_id": "candidate_c",
                "rank_score": 0.77,
                "weighted_f1_mean": 0.68,
                "default_distance": 0.20,
                "status": "ok",
            },
            {
                "candidate_id": "candidate_d",
                "rank_score": 0.70,
                "weighted_f1_mean": 0.80,
                "default_distance": 0.10,
                "status": "ok",
            },
            {
                "candidate_id": "candidate_error",
                "rank_score": np.nan,
                "weighted_f1_mean": np.nan,
                "default_distance": 0.10,
                "status": "error",
            },
        ]
    )

    promoted = stat_tuner.select_promoted_candidate_ids(rows)

    assert promoted == {"candidate_a", "candidate_b", "candidate_c"}


def test_select_best_candidate_ignores_screening_only_rows() -> None:
    rows = pd.DataFrame(
        [
            {
                "candidate_id": "screened_candidate",
                "rank_score": 0.95,
                "weighted_f1_mean": 0.95,
                "default_distance": 0.30,
                "status": "screened",
            },
            {
                "candidate_id": "promoted_candidate",
                "rank_score": 0.82,
                "weighted_f1_mean": 0.81,
                "default_distance": 0.20,
                "status": "ok",
            },
        ]
    )

    best = stat_tuner.select_best_candidate(rows)

    assert best is not None
    assert best["candidate_id"] == "promoted_candidate"


def test_screening_shortlist_marks_non_promoted_candidates_as_screened() -> None:
    empty_metrics = pd.DataFrame()
    screened_candidate = stat_tuner.CandidateSpec(
        detector_family="standard_zscore",
        candidate_id="screened_candidate",
        stage="coarse",
        params={"threshold": 2.5},
    )
    promoted_candidate = stat_tuner.CandidateSpec(
        detector_family="standard_zscore",
        candidate_id="promoted_candidate",
        stage="coarse",
        params={"threshold": 3.0},
    )

    screening = [
        stat_tuner.EvaluatedCandidate(
            candidate=screened_candidate,
            row={"candidate_id": "screened_candidate", "status": "ok", "rank_score": 0.70},
            detector_metrics=empty_metrics,
            anomaly_type_metrics=empty_metrics,
        ),
        stat_tuner.EvaluatedCandidate(
            candidate=promoted_candidate,
            row={"candidate_id": "promoted_candidate", "status": "ok", "rank_score": 0.82},
            detector_metrics=empty_metrics,
            anomaly_type_metrics=empty_metrics,
        ),
    ]
    full = {
        "promoted_candidate": stat_tuner.EvaluatedCandidate(
            candidate=promoted_candidate,
            row={"candidate_id": "promoted_candidate", "status": "ok", "rank_score": 0.80, "attempt_count": 5},
            detector_metrics=empty_metrics,
            anomaly_type_metrics=empty_metrics,
        )
    }

    shortlisted = stat_tuner._screening_shortlist_results(screening, full)

    assert shortlisted[0].row["status"] == "screened"
    assert shortlisted[0].row["promotion_status"] == "screening_only"
    assert shortlisted[1].row["status"] == "ok"
    assert shortlisted[1].row["promotion_status"] == "promoted"
    assert shortlisted[1].row["attempt_count"] == 5


def test_detector_family_artifacts_are_written_with_expected_columns(tmp_path: Path) -> None:
    scope = _build_scope_descriptor(tmp_path)
    stat_tuner.ensure_cache_snapshot(scope=scope)
    output_root = tmp_path / "results" / "tuning" / "statistical" / "unit_sweep"
    task = stat_tuner.DetectorFamilyTask(
        scope=scope,
        detector_family="standard_zscore",
        family_output_dir=output_root / scope.mh_level / scope.granularity / scope.scope_slug / "standard_zscore",
        sweep_id="unit_sweep",
        attempt_seeds=(42, 43),
    )

    result = stat_tuner.run_detector_family_task(task)

    assert result.status == "ok"
    candidate_metrics = pd.read_csv(task.family_output_dir / "candidate_metrics.csv")
    detector_metrics = pd.read_csv(task.family_output_dir / "metrics" / "detector_metrics.csv")
    progress = json.loads((task.family_output_dir / "progress.json").read_text(encoding="utf-8"))

    assert {"rank_score", "attempt_count", "new_prices_g_mean_mean", "new_products_g_mean_mean", "threshold"}.issubset(
        set(candidate_metrics.columns)
    )
    assert "promotion_status" in candidate_metrics.columns
    assert set(candidate_metrics["status"].astype(str)).issubset({"ok", "screened", "error"})
    assert progress["status"] == "complete"
    assert progress["completed_stage"] == "finalized"
    assert {"attempt_index", "attempt_seed", "candidate_id", "dataset_split", "g_mean"}.issubset(
        set(detector_metrics.columns)
    )
    assert (task.family_output_dir / "best_candidate" / "splits" / "new_prices" / "predictions.parquet").exists()
    assert (task.family_output_dir / "best_candidate" / "splits" / "new_products" / "predictions.parquet").exists()


def test_run_sweep_writes_statistical_configuration_with_all_detector_winners(tmp_path: Path) -> None:
    data_root = tmp_path / "data-subsets"
    cache_root = tmp_path / "cache"
    output_root = tmp_path / "results" / "tuning" / "statistical" / "synthetic_sweep"

    _write_scope(
        data_subsets_root=data_root,
        mh_level="mh5",
        granularity="global",
        relative_dir=".",
        stem="GLOBAL_2026-02-08",
    )
    scope = stat_tuner.discover_scopes(data_subsets_root=data_root, cache_root=cache_root)[0][0]

    args = stat_tuner.build_parser().parse_args(
        [
            "--data-subsets-root",
            str(data_root),
            "--cache-root",
            str(cache_root),
            "--output-root",
            str(output_root),
            "--attempts",
            "1",
            "--max-workers",
            "1",
        ]
    )
    exit_code = stat_tuner.run_sweep(args)

    assert exit_code == 0

    configuration = json.loads(
        (output_root / scope.mh_level / scope.granularity / scope.scope_slug / "statistical_configuration.json").read_text(
            encoding="utf-8"
        )
    )
    best_configurations = pd.read_csv(output_root / "best_configurations.csv")

    assert set(configuration["detectors"]) == set(stat_tuner.ALL_DETECTOR_FAMILIES)
    assert len(best_configurations) == len(stat_tuner.ALL_DETECTOR_FAMILIES)


def test_smoke_resume_reuses_cache_snapshot_and_outputs_on_real_scope(tmp_path: Path) -> None:
    real_scope = _find_small_real_scope()
    if real_scope is None:
        pytest.skip("No real mh5 scope available for smoke test")

    train_path, test_prices_path, test_products_path = real_scope
    repo_root = Path(__file__).resolve().parents[1]
    relative_train = train_path.relative_to(repo_root / "data-subsets")
    relative_prices = test_prices_path.relative_to(repo_root / "data-subsets")
    relative_products = test_products_path.relative_to(repo_root / "data-subsets")

    copied_root = tmp_path / "copied-data-subsets"
    (copied_root / relative_train.parent).mkdir(parents=True, exist_ok=True)
    shutil.copy2(train_path, copied_root / relative_train)
    shutil.copy2(test_prices_path, copied_root / relative_prices)
    shutil.copy2(test_products_path, copied_root / relative_products)

    cache_root = tmp_path / "cache"
    output_root = tmp_path / "results" / "tuning" / "statistical" / "resume_smoke"
    scopes, skipped = stat_tuner.discover_scopes(data_subsets_root=copied_root, cache_root=cache_root)
    assert not skipped
    assert len(scopes) == 1
    scope = scopes[0]

    first_args = stat_tuner.build_parser().parse_args(
        [
            "--data-subsets-root",
            str(copied_root),
            "--cache-root",
            str(cache_root),
            "--output-root",
            str(output_root),
            "--attempts",
            "1",
            "--detectors",
            "standard_zscore",
            "--max-workers",
            "1",
        ]
    )
    second_args = stat_tuner.build_parser().parse_args(
        [
            "--data-subsets-root",
            str(copied_root),
            "--cache-root",
            str(cache_root),
            "--output-root",
            str(output_root),
            "--attempts",
            "1",
            "--detectors",
            "standard_zscore",
            "--max-workers",
            "1",
            "--resume",
        ]
    )

    assert stat_tuner.run_sweep(first_args) == 0

    family_dir = output_root / scope.mh_level / scope.granularity / scope.scope_slug / "standard_zscore"
    cache_mtime_before = scope.cache_snapshot_path.stat().st_mtime_ns
    metrics_mtime_before = (family_dir / "candidate_metrics.csv").stat().st_mtime_ns

    assert stat_tuner.run_sweep(second_args) == 0
    assert scope.cache_snapshot_path.stat().st_mtime_ns == cache_mtime_before
    assert (family_dir / "candidate_metrics.csv").stat().st_mtime_ns == metrics_mtime_before


def test_partial_family_resume_reuses_saved_stage_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scope = _build_scope_descriptor(tmp_path)
    stat_tuner.ensure_cache_snapshot(scope=scope)
    output_root = tmp_path / "results" / "tuning" / "statistical" / "resume_unit_sweep"
    task = stat_tuner.DetectorFamilyTask(
        scope=scope,
        detector_family="standard_zscore",
        family_output_dir=output_root / scope.mh_level / scope.granularity / scope.scope_slug / "standard_zscore",
        sweep_id="resume_unit_sweep",
        attempt_seeds=(42, 43),
    )

    coarse_candidate_ids = {
        candidate.candidate_id for candidate in stat_tuner.build_coarse_candidates(task.detector_family)
    }
    original_evaluate_candidate = stat_tuner.evaluate_candidate
    interrupted_calls: list[tuple[str, int | None]] = []

    def interrupt_after_coarse_screening(
        *,
        sweep_id: str,
        detector_family: str,
        context: stat_tuner.CachedScopeContext,
        candidate: stat_tuner.CandidateSpec,
        max_attempts: int | None = None,
    ) -> stat_tuner.EvaluatedCandidate:
        interrupted_calls.append((candidate.candidate_id, max_attempts))
        if len(interrupted_calls) > len(coarse_candidate_ids):
            raise RuntimeError("interrupt after coarse screening")
        return original_evaluate_candidate(
            sweep_id=sweep_id,
            detector_family=detector_family,
            context=context,
            candidate=candidate,
            max_attempts=max_attempts,
        )

    monkeypatch.setattr(stat_tuner, "evaluate_candidate", interrupt_after_coarse_screening)

    interrupted_result = stat_tuner.run_detector_family_task(task)

    assert interrupted_result.status == "error"

    progress_path = task.family_output_dir / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    screening_metrics = pd.read_csv(task.family_output_dir / "candidate_metrics.csv")

    assert progress["status"] == "error"
    assert progress["completed_stage"] == "coarse_screening_complete"
    assert set(screening_metrics["candidate_id"]) == coarse_candidate_ids
    assert screening_metrics["attempt_count"].eq(1).all()

    resumed_calls: list[tuple[str, int | None]] = []

    def track_resume_calls(
        *,
        sweep_id: str,
        detector_family: str,
        context: stat_tuner.CachedScopeContext,
        candidate: stat_tuner.CandidateSpec,
        max_attempts: int | None = None,
    ) -> stat_tuner.EvaluatedCandidate:
        resumed_calls.append((candidate.candidate_id, max_attempts))
        return original_evaluate_candidate(
            sweep_id=sweep_id,
            detector_family=detector_family,
            context=context,
            candidate=candidate,
            max_attempts=max_attempts,
        )

    monkeypatch.setattr(stat_tuner, "evaluate_candidate", track_resume_calls)

    resumed_result = stat_tuner.run_detector_family_task(task)

    assert resumed_result.status == "ok"
    assert not any(candidate_id in coarse_candidate_ids and max_attempts == 1 for candidate_id, max_attempts in resumed_calls)

    final_progress = json.loads(progress_path.read_text(encoding="utf-8"))
    final_metrics = pd.read_csv(task.family_output_dir / "candidate_metrics.csv")

    assert final_progress["status"] == "complete"
    assert final_progress["completed_stage"] == "finalized"
    assert len(final_metrics) >= len(coarse_candidate_ids)


def test_read_family_csv_treats_whitespace_only_file_as_empty_without_warning(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    blank_csv = tmp_path / "metrics.csv"
    blank_csv.write_text("\n", encoding="utf-8")

    with caplog.at_level("WARNING", logger=stat_tuner.LOGGER.name):
        frame = stat_tuner._read_family_csv(blank_csv)

    assert frame.empty
    assert not caplog.records


def test_load_saved_evaluations_ignores_error_rows(tmp_path: Path) -> None:
    output_dir = tmp_path / "family"
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    candidate_metrics = pd.DataFrame(
        [
            {
                "candidate_id": "standard_zscore__threshold_2p000",
                "status": "error",
                "rank_score": np.nan,
            },
            {
                "candidate_id": "standard_zscore__threshold_2p500",
                "status": "ok",
                "rank_score": 0.8,
            },
        ]
    )
    detector_metrics = pd.DataFrame(
        [
            {
                "candidate_id": "standard_zscore__threshold_2p500",
                "dataset_split": "new_prices",
                "attempt_index": 0,
                "accuracy": 0.9,
                "precision": 0.8,
                "recall": 0.7,
                "tnr": 0.95,
                "f1": 0.75,
                "g_mean": 0.81,
            }
        ]
    )
    anomaly_type_metrics = pd.DataFrame(
        [
            {
                "candidate_id": "standard_zscore__threshold_2p500",
                "dataset_split": "new_prices",
                "anomaly_type": "price_spike",
                "precision": 0.8,
                "recall": 0.7,
                "f1": 0.75,
                "support": 10,
            }
        ]
    )

    candidate_metrics.to_csv(output_dir / "candidate_metrics.csv", index=False)
    detector_metrics.to_csv(metrics_dir / "detector_metrics.csv", index=False)
    anomaly_type_metrics.to_csv(metrics_dir / "anomaly_type_metrics.csv", index=False)

    candidates = [
        stat_tuner.CandidateSpec(
            detector_family="standard_zscore",
            candidate_id="standard_zscore__threshold_2p000",
            stage="coarse",
            params={"threshold": 2.0},
        ),
        stat_tuner.CandidateSpec(
            detector_family="standard_zscore",
            candidate_id="standard_zscore__threshold_2p500",
            stage="coarse",
            params={"threshold": 2.5},
        ),
    ]

    loaded = stat_tuner._load_saved_evaluations(output_dir, candidates)

    assert set(loaded) == {"standard_zscore__threshold_2p500"}
