#!/usr/bin/env python3
"""Evaluate final layered-detector finalists on all competitor scopes and mh levels.

This script is intentionally a fixed, one-off research runner:

- evaluates every competitor scope found under the sampled ``mh`` subsets
- uses competitor train/test splits for temporal history and split evaluation
- uses the retained country-level Isolation Forest configuration for each scope's country
- writes per-scope split checkpoints so interrupted runs can resume
- writes thesis-ready metric tables automatically at the end

No CLI is provided on purpose. Edit the module-level constants if the setup
needs to change.
"""

from __future__ import annotations

import gc
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

from train_isolation_forest import extract_features_vectorized, train_from_matrix
from src.anomaly.persistence import ModelPersistence
from src.research.artifacts import comparison_result_to_tables, json_dumps, resolve_git_commit
from src.research.evaluation.test_orchestrator import TestOrchestrator

from research.training.scripts.analyze_if_zscore_layered_combinations import (
    ANOMALY_TYPES,
    DATASET_SNAPSHOT,
    EVALUATION_SPLITS,
    EXPERIMENT_FAMILY,
    INJECTION_RATE,
    INJECTION_SEED,
    INJECTION_STRATEGY,
    MIN_HISTORY,
    DROP_RANGE,
    SPIKE_RANGE,
    SANITY_IF_NAME,
    SANITY_ZSCORE_GT10_IF_NAME,
    SANITY_ZSCORE_GT5_IF_NAME,
    SANITY_ZSCORE_NAME,
    _annotate_injected_frame,
    create_evaluators,
    inject_split_frame,
)
from research.training.scripts.extract_if_zscore_layered_thesis_metrics import (
    run as extract_thesis_metrics,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


COUNTRIES = ("COUNTRY_1", "COUNTRY_2", "COUNTRY_3", "COUNTRY_4")
MH_LEVELS = ("mh5", "mh10", "mh15", "mh20", "mh25", "mh30")
PARALLEL_SCOPE_WORKERS = 6
TRAIN_SPLIT = "train"
FOREST_SWEEP_ROOT = _PROJECT_ROOT / "results" / "tuning" / "forests" / "single_config_optimized_mh5_run"
RUN_ID = "all_competitors_all_countries_country_level_layered_finalists"

FINAL_COMBINATIONS = [
    SANITY_IF_NAME,
    SANITY_ZSCORE_NAME,
    SANITY_ZSCORE_GT10_IF_NAME,
    SANITY_ZSCORE_GT5_IF_NAME,
]


@dataclass(frozen=True)
class ScopeSpec:
    mh_level: str
    scope_id: str
    candidate_id: str
    country: str
    scope_market: str
    dataset_name: str
    dataset_granularity: str
    data_root: Path
    country_if_model_name: str


@dataclass(frozen=True)
class CountryIFConfig:
    mh_level: str
    country: str
    model_name: str
    threshold: float
    n_estimators: int
    max_samples: str | int
    max_features: float
    contamination: str | float
    source_json: Path


@dataclass
class ScopeRunResult:
    scope: ScopeSpec
    scope_root: Path
    resumed: bool = False


def _normalize_split_name(split_name: str) -> str:
    mapping = {
        "test_new_prices": "new_prices",
        "test_new_products": "new_products",
    }
    return mapping.get(split_name, split_name)


def _resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (_PROJECT_ROOT / path).resolve()


def _build_output_root() -> Path:
    return _PROJECT_ROOT / "results" / "detector_combinations" / RUN_ID


def _country_train_path(country: str, mh_level: str) -> Path:
    return (
        _PROJECT_ROOT
        / "data-subsets"
        / mh_level
        / "by_country"
        / country
        / f"{country}_{DATASET_SNAPSHOT}_{TRAIN_SPLIT}.parquet"
    )


def _country_best_configuration_path(country: str, mh_level: str) -> Path:
    return (
        FOREST_SWEEP_ROOT
        / mh_level
        / "by_country"
        / f"{country}_{country}_{DATASET_SNAPSHOT}"
        / "if"
        / "best_configuration.json"
    )


def _load_country_if_config(country: str, mh_level: str) -> CountryIFConfig:
    config_path = _country_best_configuration_path(country, mh_level)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    best = payload["best_candidate"]
    return CountryIFConfig(
        mh_level=mh_level,
        country=country,
        model_name=str(best["model_name"]),
        threshold=float(best["threshold"]),
        n_estimators=int(best["n_estimators"]),
        max_samples=best["max_samples"],
        max_features=float(best["max_features"]),
        contamination=best["contamination"],
        source_json=config_path,
    )


def _discover_competitor_train_files() -> list[Path]:
    files: list[Path] = []
    for mh_level in MH_LEVELS:
        root = _PROJECT_ROOT / "data-subsets" / mh_level / "by_competitor"
        files.extend(root.glob(f"COUNTRY_*/*/*_{DATASET_SNAPSHOT}_{TRAIN_SPLIT}.parquet"))
    return sorted(files)


def _build_scope_spec_from_train_path(train_path: Path) -> ScopeSpec:
    match = re.fullmatch(
        rf"(?P<scope_id>.+)_{re.escape(DATASET_SNAPSHOT)}_{TRAIN_SPLIT}\.parquet",
        train_path.name,
    )
    if match is None:
        raise ValueError(f"Unexpected competitor train filename: {train_path}")

    scope_id = match.group("scope_id")
    mh_level = train_path.parents[3].name
    country = train_path.parents[1].name
    market = train_path.parent.name
    return ScopeSpec(
        mh_level=mh_level,
        scope_id=scope_id,
        candidate_id=f"{scope_id}__{mh_level}",
        country=country,
        scope_market=market,
        dataset_name=scope_id,
        dataset_granularity="competitor",
        data_root=train_path.parent,
        country_if_model_name=f"{country}_{mh_level}",
    )


def build_scope_specs() -> list[ScopeSpec]:
    scopes = [
        _build_scope_spec_from_train_path(train_path)
        for train_path in _discover_competitor_train_files()
        if train_path.parents[1].name in COUNTRIES and train_path.parents[3].name in MH_LEVELS
    ]
    return sorted(
        scopes,
        key=lambda scope: (MH_LEVELS.index(scope.mh_level), scope.country, scope.scope_market, scope.scope_id),
    )


def build_dataset_paths(scope: ScopeSpec) -> dict[str, Path]:
    resolved_root = _resolve_repo_path(scope.data_root)
    return {
        TRAIN_SPLIT: resolved_root / f"{scope.dataset_name}_{DATASET_SNAPSHOT}_{TRAIN_SPLIT}.parquet",
        "test_new_prices": resolved_root / f"{scope.dataset_name}_{DATASET_SNAPSHOT}_test_new_prices.parquet",
        "test_new_products": resolved_root / f"{scope.dataset_name}_{DATASET_SNAPSHOT}_test_new_products.parquet",
    }


def _validate_dataset_paths(dataset_paths: dict[str, Path]) -> None:
    missing = [path for path in dataset_paths.values() if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required dataset files:\n{missing_text}")


def _scope_root(run_root: Path, scope: ScopeSpec) -> Path:
    return run_root / "scopes" / scope.candidate_id


def _scope_split_root(scope_root: Path, split_name: str) -> Path:
    return scope_root / "splits" / _normalize_split_name(split_name)


def _scope_split_paths(scope_root: Path, split_name: str) -> tuple[Path, Path]:
    split_root = _scope_split_root(scope_root, split_name)
    return split_root / "injected_rows.parquet", split_root / "predictions.parquet"


def _scope_split_complete(scope_root: Path, split_name: str) -> bool:
    injected_path, predictions_path = _scope_split_paths(scope_root, split_name)
    return injected_path.exists() and predictions_path.exists()


def _scope_run_complete(scope_root: Path) -> bool:
    return all(_scope_split_complete(scope_root, split_name) for split_name in EVALUATION_SPLITS)


def _write_scope_split_checkpoint(
    scope_root: Path,
    split_name: str,
    injected_rows: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    split_root = _scope_split_root(scope_root, split_name)
    split_root.mkdir(parents=True, exist_ok=True)
    injected_rows.to_parquet(split_root / "injected_rows.parquet", index=False)
    predictions.to_parquet(split_root / "predictions.parquet", index=False)


def _write_json_file(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json.loads(json_dumps(payload)), indent=2) + "\n", encoding="utf-8")


def _scope_extra_columns(scope: ScopeSpec, country_if_config: CountryIFConfig) -> dict[str, object]:
    return {
        "mh_level": scope.mh_level,
        "scope_id": scope.scope_id,
        "scope_kind": "competitor",
        "scope_market": scope.scope_market,
        "country_if_model_name": country_if_config.model_name,
        "country_if_config_source": str(country_if_config.source_json),
    }


def _build_scope_run_metadata(
    scope: ScopeSpec,
    dataset_paths: dict[str, Path],
    country_if_config: CountryIFConfig,
) -> dict[str, object]:
    return {
        "schema_version": "phase2.v1",
        "experiment_family": EXPERIMENT_FAMILY,
        "run_id": RUN_ID,
        "candidate_id": scope.candidate_id,
        "dataset_names": [scope.dataset_name],
        "dataset_granularity": scope.dataset_granularity,
        "dataset_splits": [_normalize_split_name(split) for split in EVALUATION_SPLITS],
        "source_dataset_paths": [str(dataset_paths[TRAIN_SPLIT])] + [
            str(dataset_paths[split]) for split in EVALUATION_SPLITS
        ],
        "random_seeds": {
            "injection_seed_base": INJECTION_SEED,
        },
        "injection_config": {
            "injection_rate": INJECTION_RATE,
            "spike_range": list(SPIKE_RANGE),
            "drop_range": list(DROP_RANGE),
            "strategy": INJECTION_STRATEGY,
            "anomaly_types": [anomaly_type.value for anomaly_type in ANOMALY_TYPES],
        },
        "detector_identifiers": list(FINAL_COMBINATIONS),
        "config_values": {
            "country": scope.country,
            "mh_level": scope.mh_level,
            "dataset_snapshot": DATASET_SNAPSHOT,
            "country_if_model_name": country_if_config.model_name,
            "country_if_threshold": country_if_config.threshold,
            "country_if_n_estimators": country_if_config.n_estimators,
            "country_if_max_samples": country_if_config.max_samples,
            "country_if_max_features": country_if_config.max_features,
            "country_if_contamination": country_if_config.contamination,
            "country_if_config_source": str(country_if_config.source_json),
            "evaluation_splits": list(EVALUATION_SPLITS),
            "min_history": MIN_HISTORY,
            "combinations": list(FINAL_COMBINATIONS),
            "parallel_scope_workers": PARALLEL_SCOPE_WORKERS,
            "scope_market": scope.scope_market,
            "if_config_granularity": "by_country",
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": resolve_git_commit(_PROJECT_ROOT),
    }


def _write_scope_metadata(
    scope_root: Path,
    scope: ScopeSpec,
    dataset_paths: dict[str, Path],
    country_if_config: CountryIFConfig,
) -> None:
    _write_json_file(
        scope_root / "run_metadata.json",
        _build_scope_run_metadata(scope, dataset_paths, country_if_config),
    )


def _build_scope_manifest_rows(
    scopes: list[ScopeSpec],
    country_if_configs: dict[tuple[str, str], CountryIFConfig],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scope in scopes:
        dataset_paths = build_dataset_paths(scope)
        country_if_config = country_if_configs[(scope.country, scope.mh_level)]
        rows.append(
            {
                "mh_level": scope.mh_level,
                "scope_id": scope.scope_id,
                "scope_kind": "competitor",
                "scope_market": scope.scope_market,
                "country": scope.country,
                "dataset_name": scope.dataset_name,
                "dataset_granularity": scope.dataset_granularity,
                "country_if_model_name": country_if_config.model_name,
                "country_if_threshold": country_if_config.threshold,
                "country_if_n_estimators": country_if_config.n_estimators,
                "country_if_max_samples": country_if_config.max_samples,
                "country_if_max_features": country_if_config.max_features,
                "country_if_contamination": country_if_config.contamination,
                "country_if_config_source": str(country_if_config.source_json),
                "train_path": str(dataset_paths[TRAIN_SPLIT]),
                "test_new_prices_path": str(dataset_paths["test_new_prices"]),
                "test_new_products_path": str(dataset_paths["test_new_products"]),
            }
        )
    return pd.DataFrame(rows)


def _build_scope_status_rows(scope_results: list[ScopeRunResult]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for result in sorted(scope_results, key=lambda item: item.scope.scope_id):
        scope_root = result.scope_root
        rows.append(
            {
                "mh_level": result.scope.mh_level,
                "scope_id": result.scope.scope_id,
                "country": result.scope.country,
                "scope_market": result.scope.scope_market,
                "resumed": result.resumed,
                "new_prices_complete": _scope_split_complete(scope_root, "test_new_prices"),
                "new_products_complete": _scope_split_complete(scope_root, "test_new_products"),
            }
        )
    return pd.DataFrame(rows)


def _load_frame(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _load_or_train_country_iforest(
    country_if_config: CountryIFConfig,
    persistence: ModelPersistence,
) -> None:
    if persistence.model_exists(country_if_config.model_name, "isolation_forest"):
        LOGGER.info("country_iforest_already_available", extra={"country": country_if_config.country})
        return

    LOGGER.info(
        "country_iforest_missing_refitting",
        extra={
            "country": country_if_config.country,
            "mh_level": country_if_config.mh_level,
            "model_name": country_if_config.model_name,
            "source_json": str(country_if_config.source_json),
        },
    )
    train_df = _load_frame(_country_train_path(country_if_config.country, country_if_config.mh_level))
    features = extract_features_vectorized(train_df)
    detector, _ = train_from_matrix(
        features,
        contamination=country_if_config.contamination,
        anomaly_threshold=country_if_config.threshold,
        n_estimators=country_if_config.n_estimators,
        max_samples=country_if_config.max_samples,
        max_features=country_if_config.max_features,
    )
    persistence.save_isolation_forest(detector, country_if_config.model_name, len(train_df))


def _ensure_country_iforest_models(country_if_configs: dict[str, CountryIFConfig]) -> None:
    persistence = ModelPersistence()
    for key in sorted(country_if_configs):
        _load_or_train_country_iforest(country_if_configs[key], persistence)


def _print_scope_summary(scope_result: ScopeRunResult) -> None:
    state = "resumed" if scope_result.resumed else "completed"
    print(f"Scope {state}: {scope_result.scope.mh_level} / {scope_result.scope.scope_id}")


def evaluate_scope(
    scope: ScopeSpec,
    run_root: Path,
    country_if_config: CountryIFConfig,
) -> ScopeRunResult:
    dataset_paths = build_dataset_paths(scope)
    _validate_dataset_paths(dataset_paths)
    scope_root = _scope_root(run_root, scope)

    if _scope_run_complete(scope_root):
        _write_scope_metadata(scope_root, scope, dataset_paths, country_if_config)
        return ScopeRunResult(scope=scope, scope_root=scope_root, resumed=True)

    print(f"Scope: {scope.scope_id}")
    print(f"  mh_level={scope.mh_level}")
    print(f"  country={scope.country}")
    print(f"  market={scope.scope_market}")
    print(f"  country_if_model={country_if_config.model_name}")

    train_df: pd.DataFrame | None = None
    orchestrator: TestOrchestrator | None = None
    detector_family_map: dict[str, str] | None = None

    for split_index, split_name in enumerate(EVALUATION_SPLITS):
        if _scope_split_complete(scope_root, split_name):
            print(f"  {_normalize_split_name(split_name)}: checkpoint found, skipping")
            continue

        if orchestrator is None:
            persistence = ModelPersistence()
            train_df = _load_frame(dataset_paths[TRAIN_SPLIT])
            iforest_detector = persistence.load_isolation_forest(country_if_config.model_name)
            evaluators = create_evaluators(
                persistence=None,
                model_name=country_if_config.model_name,
                iforest_detector=iforest_detector,
                combinations=list(FINAL_COMBINATIONS),
            )
            detector_family_map = {
                evaluator.name: evaluator.detector.name for evaluator in evaluators
            }
            orchestrator = TestOrchestrator(evaluators=evaluators, max_workers=1)
            print(f"  train_rows={len(train_df):,}")

        frame = _load_frame(dataset_paths[split_name])
        injection_kwargs = {
            "seed": INJECTION_SEED + split_index,
        }
        injected_frame, labels, injection_details = inject_split_frame(frame, split_index=split_index)
        annotated_frame = _annotate_injected_frame(
            injected_frame,
            labels=labels,
            injection_details=injection_details,
            injection_seed=int(injection_kwargs["seed"]),
        )

        print(
            f"  {split_name}: {len(frame):,} rows, "
            f"{int(np.sum(labels)):,} injected anomalies, "
            f"seed={injection_kwargs['seed']}"
        )

        comparison = orchestrator.run_comparison_with_details(
            train_df=train_df,
            test_df=annotated_frame,
            labels=labels,
            country=scope.country,
            injection_details=injection_details,
        )
        injected_rows, predictions = comparison_result_to_tables(
            comparison,
            run_id=RUN_ID,
            candidate_id=scope.candidate_id,
            experiment_family=EXPERIMENT_FAMILY,
            dataset_name=scope.dataset_name,
            dataset_granularity=scope.dataset_granularity,
            dataset_split=split_name,
            detector_family_map=detector_family_map,
            injected_row_extras=_scope_extra_columns(scope, country_if_config),
            prediction_extras=_scope_extra_columns(scope, country_if_config),
        )
        _write_scope_split_checkpoint(scope_root, split_name, injected_rows, predictions)

        del comparison
        del injected_rows
        del predictions
        del annotated_frame
        del injected_frame
        del frame
        gc.collect()

    _write_scope_metadata(scope_root, scope, dataset_paths, country_if_config)
    gc.collect()
    return ScopeRunResult(scope=scope, scope_root=scope_root, resumed=False)


def _build_run_metadata(
    scopes: list[ScopeSpec],
    country_if_configs: dict[tuple[str, str], CountryIFConfig],
) -> dict[str, object]:
    return {
        "schema_version": "phase2.v1",
        "experiment_family": EXPERIMENT_FAMILY,
        "run_id": RUN_ID,
        "candidate_ids": [scope.candidate_id for scope in scopes],
        "dataset_names": [scope.dataset_name for scope in scopes],
        "dataset_granularity": "competitor",
        "dataset_granularities": ["competitor"],
        "dataset_splits": [_normalize_split_name(split) for split in EVALUATION_SPLITS],
        "scope_count": len(scopes),
        "scope_ids": [scope.scope_id for scope in scopes],
        "countries": sorted({scope.country for scope in scopes}),
        "mh_levels": list(MH_LEVELS),
        "source_dataset_paths": [
            str(build_dataset_paths(scope)[split])
            for scope in scopes
            for split in (TRAIN_SPLIT, *EVALUATION_SPLITS)
        ],
        "random_seeds": {
            "injection_seed_base": INJECTION_SEED,
        },
        "injection_config": {
            "injection_rate": INJECTION_RATE,
            "spike_range": list(SPIKE_RANGE),
            "drop_range": list(DROP_RANGE),
            "strategy": INJECTION_STRATEGY,
            "anomaly_types": [anomaly_type.value for anomaly_type in ANOMALY_TYPES],
        },
        "detector_identifiers": list(FINAL_COMBINATIONS),
        "config_values": {
            "countries": sorted({scope.country for scope in scopes}),
            "mh_levels": list(MH_LEVELS),
            "dataset_snapshot": DATASET_SNAPSHOT,
            "country_if_model_names": {
                f"{mh_level}:{country}": config.model_name
                for (country, mh_level), config in sorted(country_if_configs.items())
            },
            "country_if_config_sources": {
                f"{mh_level}:{country}": str(config.source_json)
                for (country, mh_level), config in sorted(country_if_configs.items())
            },
            "evaluation_splits": list(EVALUATION_SPLITS),
            "min_history": MIN_HISTORY,
            "combinations": list(FINAL_COMBINATIONS),
            "parallel_scope_workers": PARALLEL_SCOPE_WORKERS,
            "if_config_granularity": "by_country",
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": resolve_git_commit(_PROJECT_ROOT),
    }


def run() -> Path:
    output_root = _build_output_root()
    output_root.mkdir(parents=True, exist_ok=True)

    scopes = build_scope_specs()
    country_if_configs = {
        (scope.country, scope.mh_level): _load_country_if_config(scope.country, scope.mh_level)
        for scope in scopes
    }
    max_workers = min(PARALLEL_SCOPE_WORKERS, len(scopes))

    print(f"Countries: {sorted({scope.country for scope in scopes})}")
    print(f"MH levels: {list(MH_LEVELS)}")
    print(f"Competitor scopes: {[scope.scope_id for scope in scopes]}")
    print(f"Scope-mh jobs: {len(scopes)}")
    print(f"Parallel scope workers: {max_workers}")
    print(f"Evaluation splits: {', '.join(EVALUATION_SPLITS)}")
    print(f"Detector finalists: {list(FINAL_COMBINATIONS)}")
    print(f"Anomaly types: {[anomaly_type.value for anomaly_type in ANOMALY_TYPES]}")

    persistence = ModelPersistence()
    print(f"ML models from: {persistence.models_root_description}")

    _ensure_country_iforest_models(country_if_configs)

    scope_results: list[ScopeRunResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {
            pool.submit(
                evaluate_scope,
                scope,
                output_root,
                country_if_configs[(scope.country, scope.mh_level)],
            ): scope
            for scope in scopes
        }
        for future in as_completed(future_map):
            scope_result = future.result()
            scope_results.append(scope_result)
            _print_scope_summary(scope_result)

    analysis_root = output_root / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    _build_scope_manifest_rows(scopes, country_if_configs).to_csv(
        analysis_root / "scope_manifest.csv",
        index=False,
    )
    _build_scope_status_rows(scope_results).to_csv(
        analysis_root / "scope_status.csv",
        index=False,
    )
    _write_json_file(output_root / "run_metadata.json", _build_run_metadata(scopes, country_if_configs))

    extract_thesis_metrics(run_root=output_root)

    print(f"Output root: {output_root}")
    return output_root


def main() -> None:
    run()


if __name__ == "__main__":
    main()
