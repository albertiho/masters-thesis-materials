#!/usr/bin/env python3
"""Sweep-style tuner for forest detectors over ``data-subsets`` scopes."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from research.training.scripts.train_eif import OPTIMAL_CONFIG as EIF_OPTIMAL_CONFIG
from research.training.scripts.train_eif import train_from_matrix as train_eif_from_matrix
from research.training.scripts.train_isolation_forest import (
    OPTIMAL_CONFIG as IF_OPTIMAL_CONFIG,
)
from research.training.scripts.train_isolation_forest import (
    extract_features_vectorized,
    train_from_matrix as train_iforest_from_matrix,
)
from research.training.scripts.train_rrcf import OPTIMAL_CONFIG as RRCF_OPTIMAL_CONFIG
from research.training.scripts.train_rrcf import train_from_matrix as train_rrcf_from_matrix
from research.training.scripts.tune_statistical import (
    ScopeDescriptor,
    discover_scopes,
    ensure_cache_snapshot,
)
from research.training.scripts.tuning_utils import TuningResult, run_tuning_trials
from research.training.scripts.validate_iforest_mh5_smoke import (
    _infer_country,
    build_quick_subset,
)
from src.anomaly.persistence import ModelPersistence
from src.research.artifacts import create_run_id, resolve_git_commit, slugify
from src.research.mh_sampling import RESEARCH_MH_LEVELS

LOGGER = logging.getLogger("tune_forests")

SCHEMA_VERSION = "phase2.v1"
EXPERIMENT_FAMILY = "tuning_forests"
DEFAULT_DATA_SUBSETS_ROOT = _PROJECT_ROOT / "data-subsets"
DEFAULT_RESULTS_ROOT = _PROJECT_ROOT / "results" / "tuning" / "forests"
DEFAULT_MODEL_ROOT = _PROJECT_ROOT / "artifacts" / "models"
DEFAULT_SAMPLED_MH_VALUES = RESEARCH_MH_LEVELS
ALL_FOREST_DETECTORS = ("if", "eif", "rrcf")
DEFAULT_FOREST_GRANULARITIES = ("global", "by_country", "by_competitor")
DEFAULT_DROP_RANGE = (0.10, 0.50)
DEFAULT_SPLIT_WEIGHTS = {"new_prices": 0.7, "new_products": 0.3}
SPLIT_METRIC_FIELDS = (
    "accuracy",
    "precision",
    "recall",
    "tnr",
    "fpr",
    "fnr",
    "f1",
    "g_mean",
)
IF_GRID = {
    "n_estimators": (100, 200),
    "max_samples": (256, 512),
    "max_features": (0.5, 0.75, 1.0),
    "contamination": (IF_OPTIMAL_CONFIG.contamination,),
}
EIF_GRID = {
    "n_estimators": (100, 200),
    "max_samples": (256, 512),
    "max_features": (0.5, 0.75, 1.0),
}
EIF_PASS1_GRID = {
    "n_estimators": (200,),
    "max_samples": (256, 512),
    "max_features": (0.5, 0.75, 1.0),
}
EIF_PASS2_GRID = {
    "n_estimators": (100, 200, 400),
    "max_samples": (256, 512),
}
EIF_PASS1_THRESHOLD_RANGE = (0.15, 0.85)
EIF_PASS1_THRESHOLD_STEP = 0.05
EIF_PASS2_THRESHOLD_RADIUS = 0.10
EIF_PASS2_THRESHOLD_STEP = 0.05
EIF_PASS2_MAX_FEATURE_CHOICES = 2
RRCF_GRID = {
    "num_trees": (20, 40, 80),
    "tree_size": (128, 256, 512),
    "warmup_samples": (RRCF_OPTIMAL_CONFIG.warmup_samples,),
}
# Freeze underperforming families to the best configs seen in the mh5 pilot
# runs so they skip the expensive stage1 hyperparameter screen entirely.
FIXED_PROMOTED_CONFIGS: dict[str, tuple[dict[str, Any], ...]] = {
    "rrcf": (
        {
            "num_trees": 80,
            "tree_size": 128,
            "warmup_samples": RRCF_OPTIMAL_CONFIG.warmup_samples,
        },
    ),
}
STAGE1_THRESHOLDS = {
    "if": 0.48,
    "eif": 0.50,
    "rrcf": 0.65,
}
STAGE2_TOP_CONFIGS = {
    "if": 3,
    "eif": 1,
    "rrcf": 1,
}
BASELINE_FOREST_PARAMS = {
    "if": {
        "n_estimators": IF_OPTIMAL_CONFIG.n_estimators,
        "max_samples": IF_OPTIMAL_CONFIG.max_samples,
        "max_features": IF_OPTIMAL_CONFIG.max_features,
        "contamination": IF_OPTIMAL_CONFIG.contamination,
    },
    "eif": {
        "n_estimators": EIF_OPTIMAL_CONFIG.n_estimators,
        "max_samples": EIF_OPTIMAL_CONFIG.max_samples,
        "max_features": EIF_OPTIMAL_CONFIG.max_features,
    },
    "rrcf": {
        "num_trees": RRCF_OPTIMAL_CONFIG.num_trees,
        "tree_size": RRCF_OPTIMAL_CONFIG.tree_size,
        "warmup_samples": RRCF_OPTIMAL_CONFIG.warmup_samples,
    },
}
BASELINE_THRESHOLDS = {
    "if": float(IF_OPTIMAL_CONFIG.anomaly_threshold),
    "eif": float(EIF_OPTIMAL_CONFIG.anomaly_threshold),
    "rrcf": float(RRCF_OPTIMAL_CONFIG.anomaly_threshold),
}
SPLIT_COUNT_FIELDS = (
    "true_positives",
    "false_positives",
    "false_negatives",
    "true_negatives",
    "n_rows",
    "n_injected",
    "n_predicted",
)


@dataclass(frozen=True)
class ForestSpec:
    detector_family: str
    storage_model_name: str
    display_name: str
    min_threshold: float
    max_threshold: float
    steps: int


FOREST_SPECS: dict[str, ForestSpec] = {
    "if": ForestSpec("if", "isolation_forest", "Isolation Forest", 0.3, 0.9, 11),
    "eif": ForestSpec("eif", "eif", "EIF", 0.15, 0.85, 15),
    "rrcf": ForestSpec("rrcf", "rrcf", "RRCF", 0.2, 0.95, 11),
}


@dataclass(frozen=True)
class FixedSearchProfile:
    name: str
    detector_family: str
    params: Mapping[str, Any]
    threshold: float
    stage_label: str = "fixed_profile"
    summary_search_strategy: str = "fixed_profile"


EIF_FIXED_SEARCH_PROFILES: dict[str, FixedSearchProfile] = {
    "fixed_gmean_winner": FixedSearchProfile(
        name="fixed_gmean_winner",
        detector_family="eif",
        params={
            "n_estimators": 100,
            "max_samples": 256,
            "max_features": 0.5,
        },
        threshold=0.20,
        summary_search_strategy="fixed_eif_gmean_winner",
    ),
}
EIF_SEARCH_STRATEGIES = ("two_pass", *EIF_FIXED_SEARCH_PROFILES)


@dataclass(frozen=True)
class ForestTask:
    scope: ScopeDescriptor
    detector_family: str
    output_dir: Path
    model_root: Path
    sweep_id: str
    attempts: int
    trial_workers: int
    target_metric: str
    min_precision: float
    injection_rate: float
    dry_run: bool
    splits: tuple[str, ...]
    max_products: int | None = None
    max_history_per_product: int | None = None
    min_test_rows: int | None = None
    max_test_rows: int | None = None
    steps_override: int | None = None
    eif_search_strategy: str = "two_pass"


@dataclass
class SplitEvaluation:
    split_name: str
    tuning_result: TuningResult
    row_count: int
    product_count: int
    train_row_count: int
    train_product_count: int


@dataclass
class ForestConfigEvaluation:
    config_key: str
    config_values: dict[str, Any]
    baseline_threshold: float
    training_time_sec: float
    split_results: list[SplitEvaluation]


@dataclass
class ForestTaskResult:
    mh_level: str
    granularity: str
    scope_id: str
    dataset_name: str
    detector_family: str
    status: str
    output_dir: str
    best_candidate: dict[str, Any] | None = None
    error: str = ""


@dataclass
class ForestSearchResult:
    candidate_metrics: pd.DataFrame
    config_evaluations: dict[str, ForestConfigEvaluation]
    current_source_metrics: pd.DataFrame | None
    current_eval: ForestConfigEvaluation | None
    screening_threshold: float
    screened_config_count: int
    promoted_config_count: int
    stage_label: str = "threshold_sweep"
    summary_metadata: dict[str, Any] = field(default_factory=dict)


def _score_column_for_target_metric(target_metric: str) -> str:
    return {
        "f1": "combined_f1",
        "precision": "combined_precision",
        "recall": "combined_recall",
        "gmean": "rank_score",
    }[target_metric]


def _normalized_split_weights(split_names: Sequence[str]) -> dict[str, float]:
    base_weights = {name: float(DEFAULT_SPLIT_WEIGHTS.get(name, 1.0)) for name in split_names}
    total = float(sum(base_weights.values()))
    if total <= 0:
        uniform = 1.0 / float(len(split_names) or 1)
        return {name: uniform for name in split_names}
    return {name: weight / total for name, weight in base_weights.items()}


def _find_threshold_metrics(results: Sequence[Mapping[str, Any]], threshold: float) -> Mapping[str, Any]:
    if not results:
        raise ValueError("No threshold results available")
    for result in results:
        if np.isclose(float(result["threshold"]), float(threshold)):
            return result
    return min(results, key=lambda row: abs(float(row["threshold"]) - float(threshold)))


def _metric_snapshot(metrics: Mapping[str, Any]) -> dict[str, Any]:
    keep = {
        "threshold",
        "accuracy",
        "accuracy_std",
        "precision",
        "precision_std",
        "recall",
        "recall_std",
        "tnr",
        "tnr_std",
        "fpr",
        "fpr_std",
        "fnr",
        "fnr_std",
        "f1",
        "f1_std",
        "g_mean",
        "g_mean_std",
        "true_positives",
        "true_positives_std",
        "false_positives",
        "false_positives_std",
        "false_negatives",
        "false_negatives_std",
        "true_negatives",
        "true_negatives_std",
        "n_rows",
        "n_rows_std",
        "n_injected",
        "n_injected_std",
        "n_predicted",
        "n_predicted_std",
        "n_trials",
    }
    return {key: _to_serializable(value) for key, value in metrics.items() if key in keep}


def _resolve_root(path_value: str | Path, *, default: Path | None = None) -> Path:
    raw_path = Path(path_value) if path_value is not None else default
    if raw_path is None:
        raise ValueError("A path value is required")
    if raw_path.is_absolute():
        return raw_path
    return (_PROJECT_ROOT / raw_path).resolve()


def _scope_matches_filter(scope: ScopeDescriptor, scope_filter: str) -> bool:
    token = scope_filter.strip().lower()
    if not token:
        return True
    return token in scope.scope_id.lower() or token in scope.dataset_name.lower()


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_serializable(payload), indent=2) + "\n", encoding="utf-8")


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    return value


def _normalize_granularities(values: Sequence[str] | None) -> list[str] | None:
    if not values:
        return list(DEFAULT_FOREST_GRANULARITIES)
    normalized = [str(value).strip() for value in values if str(value).strip()]
    unknown = sorted(set(normalized).difference(DEFAULT_FOREST_GRANULARITIES))
    if unknown:
        raise ValueError(
            f"Unsupported forest granularities {unknown!r}; expected some of {list(DEFAULT_FOREST_GRANULARITIES)!r}"
        )
    return normalized


def _iter_hyperparameter_configs(detector_family: str) -> list[dict[str, Any]]:
    if detector_family == "if":
        grid = IF_GRID
    elif detector_family == "eif":
        grid = EIF_GRID
    elif detector_family == "rrcf":
        grid = RRCF_GRID
    else:
        raise ValueError(f"Unsupported detector family: {detector_family}")

    keys = list(grid)
    return [
        {key: value for key, value in zip(keys, values, strict=True)}
        for values in product(*(grid[key] for key in keys))
    ]


def _fixed_promoted_configs(detector_family: str) -> list[dict[str, Any]] | None:
    configs = FIXED_PROMOTED_CONFIGS.get(detector_family)
    if configs is None:
        return None
    return [dict(config) for config in configs]


def _fixed_search_profile(task: ForestTask) -> FixedSearchProfile | None:
    if task.detector_family != "eif":
        return None
    if task.eif_search_strategy == "two_pass":
        return None
    return EIF_FIXED_SEARCH_PROFILES[str(task.eif_search_strategy)]


def _baseline_params(detector_family: str) -> dict[str, Any]:
    return dict(BASELINE_FOREST_PARAMS[detector_family])


def _baseline_threshold(detector_family: str) -> float:
    return float(BASELINE_THRESHOLDS[detector_family])


def _value_slug(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".").replace(".", "p")
    return str(value).replace(".", "p")


def _config_key(detector_family: str, params: Mapping[str, Any]) -> str:
    pieces = [detector_family]
    for key in sorted(params):
        pieces.append(f"{key}_{_value_slug(params[key])}")
    return "__".join(pieces)


def _candidate_id(detector_family: str, params: Mapping[str, Any], threshold: float) -> str:
    return f"{_config_key(detector_family, params)}__threshold_{_value_slug(float(threshold))}"


def _default_distance_for_candidate(
    detector_family: str,
    params: Mapping[str, Any],
    threshold: float,
) -> float:
    baseline_params = _baseline_params(detector_family)
    distance = abs(float(threshold) - _baseline_threshold(detector_family))
    for key, baseline_value in baseline_params.items():
        value = params[key]
        if isinstance(baseline_value, str):
            distance += 0.0 if value == baseline_value else 1.0
            continue
        distance += abs(float(value) - float(baseline_value)) / max(abs(float(baseline_value)), 1.0)
    return float(distance)


def _train_detector_from_matrix(
    detector_family: str,
    X_train: np.ndarray,
    params: Mapping[str, Any],
    threshold: float,
) -> Any:
    if detector_family == "if":
        detector, _ = train_iforest_from_matrix(
            X_train,
            contamination=params["contamination"],
            anomaly_threshold=threshold,
            n_estimators=int(params["n_estimators"]),
            max_samples=params["max_samples"],
            max_features=float(params["max_features"]),
        )
        return detector
    if detector_family == "eif":
        detector, _ = train_eif_from_matrix(
            X_train,
            anomaly_threshold=threshold,
            n_estimators=int(params["n_estimators"]),
            max_samples=params["max_samples"],
            max_features=float(params["max_features"]),
        )
        return detector
    if detector_family == "rrcf":
        detector, _ = train_rrcf_from_matrix(
            X_train,
            num_trees=int(params["num_trees"]),
            tree_size=int(params["tree_size"]),
            anomaly_threshold=threshold,
            warmup_samples=int(params["warmup_samples"]),
        )
        return detector
    raise ValueError(f"Unsupported detector family: {detector_family}")


def _evaluate_config_thresholds(
    *,
    task: ForestTask,
    detector_family: str,
    model_name: str,
    X_train: np.ndarray,
    train_df: pd.DataFrame,
    prepared_splits: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]],
    params: Mapping[str, Any],
    thresholds: np.ndarray,
    min_successful_trials: int,
    baseline_params: Mapping[str, Any] | None = None,
    baseline_threshold: float | None = None,
) -> tuple[pd.DataFrame, ForestConfigEvaluation]:
    config_key = _config_key(detector_family, params)
    baseline_params = dict(baseline_params) if baseline_params is not None else _baseline_params(detector_family)
    baseline_threshold = (
        float(baseline_threshold) if baseline_threshold is not None else _baseline_threshold(detector_family)
    )
    train_start = pd.Timestamp.utcnow()
    detector = _train_detector_from_matrix(
        detector_family,
        X_train,
        params,
        baseline_threshold,
    )
    training_time_sec = (pd.Timestamp.utcnow() - train_start).total_seconds()

    split_results: list[SplitEvaluation] = []
    for split_name in task.splits:
        train_input, test_input = prepared_splits[split_name]
        tuning_result = run_tuning_trials(
            detector=detector,
            detector_name=f"{model_name}::{config_key}::{split_name}",
            test_df=test_input,
            train_df=train_input,
            cache_snapshot_path=str(task.scope.cache_snapshot_path),
            thresholds=thresholds,
            current_threshold=baseline_threshold,
            n_trials=task.attempts,
            injection_rate=task.injection_rate,
            country=_infer_country(train_df) or _infer_country(test_input),
            max_workers=task.trial_workers,
            target_metric=task.target_metric,
            min_precision=task.min_precision,
            drop_range=DEFAULT_DROP_RANGE,
            min_successful_trials=min_successful_trials,
        )
        if tuning_result is None:
            raise RuntimeError(f"Tuning failed for split {split_name} under {config_key}")
        split_results.append(
            SplitEvaluation(
                split_name=split_name,
                tuning_result=tuning_result,
                row_count=len(test_input),
                product_count=int(test_input["product_id"].nunique()),
                train_row_count=len(train_input),
                train_product_count=int(train_input["product_id"].nunique()),
            )
        )

    combined_metrics = _combine_split_results(
        split_results=split_results,
        thresholds=thresholds,
        current_threshold=baseline_threshold,
        target_metric=task.target_metric,
    )
    combined_metrics["config_key"] = config_key
    combined_metrics["is_baseline_config"] = dict(params) == baseline_params
    combined_metrics["training_time_sec"] = float(training_time_sec)
    for key, value in params.items():
        combined_metrics[key] = value
    combined_metrics["default_distance"] = [
        _default_distance_for_candidate(detector_family, params, float(threshold))
        for threshold in combined_metrics["threshold"]
    ]
    combined_metrics["candidate_id"] = [
        _candidate_id(detector_family, params, float(threshold))
        for threshold in combined_metrics["threshold"]
    ]
    return combined_metrics, ForestConfigEvaluation(
        config_key=config_key,
        config_values=dict(params),
        baseline_threshold=baseline_threshold,
        training_time_sec=float(training_time_sec),
        split_results=split_results,
    )


def _run_two_pass_eif_search(
    *,
    task: ForestTask,
    spec: ForestSpec,
    model_name: str,
    X_train: np.ndarray,
    train_df: pd.DataFrame,
    prepared_splits: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]],
    min_successful_trials: int,
) -> ForestSearchResult:
    pass1_thresholds = _build_threshold_grid(
        EIF_PASS1_THRESHOLD_RANGE[0],
        EIF_PASS1_THRESHOLD_RANGE[1],
        EIF_PASS1_THRESHOLD_STEP,
    )
    pass1_frames: list[pd.DataFrame] = []
    pass1_evaluations: dict[str, ForestConfigEvaluation] = {}
    for params in _iter_eif_pass1_configs():
        combined_metrics, config_evaluation = _evaluate_config_thresholds(
            task=task,
            detector_family="eif",
            model_name=model_name,
            X_train=X_train,
            train_df=train_df,
            prepared_splits=prepared_splits,
            params=params,
            thresholds=pass1_thresholds,
            min_successful_trials=min_successful_trials,
        )
        pass1_frames.append(combined_metrics)
        pass1_evaluations[config_evaluation.config_key] = config_evaluation

    pass1_candidate_metrics = pd.concat(pass1_frames, ignore_index=True)
    pass1_best_row = _select_best_candidate_row(
        pass1_candidate_metrics,
        target_metric=task.target_metric,
        min_precision=task.min_precision,
    )
    selected_max_features = _select_promoted_param_values(
        pass1_candidate_metrics,
        column="max_features",
        limit=EIF_PASS2_MAX_FEATURE_CHOICES,
        target_metric=task.target_metric,
        min_precision=task.min_precision,
    )
    if not selected_max_features:
        raise RuntimeError("EIF two-pass search did not select any max_features values")

    pass2_thresholds = _build_centered_threshold_grid(
        float(pass1_best_row["threshold"]),
        radius=EIF_PASS2_THRESHOLD_RADIUS,
        step=EIF_PASS2_THRESHOLD_STEP,
        minimum=spec.min_threshold,
        maximum=spec.max_threshold,
        steps_override=task.steps_override,
    )

    pass2_frames: list[pd.DataFrame] = []
    pass2_evaluations: dict[str, ForestConfigEvaluation] = {}
    for params in _iter_eif_pass2_configs(selected_max_features):
        combined_metrics, config_evaluation = _evaluate_config_thresholds(
            task=task,
            detector_family="eif",
            model_name=model_name,
            X_train=X_train,
            train_df=train_df,
            prepared_splits=prepared_splits,
            params=params,
            thresholds=pass2_thresholds,
            min_successful_trials=min_successful_trials,
        )
        pass2_frames.append(combined_metrics)
        pass2_evaluations[config_evaluation.config_key] = config_evaluation

    if not pass2_frames:
        raise RuntimeError("EIF two-pass refinement did not produce any candidate metrics")

    baseline_params = _baseline_params("eif")
    baseline_config_key = _config_key("eif", baseline_params)
    current_source_metrics: pd.DataFrame | None = None
    current_eval: ForestConfigEvaluation | None = None
    if baseline_config_key not in pass2_evaluations:
        current_source_metrics, current_eval = _evaluate_config_thresholds(
            task=task,
            detector_family="eif",
            model_name=model_name,
            X_train=X_train,
            train_df=train_df,
            prepared_splits=prepared_splits,
            params=baseline_params,
            thresholds=pass2_thresholds,
            min_successful_trials=min_successful_trials,
        )

    return ForestSearchResult(
        candidate_metrics=pd.concat(pass2_frames, ignore_index=True),
        config_evaluations=pass2_evaluations,
        current_source_metrics=current_source_metrics,
        current_eval=current_eval,
        screening_threshold=float("nan"),
        screened_config_count=len(_iter_eif_pass1_configs()),
        promoted_config_count=len(pass2_evaluations),
        stage_label="two_pass_refinement",
        summary_metadata={
            "search_strategy": "two_pass_eif_calibration",
            "pass1_config_count": len(_iter_eif_pass1_configs()),
            "pass1_threshold_count": int(len(pass1_thresholds)),
            "pass1_best_config_key": str(pass1_best_row["config_key"]),
            "pass1_best_threshold": float(pass1_best_row["threshold"]),
            "pass1_selected_max_features": [float(value) for value in selected_max_features],
            "pass2_config_count": len(pass2_evaluations),
            "pass2_threshold_count": int(len(pass2_thresholds)),
            "pass2_threshold_min": float(pass2_thresholds.min()),
            "pass2_threshold_max": float(pass2_thresholds.max()),
        },
    )


def _run_fixed_profile_search(
    *,
    task: ForestTask,
    model_name: str,
    X_train: np.ndarray,
    train_df: pd.DataFrame,
    prepared_splits: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]],
    min_successful_trials: int,
    profile: FixedSearchProfile,
) -> ForestSearchResult:
    thresholds = np.asarray([float(profile.threshold)], dtype=np.float64)
    params = dict(profile.params)
    combined_metrics, config_evaluation = _evaluate_config_thresholds(
        task=task,
        detector_family=profile.detector_family,
        model_name=model_name,
        X_train=X_train,
        train_df=train_df,
        prepared_splits=prepared_splits,
        params=params,
        thresholds=thresholds,
        min_successful_trials=min_successful_trials,
        baseline_params=params,
        baseline_threshold=float(profile.threshold),
    )
    return ForestSearchResult(
        candidate_metrics=combined_metrics,
        config_evaluations={config_evaluation.config_key: config_evaluation},
        current_source_metrics=None,
        current_eval=None,
        screening_threshold=float("nan"),
        screened_config_count=1,
        promoted_config_count=1,
        stage_label=profile.stage_label,
        summary_metadata={
            "search_strategy": profile.summary_search_strategy,
            "fixed_profile_name": profile.name,
            "fixed_threshold": float(profile.threshold),
            "fixed_hyperparameters": {key: _to_serializable(value) for key, value in params.items()},
        },
    )


def _save_detector(
    persistence: ModelPersistence,
    detector_family: str,
    detector: Any,
    model_name: str,
    n_train_rows: int,
) -> str:
    if detector_family == "if":
        return persistence.save_isolation_forest(detector, model_name, n_train_rows)
    if detector_family == "eif":
        return persistence.save_eif(detector, model_name, n_train_rows)
    if detector_family == "rrcf":
        return persistence.save_rrcf(detector, model_name, n_train_rows)
    raise ValueError(f"Unsupported detector family: {detector_family}")


def _maybe_subset_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    max_products: int | None,
    max_history_per_product: int | None,
    min_test_rows: int | None,
    max_test_rows: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not any(value is not None for value in (max_products, max_history_per_product, min_test_rows, max_test_rows)):
        return train_df, test_df
    return build_quick_subset(
        train_df,
        test_df,
        max_products=max_products or 64,
        max_history_per_product=max_history_per_product or 30,
        min_test_rows=min_test_rows or 128,
        max_test_rows=max_test_rows or 512,
    )


def _combine_split_results(
    *,
    split_results: Sequence[SplitEvaluation],
    thresholds: np.ndarray,
    current_threshold: float,
    target_metric: str,
) -> pd.DataFrame:
    split_weights = _normalized_split_weights([result.split_name for result in split_results])
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        row: dict[str, Any] = {
            "threshold": float(threshold),
            "combined_accuracy": 0.0,
            "combined_precision": 0.0,
            "combined_recall": 0.0,
            "combined_tnr": 0.0,
            "combined_fpr": 0.0,
            "combined_fnr": 0.0,
            "combined_f1": 0.0,
            "combined_g_mean": 0.0,
            "split_count": len(split_results),
            "weighted_row_count": int(sum(result.row_count for result in split_results)),
            "default_distance": abs(float(threshold) - float(current_threshold)),
        }
        missing_metrics = False
        for split_result in split_results:
            metrics = _find_threshold_metrics(split_result.tuning_result.all_results, float(threshold))
            if not metrics:
                missing_metrics = True
                break
            weight = split_weights[split_result.split_name]
            row["combined_accuracy"] += weight * float(metrics["accuracy"])
            row["combined_precision"] += weight * float(metrics["precision"])
            row["combined_recall"] += weight * float(metrics["recall"])
            row["combined_tnr"] += weight * float(metrics["tnr"])
            row["combined_fpr"] += weight * float(metrics["fpr"])
            row["combined_fnr"] += weight * float(metrics["fnr"])
            row["combined_f1"] += weight * float(metrics["f1"])
            row["combined_g_mean"] += weight * float(metrics["g_mean"])
            for metric_name in SPLIT_METRIC_FIELDS:
                row[f"{split_result.split_name}_{metric_name}_mean"] = float(metrics[metric_name])
                row[f"{split_result.split_name}_{metric_name}_std"] = float(metrics[f"{metric_name}_std"])
            for metric_name in SPLIT_COUNT_FIELDS:
                row[f"{split_result.split_name}_{metric_name}_mean"] = float(metrics[metric_name])
                row[f"{split_result.split_name}_{metric_name}_std"] = float(metrics[f"{metric_name}_std"])
        if missing_metrics:
            continue
        row["weighted_f1_mean"] = row["combined_f1"]
        row["rank_score"] = row["combined_g_mean"]
        rows.append(row)
    combined = pd.DataFrame(rows)
    if combined.empty:
        raise RuntimeError("No common threshold metrics were produced across splits")
    return combined


def _rank_candidate_rows(
    candidate_metrics: pd.DataFrame,
    *,
    target_metric: str,
    min_precision: float,
) -> pd.DataFrame:
    if candidate_metrics.empty:
        raise RuntimeError("No candidate metrics were produced")
    score_column = _score_column_for_target_metric(target_metric)
    candidates = candidate_metrics.copy()
    if target_metric == "recall":
        precision_mask = candidates["combined_precision"] >= float(min_precision)
        if precision_mask.any():
            candidates = candidates[precision_mask].copy()
    return candidates.sort_values(
        [score_column, "combined_f1", "default_distance", "candidate_id"],
        ascending=[False, False, True, True],
    )


def _select_best_candidate_row(
    candidate_metrics: pd.DataFrame,
    *,
    target_metric: str,
    min_precision: float,
) -> pd.Series:
    ranked = _rank_candidate_rows(
        candidate_metrics,
        target_metric=target_metric,
        min_precision=min_precision,
    )
    return ranked.iloc[0]


def _select_baseline_candidate_row(candidate_metrics: pd.DataFrame, detector_family: str) -> pd.Series:
    baseline_rows = candidate_metrics[candidate_metrics["is_baseline_config"]].copy()
    if baseline_rows.empty:
        raise RuntimeError(f"No baseline candidate rows were produced for {detector_family}")
    baseline_threshold = _baseline_threshold(detector_family)
    baseline_rows["_threshold_delta"] = (baseline_rows["threshold"] - baseline_threshold).abs()
    ranked = baseline_rows.sort_values(
        ["_threshold_delta", "default_distance", "candidate_id"],
        ascending=[True, True, True],
    )
    return ranked.iloc[0]


def _select_promoted_config_keys(
    candidate_metrics: pd.DataFrame,
    *,
    detector_family: str,
    target_metric: str,
    min_precision: float,
) -> list[str]:
    ranked = _rank_candidate_rows(
        candidate_metrics,
        target_metric=target_metric,
        min_precision=min_precision,
    )
    limit = int(STAGE2_TOP_CONFIGS[detector_family])
    return ranked["config_key"].astype(str).drop_duplicates().head(limit).tolist()


def _select_promoted_param_values(
    candidate_metrics: pd.DataFrame,
    *,
    column: str,
    limit: int,
    target_metric: str,
    min_precision: float,
) -> list[Any]:
    ranked = _rank_candidate_rows(
        candidate_metrics,
        target_metric=target_metric,
        min_precision=min_precision,
    )
    selected: list[Any] = []
    seen: set[str] = set()
    for value in ranked[column].tolist():
        marker = repr(_to_serializable(value))
        if marker in seen:
            continue
        seen.add(marker)
        selected.append(value)
        if len(selected) >= limit:
            break
    return selected


def _build_threshold_grid(minimum: float, maximum: float, step: float) -> np.ndarray:
    count = int(round((maximum - minimum) / step)) + 1
    return np.round(np.linspace(minimum, maximum, count), 10)


def _build_centered_threshold_grid(
    center: float,
    *,
    radius: float,
    step: float,
    minimum: float,
    maximum: float,
    steps_override: int | None,
) -> np.ndarray:
    lower = max(minimum, center - radius)
    upper = min(maximum, center + radius)
    if steps_override is not None:
        if steps_override == 1 or np.isclose(lower, upper):
            return np.asarray([float(lower)], dtype=np.float64)
        return np.round(np.linspace(lower, upper, steps_override), 10)
    return _build_threshold_grid(lower, upper, step)


def _iter_eif_pass1_configs() -> list[dict[str, Any]]:
    keys = list(EIF_PASS1_GRID)
    return [
        {key: value for key, value in zip(keys, values, strict=True)}
        for values in product(*(EIF_PASS1_GRID[key] for key in keys))
    ]


def _iter_eif_pass2_configs(max_features_values: Sequence[float]) -> list[dict[str, Any]]:
    return [
        {
            "n_estimators": n_estimators,
            "max_samples": max_samples,
            "max_features": float(max_features),
        }
        for max_features in max_features_values
        for n_estimators in EIF_PASS2_GRID["n_estimators"]
        for max_samples in EIF_PASS2_GRID["max_samples"]
    ]


def _hyperparameter_columns(detector_family: str) -> list[str]:
    return list(_baseline_params(detector_family))


def _extract_hyperparameters(row: Mapping[str, Any], detector_family: str) -> dict[str, Any]:
    return {column: _to_serializable(row[column]) for column in _hyperparameter_columns(detector_family)}


def _build_task_summary(
    *,
    candidate_metrics: pd.DataFrame,
    detector_family: str,
    target_metric: str,
    min_precision: float,
    current_candidate_metrics: pd.DataFrame | None = None,
    screened_config_count: int | None = None,
    summary_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    score_column = _score_column_for_target_metric(target_metric)
    current_source = current_candidate_metrics if current_candidate_metrics is not None else candidate_metrics
    current_row = _select_baseline_candidate_row(current_source, detector_family)
    best_row = _select_best_candidate_row(
        candidate_metrics,
        target_metric=target_metric,
        min_precision=min_precision,
    )
    current_f1 = float(current_row["combined_f1"])
    best_f1 = float(best_row["combined_f1"])
    improvement_pct = ((best_f1 - current_f1) / current_f1 * 100.0) if current_f1 > 0 else 0.0
    summary = {
        "current_candidate_id": str(current_row["candidate_id"]),
        "current_config_key": str(current_row["config_key"]),
        "current_threshold": float(current_row["threshold"]),
        "current_accuracy": float(current_row["combined_accuracy"]),
        "current_precision": float(current_row["combined_precision"]),
        "current_recall": float(current_row["combined_recall"]),
        "current_tnr": float(current_row["combined_tnr"]),
        "current_fpr": float(current_row["combined_fpr"]),
        "current_fnr": float(current_row["combined_fnr"]),
        "current_f1": current_f1,
        "current_g_mean": float(current_row["combined_g_mean"]),
        "current_hyperparameters": _extract_hyperparameters(current_row, detector_family),
        "best_candidate_id": str(best_row["candidate_id"]),
        "best_config_key": str(best_row["config_key"]),
        "best_threshold": float(best_row["threshold"]),
        "best_accuracy": float(best_row["combined_accuracy"]),
        "best_precision": float(best_row["combined_precision"]),
        "best_recall": float(best_row["combined_recall"]),
        "best_tnr": float(best_row["combined_tnr"]),
        "best_fpr": float(best_row["combined_fpr"]),
        "best_fnr": float(best_row["combined_fnr"]),
        "best_f1": best_f1,
        "best_g_mean": float(best_row["combined_g_mean"]),
        "best_hyperparameters": _extract_hyperparameters(best_row, detector_family),
        "target_metric": target_metric,
        "score_column": score_column,
        "config_count": int(candidate_metrics["config_key"].nunique()),
        "screened_config_count": int(screened_config_count or candidate_metrics["config_key"].nunique()),
        "promoted_config_count": int(candidate_metrics["config_key"].nunique()),
        "candidate_count": int(len(candidate_metrics)),
        "improvement_pct": float(improvement_pct),
    }
    if summary_metadata:
        summary.update({str(key): _to_serializable(value) for key, value in summary_metadata.items()})
    return summary


def _build_split_payload(
    *,
    detector_family: str,
    config_evaluations: Mapping[str, ForestConfigEvaluation],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    current_eval = config_evaluations[str(summary["current_config_key"])]
    best_eval = config_evaluations[str(summary["best_config_key"])]
    current_by_split = {result.split_name: result for result in current_eval.split_results}
    best_by_split = {result.split_name: result for result in best_eval.split_results}
    payload: dict[str, Any] = {}
    for split_name, best_split in best_by_split.items():
        current_split = current_by_split[split_name]
        current_metrics = _find_threshold_metrics(
            current_split.tuning_result.all_results,
            float(summary["current_threshold"]),
        )
        best_metrics = _find_threshold_metrics(
            best_split.tuning_result.all_results,
            float(summary["best_threshold"]),
        )
        payload[split_name] = {
            "row_count": int(best_split.row_count),
            "product_count": int(best_split.product_count),
            "train_row_count": int(best_split.train_row_count),
            "train_product_count": int(best_split.train_product_count),
            "current_candidate_id": str(summary["current_candidate_id"]),
            "current_threshold": float(current_metrics["threshold"]),
            "current_f1": float(current_metrics["f1"]),
            "current_hyperparameters": _to_serializable(current_eval.config_values),
            "current_metrics": _metric_snapshot(current_metrics),
            "best_candidate_id": str(summary["best_candidate_id"]),
            "best_threshold": float(best_metrics["threshold"]),
            "best_f1": float(best_metrics["f1"]),
            "best_hyperparameters": _to_serializable(best_eval.config_values),
            "best_metrics": _metric_snapshot(best_metrics),
        }
    return payload


def _task_complete(output_dir: Path) -> bool:
    required = [
        output_dir / "candidate_metrics.csv",
        output_dir / "best_configuration.json",
        output_dir / "split_results.json",
        output_dir / "summary.json",
    ]
    return all(path.exists() for path in required)


def _load_existing_task_result(output_dir: Path) -> ForestTaskResult:
    payload = json.loads((output_dir / "best_configuration.json").read_text(encoding="utf-8"))
    return ForestTaskResult(
        mh_level=str(payload["mh_level"]),
        granularity=str(payload["granularity"]),
        scope_id=str(payload["scope_id"]),
        dataset_name=str(payload["dataset_name"]),
        detector_family=str(payload["detector_family"]),
        status=str(payload.get("status", "error")),
        output_dir=str(output_dir),
        best_candidate=payload.get("best_candidate"),
        error=str(payload.get("error", "")),
    )


def _write_task_outputs(
    *,
    task: ForestTask,
    spec: ForestSpec,
    model_name: str,
    candidate_metrics: pd.DataFrame,
    split_payload: Mapping[str, Any],
    summary: Mapping[str, Any],
    updated_model: bool,
    status: str,
    error: str = "",
) -> dict[str, Any]:
    task.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_metrics.to_csv(task.output_dir / "candidate_metrics.csv", index=False)
    _json_dump(task.output_dir / "split_results.json", split_payload)
    best_candidate = None
    if status == "ok" and not candidate_metrics.empty:
        best_rows = candidate_metrics
        if summary and summary.get("best_candidate_id"):
            best_rows = candidate_metrics[
                candidate_metrics["candidate_id"].astype(str) == str(summary["best_candidate_id"])
            ].copy()
        if best_rows.empty:
            best_rows = candidate_metrics.copy()
        ranked = best_rows.sort_values(
            ["rank_score", "weighted_f1_mean", "default_distance", "candidate_id"],
            ascending=[False, False, True, True],
        )
        best_candidate = _to_serializable(ranked.iloc[0].to_dict())
    payload = {
        "schema_version": SCHEMA_VERSION,
        "sweep_id": task.sweep_id,
        "experiment_family": EXPERIMENT_FAMILY,
        "mh_level": task.scope.mh_level,
        "granularity": task.scope.granularity,
        "scope_id": task.scope.scope_id,
        "dataset_name": task.scope.dataset_name,
        "detector_family": task.detector_family,
        "detector_label": spec.display_name,
        "model_name": model_name,
        "status": status,
        "updated_model": bool(updated_model),
        "best_candidate": best_candidate,
        "summary": dict(summary),
        "error": error,
        "git_commit": resolve_git_commit(_PROJECT_ROOT),
    }
    _json_dump(task.output_dir / "best_configuration.json", payload)
    _json_dump(task.output_dir / "summary.json", payload)
    (task.output_dir / "summary.md").write_text(json.dumps(_to_serializable(payload), indent=2) + "\n", encoding="utf-8")
    return payload


def run_forest_task(task: ForestTask) -> ForestTaskResult:
    spec = FOREST_SPECS[task.detector_family]
    model_name = f"{task.scope.dataset_name}_{task.scope.mh_level}"
    task_start = pd.Timestamp.utcnow()

    try:
        persistence = ModelPersistence(model_root=task.model_root)
        train_df = pd.read_parquet(task.scope.train_path)
        X_train = extract_features_vectorized(train_df)
        thresholds = np.linspace(spec.min_threshold, spec.max_threshold, task.steps_override or spec.steps)
        min_successful_trials = min(3, task.attempts)
        split_paths = {
            "new_prices": task.scope.test_new_prices_path,
            "new_products": task.scope.test_new_products_path,
        }
        prepared_splits: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
        for split_name in task.splits:
            test_df = pd.read_parquet(split_paths[split_name])
            prepared_splits[split_name] = _maybe_subset_frames(
                train_df,
                test_df,
                max_products=task.max_products,
                max_history_per_product=task.max_history_per_product,
                min_test_rows=task.min_test_rows,
                max_test_rows=task.max_test_rows,
            )

        fixed_profile = _fixed_search_profile(task)
        baseline_params = dict(fixed_profile.params) if fixed_profile is not None else _baseline_params(task.detector_family)
        baseline_config_key = _config_key(task.detector_family, baseline_params)
        if fixed_profile is not None:
            search_result = _run_fixed_profile_search(
                task=task,
                model_name=model_name,
                X_train=X_train,
                train_df=train_df,
                prepared_splits=prepared_splits,
                min_successful_trials=min_successful_trials,
                profile=fixed_profile,
            )
        elif task.detector_family == "eif":
            search_result = _run_two_pass_eif_search(
                task=task,
                spec=spec,
                model_name=model_name,
                X_train=X_train,
                train_df=train_df,
                prepared_splits=prepared_splits,
                min_successful_trials=min_successful_trials,
            )
        else:
            thresholds = np.linspace(spec.min_threshold, spec.max_threshold, task.steps_override or spec.steps)
            fixed_configs = _fixed_promoted_configs(task.detector_family)
            all_params = fixed_configs if fixed_configs is not None else _iter_hyperparameter_configs(task.detector_family)
            params_by_key = {_config_key(task.detector_family, params): params for params in all_params}
            stage1_candidate_metrics = pd.DataFrame()
            screening_threshold = np.nan
            screened_config_count = len(all_params)
            if fixed_configs is not None:
                promoted_config_keys = list(params_by_key)
                current_source_metrics: pd.DataFrame | None = None
            else:
                stage1_thresholds = np.asarray([float(STAGE1_THRESHOLDS[task.detector_family])], dtype=np.float64)
                screening_threshold = float(stage1_thresholds[0])
                stage1_frames: list[pd.DataFrame] = []
                for params in all_params:
                    stage1_metrics, _ = _evaluate_config_thresholds(
                        task=task,
                        detector_family=task.detector_family,
                        model_name=model_name,
                        X_train=X_train,
                        train_df=train_df,
                        prepared_splits=prepared_splits,
                        params=params,
                        thresholds=stage1_thresholds,
                        min_successful_trials=min_successful_trials,
                    )
                    stage1_frames.append(stage1_metrics)

                stage1_candidate_metrics = pd.concat(stage1_frames, ignore_index=True)
                screened_config_count = len(stage1_candidate_metrics["config_key"].astype(str).drop_duplicates())
                promoted_config_keys = _select_promoted_config_keys(
                    stage1_candidate_metrics,
                    detector_family=task.detector_family,
                    target_metric=task.target_metric,
                    min_precision=task.min_precision,
                )
                current_source_metrics = stage1_candidate_metrics

            current_eval: ForestConfigEvaluation | None = None
            if baseline_config_key not in set(promoted_config_keys):
                current_source_metrics, current_eval = _evaluate_config_thresholds(
                    task=task,
                    detector_family=task.detector_family,
                    model_name=model_name,
                    X_train=X_train,
                    train_df=train_df,
                    prepared_splits=prepared_splits,
                    params=baseline_params,
                    thresholds=thresholds,
                    min_successful_trials=min_successful_trials,
                )

            config_evaluations: dict[str, ForestConfigEvaluation] = {}
            candidate_frames: list[pd.DataFrame] = []
            for config_key in promoted_config_keys:
                combined_metrics, config_evaluation = _evaluate_config_thresholds(
                    task=task,
                    detector_family=task.detector_family,
                    model_name=model_name,
                    X_train=X_train,
                    train_df=train_df,
                    prepared_splits=prepared_splits,
                    params=params_by_key[config_key],
                    thresholds=thresholds,
                    min_successful_trials=min_successful_trials,
                )
                candidate_frames.append(combined_metrics)
                config_evaluations[config_key] = config_evaluation

            if not candidate_frames:
                raise RuntimeError("No candidate metrics were produced")

            search_result = ForestSearchResult(
                candidate_metrics=pd.concat(candidate_frames, ignore_index=True),
                config_evaluations=config_evaluations,
                current_source_metrics=current_source_metrics,
                current_eval=current_eval,
                screening_threshold=screening_threshold,
                screened_config_count=screened_config_count,
                promoted_config_count=len(config_evaluations),
                stage_label="threshold_sweep",
            )
        candidate_metrics = search_result.candidate_metrics.copy()
        run_id = create_run_id(f"{task.detector_family}_{slugify(task.scope.scope_id)}")
        candidate_metrics.insert(0, "sweep_id", task.sweep_id)
        candidate_metrics.insert(1, "run_id", run_id)
        if "candidate_id" in candidate_metrics.columns:
            candidate_metrics = candidate_metrics.drop(columns=["candidate_id"])
        candidate_metrics.insert(
            2,
            "candidate_id",
            [
                _candidate_id(
                    task.detector_family,
                    {column: row[column] for column in _hyperparameter_columns(task.detector_family)},
                    float(row["threshold"]),
                )
                for _, row in candidate_metrics.iterrows()
            ],
        )
        candidate_metrics.insert(3, "schema_version", SCHEMA_VERSION)
        candidate_metrics["experiment_family"] = EXPERIMENT_FAMILY
        candidate_metrics["detector_family"] = task.detector_family
        candidate_metrics["dataset_name"] = task.scope.dataset_name
        candidate_metrics["dataset_granularity"] = task.scope.granularity
        candidate_metrics["mh_level"] = task.scope.mh_level
        candidate_metrics["scope_id"] = task.scope.scope_id
        candidate_metrics["status"] = "ok"
        candidate_metrics["error"] = ""
        candidate_metrics["stage"] = search_result.stage_label
        candidate_metrics["model_name"] = model_name
        sample_evaluation = search_result.config_evaluations[next(iter(search_result.config_evaluations))]
        candidate_metrics["n_train"] = int(len(train_df))
        candidate_metrics["n_eval_prices"] = next(
            (result.row_count for result in sample_evaluation.split_results if result.split_name == "new_prices"),
            0,
        )
        candidate_metrics["n_eval_products"] = next(
            (result.row_count for result in sample_evaluation.split_results if result.split_name == "new_products"),
            0,
        )
        candidate_metrics["attempt_count"] = task.attempts
        summary = _build_task_summary(
            candidate_metrics=candidate_metrics,
            detector_family=task.detector_family,
            target_metric=task.target_metric,
            min_precision=task.min_precision,
            current_candidate_metrics=(
                None if baseline_config_key in set(search_result.config_evaluations) else search_result.current_source_metrics
            ),
            screened_config_count=search_result.screened_config_count,
            summary_metadata=search_result.summary_metadata,
        )
        candidate_metrics["current_threshold"] = float(summary["current_threshold"])
        candidate_metrics["current_f1"] = float(summary["current_f1"])
        candidate_metrics["improvement_pct"] = np.where(
            candidate_metrics["candidate_id"].astype(str) == str(summary["best_candidate_id"]),
            float(summary["improvement_pct"]),
            np.nan,
        )
        candidate_metrics["screening_threshold"] = search_result.screening_threshold
        candidate_metrics["screened_config_count"] = int(summary["screened_config_count"])
        candidate_metrics["promoted_config_count"] = int(search_result.promoted_config_count)

        split_payload = _build_split_payload(
            detector_family=task.detector_family,
            config_evaluations=(
                {**search_result.config_evaluations, search_result.current_eval.config_key: search_result.current_eval}
                if search_result.current_eval is not None
                else search_result.config_evaluations
            ),
            summary=summary,
        )

        updated_model = False
        if (
            not task.dry_run
            and float(summary["improvement_pct"]) > 1.0
            and float(summary["best_f1"]) >= 0.3
        ):
            best_detector = _train_detector_from_matrix(
                task.detector_family,
                X_train,
                summary["best_hyperparameters"],
                float(summary["best_threshold"]),
            )
            _save_detector(
                persistence,
                task.detector_family,
                best_detector,
                model_name,
                len(train_df),
            )
            updated_model = True

        payload = _write_task_outputs(
            task=task,
            spec=spec,
            model_name=model_name,
            candidate_metrics=candidate_metrics,
            split_payload=split_payload,
            summary=summary,
            updated_model=updated_model,
            status="ok",
        )
        return ForestTaskResult(
            mh_level=task.scope.mh_level,
            granularity=task.scope.granularity,
            scope_id=task.scope.scope_id,
            dataset_name=task.scope.dataset_name,
            detector_family=task.detector_family,
            status="ok",
            output_dir=str(task.output_dir),
            best_candidate=payload.get("best_candidate"),
            error="",
        )
    except Exception as exc:
        error = str(exc)
        candidate_metrics = pd.DataFrame(
            [
                {
                    "sweep_id": task.sweep_id,
                    "run_id": create_run_id(f"{task.detector_family}_{slugify(task.scope.scope_id)}"),
                    "candidate_id": "error",
                    "experiment_family": EXPERIMENT_FAMILY,
                    "detector_family": task.detector_family,
                    "dataset_name": task.scope.dataset_name,
                    "dataset_granularity": task.scope.granularity,
                    "mh_level": task.scope.mh_level,
                    "scope_id": task.scope.scope_id,
                    "status": "error",
                    "error": error,
                    "stage": "threshold_sweep",
                    "model_name": model_name,
                    "n_train": np.nan,
                    "n_eval_prices": np.nan,
                    "n_eval_products": np.nan,
                    "attempt_count": 0,
                    "training_time_sec": np.nan,
                    "combined_accuracy": np.nan,
                    "combined_precision": np.nan,
                    "combined_recall": np.nan,
                    "combined_tnr": np.nan,
                    "combined_fpr": np.nan,
                    "combined_fnr": np.nan,
                    "combined_f1": np.nan,
                    "combined_g_mean": np.nan,
                    "weighted_f1_mean": np.nan,
                    "rank_score": np.nan,
                    "default_distance": np.nan,
                    "config_key": np.nan,
                    "is_baseline_config": False,
                }
            ]
        )
        _write_task_outputs(
            task=task,
            spec=spec,
            model_name=model_name,
            candidate_metrics=candidate_metrics,
            split_payload={},
            summary={},
            updated_model=False,
            status="error",
            error=error,
        )
        return ForestTaskResult(
            mh_level=task.scope.mh_level,
            granularity=task.scope.granularity,
            scope_id=task.scope.scope_id,
            dataset_name=task.scope.dataset_name,
            detector_family=task.detector_family,
            status="error",
            output_dir=str(task.output_dir),
            best_candidate=None,
            error=error,
        )


def _run_process_pool(*, items: Sequence[Any], worker, max_workers: int) -> list[Any]:
    if not items:
        return []
    if max_workers <= 1:
        return [worker(item) for item in items]

    results: list[Any] = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(worker, item): item for item in items}
        for future in as_completed(futures):
            results.append(future.result())
    return results


def _write_sweep_summary(
    *,
    output_root: Path,
    sweep_id: str,
    best_configurations: pd.DataFrame,
    scope_status: pd.DataFrame,
    config: Mapping[str, Any],
) -> None:
    complete_count = int((scope_status["status"] == "ok").sum()) if not scope_status.empty else 0
    error_count = int((scope_status["status"] == "error").sum()) if not scope_status.empty else 0
    payload = {
        "schema_version": SCHEMA_VERSION,
        "sweep_id": sweep_id,
        "experiment_family": EXPERIMENT_FAMILY,
        "complete_task_count": complete_count,
        "error_task_count": error_count,
        "best_configuration_count": int(len(best_configurations)),
        "config": _to_serializable(config),
    }
    _json_dump(output_root / "summary.json", payload)
    (output_root / "summary.md").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune IF/EIF/RRCF across discovered data-subsets scopes.")
    parser.add_argument("--data-subsets-root", default=str(DEFAULT_DATA_SUBSETS_ROOT))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument("--scope-filter", default=None)
    parser.add_argument("--eif-search-strategy", choices=list(EIF_SEARCH_STRATEGIES), default="two_pass")
    parser.add_argument("--detectors", nargs="*", default=list(ALL_FOREST_DETECTORS))
    parser.add_argument("--mh-values", nargs="*", default=list(DEFAULT_SAMPLED_MH_VALUES))
    parser.add_argument("--granularities", nargs="*", choices=list(DEFAULT_FOREST_GRANULARITIES), default=list(DEFAULT_FOREST_GRANULARITIES))
    parser.add_argument("--splits", nargs="*", choices=["new_prices", "new_products"], default=["new_prices", "new_products"])
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--trial-workers", type=int, default=1)
    parser.add_argument("--target-metric", choices=["f1", "precision", "recall", "gmean"], default="gmean")
    parser.add_argument("--min-precision", type=float, default=0.3)
    parser.add_argument("--injection-rate", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--max-products", type=int, default=None)
    parser.add_argument("--max-history-per-product", type=int, default=None)
    parser.add_argument("--min-test-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    unknown = sorted(set(args.detectors).difference(ALL_FOREST_DETECTORS))
    if unknown:
        raise ValueError(f"Unsupported forest detectors {unknown!r}; expected some of {sorted(ALL_FOREST_DETECTORS)!r}")
    if args.attempts < 1:
        raise ValueError("--attempts must be at least 1")
    if args.max_workers < 1:
        raise ValueError("--max-workers must be at least 1")
    if args.trial_workers < 1:
        raise ValueError("--trial-workers must be at least 1")
    if args.resume and not args.output_root:
        raise ValueError("--resume requires --output-root")
    if args.steps is not None and args.steps < 2:
        raise ValueError("--steps must be at least 2")
    if getattr(args, "eif_search_strategy", "two_pass") != "two_pass" and "eif" not in set(args.detectors):
        raise ValueError("--eif-search-strategy requires 'eif' to be included in --detectors")


def run_sweep(args: argparse.Namespace) -> int:
    _validate_args(args)

    data_subsets_root = _resolve_root(args.data_subsets_root)
    model_root = _resolve_root(args.model_root)
    normalized_granularities = _normalize_granularities(args.granularities)

    if args.output_root:
        output_root = _resolve_root(args.output_root)
        sweep_id = output_root.name
    else:
        sweep_id = create_run_id("forests")
        output_root = (DEFAULT_RESULTS_ROOT / sweep_id).resolve()

    if output_root.exists() and not args.resume:
        raise FileExistsError(
            f"Output root already exists: {output_root}. Re-run with --resume to reuse it."
        )
    output_root.mkdir(parents=True, exist_ok=True)

    scopes, skipped_rows = discover_scopes(
        data_subsets_root=data_subsets_root,
        cache_root=output_root / "_cache_snapshots",
        mh_values=args.mh_values,
        granularities=normalized_granularities,
    )
    if args.scope_filter:
        scopes = [scope for scope in scopes if _scope_matches_filter(scope, str(args.scope_filter))]
        skipped_rows = [
            row
            for row in skipped_rows
            if str(args.scope_filter).strip().lower() in str(row.get("scope_id", "")).lower()
            or str(args.scope_filter).strip().lower() in str(row.get("dataset_name", "")).lower()
        ]
    if not scopes and not skipped_rows:
        if args.scope_filter:
            raise FileNotFoundError(
                f"No scopes matched --scope-filter {args.scope_filter!r} under {data_subsets_root}"
            )
        raise FileNotFoundError(f"No scopes discovered under {data_subsets_root}")

    for scope in scopes:
        ensure_cache_snapshot(scope=scope, rebuild_cache=False)

    scheduled_tasks: list[ForestTask] = []
    completed_results: list[ForestTaskResult] = []

    for scope in scopes:
        scope_output_dir = output_root / scope.mh_level / scope.granularity / scope.scope_slug
        for detector_family in args.detectors:
            task_output_dir = scope_output_dir / detector_family
            if args.resume and _task_complete(task_output_dir):
                completed_results.append(_load_existing_task_result(task_output_dir))
                continue
            scheduled_tasks.append(
                ForestTask(
                    scope=scope,
                    detector_family=detector_family,
                    output_dir=task_output_dir,
                    model_root=model_root,
                    sweep_id=sweep_id,
                    attempts=args.attempts,
                    trial_workers=args.trial_workers,
                    target_metric=args.target_metric,
                    min_precision=args.min_precision,
                    injection_rate=args.injection_rate,
                    dry_run=bool(args.dry_run),
                    splits=tuple(args.splits),
                    max_products=args.max_products,
                    max_history_per_product=args.max_history_per_product,
                    min_test_rows=args.min_test_rows,
                    max_test_rows=args.max_test_rows,
                    steps_override=args.steps,
                    eif_search_strategy=str(args.eif_search_strategy),
                )
            )

    completed_results.extend(
        _run_process_pool(items=scheduled_tasks, worker=run_forest_task, max_workers=args.max_workers)
    )

    best_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = list(skipped_rows)

    for result in completed_results:
        best_candidate = result.best_candidate or {}
        status_rows.append(
            {
                "mh_level": result.mh_level,
                "granularity": result.granularity,
                "scope_id": result.scope_id,
                "dataset_name": result.dataset_name,
                "detector_family": result.detector_family,
                "status": result.status,
                "reason": result.error,
                "output_dir": result.output_dir,
                "best_threshold": best_candidate.get("threshold"),
                "best_f1": best_candidate.get("combined_f1"),
                "best_g_mean": best_candidate.get("combined_g_mean"),
                "best_precision": best_candidate.get("combined_precision"),
                "best_recall": best_candidate.get("combined_recall"),
                "best_fpr": best_candidate.get("combined_fpr"),
                "best_fnr": best_candidate.get("combined_fnr"),
            }
        )
        if result.status == "ok" and result.best_candidate is not None:
            row = dict(result.best_candidate)
            row.update(
                {
                    "mh_level": result.mh_level,
                    "granularity": result.granularity,
                    "scope_id": result.scope_id,
                    "dataset_name": result.dataset_name,
                    "detector_family": result.detector_family,
                    "output_dir": result.output_dir,
                }
            )
            best_rows.append(row)

    best_configurations = pd.DataFrame(best_rows)
    scope_status = pd.DataFrame(status_rows)

    if best_configurations.empty:
        pd.DataFrame(columns=["mh_level", "granularity", "scope_id", "detector_family"]).to_csv(
            output_root / "best_configurations.csv",
            index=False,
        )
    else:
        best_configurations.to_csv(output_root / "best_configurations.csv", index=False)

    if scope_status.empty:
        pd.DataFrame(columns=["mh_level", "granularity", "scope_id", "detector_family", "status"]).to_csv(
            output_root / "scope_status.csv",
            index=False,
        )
    else:
        scope_status.to_csv(output_root / "scope_status.csv", index=False)

    _write_sweep_summary(
        output_root=output_root,
        sweep_id=sweep_id,
        best_configurations=best_configurations,
        scope_status=scope_status[scope_status["status"] != "skipped"].copy()
        if not scope_status.empty and "status" in scope_status.columns
        else pd.DataFrame(),
        config={
            "data_subsets_root": str(data_subsets_root),
            "model_root": str(model_root),
            "scope_filter": args.scope_filter,
            "eif_search_strategy": args.eif_search_strategy,
            "detectors": list(args.detectors),
            "mh_values": list(args.mh_values),
            "granularities": normalized_granularities,
            "splits": list(args.splits),
            "attempts": args.attempts,
            "max_workers": args.max_workers,
            "trial_workers": args.trial_workers,
            "target_metric": args.target_metric,
            "min_precision": args.min_precision,
            "injection_rate": args.injection_rate,
            "steps": args.steps,
            "max_products": args.max_products,
            "max_history_per_product": args.max_history_per_product,
            "min_test_rows": args.min_test_rows,
            "max_test_rows": args.max_test_rows,
            "dry_run": bool(args.dry_run),
            "resume": bool(args.resume),
        },
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return run_sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
