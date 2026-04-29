#!/usr/bin/env python3
"""Sweep-style tuner for statistical detectors over ``data-subsets`` scopes."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.anomaly.statistical import (
    DEFAULT_IQR_MULTIPLIER,
    DEFAULT_MODIFIED_ZSCORE_THRESHOLD,
    DEFAULT_PRICE_CHANGE_THRESHOLD,
    DEFAULT_ZSCORE_THRESHOLD,
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
from src.research.mh_sampling import RESEARCH_MH_LEVELS
from src.research.artifacts import (
    comparison_result_to_tables,
    compute_anomaly_type_metrics,
    compute_detector_metrics,
    create_run_id,
    initialize_evaluation_tracking_columns,
    reindex_split_artifacts,
    resolve_git_commit,
    slugify,
    write_evaluation_run,
    write_tuning_sweep,
)
from src.research.evaluation import DetectorEvaluator, TestOrchestrator, inject_anomalies_to_dataframe

LOGGER = logging.getLogger("tune_statistical")

SCHEMA_VERSION = "phase2.v1"
EXPERIMENT_FAMILY = "tuning"
DEFAULT_DATA_SUBSETS_ROOT = _PROJECT_ROOT / "data-subsets"
DEFAULT_CACHE_ROOT = _PROJECT_ROOT / "artifacts" / "cache" / "statistical_tuning"
DEFAULT_RESULTS_ROOT = _PROJECT_ROOT / "results" / "tuning" / "statistical"
DEFAULT_INJECTION_RATE = 0.10
DEFAULT_SPIKE_RANGE = (2.0, 5.0)
DEFAULT_DROP_RANGE = (0.10, 0.50)
DEFAULT_SAMPLED_MH_VALUES = RESEARCH_MH_LEVELS
FIXED_ATTEMPT_SEEDS = [42, 43, 44, 45, 46]
PROMOTION_TOP_K = 2
PROMOTION_WITHIN_BEST_RATIO = 0.95
COMPLETED_STAGE_SEQUENCE = (
    "initialized",
    "coarse_screening_complete",
    "coarse_complete",
    "refined_screening_complete",
    "refined_complete",
    "finalized",
)
ALL_GRANULARITIES = ("global", "by_country", "by_country_market", "by_competitor")
ALL_DETECTOR_FAMILIES = (
    "standard_zscore",
    "modified_mad",
    "modified_sn",
    "hybrid_weighted",
    "hybrid_max",
    "hybrid_avg",
    "iqr",
    "threshold",
)


@dataclass(frozen=True)
class DetectorFamilySpec:
    """Static search-space description for one detector family."""

    param_order: tuple[str, ...]
    coarse_grid: dict[str, tuple[float, ...]]
    refine_radius: dict[str, float]
    refine_step: dict[str, float]
    clip_bounds: dict[str, tuple[float, float]]
    default_params: dict[str, float]


DETECTOR_FAMILY_SPECS: dict[str, DetectorFamilySpec] = {
    "standard_zscore": DetectorFamilySpec(
        param_order=("threshold",),
        coarse_grid={"threshold": (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)},
        refine_radius={"threshold": 0.5},
        refine_step={"threshold": 0.25},
        clip_bounds={"threshold": (2.0, 5.0)},
        default_params={"threshold": DEFAULT_ZSCORE_THRESHOLD},
    ),
    "modified_mad": DetectorFamilySpec(
        param_order=("threshold",),
        coarse_grid={"threshold": (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)},
        refine_radius={"threshold": 0.5},
        refine_step={"threshold": 0.25},
        clip_bounds={"threshold": (1.0, 6.0)},
        default_params={"threshold": DEFAULT_MODIFIED_ZSCORE_THRESHOLD},
    ),
    "modified_sn": DetectorFamilySpec(
        param_order=("threshold",),
        coarse_grid={"threshold": (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)},
        refine_radius={"threshold": 0.5},
        refine_step={"threshold": 0.25},
        clip_bounds={"threshold": (1.0, 6.0)},
        default_params={"threshold": DEFAULT_MODIFIED_ZSCORE_THRESHOLD},
    ),
    "hybrid_weighted": DetectorFamilySpec(
        param_order=("threshold", "w"),
        coarse_grid={
            "threshold": (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            "w": (0.25, 0.75),
        },
        refine_radius={"threshold": 0.0, "w": 0.0},
        refine_step={"threshold": 0.25, "w": 0.25},
        clip_bounds={"threshold": (1.0, 6.0), "w": (0.0, 1.0)},
        default_params={"threshold": DEFAULT_MODIFIED_ZSCORE_THRESHOLD, "w": 0.5},
    ),
    "hybrid_max": DetectorFamilySpec(
        param_order=("threshold",),
        coarse_grid={"threshold": (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)},
        refine_radius={"threshold": 0.5},
        refine_step={"threshold": 0.25},
        clip_bounds={"threshold": (1.0, 6.0)},
        default_params={"threshold": DEFAULT_MODIFIED_ZSCORE_THRESHOLD},
    ),
    "hybrid_avg": DetectorFamilySpec(
        param_order=("threshold",),
        coarse_grid={"threshold": (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)},
        refine_radius={"threshold": 0.5},
        refine_step={"threshold": 0.25},
        clip_bounds={"threshold": (1.0, 6.0)},
        default_params={"threshold": DEFAULT_MODIFIED_ZSCORE_THRESHOLD},
    ),
    "iqr": DetectorFamilySpec(
        param_order=("multiplier",),
        coarse_grid={"multiplier": (1.0, 1.5, 2.0, 2.5, 3.0)},
        refine_radius={"multiplier": 0.5},
        refine_step={"multiplier": 0.25},
        clip_bounds={"multiplier": (1.0, 3.0)},
        default_params={"multiplier": DEFAULT_IQR_MULTIPLIER},
    ),
    "threshold": DetectorFamilySpec(
        param_order=("price_change_threshold",),
        coarse_grid={"price_change_threshold": (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)},
        refine_radius={"price_change_threshold": 0.05},
        refine_step={"price_change_threshold": 0.025},
        clip_bounds={"price_change_threshold": (0.10, 0.50)},
        default_params={"price_change_threshold": DEFAULT_PRICE_CHANGE_THRESHOLD},
    ),
}


@dataclass(frozen=True)
class ScopeDescriptor:
    """Fully resolved tuning scope."""

    mh_level: str
    granularity: str
    scope_id: str
    dataset_name: str
    train_path: Path
    test_new_prices_path: Path
    test_new_products_path: Path
    cache_snapshot_path: Path

    @property
    def cache_metadata_path(self) -> Path:
        return self.cache_snapshot_path.with_name("cache_metadata.json")

    @property
    def scope_slug(self) -> str:
        return slugify(self.scope_id)


@dataclass(frozen=True)
class CandidateSpec:
    """One concrete hyperparameter candidate."""

    detector_family: str
    candidate_id: str
    stage: str
    params: dict[str, float]


@dataclass
class SplitAttempt:
    """One deterministic injected evaluation attempt for a split."""

    split_name: str
    attempt_index: int
    seed: int
    frame: pd.DataFrame
    labels: np.ndarray
    injection_details: list[dict[str, Any]]


@dataclass
class CachedScopeContext:
    """Loaded scope data reused across all candidates in a worker."""

    scope: ScopeDescriptor
    train_df: pd.DataFrame
    template_cache: TemporalCacheManager
    new_prices_attempts: list[SplitAttempt]
    new_products_attempts: list[SplitAttempt]
    country: str | None


@dataclass
class EvaluatedCandidate:
    """Aggregate result for one candidate across both splits and all attempts."""

    candidate: CandidateSpec
    row: dict[str, Any]
    detector_metrics: pd.DataFrame
    anomaly_type_metrics: pd.DataFrame


@dataclass(frozen=True)
class DetectorFamilyTask:
    """Top-level process task for one scope-family pair."""

    scope: ScopeDescriptor
    detector_family: str
    family_output_dir: Path
    sweep_id: str
    attempt_seeds: tuple[int, ...]


@dataclass
class DetectorFamilyTaskResult:
    """Serializable result returned by a worker."""

    mh_level: str
    granularity: str
    scope_id: str
    dataset_name: str
    detector_family: str
    status: str
    family_output_dir: str
    best_candidate: dict[str, Any] | None = None
    error: str = ""


def _project_root() -> Path:
    return _PROJECT_ROOT


def _resolve_root(path_value: str | Path, *, default: Path | None = None) -> Path:
    raw_path = Path(path_value) if path_value is not None else default
    if raw_path is None:
        raise ValueError("A path value is required")
    if raw_path.is_absolute():
        return raw_path
    return (_project_root() / raw_path).resolve()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_to_serializable(payload), handle, indent=2)
        handle.write("\n")


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    return value


def _extract_dataset_name(train_path: Path) -> str:
    stem = train_path.stem
    if stem.endswith("_train"):
        stem = stem[: -len("_train")]
    stem = re.sub(r"_\d{4}-\d{2}-\d{2}$", "", stem)
    return stem


def _scope_sort_key(scope: ScopeDescriptor) -> tuple[int, str, str]:
    return int(scope.mh_level.removeprefix("mh")), scope.granularity, scope.scope_id


def _normalize_mh_values(values: Sequence[str] | None) -> set[str] | None:
    if not values:
        return None
    normalized = set()
    for value in values:
        token = str(value).strip().lower()
        if not token:
            continue
        normalized.add(token if token.startswith("mh") else f"mh{token}")
    return normalized


def _normalize_granularities(values: Sequence[str] | None) -> set[str] | None:
    if not values:
        return None
    normalized = {str(value).strip() for value in values if str(value).strip()}
    unknown = sorted(normalized.difference(ALL_GRANULARITIES))
    if unknown:
        raise ValueError(
            f"Unsupported granularities {unknown!r}; expected some of {sorted(ALL_GRANULARITIES)!r}"
        )
    return normalized


def _extract_country_token(dataset_name: str | None) -> str | None:
    if not dataset_name:
        return None
    if dataset_name.startswith("COUNTRY_"):
        parts = dataset_name.split("_")
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
    return None


def discover_scopes(
    *,
    data_subsets_root: Path,
    cache_root: Path,
    mh_values: Sequence[str] | None = None,
    granularities: Sequence[str] | None = None,
) -> tuple[list[ScopeDescriptor], list[dict[str, Any]]]:
    """Discover all complete scopes and record incomplete ones as skipped."""
    normalized_mh = _normalize_mh_values(mh_values)
    normalized_granularities = _normalize_granularities(granularities)

    scopes: list[ScopeDescriptor] = []
    skipped: list[dict[str, Any]] = []

    for mh_dir in sorted(path for path in data_subsets_root.iterdir() if path.is_dir()):
        if normalized_mh and mh_dir.name.lower() not in normalized_mh:
            continue
        if not mh_dir.name.lower().startswith("mh"):
            continue

        for granularity_dir in sorted(path for path in mh_dir.iterdir() if path.is_dir()):
            if normalized_granularities and granularity_dir.name not in normalized_granularities:
                continue
            if granularity_dir.name not in ALL_GRANULARITIES:
                continue

            for train_path in sorted(granularity_dir.rglob("*_train.parquet")):
                relative_train = train_path.relative_to(granularity_dir)
                scope_id = relative_train.as_posix().removesuffix("_train.parquet")
                dataset_name = _extract_dataset_name(train_path)
                base_name = train_path.name.removesuffix("_train.parquet")
                test_new_prices_path = train_path.with_name(f"{base_name}_test_new_prices.parquet")
                test_new_products_path = train_path.with_name(f"{base_name}_test_new_products.parquet")
                scope_slug = slugify(scope_id)

                if not test_new_prices_path.exists() or not test_new_products_path.exists():
                    skipped.append(
                        {
                            "mh_level": mh_dir.name,
                            "granularity": granularity_dir.name,
                            "scope_id": scope_id,
                            "dataset_name": dataset_name,
                            "status": "skipped",
                            "reason": "missing_test_pair",
                            "train_path": str(train_path.resolve()),
                            "test_new_prices_path": str(test_new_prices_path.resolve()),
                            "test_new_products_path": str(test_new_products_path.resolve()),
                            "has_test_new_prices": test_new_prices_path.exists(),
                            "has_test_new_products": test_new_products_path.exists(),
                        }
                    )
                    continue

                scopes.append(
                    ScopeDescriptor(
                        mh_level=mh_dir.name,
                        granularity=granularity_dir.name,
                        scope_id=scope_id,
                        dataset_name=dataset_name,
                        train_path=train_path.resolve(),
                        test_new_prices_path=test_new_prices_path.resolve(),
                        test_new_products_path=test_new_products_path.resolve(),
                        cache_snapshot_path=(
                            cache_root / mh_dir.name / granularity_dir.name / scope_slug / "template_cache.joblib"
                        ).resolve(),
                    )
                )

    scopes.sort(key=_scope_sort_key)
    skipped.sort(key=lambda row: (int(str(row["mh_level"]).removeprefix("mh")), row["granularity"], row["scope_id"]))
    return scopes, skipped


def _snapshot_source_signature(train_path: Path) -> dict[str, Any]:
    stat = train_path.stat()
    return {
        "source_path": str(train_path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _snapshot_matches_source(scope: ScopeDescriptor, metadata: Mapping[str, Any]) -> bool:
    current = _snapshot_source_signature(scope.train_path)
    return (
        str(metadata.get("source_path", "")) == current["source_path"]
        and int(metadata.get("size_bytes", -1)) == current["size_bytes"]
        and int(metadata.get("mtime_ns", -1)) == current["mtime_ns"]
    )


def _build_template_cache(train_df: pd.DataFrame) -> TemporalCacheManager:
    evaluator = DetectorEvaluator(ZScoreDetector(), name="cache_builder")
    evaluator.populate_cache(train_df)
    return evaluator.temporal_cache


def ensure_cache_snapshot(
    *,
    scope: ScopeDescriptor,
    rebuild_cache: bool = False,
) -> dict[str, Any]:
    """Create or validate a reusable cache snapshot for a scope."""
    snapshot_path = scope.cache_snapshot_path
    metadata_path = scope.cache_metadata_path
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    had_existing_snapshot = snapshot_path.exists() and metadata_path.exists()

    if (
        not rebuild_cache
        and snapshot_path.exists()
        and metadata_path.exists()
        and _snapshot_matches_source(scope, _json_load(metadata_path))
    ):
        metadata = _json_load(metadata_path)
        return {
            "mh_level": scope.mh_level,
            "granularity": scope.granularity,
            "scope_id": scope.scope_id,
            "dataset_name": scope.dataset_name,
            "status": "reused",
            "cache_snapshot_path": str(snapshot_path),
            "cache_metadata_path": str(metadata_path),
            "source_path": metadata.get("source_path"),
            "row_count": metadata.get("row_count"),
            "product_count": metadata.get("product_count"),
        }

    train_df = pd.read_parquet(scope.train_path)
    template_cache = _build_template_cache(train_df)
    cache_stats = template_cache.get_stats()
    template_cache.save_to_file(str(snapshot_path))

    metadata = {
        **_snapshot_source_signature(scope.train_path),
        "schema_version": SCHEMA_VERSION,
        "scope_id": scope.scope_id,
        "dataset_name": scope.dataset_name,
        "mh_level": scope.mh_level,
        "granularity": scope.granularity,
        "cache_snapshot_path": str(snapshot_path.resolve()),
        "row_count": int(len(train_df)),
        "product_count": int(cache_stats.get("total_products", 0)),
        "cache_observation_count": int(cache_stats.get("total_observations", 0)),
        "created_at": _utc_now_iso(),
    }
    _json_dump(metadata_path, metadata)

    return {
        "mh_level": scope.mh_level,
        "granularity": scope.granularity,
        "scope_id": scope.scope_id,
        "dataset_name": scope.dataset_name,
        "status": "rebuilt" if had_existing_snapshot else "built",
        "cache_snapshot_path": str(snapshot_path),
        "cache_metadata_path": str(metadata_path),
        "source_path": str(scope.train_path),
        "row_count": int(len(train_df)),
        "product_count": int(cache_stats.get("total_products", 0)),
    }


def _prepare_snapshot_worker(payload: dict[str, Any]) -> dict[str, Any]:
    scope = ScopeDescriptor(
        mh_level=str(payload["mh_level"]),
        granularity=str(payload["granularity"]),
        scope_id=str(payload["scope_id"]),
        dataset_name=str(payload["dataset_name"]),
        train_path=Path(payload["train_path"]),
        test_new_prices_path=Path(payload["test_new_prices_path"]),
        test_new_products_path=Path(payload["test_new_products_path"]),
        cache_snapshot_path=Path(payload["cache_snapshot_path"]),
    )
    return ensure_cache_snapshot(scope=scope, rebuild_cache=bool(payload["rebuild_cache"]))


def _annotate_injected_frame(
    frame: pd.DataFrame,
    *,
    labels: np.ndarray,
    injection_details: list[dict[str, Any]],
    injection_seed: int,
) -> pd.DataFrame:
    annotated = initialize_evaluation_tracking_columns(
        frame,
        injection_seed=injection_seed,
        injection_strategy="synthetic_dataframe_injection",
    )
    annotated["ground_truth_label"] = np.asarray(labels).astype(bool)
    annotated["is_injected"] = annotated["ground_truth_label"]
    annotated["anomaly_type"] = pd.Series([None] * len(annotated), dtype="object")
    annotated["injection_phase"] = pd.Series([pd.NA] * len(annotated), dtype="Int64")
    annotated["injection_params_json"] = "{}"

    if "__original_price__" in annotated.columns:
        annotated["original_price"] = pd.to_numeric(annotated["__original_price__"], errors="coerce")

    for detail in injection_details:
        row_index = int(detail["index"])
        if row_index not in annotated.index:
            continue

        params = {
            key: value
            for key, value in detail.items()
            if key not in {"index", "anomaly_type", "original_price", "new_price"}
        }
        annotated.at[row_index, "anomaly_type"] = detail.get("anomaly_type")
        annotated.at[row_index, "injection_phase"] = detail.get("injection_phase", pd.NA)
        annotated.at[row_index, "injection_params_json"] = json.dumps(_to_serializable(params), sort_keys=True)
        if "original_price" in detail:
            annotated.at[row_index, "original_price"] = detail["original_price"]

    return annotated


def build_split_attempts(
    *,
    split_path: Path,
    split_name: str,
    seeds: Sequence[int],
) -> list[SplitAttempt]:
    """Load a split once and materialize deterministic injected attempts."""
    base_frame = pd.read_parquet(split_path)
    attempts: list[SplitAttempt] = []

    for attempt_index, seed in enumerate(seeds, start=1):
        injected_frame, labels, injection_details = inject_anomalies_to_dataframe(
            base_frame,
            injection_rate=DEFAULT_INJECTION_RATE,
            seed=seed,
            spike_range=DEFAULT_SPIKE_RANGE,
            drop_range=DEFAULT_DROP_RANGE,
        )
        annotated = _annotate_injected_frame(
            injected_frame,
            labels=labels,
            injection_details=injection_details,
            injection_seed=seed,
        )
        attempts.append(
            SplitAttempt(
                split_name=split_name,
                attempt_index=attempt_index,
                seed=seed,
                frame=annotated,
                labels=np.asarray(labels).astype(bool),
                injection_details=injection_details,
            )
        )

    return attempts


def load_cached_scope_context(scope: ScopeDescriptor, *, attempt_seeds: Sequence[int]) -> CachedScopeContext:
    """Load the cached scope context once for a worker process."""
    train_df = pd.read_parquet(scope.train_path)
    template_cache = TemporalCacheManager()
    template_cache.load_from_file(str(scope.cache_snapshot_path))

    return CachedScopeContext(
        scope=scope,
        train_df=train_df,
        template_cache=template_cache,
        new_prices_attempts=build_split_attempts(
            split_path=scope.test_new_prices_path,
            split_name="new_prices",
            seeds=attempt_seeds,
        ),
        new_products_attempts=build_split_attempts(
            split_path=scope.test_new_products_path,
            split_name="new_products",
            seeds=attempt_seeds,
        ),
        country=_extract_country_token(scope.dataset_name),
    )


def _float_range(center: float, radius: float, step: float, lower: float, upper: float) -> tuple[float, ...]:
    values: list[float] = []
    current = center - radius
    while current <= center + radius + (step / 10):
        clipped = min(max(current, lower), upper)
        rounded = round(clipped + 1e-12, 3)
        if rounded not in values:
            values.append(rounded)
        current += step
    values.sort()
    return tuple(values)


def _candidate_id(detector_family: str, params: Mapping[str, float]) -> str:
    segments = [detector_family]
    for key in sorted(params):
        segments.append(f"{key}_{float(params[key]):0.3f}")
    return "__".join(segments).replace(".", "p")


def _expand_grid(param_order: Sequence[str], grid: Mapping[str, Sequence[float]]) -> list[dict[str, float]]:
    keys = list(param_order)
    values = [grid[key] for key in keys]
    return [{key: float(value) for key, value in zip(keys, combo, strict=True)} for combo in product(*values)]


def build_coarse_candidates(detector_family: str) -> list[CandidateSpec]:
    """Build the coarse search grid for a detector family."""
    spec = DETECTOR_FAMILY_SPECS[detector_family]
    return [
        CandidateSpec(
            detector_family=detector_family,
            candidate_id=_candidate_id(detector_family, params),
            stage="coarse",
            params=params,
        )
        for params in _expand_grid(spec.param_order, spec.coarse_grid)
    ]


def build_refined_candidates(detector_family: str, best_coarse_params: Mapping[str, float]) -> list[CandidateSpec]:
    """Build a local refinement grid around the best coarse candidate."""
    spec = DETECTOR_FAMILY_SPECS[detector_family]
    refined_grid: dict[str, tuple[float, ...]] = {}
    for param_name in spec.param_order:
        lower, upper = spec.clip_bounds[param_name]
        refined_grid[param_name] = _float_range(
            center=float(best_coarse_params[param_name]),
            radius=float(spec.refine_radius[param_name]),
            step=float(spec.refine_step[param_name]),
            lower=lower,
            upper=upper,
        )
    return [
        CandidateSpec(
            detector_family=detector_family,
            candidate_id=_candidate_id(detector_family, params),
            stage="refine",
            params=params,
        )
        for params in _expand_grid(spec.param_order, refined_grid)
    ]


def _default_distance(detector_family: str, params: Mapping[str, float]) -> float:
    """Normalized L1 distance to the repository defaults."""
    spec = DETECTOR_FAMILY_SPECS[detector_family]
    deltas: list[float] = []
    for param_name in spec.param_order:
        lower, upper = spec.clip_bounds[param_name]
        span = upper - lower
        if span <= 0:
            continue
        default_value = float(spec.default_params[param_name])
        deltas.append(abs(float(params[param_name]) - default_value) / span)
    if not deltas:
        return 0.0
    return float(sum(deltas) / len(deltas))


def _build_detector(detector_family: str, params: Mapping[str, float]) -> Any:
    if detector_family == "standard_zscore":
        return ZScoreDetector(threshold=float(params["threshold"]))
    if detector_family == "modified_mad":
        return ModifiedMADDetector(threshold=float(params["threshold"]))
    if detector_family == "modified_sn":
        return ModifiedSNDetector(threshold=float(params["threshold"]))
    if detector_family == "hybrid_weighted":
        return HybridWeightedZScoreDetector(
            threshold=float(params["threshold"]),
            w=float(params["w"]),
        )
    if detector_family == "hybrid_max":
        return HybridMaxZScoreDetector(threshold=float(params["threshold"]))
    if detector_family == "hybrid_avg":
        return HybridAvgZScoreDetector(threshold=float(params["threshold"]))
    if detector_family == "iqr":
        return IQRDetector(multiplier=float(params["multiplier"]))
    if detector_family == "threshold":
        return ThresholdDetector(threshold=float(params["price_change_threshold"]))
    raise ValueError(f"Unsupported detector family: {detector_family}")


def _evaluate_attempt(
    *,
    detector_family: str,
    candidate: CandidateSpec,
    context: CachedScopeContext,
    attempt: SplitAttempt,
    run_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detector = _build_detector(detector_family, candidate.params)
    evaluator = DetectorEvaluator(detector, name=detector_family)
    evaluator.temporal_cache.copy_from(context.template_cache)
    comparison = TestOrchestrator([evaluator], max_workers=1).run_comparison_with_details(
        train_df=None,
        test_df=attempt.frame,
        labels=attempt.labels,
        country=context.country,
        skip_cache_setup=True,
        injection_details=attempt.injection_details,
    )

    injected_rows, predictions = comparison_result_to_tables(
        comparison,
        run_id=run_id,
        candidate_id=candidate.candidate_id,
        experiment_family=EXPERIMENT_FAMILY,
        dataset_name=context.scope.dataset_name,
        dataset_granularity=context.scope.granularity,
        dataset_split=attempt.split_name,
        detector_family_map={detector_family: detector_family},
        injected_row_extras={"attempt_index": attempt.attempt_index, "attempt_seed": attempt.seed},
        prediction_extras={"attempt_index": attempt.attempt_index, "attempt_seed": attempt.seed},
    )
    detector_metrics = compute_detector_metrics(injected_rows, predictions)
    detector_metrics["attempt_index"] = attempt.attempt_index
    detector_metrics["attempt_seed"] = attempt.seed
    anomaly_type_metrics = compute_anomaly_type_metrics(injected_rows, predictions)
    anomaly_type_metrics["attempt_index"] = attempt.attempt_index
    anomaly_type_metrics["attempt_seed"] = attempt.seed
    return detector_metrics, anomaly_type_metrics


def _build_failed_candidate_row(
    *,
    sweep_id: str,
    context: CachedScopeContext,
    candidate: CandidateSpec,
    error: str,
    elapsed: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "sweep_id": sweep_id,
        "run_id": candidate.candidate_id,
        "candidate_id": candidate.candidate_id,
        "experiment_family": EXPERIMENT_FAMILY,
        "detector_family": candidate.detector_family,
        "dataset_name": context.scope.dataset_name,
        "dataset_granularity": context.scope.granularity,
        "status": "error",
        "error": error,
        "stage": candidate.stage,
        "training_time_sec": elapsed,
        "n_train": int(len(context.train_df)),
        "n_eval_prices": int(len(context.new_prices_attempts[0].frame)) if context.new_prices_attempts else 0,
        "n_eval_products": int(len(context.new_products_attempts[0].frame))
        if context.new_products_attempts
        else 0,
        "attempt_count": 0,
        "combined_precision": math.nan,
        "combined_recall": math.nan,
        "combined_f1": math.nan,
        "weighted_f1_mean": math.nan,
        "rank_score": math.nan,
        "default_distance": _default_distance(candidate.detector_family, candidate.params),
    }
    for key, value in candidate.params.items():
        row[key] = value
    return row


def _metric_mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def _build_candidate_row(
    *,
    sweep_id: str,
    context: CachedScopeContext,
    candidate: CandidateSpec,
    detector_metrics: pd.DataFrame,
    elapsed: float,
) -> dict[str, Any]:
    split_metrics: dict[str, dict[str, tuple[float, float]]] = {}
    for split_name in ("new_prices", "new_products"):
        split_rows = detector_metrics[detector_metrics["dataset_split"] == split_name]
        split_metrics[split_name] = {}
        for metric_name in ("accuracy", "precision", "recall", "tnr", "f1", "g_mean"):
            split_metrics[split_name][metric_name] = _metric_mean_std(split_rows[metric_name].tolist())

    new_prices_precision_mean, _ = split_metrics["new_prices"]["precision"]
    new_products_precision_mean, _ = split_metrics["new_products"]["precision"]
    new_prices_recall_mean, _ = split_metrics["new_prices"]["recall"]
    new_products_recall_mean, _ = split_metrics["new_products"]["recall"]
    new_prices_f1_mean, _ = split_metrics["new_prices"]["f1"]
    new_products_f1_mean, _ = split_metrics["new_products"]["f1"]
    new_prices_g_mean_mean, new_prices_g_mean_std = split_metrics["new_prices"]["g_mean"]
    new_products_g_mean_mean, new_products_g_mean_std = split_metrics["new_products"]["g_mean"]

    weighted_precision_mean = (0.7 * new_prices_precision_mean) + (0.3 * new_products_precision_mean)
    weighted_recall_mean = (0.7 * new_prices_recall_mean) + (0.3 * new_products_recall_mean)
    weighted_f1_mean = (0.7 * new_prices_f1_mean) + (0.3 * new_products_f1_mean)
    rank_score = (0.7 * new_prices_g_mean_mean) + (0.3 * new_products_g_mean_mean)

    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "sweep_id": sweep_id,
        "run_id": candidate.candidate_id,
        "candidate_id": candidate.candidate_id,
        "experiment_family": EXPERIMENT_FAMILY,
        "detector_family": candidate.detector_family,
        "dataset_name": context.scope.dataset_name,
        "dataset_granularity": context.scope.granularity,
        "status": "ok",
        "error": "",
        "stage": candidate.stage,
        "training_time_sec": elapsed,
        "n_train": int(len(context.train_df)),
        "n_eval_prices": int(len(context.new_prices_attempts[0].frame)) if context.new_prices_attempts else 0,
        "n_eval_products": int(len(context.new_products_attempts[0].frame))
        if context.new_products_attempts
        else 0,
        "attempt_count": int(detector_metrics["attempt_index"].nunique()),
        "combined_precision": weighted_precision_mean,
        "combined_recall": weighted_recall_mean,
        "combined_f1": weighted_f1_mean,
        "weighted_f1_mean": weighted_f1_mean,
        "rank_score": rank_score,
        "default_distance": _default_distance(candidate.detector_family, candidate.params),
        "new_prices_accuracy_mean": split_metrics["new_prices"]["accuracy"][0],
        "new_prices_accuracy_std": split_metrics["new_prices"]["accuracy"][1],
        "new_prices_precision_mean": split_metrics["new_prices"]["precision"][0],
        "new_prices_precision_std": split_metrics["new_prices"]["precision"][1],
        "new_prices_recall_mean": split_metrics["new_prices"]["recall"][0],
        "new_prices_recall_std": split_metrics["new_prices"]["recall"][1],
        "new_prices_f1_mean": split_metrics["new_prices"]["f1"][0],
        "new_prices_f1_std": split_metrics["new_prices"]["f1"][1],
        "new_prices_g_mean_mean": new_prices_g_mean_mean,
        "new_prices_g_mean_std": new_prices_g_mean_std,
        "new_products_accuracy_mean": split_metrics["new_products"]["accuracy"][0],
        "new_products_accuracy_std": split_metrics["new_products"]["accuracy"][1],
        "new_products_precision_mean": split_metrics["new_products"]["precision"][0],
        "new_products_precision_std": split_metrics["new_products"]["precision"][1],
        "new_products_recall_mean": split_metrics["new_products"]["recall"][0],
        "new_products_recall_std": split_metrics["new_products"]["recall"][1],
        "new_products_f1_mean": split_metrics["new_products"]["f1"][0],
        "new_products_f1_std": split_metrics["new_products"]["f1"][1],
        "new_products_g_mean_mean": new_products_g_mean_mean,
        "new_products_g_mean_std": new_products_g_mean_std,
    }
    for key, value in candidate.params.items():
        row[key] = value
    return row


def evaluate_candidate(
    *,
    sweep_id: str,
    detector_family: str,
    context: CachedScopeContext,
    candidate: CandidateSpec,
    max_attempts: int | None = None,
) -> EvaluatedCandidate:
    """Evaluate one candidate sequentially over both splits and all attempts."""
    start = time.perf_counter()
    detector_metric_frames: list[pd.DataFrame] = []
    anomaly_type_frames: list[pd.DataFrame] = []
    prices_attempts = context.new_prices_attempts[:max_attempts] if max_attempts is not None else context.new_prices_attempts
    product_attempts = (
        context.new_products_attempts[:max_attempts] if max_attempts is not None else context.new_products_attempts
    )

    try:
        for attempts in (prices_attempts, product_attempts):
            for attempt in attempts:
                run_id = f"{candidate.candidate_id}__attempt_{attempt.attempt_index:02d}"
                detector_metrics, anomaly_type_metrics = _evaluate_attempt(
                    detector_family=detector_family,
                    candidate=candidate,
                    context=context,
                    attempt=attempt,
                    run_id=run_id,
                )
                for key, value in candidate.params.items():
                    detector_metrics[key] = value
                    anomaly_type_metrics[key] = value
                detector_metrics["stage"] = candidate.stage
                anomaly_type_metrics["stage"] = candidate.stage
                detector_metric_frames.append(detector_metrics)
                anomaly_type_frames.append(anomaly_type_metrics)

        detector_metrics_frame = pd.concat(detector_metric_frames, ignore_index=True)
        anomaly_type_metrics_frame = (
            pd.concat(anomaly_type_frames, ignore_index=True)
            if anomaly_type_frames
            else pd.DataFrame(columns=["run_id", "candidate_id", "dataset_split"])
        )
        elapsed = time.perf_counter() - start
        row = _build_candidate_row(
            sweep_id=sweep_id,
            context=context,
            candidate=candidate,
            detector_metrics=detector_metrics_frame,
            elapsed=elapsed,
        )
        return EvaluatedCandidate(
            candidate=candidate,
            row=row,
            detector_metrics=detector_metrics_frame,
            anomaly_type_metrics=anomaly_type_metrics_frame,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return EvaluatedCandidate(
            candidate=candidate,
            row=_build_failed_candidate_row(
                sweep_id=sweep_id,
                context=context,
                candidate=candidate,
                error=str(exc),
                elapsed=elapsed,
            ),
            detector_metrics=pd.DataFrame(),
            anomaly_type_metrics=pd.DataFrame(),
        )


def select_promoted_candidate_ids(
    candidate_metrics: pd.DataFrame,
    *,
    top_k: int = PROMOTION_TOP_K,
    within_best_ratio: float = PROMOTION_WITHIN_BEST_RATIO,
) -> set[str]:
    """Select candidates that should receive the full multi-attempt evaluation."""
    if candidate_metrics.empty:
        return set()

    valid = candidate_metrics.copy()
    valid = valid[valid["rank_score"].notna()]
    valid = valid[valid["status"].fillna("").astype(str).str.lower().isin({"ok", "success", ""})]
    if valid.empty:
        return set()

    ordered = valid.sort_values(
        ["rank_score", "weighted_f1_mean", "default_distance", "candidate_id"],
        ascending=[False, False, True, True],
    )
    best_rank_score = float(ordered.iloc[0]["rank_score"])
    threshold = best_rank_score * within_best_ratio

    promoted = set(ordered.head(max(top_k, 1))["candidate_id"].astype(str).tolist())
    promoted.update(ordered[ordered["rank_score"] >= threshold]["candidate_id"].astype(str).tolist())
    return promoted


def _with_candidate_status(
    evaluation: EvaluatedCandidate,
    *,
    status: str,
    promotion_status: str,
) -> EvaluatedCandidate:
    row = dict(evaluation.row)
    row["status"] = status
    row["promotion_status"] = promotion_status
    return EvaluatedCandidate(
        candidate=evaluation.candidate,
        row=row,
        detector_metrics=evaluation.detector_metrics,
        anomaly_type_metrics=evaluation.anomaly_type_metrics,
    )


def _screening_shortlist_results(
    screening_evaluations: Sequence[EvaluatedCandidate],
    full_evaluations: Mapping[str, EvaluatedCandidate],
) -> list[EvaluatedCandidate]:
    shortlisted: list[EvaluatedCandidate] = []
    for evaluation in screening_evaluations:
        candidate_id = evaluation.candidate.candidate_id
        if candidate_id in full_evaluations:
            shortlisted.append(_with_candidate_status(full_evaluations[candidate_id], status="ok", promotion_status="promoted"))
        elif str(evaluation.row.get("status", "")).lower() in {"ok", "success", ""}:
            shortlisted.append(
                _with_candidate_status(evaluation, status="screened", promotion_status="screening_only")
            )
        else:
            shortlisted.append(_with_candidate_status(evaluation, status="error", promotion_status="error"))
    return shortlisted


def _completed_stage_rank(stage: str | None) -> int:
    if stage in COMPLETED_STAGE_SEQUENCE:
        return COMPLETED_STAGE_SEQUENCE.index(stage)
    return 0


def _family_progress_path(output_dir: Path) -> Path:
    return output_dir / "progress.json"


def _read_family_progress(output_dir: Path) -> dict[str, Any]:
    path = _family_progress_path(output_dir)
    if not path.exists():
        return {}
    try:
        return _json_load(path)
    except Exception as exc:
        LOGGER.warning("Failed to read family progress from %s: %s", path, exc)
        return {}


def _read_family_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        if path.stat().st_size == 0:
            return pd.DataFrame()
        with path.open("r", encoding="utf-8", newline="") as handle:
            if not handle.read(1024).strip():
                return pd.DataFrame()
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()
    except Exception as exc:
        LOGGER.warning("Failed to read existing progress table %s: %s", path, exc)
        return pd.DataFrame()


def _read_family_tables(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_dir = output_dir / "metrics"
    return (
        _read_family_csv(output_dir / "candidate_metrics.csv"),
        _read_family_csv(metrics_dir / "detector_metrics.csv"),
        _read_family_csv(metrics_dir / "anomaly_type_metrics.csv"),
    )


def _materialize_family_tables(
    evaluations: Sequence[EvaluatedCandidate],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_metrics = pd.DataFrame([evaluation.row for evaluation in evaluations])
    detector_metric_frames = [evaluation.detector_metrics for evaluation in evaluations if not evaluation.detector_metrics.empty]
    anomaly_type_metric_frames = [
        evaluation.anomaly_type_metrics for evaluation in evaluations if not evaluation.anomaly_type_metrics.empty
    ]
    detector_metrics = pd.concat(detector_metric_frames, ignore_index=True) if detector_metric_frames else pd.DataFrame()
    anomaly_type_metrics = (
        pd.concat(anomaly_type_metric_frames, ignore_index=True) if anomaly_type_metric_frames else pd.DataFrame()
    )
    return candidate_metrics, detector_metrics, anomaly_type_metrics


def _write_family_snapshot_tables(
    *,
    task: DetectorFamilyTask,
    evaluations: Sequence[EvaluatedCandidate],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    family_output_dir = task.family_output_dir
    family_output_dir.mkdir(parents=True, exist_ok=True)

    candidate_metrics, detector_metrics, anomaly_type_metrics = _materialize_family_tables(evaluations)
    write_tuning_sweep(
        sweep_root=family_output_dir,
        sweep_metadata=_family_sweep_metadata(task=task, candidates=candidate_metrics),
        candidate_metrics=candidate_metrics,
    )

    metrics_dir = family_output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    detector_metrics.to_csv(metrics_dir / "detector_metrics.csv", index=False)
    anomaly_type_metrics.to_csv(metrics_dir / "anomaly_type_metrics.csv", index=False)
    return candidate_metrics, detector_metrics, anomaly_type_metrics


def _load_saved_evaluations(
    output_dir: Path,
    candidates: Sequence[CandidateSpec],
) -> dict[str, EvaluatedCandidate]:
    candidate_metrics, detector_metrics, anomaly_type_metrics = _read_family_tables(output_dir)
    if candidate_metrics.empty or "candidate_id" not in candidate_metrics.columns:
        return {}

    candidate_frame = candidate_metrics.copy()
    candidate_frame["candidate_id"] = candidate_frame["candidate_id"].fillna("").astype(str)
    candidate_frame = candidate_frame.drop_duplicates(subset=["candidate_id"], keep="last")
    rows_by_id = {str(row["candidate_id"]): row.to_dict() for _, row in candidate_frame.iterrows()}

    detector_groups: dict[str, pd.DataFrame] = {}
    if not detector_metrics.empty and "candidate_id" in detector_metrics.columns:
        detector_groups = {
            str(candidate_id): group.copy()
            for candidate_id, group in detector_metrics.groupby("candidate_id", dropna=False, sort=False)
        }

    anomaly_groups: dict[str, pd.DataFrame] = {}
    if not anomaly_type_metrics.empty and "candidate_id" in anomaly_type_metrics.columns:
        anomaly_groups = {
            str(candidate_id): group.copy()
            for candidate_id, group in anomaly_type_metrics.groupby("candidate_id", dropna=False, sort=False)
        }

    loaded: dict[str, EvaluatedCandidate] = {}
    for candidate in candidates:
        row = rows_by_id.get(candidate.candidate_id)
        if row is None:
            continue
        status = str(row.get("status", "")).strip().lower()
        if status == "error":
            continue
        loaded[candidate.candidate_id] = EvaluatedCandidate(
            candidate=candidate,
            row=row,
            detector_metrics=detector_groups.get(candidate.candidate_id, pd.DataFrame()),
            anomaly_type_metrics=anomaly_groups.get(candidate.candidate_id, pd.DataFrame()),
        )
    return loaded


def _ordered_evaluations(
    candidates: Sequence[CandidateSpec],
    evaluations_by_id: Mapping[str, EvaluatedCandidate],
) -> list[EvaluatedCandidate]:
    return [evaluations_by_id[candidate.candidate_id] for candidate in candidates if candidate.candidate_id in evaluations_by_id]


def _write_family_progress(
    *,
    task: DetectorFamilyTask,
    status: str,
    completed_stage: str,
    active_stage: str,
    evaluations: Sequence[EvaluatedCandidate] | None = None,
    error: str = "",
    started_at: str | None = None,
) -> dict[str, Any]:
    family_output_dir = task.family_output_dir
    family_output_dir.mkdir(parents=True, exist_ok=True)

    if evaluations is not None:
        candidate_metrics, _, _ = _write_family_snapshot_tables(task=task, evaluations=evaluations)
    else:
        candidate_metrics, _, _ = _read_family_tables(family_output_dir)

    existing = _read_family_progress(family_output_dir)
    best_row = select_best_candidate(candidate_metrics) if not candidate_metrics.empty else None
    now = _utc_now_iso()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "sweep_id": task.sweep_id,
        "mh_level": task.scope.mh_level,
        "granularity": task.scope.granularity,
        "scope_id": task.scope.scope_id,
        "dataset_name": task.scope.dataset_name,
        "detector_family": task.detector_family,
        "family_output_dir": str(family_output_dir),
        "cache_snapshot_path": str(task.scope.cache_snapshot_path),
        "attempt_seeds": list(task.attempt_seeds),
        "status": status,
        "completed_stage": completed_stage,
        "active_stage": active_stage,
        "started_at": started_at or str(existing.get("started_at", now)),
        "updated_at": now,
        "completed_at": now if status == "complete" else "",
        "error": error,
        "candidate_count": int(len(candidate_metrics)),
        "candidate_ids": sorted(candidate_metrics["candidate_id"].astype(str).tolist())
        if not candidate_metrics.empty and "candidate_id" in candidate_metrics.columns
        else [],
        "best_candidate_id": str(best_row["candidate_id"]) if best_row is not None else "",
    }
    _json_dump(_family_progress_path(family_output_dir), payload)
    return payload


def select_best_candidate(candidate_metrics: pd.DataFrame) -> pd.Series | None:
    """Apply the deterministic candidate ranking and return the winner."""
    if candidate_metrics.empty:
        return None
    valid = candidate_metrics.copy()
    valid = valid[valid["rank_score"].notna()]
    valid = valid[valid["status"].fillna("").astype(str).str.lower().isin({"ok", "success", ""})]
    if valid.empty:
        return None
    return valid.sort_values(
        ["rank_score", "weighted_f1_mean", "default_distance", "candidate_id"],
        ascending=[False, False, True, True],
    ).iloc[0]


def _candidate_lookup(candidates: Sequence[CandidateSpec]) -> dict[str, CandidateSpec]:
    return {candidate.candidate_id: candidate for candidate in candidates}


def _combine_attempt_artifacts(
    split_artifacts: Sequence[tuple[pd.DataFrame, pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reindexed = reindex_split_artifacts(split_artifacts)
    injected_rows = pd.concat([rows for rows, _ in reindexed], ignore_index=True)
    predictions = pd.concat([rows for _, rows in reindexed], ignore_index=True)
    return injected_rows, predictions


def build_best_candidate_split_artifacts(
    *,
    detector_family: str,
    context: CachedScopeContext,
    candidate: CandidateSpec,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Rerun the winning candidate and pool row-level outputs across attempts."""
    split_artifacts: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    for split_name, attempts in (
        ("new_prices", context.new_prices_attempts),
        ("new_products", context.new_products_attempts),
    ):
        per_attempt_artifacts: list[tuple[pd.DataFrame, pd.DataFrame]] = []
        for attempt in attempts:
            detector = _build_detector(detector_family, candidate.params)
            evaluator = DetectorEvaluator(detector, name=detector_family)
            evaluator.temporal_cache.copy_from(context.template_cache)
            comparison = TestOrchestrator([evaluator], max_workers=1).run_comparison_with_details(
                train_df=None,
                test_df=attempt.frame,
                labels=attempt.labels,
                country=context.country,
                skip_cache_setup=True,
                injection_details=attempt.injection_details,
            )
            per_attempt_artifacts.append(
                comparison_result_to_tables(
                    comparison,
                    run_id=candidate.candidate_id,
                    candidate_id=candidate.candidate_id,
                    experiment_family=EXPERIMENT_FAMILY,
                    dataset_name=context.scope.dataset_name,
                    dataset_granularity=context.scope.granularity,
                    dataset_split=split_name,
                    detector_family_map={detector_family: detector_family},
                    injected_row_extras={"attempt_index": attempt.attempt_index, "attempt_seed": attempt.seed},
                    prediction_extras={"attempt_index": attempt.attempt_index, "attempt_seed": attempt.seed},
                )
            )
        split_artifacts[split_name] = _combine_attempt_artifacts(per_attempt_artifacts)

    return split_artifacts


def _family_sweep_metadata(
    *,
    task: DetectorFamilyTask,
    candidates: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_family": EXPERIMENT_FAMILY,
        "detector_family": task.detector_family,
        "sweep_id": task.sweep_id,
        "mh_level": task.scope.mh_level,
        "granularity": task.scope.granularity,
        "scope_id": task.scope.scope_id,
        "dataset_name": task.scope.dataset_name,
        "source_dataset_paths": [
            str(task.scope.train_path),
            str(task.scope.test_new_prices_path),
            str(task.scope.test_new_products_path),
        ],
        "dataset_granularity": task.scope.granularity,
        "dataset_splits": ["new_prices", "new_products"],
        "cache_snapshot_path": str(task.scope.cache_snapshot_path),
        "cache_metadata_path": str(task.scope.cache_metadata_path),
        "attempt_seeds": list(task.attempt_seeds),
        "attempt_count": len(task.attempt_seeds),
        "injection_config": {
            "injection_rate": DEFAULT_INJECTION_RATE,
            "spike_range": list(DEFAULT_SPIKE_RANGE),
            "drop_range": list(DEFAULT_DROP_RANGE),
        },
        "search_space": _to_serializable(asdict(DETECTOR_FAMILY_SPECS[task.detector_family])),
        "candidate_count": int(len(candidates)),
        "generated_at": _utc_now_iso(),
        "git_commit": resolve_git_commit(_project_root()),
    }


def _best_candidate_run_metadata(
    *,
    task: DetectorFamilyTask,
    candidate: CandidateSpec,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_family": EXPERIMENT_FAMILY,
        "run_id": candidate.candidate_id,
        "candidate_id": candidate.candidate_id,
        "sweep_id": task.sweep_id,
        "mh_level": task.scope.mh_level,
        "granularity": task.scope.granularity,
        "scope_id": task.scope.scope_id,
        "dataset_name": task.scope.dataset_name,
        "source_dataset_paths": [
            str(task.scope.train_path),
            str(task.scope.test_new_prices_path),
            str(task.scope.test_new_products_path),
        ],
        "dataset_names": [task.scope.dataset_name],
        "dataset_granularity": task.scope.granularity,
        "dataset_splits": ["new_prices", "new_products"],
        "cache_snapshot_path": str(task.scope.cache_snapshot_path),
        "random_seeds": {"attempt_seeds": list(task.attempt_seeds)},
        "injection_config": {
            "injection_rate": DEFAULT_INJECTION_RATE,
            "spike_range": list(DEFAULT_SPIKE_RANGE),
            "drop_range": list(DEFAULT_DROP_RANGE),
        },
        "detector_identifiers": [task.detector_family],
        "config_values": candidate.params,
        "generated_at": _utc_now_iso(),
        "git_commit": resolve_git_commit(_project_root()),
    }


def _write_detector_family_outputs(
    *,
    task: DetectorFamilyTask,
    context: CachedScopeContext,
    evaluations: Sequence[EvaluatedCandidate],
) -> dict[str, Any]:
    family_output_dir = task.family_output_dir
    candidate_metrics, _, _ = _write_family_snapshot_tables(task=task, evaluations=evaluations)

    best_row = select_best_candidate(candidate_metrics)
    if best_row is None:
        best_configuration = {
            "schema_version": SCHEMA_VERSION,
            "sweep_id": task.sweep_id,
            "mh_level": task.scope.mh_level,
            "granularity": task.scope.granularity,
            "scope_id": task.scope.scope_id,
            "dataset_name": task.scope.dataset_name,
            "detector_family": task.detector_family,
            "status": "error",
            "cache_snapshot_path": str(task.scope.cache_snapshot_path),
            "attempt_seeds": list(task.attempt_seeds),
            "error": "No successful candidates were produced",
            "generated_at": _utc_now_iso(),
        }
        _json_dump(family_output_dir / "best_configuration.json", best_configuration)
        return best_configuration

    candidates_by_id = _candidate_lookup([evaluation.candidate for evaluation in evaluations])
    best_candidate = candidates_by_id[str(best_row["candidate_id"])]
    best_split_artifacts = build_best_candidate_split_artifacts(
        detector_family=task.detector_family,
        context=context,
        candidate=best_candidate,
    )
    write_evaluation_run(
        run_root=family_output_dir / "best_candidate",
        run_metadata=_best_candidate_run_metadata(task=task, candidate=best_candidate),
        split_artifacts=best_split_artifacts,
    )

    best_configuration = {
        "schema_version": SCHEMA_VERSION,
        "sweep_id": task.sweep_id,
        "mh_level": task.scope.mh_level,
        "granularity": task.scope.granularity,
        "scope_id": task.scope.scope_id,
        "dataset_name": task.scope.dataset_name,
        "detector_family": task.detector_family,
        "cache_snapshot_path": str(task.scope.cache_snapshot_path),
        "attempt_seeds": list(task.attempt_seeds),
        "best_candidate": _to_serializable(best_row.to_dict()),
        "configuration": _to_serializable(best_candidate.params),
        "generated_at": _utc_now_iso(),
    }
    _json_dump(family_output_dir / "best_configuration.json", best_configuration)
    return best_configuration


def run_detector_family_task(task: DetectorFamilyTask) -> DetectorFamilyTaskResult:
    """Worker entry-point for one scope-family pair."""
    completed_stage = "initialized"
    active_stage = "starting"
    progress = _read_family_progress(task.family_output_dir)
    started_at = str(progress.get("started_at", _utc_now_iso()))
    try:
        completed_stage = str(progress.get("completed_stage", completed_stage))
        _write_family_progress(
            task=task,
            status="running",
            completed_stage=completed_stage,
            active_stage=active_stage,
            evaluations=None,
            started_at=started_at,
        )
        context = load_cached_scope_context(task.scope, attempt_seeds=list(task.attempt_seeds))
        coarse_candidates = build_coarse_candidates(task.detector_family)

        if _completed_stage_rank(completed_stage) >= _completed_stage_rank("coarse_screening_complete"):
            coarse_screening_by_id = _load_saved_evaluations(task.family_output_dir, coarse_candidates)
        else:
            coarse_screening_by_id = {}

        missing_coarse_screening = [
            candidate for candidate in coarse_candidates if candidate.candidate_id not in coarse_screening_by_id
        ]
        if missing_coarse_screening:
            active_stage = "coarse_screening"
            _write_family_progress(
                task=task,
                status="running",
                completed_stage=completed_stage,
                active_stage=active_stage,
                evaluations=None,
                started_at=started_at,
            )
            for candidate in missing_coarse_screening:
                coarse_screening_by_id[candidate.candidate_id] = evaluate_candidate(
                    sweep_id=task.sweep_id,
                    detector_family=task.detector_family,
                    context=context,
                    candidate=candidate,
                    max_attempts=1,
                )

        coarse_screening = _ordered_evaluations(coarse_candidates, coarse_screening_by_id)
        if missing_coarse_screening or _completed_stage_rank(completed_stage) < _completed_stage_rank("coarse_screening_complete"):
            completed_stage = "coarse_screening_complete"
            active_stage = "idle"
            _write_family_progress(
                task=task,
                status="running",
                completed_stage=completed_stage,
                active_stage=active_stage,
                evaluations=coarse_screening,
                started_at=started_at,
            )

        coarse_metrics = pd.DataFrame([evaluation.row for evaluation in coarse_screening])
        best_coarse_row = select_best_candidate(coarse_metrics)
        if best_coarse_row is None:
            best_configuration = _write_detector_family_outputs(
                task=task,
                context=context,
                evaluations=coarse_screening,
            )
            return DetectorFamilyTaskResult(
                mh_level=task.scope.mh_level,
                granularity=task.scope.granularity,
                scope_id=task.scope.scope_id,
                dataset_name=task.scope.dataset_name,
                detector_family=task.detector_family,
                status="error",
                family_output_dir=str(task.family_output_dir),
                best_candidate=None,
                error=str(best_configuration.get("error", "No successful candidates were produced")),
            )

        if _completed_stage_rank(completed_stage) >= _completed_stage_rank("coarse_complete"):
            coarse_evaluations = _ordered_evaluations(
                coarse_candidates,
                _load_saved_evaluations(task.family_output_dir, coarse_candidates),
            )
        elif len(task.attempt_seeds) > 1:
            promoted_coarse = select_promoted_candidate_ids(coarse_metrics)
            active_stage = "coarse_full"
            _write_family_progress(
                task=task,
                status="running",
                completed_stage=completed_stage,
                active_stage=active_stage,
                evaluations=coarse_screening,
                started_at=started_at,
            )
            coarse_full = {
                candidate.candidate_id: evaluate_candidate(
                    sweep_id=task.sweep_id,
                    detector_family=task.detector_family,
                    context=context,
                    candidate=candidate,
                )
                for candidate in coarse_candidates
                if candidate.candidate_id in promoted_coarse
            }
            coarse_evaluations = _screening_shortlist_results(coarse_screening, coarse_full)
        else:
            coarse_evaluations = coarse_screening

        if _completed_stage_rank(completed_stage) < _completed_stage_rank("coarse_complete"):
            completed_stage = "coarse_complete"
            active_stage = "idle"
            _write_family_progress(
                task=task,
                status="running",
                completed_stage=completed_stage,
                active_stage=active_stage,
                evaluations=coarse_evaluations,
                started_at=started_at,
            )

        coarse_metrics = pd.DataFrame([evaluation.row for evaluation in coarse_evaluations])
        best_coarse_row = select_best_candidate(coarse_metrics)
        if best_coarse_row is None:
            best_configuration = _write_detector_family_outputs(
                task=task,
                context=context,
                evaluations=coarse_evaluations,
            )
            return DetectorFamilyTaskResult(
                mh_level=task.scope.mh_level,
                granularity=task.scope.granularity,
                scope_id=task.scope.scope_id,
                dataset_name=task.scope.dataset_name,
                detector_family=task.detector_family,
                status="error",
                family_output_dir=str(task.family_output_dir),
                best_candidate=None,
                error=str(best_configuration.get("error", "No successful candidates were produced")),
            )

        refined_candidates = build_refined_candidates(task.detector_family, dict(best_coarse_row))
        seen_candidate_ids = {candidate.candidate_id for candidate in coarse_candidates}
        refined_candidates = [candidate for candidate in refined_candidates if candidate.candidate_id not in seen_candidate_ids]

        if _completed_stage_rank(completed_stage) >= _completed_stage_rank("refined_complete"):
            refined_evaluations = _ordered_evaluations(
                refined_candidates,
                _load_saved_evaluations(task.family_output_dir, refined_candidates),
            )
        else:
            if _completed_stage_rank(completed_stage) >= _completed_stage_rank("refined_screening_complete"):
                refined_screening_by_id = _load_saved_evaluations(task.family_output_dir, refined_candidates)
            else:
                refined_screening_by_id = {}

            missing_refined_screening = [
                candidate for candidate in refined_candidates if candidate.candidate_id not in refined_screening_by_id
            ]
            if missing_refined_screening:
                active_stage = "refined_screening"
                _write_family_progress(
                    task=task,
                    status="running",
                    completed_stage=completed_stage,
                    active_stage=active_stage,
                    evaluations=coarse_evaluations,
                    started_at=started_at,
                )
                for candidate in missing_refined_screening:
                    refined_screening_by_id[candidate.candidate_id] = evaluate_candidate(
                        sweep_id=task.sweep_id,
                        detector_family=task.detector_family,
                        context=context,
                        candidate=candidate,
                        max_attempts=1,
                    )

            refined_screening = _ordered_evaluations(refined_candidates, refined_screening_by_id)
            if (
                missing_refined_screening
                or _completed_stage_rank(completed_stage) < _completed_stage_rank("refined_screening_complete")
            ):
                completed_stage = "refined_screening_complete"
                active_stage = "idle"
                _write_family_progress(
                    task=task,
                    status="running",
                    completed_stage=completed_stage,
                    active_stage=active_stage,
                    evaluations=coarse_evaluations + refined_screening,
                    started_at=started_at,
                )

            if len(task.attempt_seeds) > 1 and refined_screening:
                refined_metrics = pd.DataFrame([evaluation.row for evaluation in refined_screening])
                promoted_refined = select_promoted_candidate_ids(refined_metrics)
                active_stage = "refined_full"
                _write_family_progress(
                    task=task,
                    status="running",
                    completed_stage=completed_stage,
                    active_stage=active_stage,
                    evaluations=coarse_evaluations + refined_screening,
                    started_at=started_at,
                )
                refined_full = {
                    candidate.candidate_id: evaluate_candidate(
                        sweep_id=task.sweep_id,
                        detector_family=task.detector_family,
                        context=context,
                        candidate=candidate,
                    )
                    for candidate in refined_candidates
                    if candidate.candidate_id in promoted_refined
                }
                refined_evaluations = _screening_shortlist_results(refined_screening, refined_full)
            else:
                refined_evaluations = refined_screening

            completed_stage = "refined_complete"
            active_stage = "idle"
            _write_family_progress(
                task=task,
                status="running",
                completed_stage=completed_stage,
                active_stage=active_stage,
                evaluations=coarse_evaluations + refined_evaluations,
                started_at=started_at,
            )

        all_evaluations = coarse_evaluations + refined_evaluations
        active_stage = "finalize"
        _write_family_progress(
            task=task,
            status="running",
            completed_stage=completed_stage,
            active_stage=active_stage,
            evaluations=all_evaluations,
            started_at=started_at,
        )
        best_configuration = _write_detector_family_outputs(
            task=task,
            context=context,
            evaluations=all_evaluations,
        )
        completed_stage = "finalized"
        active_stage = "idle"
        _write_family_progress(
            task=task,
            status="complete",
            completed_stage=completed_stage,
            active_stage=active_stage,
            evaluations=all_evaluations,
            started_at=started_at,
        )
        best_candidate = best_configuration.get("best_candidate")
        status = "ok" if best_candidate else "error"
        return DetectorFamilyTaskResult(
            mh_level=task.scope.mh_level,
            granularity=task.scope.granularity,
            scope_id=task.scope.scope_id,
            dataset_name=task.scope.dataset_name,
            detector_family=task.detector_family,
            status=status,
            family_output_dir=str(task.family_output_dir),
            best_candidate=best_candidate if isinstance(best_candidate, dict) else None,
            error=str(best_configuration.get("error", "")),
        )
    except Exception as exc:
        _write_family_progress(
            task=task,
            status="error",
            completed_stage=completed_stage,
            active_stage=active_stage,
            evaluations=None,
            error=str(exc),
            started_at=started_at,
        )
        return DetectorFamilyTaskResult(
            mh_level=task.scope.mh_level,
            granularity=task.scope.granularity,
            scope_id=task.scope.scope_id,
            dataset_name=task.scope.dataset_name,
            detector_family=task.detector_family,
            status="error",
            family_output_dir=str(task.family_output_dir),
            best_candidate=None,
            error=str(exc),
        )


def _family_complete(output_dir: Path) -> bool:
    required_paths = [
        output_dir / "candidate_metrics.csv",
        output_dir / "metrics" / "detector_metrics.csv",
        output_dir / "metrics" / "anomaly_type_metrics.csv",
        output_dir / "summary.json",
        output_dir / "summary.md",
        output_dir / "sweep_metadata.json",
        output_dir / "best_configuration.json",
        output_dir / "best_candidate" / "run_metadata.json",
        output_dir / "best_candidate" / "summary.json",
        output_dir / "best_candidate" / "summary.md",
        output_dir / "best_candidate" / "metrics" / "detector_metrics.csv",
        output_dir / "best_candidate" / "metrics" / "anomaly_type_metrics.csv",
        output_dir / "best_candidate" / "splits" / "new_prices" / "injected_rows.parquet",
        output_dir / "best_candidate" / "splits" / "new_prices" / "predictions.parquet",
        output_dir / "best_candidate" / "splits" / "new_products" / "injected_rows.parquet",
        output_dir / "best_candidate" / "splits" / "new_products" / "predictions.parquet",
    ]
    return all(path.exists() for path in required_paths)


def _load_existing_family_result(output_dir: Path) -> DetectorFamilyTaskResult:
    payload = _json_load(output_dir / "best_configuration.json")
    return DetectorFamilyTaskResult(
        mh_level=str(payload["mh_level"]),
        granularity=str(payload["granularity"]),
        scope_id=str(payload["scope_id"]),
        dataset_name=str(payload["dataset_name"]),
        detector_family=str(payload["detector_family"]),
        status="ok" if payload.get("best_candidate") else "error",
        family_output_dir=str(output_dir),
        best_candidate=payload.get("best_candidate"),
        error=str(payload.get("error", "")),
    )


def _scope_output_dir(output_root: Path, scope: ScopeDescriptor) -> Path:
    return output_root / scope.mh_level / scope.granularity / scope.scope_slug


def _write_scope_configuration(
    *,
    scope: ScopeDescriptor,
    scope_output_dir: Path,
    sweep_id: str,
    attempt_seeds: Sequence[int],
    selected_detector_families: Sequence[str],
    family_results: Sequence[DetectorFamilyTaskResult],
) -> dict[str, Any]:
    detectors: dict[str, Any] = {}
    aggregate_metrics: dict[str, Any] = {}
    for result in family_results:
        if result.best_candidate is None:
            continue
        family_dir = Path(result.family_output_dir)
        payload = _json_load(family_dir / "best_configuration.json")
        detectors[result.detector_family] = payload.get("configuration", {})
        aggregate_metrics[result.detector_family] = payload.get("best_candidate", {})

    status = "complete" if len(detectors) == len(selected_detector_families) else "partial"
    configuration = {
        "schema_version": SCHEMA_VERSION,
        "sweep_id": sweep_id,
        "mh_level": scope.mh_level,
        "granularity": scope.granularity,
        "scope_id": scope.scope_id,
        "dataset_name": scope.dataset_name,
        "status": status,
        "cache_snapshot_path": str(scope.cache_snapshot_path),
        "attempt_seeds": list(attempt_seeds),
        "detector_families_requested": list(selected_detector_families),
        "detectors": detectors,
        "aggregate_metrics": aggregate_metrics,
        "generated_at": _utc_now_iso(),
    }
    _json_dump(scope_output_dir / "statistical_configuration.json", configuration)
    return configuration


def _scope_status_row(
    *,
    scope: ScopeDescriptor,
    scope_output_dir: Path,
    selected_detector_families: Sequence[str],
    family_results: Sequence[DetectorFamilyTaskResult],
    cache_status: Mapping[str, Any] | None,
) -> dict[str, Any]:
    succeeded = sum(1 for result in family_results if result.status == "ok" and result.best_candidate is not None)
    failed_results = [result for result in family_results if result.status != "ok" or result.best_candidate is None]
    status = "complete" if succeeded == len(selected_detector_families) else ("error" if failed_results else "pending")
    return {
        "mh_level": scope.mh_level,
        "granularity": scope.granularity,
        "scope_id": scope.scope_id,
        "dataset_name": scope.dataset_name,
        "status": status,
        "reason": "; ".join(filter(None, [result.error for result in failed_results])),
        "detectors_requested": len(selected_detector_families),
        "detectors_succeeded": succeeded,
        "detectors_failed": len(failed_results),
        "scope_output_dir": str(scope_output_dir),
        "cache_snapshot_path": str(scope.cache_snapshot_path),
        "cache_status": cache_status.get("status") if cache_status else "",
        "train_path": str(scope.train_path),
        "test_new_prices_path": str(scope.test_new_prices_path),
        "test_new_products_path": str(scope.test_new_products_path),
    }


def _write_sweep_summary(
    *,
    output_root: Path,
    sweep_id: str,
    selected_detector_families: Sequence[str],
    scope_status: pd.DataFrame,
    best_configurations: pd.DataFrame,
    skipped_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> None:
    complete_scope_count = int((scope_status["status"] == "complete").sum()) if not scope_status.empty else 0
    error_scope_count = int((scope_status["status"] == "error").sum()) if not scope_status.empty else 0
    summary = {
        "schema_version": SCHEMA_VERSION,
        "sweep_id": sweep_id,
        "experiment_family": EXPERIMENT_FAMILY,
        "detector_families": list(selected_detector_families),
        "generated_at": _utc_now_iso(),
        "scope_count": int(len(scope_status)),
        "complete_scope_count": complete_scope_count,
        "error_scope_count": error_scope_count,
        "skipped_scope_count": int(len(skipped_rows)),
        "best_configuration_count": int(len(best_configurations)),
        "config": _to_serializable(config),
    }
    _json_dump(output_root / "summary.json", summary)

    lines = [
        "# Statistical Sweep Summary",
        "",
        f"- Sweep ID: `{sweep_id}`",
        f"- Detector families: {', '.join(selected_detector_families)}",
        f"- Complete scopes: {complete_scope_count}",
        f"- Error scopes: {error_scope_count}",
        f"- Skipped scopes: {len(skipped_rows)}",
        f"- Best configurations: {len(best_configurations)}",
        "",
    ]
    if not scope_status.empty:
        lines.extend(
            [
                "| Scope | Status | Succeeded | Failed |",
                "| --- | --- | ---: | ---: |",
            ]
        )
        for _, row in scope_status.sort_values(["mh_level", "granularity", "scope_id"]).iterrows():
            lines.append(
                "| "
                f"{row['mh_level']}/{row['granularity']}/{row['scope_id']} | "
                f"{row['status']} | {int(row['detectors_succeeded'])} | {int(row['detectors_failed'])} |"
            )
        lines.append("")

    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_process_pool(
    *,
    items: Sequence[Any],
    worker,
    max_workers: int,
) -> list[Any]:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune statistical detectors across all discovered data-subsets scopes."
    )
    parser.add_argument(
        "--data-subsets-root",
        default=str(DEFAULT_DATA_SUBSETS_ROOT),
        help="Root containing mhX data-subset directories.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Sweep output root. Defaults to results/tuning/statistical/<sweep_id>.",
    )
    parser.add_argument(
        "--cache-root",
        default=str(DEFAULT_CACHE_ROOT),
        help="Reusable cache snapshot root.",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=5,
        help="Number of deterministic attempts to use. Must be between 1 and 5.",
    )
    parser.add_argument(
        "--detectors",
        nargs="*",
        default=list(ALL_DETECTOR_FAMILIES),
        help=f"Detector families to tune. Defaults to all: {', '.join(ALL_DETECTOR_FAMILIES)}",
    )
    parser.add_argument(
        "--mh-values",
        nargs="*",
        default=list(DEFAULT_SAMPLED_MH_VALUES),
        help=(
            "mh filters to evaluate. Defaults to the sampled set "
            f"{', '.join(DEFAULT_SAMPLED_MH_VALUES)}; override explicitly to tune other mh values."
        ),
    )
    parser.add_argument(
        "--granularities",
        nargs="*",
        default=None,
        help=f"Optional granularity filters: {', '.join(ALL_GRANULARITIES)}",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Top-level process parallelism for snapshot prep and scope-family workers.",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse an existing output root.")
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Force cache snapshot rebuilds even when metadata matches.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default: INFO).",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.attempts < 1 or args.attempts > len(FIXED_ATTEMPT_SEEDS):
        raise ValueError(
            f"--attempts must be between 1 and {len(FIXED_ATTEMPT_SEEDS)}, got {args.attempts}"
        )

    unknown_detectors = sorted(set(args.detectors).difference(ALL_DETECTOR_FAMILIES))
    if unknown_detectors:
        raise ValueError(
            f"Unsupported detectors {unknown_detectors!r}; expected some of {sorted(ALL_DETECTOR_FAMILIES)!r}"
        )

    if args.max_workers < 1:
        raise ValueError("--max-workers must be at least 1")

    if args.resume and not args.output_root:
        raise ValueError("--resume requires --output-root so an existing sweep directory can be reused")


def run_sweep(args: argparse.Namespace) -> int:
    """Execute the statistical tuning sweep."""
    _validate_args(args)
    data_subsets_root = _resolve_root(args.data_subsets_root)
    cache_root = _resolve_root(args.cache_root)
    selected_detector_families = list(args.detectors)
    attempt_seeds = tuple(FIXED_ATTEMPT_SEEDS[: args.attempts])

    if args.output_root:
        output_root = _resolve_root(args.output_root)
        sweep_id = output_root.name
    else:
        sweep_id = create_run_id("statistical")
        output_root = (DEFAULT_RESULTS_ROOT / sweep_id).resolve()

    if output_root.exists() and not args.resume:
        raise FileExistsError(
            f"Output root already exists: {output_root}. Re-run with --resume to reuse it."
        )
    output_root.mkdir(parents=True, exist_ok=True)

    scopes, skipped_rows = discover_scopes(
        data_subsets_root=data_subsets_root,
        cache_root=cache_root,
        mh_values=args.mh_values,
        granularities=args.granularities,
    )
    if not scopes and not skipped_rows:
        raise FileNotFoundError(f"No scopes discovered under {data_subsets_root}")

    snapshot_payloads = [
        {
            "mh_level": scope.mh_level,
            "granularity": scope.granularity,
            "scope_id": scope.scope_id,
            "dataset_name": scope.dataset_name,
            "train_path": str(scope.train_path),
            "test_new_prices_path": str(scope.test_new_prices_path),
            "test_new_products_path": str(scope.test_new_products_path),
            "cache_snapshot_path": str(scope.cache_snapshot_path),
            "rebuild_cache": bool(args.rebuild_cache),
        }
        for scope in scopes
    ]
    snapshot_results = _run_process_pool(
        items=snapshot_payloads,
        worker=_prepare_snapshot_worker,
        max_workers=args.max_workers,
    )
    cache_status_by_scope = {
        (result["mh_level"], result["granularity"], result["scope_id"]): result for result in snapshot_results
    }

    family_results_by_scope: dict[tuple[str, str, str], list[DetectorFamilyTaskResult]] = {
        (scope.mh_level, scope.granularity, scope.scope_id): [] for scope in scopes
    }
    scheduled_tasks: list[DetectorFamilyTask] = []

    for scope in scopes:
        scope_output_dir = _scope_output_dir(output_root, scope)
        for detector_family in selected_detector_families:
            family_output_dir = scope_output_dir / detector_family
            if args.resume and _family_complete(family_output_dir):
                family_results_by_scope[(scope.mh_level, scope.granularity, scope.scope_id)].append(
                    _load_existing_family_result(family_output_dir)
                )
                continue
            scheduled_tasks.append(
                DetectorFamilyTask(
                    scope=scope,
                    detector_family=detector_family,
                    family_output_dir=family_output_dir,
                    sweep_id=sweep_id,
                    attempt_seeds=attempt_seeds,
                )
            )

    worker_results = _run_process_pool(
        items=scheduled_tasks,
        worker=run_detector_family_task,
        max_workers=args.max_workers,
    )
    for result in worker_results:
        family_results_by_scope[(result.mh_level, result.granularity, result.scope_id)].append(result)

    best_configuration_rows: list[dict[str, Any]] = []
    scope_status_rows = list(skipped_rows)

    for scope in scopes:
        key = (scope.mh_level, scope.granularity, scope.scope_id)
        scope_output_dir = _scope_output_dir(output_root, scope)
        scope_output_dir.mkdir(parents=True, exist_ok=True)
        family_results = sorted(
            family_results_by_scope[key],
            key=lambda item: selected_detector_families.index(item.detector_family),
        )

        _write_scope_configuration(
            scope=scope,
            scope_output_dir=scope_output_dir,
            sweep_id=sweep_id,
            attempt_seeds=attempt_seeds,
            selected_detector_families=selected_detector_families,
            family_results=family_results,
        )

        for result in family_results:
            if result.best_candidate is None:
                continue
            best_row = dict(result.best_candidate)
            best_row.update(
                {
                    "mh_level": scope.mh_level,
                    "granularity": scope.granularity,
                    "scope_id": scope.scope_id,
                    "dataset_name": scope.dataset_name,
                    "family_output_dir": result.family_output_dir,
                    "cache_snapshot_path": str(scope.cache_snapshot_path),
                }
            )
            best_configuration_rows.append(best_row)

        scope_status_rows.append(
            _scope_status_row(
                scope=scope,
                scope_output_dir=scope_output_dir,
                selected_detector_families=selected_detector_families,
                family_results=family_results,
                cache_status=cache_status_by_scope.get(key),
            )
        )

    best_configurations = pd.DataFrame(best_configuration_rows)
    scope_status = pd.DataFrame(scope_status_rows)
    if not best_configurations.empty:
        best_configurations.to_csv(output_root / "best_configurations.csv", index=False)
    else:
        pd.DataFrame(columns=["mh_level", "granularity", "scope_id", "detector_family"]).to_csv(
            output_root / "best_configurations.csv",
            index=False,
        )
    if not scope_status.empty:
        scope_status.to_csv(output_root / "scope_status.csv", index=False)
    else:
        pd.DataFrame(columns=["mh_level", "granularity", "scope_id", "status"]).to_csv(
            output_root / "scope_status.csv",
            index=False,
        )

    _write_sweep_summary(
        output_root=output_root,
        sweep_id=sweep_id,
        selected_detector_families=selected_detector_families,
        scope_status=scope_status[scope_status["status"] != "skipped"].copy()
        if not scope_status.empty and "status" in scope_status.columns
        else pd.DataFrame(),
        best_configurations=best_configurations,
        skipped_rows=skipped_rows,
        config={
            "data_subsets_root": str(data_subsets_root),
            "cache_root": str(cache_root),
            "output_root": str(output_root),
            "attempt_seeds": list(attempt_seeds),
            "detectors": selected_detector_families,
            "mh_values": args.mh_values,
            "granularities": args.granularities,
            "max_workers": args.max_workers,
            "resume": bool(args.resume),
            "rebuild_cache": bool(args.rebuild_cache),
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
