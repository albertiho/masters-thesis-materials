#!/usr/bin/env python3
"""Run a fixed COUNTRY_1 layered IF/Z-score/sanity evaluation.

This script is intentionally closed-system and reproducible:

- dataset scope is fixed to COUNTRY_1 at country-level granularity
- train/evaluation parquet files are hardcoded
- the Isolation Forest model name is hardcoded
- the anomaly injection set is explicit and local to this file

Comment out items in ``ANOMALY_TYPES`` to narrow the injected anomaly mix.
"""

from __future__ import annotations

import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add project root to path (4 levels up from research/training/scripts/)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = Path(os.path.dirname(os.path.dirname(os.path.dirname(_script_dir))))
sys.path.insert(0, str(_project_root))
sys.path.insert(0, _script_dir)

from train_isolation_forest import extract_features_vectorized, train_from_matrix
from src.anomaly.combined import CombinedDetector, CombinedDetectorConfig, DetectionContext, DetectorLayer
from src.anomaly.persistence import ModelPersistence
from src.anomaly.statistical import (
    AnomalyResult,
    AnomalySeverity,
    AnomalyType,
    SanityCheckDetector,
    ZScoreDetector,
)
from src.research.artifacts import (
    comparison_result_to_tables,
    initialize_evaluation_tracking_columns,
    json_dumps,
    resolve_git_commit,
)
from src.research.evaluation.detector_evaluator import DetectorEvaluator
from src.research.evaluation.synthetic import SyntheticAnomalyType, inject_anomalies_to_dataframe
from src.research.evaluation.test_orchestrator import ComparisonResult, TestOrchestrator

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


COUNTRY_NAME = "COUNTRY_1"
DATASET_LEVEL = "mh5"
DATASET_SNAPSHOT = "2026-02-08"
TRAIN_SPLIT = "train"
EVALUATION_SPLITS = ("test_new_prices", "test_new_products")
PARALLEL_SCOPE_WORKERS = 6

MIN_HISTORY = 5
INJECTION_RATE = 0.1
INJECTION_SEED = 42
SPIKE_RANGE = (2.0, 5.0)
DROP_RANGE = (0.1, 0.5)
INJECTION_STRATEGY = "synthetic_dataframe_injection"

EXPERIMENT_FAMILY = "detector_combinations"
RUN_ID = "country1_all_scopes_if_zscore_layered_combinations"

# Comment out anomaly types here to narrow the injection mix for follow-up runs.
ANOMALY_TYPES = [
    SyntheticAnomalyType.PRICE_SPIKE,
    SyntheticAnomalyType.PRICE_DROP,
    SyntheticAnomalyType.PRICE_NOISE,
    SyntheticAnomalyType.LIST_PRICE_VIOLATION,
    SyntheticAnomalyType.ZERO_PRICE,
    SyntheticAnomalyType.NEGATIVE_PRICE,
    SyntheticAnomalyType.EXTREME_OUTLIER,
    SyntheticAnomalyType.DECIMAL_SHIFT,
    SyntheticAnomalyType.CURRENCY_SWAP,
]


ZSCORE_ONLY_NAME = "Z-score"
IF_ONLY_NAME = "IF"
SANITY_ONLY_NAME = "Sanity"
SANITY_IF_NAME = "Sanity -> IF"
SANITY_ZSCORE_NAME = "Sanity -> Z-score"
IF_ZSCORE_5050_NAME = "IF -> Z-score (50/50)"
SANITY_ZSCORE_IF_5050_NAME = "Sanity -> Z-score -> IF (50/50)"
SANITY_ZSCORE_GT5_IF_NAME = "Sanity -> Z-score (>=5) -> IF"
SANITY_ZSCORE_GT10_IF_NAME = "Sanity -> Z-score (>=10) -> IF"

DEFAULT_COMBINATIONS = [
    IF_ONLY_NAME,
    ZSCORE_ONLY_NAME,
    SANITY_ONLY_NAME,
    SANITY_IF_NAME,
    SANITY_ZSCORE_NAME,
    IF_ZSCORE_5050_NAME,
    SANITY_ZSCORE_IF_5050_NAME,
    SANITY_ZSCORE_GT5_IF_NAME,
    SANITY_ZSCORE_GT10_IF_NAME,
]
IF_REQUIRED_COMBINATIONS = {
    IF_ONLY_NAME,
    SANITY_IF_NAME,
    IF_ZSCORE_5050_NAME,
    SANITY_ZSCORE_IF_5050_NAME,
    SANITY_ZSCORE_GT5_IF_NAME,
    SANITY_ZSCORE_GT10_IF_NAME,
}


@dataclass(frozen=True)
class ScopeSpec:
    scope_id: str
    scope_kind: str
    dataset_name: str
    dataset_granularity: str
    country: str
    data_root: Path
    iforest_model_name: str
    scope_market: str = ""


def _severity_from_score(score: float) -> AnomalySeverity | None:
    """Map a normalized 0-1 ensemble score to the shared severity bands."""
    if score >= 0.9:
        return AnomalySeverity.CRITICAL
    if score >= 0.8:
        return AnomalySeverity.HIGH
    if score >= 0.7:
        return AnomalySeverity.MEDIUM
    if score >= 0.5:
        return AnomalySeverity.LOW
    return None


class WeightedScoreCombinedDetector(CombinedDetector):
    """Combined detector that averages normalized detector scores by weight."""

    def __init__(
        self,
        config: CombinedDetectorConfig,
        layers: list[DetectorLayer],
        *,
        decision_threshold: float = 0.5,
    ) -> None:
        super().__init__(config=config, layers=layers)
        self.decision_threshold = decision_threshold

    def _aggregate_results(self, context: DetectionContext) -> AnomalyResult:
        all_results = context.get_all_results()

        if not all_results:
            return AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                anomaly_types=[],
                severity=None,
                details={"no_results": True, "route_path": context.route_path},
                detector=self.name,
                competitor_product_id=context.numeric_features.competitor_product_id,
                competitor=context.numeric_features.competitor,
            )

        scored_results: list[tuple[AnomalyResult, float]] = []
        skipped_results: list[dict[str, Any]] = []

        for result in all_results:
            if result.details.get("insufficient_history"):
                skipped_results.append(
                    {
                        "detector": result.detector,
                        "reason": "insufficient_history",
                    }
                )
                continue

            weight = float(self.config.detector_weights.get(result.detector, 1.0))
            if weight <= 0:
                skipped_results.append(
                    {
                        "detector": result.detector,
                        "reason": "non_positive_weight",
                        "weight": weight,
                    }
                )
                continue

            scored_results.append((result, weight))

        details: dict[str, Any] = {
            "route_path": context.route_path,
            "ensemble_mode": "weighted_score_average",
            "decision_threshold": self.decision_threshold,
            "total_detectors": len(context.votes),
            "flagged_detectors": context.get_flagged_detectors(),
            "confidence": context.confidence,
            "short_circuited": context.short_circuited,
            "skipped_results": skipped_results,
        }

        if context.short_circuited:
            details["short_circuit_reason"] = context.short_circuit_reason

        for layer_name, layer_results in context.layer_results.items():
            details[layer_name] = [result.details for result in layer_results]

        if not scored_results:
            details["no_scored_results"] = True
            return AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                anomaly_types=[],
                severity=None,
                details=details,
                detector=self.name,
                competitor_product_id=context.numeric_features.competitor_product_id,
                competitor=context.numeric_features.competitor,
            )

        total_weight = sum(weight for _, weight in scored_results)
        weighted_score = sum(result.anomaly_score * weight for result, weight in scored_results) / total_weight
        is_anomaly = weighted_score >= self.decision_threshold

        anomaly_types: set[AnomalyType] = set()
        if is_anomaly:
            for result, _ in scored_results:
                anomaly_types.update(result.anomaly_types)

        details["weighted_score"] = weighted_score
        details["active_detectors"] = [result.detector for result, _ in scored_results]
        details["active_detector_weights"] = {
            result.detector: weight for result, weight in scored_results
        }
        details["detector_scores"] = {
            result.detector: result.anomaly_score for result, _ in scored_results
        }

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=weighted_score,
            anomaly_types=list(anomaly_types),
            severity=_severity_from_score(weighted_score) if is_anomaly else None,
            details=details,
            detector=self.name,
            competitor_product_id=context.numeric_features.competitor_product_id,
            competitor=context.numeric_features.competitor,
        )


def _base_config(name: str, *, detector_weights: dict[str, float] | None = None) -> CombinedDetectorConfig:
    return CombinedDetectorConfig(
        name=name,
        min_history_cold=1,
        min_history_warm=MIN_HISTORY,
        detector_weights=detector_weights or {},
    )


def _sanity_layer() -> DetectorLayer:
    return DetectorLayer(
        name="sanity",
        detectors=[SanityCheckDetector()],
        is_gate=True,
        required_history=0,
        layer_type="sanity",
    )


def _zscore_layer(*, required_history: int = 0) -> DetectorLayer:
    return DetectorLayer(
        name="zscore",
        detectors=[ZScoreDetector()],
        required_history=required_history,
        layer_type="statistical",
    )


def _if_layer(iforest_detector: Any) -> DetectorLayer:
    return DetectorLayer(
        name="iforest",
        detectors=[iforest_detector],
        required_history=0,
        layer_type="ml",
    )


def create_zscore_only_detector(name: str = ZSCORE_ONLY_NAME) -> CombinedDetector:
    """Create a single-layer Z-score detector."""
    return CombinedDetector(
        config=_base_config(name),
        layers=[_zscore_layer()],
    )


def create_if_only_detector(
    iforest_detector: Any,
    name: str = IF_ONLY_NAME,
) -> CombinedDetector:
    """Create a single-layer Isolation Forest detector."""
    return CombinedDetector(
        config=_base_config(name),
        layers=[_if_layer(iforest_detector)],
    )


def create_sanity_only_detector(name: str = SANITY_ONLY_NAME) -> CombinedDetector:
    """Create a single-layer sanity gate detector."""
    return CombinedDetector(
        config=_base_config(name),
        layers=[_sanity_layer()],
    )


def create_sanity_if_detector(
    iforest_detector: Any,
    name: str = SANITY_IF_NAME,
) -> CombinedDetector:
    """Create the requested Sanity -> IF cascade."""
    return CombinedDetector(
        config=_base_config(name),
        layers=[_sanity_layer(), _if_layer(iforest_detector)],
    )


def create_sanity_zscore_detector(
    name: str = SANITY_ZSCORE_NAME,
) -> CombinedDetector:
    """Create the requested Sanity -> Z-score cascade."""
    return CombinedDetector(
        config=_base_config(name),
        layers=[_sanity_layer(), _zscore_layer()],
    )


def create_if_zscore_5050_detector(
    iforest_detector: Any,
    name: str = IF_ZSCORE_5050_NAME,
) -> WeightedScoreCombinedDetector:
    """Create a 50/50 normalized-score ensemble of IF and Z-score."""
    return WeightedScoreCombinedDetector(
        config=_base_config(
            name,
            detector_weights={"isolation_forest": 0.5, "zscore": 0.5},
        ),
        layers=[
            DetectorLayer(
                name="if_zscore_5050",
                detectors=[iforest_detector, ZScoreDetector()],
                required_history=0,
                layer_type="ensemble",
            )
        ],
        decision_threshold=0.5,
    )


def create_sanity_zscore_if_5050_detector(
    iforest_detector: Any,
    name: str = SANITY_ZSCORE_IF_5050_NAME,
) -> WeightedScoreCombinedDetector:
    """Create Sanity -> Z-score -> IF with 50/50 IF/Z-score aggregation."""
    return WeightedScoreCombinedDetector(
        config=_base_config(
            name,
            detector_weights={
                "sanity": 0.0,
                "isolation_forest": 0.5,
                "zscore": 0.5,
            },
        ),
        layers=[
            _sanity_layer(),
            _zscore_layer(),
            _if_layer(iforest_detector),
        ],
        decision_threshold=0.5,
    )


def create_sanity_zscore_if_detector(
    iforest_detector: Any,
    *,
    zscore_required_history: int,
    name: str,
) -> CombinedDetector:
    """Create Sanity -> Z-score(history gate) -> IF."""
    return CombinedDetector(
        config=_base_config(name),
        layers=[
            _sanity_layer(),
            _zscore_layer(required_history=zscore_required_history),
            _if_layer(iforest_detector),
        ],
    )


def _requires_iforest(combinations: list[str]) -> bool:
    return any(name in IF_REQUIRED_COMBINATIONS for name in combinations)


def create_evaluators(
    persistence: ModelPersistence | None,
    model_name: str,
    *,
    iforest_model: str | None = None,
    iforest_detector: Any | None = None,
    combinations: list[str] | None = None,
) -> list[DetectorEvaluator]:
    """Create the requested detector-combination evaluators."""
    selected = combinations or list(DEFAULT_COMBINATIONS)

    unknown = [name for name in selected if name not in DEFAULT_COMBINATIONS]
    if unknown:
        raise ValueError(f"Unknown combinations requested: {unknown}")

    if _requires_iforest(selected):
        if iforest_detector is None:
            if persistence is None:
                raise ValueError("Isolation Forest combinations require model persistence to be available.")
            iforest_detector = persistence.load_isolation_forest(iforest_model or model_name)

    evaluators: list[DetectorEvaluator] = []
    for combination in selected:
        if combination == ZSCORE_ONLY_NAME:
            evaluators.append(DetectorEvaluator(create_zscore_only_detector(), combination))
        elif combination == IF_ONLY_NAME:
            evaluators.append(DetectorEvaluator(create_if_only_detector(iforest_detector), combination))
        elif combination == SANITY_ONLY_NAME:
            evaluators.append(DetectorEvaluator(create_sanity_only_detector(), combination))
        elif combination == SANITY_IF_NAME:
            evaluators.append(DetectorEvaluator(create_sanity_if_detector(iforest_detector), combination))
        elif combination == SANITY_ZSCORE_NAME:
            evaluators.append(DetectorEvaluator(create_sanity_zscore_detector(), combination))
        elif combination == IF_ZSCORE_5050_NAME:
            evaluators.append(
                DetectorEvaluator(create_if_zscore_5050_detector(iforest_detector), combination)
            )
        elif combination == SANITY_ZSCORE_IF_5050_NAME:
            evaluators.append(
                DetectorEvaluator(create_sanity_zscore_if_5050_detector(iforest_detector), combination)
            )
        elif combination == SANITY_ZSCORE_GT5_IF_NAME:
            evaluators.append(
                DetectorEvaluator(
                    create_sanity_zscore_if_detector(
                        iforest_detector,
                        zscore_required_history=5,
                        name=SANITY_ZSCORE_GT5_IF_NAME,
                    ),
                    combination,
                )
            )
        elif combination == SANITY_ZSCORE_GT10_IF_NAME:
            evaluators.append(
                DetectorEvaluator(
                    create_sanity_zscore_if_detector(
                        iforest_detector,
                        zscore_required_history=10,
                        name=SANITY_ZSCORE_GT10_IF_NAME,
                    ),
                    combination,
                )
            )

    return evaluators


def _resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (_project_root / path).resolve()


def build_scope_specs() -> list[ScopeSpec]:
    """Return the fixed COUNTRY_1 country + competitor scope list."""
    scopes: list[ScopeSpec] = [
        ScopeSpec(
            scope_id=COUNTRY_NAME,
            scope_kind="country",
            dataset_name=COUNTRY_NAME,
            dataset_granularity="country",
            country=COUNTRY_NAME,
            data_root=Path("data-subsets") / DATASET_LEVEL / "by_country" / COUNTRY_NAME,
            iforest_model_name=f"{COUNTRY_NAME}_{DATASET_LEVEL}",
        )
    ]

    scopes.append(
        ScopeSpec(
            scope_id="COMPETITOR_1_COUNTRY_1",
            scope_kind="competitor",
            dataset_name="COMPETITOR_1_COUNTRY_1",
            dataset_granularity="competitor",
            country=COUNTRY_NAME,
            scope_market="B2B",
            data_root=(
                Path("data-subsets")
                / DATASET_LEVEL
                / "by_competitor"
                / COUNTRY_NAME
                / "B2B"
            ),
            iforest_model_name="COMPETITOR_1_COUNTRY_1_mh5",
        )
    )
    scopes.append(
        ScopeSpec(
            scope_id="COMPETITOR_2_COUNTRY_1",
            scope_kind="competitor",
            dataset_name="COMPETITOR_2_COUNTRY_1",
            dataset_granularity="competitor",
            country=COUNTRY_NAME,
            scope_market="B2C",
            data_root=(
                Path("data-subsets")
                / DATASET_LEVEL
                / "by_competitor"
                / COUNTRY_NAME
                / "B2C"
            ),
            iforest_model_name="COMPETITOR_2_COUNTRY_1_mh5",
        )
    )
    scopes.append(
        ScopeSpec(
            scope_id="COMPETITOR_3_COUNTRY_1",
            scope_kind="competitor",
            dataset_name="COMPETITOR_3_COUNTRY_1",
            dataset_granularity="competitor",
            country=COUNTRY_NAME,
            scope_market="B2C",
            data_root=(
                Path("data-subsets")
                / DATASET_LEVEL
                / "by_competitor"
                / COUNTRY_NAME
                / "B2C"
            ),
            iforest_model_name="COMPETITOR_3_COUNTRY_1_mh5",
        )
    )
    return scopes


def build_dataset_paths(scope: ScopeSpec) -> dict[str, Path]:
    """Build the parquet paths for one fixed scope."""
    resolved_root = _resolve_repo_path(scope.data_root)
    return {
        TRAIN_SPLIT: resolved_root / f"{scope.dataset_name}_{DATASET_SNAPSHOT}_{TRAIN_SPLIT}.parquet",
        "test_new_prices": resolved_root / f"{scope.dataset_name}_{DATASET_SNAPSHOT}_test_new_prices.parquet",
        "test_new_products": resolved_root / f"{scope.dataset_name}_{DATASET_SNAPSHOT}_test_new_products.parquet",
    }


def build_injection_kwargs(split_index: int) -> dict[str, Any]:
    """Build the explicit injection configuration for one evaluation split."""
    return {
        "injection_rate": INJECTION_RATE,
        "seed": INJECTION_SEED + split_index,
        "spike_range": SPIKE_RANGE,
        "drop_range": DROP_RANGE,
        "anomaly_types": list(ANOMALY_TYPES),
    }


def _load_frame(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _normalize_split_name(split_name: str) -> str:
    mapping = {
        "test_new_prices": "new_prices",
        "test_new_products": "new_products",
    }
    return mapping.get(split_name, split_name)


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
        injection_strategy=INJECTION_STRATEGY,
    )
    annotated["ground_truth_label"] = np.asarray(labels).astype(bool)
    annotated["is_injected"] = annotated["ground_truth_label"]
    annotated["anomaly_type"] = pd.Series([None] * len(annotated), dtype="object")
    annotated["injection_phase"] = pd.Series([pd.NA] * len(annotated), dtype="Int64")
    annotated["injection_params_json"] = "{}"

    if "__original_price__" in annotated.columns:
        annotated["original_price"] = pd.to_numeric(
            annotated["__original_price__"],
            errors="coerce",
        )

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
        annotated.at[row_index, "injection_params_json"] = json_dumps(params)
        if "original_price" in detail:
            annotated.at[row_index, "original_price"] = detail["original_price"]

    return annotated


def inject_split_frame(
    frame: pd.DataFrame,
    *,
    split_index: int,
) -> tuple[pd.DataFrame, np.ndarray, list[dict[str, Any]]]:
    """Inject anomalies into one split using the explicit script-local config."""
    return inject_anomalies_to_dataframe(
        frame,
        **build_injection_kwargs(split_index),
    )


def load_or_train_iforest(
    train_df: pd.DataFrame,
    *,
    persistence: ModelPersistence,
    model_name: str,
) -> Any:
    """Load the fixed IF model, or train and persist it from the fixed train split."""
    try:
        detector = persistence.load_isolation_forest(model_name)
        print(f"Loaded Isolation Forest model: {model_name}")
        return detector
    except FileNotFoundError:
        print(f"Isolation Forest model missing for {model_name}; training from fixed train split.")

    features = extract_features_vectorized(train_df)
    detector, _ = train_from_matrix(features)

    try:
        model_uri = persistence.save_isolation_forest(detector, model_name, len(train_df))
        print(f"Saved Isolation Forest model: {model_uri}")
    except Exception as exc:
        logger.warning("Failed to persist trained Isolation Forest model %s: %s", model_name, exc)

    return detector


@dataclass
class ScopeRunResult:
    scope: ScopeSpec
    scope_root: Path
    detector_names: list[str]
    resumed: bool = False


def _scope_extra_columns(scope: ScopeSpec) -> dict[str, object]:
    return {
        "scope_id": scope.scope_id,
        "scope_kind": scope.scope_kind,
        "scope_market": scope.scope_market,
    }


def _scope_manifest_rows(scopes: list[ScopeSpec]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scope in scopes:
        dataset_paths = build_dataset_paths(scope)
        rows.append(
            {
                "scope_id": scope.scope_id,
                "scope_kind": scope.scope_kind,
                "scope_market": scope.scope_market,
                "dataset_name": scope.dataset_name,
                "dataset_granularity": scope.dataset_granularity,
                "country": scope.country,
                "iforest_model_name": scope.iforest_model_name,
                "train_path": str(dataset_paths[TRAIN_SPLIT]),
                "test_new_prices_path": str(dataset_paths["test_new_prices"]),
                "test_new_products_path": str(dataset_paths["test_new_products"]),
            }
        )
    return pd.DataFrame(rows)


def _scope_root(run_root: Path, scope: ScopeSpec) -> Path:
    return run_root / "scopes" / scope.scope_id


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


def _load_scope_split_artifact(
    scope_root: Path,
    split_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    injected_path, predictions_path = _scope_split_paths(scope_root, split_name)
    return pd.read_parquet(injected_path), pd.read_parquet(predictions_path)


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


def _legacy_split_paths(run_root: Path, split_name: str) -> tuple[Path, Path]:
    split_root = run_root / "splits" / _normalize_split_name(split_name)
    return split_root / "injected_rows.parquet", split_root / "predictions.parquet"


def _read_legacy_scope_rows(path: Path, scope_id: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, filters=[("candidate_id", "==", scope_id)])
    except Exception:
        frame = pd.read_parquet(path)
        return frame[frame["candidate_id"].fillna("").astype(str).eq(scope_id)].copy()


def _hydrate_scope_checkpoints_from_legacy_split(
    run_root: Path,
    scopes: list[ScopeSpec],
    split_name: str,
) -> None:
    injected_path, predictions_path = _legacy_split_paths(run_root, split_name)
    if not injected_path.exists() or not predictions_path.exists():
        return

    if all(_scope_split_complete(_scope_root(run_root, scope), split_name) for scope in scopes):
        return

    for scope in scopes:
        scope_root = _scope_root(run_root, scope)
        if _scope_split_complete(scope_root, split_name):
            continue

        print(
            "Hydrating per-scope checkpoint from legacy "
            f"{_normalize_split_name(split_name)} artifact for {scope.scope_id}..."
        )
        scope_injected = _read_legacy_scope_rows(injected_path, scope.scope_id)
        scope_predictions = _read_legacy_scope_rows(predictions_path, scope.scope_id)
        if scope_injected.empty or scope_predictions.empty:
            continue
        _write_scope_split_checkpoint(scope_root, split_name, scope_injected, scope_predictions)


def _hydrate_scope_checkpoints_from_legacy_root(
    run_root: Path,
    scopes: list[ScopeSpec],
) -> None:
    for split_name in EVALUATION_SPLITS:
        _hydrate_scope_checkpoints_from_legacy_split(run_root, scopes, split_name)


def _print_scope_summary(scope_result: ScopeRunResult) -> None:
    state = "resumed" if scope_result.resumed else "completed"
    print(f"Scope {state}: {scope_result.scope.scope_id}")


def _build_scope_run_metadata(
    scope: ScopeSpec,
    dataset_paths: dict[str, Path],
    detector_names: list[str],
) -> dict[str, object]:
    return {
        "schema_version": "phase2.v1",
        "experiment_family": EXPERIMENT_FAMILY,
        "run_id": RUN_ID,
        "candidate_id": scope.scope_id,
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
        "detector_identifiers": detector_names,
        "config_values": {
            "country_name": COUNTRY_NAME,
            "dataset_level": DATASET_LEVEL,
            "dataset_snapshot": DATASET_SNAPSHOT,
            "iforest_model_name": scope.iforest_model_name,
            "evaluation_splits": list(EVALUATION_SPLITS),
            "min_history": MIN_HISTORY,
            "combinations": list(DEFAULT_COMBINATIONS),
            "parallel_scope_workers": PARALLEL_SCOPE_WORKERS,
            "scope_kind": scope.scope_kind,
            "scope_market": scope.scope_market,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": resolve_git_commit(_project_root),
    }


def _write_json_file(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json.loads(json_dumps(payload)), indent=2) + "\n", encoding="utf-8")


def _write_scope_metadata(
    scope_root: Path,
    scope: ScopeSpec,
    dataset_paths: dict[str, Path],
    detector_names: list[str],
) -> None:
    _write_json_file(
        scope_root / "run_metadata.json",
        _build_scope_run_metadata(scope, dataset_paths, detector_names),
    )


def _build_scope_status_rows(scope_results: list[ScopeRunResult]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for result in sorted(scope_results, key=lambda item: item.scope.scope_id):
        scope_root = result.scope_root
        rows.append(
            {
                "scope_id": result.scope.scope_id,
                "scope_kind": result.scope.scope_kind,
                "scope_market": result.scope.scope_market,
                "dataset_name": result.scope.dataset_name,
                "dataset_granularity": result.scope.dataset_granularity,
                "iforest_model_name": result.scope.iforest_model_name,
                "resumed": result.resumed,
                "new_prices_complete": _scope_split_complete(scope_root, "test_new_prices"),
                "new_products_complete": _scope_split_complete(scope_root, "test_new_products"),
            }
        )
    return pd.DataFrame(rows)


def evaluate_scope(scope: ScopeSpec, run_root: Path) -> ScopeRunResult:
    dataset_paths = build_dataset_paths(scope)
    _validate_dataset_paths(dataset_paths)
    scope_root = _scope_root(run_root, scope)
    detector_names = list(DEFAULT_COMBINATIONS)

    if _scope_run_complete(scope_root):
        _write_scope_metadata(scope_root, scope, dataset_paths, detector_names)
        return ScopeRunResult(
            scope=scope,
            scope_root=scope_root,
            detector_names=detector_names,
            resumed=True,
        )

    print(f"Scope: {scope.scope_id}")
    print(f"  granularity={scope.dataset_granularity}")
    print(f"  data_root={dataset_paths[TRAIN_SPLIT].parent}")
    print(f"  iforest_model={scope.iforest_model_name}")

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
            iforest_detector = load_or_train_iforest(
                train_df,
                persistence=persistence,
                model_name=scope.iforest_model_name,
            )
            evaluators = create_evaluators(
                persistence,
                scope.iforest_model_name,
                iforest_detector=iforest_detector,
                combinations=detector_names,
            )
            detector_family_map = {
                evaluator.name: evaluator.detector.name for evaluator in evaluators
            }
            orchestrator = TestOrchestrator(evaluators=evaluators, max_workers=1)
            print(f"  train_rows={len(train_df):,}")

        frame = _load_frame(dataset_paths[split_name])
        injection_kwargs = build_injection_kwargs(split_index)
        injected_frame, labels, injection_details = inject_split_frame(
            frame,
            split_index=split_index,
        )
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
            candidate_id=scope.scope_id,
            experiment_family=EXPERIMENT_FAMILY,
            dataset_name=scope.dataset_name,
            dataset_granularity=scope.dataset_granularity,
            dataset_split=split_name,
            detector_family_map=detector_family_map,
            injected_row_extras=_scope_extra_columns(scope),
            prediction_extras=_scope_extra_columns(scope),
        )
        _write_scope_split_checkpoint(scope_root, split_name, injected_rows, predictions)

        del comparison
        del injected_rows
        del predictions
        del annotated_frame
        del injected_frame
        del frame
        gc.collect()

    _write_scope_metadata(scope_root, scope, dataset_paths, detector_names)
    gc.collect()

    return ScopeRunResult(
        scope=scope,
        scope_root=scope_root,
        detector_names=detector_names,
        resumed=False,
    )


def _build_run_metadata(
    scopes: list[ScopeSpec],
    detector_names: list[str],
) -> dict[str, object]:
    return {
        "schema_version": "phase2.v1",
        "experiment_family": EXPERIMENT_FAMILY,
        "run_id": RUN_ID,
        "candidate_ids": [scope.scope_id for scope in scopes],
        "dataset_names": [scope.dataset_name for scope in scopes],
        "dataset_granularity": "mixed",
        "dataset_granularities": sorted({scope.dataset_granularity for scope in scopes}),
        "dataset_splits": [_normalize_split_name(split) for split in EVALUATION_SPLITS],
        "scope_count": len(scopes),
        "scope_ids": [scope.scope_id for scope in scopes],
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
        "detector_identifiers": detector_names,
        "config_values": {
            "country_name": COUNTRY_NAME,
            "dataset_level": DATASET_LEVEL,
            "dataset_snapshot": DATASET_SNAPSHOT,
            "iforest_model_names": [scope.iforest_model_name for scope in scopes],
            "evaluation_splits": list(EVALUATION_SPLITS),
            "min_history": MIN_HISTORY,
            "combinations": list(DEFAULT_COMBINATIONS),
            "parallel_scope_workers": PARALLEL_SCOPE_WORKERS,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": resolve_git_commit(_project_root),
    }


def _build_output_root(base_root: Path | None = None) -> Path:
    if base_root is not None:
        return base_root
    return _project_root / "results" / "detector_combinations" / RUN_ID


def _validate_dataset_paths(dataset_paths: dict[str, Path]) -> None:
    missing = [path for path in dataset_paths.values() if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required dataset files:\n{missing_text}")


def run(*, output_root: Path | None = None) -> Path:
    resolved_output_root = _build_output_root(output_root)
    resolved_output_root.mkdir(parents=True, exist_ok=True)
    scope_specs = build_scope_specs()
    max_workers = min(PARALLEL_SCOPE_WORKERS, len(scope_specs))

    print(f"Country: {COUNTRY_NAME}")
    print(f"Scopes: {[scope.scope_id for scope in scope_specs]}")
    print(f"Parallel scope workers: {max_workers}")
    print(f"Evaluation splits: {', '.join(EVALUATION_SPLITS)}")
    print(f"Anomaly types: {[anomaly_type.value for anomaly_type in ANOMALY_TYPES]}")

    persistence = ModelPersistence()
    print(f"ML models from: {persistence.models_root_description}")
    _hydrate_scope_checkpoints_from_legacy_root(resolved_output_root, scope_specs)

    scope_results: list[ScopeRunResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {
            pool.submit(evaluate_scope, scope, resolved_output_root): scope
            for scope in scope_specs
        }
        for future in as_completed(future_map):
            scope_result = future.result()
            scope_results.append(scope_result)
            _print_scope_summary(scope_result)

    scope_results.sort(key=lambda result: result.scope.scope_id)
    analysis_root = resolved_output_root / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    _scope_manifest_rows(scope_specs).to_csv(analysis_root / "scope_manifest.csv", index=False)
    _build_scope_status_rows(scope_results).to_csv(analysis_root / "scope_status.csv", index=False)
    _write_json_file(
        resolved_output_root / "run_metadata.json",
        _build_run_metadata(
            scope_specs,
            scope_results[0].detector_names if scope_results else list(DEFAULT_COMBINATIONS),
        ),
    )

    print(f"Output root: {resolved_output_root}")
    return resolved_output_root


def main() -> None:
    run()


if __name__ == "__main__":
    main()
