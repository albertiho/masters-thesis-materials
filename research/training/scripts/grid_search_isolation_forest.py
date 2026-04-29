#!/usr/bin/env python3
"""Single-run Isolation Forest hyperparameter grid search for research/documentation.

Runs a full Cartesian grid search over Isolation Forest hyperparameters on one dataset,
compares variants, and writes research-document-ready outputs (full results CSV +
summary with optimal config). Production training with the chosen config
is done later via train_isolation_forest.py.

Uses TestOrchestrator + DetectorEvaluator for proper batch processing with cache
updates between rounds (via BatchRoundProcessor), ensuring products with multiple
test observations have correct temporal features.

Uses hardcoded paths: data/training/ with _train_mh4 suffix.

Usage:
    # Run on specific model (e.g., NO_B2C)
    python scripts/grid_search_isolation_forest.py --model-filter NO_B2C --output-dir results/iforest_grid_search

    # Resume after interrupt
    python scripts/grid_search_isolation_forest.py --model-filter NO_B2C --output-dir results/iforest_grid_search --resume

    # Dry run (show configs without training)
    python scripts/grid_search_isolation_forest.py --dry-run
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path (4 levels up from research/training/scripts/)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_script_dir)))
sys.path.insert(0, _project_root)
sys.path.insert(0, _script_dir)  # For importing sibling scripts

from dotenv import load_dotenv

# Generic utilities (shared across all training scripts)
from train_autoencoder import (
    extract_model_name,
    find_matching_test_file,
    find_parquet_files,
    load_parquet_file,
)
# Isolation Forest-specific features, caching, and training
from train_isolation_forest import (
    extract_features_vectorized,
    get_cache_path,
    load_cached_features,
    save_cached_features,
    train_from_matrix,
)
# Evaluation framework - inject anomalies at DataFrame level
from src.research.evaluation.synthetic import inject_anomalies_to_dataframe
from src.research.evaluation.detector_evaluator import DetectorEvaluator
from src.research.evaluation.test_orchestrator import ComparisonResult, TestOrchestrator
from src.research.artifacts import (
    comparison_result_to_tables,
    empty_injected_rows_table,
    empty_predictions_table,
    resolve_git_commit,
    write_evaluation_run,
    write_tuning_sweep,
)
from compare_detectors import extract_country

from src.anomaly.ml.isolation_forest import IsolationForestConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Grid search hyperparameters (full Cartesian grid)
# Based on sklearn IsolationForest and current fixed values in IsolationForestConfig
N_ESTIMATORS_OPTIONS = [50, 100, 200]  # Currently fixed at 100
MAX_SAMPLES_OPTIONS = ["auto", 128, 256, 512]  # "auto" = min(256, n_samples)
MAX_FEATURES_OPTIONS = [0.5, 0.75, 1.0]  # Currently 1.0 (all features)
ANOMALY_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]  # Score threshold for flagging anomalies

# Fixed parameters
DEFAULT_CONTAMINATION = "auto"  # Clean training data assumption
RANDOM_STATE = 42
INJECTION_RATE = 0.1
# inject_anomalies_to_dataframe defaults: spike_range=(2.0, 5.0), drop_range=(0.1, 0.5)
SPIKE_RANGE = (2.0, 5.0)
DROP_RANGE = (0.1, 0.5)

# Default train/test suffixes: train on mh5, evaluate on both test_new_prices_mh5 and test_new_products_mh5
DEFAULT_FILE_SUFFIX = "_train_mh5"
TEST_NEW_PRICES_SUFFIX = "_test_new_prices_mh5"
TEST_NEW_PRODUCTS_SUFFIX = "_test_new_products_mh5"


@dataclass
class GridRow:
    """One evaluated configuration (one row in results CSV)."""

    run_id: str
    n_estimators: int
    max_samples: str | int
    max_features: float
    anomaly_threshold: float
    contamination: str | float
    precision: float
    recall: float
    f1_score: float
    precision_new_prices: float
    recall_new_prices: float
    f1_new_prices: float
    precision_new_products: float
    recall_new_products: float
    f1_new_products: float
    training_time_sec: float
    dataset_name: str = ""
    n_train: int = 0
    n_eval_prices: int = 0
    n_eval_products: int = 0
    error: str = ""


@dataclass
class CandidateEvaluationResult:
    """Full candidate output including row-level comparison artifacts."""

    row: GridRow
    comparisons: dict[str, ComparisonResult]
    config: dict[str, object]


def _resolve_train_file(model_filter: str | None, granularity: str) -> str:
    """Resolve single train Parquet path.
    
    Uses hardcoded data path and file suffix for consistency.
    
    Args:
        model_filter: Optional model name filter (e.g., 'NO_B2C', 'DK_B2B').
        granularity: 'country_segment' or 'competitor'.
    
    Returns:
        Path to the training Parquet file.
    """
    data_path = "data/training"
    file_suffix = DEFAULT_FILE_SUFFIX  # _train_mh4
    files = find_parquet_files(data_path, granularity, file_suffix)
    
    # Filter by model name if specified
    if model_filter:
        files = [f for f in files if model_filter in extract_model_name(f)]
    
    if not files:
        filter_msg = f", model_filter={model_filter!r}" if model_filter else ""
        raise FileNotFoundError(
            f"No Parquet files in {data_path} (granularity={granularity}, suffix={file_suffix!r}{filter_msg})"
        )
    return files[0]


def _resolve_test_files(train_file: str) -> tuple[str, str]:
    """Resolve test_new_prices_mh5 and test_new_products_mh5 paths.
    
    Args:
        train_file: Path to training file (used to find matching test files).
    
    Returns:
        Tuple of (test_new_prices path, test_new_products path).
    
    Raises:
        FileNotFoundError: If either test file is missing.
    """
    data_path = os.path.dirname(train_file)
    path_prices = find_matching_test_file(train_file, TEST_NEW_PRICES_SUFFIX, data_path)
    path_products = find_matching_test_file(train_file, TEST_NEW_PRODUCTS_SUFFIX, data_path)
    if not path_prices:
        raise FileNotFoundError(
            f"Test file not found for suffix {TEST_NEW_PRICES_SUFFIX!r} (train: {train_file})"
        )
    if not path_products:
        raise FileNotFoundError(
            f"Test file not found for suffix {TEST_NEW_PRODUCTS_SUFFIX!r} (train: {train_file})"
        )
    return path_prices, path_products


def _load_features(
    filepath: str,
    use_cache: bool = True,
) -> tuple[np.ndarray, int]:
    """Load or compute feature matrix; return (X, n_rows). Uses cache first to avoid Parquet read on hit."""
    cache_path = get_cache_path(filepath)
    if use_cache:
        X = load_cached_features(cache_path)
        if X is not None:
            logger.info("Using cached features for %s (%d samples)", os.path.basename(filepath), X.shape[0])
            return X, int(X.shape[0])
    df = load_parquet_file(filepath)
    n_rows = len(df)
    X = extract_features_vectorized(df)
    if use_cache:
        save_cached_features(cache_path, X)
    return X, n_rows


def _load_eval_data(filepath: str) -> tuple[pd.DataFrame, int]:
    """Load DataFrame for evaluation (anomaly injection happens at DataFrame level).
    
    Unlike _load_features, this always loads the raw DataFrame since we need it
    for inject_anomalies_to_dataframe to work at the raw data level.
    """
    df = load_parquet_file(filepath)
    n_rows = len(df)
    logger.info("Loaded eval data: %s (%d samples)", os.path.basename(filepath), n_rows)
    return df, n_rows


def _load_train_data(filepath: str) -> pd.DataFrame:
    """Load training DataFrame for TestOrchestrator cache population.
    
    The training DataFrame is needed to populate the temporal cache before
    running detection on test data. This simulates the "warm cache" scenario
    where the detector has seen historical data for the products.
    """
    df = load_parquet_file(filepath)
    logger.info("Loaded train data: %s (%d samples)", os.path.basename(filepath), len(df))
    return df


def _build_grid_combos() -> list[dict]:
    """Build full Cartesian grid of hyperparameter combinations.
    
    Grid: n_estimators x max_samples x max_features x anomaly_threshold
    Total: 3 x 4 x 3 x 5 = 180 combinations
    """
    combos = []
    for n_est, max_samp, max_feat, thresh in product(
        N_ESTIMATORS_OPTIONS,
        MAX_SAMPLES_OPTIONS,
        MAX_FEATURES_OPTIONS,
        ANOMALY_THRESHOLDS,
    ):
        combos.append({
            "n_estimators": n_est,
            "max_samples": max_samp,
            "max_features": max_feat,
            "anomaly_threshold": thresh,
            "contamination": DEFAULT_CONTAMINATION,
        })
    return combos


def _run_one(
    combo: dict,
    X_train: np.ndarray,
    train_df: pd.DataFrame,
    df_eval_prices: pd.DataFrame,
    df_eval_products: pd.DataFrame,
    run_id: str,
    dataset_name: str,
    n_train: int,
    n_eval_prices: int,
    n_eval_products: int,
    country: str | None = None,
    combo_idx: int = 0,
    total_combos: int = 0,
) -> CandidateEvaluationResult:
    """Train once, evaluate on both test_new_prices and test_new_products; return GridRow.
    
    Uses TestOrchestrator with DetectorEvaluator for proper evaluation:
    1. Inject anomalies at DataFrame level (anomalies flow through feature extraction)
    2. Wrap detector in DetectorEvaluator with isolated cache
    3. Use TestOrchestrator.run_comparison() for evaluation with proper cache handling
    
    The TestOrchestrator internally uses BatchRoundProcessor which:
    - Processes test data in rounds (one observation per product per round)
    - Updates cache after each round with non-anomalous observations
    - Ensures products with multiple observations see their earlier observations in history
    
    Args:
        combo: Hyperparameter combination dict.
        X_train: Training feature matrix (for train_from_matrix).
        train_df: Training DataFrame (for TestOrchestrator cache population).
        df_eval_prices: Test DataFrame for new prices evaluation.
        df_eval_products: Test DataFrame for new products evaluation.
        run_id: Unique run identifier.
        dataset_name: Name of the dataset being evaluated.
        n_train: Number of training samples.
        n_eval_prices: Number of test samples (prices).
        n_eval_products: Number of test samples (products).
        country: Country code for numeric features (e.g., 'DK', 'NO').
        combo_idx: Current combination index (1-based) for progress display.
        total_combos: Total number of combinations for progress display.
    
    Returns:
        CandidateEvaluationResult with evaluation metrics and row-level artifacts.
    """
    nan_f = float("nan")
    progress_str = f"[{combo_idx}/{total_combos}]" if total_combos > 0 else ""
    
    try:
        train_start = time.perf_counter()
        print(
            f"\n{'='*60}\n"
            f"{progress_str} Training Isolation Forest\n"
            f"  n_estimators={combo['n_estimators']}, max_samples={combo['max_samples']}, "
            f"max_features={combo['max_features']}, threshold={combo['anomaly_threshold']}\n"
            f"{'='*60}",
            flush=True,
        )
        
        # Train using train_from_matrix with config parameters
        detector, _ = train_from_matrix(
            X_train,
            contamination=combo["contamination"],
            anomaly_threshold=combo["anomaly_threshold"],
            n_estimators=combo["n_estimators"],
            max_samples=combo["max_samples"],
            max_features=combo["max_features"],
            random_state=RANDOM_STATE,
        )
        
        training_time_sec = time.perf_counter() - train_start
        print(f"  [TRAIN] Completed in {training_time_sec:.1f}s", flush=True)
        
        # Inject anomalies at DataFrame level (anomalies flow through feature extraction)
        print(f"  [INJECT] Injecting anomalies (rate={INJECTION_RATE:.0%}, spike={SPIKE_RANGE}, drop={DROP_RANGE})...", flush=True)
        inject_start = time.perf_counter()
        
        df_prices_injected, labels_prices, _ = inject_anomalies_to_dataframe(
            df_eval_prices,
            injection_rate=INJECTION_RATE,
            seed=RANDOM_STATE,
            spike_range=SPIKE_RANGE,
            drop_range=DROP_RANGE,
        )
        n_injected_prices = int(labels_prices.sum())
        
        df_products_injected, labels_products, _ = inject_anomalies_to_dataframe(
            df_eval_products,
            injection_rate=INJECTION_RATE,
            seed=RANDOM_STATE + 1,  # Different seed for independence
            spike_range=SPIKE_RANGE,
            drop_range=DROP_RANGE,
        )
        n_injected_products = int(labels_products.sum())
        
        inject_time = time.perf_counter() - inject_start
        print(
            f"  [INJECT] Done in {inject_time:.1f}s: "
            f"{n_injected_prices} anomalies in prices, {n_injected_products} in products",
            flush=True,
        )
        
        # Create DetectorEvaluator wrapper for proper cache handling
        evaluator = DetectorEvaluator(
            detector=detector,
            name="iforest",
            enable_persistence_acceptance=False,  # Disable for clean evaluation
        )
        
        # Create TestOrchestrator for proper evaluation
        orchestrator = TestOrchestrator(evaluators=[evaluator], max_workers=1)
        
        # Evaluate on new prices (warm cache from train_df)
        print(f"  [EVAL] Evaluating on test_new_prices ({n_eval_prices} samples)...", flush=True)
        eval_prices_start = time.perf_counter()
        
        comparison_prices = orchestrator.run_comparison_with_details(
            train_df=train_df,
            test_df=df_prices_injected,
            labels=labels_prices,
            country=country,
        )
        res_prices = comparison_prices.metrics["iforest"]
        eval_prices_time = time.perf_counter() - eval_prices_start
        
        print(
            f"  [EVAL] test_new_prices ({eval_prices_time:.1f}s):\n"
            f"         Precision={res_prices.precision:.4f}  Recall={res_prices.recall:.4f}  F1={res_prices.f1:.4f}",
            flush=True,
        )
        
        # Evaluate on new products (warm cache from train_df)
        # Need to create fresh evaluator/orchestrator to reset cache
        evaluator2 = DetectorEvaluator(
            detector=detector,
            name="iforest",
            enable_persistence_acceptance=False,
        )
        orchestrator2 = TestOrchestrator(evaluators=[evaluator2], max_workers=1)
        
        print(f"  [EVAL] Evaluating on test_new_products ({n_eval_products} samples)...", flush=True)
        eval_products_start = time.perf_counter()
        
        comparison_products = orchestrator2.run_comparison_with_details(
            train_df=train_df,
            test_df=df_products_injected,
            labels=labels_products,
            country=country,
        )
        res_products = comparison_products.metrics["iforest"]
        eval_products_time = time.perf_counter() - eval_products_start
        
        print(
            f"  [EVAL] test_new_products ({eval_products_time:.1f}s):\n"
            f"         Precision={res_products.precision:.4f}  Recall={res_products.recall:.4f}  F1={res_products.f1:.4f}",
            flush=True,
        )
        
        # Compute combined metrics
        f1_p = res_prices.f1
        f1_prod = res_products.f1
        f1_combined = (f1_p + f1_prod) / 2.0
        prec_combined = (res_prices.precision + res_products.precision) / 2.0
        rec_combined = (res_prices.recall + res_products.recall) / 2.0
        
        total_time = time.perf_counter() - train_start
        
        print(
            f"\n  [SUMMARY] {progress_str} Total time: {total_time:.1f}s\n"
            f"  Combined: Precision={prec_combined:.4f}  Recall={rec_combined:.4f}  F1={f1_combined:.4f}\n"
            f"  {'='*56}",
            flush=True,
        )
        
        return CandidateEvaluationResult(
            row=GridRow(
                run_id=run_id,
                n_estimators=combo["n_estimators"],
                max_samples=combo["max_samples"],
                max_features=combo["max_features"],
                anomaly_threshold=combo["anomaly_threshold"],
                contamination=combo["contamination"],
                precision=prec_combined,
                recall=rec_combined,
                f1_score=f1_combined,
                precision_new_prices=res_prices.precision,
                recall_new_prices=res_prices.recall,
                f1_new_prices=f1_p,
                precision_new_products=res_products.precision,
                recall_new_products=res_products.recall,
                f1_new_products=f1_prod,
                training_time_sec=training_time_sec,
                dataset_name=dataset_name,
                n_train=n_train,
                n_eval_prices=n_eval_prices,
                n_eval_products=n_eval_products,
            ),
            comparisons={
                "new_prices": comparison_prices,
                "new_products": comparison_products,
            },
            config=combo.copy(),
        )
    except Exception as e:
        logger.warning("Combo %s failed: %s", run_id, e)
        return CandidateEvaluationResult(
            row=GridRow(
                run_id=run_id,
                n_estimators=combo["n_estimators"],
                max_samples=combo["max_samples"],
                max_features=combo["max_features"],
                anomaly_threshold=combo["anomaly_threshold"],
                contamination=combo["contamination"],
                precision=nan_f,
                recall=nan_f,
                f1_score=nan_f,
                precision_new_prices=nan_f,
                recall_new_prices=nan_f,
                f1_new_prices=nan_f,
                precision_new_products=nan_f,
                recall_new_products=nan_f,
                f1_new_products=nan_f,
                training_time_sec=0.0,
                dataset_name=dataset_name,
                n_train=n_train,
                n_eval_prices=n_eval_prices,
                n_eval_products=n_eval_products,
                error=str(e),
            ),
            comparisons={},
            config=combo.copy(),
        )


def build_candidate_metrics_frame(
    rows: list[GridRow],
    *,
    sweep_id: str,
    dataset_name: str,
    dataset_granularity: str,
) -> pd.DataFrame:
    """Build the canonical candidate_metrics.csv table."""
    records: list[dict[str, object]] = []
    for row in rows:
        records.append(
            {
                "schema_version": "phase2.v1",
                "sweep_id": sweep_id,
                "run_id": row.run_id,
                "candidate_id": row.run_id,
                "experiment_family": "tuning",
                "detector_family": "isolation_forest",
                "dataset_name": dataset_name,
                "dataset_granularity": dataset_granularity,
                "status": "error" if row.error else "ok",
                "error": row.error,
                "training_time_sec": row.training_time_sec,
                "n_train": row.n_train,
                "n_eval_prices": row.n_eval_prices,
                "n_eval_products": row.n_eval_products,
                "n_estimators": row.n_estimators,
                "max_samples": row.max_samples,
                "max_features": row.max_features,
                "anomaly_threshold": row.anomaly_threshold,
                "contamination": row.contamination,
                "combined_precision": row.precision,
                "combined_recall": row.recall,
                "combined_f1": row.f1_score,
                "new_prices_precision": row.precision_new_prices,
                "new_prices_recall": row.recall_new_prices,
                "new_prices_f1": row.f1_new_prices,
                "new_products_precision": row.precision_new_products,
                "new_products_recall": row.recall_new_products,
                "new_products_f1": row.f1_new_products,
            }
        )
    return pd.DataFrame(records)


def load_candidate_metrics(path: str) -> tuple[list[GridRow], str | None]:
    """Load existing candidate_metrics.csv for resume support."""
    if not os.path.exists(path):
        return [], None

    df = pd.read_csv(path)
    if df.empty:
        return [], None

    rows: list[GridRow] = []
    for _, record in df.iterrows():
        rows.append(
            GridRow(
                run_id=str(record.get("run_id", "")),
                n_estimators=int(record.get("n_estimators", 0)),
                max_samples=record.get("max_samples", "auto"),
                max_features=float(record.get("max_features", 0)),
                anomaly_threshold=float(record.get("anomaly_threshold", 0)),
                contamination=record.get("contamination", "auto"),
                precision=float(record.get("combined_precision", math.nan)),
                recall=float(record.get("combined_recall", math.nan)),
                f1_score=float(record.get("combined_f1", math.nan)),
                precision_new_prices=float(record.get("new_prices_precision", math.nan)),
                recall_new_prices=float(record.get("new_prices_recall", math.nan)),
                f1_new_prices=float(record.get("new_prices_f1", math.nan)),
                precision_new_products=float(record.get("new_products_precision", math.nan)),
                recall_new_products=float(record.get("new_products_recall", math.nan)),
                f1_new_products=float(record.get("new_products_f1", math.nan)),
                training_time_sec=float(record.get("training_time_sec", 0)),
                dataset_name=str(record.get("dataset_name", "")),
                n_train=int(record.get("n_train", 0)),
                n_eval_prices=int(record.get("n_eval_prices", 0)),
                n_eval_products=int(record.get("n_eval_products", 0)),
                error=str(record.get("error", "")) if pd.notna(record.get("error", "")) else "",
            )
        )
    sweep_id = str(df["sweep_id"].iloc[0]) if "sweep_id" in df.columns else None
    return rows, sweep_id


def write_candidate_run_artifacts(
    *,
    sweep_root: Path,
    candidate: CandidateEvaluationResult,
    dataset_name: str,
    dataset_granularity: str,
    train_file: str,
    test_file_prices: str,
    test_file_products: str,
    sweep_id: str,
) -> None:
    """Write canonical row-level artifacts for one sweep candidate."""
    candidate_id = candidate.row.run_id
    split_artifacts: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for split_name, comparison in candidate.comparisons.items():
        injected_rows, predictions = comparison_result_to_tables(
            comparison,
            run_id=candidate_id,
            candidate_id=candidate_id,
            experiment_family="tuning",
            dataset_name=dataset_name,
            dataset_granularity=dataset_granularity,
            dataset_split=split_name,
        )
        split_artifacts[split_name] = (injected_rows, predictions)

    expected_splits = ["new_prices", "new_products"]
    for split_name in expected_splits:
        split_artifacts.setdefault(
            split_name,
            (empty_injected_rows_table(), empty_predictions_table()),
        )

    run_metadata = {
        "schema_version": "phase2.v1",
        "experiment_family": "tuning",
        "run_id": candidate_id,
        "candidate_id": candidate_id,
        "sweep_id": sweep_id,
        "source_dataset_paths": [train_file, test_file_prices, test_file_products],
        "dataset_names": [dataset_name],
        "dataset_granularity": dataset_granularity,
        "dataset_splits": sorted(split_artifacts.keys()),
        "random_seeds": {"training_seed": RANDOM_STATE, "injection_seed": RANDOM_STATE},
        "injection_config": {
            "injection_rate": INJECTION_RATE,
            "spike_range": SPIKE_RANGE,
            "drop_range": DROP_RANGE,
        },
        "detector_identifiers": ["iforest"],
        "config_values": candidate.config,
        "training_time_sec": candidate.row.training_time_sec,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": resolve_git_commit(Path(_project_root)),
    }
    write_evaluation_run(
        run_root=sweep_root / "candidates" / candidate_id,
        run_metadata=run_metadata,
        split_artifacts=split_artifacts,
    )


def persist_candidate_run_artifacts(
    *,
    sweep_root: Path,
    candidate: CandidateEvaluationResult,
    dataset_name: str,
    dataset_granularity: str,
    train_file: str,
    test_file_prices: str,
    test_file_products: str,
    sweep_id: str,
    written_candidates: set[str],
) -> bool:
    """Persist one candidate directory unless it has already been written."""
    candidate_id = candidate.row.run_id
    if candidate_id in written_candidates:
        return False

    write_candidate_run_artifacts(
        sweep_root=sweep_root,
        candidate=candidate,
        dataset_name=dataset_name,
        dataset_granularity=dataset_granularity,
        train_file=train_file,
        test_file_prices=test_file_prices,
        test_file_products=test_file_products,
        sweep_id=sweep_id,
    )
    written_candidates.add(candidate_id)
    return True


CSV_FIELDNAMES = [
    "run_id", "n_estimators", "max_samples", "max_features", "anomaly_threshold",
    "contamination", "precision", "recall", "f1_score",
    "precision_new_prices", "recall_new_prices", "f1_new_prices",
    "precision_new_products", "recall_new_products", "f1_new_products",
    "training_time_sec",
    "dataset_name", "n_train", "n_eval_prices", "n_eval_products", "error",
]


def _parse_float(s: str) -> float:
    if s is None or (isinstance(s, str) and not s.strip()):
        return float("nan")
    try:
        return float(s)
    except (ValueError, TypeError):
        return float("nan")


def _parse_int(s: str) -> int:
    if s is None or (isinstance(s, str) and not s.strip()):
        return 0
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return 0


def _parse_max_samples(s: str) -> str | int:
    """Parse max_samples which can be 'auto' or an integer."""
    if s is None or (isinstance(s, str) and not s.strip()):
        return "auto"
    if s == "auto":
        return "auto"
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return "auto"


def _load_results_csv(path: str) -> tuple[list[GridRow], str | None]:
    """Load existing results CSV; return (rows, run_id_prefix). Prefix is None if empty."""
    if not os.path.exists(path):
        return [], None
    rows: list[GridRow] = []
    prefix: str | None = None
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f, fieldnames=CSV_FIELDNAMES)
        next(r, None)  # skip header
        for d in r:
            if not d.get("run_id"):
                continue
            rows.append(GridRow(
                run_id=d["run_id"],
                n_estimators=_parse_int(d.get("n_estimators", 0)),
                max_samples=_parse_max_samples(d.get("max_samples", "auto")),
                max_features=_parse_float(d.get("max_features", "")),
                anomaly_threshold=_parse_float(d.get("anomaly_threshold", "")),
                contamination=d.get("contamination", "auto"),
                precision=_parse_float(d.get("precision", "")),
                recall=_parse_float(d.get("recall", "")),
                f1_score=_parse_float(d.get("f1_score", "")),
                precision_new_prices=_parse_float(d.get("precision_new_prices", "")),
                recall_new_prices=_parse_float(d.get("recall_new_prices", "")),
                f1_new_prices=_parse_float(d.get("f1_new_prices", "")),
                precision_new_products=_parse_float(d.get("precision_new_products", "")),
                recall_new_products=_parse_float(d.get("recall_new_products", "")),
                f1_new_products=_parse_float(d.get("f1_new_products", "")),
                training_time_sec=_parse_float(d.get("training_time_sec", "0")),
                dataset_name=d.get("dataset_name", ""),
                n_train=_parse_int(d.get("n_train", 0)),
                n_eval_prices=_parse_int(d.get("n_eval_prices", 0)),
                n_eval_products=_parse_int(d.get("n_eval_products", 0)),
                error=d.get("error", ""),
            ))
            if prefix is None and rows:
                # e.g. "202601262200_1" -> "202601262200"
                prefix = rows[-1].run_id.rsplit("_", 1)[0] if "_" in rows[-1].run_id else None
    return rows, prefix


def _write_results_csv(rows: list[GridRow], path: str) -> None:
    """Write full results table CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            d = {
                "run_id": r.run_id,
                "n_estimators": r.n_estimators,
                "max_samples": r.max_samples,
                "max_features": r.max_features,
                "anomaly_threshold": r.anomaly_threshold,
                "contamination": r.contamination,
                "precision": r.precision,
                "recall": r.recall,
                "f1_score": r.f1_score,
                "precision_new_prices": r.precision_new_prices,
                "recall_new_prices": r.recall_new_prices,
                "f1_new_prices": r.f1_new_prices,
                "precision_new_products": r.precision_new_products,
                "recall_new_products": r.recall_new_products,
                "f1_new_products": r.f1_new_products,
                "training_time_sec": r.training_time_sec,
                "dataset_name": r.dataset_name,
                "n_train": r.n_train,
                "n_eval_prices": r.n_eval_prices,
                "n_eval_products": r.n_eval_products,
                "error": r.error,
            }
            w.writerow(d)


def _best_config(rows: list[GridRow], top_k: int = 5) -> tuple[GridRow | None, list[GridRow]]:
    """Return best row (by F1) and top-k list; ignore NaN."""
    valid = [r for r in rows if not (r.error or (r.f1_score != r.f1_score))]
    if not valid:
        return None, []
    valid.sort(key=lambda r: r.f1_score, reverse=True)
    return valid[0], valid[:top_k]


def _config_to_dict(r: GridRow) -> dict:
    return {
        "n_estimators": r.n_estimators,
        "max_samples": r.max_samples,
        "max_features": r.max_features,
        "anomaly_threshold": r.anomaly_threshold,
        "contamination": r.contamination,
    }


def _write_summary(
    output_dir: str,
    rows: list[GridRow],
    train_file: str,
    test_file_prices: str,
    test_file_products: str,
    n_train: int,
    n_eval_prices: int,
    n_eval_products: int,
    run_id_prefix: str,
) -> None:
    """Write grid_search_summary.md and grid_search_summary.json."""
    best, top_k = _best_config(rows)
    search_space = {
        "n_estimators": N_ESTIMATORS_OPTIONS,
        "max_samples": MAX_SAMPLES_OPTIONS,
        "max_features": MAX_FEATURES_OPTIONS,
        "anomaly_threshold": ANOMALY_THRESHOLDS,
        "contamination": [DEFAULT_CONTAMINATION],
    }
    run_metadata = {
        "dataset_path": train_file,
        "test_new_prices_path": test_file_prices,
        "test_new_products_path": test_file_products,
        "n_train": n_train,
        "n_eval_prices": n_eval_prices,
        "n_eval_products": n_eval_products,
        "evaluation": "DataFrame-level injection (inject_anomalies_to_dataframe) + TestOrchestrator",
        "injection_rate": INJECTION_RATE,
        "spike_range": SPIKE_RANGE,
        "drop_range": DROP_RANGE,
        "run_date_utc": datetime.now(timezone.utc).isoformat(),
        "run_id_prefix": run_id_prefix,
    }
    grid_info = {"total_combos": len(rows)}

    summary = {
        "search_space": search_space,
        "run_metadata": run_metadata,
        "grid": grid_info,
        "optimal_config": _config_to_dict(best) if best else None,
        "optimal_metrics": {
            "precision": best.precision,
            "recall": best.recall,
            "f1_score": best.f1_score,
            "f1_new_prices": best.f1_new_prices,
            "f1_new_products": best.f1_new_products,
        } if best else None,
        "top_k_configs": [_config_to_dict(r) for r in top_k],
        "top_k_metrics": [
            {
                "f1_score": r.f1_score,
                "f1_new_prices": r.f1_new_prices,
                "f1_new_products": r.f1_new_products,
                "precision": r.precision,
                "recall": r.recall,
            }
            for r in top_k
        ],
    }

    json_path = os.path.join(output_dir, "grid_search_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Narrative and Markdown
    n_total = len(rows)
    if best:
        opt_str = ", ".join(f"{k}={v}" for k, v in _config_to_dict(best).items())
        narrative = (
            f"Optimal Isolation Forest configuration: {opt_str} (F1={best.f1_score:.4f}). "
            f"Selected from {n_total} configurations via full Cartesian grid search."
        )
    else:
        narrative = "No valid configurations completed successfully."

    md_lines = [
        "# Isolation Forest Grid Search Summary",
        "",
        "## Search space (full Cartesian grid)",
        f"- n_estimators: {N_ESTIMATORS_OPTIONS}",
        f"- max_samples: {MAX_SAMPLES_OPTIONS}",
        f"- max_features: {MAX_FEATURES_OPTIONS}",
        f"- anomaly_threshold: {ANOMALY_THRESHOLDS}",
        f"- contamination: {DEFAULT_CONTAMINATION}",
        f"- Total combinations: {n_total}",
        "",
        "## Run metadata",
        f"- Train: {train_file}",
        f"- Test (new_prices): {test_file_prices} (n={n_eval_prices})",
        f"- Test (new_products): {test_file_products} (n={n_eval_products})",
        f"- n_train={n_train}",
        f"- Evaluation: DataFrame-level injection (rate={INJECTION_RATE}, spike={SPIKE_RANGE}, drop={DROP_RANGE})",
        f"- Uses TestOrchestrator + BatchRoundProcessor for proper cache handling",
        f"- Run date: {run_metadata['run_date_utc']}",
        "",
        "## Optimal configuration",
    ]
    if best:
        md_lines.append(f"- Config: {_config_to_dict(best)}")
        md_lines.append(
            f"- F1 (combined): {best.f1_score:.4f} | new_prices: {best.f1_new_prices:.4f} | new_products: {best.f1_new_products:.4f}"
        )
        md_lines.append(f"- Precision: {best.precision:.4f}, Recall: {best.recall:.4f}")
    else:
        md_lines.append("- None (all runs failed or had NaN F1).")
    md_lines.extend(["", "## One-line narrative", "", narrative])

    md_path = os.path.join(output_dir, "grid_search_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    logger.info("Wrote summary: %s and %s", json_path, md_path)


def main() -> None:
    """Single-run grid search: load data, run full Cartesian grid, write CSV + summary."""
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Isolation Forest hyperparameter grid search (single run, research outputs)"
    )
    parser.add_argument(
        "--granularity",
        type=str,
        choices=["country_segment", "competitor"],
        default="country_segment",
        help="Granularity for file resolution",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Canonical sweep root. Defaults to results/tuning/isolation_forest/<sweep_id>",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not use cached features",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing candidate_metrics.csv in the sweep directory",
    )
    parser.add_argument(
        "--sweep-id",
        type=str,
        default=None,
        help="Optional sweep id. Used when --output-dir is not provided.",
    )
    parser.add_argument(
        "--model-filter",
        type=str,
        default="NO_B2C",
        help="Filter to select specific model (e.g., NO_B2C, DK_B2B). Default: NO_B2C",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show grid combinations without training",
    )
    args = parser.parse_args()

    train_file = _resolve_train_file(args.model_filter, args.granularity)
    test_file_prices, test_file_products = _resolve_test_files(train_file)
    dataset_name = extract_model_name(train_file)
    use_cache = not args.no_cache
    sweep_id = args.sweep_id or datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    if args.output_dir:
        sweep_root = Path(args.output_dir)
        sweep_id = args.sweep_id or sweep_root.name
    else:
        sweep_root = Path("results") / "tuning" / "isolation_forest" / sweep_id

    print("=" * 70)
    print("Isolation Forest Grid Search")
    print("=" * 70)
    print(f"Train file: {train_file}")
    print(f"Test (new_prices): {test_file_prices}")
    print(f"Test (new_products): {test_file_products}")
    print(f"Dataset: {dataset_name}")
    print(f"Sweep root: {sweep_root}")
    print(f"Sweep ID: {sweep_id}")
    print()
    print("Search space:")
    print(f"  n_estimators: {N_ESTIMATORS_OPTIONS}")
    print(f"  max_samples: {MAX_SAMPLES_OPTIONS}")
    print(f"  max_features: {MAX_FEATURES_OPTIONS}")
    print(f"  anomaly_threshold: {ANOMALY_THRESHOLDS}")
    print(f"  contamination: {DEFAULT_CONTAMINATION}")
    n_combos = len(N_ESTIMATORS_OPTIONS) * len(MAX_SAMPLES_OPTIONS) * len(MAX_FEATURES_OPTIONS) * len(ANOMALY_THRESHOLDS)
    print(f"  Total combinations: {n_combos}")
    print()
    print("Evaluation:")
    print(f"  injection_rate: {INJECTION_RATE}")
    print(f"  spike_range: {SPIKE_RANGE}")
    print(f"  drop_range: {DROP_RANGE}")
    print(f"  Uses TestOrchestrator + BatchRoundProcessor for proper cache handling")
    print("=" * 70)
    
    if args.dry_run:
        print("\n[DRY RUN] Grid combinations:")
        combos = _build_grid_combos()
        for i, combo in enumerate(combos[:10]):  # Show first 10
            print(f"  {i+1}. n_est={combo['n_estimators']}, max_samp={combo['max_samples']}, max_feat={combo['max_features']}, thresh={combo['anomaly_threshold']}")
        if len(combos) > 10:
            print(f"  ... and {len(combos) - 10} more")
        print(f"\nTotal: {len(combos)} combinations")
        print("\nEstimated time: ~10-12s per combo = ~30-35 min total")
        return
    
    print("\n>>> Loading data...", flush=True)
    load_start = time.perf_counter()
    
    # Load training features (cached for fast training)
    print(f"  [1/4] Loading training features from {os.path.basename(train_file)}...", flush=True)
    X_train, n_train = _load_features(train_file, use_cache=use_cache)
    print(f"        Loaded {n_train:,} training samples, {X_train.shape[1]} features", flush=True)
    
    # Load training DataFrame for TestOrchestrator cache population
    print(f"  [2/4] Loading training DataFrame for cache population...", flush=True)
    train_df = _load_train_data(train_file)
    print(f"        Loaded {len(train_df):,} rows", flush=True)
    
    # Load evaluation DataFrames (raw data needed for proper anomaly injection)
    # inject_anomalies_to_dataframe works at DataFrame level so anomalies flow
    # through the full feature extraction pipeline
    print(f"  [3/4] Loading test_new_prices from {os.path.basename(test_file_prices)}...", flush=True)
    df_eval_prices, n_eval_prices = _load_eval_data(test_file_prices)
    print(f"        Loaded {n_eval_prices:,} samples", flush=True)
    
    print(f"  [4/4] Loading test_new_products from {os.path.basename(test_file_products)}...", flush=True)
    df_eval_products, n_eval_products = _load_eval_data(test_file_products)
    print(f"        Loaded {n_eval_products:,} samples", flush=True)
    
    load_time = time.perf_counter() - load_start
    print(f"\n>>> Data loading completed in {load_time:.1f}s", flush=True)
    print(f"    Train: {n_train:,} | Test (prices): {n_eval_prices:,} | Test (products): {n_eval_products:,}", flush=True)
    
    # Extract country code from model name for numeric features
    country = extract_country(dataset_name)
    print(f"    Country: {country}", flush=True)

    os.makedirs(sweep_root, exist_ok=True)
    csv_path = os.path.join(sweep_root, "candidate_metrics.csv")
    print(f"\n>>> Output directory: {sweep_root}", flush=True)

    # Resume: load existing candidate_metrics.csv and reuse sweep_id so we can skip completed combos
    completed_by_run_id: dict[str, GridRow] = {}
    if getattr(args, "resume", False) and os.path.exists(csv_path):
        print(f">>> Checking for existing results to resume...", flush=True)
        loaded_rows, saved_prefix = load_candidate_metrics(csv_path)
        if loaded_rows and saved_prefix:
            completed_by_run_id = {r.run_id: r for r in loaded_rows}
            sweep_id = saved_prefix
            print(f"    Found {len(completed_by_run_id)} completed runs (sweep {sweep_id}), will resume", flush=True)
        else:
            print(f"    No valid prior results found, starting fresh (sweep {sweep_id})", flush=True)

    all_rows: list[GridRow] = []
    written_candidates: set[str] = set()

    def _save_progress() -> None:
        try:
            candidate_metrics = build_candidate_metrics_frame(
                all_rows,
                sweep_id=sweep_id,
                dataset_name=dataset_name,
                dataset_granularity=args.granularity,
            )
            sweep_metadata = {
                "schema_version": "phase2.v1",
                "experiment_family": "tuning",
                "detector_family": "isolation_forest",
                "sweep_id": sweep_id,
                "source_dataset_paths": [train_file, test_file_prices, test_file_products],
                "dataset_names": [dataset_name],
                "dataset_granularity": args.granularity,
                "dataset_splits": ["new_prices", "new_products"],
                "random_seeds": {"training_seed": RANDOM_STATE, "injection_seed": RANDOM_STATE},
                "injection_config": {
                    "injection_rate": INJECTION_RATE,
                    "spike_range": SPIKE_RANGE,
                    "drop_range": DROP_RANGE,
                },
                "detector_identifiers": ["iforest"],
                "config_values": {
                    "model_filter": args.model_filter,
                    "no_cache": args.no_cache,
                },
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": resolve_git_commit(Path(_project_root)),
            }
            write_tuning_sweep(
                sweep_root=sweep_root,
                sweep_metadata=sweep_metadata,
                candidate_metrics=candidate_metrics,
            )
        except OSError as e:
            logger.warning("Failed to save progress to %s: %s (will retry next combo)", csv_path, e)

    # Full Cartesian grid search
    grid_combos = _build_grid_combos()
    n_combos = len(grid_combos)
    n_completed = sum(1 for i in range(n_combos) if f"{sweep_id}_{i+1}" in completed_by_run_id)
    
    print(f"\n>>> Starting grid search: {n_combos} total combinations", flush=True)
    if n_completed > 0:
        print(f"    Resuming: {n_completed} already completed, {n_combos - n_completed} remaining", flush=True)
    print(f"    Candidate ID prefix: {sweep_id}", flush=True)
    
    grid_start = time.perf_counter()
    combo_times: list[float] = []  # Track times for ETA calculation
    best_f1_so_far: float = -1.0
    best_config_so_far: dict | None = None
    
    for i, combo in enumerate(grid_combos):
        run_id = f"{sweep_id}_{i+1}"
        if run_id in completed_by_run_id:
            all_rows.append(completed_by_run_id[run_id])
            # Check if this resumed row is the best
            row = completed_by_run_id[run_id]
            if not row.error and row.f1_score == row.f1_score and row.f1_score > best_f1_so_far:
                best_f1_so_far = row.f1_score
                best_config_so_far = {
                    "n_estimators": row.n_estimators,
                    "max_samples": row.max_samples,
                    "max_features": row.max_features,
                    "anomaly_threshold": row.anomaly_threshold,
                }
            continue
        
        # Calculate ETA based on previous combos
        remaining = n_combos - i
        if combo_times:
            avg_time = sum(combo_times) / len(combo_times)
            eta_min = (avg_time * remaining) / 60.0
            elapsed_min = (time.perf_counter() - grid_start) / 60.0
            print(
                f"\n{'#'*70}\n"
                f">>> PROGRESS: {i}/{n_combos} combos done ({i*100//n_combos}%) | "
                f"Elapsed: {elapsed_min:.1f} min | ETA: ~{eta_min:.1f} min\n"
                f">>> Current best F1: {best_f1_so_far:.4f}" + 
                (f" (n_est={best_config_so_far['n_estimators']}, max_samp={best_config_so_far['max_samples']}, "
                 f"max_feat={best_config_so_far['max_features']}, thresh={best_config_so_far['anomaly_threshold']})"
                 if best_config_so_far else " (none yet)") +
                f"\n{'#'*70}",
                flush=True,
            )
        
        combo_start = time.perf_counter()
        candidate_result = _run_one(
            combo,
            X_train,
            train_df,
            df_eval_prices,
            df_eval_products,
            run_id,
            dataset_name,
            n_train,
            n_eval_prices,
            n_eval_products,
            country=country,
            combo_idx=i + 1,
            total_combos=n_combos,
        )
        combo_time = time.perf_counter() - combo_start
        combo_times.append(combo_time)

        row = candidate_result.row
        all_rows.append(row)
        persist_candidate_run_artifacts(
            sweep_root=sweep_root,
            candidate=candidate_result,
            dataset_name=dataset_name,
            dataset_granularity=args.granularity,
            train_file=train_file,
            test_file_prices=test_file_prices,
            test_file_products=test_file_products,
            sweep_id=sweep_id,
            written_candidates=written_candidates,
        )
        _save_progress()
        
        # Check if this is a new best
        if not row.error and row.f1_score == row.f1_score:  # NaN check
            if row.f1_score > best_f1_so_far:
                improvement = row.f1_score - best_f1_so_far if best_f1_so_far > 0 else row.f1_score
                print(
                    f"\n  *** NEW BEST! F1={row.f1_score:.4f} (+{improvement:.4f}) ***\n"
                    f"      Config: n_est={combo['n_estimators']}, max_samp={combo['max_samples']}, "
                    f"max_feat={combo['max_features']}, thresh={combo['anomaly_threshold']}",
                    flush=True,
                )
                best_f1_so_far = row.f1_score
                best_config_so_far = combo.copy()
        
        # After first combo: estimate total time
        if i == 0 and not row.error:
            total_est_min = (combo_time * n_combos) / 60.0
            print(
                f"\n>>> First combo took {combo_time:.1f}s. Estimated total: ~{total_est_min:.0f} min for {n_combos} combos",
                flush=True,
            )
    
    grid_elapsed = time.perf_counter() - grid_start
    print(
        f"\n{'#'*70}\n"
        f">>> GRID SEARCH COMPLETED\n"
        f">>> Total time: {grid_elapsed / 60.0:.1f} min ({grid_elapsed:.0f}s)\n"
        f">>> Combos evaluated: {len([r for r in all_rows if not r.error])}/{n_combos}\n"
        f"{'#'*70}",
        flush=True,
    )

    try:
        _save_progress()
        print(f">>> Results saved to: {csv_path}", flush=True)
    except OSError as e:
        logger.error("FAILED to write final results to %s: %s", csv_path, e)

    best, _ = _best_config(all_rows)
    print()
    print("=" * 60)
    if best:
        print("Optimal config:", _config_to_dict(best))
        print(
            f"  F1 (combined)={best.f1_score:.4f}  new_prices={best.f1_new_prices:.4f}  new_products={best.f1_new_products:.4f}"
        )
    else:
        print("No valid optimal config (all runs failed or NaN F1).")
    print("Summary:", sweep_root / "summary.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
