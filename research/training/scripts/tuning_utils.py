#!/usr/bin/env python3
"""Shared utilities for threshold tuning scripts.

This module contains shared functions used by detector-specific tuning scripts
(tune_isolation_forest.py, tune_autoencoder.py, tune_statistical.py). Provides
two tuning approaches for different detector types.

Key Components:
    ML Detector Tuning (score cutoff based):
        - get_anomaly_scores: Extract raw scores from a detector
        - evaluate_at_threshold: Compute metrics at a specific score cutoff
        - run_single_trial: Run one tuning trial for ML detectors
        - run_tuning_trials: Orchestrate multiple parallel trials for ML detectors

    Statistical Detector Tuning (threshold parameter based):
        - get_is_anomaly_predictions: Extract binary is_anomaly flags
        - evaluate_predictions: Compute metrics from binary predictions
        - run_single_statistical_trial: Run one trial for statistical detectors
        - run_statistical_tuning_trials: Orchestrate parallel trials for statistical detectors

    Shared:
        - TuningResult: Dataclass for tuning results
        - File utilities: find_parquet_files, find_train_file, extract_model_name

Design Decision:
    Two separate tuning approaches exist because ML detectors and statistical
    detectors have fundamentally different tuning needs. See the comment block
    above run_statistical_tuning_trials() for detailed explanation.

Usage:
    # For ML detectors (Autoencoder, IsolationForest) - tune score cutoffs
    from tuning_utils import run_tuning_trials
    result = run_tuning_trials(
        detector=detector,
        detector_name=model_name,
        test_df=test_df,
        train_df=train_df,
        thresholds=np.linspace(0.3, 0.9, 30),  # Score cutoffs
        current_threshold=detector.config.anomaly_threshold,
    )

    # For statistical detectors (ZScore, IQR, Threshold) - tune detector params
    from tuning_utils import run_statistical_tuning_trials
    result = run_statistical_tuning_trials(
        detector_type="zscore",  # or "iqr" or "threshold"
        detector_name="zscore",
        test_df=test_df,
        train_df=train_df,
        thresholds=np.linspace(2.0, 5.0, 20),  # Detector parameter values
        current_threshold=3.0,
    )
"""

import glob
import math
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.research.evaluation import DetectorEvaluator, inject_anomalies_to_dataframe

logger = logging.getLogger(__name__)


def _compute_binary_metrics(
    *,
    true_positives: int | float,
    false_positives: int | float,
    false_negatives: int | float,
    true_negatives: int | float,
) -> dict[str, Any]:
    """Compute a consistent binary-classification metric set from confusion counts."""
    total = true_positives + false_positives + false_negatives + true_negatives
    predicted_positives = true_positives + false_positives
    actual_positives = true_positives + false_negatives
    actual_negatives = false_positives + true_negatives

    precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
    recall = true_positives / actual_positives if actual_positives > 0 else 0.0
    tnr = true_negatives / actual_negatives if actual_negatives > 0 else 0.0
    fpr = false_positives / actual_negatives if actual_negatives > 0 else 0.0
    fnr = false_negatives / actual_positives if actual_positives > 0 else 0.0
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    g_mean = math.sqrt(recall * tnr) if recall > 0 and tnr > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "tnr": tnr,
        "fpr": fpr,
        "fnr": fnr,
        "f1": f1,
        "g_mean": g_mean,
        "true_positives": int(true_positives),
        "false_positives": int(false_positives),
        "true_negatives": int(true_negatives),
        "false_negatives": int(false_negatives),
        "n_rows": int(total),
        "n_injected": int(actual_positives),
        "n_predicted": int(predicted_positives),
    }


@dataclass
class TuningResult:
    """Result from tuning a single model.

    Attributes:
        model_name: Identifier for the model being tuned.
        granularity: Model granularity ('country_segment', 'competitor', or 'global').
        current_threshold: Threshold value before tuning.
        best_threshold: Optimal threshold found during tuning.
        current_f1: F1 score at the current threshold.
        best_f1: F1 score at the best threshold.
        best_precision: Precision at the best threshold.
        best_recall: Recall at the best threshold.
        improvement_pct: Percentage improvement in F1 (positive = better).
        all_results: List of results at each threshold evaluated.
        row_count: Number of rows in the test data used for tuning.
    """

    model_name: str
    granularity: str
    current_threshold: float
    best_threshold: float
    current_f1: float
    best_f1: float
    best_precision: float
    best_recall: float
    improvement_pct: float
    all_results: list[dict]
    row_count: int = 0


# =============================================================================
# File Discovery Utilities
# =============================================================================


def find_parquet_files(
    data_path: str,
    granularity: str,
    file_suffix: str = "_test",
) -> list[str]:
    """Find Parquet files for the specified granularity.

    Args:
        data_path: Base data directory.
        granularity: 'country_segment', 'competitor', or 'global'.
        file_suffix: Suffix filter (default: '_test').

    Returns:
        List of Parquet file paths, sorted alphabetically.

    Raises:
        ValueError: If granularity is invalid.
    """
    if granularity == "country_segment":
        subdir = "by_country_segment"
    elif granularity == "competitor":
        subdir = "by_competitor"
    elif granularity == "global":
        subdir = "global"
    else:
        raise ValueError(f"Invalid granularity: {granularity}")

    file_pattern = f"*{file_suffix}.parquet"

    if granularity == "competitor":
        pattern = os.path.join(data_path, subdir, "**", file_pattern)
        return sorted(glob.glob(pattern, recursive=True))

    pattern = os.path.join(data_path, subdir, file_pattern)
    return sorted(glob.glob(pattern))


def find_train_file(test_file: str) -> str | None:
    """Find corresponding train file for a test file.

    Handles various naming patterns:
        'DK_B2C_2026-01-18_test.parquet' -> 'DK_B2C_2026-01-18_train.parquet'
        'POWER_DK_B2C_INTERNAL_2026-01-22_test_new_prices.parquet' ->
            'POWER_DK_B2C_INTERNAL_2026-01-22_train.parquet'
        'POWER_DK_B2C_INTERNAL_2026-01-22_test_new_prices_mh5.parquet' ->
            'POWER_DK_B2C_INTERNAL_2026-01-22_train_mh5.parquet'

    Args:
        test_file: Path to test file.

    Returns:
        Path to train file if exists, None otherwise.
    """
    # Extract any _mh suffix (min-history variants like _mh4, _mh5)
    mh_match = re.search(r"(_mh\d+)\.parquet$", test_file)
    mh_suffix = mh_match.group(1) if mh_match else ""

    # Remove the mh suffix temporarily for replacement
    base = re.sub(r"_mh\d+\.parquet$", ".parquet", test_file)

    # Replace test patterns with train
    train_file = re.sub(
        r"_(test_new_prices|test_new_products|test)\.parquet$",
        "_train.parquet",
        base,
    )

    # Add back the mh suffix
    if mh_suffix:
        train_file = train_file.replace(".parquet", f"{mh_suffix}.parquet")

    if os.path.exists(train_file):
        return train_file
    return None


def extract_model_name(filepath: str) -> str:
    """Extract model name from Parquet filename.

    Handles various file naming patterns:
        'DK_B2C_2026-01-18_test.parquet' -> 'DK_B2C'
        'POWER_DK_B2C_INTERNAL_2026-01-18_test.parquet' -> 'POWER_DK_B2C_INTERNAL'
        'POWER_DK_B2C_INTERNAL_2026-01-22_test_new_prices.parquet' ->
            'POWER_DK_B2C_INTERNAL'
        'POWER_DK_B2C_INTERNAL_2026-01-22_test_new_prices_mh5.parquet' ->
            'POWER_DK_B2C_INTERNAL'
        'POWER_DK_B2C_INTERNAL_2026-01-22_train.parquet' -> 'POWER_DK_B2C_INTERNAL'

    Args:
        filepath: Path to Parquet file.

    Returns:
        Model name extracted from filename.
    """
    filename = os.path.basename(filepath)
    name = filename.replace(".parquet", "")

    # Remove suffixes like _mh4, _mh5, _mh6 (min-history variants)
    name = re.sub(r"_mh\d+$", "", name)

    # Remove _test_new_prices, _test_new_products, _train, _test suffixes
    name = re.sub(r"_(test_new_prices|test_new_products|train|test)$", "", name)

    # Remove date suffix like _2026-01-18 (YYYY-MM-DD format)
    name = re.sub(r"_\d{4}-\d{2}-\d{2}$", "", name)

    return name


# =============================================================================
# Score Extraction and Evaluation
# =============================================================================


def get_anomaly_scores(
    evaluator: DetectorEvaluator,
    df: pd.DataFrame,
    country: str | None = None,
    log_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract raw anomaly scores from evaluator.

    Uses DetectorEvaluator's batched path when the wrapped detector supports it,
    falling back to per-row processing otherwise. This keeps ML tuning aligned
    with the rounds-based batch orchestration used elsewhere in the evaluation
    stack while preserving compatibility with sequential-only detectors.

    Args:
        evaluator: DetectorEvaluator instance (already has cache populated).
        df: DataFrame with test data. Must have columns: product_id,
            competitor_id, price. If 'is_anomaly' column exists, it's used
            for labels; otherwise labels are all 0.
        country: Optional country code for numeric features.
        log_progress: Whether to log progress every 10k rows.

    Returns:
        Tuple of (scores, labels) as numpy arrays.
    """
    col_map = {col: idx for idx, col in enumerate(df.columns)}
    has_labels = "is_anomaly" in df.columns

    # Sort by time if time column exists (chronological processing)
    time_col = _get_time_column(df)
    if time_col:
        df = df.sort_values(time_col).reset_index(drop=True)
        # Rebuild col_map after sort
        col_map = {col: idx for idx, col in enumerate(df.columns)}

    total_rows = len(df)
    if has_labels:
        labels = df["is_anomaly"].astype(int).to_numpy(dtype=np.int64, copy=False)
    else:
        labels = np.zeros(total_rows, dtype=np.int64)

    supports_batch = callable(getattr(evaluator, "supports_batch", None)) and evaluator.supports_batch()
    if supports_batch:
        rows = list(df.itertuples(index=False))
        if log_progress:
            print(f"  Score extraction mode: batch ({total_rows:,} rows)")
        try:
            results = evaluator.process_batch(rows, col_map, country)
        except Exception as exc:
            if log_progress:
                print(f"  Batch score extraction failed, falling back to sequential: {exc}")
        else:
            scores = np.asarray([result.anomaly_score for result in results], dtype=np.float64)
            if log_progress:
                print(f"  Score extraction progress: {total_rows:,}/{total_rows:,} rows (100%)")
            return scores, labels

    scores = []
    log_interval = max(10000, total_rows // 10)  # Log every 10k rows or 10% progress

    for i, row in enumerate(df.itertuples(index=False)):
        result = evaluator.process_row(row, col_map, country)
        scores.append(result.anomaly_score)

        # Log progress (use print for child process visibility)
        if log_progress and (i + 1) % log_interval == 0:
            pct = (i + 1) / total_rows * 100
            print(f"  Score extraction progress: {i + 1:,}/{total_rows:,} rows ({pct:.0f}%)")

    return np.asarray(scores, dtype=np.float64), labels


def evaluate_at_threshold(
    scores: np.ndarray,
    threshold: float,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Evaluate detection metrics at a specific threshold.

    Args:
        scores: Anomaly scores (higher = more anomalous).
        threshold: Threshold to evaluate (scores > threshold = anomaly).
        labels: Ground truth labels (1 = anomaly, 0 = normal).

    Returns:
        Dictionary with threshold, precision, recall, f1, and confusion
        matrix values (true_positives, false_positives, etc.).
    """
    predictions = scores > threshold

    true_positives = np.sum(predictions & (labels == 1))
    false_positives = np.sum(predictions & (labels == 0))
    false_negatives = np.sum(~predictions & (labels == 1))
    true_negatives = np.sum(~predictions & (labels == 0))

    metrics = _compute_binary_metrics(
        true_positives=int(true_positives),
        false_positives=int(false_positives),
        false_negatives=int(false_negatives),
        true_negatives=int(true_negatives),
    )
    metrics["threshold"] = threshold
    return metrics


# =============================================================================
# Trial Execution (Parallel-friendly)
# =============================================================================


def run_single_trial(
    detector: Any,
    detector_name: str,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame | None,
    cache_snapshot_path: str | None,
    thresholds: np.ndarray,
    current_threshold: float,
    injection_rate: float,
    seed: int,
    country: str | None = None,
    spike_range: tuple[float, float] = (2.0, 5.0),
    drop_range: tuple[float, float] = (0.3, 0.8),
) -> dict[str, Any]:
    """Run single tuning trial - designed for parallel execution.

    Creates a fresh DetectorEvaluator with isolated cache for this trial,
    injects anomalies with the given seed, and evaluates at all thresholds.

    Args:
        detector: Detector instance (must have detect() method).
        detector_name: Name for the evaluator.
        test_df: Test DataFrame (without injected anomalies).
        train_df: Training DataFrame for cache population (None for cold-start).
        cache_snapshot_path: Optional path to a saved temporal cache snapshot.
            When provided, the evaluator loads the snapshot instead of rebuilding
            the cache from ``train_df``.
        thresholds: Array of threshold values to evaluate.
        current_threshold: Current threshold for comparison.
        injection_rate: Fraction of data to inject as anomalies.
        seed: Random seed for anomaly injection (different per trial).
        country: Optional country code for numeric features.
        spike_range: (min_multiplier, max_multiplier) for price spikes.
        drop_range: (min_ratio, max_ratio) for price drops.
            Note: drop_range (0.3, 0.8) means drops to 30%-80% of original,
            which excludes normal 10-30% discounts.

    Returns:
        Dictionary with:
            - threshold_results: List of metrics dicts for each threshold
            - current_result: Metrics dict at current_threshold
    """
    import time
    trial_start = time.time()

    # Use print() instead of logger - child processes don't propagate logs to parent
    print(f"[Trial seed={seed}] Starting: {len(test_df):,} test rows, {len(train_df) if train_df is not None else 0:,} train rows")

    # Create fresh evaluator with isolated cache for this trial
    evaluator = DetectorEvaluator(detector, detector_name)

    # Populate cache from a reusable snapshot when available; otherwise build it
    # from the filtered training data.
    if cache_snapshot_path:
        cache_start = time.time()
        print(f"[Trial seed={seed}] Loading cache snapshot from {cache_snapshot_path}...")
        evaluator.temporal_cache.load_from_file(cache_snapshot_path)
        print(f"[Trial seed={seed}] Cache loaded in {time.time() - cache_start:.1f}s")
    elif train_df is not None:
        cache_start = time.time()
        
        # Filter train data to only products in test set
        test_products = set(test_df["product_id"].unique())
        original_train_rows = len(train_df)
        train_df_filtered = train_df[train_df["product_id"].isin(test_products)]
        filtered_train_rows = len(train_df_filtered)
        
        print(
            f"[Trial seed={seed}] Filtered train data: {filtered_train_rows:,} rows "
            f"(was {original_train_rows:,}, {filtered_train_rows/original_train_rows:.1%} kept) "
            f"for {len(test_products):,} test products"
        )
        
        print(f"[Trial seed={seed}] Populating cache from {filtered_train_rows:,} train rows...")
        evaluator.populate_cache(train_df_filtered)
        print(f"[Trial seed={seed}] Cache populated in {time.time() - cache_start:.1f}s")

    # Inject anomalies at DataFrame level (before feature extraction)
    inject_start = time.time()
    print(f"[Trial seed={seed}] Injecting anomalies at {injection_rate:.0%} rate...")
    df_injected, inject_mask, _ = inject_anomalies_to_dataframe(
        test_df,
        injection_rate=injection_rate,
        seed=seed,
        spike_range=spike_range,
        drop_range=drop_range,
    )
    df_injected["is_anomaly"] = inject_mask.astype(int)
    n_injected = inject_mask.sum()
    print(f"[Trial seed={seed}] Injected {n_injected:,} anomalies in {time.time() - inject_start:.1f}s")

    # Get scores using the production-like evaluation path
    score_start = time.time()
    print(f"[Trial seed={seed}] Extracting anomaly scores from {len(df_injected):,} rows...")
    scores, labels = get_anomaly_scores(evaluator, df_injected, country, log_progress=True)
    print(f"[Trial seed={seed}] Score extraction complete in {time.time() - score_start:.1f}s")

    if len(scores) == 0:
        print(f"[Trial seed={seed}] WARNING: No scores extracted!")
        return {"threshold_results": [], "current_result": None}

    # Debug: Print score distribution (print() works across processes, logger doesn't)
    n_anomaly_labels = int(labels.sum())
    n_normal_labels = len(labels) - n_anomaly_labels
    print(
        f"[Trial seed={seed}] Score stats: min={scores.min():.3f}, max={scores.max():.3f}, "
        f"mean={scores.mean():.3f}, std={scores.std():.3f}"
    )
    print(
        f"[Trial seed={seed}] Labels: {n_anomaly_labels:,} anomalies, {n_normal_labels:,} normal "
        f"({n_anomaly_labels/len(labels):.1%} anomaly rate)"
    )
    
    # Check scores for anomaly vs normal labels
    if n_anomaly_labels > 0:
        anomaly_scores = scores[labels == 1]
        normal_scores = scores[labels == 0]
        print(
            f"[Trial seed={seed}] Anomaly scores: min={anomaly_scores.min():.3f}, "
            f"max={anomaly_scores.max():.3f}, mean={anomaly_scores.mean():.3f}"
        )
        print(
            f"[Trial seed={seed}] Normal scores: min={normal_scores.min():.3f}, "
            f"max={normal_scores.max():.3f}, mean={normal_scores.mean():.3f}"
        )

    # Evaluate at all thresholds
    eval_start = time.time()
    threshold_results = [evaluate_at_threshold(scores, th, labels) for th in thresholds]
    print(f"[Trial seed={seed}] Evaluated {len(thresholds)} thresholds in {time.time() - eval_start:.2f}s")
    
    # Print detailed results per threshold
    for r in threshold_results:
        print(
            f"[Trial seed={seed}] Threshold={r['threshold']:.3f}: "
            f"TP={r['true_positives']}, FP={r['false_positives']}, "
            f"FN={r['false_negatives']}, TN={r['true_negatives']}, F1={r['f1']:.1%}"
        )

    # Also evaluate current threshold
    current_result = evaluate_at_threshold(scores, current_threshold, labels)

    total_time = time.time() - trial_start
    best_f1 = max(r["f1"] for r in threshold_results)
    print(f"[Trial seed={seed}] Complete in {total_time:.1f}s, best F1={best_f1:.1%}")

    return {"threshold_results": threshold_results, "current_result": current_result}


def run_tuning_trials(
    detector: Any,
    detector_name: str,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame | None,
    thresholds: np.ndarray,
    current_threshold: float,
    cache_snapshot_path: str | None = None,
    n_trials: int = 5,
    injection_rate: float = 0.1,
    country: str | None = None,
    max_workers: int | None = None,
    target_metric: str = "f1",
    min_precision: float = 0.3,
    spike_range: tuple[float, float] = (2.0, 5.0),
    drop_range: tuple[float, float] = (0.3, 0.8),
    min_successful_trials: int = 3,
) -> TuningResult | None:
    """Run parallel tuning trials and aggregate results.

    Executes multiple trials with different random seeds in parallel,
    then averages results across trials to find the best threshold.

    Args:
        detector: Detector instance (must have detect() method).
        detector_name: Name for logging and results.
        test_df: Test DataFrame (without injected anomalies).
        train_df: Training DataFrame for cache population (None for cold-start).
        cache_snapshot_path: Optional path to a saved temporal cache snapshot.
        thresholds: Array of threshold values to evaluate.
        current_threshold: Current threshold for comparison.
        n_trials: Number of trials with different seeds to average.
        injection_rate: Fraction of data to inject as anomalies.
        country: Optional country code for numeric features.
        max_workers: Maximum parallel workers (defaults to n_trials).
        target_metric: Metric to optimize ('f1', 'precision', 'recall', 'gmean').
        min_precision: Minimum acceptable precision when optimizing recall.
        spike_range: (min_multiplier, max_multiplier) for price spikes.
        drop_range: (min_ratio, max_ratio) for price drops.
        min_successful_trials: Minimum successful trials required for valid results.
            If fewer trials succeed, returns None.

    Returns:
        TuningResult with best threshold and metrics, or None if tuning fails
        or fewer than min_successful_trials succeed.
    """
    seeds = [1000 + i * 17 for i in range(n_trials)]
    workers = min(max_workers or n_trials, n_trials)

    # Run trials in parallel
    # Try ProcessPoolExecutor first (true parallelism for CPU-bound work)
    # Fall back to ThreadPoolExecutor if objects aren't picklable
    trial_results: list[dict] = []

    def execute_trial(seed: int) -> dict[str, Any]:
        return run_single_trial(
            detector,
            detector_name,
            test_df,
            train_df,
            cache_snapshot_path,
            thresholds,
            current_threshold,
            injection_rate,
            seed,
            country,
            spike_range,
            drop_range,
        )

    def record_trial_result(result: dict[str, Any], completed_count: int) -> None:
        if result["threshold_results"]:
            trial_results.append(result)
            logger.debug(f"Trial {completed_count}/{n_trials} completed successfully")
        else:
            logger.warning(f"Trial {completed_count}/{n_trials} returned empty results")

    def run_trials_with_executor(executor_class, executor_name: str) -> bool:
        """Run trials with given executor. Returns True if successful."""
        nonlocal trial_results
        trial_results = []
        completed_count = 0

        logger.info(
            f"Starting {n_trials} parallel trials with {workers} workers "
            f"using {executor_name}"
        )

        try:
            with executor_class(max_workers=workers) as pool:
                futures = [
                    pool.submit(execute_trial, seed)
                    for seed in seeds
                ]
                for future in as_completed(futures):
                    completed_count += 1
                    try:
                        result = future.result()
                        record_trial_result(result, completed_count)
                    except Exception as e:
                        logger.warning(f"Trial {completed_count}/{n_trials} failed: {e}")
            return True
        except Exception as e:
            logger.warning(f"{executor_name} failed: {e}")
            return False

    if n_trials == 1:
        logger.info("Running 1 tuning trial inline without executor overhead")
        try:
            record_trial_result(execute_trial(seeds[0]), 1)
        except Exception as e:
            logger.warning(f"Single inline trial failed: {e}")
    elif workers <= 1:
        logger.info(f"Running {n_trials} tuning trials sequentially with 1 worker")
        for completed_count, seed in enumerate(seeds, start=1):
            try:
                record_trial_result(execute_trial(seed), completed_count)
            except Exception as e:
                logger.warning(f"Trial {completed_count}/{n_trials} failed: {e}")
    else:
        # Try ProcessPoolExecutor first (bypasses GIL for true parallelism)
        if not run_trials_with_executor(ProcessPoolExecutor, "ProcessPoolExecutor"):
            # Fall back to ThreadPoolExecutor if pickling fails
            logger.info("Falling back to ThreadPoolExecutor (GIL-limited, single-core effective)")
            run_trials_with_executor(ThreadPoolExecutor, "ThreadPoolExecutor")

    if not trial_results:
        logger.warning(f"No successful trials for {detector_name}")
        return None

    # Validate minimum successful trials
    successful_count = len(trial_results)
    if successful_count < min_successful_trials:
        logger.warning(
            f"Insufficient successful trials for {detector_name}: "
            f"{successful_count}/{n_trials} succeeded, need at least {min_successful_trials}"
        )
        return None

    logger.info(
        f"Trial validation passed for {detector_name}: "
        f"{successful_count}/{n_trials} trials succeeded (min required: {min_successful_trials})"
    )

    # Aggregate results across trials
    # Initialize accumulators for each threshold
    threshold_metrics: dict[int, list[dict[str, Any]]] = {
        i: [] for i in range(len(thresholds))
    }
    current_threshold_metrics: list[dict[str, Any]] = []

    for trial in trial_results:
        for i, result in enumerate(trial["threshold_results"]):
            threshold_metrics[i].append(result)

        if trial["current_result"]:
            current_threshold_metrics.append(trial["current_result"])

    # Average results across trials
    all_results = []
    for i, threshold in enumerate(thresholds):
        metrics = threshold_metrics[i]
        if not metrics:
            continue

        aggregate_fields = (
            "accuracy",
            "precision",
            "recall",
            "tnr",
            "fpr",
            "fnr",
            "f1",
            "g_mean",
            "true_positives",
            "false_positives",
            "false_negatives",
            "true_negatives",
            "n_rows",
            "n_injected",
            "n_predicted",
        )
        aggregated: dict[str, Any] = {"threshold": threshold}
        for field in aggregate_fields:
            values = [float(metric[field]) for metric in metrics]
            aggregated[field] = float(np.mean(values))
            aggregated[f"{field}_std"] = float(np.std(values))
        aggregated["n_trials"] = len(metrics)
        all_results.append(aggregated)

    if not all_results:
        return None

    # Average current threshold results
    avg_current_f1 = (
        np.mean([metric["f1"] for metric in current_threshold_metrics])
        if current_threshold_metrics
        else 0
    )

    # Find best threshold based on target metric
    best_result = None
    best_score = -1

    for result in all_results:
        score_key = {"f1": "f1", "precision": "precision", "recall": "recall", "gmean": "g_mean"}[target_metric]
        score = result[score_key]

        # If optimizing for recall, enforce minimum precision
        if target_metric == "recall" and result["precision"] < min_precision:
            continue

        if score > best_score:
            best_score = score
            best_result = result

    if best_result is None:
        # Fall back to best F1 if no result meets min_precision
        best_result = max(all_results, key=lambda r: r["f1"])

    # Calculate improvement
    improvement = (
        (best_result["f1"] - avg_current_f1) / avg_current_f1 * 100
        if avg_current_f1 > 0
        else 0
    )

    return TuningResult(
        model_name=detector_name,
        granularity="",  # Set by caller
        current_threshold=current_threshold,
        best_threshold=best_result["threshold"],
        current_f1=avg_current_f1,
        best_f1=best_result["f1"],
        best_precision=best_result["precision"],
        best_recall=best_result["recall"],
        improvement_pct=improvement,
        all_results=all_results,
        row_count=len(test_df),
    )


# =============================================================================
# Private Helpers
# =============================================================================


def _get_time_column(df: pd.DataFrame) -> str | None:
    """Find the timestamp column in the DataFrame.

    Args:
        df: DataFrame to search.

    Returns:
        Column name or None if not found.
    """
    for col in ["first_seen_at", "scraped_at", "timestamp"]:
        if col in df.columns:
            return col
    return None


# =============================================================================
# Statistical Detector Tuning (Threshold-Based)
# =============================================================================
#
# WHY DO WE HAVE BOTH run_tuning_trials() AND run_statistical_tuning_trials()?
#
# These functions serve fundamentally different tuning approaches:
#
# 1. run_tuning_trials() - For ML detectors (Autoencoder, IsolationForest)
#    - Creates ONE detector instance with fixed hyperparameters
#    - Extracts anomaly_score from each prediction
#    - Tests different SCORE CUTOFFS (e.g., 0.3, 0.5, 0.7) to find optimal threshold
#    - The score IS the decision metric (e.g., reconstruction error)
#    - Use this for: Autoencoder, IsolationForest, any detector where you tune
#      the decision threshold on a continuous score output
#
# 2. run_statistical_tuning_trials() - For statistical detectors (ZScore, IQR, Threshold)
#    - Creates MULTIPLE detector instances, one per threshold value
#    - Uses is_anomaly directly from each detector (no score cutoff)
#    - Tests different DETECTOR PARAMETERS (e.g., zscore threshold 2.0, 2.5, 3.0)
#    - The threshold IS the detector parameter, not a post-hoc cutoff
#    - Use this for: ZScoreDetector, IQRDetector, ThresholdDetector
#
# The key insight: Statistical detectors have a built-in threshold parameter that
# determines is_anomaly. Testing score cutoffs on these detectors is meaningless
# because anomaly_score is derived from the threshold (e.g., zscore/threshold).
# =============================================================================


def get_is_anomaly_predictions(
    evaluator: "DetectorEvaluator",
    df: pd.DataFrame,
    country: str | None = None,
    log_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract is_anomaly predictions and labels from evaluator.

    Unlike get_anomaly_scores() which extracts continuous scores for ML detectors,
    this function extracts binary is_anomaly flags for statistical detectors
    where the threshold is baked into the detector itself.

    Args:
        evaluator: DetectorEvaluator instance with detector configured.
        df: DataFrame with test data. Must have columns: product_id,
            competitor_id, price. If 'is_anomaly' column exists, it's used
            for labels; otherwise labels are all 0.
        country: Optional country code for numeric features.
        log_progress: Whether to log progress every 10k rows.

    Returns:
        Tuple of (predictions, labels) as numpy arrays of 0/1 values.
    """
    col_map = {col: idx for idx, col in enumerate(df.columns)}
    has_labels = "is_anomaly" in df.columns

    predictions = []
    labels = []

    # Sort by time if time column exists (chronological processing)
    time_col = _get_time_column(df)
    if time_col:
        df = df.sort_values(time_col).reset_index(drop=True)
        col_map = {col: idx for idx, col in enumerate(df.columns)}

    total_rows = len(df)
    log_interval = max(10000, total_rows // 10)

    for i, row in enumerate(df.itertuples(index=False)):
        result = evaluator.process_row(row, col_map, country)
        predictions.append(1 if result.is_anomaly else 0)

        if has_labels:
            labels.append(int(row[col_map["is_anomaly"]]))
        else:
            labels.append(0)

        if log_progress and (i + 1) % log_interval == 0:
            pct = (i + 1) / total_rows * 100
            print(f"  Prediction extraction progress: {i + 1:,}/{total_rows:,} rows ({pct:.0f}%)")

    return np.array(predictions), np.array(labels)


def evaluate_predictions(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Evaluate binary predictions against labels.

    Args:
        predictions: Binary predictions (0/1).
        labels: Ground truth labels (1 = anomaly, 0 = normal).

    Returns:
        Dictionary with precision, recall, f1, and confusion matrix values.
    """
    true_positives = np.sum((predictions == 1) & (labels == 1))
    false_positives = np.sum((predictions == 1) & (labels == 0))
    false_negatives = np.sum((predictions == 0) & (labels == 1))
    true_negatives = np.sum((predictions == 0) & (labels == 0))

    return _compute_binary_metrics(
        true_positives=int(true_positives),
        false_positives=int(false_positives),
        false_negatives=int(false_negatives),
        true_negatives=int(true_negatives),
    )


def _create_statistical_detector(detector_type: str, threshold: float):
    """Module-level factory for statistical detectors (picklable for ProcessPoolExecutor).

    This function is defined at module level so it can be pickled and sent to
    worker processes. Lambda functions defined inside other functions cannot
    be pickled, which breaks ProcessPoolExecutor.

    Args:
        detector_type: One of 'zscore', 'iqr', 'threshold'.
        threshold: Threshold value for the detector.

    Returns:
        Configured detector instance.
    """
    # Import here to avoid circular imports in multiprocessing
    from src.anomaly.statistical import ZScoreDetector, IQRDetector, ThresholdDetector

    if detector_type == "zscore":
        return ZScoreDetector(threshold=threshold)
    elif detector_type == "iqr":
        return IQRDetector(multiplier=threshold)
    elif detector_type == "threshold":
        return ThresholdDetector(threshold=threshold)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")


def run_single_statistical_trial(
    detector_type: str,
    detector_name: str,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame | None,
    threshold: float,
    injection_rate: float,
    seed: int,
    country: str | None = None,
    spike_range: tuple[float, float] = (2.0, 5.0),
    drop_range: tuple[float, float] = (0.3, 0.8),
) -> dict[str, Any]:
    """Run single tuning trial for a statistical detector at a specific threshold.

    Creates a fresh detector with the given threshold, evaluates using is_anomaly
    directly (not score cutoffs). Each (threshold, trial) combination gets its
    own isolated evaluator and cache for production-accurate simulation.

    Args:
        detector_type: Type of detector ('zscore', 'iqr', 'threshold').
        detector_name: Name for logging.
        test_df: Test DataFrame (without injected anomalies).
        train_df: Training DataFrame for cache population (None for cold-start).
        threshold: Threshold value to create detector with.
        injection_rate: Fraction of data to inject as anomalies.
        seed: Random seed for anomaly injection.
        country: Optional country code for numeric features.
        spike_range: (min_multiplier, max_multiplier) for price spikes.
        drop_range: (min_ratio, max_ratio) for price drops.

    Returns:
        Dictionary with metrics for this (threshold, trial) combination.
    """
    import time
    trial_start = time.time()

    print(
        f"[Trial seed={seed}, thresh={threshold:.3f}] Starting: "
        f"{len(test_df):,} test rows, "
        f"{len(train_df) if train_df is not None else 0:,} train rows"
    )

    # Create detector with this specific threshold using module-level factory
    detector = _create_statistical_detector(detector_type, threshold)

    # Create fresh evaluator with isolated cache
    evaluator = DetectorEvaluator(detector, detector_name)

    # Populate cache from training data
    if train_df is not None:
        cache_start = time.time()

        # Filter train data to only products in test set
        test_products = set(test_df["product_id"].unique())
        original_train_rows = len(train_df)
        train_df_filtered = train_df[train_df["product_id"].isin(test_products)]
        filtered_train_rows = len(train_df_filtered)

        print(
            f"[Trial seed={seed}, thresh={threshold:.3f}] Filtered train: "
            f"{filtered_train_rows:,}/{original_train_rows:,} rows "
            f"({filtered_train_rows/original_train_rows:.1%}) for {len(test_products):,} products"
        )

        evaluator.populate_cache(train_df_filtered)
        print(f"[Trial seed={seed}, thresh={threshold:.3f}] Cache populated in {time.time() - cache_start:.1f}s")

    # Inject anomalies
    inject_start = time.time()
    df_injected, inject_mask, _ = inject_anomalies_to_dataframe(
        test_df,
        injection_rate=injection_rate,
        seed=seed,
        spike_range=spike_range,
        drop_range=drop_range,
    )
    df_injected["is_anomaly"] = inject_mask.astype(int)
    n_injected = inject_mask.sum()
    print(
        f"[Trial seed={seed}, thresh={threshold:.3f}] Injected {n_injected:,} anomalies "
        f"in {time.time() - inject_start:.1f}s"
    )

    # Get is_anomaly predictions directly (not scores)
    pred_start = time.time()
    predictions, labels = get_is_anomaly_predictions(
        evaluator, df_injected, country, log_progress=True
    )
    print(
        f"[Trial seed={seed}, thresh={threshold:.3f}] Predictions extracted "
        f"in {time.time() - pred_start:.1f}s"
    )

    if len(predictions) == 0:
        print(f"[Trial seed={seed}, thresh={threshold:.3f}] WARNING: No predictions!")
        return {"threshold": threshold, "metrics": None}

    # Evaluate predictions
    metrics = evaluate_predictions(predictions, labels)

    # Add debug info
    n_anomaly_labels = int(labels.sum())
    n_predicted_anomalies = int(predictions.sum())
    print(
        f"[Trial seed={seed}, thresh={threshold:.3f}] Labels: {n_anomaly_labels:,} anomalies, "
        f"Predicted: {n_predicted_anomalies:,} anomalies"
    )
    print(
        f"[Trial seed={seed}, thresh={threshold:.3f}] TP={metrics['true_positives']}, "
        f"FP={metrics['false_positives']}, FN={metrics['false_negatives']}, "
        f"TN={metrics['true_negatives']}, F1={metrics['f1']:.1%}"
    )

    total_time = time.time() - trial_start
    print(f"[Trial seed={seed}, thresh={threshold:.3f}] Complete in {total_time:.1f}s")

    return {
        "threshold": threshold,
        "seed": seed,
        "metrics": metrics,
    }


def run_single_statistical_trial_optimized(
    detector_type: str,
    detector_name: str,
    pre_injected_df: pd.DataFrame,
    train_df_filtered: pd.DataFrame | None,
    threshold: float,
    seed: int,
    country: str | None = None,
    pre_populated_cache: dict | None = None,
) -> dict[str, Any]:
    """Run single tuning trial with pre-injected data (optimized version).

    This version skips anomaly injection since it receives pre-injected data.
    Also receives pre-filtered train_df to avoid redundant filtering.
    Optionally receives pre-populated cache to skip cache population entirely.

    Args:
        detector_type: Type of detector ('zscore', 'iqr', 'threshold').
        detector_name: Name for logging.
        pre_injected_df: DataFrame with anomalies already injected and 'is_anomaly' column.
        train_df_filtered: Pre-filtered training DataFrame (only products in test set).
            Ignored if pre_populated_cache is provided.
        threshold: Threshold value to create detector with.
        seed: Seed used for injection (for logging only).
        country: Optional country code for numeric features.
        pre_populated_cache: Optional pre-populated cache dict to restore instead of
            populating from train_df. This is a deep copy of TemporalCacheManager._caches.

    Returns:
        Dictionary with metrics for this (threshold, trial) combination.
    """
    import copy
    import time
    trial_start = time.time()

    n_anomalies = int(pre_injected_df["is_anomaly"].sum()) if "is_anomaly" in pre_injected_df.columns else 0
    print(
        f"[Trial seed={seed}, thresh={threshold:.3f}] Starting (optimized): "
        f"{len(pre_injected_df):,} rows, {n_anomalies:,} anomalies pre-injected"
    )

    # Create detector with this specific threshold using module-level factory
    detector = _create_statistical_detector(detector_type, threshold)

    # Create fresh evaluator with isolated cache
    evaluator = DetectorEvaluator(detector, detector_name)

    # Restore pre-populated cache OR populate from train_df
    if pre_populated_cache is not None:
        cache_start = time.time()
        # Deep copy the cache so this trial has its own isolated copy
        evaluator.temporal_cache._caches = copy.deepcopy(pre_populated_cache)
        cache_stats = evaluator.temporal_cache.get_stats()
        print(
            f"[Trial seed={seed}, thresh={threshold:.3f}] Cache restored "
            f"({cache_stats['total_products']:,} products) in {time.time() - cache_start:.1f}s"
        )
    elif train_df_filtered is not None and not train_df_filtered.empty:
        cache_start = time.time()
        evaluator.populate_cache(train_df_filtered)
        print(
            f"[Trial seed={seed}, thresh={threshold:.3f}] Cache populated "
            f"({len(train_df_filtered):,} rows) in {time.time() - cache_start:.1f}s"
        )

    # Get is_anomaly predictions directly (no injection needed!)
    pred_start = time.time()
    predictions, labels = get_is_anomaly_predictions(
        evaluator, pre_injected_df, country, log_progress=True
    )
    print(
        f"[Trial seed={seed}, thresh={threshold:.3f}] Predictions extracted "
        f"in {time.time() - pred_start:.1f}s"
    )

    if len(predictions) == 0:
        print(f"[Trial seed={seed}, thresh={threshold:.3f}] WARNING: No predictions!")
        return {"threshold": threshold, "metrics": None}

    # Evaluate predictions
    metrics = evaluate_predictions(predictions, labels)

    # Add debug info
    n_anomaly_labels = int(labels.sum())
    n_predicted_anomalies = int(predictions.sum())
    print(
        f"[Trial seed={seed}, thresh={threshold:.3f}] Labels: {n_anomaly_labels:,} anomalies, "
        f"Predicted: {n_predicted_anomalies:,} anomalies"
    )
    print(
        f"[Trial seed={seed}, thresh={threshold:.3f}] TP={metrics['true_positives']}, "
        f"FP={metrics['false_positives']}, FN={metrics['false_negatives']}, "
        f"TN={metrics['true_negatives']}, F1={metrics['f1']:.1%}"
    )

    total_time = time.time() - trial_start
    print(f"[Trial seed={seed}, thresh={threshold:.3f}] Complete in {total_time:.1f}s")

    return {
        "threshold": threshold,
        "seed": seed,
        "metrics": metrics,
    }


def _pre_inject_single_seed(
    test_df: pd.DataFrame,
    seed: int,
    injection_rate: float,
    spike_range: tuple[float, float],
    drop_range: tuple[float, float],
) -> tuple[int, pd.DataFrame]:
    """Pre-inject anomalies for a single seed (picklable for parallel execution).

    Args:
        test_df: Original test DataFrame.
        seed: Random seed for injection.
        injection_rate: Fraction of data to inject.
        spike_range: Range for price spikes.
        drop_range: Range for price drops.

    Returns:
        Tuple of (seed, injected_df with is_anomaly column).
    """
    import time
    start = time.time()

    df_injected, inject_mask, _ = inject_anomalies_to_dataframe(
        test_df,
        injection_rate=injection_rate,
        seed=seed,
        spike_range=spike_range,
        drop_range=drop_range,
    )
    df_injected["is_anomaly"] = inject_mask.astype(int)

    duration = time.time() - start
    n_injected = inject_mask.sum()
    print(f"[Pre-inject seed={seed}] Injected {n_injected:,} anomalies in {duration:.1f}s")

    return seed, df_injected


def pre_inject_all_seeds(
    test_df: pd.DataFrame,
    seeds: list[int],
    injection_rate: float,
    spike_range: tuple[float, float] = (2.0, 5.0),
    drop_range: tuple[float, float] = (0.3, 0.8),
    max_workers: int | None = None,
) -> dict[int, pd.DataFrame]:
    """Pre-inject anomalies for all seeds in parallel.

    This is a major optimization: instead of injecting anomalies for each
    (threshold, seed) combination, we inject once per seed and reuse across
    all threshold evaluations.

    Args:
        test_df: Original test DataFrame.
        seeds: List of random seeds to use.
        injection_rate: Fraction of data to inject as anomalies.
        spike_range: Range for price spike multipliers.
        drop_range: Range for price drop multipliers.
        max_workers: Max parallel workers (default: len(seeds)).

    Returns:
        Dict mapping seed -> pre-injected DataFrame.
    """
    import time
    start = time.time()
    workers = max_workers or len(seeds)

    print(f"Pre-injecting anomalies for {len(seeds)} seeds using {workers} workers...")
    logger.info(f"Pre-injecting anomalies: {len(seeds)} seeds, {len(test_df):,} rows")

    pre_injected: dict[int, pd.DataFrame] = {}

    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _pre_inject_single_seed,
                    test_df,
                    seed,
                    injection_rate,
                    spike_range,
                    drop_range,
                )
                for seed in seeds
            ]

            for future in as_completed(futures):
                try:
                    seed, df_injected = future.result()
                    pre_injected[seed] = df_injected
                except Exception as e:
                    logger.warning(f"Pre-injection failed: {e}")

    except Exception as e:
        logger.warning(f"ProcessPoolExecutor failed for pre-injection: {e}, falling back to sequential")
        # Fall back to sequential
        for seed in seeds:
            _, df_injected = _pre_inject_single_seed(
                test_df, seed, injection_rate, spike_range, drop_range
            )
            pre_injected[seed] = df_injected

    duration = time.time() - start
    print(f"Pre-injection complete: {len(pre_injected)} seeds in {duration:.1f}s")
    logger.info(f"Pre-injection complete: {len(pre_injected)} seeds in {duration:.1f}s")

    return pre_injected


def pre_populate_cache(
    train_df_filtered: pd.DataFrame,
    detector_type: str = "zscore",
    detector_name: str = "cache_builder",
) -> dict:
    """Pre-populate the temporal cache from training data.

    Creates a DetectorEvaluator, populates its cache, and returns the internal
    cache dict for sharing across trials.

    Args:
        train_df_filtered: Pre-filtered training DataFrame.
        detector_type: Detector type (doesn't matter for cache, just for evaluator init).
        detector_name: Name for the evaluator.

    Returns:
        The internal _caches dict from TemporalCacheManager, ready for deep-copying.
    """
    import time
    start = time.time()

    print(f"Pre-populating cache from {len(train_df_filtered):,} train rows...")
    logger.info(f"Pre-populating cache: {len(train_df_filtered):,} rows")

    # Create a temporary evaluator just to populate the cache
    detector = _create_statistical_detector(detector_type, 3.0)  # threshold doesn't matter
    evaluator = DetectorEvaluator(detector, detector_name)
    evaluator.populate_cache(train_df_filtered)

    # Get cache statistics
    cache_stats = evaluator.temporal_cache.get_stats()
    duration = time.time() - start

    print(
        f"Cache pre-populated: {cache_stats['total_products']:,} products, "
        f"{cache_stats['total_observations']:,} observations in {duration:.1f}s"
    )
    logger.info(
        f"Cache pre-populated: {cache_stats['total_products']:,} products, "
        f"{cache_stats['total_observations']:,} observations in {duration:.1f}s"
    )

    # Return the internal cache dict (will be deep-copied for each trial)
    return evaluator.temporal_cache._caches


def run_statistical_tuning_trials(
    detector_type: str,
    detector_name: str,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame | None,
    thresholds: np.ndarray,
    current_threshold: float,
    n_trials: int = 5,
    injection_rate: float = 0.1,
    country: str | None = None,
    max_workers: int | None = None,
    target_metric: str = "f1",
    min_precision: float = 0.3,
    spike_range: tuple[float, float] = (2.0, 5.0),
    drop_range: tuple[float, float] = (0.3, 0.8),
    min_successful_trials: int = 3,
) -> TuningResult | None:
    """Run parallel tuning trials for statistical detectors.

    Unlike run_tuning_trials() which tests score cutoffs on a single detector,
    this function creates fresh detectors with different threshold values and
    evaluates using is_anomaly directly.

    OPTIMIZATION (2026-01-26): Pre-injects anomalies for all seeds upfront and
    pre-filters train_df once, eliminating redundant work across threshold
    evaluations. This reduces runtime from O(thresholds × trials × injection_time)
    to O(trials × injection_time + thresholds × trials × prediction_time).

    Parallelization strategy: Runs trials in parallel for each threshold value.
    Each (threshold, trial) gets its own isolated evaluator and cache.

    Args:
        detector_type: Type of detector ('zscore', 'iqr', 'threshold').
            Used with _create_statistical_detector() to create fresh instances.
        detector_name: Name for logging and results.
        test_df: Test DataFrame (without injected anomalies).
        train_df: Training DataFrame for cache population (None for cold-start).
        thresholds: Array of threshold values to test.
        current_threshold: Current threshold for comparison metrics.
        n_trials: Number of trials with different injection seeds per threshold.
        injection_rate: Fraction of data to inject as anomalies.
        country: Optional country code for numeric features.
        max_workers: Maximum parallel workers (defaults to n_trials).
        target_metric: Metric to optimize ('f1', 'precision', 'recall').
        min_precision: Minimum acceptable precision when optimizing recall.
        spike_range: (min_multiplier, max_multiplier) for price spikes.
        drop_range: (min_ratio, max_ratio) for price drops.
        min_successful_trials: Minimum successful trials required per threshold.

    Returns:
        TuningResult with best threshold and metrics, or None if tuning fails.
    """
    import time
    tuning_start = time.time()

    seeds = [1000 + i * 17 for i in range(n_trials)]
    workers = min(max_workers or n_trials, n_trials)  # Cap at n_trials for trial-level parallelism

    logger.info(
        f"Starting statistical tuning (optimized): {len(thresholds)} thresholds × {n_trials} trials "
        f"= {len(thresholds) * n_trials} evaluations, {workers} workers"
    )

    # ==========================================================================
    # OPTIMIZATION PHASE 1: Pre-inject anomalies for all seeds (done once!)
    # ==========================================================================
    pre_injected = pre_inject_all_seeds(
        test_df=test_df,
        seeds=seeds,
        injection_rate=injection_rate,
        spike_range=spike_range,
        drop_range=drop_range,
        max_workers=workers,
    )

    if len(pre_injected) < min_successful_trials:
        logger.error(f"Pre-injection failed: only {len(pre_injected)} seeds succeeded")
        return None

    # ==========================================================================
    # OPTIMIZATION PHASE 2: Pre-filter train_df and pre-populate cache (done once!)
    # ==========================================================================
    train_df_filtered = None
    pre_populated_cache = None

    if train_df is not None and not train_df.empty:
        # Step 2a: Pre-filter train_df
        filter_start = time.time()
        test_products = set(test_df["product_id"].unique())
        train_df_filtered = train_df[train_df["product_id"].isin(test_products)]
        filter_duration = time.time() - filter_start
        print(
            f"Pre-filtered train data: {len(train_df_filtered):,}/{len(train_df):,} rows "
            f"({len(train_df_filtered)/len(train_df):.1%}) in {filter_duration:.1f}s"
        )
        logger.info(
            f"Pre-filtered train data: {len(train_df_filtered):,} rows for "
            f"{len(test_products):,} test products in {filter_duration:.1f}s"
        )

        # Step 2b: Pre-populate cache (NEW OPTIMIZATION!)
        pre_populated_cache = pre_populate_cache(
            train_df_filtered=train_df_filtered,
            detector_type=detector_type,
            detector_name=detector_name,
        )

    # ==========================================================================
    # THRESHOLD EVALUATION PHASE: Use pre-computed data
    # ==========================================================================
    # Results storage: threshold -> list of trial metrics
    threshold_results: dict[float, list[dict]] = {float(t): [] for t in thresholds}

    # Process each threshold, running trials in parallel
    for thresh_idx, threshold in enumerate(thresholds):
        threshold = float(threshold)
        logger.info(
            f"Threshold {thresh_idx + 1}/{len(thresholds)}: {threshold:.3f} "
            f"- running {n_trials} trials"
        )

        trial_results: list[dict] = []

        # Run trials in parallel for this threshold using pre-injected data + pre-populated cache
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(
                        run_single_statistical_trial_optimized,
                        detector_type,
                        detector_name,
                        pre_injected[seed],
                        train_df_filtered,  # Kept for fallback, but cache is preferred
                        threshold,
                        seed,
                        country,
                        pre_populated_cache,  # NEW: Pass pre-populated cache
                    )
                    for seed in seeds
                    if seed in pre_injected  # Only use successfully injected seeds
                ]

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result.get("metrics") is not None:
                            trial_results.append(result)
                    except Exception as e:
                        logger.warning(f"Trial failed for threshold {threshold}: {e}")

        except Exception as e:
            logger.warning(f"ProcessPoolExecutor failed for threshold {threshold}: {e}")
            # Fall back to ThreadPoolExecutor
            try:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [
                        pool.submit(
                            run_single_statistical_trial_optimized,
                            detector_type,
                            detector_name,
                            pre_injected[seed],
                            train_df_filtered,
                            threshold,
                            seed,
                            country,
                            pre_populated_cache,  # NEW: Pass pre-populated cache
                        )
                        for seed in seeds
                        if seed in pre_injected
                    ]

                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            if result.get("metrics") is not None:
                                trial_results.append(result)
                        except Exception as e:
                            logger.warning(f"Trial failed for threshold {threshold}: {e}")

            except Exception as e:
                logger.error(f"Both executors failed for threshold {threshold}: {e}")

        threshold_results[threshold] = trial_results

        # Log progress
        successful = len(trial_results)
        if successful >= min_successful_trials:
            avg_f1 = np.mean([r["metrics"]["f1"] for r in trial_results])
            logger.info(
                f"Threshold {threshold:.3f}: {successful}/{n_trials} trials succeeded, "
                f"avg F1={avg_f1:.1%}"
            )
        else:
            logger.warning(
                f"Threshold {threshold:.3f}: Only {successful}/{n_trials} trials succeeded "
                f"(need {min_successful_trials})"
            )

    # Aggregate results across trials for each threshold
    all_results = []
    current_result_metrics = None

    for threshold, trials in threshold_results.items():
        if len(trials) < min_successful_trials:
            logger.warning(
                f"Skipping threshold {threshold:.3f}: insufficient trials "
                f"({len(trials)} < {min_successful_trials})"
            )
            continue

        # Average metrics across trials
        avg_precision = np.mean([t["metrics"]["precision"] for t in trials])
        avg_recall = np.mean([t["metrics"]["recall"] for t in trials])
        avg_f1 = np.mean([t["metrics"]["f1"] for t in trials])
        std_f1 = np.std([t["metrics"]["f1"] for t in trials])

        result = {
            "threshold": threshold,
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1,
            "f1_std": std_f1,
            "n_trials": len(trials),
        }
        all_results.append(result)

        # Track current threshold result for comparison
        if abs(threshold - current_threshold) < 0.001:
            current_result_metrics = result

    if not all_results:
        logger.warning(f"No valid results for {detector_name}")
        return None

    # Find current threshold F1 (or closest if not in threshold list)
    if current_result_metrics is not None:
        avg_current_f1 = current_result_metrics["f1"]
    else:
        # Find closest threshold to current
        closest = min(all_results, key=lambda r: abs(r["threshold"] - current_threshold))
        avg_current_f1 = closest["f1"]
        logger.info(
            f"Current threshold {current_threshold:.3f} not in test set, "
            f"using closest {closest['threshold']:.3f} for comparison"
        )

    # Find best threshold based on target metric
    best_result = None
    best_score = -1

    for result in all_results:
        score = result[target_metric]

        # If optimizing for recall, enforce minimum precision
        if target_metric == "recall" and result["precision"] < min_precision:
            continue

        if score > best_score:
            best_score = score
            best_result = result

    if best_result is None:
        # Fall back to best F1 if no result meets min_precision
        best_result = max(all_results, key=lambda r: r["f1"])

    # Calculate improvement
    improvement = (
        (best_result["f1"] - avg_current_f1) / avg_current_f1 * 100
        if avg_current_f1 > 0
        else 0
    )

    logger.info(
        f"Statistical tuning complete for {detector_name}: "
        f"best_threshold={best_result['threshold']:.3f}, "
        f"best_f1={best_result['f1']:.1%}, "
        f"improvement={improvement:+.1f}%"
    )

    return TuningResult(
        model_name=detector_name,
        granularity="",  # Set by caller
        current_threshold=current_threshold,
        best_threshold=best_result["threshold"],
        current_f1=avg_current_f1,
        best_f1=best_result["f1"],
        best_precision=best_result["precision"],
        best_recall=best_result["recall"],
        improvement_pct=improvement,
        all_results=all_results,
        row_count=len(test_df),
    )
