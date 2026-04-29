"""TestOrchestrator - Orchestrates detector comparison with parallel execution.

This class manages multiple DetectorEvaluator instances and coordinates their
evaluation on the same test data. Unlike SequentialEvaluator which shares a
single cache across detectors, TestOrchestrator gives each detector its own
isolated cache through DetectorEvaluator.

Key Features:
    - Each detector has isolated cache (no cross-contamination)
    - Parallel execution for clearing, populating, and per-row processing
    - Centralized metrics computation
    - Data-driven test scenarios (train_df presence determines scenario)

Usage:
    # Create evaluators with isolated caches
    evaluators = [
        DetectorEvaluator(ZScoreDetector(), "zscore"),
        DetectorEvaluator(IQRDetector(), "iqr"),
        DetectorEvaluator(AutoencoderDetector(...), "autoencoder"),
    ]
    
    # Create orchestrator
    orchestrator = TestOrchestrator(evaluators, max_workers=len(evaluators))
    
    # Run comparison - train_df determines scenario
    # With train_df: known products (warm cache)
    # Without train_df: new products (cold start)
    results = orchestrator.run_comparison(train_df, test_df, labels)
    
    # Results is dict[str, DetectorMetrics] with P/R/F1 for each evaluator
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.anomaly.persistence import StatisticalConfig
from src.anomaly.statistical import AnomalyResult, StatisticalEnsemble
from src.research.evaluation.detector_evaluator import DetectorEvaluator

logger = logging.getLogger(__name__)


@dataclass
class TypeDetectionMetrics:
    """Detection metrics for a single anomaly type.
    
    Attributes:
        anomaly_type: The type of anomaly (e.g., 'price_spike', 'decimal_shift').
        injected: How many of this type were injected.
        detected: How many were detected by this detector.
        rate: Detection rate (detected / injected).
    """
    
    anomaly_type: str
    injected: int
    detected: int
    rate: float


@dataclass
class DetectorMetrics:
    """Metrics from evaluating a single detector.
    
    Attributes:
        detector_name: Name of the detector/evaluator.
        precision: True positives / (True positives + False positives).
        recall: True positives / (True positives + False negatives).
        f1: Harmonic mean of precision and recall.
        true_positives: Count of correctly identified anomalies.
        false_positives: Count of normal records flagged as anomalies.
        false_negatives: Count of missed anomalies.
        n_samples: Total number of samples evaluated.
        predictions: Raw boolean predictions array.
        scores: Raw anomaly scores array (if available).
        detection_by_type: Per-anomaly-type detection metrics.
    """

    detector_name: str
    precision: float
    recall: float
    f1: float
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    n_samples: int = 0
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    scores: np.ndarray = field(default_factory=lambda: np.array([]))
    detection_by_type: dict[str, TypeDetectionMetrics] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Full result from a detector comparison run.
    
    Contains both aggregate metrics and detailed per-row results for
    downstream analysis (overlap, cascade effectiveness, etc.).
    
    Attributes:
        metrics: Dict mapping detector names to DetectorMetrics.
        raw_results: Dict mapping detector names to list of AnomalyResult.
        observation_counts: Array of observation counts per test row.
        labels: Ground truth labels used for evaluation.
        df_sorted: The test DataFrame after time-sorting (with injected anomaly
            metadata columns). Aligned with raw_results and labels.
    """
    
    metrics: dict[str, DetectorMetrics]
    raw_results: dict[str, list[AnomalyResult]]
    observation_counts: np.ndarray
    labels: np.ndarray
    df_sorted: pd.DataFrame | None = None

    def __repr__(self) -> str:
        detector_names = list(self.metrics.keys())
        return f"ComparisonResult(detectors={detector_names}, n_samples={len(self.labels)})"


class TestOrchestrator:
    """Orchestrates detector comparison with parallel execution.
    
    This class coordinates multiple DetectorEvaluator instances, running them
    in parallel where possible and computing aggregate metrics.
    
    Each evaluator maintains its own isolated cache, so anomaly decisions made
    by one detector don't affect another detector's baseline statistics.
    
    Attributes:
        evaluators: List of DetectorEvaluator instances to compare.
        max_workers: Maximum parallel workers for ThreadPoolExecutor.
    """

    def __init__(
        self,
        evaluators: list[DetectorEvaluator],
        max_workers: int = 4,
    ) -> None:
        """Initialize the orchestrator.
        
        Args:
            evaluators: List of DetectorEvaluator instances to compare.
            max_workers: Maximum number of parallel workers. Set to
                len(evaluators) for maximum parallelism.
        """
        if not evaluators:
            raise ValueError("At least one evaluator is required")
        
        self.evaluators = evaluators
        self.max_workers = max_workers
        
        # Verify evaluator names are unique
        names = [e.name for e in evaluators]
        if len(names) != len(set(names)):
            raise ValueError(f"Evaluator names must be unique, got: {names}")

    def run_comparison(
        self,
        train_df: pd.DataFrame | None,
        test_df: pd.DataFrame,
        labels: np.ndarray,
        country: str | None = None,
        skip_cache_setup: bool = False,
        injection_details: list[dict] | None = None,
    ) -> dict[str, DetectorMetrics]:
        """Run full comparison across all evaluators.
        
        This method orchestrates the entire evaluation flow:
        1. Clear all caches (parallel) - skipped if skip_cache_setup=True
        2. Populate caches from train_df (parallel) - skipped if skip_cache_setup=True
        3. Process all test rows (each evaluator processes same row)
        4. Compute metrics for each evaluator
        
        Args:
            train_df: Historical data for cache (empty/None for cold start).
                If provided, simulates "known products" scenario.
                If None or empty, simulates "new products" scenario.
                Ignored if skip_cache_setup=True.
            test_df: Test data to evaluate. Must have columns:
                product_id, competitor_id, price, and optionally a time column.
            labels: Ground truth anomaly labels (boolean array, same length as test_df).
            country: Optional country code for numeric features.
            skip_cache_setup: If True, skip cache clearing and population.
                Use when evaluator caches are pre-populated (e.g., grid search
                with a template cache cloned via temporal_cache.copy_from()).
            injection_details: Optional list of injection details from
                inject_anomalies_to_dataframe(). Used to compute per-anomaly-type
                detection rates.
        
        Returns:
            Dict mapping evaluator names to DetectorMetrics.
        
        Raises:
            ValueError: If labels length doesn't match test_df.
        """
        result = self.run_comparison_with_details(
            train_df, test_df, labels, country, skip_cache_setup, injection_details
        )
        return result.metrics

    def run_comparison_with_details(
        self,
        train_df: pd.DataFrame | None,
        test_df: pd.DataFrame,
        labels: np.ndarray,
        country: str | None = None,
        skip_cache_setup: bool = False,
        injection_details: list[dict] | None = None,
    ) -> ComparisonResult:
        """Run full comparison and return detailed results for analysis.
        
        This method extends run_comparison() by also returning:
        - Raw AnomalyResult lists for each detector (for overlap analysis)
        - Observation counts per row (for history bucket analysis)
        
        Args:
            train_df: Historical data for cache (empty/None for cold start).
                Ignored if skip_cache_setup=True.
            test_df: Test data to evaluate.
            labels: Ground truth anomaly labels.
            country: Optional country code for numeric features.
            skip_cache_setup: If True, skip cache clearing and population.
                Use when evaluator caches are pre-populated.
            injection_details: Optional list of injection details from
                inject_anomalies_to_dataframe(). Used to compute per-anomaly-type
                detection rates. Each dict should contain 'index' and 'anomaly_type'.
        
        Returns:
            ComparisonResult with metrics, raw results, and observation counts.
        """
        import time
        
        if len(labels) != len(test_df):
            raise ValueError(
                f"Labels length ({len(labels)}) must match test_df ({len(test_df)})"
            )

        test_df = test_df.copy()
        if "source_row_index" not in test_df.columns:
            test_df["source_row_index"] = np.arange(len(test_df), dtype=np.int64)
        test_df["ground_truth_label"] = np.asarray(labels).astype(bool)
        if "is_injected" not in test_df.columns:
            test_df["is_injected"] = test_df["ground_truth_label"]

        print(f"[Orchestrator] Starting comparison with {len(self.evaluators)} evaluators")
        if skip_cache_setup:
            print("[Orchestrator] Using pre-populated cache (skip_cache_setup=True)")
        else:
            print(f"[Orchestrator] Train: {len(train_df) if train_df is not None else 0:,} rows, Test: {len(test_df):,} rows")
        
        # Sort test data by time for chronological processing
        # CRITICAL: Must sort labels along with DataFrame to maintain alignment
        time_col = self._get_time_column(test_df)
        if time_col:
            # Add labels as column, sort, then extract back
            test_df["__labels__"] = labels
            test_df = test_df.sort_values(time_col).reset_index(drop=True)
            labels = test_df["__labels__"].values
            test_df = test_df.drop(columns=["__labels__"])
            print(f"[Orchestrator] Sorted by {time_col} (labels aligned)")

        test_df["evaluation_row_id"] = np.arange(len(test_df), dtype=np.int64)
        test_df["ground_truth_label"] = np.asarray(labels).astype(bool)

        if not skip_cache_setup:
            # Step 0: Clear all caches (parallel)
            t0 = time.time()
            print("[Orchestrator] Step 0: Clearing caches...")
            self._clear_all()
            print(f"[Orchestrator] Step 0 done in {time.time() - t0:.1f}s")
            
            # Step 1: Populate caches (parallel) - no-op if train_df empty
            t0 = time.time()
            print("[Orchestrator] Step 1: Populating caches from train data...")
            self._populate_all(train_df)
            print(f"[Orchestrator] Step 1 done in {time.time() - t0:.1f}s")
        
        # Step 2: Process all rows and collect results per evaluator
        t0 = time.time()
        print("[Orchestrator] Step 2: Processing test rows...")
        results_per_evaluator, observation_counts = self._process_all_rows_with_obs(test_df, country)
        print(f"[Orchestrator] Step 2 done in {time.time() - t0:.1f}s")
        
        # Step 3: Compute metrics for each evaluator
        t0 = time.time()
        print("[Orchestrator] Step 3: Computing metrics...")
        metrics = self._compute_all_metrics(
            results_per_evaluator, labels, injection_details, test_df
        )
        print(f"[Orchestrator] Step 3 done in {time.time() - t0:.1f}s")
        
        logger.info(
            "test_orchestrator_run_comparison_complete",
            extra={
                "metrics": {name: m.f1 for name, m in metrics.items()},
            },
        )
        
        return ComparisonResult(
            metrics=metrics,
            raw_results=results_per_evaluator,
            observation_counts=observation_counts,
            labels=np.asarray(labels).astype(bool),
            df_sorted=test_df,
        )

    def _clear_all(self) -> None:
        """Clear all evaluator caches in parallel."""
        if len(self.evaluators) == 1:
            self.evaluators[0].clear()
            return
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [pool.submit(e.clear) for e in self.evaluators]
            # Wait for all to complete
            for future in as_completed(futures):
                future.result()  # Raises any exceptions
        
        logger.debug(
            "test_orchestrator_caches_cleared",
            extra={"evaluator_count": len(self.evaluators)},
        )

    def _populate_all(self, train_df: pd.DataFrame | None) -> None:
        """Populate all evaluator caches from training data.
        
        Optimization: Populate one cache, then copy to all others (much faster
        than populating each cache separately from the same data).
        
        Args:
            train_df: Training data for cache population. No-op if None or empty.
        """
        import time
        
        if train_df is None or train_df.empty:
            print("[Orchestrator] Populate skipped (no train data)")
            return
        
        print(f"[Orchestrator] Populating cache with {len(train_df):,} rows (populate once, copy to {len(self.evaluators)} evaluators)...")
        
        # Populate first evaluator's cache
        t0 = time.time()
        first_eval = self.evaluators[0]
        first_eval.populate_cache(train_df)
        populate_time = time.time() - t0
        print(f"  [Populate] {first_eval.name} done in {populate_time:.1f}s")
        
        # DEBUG: Check cache statistics after population
        cache = first_eval.temporal_cache
        total_products = 0
        total_with_history = 0
        obs_counts = []
        for competitor_id, competitor_cache in cache._caches.items():
            for product_id, entry in competitor_cache.items():
                total_products += 1
                obs_counts.append(entry.observation_count)
                if entry.observation_count >= 3:
                    total_with_history += 1
        
        if obs_counts:
            import numpy as np
            obs_arr = np.array(obs_counts)
            print(f"  [DEBUG] Cache after populate:")
            print(f"  [DEBUG]   Total products in cache: {total_products:,}")
            print(f"  [DEBUG]   Products with >= 3 obs (sufficient history): {total_with_history:,} ({total_with_history/total_products*100:.1f}%)")
            print(f"  [DEBUG]   Observation counts: min={obs_arr.min()}, max={obs_arr.max()}, mean={obs_arr.mean():.1f}")
        
        # Copy to remaining evaluators
        if len(self.evaluators) > 1:
            t0 = time.time()
            for evaluator in self.evaluators[1:]:
                evaluator.temporal_cache.copy_from(first_eval.temporal_cache)
            copy_time = time.time() - t0
            print(f"  [Copy] Copied cache to {len(self.evaluators) - 1} evaluators in {copy_time:.1f}s")

    def _process_all_rows(
        self,
        test_df: pd.DataFrame,
        country: str | None,
    ) -> dict[str, list[AnomalyResult]]:
        """Process all test rows through all evaluators.
        
        For ML detectors (Autoencoder and Isolation Forest), uses batch
        detection which is ~50-100x faster. Statistical detectors continue
        with sequential processing to maintain cache evolution.
        
        Args:
            test_df: Test DataFrame to process.
            country: Optional country code for numeric features.
        
        Returns:
            Dict mapping evaluator names to list of AnomalyResult.
        """
        # Delegate to the extended method and discard observation counts
        results, _ = self._process_all_rows_with_obs(test_df, country)
        return results

    def _process_all_rows_with_obs(
        self,
        test_df: pd.DataFrame,
        country: str | None,
    ) -> tuple[dict[str, list[AnomalyResult]], np.ndarray]:
        """Process all test rows and capture observation counts.
        
        Extended version of _process_all_rows that also returns observation
        counts for each test row (from the first evaluator's cache, since
        all caches are populated identically before test processing).
        
        Args:
            test_df: Test DataFrame to process.
            country: Optional country code for numeric features.
        
        Returns:
            Tuple of (results dict, observation_counts array).
        """
        import time
        
        results: dict[str, list[AnomalyResult]] = {e.name: [] for e in self.evaluators}
        
        if test_df.empty:
            return results, np.array([], dtype=np.int32)
        
        # Note: Sorting is now done in run_comparison_with_details() to keep
        # labels aligned. The test_df passed here is already sorted.
        
        # Build column map for fast access
        col_map = {col: idx for idx, col in enumerate(test_df.columns)}
        
        # Convert to list of tuples for faster iteration
        print("[Orchestrator] Converting DataFrame to tuples...")
        t0 = time.time()
        rows = list(test_df.itertuples(index=False))
        n_rows = len(rows)
        print(f"[Orchestrator] Converted {n_rows:,} rows in {time.time() - t0:.1f}s")
        
        # Capture observation counts from first evaluator's cache BEFORE processing
        # (shows historical depth available for each product at detection time)
        print("[Orchestrator] Capturing observation counts...")
        observation_counts = self._capture_observation_counts(rows, col_map)
        print(f"[Orchestrator] Captured {len(observation_counts):,} observation counts")
        
        # Separate batch-capable and sequential evaluators
        batch_evaluators = [e for e in self.evaluators if e.supports_batch()]
        sequential_evaluators = [e for e in self.evaluators if not e.supports_batch()]
        
        if batch_evaluators:
            print(f"[Orchestrator] Batch detection: {[e.name for e in batch_evaluators]}")
        if sequential_evaluators:
            print(f"[Orchestrator] Sequential detection: {[e.name for e in sequential_evaluators]}")
        
        def process_evaluator_batch(evaluator: DetectorEvaluator) -> tuple[str, list[AnomalyResult]]:
            """Process all rows through a batch-capable evaluator."""
            t0 = time.time()
            print(f"  [Batch] {evaluator.name} starting ({n_rows:,} rows)...")
            eval_results = evaluator.process_batch(rows, col_map, country)
            elapsed = time.time() - t0
            rate = n_rows / elapsed if elapsed > 0 else 0
            print(f"  [Batch] {evaluator.name} done in {elapsed:.1f}s ({rate:,.0f} rows/s)")
            # Log cache stats after batch processing
            self._log_cache_stats(evaluator, "post-batch")
            return evaluator.name, eval_results
        
        def process_evaluator_sequential(evaluator: DetectorEvaluator) -> tuple[str, list[AnomalyResult]]:
            """Process all rows through a sequential evaluator."""
            t0 = time.time()
            print(f"  [Sequential] {evaluator.name} starting...")
            eval_results = []
            for i, row in enumerate(rows):
                result = evaluator.process_row(row, col_map, country)
                eval_results.append(result)
                # Log progress and cache stats every 5K rows
                if (i + 1) % 5000 == 0:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    eta = (n_rows - i - 1) / rate if rate > 0 else 0
                    print(f"  [Sequential] {evaluator.name}: {i + 1:,}/{n_rows:,} ({rate:.0f} rows/s, ETA {eta:.0f}s)")
                    # Log cache stats at each progress checkpoint
                    self._log_cache_stats(evaluator, f"row-{i+1}")
            elapsed = time.time() - t0
            final_rate = n_rows / elapsed if elapsed > 0 else 0.0
            print(f"  [Sequential] {evaluator.name} done in {elapsed:.1f}s ({final_rate:.0f} rows/s)")
            # Log final cache stats after sequential processing
            self._log_cache_stats(evaluator, "post-sequential")
            return evaluator.name, eval_results
        
        # Process batch evaluators first (they're faster)
        print(f"[Orchestrator] Processing {n_rows:,} rows through {len(self.evaluators)} evaluators...")
        
        for evaluator in batch_evaluators:
            name, eval_results = process_evaluator_batch(evaluator)
            results[name] = eval_results
            print(f"[Orchestrator] {name} complete ({len(eval_results):,} results)")
        
        # Process sequential evaluators
        for evaluator in sequential_evaluators:
            name, eval_results = process_evaluator_sequential(evaluator)
            results[name] = eval_results
            print(f"[Orchestrator] {name} complete ({len(eval_results):,} results)")
        
        return results, observation_counts

    def _capture_observation_counts(
        self,
        rows: list[tuple],
        col_map: dict[str, int],
    ) -> np.ndarray:
        """Capture observation counts from the first evaluator's cache.
        
        Uses the first evaluator's cache (all caches are populated identically
        from train data) to get observation counts for each test row.
        
        Args:
            rows: List of tuples from df.itertuples(index=False).
            col_map: Dict mapping column names to indices.
        
        Returns:
            Array of observation counts, one per row.
        """
        import pandas as pd
        
        cache = self.evaluators[0].temporal_cache
        counts = np.zeros(len(rows), dtype=np.int32)
        
        # DEBUG counters
        n_found = 0
        n_not_found = 0
        
        for i, row in enumerate(rows):
            product_id = str(row[col_map["product_id"]])
            competitor_id = str(row[col_map["competitor_id"]])
            
            entry = cache.get(product_id, competitor_id)
            if entry is not None:
                counts[i] = entry.observation_count
                n_found += 1
            else:
                n_not_found += 1
        
        # DEBUG output
        print(f"  [DEBUG] Observation counts capture:")
        print(f"  [DEBUG]   Rows with cache entry: {n_found:,} ({n_found/len(rows)*100:.1f}%)")
        print(f"  [DEBUG]   Rows without cache entry: {n_not_found:,} ({n_not_found/len(rows)*100:.1f}%)")
        print(f"  [DEBUG]   Rows with >= 3 obs (sufficient history): {(counts >= 3).sum():,} ({(counts >= 3).sum()/len(rows)*100:.1f}%)")
        print(f"  [DEBUG]   Observation count distribution: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")
        
        return counts

    def _log_cache_stats(self, evaluator: DetectorEvaluator, stage: str) -> None:
        """Log cache statistics for an evaluator.
        
        Logs:
            - Count of products in the cache
            - Average history length (observation count) per product
        
        Args:
            evaluator: The evaluator to inspect.
            stage: Description of when this is being logged (e.g., "post-batch", "row-5000").
        """
        cache = evaluator.temporal_cache
        
        total_products = 0
        total_obs = 0
        obs_counts = []
        
        # Iterate through all cache entries
        for competitor_id, competitor_cache in cache._caches.items():
            for product_id, entry in competitor_cache.items():
                total_products += 1
                obs_counts.append(entry.observation_count)
                total_obs += entry.observation_count
        
        if total_products > 0:
            avg_history = total_obs / total_products
            obs_arr = np.array(obs_counts)
            print(f"  [CACHE {stage}] {evaluator.name}:")
            print(f"  [CACHE {stage}]   Products in cache: {total_products:,}")
            print(f"  [CACHE {stage}]   Avg history length per product: {avg_history:.2f}")
            print(f"  [CACHE {stage}]   History distribution: min={obs_arr.min()}, max={obs_arr.max()}, median={np.median(obs_arr):.0f}")
        else:
            print(f"  [CACHE {stage}] {evaluator.name}: Cache is empty (0 products)")

    def _compute_all_metrics(
        self,
        results_per_evaluator: dict[str, list[AnomalyResult]],
        labels: np.ndarray,
        injection_details: list[dict] | None = None,
        df_injected: pd.DataFrame | None = None,
    ) -> dict[str, DetectorMetrics]:
        """Compute metrics for all evaluators.
        
        Args:
            results_per_evaluator: Dict mapping evaluator names to result lists.
            labels: Ground truth labels.
            injection_details: Optional list of injection details from 
                inject_anomalies_to_dataframe(). Each dict contains 'index',
                'anomaly_type', 'original_price', 'new_price', etc.
            df_injected: Optional DataFrame with __injected_anomaly_type__ column.
                When provided, per-type detection reads from aligned DataFrame columns.
        
        Returns:
            Dict mapping evaluator names to DetectorMetrics.
        """
        metrics = {}
        
        for name, results in results_per_evaluator.items():
            metrics[name] = self._compute_metrics(
                name, results, labels, injection_details, df_injected
            )
        
        return metrics

    def _compute_metrics(
        self,
        detector_name: str,
        results: list[AnomalyResult],
        labels: np.ndarray,
        injection_details: list[dict] | None = None,
        df_injected: pd.DataFrame | None = None,
    ) -> DetectorMetrics:
        """Compute precision, recall, F1 for a single evaluator.
        
        Args:
            detector_name: Name of the detector/evaluator.
            results: List of AnomalyResult from processing.
            labels: Ground truth labels (boolean array).
            injection_details: Optional list of injection details from
                inject_anomalies_to_dataframe(). Each dict contains 'index',
                'anomaly_type', 'original_price', 'new_price', etc.
            df_injected: Optional DataFrame with __injected_anomaly_type__ column.
                When provided, per-type detection reads from aligned DataFrame columns.
        
        Returns:
            DetectorMetrics with computed metrics including per-type detection rates.
        """
        if not results:
            return DetectorMetrics(
                detector_name=detector_name,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                n_samples=0,
            )
        
        # Extract predictions and scores
        predictions = np.array([r.is_anomaly for r in results])
        scores = np.array([r.anomaly_score for r in results])
        
        # Handle NaN predictions
        predictions = np.nan_to_num(predictions, nan=False).astype(bool)
        labels_bool = np.asarray(labels).astype(bool)
        
        # Compute confusion matrix values
        tp = int(np.sum(predictions & labels_bool))
        fp = int(np.sum(predictions & ~labels_bool))
        fn = int(np.sum(~predictions & labels_bool))
        
        # Compute metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Compute per-type detection rates
        detection_by_type = self._compute_per_type_detection(
            predictions, injection_details, df_injected
        )
        
        return DetectorMetrics(
            detector_name=detector_name,
            precision=precision,
            recall=recall,
            f1=f1,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            n_samples=len(results),
            predictions=predictions,
            scores=scores,
            detection_by_type=detection_by_type,
        )

    def _compute_per_type_detection(
        self,
        predictions: np.ndarray,
        injection_details: list[dict] | None,
        df_injected: pd.DataFrame | None = None,
    ) -> dict[str, TypeDetectionMetrics]:
        """Compute detection rates for each anomaly type.
        
        Args:
            predictions: Boolean array of detector predictions.
            injection_details: List of injection details (legacy fallback).
            df_injected: DataFrame with __injected_anomaly_type__ column (preferred).
                When provided, reads anomaly types from DataFrame columns which stay
                aligned with predictions after time-based sorting.
        
        Returns:
            Dict mapping anomaly type names to TypeDetectionMetrics.
        """
        # Prefer DataFrame columns over injection_details (aligned after sorting)
        anomaly_type_column = None
        if df_injected is not None and "anomaly_type" in df_injected.columns:
            anomaly_type_column = "anomaly_type"
        elif df_injected is not None and "__injected_anomaly_type__" in df_injected.columns:
            anomaly_type_column = "__injected_anomaly_type__"

        if df_injected is not None and anomaly_type_column is not None:
            # Build type counts from DataFrame columns (aligned after sorting)
            type_counts: dict[str, dict[str, int]] = {}
            
            for i in range(len(predictions)):
                anomaly_type = df_injected.iloc[i][anomaly_type_column]
                if pd.isna(anomaly_type):
                    continue  # Not an injected row
                
                if anomaly_type not in type_counts:
                    type_counts[anomaly_type] = {"injected": 0, "detected": 0}
                
                type_counts[anomaly_type]["injected"] += 1
                if predictions[i]:
                    type_counts[anomaly_type]["detected"] += 1
            
            # Convert to TypeDetectionMetrics
            detection_by_type: dict[str, TypeDetectionMetrics] = {}
            for anomaly_type, counts in type_counts.items():
                injected = counts["injected"]
                detected = counts["detected"]
                rate = detected / injected if injected > 0 else 0.0
                detection_by_type[anomaly_type] = TypeDetectionMetrics(
                    anomaly_type=anomaly_type,
                    injected=injected,
                    detected=detected,
                    rate=rate,
                )
            return detection_by_type
        
        # No injection columns available - return empty
        return {}

    def _get_time_column(self, df: pd.DataFrame) -> str | None:
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

    def get_evaluator(self, name: str) -> DetectorEvaluator | None:
        """Get an evaluator by name.
        
        Args:
            name: Evaluator name to find.
        
        Returns:
            DetectorEvaluator or None if not found.
        """
        for e in self.evaluators:
            if e.name == name:
                return e
        return None

    def get_cache_stats(self) -> dict[str, dict[str, Any]]:
        """Get cache statistics for all evaluators.
        
        Returns:
            Dict mapping evaluator names to cache stats.
        """
        return {e.name: e.get_cache_stats() for e in self.evaluators}


def create_statistical_evaluators(
    configs: dict[str, StatisticalConfig],
) -> list[DetectorEvaluator]:
    """Create DetectorEvaluators for multiple StatisticalConfig variants.

    This helper function creates evaluators for comparing different statistical
    config settings side-by-side. Each config becomes a separate evaluator with
    its own isolated cache.

    Example:
        # Compare global vs per-competitor configs
        configs = {
            "global": global_config,
            "DK_B2C": dk_segment_config,
            "PROSHOP_DK": proshop_config,
        }
        evaluators = create_statistical_evaluators(configs)
        orchestrator = TestOrchestrator(evaluators)
        results = orchestrator.run_comparison(train_df, test_df, labels)

    Args:
        configs: Dict mapping config names to StatisticalConfig instances.

    Returns:
        List of DetectorEvaluator instances, one per config.
    """
    evaluators = []

    for name, config in configs.items():
        # Create ensemble from config
        ensemble = StatisticalEnsemble.from_config(config)

        # Create evaluator with descriptive name
        evaluator = DetectorEvaluator(
            detector=ensemble,
            name=f"Statistical_{name}",
        )
        evaluators.append(evaluator)

    logger.info(
        "statistical_evaluators_created",
        extra={
            "count": len(evaluators),
            "names": [e.name for e in evaluators],
        },
    )

    return evaluators


def create_expanded_statistical_evaluators(
    *,
    zscore_threshold: float = 3.0,
    robust_threshold: float = 2.0,
    hybrid_weight: float = 0.5,
    iqr_multiplier: float = 1.5,
    change_threshold: float = 0.20,
) -> list[DetectorEvaluator]:
    """Create evaluators for the full expanded statistical comparison roster.

    This helper is intended for local experimentation and smoke testing of the
    evaluation stack. It deliberately does not modify the published manifest-
    driven detector comparison surfaces.

    Returns:
        List of DetectorEvaluator instances for the expanded statistical roster.
    """
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

    return [
        DetectorEvaluator(ZScoreDetector(threshold=zscore_threshold), "Z-score"),
        DetectorEvaluator(ModifiedMADDetector(threshold=robust_threshold), "ModifiedMAD"),
        DetectorEvaluator(ModifiedSNDetector(threshold=robust_threshold), "ModifiedSN"),
        DetectorEvaluator(
            HybridWeightedZScoreDetector(
                threshold=robust_threshold,
                w=hybrid_weight,
            ),
            "HybridWeighted",
        ),
        DetectorEvaluator(HybridMaxZScoreDetector(threshold=robust_threshold), "HybridMax"),
        DetectorEvaluator(HybridAvgZScoreDetector(threshold=robust_threshold), "HybridAvg"),
        DetectorEvaluator(IQRDetector(multiplier=iqr_multiplier), "IQR"),
        DetectorEvaluator(ThresholdDetector(threshold=change_threshold), "Threshold"),
    ]
