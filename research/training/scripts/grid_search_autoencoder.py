#!/usr/bin/env python3
"""Autoencoder hyperparameter grid search for thesis research.

Runs a phased grid search over autoencoder hyperparameters on one dataset,
comparing architecture variants, regularization, and threshold settings.

Phased Approach (reduces search space from 2000+ to ~80 configs):
    Phase 1: Architecture search (hidden_dims × latent_dim) - 16 combos
             Keep top 4 by F1
    Phase 2: Regularization (dropout × learning_rate) - 12 combos
             Keep top 4 by F1
    Phase 3: Threshold tuning (threshold_percentile) - 16 combos
             Select best overall

Usage:
    # Fast search on small dataset (recommended for initial exploration)
    python scripts/grid_search_autoencoder.py --model-filter DK_B2B

    # Full search on larger dataset
    python scripts/grid_search_autoencoder.py --model-filter DK_B2C

    # Resume interrupted search
    python scripts/grid_search_autoencoder.py --model-filter DK_B2B --resume

    # Full Cartesian grid (no phases, slower)
    python scripts/grid_search_autoencoder.py --model-filter DK_B2B --no-phases
"""

import argparse
import csv
from copy import deepcopy
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path (4 levels up from research/training/scripts/)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_script_dir)))
sys.path.insert(0, _project_root)
sys.path.insert(0, _script_dir)  # For importing sibling scripts

from dotenv import load_dotenv

# Reuse data/feature loading AND training from existing scripts
from train_autoencoder import (
    FEATURE_NAMES,
    extract_features_vectorized,
    extract_model_name,
    find_matching_test_file,
    find_parquet_files,
    get_cache_path,
    load_cached_features,
    load_parquet_file,
    save_cached_features,
    train_from_matrix,
)

# Production evaluation components
from compare_detectors import extract_country
from src.anomaly.ml.autoencoder import AutoencoderConfig
from src.anomaly.statistical import AnomalySeverity, AnomalyType
from src.research.artifacts import (
    comparison_result_to_tables,
    create_run_id,
    empty_injected_rows_table,
    empty_predictions_table,
    resolve_git_commit,
    write_evaluation_run,
    write_tuning_sweep,
)
from src.research.evaluation.synthetic import inject_anomalies_to_dataframe
from src.research.evaluation.detector_evaluator import DetectorEvaluator
from src.research.evaluation.test_orchestrator import ComparisonResult, TestOrchestrator
from src.features.temporal import TemporalCacheManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Grid Search Configuration
# =============================================================================

# Phase 1: Architecture
PHASE1_HIDDEN_DIMS = [
    [64, 32],        # Original (simple)
    [128, 64],       # Wider
    [256, 128],      # Even wider
    [128, 64, 32],   # Deeper
]
PHASE1_LATENT_DIM = [4, 8, 16, 32]

# Phase 2: Regularization (applied to top 4 from Phase 1)
PHASE2_DROPOUT = [0.0, 0.1, 0.2]
PHASE2_LEARNING_RATE = [0.0001, 0.001, 0.01]

# Phase 3: Threshold (applied to top 4 from Phase 2)
PHASE3_THRESHOLD_PERCENTILE = [90, 93, 95, 97, 99]

# Fixed parameters
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_DROPOUT = 0.1
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_THRESHOLD_PERCENTILE = 95
DEFAULT_WORKERS = 1  # Parallel workers for training

# Evaluation
INJECTION_RATE = 0.1
RANDOM_STATE = 42

# File suffixes
DEFAULT_FILE_SUFFIX = "_train_mh4"
TEST_NEW_PRICES_SUFFIX = "_test_new_prices_mh4"
TEST_NEW_PRODUCTS_SUFFIX = "_test_new_products_mh4"


@dataclass
class GridRow:
    """One evaluated configuration."""
    
    run_id: str
    phase: int
    hidden_dims: str  # JSON string for CSV compatibility
    latent_dim: int
    dropout: float
    learning_rate: float
    epochs: int
    threshold_percentile: float
    threshold_value: float
    # Metrics on test_new_prices
    precision_new_prices: float
    recall_new_prices: float
    f1_new_prices: float
    # Metrics on test_new_products
    precision_new_products: float
    recall_new_products: float
    f1_new_products: float
    # Combined F1 (average)
    f1_combined: float
    # Metadata
    training_time_sec: float
    dataset_name: str = ""
    n_train: int = 0
    n_eval_prices: int = 0
    n_eval_products: int = 0
    mean_reconstruction_error: float = 0.0
    error: str = ""


@dataclass
class CachedEvaluationData:
    """Pre-computed data for efficient grid search evaluation.
    
    Holds all data that can be computed once and reused across all
    hyperparameter configurations during grid search.
    """
    
    X_train: np.ndarray  # Training features (cached as .npz)
    train_df: pd.DataFrame  # For cache population
    df_test_prices_injected: pd.DataFrame | None  # Pre-injected anomalies
    labels_prices: np.ndarray | None  # Ground truth for prices test
    df_test_products_injected: pd.DataFrame | None
    labels_products: np.ndarray | None
    template_cache: TemporalCacheManager  # Pre-populated, clone for each config
    country: str | None
    dataset_name: str  # e.g., "NO_B2C" for results logging


@dataclass
class CandidateEvaluationResult:
    """Full candidate output including row-level comparison artifacts."""

    row: GridRow
    comparisons: dict[str, ComparisonResult]
    config: dict[str, object]


def _resolve_train_file(args: argparse.Namespace) -> str:
    """Resolve single train Parquet path from CLI."""
    data_path = getattr(args, "data_path", "data/training")
    file_suffix = getattr(args, "file_suffix", DEFAULT_FILE_SUFFIX)
    granularity = getattr(args, "granularity", "country_segment")
    model_filter = getattr(args, "model_filter", None)
    
    files = find_parquet_files(data_path, granularity, file_suffix)
    
    if model_filter:
        files = [f for f in files if model_filter in extract_model_name(f)]
    
    if not files:
        filter_msg = f", model_filter={model_filter!r}" if model_filter else ""
        raise FileNotFoundError(
            f"No Parquet files in {data_path} (granularity={granularity}, "
            f"suffix={file_suffix!r}{filter_msg})"
        )
    return files[0]


def _resolve_test_files(train_file: str, data_path: str) -> tuple[str | None, str | None]:
    """Find test_new_prices and test_new_products files matching train file."""
    test_prices = find_matching_test_file(train_file, TEST_NEW_PRICES_SUFFIX, data_path)
    test_products = find_matching_test_file(train_file, TEST_NEW_PRODUCTS_SUFFIX, data_path)
    return test_prices, test_products


def load_or_extract_features(filepath: str, use_cache: bool = True) -> np.ndarray:
    """Load cached features or extract from parquet."""
    cache_path = get_cache_path(filepath)
    
    if use_cache:
        X = load_cached_features(cache_path)
        if X is not None:
            return X
    
    df = load_parquet_file(filepath)
    X = extract_features_vectorized(df)
    
    if use_cache:
        save_cached_features(cache_path, X)
    
    return X


def _get_time_column(df: pd.DataFrame) -> str | None:
    """Get the time column name from DataFrame."""
    for col in ["first_seen_at", "observed_at", "timestamp", "created_at"]:
        if col in df.columns:
            return col
    return None


def _populate_cache_from_df(cache: TemporalCacheManager, df: pd.DataFrame) -> None:
    """Populate a TemporalCacheManager from DataFrame.
    
    This replicates the logic from DetectorEvaluator._populate_from_df()
    but operates directly on a TemporalCacheManager instance.
    
    Args:
        cache: TemporalCacheManager to populate.
        df: DataFrame with historical price data.
    """
    if df.empty:
        return
    
    # Determine time column
    time_col = _get_time_column(df)
    
    # Sort by time for correct chronological order
    if time_col:
        df = df.sort_values(time_col)
    
    # Group by product and competitor
    grouped = df.groupby(["product_id", "competitor_id"])
    
    for (product_id, competitor_id), group in grouped:
        # Sort group by time
        if time_col:
            group = group.sort_values(time_col)
        
        # Extract prices in chronological order
        prices = group["price"].dropna().tolist()
        
        if prices:
            # Create cache entry with price history
            cache._create_entry(
                product_id=str(product_id),
                competitor_id=str(competitor_id),
                prices=prices,
            )


def setup_cached_evaluation(
    train_file: str,
    test_prices_file: str | None,
    test_products_file: str | None,
    model_name: str,
    use_cache: bool = True,
) -> CachedEvaluationData:
    """One-time setup: load data, inject anomalies, populate template cache.
    
    This function pre-computes all expensive operations that can be reused
    across all hyperparameter configurations during grid search.
    
    Args:
        train_file: Path to training parquet file.
        test_prices_file: Path to test_new_prices parquet file (optional).
        test_products_file: Path to test_new_products parquet file (optional).
        model_name: Model name for country extraction.
        use_cache: Whether to use cached features.
        
    Returns:
        CachedEvaluationData with all pre-computed data.
    """
    logger.info("Setting up cached evaluation data...")
    
    # 1. Load training features (cached as .npz)
    logger.info(f"  Loading training features from {os.path.basename(train_file)}")
    X_train = load_or_extract_features(train_file, use_cache)
    logger.info(f"    {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    
    # 2. Load DataFrames
    logger.info("  Loading DataFrames...")
    train_df = load_parquet_file(train_file)
    df_test_prices = load_parquet_file(test_prices_file) if test_prices_file else None
    df_test_products = load_parquet_file(test_products_file) if test_products_file else None
    
    # 3. Pre-inject anomalies (once per test set) using unified injection (6 anomaly types)
    logger.info("  Injecting anomalies into test sets (6 types: SPIKE, DROP, ZERO, NEGATIVE, EXTREME_OUTLIER, DECIMAL_SHIFT)...")
    df_prices_injected = None
    labels_prices = None
    df_products_injected = None
    labels_products = None
    
    if df_test_prices is not None:
        df_prices_injected, labels_prices, _ = inject_anomalies_to_dataframe(
            df_test_prices, injection_rate=0.1, seed=42
        )
        logger.info(f"    test_new_prices: {len(df_prices_injected):,} rows, {labels_prices.sum():,} anomalies")
    
    if df_test_products is not None:
        df_products_injected, labels_products, _ = inject_anomalies_to_dataframe(
            df_test_products, injection_rate=0.1, seed=42
        )
        logger.info(f"    test_new_products: {len(df_products_injected):,} rows, {labels_products.sum():,} anomalies")
    
    # 4. Create and populate template cache (once)
    logger.info("  Populating template cache from training data...")
    template_cache = TemporalCacheManager()
    _populate_cache_from_df(template_cache, train_df)
    cache_stats = template_cache.get_stats()
    logger.info(f"    Cache populated: {cache_stats.get('total_products', 0):,} entries")
    
    # Extract country from model name
    country = extract_country(model_name)
    logger.info(f"  Country: {country}")
    
    return CachedEvaluationData(
        X_train=X_train,
        train_df=train_df,
        df_test_prices_injected=df_prices_injected,
        labels_prices=labels_prices,
        df_test_products_injected=df_products_injected,
        labels_products=labels_products,
        template_cache=template_cache,
        country=country,
        dataset_name=model_name,
    )


def _train_config_worker(
    worker_args: dict,
) -> dict:
    """Worker function for parallel training.
    
    Runs in a subprocess - receives serializable data, returns serializable results.
    PyTorch model is created fresh in each worker to avoid serialization issues.
    
    Args:
        worker_args: Dict with all config and data needed for training/eval.
        
    Returns:
        Dict with GridRow fields (serializable).
    """
    import time
    
    import numpy as np
    
    # Import inside worker to ensure fresh PyTorch context per process
    from train_autoencoder import train_from_matrix
    from src.anomaly.ml.autoencoder import AutoencoderConfig
    
    # Unpack args
    X_train = worker_args["X_train"]
    hidden_dims = worker_args["hidden_dims"]
    latent_dim = worker_args["latent_dim"]
    dropout = worker_args["dropout"]
    learning_rate = worker_args["learning_rate"]
    epochs = worker_args["epochs"]
    threshold_percentile = worker_args["threshold_percentile"]
    batch_size = worker_args.get("batch_size", DEFAULT_BATCH_SIZE)
    run_id = worker_args["run_id"]
    phase = worker_args["phase"]
    log_interval = worker_args.get("log_interval", 5)
    dataset_name = worker_args.get("dataset_name", "")
    
    start_time = time.perf_counter()
    
    try:
        # Train autoencoder
        config = AutoencoderConfig(
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            dropout=dropout,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
        )
        detector, _, mean_error, train_errors = train_from_matrix(
            X_train, config, log_interval=log_interval
        )
        
        train_elapsed = time.perf_counter() - start_time
        
        # Set threshold at specified percentile
        threshold = float(np.percentile(train_errors, threshold_percentile))
        detector._threshold = threshold
        
        elapsed = time.perf_counter() - start_time
        
        # Return results - evaluation will happen in main process
        return {
            "success": True,
            "run_id": run_id,
            "phase": phase,
            "hidden_dims": hidden_dims,
            "latent_dim": latent_dim,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "threshold_percentile": threshold_percentile,
            "threshold_value": threshold,
            "training_time_sec": elapsed,
            "mean_reconstruction_error": mean_error,
            "dataset_name": dataset_name,
            "n_train": len(X_train),
            "train_errors": train_errors.tolist(),  # For threshold sweep
            # Detector state for evaluation
            "detector_mean": detector._mean.tolist(),
            "detector_std": detector._std.tolist(),
            "detector_config": {
                "input_dim": detector.config.input_dim,
                "hidden_dims": detector.config.hidden_dims,
                "latent_dim": detector.config.latent_dim,
                "dropout": detector.config.dropout,
                "learning_rate": detector.config.learning_rate,
                "epochs": detector.config.epochs,
                "batch_size": detector.config.batch_size,
            },
            "encoder_state": detector._model.encoder.state_dict(),
            "decoder_state": detector._model.decoder.state_dict(),
        }
        
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        return {
            "success": False,
            "run_id": run_id,
            "phase": phase,
            "hidden_dims": hidden_dims,
            "latent_dim": latent_dim,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "threshold_percentile": threshold_percentile,
            "training_time_sec": elapsed,
            "dataset_name": dataset_name,
            "error": str(e),
        }


def _restore_detector_from_worker_result(worker_result: dict):
    """Restore a trained detector from worker result for evaluation.
    
    Args:
        worker_result: Dict returned by _train_config_worker.
        
    Returns:
        Trained AutoencoderDetector ready for evaluation.
    """
    import torch
    
    from src.anomaly.ml.autoencoder import AutoencoderConfig, AutoencoderDetector, AutoencoderModel
    
    config = AutoencoderConfig(**worker_result["detector_config"])
    
    detector = AutoencoderDetector(config)
    detector._model = AutoencoderModel(config)
    detector._model.encoder.load_state_dict(worker_result["encoder_state"])
    detector._model.decoder.load_state_dict(worker_result["decoder_state"])
    detector._model.encoder.eval()
    detector._model.decoder.eval()
    
    detector._mean = np.array(worker_result["detector_mean"])
    detector._std = np.array(worker_result["detector_std"])
    detector._threshold = worker_result["threshold_value"]
    detector._is_fitted = True
    
    return detector


def run_single_config(
    cached_data: CachedEvaluationData,
    hidden_dims: list[int],
    latent_dim: int,
    dropout: float,
    learning_rate: float,
    epochs: int,
    threshold_percentile: float,
    run_id: str,
    phase: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    log_interval: int = 5,
) -> CandidateEvaluationResult:
    """Train and evaluate a single configuration using production evaluation.
    
    Uses train_from_matrix() for training and TestOrchestrator for evaluation,
    with pre-heated cache cloning for efficiency.
    
    Args:
        cached_data: Pre-computed evaluation data (injected test sets, template cache).
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent space dimension.
        dropout: Dropout rate.
        learning_rate: Learning rate.
        epochs: Number of training epochs.
        threshold_percentile: Percentile for threshold computation.
        run_id: Unique identifier for this run.
        phase: Phase number (1, 2, or 3).
        log_interval: Log training progress every N epochs (default: 5).
        
    Returns:
        CandidateEvaluationResult with metrics and row-level artifacts.
    """
    start_time = time.perf_counter()
    config_values = {
        "phase": phase,
        "hidden_dims": list(hidden_dims),
        "latent_dim": latent_dim,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "threshold_percentile": threshold_percentile,
    }
    
    try:
        # Train with production function
        logger.info(f"  Training autoencoder ({cached_data.X_train.shape[0]:,} samples, {epochs} epochs)...")
        config = AutoencoderConfig(
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            dropout=dropout,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
        )
        detector, _, mean_error, train_errors = train_from_matrix(
            cached_data.X_train, config, log_interval=log_interval
        )
        
        train_elapsed = time.perf_counter() - start_time
        logger.info(f"  Training completed in {train_elapsed:.1f}s")
        
        # Set threshold at specified percentile
        threshold = float(np.percentile(train_errors, threshold_percentile))
        detector._threshold = threshold
        
        # --- Evaluate on test_new_prices ---
        logger.info("  Evaluating on test_new_prices...")
        eval_start = time.perf_counter()
        metrics_prices, comparison_prices = _evaluate_with_preheated_cache(
            detector, cached_data, "prices"
        )
        eval_prices_time = time.perf_counter() - eval_start
        logger.info(f"    test_new_prices: F1={metrics_prices['f1']:.1%} ({eval_prices_time:.1f}s)")
        
        # --- Evaluate on test_new_products ---
        logger.info("  Evaluating on test_new_products...")
        eval_start = time.perf_counter()
        metrics_products, comparison_products = _evaluate_with_preheated_cache(
            detector, cached_data, "products"
        )
        eval_products_time = time.perf_counter() - eval_start
        logger.info(f"    test_new_products: F1={metrics_products['f1']:.1%} ({eval_products_time:.1f}s)")
        
        # Combined F1 (average of both test sets)
        f1_combined = (metrics_prices["f1"] + metrics_products["f1"]) / 2
        
        elapsed = time.perf_counter() - start_time
        logger.info(f"  Config {run_id} complete: F1_combined={f1_combined:.1%} (total: {elapsed:.1f}s)")
        
        return CandidateEvaluationResult(
            row=GridRow(
                run_id=run_id,
                phase=phase,
                hidden_dims=json.dumps(hidden_dims),
                latent_dim=latent_dim,
                dropout=dropout,
                learning_rate=learning_rate,
                epochs=epochs,
                threshold_percentile=threshold_percentile,
                threshold_value=threshold,
                precision_new_prices=metrics_prices["precision"],
                recall_new_prices=metrics_prices["recall"],
                f1_new_prices=metrics_prices["f1"],
                precision_new_products=metrics_products["precision"],
                recall_new_products=metrics_products["recall"],
                f1_new_products=metrics_products["f1"],
                f1_combined=f1_combined,
                training_time_sec=elapsed,
                mean_reconstruction_error=mean_error,
                dataset_name=cached_data.dataset_name,
                n_train=len(cached_data.X_train),
                n_eval_prices=len(cached_data.df_test_prices_injected) if cached_data.df_test_prices_injected is not None else 0,
                n_eval_products=len(cached_data.df_test_products_injected) if cached_data.df_test_products_injected is not None else 0,
            ),
            comparisons={
                "new_prices": comparison_prices,
                "new_products": comparison_products,
            },
            config={**config_values, "threshold_value": threshold},
        )
        
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        logger.error(f"Config {run_id} failed: {e}")
        return CandidateEvaluationResult(
            row=GridRow(
                run_id=run_id,
                phase=phase,
                hidden_dims=json.dumps(hidden_dims),
                latent_dim=latent_dim,
                dropout=dropout,
                learning_rate=learning_rate,
                epochs=epochs,
                threshold_percentile=threshold_percentile,
                threshold_value=0,
                precision_new_prices=0,
                recall_new_prices=0,
                f1_new_prices=0,
                precision_new_products=0,
                recall_new_products=0,
                f1_new_products=0,
                f1_combined=0,
                training_time_sec=elapsed,
                dataset_name=cached_data.dataset_name,
                error=str(e),
            ),
            comparisons={},
            config=config_values,
        )


def _evaluate_with_preheated_cache(
    detector,
    cached_data: CachedEvaluationData,
    test_set: str,
) -> tuple[dict[str, float], ComparisonResult | None]:
    """Evaluate detector on a test set using pre-heated cache.
    
    Args:
        detector: Trained autoencoder detector.
        cached_data: Pre-computed evaluation data.
        test_set: Which test set to use ("prices" or "products").
        
    Returns:
        Tuple of metrics dict and ComparisonResult.
    """
    if test_set == "prices":
        df_injected = cached_data.df_test_prices_injected
        labels = cached_data.labels_prices
    else:
        df_injected = cached_data.df_test_products_injected
        labels = cached_data.labels_products
    
    if df_injected is None or labels is None:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}, None
    
    # Create evaluator and clone template cache
    evaluator = DetectorEvaluator(detector, name="Autoencoder")
    evaluator.temporal_cache.copy_from(cached_data.template_cache)
    
    # Run evaluation with skip_cache_setup (cache already populated)
    orchestrator = TestOrchestrator([evaluator])
    comparison = orchestrator.run_comparison_with_details(
        train_df=None,  # Ignored when skip_cache_setup=True
        test_df=df_injected,
        labels=labels,
        country=cached_data.country,
        skip_cache_setup=True,
    )

    result = comparison.metrics["Autoencoder"]
    return (
        {
            "precision": result.precision,
            "recall": result.recall,
            "f1": result.f1,
        },
        comparison,
    )


def run_configs_parallel(
    cached_data: CachedEvaluationData,
    configs: list[dict],
    max_workers: int = 2,
    completed_ids: set[str] | None = None,
) -> list[CandidateEvaluationResult]:
    """Run multiple configurations in parallel.
    
    Training happens in parallel worker processes, evaluation happens
    sequentially in the main process (to reuse cached data efficiently).
    
    Args:
        cached_data: Pre-computed evaluation data.
        configs: List of config dicts with keys:
            - hidden_dims, latent_dim, dropout, learning_rate
            - epochs, threshold_percentile, run_id, phase
        max_workers: Maximum parallel training processes.
        completed_ids: Set of already-completed run_ids to skip.
        
    Returns:
        List of CandidateEvaluationResult values.
    """
    if completed_ids is None:
        completed_ids = set()
    
    # Filter out already-completed configs
    pending_configs = [c for c in configs if c["run_id"] not in completed_ids]
    
    if not pending_configs:
        logger.info("All configs already completed")
        return []
    
    logger.info(f"Running {len(pending_configs)} configs with {max_workers} parallel workers")
    
    results: list[CandidateEvaluationResult] = []
    
    # Prepare worker args - include X_train for each worker
    worker_args_list = []
    for config in pending_configs:
        worker_args = {
            "X_train": cached_data.X_train,  # Will be copied to each process
            "hidden_dims": config["hidden_dims"],
            "latent_dim": config["latent_dim"],
            "dropout": config["dropout"],
            "learning_rate": config["learning_rate"],
            "epochs": config["epochs"],
            "batch_size": config.get("batch_size", DEFAULT_BATCH_SIZE),
            "threshold_percentile": config["threshold_percentile"],
            "run_id": config["run_id"],
            "phase": config["phase"],
            "log_interval": config.get("log_interval", 5),
            "dataset_name": cached_data.dataset_name,
        }
        worker_args_list.append(worker_args)
    
    # Use 'spawn' context for clean process isolation (safer with PyTorch)
    ctx = get_context("spawn")
    
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        # Submit all training jobs
        future_to_config = {
            executor.submit(_train_config_worker, args): args["run_id"]
            for args in worker_args_list
        }
        
        # Process results as they complete
        for future in as_completed(future_to_config):
            run_id = future_to_config[future]
            
            try:
                worker_result = future.result()
            except Exception as e:
                logger.error(f"Config {run_id} worker failed: {e}")
                results.append(
                    CandidateEvaluationResult(
                        row=GridRow(
                            run_id=run_id,
                            phase=0,
                            hidden_dims="[]",
                            latent_dim=0,
                            dropout=0,
                            learning_rate=0,
                            epochs=0,
                            threshold_percentile=0,
                            threshold_value=0,
                            precision_new_prices=0,
                            recall_new_prices=0,
                            f1_new_prices=0,
                            precision_new_products=0,
                            recall_new_products=0,
                            f1_new_products=0,
                            f1_combined=0,
                            training_time_sec=0,
                            error=str(e),
                        ),
                        comparisons={},
                        config={},
                    )
                )
                continue
            
            if not worker_result.get("success", False):
                logger.error(f"Config {run_id} failed: {worker_result.get('error', 'unknown')}")
                results.append(
                    CandidateEvaluationResult(
                        row=GridRow(
                            run_id=run_id,
                            phase=worker_result.get("phase", 0),
                            hidden_dims=json.dumps(worker_result.get("hidden_dims", [])),
                            latent_dim=worker_result.get("latent_dim", 0),
                            dropout=worker_result.get("dropout", 0),
                            learning_rate=worker_result.get("learning_rate", 0),
                            epochs=worker_result.get("epochs", 0),
                            threshold_percentile=worker_result.get("threshold_percentile", 0),
                            threshold_value=0,
                            precision_new_prices=0,
                            recall_new_prices=0,
                            f1_new_prices=0,
                            precision_new_products=0,
                            recall_new_products=0,
                            f1_new_products=0,
                            f1_combined=0,
                            training_time_sec=worker_result.get("training_time_sec", 0),
                            dataset_name=worker_result.get("dataset_name", ""),
                            error=worker_result.get("error", ""),
                        ),
                        comparisons={},
                        config={
                            "phase": worker_result.get("phase", 0),
                            "hidden_dims": worker_result.get("hidden_dims", []),
                            "latent_dim": worker_result.get("latent_dim", 0),
                            "dropout": worker_result.get("dropout", 0),
                            "learning_rate": worker_result.get("learning_rate", 0),
                            "epochs": worker_result.get("epochs", 0),
                            "batch_size": worker_result.get("batch_size", DEFAULT_BATCH_SIZE),
                            "threshold_percentile": worker_result.get("threshold_percentile", 0),
                        },
                    )
                )
                continue
            
            # Training succeeded - now evaluate in main process
            logger.info(f"Config {run_id} trained in {worker_result['training_time_sec']:.1f}s, evaluating...")
            
            try:
                # Restore detector from worker result
                detector = _restore_detector_from_worker_result(worker_result)
                
                # Evaluate on test sets
                eval_start = time.perf_counter()
                metrics_prices, comparison_prices = _evaluate_with_preheated_cache(detector, cached_data, "prices")
                eval_prices_time = time.perf_counter() - eval_start
                logger.info(f"  {run_id} test_new_prices: F1={metrics_prices['f1']:.1%} ({eval_prices_time:.1f}s)")
                
                eval_start = time.perf_counter()
                metrics_products, comparison_products = _evaluate_with_preheated_cache(detector, cached_data, "products")
                eval_products_time = time.perf_counter() - eval_start
                logger.info(f"  {run_id} test_new_products: F1={metrics_products['f1']:.1%} ({eval_products_time:.1f}s)")
                
                f1_combined = (metrics_prices["f1"] + metrics_products["f1"]) / 2
                total_time = worker_result["training_time_sec"] + eval_prices_time + eval_products_time
                
                logger.info(f"Config {run_id} complete: F1_combined={f1_combined:.1%} (total: {total_time:.1f}s)")
                
                results.append(
                    CandidateEvaluationResult(
                        row=GridRow(
                            run_id=run_id,
                            phase=worker_result["phase"],
                            hidden_dims=json.dumps(worker_result["hidden_dims"]),
                            latent_dim=worker_result["latent_dim"],
                            dropout=worker_result["dropout"],
                            learning_rate=worker_result["learning_rate"],
                            epochs=worker_result["epochs"],
                            threshold_percentile=worker_result["threshold_percentile"],
                            threshold_value=worker_result["threshold_value"],
                            precision_new_prices=metrics_prices["precision"],
                            recall_new_prices=metrics_prices["recall"],
                            f1_new_prices=metrics_prices["f1"],
                            precision_new_products=metrics_products["precision"],
                            recall_new_products=metrics_products["recall"],
                            f1_new_products=metrics_products["f1"],
                            f1_combined=f1_combined,
                            training_time_sec=total_time,
                            mean_reconstruction_error=worker_result["mean_reconstruction_error"],
                            dataset_name=worker_result["dataset_name"],
                            n_train=worker_result["n_train"],
                            n_eval_prices=len(cached_data.df_test_prices_injected) if cached_data.df_test_prices_injected is not None else 0,
                            n_eval_products=len(cached_data.df_test_products_injected) if cached_data.df_test_products_injected is not None else 0,
                        ),
                        comparisons={
                            "new_prices": comparison_prices,
                            "new_products": comparison_products,
                        },
                        config={
                            "phase": worker_result["phase"],
                            "hidden_dims": worker_result["hidden_dims"],
                            "latent_dim": worker_result["latent_dim"],
                            "dropout": worker_result["dropout"],
                            "learning_rate": worker_result["learning_rate"],
                            "epochs": worker_result["epochs"],
                            "batch_size": worker_result.get("batch_size", DEFAULT_BATCH_SIZE),
                            "threshold_percentile": worker_result["threshold_percentile"],
                            "threshold_value": worker_result["threshold_value"],
                        },
                    )
                )
                
            except Exception as e:
                logger.error(f"Config {run_id} evaluation failed: {e}")
                results.append(
                    CandidateEvaluationResult(
                        row=GridRow(
                            run_id=run_id,
                            phase=worker_result["phase"],
                            hidden_dims=json.dumps(worker_result["hidden_dims"]),
                            latent_dim=worker_result["latent_dim"],
                            dropout=worker_result["dropout"],
                            learning_rate=worker_result["learning_rate"],
                            epochs=worker_result["epochs"],
                            threshold_percentile=worker_result["threshold_percentile"],
                            threshold_value=worker_result.get("threshold_value", 0),
                            precision_new_prices=0,
                            recall_new_prices=0,
                            f1_new_prices=0,
                            precision_new_products=0,
                            recall_new_products=0,
                            f1_new_products=0,
                            f1_combined=0,
                            training_time_sec=worker_result["training_time_sec"],
                            mean_reconstruction_error=worker_result.get("mean_reconstruction_error", 0),
                            dataset_name=worker_result["dataset_name"],
                            n_train=worker_result["n_train"],
                            error=f"Evaluation failed: {e}",
                        ),
                        comparisons={},
                        config={
                            "phase": worker_result["phase"],
                            "hidden_dims": worker_result["hidden_dims"],
                            "latent_dim": worker_result["latent_dim"],
                            "dropout": worker_result["dropout"],
                            "learning_rate": worker_result["learning_rate"],
                            "epochs": worker_result["epochs"],
                            "batch_size": worker_result.get("batch_size", DEFAULT_BATCH_SIZE),
                            "threshold_percentile": worker_result["threshold_percentile"],
                            "threshold_value": worker_result.get("threshold_value", 0),
                        },
                    )
                )
    
    return results


def run_threshold_sweep(
    cached_data: CachedEvaluationData,
    best_config: dict,
    threshold_percentiles: list[float],
    base_run_id: str,
    log_interval: int = 5,
) -> list[CandidateEvaluationResult]:
    """Efficiently sweep thresholds without re-evaluation.
    
    Trains once with the best config, evaluates once per test set to get raw
    reconstruction errors, then computes metrics at multiple thresholds using
    simple array operations (no detector re-runs).
    
    Args:
        cached_data: Pre-computed evaluation data.
        best_config: Best hyperparameters from Phase 2 (hidden_dims, latent_dim, etc.).
        threshold_percentiles: List of percentiles to sweep.
        base_run_id: Base run ID for numbering (e.g., "p3").
        log_interval: Log training progress every N epochs.
        
    Returns:
        List of CandidateEvaluationResult, one per threshold percentile.
    """
    logger.info(f"Threshold sweep: {len(threshold_percentiles)} percentiles")
    start_time = time.perf_counter()
    
    # Train once with best config
    logger.info(f"  Training autoencoder ({cached_data.X_train.shape[0]:,} samples, {best_config['epochs']} epochs)...")
    config = AutoencoderConfig(
        hidden_dims=best_config["hidden_dims"],
        latent_dim=best_config["latent_dim"],
        dropout=best_config["dropout"],
        learning_rate=best_config["learning_rate"],
        epochs=best_config["epochs"],
    )
    detector, _, mean_error, train_errors = train_from_matrix(
        cached_data.X_train, config, log_interval=log_interval
    )
    
    training_time = time.perf_counter() - start_time
    logger.info(f"  Training done in {training_time:.1f}s")
    
    base_metrics_prices, base_comparison_prices = _evaluate_with_preheated_cache(
        detector, cached_data, "prices"
    )
    logger.info(f"  test_new_prices base evaluation: F1={base_metrics_prices['f1']:.1%}")
    base_metrics_products, base_comparison_products = _evaluate_with_preheated_cache(
        detector, cached_data, "products"
    )
    logger.info(f"  test_new_products base evaluation: F1={base_metrics_products['f1']:.1%}")
    
    eval_time = time.perf_counter() - start_time - training_time
    logger.info(f"  Evaluation done in {eval_time:.1f}s")

    def _extract_errors_and_labels(
        comparison: ComparisonResult | None,
        split_name: str,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if comparison is None:
            return None, None

        raw_results = comparison.raw_results.get("Autoencoder", [])
        errors = np.array(
            [float((result.details or {}).get("reconstruction_error", 0.0)) for result in raw_results],
            dtype=float,
        )
        logger.info("  %s: %s errors extracted", split_name, f"{len(errors):,}")
        return errors, np.asarray(comparison.labels).astype(bool)

    errors_prices, labels_prices = _extract_errors_and_labels(
        base_comparison_prices,
        "test_new_prices",
    )
    errors_products, labels_products = _extract_errors_and_labels(
        base_comparison_products,
        "test_new_products",
    )
    
    # Compute metrics at each threshold percentile (fast array ops, no re-eval!)
    results: list[CandidateEvaluationResult] = []
    for idx, pct in enumerate(threshold_percentiles, 1):
        threshold = float(np.percentile(train_errors, pct))
        
        # Metrics for prices
        if errors_prices is not None:
            pred_prices = errors_prices > threshold
            p_prices, r_prices, f1_prices = _compute_prf1(pred_prices, labels_prices)
        else:
            p_prices, r_prices, f1_prices = 0.0, 0.0, 0.0
        
        # Metrics for products
        if errors_products is not None:
            pred_products = errors_products > threshold
            p_products, r_products, f1_products = _compute_prf1(pred_products, labels_products)
        else:
            p_products, r_products, f1_products = 0.0, 0.0, 0.0
        
        # Combined F1
        f1_combined = (f1_prices + f1_products) / 2
        
        run_id = f"{base_run_id}_{idx:02d}"
        
        comparisons: dict[str, ComparisonResult] = {}
        if base_comparison_prices is not None:
            comparisons["new_prices"] = _build_threshold_comparison(
                detector=detector,
                base_comparison=base_comparison_prices,
                threshold=threshold,
            )
        if base_comparison_products is not None:
            comparisons["new_products"] = _build_threshold_comparison(
                detector=detector,
                base_comparison=base_comparison_products,
                threshold=threshold,
            )

        results.append(
            CandidateEvaluationResult(
                row=GridRow(
                    run_id=run_id,
                    phase=3,
                    hidden_dims=json.dumps(best_config["hidden_dims"]),
                    latent_dim=best_config["latent_dim"],
                    dropout=best_config["dropout"],
                    learning_rate=best_config["learning_rate"],
                    epochs=best_config["epochs"],
                    threshold_percentile=pct,
                    threshold_value=threshold,
                    precision_new_prices=p_prices,
                    recall_new_prices=r_prices,
                    f1_new_prices=f1_prices,
                    precision_new_products=p_products,
                    recall_new_products=r_products,
                    f1_new_products=f1_products,
                    f1_combined=f1_combined,
                    training_time_sec=training_time / len(threshold_percentiles),  # Amortized
                    mean_reconstruction_error=mean_error,
                    dataset_name=cached_data.dataset_name,
                    n_train=len(cached_data.X_train),
                    n_eval_prices=len(errors_prices) if errors_prices is not None else 0,
                    n_eval_products=len(errors_products) if errors_products is not None else 0,
                ),
                comparisons=comparisons,
                config={
                    "phase": 3,
                    "hidden_dims": best_config["hidden_dims"],
                    "latent_dim": best_config["latent_dim"],
                    "dropout": best_config["dropout"],
                    "learning_rate": best_config["learning_rate"],
                    "epochs": best_config["epochs"],
                    "threshold_percentile": pct,
                    "threshold_value": threshold,
                },
            )
        )
        
        logger.info(f"  {run_id}: thresh={pct}% -> F1_combined={f1_combined:.1%}")
    
    total_time = time.perf_counter() - start_time
    logger.info(f"Threshold sweep complete in {total_time:.1f}s")
    
    return results


def _severity_from_error(error: float, threshold: float) -> AnomalySeverity | None:
    """Mirror the autoencoder detector severity thresholds."""
    if error <= threshold:
        return None
    if error > threshold * 3:
        return AnomalySeverity.CRITICAL
    if error > threshold * 2:
        return AnomalySeverity.HIGH
    if error > threshold * 1.5:
        return AnomalySeverity.MEDIUM
    return AnomalySeverity.LOW


def _build_threshold_comparison(
    *,
    detector,
    base_comparison: ComparisonResult,
    threshold: float,
) -> ComparisonResult:
    """Materialize a ComparisonResult for a threshold candidate from cached raw errors."""
    updated_results = []
    for result in base_comparison.raw_results.get("Autoencoder", []):
        cloned = deepcopy(result)
        details = dict(cloned.details or {})
        error = float(details.get("reconstruction_error", 0.0))
        is_anomaly = error > threshold
        details["threshold"] = threshold

        cloned.is_anomaly = is_anomaly
        cloned.anomaly_score = detector.normalize_score(error, threshold)
        cloned.details = details
        if is_anomaly:
            cloned.anomaly_types = [AnomalyType.PRICE_ZSCORE]
            cloned.severity = _severity_from_error(error, threshold)
        else:
            cloned.anomaly_types = []
            cloned.severity = None
        updated_results.append(cloned)

    return ComparisonResult(
        metrics={},
        raw_results={"Autoencoder": updated_results},
        observation_counts=np.asarray(base_comparison.observation_counts).copy(),
        labels=np.asarray(base_comparison.labels).copy(),
        df_sorted=base_comparison.df_sorted.copy() if base_comparison.df_sorted is not None else None,
    )


def _compute_prf1(predictions: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    """Compute precision, recall, F1 from boolean arrays."""
    predictions = np.asarray(predictions).astype(bool)
    labels = np.asarray(labels).astype(bool)
    
    tp = np.sum(predictions & labels)
    fp = np.sum(predictions & ~labels)
    fn = np.sum(~predictions & labels)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


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
                "detector_family": "autoencoder",
                "dataset_name": dataset_name,
                "dataset_granularity": dataset_granularity,
                "status": "error" if row.error else "ok",
                "error": row.error,
                "phase": row.phase,
                "hidden_dims": row.hidden_dims,
                "latent_dim": row.latent_dim,
                "dropout": row.dropout,
                "learning_rate": row.learning_rate,
                "epochs": row.epochs,
                "threshold_percentile": row.threshold_percentile,
                "threshold_value": row.threshold_value,
                "training_time_sec": row.training_time_sec,
                "n_train": row.n_train,
                "n_eval_prices": row.n_eval_prices,
                "n_eval_products": row.n_eval_products,
                "mean_reconstruction_error": row.mean_reconstruction_error,
                "combined_precision": (row.precision_new_prices + row.precision_new_products) / 2,
                "combined_recall": (row.recall_new_prices + row.recall_new_products) / 2,
                "combined_f1": row.f1_combined,
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
        hidden_dims = record.get("hidden_dims", "[]")
        error = record.get("error", "")
        rows.append(
            GridRow(
                run_id=str(record.get("run_id", "")),
                phase=int(record.get("phase", 0)),
                hidden_dims=str(hidden_dims) if pd.notna(hidden_dims) else "[]",
                latent_dim=int(record.get("latent_dim", 0)),
                dropout=float(record.get("dropout", 0)),
                learning_rate=float(record.get("learning_rate", 0)),
                epochs=int(record.get("epochs", 0)),
                threshold_percentile=float(record.get("threshold_percentile", 0)),
                threshold_value=float(record.get("threshold_value", 0)),
                precision_new_prices=float(record.get("new_prices_precision", 0)),
                recall_new_prices=float(record.get("new_prices_recall", 0)),
                f1_new_prices=float(record.get("new_prices_f1", 0)),
                precision_new_products=float(record.get("new_products_precision", 0)),
                recall_new_products=float(record.get("new_products_recall", 0)),
                f1_new_products=float(record.get("new_products_f1", 0)),
                f1_combined=float(record.get("combined_f1", 0)),
                training_time_sec=float(record.get("training_time_sec", 0)),
                dataset_name=str(record.get("dataset_name", "")),
                n_train=int(record.get("n_train", 0)),
                n_eval_prices=int(record.get("n_eval_prices", 0)),
                n_eval_products=int(record.get("n_eval_products", 0)),
                mean_reconstruction_error=float(record.get("mean_reconstruction_error", 0)),
                error=str(error) if pd.notna(error) else "",
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
    test_file_prices: str | None,
    test_file_products: str | None,
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

    expected_splits: list[str] = []
    if test_file_prices:
        expected_splits.append("new_prices")
    if test_file_products:
        expected_splits.append("new_products")
    for split_name in expected_splits:
        split_artifacts.setdefault(
            split_name,
            (empty_injected_rows_table(), empty_predictions_table()),
        )

    source_dataset_paths = [train_file]
    if test_file_prices:
        source_dataset_paths.append(test_file_prices)
    if test_file_products:
        source_dataset_paths.append(test_file_products)

    run_metadata = {
        "schema_version": "phase2.v1",
        "experiment_family": "tuning",
        "run_id": candidate_id,
        "candidate_id": candidate_id,
        "sweep_id": sweep_id,
        "source_dataset_paths": source_dataset_paths,
        "dataset_names": [dataset_name],
        "dataset_granularity": dataset_granularity,
        "dataset_splits": sorted(split_artifacts.keys()),
        "random_seeds": {"training_seed": RANDOM_STATE, "injection_seed": RANDOM_STATE},
        "injection_config": {"injection_rate": INJECTION_RATE},
        "detector_identifiers": ["Autoencoder"],
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
    test_file_prices: str | None,
    test_file_products: str | None,
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


def write_results_csv(results: list[GridRow], output_path: str) -> None:
    """Write results to CSV."""
    if not results:
        return
    
    fieldnames = [
        "run_id", "phase", "hidden_dims", "latent_dim", "dropout", "learning_rate",
        "epochs", "threshold_percentile", "threshold_value",
        "precision_new_prices", "recall_new_prices", "f1_new_prices",
        "precision_new_products", "recall_new_products", "f1_new_products",
        "f1_combined", "training_time_sec", "dataset_name",
        "n_train", "n_eval_prices", "n_eval_products",
        "mean_reconstruction_error", "error",
    ]
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({
                "run_id": row.run_id,
                "phase": row.phase,
                "hidden_dims": row.hidden_dims,
                "latent_dim": row.latent_dim,
                "dropout": row.dropout,
                "learning_rate": row.learning_rate,
                "epochs": row.epochs,
                "threshold_percentile": row.threshold_percentile,
                "threshold_value": row.threshold_value,
                "precision_new_prices": f"{row.precision_new_prices:.4f}",
                "recall_new_prices": f"{row.recall_new_prices:.4f}",
                "f1_new_prices": f"{row.f1_new_prices:.4f}",
                "precision_new_products": f"{row.precision_new_products:.4f}",
                "recall_new_products": f"{row.recall_new_products:.4f}",
                "f1_new_products": f"{row.f1_new_products:.4f}",
                "f1_combined": f"{row.f1_combined:.4f}",
                "training_time_sec": f"{row.training_time_sec:.2f}",
                "dataset_name": row.dataset_name,
                "n_train": row.n_train,
                "n_eval_prices": row.n_eval_prices,
                "n_eval_products": row.n_eval_products,
                "mean_reconstruction_error": f"{row.mean_reconstruction_error:.6f}",
                "error": row.error,
            })


def write_summary(results: list[GridRow], output_dir: str, dataset_name: str) -> None:
    """Write summary markdown and JSON."""
    if not results:
        return
    
    # Find best config
    valid_results = [r for r in results if r.error == "" and r.f1_combined > 0]
    if not valid_results:
        logger.warning("No valid results to summarize")
        return
    
    best = max(valid_results, key=lambda r: r.f1_combined)
    
    # Summary JSON
    summary = {
        "dataset": dataset_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_configs_evaluated": len(results),
        "valid_configs": len(valid_results),
        "best_config": {
            "hidden_dims": json.loads(best.hidden_dims),
            "latent_dim": best.latent_dim,
            "dropout": best.dropout,
            "learning_rate": best.learning_rate,
            "epochs": best.epochs,
            "threshold_percentile": best.threshold_percentile,
            "threshold_value": best.threshold_value,
        },
        "best_metrics": {
            "f1_combined": best.f1_combined,
            "f1_new_prices": best.f1_new_prices,
            "f1_new_products": best.f1_new_products,
            "precision_new_prices": best.precision_new_prices,
            "recall_new_prices": best.recall_new_prices,
            "precision_new_products": best.precision_new_products,
            "recall_new_products": best.recall_new_products,
        },
    }
    
    json_path = os.path.join(output_dir, "grid_search_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Summary markdown
    md_path = os.path.join(output_dir, "grid_search_summary.md")
    with open(md_path, "w") as f:
        f.write("# Autoencoder Grid Search Results\n\n")
        f.write(f"**Dataset:** {dataset_name}\n")
        f.write(f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")
        f.write(f"**Configs evaluated:** {len(results)} ({len(valid_results)} valid)\n\n")
        
        f.write("## Best Configuration\n\n")
        f.write("| Parameter | Value |\n")
        f.write("|-----------|-------|\n")
        f.write(f"| hidden_dims | {json.loads(best.hidden_dims)} |\n")
        f.write(f"| latent_dim | {best.latent_dim} |\n")
        f.write(f"| dropout | {best.dropout} |\n")
        f.write(f"| learning_rate | {best.learning_rate} |\n")
        f.write(f"| epochs | {best.epochs} |\n")
        f.write(f"| threshold_percentile | {best.threshold_percentile} |\n")
        f.write(f"| threshold_value | {best.threshold_value:.6f} |\n\n")
        
        f.write("## Best Metrics\n\n")
        f.write("| Metric | New Prices | New Products | Combined |\n")
        f.write("|--------|------------|--------------|----------|\n")
        f.write(f"| Precision | {best.precision_new_prices:.1%} | {best.precision_new_products:.1%} | - |\n")
        f.write(f"| Recall | {best.recall_new_prices:.1%} | {best.recall_new_products:.1%} | - |\n")
        f.write(f"| F1 | {best.f1_new_prices:.1%} | {best.f1_new_products:.1%} | **{best.f1_combined:.1%}** |\n\n")
        
        # Top 5 configs
        f.write("## Top 5 Configurations\n\n")
        f.write("| Rank | Hidden Dims | Latent | Dropout | LR | Threshold% | F1 Combined |\n")
        f.write("|------|-------------|--------|---------|-----|------------|-------------|\n")
        
        top5 = sorted(valid_results, key=lambda r: r.f1_combined, reverse=True)[:5]
        for i, r in enumerate(top5, 1):
            f.write(
                f"| {i} | {json.loads(r.hidden_dims)} | {r.latent_dim} | "
                f"{r.dropout} | {r.learning_rate} | {r.threshold_percentile} | "
                f"{r.f1_combined:.1%} |\n"
            )
    
    logger.info(f"Summary written to {md_path}")


def load_completed_run_ids(output_dir: str) -> set[str]:
    """Load run IDs from existing CSV for resume support."""
    csv_path = os.path.join(output_dir, "grid_search_results.csv")
    if not os.path.exists(csv_path):
        return set()
    
    completed = set()
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add(row["run_id"])
    
    return completed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Autoencoder hyperparameter grid search"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/training",
        help="Data directory (default: data/training)",
    )
    parser.add_argument(
        "--granularity",
        type=str,
        default="country_segment",
        help="Data granularity (default: country_segment)",
    )
    parser.add_argument(
        "--file-suffix",
        type=str,
        default=DEFAULT_FILE_SUFFIX,
        help=f"Training file suffix (default: {DEFAULT_FILE_SUFFIX})",
    )
    parser.add_argument(
        "--model-filter",
        type=str,
        default=None,
        help="Filter to specific model (e.g., 'DK_B2B' for fast testing)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Canonical sweep root. Defaults to results/tuning/autoencoder/<sweep_id>",
    )
    parser.add_argument(
        "--no-phases",
        action="store_true",
        help="Run full Cartesian grid (no phased pruning)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing results (skip completed run_ids)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Don't use cached features",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of top configs to keep between phases (default: 4)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without actually training",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel training workers (default: {DEFAULT_WORKERS}). "
             "Set to 2+ for parallel training. Each worker uses ~2-3GB RAM.",
    )
    parser.add_argument(
        "--sweep-id",
        type=str,
        default=None,
        help="Optional sweep id. Used when --output-dir is not provided.",
    )
    args = parser.parse_args()
    
    load_dotenv()
    sweep_id = args.sweep_id or create_run_id("autoencoder")
    if args.output_dir:
        sweep_root = Path(args.output_dir)
        sweep_id = args.sweep_id or sweep_root.name
    else:
        sweep_root = Path("results") / "tuning" / "autoencoder" / sweep_id
    os.makedirs(sweep_root, exist_ok=True)
    
    print("=" * 70)
    print("Autoencoder Hyperparameter Grid Search")
    print("=" * 70)
    print(f"Data path: {args.data_path}")
    print(f"Granularity: {args.granularity}")
    print(f"File suffix: {args.file_suffix}")
    print(f"Model filter: {args.model_filter or '(all)'}")
    print(f"Sweep root: {sweep_root}")
    print(f"Sweep ID: {sweep_id}")
    print(f"Phased search: {not args.no_phases}")
    print(f"Resume: {args.resume}")
    print(f"Top-K per phase: {args.top_k}")
    print(f"Parallel workers: {args.workers}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 70)
    print()
    
    # Dry run - just show what would be done
    if args.dry_run:
        if args.no_phases:
            n_configs = (
                len(PHASE1_HIDDEN_DIMS) * len(PHASE1_LATENT_DIM) *
                len(PHASE2_DROPOUT) * len(PHASE2_LEARNING_RATE) *
                len(PHASE3_THRESHOLD_PERCENTILE)
            )
            print(f"FULL CARTESIAN GRID: {n_configs} configs")
        else:
            p1 = len(PHASE1_HIDDEN_DIMS) * len(PHASE1_LATENT_DIM)
            p2 = args.top_k * len(PHASE2_DROPOUT) * len(PHASE2_LEARNING_RATE)
            p3 = args.top_k * len(PHASE3_THRESHOLD_PERCENTILE)
            total = p1 + p2 + p3
            print(f"PHASED SEARCH:")
            print(f"  Phase 1 (Architecture): {p1} configs")
            print(f"  Phase 2 (Regularization): {p2} configs (top {args.top_k} × {len(PHASE2_DROPOUT)} × {len(PHASE2_LEARNING_RATE)})")
            print(f"  Phase 3 (Threshold): {p3} configs (top {args.top_k} × {len(PHASE3_THRESHOLD_PERCENTILE)})")
            print(f"  TOTAL: {total} configs")
        
        print()
        print("Estimated time per config: ~30-60 sec (DK_B2B) or ~5-7 min (DK_B2C)")
        if args.workers > 1:
            print(f"With {args.workers} parallel workers, Phase 1 could complete ~{args.workers}x faster")
        print()
        print("Phase 1 configs to test:")
        for hd in PHASE1_HIDDEN_DIMS:
            for ld in PHASE1_LATENT_DIM:
                print(f"  hidden_dims={hd}, latent_dim={ld}")
        
        print()
        print("[DRY RUN] No training performed.")
        sys.exit(0)
    
    # Resolve files
    try:
        train_file = _resolve_train_file(args)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    
    dataset_name = extract_model_name(train_file)
    test_prices_file, test_products_file = _resolve_test_files(train_file, args.data_path)
    
    print(f"Train file: {os.path.basename(train_file)}")
    print(f"Test (new prices): {os.path.basename(test_prices_file) if test_prices_file else 'NOT FOUND'}")
    print(f"Test (new products): {os.path.basename(test_products_file) if test_products_file else 'NOT FOUND'}")
    print()
    
    if not test_prices_file and not test_products_file:
        logger.error("No test files found - cannot evaluate")
        sys.exit(1)
    
    # Setup cached evaluation data (one-time expensive operations)
    print("Setting up cached evaluation data...")
    use_cache = not args.no_cache
    
    cached_data = setup_cached_evaluation(
        train_file=train_file,
        test_prices_file=test_prices_file,
        test_products_file=test_products_file,
        model_name=dataset_name,
        use_cache=use_cache,
    )
    
    print(f"  Train: {cached_data.X_train.shape[0]:,} samples, {cached_data.X_train.shape[1]} features")
    if cached_data.df_test_prices_injected is not None:
        print(f"  Test (new prices): {len(cached_data.df_test_prices_injected):,} rows ({cached_data.labels_prices.sum():,} anomalies)")
    if cached_data.df_test_products_injected is not None:
        print(f"  Test (new products): {len(cached_data.df_test_products_injected):,} rows ({cached_data.labels_products.sum():,} anomalies)")
    print()
    
    csv_path = os.path.join(sweep_root, "candidate_metrics.csv")
    if args.resume:
        loaded_rows, saved_sweep_id = load_candidate_metrics(csv_path)
        if loaded_rows:
            all_results = loaded_rows
            if saved_sweep_id:
                sweep_id = saved_sweep_id
            print(f"Resuming: {len(all_results)} configs already completed")
        else:
            all_results = []
    else:
        all_results = []
    completed_ids = {row.run_id for row in all_results}
    written_candidates = set(completed_ids)

    def _save_progress() -> None:
        candidate_metrics = build_candidate_metrics_frame(
            all_results,
            sweep_id=sweep_id,
            dataset_name=dataset_name,
            dataset_granularity=args.granularity,
        )
        sweep_metadata = {
            "schema_version": "phase2.v1",
            "experiment_family": "tuning",
            "detector_family": "autoencoder",
            "sweep_id": sweep_id,
            "source_dataset_paths": [path for path in [train_file, test_prices_file, test_products_file] if path],
            "dataset_names": [dataset_name],
            "dataset_granularity": args.granularity,
            "dataset_splits": ["new_prices", "new_products"],
            "random_seeds": {"training_seed": RANDOM_STATE, "injection_seed": RANDOM_STATE},
            "injection_config": {"injection_rate": INJECTION_RATE},
            "detector_identifiers": ["Autoencoder"],
            "config_values": {
                "data_path": args.data_path,
                "file_suffix": args.file_suffix,
                "model_filter": args.model_filter,
                "no_phases": args.no_phases,
                "top_k": args.top_k,
                "workers": args.workers,
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

    def _record_candidates(candidates: list[CandidateEvaluationResult]) -> None:
        if not candidates:
            return
        for candidate in candidates:
            all_results.append(candidate.row)
            completed_ids.add(candidate.row.run_id)
            persist_candidate_run_artifacts(
                sweep_root=sweep_root,
                candidate=candidate,
                dataset_name=dataset_name,
                dataset_granularity=args.granularity,
                train_file=train_file,
                test_file_prices=test_prices_file,
                test_file_products=test_products_file,
                sweep_id=sweep_id,
                written_candidates=written_candidates,
            )
        _save_progress()
    
    total_start = time.perf_counter()
    
    if args.no_phases:
        # Full Cartesian grid
        print("Running FULL Cartesian grid (this will take a while)...")
        configs = list(product(
            PHASE1_HIDDEN_DIMS,
            PHASE1_LATENT_DIM,
            PHASE2_DROPOUT,
            PHASE2_LEARNING_RATE,
            PHASE3_THRESHOLD_PERCENTILE,
        ))
        print(f"Total configs: {len(configs)}")
        
        for i, (hidden_dims, latent_dim, dropout, lr, thresh_pct) in enumerate(configs, 1):
            run_id = f"full_{i:04d}"
            
            if run_id in completed_ids:
                continue
            
            print(f"[{i}/{len(configs)}] {hidden_dims}, lat={latent_dim}, drop={dropout}, lr={lr}, thresh={thresh_pct}%")
            
            candidate = run_single_config(
                cached_data=cached_data,
                hidden_dims=hidden_dims,
                latent_dim=latent_dim,
                dropout=dropout,
                learning_rate=lr,
                epochs=DEFAULT_EPOCHS,
                threshold_percentile=thresh_pct,
                run_id=run_id,
                phase=0,
            )
            _record_candidates([candidate])
            result = candidate.row
            
            if result.error:
                print(f"  ERROR: {result.error}")
            else:
                print(f"  F1: {result.f1_combined:.1%} (prices={result.f1_new_prices:.1%}, products={result.f1_new_products:.1%})")
    
    else:
        # Phased search
        print("=" * 60)
        print("PHASE 1: Architecture Search")
        print("=" * 60)
        
        phase1_configs = list(product(PHASE1_HIDDEN_DIMS, PHASE1_LATENT_DIM))
        print(f"Phase 1 configs: {len(phase1_configs)}")
        
        # Collect already-completed phase1 results
        phase1_results = [r for r in all_results if r.run_id.startswith("p1_")]
        
        if args.workers > 1:
            # Parallel execution
            print(f"Running Phase 1 with {args.workers} parallel workers...")
            
            # Build config dicts for parallel runner
            phase1_config_dicts = []
            for i, (hidden_dims, latent_dim) in enumerate(phase1_configs, 1):
                run_id = f"p1_{i:02d}"
                phase1_config_dicts.append({
                    "hidden_dims": hidden_dims,
                    "latent_dim": latent_dim,
                    "dropout": DEFAULT_DROPOUT,
                    "learning_rate": DEFAULT_LEARNING_RATE,
                    "epochs": DEFAULT_EPOCHS,
                    "threshold_percentile": DEFAULT_THRESHOLD_PERCENTILE,
                    "run_id": run_id,
                    "phase": 1,
                })
            
            # Run in parallel
            new_results = run_configs_parallel(
                cached_data=cached_data,
                configs=phase1_config_dicts,
                max_workers=args.workers,
                completed_ids=completed_ids,
            )
            
            _record_candidates(new_results)
            phase1_results.extend(candidate.row for candidate in new_results)
            
        else:
            # Sequential execution (original behavior)
            for i, (hidden_dims, latent_dim) in enumerate(phase1_configs, 1):
                run_id = f"p1_{i:02d}"
                
                if run_id in completed_ids:
                    continue
                
                print(f"[{i}/{len(phase1_configs)}] hidden={hidden_dims}, latent={latent_dim}")
                
                candidate = run_single_config(
                    cached_data=cached_data,
                    hidden_dims=hidden_dims,
                    latent_dim=latent_dim,
                    dropout=DEFAULT_DROPOUT,
                    learning_rate=DEFAULT_LEARNING_RATE,
                    epochs=DEFAULT_EPOCHS,
                    threshold_percentile=DEFAULT_THRESHOLD_PERCENTILE,
                    run_id=run_id,
                    phase=1,
                )
                _record_candidates([candidate])
                result = candidate.row
                phase1_results.append(result)
                
                if result.error:
                    print(f"  ERROR: {result.error}")
                else:
                    print(f"  F1: {result.f1_combined:.1%} ({result.training_time_sec:.1f}s)")
        
        # Select top-K from Phase 1
        phase1_valid = [r for r in phase1_results if r.error == ""]
        phase1_top = sorted(phase1_valid, key=lambda r: r.f1_combined, reverse=True)[:args.top_k]
        if not phase1_top:
            _save_progress()
            print("No valid Phase 1 candidates completed.")
            print(f"Results saved to: {sweep_root}/")
            return
        
        print(f"\nPhase 1 Top {args.top_k}:")
        for r in phase1_top:
            print(f"  {json.loads(r.hidden_dims)}, lat={r.latent_dim} -> F1={r.f1_combined:.1%}")
        
        # Phase 2: Regularization
        print()
        print("=" * 60)
        print("PHASE 2: Regularization Search")
        print("=" * 60)
        
        # Collect already-completed phase2 results
        phase2_results = [r for r in all_results if r.run_id.startswith("p2_")]
        
        if args.workers > 1:
            # Parallel execution
            print(f"Running Phase 2 with {args.workers} parallel workers...")
            
            # Build config dicts for parallel runner
            phase2_config_dicts = []
            phase2_idx = 0
            
            for base_config in phase1_top:
                hidden_dims = json.loads(base_config.hidden_dims)
                latent_dim = base_config.latent_dim
                
                for dropout, lr in product(PHASE2_DROPOUT, PHASE2_LEARNING_RATE):
                    phase2_idx += 1
                    run_id = f"p2_{phase2_idx:02d}"
                    phase2_config_dicts.append({
                        "hidden_dims": hidden_dims,
                        "latent_dim": latent_dim,
                        "dropout": dropout,
                        "learning_rate": lr,
                        "epochs": DEFAULT_EPOCHS,
                        "threshold_percentile": DEFAULT_THRESHOLD_PERCENTILE,
                        "run_id": run_id,
                        "phase": 2,
                    })
            
            # Run in parallel
            new_results = run_configs_parallel(
                cached_data=cached_data,
                configs=phase2_config_dicts,
                max_workers=args.workers,
                completed_ids=completed_ids,
            )
            
            _record_candidates(new_results)
            phase2_results.extend(candidate.row for candidate in new_results)
            
        else:
            # Sequential execution (original behavior)
            phase2_idx = 0
            
            for base_config in phase1_top:
                hidden_dims = json.loads(base_config.hidden_dims)
                latent_dim = base_config.latent_dim
                
                for dropout, lr in product(PHASE2_DROPOUT, PHASE2_LEARNING_RATE):
                    phase2_idx += 1
                    run_id = f"p2_{phase2_idx:02d}"
                    
                    if run_id in completed_ids:
                        continue
                    
                    print(f"[{phase2_idx}] hidden={hidden_dims}, lat={latent_dim}, drop={dropout}, lr={lr}")
                    
                    candidate = run_single_config(
                        cached_data=cached_data,
                        hidden_dims=hidden_dims,
                        latent_dim=latent_dim,
                        dropout=dropout,
                        learning_rate=lr,
                        epochs=DEFAULT_EPOCHS,
                        threshold_percentile=DEFAULT_THRESHOLD_PERCENTILE,
                        run_id=run_id,
                        phase=2,
                    )
                    _record_candidates([candidate])
                    result = candidate.row
                    phase2_results.append(result)
                    
                    if result.error:
                        print(f"  ERROR: {result.error}")
                    else:
                        print(f"  F1: {result.f1_combined:.1%} ({result.training_time_sec:.1f}s)")
        
        # Select top-K from Phase 2
        phase2_valid = [r for r in phase2_results if r.error == ""]
        phase2_top = sorted(phase2_valid, key=lambda r: r.f1_combined, reverse=True)[:args.top_k]
        if not phase2_top:
            _save_progress()
            print("No valid Phase 2 candidates completed.")
            print(f"Results saved to: {sweep_root}/")
            return
        
        print(f"\nPhase 2 Top {args.top_k}:")
        for r in phase2_top:
            print(f"  {json.loads(r.hidden_dims)}, lat={r.latent_dim}, drop={r.dropout}, lr={r.learning_rate} -> F1={r.f1_combined:.1%}")
        
        # Phase 3: Threshold tuning (optimized - train once per config, sweep thresholds)
        print()
        print("=" * 60)
        print("PHASE 3: Threshold Tuning (Optimized)")
        print("=" * 60)
        
        for config_idx, base_config in enumerate(phase2_top, 1):
            base_run_id = f"p3_c{config_idx}"
            expected_run_ids = {
                f"{base_run_id}_{threshold_idx:02d}"
                for threshold_idx, _ in enumerate(PHASE3_THRESHOLD_PERCENTILE, 1)
            }
            if expected_run_ids.issubset(completed_ids):
                print(f"[Config {config_idx}] Already completed, skipping")
                continue
            
            hidden_dims = json.loads(base_config.hidden_dims)
            print(f"\n[Config {config_idx}/{len(phase2_top)}] hidden={hidden_dims}, lat={base_config.latent_dim}, "
                  f"drop={base_config.dropout}, lr={base_config.learning_rate}")
            
            best_config = {
                "hidden_dims": hidden_dims,
                "latent_dim": base_config.latent_dim,
                "dropout": base_config.dropout,
                "learning_rate": base_config.learning_rate,
                "epochs": DEFAULT_EPOCHS,
            }
            
            # Run threshold sweep (trains once, evaluates once per test set, sweeps thresholds)
            sweep_results = run_threshold_sweep(
                cached_data=cached_data,
                best_config=best_config,
                threshold_percentiles=PHASE3_THRESHOLD_PERCENTILE,
                base_run_id=base_run_id,
            )
            pending_results = [
                candidate for candidate in sweep_results
                if candidate.row.run_id not in completed_ids
            ]
            _record_candidates(pending_results)
    
    total_elapsed = time.perf_counter() - total_start
    
    # Final summary
    print()
    print("=" * 70)
    print("GRID SEARCH COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_elapsed / 60:.1f} minutes")
    print(f"Configs evaluated: {len(all_results)}")
    
    # Write final outputs
    _save_progress()
    
    # Print best config
    valid_results = [r for r in all_results if r.error == "" and r.f1_combined > 0]
    if valid_results:
        best = max(valid_results, key=lambda r: r.f1_combined)
        print()
        print("BEST CONFIGURATION:")
        print(f"  hidden_dims: {json.loads(best.hidden_dims)}")
        print(f"  latent_dim: {best.latent_dim}")
        print(f"  dropout: {best.dropout}")
        print(f"  learning_rate: {best.learning_rate}")
        print(f"  threshold_percentile: {best.threshold_percentile}")
        print()
        print("BEST METRICS:")
        print(f"  F1 (combined): {best.f1_combined:.1%}")
        print(f"  F1 (new_prices): {best.f1_new_prices:.1%}")
        print(f"  F1 (new_products): {best.f1_new_products:.1%}")
        print(f"  Precision (new_prices): {best.precision_new_prices:.1%}")
        print(f"  Recall (new_prices): {best.recall_new_prices:.1%}")
    
    print()
    print(f"Results saved to: {sweep_root}/")
    print(f"Summary: {sweep_root / 'summary.md'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
