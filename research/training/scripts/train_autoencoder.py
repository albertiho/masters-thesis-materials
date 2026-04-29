#!/usr/bin/env python3
"""Train autoencoder models from local Parquet files.

This script trains price anomaly detection autoencoders at two granularity levels:
1. country_segment: One model per country+segment (8 models)
2. competitor: One model per competitor (N models)

The script reads from local Parquet files and saves trained models under the
local artifacts directory.

Usage:
    # Train country+segment models (8 models)
    python scripts/train_autoencoder.py --granularity country_segment
    
    # Train per-competitor models
    python scripts/train_autoencoder.py --granularity competitor
    
    # Train both and compare
    python scripts/train_autoencoder.py --granularity both --evaluate
    
    # Resume training (skip models that already exist locally)
    python scripts/train_autoencoder.py --resume
    
    # Force recompute features (ignore cache)
    python scripts/train_autoencoder.py --no-cache
    
    # Dry run (list files, don't train)
    python scripts/train_autoencoder.py --dry-run

Architecture:
    Parquet Files -> Feature Extraction (cached) -> Autoencoder Training -> Local Models
    
    Feature matrices are cached as .autoencoder_features.npz files for fast subsequent runs.
    Each model type uses a distinct cache extension to prevent conflicts:
        - Autoencoder: .autoencoder_features.npz (9 features)
        - Isolation Forest: .iforest_features.npz (12 features)
    The published workflow uses only local filesystem paths.
"""

import argparse
import glob
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

try:
    import numpy as np
    import pandas as pd
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install pandas numpy")
    sys.exit(1)

# Add project root to path (4 levels up from research/training/scripts/)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_script_dir)))
sys.path.insert(0, _project_root)
sys.path.insert(0, _script_dir)  # For importing sibling scripts

from dotenv import load_dotenv

from src.anomaly.persistence import ModelPersistence
from src.research.evaluation.synthetic import inject_anomalies_to_dataframe
from src.features.temporal import DEFAULT_HISTORY_DEPTH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Rolling window size - must match DEFAULT_HISTORY_DEPTH in temporal.py (30 observations)
# This ensures training computes statistics the same way as inference.
ROLLING_WINDOW_SIZE = DEFAULT_HISTORY_DEPTH

# Feature schema version - bump when feature computation logic changes.
# This auto-invalidates cached features when the computation method changes,
# even if the feature names stay the same.
# Format: "YYYY-MM-DD-vN" where N increments for same-day changes.
# History:
#   2026-01-22-v1: Initial version after rolling window fix (was all-time aggregation)
#   2026-01-29-v1: Added shift(1) to rolling stats to exclude current price (train-serve skew fix)
FEATURE_SCHEMA_VERSION = "2026-01-29-v1"

# Feature names matching AutoencoderDetector._prepare_features()
FEATURE_NAMES = [
    "price",
    "price_log", 
    "price_ratio",
    "has_list_price",
    "rolling_mean",
    "rolling_std",
    "price_zscore",
    "price_change_pct",
    "price_vs_mean_ratio",
]


@dataclass
class TrainingResult:
    """Result from training a single model."""
    
    model_name: str
    granularity: str
    n_samples: int
    n_products: int
    training_time_sec: float
    threshold: float
    mean_reconstruction_error: float
    # Evaluation metrics (optional)
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    model_uri: str | None = None


def find_parquet_files(data_path: str, granularity: str, file_suffix: str = "") -> list[str]:
    """Find Parquet files for the specified granularity.
    
    Args:
        data_path: Base data directory.
        granularity: 'country_segment' or 'competitor'
        file_suffix: Optional suffix filter (e.g., '_train' to match only *_train.parquet)
    
    Returns:
        List of Parquet file paths
    """
    if granularity == "country_segment":
        subdir = "by_country_segment"
    elif granularity == "competitor":
        subdir = "by_competitor"
    else:
        raise ValueError(f"Invalid granularity: {granularity}")
    
    # Build glob pattern with optional suffix
    file_pattern = f"*{file_suffix}.parquet" if file_suffix else "*.parquet"
    
    if granularity == "competitor":
        pattern = os.path.join(data_path, subdir, "**", file_pattern)
        files = glob.glob(pattern, recursive=True)
    else:
        pattern = os.path.join(data_path, subdir, file_pattern)
        files = glob.glob(pattern)
    
    return sorted(files)


def load_parquet_file(path: str) -> pd.DataFrame:
    """Load a local Parquet file."""
    return pd.read_parquet(path)


def extract_base_name(filepath: str) -> str:
    """Extract base name without suffix (for matching train/test files).

    Examples:
        'DK_B2C_2026-01-18_train_mh5.parquet' -> 'DK_B2C_2026-01-18'
        'DK_B2C_2026-01-18_train.parquet' -> 'DK_B2C_2026-01-18'
        'POWER_DK_B2C_INTERNAL_2026-01-18_train_mh4.parquet' -> 'POWER_DK_B2C_INTERNAL_2026-01-18'
    """
    import re
    
    filename = os.path.basename(filepath)
    name = filename.replace(".parquet", "")

    # Remove experiment suffix (e.g., _mh5) if present
    suffix_match = re.search(r"_(?:train|test(?:_new_products|_new_prices)?)(_[a-z0-9]+)$", name)
    if suffix_match:
        name = name[: suffix_match.start()]
    else:
        # Remove _train or _test suffix if present
        train_test_match = re.search(r"_(?:train|test(?:_new_products|_new_prices)?)$", name)
        if train_test_match:
            name = name[: train_test_match.start()]

    return name


def find_matching_test_file(train_file: str, test_suffix: str, data_path: str) -> str | None:
    """Find test file matching a train file with different suffix.

    Args:
        train_file: Path to training file (e.g., data/training/by_country_segment/DK_B2C_2026-01-18_train_mh5.parquet)
        test_suffix: Suffix for test file (e.g., '_train' for unfiltered data)
        data_path: Base data directory

    Returns:
        Path to matching test file or None if not found
    """
    import glob as glob_module
    
    base_name = extract_base_name(train_file)
    parent_dir = os.path.dirname(train_file)

    # Build expected test filename
    test_filename = f"{base_name}{test_suffix}.parquet"
    test_path = os.path.join(parent_dir, test_filename)

    if os.path.exists(test_path):
        return test_path

    # Try with glob in case of slight naming variations
    pattern = os.path.join(parent_dir, f"*{test_suffix}.parquet")
    matches = [f for f in glob_module.glob(pattern) if extract_base_name(f) == base_name]
    if matches:
        return matches[0]

    return None


def extract_model_name(filepath: str) -> str:
    """Extract model name from Parquet filename.
    
    Examples:
        'DK_B2C_2026-01-18.parquet' -> 'DK_B2C'
        'POWER_DK_B2C_INTERNAL_2026-01-18.parquet' -> 'POWER_DK_B2C_INTERNAL'
        'DK_B2C_2026-01-18_train.parquet' -> 'DK_B2C'
        'DK_B2C_2026-01-18_train_mh5.parquet' -> 'DK_B2C_mh5'
        'DK_B2C_2026-01-18_test_new_prices_mh5.parquet' -> 'DK_B2C_mh5'
        'POWER_DK_B2C_INTERNAL_2026-01-18_train_mh4.parquet' -> 'POWER_DK_B2C_INTERNAL_mh4'
    """
    import re
    
    filename = os.path.basename(filepath)
    name = filename.replace(".parquet", "")
    
    # Extract experiment suffix (e.g., _mh5, _mh4) if present after _train or _test
    # Pattern: _train_SUFFIX or _test_*_SUFFIX where SUFFIX doesn't contain underscores
    suffix = ""
    suffix_match = re.search(r"_(?:train|test(?:_new_products|_new_prices)?)(_[a-z0-9]+)$", name)
    if suffix_match:
        suffix = suffix_match.group(1)  # e.g., "_mh5"
        # Remove the entire train/test part including suffix
        name = name[:suffix_match.start()]
    else:
        # Remove _train or _test suffix if present (no experiment suffix)
        train_test_match = re.search(r"_(?:train|test(?:_new_products|_new_prices)?)$", name)
        if train_test_match:
            name = name[:train_test_match.start()]
    
    # Remove date suffix like _2026-01-18 (YYYY-MM-DD format with dashes)
    date_pattern = r"_\d{4}-\d{2}-\d{2}$"
    name = re.sub(date_pattern, "", name)
    
    # Append experiment suffix to model name
    return name + suffix


def get_cache_path(parquet_path: str) -> str:
    """Get the cache file path for autoencoder features.
    
    Cache files are stored alongside parquet files with model-specific extensions
    to prevent conflicts between different models (autoencoder vs isolation forest).
    
    Cache extensions by model:
        - Autoencoder: .autoencoder_features.npz (9 features)
        - Isolation Forest: .iforest_features.npz (12 features)
    """
    return parquet_path.replace(".parquet", ".autoencoder_features.npz")


def load_cached_features(cache_path: str) -> np.ndarray | None:
    """Load cached feature matrix if it exists and is valid.
    
    Validates both feature names AND schema version to ensure cache
    invalidates when computation logic changes (not just feature names).
    
    Args:
        cache_path: Path to .npz cache file
        
    Returns:
        Feature matrix or None if cache doesn't exist/is invalid
    """
    if not os.path.exists(cache_path):
        return None
    
    try:
        data = np.load(cache_path)
        X = data["features"]
        cached_names = list(data["feature_names"])
        
        # Validate feature names match
        if cached_names != FEATURE_NAMES:
            logger.warning(f"  Cache feature mismatch, recomputing")
            return None
        
        # Validate schema version (auto-invalidate when computation changes)
        cached_version = str(data.get("schema_version", ""))
        if cached_version != FEATURE_SCHEMA_VERSION:
            logger.warning(
                f"  Cache schema version mismatch (cached={cached_version!r}, "
                f"current={FEATURE_SCHEMA_VERSION!r}), recomputing"
            )
            return None
            
        return X
    except Exception as e:
        logger.warning(f"  Cache load failed: {e}")
        return None


def save_cached_features(cache_path: str, X: np.ndarray) -> None:
    """Save feature matrix to cache file.
    
    Includes schema version for automatic cache invalidation when
    computation logic changes.
    
    Args:
        cache_path: Path to .npz cache file
        X: Feature matrix to cache
    """
    try:
        np.savez_compressed(
            cache_path,
            features=X,
            feature_names=FEATURE_NAMES,
            schema_version=FEATURE_SCHEMA_VERSION,
        )
        logger.info(f"  Cached features to {os.path.basename(cache_path)} (schema={FEATURE_SCHEMA_VERSION})")
    except Exception as e:
        logger.warning(f"  Cache save failed: {e}")


def extract_features_vectorized(df: pd.DataFrame, window_size: int = ROLLING_WINDOW_SIZE) -> np.ndarray:
    """Extract features from DataFrame using vectorized pandas operations.
    
    IMPORTANT: Uses rolling window of `window_size` observations (default: 30) to match
    inference behavior in src/features/temporal.py. This prevents train-serve skew where
    training would use all-time statistics but inference only uses the last 30 observations.
    
    Args:
        df: DataFrame with price data (must have: price, list_price, product_id)
            Optionally: first_seen_at for sequential price_change_pct computation
        window_size: Rolling window size for statistics (default: 30, matching inference)
    
    Returns:
        Feature matrix of shape (n_samples, 9) matching FEATURE_NAMES
    """
    n = len(df)
    df = df.copy()
    
    # Sort by product and time for correct rolling window computation
    # This is required for both rolling stats and price_change_pct
    if "first_seen_at" in df.columns:
        df = df.sort_values(["product_id", "first_seen_at"])
    else:
        # If no timestamp, at least group by product for consistent ordering
        df = df.sort_values(["product_id"])
    
    # Compute rolling statistics with window_size observations (default: 30)
    # This matches inference behavior in temporal.py which uses last 30 observations
    # Using min_periods=1 to handle products with fewer observations
    # IMPORTANT: shift(1) excludes current price from rolling stats to match inference behavior
    df["rolling_mean"] = df.groupby("product_id")["price"].transform(
        lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean()
    )
    df["rolling_std"] = df.groupby("product_id")["price"].transform(
        lambda x: x.shift(1).rolling(window=window_size, min_periods=1).std()
    )
    
    # Observation count within the rolling window (capped at window_size)
    # shift(1) ensures we count only prior observations (not including current)
    df["obs_count"] = df.groupby("product_id")["price"].transform(
        lambda x: x.shift(1).rolling(window=window_size, min_periods=1).count()
    ).fillna(0).astype(int)
    
    # Compute price_change_pct using sequential data
    # Note: df is already sorted by (product_id, first_seen_at) above
    if "first_seen_at" in df.columns:
        # Get previous price within each product group
        df["prev_price"] = df.groupby("product_id")["price"].shift(1)
        # Compute percentage change: (current - previous) / previous
        df["price_change_pct"] = np.where(
            (df["prev_price"].notna()) & (df["prev_price"] > 0),
            (df["price"] - df["prev_price"]) / df["prev_price"],
            0.0,
        )
    else:
        # Fallback if no timestamp available
        df["price_change_pct"] = 0.0
    
    # Extract base columns as numpy arrays
    price = df["price"].fillna(0).values.astype(np.float64)
    list_price = df["list_price"].values.astype(np.float64)
    rolling_mean = df["rolling_mean"].values.astype(np.float64)
    rolling_std = df["rolling_std"].fillna(0).values.astype(np.float64)
    obs_count = df["obs_count"].fillna(0).values.astype(np.int32)
    
    # Compute derived features (all vectorized)
    # 1. price (direct)
    feat_price = price
    
    # 2. price_log
    feat_price_log = np.log(np.maximum(price, 1e-10) + 1)
    
    # 3. price_ratio (price / list_price, default 1.0)
    valid_list = ~np.isnan(list_price) & (list_price > 0)
    feat_price_ratio = np.ones(n, dtype=np.float64)
    feat_price_ratio[valid_list] = price[valid_list] / list_price[valid_list]
    
    # 4. has_list_price (binary)
    feat_has_list_price = (~np.isnan(list_price)).astype(np.float64)
    
    # 5. rolling_mean (direct, fill NaN with 0)
    feat_rolling_mean = np.nan_to_num(rolling_mean, nan=0.0)
    
    # 6. rolling_std (direct, fill NaN with 0)
    feat_rolling_std = np.nan_to_num(rolling_std, nan=0.0)
    
    # 7. price_zscore: (price - mean) / std, only if has_history
    has_history = (obs_count >= 3) & (rolling_std > 0)
    feat_price_zscore = np.zeros(n, dtype=np.float64)
    feat_price_zscore[has_history] = (
        (price[has_history] - rolling_mean[has_history]) / rolling_std[has_history]
    )
    
    # 8. price_change_pct (computed from sequential data above)
    feat_price_change_pct = df["price_change_pct"].fillna(0).values.astype(np.float64)
    
    # 9. price_vs_mean_ratio
    valid_mean = rolling_mean > 0
    feat_price_vs_mean = np.ones(n, dtype=np.float64)
    feat_price_vs_mean[valid_mean] = price[valid_mean] / rolling_mean[valid_mean]
    
    # Stack into feature matrix
    X = np.column_stack([
        feat_price,
        feat_price_log,
        feat_price_ratio,
        feat_has_list_price,
        feat_rolling_mean,
        feat_rolling_std,
        feat_price_zscore,
        feat_price_change_pct,
        feat_price_vs_mean,
    ])
    
    # Replace any remaining NaN/Inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X


def train_from_matrix(
    X: np.ndarray,
    config: "AutoencoderConfig",
    log_interval: int = 5,
) -> tuple["AutoencoderDetector", float, float, np.ndarray]:
    """Train autoencoder directly from feature matrix.
    
    This bypasses the slow dataclass conversion in detector.fit().
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        config: Autoencoder configuration
        log_interval: Log training progress every N epochs (default: 5). Set to 0 to disable.
        
    Returns:
        Tuple of (detector, threshold, mean_error, train_errors)
    """
    from src.anomaly.ml.autoencoder import AutoencoderConfig, AutoencoderDetector, AutoencoderModel
    
    # Filter out invalid rows (all zeros or NaN)
    valid_mask = ~(np.all(X == 0, axis=1) | np.any(np.isnan(X), axis=1))
    X_valid = X[valid_mask]
    
    if len(X_valid) < 50:
        raise ValueError(f"Need at least 50 valid samples, got {len(X_valid)}")
    
    # Normalize
    mean = X_valid.mean(axis=0)
    std = X_valid.std(axis=0)
    std[std == 0] = 1.0
    X_normalized = (X_valid - mean) / std
    
    # Update config with input dimension
    config.input_dim = X_valid.shape[1]
    
    # Create and train model
    model = AutoencoderModel(config)
    model.fit(X_normalized, verbose=False, log_interval=log_interval)
    
    # Compute threshold (optimal: 99th percentile from grid search)
    train_errors = model.get_reconstruction_error(X_normalized)
    threshold = float(np.percentile(train_errors, 99))  # Optimal: 99 (was 95)
    mean_error = float(train_errors.mean())
    
    # Create detector and set internal state
    detector = AutoencoderDetector(config)
    detector._model = model
    detector._is_fitted = True
    detector._threshold = threshold
    detector._feature_names = FEATURE_NAMES
    detector._mean = mean
    detector._std = std
    
    return detector, threshold, mean_error, train_errors


def train_single_model(
    filepath: str,
    model_name: str,
    granularity: str,
    evaluate: bool = False,
    save_models: bool = True,
    persistence: ModelPersistence | None = None,
    use_cache: bool = True,
    test_filepath: str | None = None,
    auto_tune: bool = False,
    tune_steps: int = 30,
    tune_trials: int = 10,
) -> TrainingResult | None:
    """Train a single autoencoder model from a Parquet file.
    
    Uses cached feature matrices when available for faster subsequent runs.
    
    Args:
        filepath: Path to Parquet file (training data)
        model_name: Name for the model
        granularity: 'country_segment' or 'competitor'
        evaluate: Whether to evaluate with synthetic anomaly injection
        save_models: Whether to save models locally
        persistence: ModelPersistence instance for saving
        use_cache: Whether to use cached features (default: True)
        test_filepath: Optional separate file for evaluation (e.g., unfiltered data).
                      If provided, threshold tuning uses this data instead of training data.
        auto_tune: Whether to auto-tune the threshold using grid search (default: False)
        tune_steps: Number of threshold percentile values to test (default: 30)
        tune_trials: Number of trials for robust evaluation (default: 10)
    
    Returns:
        TrainingResult or None if training failed
    """
    try:
        from src.anomaly.ml.autoencoder import AutoencoderConfig, AutoencoderDetector
    except ImportError:
        logger.error("PyTorch not available. Install with: pip install torch")
        return None
    
    logger.info(f"Training model: {model_name}")
    start_time = datetime.now(timezone.utc)
    
    # Load data
    df = load_parquet_file(filepath)
    n_rows = len(df)
    n_products = df["product_id"].nunique()
    logger.info(f"  Loaded {n_rows:,} rows, {n_products:,} products")
    
    if n_rows < 100:
        logger.warning(f"  Skipping {model_name}: insufficient data ({n_rows} rows)")
        return None
    
    # Try to load cached features
    cache_path = get_cache_path(filepath)
    X = None
    
    if use_cache:
        X = load_cached_features(cache_path)
        if X is not None:
            logger.info(f"  Using cached features ({X.shape[0]:,} samples)")
    
    # Extract features if not cached
    if X is None:
        logger.info(f"  Extracting features (vectorized)...")
        extract_start = datetime.now(timezone.utc)
        X = extract_features_vectorized(df)
        extract_time = (datetime.now(timezone.utc) - extract_start).total_seconds()
        logger.info(f"  Extracted {X.shape[0]:,} x {X.shape[1]} features in {extract_time:.1f}s")
        
        # Cache for next time
        if use_cache:
            save_cached_features(cache_path, X)
    
    # Configure autoencoder (optimal hyperparameters from grid search)
    config = AutoencoderConfig(
        epochs=50,
        batch_size=32,
        learning_rate=0.0001,  # Optimal: 0.0001 (was 0.001)
        latent_dim=4,          # Optimal: 4 (was 8)
        hidden_dims=[128, 64, 32],  # Optimal architecture
        dropout=0.1,
    )
    
    # Train directly from matrix (fast path)
    try:
        train_start = datetime.now(timezone.utc)
        detector, threshold, mean_error, train_errors = train_from_matrix(X, config)
        train_time = (datetime.now(timezone.utc) - train_start).total_seconds()
        logger.info(f"  Trained in {train_time:.1f}s, threshold={threshold:.4f}")
    except Exception as e:
        logger.warning(f"  Training failed: {e}")
        return None
    
    total_time = (datetime.now(timezone.utc) - start_time).total_seconds()
    
    result = TrainingResult(
        model_name=model_name,
        granularity=granularity,
        n_samples=n_rows,
        n_products=n_products,
        training_time_sec=total_time,
        threshold=threshold,
        mean_reconstruction_error=mean_error,
    )
    
    # Auto-tune threshold if enabled
    if auto_tune:
        # Use test file DataFrame if provided, otherwise use training DataFrame
        # Note: tune_threshold() now uses DataFrame-level injection (production-like)
        df_eval = df
        if test_filepath:
            logger.info(f"  Loading separate test file for evaluation: {os.path.basename(test_filepath)}")
            df_eval = load_parquet_file(test_filepath)
            logger.info(f"    Test data: {len(df_eval):,} rows, {df_eval['product_id'].nunique():,} products")

        # train_errors already returned from train_from_matrix()
        logger.info(f"  Auto-tuning threshold ({tune_steps} steps, {tune_trials} trials)...")
        logger.info(f"    Using DataFrame-level injection (production-like)")
        tune_start = datetime.now(timezone.utc)
        best_threshold, metrics = tune_threshold(
            detector,
            df_eval,
            train_errors,
            min_percentile=80.0,
            max_percentile=99.5,
            steps=tune_steps,
            n_trials=tune_trials,
            target_metric="f1",
        )
        tune_time = (datetime.now(timezone.utc) - tune_start).total_seconds()

        # Log before/after comparison
        default_f1 = metrics["default_f1"]
        best_f1 = metrics["f1"]
        improvement = (best_f1 - default_f1) / default_f1 * 100 if default_f1 > 0 else 0

        logger.info(
            f"  Default ({threshold:.4f}): F1={default_f1:.1%}, "
            f"P={metrics['default_precision']:.1%}, R={metrics['default_recall']:.1%}"
        )
        logger.info(
            f"  Best    ({best_threshold:.4f}): F1={best_f1:.1%}, "
            f"P={metrics['precision']:.1%}, R={metrics['recall']:.1%}  "
            f"[{improvement:+.1f}% improvement] ({tune_time:.1f}s)"
        )

        # Update detector and result with best threshold
        detector._threshold = best_threshold
        result.threshold = best_threshold
        result.precision = metrics["precision"]
        result.recall = metrics["recall"]
        result.f1_score = metrics["f1"]
    
    # Evaluate with synthetic anomaly injection (if not auto-tuning)
    elif evaluate:
        # Use test DataFrame if provided, otherwise use training DataFrame
        df_eval = df
        if test_filepath:
            logger.info(f"  Loading separate test file for evaluation: {os.path.basename(test_filepath)}")
            df_eval = load_parquet_file(test_filepath)
            logger.info(f"    Test data: {len(df_eval):,} rows, {df_eval['product_id'].nunique():,} products")
        
        logger.info("  Evaluating with production-like anomaly injection...")
        eval_start = datetime.now(timezone.utc)
        eval_result = evaluate_production_like(detector, df_eval)
        eval_time = (datetime.now(timezone.utc) - eval_start).total_seconds()
        result.precision = eval_result["precision"]
        result.recall = eval_result["recall"]
        result.f1_score = eval_result["f1_score"]
        logger.info(
            f"  P={result.precision:.1%}, R={result.recall:.1%}, F1={result.f1_score:.1%} ({eval_time:.1f}s)"
        )
    
    # Save model
    if save_models and persistence:
        try:
            model_uri = persistence.save_autoencoder(
                detector,
                model_name,
                n_rows,
            )
            result.model_uri = model_uri
            logger.info(f"  Saved to {model_uri}")
        except Exception as e:
            logger.warning(f"  Failed to save model: {e}")
    
    return result


def evaluate_production_like(
    detector,
    df: pd.DataFrame,
    injection_rate: float = 0.1,
    seed: int = 42,
) -> dict:
    """Evaluate detector using production-like anomaly injection.
    
    Injects anomalies at the DataFrame level and re-extracts features,
    ensuring rolling statistics are computed correctly (no look-ahead bias).
    
    Args:
        detector: Trained detector with _model, _mean, _std, _threshold
        df: DataFrame with price data (will be modified with injected anomalies)
        injection_rate: Fraction of records to inject anomalies
        seed: Random seed for reproducibility
    
    Returns:
        Dict with precision, recall, f1_score
    """
    # Inject anomalies at DataFrame level and extract features
    X_modified, inject_mask = inject_anomalies_production_like(
        df,
        seed=seed,
        injection_rate=injection_rate,
    )
    
    # Normalize using detector's normalization params
    X_normalized = (X_modified - detector._mean) / detector._std
    
    # Get reconstruction errors (batch)
    errors = detector._model.get_reconstruction_error(X_normalized)
    
    # Classify as anomaly if error > threshold
    predictions = errors > detector._threshold
    
    # Calculate metrics
    true_positives = np.sum(predictions & inject_mask)
    false_positives = np.sum(predictions & ~inject_mask)
    false_negatives = np.sum(~predictions & inject_mask)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "true_positives": int(true_positives),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "n_injected": int(np.sum(inject_mask)),
    }


def inject_anomalies_production_like(
    df: pd.DataFrame,
    seed: int = 42,
    injection_rate: float = 0.1,
    spike_range: tuple[float, float] = (2.0, 5.0),
    drop_range: tuple[float, float] = (0.1, 0.5),
) -> tuple[np.ndarray, np.ndarray]:
    """Inject anomalies at DataFrame level, return feature matrix and mask.

    Production-like evaluation: anomalies are injected at raw data level
    and flow through full feature extraction pipeline. This matches how
    anomalies would appear in production (affecting rolling stats, etc.).

    Uses inject_anomalies_to_dataframe() from synthetic.py which includes
    6 anomaly types: PRICE_SPIKE, PRICE_DROP, ZERO_PRICE, NEGATIVE_PRICE,
    EXTREME_OUTLIER, DECIMAL_SHIFT.

    Args:
        df: DataFrame with price data (must have: price, list_price, product_id).
        seed: Random seed for reproducibility.
        injection_rate: Fraction of records to inject anomalies.
        spike_range: (min, max) multiplier for price spikes (e.g., 2x-5x).
        drop_range: (min, max) multiplier for price drops (e.g., 0.1x-0.5x).

    Returns:
        Tuple of (feature matrix from injected DataFrame, boolean mask of injected indices).
    """
    df_injected, labels, _ = inject_anomalies_to_dataframe(
        df,
        injection_rate=injection_rate,
        seed=seed,
        spike_range=spike_range,
        drop_range=drop_range,
    )

    # Extract features for autoencoder (9-feature schema)
    X_modified = extract_features_vectorized(df_injected)

    return X_modified, labels


def tune_threshold(
    detector,
    df: pd.DataFrame,
    train_errors: np.ndarray,
    min_percentile: float = 80.0,
    max_percentile: float = 99.5,
    steps: int = 30,
    n_trials: int = 10,
    target_metric: str = "f1",
    injection_rate: float = 0.1,
) -> tuple[float, dict]:
    """Grid search for optimal reconstruction error threshold.

    Evaluates multiple threshold percentiles across multiple trials with different
    random seeds to find the threshold that maximizes the target metric.

    Uses DataFrame-level anomaly injection (production-like) where anomalies are
    injected at raw data level and flow through full feature extraction pipeline.

    Args:
        detector: Trained AutoencoderDetector with _model, _mean, _std.
        df: DataFrame with price data for evaluation (anomalies will be injected).
        train_errors: Reconstruction errors from training data (for percentile thresholds).
        min_percentile: Minimum threshold percentile to try (default: 80).
        max_percentile: Maximum threshold percentile to try (default: 99.5).
        steps: Number of percentile values to test (default: 30).
        n_trials: Number of trials with different seeds to average (default: 10).
        target_metric: Metric to optimize ("f1", "precision", "recall").
        injection_rate: Fraction of data to inject as anomalies (default: 0.1).

    Returns:
        Tuple of (best_threshold, metrics_dict with precision, recall, f1, default_metrics).
    """
    percentiles = np.linspace(min_percentile, max_percentile, steps)
    thresholds = [float(np.percentile(train_errors, p)) for p in percentiles]

    # Accumulate results across trials
    threshold_metrics: dict[int, list[tuple[float, float, float]]] = {
        i: [] for i in range(len(thresholds))
    }
    default_metrics: list[tuple[float, float, float]] = []
    default_threshold = detector._threshold

    # Run multiple trials with different seeds
    for trial in range(n_trials):
        seed = 1000 + trial * 17

        # Inject anomalies at DataFrame level (production-like)
        # This uses inject_anomalies_to_dataframe() internally
        X_modified, inject_mask = inject_anomalies_production_like(
            df,
            seed=seed,
            injection_rate=injection_rate,
            spike_range=(2.0, 5.0),
            drop_range=(0.1, 0.5),
        )

        # Normalize and get errors
        X_normalized = (X_modified - detector._mean) / detector._std
        errors = detector._model.get_reconstruction_error(X_normalized)

        # Evaluate at each threshold
        for i, threshold in enumerate(thresholds):
            predictions = errors > threshold
            tp = np.sum(predictions & inject_mask)
            fp = np.sum(predictions & ~inject_mask)
            fn = np.sum(~predictions & inject_mask)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            threshold_metrics[i].append((precision, recall, f1))

        # Also evaluate default threshold
        predictions = errors > default_threshold
        tp = np.sum(predictions & inject_mask)
        fp = np.sum(predictions & ~inject_mask)
        fn = np.sum(~predictions & inject_mask)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        default_metrics.append((precision, recall, f1))

    # Average results across trials and find best
    best_threshold = default_threshold
    best_score = -1
    best_metrics = {"precision": 0, "recall": 0, "f1": 0}

    for i, threshold in enumerate(thresholds):
        metrics = threshold_metrics[i]
        avg_precision = np.mean([m[0] for m in metrics])
        avg_recall = np.mean([m[1] for m in metrics])
        avg_f1 = np.mean([m[2] for m in metrics])

        score = {"f1": avg_f1, "precision": avg_precision, "recall": avg_recall}[target_metric]

        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = {"precision": avg_precision, "recall": avg_recall, "f1": avg_f1}

    # Compute default threshold average metrics
    avg_default_precision = np.mean([m[0] for m in default_metrics])
    avg_default_recall = np.mean([m[1] for m in default_metrics])
    avg_default_f1 = np.mean([m[2] for m in default_metrics])

    return best_threshold, {
        "precision": best_metrics["precision"],
        "recall": best_metrics["recall"],
        "f1": best_metrics["f1"],
        "default_precision": avg_default_precision,
        "default_recall": avg_default_recall,
        "default_f1": avg_default_f1,
    }


def eval_saved_model(
    model_name: str,
    granularity: str,
    test_filepath: str,
    persistence: "ModelPersistence",
    use_cache: bool = True,
) -> TrainingResult | None:
    """Evaluate an existing local model against test data.
    
    Args:
        model_name: Name of the model to load (e.g., 'DK_B2C_mh4')
        granularity: 'country_segment' or 'competitor'
        test_filepath: Path to test data file
        persistence: ModelPersistence instance for loading
        use_cache: Whether to use cached features
    
    Returns:
        TrainingResult with evaluation metrics or None if failed
    """
    logger.info(f"Evaluating saved model: {model_name}")
    start_time = datetime.now(timezone.utc)
    
    # Load model from local storage
    try:
        logger.info("  Loading from local storage...")
        detector = persistence.load_autoencoder(model_name)
        logger.info(f"  Loaded model with threshold={detector._threshold:.4f}")
    except Exception as e:
        logger.warning(f"  Failed to load model: {e}")
        return None
    
    # Load test data
    logger.info(f"  Loading test file: {os.path.basename(test_filepath)}")
    df_test = load_parquet_file(test_filepath)
    n_rows = len(df_test)
    n_products = df_test["product_id"].nunique()
    logger.info(f"    Test data: {n_rows:,} rows, {n_products:,} products")
    
    # Evaluate with production-like anomaly injection
    logger.info("  Evaluating with production-like anomaly injection...")
    eval_start = datetime.now(timezone.utc)
    eval_result = evaluate_production_like(detector, df_test)
    eval_time = (datetime.now(timezone.utc) - eval_start).total_seconds()
    
    total_time = (datetime.now(timezone.utc) - start_time).total_seconds()
    
    logger.info(
        f"  P={eval_result['precision']:.1%}, R={eval_result['recall']:.1%}, "
        f"F1={eval_result['f1_score']:.1%} ({eval_time:.1f}s)"
    )
    
    return TrainingResult(
        model_name=model_name,
        granularity=granularity,
        n_samples=n_rows,
        n_products=n_products,
        training_time_sec=total_time,
        threshold=detector._threshold,
        mean_reconstruction_error=0.0,  # Not available for loaded models
        precision=eval_result["precision"],
        recall=eval_result["recall"],
        f1_score=eval_result["f1_score"],
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train autoencoder models from Parquet files"
    )
    parser.add_argument(
        "--granularity",
        type=str,
        choices=["country_segment", "competitor", "both"],
        default="both",
        help="Model granularity to train (default: both)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/training/derived",
        help="Data directory (default: data/training/derived)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate models with synthetic anomaly injection",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save trained models",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files without training",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Don't use cached features, recompute from parquet",
    )
    parser.add_argument(
        "--model-filter",
        type=str,
        default=None,
        help="Only train models matching this substring",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip models that already exist in the local model directory",
    )
    parser.add_argument(
        "--file-suffix",
        type=str,
        default="",
        help="Filter files by suffix (e.g., '_train_mh5' for *_train_mh5.parquet files)",
    )
    parser.add_argument(
        "--test-suffix",
        type=str,
        default=None,
        help="Separate suffix for test files (e.g., '_train' to test mh5 models on unfiltered data). "
             "If not set, uses same suffix as --file-suffix.",
    )
    parser.add_argument(
        "--auto-tune",
        action="store_true",
        help="[DEPRECATED] Fast approximate threshold tuning. Uses vectorized feature extraction "
             "with look-ahead bias. For production-accurate tuning, use tune_autoencoder.py instead.",
    )
    parser.add_argument(
        "--tune-steps",
        type=int,
        default=30,
        help="Number of threshold percentile values to test (default: 30)",
    )
    parser.add_argument(
        "--tune-trials",
        type=int,
        default=10,
        help="Number of trials for robust evaluation (default: 10)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Load existing local models and evaluate them without retraining",
    )
    parser.add_argument(
        "--model-root",
        type=str,
        default="artifacts/models",
        help="Local model root for saved models (default: artifacts/models)",
    )
    args = parser.parse_args()
    
    # Load environment
    load_dotenv()
    
    print("=" * 70)
    print("Autoencoder Training")
    print("=" * 70)
    print(f"Data path: {args.data_path}")
    print(f"Granularity: {args.granularity}")
    print(f"Evaluate: {args.evaluate}")
    print(f"Save models: {not args.no_save}")
    print(f"Use cache: {not args.no_cache}")
    print(f"Dry run: {args.dry_run}")
    print(f"Resume (skip existing): {args.resume}")
    print(f"Model root: {args.model_root}")
    if args.model_filter:
        print(f"Model filter: {args.model_filter}")
    if args.file_suffix:
        print(f"File suffix filter: {args.file_suffix}")
    if args.test_suffix is not None:
        print(f"Test suffix (separate): {args.test_suffix}")
    if args.auto_tune:
        print(f"Auto-tune: {args.auto_tune}")
        print(f"  Tune steps: {args.tune_steps}")
        print(f"  Tune trials: {args.tune_trials}")
        print()
        print("  [WARNING] --auto-tune is deprecated. It uses vectorized feature extraction")
        print("            with look-ahead bias. For production-accurate thresholds, use:")
        print("            python scripts/tune_autoencoder.py --file-suffix \"_test_new_prices\"")
    if args.eval_only:
        print("Eval-only: True (loading models from the local model directory)")
    print("=" * 70)
    print()
    
    # Determine granularities to process
    if args.granularity == "both":
        granularities = ["country_segment", "competitor"]
    else:
        granularities = [args.granularity]
    
    # Initialize persistence if saving, resuming, or eval-only
    persistence = None
    if (not args.no_save or args.resume or args.eval_only) and not args.dry_run:
        persistence = ModelPersistence(model_root=args.model_root)
        if args.eval_only:
            print(f"Loading models from: {persistence.models_root_description}")
        elif not args.no_save:
            print(f"Models will be saved to: {persistence.models_root_description}")
        if args.resume:
            print("Resume mode: checking for existing models in the local model directory")
        print()
    
    all_results: list[TrainingResult] = []
    
    for granularity in granularities:
        print(f"\n{'='*60}")
        print(f"Granularity: {granularity}")
        print(f"{'='*60}")
        
        # Find files
        files = find_parquet_files(args.data_path, granularity, args.file_suffix)
        
        if not files:
            print(f"No Parquet files found in {args.data_path}/by_{granularity}/")
            continue
        
        print(f"Found {len(files)} Parquet files:")
        for f in files:
            model_name = extract_model_name(f)
            print(f"  - {model_name}")
        
        if args.dry_run:
            continue
        
        # Process each model
        skipped_existing = 0
        for filepath in files:
            model_name = extract_model_name(filepath)
            
            # Apply filter if specified
            if args.model_filter and args.model_filter not in model_name:
                logger.info(f"Skipping {model_name} (doesn't match filter)")
                continue
            
            # Find test file (required for eval-only, optional otherwise)
            test_filepath = None
            if args.test_suffix is not None:
                test_filepath = find_matching_test_file(filepath, args.test_suffix, args.data_path)
                if test_filepath is None:
                    logger.warning(f"  No test file found for {model_name} with suffix '{args.test_suffix}'")
                    if args.eval_only:
                        continue  # Skip if eval-only and no test file
            
            # Eval-only mode: load the saved local model and evaluate it
            if args.eval_only:
                if not persistence:
                    logger.error("Persistence required for --eval-only mode")
                    continue
                if not test_filepath:
                    logger.warning(f"  Skipping {model_name}: --eval-only requires --test-suffix with valid test file")
                    continue
                    
                result = eval_saved_model(
                    model_name=model_name,
                    granularity=granularity,
                    test_filepath=test_filepath,
                    persistence=persistence,
                    use_cache=not args.no_cache,
                )
            else:
                # Normal training mode
                # Check if model already exists (resume mode)
                if args.resume and persistence:
                    if persistence.model_exists(model_name, "autoencoder"):
                        logger.info(f"Skipping {model_name} (already exists locally)")
                        skipped_existing += 1
                        continue

                result = train_single_model(
                    filepath=filepath,
                    model_name=model_name,
                    granularity=granularity,
                    evaluate=args.evaluate,
                    save_models=not args.no_save,
                    persistence=persistence,
                    use_cache=not args.no_cache,
                    test_filepath=test_filepath,
                    auto_tune=args.auto_tune,
                    tune_steps=args.tune_steps,
                    tune_trials=args.tune_trials,
                )
            
            if result:
                all_results.append(result)
        
        if skipped_existing > 0:
            print(f"\n  Skipped {skipped_existing} existing models (--resume)")
    
    # Summary
    if all_results:
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY" if args.eval_only else "TRAINING SUMMARY")
        print("=" * 70)
        
        # Group by granularity
        by_granularity: dict[str, list[TrainingResult]] = {}
        for r in all_results:
            if r.granularity not in by_granularity:
                by_granularity[r.granularity] = []
            by_granularity[r.granularity].append(r)
        
        for granularity, results in by_granularity.items():
            print(f"\n{granularity} ({len(results)} models):")
            print("-" * 50)
            
            for r in results:
                eval_str = ""
                if r.f1_score is not None:
                    eval_str = f", F1={r.f1_score:.1%}"
                print(
                    f"  {r.model_name}: {r.n_samples:,} samples, "
                    f"{r.n_products:,} products, "
                    f"threshold={r.threshold:.4f}{eval_str}"
                )
            
            # Averages
            if results[0].f1_score is not None:
                avg_precision = sum(r.precision for r in results) / len(results)
                avg_recall = sum(r.recall for r in results) / len(results)
                avg_f1 = sum(r.f1_score for r in results) / len(results)
                print(f"\n  Average: P={avg_precision:.1%}, R={avg_recall:.1%}, F1={avg_f1:.1%}")
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
