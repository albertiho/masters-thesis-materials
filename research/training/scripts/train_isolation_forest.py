#!/usr/bin/env python3
"""Train Isolation Forest models from local Parquet files.

This script trains price anomaly detection Isolation Forest models at three granularity levels:
1. country_segment: One model per country+segment (8 models)
2. competitor: One model per competitor (N models)
3. global: One model across all competitors/products for a dataset snapshot

The script reads from local Parquet files and saves trained models under the
local artifacts directory.
Thresholds and hyperparameters are fixed to grid-search optimal values (no per-model tuning).

Usage:
    # Train all models (both granularities)
    python research/training/scripts/train_isolation_forest.py
    
    # Train only country_segment models
    python research/training/scripts/train_isolation_forest.py --granularity country_segment
    
    # Resume training (skip models that already exist locally)
    python research/training/scripts/train_isolation_forest.py --resume
    
    # Force recompute features (ignore cache)
    python research/training/scripts/train_isolation_forest.py --no-cache
    
    # Dry run (list files, don't train)
    python research/training/scripts/train_isolation_forest.py --dry-run
    
    # Train only specific model
    python research/training/scripts/train_isolation_forest.py --model-filter "DK_B2C"

Architecture:
    Parquet Files -> Feature Extraction (cached) -> Isolation Forest Training -> Local Models
    
    Feature matrices are cached as .iforest_features.npz files for fast subsequent runs.
    Each model type uses a distinct cache extension to prevent conflicts:
        - Autoencoder: .autoencoder_features.npz (9 features)
        - Isolation Forest: .iforest_features.npz (12 features)

Threshold:
    Uses a fixed anomaly_threshold from grid search (no per-model tuning).

Fixed Parameters:
    - contamination: "auto" (data is clean)
    - n_estimators: 200
    - max_samples: 512
    - max_features: 0.75
    - baseline_threshold: 0.4 (fixed)
    - file_suffix: "_train_mh{min_history}" (from tuning_config)

Key Differences from Autoencoder:
    - Uses 12 features (vs 9 for autoencoder) - includes rolling_min, rolling_max, price_range_position
    - Scikit-learn based (faster training, no GPU needed)
    - Uses contamination parameter to set expected anomaly rate
    - anomaly_threshold controls the detection cutoff
"""

import argparse
import glob
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

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
from src.anomaly.ml.tree_features import (
    TREE_FEATURE_NAMES,
    TREE_FEATURE_SCHEMA_VERSION,
    extract_tree_features_vectorized,
    infer_tree_training_valid_mask,
)
from src.features.temporal import DEFAULT_HISTORY_DEPTH
from src.tuning_config import get_min_history

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Rolling window size - must match DEFAULT_HISTORY_DEPTH in temporal.py (30 observations)
# This ensures training computes statistics the same way as inference.
ROLLING_WINDOW_SIZE = DEFAULT_HISTORY_DEPTH

FEATURE_SCHEMA_VERSION = TREE_FEATURE_SCHEMA_VERSION
FEATURE_NAMES = list(TREE_FEATURE_NAMES)

DATE_TOKEN_PATTERN = re.compile(r"_(\d{4}-\d{2}-\d{2})(?:_|$)")
EXPECTED_COUNTRY_SEGMENT_MODELS = 8


@dataclass(frozen=True)
class OptimalIForestConfig:
    """Isolation Forest hyperparameters from grid search."""

    n_estimators: int
    max_samples: int | str
    max_features: float
    anomaly_threshold: float
    contamination: str | float


OPTIMAL_CONFIG = OptimalIForestConfig(
    n_estimators=200,
    max_samples=512,
    max_features=0.75,
    anomaly_threshold=0.4,
    contamination="auto",
)


@dataclass
class TrainingResult:
    """Result from training a single model."""

    model_name: str
    granularity: str
    n_samples: int
    n_products: int
    training_time_sec: float
    contamination: str | float
    anomaly_threshold: float
    # Evaluation metrics (optional)
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    model_uri: str | None = None


def find_parquet_files(data_path: str, granularity: str, file_suffix: str = "") -> list[str]:
    """Find Parquet files for the specified granularity.

    Args:
        data_path: Base data directory.
        granularity: 'country_segment', 'competitor', or 'global'
        file_suffix: Optional suffix filter (e.g., '_train' to match only *_train.parquet)

    Returns:
        List of Parquet file paths
    """
    if granularity == "country_segment":
        subdir = "by_country_segment"
    elif granularity == "competitor":
        subdir = "by_competitor"
    elif granularity == "global":
        subdir = "global"
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
    base_name = extract_base_name(train_file)
    parent_dir = os.path.dirname(train_file)

    # Build expected test filename
    test_filename = f"{base_name}{test_suffix}.parquet"
    test_path = os.path.join(parent_dir, test_filename)

    if os.path.exists(test_path):
        return test_path

    # Try with glob in case of slight naming variations
    pattern = os.path.join(parent_dir, f"*{test_suffix}.parquet")
    matches = [f for f in glob.glob(pattern) if extract_base_name(f) == base_name]
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
    """
    filename = os.path.basename(filepath)
    name = filename.replace(".parquet", "")

    # Extract experiment suffix (e.g., _mh5, _mh4) if present after _train or _test
    suffix = ""
    suffix_match = re.search(r"_(?:train|test(?:_new_products|_new_prices)?)(_[a-z0-9]+)$", name)
    if suffix_match:
        suffix = suffix_match.group(1)  # e.g., "_mh5"
        name = name[: suffix_match.start()]
    else:
        # Remove _train or _test suffix if present (no experiment suffix)
        train_test_match = re.search(r"_(?:train|test(?:_new_products|_new_prices)?)$", name)
        if train_test_match:
            name = name[: train_test_match.start()]

    # Remove date suffix like _2026-01-18
    date_pattern = r"_\d{4}-\d{2}-\d{2}$"
    name = re.sub(date_pattern, "", name)

    return name + suffix


def extract_date_token(filepath: str) -> datetime | None:
    """Extract YYYY-MM-DD token from filename for recency selection."""
    filename = os.path.basename(filepath)
    match = DATE_TOKEN_PATTERN.search(filename)
    if not match:
        return None
    date_str = match.group(1)
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        logger.warning("Invalid date token in parquet filename: %s", filename)
        return None


def select_latest_per_model(files: list[str], granularity: str) -> list[str]:
    """Select the latest dated Parquet file per model name."""
    if not files:
        return []

    grouped: dict[str, list[tuple[datetime | None, str]]] = {}
    missing_date_files: list[str] = []

    for filepath in files:
        model_name = extract_model_name(filepath)
        date_token = extract_date_token(filepath)
        if date_token is None:
            missing_date_files.append(filepath)
        grouped.setdefault(model_name, []).append((date_token, filepath))

    for filepath in missing_date_files:
        logger.warning(
            "Missing date token in parquet filename: %s",
            os.path.basename(filepath),
        )

    selected: list[str] = []
    for model_name, candidates in grouped.items():
        dated_candidates = [c for c in candidates if c[0] is not None]
        if dated_candidates:
            latest_date, latest_path = max(dated_candidates, key=lambda item: item[0])
            logger.info(
                "Selected latest file for %s: %s (%s)",
                model_name,
                os.path.basename(latest_path),
                latest_date.date(),
            )
            selected.append(latest_path)
        else:
            fallback = sorted([c[1] for c in candidates])[-1]
            logger.warning(
                "No dated files found for model %s; falling back to %s",
                model_name,
                os.path.basename(fallback),
            )
            selected.append(fallback)

    selected = sorted(selected)
    logger.info(
        "Selected %d latest files from %d candidates for %s",
        len(selected),
        len(files),
        granularity,
    )

    if granularity == "country_segment" and len(selected) != EXPECTED_COUNTRY_SEGMENT_MODELS:
        logger.warning(
            "Expected %d country_segment models, found %d",
            EXPECTED_COUNTRY_SEGMENT_MODELS,
            len(selected),
        )

    return selected


def get_cache_path(parquet_path: str) -> str:
    """Get the cache file path for Isolation Forest features.

    Cache files are stored alongside parquet files with model-specific extensions
    to prevent conflicts between different models (autoencoder vs isolation forest).

    Cache extensions by model:
        - Autoencoder: .autoencoder_features.npz (9 features)
        - Isolation Forest: .iforest_features.npz (12 features)
    """
    return parquet_path.replace(".parquet", ".iforest_features.npz")


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
    """Extract the shared tree feature matrix used by IF, EIF, and RRCF."""
    return extract_tree_features_vectorized(df, window_size=window_size)


def train_from_matrix(
    X: np.ndarray,
    contamination: str | float = "auto",
    anomaly_threshold: float = 0.4,  # Optimal: 0.4 (was 0.6)
    n_estimators: int = 200,         # Optimal: 200 (was 100)
    max_samples: int = 512,          # Optimal: 512 (was "auto")
    max_features: float = 0.75,      # Optimal: 0.75 (was 1.0)
    random_state: int = 42,
) -> tuple["IsolationForestDetector", float]:
    """Train Isolation Forest directly from feature matrix.

    This bypasses the slow dataclass conversion in detector.fit().

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        contamination: Expected proportion of anomalies ('auto' for clean data, or 0-0.5)
        anomaly_threshold: Score threshold for flagging anomalies (0-1)
        n_estimators: Number of trees in the forest
        max_samples: Number of samples to train each tree ('auto' = min(256, n_samples))
        max_features: Fraction of features to use per tree (0-1)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (detector, anomaly_threshold)
    """
    from src.anomaly.ml.isolation_forest import IsolationForestConfig, IsolationForestDetector

    # Filter out rows that do not satisfy the shared tree detector validity policy.
    valid_mask = infer_tree_training_valid_mask(X)
    X_valid = X[valid_mask]

    if len(X_valid) < 50:
        raise ValueError(f"Need at least 50 valid samples, got {len(X_valid)}")

    # Create config
    config = IsolationForestConfig(
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        contamination=contamination,
        anomaly_threshold=anomaly_threshold,
        random_state=random_state,
    )

    # Create detector
    detector = IsolationForestDetector(config)

    detector.fit_from_matrix(X_valid, feature_names=FEATURE_NAMES)

    logger.info(
        f"  Score normalization: offset={detector._score_offset:.4f}, scale={detector._score_scale:.4f}"
    )

    return detector, anomaly_threshold


def train_single_model(
    filepath: str,
    model_name: str,
    granularity: str,
    contamination: str | float = OPTIMAL_CONFIG.contamination,
    n_estimators: int = OPTIMAL_CONFIG.n_estimators,
    max_samples: int | str = OPTIMAL_CONFIG.max_samples,
    max_features: float = OPTIMAL_CONFIG.max_features,
    anomaly_threshold: float = OPTIMAL_CONFIG.anomaly_threshold,
    save_models: bool = True,
    persistence: ModelPersistence | None = None,
    use_cache: bool = True,
) -> TrainingResult | None:
    """Train a single Isolation Forest model from a Parquet file.

    Uses cached feature matrices when available for faster subsequent runs.
    Does not perform threshold tuning; uses fixed threshold from grid search.

    Args:
        filepath: Path to Parquet file (training data)
        model_name: Name for the model
        granularity: 'country_segment' or 'competitor'
        contamination: Expected anomaly rate ('auto' for clean data)
        n_estimators: Number of trees
        max_samples: Number of samples to train each tree
        max_features: Fraction of features to use per tree
        anomaly_threshold: Fixed threshold from grid search
        save_models: Whether to save models locally
        persistence: ModelPersistence instance for saving
        use_cache: Whether to use cached features (default: True)

    Returns:
        TrainingResult or None if training failed
    """
    try:
        from sklearn.ensemble import IsolationForest  # noqa: F401
    except ImportError:
        logger.error("scikit-learn not available. Install with: pip install scikit-learn")
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
            logger.info(f"  Using cached features ({X.shape[0]:,} samples, {X.shape[1]} features)")

    # Extract features if not cached
    if X is None:
        logger.info(f"  Extracting features (vectorized, 12 features)...")
        extract_start = datetime.now(timezone.utc)
        X = extract_features_vectorized(df)
        extract_time = (datetime.now(timezone.utc) - extract_start).total_seconds()
        logger.info(f"  Extracted {X.shape[0]:,} x {X.shape[1]} features in {extract_time:.1f}s")

        # Cache for next time
        if use_cache:
            save_cached_features(cache_path, X)

    # Train Isolation Forest
    try:
        train_start = datetime.now(timezone.utc)
        detector, threshold = train_from_matrix(
            X,
            contamination=contamination,
            anomaly_threshold=anomaly_threshold,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
        )
        train_time = (datetime.now(timezone.utc) - train_start).total_seconds()
        logger.info(
            f"  Trained in {train_time:.1f}s, contamination={contamination}"
        )
    except Exception as e:
        logger.warning(f"  Training failed: {e}")
        return None

    threshold = detector.config.anomaly_threshold
    logger.info(f"  Using fixed threshold: {threshold:.2f}")

    total_time = (datetime.now(timezone.utc) - start_time).total_seconds()

    result = TrainingResult(
        model_name=model_name,
        granularity=granularity,
        n_samples=n_rows,
        n_products=n_products,
        training_time_sec=total_time,
        contamination=contamination,
        anomaly_threshold=threshold,
    )

    # Save model
    if save_models and persistence:
        try:
            model_uri = persistence.save_isolation_forest(
                detector,
                model_name,
                n_rows,
            )
            result.model_uri = model_uri
            logger.info(f"  Saved to {model_uri}")
        except Exception as e:
            logger.warning(f"  Failed to save model: {e}")

    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train Isolation Forest models from Parquet files")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/training/derived",
        help="Data directory containing by_competitor/, by_country_segment/, and global/ (default: data/training/derived)",
    )
    parser.add_argument(
        "--granularity",
        type=str,
        choices=["country_segment", "competitor", "global", "both"],
        default="both",
        help="Model granularity to train (default: both)",
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
        "--model-root",
        type=str,
        default="artifacts/models",
        help="Local model root for saved models (default: artifacts/models)",
    )
    args = parser.parse_args()

    # Fixed parameters (no longer configurable)
    # Optimal hyperparameters from grid search (F1: 62.2%)
    DATA_PATH = args.data_path
    MIN_HISTORY = get_min_history("isolation_forest")
    FILE_SUFFIX = f"_train_mh{MIN_HISTORY}"
    CONTAMINATION = OPTIMAL_CONFIG.contamination
    N_ESTIMATORS = OPTIMAL_CONFIG.n_estimators
    MAX_SAMPLES = OPTIMAL_CONFIG.max_samples
    MAX_FEATURES = OPTIMAL_CONFIG.max_features
    BASELINE_THRESHOLD = OPTIMAL_CONFIG.anomaly_threshold

    # Load environment
    load_dotenv()

    print("=" * 70)
    print("Isolation Forest Training")
    print("=" * 70)
    print(f"Data path: {DATA_PATH}")
    print(f"Granularity: {args.granularity}")
    print(f"Min history: {MIN_HISTORY}")
    print(f"Contamination: {CONTAMINATION}")
    print(f"N estimators: {N_ESTIMATORS}")
    print(f"Max samples: {MAX_SAMPLES}")
    print(f"Max features: {MAX_FEATURES}")
    print(f"Baseline threshold: {BASELINE_THRESHOLD}")
    print("Threshold tuning: disabled (fixed threshold)")
    print(f"File suffix: {FILE_SUFFIX}")
    print(f"Save models: {not args.no_save}")
    print(f"Use cache: {not args.no_cache}")
    print(f"Dry run: {args.dry_run}")
    print(f"Resume (skip existing): {args.resume}")
    print(f"Model root: {args.model_root}")
    if args.model_filter:
        print(f"Model filter: {args.model_filter}")
    print("=" * 70)
    print()

    # Determine granularities to process
    if args.granularity == "both":
        granularities = ["country_segment", "competitor", "global"]
    else:
        granularities = [args.granularity]

    # Initialize persistence if saving or resuming
    persistence = None
    if (not args.no_save or args.resume) and not args.dry_run:
        persistence = ModelPersistence(model_root=args.model_root)
        if not args.no_save:
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
        candidate_files = find_parquet_files(DATA_PATH, granularity, FILE_SUFFIX)

        if not candidate_files:
            raise FileNotFoundError(
                f"No Parquet files found for granularity {granularity!r} under {DATA_PATH} with suffix '{FILE_SUFFIX}'"
            )

        files = select_latest_per_model(candidate_files, granularity)

        print(f"Found {len(candidate_files)} Parquet files, selected {len(files)} latest:")
        for f in files:
            model_name = extract_model_name(f)
            print(f"  - {model_name} ({os.path.basename(f)})")

        if args.dry_run:
            continue

        # Train each model
        skipped_existing = 0
        for filepath in files:
            model_name = extract_model_name(filepath)

            # Apply filter if specified
            if args.model_filter and args.model_filter not in model_name:
                logger.info(f"Skipping {model_name} (doesn't match filter)")
                continue

            # Check if model already exists (resume mode)
            if args.resume and persistence:
                if persistence.model_exists(model_name, "isolation_forest"):
                    logger.info(f"Skipping {model_name} (already exists locally)")
                    skipped_existing += 1
                    continue

            result = train_single_model(
                filepath=filepath,
                model_name=model_name,
                granularity=granularity,
                contamination=CONTAMINATION,
                n_estimators=N_ESTIMATORS,
                max_samples=MAX_SAMPLES,
                max_features=MAX_FEATURES,
                anomaly_threshold=BASELINE_THRESHOLD,
                save_models=not args.no_save,
                persistence=persistence,
                use_cache=not args.no_cache,
            )

            if result:
                all_results.append(result)

        if skipped_existing > 0:
            print(f"\n  Skipped {skipped_existing} existing models (--resume)")

    # Summary
    if all_results:
        print("\n" + "=" * 70)
        print("TRAINING SUMMARY")
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
                f1_display = f"{r.f1_score:.1%}" if r.f1_score is not None else "n/a"
                print(
                    f"  {r.model_name}: {r.n_samples:,} samples, "
                    f"{r.n_products:,} products, "
                    f"threshold={r.anomaly_threshold:.2f}, F1={f1_display}"
                )

            if all(r.f1_score is not None for r in results):
                avg_precision = sum(r.precision for r in results) / len(results)
                avg_recall = sum(r.recall for r in results) / len(results)
                avg_f1 = sum(r.f1_score for r in results) / len(results)
                print(f"\n  Average: P={avg_precision:.1%}, R={avg_recall:.1%}, F1={avg_f1:.1%}")
            else:
                print("\n  Average metrics: unavailable (threshold tuning disabled)")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
