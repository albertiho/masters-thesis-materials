#!/usr/bin/env python3
"""Train local RRCF detector models from Parquet feature matrices."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from dotenv import load_dotenv

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_script_dir)))
sys.path.insert(0, _project_root)
sys.path.insert(0, _script_dir)

from src.anomaly.ml.rrcf import RRCFDetector, RRCFDetectorConfig
from src.anomaly.ml.tree_features import infer_tree_training_valid_mask
from src.anomaly.persistence import ModelPersistence
from src.tuning_config import get_min_history
from train_isolation_forest import (
    TrainingResult,
    extract_features_vectorized,
    extract_model_name,
    find_parquet_files,
    get_cache_path,
    load_cached_features,
    load_parquet_file,
    save_cached_features,
    select_latest_per_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptimalRRCFConfig:
    num_trees: int
    tree_size: int
    anomaly_threshold: float
    warmup_samples: int


OPTIMAL_CONFIG = OptimalRRCFConfig(
    num_trees=40,
    tree_size=256,
    anomaly_threshold=0.8,
    warmup_samples=32,
)


def train_from_matrix(
    X: np.ndarray,
    *,
    num_trees: int = OPTIMAL_CONFIG.num_trees,
    tree_size: int = OPTIMAL_CONFIG.tree_size,
    anomaly_threshold: float = OPTIMAL_CONFIG.anomaly_threshold,
    warmup_samples: int = OPTIMAL_CONFIG.warmup_samples,
    random_state: int = 42,
) -> tuple[RRCFDetector, float]:
    """Train RRCF directly from a shared tree feature matrix."""
    valid_mask = infer_tree_training_valid_mask(X)
    X_valid = X[valid_mask]
    if len(X_valid) < 50:
        raise ValueError(f"Need at least 50 valid samples, got {len(X_valid)}")

    detector = RRCFDetector(
        RRCFDetectorConfig(
            num_trees=num_trees,
            tree_size=tree_size,
            anomaly_threshold=anomaly_threshold,
            random_state=random_state,
            warmup_samples=warmup_samples,
        )
    )
    detector.fit_from_matrix(X_valid)
    return detector, anomaly_threshold


def train_single_model(
    filepath: str,
    model_name: str,
    granularity: str,
    *,
    num_trees: int = OPTIMAL_CONFIG.num_trees,
    tree_size: int = OPTIMAL_CONFIG.tree_size,
    anomaly_threshold: float = OPTIMAL_CONFIG.anomaly_threshold,
    warmup_samples: int = OPTIMAL_CONFIG.warmup_samples,
    save_models: bool = True,
    persistence: ModelPersistence | None = None,
    use_cache: bool = True,
) -> TrainingResult | None:
    logger.info("Training RRCF model: %s", model_name)
    start_time = datetime.now(timezone.utc)

    df = load_parquet_file(filepath)
    n_rows = len(df)
    n_products = df["product_id"].nunique()
    if n_rows < 100:
        logger.warning("Skipping %s: insufficient data (%d rows)", model_name, n_rows)
        return None

    cache_path = get_cache_path(filepath)
    X = load_cached_features(cache_path) if use_cache else None
    if X is None:
        X = extract_features_vectorized(df)
        if use_cache:
            save_cached_features(cache_path, X)

    try:
        detector, threshold = train_from_matrix(
            X,
            num_trees=num_trees,
            tree_size=tree_size,
            anomaly_threshold=anomaly_threshold,
            warmup_samples=warmup_samples,
        )
    except Exception as exc:
        logger.warning("RRCF training failed for %s: %s", model_name, exc)
        return None

    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
    result = TrainingResult(
        model_name=model_name,
        granularity=granularity,
        n_samples=n_rows,
        n_products=n_products,
        training_time_sec=elapsed,
        contamination="n/a",
        anomaly_threshold=threshold,
    )

    if save_models and persistence is not None:
        try:
            result.model_uri = persistence.save_rrcf(detector, model_name, n_rows)
        except Exception as exc:
            logger.warning("Failed to save RRCF model %s: %s", model_name, exc)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RRCF detector models from Parquet files")
    parser.add_argument("--data-path", type=str, default="data/training/derived")
    parser.add_argument("--granularity", type=str, choices=["country_segment", "competitor", "global", "both"], default="both")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--model-filter", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model-root", type=str, default="artifacts/models")
    args = parser.parse_args()

    load_dotenv()
    min_history = get_min_history("isolation_forest")
    file_suffix = f"_train_mh{min_history}"
    granularities = ["country_segment", "competitor", "global"] if args.granularity == "both" else [args.granularity]

    print("=" * 70)
    print("RRCF Training")
    print("=" * 70)
    print(f"Data path: {args.data_path}")
    print(f"Granularity: {args.granularity}")
    print(f"Min history: {min_history}")
    print(f"Num trees: {OPTIMAL_CONFIG.num_trees}")
    print(f"Tree size: {OPTIMAL_CONFIG.tree_size}")
    print(f"Warmup samples: {OPTIMAL_CONFIG.warmup_samples}")
    print(f"Baseline threshold: {OPTIMAL_CONFIG.anomaly_threshold}")
    print(f"File suffix: {file_suffix}")
    print(f"Save models: {not args.no_save}")
    print(f"Use cache: {not args.no_cache}")
    print(f"Dry run: {args.dry_run}")
    print(f"Resume (skip existing): {args.resume}")
    print(f"Model root: {args.model_root}")
    if args.model_filter:
        print(f"Model filter: {args.model_filter}")
    print("=" * 70)

    persistence = None
    if (not args.no_save or args.resume) and not args.dry_run:
        persistence = ModelPersistence(model_root=args.model_root)
        print(f"Models will be saved to: {persistence.models_root_description}")

    all_results: list[TrainingResult] = []
    for granularity in granularities:
        candidate_files = find_parquet_files(args.data_path, granularity, file_suffix)
        if not candidate_files:
            raise FileNotFoundError(
                f"No Parquet files found for granularity {granularity!r} under "
                f"{args.data_path} with suffix '{file_suffix}'"
            )

        files = select_latest_per_model(candidate_files, granularity)
        print(f"\nGranularity: {granularity}")
        print(f"Found {len(candidate_files)} Parquet files, selected {len(files)} latest")
        for filepath in files:
            model_name = extract_model_name(filepath)
            print(f"  - {model_name} ({os.path.basename(filepath)})")

        if args.dry_run:
            continue

        skipped_existing = 0
        for filepath in files:
            model_name = extract_model_name(filepath)
            if args.model_filter and args.model_filter not in model_name:
                continue
            if args.resume and persistence and persistence.model_exists(model_name, "rrcf"):
                skipped_existing += 1
                continue

            result = train_single_model(
                filepath=filepath,
                model_name=model_name,
                granularity=granularity,
                num_trees=OPTIMAL_CONFIG.num_trees,
                tree_size=OPTIMAL_CONFIG.tree_size,
                anomaly_threshold=OPTIMAL_CONFIG.anomaly_threshold,
                warmup_samples=OPTIMAL_CONFIG.warmup_samples,
                save_models=not args.no_save,
                persistence=persistence,
                use_cache=not args.no_cache,
            )
            if result is not None:
                all_results.append(result)

        if skipped_existing:
            print(f"  Skipped {skipped_existing} existing RRCF models (--resume)")

    if all_results:
        print("\n" + "=" * 70)
        print("TRAINING SUMMARY")
        print("=" * 70)
        for result in all_results:
            print(
                f"{result.model_name}: {result.n_samples:,} samples, "
                f"{result.n_products:,} products, "
                f"threshold={result.anomaly_threshold:.2f}"
            )

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
