#!/usr/bin/env python3
"""Tune Autoencoder thresholds using held-out test data.

This script finds optimal thresholds for trained Autoencoder models by
evaluating them on test data with synthetic anomaly injection.

Key Features:
    - Production-like feature extraction via DetectorEvaluator
    - Anomaly injection at DataFrame level (before feature extraction)
    - Sequential processing eliminates look-ahead bias
    - Parallel trial execution for ~5x speedup
    - Log-space threshold search (appropriate for reconstruction errors)
    - Supports both "known products" and "new products" scenarios

The script:
1. Loads trained models from local storage
2. Loads test data from local Parquet files
3. Optionally loads train data to populate price history cache
4. Injects synthetic anomalies at the raw price level
5. Processes sequentially through production feature extraction
6. Grid searches over threshold values (parallel trials)
7. Selects threshold that maximizes target metric (F1 by default)
8. Updates saved local model metadata (unless --dry-run)

Usage:
    # Tune all models using test data (known products scenario)
    python scripts/tune_autoencoder.py --file-suffix "_test"

    # Cold-start scenario (no price history)
    python scripts/tune_autoencoder.py --file-suffix "_test" --cold-start

    # Dry run (show results without updating saved models)
    python scripts/tune_autoencoder.py --file-suffix "_test" --dry-run

    # Custom threshold range
    python scripts/tune_autoencoder.py --file-suffix "_test" --min-threshold 0.001 --max-threshold 0.5
"""

import argparse
import io
import logging
import os
import sys

import numpy as np

# Add project root to path (4 levels up from research/training/scripts/)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_script_dir)))
sys.path.insert(0, _project_root)
sys.path.insert(0, _script_dir)  # For importing sibling scripts

try:
    import pandas as pd
except ImportError:
    print("Missing dependency: pandas")
    print("Install with: pip install pandas")
    sys.exit(1)

from dotenv import load_dotenv

from tuning_utils import (
    TuningResult,
    extract_model_name,
    find_parquet_files,
    find_train_file,
    run_tuning_trials,
)
from src.anomaly.persistence import ModelPersistence

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def update_model_threshold(
    persistence: ModelPersistence,
    model_name: str,
    new_threshold: float,
) -> str:
    """Update the threshold in a saved local model.

    Args:
        persistence: Model persistence instance.
        model_name: Model identifier.
        new_threshold: New threshold value.

    Returns:
        Absolute path of the updated model.
    """
    import torch

    # Update the model file
    model_path = persistence._get_model_path(model_name, "autoencoder", "model.pt")
    model_bytes = persistence._download_bytes(model_path)

    buffer = io.BytesIO(model_bytes)
    saved_data = torch.load(buffer, map_location="cpu", weights_only=False)

    # Update threshold in saved data
    old_threshold = saved_data.get("threshold", "unknown")
    saved_data["threshold"] = new_threshold

    # Re-upload model
    buffer = io.BytesIO()
    torch.save(saved_data, buffer)
    model_uri = persistence._upload_bytes(buffer.getvalue(), model_path)

    logger.info(f"  Updated threshold: {old_threshold:.6f} -> {new_threshold:.6f}")

    return model_uri


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Tune Autoencoder anomaly detection thresholds")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/training",
        help="Data directory (default: data/training)",
    )
    parser.add_argument(
        "--file-suffix",
        type=str,
        default="_test",
        help="Suffix for test files (default: _test)",
    )
    parser.add_argument(
        "--granularity",
        type=str,
        choices=["country_segment", "competitor", "both"],
        default="both",
        help="Model granularity to tune (default: both)",
    )
    parser.add_argument(
        "--min-threshold",
        type=float,
        default=0.001,
        help="Minimum threshold to test (default: 0.001)",
    )
    parser.add_argument(
        "--max-threshold",
        type=float,
        default=0.5,
        help="Maximum threshold to test (default: 0.5)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of threshold values to test (default: 30)",
    )
    parser.add_argument(
        "--target-metric",
        type=str,
        default="f1",
        choices=["f1", "precision", "recall"],
        help="Metric to optimize (default: f1)",
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.3,
        help="Minimum acceptable precision when optimizing recall (default: 0.3)",
    )
    parser.add_argument(
        "--injection-rate",
        type=float,
        default=0.1,
        help="Fraction of data to inject as anomalies (default: 0.1)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=5,
        help="Number of trials with different seeds (default: 5)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Max parallel workers for trials (default: n_trials)",
    )
    parser.add_argument(
        "--cold-start",
        action="store_true",
        help="Evaluate in cold-start mode (no price history)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't update saved models, just show results",
    )
    parser.add_argument(
        "--model-filter",
        type=str,
        default=None,
        help="Only tune models matching this substring",
    )
    parser.add_argument(
        "--model-suffix",
        type=str,
        default="",
        help="Suffix to append when loading an alternate local model name (e.g., '_2026-01-18')",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed results for each threshold",
    )
    args = parser.parse_args()

    # Load environment
    load_dotenv()

    print("=" * 70)
    print("Autoencoder Threshold Tuning (Parallel Trials)")
    print("=" * 70)
    print(f"Data path: {args.data_path}")
    print(f"File suffix: {args.file_suffix}")
    print(f"Granularity: {args.granularity}")
    print(f"Threshold range: {args.min_threshold} - {args.max_threshold} ({args.steps} steps, log-space)")
    print(f"Target metric: {args.target_metric}")
    print(f"Min precision: {args.min_precision}")
    print(f"Injection rate: {args.injection_rate:.1%}")
    print(f"Trials: {args.n_trials} (max_workers: {args.max_workers or args.n_trials})")
    print(f"Cold-start mode: {args.cold_start}")
    print(f"Dry run: {args.dry_run}")
    if args.model_filter:
        print(f"Model filter: {args.model_filter}")
    if args.model_suffix:
        print(f"Model suffix: {args.model_suffix}")
    print("=" * 70)
    print()

    # Initialize persistence
    persistence = ModelPersistence()
    print(f"Models location: {persistence.models_root_description}")
    print()

    # Determine granularities to process
    if args.granularity == "both":
        granularities = ["country_segment", "competitor"]
    else:
        granularities = [args.granularity]

    all_results: list[TuningResult] = []
    updated_models: list[str] = []

    for granularity in granularities:
        print(f"\n{'='*60}")
        print(f"Granularity: {granularity}")
        print(f"{'='*60}")

        # Find test files
        files = find_parquet_files(args.data_path, granularity, args.file_suffix)

        if not files:
            print(f"No test files found in {args.data_path}/by_{granularity}/*{args.file_suffix}.parquet")
            continue

        print(f"Found {len(files)} test files")

        for filepath in files:
            model_name = extract_model_name(filepath)

            # Apply filter if specified
            if args.model_filter and args.model_filter not in model_name:
                continue

            # Build saved model name (may include suffix for existing dated models)
            saved_model_name = model_name + args.model_suffix

            print(f"\n  Tuning: {model_name}" + (f" (loading {saved_model_name})" if args.model_suffix else ""))

            # Load model from local storage
            try:
                detector = persistence.load_autoencoder(saved_model_name)
            except Exception as e:
                print(f"    [SKIP] Could not load model: {e}")
                continue

            # Load test data
            print("    Loading test data...")
            test_df = pd.read_parquet(filepath)
            print(f"    {len(test_df):,} rows, {test_df['product_id'].nunique():,} products")

            # Load train data for cache population (unless cold-start)
            train_df = None
            if not args.cold_start:
                train_file = find_train_file(filepath)
                if train_file:
                    print("    Loading train data for cache...")
                    train_df = pd.read_parquet(train_file)
                    print(f"    {len(train_df):,} train rows")
                else:
                    print("    [NOTE] No train file found, using cold-start mode")

            if len(test_df) < 100:
                print("    [SKIP] Insufficient test data")
                continue

            # Use log-space for autoencoder reconstruction errors
            thresholds = np.logspace(
                np.log10(args.min_threshold),
                np.log10(args.max_threshold),
                args.steps,
            )
            current_threshold = detector._threshold

            # Tune threshold using parallel trials
            print(f"    Running grid search ({args.steps} thresholds, {args.n_trials} trials)...")
            result = run_tuning_trials(
                detector=detector,
                detector_name=model_name,
                test_df=test_df,
                train_df=train_df,
                thresholds=thresholds,
                current_threshold=current_threshold,
                n_trials=args.n_trials,
                injection_rate=args.injection_rate,
                max_workers=args.max_workers,
                target_metric=args.target_metric,
                min_precision=args.min_precision,
                # AE uses drop_range (0.1, 0.5) - original script's values
                drop_range=(0.1, 0.5),
            )

            if result is None:
                print("    [SKIP] Tuning failed")
                continue

            result.granularity = granularity
            all_results.append(result)

            # Print results
            print(f"    Current: threshold={result.current_threshold:.6f}, F1={result.current_f1:.1%}")
            print(f"    Best:    threshold={result.best_threshold:.6f}, F1={result.best_f1:.1%}")
            print(f"    Best P={result.best_precision:.1%}, R={result.best_recall:.1%}")

            if args.verbose:
                print("\n    Threshold sweep results:")
                print(f"    {'Threshold':>12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
                print(f"    {'-'*44}")
                for r in result.all_results:
                    marker = " *" if abs(r["threshold"] - result.best_threshold) < 0.0001 else ""
                    print(f"    {r['threshold']:>12.6f} {r['precision']:>9.1%} {r['recall']:>9.1%} {r['f1']:>9.1%}{marker}")

            if result.improvement_pct > 0:
                print(f"    [+] Improvement: +{result.improvement_pct:.1f}% F1")
            elif result.improvement_pct < 0:
                print(f"    [-] Degradation: {result.improvement_pct:.1f}% F1")
            else:
                print("    [=] No change")

            # Update the saved local model if improvement and not dry run
            should_update = result.improvement_pct > 1 and result.best_f1 >= 0.3
            if not args.dry_run and should_update:
                try:
                    update_model_threshold(persistence, saved_model_name, result.best_threshold)
                    updated_models.append(saved_model_name)
                except Exception as e:
                    print(f"    [ERROR] Failed to update: {e}")
            elif args.dry_run and should_update:
                print("    [DRY RUN] Would update the saved local model")
            elif result.improvement_pct > 1 and result.best_f1 < 0.3:
                print(f"    [SKIP UPDATE] F1 too low ({result.best_f1:.1%})")

    # Summary
    print("\n")
    print("=" * 70)
    print("TUNING SUMMARY")
    print("=" * 70)

    if all_results:
        print(f"\n{'Model':<30} {'Current':>12} {'Best':>12} {'F1':>8} {'Improvement':>12}")
        print("-" * 74)
        for r in all_results:
            imp_str = f"+{r.improvement_pct:.1f}%" if r.improvement_pct > 0 else f"{r.improvement_pct:.1f}%"
            print(
                f"{r.model_name:<30} {r.current_threshold:>12.6f} {r.best_threshold:>12.6f} "
                f"{r.best_f1:>7.1%} {imp_str:>12}"
            )

        avg_improvement = sum(r.improvement_pct for r in all_results) / len(all_results)
        avg_f1 = sum(r.best_f1 for r in all_results) / len(all_results)
        print()
        print(f"Average F1: {avg_f1:.1%}")
        print(f"Average improvement: {avg_improvement:+.1f}%")

    print()
    if updated_models:
        print(f"Updated {len(updated_models)} saved model(s):")
        for m in updated_models:
            print(f"  - {m}")
    elif not args.dry_run and all_results:
        print("No models updated (no significant improvements found)")
    elif args.dry_run:
        print("[DRY RUN] No local model files were changed")

    print()
    print("=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
