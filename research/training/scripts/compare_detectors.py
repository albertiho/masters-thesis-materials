#!/usr/bin/env python3
"""Unified comparison of all anomaly detection methods for thesis.

This script evaluates all detector types on the same test data with identical
anomaly injection methodology, enabling fair method comparison.

Key Features:
    - Uses TestOrchestrator with isolated caches per detector
    - Each detector maintains its own baseline (no cross-contamination)
    - Parallel execution across detectors for each row
    - File-level parallelization via ThreadPoolExecutor

Detectors Compared:
    Statistical (5 methods):
        - Z-score: Price deviates from rolling mean by N standard deviations
        - IQR: Price outside interquartile range bounds
        - Threshold: Percentage price change exceeds threshold
        - Sanity: Business rule violations (sale > list price)
        - StatisticalEnsemble: Combines Z-score, IQR, Threshold

    ML Methods (4 methods):
        - Isolation Forest: Tree-based anomaly detection
        - EIF: Hyperplane-based tree anomaly detection
        - RRCF: Streaming random cut forest anomaly detection
        - Autoencoder: Neural network reconstruction error

Output:
    - Console summary table with P/R/F1 for each method
    - CSV file for thesis figures (per-model and aggregate)
    - Detailed breakdown by model if requested

Usage:
    # Run full comparison on test data
    python scripts/compare_detectors.py --file-suffix "_test"

    # Save results to CSV
    python scripts/compare_detectors.py --file-suffix "_test" --output thesis_comparison.csv

    # Only compare specific granularity
    python scripts/compare_detectors.py --file-suffix "_test" --granularity country_segment

    # Skip ML models (statistical only)
    python scripts/compare_detectors.py --file-suffix "_test" --skip-ml

    # Verbose output with per-model breakdown
    python scripts/compare_detectors.py --file-suffix "_test" --verbose

    # Control parallelization
    python scripts/compare_detectors.py --file-suffix "_test" --workers 4
"""

import argparse
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Add project root to path (4 levels up from research/training/scripts/)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_script_dir)))
sys.path.insert(0, _project_root)
sys.path.insert(0, _script_dir)  # For importing sibling scripts

from src.anomaly.persistence import ModelPersistence
from src.research.artifacts import (
    comparison_result_to_tables,
    create_run_id,
    normalize_dataset_split_name,
    reindex_split_artifacts,
    resolve_git_commit,
    write_evaluation_run,
)
from src.anomaly.statistical import (
    ZScoreDetector,
    IQRDetector,
    ThresholdDetector,
    SanityCheckDetector,
    StatisticalEnsemble,
)
from src.anomaly.statistical import AnomalyResult
from src.research.evaluation.detector_evaluator import DetectorEvaluator
from src.research.evaluation.synthetic import inject_anomalies_to_dataframe
from src.research.evaluation.test_orchestrator import ComparisonResult, DetectorMetrics, TestOrchestrator

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class ModelComparison:
    """Comparison results for a single model/data split."""

    model_name: str
    granularity: str
    n_samples: int
    n_products: int
    results: dict[str, DetectorMetrics] = field(default_factory=dict)
    raw_results: dict[str, list[AnomalyResult]] | None = None
    injection_details: list[dict] | None = None
    comparison_result: ComparisonResult | None = None


# =============================================================================
# File Discovery
# =============================================================================


def find_parquet_files(data_path: str, granularity: str, file_suffix: str = "") -> list[str]:
    """Find Parquet files matching the granularity and suffix."""
    if granularity == "country_segment":
        pattern = os.path.join(data_path, "by_country_segment", f"*{file_suffix}.parquet")
        files = glob(pattern)
    elif granularity == "competitor":
        pattern = os.path.join(
            data_path, "by_competitor", "**", f"*{file_suffix}.parquet"
        )
        files = glob(pattern, recursive=True)
    elif granularity == "global":
        pattern = os.path.join(data_path, "global", f"*{file_suffix}.parquet")
        files = glob(pattern)
    else:
        raise ValueError(f"Unknown granularity: {granularity}")
    # Exclude files that are splits if we're not using a suffix
    if not file_suffix:
        files = [f for f in files if not f.endswith("_train.parquet") and not f.endswith("_test.parquet")]

    return sorted(files)


def find_matching_train_file(test_filepath: str, file_suffix: str) -> str | None:
    """Find the matching train file for a test file.
    
    For _test_new_prices files, we need the corresponding _train file to provide
    historical context for computing rolling statistics.
    
    Args:
        test_filepath: Path to test file (e.g., DK_B2C_2026-01-18_test_new_prices.parquet)
        file_suffix: The suffix used to find test files (e.g., "_test_new_prices")
    
    Returns:
        Path to matching train file, or None if not found
    """
    basename = os.path.basename(test_filepath)
    name = basename.replace(".parquet", "")
    
    # Check for experiment suffix after the test suffix
    exp_suffix = ""
    if file_suffix in name:
        after_test = name.split(file_suffix)[-1]
        if after_test and after_test.startswith("_"):
            exp_suffix = after_test
    
    # Build train filename
    base_without_suffix = name.replace(file_suffix + exp_suffix, "")
    train_filename = f"{base_without_suffix}_train{exp_suffix}.parquet"
    train_filepath = os.path.join(os.path.dirname(test_filepath), train_filename)
    
    if os.path.exists(train_filepath):
        return train_filepath
    
    return None


def extract_model_name(filepath: str) -> str:
    """Extract model name from Parquet filename.
    
    Handles various filename patterns:
        - NO_B2C_train_mh4.parquet -> NO_B2C_mh4
        - NO_B2C_2026-01-22_test_new_prices_mh4.parquet -> NO_B2C_mh4
        - NO_B2C_test.parquet -> NO_B2C
    """
    filename = os.path.basename(filepath)
    name = filename.replace(".parquet", "")

    # Extract min_history suffix if present (e.g., _mh4, _mh5)
    mh_suffix = ""
    mh_match = re.search(r"(_mh\d+)$", name)
    if mh_match:
        mh_suffix = mh_match.group(1)
        name = name[: -len(mh_suffix)]

    # Remove various test suffixes
    for suffix in ["_train", "_test", "_test_new_products", "_test_new_prices"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break

    # Remove date suffix like _2026-01-18
    date_pattern = r"_\d{4}-\d{2}-\d{2}$"
    name = re.sub(date_pattern, "", name)

    # Re-append min_history suffix
    return name + mh_suffix


def extract_country(model_name: str | None) -> str | None:
    """Extract country code from model name.

    Model names follow pattern: {COUNTRY}_{SEGMENT}_{DATE}
    Example: DK_B2C_2026-01-18 -> DK

    Args:
        model_name: Model name string.

    Returns:
        Country code or None.
    """
    if not model_name:
        return None

    match = re.match(r"^([A-Z]{2})_", model_name)
    if match:
        return match.group(1)

    return None


# =============================================================================
# Main Comparison Logic
# =============================================================================


def create_evaluators(
    persistence: ModelPersistence | None,
    model_name: str,
    skip_ml: bool = False,
) -> list[DetectorEvaluator]:
    """Create DetectorEvaluator instances for all detectors.
    
    Args:
        persistence: ModelPersistence for loading ML models (None to skip ML).
        model_name: Model name for loading ML models.
        skip_ml: Skip ML model evaluation if True.
    
    Returns:
        List of DetectorEvaluator instances.
    """
    evaluators = []
    
    # Statistical detectors (always included)
    evaluators.append(DetectorEvaluator(StatisticalEnsemble(), "StatisticalEnsemble"))
    evaluators.append(DetectorEvaluator(ZScoreDetector(), "Z-score"))
    evaluators.append(DetectorEvaluator(IQRDetector(), "IQR"))
    evaluators.append(DetectorEvaluator(ThresholdDetector(), "Threshold"))
    evaluators.append(DetectorEvaluator(SanityCheckDetector(), "Sanity"))
    
    # ML detectors (if available)
    if not skip_ml and persistence is not None:
        # Isolation Forest
        try:
            if_detector = persistence.load_isolation_forest(model_name)
            evaluators.append(DetectorEvaluator(if_detector, "Isolation Forest"))
        except Exception as e:
            logger.warning(f"Could not load Isolation Forest for {model_name}: {e}")

        # EIF
        try:
            eif_detector = persistence.load_eif(model_name)
            evaluators.append(DetectorEvaluator(eif_detector, "EIF"))
        except Exception as e:
            logger.warning(f"Could not load EIF for {model_name}: {e}")

        # RRCF
        try:
            rrcf_detector = persistence.load_rrcf(model_name)
            evaluators.append(DetectorEvaluator(rrcf_detector, "RRCF"))
        except Exception as e:
            logger.warning(f"Could not load RRCF for {model_name}: {e}")

        # Autoencoder
        try:
            ae_detector = persistence.load_autoencoder(model_name)
            evaluators.append(DetectorEvaluator(ae_detector, "Autoencoder"))
        except Exception as e:
            logger.warning(f"Could not load Autoencoder for {model_name}: {e}")

    return evaluators


def compare_on_file(
    filepath: str,
    persistence: ModelPersistence | None,
    injection_rate: float = 0.1,
    skip_ml: bool = False,
    file_suffix: str = "_test",
    save_matrix: bool = False,
    matrix_dir: str | None = None,
) -> ModelComparison | None:
    """Run full detector comparison on a single test file.

    Uses TestOrchestrator to run all detectors with isolated caches in parallel,
    eliminating cross-contamination and look-ahead bias.

    For _test_new_prices files, this function automatically loads the corresponding
    training file to provide historical context for computing rolling statistics.

    Uses inject_anomalies_to_dataframe() with 6 anomaly types:
    PRICE_SPIKE, PRICE_DROP, ZERO_PRICE, NEGATIVE_PRICE, EXTREME_OUTLIER, DECIMAL_SHIFT

    Args:
        filepath: Path to test parquet file.
        persistence: ModelPersistence for loading ML models (None to skip ML).
        injection_rate: Fraction of data to inject as anomalies.
        skip_ml: Skip ML model evaluation if True.
        file_suffix: Suffix pattern for test files (e.g., "_test_new_prices").
        save_matrix: If True, save per-anomaly detection matrix to CSV.
        matrix_dir: Directory to save anomaly matrices (required if save_matrix=True).

    Returns:
        ModelComparison with results for all evaluated detectors, or None if skipped.
    """
    start_time = time.time()
    model_name = extract_model_name(filepath)
    logger.info(f"Comparing detectors on: {model_name}")

    # Load test data
    df_test = pd.read_parquet(filepath)
    n_test_samples = len(df_test)
    n_test_products = df_test["product_id"].nunique()
    logger.info(f"  {n_test_samples:,} rows, {n_test_products:,} products")

    if n_test_samples < 100:
        logger.warning(f"  Skipping {model_name}: insufficient data")
        return None

    # Determine if cold start (no train data)
    train_df = None
    if "_test_new_prices" in file_suffix:
        train_filepath = find_matching_train_file(filepath, file_suffix)
        if train_filepath:
            train_df = pd.read_parquet(train_filepath)
            logger.info(f"  Loaded historical context: {len(train_df):,} train rows")
        else:
            logger.warning(f"  No matching train file found for {filepath}")
            logger.warning("  Statistical detectors will operate in cold-start mode")

    # Inject anomalies into test data using unified injection (6 anomaly types)
    df_injected, labels, injection_details = inject_anomalies_to_dataframe(
        df_test,
        injection_rate=injection_rate,
        seed=42,
        spike_range=(2.0, 5.0),
        drop_range=(0.1, 0.5),
    )
    logger.info(f"  Injected {np.sum(labels):,} anomalies ({injection_rate:.0%} rate)")

    # Create evaluators
    evaluators = create_evaluators(persistence, model_name, skip_ml)
    logger.info(f"  Evaluating {len(evaluators)} detectors: {[e.name for e in evaluators]}")

    # Create orchestrator and run comparison with details (for raw results)
    orchestrator = TestOrchestrator(evaluators, max_workers=len(evaluators))
    country = extract_country(model_name)
    
    try:
        comparison_result = orchestrator.run_comparison_with_details(
            train_df, df_injected, labels, country,
            injection_details=injection_details,
        )
    except Exception as e:
        logger.error(f"  Comparison failed for {model_name}: {e}")
        return None

    # Save anomaly matrix if requested
    if save_matrix and matrix_dir:
        save_anomaly_matrix(
            model_name=model_name,
            injection_details=injection_details,
            raw_results=comparison_result.raw_results,
            output_dir=matrix_dir,
            df_sorted=comparison_result.df_sorted,
        )

    # Create comparison result
    comparison = ModelComparison(
        model_name=model_name,
        granularity="",  # Set by caller
        n_samples=n_test_samples,
        n_products=n_test_products,
        results=comparison_result.metrics,
        raw_results=comparison_result.raw_results if save_matrix else None,
        injection_details=injection_details if save_matrix else None,
        comparison_result=comparison_result,
    )

    elapsed = time.time() - start_time
    logger.info(f"  Completed {model_name} in {elapsed:.1f}s")
    return comparison


def save_anomaly_matrix(
    model_name: str,
    injection_details: list[dict],
    raw_results: dict[str, list[AnomalyResult]],
    output_dir: str,
    df_sorted: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Save full anomaly detection matrix to CSV.
    
    Creates one row per injected anomaly with detection status from each detector.
    This enables deep analysis of which specific anomalies escape which detectors.
    
    Args:
        model_name: Model name for the output file.
        injection_details: List of injection details from inject_anomalies_to_dataframe().
            Each dict contains 'index', 'anomaly_type', 'original_price', 'new_price', etc.
        raw_results: Dict mapping detector names to list of AnomalyResult.
        output_dir: Directory to save the CSV file.
        df_sorted: Optional sorted DataFrame with __injected_anomaly_type__ column.
            When provided, reads anomaly info from aligned DataFrame columns.
    
    Returns:
        DataFrame with the anomaly matrix (also saved to CSV).
    """
    # Check if new columns exist (backward compatibility)
    has_injection_columns = (
        df_sorted is not None and 
        "__injected_anomaly_type__" in df_sorted.columns
    )
    
    if has_injection_columns:
        # Build rows from DataFrame (aligned after sorting)
        rows = []
        detector_names = list(raw_results.keys())
        price_column = "unit_price" if "unit_price" in df_sorted.columns else "price"
        
        for i in range(len(df_sorted)):
            anomaly_type = df_sorted.iloc[i]["__injected_anomaly_type__"]
            if anomaly_type is None:
                continue  # Not an injected anomaly
            
            original_price = df_sorted.iloc[i]["__original_price__"]
            injected_price = df_sorted.iloc[i][price_column] if price_column in df_sorted.columns else original_price
            
            row = {
                "model": model_name,
                "index": i,
                "product_id": df_sorted.iloc[i].get("product_id", ""),
                "competitor_id": df_sorted.iloc[i].get("competitor_id", ""),
                "anomaly_type": anomaly_type,
                "original_price": original_price,
                "injected_price": injected_price,
            }
            
            # Calculate change percentage
            if original_price and original_price > 0:
                row["change_pct"] = (injected_price - original_price) / original_price
            else:
                row["change_pct"] = None
            
            # Add detection status for each detector
            detected_count = 0
            for detector_name in detector_names:
                results_list = raw_results[detector_name]
                is_detected = results_list[i].is_anomaly if i < len(results_list) else False
                col_name = f"{detector_name.replace(' ', '_').replace('-', '_')}_detected"
                row[col_name] = is_detected
                if is_detected:
                    detected_count += 1
            
            row["any_detected"] = detected_count > 0
            row["all_detected"] = detected_count == len(detector_names)
            row["num_detectors_caught"] = detected_count
            rows.append(row)
        
        df = pd.DataFrame(rows)
    
    if df.empty:
        logger.warning(f"  No anomalies to save for {model_name}")
        return df
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_name}_anomaly_matrix.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"  Saved anomaly matrix: {output_path} ({len(df)} anomalies)")
    
    return df


def aggregate_results(comparisons: list[ModelComparison]) -> dict[str, DetectorMetrics]:
    """Aggregate results across all models."""
    detector_names = [
        "Z-score", "IQR", "Threshold", "Sanity", "StatisticalEnsemble",
        "Autoencoder", "Isolation Forest", "EIF", "RRCF"
    ]

    aggregated = {}
    for name in detector_names:
        f1_scores = []
        precision_scores = []
        recall_scores = []
        total_samples = 0

        for comp in comparisons:
            if name in comp.results and comp.results[name].f1 > 0:
                f1_scores.append(comp.results[name].f1)
                precision_scores.append(comp.results[name].precision)
                recall_scores.append(comp.results[name].recall)
                total_samples += comp.results[name].n_samples

        if f1_scores:
            aggregated[name] = DetectorMetrics(
                detector_name=name,
                precision=float(np.mean(precision_scores)),
                recall=float(np.mean(recall_scores)),
                f1=float(np.mean(f1_scores)),
                n_samples=total_samples,
            )
        else:
            aggregated[name] = DetectorMetrics(name, 0.0, 0.0, 0.0)

    return aggregated


def build_canonical_split_artifacts(
    comparisons: list[ModelComparison],
    *,
    run_id: str,
    dataset_split: str,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Convert per-file comparison results into canonical split artifacts."""
    injected_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []

    for comparison in comparisons:
        if comparison.comparison_result is None:
            continue
        injected_rows, predictions = comparison_result_to_tables(
            comparison.comparison_result,
            run_id=run_id,
            candidate_id="",
            experiment_family="comparison",
            dataset_name=comparison.model_name,
            dataset_granularity=comparison.granularity,
            dataset_split=dataset_split,
        )
        injected_frames.append(injected_rows)
        prediction_frames.append(predictions)

    if not injected_frames:
        return {}

    injected_frames, prediction_frames = zip(
        *reindex_split_artifacts(list(zip(injected_frames, prediction_frames)))
    )
    split_name = normalize_dataset_split_name(dataset_split)
    return {
        split_name: (
            pd.concat(injected_frames, ignore_index=True),
            pd.concat(prediction_frames, ignore_index=True),
        )
    }


def build_comparison_run_metadata(
    *,
    run_id: str,
    source_dataset_paths: list[str],
    comparisons: list[ModelComparison],
    dataset_granularity: str,
    dataset_split: str,
    injection_rate: float,
    skip_ml: bool,
    workers: int,
    model_filter: str | None,
) -> dict[str, object]:
    """Build canonical run metadata for the comparison workflow."""
    return {
        "schema_version": "phase2.v1",
        "experiment_family": "comparison",
        "run_id": run_id,
        "source_dataset_paths": sorted(set(source_dataset_paths)),
        "dataset_names": sorted({comp.model_name for comp in comparisons}),
        "dataset_granularity": dataset_granularity,
        "dataset_splits": [normalize_dataset_split_name(dataset_split)],
        "random_seeds": {"injection_seed": 42},
        "injection_config": {
            "injection_rate": injection_rate,
            "file_suffix": dataset_split,
        },
        "detector_identifiers": sorted(
            {
                detector_name
                for comparison in comparisons
                for detector_name in comparison.results.keys()
            }
        ),
        "config_values": {
            "skip_ml": skip_ml,
            "workers": workers,
            "model_filter": model_filter,
        },
        "generated_at": datetime.now().astimezone().isoformat(),
        "git_commit": resolve_git_commit(Path(_project_root)),
    }


def results_to_csv(
    comparisons: list[ModelComparison],
    aggregated: dict[str, DetectorMetrics],
    output_path: str,
) -> None:
    """Save results to CSV for thesis figures.
    
    Includes per-anomaly-type detection rates when available.
    """
    # Define the 6 anomaly types for consistent column ordering
    anomaly_types = [
        "price_spike",
        "price_drop",
        "zero_price",
        "negative_price",
        "extreme_outlier",
        "decimal_shift",
    ]
    
    rows = []

    # Per-model results
    for comp in comparisons:
        for detector_name, result in comp.results.items():
            row = {
                "model": comp.model_name,
                "granularity": comp.granularity,
                "detector": detector_name,
                "precision": result.precision,
                "recall": result.recall,
                "f1": result.f1,
                "n_samples": result.n_samples,
            }
            
            # Add per-type detection rates
            for atype in anomaly_types:
                col_name = f"{atype}_rate"
                if atype in result.detection_by_type:
                    row[col_name] = result.detection_by_type[atype].rate
                else:
                    row[col_name] = None
            
            rows.append(row)

    # Aggregate results
    for detector_name, result in aggregated.items():
        row = {
            "model": "AGGREGATE",
            "granularity": "all",
            "detector": detector_name,
            "precision": result.precision,
            "recall": result.recall,
            "f1": result.f1,
            "n_samples": result.n_samples,
        }
        
        # Add per-type detection rates (may be empty for aggregated)
        for atype in anomaly_types:
            col_name = f"{atype}_rate"
            if atype in result.detection_by_type:
                row[col_name] = result.detection_by_type[atype].rate
            else:
                row[col_name] = None
        
        rows.append(row)

    df = pd.DataFrame(rows)
    
    # Ensure parent directory exists
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    logger.info(f"Saved results to {output_path}")


def print_summary_table(aggregated: dict[str, DetectorMetrics]) -> None:
    """Print summary comparison table."""
    print("\n" + "=" * 70)
    print("DETECTOR COMPARISON SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Detector':<20} {'Precision':>12} {'Recall':>12} {'F1':>12}")
    print("-" * 60)

    # Sort by F1 descending
    sorted_results = sorted(aggregated.items(), key=lambda x: x[1].f1, reverse=True)

    for name, result in sorted_results:
        if result.f1 > 0:
            print(f"{name:<20} {result.precision:>11.1%} {result.recall:>12.1%} {result.f1:>12.1%}")
        else:
            print(f"{name:<20} {'N/A':>12} {'N/A':>12} {'N/A':>12}")

    print("-" * 60)
    print()
    
    # Print per-type detection breakdown
    print_per_type_breakdown(aggregated)


def find_completed_models(matrix_dir: str) -> set[str]:
    """Find model names with existing anomaly matrix files.
    
    Scans the matrix directory for existing *_anomaly_matrix.csv files
    and extracts the model names, enabling resume functionality.
    
    Args:
        matrix_dir: Directory containing anomaly matrix CSV files.
    
    Returns:
        Set of model names that have completed matrix files.
    """
    if not matrix_dir or not os.path.exists(matrix_dir):
        return set()
    
    completed = set()
    for filename in os.listdir(matrix_dir):
        if filename.endswith("_anomaly_matrix.csv"):
            model_name = filename.replace("_anomaly_matrix.csv", "")
            completed.add(model_name)
    return completed


def print_per_type_breakdown(aggregated: dict[str, DetectorMetrics]) -> None:
    """Print detection rates by anomaly type for each detector.
    
    Shows a matrix of detectors vs anomaly types with detection rates,
    helping identify which anomaly types escape which detectors.
    """
    # Define anomaly types with short display names
    anomaly_types = [
        ("price_spike", "SPIKE"),
        ("price_drop", "DROP"),
        ("zero_price", "ZERO"),
        ("negative_price", "NEG"),
        ("extreme_outlier", "OUTLIER"),
        ("decimal_shift", "DECIMAL"),
    ]
    
    # Check if any detector has per-type data
    has_type_data = any(
        result.detection_by_type
        for result in aggregated.values()
        if result.f1 > 0
    )
    
    if not has_type_data:
        return
    
    print("=" * 80)
    print("DETECTION BY ANOMALY TYPE")
    print("=" * 80)
    print()
    
    # Header row
    header = f"{'Detector':<20}"
    for _, short_name in anomaly_types:
        header += f"{short_name:>10}"
    print(header)
    print("-" * 80)
    
    # Sort by F1 descending (same order as main table)
    sorted_results = sorted(aggregated.items(), key=lambda x: x[1].f1, reverse=True)
    
    for name, result in sorted_results:
        if result.f1 <= 0:
            continue
        
        row = f"{name:<20}"
        for atype, _ in anomaly_types:
            if atype in result.detection_by_type:
                rate = result.detection_by_type[atype].rate
                row += f"{rate:>9.0%} "
            else:
                row += f"{'N/A':>10}"
        print(row)
    
    print("-" * 80)
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Compare all anomaly detection methods")
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
        choices=["country_segment", "competitor", "global", "both"],
        default="both",
        help="Model granularity to evaluate (default: both)",
    )
    parser.add_argument(
        "--injection-rate",
        type=float,
        default=0.1,
        help="Fraction of data to inject as anomalies (default: 0.1)",
    )
    parser.add_argument(
        "--skip-ml",
        action="store_true",
        help="Skip ML models (statistical only)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed per-model breakdown",
    )
    parser.add_argument(
        "--model-filter",
        type=str,
        default=None,
        help="Only evaluate models matching this substring",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for file processing (default: 1)",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="results/comparison",
        help="Canonical results root (default: results/comparison)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id for the canonical output directory",
    )
    args = parser.parse_args()

    load_dotenv()

    print("=" * 70)
    print("Unified Detector Comparison - Thesis Evaluation")
    print("=" * 70)
    print(f"Data path: {args.data_path}")
    print(f"File suffix: {args.file_suffix}")
    print(f"Granularity: {args.granularity}")
    print(f"Injection rate: {args.injection_rate:.1%}")
    print(f"Anomaly types: 6 (SPIKE, DROP, ZERO, NEGATIVE, EXTREME_OUTLIER, DECIMAL_SHIFT)")
    print(f"Skip ML: {args.skip_ml}")
    print(f"Workers: {args.workers}")
    print(f"Results root: {args.results_root}")
    if args.run_id:
        print(f"Run ID: {args.run_id}")
    if args.model_filter:
        print(f"Model filter: {args.model_filter}")
    print()
    print("Detectors:")
    print("  Statistical: Z-score, IQR, Threshold, Sanity, StatisticalEnsemble")
    if not args.skip_ml:
        print("  ML: Autoencoder, Isolation Forest, EIF, RRCF")
    print()
    print("Architecture: TestOrchestrator with isolated DetectorEvaluator caches")
    print("=" * 70)
    print()

    # Initialize persistence if needed
    persistence = None
    if not args.skip_ml:
        try:
            persistence = ModelPersistence()
            print(f"ML models: {persistence.models_root_description}")
            print()
        except Exception as e:
            logger.warning(f"Could not initialize model persistence: {e}")
            logger.warning("Continuing without ML models")

    # Determine granularities
    if args.granularity == "both":
        granularities = ["country_segment", "competitor", "global"]
    else:
        granularities = [args.granularity]

    all_comparisons: list[ModelComparison] = []
    evaluated_files: list[str] = []

    total_start_time = time.time()

    for granularity in granularities:
        print(f"\n{'='*60}")
        print(f"Granularity: {granularity}")
        print(f"{'='*60}")

        files = find_parquet_files(args.data_path, granularity, args.file_suffix)

        if not files:
            print("No test files found")
            continue

        # Apply model filter
        if args.model_filter:
            files = [f for f in files if args.model_filter in extract_model_name(f)]

        print(f"Found {len(files)} test files to process")

        # Process files with optional parallelization
        if args.workers > 1 and len(files) > 1:
            print(f"Processing with {args.workers} parallel workers...")
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {}
                for filepath in files:
                    future = executor.submit(
                        compare_on_file,
                        filepath,
                        persistence,
                        args.injection_rate,
                        args.skip_ml,
                        args.file_suffix,
                        False,
                        None,
                    )
                    futures[future] = filepath

                completed = 0
                for future in as_completed(futures):
                    filepath = futures[future]
                    completed += 1
                    try:
                        comparison = future.result()
                        if comparison:
                            comparison.granularity = granularity
                            all_comparisons.append(comparison)
                            evaluated_files.append(filepath)
                            print(f"  [{completed}/{len(files)}] Completed: {comparison.model_name}")

                            if args.verbose:
                                for name, result in comparison.results.items():
                                    if result.f1 > 0:
                                        print(
                                            f"    {name:<18} P={result.precision:.1%}, "
                                            f"R={result.recall:.1%}, F1={result.f1:.1%}"
                                        )
                    except Exception as e:
                        logger.error(f"  [{completed}/{len(files)}] Failed: {filepath}: {e}")
        else:
            # Sequential processing
            for i, filepath in enumerate(files, 1):
                comparison = compare_on_file(
                    filepath,
                    persistence,
                    injection_rate=args.injection_rate,
                    skip_ml=args.skip_ml,
                    file_suffix=args.file_suffix,
                    save_matrix=False,
                    matrix_dir=None,
                )

                if comparison:
                    comparison.granularity = granularity
                    all_comparisons.append(comparison)
                    evaluated_files.append(filepath)
                    print(f"  [{i}/{len(files)}] Completed: {comparison.model_name}")

                    if args.verbose:
                        for name, result in comparison.results.items():
                            if result.f1 > 0:
                                print(
                                    f"    {name:<18} P={result.precision:.1%}, "
                                    f"R={result.recall:.1%}, F1={result.f1:.1%}"
                                )

    total_elapsed = time.time() - total_start_time
    print(f"\nTotal processing time: {total_elapsed:.1f}s")

    # Aggregate and summarize
    if all_comparisons:
        aggregated = aggregate_results(all_comparisons)
        print_summary_table(aggregated)
        run_id = args.run_id or create_run_id("comparison")
        split_artifacts = build_canonical_split_artifacts(
            all_comparisons,
            run_id=run_id,
            dataset_split=args.file_suffix,
        )
        if split_artifacts:
            run_root = Path(args.results_root) / run_id
            run_metadata = build_comparison_run_metadata(
                run_id=run_id,
                source_dataset_paths=evaluated_files,
                comparisons=all_comparisons,
                dataset_granularity=args.granularity,
                dataset_split=args.file_suffix,
                injection_rate=args.injection_rate,
                skip_ml=args.skip_ml,
                workers=args.workers,
                model_filter=args.model_filter,
            )
            write_evaluation_run(
                run_root=run_root,
                run_metadata=run_metadata,
                split_artifacts=split_artifacts,
            )
            print(f"\nCanonical results written to {run_root}")

        # Print detailed breakdown if verbose
        if args.verbose:
            print("\n" + "=" * 70)
            print("DETAILED RESULTS BY MODEL")
            print("=" * 70)

            for comp in all_comparisons:
                print(f"\n{comp.model_name} ({comp.granularity}, {comp.n_samples:,} samples):")
                for name in [
                    "Z-score",
                    "IQR",
                    "Threshold",
                    "Sanity",
                    "StatisticalEnsemble",
                    "Autoencoder",
                    "Isolation Forest",
                    "EIF",
                    "RRCF",
                ]:
                    if name in comp.results:
                        r = comp.results[name]
                        if r.f1 > 0:
                            print(f"  {name:<18} P={r.precision:.1%}, R={r.recall:.1%}, F1={r.f1:.1%}")
                        else:
                            print(f"  {name:<18} [not available]")

    else:
        print("\nNo results to report")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
