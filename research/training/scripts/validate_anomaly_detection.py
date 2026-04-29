#!/usr/bin/env python3
"""Validate anomaly detection pipeline against all production anomaly types.

This script creates deterministic test data where each of the 7 production
anomaly types is represented exactly once, then runs the detection pipeline
to verify each type is properly detected.

Usage:
    # Run validation with statistical detectors only (fast)
    python scripts/validate_anomaly_detection.py --skip-ml

    # Run with ML detectors (requires trained models)
    python scripts/validate_anomaly_detection.py --model DK_B2C_mh4

    # Verbose output with detailed scores
    python scripts/validate_anomaly_detection.py --skip-ml -v
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Add project root to path (4 levels up from research/training/scripts/)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_script_dir)))
sys.path.insert(0, _project_root)
sys.path.insert(0, _script_dir)  # For importing sibling scripts

from src.anomaly.combined import (
    CombinedDetector,
    CombinedDetectorConfig,
    DetectorLayer,
)
from src.anomaly.persistence import ModelPersistence
from src.anomaly.statistical import (
    SanityCheckDetector,
    ZScoreDetector,
)
from src.research.artifacts import (
    comparison_result_to_tables,
    create_run_id,
    resolve_git_commit,
    write_evaluation_run,
)
from src.research.evaluation.detector_evaluator import DetectorEvaluator
from src.research.evaluation.synthetic import (
    PRODUCTION_ANOMALY_TYPES,
    SyntheticAnomalyType,
    generate_all_anomaly_variants,
)
from src.research.evaluation.test_orchestrator import ComparisonResult, TestOrchestrator

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Detector Factory Functions (reused from analyze_detector_combinations.py)
# =============================================================================


def create_sanity_zscore_detector(name: str = "Sanity+Zscore") -> CombinedDetector:
    """Create two-layer detector: Sanity (gate) -> Z-score."""
    return CombinedDetector(
        config=CombinedDetectorConfig(name=name, min_history_cold=1, min_history_warm=1),
        layers=[
            DetectorLayer(
                name="sanity",
                detectors=[SanityCheckDetector()],
                is_gate=True,
                required_history=0,
            ),
            DetectorLayer(
                name="zscore",
                detectors=[ZScoreDetector()],
                required_history=0,
            ),
        ],
    )


def create_sanity_iforest_detector(iforest_detector, name: str = "Sanity+IForest") -> CombinedDetector:
    """Create two-layer detector: Sanity (gate) -> Isolation Forest."""
    return CombinedDetector(
        config=CombinedDetectorConfig(name=name, min_history_cold=1, min_history_warm=1),
        layers=[
            DetectorLayer(
                name="sanity",
                detectors=[SanityCheckDetector()],
                is_gate=True,
                required_history=0,
            ),
            DetectorLayer(
                name="iforest",
                detectors=[iforest_detector],
                required_history=0,
            ),
        ],
    )


def create_sanity_autoencoder_detector(ae_detector, name: str = "Sanity+AE") -> CombinedDetector:
    """Create two-layer detector: Sanity (gate) -> Autoencoder."""
    return CombinedDetector(
        config=CombinedDetectorConfig(name=name, min_history_cold=1, min_history_warm=1),
        layers=[
            DetectorLayer(
                name="sanity",
                detectors=[SanityCheckDetector()],
                is_gate=True,
                required_history=0,
            ),
            DetectorLayer(
                name="autoencoder",
                detectors=[ae_detector],
                required_history=0,
            ),
        ],
    )


# =============================================================================
# Test Data Generation
# =============================================================================


def create_synthetic_base_record(
    price: float = 1000.0,
    list_price: float = 1200.0,
    observation_count: int = 10,
) -> pd.Series:
    """Create a synthetic base record for validation testing.

    Args:
        price: Base price value (allows meaningful anomaly injection).
        list_price: List price (enables LIST_PRICE_VIOLATION detection).
        observation_count: Simulated observation count for warm product.

    Returns:
        pd.Series with columns needed for feature extraction.
    """
    now = datetime.now(timezone.utc)
    
    return pd.Series({
        # Required identifiers
        "product_id": "VALIDATION_TEST_001",
        "competitor_id": "TEST_COMPETITOR",
        "competitor_product_id": "TEST_001",
        # Price fields
        "price": price,
        "unit_price": price,
        "list_price": list_price,
        "currency": "DKK",
        # Temporal fields
        "first_seen_at": now,
        "scraped_at": now,
        # Context fields
        "country": "DK",
        "segment": "B2C",
        "observation_count": observation_count,
        # Product info (for feature extraction)
        "product_name": "Test Product for Validation",
        "brand": "TestBrand",
    })


def create_training_history(
    base_row: pd.Series,
    n_observations: int = 15,
    price_std_pct: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic training history for cache population.

    Generates stable price observations around the base price to provide
    temporal context for detectors.

    Args:
        base_row: Base record to generate history for.
        n_observations: Number of historical observations.
        price_std_pct: Price variation as percentage of base price.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with historical price observations.
    """
    rng = np.random.default_rng(seed)
    
    base_price = float(base_row["price"])
    rows = []
    
    # Generate observations going back in time
    base_time = base_row["first_seen_at"]
    if pd.isna(base_time):
        base_time = datetime.now(timezone.utc)
    
    for i in range(n_observations):
        row = base_row.to_dict()
        # Stable prices with small variation
        row["price"] = base_price * (1 + rng.normal(0, price_std_pct))
        row["unit_price"] = row["price"]
        # Go back in time
        row["first_seen_at"] = base_time - pd.Timedelta(days=n_observations - i)
        row["scraped_at"] = row["first_seen_at"]
        rows.append(row)
    
    return pd.DataFrame(rows)


# =============================================================================
# Evaluator Creation
# =============================================================================


def create_evaluators(
    persistence: ModelPersistence | None,
    model_name: str,
    skip_ml: bool = False,
) -> list[DetectorEvaluator]:
    """Create DetectorEvaluator instances for validation.

    Args:
        persistence: ModelPersistence instance for loading ML models.
        model_name: Model name for loading ML detectors.
        skip_ml: If True, skip ML detectors (statistical only).

    Returns:
        List of DetectorEvaluator instances.
    """
    evaluators = []

    # Statistical: Sanity + Z-score (our recommended cascade)
    evaluators.append(DetectorEvaluator(create_sanity_zscore_detector(), "Sanity+Zscore"))

    if not skip_ml and persistence is not None:
        # Isolation Forest
        try:
            if_detector = persistence.load_isolation_forest(model_name)
            evaluators.append(
                DetectorEvaluator(
                    create_sanity_iforest_detector(if_detector),
                    "Sanity+IForest",
                )
            )
        except Exception as e:
            logger.warning(f"Could not load Isolation Forest for {model_name}: {e}")

        # Autoencoder
        try:
            ae_detector = persistence.load_autoencoder(model_name)
            evaluators.append(
                DetectorEvaluator(
                    create_sanity_autoencoder_detector(ae_detector),
                    "Sanity+AE",
                )
            )
        except Exception as e:
            logger.warning(f"Could not load Autoencoder for {model_name}: {e}")

    return evaluators


# =============================================================================
# Validation Logic
# =============================================================================


def run_validation(
    evaluators: list[DetectorEvaluator],
    base_row: pd.Series,
    train_df: pd.DataFrame | None,
    verbose: bool = False,
) -> tuple[dict[str, dict[str, bool]], ComparisonResult, pd.DataFrame, list[dict]]:
    """Run validation and return per-type detection results.

    Args:
        evaluators: List of detector evaluators to test.
        base_row: Base record for generating variants.
        train_df: Training data for cache population.
        verbose: If True, print detailed scores.

    Returns:
        Tuple of:
            - detection_matrix: Dict mapping detector names to anomaly_type -> detected
            - result: ComparisonResult from orchestrator
            - df_test: Test DataFrame with injected anomalies
            - injection_details: List of injection detail dicts
    """
    # Generate test data: original + one variant per anomaly type
    df_test, labels, injection_details = generate_all_anomaly_variants(base_row)

    # Add injection columns to DataFrame for alignment after orchestrator sorting
    df_test["__injected_anomaly_type__"] = None
    df_test["__original_price__"] = df_test["price"].copy()
    for detail in injection_details:
        idx = detail["index"]
        df_test.loc[idx, "__injected_anomaly_type__"] = detail["anomaly_type"]
        df_test.loc[idx, "__original_price__"] = detail["original_price"]

    print(f"\nGenerated {len(df_test)} rows: 1 original + {len(injection_details)} anomaly variants")
    print(f"Base price: {base_row['price']:.2f}, List price: {base_row.get('list_price', 'N/A')}")

    # Run orchestrator
    orchestrator = TestOrchestrator(evaluators, max_workers=len(evaluators))
    result = orchestrator.run_comparison_with_details(
        train_df=train_df,
        test_df=df_test,
        labels=labels,
        country="DK",
        injection_details=injection_details,
    )

    # Build per-type detection matrix using aligned DataFrame from result
    detection_matrix: dict[str, dict[str, bool]] = {}
    df_sorted = result.df_sorted if result.df_sorted is not None else df_test

    for detector_name, anomaly_results in result.raw_results.items():
        detection_matrix[detector_name] = {}

        for i, row in df_sorted.iterrows():
            anomaly_type = row["__injected_anomaly_type__"]
            if anomaly_type is None:
                continue  # Skip original (non-injected) row
            
            # Get the position in anomaly_results (use enumerate index, not DataFrame index)
            pos = df_sorted.index.get_loc(i)
            is_detected = anomaly_results[pos].is_anomaly

            detection_matrix[detector_name][anomaly_type] = is_detected

            if verbose:
                score = anomaly_results[pos].anomaly_score
                detected_types = [t.value for t in anomaly_results[pos].anomaly_types]
                status = "DETECTED" if is_detected else "MISSED"
                print(f"  [{detector_name}] {anomaly_type}: {status} (score={score:.3f}, detected_types={detected_types})")

    return detection_matrix, result, df_test, injection_details


def print_results(
    detection_matrix: dict[str, dict[str, bool]],
    base_price: float,
    list_price: float | None,
) -> None:
    """Print validation results in a formatted table.

    Args:
        detection_matrix: Dict mapping detector -> anomaly_type -> detected.
        base_price: Base price used for test.
        list_price: List price used for test.
    """
    detector_names = list(detection_matrix.keys())
    anomaly_types = [t.value for t in PRODUCTION_ANOMALY_TYPES]

    print("\n" + "=" * 70)
    print("ANOMALY DETECTION VALIDATION RESULTS")
    print("=" * 70)
    print(f"Base price: {base_price:.2f}, List price: {list_price or 'N/A'}")
    print(f"Testing {len(anomaly_types)} anomaly types against {len(detector_names)} detector(s)")
    print()

    # Print results per anomaly type
    print("Results:")
    for anomaly_type in anomaly_types:
        # Check if detected by any detector
        detected_by = [
            name for name in detector_names
            if detection_matrix.get(name, {}).get(anomaly_type, False)
        ]

        if detected_by:
            status = f"[OK] Detected by {', '.join(detected_by)}"
        else:
            status = "[MISSED] Not detected by any detector"

        # Pad anomaly type for alignment
        print(f"  {anomaly_type:25s} {status}")

    # Print summary per detector
    print("\nSummary per detector:")
    for detector_name in detector_names:
        detected_count = sum(
            1 for t in anomaly_types
            if detection_matrix.get(detector_name, {}).get(t, False)
        )
        total = len(anomaly_types)
        pct = 100 * detected_count / total if total > 0 else 0
        print(f"  {detector_name}: {detected_count}/{total} ({pct:.0f}%)")

    # Overall summary
    print()
    all_detected = all(
        any(detection_matrix.get(d, {}).get(t, False) for d in detector_names)
        for t in anomaly_types
    )
    
    if all_detected:
        print("PASS: All anomaly types detected by at least one detector")
    else:
        missed = [
            t for t in anomaly_types
            if not any(detection_matrix.get(d, {}).get(t, False) for d in detector_names)
        ]
        print(f"FAIL: {len(missed)} anomaly type(s) not detected: {missed}")

    print("=" * 70)


def build_detection_matrix_frame(
    detection_matrix: dict[str, dict[str, bool]],
) -> pd.DataFrame:
    """Convert the validation detection matrix into a flat analysis table."""
    rows: list[dict[str, object]] = []
    for detector_name, per_type in detection_matrix.items():
        for anomaly_type, detected in per_type.items():
            rows.append(
                {
                    "detector_name": detector_name,
                    "anomaly_type": anomaly_type,
                    "detected": bool(detected),
                }
            )
    return pd.DataFrame(rows)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate anomaly detection against all production anomaly types"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="DK_B2C_mh4",
        help="Model name for ML detectors (default: DK_B2C_mh4)",
    )
    parser.add_argument(
        "--skip-ml",
        action="store_true",
        help="Skip ML detectors (statistical only, faster)",
    )
    parser.add_argument(
        "--base-price",
        type=float,
        default=1000.0,
        help="Base price for test record (default: 1000.0)",
    )
    parser.add_argument(
        "--list-price",
        type=float,
        default=1200.0,
        help="List price for test record (default: 1200.0)",
    )
    parser.add_argument(
        "--history-size",
        type=int,
        default=15,
        help="Number of historical observations (default: 15)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/validation",
        help="Canonical results root (default: results/validation)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id for the canonical output directory",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed detection scores",
    )
    args = parser.parse_args()

    load_dotenv()

    print("=" * 70)
    print("Anomaly Detection Validation")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Skip ML: {args.skip_ml}")
    print(f"Base price: {args.base_price}")
    print(f"List price: {args.list_price}")
    print(f"History size: {args.history_size}")
    print(f"Output dir: {args.output_dir}")

    # Initialize persistence for ML models
    persistence = None
    if not args.skip_ml:
        try:
            persistence = ModelPersistence()
            print(f"ML models from: {persistence.models_root_description}")
        except Exception as e:
            print(f"Warning: Could not initialize ML persistence: {e}")
            print("Falling back to statistical detectors only")
            args.skip_ml = True

    # Create evaluators
    evaluators = create_evaluators(persistence, args.model, args.skip_ml)
    print(f"Evaluators: {[e.name for e in evaluators]}")

    if not evaluators:
        print("Error: No evaluators available")
        sys.exit(1)

    # Create test data
    base_row = create_synthetic_base_record(
        price=args.base_price,
        list_price=args.list_price,
        observation_count=args.history_size,
    )
    train_df = create_training_history(base_row, n_observations=args.history_size)

    # Run validation
    detection_matrix, result, df_test, injection_details = run_validation(
        evaluators=evaluators,
        base_row=base_row,
        train_df=train_df,
        verbose=args.verbose,
    )

    # Print results
    print_results(
        detection_matrix=detection_matrix,
        base_price=args.base_price,
        list_price=args.list_price,
    )

    run_id = args.run_id or create_run_id("validation")
    run_root = Path(args.output_dir) / run_id
    injected_rows, predictions = comparison_result_to_tables(
        result,
        run_id=run_id,
        candidate_id="validation",
        experiment_family="validation",
        dataset_name="validation_synthetic",
        dataset_granularity="synthetic",
        dataset_split="validation",
    )
    run_metadata = {
        "schema_version": "phase2.v1",
        "experiment_family": "validation",
        "run_id": run_id,
        "candidate_id": "validation",
        "source_dataset_paths": [],
        "dataset_names": ["validation_synthetic"],
        "dataset_granularity": "synthetic",
        "dataset_splits": ["validation"],
        "random_seeds": {"generation_seed": 42},
        "injection_config": {
            "base_price": args.base_price,
            "list_price": args.list_price,
            "history_size": args.history_size,
            "anomaly_types": [t.value for t in PRODUCTION_ANOMALY_TYPES],
        },
        "detector_identifiers": [e.name for e in evaluators],
        "config_values": {
            "model": args.model,
            "skip_ml": args.skip_ml,
            "verbose": args.verbose,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": resolve_git_commit(Path(_project_root)),
    }
    write_evaluation_run(
        run_root=run_root,
        run_metadata=run_metadata,
        split_artifacts={"validation": (injected_rows, predictions)},
        analysis_artifacts={
            "validation_detection_matrix.csv": build_detection_matrix_frame(detection_matrix),
        },
    )

    print(f"\nResults saved to {run_root}")


if __name__ == "__main__":
    main()
