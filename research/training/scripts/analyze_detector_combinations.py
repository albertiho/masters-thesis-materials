#!/usr/bin/env python3
"""Analyze detector combinations for CombinedDetector design.

This script captures per-row predictions from all detectors and computes
metrics to inform the CombinedDetector layer composition and ordering.

Outputs:
    1. Overlap Matrix (Jaccard similarity) - Which detectors are complementary?
    2. Cascade Effectiveness - What's the cumulative recall at each layer?
    3. Optimal Ordering - Greedy search for best F1 ordering
    4. History Bucket Analysis - How does performance vary by observation count?

Usage:
    # Run full analysis on both test sets (default)
    python scripts/analyze_detector_combinations.py --model-filter DK_B2B_mh4
    
    # Load ML models from different local paths
    python scripts/analyze_detector_combinations.py \\
        --model-filter DK_B2B_mh4 \\
        --autoencoder-model DK_B2B_mh4 \\
        --iforest-model DK_B2B_mh5
    
    # Test only specific suffix(es)
    python scripts/analyze_detector_combinations.py \\
        --model-filter DK_B2B_mh4 \\
        --file-suffix _test_new_prices _test_new_products
    
    # Save results to specific directory
    python scripts/analyze_detector_combinations.py --output-dir results/detector_combinations/
    
    # Run quietly (save CSV only, no terminal stats)
    python scripts/analyze_detector_combinations.py --model-filter DK_B2B_mh4 --quiet
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from glob import glob
from itertools import permutations
from pathlib import Path
from typing import Any

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
from src.research.artifacts import (
    comparison_result_to_tables,
    create_run_id,
    reindex_split_artifacts,
    resolve_git_commit,
    write_evaluation_run,
)
from src.anomaly.statistical import (
    ZScoreDetector,
    IQRDetector,
    ThresholdDetector,
    SanityCheckDetector,
)
from src.research.evaluation.detector_evaluator import DetectorEvaluator
from src.research.evaluation.synthetic import inject_anomalies_to_dataframe
from src.research.evaluation.test_orchestrator import (
    ComparisonResult,
    DetectorMetrics,
    TestOrchestrator,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class CascadeLayerStats:
    """Statistics for a single layer in a cascade."""
    
    layer_name: str
    layer_position: int
    cumulative_recall: float
    unique_catches: int  # Anomalies caught ONLY by this layer
    unique_catch_rate: float  # unique_catches / total_true_anomalies
    short_circuit_rate: float  # What % flagged at this layer would also be flagged later
    marginal_f1_gain: float  # F1 improvement from adding this layer


@dataclass
class CascadeAnalysis:
    """Full analysis of a cascade layer ordering."""
    
    layer_order: list[str]
    layers: list[CascadeLayerStats]
    final_recall: float
    final_precision: float
    final_f1: float


@dataclass
class AnalysisRunResult:
    """Canonical input bundle for one analyzed file."""

    model_name: str
    dataset_granularity: str
    dataset_split: str
    comparison_result: ComparisonResult


# =============================================================================
# Combined Detector Factory Functions
# =============================================================================


def create_zscore_detector(name: str = "Z-score") -> CombinedDetector:
    """Create Z-score detector wrapped in combined framework."""
    return CombinedDetector(
        config=CombinedDetectorConfig(name=name, min_history_cold=1, min_history_warm=1),
        layers=[DetectorLayer(name="zscore", detectors=[ZScoreDetector()], required_history=0)],
    )


def create_iqr_detector(name: str = "IQR") -> CombinedDetector:
    """Create IQR detector wrapped in combined framework."""
    return CombinedDetector(
        config=CombinedDetectorConfig(name=name, min_history_cold=1, min_history_warm=1),
        layers=[DetectorLayer(name="iqr", detectors=[IQRDetector()], required_history=0)],
    )


def create_threshold_detector(name: str = "Threshold") -> CombinedDetector:
    """Create Threshold detector wrapped in combined framework."""
    return CombinedDetector(
        config=CombinedDetectorConfig(name=name, min_history_cold=1, min_history_warm=1),
        layers=[DetectorLayer(name="threshold", detectors=[ThresholdDetector()], required_history=0)],
    )


def create_sanity_detector(name: str = "Sanity") -> CombinedDetector:
    """Create Sanity detector wrapped in combined framework."""
    return CombinedDetector(
        config=CombinedDetectorConfig(name=name, min_history_cold=1, min_history_warm=1),
        layers=[DetectorLayer(name="sanity", detectors=[SanityCheckDetector()], required_history=0)],
    )


def create_sanity_zscore_detector(name: str = "Sanity+Zscore") -> CombinedDetector:
    """Create two-layer detector: Sanity (gate) -> Z-score.
    
    Architecture:
        Layer 1 (gate): Sanity - catches extreme/impossible values, short-circuits on hit
        Layer 2: Z-score - statistical detection on remaining records
    
    This combination prioritizes catching obvious errors first (zero prices, negative
    prices, extreme spikes) before applying statistical detection.
    """
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


def create_ml_detector(name: str, ml_detector: Any) -> CombinedDetector:
    """Create ML detector wrapped in combined framework.
    
    Wraps any ML detector (Autoencoder or IsolationForest) in the combined
    framework for consistent evaluation.
    """
    return CombinedDetector(
        config=CombinedDetectorConfig(name=name, min_history_cold=1, min_history_warm=1),
        layers=[DetectorLayer(name="ml", detectors=[ml_detector], required_history=0)],
    )


def create_sanity_iforest_detector(iforest_detector: Any, name: str = "Sanity+IForest") -> CombinedDetector:
    """Create two-layer detector: Sanity (gate) -> Isolation Forest.
    
    Architecture:
        Layer 1 (gate): Sanity - catches extreme/impossible values, short-circuits on hit
        Layer 2: Isolation Forest - ML-based anomaly detection on remaining records
    
    This combination prioritizes catching obvious errors first (zero prices, negative
    prices, extreme spikes) before applying ML detection. Short-circuiting removes
    anomalies from the batch before the expensive ML inference.
    """
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


def create_sanity_autoencoder_detector(ae_detector: Any, name: str = "Sanity+AE") -> CombinedDetector:
    """Create two-layer detector: Sanity (gate) -> Autoencoder.
    
    Architecture:
        Layer 1 (gate): Sanity - catches extreme/impossible values, short-circuits on hit
        Layer 2: Autoencoder - reconstruction-based anomaly detection on remaining records
    
    This combination prioritizes catching obvious errors first (zero prices, negative
    prices, extreme spikes) before applying ML detection. Short-circuiting removes
    anomalies from the batch before the expensive ML inference.
    """
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


def create_sanity_ml_ensemble_detector(
    iforest_detector: Any,
    ae_detector: Any,
    name: str = "Sanity+IF+AE",
) -> CombinedDetector:
    """Create two-layer detector: Sanity (gate) -> [Isolation Forest + Autoencoder].
    
    Architecture:
        Layer 1 (gate): Sanity - catches extreme/impossible values, short-circuits on hit
        Layer 2: Both IF and AE run in parallel, votes aggregated
    
    This combination uses sanity as a fast filter, then runs both ML detectors
    on the remaining records. Both detectors contribute votes to the final decision.
    """
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
                name="ml_ensemble",
                detectors=[iforest_detector, ae_detector],
                required_history=0,
            ),
        ],
    )


def create_sanity_zscore_ae_detector(
    ae_detector: Any,
    name: str = "Sanity+Zscore+AE",
) -> CombinedDetector:
    """Create three-layer detector: Sanity (gate) -> Z-score (gate) -> Autoencoder.
    
    Architecture:
        Layer 1 (gate): Sanity - catches extreme/impossible values, short-circuits on hit
        Layer 2 (gate): Z-score - statistical detection, short-circuits on hit
        Layer 3: Autoencoder - ML-based detection on remaining records
    
    This combination filters obvious errors first, then statistical anomalies,
    before applying expensive ML inference on the remaining records.
    """
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


def create_sanity_zscore_iforest_detector(
    iforest_detector: Any,
    name: str = "Sanity+cold_iforest+warm_zscore",
) -> CombinedDetector:
    """Create three-layer detector: Sanity (gate) -> Z-score (gate) -> Isolation Forest.
    
    Architecture:
        Layer 1 (gate): Sanity - catches extreme/impossible values, short-circuits on hit
        Layer 2 (gate): Z-score - statistical detection, short-circuits on hit
        Layer 3: Isolation Forest - ML-based detection on remaining records
    
    This combination filters obvious errors first, then statistical anomalies,
    before applying expensive ML inference on the remaining records.
    """
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
               # maximum_history=10,
            ),
            DetectorLayer(
                name="zscore",
                detectors=[ZScoreDetector()],
                required_history=3,
            ),
        ],
    )


# =============================================================================
# File Discovery (copied from compare_detectors.py)
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
    else:
        raise ValueError(f"Unknown granularity: {granularity}")
    if not file_suffix:
        files = [f for f in files if not f.endswith("_train.parquet") and not f.endswith("_test.parquet")]

    return sorted(files)


def find_matching_train_file(test_filepath: str, file_suffix: str) -> str | None:
    """Find the matching train file for a test file.
    
    Examples:
        test_filepath: data/training/by_country_segment/NO_B2C_2026-01-22_test_new_prices_mh4.parquet
        file_suffix: _test_new_prices_mh4
        -> Should find: NO_B2C_2026-01-22_train_mh4.parquet
    """
    import re
    
    basename = os.path.basename(test_filepath)
    name = basename.replace(".parquet", "")
    
    # Extract _mhN suffix from file_suffix (e.g., _test_new_prices_mh4 -> _mh4)
    mh_match = re.search(r"(_mh\d+)$", file_suffix)
    mh_suffix = mh_match.group(1) if mh_match else ""
    
    # Remove _mhN from file_suffix for matching (e.g., _test_new_prices_mh4 -> _test_new_prices)
    test_suffix_without_mh = re.sub(r"_mh\d+$", "", file_suffix)
    
    # Build the base name by removing test suffix and mh suffix
    # e.g., NO_B2C_2026-01-22_test_new_prices_mh4 -> NO_B2C_2026-01-22
    base_name = name
    if mh_suffix and base_name.endswith(mh_suffix):
        base_name = base_name[:-len(mh_suffix)]
    if test_suffix_without_mh and base_name.endswith(test_suffix_without_mh):
        base_name = base_name[:-len(test_suffix_without_mh)]
    
    # Construct train filename with mh suffix
    train_filename = f"{base_name}_train{mh_suffix}.parquet"
    train_filepath = os.path.join(os.path.dirname(test_filepath), train_filename)
    
    print(f"[DEBUG] Test file: {basename}")
    print(f"[DEBUG] file_suffix: {file_suffix}, mh_suffix: {mh_suffix}")
    print(f"[DEBUG] Looking for train file: {train_filename}")
    print(f"[DEBUG] Full path: {train_filepath}")
    print(f"[DEBUG] Exists: {os.path.exists(train_filepath)}")
    
    if os.path.exists(train_filepath):
        return train_filepath
    
    return None


def extract_model_name(filepath: str) -> str:
    """Extract model name from Parquet filename."""
    import re
    
    filename = os.path.basename(filepath)
    name = filename.replace(".parquet", "")

    mh_suffix = ""
    mh_match = re.search(r"(_mh\d+)$", name)
    if mh_match:
        mh_suffix = mh_match.group(1)
        name = name[: -len(mh_suffix)]

    for suffix in ["_train", "_test", "_test_new_products", "_test_new_prices"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break

    date_pattern = r"_\d{4}-\d{2}-\d{2}$"
    name = re.sub(date_pattern, "", name)

    return name + mh_suffix


def extract_country(model_name: str | None) -> str | None:
    """Extract country code from model name."""
    import re
    
    if not model_name:
        return None

    match = re.match(r"^([A-Z]{2})_", model_name)
    if match:
        return match.group(1)

    return None


# =============================================================================
# Analysis Functions
# =============================================================================


def compute_jaccard_similarity(pred_a: np.ndarray, pred_b: np.ndarray) -> float:
    """Compute Jaccard similarity between two boolean arrays.
    
    Jaccard = |intersection| / |union|
    """
    intersection = np.sum(pred_a & pred_b)
    union = np.sum(pred_a | pred_b)
    
    if union == 0:
        return 0.0
    
    return intersection / union


def compute_overlap_matrix(
    predictions: dict[str, np.ndarray],
    labels: np.ndarray,
) -> pd.DataFrame:
    """Compute Jaccard similarity matrix between detector TRUE POSITIVE sets.
    
    We care about overlap of TRUE POSITIVES (correctly caught anomalies),
    not all predictions, since that's what determines complementarity.
    
    Args:
        predictions: Dict mapping detector names to boolean prediction arrays.
        labels: Ground truth labels.
    
    Returns:
        DataFrame with Jaccard similarity between each detector pair.
    """
    detector_names = list(predictions.keys())
    n = len(detector_names)
    
    # Compute true positive sets (predictions AND labels)
    tp_sets = {
        name: preds & labels
        for name, preds in predictions.items()
    }
    
    # Compute pairwise Jaccard similarity
    matrix = np.zeros((n, n))
    for i, name_i in enumerate(detector_names):
        for j, name_j in enumerate(detector_names):
            matrix[i, j] = compute_jaccard_similarity(tp_sets[name_i], tp_sets[name_j])
    
    return pd.DataFrame(matrix, index=detector_names, columns=detector_names)


def compute_f1_from_predictions(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float, float]:
    """Compute precision, recall, F1 from predictions and labels."""
    tp = np.sum(predictions & labels)
    fp = np.sum(predictions & ~labels)
    fn = np.sum(~predictions & labels)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return float(precision), float(recall), float(f1)


def analyze_cascade(
    predictions: dict[str, np.ndarray],
    labels: np.ndarray,
    layer_order: list[str],
) -> CascadeAnalysis:
    """Analyze cascade effectiveness for a given layer order.
    
    For each layer, computes:
    - Cumulative recall: % of true anomalies caught by layers 1..N
    - Unique catches: Anomalies caught ONLY by this layer (not by prior layers)
    - Short-circuit rate: What % flagged at this layer would be flagged by later layers
    - Marginal F1 gain: F1 improvement from adding this layer
    
    Args:
        predictions: Dict mapping detector names to boolean prediction arrays.
        labels: Ground truth labels.
        layer_order: List of detector names in cascade order.
    
    Returns:
        CascadeAnalysis with per-layer statistics.
    """
    total_true_anomalies = np.sum(labels)
    if total_true_anomalies == 0:
        return CascadeAnalysis(
            layer_order=layer_order,
            layers=[],
            final_recall=0.0,
            final_precision=0.0,
            final_f1=0.0,
        )
    
    layers: list[CascadeLayerStats] = []
    cumulative_predictions = np.zeros(len(labels), dtype=bool)
    prev_f1 = 0.0
    
    for i, layer_name in enumerate(layer_order):
        if layer_name not in predictions:
            continue
        
        layer_preds = predictions[layer_name]
        
        # Unique catches: TP in this layer but NOT in prior layers
        unique_catches = np.sum(layer_preds & labels & ~cumulative_predictions)
        unique_catch_rate = unique_catches / total_true_anomalies
        
        # Short-circuit rate: What % of this layer's flags would be caught by later layers
        remaining_layers = layer_order[i + 1:]
        if remaining_layers:
            later_preds = np.zeros(len(labels), dtype=bool)
            for later_name in remaining_layers:
                if later_name in predictions:
                    later_preds |= predictions[later_name]
            
            this_layer_flags = np.sum(layer_preds)
            would_be_caught_later = np.sum(layer_preds & later_preds)
            short_circuit_rate = would_be_caught_later / this_layer_flags if this_layer_flags > 0 else 0.0
        else:
            short_circuit_rate = 0.0  # Last layer - nothing to short-circuit
        
        # Update cumulative predictions
        cumulative_predictions |= layer_preds
        
        # Compute cumulative recall and F1
        _, recall, _ = compute_f1_from_predictions(cumulative_predictions, labels)
        precision, _, f1 = compute_f1_from_predictions(cumulative_predictions, labels)
        
        marginal_f1_gain = f1 - prev_f1
        prev_f1 = f1
        
        layers.append(CascadeLayerStats(
            layer_name=layer_name,
            layer_position=i + 1,
            cumulative_recall=recall,
            unique_catches=int(unique_catches),
            unique_catch_rate=unique_catch_rate,
            short_circuit_rate=short_circuit_rate,
            marginal_f1_gain=marginal_f1_gain,
        ))
    
    # Final metrics
    final_precision, final_recall, final_f1 = compute_f1_from_predictions(
        cumulative_predictions, labels
    )
    
    return CascadeAnalysis(
        layer_order=layer_order,
        layers=layers,
        final_recall=final_recall,
        final_precision=final_precision,
        final_f1=final_f1,
    )


def find_optimal_ordering(
    predictions: dict[str, np.ndarray],
    labels: np.ndarray,
) -> list[tuple[str, float, float]]:
    """Find optimal detector ordering using greedy algorithm.
    
    Algorithm:
    1. Start with best single detector (highest F1)
    2. Add detector that maximizes F1 gain (union of predictions)
    3. Repeat until no improvement or all detectors added
    
    Args:
        predictions: Dict mapping detector names to boolean prediction arrays.
        labels: Ground truth labels.
    
    Returns:
        List of (detector_name, cumulative_f1, marginal_gain) tuples in recommended order.
    """
    detector_names = list(predictions.keys())
    remaining = set(detector_names)
    order: list[tuple[str, float, float]] = []
    cumulative_preds = np.zeros(len(labels), dtype=bool)
    prev_f1 = 0.0
    
    while remaining:
        best_name = None
        best_f1 = prev_f1
        best_gain = 0.0
        
        for name in remaining:
            # Try adding this detector
            test_preds = cumulative_preds | predictions[name]
            _, _, f1 = compute_f1_from_predictions(test_preds, labels)
            
            gain = f1 - prev_f1
            if f1 > best_f1:
                best_name = name
                best_f1 = f1
                best_gain = gain
        
        if best_name is None:
            # No improvement - stop
            break
        
        remaining.remove(best_name)
        cumulative_preds |= predictions[best_name]
        order.append((best_name, best_f1, best_gain))
        prev_f1 = best_f1
        
        # Stop if no marginal gain (diminishing returns)
        if best_gain < 0.001:  # 0.1% threshold
            # Still add remaining for completeness but mark as low-value
            for name in sorted(remaining):
                test_preds = cumulative_preds | predictions[name]
                _, _, f1 = compute_f1_from_predictions(test_preds, labels)
                order.append((name, f1, f1 - prev_f1))
                cumulative_preds |= predictions[name]
                prev_f1 = f1
            break
    
    return order


def analyze_by_history_bucket(
    predictions: dict[str, np.ndarray],
    labels: np.ndarray,
    observation_counts: np.ndarray,
    buckets: list[tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """Analyze detector performance by history depth bucket.
    
    Args:
        predictions: Dict mapping detector names to boolean prediction arrays.
        labels: Ground truth labels.
        observation_counts: Observation count for each row.
        buckets: List of (min_obs, max_obs) tuples defining buckets.
            Default: [(0, 3), (3, 5), (5, 10), (10, 30), (30, inf)]
    
    Returns:
        DataFrame with F1 per detector per bucket.
    """
    if buckets is None:
        buckets = [
            (0, 3),    # Cold start (min_history_cold)
            (3, 5),    # Limited history
            (5, 10),   # Warm (min_history_warm)
            (10, 30),  # Full history
            (30, np.inf),  # Very established products
        ]
    
    rows = []
    detector_names = list(predictions.keys())
    
    for min_obs, max_obs in buckets:
        bucket_mask = (observation_counts >= min_obs) & (observation_counts < max_obs)
        n_samples = np.sum(bucket_mask)
        n_anomalies = np.sum(labels[bucket_mask])
        
        if n_samples == 0:
            continue
        
        bucket_name = f"{min_obs}-{int(max_obs) if max_obs < np.inf else '+'}"
        
        for detector_name in detector_names:
            bucket_preds = predictions[detector_name][bucket_mask]
            bucket_labels = labels[bucket_mask]
            
            precision, recall, f1 = compute_f1_from_predictions(bucket_preds, bucket_labels)
            
            rows.append({
                "bucket": bucket_name,
                "min_obs": min_obs,
                "max_obs": max_obs if max_obs < np.inf else 999,
                "detector": detector_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "n_samples": n_samples,
                "n_anomalies": n_anomalies,
            })
    
    return pd.DataFrame(rows)


# =============================================================================
# Evaluator Creation
# =============================================================================


def create_evaluators(
    persistence: ModelPersistence | None,
    model_name: str,
    skip_ml: bool = False,
    autoencoder_model: str | None = None,
    iforest_model: str | None = None,
) -> list[DetectorEvaluator]:
    """Create DetectorEvaluator instances for all detectors.
    
    All detectors are wrapped in the combined framework (single-layer) to validate
    that the combined framework produces identical results to raw detectors.
    
    Args:
        persistence: ModelPersistence instance for loading ML models.
        model_name: Default model name (derived from test file).
        skip_ml: If True, skip ML detectors.
        autoencoder_model: Override model name for Autoencoder (default: model_name).
        iforest_model: Override model name for Isolation Forest (default: model_name).
    
    Returns:
        List of DetectorEvaluator instances.
    """
    evaluators = []
    
    
    # Statistical detectors wrapped in combined framework (single layer each)
    # evaluators.append(DetectorEvaluator(create_zscore_detector(), "Z-score"))
    # evaluators.append(DetectorEvaluator(create_iqr_detector(), "IQR"))
    # evaluators.append(DetectorEvaluator(create_threshold_detector(), "Threshold"))
    # evaluators.append(DetectorEvaluator(create_sanity_detector(), "Sanity"))
    
    # Combined multi-layer detectors
    evaluators.append(DetectorEvaluator(create_sanity_zscore_detector(), "Sanity+Zscore"))
    
    # ML detectors wrapped in combined framework (if available)
    if not skip_ml and persistence is not None:
        ae_detector = None
        if_detector = None
        
        # Autoencoder
        ae_model = autoencoder_model or model_name
        try:
            ae_detector = persistence.load_autoencoder(ae_model)
            # evaluators.append(DetectorEvaluator(create_ml_detector("Autoencoder", ae_detector), "Autoencoder"))
        except Exception as e:
            logger.warning(f"Could not load Autoencoder for {ae_model}: {e}")
        
        # Isolation Forest
        if_model = iforest_model or model_name
        try:
            if_detector = persistence.load_isolation_forest(if_model)
            # evaluators.append(DetectorEvaluator(create_ml_detector("IsolationForest", if_detector), "IsolationForest"))
        except Exception as e:
            logger.warning(f"Could not load Isolation Forest for {if_model}: {e}")

        # Combined: Sanity (gate) -> Isolation Forest
        if if_detector is not None:
            evaluators.append(
                DetectorEvaluator(
                    create_sanity_iforest_detector(if_detector),
                    "Sanity+IForest",
                )
            )
        

        if if_detector is not None:
            evaluators.append(
                DetectorEvaluator(
                    create_sanity_zscore_iforest_detector(if_detector),
                    "Sanity+cold_iforest+warm_zscore",
                )
            )
        
    
    return evaluators
        



# =============================================================================
# Output Functions
# =============================================================================


def save_overlap_matrix(overlap_df: pd.DataFrame, output_dir: str) -> None:
    """Save overlap matrix to CSV."""
    path = os.path.join(output_dir, "overlap_matrix.csv")
    overlap_df.to_csv(path)
    print(f"Saved overlap matrix to {path}")


def save_cascade_analysis(
    cascade: CascadeAnalysis,
    output_dir: str,
    filename: str = "cascade_analysis.md",
) -> None:
    """Save cascade analysis as markdown."""
    path = os.path.join(output_dir, filename)
    
    lines = [
        "# Cascade Effectiveness Analysis",
        "",
        f"Layer Order: {' -> '.join(cascade.layer_order)}",
        "",
        f"Final Performance: P={cascade.final_precision:.1%}, R={cascade.final_recall:.1%}, F1={cascade.final_f1:.1%}",
        "",
        "## Per-Layer Statistics",
        "",
        "| Layer | Position | Cumulative Recall | Unique Catches | Unique Rate | Short-Circuit | Marginal F1 |",
        "|-------|----------|-------------------|----------------|-------------|---------------|-------------|",
    ]
    
    for layer in cascade.layers:
        lines.append(
            f"| {layer.layer_name} | {layer.layer_position} | "
            f"{layer.cumulative_recall:.1%} | {layer.unique_catches} | "
            f"{layer.unique_catch_rate:.1%} | {layer.short_circuit_rate:.1%} | "
            f"{'+' if layer.marginal_f1_gain >= 0 else ''}{layer.marginal_f1_gain:.1%} |"
        )
    
    with open(path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"Saved cascade analysis to {path}")


def save_optimal_order(
    order: list[tuple[str, float, float]],
    output_dir: str,
) -> None:
    """Save optimal ordering as JSON."""
    path = os.path.join(output_dir, "optimal_order.json")
    
    data = {
        "recommended_order": [item[0] for item in order],
        "details": [
            {
                "detector": name,
                "cumulative_f1": f1,
                "marginal_gain": gain,
            }
            for name, f1, gain in order
        ],
        "analysis_timestamp": datetime.now().isoformat(),
    }
    
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved optimal order to {path}")


def save_history_buckets(history_df: pd.DataFrame, output_dir: str) -> None:
    """Save history bucket analysis to CSV."""
    path = os.path.join(output_dir, "history_buckets.csv")
    history_df.to_csv(path, index=False)
    print(f"Saved history buckets to {path}")


def save_detector_predictions(
    result: ComparisonResult,
    injection_details: list[dict],
    df_injected: pd.DataFrame,
    model_name: str,
    file_suffix: str,
    output_dir: str,
) -> None:
    """Save per-row predictions for each detector as separate CSVs.
    
    Creates one CSV per detector with columns:
    - index: Row index in the test DataFrame
    - competitor_product_id: Product identifier
    - anomaly_type: Type of injected anomaly (or None if not injected)
    - price_before: Original price before injection (or current price if not injected)
    - price_after: Price after injection (or current price if not injected)
    - label: Ground truth (1=anomaly, 0=normal)
    - is_anomaly: Detector prediction (1=flagged, 0=not flagged)
    - anomaly_score: Detector confidence score
    
    Args:
        result: ComparisonResult containing raw_results, labels, and df_sorted.
        injection_details: List of dicts with 'index', 'anomaly_type', 'original_price', 'new_price'.
        df_injected: DataFrame with injected anomalies (fallback for prices).
        model_name: Model name for filename prefix.
        file_suffix: Test file suffix (e.g., "_test_new_prices_mh4") for filename.
        output_dir: Directory to save CSVs to.
    """
    import re
    
    # Create predictions subdirectory
    predictions_dir = os.path.join(output_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    
    # Prefer sorted DataFrame from result (aligned with raw_results after time sorting)
    df_aligned = result.df_sorted if result.df_sorted is not None else df_injected
    
    # Get price column name
    price_column = "unit_price" if "unit_price" in df_aligned.columns else "price"
    
    # Extract test type from suffix for filename (e.g., "_test_new_prices_mh4" -> "test_new_prices")
    test_type = file_suffix.lstrip("_")
    test_type = re.sub(r"_mh\d+$", "", test_type)
    
    # Save one CSV per detector
    for detector_name, anomaly_results in result.raw_results.items():
        rows = []
        for i, ar in enumerate(anomaly_results):
            # Read directly from DataFrame columns - data is aligned after sorting
            anomaly_type = df_aligned.iloc[i]["__injected_anomaly_type__"]
            price_before = df_aligned.iloc[i]["__original_price__"]
            price_after = df_aligned.iloc[i][price_column] if price_column in df_aligned.columns else price_before
            
            # Extract is_valid from details (different keys for different detectors)
            is_valid = ar.details.get("is_valid_input", ar.details.get("feature_vector_valid", True))
            
            rows.append({
                "index": i,
                "competitor_product_id": ar.competitor_product_id,
                "anomaly_type": anomaly_type,
                "price_before": price_before,
                "price_after": price_after,
                "label": int(result.labels[i]),
                "is_anomaly": int(ar.is_anomaly),
                "anomaly_score": ar.anomaly_score,
                "is_valid": int(is_valid),
            })
        
        df = pd.DataFrame(rows)
        
        # Sanitize detector name for filename
        safe_detector_name = detector_name.replace("+", "_").replace(" ", "_")
        filename = f"{model_name}_{test_type}_{safe_detector_name}.csv"
        path = os.path.join(predictions_dir, filename)
        df.to_csv(path, index=False)
    
    print(f"Saved {len(result.raw_results)} detector predictions to {predictions_dir}/")


def compute_detector_metrics_by_type(
    results_by_suffix: dict[str, list[Any]],
) -> pd.DataFrame:
    """Compute precision, recall, F1 per detector per test type.
    
    Args:
        results_by_suffix: Dict mapping suffix to analyzed results.
    
    Returns:
        DataFrame with columns: detector, test_type, precision, recall, f1
    """
    rows = []
    
    for suffix, results in results_by_suffix.items():
        if not results:
            continue
        
        # Extract test type from suffix (e.g., "_test_new_prices_mh4" -> "new_prices")
        import re
        test_type = suffix.lstrip("_")
        test_type = re.sub(r"_mh\d+$", "", test_type)  # Remove _mhN
        test_type = test_type.replace("test_", "")  # Remove test_ prefix
        
        # Aggregate predictions and labels for this suffix
        predictions, labels, _ = aggregate_comparison_results(results)
        
        # Compute metrics per detector
        for detector_name, preds in predictions.items():
            precision, recall, f1 = compute_f1_from_predictions(preds, labels)
            rows.append({
                "detector": detector_name,
                "test_type": test_type,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            })
    
    return pd.DataFrame(rows)


def print_analysis_summary(
    overlap_df: pd.DataFrame,
    cascade: CascadeAnalysis,
    optimal_order: list[tuple[str, float, float]],
    history_df: pd.DataFrame,
    results_by_suffix: dict[str, list[Any]] | None = None,
) -> None:
    """Print summary of analysis results."""
    print("\n" + "=" * 70)
    print("DETECTOR COMBINATION ANALYSIS SUMMARY")
    print("=" * 70)
    
    # Per-detector metrics by test type (new detailed summary)
    if results_by_suffix:
        metrics_df = compute_detector_metrics_by_type(results_by_suffix)
        
        if not metrics_df.empty:
            print("\n## Detector Performance by Test Type")
            print("-" * 90)
            
            # Get unique detectors and test types
            detectors = sorted(metrics_df["detector"].unique())
            test_types = sorted(metrics_df["test_type"].unique())
            
            # Build header
            header_parts = ["Detector".ljust(20)]
            for tt in test_types:
                header_parts.append(f"{tt} P/R/F1".center(20))
            header_parts.append("Combined F1".center(12))
            print(" | ".join(header_parts))
            print("-" * 90)
            
            # Compute combined metrics for each detector
            combined_predictions, combined_labels, _ = aggregate_comparison_results(
                [r for results in results_by_suffix.values() for r in results]
            )
            
            # Print row for each detector
            for detector in detectors:
                row_parts = [detector.ljust(20)]
                
                for tt in test_types:
                    row = metrics_df[(metrics_df["detector"] == detector) & (metrics_df["test_type"] == tt)]
                    if not row.empty:
                        p = row["precision"].values[0]
                        r = row["recall"].values[0]
                        f = row["f1"].values[0]
                        row_parts.append(f"{p:.1%}/{r:.1%}/{f:.1%}".center(20))
                    else:
                        row_parts.append("N/A".center(20))
                
                # Combined F1
                if detector in combined_predictions:
                    _, _, combined_f1 = compute_f1_from_predictions(
                        combined_predictions[detector], combined_labels
                    )
                    row_parts.append(f"{combined_f1:.1%}".center(12))
                else:
                    row_parts.append("N/A".center(12))
                
                print(" | ".join(row_parts))
            
            print("-" * 90)
            print("Format: Precision/Recall/F1")
    
    # Overlap matrix summary
    print("\n## Overlap Matrix (Jaccard Similarity of True Positives)")
    print("-" * 60)
    print(overlap_df.round(2).to_string())
    
    # Find complementary pairs (low overlap)
    print("\n## Most Complementary Pairs (Lowest Overlap)")
    pairs = []
    detectors = list(overlap_df.index)
    for i, d1 in enumerate(detectors):
        for j, d2 in enumerate(detectors):
            if i < j:
                pairs.append((d1, d2, overlap_df.loc[d1, d2]))
    
    pairs.sort(key=lambda x: x[2])
    for d1, d2, overlap in pairs[:5]:
        print(f"  {d1} + {d2}: {overlap:.2f}")
    
    # Optimal ordering
    print("\n## Recommended Cascade Order (Greedy Optimization)")
    print("-" * 60)
    for i, (name, f1, gain) in enumerate(optimal_order, 1):
        gain_str = f"+{gain:.1%}" if gain >= 0 else f"{gain:.1%}"
        print(f"  {i}. {name}: F1={f1:.1%} (marginal: {gain_str})")
    
    # Cascade analysis for recommended order
    print(f"\n## Cascade Analysis for Recommended Order")
    print("-" * 60)
    print(f"Final: P={cascade.final_precision:.1%}, R={cascade.final_recall:.1%}, F1={cascade.final_f1:.1%}")
    
    for layer in cascade.layers:
        print(f"  Layer {layer.layer_position} ({layer.layer_name}):")
        print(f"    Cumulative Recall: {layer.cumulative_recall:.1%}")
        print(f"    Unique Catches: {layer.unique_catches} ({layer.unique_catch_rate:.1%})")
        print(f"    Short-Circuit Rate: {layer.short_circuit_rate:.1%}")
    
    # History bucket summary
    print("\n## Performance by History Bucket (F1 scores)")
    print("-" * 60)
    
    pivot = history_df.pivot(index="detector", columns="bucket", values="f1")
    if not pivot.empty:
        print(pivot.round(2).to_string())
    
    # Recommendations
    print("\n## Recommendations")
    print("-" * 60)
    
    if optimal_order:
        top_3 = [name for name, _, _ in optimal_order[:3]]
        print(f"1. Primary ensemble layers: {' -> '.join(top_3)}")
    
    if len(optimal_order) >= 4:
        diminishing = [(name, gain) for name, _, gain in optimal_order if gain < 0.01]
        if diminishing:
            print(f"2. Low marginal value (consider removing): {[name for name, _ in diminishing]}")
    
    # Check cold start performance
    cold_start_df = history_df[history_df["bucket"].str.startswith("0")]
    if not cold_start_df.empty:
        best_cold = cold_start_df.loc[cold_start_df["f1"].idxmax()]
        print(f"3. Best cold-start detector: {best_cold['detector']} (F1={best_cold['f1']:.1%})")
    
    print("\n" + "=" * 70)


# =============================================================================
# Main Analysis
# =============================================================================


def run_analysis_on_file(
    filepath: str,
    persistence: ModelPersistence | None,
    file_suffix: str,
    dataset_granularity: str,
    injection_rate: float = 0.1,
    skip_ml: bool = False,
    autoencoder_model: str | None = None,
    iforest_model: str | None = None,
) -> AnalysisRunResult | None:
    """Run detector comparison on a single file and return detailed results.
    
    Args:
        filepath: Path to test parquet file.
        persistence: ModelPersistence instance for loading ML models.
        file_suffix: Suffix of the test file (e.g., "_test_new_prices").
        dataset_granularity: Dataset granularity for canonical metadata.
        injection_rate: Fraction of data to inject as anomalies.
        skip_ml: If True, skip ML detectors.
        autoencoder_model: Override model name for Autoencoder.
        iforest_model: Override model name for Isolation Forest.
    
    Returns:
        AnalysisRunResult or None if analysis failed.
    """
    model_name = extract_model_name(filepath)
    print(f"\n{'='*60}")
    print(f"Analyzing: {model_name}")
    print(f"{'='*60}")
    
    # Load test data
    df_test = pd.read_parquet(filepath)
    n_test_samples = len(df_test)
    n_test_products = df_test["product_id"].nunique()
    print(f"Test data: {n_test_samples:,} rows, {n_test_products:,} products")
    
    if n_test_samples < 100:
        print(f"Skipping {model_name}: insufficient data")
        return None
    
    # Load train data for cache population
    train_df = None
    if "_test_new_prices" in file_suffix:
        print(f"\n[DEBUG] === TRAIN DATA LOADING ===")
        train_filepath = find_matching_train_file(filepath, file_suffix)
        if train_filepath:
            train_df = pd.read_parquet(train_filepath)
            print(f"[DEBUG] Training data: {len(train_df):,} rows")
            
            # Check product overlap
            train_products = set(train_df["product_id"].unique())
            test_products = set(df_test["product_id"].unique())
            overlap = train_products & test_products
            print(f"[DEBUG] Train products: {len(train_products):,}")
            print(f"[DEBUG] Test products: {len(test_products):,}")
            print(f"[DEBUG] Overlap: {len(overlap):,} ({len(overlap)/len(test_products)*100:.1f}% of test)")
            
            # Check observation counts per product in train
            obs_per_product = train_df.groupby("product_id").size()
            print(f"[DEBUG] Observations per product in train:")
            print(f"[DEBUG]   Min: {obs_per_product.min()}, Max: {obs_per_product.max()}, Mean: {obs_per_product.mean():.1f}")
            print(f"[DEBUG]   Products with >= 3 obs: {(obs_per_product >= 3).sum()} ({(obs_per_product >= 3).sum()/len(obs_per_product)*100:.1f}%)")
        else:
            print(f"[DEBUG] WARNING: No train file found!")
    else:
        print(f"[DEBUG] Skipping train data loading (not new_prices test)")
    
    # Inject anomalies using unified injection (6 anomaly types)
    df_injected, labels, injection_details = inject_anomalies_to_dataframe(
        df_test, injection_rate=injection_rate, seed=42
    )
    print(f"Injected {np.sum(labels):,} anomalies ({injection_rate:.0%} rate)")
    
    # Create evaluators
    evaluators = create_evaluators(
        persistence,
        model_name,
        skip_ml,
        autoencoder_model=autoencoder_model,
        iforest_model=iforest_model,
    )
    print(f"Evaluating {len(evaluators)} detectors: {[e.name for e in evaluators]}")
    
    # Run comparison with detailed results
    orchestrator = TestOrchestrator(evaluators, max_workers=len(evaluators))
    country = extract_country(model_name)
    
    try:
        result = orchestrator.run_comparison_with_details(
            train_df, df_injected, labels, country
        )
        return AnalysisRunResult(
            model_name=model_name,
            dataset_granularity=dataset_granularity,
            dataset_split=file_suffix,
            comparison_result=result,
        )
    except Exception as e:
        logger.error(f"Analysis failed for {model_name}: {e}")
        return None


def aggregate_comparison_results(
    results: list[Any],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Aggregate results from multiple files into single arrays.
    
    Args:
        results: List of ComparisonResult from different files.
    
    Returns:
        Tuple of (predictions_dict, combined_labels, combined_obs_counts)
    """
    if not results:
        return {}, np.array([]), np.array([])
    
    # Get all detector names from first result
    all_detector_names = set()
    normalized_results: list[ComparisonResult] = []
    for item in results:
        result = item.comparison_result if hasattr(item, "comparison_result") else item
        normalized_results.append(result)
        all_detector_names.update(result.raw_results.keys())
    
    # Initialize aggregated arrays
    all_predictions: dict[str, list[np.ndarray]] = {name: [] for name in all_detector_names}
    all_labels: list[np.ndarray] = []
    all_obs_counts: list[np.ndarray] = []
    
    for i, result in enumerate(normalized_results):
        all_labels.append(result.labels)
        all_obs_counts.append(result.observation_counts)
        
        for name in all_detector_names:
            if name in result.raw_results:
                preds = np.array([r.is_anomaly for r in result.raw_results[name]])
            else:
                # Detector not available for this file - use False
                preds = np.zeros(len(result.labels), dtype=bool)
            all_predictions[name].append(preds)
    
    # Concatenate
    combined_predictions = {
        name: np.concatenate(preds) for name, preds in all_predictions.items()
    }
    combined_labels = np.concatenate(all_labels)
    combined_obs_counts = np.concatenate(all_obs_counts)
    
    return combined_predictions, combined_labels, combined_obs_counts


def build_canonical_run_artifacts(
    results_by_suffix: dict[str, list[AnalysisRunResult]],
    run_id: str,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Build canonical split artifacts from analyzed files."""
    split_artifacts: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    for suffix, results in results_by_suffix.items():
        injected_frames: list[pd.DataFrame] = []
        prediction_frames: list[pd.DataFrame] = []

        for result in results:
            observation_counts = result.comparison_result.observation_counts
            injected_rows, predictions = comparison_result_to_tables(
                result.comparison_result,
                run_id=run_id,
                candidate_id="",
                experiment_family="detector_combinations",
                dataset_name=result.model_name,
                dataset_granularity=result.dataset_granularity,
                dataset_split=suffix,
                injected_row_extras={"observation_count": observation_counts},
            )
            injected_frames.append(injected_rows)
            prediction_frames.append(predictions)

        if injected_frames:
            injected_frames, prediction_frames = zip(
                *reindex_split_artifacts(list(zip(injected_frames, prediction_frames)))
            )
            split_artifacts[suffix] = (
                pd.concat(injected_frames, ignore_index=True),
                pd.concat(prediction_frames, ignore_index=True),
            )

    return split_artifacts


def aggregate_from_canonical_tables(
    split_artifacts: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Aggregate prediction arrays from canonical row tables."""
    if not split_artifacts:
        return {}, np.array([]), np.array([])

    injected = pd.concat([tables[0] for tables in split_artifacts.values()], ignore_index=True)
    predictions = pd.concat([tables[1] for tables in split_artifacts.values()], ignore_index=True)
    sort_columns = ["dataset_split", "dataset_name", "candidate_id", "evaluation_row_id"]
    join_columns = ["dataset_split", "candidate_id", "evaluation_row_id"]

    injected_sorted = injected.sort_values(sort_columns).reset_index(drop=True)
    labels = injected_sorted["ground_truth_label"].fillna(False).astype(bool).to_numpy()
    if "observation_count" in injected_sorted.columns:
        obs_counts = (
            pd.to_numeric(injected_sorted["observation_count"], errors="coerce")
            .fillna(0)
            .astype(np.int32)
            .to_numpy()
        )
    else:
        obs_counts = np.zeros(len(injected_sorted), dtype=np.int32)

    predictions_sorted = predictions.merge(
        injected_sorted[join_columns],
        on=join_columns,
        how="right",
        validate="many_to_one",
    )
    prediction_map: dict[str, np.ndarray] = {}
    for detector_name, group in predictions_sorted.groupby("detector_name", sort=True):
        detector_group = group.sort_values(join_columns).reset_index(drop=True)
        prediction_map[detector_name] = (
            detector_group["predicted_is_anomaly"].fillna(False).astype(bool).to_numpy()
        )

    return prediction_map, labels, obs_counts


def cascade_to_dict(cascade: CascadeAnalysis) -> dict[str, Any]:
    """Convert cascade analysis to a JSON-safe dictionary."""
    return {
        "layer_order": cascade.layer_order,
        "final_recall": cascade.final_recall,
        "final_precision": cascade.final_precision,
        "final_f1": cascade.final_f1,
        "layers": [
            {
                "layer_name": layer.layer_name,
                "layer_position": layer.layer_position,
                "cumulative_recall": layer.cumulative_recall,
                "unique_catches": layer.unique_catches,
                "unique_catch_rate": layer.unique_catch_rate,
                "short_circuit_rate": layer.short_circuit_rate,
                "marginal_f1_gain": layer.marginal_f1_gain,
            }
            for layer in cascade.layers
        ],
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze detector combinations")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/training",
        help="Data directory (default: data/training)",
    )
    parser.add_argument(
        "--file-suffix",
        type=str,
        nargs="+",
        default=["_test_new_prices_mh5", "_test_new_products_mh5"],
        help="Suffix(es) for test files (default: both _test_new_prices_mh5 and _test_new_products_mh5)",
    )
    parser.add_argument(
        "--granularity",
        type=str,
        choices=["country_segment", "competitor"],
        default="country_segment",
        help="Model granularity to analyze (default: country_segment)",
    )
    parser.add_argument(
        "--model-filter",
        type=str,
        default=None,
        help="Only analyze models matching this substring",
    )
    parser.add_argument(
        "--injection-rate",
        type=float,
        default=0.1,
        help="Fraction of data to inject as anomalies (default: 0.1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/detector_combinations",
        help="Output directory for results (default: results/detector_combinations)",
    )
    parser.add_argument(
        "--skip-ml",
        action="store_true",
        help="Skip ML models (statistical only)",
    )
    parser.add_argument(
        "--autoencoder-model",
        type=str,
        default="DK_B2B_mh4",
        help="Model name to load Autoencoder from (default: DK_B2B_mh4)",
    )
    parser.add_argument(
        "--iforest-model",
        type=str,
        default="DK_B2B_mh5",
        help="Model name to load Isolation Forest from (default: DK_B2B_mh5)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Skip displaying stats on terminal, just save results to CSV",
    )
    args = parser.parse_args()
    
    load_dotenv()
    
    print("=" * 70)
    print("Detector Combination Analysis")
    print("=" * 70)
    print(f"Data path: {args.data_path}")
    print(f"File suffix(es): {args.file_suffix}")
    print(f"Granularity: {args.granularity}")
    print(f"Output: {args.output_dir}")
    print(f"Skip ML: {args.skip_ml}")
    if args.model_filter:
        print(f"Model filter: {args.model_filter}")
    print(f"Autoencoder model: {args.autoencoder_model}")
    print(f"Isolation Forest model: {args.iforest_model}")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize persistence
    persistence = None
    if not args.skip_ml:
        try:
            persistence = ModelPersistence()
            print(f"ML models from: {persistence.models_root_description}")
        except Exception as e:
            logger.warning(f"Could not initialize persistence: {e}")
    
    # Run analysis on each suffix, tracking results by suffix
    all_results: list[AnalysisRunResult] = []
    results_by_suffix: dict[str, list[AnalysisRunResult]] = {}
    
    for suffix in args.file_suffix:
        print(f"\n{'='*70}")
        print(f"Processing suffix: {suffix}")
        print(f"{'='*70}")
        
        results_by_suffix[suffix] = []
        
        # Find test files for this suffix
        files = find_parquet_files(args.data_path, args.granularity, suffix)
        
        if args.model_filter:
            files = [f for f in files if args.model_filter in extract_model_name(f)]
        
        if not files:
            print(f"No test files found for suffix '{suffix}'!")
            continue
        
        print(f"Found {len(files)} test files for suffix '{suffix}'")
        
        # Run analysis on each file
        for filepath in files:
            result = run_analysis_on_file(
                filepath=filepath,
                persistence=persistence,
                file_suffix=suffix,
                dataset_granularity=args.granularity,
                injection_rate=args.injection_rate,
                skip_ml=args.skip_ml,
                autoencoder_model=args.autoencoder_model,
                iforest_model=args.iforest_model,
            )
            if result is not None:
                all_results.append(result)
                results_by_suffix[suffix].append(result)
    
    if not all_results:
        print("\nNo results to analyze!")
        return
    
    run_id = create_run_id("detector_combinations")
    run_root = Path(args.output_dir) / run_id
    split_artifacts = build_canonical_run_artifacts(results_by_suffix, run_id)

    # Aggregate results from canonical tables
    print(f"\n{'='*60}")
    print("Aggregating results from canonical prediction tables...")
    print(f"{'='*60}")

    predictions, labels, obs_counts = aggregate_from_canonical_tables(split_artifacts)
    
    print(f"Total samples: {len(labels):,}")
    print(f"Total anomalies: {np.sum(labels):,}")
    print(f"Detectors: {list(predictions.keys())}")
    
    # Run analysis
    print("\n## Computing overlap matrix...")
    overlap_df = compute_overlap_matrix(predictions, labels)
    
    print("\n## Finding optimal ordering...")
    optimal_order = find_optimal_ordering(predictions, labels)
    
    print("\n## Analyzing cascade for optimal order...")
    optimal_order_names = [name for name, _, _ in optimal_order]
    cascade = analyze_cascade(predictions, labels, optimal_order_names)
    
    print("\n## Analyzing by history bucket...")
    history_df = analyze_by_history_bucket(predictions, labels, obs_counts)
    
    analysis_artifacts = {
        "overlap_matrix.csv": overlap_df,
        "cascade_analysis.md": "\n".join(
            [
                "# Cascade Effectiveness Analysis",
                "",
                f"Layer Order: {' -> '.join(cascade.layer_order)}",
                "",
                f"Final Performance: P={cascade.final_precision:.1%}, R={cascade.final_recall:.1%}, F1={cascade.final_f1:.1%}",
            ]
        )
        + "\n",
        "cascade_analysis.json": cascade_to_dict(cascade),
        "optimal_order.json": {
            "recommended_order": [item[0] for item in optimal_order],
            "details": [
                {
                    "detector": name,
                    "cumulative_f1": f1,
                    "marginal_gain": gain,
                }
                for name, f1, gain in optimal_order
            ],
            "analysis_timestamp": datetime.now().isoformat(),
        },
        "history_buckets.csv": history_df,
    }

    run_metadata = {
        "schema_version": "phase2.v1",
        "experiment_family": "detector_combinations",
        "run_id": run_id,
        "source_dataset_paths": sorted(
            {
                filepath
                for suffix in args.file_suffix
                for filepath in find_parquet_files(args.data_path, args.granularity, suffix)
                if not args.model_filter or args.model_filter in extract_model_name(filepath)
            }
        ),
        "dataset_names": sorted({result.model_name for result in all_results}),
        "dataset_granularity": args.granularity,
        "dataset_splits": sorted({result.dataset_split for result in all_results}),
        "random_seeds": {"injection_seed": 42},
        "injection_config": {
            "injection_rate": args.injection_rate,
            "file_suffixes": args.file_suffix,
        },
        "detector_identifiers": sorted(predictions.keys()),
        "config_values": {
            "skip_ml": args.skip_ml,
            "autoencoder_model": args.autoencoder_model,
            "iforest_model": args.iforest_model,
            "model_filter": args.model_filter,
        },
        "generated_at": datetime.now().astimezone().isoformat(),
        "git_commit": resolve_git_commit(Path(_project_root)),
    }
    write_evaluation_run(
        run_root=run_root,
        run_metadata=run_metadata,
        split_artifacts=split_artifacts,
        analysis_artifacts=analysis_artifacts,
    )
    print(f"\nCanonical results written to {run_root}")

    # Print summary
    if not args.quiet:
        print_analysis_summary(overlap_df, cascade, optimal_order, history_df, results_by_suffix)

    print("\nDone!")


if __name__ == "__main__":
    main()
