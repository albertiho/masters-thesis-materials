#!/usr/bin/env python3
"""Analyze optimal detector cascade order per anomaly type.

Calculates conditional detection probabilities:
- P(Detector B catches | Detector A missed)

This helps determine the optimal cascade order for each anomaly type.

Usage:
    python scripts/analyze_cascade_order.py
    
    # Specific parquet file
    python scripts/analyze_cascade_order.py --input results/detector_combinations/20260131_183834_combined_results.parquet
"""

import argparse
import sys
from pathlib import Path
from itertools import permutations

import pandas as pd
import numpy as np


def find_latest_parquet(directory: Path) -> Path:
    """Find the most recent combined_results parquet file."""
    parquet_files = sorted(directory.glob("*_combined_results.parquet"), reverse=True)
    if not parquet_files:
        raise FileNotFoundError(f"No combined_results.parquet files found in {directory}")
    return parquet_files[0]


def get_detection_matrix(df: pd.DataFrame, anomaly_type: str = None) -> pd.DataFrame:
    """Create detection matrix: rows=products, columns=detectors, values=is_anomaly.
    
    Args:
        df: DataFrame with anomalies (label=1)
        anomaly_type: Optional filter for specific anomaly type
        
    Returns:
        Pivot table with products as rows, detectors as columns
    """
    if anomaly_type:
        df = df[df["anomaly_type"] == anomaly_type]
    
    # Pivot: one row per product, one column per detector
    matrix = df.pivot_table(
        index="competitor_product_id",
        columns="detector_combo",
        values="is_anomaly",
        aggfunc="max"
    )
    return matrix


def calculate_conditional_probs(matrix: pd.DataFrame) -> pd.DataFrame:
    """Calculate P(B catches | A missed) for all detector pairs.
    
    Args:
        matrix: Detection matrix (products x detectors)
        
    Returns:
        DataFrame with conditional probabilities
    """
    detectors = list(matrix.columns)
    results = []
    
    for det_a in detectors:
        for det_b in detectors:
            if det_a == det_b:
                continue
            
            # Cases where A missed (is_anomaly=0)
            missed_by_a = matrix[matrix[det_a] == 0]
            n_missed = len(missed_by_a)
            
            if n_missed == 0:
                # A catches everything
                prob = np.nan
            else:
                # Of those A missed, how many did B catch?
                caught_by_b = (missed_by_a[det_b] == 1).sum()
                prob = caught_by_b / n_missed
            
            results.append({
                "missed_by": det_a,
                "caught_by": det_b,
                "n_missed_by_first": n_missed,
                "n_caught_by_second": caught_by_b if n_missed > 0 else 0,
                "conditional_prob": prob,
            })
    
    return pd.DataFrame(results)


def calculate_cascade_recall(matrix: pd.DataFrame, order: list) -> dict:
    """Calculate cumulative recall for a specific detector order.
    
    Args:
        matrix: Detection matrix (products x detectors)
        order: List of detector names in cascade order
        
    Returns:
        Dict with recall at each stage and marginal contribution
    """
    n_total = len(matrix)
    caught_so_far = pd.Series(False, index=matrix.index)
    
    results = {"order": order, "stages": []}
    
    for i, detector in enumerate(order):
        # This detector catches these
        catches = matrix[detector] == 1
        # New catches (not already caught)
        new_catches = catches & ~caught_so_far
        
        caught_so_far = caught_so_far | catches
        cumulative_recall = caught_so_far.sum() / n_total
        marginal = new_catches.sum() / n_total
        
        results["stages"].append({
            "detector": detector,
            "position": i + 1,
            "cumulative_recall": cumulative_recall,
            "marginal_contribution": marginal,
            "new_catches": new_catches.sum(),
        })
    
    results["final_recall"] = caught_so_far.sum() / n_total
    results["total_caught"] = caught_so_far.sum()
    results["total_anomalies"] = n_total
    
    return results


def find_optimal_order(matrix: pd.DataFrame, max_detectors: int = None) -> dict:
    """Find the optimal detector order by greedy selection.
    
    At each step, pick the detector that catches the most remaining anomalies.
    
    Args:
        matrix: Detection matrix (products x detectors)
        max_detectors: Max number of detectors to include (None = all)
        
    Returns:
        Dict with optimal order and recall at each stage
    """
    detectors = list(matrix.columns)
    n_total = len(matrix)
    
    if max_detectors is None:
        max_detectors = len(detectors)
    
    remaining_detectors = set(detectors)
    caught_so_far = pd.Series(False, index=matrix.index)
    
    order = []
    stages = []
    
    for i in range(min(max_detectors, len(detectors))):
        best_detector = None
        best_new_catches = -1
        
        for det in remaining_detectors:
            catches = matrix[det] == 1
            new_catches = (catches & ~caught_so_far).sum()
            
            if new_catches > best_new_catches:
                best_new_catches = new_catches
                best_detector = det
        
        if best_detector is None or best_new_catches == 0:
            break
        
        # Add this detector
        order.append(best_detector)
        remaining_detectors.remove(best_detector)
        
        catches = matrix[best_detector] == 1
        caught_so_far = caught_so_far | catches
        
        stages.append({
            "detector": best_detector,
            "position": i + 1,
            "cumulative_recall": caught_so_far.sum() / n_total,
            "marginal_contribution": best_new_catches / n_total,
            "new_catches": best_new_catches,
        })
    
    return {
        "order": order,
        "stages": stages,
        "final_recall": caught_so_far.sum() / n_total,
        "total_caught": caught_so_far.sum(),
        "total_anomalies": n_total,
    }


def print_conditional_probs(cond_probs: pd.DataFrame, title: str):
    """Print conditional probability matrix."""
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print("\nP(row catches | column missed)")
    print("Read as: 'If [column] missed it, [row] catches it with probability X'\n")
    
    # Pivot for display
    pivot = cond_probs.pivot(
        index="caught_by",
        columns="missed_by",
        values="conditional_prob"
    )
    
    # Format as percentages
    formatted = pivot.apply(lambda x: x.map(lambda v: f"{v*100:.1f}%" if pd.notna(v) else "-"))
    print(formatted.to_string())


def print_cascade_result(result: dict, title: str):
    """Print cascade analysis result."""
    print(f"\n{title}")
    print("-" * 80)
    print(f"Order: {' -> '.join(result['order'])}")
    print(f"Total anomalies: {result['total_anomalies']}, Caught: {result['total_caught']} ({result['final_recall']*100:.1f}%)")
    print("\nStage-by-stage:")
    for stage in result["stages"]:
        det = stage["detector"].replace("Sanity_", "")
        print(f"  {stage['position']}. {det:12s}: +{stage['new_catches']:4d} ({stage['marginal_contribution']*100:5.1f}%) -> cumulative {stage['cumulative_recall']*100:5.1f}%")


def analyze_anomaly_type(df: pd.DataFrame, anomaly_type: str, test_scenario: str = None):
    """Full analysis for one anomaly type."""
    # Filter data
    filtered = df.copy()
    if test_scenario:
        filtered = filtered[filtered["test_scenario"] == test_scenario]
    
    matrix = get_detection_matrix(filtered, anomaly_type)
    
    if len(matrix) == 0:
        print(f"No data for {anomaly_type}")
        return None
    
    # Calculate conditional probabilities
    cond_probs = calculate_conditional_probs(matrix)
    
    # Find optimal order
    optimal = find_optimal_order(matrix)
    
    return {
        "anomaly_type": anomaly_type,
        "test_scenario": test_scenario or "all",
        "n_anomalies": len(matrix),
        "conditional_probs": cond_probs,
        "optimal_order": optimal,
        "detection_matrix": matrix,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze optimal cascade order per anomaly type")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to combined_results.parquet file (defaults to latest)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        choices=["test_new_prices", "test_new_products"],
        help="Filter to specific test scenario",
    )
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    
    if args.input:
        input_file = args.input if args.input.is_absolute() else project_root / args.input
    else:
        input_file = find_latest_parquet(project_root / "results" / "detector_combinations")
    
    if not input_file.exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)
    
    print(f"Loading {input_file.name}...")
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df):,} rows")
    
    # Filter to actual anomalies
    anomalies_df = df[df["label"] == 1].copy()
    print(f"Anomaly rows: {len(anomalies_df):,}")
    
    # Get anomaly types
    anomaly_types = sorted(anomalies_df["anomaly_type"].dropna().unique())
    print(f"Anomaly types: {anomaly_types}")
    
    # Analyze each anomaly type
    all_results = []
    
    for anomaly_type in anomaly_types:
        result = analyze_anomaly_type(anomalies_df, anomaly_type, args.scenario)
        if result:
            all_results.append(result)
            
            print("\n" + "=" * 100)
            print(f"ANOMALY TYPE: {anomaly_type.upper()}")
            print(f"Test scenario: {result['test_scenario']}, Anomalies: {result['n_anomalies']:,}")
            print("=" * 100)
            
            # Print conditional probabilities
            print_conditional_probs(result["conditional_probs"], "Conditional Detection Probabilities")
            
            # Print optimal order
            print_cascade_result(result["optimal_order"], "OPTIMAL CASCADE ORDER (greedy)")
    
    # Summary: optimal orders per anomaly type
    print("\n" + "=" * 100)
    print("SUMMARY: OPTIMAL CASCADE ORDER PER ANOMALY TYPE")
    print("=" * 100)
    
    summary_rows = []
    for result in all_results:
        opt = result["optimal_order"]
        order_str = " -> ".join([d.replace("Sanity_", "") for d in opt["order"]])
        
        # Get recall after first 2 detectors
        recall_2 = opt["stages"][1]["cumulative_recall"] if len(opt["stages"]) > 1 else opt["stages"][0]["cumulative_recall"]
        
        print(f"\n{result['anomaly_type']:20s}: {order_str}")
        print(f"  Final recall: {opt['final_recall']*100:.1f}%, After 2 detectors: {recall_2*100:.1f}%")
        
        summary_rows.append({
            "anomaly_type": result["anomaly_type"],
            "optimal_order": " -> ".join(opt["order"]),
            "first_detector": opt["order"][0] if opt["order"] else None,
            "second_detector": opt["order"][1] if len(opt["order"]) > 1 else None,
            "recall_after_1": opt["stages"][0]["cumulative_recall"] if opt["stages"] else 0,
            "recall_after_2": recall_2,
            "final_recall": opt["final_recall"],
            "n_anomalies": result["n_anomalies"],
        })
    
    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    output_file = input_file.parent / "cascade_order_by_anomaly_type.csv"
    summary_df.to_csv(output_file, index=False)
    print(f"\n\nSaved summary to: {output_file}")
    
    # Overall optimal order (all anomaly types combined)
    print("\n" + "=" * 100)
    print("OVERALL OPTIMAL ORDER (all anomaly types)")
    print("=" * 100)
    
    overall_matrix = get_detection_matrix(anomalies_df)
    overall_optimal = find_optimal_order(overall_matrix)
    print_cascade_result(overall_optimal, "Combined optimal order")


if __name__ == "__main__":
    main()
