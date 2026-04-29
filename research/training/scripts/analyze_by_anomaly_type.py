#!/usr/bin/env python3
"""Analyze detection rates by anomaly type across all detectors.

Usage:
    python scripts/analyze_by_anomaly_type.py
    
    # Specific parquet file
    python scripts/analyze_by_anomaly_type.py --input results/detector_combinations/20260131_172738_combined_results.parquet
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def find_latest_parquet(directory: Path) -> Path:
    """Find the most recent combined_results parquet file."""
    parquet_files = sorted(directory.glob("*_combined_results.parquet"), reverse=True)
    if not parquet_files:
        raise FileNotFoundError(f"No combined_results.parquet files found in {directory}")
    return parquet_files[0]


def analyze_by_anomaly_type(df: pd.DataFrame, test_scenario: str = None) -> pd.DataFrame:
    """Analyze detection rates by anomaly type for each detector.
    
    Args:
        df: Combined predictions DataFrame
        test_scenario: Optional filter for specific test scenario
        
    Returns:
        Summary DataFrame with detection rates per anomaly type per detector
    """
    # Filter by test scenario if specified
    if test_scenario:
        df = df[df["test_scenario"] == test_scenario].copy()
    
    # Get unique detectors
    detectors = sorted(df["detector_combo"].unique())
    
    # Filter to only rows with actual anomalies (label=1)
    anomalies_df = df[df["label"] == 1].copy()
    
    # Get unique anomaly types
    anomaly_types = anomalies_df["anomaly_type"].dropna().unique()
    
    results = []
    
    for anomaly_type in sorted(anomaly_types):
        row = {"anomaly_type": anomaly_type}
        
        # Get all rows for this anomaly type (across all detectors, but count unique instances)
        type_df = anomalies_df[anomalies_df["anomaly_type"] == anomaly_type]
        
        # Count unique anomaly instances by competitor_product_id
        # Use first detector's data to count unique products
        first_detector = detectors[0]
        first_detector_df = type_df[type_df["detector_combo"] == first_detector]
        unique_count = first_detector_df["competitor_product_id"].nunique()
        row["count"] = unique_count
        
        # Calculate detection rate for each detector
        for detector in detectors:
            detector_df = type_df[type_df["detector_combo"] == detector]
            # Group by product to handle any duplicates
            product_detected = detector_df.groupby("competitor_product_id")["is_anomaly"].max()
            detected = product_detected.sum()
            total = len(product_detected)
            rate = (detected / total * 100) if total > 0 else 0
            row[detector] = rate
        
        # Calculate overall detected (by ANY detector)
        # Group by competitor_product_id to find if at least one detector caught it
        detected_by_any = type_df.groupby("competitor_product_id")["is_anomaly"].max().sum()
        row["detected_any"] = detected_by_any
        row["detected_any_pct"] = (detected_by_any / unique_count * 100) if unique_count > 0 else 0
        
        results.append(row)
    
    # Add totals row
    total_row = {"anomaly_type": "TOTAL"}
    first_detector = detectors[0]
    first_detector_df = anomalies_df[anomalies_df["detector_combo"] == first_detector]
    total_anomalies = first_detector_df["competitor_product_id"].nunique()
    total_row["count"] = total_anomalies
    
    for detector in detectors:
        detector_df = anomalies_df[anomalies_df["detector_combo"] == detector]
        # Group by product to handle any duplicates
        product_detected = detector_df.groupby("competitor_product_id")["is_anomaly"].max()
        detected = product_detected.sum()
        total = len(product_detected)
        rate = (detected / total * 100) if total > 0 else 0
        total_row[detector] = rate
    
    detected_by_any = anomalies_df.groupby("competitor_product_id")["is_anomaly"].max().sum()
    total_row["detected_any"] = detected_by_any
    total_row["detected_any_pct"] = (detected_by_any / total_anomalies * 100) if total_anomalies > 0 else 0
    
    results.append(total_row)
    
    return pd.DataFrame(results)


def print_results(results: pd.DataFrame, title: str):
    """Print formatted results table."""
    print("=" * 120)
    print(title)
    print("=" * 120)
    
    # Get detector columns for formatting
    detector_cols = [c for c in results.columns if c.startswith("Sanity")]
    
    # Print header
    header = f"{'Anomaly Type':<20} {'Count':>8} {'Detected':>8} {'Any%':>6}"
    for det in detector_cols:
        short_name = det.replace("Sanity_", "")
        header += f" {short_name:>10}"
    print(header)
    print("-" * 120)
    
    # Print rows
    for _, row in results.iterrows():
        line = f"{row['anomaly_type']:<20} {int(row['count']):>8} {int(row['detected_any']):>8} {row['detected_any_pct']:>5.1f}%"
        for det in detector_cols:
            line += f" {row[det]:>9.1f}%"
        print(line)
    
    print("=" * 120)


def main():
    parser = argparse.ArgumentParser(description="Analyze detection by anomaly type")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to combined_results.parquet file (defaults to latest)",
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
    print(f"Loaded {len(df):,} rows\n")
    
    # Get unique test scenarios
    test_scenarios = sorted(df["test_scenario"].unique())
    
    all_results = {}
    
    # Run analysis for each test scenario
    for scenario in test_scenarios:
        scenario_label = scenario.replace("test_", "").replace("_", " ").upper()
        results = analyze_by_anomaly_type(df, test_scenario=scenario)
        results["test_scenario"] = scenario
        all_results[scenario] = results
        print_results(results, f"DETECTION RATES BY ANOMALY TYPE - {scenario_label}")
        print()
    
    # Also run combined analysis
    combined_results = analyze_by_anomaly_type(df)
    combined_results["test_scenario"] = "all"
    all_results["all"] = combined_results
    print_results(combined_results, "DETECTION RATES BY ANOMALY TYPE - COMBINED (ALL TEST SCENARIOS)")
    
    # Save all results to CSV
    all_df = pd.concat(all_results.values(), ignore_index=True)
    output_file = input_file.parent / "anomaly_type_detection_rates.csv"
    all_df.to_csv(output_file, index=False)
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
