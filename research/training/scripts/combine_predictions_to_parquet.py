#!/usr/bin/env python3
"""Combine prediction CSV files into a single Parquet file.

Reads all CSV files from the predictions folder and combines them into
a single Parquet file with metadata columns extracted from filenames.

Usage:
    python scripts/combine_predictions_to_parquet.py
    
    # Custom input/output paths
    python scripts/combine_predictions_to_parquet.py \
        --input-dir results/detector_combinations/predictions \
        --output-dir results/detector_combinations
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_filename(filename: str) -> dict:
    """Extract metadata from prediction filename.
    
    Expected format: NO_B2C_mh4_test_new_prices_Sanity_AE.csv
    
    Returns:
        dict with keys: model_filter, test_scenario, detector_combo
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    
    # Find where test scenario starts (test_new_prices or test_new_products)
    test_idx = None
    for i, part in enumerate(parts):
        if part == "test":
            test_idx = i
            break
    
    if test_idx is None:
        logger.warning(f"Could not parse filename: {filename}")
        return {
            "model_filter": stem,
            "test_scenario": "unknown",
            "detector_combo": "unknown",
        }
    
    model_filter = "_".join(parts[:test_idx])
    test_scenario = "_".join(parts[test_idx:test_idx + 3])  # test_new_prices or test_new_products
    detector_combo = "_".join(parts[test_idx + 3:])
    
    return {
        "model_filter": model_filter,
        "test_scenario": test_scenario,
        "detector_combo": detector_combo,
    }


def combine_predictions(input_dir: Path) -> pd.DataFrame:
    """Load and combine all CSV files from input directory.
    
    Args:
        input_dir: Path to directory containing prediction CSVs
        
    Returns:
        Combined DataFrame with metadata columns
    """
    csv_files = sorted(input_dir.glob("*.csv"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {input_dir}")
    
    logger.info(f"Found {len(csv_files)} CSV files to combine")
    
    dfs = []
    for csv_file in csv_files:
        logger.info(f"Reading {csv_file.name}...")
        df = pd.read_csv(csv_file)
        
        # Add metadata columns from filename
        metadata = parse_filename(csv_file.name)
        df["model_filter"] = metadata["model_filter"]
        df["test_scenario"] = metadata["test_scenario"]
        df["detector_combo"] = metadata["detector_combo"]
        df["source_file"] = csv_file.name
        
        dfs.append(df)
        logger.info(f"  -> {len(df):,} rows")
    
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined total: {len(combined):,} rows")
    
    return combined


def main():
    parser = argparse.ArgumentParser(description="Combine prediction CSVs to Parquet")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/detector_combinations/predictions"),
        help="Directory containing prediction CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/detector_combinations"),
        help="Directory to write output Parquet file",
    )
    args = parser.parse_args()
    
    # Resolve paths relative to project root if needed
    project_root = Path(__file__).parent.parent
    input_dir = args.input_dir if args.input_dir.is_absolute() else project_root / args.input_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine all CSVs
    combined_df = combine_predictions(input_dir)
    
    # Generate timestamped output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{timestamp}_combined_results.parquet"
    
    # Save to Parquet
    logger.info(f"Writing to {output_file}...")
    combined_df.to_parquet(output_file, index=False)
    
    # Report file size
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.info(f"Done! Output file: {output_file} ({file_size_mb:.2f} MB)")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total rows: {len(combined_df):,}")
    print(f"Columns: {list(combined_df.columns)}")
    print(f"\nBreakdown by test scenario:")
    print(combined_df.groupby("test_scenario").size().to_string())
    print(f"\nBreakdown by detector combo:")
    print(combined_df.groupby("detector_combo").size().to_string())
    print(f"\nOutput: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
