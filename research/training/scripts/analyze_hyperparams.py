#!/usr/bin/env python3
"""Analyze training data to guide hyperparameter tuning for split_training_data.py.

This script answers key questions to help tune:
- --last-n: How many observations per product go to test_new_prices
- --min-history: Minimum observations required to include a product

Usage:
    # Analyze extracted training data
    python scripts/analyze_hyperparams.py
    
    # Analyze specific country/segment
    python scripts/analyze_hyperparams.py --file data/training/by_country_segment/DK_B2C_2026-01-18.parquet
    
    # Test different min-history thresholds
    python scripts/analyze_hyperparams.py --test-min-history 3,4,5,6,7

Output:
    Diagnostic results for each question (L1-L5, H1-H6) with recommendations.
"""

import argparse
import logging
import sys
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install pandas numpy")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_training_data(data_dir: str, file_path: str | None = None) -> pd.DataFrame:
    """Load training data from Parquet files.
    
    Args:
        data_dir: Base data directory
        file_path: Optional specific file path
    
    Returns:
        Combined DataFrame
    """
    if file_path:
        logger.info(f"Loading: {file_path}")
        return pd.read_parquet(file_path)
    
    # Load all by_country_segment files (to avoid double-counting)
    data_path = Path(data_dir) / "by_country_segment"
    files = list(data_path.glob("*.parquet"))
    
    # Filter out split files
    files = [f for f in files if "_train" not in f.name and "_test" not in f.name]
    
    if not files:
        raise FileNotFoundError(f"No Parquet files found in {data_path}")
    
    logger.info(f"Loading {len(files)} files from {data_path}")
    
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        df["_source_file"] = f.name
        dfs.append(df)
        logger.info(f"  {f.name}: {len(df):,} rows")
    
    return pd.concat(dfs, ignore_index=True)


def analyze_last_n(df: pd.DataFrame, current_last_n: int = 2, current_min_history: int = 5) -> dict:
    """Analyze --last-n parameter (L1-L5 questions).
    
    Args:
        df: Training data
        current_last_n: Current last-n setting
        current_min_history: Current min-history setting
    
    Returns:
        Dict with analysis results
    """
    print("\n" + "=" * 70)
    print("ANALYSIS: --last-n parameter (currently {})".format(current_last_n))
    print("Core question: Is {} observations per product enough for meaningful test evaluation?".format(current_last_n))
    print("=" * 70)
    
    # Get time column
    time_col = "first_seen_at" if "first_seen_at" in df.columns else "scraped_at"
    
    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Observations per product
    obs_per_product = df.groupby("product_id").size()
    
    # L1: Median observations per product
    median_obs = obs_per_product.median()
    mean_obs = obs_per_product.mean()
    p25, p75 = obs_per_product.quantile([0.25, 0.75]).values
    
    print(f"\nL1: What's the median observations per product?")
    print(f"    Median: {median_obs:.0f}")
    print(f"    Mean: {mean_obs:.1f}")
    print(f"    P25/P75: {p25:.0f} / {p75:.0f}")
    print(f"    => last-{current_last_n} is {(current_last_n / median_obs * 100):.1f}% of median product's data")
    
    # L2: Distribution shape
    print(f"\nL2: What's the distribution shape?")
    print(f"    Min: {obs_per_product.min()}")
    print(f"    Max: {obs_per_product.max()}")
    print(f"    Std: {obs_per_product.std():.1f}")
    skewness = ((obs_per_product - mean_obs) ** 3).mean() / (obs_per_product.std() ** 3)
    print(f"    Skewness: {skewness:.2f} ({'right-skewed' if skewness > 0 else 'left-skewed'})")
    
    # Distribution buckets
    buckets = [0, 5, 10, 20, 50, 100, float("inf")]
    bucket_labels = ["1-4", "5-9", "10-19", "20-49", "50-99", "100+"]
    bucket_counts = pd.cut(obs_per_product, bins=buckets, labels=bucket_labels).value_counts().sort_index()
    print(f"\n    Observation count distribution:")
    for label, count in bucket_counts.items():
        pct = count / len(obs_per_product) * 100
        print(f"      {label:>6} obs: {count:>6,} products ({pct:>5.1f}%)")
    
    # L3: % of train data going to test_new_prices
    # Simulate the split
    valid_products = obs_per_product[obs_per_product >= current_min_history].index
    valid_df = df[df["product_id"].isin(valid_products)]
    
    # Calculate train vs test rows
    train_rows = 0
    test_rows = 0
    for pid, group in valid_df.groupby("product_id"):
        n = len(group)
        train_rows += max(0, n - current_last_n)
        test_rows += min(n, current_last_n)
    
    train_pct = train_rows / (train_rows + test_rows) * 100
    test_pct = test_rows / (train_rows + test_rows) * 100
    
    print(f"\nL3: What % of train data goes to new_prices test?")
    print(f"    Train rows: {train_rows:,} ({train_pct:.1f}%)")
    print(f"    Test (new_prices) rows: {test_rows:,} ({test_pct:.1f}%)")
    
    # L4: Time span of last-n observations
    df_sorted = valid_df.sort_values(["product_id", time_col])
    
    # Get time span for last-n observations per product
    def get_last_n_span(group):
        if len(group) < current_last_n + 1:
            return pd.NaT
        times = group[time_col].values
        # Time between last and (last-n)th observation
        return pd.Timestamp(times[-1]) - pd.Timestamp(times[-current_last_n])
    
    time_spans = df_sorted.groupby("product_id").apply(get_last_n_span, include_groups=False).dropna()
    
    if len(time_spans) > 0:
        median_span = time_spans.median()
        mean_span = time_spans.mean()
        print(f"\nL4: What's the time span of last {current_last_n} observations?")
        print(f"    Median: {median_span}")
        print(f"    Mean: {mean_span}")
        
        # Categorize
        spans_days = time_spans.dt.total_seconds() / 86400
        print(f"    Distribution:")
        print(f"      < 1 day:   {(spans_days < 1).mean() * 100:.1f}%")
        print(f"      1-7 days:  {((spans_days >= 1) & (spans_days < 7)).mean() * 100:.1f}%")
        print(f"      1-4 weeks: {((spans_days >= 7) & (spans_days < 28)).mean() * 100:.1f}%")
        print(f"      > 4 weeks: {(spans_days >= 28).mean() * 100:.1f}%")
    
    # L5: Price change frequency
    df_sorted = df.sort_values(["product_id", time_col])
    df_sorted["price_changed"] = df_sorted.groupby("product_id")["price"].diff() != 0
    
    # Exclude first observation per product (can't have a change)
    first_obs_mask = df_sorted.groupby("product_id").cumcount() == 0
    price_change_rate = df_sorted.loc[~first_obs_mask, "price_changed"].mean()
    
    print(f"\nL5: How often does price change between consecutive observations?")
    print(f"    Price change rate: {price_change_rate * 100:.1f}%")
    
    # Price changes in last-n observations
    def last_n_has_change(group):
        if len(group) < current_last_n + 1:
            return False
        prices = group["price"].values
        return len(set(prices[-current_last_n - 1:])) > 1
    
    products_with_changes = df_sorted.groupby("product_id").apply(last_n_has_change, include_groups=False).mean()
    print(f"    Products with price change in last {current_last_n + 1} obs: {products_with_changes * 100:.1f}%")
    
    # Recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS for --last-n")
    print("=" * 70)
    
    recommendations = []
    
    if test_pct > 30:
        recommendations.append(f"[!] L3: Test split is {test_pct:.0f}% (>30%) => Consider INCREASING last-n to get more train data")
    elif test_pct < 10:
        recommendations.append(f"[OK] L3: Test split is {test_pct:.0f}% (<10%) => Current last-n={current_last_n} is fine or DECREASE")
    else:
        recommendations.append(f"[OK] L3: Test split is {test_pct:.0f}% (10-30%) => last-n={current_last_n} is reasonable")
    
    if price_change_rate < 0.05:
        recommendations.append(f"[!] L5: Price change rate is {price_change_rate * 100:.1f}% (<5%) => Test set may not contain interesting anomalies")
    else:
        recommendations.append(f"[OK] L5: Price change rate is {price_change_rate * 100:.1f}% => Test set should have price variation")
    
    if median_obs < 10:
        recommendations.append(f"[!] L1: Median obs/product is {median_obs:.0f} (<10) => Keep last-n low (2-3)")
    elif median_obs > 30:
        recommendations.append(f"[OK] L1: Median obs/product is {median_obs:.0f} (>30) => Could increase last-n to 3-5")
    else:
        recommendations.append(f"[OK] L1: Median obs/product is {median_obs:.0f} => last-n=2 is reasonable")
    
    for rec in recommendations:
        print(f"  {rec}")
    
    return {
        "median_obs": median_obs,
        "mean_obs": mean_obs,
        "test_pct": test_pct,
        "price_change_rate": price_change_rate,
        "products_with_changes": products_with_changes,
    }


def analyze_min_history(df: pd.DataFrame, current_min_history: int = 5) -> dict:
    """Analyze --min-history parameter (H1-H6 questions).
    
    Args:
        df: Training data
        current_min_history: Current min-history setting
    
    Returns:
        Dict with analysis results
    """
    print("\n" + "=" * 70)
    print("ANALYSIS: --min-history parameter (currently {})".format(current_min_history))
    print("Core question: Are we excluding important products, or just noise?")
    print("=" * 70)
    
    # Get time column
    time_col = "first_seen_at" if "first_seen_at" in df.columns else "scraped_at"
    
    # Observations per product
    obs_per_product = df.groupby("product_id").size()
    total_products = len(obs_per_product)
    
    # H1: What % of products are excluded?
    excluded_mask = obs_per_product < current_min_history
    excluded_products = excluded_mask.sum()
    excluded_pct = excluded_products / total_products * 100
    
    print(f"\nH1: What % of products are excluded?")
    print(f"    Total products: {total_products:,}")
    print(f"    Excluded (< {current_min_history} obs): {excluded_products:,} ({excluded_pct:.1f}%)")
    print(f"    Included: {total_products - excluded_products:,} ({100 - excluded_pct:.1f}%)")
    
    # H2: Distribution of excluded products by obs count
    excluded_counts = obs_per_product[excluded_mask].value_counts().sort_index()
    
    print(f"\nH2: Distribution of excluded products by observation count:")
    for obs_count, product_count in excluded_counts.items():
        pct = product_count / total_products * 100
        print(f"    {obs_count} obs: {product_count:>6,} products ({pct:>5.1f}% of total)")
    
    # H3: Are excluded products newer or just low-volume?
    if time_col in df.columns:
        # Get first observation time per product
        first_seen = df.groupby("product_id")[time_col].min()
        
        excluded_first_seen = first_seen[excluded_mask]
        included_first_seen = first_seen[~excluded_mask]
        
        if len(excluded_first_seen) > 0 and len(included_first_seen) > 0:
            print(f"\nH3: Are excluded products newer or just low-volume?")
            print(f"    Excluded products:")
            print(f"      Median first_seen: {excluded_first_seen.median()}")
            print(f"      Min: {excluded_first_seen.min()}")
            print(f"      Max: {excluded_first_seen.max()}")
            print(f"    Included products:")
            print(f"      Median first_seen: {included_first_seen.median()}")
            print(f"      Min: {included_first_seen.min()}")
            print(f"      Max: {included_first_seen.max()}")
            
            # Is excluded median after included median?
            if excluded_first_seen.median() > included_first_seen.median():
                print(f"    => Excluded products tend to be NEWER (expected)")
            else:
                print(f"    => Excluded products are NOT newer => may indicate data issue")
    
    # H4: Are excluded products from specific competitors?
    if "competitor_id" in df.columns:
        products_per_competitor = df.drop_duplicates("product_id").groupby("competitor_id")["product_id"].count()
        
        excluded_product_ids = obs_per_product[excluded_mask].index
        excluded_competitor_counts = df[df["product_id"].isin(excluded_product_ids)].drop_duplicates("product_id").groupby("competitor_id")["product_id"].count()
        
        competitor_exclusion_rate = (excluded_competitor_counts / products_per_competitor * 100).sort_values(ascending=False)
        
        print(f"\nH4: Are excluded products from specific competitors?")
        print(f"    Top 10 competitors by exclusion rate:")
        for comp_id, rate in competitor_exclusion_rate.head(10).items():
            total = products_per_competitor.get(comp_id, 0)
            excluded = excluded_competitor_counts.get(comp_id, 0)
            print(f"      {comp_id}: {rate:.1f}% excluded ({excluded:,} of {total:,})")
        
        # Check if any competitor has >60% exclusion
        high_exclusion = competitor_exclusion_rate[competitor_exclusion_rate > 60]
        if len(high_exclusion) > 0:
            print(f"    [!] {len(high_exclusion)} competitors have >60% exclusion rate => investigate data quality")
    
    # H5: Can a model learn anything from 3-4 observations?
    print(f"\nH5: Can a model learn anything from 3-4 observations?")
    print(f"    With min-history={current_min_history} and last-n=2:")
    print(f"      Minimum training points per product: {current_min_history - 2}")
    print(f"    Theoretical minimum: 2 training points + 1 test point")
    
    if current_min_history < 4:
        print(f"    [!] min-history={current_min_history} may not provide enough training data per product")
    else:
        print(f"    [OK] min-history={current_min_history} provides {current_min_history - 2}+ training points")
    
    # H6: Price variance of excluded products
    price_stats = df.groupby("product_id")["price"].agg(["std", "mean"])
    price_stats["cv"] = price_stats["std"] / price_stats["mean"]  # Coefficient of variation
    
    excluded_cv = price_stats.loc[excluded_mask, "cv"].dropna()
    included_cv = price_stats.loc[~excluded_mask, "cv"].dropna()
    
    print(f"\nH6: Price variance of excluded products vs included:")
    if len(excluded_cv) > 0:
        print(f"    Excluded products:")
        print(f"      Median CV (std/mean): {excluded_cv.median():.3f}")
        print(f"      Mean CV: {excluded_cv.mean():.3f}")
    print(f"    Included products:")
    print(f"      Median CV (std/mean): {included_cv.median():.3f}")
    print(f"      Mean CV: {included_cv.mean():.3f}")
    
    if len(excluded_cv) > 0 and excluded_cv.median() > included_cv.median() * 1.5:
        print(f"    [!] Excluded products have HIGHER price variance => might be interesting anomaly candidates")
    else:
        print(f"    [OK] Excluded products have similar or lower variance => exclusion is fine")
    
    # Test different min-history thresholds
    print(f"\n{'-'*70}")
    print("Sensitivity analysis: Impact of different --min-history values")
    print("-" * 70)
    
    for threshold in [3, 4, 5, 6, 7, 10]:
        excluded = (obs_per_product < threshold).sum()
        excluded_pct = excluded / total_products * 100
        included_rows = df[df["product_id"].isin(obs_per_product[obs_per_product >= threshold].index)]
        print(f"    min-history={threshold}: Excludes {excluded:>6,} products ({excluded_pct:>5.1f}%), keeps {len(included_rows):>8,} rows")
    
    # Recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS for --min-history")
    print("=" * 70)
    
    recommendations = []
    
    if excluded_pct > 40:
        recommendations.append(f"[!] H1: {excluded_pct:.0f}% excluded (>40%) => Consider LOWERING min-history to 4 or 3")
    elif excluded_pct < 20:
        recommendations.append(f"[OK] H1: {excluded_pct:.0f}% excluded (<20%) => Keep min-history={current_min_history}")
    else:
        recommendations.append(f"[OK] H1: {excluded_pct:.0f}% excluded (20-40%) => min-history={current_min_history} is reasonable")
    
    if current_min_history >= 5:
        recommendations.append(f"[OK] H5: min-history={current_min_history} provides {current_min_history - 2}+ training points per product")
    else:
        recommendations.append(f"[!] H5: min-history={current_min_history} provides only {current_min_history - 2} training points => may be too low")
    
    for rec in recommendations:
        print(f"  {rec}")
    
    return {
        "total_products": total_products,
        "excluded_products": excluded_products,
        "excluded_pct": excluded_pct,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze training data to guide hyperparameter tuning"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/training",
        help="Base data directory (default: data/training)",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Specific Parquet file to analyze (optional)",
    )
    parser.add_argument(
        "--last-n",
        type=int,
        default=2,
        help="Current last-n setting to analyze (default: 2)",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=5,
        help="Current min-history setting to analyze (default: 5)",
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("Hyperparameter Tuning Analysis")
    print("=" * 70)
    print(f"Data directory: {args.data_dir}")
    print(f"Current --last-n: {args.last_n}")
    print(f"Current --min-history: {args.min_history}")
    print("=" * 70)
    
    try:
        df = load_training_data(args.data_dir, args.file)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nTo extract training data, run:")
        print("  python scripts/extract_training_data.py --env prod")
        return
    
    print(f"\nLoaded {len(df):,} rows, {df['product_id'].nunique():,} unique products")
    
    # Run analyses
    last_n_results = analyze_last_n(df, args.last_n, args.min_history)
    min_history_results = analyze_min_history(df, args.min_history)
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    print("\nKey metrics:")
    print(f"  Median observations per product: {last_n_results['median_obs']:.0f}")
    print(f"  Price change rate: {last_n_results['price_change_rate'] * 100:.1f}%")
    print(f"  Products excluded: {min_history_results['excluded_pct']:.1f}%")
    print(f"  Test split (new_prices): {last_n_results['test_pct']:.1f}%")
    
    print("\nDecision matrix:")
    print("  +-------------------------+---------------------------------------+")
    print("  | If...                   | Then...                               |")
    print("  +-------------------------+---------------------------------------+")
    print("  | Test split >30%         | Increase --last-n (more train data)   |")
    print("  | Test split <10%         | Current --last-n is fine or decrease  |")
    print("  | Excluded >40%           | Lower --min-history to 4 or 3         |")
    print("  | Excluded <20%           | Keep --min-history as is              |")
    print("  | Price change <5%        | Test set may be weak regardless       |")
    print("  +-------------------------+---------------------------------------+")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
