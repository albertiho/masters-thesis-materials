"""Synthetic Anomaly Injection for Evaluation.

Since we have no labeled anomalies, we evaluate detection methods by:
1. Assuming existing data is "mostly normal"
2. Injecting known synthetic anomalies
3. Measuring how well each method detects them

This module provides synthetic anomaly generators that mimic
real-world anomaly patterns observed in competitor data.

Anomaly Types:
    - Price spikes: Sudden large price increases
    - Price drops: Sudden large price decreases  
    - Price noise: Random noise added to prices
    - List price violations: Sale price > list price
    - Zero prices: Price set to zero
    - Extreme outliers: Prices far outside normal range
"""

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from src.ingestion.parser import ProductRecord

logger = logging.getLogger(__name__)


# Closed-vocabulary currency swap map for anonymized thesis datasets.
# We intentionally route to CURRENCY_5 so injected currency values do not
# overlap with the source COUNTRY_1 data.
CURRENCY_SWAP_TRANSFORMS: dict[str, tuple[str, float]] = {
    "CURRENCY_1": ("CURRENCY_5", 10.0),
    "CURRENCY_2": ("CURRENCY_5", 0.1),
    "CURRENCY_3": ("CURRENCY_5", 1.4),
    "CURRENCY_4": ("CURRENCY_5", 0.7),
    "CURRENCY_5": ("CURRENCY_1", 0.1),
}

MIN_REALISTIC_PRICE = 1.0
LIST_PRICE_COSCALE_TYPES = {
    "price_spike",
    "price_drop",
    "price_noise",
    "extreme_outlier",
    "decimal_shift",
    "currency_swap",
    "transient_spike",
    "persistent_change",
}


def _resolve_currency_swap(original_currency: str | None) -> tuple[str, float]:
    """Return a deterministic synthetic currency swap target and price factor."""
    if isinstance(original_currency, str):
        normalized = original_currency.strip().upper()
        if normalized in CURRENCY_SWAP_TRANSFORMS:
            return CURRENCY_SWAP_TRANSFORMS[normalized]
    return ("CURRENCY_5", 10.0)


class SyntheticAnomalyType(Enum):
    """Types of synthetic anomalies we can inject."""

    # Price anomalies
    PRICE_SPIKE = "price_spike"  # Large sudden increase
    PRICE_DROP = "price_drop"  # Large sudden decrease
    PRICE_NOISE = "price_noise"  # Random noise
    LIST_PRICE_VIOLATION = "list_price_violation"  # Sale > list
    ZERO_PRICE = "zero_price"  # Price = 0
    NEGATIVE_PRICE = "negative_price"  # Price < 0
    EXTREME_OUTLIER = "extreme_outlier"  # Far outside range

    # Scraper bug patterns (new for scope expansion)
    TITLE_COLLAPSE = "title_collapse"  # Title shortened + price changed
    DECIMAL_SHIFT = "decimal_shift"  # Price x100 or /100 (common bug)
    CURRENCY_SWAP = "currency_swap"  # Wrong currency (DKK vs EUR confusion)

    # Temporal patterns (for persistence testing)
    TRANSIENT_SPIKE = "transient_spike"  # One-time spike (should be filtered)
    PERSISTENT_CHANGE = "persistent_change"  # Lasting change (should pass)


@dataclass
class AnomalyInjectionConfig:
    """Configuration for synthetic anomaly injection.

    Attributes:
        injection_rate: Fraction of records to inject anomalies into.
        spike_magnitude: Multiplier for price spikes (e.g., 2.0 = double).
        drop_magnitude: Multiplier for price drops (e.g., 0.3 = 70% off).
        noise_std: Standard deviation for noise injection.
        outlier_std: Number of std devs for extreme outliers.
        random_seed: Random seed for reproducibility.
    """

    injection_rate: float = 0.05  # 5% of records get anomalies
    spike_magnitude: float = 2.0  # Double the price
    drop_magnitude: float = 0.3  # 70% discount
    noise_std: float = 0.2  # 20% noise
    outlier_std: float = 5.0  # 5 std devs
    title_collapse_ratio: float = 0.2  # Keep 20% of title
    random_seed: int = 42

    # Weights for each anomaly type (must sum to 1.0)
    type_weights: dict[SyntheticAnomalyType, float] = field(
        default_factory=lambda: {
            SyntheticAnomalyType.PRICE_SPIKE: 0.15,
            SyntheticAnomalyType.PRICE_DROP: 0.15,
            SyntheticAnomalyType.PRICE_NOISE: 0.10,
            SyntheticAnomalyType.LIST_PRICE_VIOLATION: 0.08,
            SyntheticAnomalyType.ZERO_PRICE: 0.07,
            SyntheticAnomalyType.NEGATIVE_PRICE: 0.05,
            SyntheticAnomalyType.EXTREME_OUTLIER: 0.10,
            # Scraper bug patterns
            SyntheticAnomalyType.TITLE_COLLAPSE: 0.10,
            SyntheticAnomalyType.DECIMAL_SHIFT: 0.10,
            SyntheticAnomalyType.CURRENCY_SWAP: 0.05,
            # Temporal patterns (for classifier evaluation)
            SyntheticAnomalyType.TRANSIENT_SPIKE: 0.03,
            SyntheticAnomalyType.PERSISTENT_CHANGE: 0.02,
        }
    )


@dataclass
class InjectedAnomaly:
    """Record of an injected synthetic anomaly.

    Used as ground truth for evaluation.

    Attributes:
        record_index: Index of the record in the dataset.
        competitor_product_id: Product identifier.
        competitor: Competitor identifier.
        anomaly_type: Type of anomaly injected.
        original_price: Original price before injection.
        injected_price: Price after injection.
        injection_params: Parameters used for injection.
    """

    record_index: int
    competitor_product_id: str
    competitor: str
    anomaly_type: SyntheticAnomalyType
    original_price: float
    injected_price: float
    injection_params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "record_index": self.record_index,
            "competitor_product_id": self.competitor_product_id,
            "competitor": self.competitor,
            "anomaly_type": self.anomaly_type.value,
            "original_price": self.original_price,
            "injected_price": self.injected_price,
            **self.injection_params,
        }


class SyntheticAnomalyInjector:
    """Inject synthetic anomalies into product records for evaluation.

    Usage:
        injector = SyntheticAnomalyInjector()

        # Inject anomalies
        modified_records, injected = injector.inject(records)

        # Run detection
        results = detector.detect_batch(modified_records)

        # Evaluate
        metrics = evaluate_detection(results, injected)
    """

    def __init__(self, config: AnomalyInjectionConfig | None = None):
        """Initialize the injector.

        Args:
            config: Configuration for injection. Uses defaults if None.
        """
        self.config = config or AnomalyInjectionConfig()
        self._rng = np.random.RandomState(self.config.random_seed)

    def inject(
        self,
        records: list[ProductRecord],
        compute_stats: bool = True,
    ) -> tuple[list[ProductRecord], list[InjectedAnomaly]]:
        """Inject synthetic anomalies into records.

        Args:
            records: List of original product records.
            compute_stats: Whether to compute price stats for outlier injection.

        Returns:
            Tuple of (modified records, list of injected anomalies).
        """
        if not records:
            return [], []

        # Compute price statistics for realistic injections
        prices = [r.price for r in records if r.price is not None and r.price > 0]
        if not prices:
            logger.warning("no_valid_prices_for_injection")
            return records.copy(), []

        price_mean = np.mean(prices)
        price_std = np.std(prices)
        price_min = np.min(prices)
        price_max = np.max(prices)

        logger.info(
            "injection_stats",
            extra={
                "total_records": len(records),
                "valid_prices": len(prices),
                "price_mean": price_mean,
                "price_std": price_std,
                "price_range": f"{price_min:.2f} - {price_max:.2f}",
            },
        )

        # Determine how many records to inject
        n_inject = max(1, int(len(records) * self.config.injection_rate))

        # Randomly select records to inject (only those with valid prices)
        valid_indices = [i for i, r in enumerate(records) if r.price is not None and r.price > 0]

        if len(valid_indices) < n_inject:
            n_inject = len(valid_indices)

        inject_indices = self._rng.choice(valid_indices, size=n_inject, replace=False)

        # Copy records (don't modify originals)
        modified_records = [self._copy_record(r) for r in records]
        injected_anomalies: list[InjectedAnomaly] = []

        # Inject anomalies
        for idx in inject_indices:
            record = modified_records[idx]
            original_price = record.price

            # Select anomaly type based on weights
            anomaly_type = self._select_anomaly_type()

            # Inject the anomaly
            injected_price, params = self._inject_anomaly(
                record=record,
                anomaly_type=anomaly_type,
                price_mean=price_mean,
                price_std=price_std,
            )

            # Check if we need to collapse the title
            collapse_title = anomaly_type == SyntheticAnomalyType.TITLE_COLLAPSE

            # Update record
            modified_records[idx] = self._update_record_price(
                record,
                injected_price,
                new_currency=params.get("new_currency"),
                collapse_title=collapse_title,
                collapse_ratio=self.config.title_collapse_ratio,
            )

            # Track injection
            injected_anomalies.append(
                InjectedAnomaly(
                    record_index=idx,
                    competitor_product_id=record.competitor_product_id,
                    competitor=record.competitor,
                    anomaly_type=anomaly_type,
                    original_price=original_price,
                    injected_price=injected_price,
                    injection_params=params,
                )
            )

        logger.info(
            "anomalies_injected",
            extra={
                "total_records": len(records),
                "anomalies_injected": len(injected_anomalies),
                "injection_rate": len(injected_anomalies) / len(records),
                "types_injected": {
                    t.value: sum(1 for a in injected_anomalies if a.anomaly_type == t)
                    for t in SyntheticAnomalyType
                },
            },
        )

        return modified_records, injected_anomalies

    def _select_anomaly_type(self) -> SyntheticAnomalyType:
        """Select an anomaly type based on configured weights."""
        types = list(self.config.type_weights.keys())
        weights = [self.config.type_weights[t] for t in types]
        return self._rng.choice(types, p=weights)

    def _inject_anomaly(
        self,
        record: ProductRecord,
        anomaly_type: SyntheticAnomalyType,
        price_mean: float,
        price_std: float,
    ) -> tuple[float, dict[str, Any]]:
        """Inject a specific type of anomaly.

        Returns:
            Tuple of (new price, injection parameters).
        """
        original_price = record.price
        params: dict[str, Any] = {"anomaly_type": anomaly_type.value}

        if anomaly_type == SyntheticAnomalyType.PRICE_SPIKE:
            # Multiply price by spike magnitude (with some randomness)
            multiplier = self.config.spike_magnitude * (1 + self._rng.uniform(-0.2, 0.2))
            new_price = original_price * multiplier
            params["multiplier"] = multiplier

        elif anomaly_type == SyntheticAnomalyType.PRICE_DROP:
            # Multiply price by drop magnitude
            multiplier = self.config.drop_magnitude * (1 + self._rng.uniform(-0.2, 0.2))
            new_price = original_price * multiplier
            params["multiplier"] = multiplier

        elif anomaly_type == SyntheticAnomalyType.PRICE_NOISE:
            # Add random noise
            noise = self._rng.normal(0, original_price * self.config.noise_std)
            new_price = original_price + noise
            params["noise"] = noise

        elif anomaly_type == SyntheticAnomalyType.LIST_PRICE_VIOLATION:
            # Make price higher than list price
            if record.list_price and record.list_price > 0:
                new_price = record.list_price * (1.1 + self._rng.uniform(0, 0.3))
            else:
                # No list price, create one and violate it
                new_price = original_price * 1.5
            params["list_price"] = record.list_price

        elif anomaly_type == SyntheticAnomalyType.ZERO_PRICE:
            new_price = 0.0

        elif anomaly_type == SyntheticAnomalyType.NEGATIVE_PRICE:
            new_price = -abs(original_price * self._rng.uniform(0.1, 0.5))

        elif anomaly_type == SyntheticAnomalyType.EXTREME_OUTLIER:
            # Go far outside normal range
            direction = self._rng.choice([-1, 1])
            deviation = self.config.outlier_std * price_std
            new_price = price_mean + direction * deviation
            new_price = max(0.01, new_price)  # Keep positive for most outliers
            params["direction"] = "above" if direction > 0 else "below"
            params["deviation_std"] = self.config.outlier_std

        elif anomaly_type == SyntheticAnomalyType.TITLE_COLLAPSE:
            # Title collapse pattern: truncate title AND change price
            # This is a strong indicator of scraper bug
            new_price = original_price * self._rng.uniform(0.5, 1.5)
            params["title_collapse"] = True
            params["title_collapse_ratio"] = self.config.title_collapse_ratio

        elif anomaly_type == SyntheticAnomalyType.DECIMAL_SHIFT:
            # Common bug: decimal point error (x100 or /100)
            shift_direction = self._rng.choice([100, 0.01])
            new_price = original_price * shift_direction
            params["decimal_shift"] = shift_direction
            params["shift_type"] = "x100" if shift_direction == 100 else "/100"

        elif anomaly_type == SyntheticAnomalyType.CURRENCY_SWAP:
            new_currency, swap_factor = _resolve_currency_swap(record.currency)
            new_price = original_price * swap_factor
            params["original_currency"] = record.currency
            params["new_currency"] = new_currency
            params["currency_swap_factor"] = swap_factor

        elif anomaly_type == SyntheticAnomalyType.TRANSIENT_SPIKE:
            # One-time spike (should be filtered by persistence)
            new_price = original_price * self._rng.uniform(1.5, 3.0)
            params["is_transient"] = True
            params["expected_persistence"] = 1  # Should revert next run

        elif anomaly_type == SyntheticAnomalyType.PERSISTENT_CHANGE:
            # Lasting price change (should be trusted after persistence)
            new_price = original_price * self._rng.uniform(0.8, 1.2)
            params["is_persistent"] = True
            params["expected_persistence"] = 3  # Should persist

        else:
            new_price = original_price

        params["original_price"] = original_price
        params["new_price"] = new_price
        params["change_pct"] = (
            (new_price - original_price) / original_price if original_price != 0 else 0
        )

        return new_price, params

    def _copy_record(self, record: ProductRecord) -> ProductRecord:
        """Create a copy of a ProductRecord."""
        return ProductRecord(
            competitor_product_id=record.competitor_product_id,
            competitor=record.competitor,
            price=record.price,
            currency=record.currency,
            scraped_at=record.scraped_at,
            scrape_run_id=record.scrape_run_id,
            country=record.country,
            channel=record.channel,
            source=record.source,
            product_name=record.product_name,
            brand=record.brand,
            availability_status=record.availability_status,
            product_url=record.product_url,
            mpn=record.mpn,
            ean=record.ean,
            list_price=record.list_price,
        )

    def _update_record_price(
        self,
        record: ProductRecord,
        new_price: float,
        new_currency: str | None = None,
        collapse_title: bool = False,
        collapse_ratio: float = 0.2,
    ) -> ProductRecord:
        """Create a new record with updated price and optionally collapsed title."""
        product_name = record.product_name

        # Collapse title if requested (for title_collapse anomaly type)
        if collapse_title and product_name:
            # Keep only first N% of title
            keep_len = max(5, int(len(product_name) * collapse_ratio))
            product_name = product_name[:keep_len]

        return ProductRecord(
            competitor_product_id=record.competitor_product_id,
            competitor=record.competitor,
            price=new_price,
            currency=new_currency if new_currency is not None else record.currency,
            scraped_at=record.scraped_at,
            scrape_run_id=record.scrape_run_id,
            country=record.country,
            channel=record.channel,
            source=record.source,
            product_name=product_name,
            brand=record.brand,
            availability_status=record.availability_status,
            product_url=record.product_url,
            mpn=record.mpn,
            ean=record.ean,
            list_price=record.list_price,
        )


# Production anomaly types used for validation testing
PRODUCTION_ANOMALY_TYPES = [
    SyntheticAnomalyType.PRICE_SPIKE,
    SyntheticAnomalyType.PRICE_DROP,
    SyntheticAnomalyType.ZERO_PRICE,
    SyntheticAnomalyType.NEGATIVE_PRICE,
    SyntheticAnomalyType.EXTREME_OUTLIER,
    SyntheticAnomalyType.DECIMAL_SHIFT,
    SyntheticAnomalyType.LIST_PRICE_VIOLATION,
]


def generate_all_anomaly_variants(
    base_row: pd.Series,
    anomaly_types: list[SyntheticAnomalyType] | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, list[dict]]:
    """Generate one variant per anomaly type for validation testing.

    Creates a deterministic test dataset where each anomaly type is represented
    exactly once. This is useful for validating that the detection pipeline
    correctly identifies all supported anomaly types.

    Args:
        base_row: A single row (pd.Series) with at minimum 'price'. Optional
            columns: 'list_price' (for LIST_PRICE_VIOLATION), and any other
            columns needed for feature extraction.
        anomaly_types: List of anomaly types to generate. Defaults to the 7
            production types (PRICE_SPIKE, PRICE_DROP, ZERO_PRICE, NEGATIVE_PRICE,
            EXTREME_OUTLIER, DECIMAL_SHIFT, LIST_PRICE_VIOLATION).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of:
            - DataFrame with original row (index 0) + one row per anomaly type
            - Boolean labels array (False for original, True for anomalies)
            - Injection details list (one dict per anomaly with type, prices, etc.)

    Example:
        >>> base = pd.Series({'price': 1000.0, 'list_price': 1200.0, 'product_id': 'TEST'})
        >>> df, labels, details = generate_all_anomaly_variants(base)
        >>> print(f"Generated {len(df)} rows: 1 original + {len(details)} anomalies")
        Generated 8 rows: 1 original + 7 anomalies
    """
    if anomaly_types is None:
        anomaly_types = PRODUCTION_ANOMALY_TYPES.copy()

    rng = np.random.default_rng(seed)

    # Extract base price and list_price
    original_price = float(base_row.get("price", 1000.0))
    list_price = base_row.get("list_price")
    if pd.notna(list_price):
        list_price = float(list_price)
    else:
        list_price = None

    # Use base price for mean/std (single row, so use reasonable defaults)
    price_mean = original_price
    price_std = original_price * 0.1  # 10% std dev for outlier calculation

    # Build rows: original + one per anomaly type
    rows = [base_row.to_dict()]  # Row 0: original
    labels = [False]  # Not an anomaly
    injection_details: list[dict] = []

    for i, anomaly_type in enumerate(anomaly_types):
        # Apply anomaly to get new price
        new_price, detail = _apply_anomaly_to_price(
            original_price=original_price,
            anomaly_type=anomaly_type,
            rng=rng,
            spike_range=(2.0, 5.0),
            drop_range=(0.1, 0.5),
            price_mean=price_mean,
            price_std=price_std,
            list_price=list_price,
            currency=base_row.get("currency"),
        )

        # Create modified row
        modified_row = base_row.to_dict()
        modified_row["price"] = new_price
        if anomaly_type == SyntheticAnomalyType.CURRENCY_SWAP and "new_currency" in detail:
            modified_row["currency"] = detail["new_currency"]

        rows.append(modified_row)
        labels.append(True)  # Is an anomaly

        # Track injection details
        detail["index"] = i + 1  # +1 because row 0 is original
        detail["anomaly_type"] = anomaly_type.value
        detail["original_price"] = original_price
        detail["new_price"] = new_price
        injection_details.append(detail)

    # Convert to DataFrame
    df = pd.DataFrame(rows)
    labels_array = np.array(labels, dtype=bool)

    logger.info(
        "generated_anomaly_variants",
        extra={
            "n_rows": len(df),
            "n_anomalies": len(injection_details),
            "anomaly_types": [d["anomaly_type"] for d in injection_details],
        },
    )

    return df, labels_array, injection_details


def inject_anomalies_to_dataframe(
    df: pd.DataFrame,
    injection_rate: float = 0.1,
    seed: int = 42,
    spike_range: tuple[float, float] = (2.0, 5.0),
    drop_range: tuple[float, float] = (0.1, 0.5),
    anomaly_types: list[SyntheticAnomalyType] | None = None,
    type_weights: dict[SyntheticAnomalyType, float] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, list[dict]]:
    """Inject price anomalies at the raw data level (DataFrame).

    This function modifies the price column directly so anomalies flow through
    the full feature extraction pipeline. Use this for production-like evaluation
    where you want features to be computed from anomalous prices.

    Args:
        df: DataFrame with at least 'price' column. Also handles 'list_price',
            'product_name' columns if present for relevant anomaly types.
        injection_rate: Fraction of records to inject anomalies into (0-1).
        seed: Random seed for reproducibility.
        spike_range: (min, max) multiplier for price spikes (e.g., 2x-5x).
        drop_range: (min, max) multiplier for price drops (e.g., 0.1x-0.5x).
        anomaly_types: List of anomaly types to use. If None, uses all price-related types.
        type_weights: Custom weights for anomaly types. If None, uses equal weights.

    Returns:
        Tuple of (modified DataFrame, boolean mask of injected rows, injection details).
            - df: Modified DataFrame with anomalies injected
            - mask: Boolean array where True = anomaly was injected at this index
            - details: List of dicts with injection details for each anomaly
    """
    if df.empty:
        return df.copy(), np.zeros(0, dtype=bool), []

    rng = np.random.default_rng(seed)

    # Default to 6 core anomaly types for production-like evaluation.
    # Excluded: PRICE_NOISE (legitimate market fluctuations), CURRENCY_SWAP (rare, caught by DECIMAL_SHIFT)
    if anomaly_types is None:
        anomaly_types = [
            SyntheticAnomalyType.PRICE_SPIKE,
            SyntheticAnomalyType.PRICE_DROP,
#            SyntheticAnomalyType.ZERO_PRICE,
#            SyntheticAnomalyType.NEGATIVE_PRICE,
#            SyntheticAnomalyType.EXTREME_OUTLIER,
            SyntheticAnomalyType.DECIMAL_SHIFT,
        ]
        # Add list_price_violation if list_price column exists
        if "list_price" in df.columns:
            anomaly_types.append(SyntheticAnomalyType.LIST_PRICE_VIOLATION)
        # Note: TITLE_COLLAPSE excluded - none of our detectors target it

    # Set up weights
    if type_weights is None:
        type_weights = {t: 1.0 / len(anomaly_types) for t in anomaly_types}
    else:
        # Normalize provided weights
        total = sum(type_weights.get(t, 0) for t in anomaly_types)
        if total > 0:
            type_weights = {t: type_weights.get(t, 0) / total for t in anomaly_types}
        else:
            type_weights = {t: 1.0 / len(anomaly_types) for t in anomaly_types}

    # Compute price statistics for realistic anomaly generation
    valid_prices = df["price"].dropna()
    valid_prices = valid_prices[valid_prices > 0]
    if len(valid_prices) == 0:
        logger.warning("inject_anomalies_to_dataframe: no valid prices found")
        return df.copy(), np.zeros(len(df), dtype=bool), []

    price_mean = float(valid_prices.mean())
    price_std = float(valid_prices.std()) if len(valid_prices) > 1 else price_mean * 0.1

    # Determine how many and which records to inject
    n_records = len(df)
    n_to_inject = max(1, int(n_records * injection_rate))

    # Only inject into rows with valid prices
    valid_mask = df["price"].notna() & (df["price"] > 0)
    valid_indices = df.index[valid_mask].tolist()

    if len(valid_indices) < n_to_inject:
        n_to_inject = len(valid_indices)

    if n_to_inject == 0:
        return df.copy(), np.zeros(n_records, dtype=bool), []

    # Create modified DataFrame
    df_modified = df.copy()
    inject_mask = np.zeros(n_records, dtype=bool)
    injection_details: list[dict] = []

    # Add columns for anomaly tracking - will travel with rows through sorting
    df_modified["__injected_anomaly_type__"] = None
    df_modified["__original_price__"] = df_modified["price"].copy()

    # ========================================================================
    # TWO-PHASE INJECTION: Inject LIST_PRICE_VIOLATION first into eligible rows
    # This ensures LIST_PRICE_VIOLATION is only injected where list_price > 0,
    # so Sanity detector can actually detect it.
    # ========================================================================

    phase1_count = 0
    phase1_eligible_count = 0
    phase2_count = 0
    injected_indices_set: set[int] = set()

    # Phase 1: Inject LIST_PRICE_VIOLATION into eligible rows
    if SyntheticAnomalyType.LIST_PRICE_VIOLATION in anomaly_types and "list_price" in df.columns:
        # Find eligible rows: valid price AND list_price > 0
        list_price_eligible = df.index[
            valid_mask & df["list_price"].notna() & (df["list_price"] > 0)
        ].tolist()
        phase1_eligible_count = len(list_price_eligible)

        if phase1_eligible_count > 0:
            # Allocate fair share to LIST_PRICE_VIOLATION (1/N of budget)
            n_list_price_inject = min(
                phase1_eligible_count,  # Can't inject more than eligible
                n_to_inject // len(anomaly_types),  # Fair share
            )

            if n_list_price_inject > 0:
                phase1_indices = rng.choice(
                    list_price_eligible, size=n_list_price_inject, replace=False
                )

                for idx in phase1_indices:
                    pos_idx = df_modified.index.get_loc(idx)
                    inject_mask[pos_idx] = True
                    injected_indices_set.add(idx)

                    original_price = float(df_modified.loc[idx, "price"])
                    new_price, detail = _apply_anomaly_to_price(
                        original_price=original_price,
                        anomaly_type=SyntheticAnomalyType.LIST_PRICE_VIOLATION,
                        rng=rng,
                        spike_range=spike_range,
                        drop_range=drop_range,
                        price_mean=price_mean,
                        price_std=price_std,
                        list_price=float(df_modified.loc[idx, "list_price"]),
                        currency=df_modified.loc[idx, "currency"] if "currency" in df.columns else None,
                    )

                    df_modified.loc[idx, "price"] = new_price
                    df_modified.loc[idx, "__injected_anomaly_type__"] = SyntheticAnomalyType.LIST_PRICE_VIOLATION.value

                    detail["index"] = int(idx)
                    detail["original_price"] = original_price
                    detail["new_price"] = new_price
                    detail["anomaly_type"] = SyntheticAnomalyType.LIST_PRICE_VIOLATION.value
                    detail["injection_phase"] = 1
                    injection_details.append(detail)

                phase1_count = len(phase1_indices)

    # Phase 2: Inject remaining anomaly types into other rows
    phase2_types = [t for t in anomaly_types if t != SyntheticAnomalyType.LIST_PRICE_VIOLATION]
    n_phase2_inject = n_to_inject - phase1_count

    if n_phase2_inject > 0 and phase2_types:
        # Select from rows NOT already injected in phase 1
        phase2_eligible = [i for i in valid_indices if i not in injected_indices_set]

        if len(phase2_eligible) < n_phase2_inject:
            n_phase2_inject = len(phase2_eligible)

        if n_phase2_inject > 0:
            phase2_indices = rng.choice(phase2_eligible, size=n_phase2_inject, replace=False)

            # Equal weights among phase 2 types
            phase2_weights = [1.0 / len(phase2_types)] * len(phase2_types)

            for idx in phase2_indices:
                pos_idx = df_modified.index.get_loc(idx)
                inject_mask[pos_idx] = True

                original_price = float(df_modified.loc[idx, "price"])
                original_list_price = None
                if "list_price" in df.columns:
                    list_price_val = df_modified.loc[idx, "list_price"]
                    if pd.notna(list_price_val):
                        original_list_price = float(list_price_val)

                # Select anomaly type from phase 2 types
                anomaly_type = rng.choice(phase2_types, p=phase2_weights)

                new_price, detail = _apply_anomaly_to_price(
                    original_price=original_price,
                    anomaly_type=anomaly_type,
                    rng=rng,
                    spike_range=spike_range,
                    drop_range=drop_range,
                    price_mean=price_mean,
                    price_std=price_std,
                    list_price=original_list_price,
                    currency=df_modified.loc[idx, "currency"] if "currency" in df.columns else None,
                )

                df_modified.loc[idx, "price"] = new_price
                new_list_price, list_price_detail = _co_scale_list_price_if_needed(
                    anomaly_type=anomaly_type,
                    original_price=original_price,
                    new_price=new_price,
                    original_list_price=original_list_price,
                )
                if "list_price" in df.columns and new_list_price is not None:
                    df_modified.loc[idx, "list_price"] = new_list_price
                detail.update(list_price_detail)
                if anomaly_type == SyntheticAnomalyType.CURRENCY_SWAP and "currency" in df.columns:
                    new_currency = detail.get("new_currency")
                    if isinstance(new_currency, str):
                        df_modified.loc[idx, "currency"] = new_currency
                df_modified.loc[idx, "__injected_anomaly_type__"] = anomaly_type.value

                # Handle title collapse
                if anomaly_type == SyntheticAnomalyType.TITLE_COLLAPSE and "product_name" in df.columns:
                    original_name = df_modified.loc[idx, "product_name"]
                    if pd.notna(original_name) and len(str(original_name)) > 5:
                        collapsed_name = str(original_name)[:max(5, len(str(original_name)) // 5)]
                        df_modified.loc[idx, "product_name"] = collapsed_name
                        detail["original_name"] = original_name
                        detail["collapsed_name"] = collapsed_name

                detail["index"] = int(idx)
                detail["original_price"] = original_price
                detail["new_price"] = new_price
                detail["anomaly_type"] = anomaly_type.value
                detail["injection_phase"] = 2
                injection_details.append(detail)

            phase2_count = len(phase2_indices)

    total_injected = phase1_count + phase2_count

    # Log phase breakdown
    logger.info(
        "anomaly_injection_phases",
        extra={
            "phase1_list_price_violations": phase1_count,
            "phase1_eligible_rows": phase1_eligible_count,
            "phase2_other_anomalies": phase2_count,
            "total_injected": total_injected,
        },
    )

    logger.info(
        "anomalies_injected_to_dataframe",
        extra={
            "total_records": n_records,
            "injected_count": total_injected,
            "injection_rate": total_injected / n_records if n_records > 0 else 0,
            "types_used": {t.value: sum(1 for d in injection_details if d["anomaly_type"] == t.value) for t in anomaly_types},
        },
    )

    return df_modified, inject_mask, injection_details


def _apply_anomaly_to_price(
    original_price: float,
    anomaly_type: SyntheticAnomalyType,
    rng: np.random.Generator,
    spike_range: tuple[float, float],
    drop_range: tuple[float, float],
    price_mean: float,
    price_std: float,
    list_price: float | None,
    currency: str | None = None,
) -> tuple[float, dict]:
    """Apply a specific anomaly type to a price value.

    Args:
        original_price: Original price value.
        anomaly_type: Type of anomaly to apply.
        rng: Random number generator.
        spike_range: (min, max) for spike multiplier.
        drop_range: (min, max) for drop multiplier.
        price_mean: Mean price in dataset (for outlier calculation).
        price_std: Std dev of prices (for outlier calculation).
        list_price: List price if available (for list_price_violation).
        currency: Currency token if available (for currency_swap).

    Returns:
        Tuple of (new_price, detail_dict).
    """
    detail: dict = {}

    if anomaly_type == SyntheticAnomalyType.PRICE_SPIKE:
        multiplier = rng.uniform(spike_range[0], spike_range[1])
        new_price = original_price * multiplier
        detail["multiplier"] = multiplier

    elif anomaly_type == SyntheticAnomalyType.PRICE_DROP:
        multiplier = rng.uniform(drop_range[0], drop_range[1])
        new_price = _apply_minimum_price(original_price * multiplier)
        detail["multiplier"] = multiplier

    elif anomaly_type == SyntheticAnomalyType.PRICE_NOISE:
        noise_pct = rng.uniform(-0.3, 0.3)
        new_price = _apply_minimum_price(original_price * (1 + noise_pct))
        detail["noise_pct"] = noise_pct

    elif anomaly_type == SyntheticAnomalyType.LIST_PRICE_VIOLATION:
        if list_price and list_price > 0:
            # Make sale price higher than list price
            new_price = list_price * rng.uniform(1.1, 1.5)
        else:
            # No list price, just spike the price
            new_price = original_price * rng.uniform(1.3, 1.8)
        detail["list_price"] = list_price

    elif anomaly_type == SyntheticAnomalyType.ZERO_PRICE:
        new_price = 0.0

    elif anomaly_type == SyntheticAnomalyType.NEGATIVE_PRICE:
        new_price = -abs(original_price * rng.uniform(0.1, 0.5))

    elif anomaly_type == SyntheticAnomalyType.EXTREME_OUTLIER:
        direction = rng.choice([-1, 1])
        deviation = rng.uniform(4, 8) * price_std
        new_price = _apply_minimum_price(price_mean + direction * deviation)
        detail["direction"] = "above" if direction > 0 else "below"
        detail["std_devs"] = deviation / price_std if price_std > 0 else 0

    elif anomaly_type == SyntheticAnomalyType.TITLE_COLLAPSE:
        # Title collapse usually comes with a price change
        new_price = original_price * rng.uniform(0.5, 1.5)
        detail["title_collapsed"] = True

    elif anomaly_type == SyntheticAnomalyType.DECIMAL_SHIFT:
        shift = rng.choice([100.0, 0.01])
        new_price = _apply_minimum_price(original_price * shift)
        detail["decimal_shift"] = "x100" if shift == 100.0 else "/100"

    elif anomaly_type == SyntheticAnomalyType.CURRENCY_SWAP:
        new_currency, swap_factor = _resolve_currency_swap(currency)
        new_price = _apply_minimum_price(original_price * swap_factor)
        detail["original_currency"] = currency
        detail["new_currency"] = new_currency
        detail["currency_swap_factor"] = swap_factor

    elif anomaly_type == SyntheticAnomalyType.TRANSIENT_SPIKE:
        new_price = original_price * rng.uniform(1.5, 3.0)
        detail["is_transient"] = True

    elif anomaly_type == SyntheticAnomalyType.PERSISTENT_CHANGE:
        new_price = _apply_minimum_price(original_price * rng.uniform(0.7, 1.3))
        detail["is_persistent"] = True

    else:
        # Unknown type, no change
        new_price = original_price

    detail["change_pct"] = (new_price - original_price) / original_price if original_price != 0 else 0

    return new_price, detail


def _apply_minimum_price(price: float, minimum: float = MIN_REALISTIC_PRICE) -> float:
    """Clamp reduced-price anomalies to a realistic positive floor."""
    return max(minimum, price)


def _co_scale_list_price_if_needed(
    anomaly_type: SyntheticAnomalyType,
    original_price: float,
    new_price: float,
    original_list_price: float | None,
) -> tuple[float | None, dict[str, float]]:
    """Scale list_price with price for non-sanity-targeted price mutations."""
    if (
        anomaly_type.value not in LIST_PRICE_COSCALE_TYPES
        or original_list_price is None
        or original_list_price <= 0
        or original_price <= 0
    ):
        return original_list_price, {}

    actual_multiplier = new_price / original_price
    new_list_price = original_list_price * actual_multiplier
    if actual_multiplier < 1.0:
        new_list_price = _apply_minimum_price(new_list_price)

    return new_list_price, {
        "original_list_price": original_list_price,
        "new_list_price": new_list_price,
        "list_price_multiplier": actual_multiplier,
    }


def evaluate_detection(
    detection_results: list[Any],  # AnomalyResult
    injected_anomalies: list[InjectedAnomaly],
    score_threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate detection results against known injected anomalies.

    Args:
        detection_results: List of AnomalyResult from detector.
        injected_anomalies: List of InjectedAnomaly (ground truth).
        score_threshold: Score threshold for counting as detected.

    Returns:
        Dictionary with evaluation metrics.
    """
    # Build set of injected indices for fast lookup
    injected_indices = {a.record_index for a in injected_anomalies}

    # Count detection outcomes
    true_positives = 0  # Injected and detected
    false_positives = 0  # Not injected but detected
    false_negatives = 0  # Injected but not detected
    true_negatives = 0  # Not injected and not detected

    detected_anomalies = []
    missed_anomalies = []

    for i, result in enumerate(detection_results):
        is_injected = i in injected_indices
        is_detected = result.is_anomaly and result.anomaly_score >= score_threshold

        if is_injected and is_detected:
            true_positives += 1
            detected_anomalies.append(i)
        elif is_injected and not is_detected:
            false_negatives += 1
            missed_anomalies.append(i)
        elif not is_injected and is_detected:
            false_positives += 1
        else:
            true_negatives += 1

    # Calculate metrics
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Detection rate by anomaly type
    type_detection = {}
    for anomaly_type in SyntheticAnomalyType:
        type_anomalies = [a for a in injected_anomalies if a.anomaly_type == anomaly_type]
        if type_anomalies:
            type_detected = sum(
                1 for a in type_anomalies if detection_results[a.record_index].is_anomaly
            )
            type_detection[anomaly_type.value] = {
                "injected": len(type_anomalies),
                "detected": type_detected,
                "rate": type_detected / len(type_anomalies),
            }

    metrics = {
        "total_records": len(detection_results),
        "injected_anomalies": len(injected_anomalies),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "detection_rate": recall,  # Same as recall
        "false_positive_rate": (
            false_positives / (false_positives + true_negatives)
            if (false_positives + true_negatives) > 0
            else 0
        ),
        "detection_by_type": type_detection,
        "detected_indices": detected_anomalies,
        "missed_indices": missed_anomalies,
    }

    logger.info(
        "evaluation_complete",
        extra={
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        },
    )

    return metrics


def evaluate_classifier(
    classification_results: list[Any],  # ScrapeIssueClassification
    injected_anomalies: list[InjectedAnomaly],
    scrape_bug_types: set[SyntheticAnomalyType] | None = None,
) -> dict[str, Any]:
    """Evaluate scrape-bug classifier against known injections.

    Args:
        classification_results: List of ScrapeIssueClassification results.
        injected_anomalies: List of InjectedAnomaly (ground truth).
        scrape_bug_types: Set of anomaly types that are scrape bugs (vs real events).
            Defaults to TITLE_COLLAPSE, DECIMAL_SHIFT, CURRENCY_SWAP, ZERO_PRICE, NEGATIVE_PRICE.

    Returns:
        Dictionary with classifier evaluation metrics.
    """
    if scrape_bug_types is None:
        scrape_bug_types = {
            SyntheticAnomalyType.TITLE_COLLAPSE,
            SyntheticAnomalyType.DECIMAL_SHIFT,
            SyntheticAnomalyType.CURRENCY_SWAP,
            SyntheticAnomalyType.ZERO_PRICE,
            SyntheticAnomalyType.NEGATIVE_PRICE,
        }

    # Build lookup for injected indices and their types
    injected_map = {a.record_index: a for a in injected_anomalies}

    # Evaluate classifier
    true_positives = 0  # Correctly identified scrape bugs
    false_positives = 0  # Real events classified as scrape bugs
    true_negatives = 0  # Correctly identified real events
    false_negatives = 0  # Scrape bugs classified as real events

    for i, result in enumerate(classification_results):
        is_injected = i in injected_map
        is_scrape_bug = is_injected and injected_map[i].anomaly_type in scrape_bug_types

        # Classification decision (suppress_downstream indicates classifier thinks it's a bug)
        classified_as_bug = result.suppress_downstream

        if is_scrape_bug and classified_as_bug:
            true_positives += 1
        elif is_scrape_bug and not classified_as_bug:
            false_negatives += 1
        elif not is_scrape_bug and classified_as_bug:
            false_positives += 1
        else:
            true_negatives += 1

    # Calculate metrics
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Classification by type
    type_classification = {}
    for anomaly_type in SyntheticAnomalyType:
        type_anomalies = [a for a in injected_anomalies if a.anomaly_type == anomaly_type]
        if type_anomalies:
            is_bug_type = anomaly_type in scrape_bug_types
            correctly_classified = sum(
                1
                for a in type_anomalies
                if classification_results[a.record_index].suppress_downstream == is_bug_type
            )
            type_classification[anomaly_type.value] = {
                "count": len(type_anomalies),
                "is_scrape_bug_type": is_bug_type,
                "correctly_classified": correctly_classified,
                "accuracy": correctly_classified / len(type_anomalies),
            }

    metrics = {
        "total_classified": len(classification_results),
        "scrape_bugs_injected": sum(
            1 for a in injected_anomalies if a.anomaly_type in scrape_bug_types
        ),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": (
            (true_positives + true_negatives) / len(classification_results)
            if classification_results
            else 0.0
        ),
        "classification_by_type": type_classification,
    }

    logger.info(
        "classifier_evaluation_complete",
        extra={
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": true_positives,
            "false_negatives": false_negatives,
        },
    )

    return metrics
