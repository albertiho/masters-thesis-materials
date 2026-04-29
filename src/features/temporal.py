"""Temporal feature extraction for anomaly detection.

Enhanced version with:
- NumPy arrays for efficient computation
- Robust statistics (median, MAD, percentiles)
- Per-competitor caches with 100% coverage (no LRU eviction)
- Local in-memory cache misses initialized on demand
- Stats recomputation only on price change

Architecture:
    Per-competitor caches (unbounded - memory is not a constraint):
    - Each competitor has its own dict of ProductTemporalCache
    - Each product has a numpy array of prices (last 30 observations)
    - Create empty cache entries on cache miss
    - Stats only recomputed when price changes

TODO (Dual-Cache Architecture):
    Current: Single cache updates unconditionally with every price observation.
    
    Problem: A legitimate 40% price drop (e.g., TV sale) will always appear as an
    anomaly because the rolling stats baseline never incorporates the new price.
    
    Proposed Enhancement: Two caches per product:
    1. All prices cache: Every observation, regardless of anomaly status.
       Used to detect price stability ("price has been $X for N days").
    2. Clean prices cache: Only non-anomalous prices.
       Used for computing baseline rolling stats for anomaly detection.
    
    This would allow distinguishing:
    - Scraper bug: Price spikes randomly, changes next day → anomaly
    - Real price change: Price drops and stays for 4+ days → likely legitimate
    
    Impact: Each detector would have its own clean-cache evolution since different
    detectors flag different observations as anomalies.
    
    See: todo.md "Dual-cache architecture for temporal features"
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from src.config import get_price_persist_hours, get_price_persist_threshold
from src.utils.memory import log_memory

logger = logging.getLogger(__name__)

# Default rolling window size (number of observations)
DEFAULT_HISTORY_DEPTH = 30

# Minimum observations required for meaningful statistics
MIN_OBSERVATIONS = 3

# MAD scaling factor for normal distribution equivalence
MAD_SCALE_FACTOR = 1.4826


def _prices_within_threshold(price1: float, price2: float, threshold: float) -> bool:
    """Check if two prices are within the persistence threshold.

    Used for anomaly persistence tracking to allow small price variations
    to be treated as "the same price" for persistence purposes.

    Args:
        price1: Reference price (typically the tracked anomaly price).
        price2: Current price to compare.
        threshold: Percentage tolerance as decimal (0.03 = 3%).

    Returns:
        True if prices are within threshold of each other.
    """
    if threshold == 0.0:
        return price1 == price2

    # Use the reference (tracked) price as the basis for percentage calculation
    max_diff = price1 * threshold
    return abs(price1 - price2) <= max_diff


@dataclass
class ProductTemporalCache:
    """Per-product temporal cache with robust statistics.

    Uses numpy arrays for efficient median/percentile computation.
    Stats are pre-computed and only updated when price changes.

    Attributes:
        product_id: Product identifier (int for new schema, str for legacy).
        competitor_id: Competitor identifier.
        price_history: NumPy array of recent prices (max 30).
        median: Median price.
        mad: Median Absolute Deviation.
        mean: Mean price.
        std: Standard deviation.
        percentile_5: 5th percentile.
        percentile_95: 95th percentile.
        min_price: Minimum price in history.
        max_price: Maximum price in history.
        last_price: Most recent price.
        last_change_at: When price last changed.
        consecutive_unchanged: Runs with same price.
        last_seen_at: Last time this product was observed.
        observation_count: Total observations in history.
    """

    product_id: int | str
    competitor_id: str

    # NumPy array for efficient computation
    price_history: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))

    # Robust statistics
    median: float | None = None
    mad: float | None = None
    percentile_5: float | None = None
    percentile_95: float | None = None

    # Standard statistics (backward compat)
    mean: float | None = None
    std: float | None = None
    min_price: float | None = None
    max_price: float | None = None

    # Change tracking
    last_price: float | None = None
    last_change_at: datetime | None = None
    consecutive_unchanged: int = 0
    last_seen_at: datetime | None = None

    observation_count: int = 0

    # Anomaly persistence tracking
    # Used to detect when a flagged price has persisted long enough to be accepted
    anomaly_price: float | None = None  # Price that was flagged as anomalous
    anomaly_first_seen_at: datetime | None = None  # When the anomaly price was first observed

    def compute_robust_zscore(self, current_price: float) -> float | None:
        """Compute robust z-score for current price.

        Uses MAD-based z-score: (price - median) / (1.4826 * MAD)
        The 1.4826 factor makes MAD comparable to std for normal distributions.

        Args:
            current_price: Price to compute z-score for.

        Returns:
            Robust z-score or None if insufficient data.
        """
        if self.median is None or self.mad is None:
            return None
        if self.mad == 0:
            # No variation - return 0 if at median, else large value
            return 0.0 if current_price == self.median else 10.0
        return (current_price - self.median) / (MAD_SCALE_FACTOR * self.mad)

    def get_percentile_position(self, current_price: float) -> float | None:
        """Get position of current price in the p5-p95 range.

        Returns:
            0-1 value where 0 = at p5, 1 = at p95, None if insufficient data.
        """
        if self.percentile_5 is None or self.percentile_95 is None:
            return None
        range_size = self.percentile_95 - self.percentile_5
        if range_size == 0:
            return 0.5
        position = (current_price - self.percentile_5) / range_size
        return max(0.0, min(1.0, position))  # Clamp to 0-1

    @property
    def has_sufficient_history(self) -> bool:
        """Whether enough history exists for reliable statistics."""
        return self.observation_count >= MIN_OBSERVATIONS


def recompute_stats(entry: ProductTemporalCache) -> None:
    """Recompute all statistics from price history.

    Called only when price changes. Uses numpy for efficiency.

    Args:
        entry: Cache entry to update.
    """
    if len(entry.price_history) == 0:
        entry.observation_count = 0
        entry.median = None
        entry.mad = None
        entry.mean = None
        entry.std = None
        entry.percentile_5 = None
        entry.percentile_95 = None
        entry.min_price = None
        entry.max_price = None
        return

    prices = entry.price_history
    entry.observation_count = len(prices)

    # Robust statistics (numpy built-ins)
    entry.median = float(np.median(prices))
    entry.percentile_5 = float(np.percentile(prices, 5))
    entry.percentile_95 = float(np.percentile(prices, 95))

    # MAD: Median Absolute Deviation
    entry.mad = float(np.median(np.abs(prices - entry.median)))

    # Standard statistics
    entry.mean = float(np.mean(prices))
    entry.std = float(np.std(prices, ddof=1)) if len(prices) > 1 else 0.0
    entry.min_price = float(np.min(prices))
    entry.max_price = float(np.max(prices))


class TemporalCacheManager:
    """Per-competitor temporal caches for local execution.

    Features:
    - 100% cache coverage (no LRU eviction for the standalone workflow)
    - Missing products initialized as empty histories on demand
    - Stats only recomputed when price changes
    - NumPy arrays for efficient computation
    """

    HISTORY_DEPTH = DEFAULT_HISTORY_DEPTH

    def __init__(self):
        """Initialize the cache manager."""
        # Per-competitor caches (unbounded)
        self._caches: dict[str, dict[int | str, ProductTemporalCache]] = {}

        # Stats for monitoring
        self._cache_hits = 0
        self._cache_misses = 0

    def get(self, product_id: int | str, competitor_id: str) -> ProductTemporalCache | None:
        """Get cache entry for a product.

        Args:
            product_id: Product identifier.
            competitor_id: Competitor identifier.

        Returns:
            Cache entry or None if not in cache.
        """
        cache = self._caches.get(competitor_id)
        if cache is None:
            return None
        return cache.get(product_id)

    def get_many(
        self,
        keys: list[tuple[int | str, str]],
    ) -> dict[tuple[int | str, str], ProductTemporalCache | None]:
        """Check cache for multiple products.

        Args:
            keys: List of (product_id, competitor_id) tuples.

        Returns:
            Dict mapping keys to cache entries (None for misses).
        """
        results = {}
        for product_id, competitor_id in keys:
            entry = self.get(product_id, competitor_id)
            results[(product_id, competitor_id)] = entry
            if entry is not None:
                self._cache_hits += 1
            else:
                self._cache_misses += 1
        return results

    async def ensure_cache_populated(
        self,
        keys: list[tuple[int | str, str]],
    ) -> dict[str, Any]:
        """Ensure all products are represented in the local cache.

        Args:
            keys: List of (product_id, competitor_id) tuples.

        Returns:
            Dict with load statistics.
        """
        # Check cache for all products
        cache_results = self.get_many(keys)

        # Collect misses
        missing = [k for k, v in cache_results.items() if v is None]

        if not missing:
            return {
                "status": "all_cached",
                "products_requested": len(keys),
                "cache_hits": len(keys),
                "cache_misses": 0,
                "products_loaded": 0,
            }

        for product_id, competitor_id in missing:
            self._create_entry(product_id, competitor_id, [])

        total_entries = sum(len(c) for c in self._caches.values())
        cache_bytes = self.estimate_cache_bytes()
        log_memory(
            "temporal_cache_loaded",
            cache_entries=total_entries,
            cache_estimated_mb=round(cache_bytes / (1024 * 1024), 1),
            products_loaded=0,
        )

        return {
            "status": "initialized",
            "products_requested": len(keys),
            "cache_hits": len(keys) - len(missing),
            "cache_misses": len(missing),
            "products_loaded": 0,
            "products_without_history": len(missing),
            "observations_loaded": 0,
            "duration_ms": 0,
        }

    def _create_entry(
        self,
        product_id: int | str,
        competitor_id: str,
        prices: list[float],
    ) -> ProductTemporalCache:
        """Create a cache entry with computed stats.

        Args:
            product_id: Product identifier.
            competitor_id: Competitor identifier.
            prices: List of historical prices (chronological order).

        Returns:
            Created cache entry.
        """
        cache = self._caches.setdefault(competitor_id, {})

        price_array = np.array(prices[-self.HISTORY_DEPTH :], dtype=np.float64)

        entry = ProductTemporalCache(
            product_id=product_id,
            competitor_id=competitor_id,
            price_history=price_array,
            last_price=prices[-1] if prices else None,
        )
        recompute_stats(entry)

        cache[product_id] = entry
        return entry

    def update_if_changed(
        self,
        product_id: int | str,
        competitor_id: str,
        price: float,
        scraped_at: datetime,
    ) -> bool:
        """Update cache if price changed. Returns True if price was new.

        Only recomputes stats when price actually changes.

        Args:
            product_id: Product identifier.
            competitor_id: Competitor identifier.
            price: Current price.
            scraped_at: When the price was observed.

        Returns:
            True if price changed (stats recomputed), False otherwise.
        """
        entry = self.get(product_id, competitor_id)

        if entry is None:
            # New product - create entry
            entry = self._create_entry(product_id, competitor_id, [price])
            entry.last_change_at = scraped_at
            entry.last_seen_at = scraped_at
            return True

        entry.last_seen_at = scraped_at

        # Check if price unchanged
        if entry.last_price is not None and entry.last_price == price:
            entry.consecutive_unchanged += 1
            return False

        # Price changed - update history and recompute stats
        entry.price_history = np.append(entry.price_history, price)[-self.HISTORY_DEPTH :]
        entry.last_price = price
        entry.last_change_at = scraped_at
        entry.consecutive_unchanged = 0
        recompute_stats(entry)

        return True

    def record_anomaly(
        self,
        product_id: int | str,
        competitor_id: str,
        price: float,
        timestamp: datetime,
    ) -> None:
        """Record that a price was flagged as anomalous.

        Called after anomaly detection flags a price. This starts the persistence
        tracking clock for the price.

        If the price is different from the currently tracked anomaly price
        (beyond the tolerance threshold), resets the tracking (new anomaly).

        Args:
            product_id: Product identifier.
            competitor_id: Competitor identifier.
            price: The anomalous price.
            timestamp: When the anomaly was observed.
        """
        entry = self.get(product_id, competitor_id)

        if entry is None:
            # Product not in cache - create entry first
            entry = self._create_entry(product_id, competitor_id, [])
            entry.last_seen_at = timestamp

        # Check if this is a new anomaly price or same as tracked (within threshold)
        threshold = get_price_persist_threshold()
        if entry.anomaly_price is None or not _prices_within_threshold(
            entry.anomaly_price, price, threshold
        ):
            # New anomaly price - start tracking
            entry.anomaly_price = price
            entry.anomaly_first_seen_at = timestamp

            logger.debug(
                "anomaly_tracking_started",
                extra={
                    "product_id": product_id,
                    "competitor_id": competitor_id,
                    "anomaly_price": price,
                    "first_seen_at": timestamp.isoformat(),
                    "threshold": threshold,
                },
            )
        # If same price (within threshold), keep the original first_seen_at (don't reset the clock)

    def check_and_accept_persisted_price(
        self,
        product_id: int | str,
        competitor_id: str,
        current_price: float,
        current_time: datetime,
    ) -> bool:
        """Check if an anomaly price has persisted long enough to accept.

        If the current price matches the tracked anomaly price (within the
        tolerance threshold) and the persistence threshold (PRICE_PERSIST_HOURS)
        has been exceeded, automatically adds the price to the baseline history
        and clears anomaly tracking.

        Args:
            product_id: Product identifier.
            competitor_id: Competitor identifier.
            current_price: Current price observation.
            current_time: Current timestamp.

        Returns:
            True if the price was accepted into baseline (caller should
            skip anomaly flagging), False otherwise.
        """
        entry = self.get(product_id, competitor_id)

        if entry is None:
            return False

        # Check if we have an anomaly being tracked
        if entry.anomaly_price is None or entry.anomaly_first_seen_at is None:
            return False

        # Check if current price matches the tracked anomaly price (within threshold)
        threshold = get_price_persist_threshold()
        if not _prices_within_threshold(entry.anomaly_price, current_price, threshold):
            # Price changed beyond threshold - this is a different situation
            # Clear anomaly tracking since the anomaly is gone
            entry.anomaly_price = None
            entry.anomaly_first_seen_at = None
            return False

        # Calculate hours since anomaly was first seen
        hours_elapsed = (current_time - entry.anomaly_first_seen_at).total_seconds() / 3600
        threshold_hours = get_price_persist_hours()

        if hours_elapsed >= threshold_hours:
            # Price has persisted long enough - accept into baseline
            logger.info(
                "anomaly_price_accepted_via_persistence",
                extra={
                    "product_id": product_id,
                    "competitor_id": competitor_id,
                    "price": current_price,
                    "tracked_anomaly_price": entry.anomaly_price,
                    "hours_elapsed": round(hours_elapsed, 2),
                    "threshold_hours": threshold_hours,
                    "price_threshold": threshold,
                    "first_seen_at": entry.anomaly_first_seen_at.isoformat(),
                },
            )

            # Add to baseline history and recompute stats
            entry.price_history = np.append(entry.price_history, current_price)[
                -self.HISTORY_DEPTH :
            ]
            entry.last_price = current_price
            entry.last_change_at = current_time
            entry.consecutive_unchanged = 0
            recompute_stats(entry)

            # Clear anomaly tracking
            entry.anomaly_price = None
            entry.anomaly_first_seen_at = None

            return True

        # Not enough time has passed yet
        return False

    def clear_anomaly_tracking(
        self,
        product_id: int | str,
        competitor_id: str,
    ) -> None:
        """Clear anomaly tracking for a product.

        Called when a price is no longer flagged as anomalous (e.g., after
        manual review or if the price returns to normal).

        Args:
            product_id: Product identifier.
            competitor_id: Competitor identifier.
        """
        entry = self.get(product_id, competitor_id)

        if entry is not None:
            entry.anomaly_price = None
            entry.anomaly_first_seen_at = None

    def estimate_cache_bytes(self) -> int:
        """Estimate total memory consumed by the temporal caches.

        Counts numpy array storage plus a per-entry overhead estimate for
        the Python dataclass, dict slots, and string keys.
        """
        PER_ENTRY_OVERHEAD = 500  # dict slot + dataclass fields + key strings
        total = 0
        for cache in self._caches.values():
            for entry in cache.values():
                total += PER_ENTRY_OVERHEAD
                if entry.price_history is not None:
                    total += entry.price_history.nbytes
        return total

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics for monitoring.

        Returns:
            Dict with cache statistics.
        """
        total_products = 0
        total_observations = 0
        per_competitor = {}

        for competitor_id, cache in self._caches.items():
            products = len(cache)
            observations = sum(e.observation_count for e in cache.values())

            per_competitor[competitor_id] = {
                "products": products,
                "observations": observations,
            }

            total_products += products
            total_observations += observations

        return {
            "total_products": total_products,
            "total_observations": total_observations,
            "competitors": len(self._caches),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "per_competitor": per_competitor,
        }

    def update_batch(
        self,
        updates: list[tuple[int | str, str, float, datetime]],
    ) -> dict[str, int]:
        """Batch update cache for multiple products.
        
        More efficient than calling update_if_changed() repeatedly because it
        avoids repeated dict lookups and can be optimized for bulk operations.
        
        Args:
            updates: List of (product_id, competitor_id, price, scraped_at) tuples.
        
        Returns:
            Dict with counts: {"updated": N, "unchanged": M, "new": P}
        """
        counts = {"updated": 0, "unchanged": 0, "new": 0}
        
        for product_id, competitor_id, price, scraped_at in updates:
            entry = self.get(product_id, competitor_id)
            
            if entry is None:
                # New product - create entry
                entry = self._create_entry(product_id, competitor_id, [price])
                entry.last_change_at = scraped_at
                entry.last_seen_at = scraped_at
                counts["new"] += 1
                continue
            
            entry.last_seen_at = scraped_at
            
            # Check if price unchanged
            if entry.last_price is not None and entry.last_price == price:
                entry.consecutive_unchanged += 1
                counts["unchanged"] += 1
                continue
            
            # Price changed - update history and recompute stats
            entry.price_history = np.append(entry.price_history, price)[-self.HISTORY_DEPTH :]
            entry.last_price = price
            entry.last_change_at = scraped_at
            entry.consecutive_unchanged = 0
            recompute_stats(entry)
            counts["updated"] += 1
        
        return counts

    def clear(self) -> None:
        """Clear all cached data."""
        self._caches.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("temporal_cache_cleared")

    def copy_from(self, other: "TemporalCacheManager") -> None:
        """Deep copy all cache entries from another manager.
        
        This is useful for initializing multiple evaluators with the same
        baseline data without reprocessing the historical data multiple times.
        
        Args:
            other: Source cache manager to copy from.
        """
        self._caches.clear()
        for competitor_id, product_cache in other._caches.items():
            self._caches[competitor_id] = {}
            for product_id, entry in product_cache.items():
                # Deep copy the entry, including numpy array
                new_entry = ProductTemporalCache(
                    product_id=entry.product_id,
                    competitor_id=entry.competitor_id,
                    price_history=entry.price_history.copy(),
                    median=entry.median,
                    mad=entry.mad,
                    percentile_5=entry.percentile_5,
                    percentile_95=entry.percentile_95,
                    mean=entry.mean,
                    std=entry.std,
                    min_price=entry.min_price,
                    max_price=entry.max_price,
                    last_price=entry.last_price,
                    last_change_at=entry.last_change_at,
                    consecutive_unchanged=entry.consecutive_unchanged,
                    last_seen_at=entry.last_seen_at,
                    observation_count=entry.observation_count,
                    anomaly_price=entry.anomaly_price,
                    anomaly_first_seen_at=entry.anomaly_first_seen_at,
                )
                self._caches[competitor_id][product_id] = new_entry
        
        # Copy stats
        self._cache_hits = other._cache_hits
        self._cache_misses = other._cache_misses

    def save_to_file(self, path: str) -> dict[str, int]:
        """Save cache to a file for persistence or debugging.
        
        Uses joblib for efficient numpy array serialization.
        
        Args:
            path: File path to save to (should end with .joblib).
        
        Returns:
            Dict with save statistics.
        """
        from joblib import dump as joblib_dump
        
        # Convert to serializable format
        data = {
            "caches": {},
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
        }
        
        total_products = 0
        total_observations = 0
        
        for competitor_id, product_cache in self._caches.items():
            data["caches"][competitor_id] = {}
            for product_id, entry in product_cache.items():
                data["caches"][competitor_id][product_id] = {
                    "product_id": entry.product_id,
                    "competitor_id": entry.competitor_id,
                    "price_history": entry.price_history,  # numpy array
                    "median": entry.median,
                    "mad": entry.mad,
                    "percentile_5": entry.percentile_5,
                    "percentile_95": entry.percentile_95,
                    "mean": entry.mean,
                    "std": entry.std,
                    "min_price": entry.min_price,
                    "max_price": entry.max_price,
                    "last_price": entry.last_price,
                    "last_change_at": entry.last_change_at,
                    "consecutive_unchanged": entry.consecutive_unchanged,
                    "last_seen_at": entry.last_seen_at,
                    "observation_count": entry.observation_count,
                    "anomaly_price": entry.anomaly_price,
                    "anomaly_first_seen_at": entry.anomaly_first_seen_at,
                }
                total_products += 1
                total_observations += entry.observation_count
        
        joblib_dump(data, path)
        
        logger.info(
            "temporal_cache_saved",
            extra={
                "path": path,
                "n_competitors": len(self._caches),
                "n_products": total_products,
                "n_observations": total_observations,
            },
        )
        
        return {
            "n_competitors": len(self._caches),
            "n_products": total_products,
            "n_observations": total_observations,
        }

    def load_from_file(self, path: str) -> dict[str, int]:
        """Load cache from a file.
        
        Args:
            path: File path to load from.
        
        Returns:
            Dict with load statistics.
        """
        from joblib import load as joblib_load
        
        data = joblib_load(path)
        
        self._caches.clear()
        total_products = 0
        total_observations = 0
        
        for competitor_id, product_cache in data["caches"].items():
            self._caches[competitor_id] = {}
            for product_id, entry_data in product_cache.items():
                entry = ProductTemporalCache(
                    product_id=entry_data["product_id"],
                    competitor_id=entry_data["competitor_id"],
                    price_history=entry_data["price_history"],
                    median=entry_data["median"],
                    mad=entry_data["mad"],
                    percentile_5=entry_data["percentile_5"],
                    percentile_95=entry_data["percentile_95"],
                    mean=entry_data["mean"],
                    std=entry_data["std"],
                    min_price=entry_data["min_price"],
                    max_price=entry_data["max_price"],
                    last_price=entry_data["last_price"],
                    last_change_at=entry_data["last_change_at"],
                    consecutive_unchanged=entry_data["consecutive_unchanged"],
                    last_seen_at=entry_data["last_seen_at"],
                    observation_count=entry_data["observation_count"],
                    anomaly_price=entry_data["anomaly_price"],
                    anomaly_first_seen_at=entry_data["anomaly_first_seen_at"],
                )
                self._caches[competitor_id][product_id] = entry
                total_products += 1
                total_observations += entry.observation_count
        
        self._cache_hits = data.get("cache_hits", 0)
        self._cache_misses = data.get("cache_misses", 0)
        
        logger.info(
            "temporal_cache_loaded",
            extra={
                "path": path,
                "n_competitors": len(self._caches),
                "n_products": total_products,
                "n_observations": total_observations,
            },
        )
        
        return {
            "n_competitors": len(self._caches),
            "n_products": total_products,
            "n_observations": total_observations,
        }


# =============================================================================
# Legacy compatibility - keep old classes for existing code
# =============================================================================


@dataclass
class TemporalFeatures:
    """Container for temporal/rolling features (legacy compatibility).

    Attributes:
        rolling_mean: Mean price over the rolling window.
        rolling_std: Standard deviation over the rolling window.
        rolling_min: Minimum price in the rolling window.
        rolling_max: Maximum price in the rolling window.
        price_zscore: Z-score of current price vs rolling distribution.
        price_change_pct: Percentage change from previous observation.
        days_since_change: Days since last significant price change.
        observation_count: Number of historical observations available.
        has_sufficient_history: Whether enough history exists for statistics.
    """

    rolling_mean: float | None
    rolling_std: float | None
    rolling_min: float | None
    rolling_max: float | None
    price_zscore: float | None
    price_change_pct: float | None
    days_since_change: float | None
    observation_count: int
    has_sufficient_history: bool

    # Identifiers
    competitor_product_id: str
    competitor: str

    # New robust stats (optional for backward compat)
    rolling_median: float | None = None
    rolling_mad: float | None = None
    robust_zscore: float | None = None
    percentile_5: float | None = None
    percentile_95: float | None = None
    consecutive_unchanged: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            "competitor_product_id": self.competitor_product_id,
            "competitor": self.competitor,
            "rolling_mean": self.rolling_mean,
            "rolling_std": self.rolling_std,
            "rolling_min": self.rolling_min,
            "rolling_max": self.rolling_max,
            "price_zscore": self.price_zscore,
            "price_change_pct": self.price_change_pct,
            "days_since_change": self.days_since_change,
            "observation_count": self.observation_count,
            "has_sufficient_history": self.has_sufficient_history,
            "rolling_median": self.rolling_median,
            "rolling_mad": self.rolling_mad,
            "robust_zscore": self.robust_zscore,
            "percentile_5": self.percentile_5,
            "percentile_95": self.percentile_95,
            "consecutive_unchanged": self.consecutive_unchanged,
        }

    @classmethod
    def from_cache(
        cls,
        cache: ProductTemporalCache,
        current_price: float,
        current_timestamp: datetime | None = None,
    ) -> "TemporalFeatures":
        """Create TemporalFeatures from ProductTemporalCache.

        Bridge method for legacy code.

        Args:
            cache: ProductTemporalCache entry.
            current_price: Current price for z-score computation.
            current_timestamp: Current timestamp for days_since_change.

        Returns:
            TemporalFeatures instance.
        """
        # Compute price_change_pct
        price_change_pct = None
        if cache.last_price is not None and cache.last_price != current_price:
            if cache.last_price > 0:
                price_change_pct = (current_price - cache.last_price) / cache.last_price

        # Compute days_since_change
        days_since_change = None
        if current_timestamp and cache.last_change_at:
            delta = current_timestamp - cache.last_change_at
            days_since_change = delta.total_seconds() / 86400

        # Compute standard z-score
        price_zscore = None
        if cache.mean is not None and cache.std is not None and cache.std > 0:
            price_zscore = (current_price - cache.mean) / cache.std

        return cls(
            rolling_mean=cache.mean,
            rolling_std=cache.std,
            rolling_min=cache.min_price,
            rolling_max=cache.max_price,
            price_zscore=price_zscore,
            price_change_pct=price_change_pct,
            days_since_change=days_since_change,
            observation_count=cache.observation_count,
            has_sufficient_history=cache.has_sufficient_history,
            competitor_product_id=str(cache.product_id),
            competitor=cache.competitor_id,
            rolling_median=cache.median,
            rolling_mad=cache.mad,
            robust_zscore=cache.compute_robust_zscore(current_price),
            percentile_5=cache.percentile_5,
            percentile_95=cache.percentile_95,
            consecutive_unchanged=cache.consecutive_unchanged,
        )


# Legacy class kept for backward compatibility
class TemporalFeatureStore:
    """Legacy temporal feature store.

    This class is kept for backward compatibility with existing code.
    New code should use TemporalCacheManager instead.
    """

    def __init__(self, window_size: int = DEFAULT_HISTORY_DEPTH):
        """Initialize the feature store."""
        self._manager = TemporalCacheManager()
        self.window_size = window_size

    def add_observation(
        self,
        competitor_product_id: str,
        competitor: str,
        price: float,
        timestamp: datetime,
        scraped_at: datetime | None = None,
    ) -> None:
        """Add a price observation for a product."""
        self._manager.update_if_changed(
            product_id=competitor_product_id,
            competitor_id=competitor,
            price=price,
            scraped_at=scraped_at or timestamp,
        )

    def get_temporal_features(
        self,
        competitor_product_id: str,
        competitor: str,
        current_price: float,
        current_timestamp: datetime | None = None,
    ) -> TemporalFeatures:
        """Compute temporal features for a product."""
        cache = self._manager.get(competitor_product_id, competitor)

        if cache is None:
            return TemporalFeatures(
                rolling_mean=None,
                rolling_std=None,
                rolling_min=None,
                rolling_max=None,
                price_zscore=None,
                price_change_pct=None,
                days_since_change=None,
                observation_count=0,
                has_sufficient_history=False,
                competitor_product_id=competitor_product_id,
                competitor=competitor,
            )

        return TemporalFeatures.from_cache(cache, current_price, current_timestamp)

    def get_store_stats(self) -> dict[str, Any]:
        """Get statistics about the feature store."""
        return self._manager.get_stats()

    def ensure_history_for_products(
        self,
        products: list[tuple[str, str]],
        days: int = 30,
    ) -> dict[str, Any]:
        """Ensure products have local cache entries before feature extraction."""
        del days  # maintained for backward compatibility

        keys = [(p[0], p[1]) for p in products]
        missing = 0
        for product_id, competitor in keys:
            if self._manager.get(product_id, competitor) is None:
                self._manager._create_entry(product_id, competitor, [])
                missing += 1

        return {
            "status": "initialized",
            "products_requested": len(keys),
            "cache_hits": len(keys) - missing,
            "cache_misses": missing,
            "products_loaded": 0,
            "products_without_history": missing,
            "observations_loaded": 0,
            "duration_ms": 0,
        }

    def clear(self) -> None:
        """Clear all cached data."""
        self._manager.clear()


def compute_rolling_statistics(
    prices: list[float],
    window_size: int = DEFAULT_HISTORY_DEPTH,
) -> dict[str, float | None]:
    """Compute rolling statistics for a list of prices.

    Utility function for batch processing without the store.

    Args:
        prices: List of historical prices (chronological order).
        window_size: Number of observations to use.

    Returns:
        Dictionary with rolling statistics including robust stats.
    """
    if not prices:
        return {
            "rolling_mean": None,
            "rolling_std": None,
            "rolling_min": None,
            "rolling_max": None,
            "rolling_median": None,
            "rolling_mad": None,
            "percentile_5": None,
            "percentile_95": None,
        }

    # Use most recent observations
    recent = np.array(prices[-window_size:], dtype=np.float64)

    median = float(np.median(recent))
    mad = float(np.median(np.abs(recent - median)))

    return {
        "rolling_mean": float(np.mean(recent)),
        "rolling_std": float(np.std(recent, ddof=1)) if len(recent) > 1 else 0.0,
        "rolling_min": float(np.min(recent)),
        "rolling_max": float(np.max(recent)),
        "rolling_median": median,
        "rolling_mad": mad,
        "percentile_5": float(np.percentile(recent, 5)),
        "percentile_95": float(np.percentile(recent, 95)),
    }
