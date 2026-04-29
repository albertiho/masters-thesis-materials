"""DetectorEvaluator - Wraps a detector with isolated temporal cache.

This class provides each detector with its own isolated cache and implements
conditional cache updates based on anomaly detection results. Unlike the
SequentialEvaluator which shares a single cache across all detectors and
updates unconditionally, DetectorEvaluator:

1. Gives each detector its own cache (isolated evolution of "clean" prices)
2. Only updates the cache when a price is NOT flagged as anomaly
3. Records anomalies for persistence tracking (without polluting baseline)

This prevents anomalous prices from polluting the baseline statistics used
for future anomaly detection.

Usage:
    # Create evaluator with detector
    evaluator = DetectorEvaluator(ZScoreDetector(), name="zscore")
    
    # Clear and populate cache
    evaluator.clear()
    evaluator.populate_cache(historical_df)
    
    # Process records sequentially
    for row in test_df.itertuples(index=False):
        result = evaluator.process_row(row, col_map)
"""

import logging
import math
import re
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.anomaly.batch_processor import BatchRoundProcessor
from src.anomaly.statistical import AnomalyResult
from src.features.numeric import NumericFeatures
from src.features.temporal import (
    TemporalCacheManager,
    TemporalFeatures,
)

logger = logging.getLogger(__name__)


class DetectorEvaluator:
    """Wraps a detector with its own isolated temporal cache.
    
    Each DetectorEvaluator maintains an independent cache that only includes
    prices that this detector did NOT flag as anomalous. This ensures each
    detector's baseline evolves based on its own decisions.
    
    Attributes:
        name: Human-readable name for this evaluator.
        detector: The underlying detector with detect() method.
        temporal_cache: Isolated TemporalCacheManager for this detector.
        enable_persistence_acceptance: Whether to enable persistence-based acceptance.
    """

    # Default rolling window size (matches temporal.py)
    HISTORY_DEPTH = 30

    def __init__(
        self,
        detector: Any,
        name: str | None = None,
        enable_persistence_acceptance: bool = True,
    ) -> None:
        """Initialize the evaluator with a detector.
        
        Args:
            detector: Any detector with a detect() method.
            name: Optional name for this evaluator. Defaults to detector.name
                or the class name.
            enable_persistence_acceptance: If True, prices that persist longer
                than PRICE_PERSIST_HOURS are automatically accepted into the
                baseline. Set to False to evaluate without persistence acceptance.
        """
        self.detector = detector
        self.name = name or getattr(detector, "name", type(detector).__name__)
        # Create isolated cache and populate it directly from DataFrames.
        self.temporal_cache = TemporalCacheManager()
        self.enable_persistence_acceptance = enable_persistence_acceptance
        
        # Pre-detect detector type for efficient dispatch
        self._detector_type = self._detect_detector_type(detector)

    def clear(self) -> None:
        """Reset cache to empty state."""
        self.temporal_cache.clear()
        logger.debug(
            "detector_evaluator_cache_cleared",
            extra={"evaluator": self.name},
        )

    def populate_cache(self, historical_df: pd.DataFrame | None) -> None:
        """Populate cache from historical data.
        
        No-op if df is empty or None (for cold-start scenarios).
        
        Args:
            historical_df: DataFrame with historical price data containing
                product_id, competitor_id, price, and optionally a time column.
        """
        if historical_df is None or historical_df.empty:
            logger.debug(
                "detector_evaluator_populate_skipped",
                extra={"evaluator": self.name, "reason": "empty_or_none"},
            )
            return
        
        self._populate_from_df(historical_df)
        
        logger.info(
            "detector_evaluator_cache_populated",
            extra={
                "evaluator": self.name,
                "records_processed": len(historical_df),
                "cache_stats": self.temporal_cache.get_stats(),
            },
        )

    def process_row(
        self,
        row: tuple,
        col_map: dict[str, int],
        country: str | None = None,
    ) -> AnomalyResult:
        """Process a single row: detect anomaly and conditionally update cache.
        
        Processing steps:
        1. Check persistence acceptance (if enabled)
        2. Get temporal features from cache (BEFORE any update)
        3. Create numeric features from row
        4. Call detector.detect()
        5. If NOT anomaly: add price to cache
        6. If anomaly: record for persistence tracking (don't add to baseline)
        
        Args:
            row: Named tuple from df.itertuples(index=False).
            col_map: Dict mapping column names to indices.
            country: Optional country code for numeric features.
            
        Returns:
            AnomalyResult from the detector.
        """
        # Extract identifiers
        product_id = str(row[col_map["product_id"]])
        competitor_id = str(row[col_map["competitor_id"]])
        price_val = row[col_map["price"]]
        price = float(price_val) if pd.notna(price_val) else 0.0
        
        # Get timestamp
        current_timestamp = self._extract_timestamp(row, col_map)
        
        # Step 1: Check persistence acceptance BEFORE detection (if enabled)
        accepted_via_persistence = False
        if self.enable_persistence_acceptance and current_timestamp:
            accepted_via_persistence = self.temporal_cache.check_and_accept_persisted_price(
                product_id=product_id,
                competitor_id=competitor_id,
                current_price=price,
                current_time=current_timestamp,
            )
        
        # Step 2: Get temporal features from cache BEFORE any update (no look-ahead)
        temporal_features = self._get_temporal_features(
            product_id=product_id,
            competitor_id=competitor_id,
            current_price=price,
            current_timestamp=current_timestamp,
        )
        
        # Step 3: Create numeric features from row
        numeric_features = self._numeric_features_from_tuple(row, col_map, country)
        
        # Get price history for history-dependent detectors
        cache_entry = self.temporal_cache.get(product_id, competitor_id)
        price_history = None
        if cache_entry is not None and len(cache_entry.price_history) > 0:
            price_history = cache_entry.price_history.tolist()
        
        # Step 4: Call detector.detect()
        result = self._detect(
            numeric_features=numeric_features,
            temporal_features=temporal_features,
            price_history=price_history,
        )
        
        # Step 5 & 6: Handle persistence acceptance and conditional cache update
        if self.enable_persistence_acceptance and accepted_via_persistence:
            # Price was accepted via persistence - override anomaly flag
            result = AnomalyResult(
                is_anomaly=False,
                anomaly_score=result.anomaly_score,
                anomaly_types=result.anomaly_types,
                severity=result.severity,
                details={**result.details, "accepted_via_persistence": True},
                detector=result.detector,
                competitor_product_id=result.competitor_product_id,
                competitor=result.competitor,
            )
            # Cache was already updated by check_and_accept_persisted_price
        elif result.is_anomaly:
            # ANOMALY: Record for persistence tracking, but DON'T add to baseline
            if self.enable_persistence_acceptance and current_timestamp:
                self.temporal_cache.record_anomaly(
                    product_id=product_id,
                    competitor_id=competitor_id,
                    price=price,
                    timestamp=current_timestamp,
                )
            # Note: We deliberately do NOT call update_if_changed here
        else:
            # NOT ANOMALY: Add price to baseline cache
            self.temporal_cache.update_if_changed(
                product_id=product_id,
                competitor_id=competitor_id,
                price=price,
                scraped_at=current_timestamp or datetime.now(timezone.utc),
            )
        
        return result

    def _populate_from_df(self, df: pd.DataFrame) -> None:
        """Load historical prices into TemporalCacheManager.
        
        Groups data by (product_id, competitor_id), sorts by timestamp,
        and creates cache entries with price history.
        
        Args:
            df: DataFrame with price history to load.
        """
        if df.empty:
            return
        
        # Determine time column
        time_col = self._get_time_column(df)
        
        # Sort by time for correct chronological order
        if time_col:
            df = df.sort_values(time_col)
        
        # Group by product and competitor
        grouped = df.groupby(["product_id", "competitor_id"])
        total_groups = len(grouped)
        log_interval = max(10000, total_groups // 10)  # Log every 10k groups or 10%
        
        logger.info(f"Populating cache: {len(df):,} rows, {total_groups:,} product-competitor groups")
        
        for i, ((product_id, competitor_id), group) in enumerate(grouped):
            # Sort group by time
            if time_col:
                group = group.sort_values(time_col)
            
            # Extract prices in chronological order
            prices = group["price"].dropna().tolist()
            
            if prices:
                # Create cache entry with price history
                self.temporal_cache._create_entry(
                    product_id=str(product_id),
                    competitor_id=str(competitor_id),
                    prices=prices,
                )
                
                # Set last_seen_at from latest record
                if time_col and time_col in group.columns:
                    last_time = group[time_col].iloc[-1]
                    cache_entry = self.temporal_cache.get(str(product_id), str(competitor_id))
                    if cache_entry and pd.notna(last_time):
                        if isinstance(last_time, pd.Timestamp):
                            cache_entry.last_seen_at = last_time.to_pydatetime()
                        elif isinstance(last_time, datetime):
                            cache_entry.last_seen_at = last_time
            
            # Log progress
            if (i + 1) % log_interval == 0:
                pct = (i + 1) / total_groups * 100
                logger.info(f"  Cache population progress: {i + 1:,}/{total_groups:,} groups ({pct:.0f}%)")

    def _get_temporal_features(
        self,
        product_id: str,
        competitor_id: str,
        current_price: float,
        current_timestamp: datetime | None,
    ) -> TemporalFeatures:
        """Get TemporalFeatures from cache entry.
        
        Args:
            product_id: Product identifier.
            competitor_id: Competitor identifier.
            current_price: Current price for z-score computation.
            current_timestamp: Current timestamp for days_since_change.
            
        Returns:
            TemporalFeatures instance (empty if no cache entry).
        """
        cache_entry = self.temporal_cache.get(product_id, competitor_id)
        
        if cache_entry is None:
            # No history - return empty features
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
                competitor_product_id=product_id,
                competitor=competitor_id,
            )
        
        # Use the existing from_cache method
        return TemporalFeatures.from_cache(cache_entry, current_price, current_timestamp)

    def _detect(
        self,
        numeric_features: NumericFeatures,
        temporal_features: TemporalFeatures,
        price_history: list[float] | None = None,
    ) -> AnomalyResult:
        """Call appropriate detect method based on detector type.
        
        Args:
            numeric_features: Numeric features from the record.
            temporal_features: Temporal features from cache.
            price_history: Optional price history for history-dependent detectors.
            
        Returns:
            AnomalyResult from the detector.
        """
        if self._detector_type == "combined":
            # Combined detectors use DetectionContext
            from src.anomaly.combined import DetectionContext
            
            context = DetectionContext.from_features(
                numeric_features=numeric_features,
                temporal_features=temporal_features,
                price_history=price_history,
                observation_count=temporal_features.observation_count,
            )
            return self.detector.detect(context)
        
        elif self._detector_type == "statistical_ensemble":
            # StatisticalEnsemble needs price_history for IQR detector
            return self.detector.detect(
                numeric_features=numeric_features,
                temporal_features=temporal_features,
                price_history=price_history,
                invariant_context=None,
            )
        
        elif getattr(self.detector, "requires_price_history", False):
            # History-dependent statistical detectors consume raw prior history.
            return self.detector.detect(
                numeric_features=numeric_features,
                temporal_features=temporal_features,
                price_history=price_history,
            )
        
        elif self._detector_type == "sanity":
            # SanityCheckDetector only takes numeric_features (no temporal!)
            return self.detector.detect(numeric_features=numeric_features)
        
        elif self._detector_type in ("isolation_forest", "autoencoder"):
            # ML detectors take numeric and temporal features
            return self.detector.detect(
                numeric_features=numeric_features,
                temporal_features=temporal_features,
            )
        
        else:
            # Other statistical detectors (zscore, threshold)
            return self.detector.detect(
                numeric_features=numeric_features,
                temporal_features=temporal_features,
            )

    def _numeric_features_from_tuple(
        self,
        row: tuple,
        col_map: dict[str, int],
        country: str | None,
    ) -> NumericFeatures:
        """Create NumericFeatures from itertuples row.
        
        Args:
            row: Named tuple from df.itertuples(index=False).
            col_map: Dict mapping column names to indices.
            country: Country code (derived from model name).
            
        Returns:
            NumericFeatures instance.
        """
        price_val = row[col_map["price"]]
        price = float(price_val) if pd.notna(price_val) else 0.0
        
        list_price = None
        if "list_price" in col_map:
            lp_val = row[col_map["list_price"]]
            if pd.notna(lp_val):
                list_price = float(lp_val)
        
        # Compute price_ratio
        if list_price and list_price > 0:
            price_ratio = price / list_price
        else:
            price_ratio = 1.0
        
        # Compute price_log
        price_log = math.log(price + 1) if price >= 0 else 0.0
        
        # Validation
        validation_errors: list[str] = []
        is_valid = True
        
        if price <= 0:
            validation_errors.append("invalid_price")
            is_valid = False
        
        # Get identifiers
        competitor_product_id = ""
        if "competitor_product_id" in col_map:
            competitor_product_id = str(row[col_map["competitor_product_id"]])
        elif "product_id" in col_map:
            competitor_product_id = str(row[col_map["product_id"]])
        
        competitor = ""
        if "competitor_id" in col_map:
            competitor = str(row[col_map["competitor_id"]])
        
        return NumericFeatures(
            price=price,
            list_price=list_price,
            price_ratio=price_ratio,
            has_list_price=list_price is not None,
            price_log=price_log,
            is_valid=is_valid,
            validation_errors=validation_errors,
            competitor_product_id=competitor_product_id,
            competitor=competitor,
            country=country,
        )

    def _extract_timestamp(
        self,
        row: tuple,
        col_map: dict[str, int],
    ) -> datetime | None:
        """Extract timestamp from row.
        
        Args:
            row: Named tuple from df.itertuples(index=False).
            col_map: Dict mapping column names to indices.
            
        Returns:
            Datetime or None if not found.
        """
        for col_name in ["first_seen_at", "scraped_at", "timestamp"]:
            if col_name in col_map:
                ts = row[col_map[col_name]]
                if pd.notna(ts):
                    if isinstance(ts, pd.Timestamp):
                        return ts.to_pydatetime()
                    elif isinstance(ts, datetime):
                        return ts
        return None

    def _detect_detector_type(self, detector: Any) -> str:
        """Detect the type of detector for appropriate method dispatch.
        
        Args:
            detector: The detector instance.
            
        Returns:
            Detector type string: "combined", "statistical_ensemble", "sanity",
            "isolation_forest", "autoencoder", or "statistical".
        """
        # Check if it's a BaseCombinedDetector
        if hasattr(detector, "get_layers") and hasattr(detector, "route_by_history"):
            return "combined"
        
        # Check for StatisticalEnsemble (has zscore_detector and iqr_detector)
        if hasattr(detector, "zscore_detector") and hasattr(detector, "iqr_detector"):
            return "statistical_ensemble"
        
        # Check by name attribute
        name = getattr(detector, "name", "")
        
        if name == "isolation_forest":
            return "isolation_forest"
        elif name == "autoencoder":
            return "autoencoder"
        elif name == "sanity":
            return "sanity"
        elif name in ("zscore", "threshold", "invariant"):
            return "statistical"
        
        # Default to statistical
        return "statistical"

    def _get_time_column(self, df: pd.DataFrame) -> str | None:
        """Find the timestamp column in the DataFrame.
        
        Args:
            df: DataFrame to search.
            
        Returns:
            Column name or None if not found.
        """
        for col in ["first_seen_at", "scraped_at", "timestamp"]:
            if col in df.columns:
                return col
        return None

    def supports_batch(self) -> bool:
        """Check if the detector supports batch detection.
        
        Returns:
            True if detector has a detect_batch() method.
        """
        if self._detector_type in ("autoencoder", "isolation_forest"):
            return hasattr(self.detector, "detect_batch")
        if self._detector_type == "combined":
            return hasattr(self.detector, "detect_batch")
        return False

    def process_batch(
        self,
        rows: list[tuple],
        col_map: dict[str, int],
        country: str | None = None,
    ) -> list[AnomalyResult]:
        """Process a batch of rows using rounds-based batch detection.
        
        Uses BatchRoundProcessor for train-serve consistency: processes rows in
        rounds (one observation per product per round), updating the cache between
        rounds so later observations see earlier ones in their temporal features.
        
        This is critical for products with multiple test observations. Without
        rounds-based processing, obs2 wouldn't see obs1 in its history, causing
        25%+ missing history and train-serve skew.
        
        Performance: Still ~10-50x faster than sequential process_row() because
        each round uses batch detection. Products with single observations are
        unaffected.
        
        Args:
            rows: List of tuples from df.itertuples(index=False).
            col_map: Dict mapping column names to indices.
            country: Optional country code for numeric features.
            
        Returns:
            List of AnomalyResult, one per row, in original order.
        """
        if not self.supports_batch():
            raise ValueError(f"Detector {self.name} does not support batch processing")
        
        # Combined detectors use DetectionContext - need different processing
        if self._detector_type == "combined":
            return self._process_batch_combined(rows, col_map, country)
        
        # Create processor with closures that capture country
        processor = BatchRoundProcessor(
            detector=self.detector,
            cache=self.temporal_cache,
            get_temporal_features=self._get_temporal_features,
            get_numeric_features=lambda row, cm: self._numeric_features_from_tuple(row, cm, country),
        )
        
        return processor.process(rows, col_map)

    def _process_batch_combined(
        self,
        rows: list[tuple],
        col_map: dict[str, int],
        country: str | None = None,
    ) -> list[AnomalyResult]:
        """Process batch for combined detectors using DetectionContext.
        
        Combined detectors expect detect_batch(contexts: list[DetectionContext])
        rather than separate numeric/temporal feature lists. Uses rounds-based
        processing for train-serve consistency.
        
        Args:
            rows: List of tuples from df.itertuples(index=False).
            col_map: Dict mapping column names to indices.
            country: Optional country code for numeric features.
            
        Returns:
            List of AnomalyResult, one per row, in original order.
        """
        from collections import defaultdict
        from src.anomaly.combined import DetectionContext
        
        if not rows:
            return []
        
        # Build row contexts and group by product
        row_contexts = self._build_row_contexts(rows, col_map)
        product_queues = self._group_rows_by_product(row_contexts)
        
        # Track results by original index
        results_by_index: dict[int, AnomalyResult] = {}
        
        # Track stats
        total_rounds = 0
        total_cache_updates = 0
        
        # Process rounds until all queues are empty
        while any(queue for queue in product_queues.values()):
            # Pop one observation per product
            round_rows = self._pop_round_combined(product_queues)
            if not round_rows:
                break
            
            # Build DetectionContext for each row in this round
            contexts: list[DetectionContext] = []
            for row_ctx in round_rows:
                # Get temporal features from cache (BEFORE any update)
                temporal_features = self._get_temporal_features(
                    row_ctx["product_id"],
                    row_ctx["competitor_id"],
                    row_ctx["price"],
                    row_ctx["timestamp"],
                )
                
                # Get numeric features
                numeric_features = self._numeric_features_from_tuple(
                    row_ctx["row"], col_map, country
                )
                
                # Get price history for history-dependent detectors within combined
                cache_entry = self.temporal_cache.get(
                    row_ctx["product_id"], row_ctx["competitor_id"]
                )
                price_history = None
                if cache_entry is not None and len(cache_entry.price_history) > 0:
                    price_history = cache_entry.price_history.tolist()
                
                # Build DetectionContext
                context = DetectionContext.from_features(
                    numeric_features=numeric_features,
                    temporal_features=temporal_features,
                    price_history=price_history,
                    observation_count=temporal_features.observation_count,
                )
                contexts.append(context)
            
            # Run batch detection on combined detector
            results = self.detector.detect_batch(contexts)
            
            # Count anomalies in this round
            round_anomalies = sum(1 for r in results if r.is_anomaly)
            
            # Store results and prepare cache updates
            non_anomaly_updates: list[tuple[str, str, float, datetime]] = []
            for row_ctx, result in zip(round_rows, results, strict=True):
                results_by_index[row_ctx["original_index"]] = result
                
                if not result.is_anomaly:
                    non_anomaly_updates.append((
                        row_ctx["product_id"],
                        row_ctx["competitor_id"],
                        row_ctx["price"],
                        row_ctx["timestamp"] or datetime.now(timezone.utc),
                    ))
            
            # Update cache with non-anomalous prices
            if non_anomaly_updates:
                self.temporal_cache.update_batch(non_anomaly_updates)
                total_cache_updates += len(non_anomaly_updates)
            
            total_rounds += 1
            
            # Log cache stats after each round
            self._log_round_cache_stats(
                round_num=total_rounds,
                round_size=len(round_rows),
                round_anomalies=round_anomalies,
                cache_updates=len(non_anomaly_updates),
            )
        
        logger.debug(
            "combined_batch_round_processing_complete",
            extra={
                "evaluator": self.name,
                "total_rows": len(rows),
                "total_rounds": total_rounds,
                "unique_products": len(product_queues),
                "cache_updates": total_cache_updates,
            },
        )
        
        # Reconstruct results in original order
        return [results_by_index[i] for i in range(len(rows))]

    def _build_row_contexts(
        self,
        rows: list[tuple],
        col_map: dict[str, int],
    ) -> list[dict[str, Any]]:
        """Build context dicts for batch processing.
        
        Args:
            rows: List of row tuples.
            col_map: Column name to index mapping.
            
        Returns:
            List of context dicts with row data.
        """
        contexts = []
        for i, row in enumerate(rows):
            product_id = str(row[col_map["product_id"]])
            competitor_id = str(row[col_map["competitor_id"]])
            
            price_val = row[col_map["price"]]
            price = float(price_val) if pd.notna(price_val) else 0.0
            
            timestamp = self._extract_timestamp(row, col_map)
            
            contexts.append({
                "original_index": i,
                "product_id": product_id,
                "competitor_id": competitor_id,
                "price": price,
                "timestamp": timestamp,
                "row": row,
            })
        
        return contexts

    def _group_rows_by_product(
        self,
        contexts: list[dict[str, Any]],
    ) -> dict[tuple[str, str], list[dict[str, Any]]]:
        """Group row contexts by (product_id, competitor_id) and sort by timestamp.
        
        Args:
            contexts: List of row context dicts.
            
        Returns:
            Dict mapping (product_id, competitor_id) to sorted list of contexts.
        """
        from collections import defaultdict
        
        product_queues: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        
        for ctx in contexts:
            key = (ctx["product_id"], ctx["competitor_id"])
            product_queues[key].append(ctx)
        
        # Sort each queue by timestamp (chronological order)
        for key in product_queues:
            product_queues[key].sort(
                key=lambda x: (x["timestamp"] is None, x["timestamp"] or datetime.min)
            )
        
        return dict(product_queues)

    def _pop_round_combined(
        self,
        product_queues: dict[tuple[str, str], list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Pop one observation from each non-empty product queue.
        
        Args:
            product_queues: Dict of product queues (modified in place).
            
        Returns:
            List of context dicts for this round.
        """
        round_contexts = []
        
        for key, queue in product_queues.items():
            if queue:
                ctx = queue.pop(0)
                round_contexts.append(ctx)
        
        return round_contexts

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics for monitoring.
        
        Returns:
            Dict with cache statistics.
        """
        return self.temporal_cache.get_stats()

    def _log_round_cache_stats(
        self,
        round_num: int,
        round_size: int,
        round_anomalies: int,
        cache_updates: int,
    ) -> None:
        """Log cache statistics after a batch round.
        
        Args:
            round_num: Current round number.
            round_size: Number of rows processed in this round.
            round_anomalies: Number of anomalies detected in this round.
            cache_updates: Number of cache entries updated this round.
        """
        import numpy as np
        
        # Compute cache statistics
        total_products = 0
        total_obs = 0
        obs_counts = []
        
        for competitor_id, competitor_cache in self.temporal_cache._caches.items():
            for product_id, entry in competitor_cache.items():
                total_products += 1
                obs_counts.append(entry.observation_count)
                total_obs += entry.observation_count
        
        if total_products > 0:
            avg_history = total_obs / total_products
            obs_arr = np.array(obs_counts)
            print(
                f"    [ROUND {round_num}] rows={round_size:,}, anomalies={round_anomalies}, "
                f"cache_updates={cache_updates} | "
                f"Cache: {total_products:,} products, avg_history={avg_history:.2f}, "
                f"min={obs_arr.min()}, max={obs_arr.max()}, median={np.median(obs_arr):.0f}"
            )
        else:
            print(
                f"    [ROUND {round_num}] rows={round_size:,}, anomalies={round_anomalies}, "
                f"cache_updates={cache_updates} | Cache: empty"
            )
