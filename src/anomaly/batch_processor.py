"""BatchRoundProcessor - Rounds-based batch processing for train-serve consistency.

This module solves the "cache not updating mid-batch" problem where products
with multiple test observations don't see their earlier observations in the
temporal cache when computing features for later observations.

The Problem:
    For products with multiple test observations (e.g., obs1 and obs2):
    - Training: obs2's features are computed from history that includes obs1
    - Batch inference: Both obs1 and obs2 features computed from initial cache
    - Result: 25%+ missing history for obs2, causing train-serve skew

The Solution:
    Process batch in "rounds" where each round contains at most one observation
    per product. After each round:
    1. Compute features from current cache state
    2. Run batch detection
    3. Update cache with non-anomalous prices
    4. Proceed to next round with updated cache

This ensures obs2's features include obs1 (if obs1 was not anomalous),
matching the training data generation process.

Usage:
    processor = BatchRoundProcessor(
        detector=evaluator.detector,
        cache=evaluator.temporal_cache,
        get_temporal_features=evaluator._get_temporal_features,
        get_numeric_features=evaluator._numeric_features_from_tuple,
    )
    results = processor.process(rows, col_map)
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd

from src.anomaly.statistical import AnomalyResult
from src.features.numeric import NumericFeatures
from src.features.temporal import TemporalCacheManager, TemporalFeatures

logger = logging.getLogger(__name__)


@dataclass
class RowContext:
    """Context for a single row being processed.
    
    Stores the original row data and its position in the input for
    reconstructing results in original order.
    
    Attributes:
        original_index: Position in the original input list.
        product_id: Product identifier.
        competitor_id: Competitor identifier.
        price: Current price.
        timestamp: Observation timestamp.
        row: Original row tuple.
    """
    original_index: int
    product_id: str
    competitor_id: str
    price: float
    timestamp: datetime | None
    row: tuple


@dataclass
class RoundResult:
    """Result from processing a single round.
    
    Attributes:
        contexts: Row contexts processed in this round.
        results: AnomalyResult for each row.
        non_anomaly_updates: Cache updates for non-anomalous prices.
    """
    contexts: list[RowContext]
    results: list[AnomalyResult]
    non_anomaly_updates: list[tuple[str, str, float, datetime]]


class BatchRoundProcessor:
    """Process batches in rounds for train-serve consistency.
    
    Each round processes at most one observation per product, then updates
    the cache before the next round. This ensures later observations within
    a batch see earlier observations in their temporal features.
    
    Attributes:
        detector: Detector with detect_batch() method.
        cache: TemporalCacheManager for the detector.
        get_temporal_features: Function to get TemporalFeatures from cache.
        get_numeric_features: Function to create NumericFeatures from row.
    """
    
    def __init__(
        self,
        detector: Any,
        cache: TemporalCacheManager,
        get_temporal_features: Callable[
            [str, str, float, datetime | None], TemporalFeatures
        ],
        get_numeric_features: Callable[[tuple, dict[str, int]], NumericFeatures],
        get_price_history: Callable[[str, str], list[float] | None] | None = None,
    ) -> None:
        """Initialize the batch processor.
        
        Args:
            detector: Detector with detect_batch() method.
            cache: TemporalCacheManager for temporal features and updates.
            get_temporal_features: Function(product_id, competitor_id, price, timestamp)
                that returns TemporalFeatures from the cache.
            get_numeric_features: Function(row, col_map) that returns NumericFeatures.
            get_price_history: Optional function(product_id, competitor_id) that
                returns raw historical prices for history-dependent detectors.
        """
        self.detector = detector
        self.cache = cache
        self.get_temporal_features = get_temporal_features
        self.get_numeric_features = get_numeric_features
        self.get_price_history = get_price_history
    
    def process(
        self,
        rows: list[tuple],
        col_map: dict[str, int],
    ) -> list[AnomalyResult]:
        """Process a batch of rows using rounds-based processing.
        
        Main entry point. Processes rows in rounds, updating cache between
        rounds, and returns results in the original input order.
        
        Args:
            rows: List of tuples from df.itertuples(index=False).
            col_map: Dict mapping column names to indices.
            
        Returns:
            List of AnomalyResult, one per row, in original order.
        """
        if not rows:
            return []
        
        # Step 1: Build contexts and group by product
        contexts = self._build_contexts(rows, col_map)
        product_queues = self._group_by_product(contexts)
        
        # Track results by original index
        results_by_index: dict[int, AnomalyResult] = {}
        
        # Track stats
        total_rounds = 0
        total_cache_updates = 0
        
        # Step 2: Process rounds until all queues are empty
        while any(queue for queue in product_queues.values()):
            round_result = self._process_round(product_queues, col_map)
            
            # Store results by original index
            for ctx, result in zip(round_result.contexts, round_result.results, strict=True):
                results_by_index[ctx.original_index] = result
            
            # Update cache with non-anomalous prices
            if round_result.non_anomaly_updates:
                self.cache.update_batch(round_result.non_anomaly_updates)
                total_cache_updates += len(round_result.non_anomaly_updates)
            
            total_rounds += 1
        
        # Log processing stats
        logger.debug(
            "batch_round_processing_complete",
            extra={
                "total_rows": len(rows),
                "total_rounds": total_rounds,
                "unique_products": len(product_queues),
                "cache_updates": total_cache_updates,
            },
        )
        
        # Step 3: Reconstruct results in original order
        return [results_by_index[i] for i in range(len(rows))]
    
    def _build_contexts(
        self,
        rows: list[tuple],
        col_map: dict[str, int],
    ) -> list[RowContext]:
        """Extract context from each row.
        
        Args:
            rows: List of row tuples.
            col_map: Column name to index mapping.
            
        Returns:
            List of RowContext objects.
        """
        contexts = []
        for i, row in enumerate(rows):
            product_id = str(row[col_map["product_id"]])
            competitor_id = str(row[col_map["competitor_id"]])
            
            price_val = row[col_map["price"]]
            price = float(price_val) if pd.notna(price_val) else 0.0
            
            timestamp = self._extract_timestamp(row, col_map)
            
            contexts.append(RowContext(
                original_index=i,
                product_id=product_id,
                competitor_id=competitor_id,
                price=price,
                timestamp=timestamp,
                row=row,
            ))
        
        return contexts
    
    def _group_by_product(
        self,
        contexts: list[RowContext],
    ) -> dict[tuple[str, str], list[RowContext]]:
        """Group contexts by (product_id, competitor_id) and sort by timestamp.
        
        Args:
            contexts: List of row contexts.
            
        Returns:
            Dict mapping (product_id, competitor_id) to sorted list of contexts.
        """
        product_queues: dict[tuple[str, str], list[RowContext]] = defaultdict(list)
        
        for ctx in contexts:
            key = (ctx.product_id, ctx.competitor_id)
            product_queues[key].append(ctx)
        
        # Sort each queue by timestamp (chronological order)
        # None timestamps sort to the end
        for key in product_queues:
            product_queues[key].sort(
                key=lambda x: (x.timestamp is None, x.timestamp or datetime.min)
            )
        
        return dict(product_queues)
    
    def _pop_round(
        self,
        product_queues: dict[tuple[str, str], list[RowContext]],
    ) -> list[RowContext]:
        """Pop one observation from each non-empty product queue.
        
        Args:
            product_queues: Dict of product queues (modified in place).
            
        Returns:
            List of contexts for this round (one per product with remaining data).
        """
        round_contexts = []
        
        for key, queue in product_queues.items():
            if queue:
                # Pop first element (earliest timestamp)
                ctx = queue.pop(0)
                round_contexts.append(ctx)
        
        return round_contexts
    
    def _process_round(
        self,
        product_queues: dict[tuple[str, str], list[RowContext]],
        col_map: dict[str, int],
    ) -> RoundResult:
        """Process one round of observations.
        
        Gets one observation per product, builds features, runs detection,
        and prepares cache updates.
        
        Args:
            product_queues: Product queues (modified in place).
            col_map: Column name to index mapping.
            
        Returns:
            RoundResult with contexts, results, and cache updates.
        """
        # Pop one observation per product
        round_contexts = self._pop_round(product_queues)
        
        if not round_contexts:
            return RoundResult(contexts=[], results=[], non_anomaly_updates=[])
        
        # Build features for this round
        numeric_features_list: list[NumericFeatures] = []
        temporal_features_list: list[TemporalFeatures] = []
        
        for ctx in round_contexts:
            # Get temporal features from cache (BEFORE any update)
            temporal_features = self.get_temporal_features(
                ctx.product_id,
                ctx.competitor_id,
                ctx.price,
                ctx.timestamp,
            )
            temporal_features_list.append(temporal_features)
            
            # Get numeric features from row
            numeric_features = self.get_numeric_features(ctx.row, col_map)
            numeric_features_list.append(numeric_features)
        
        detect_batch_kwargs: dict[str, Any] = {
            "numeric_features_list": numeric_features_list,
            "temporal_features_list": temporal_features_list,
        }
        if self.get_price_history is not None and getattr(self.detector, "requires_price_history", False):
            detect_batch_kwargs["price_history_list"] = [
                self.get_price_history(ctx.product_id, ctx.competitor_id)
                for ctx in round_contexts
            ]

        # Run batch detection
        results = self.detector.detect_batch(**detect_batch_kwargs)
        
        # Collect non-anomalous prices for cache update
        non_anomaly_updates: list[tuple[str, str, float, datetime]] = []
        for ctx, result in zip(round_contexts, results, strict=True):
            if not result.is_anomaly:
                non_anomaly_updates.append((
                    ctx.product_id,
                    ctx.competitor_id,
                    ctx.price,
                    ctx.timestamp or datetime.now(timezone.utc),
                ))
        
        return RoundResult(
            contexts=round_contexts,
            results=results,
            non_anomaly_updates=non_anomaly_updates,
        )
    
    def _extract_timestamp(
        self,
        row: tuple,
        col_map: dict[str, int],
    ) -> datetime | None:
        """Extract timestamp from row.
        
        Args:
            row: Row tuple.
            col_map: Column name to index mapping.
            
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
