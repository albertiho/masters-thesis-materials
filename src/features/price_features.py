"""Price Feature Extractor - Extract features for price anomaly detection (Layer 2).

This extractor produces ~22 price-focused features for anomaly detection.
Features are designed to capture price behavior, context, and anomalies.

Includes both standard statistics and robust statistics (median/MAD) from
ProductTemporalCache for better outlier resistance.

Usage:
    extractor = PriceFeatureExtractor()
    
    features = extractor.extract(
        product_id=123,
        competitor_id="PROSHOP_DK",
        price_data=PriceData(...),
        temporal_cache=cache_manager.get(123, "PROSHOP_DK"),
        context_data=ContextData(...),
    )
"""

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from src.features.base import (
    FeatureExtractor,
    FeatureVector,
    safe_divide,
    safe_pct_change,
    safe_ratio,
)

if TYPE_CHECKING:
    from src.features.temporal import ProductTemporalCache

logger = logging.getLogger(__name__)


@dataclass
class PriceData:
    """Current price observation from product_prices table.
    
    Attributes:
        price: Current selling price.
        list_price: Original/before-discount price.
        currency: Currency code.
        price_status: confirmed, unavailable, hidden, free.
        seller_type: retailer, marketplace_first_party, marketplace_third_party.
        is_club_price: Whether this is a member-only price.
        show_discount: Whether discount is displayed.
    """
    
    price: float | None
    list_price: float | None = None
    currency: str = "DKK"
    price_status: str = "confirmed"
    seller_type: str = "retailer"
    is_club_price: bool = False
    show_discount: bool = False


@dataclass
class TemporalData:
    """Historical price data for temporal features (legacy).
    
    New code should use ProductTemporalCache directly.
    This class is kept for backward compatibility.
    
    Attributes:
        previous_price: Last observed price.
        rolling_mean: Rolling average (e.g., 7-day).
        rolling_std: Rolling standard deviation.
        rolling_min: Rolling minimum.
        rolling_max: Rolling maximum.
        observation_count: Number of historical observations.
        rolling_median: Rolling median (robust).
        rolling_mad: Median Absolute Deviation (robust).
        percentile_5: 5th percentile.
        percentile_95: 95th percentile.
        last_change_at: When price last changed.
        consecutive_unchanged: Runs with same price.
    """
    
    previous_price: float | None = None
    rolling_mean: float | None = None
    rolling_std: float | None = None
    rolling_min: float | None = None
    rolling_max: float | None = None
    observation_count: int = 0
    
    # New robust stats
    rolling_median: float | None = None
    rolling_mad: float | None = None
    percentile_5: float | None = None
    percentile_95: float | None = None
    
    # Change tracking
    last_change_at: datetime | None = None
    consecutive_unchanged: int = 0
    
    @property
    def has_history(self) -> bool:
        """Whether we have enough history for temporal features."""
        return self.observation_count >= 3 and self.rolling_mean is not None


@dataclass 
class ContextData:
    """Contextual data from related tables.
    
    Attributes:
        has_promotion: Whether product has active promotion.
        discount_amount: Promotion discount amount.
        stock_status: Stock status (in_stock, out_of_stock, etc.).
        is_outlet: Whether this is an outlet product.
        item_condition: Product condition (new, refurbished, etc.).
        category_median_price: Median price in same category.
    """
    
    has_promotion: bool = False
    discount_amount: float | None = None
    stock_status: str = "in_stock"
    is_outlet: bool = False
    item_condition: str = "new"
    category_median_price: float | None = None


# MAD scaling factor for normal distribution equivalence
MAD_SCALE_FACTOR = 1.4826

# Threshold for "persistent" change (consecutive observations)
PERSISTENT_CHANGE_THRESHOLD = 2


class PriceFeatureExtractor(FeatureExtractor):
    """Extract features for price anomaly detection (Layer 2).
    
    Features (22 total):
        Core Price (4):
        - price: Current price value
        - price_log: Log of price (handles scale differences)
        - price_ratio: price / list_price (discount depth)
        - price_change_pct: Change from previous observation
        
        Standard Statistics (3):
        - price_zscore: Standard deviations from rolling mean
        - price_vs_mean_ratio: Current price / rolling mean
        - price_range_position: Position within rolling min/max range
        
        Robust Statistics (4):
        - rolling_median: Median of historical prices
        - rolling_mad: Median Absolute Deviation
        - robust_zscore: (price - median) / (1.4826 * MAD)
        - price_percentile_position: Position in p5-p95 range
        
        Change Tracking (3):
        - days_since_change: Days since last price change
        - consecutive_unchanged: Number of runs with same price
        - is_persistent_change: 1 if same new price for >= 2 observations
        
        Context (8):
        - price_vs_category: Ratio to category median
        - seller_type_encoded: 0=retailer, 1=marketplace_1p, 2=marketplace_3p
        - is_club_price: Binary club price indicator
        - has_promotion: Binary promotion indicator
        - expected_discount: Expected discount based on promotion
        - stock_status_encoded: 0=in_stock, 1=limited, 2=out_of_stock, 3=other
        - is_outlet: Binary outlet indicator
        - condition_encoded: 0=new, 1=refurbished, 2=used, 3=other
    """
    
    FEATURE_NAMES = [
        # Core price features
        "price",
        "price_log",
        "price_ratio",
        "price_change_pct",
        # Standard statistics
        "price_zscore",
        "price_vs_mean_ratio",
        "price_range_position",
        # Robust statistics
        "rolling_median",
        "rolling_mad",
        "robust_zscore",
        "price_percentile_position",
        # Change tracking
        "days_since_change",
        "consecutive_unchanged",
        "is_persistent_change",
        # Context
        "price_vs_category",
        "seller_type_encoded",
        "is_club_price",
        "has_promotion",
        "expected_discount",
        "stock_status_encoded",
        "is_outlet",
        "condition_encoded",
    ]
    
    SELLER_TYPE_MAP = {
        "retailer": 0,
        "marketplace_first_party": 1,
        "marketplace_third_party": 2,
        "unknown": 0,
    }
    
    STOCK_STATUS_MAP = {
        "in_stock": 0,
        "limited": 1,
        "out_of_stock": 2,
        "preorder": 3,
        "backorder": 3,
        "unknown": 3,
    }
    
    CONDITION_MAP = {
        "new": 0,
        "refurbished": 1,
        "used": 2,
        "open_box": 1,
        "demo": 1,
        "outlet": 1,
        "unknown": 0,
    }
    
    @property
    def name(self) -> str:
        return "price_features"
    
    @property
    def layer(self) -> str:
        return "price_anomaly"
    
    @property
    def feature_names(self) -> list[str]:
        return self.FEATURE_NAMES.copy()
    
    def extract(
        self,
        product_id: int,
        competitor_id: str,
        raw_record_id: int | None = None,
        scrape_run_id: str | None = None,
        price_data: PriceData | None = None,
        temporal_cache: "ProductTemporalCache | None" = None,
        temporal_data: TemporalData | None = None,
        context_data: ContextData | None = None,
        current_timestamp: datetime | None = None,
        **kwargs: Any,
    ) -> FeatureVector:
        """Extract price features for a single product.
        
        Args:
            product_id: Product identifier.
            competitor_id: Competitor identifier.
            raw_record_id: Optional FK to raw_scrape_records.
            scrape_run_id: Optional FK to scrape_runs.
            price_data: Current price observation.
            temporal_cache: ProductTemporalCache from TemporalCacheManager (preferred).
            temporal_data: Legacy TemporalData (for backward compatibility).
            context_data: Contextual data.
            current_timestamp: Current time for days_since_change computation.
            **kwargs: Additional arguments (ignored).
            
        Returns:
            FeatureVector with price features.
        """
        # Use defaults if not provided
        price_data = price_data or PriceData(price=None)
        context_data = context_data or ContextData()
        current_timestamp = current_timestamp or datetime.now(timezone.utc)
        
        # Convert temporal_cache to temporal_data if provided
        if temporal_cache is not None and temporal_data is None:
            temporal_data = self._cache_to_temporal_data(temporal_cache)
        elif temporal_data is None:
            temporal_data = TemporalData()
        
        features: dict[str, float | None] = {}
        missing_features: list[str] = []
        is_valid = True
        
        # ===== Core price features =====
        price = price_data.price
        
        if price is None or price <= 0:
            is_valid = False
            missing_features.append("price")
            features["price"] = None
            features["price_log"] = None
        else:
            features["price"] = price
            features["price_log"] = math.log(price) if price > 0 else None
        
        # Price ratio (discount depth)
        if price and price_data.list_price and price_data.list_price > 0:
            features["price_ratio"] = price / price_data.list_price
        else:
            features["price_ratio"] = 1.0  # No discount
            if price_data.list_price is None:
                missing_features.append("price_ratio")
        
        # Price change from previous
        if temporal_data.has_history and price:
            features["price_change_pct"] = safe_pct_change(
                price, temporal_data.previous_price, default=0.0
            )
        else:
            features["price_change_pct"] = 0.0
            if not temporal_data.has_history:
                missing_features.append("price_change_pct")
        
        # ===== Standard statistics =====
        if temporal_data.has_history and price:
            # Z-score
            if temporal_data.rolling_std and temporal_data.rolling_std > 0:
                features["price_zscore"] = (
                    (price - temporal_data.rolling_mean) / temporal_data.rolling_std
                )
            else:
                features["price_zscore"] = 0.0
            
            # Ratio to mean
            features["price_vs_mean_ratio"] = safe_ratio(
                price, temporal_data.rolling_mean, default=1.0
            )
            
            # Position in range
            if temporal_data.rolling_max and temporal_data.rolling_min:
                price_range = temporal_data.rolling_max - temporal_data.rolling_min
                if price_range > 0:
                    features["price_range_position"] = (
                        (price - temporal_data.rolling_min) / price_range
                    )
                else:
                    features["price_range_position"] = 0.5
            else:
                features["price_range_position"] = 0.5
        else:
            # No history - use defaults
            features["price_zscore"] = 0.0
            features["price_vs_mean_ratio"] = 1.0
            features["price_range_position"] = 0.5
            if not temporal_data.has_history:
                missing_features.extend([
                    "price_zscore", "price_vs_mean_ratio", "price_range_position"
                ])
        
        # ===== Robust statistics =====
        features["rolling_median"] = temporal_data.rolling_median
        features["rolling_mad"] = temporal_data.rolling_mad
        
        # Robust z-score
        if price and temporal_data.rolling_median is not None:
            if temporal_data.rolling_mad and temporal_data.rolling_mad > 0:
                features["robust_zscore"] = (
                    (price - temporal_data.rolling_median) / 
                    (MAD_SCALE_FACTOR * temporal_data.rolling_mad)
                )
            else:
                # No variation - return 0 if at median, else large value
                features["robust_zscore"] = (
                    0.0 if price == temporal_data.rolling_median else 10.0
                )
        else:
            features["robust_zscore"] = 0.0
            if temporal_data.rolling_median is None:
                missing_features.append("robust_zscore")
        
        # Percentile position
        if price and temporal_data.percentile_5 is not None and temporal_data.percentile_95 is not None:
            range_size = temporal_data.percentile_95 - temporal_data.percentile_5
            if range_size > 0:
                position = (price - temporal_data.percentile_5) / range_size
                features["price_percentile_position"] = max(0.0, min(1.0, position))
            else:
                features["price_percentile_position"] = 0.5
        else:
            features["price_percentile_position"] = 0.5
            if temporal_data.percentile_5 is None:
                missing_features.append("price_percentile_position")
        
        # ===== Change tracking =====
        # Days since change
        if temporal_data.last_change_at:
            delta = current_timestamp - temporal_data.last_change_at
            features["days_since_change"] = delta.total_seconds() / 86400
        else:
            features["days_since_change"] = None
            missing_features.append("days_since_change")
        
        # Consecutive unchanged
        features["consecutive_unchanged"] = float(temporal_data.consecutive_unchanged)
        
        # Is persistent change (same new price for >= 2 observations)
        # This is 1 if we have >= 2 observations of the same price after a change
        # We infer this from consecutive_unchanged: if unchanged >= 1, the current price
        # has persisted for at least 2 observations
        features["is_persistent_change"] = (
            1.0 if temporal_data.consecutive_unchanged >= PERSISTENT_CHANGE_THRESHOLD - 1 else 0.0
        )
        
        # ===== Context features =====
        # Category comparison
        if price and context_data.category_median_price:
            features["price_vs_category"] = safe_ratio(
                price, context_data.category_median_price, default=1.0
            )
        else:
            features["price_vs_category"] = 1.0
            if context_data.category_median_price is None:
                missing_features.append("price_vs_category")
        
        # Seller type encoding
        features["seller_type_encoded"] = float(
            self.SELLER_TYPE_MAP.get(price_data.seller_type.lower(), 0)
        )
        
        # Club price
        features["is_club_price"] = 1.0 if price_data.is_club_price else 0.0
        
        # Promotion features
        features["has_promotion"] = 1.0 if context_data.has_promotion else 0.0
        
        # Expected discount (if promotion exists)
        if context_data.has_promotion and context_data.discount_amount and price:
            features["expected_discount"] = context_data.discount_amount / price
        else:
            features["expected_discount"] = 0.0
        
        # Stock status encoding
        features["stock_status_encoded"] = float(
            self.STOCK_STATUS_MAP.get(context_data.stock_status.lower(), 3)
        )
        
        # Outlet flag
        features["is_outlet"] = 1.0 if context_data.is_outlet else 0.0
        
        # Condition encoding
        features["condition_encoded"] = float(
            self.CONDITION_MAP.get(context_data.item_condition.lower(), 0)
        )
        
        return FeatureVector(
            features=features,
            product_id=product_id,
            competitor_id=competitor_id,
            raw_record_id=raw_record_id,
            scrape_run_id=scrape_run_id,
            is_valid=is_valid,
            missing_features=missing_features,
            metadata={
                "has_temporal_history": temporal_data.has_history,
                "observation_count": temporal_data.observation_count,
                "price_status": price_data.price_status,
                "has_robust_stats": temporal_data.rolling_median is not None,
            },
        )
    
    def _cache_to_temporal_data(
        self,
        cache: "ProductTemporalCache",
    ) -> TemporalData:
        """Convert ProductTemporalCache to TemporalData for extraction.
        
        Args:
            cache: ProductTemporalCache from TemporalCacheManager.
            
        Returns:
            TemporalData populated from cache.
        """
        # Get previous price from history
        previous_price = None
        if len(cache.price_history) > 1:
            previous_price = float(cache.price_history[-2])
        elif len(cache.price_history) == 1:
            previous_price = float(cache.price_history[-1])
        
        return TemporalData(
            previous_price=previous_price,
            rolling_mean=cache.mean,
            rolling_std=cache.std,
            rolling_min=cache.min_price,
            rolling_max=cache.max_price,
            observation_count=cache.observation_count,
            rolling_median=cache.median,
            rolling_mad=cache.mad,
            percentile_5=cache.percentile_5,
            percentile_95=cache.percentile_95,
            last_change_at=cache.last_change_at,
            consecutive_unchanged=cache.consecutive_unchanged,
        )