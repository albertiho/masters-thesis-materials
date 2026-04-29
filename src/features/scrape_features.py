"""Scrape Feature Extractor - Extract features for scrape bug detection (Layer 1).

This extractor produces content-focused features to detect parser failures,
CSS changes, site redesigns, and other scraper bugs.

Usage:
    extractor = ScrapeFeatureExtractor()
    
    features = extractor.extract(
        product_id=123,
        competitor_id="PROSHOP_DK",
        content_data=ContentData(...),
        previous_content=ContentData(...),
        run_data=RunData(...),
    )
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from src.features.base import (
    FeatureExtractor,
    FeatureVector,
    safe_divide,
)

logger = logging.getLogger(__name__)


@dataclass
class ContentData:
    """Content observation from product_attributes and related tables.
    
    Attributes:
        title: Product title/name.
        description: Product description.
        brand: Product brand.
        ean: EAN identifier.
        mpn: MPN identifier.
        bullet_count: Number of bullet points.
        image_count: Number of images.
        price_status: Price status (confirmed, hidden, etc.).
    """
    
    title: str | None = None
    description: str | None = None
    brand: str | None = None
    ean: str | None = None
    mpn: str | None = None
    bullet_count: int = 0
    image_count: int = 0
    price_status: str = "confirmed"


@dataclass
class RunData:
    """Run-level statistics from scrape_runs table.
    
    Attributes:
        record_count: Total records in run.
        rejected_count: Records rejected during processing.
        products_missing_pct: Percentage of products missing vs previous run.
    """
    
    record_count: int = 0
    rejected_count: int = 0
    products_missing_pct: float = 0.0
    
    @property
    def parse_error_rate(self) -> float:
        """Calculate parse error rate."""
        if self.record_count == 0:
            return 0.0
        return self.rejected_count / self.record_count


class ScrapeFeatureExtractor(FeatureExtractor):
    """Extract features for scrape bug detection (Layer 1).
    
    Features focus on content stability - things that shouldn't change
    between scrapes unless there's a bug.
    
    Features:
        - title_length: Current title length
        - title_length_ratio: Current / previous title length
        - title_changed: Binary - did title change?
        - title_collapsed: Binary - title significantly shorter?
        - description_present: Binary - has description?
        - description_disappeared: Binary - had description, now gone?
        - brand_changed: Binary - brand changed (shouldn't happen)?
        - identifier_changed: Binary - EAN or MPN changed?
        - bullet_count_change: Change in bullet point count
        - image_count_change: Change in image count
        - price_status_changed: Binary - status changed to hidden?
        - run_parse_error_rate: Run-level error rate
        - run_products_missing_pct: Run-level missing products rate
    """
    
    FEATURE_NAMES = [
        "title_length",
        "title_length_ratio",
        "title_changed",
        "title_collapsed",
        "description_present",
        "description_disappeared",
        "brand_changed",
        "identifier_changed",
        "bullet_count_change",
        "image_count_change",
        "price_status_changed",
        "run_parse_error_rate",
        "run_products_missing_pct",
    ]
    
    # Threshold for title collapse detection
    TITLE_COLLAPSE_THRESHOLD = 0.3  # If title is <30% of previous, it's collapsed
    
    @property
    def name(self) -> str:
        return "scrape_features"
    
    @property
    def layer(self) -> str:
        return "scrape_bug"
    
    @property
    def feature_names(self) -> list[str]:
        return self.FEATURE_NAMES.copy()
    
    def extract(
        self,
        product_id: int,
        competitor_id: str,
        raw_record_id: int | None = None,
        scrape_run_id: str | None = None,
        content_data: ContentData | None = None,
        previous_content: ContentData | None = None,
        run_data: RunData | None = None,
        **kwargs: Any,
    ) -> FeatureVector:
        """Extract scrape bug features for a single product.
        
        Args:
            product_id: Product identifier.
            competitor_id: Competitor identifier.
            raw_record_id: Optional FK to raw_scrape_records.
            scrape_run_id: Optional FK to scrape_runs.
            content_data: Current content observation.
            previous_content: Previous content observation (for comparison).
            run_data: Run-level statistics.
            **kwargs: Additional arguments (ignored).
            
        Returns:
            FeatureVector with scrape bug features.
        """
        # Use defaults if not provided
        content_data = content_data or ContentData()
        previous_content = previous_content or ContentData()
        run_data = run_data or RunData()
        
        features: dict[str, float | None] = {}
        missing_features: list[str] = []
        is_valid = True
        
        # Title features
        current_title_len = len(content_data.title) if content_data.title else 0
        previous_title_len = len(previous_content.title) if previous_content.title else 0
        
        features["title_length"] = float(current_title_len)
        
        # Title length ratio
        if previous_title_len > 0:
            features["title_length_ratio"] = current_title_len / previous_title_len
        else:
            features["title_length_ratio"] = 1.0
            if previous_content.title is None:
                missing_features.append("title_length_ratio")
        
        # Title changed
        if content_data.title and previous_content.title:
            features["title_changed"] = 0.0 if content_data.title == previous_content.title else 1.0
        else:
            features["title_changed"] = 0.0
        
        # Title collapsed (significantly shorter)
        if previous_title_len > 0:
            ratio = current_title_len / previous_title_len
            features["title_collapsed"] = 1.0 if ratio < self.TITLE_COLLAPSE_THRESHOLD else 0.0
        else:
            features["title_collapsed"] = 0.0
        
        # Description features
        has_description = bool(content_data.description and len(content_data.description) > 0)
        had_description = bool(previous_content.description and len(previous_content.description) > 0)
        
        features["description_present"] = 1.0 if has_description else 0.0
        features["description_disappeared"] = 1.0 if (had_description and not has_description) else 0.0
        
        # Brand changed (shouldn't happen for same product)
        if content_data.brand and previous_content.brand:
            features["brand_changed"] = 0.0 if content_data.brand == previous_content.brand else 1.0
        else:
            features["brand_changed"] = 0.0
        
        # Identifier changed (EAN/MPN shouldn't change)
        ean_changed = (
            content_data.ean and previous_content.ean and 
            content_data.ean != previous_content.ean
        )
        mpn_changed = (
            content_data.mpn and previous_content.mpn and
            content_data.mpn != previous_content.mpn
        )
        features["identifier_changed"] = 1.0 if (ean_changed or mpn_changed) else 0.0
        
        # Bullet count change
        if previous_content.bullet_count > 0:
            features["bullet_count_change"] = (
                content_data.bullet_count - previous_content.bullet_count
            ) / previous_content.bullet_count
        else:
            features["bullet_count_change"] = 0.0
        
        # Image count change
        if previous_content.image_count > 0:
            features["image_count_change"] = (
                content_data.image_count - previous_content.image_count
            ) / previous_content.image_count
        else:
            features["image_count_change"] = 0.0
        
        # Price status changed to hidden (might indicate scraper issue)
        status_to_hidden = (
            previous_content.price_status == "confirmed" and
            content_data.price_status == "hidden"
        )
        features["price_status_changed"] = 1.0 if status_to_hidden else 0.0
        
        # Run-level features
        features["run_parse_error_rate"] = run_data.parse_error_rate
        features["run_products_missing_pct"] = run_data.products_missing_pct
        
        # Determine validity - if we're missing previous content, features are less reliable
        if previous_content.title is None:
            missing_features.extend([
                "title_changed", "title_collapsed", "brand_changed", 
                "identifier_changed", "bullet_count_change", "image_count_change"
            ])
        
        return FeatureVector(
            features=features,
            product_id=product_id,
            competitor_id=competitor_id,
            raw_record_id=raw_record_id,
            scrape_run_id=scrape_run_id,
            is_valid=is_valid,
            missing_features=missing_features,
            metadata={
                "has_previous_content": previous_content.title is not None,
                "current_title_length": current_title_len,
                "previous_title_length": previous_title_len,
                "run_record_count": run_data.record_count,
            },
        )
