"""Coherence Feature Extractor - Features for scrape bug vs real move classification.

Extracts features that help distinguish scraper failures from genuine market changes.
These features capture the "coherence" of a data change - whether it makes sense
as a real market event or looks like a data artifact.

Key insight: Real price changes rarely come with content degradation.
When price changes AND content breaks, it's usually a scraper bug.

Run-level features:
- Missing field rates
- Parse error rates
- Distribution drift

Product-level features:
- Price changed
- Content changed/degraded
- Change persistence across runs
- Cross-competitor agreement
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.ingestion.parser import ParseResult, ProductRecord
from src.quality.run_health import RunHealth

logger = logging.getLogger(__name__)


@dataclass
class RunLevelFeatures:
    """Run-level features aggregated across all products in a scrape run."""

    # Identifiers
    run_id: str
    competitor: str
    country: str

    # Missing field rates
    missing_title_rate: float
    missing_image_rate: float
    missing_price_rate: float
    missing_ean_rate: float

    # Parse metrics
    parse_error_rate: float
    total_records: int
    parsed_records: int

    # Distribution metrics
    price_mean: float | None
    price_std: float | None

    # Drift metrics (vs historical)
    row_count_drift: float | None
    price_distribution_drift: float | None

    # Health score
    health_score: float
    is_healthy: bool

    @classmethod
    def from_run_health(cls, run_health: RunHealth) -> "RunLevelFeatures":
        """Create from RunHealth object."""
        return cls(
            run_id=run_health.run_id,
            competitor=run_health.competitor,
            country=run_health.country,
            missing_title_rate=run_health.missing_title_rate,
            missing_image_rate=run_health.missing_image_rate,
            missing_price_rate=run_health.missing_price_rate,
            missing_ean_rate=run_health.missing_ean_rate,
            parse_error_rate=run_health.parse_error_rate,
            total_records=run_health.total_records,
            parsed_records=run_health.parsed_records,
            price_mean=run_health.price_mean,
            price_std=run_health.price_std,
            row_count_drift=run_health.row_count_drift_pct,
            price_distribution_drift=run_health.price_distribution_drift,
            health_score=run_health.health_score,
            is_healthy=run_health.is_healthy,
        )


@dataclass
class ProductLevelFeatures:
    """Product-level features for scrape-bug classification."""

    # Identifiers
    competitor_product_id: str
    competitor: str
    country: str | None

    # Price change features
    price_changed: bool
    price_change_pct: float | None
    price_change_direction: str | None  # "increase", "decrease", None

    # Content change features
    title_changed: bool
    title_similarity: float | None  # 0-1, 1 = identical
    content_degraded: bool  # Title shortened, fields missing, etc.

    # Persistence features
    change_persisted_runs: int  # How many consecutive runs with this change
    first_seen_at: str | None  # Timestamp when change first appeared

    # Cross-competitor features
    cross_competitor_agreement: float | None  # % of other competitors with similar change
    competitors_with_similar_change: list[str] = field(default_factory=list)

    # Anomaly features (from detectors)
    price_anomaly_score: float | None = None
    is_price_anomaly: bool = False

    # Run context
    run_health_score: float | None = None
    run_is_healthy: bool | None = None

    def to_feature_vector(self) -> np.ndarray:
        """Convert to numeric feature vector for ML classifier."""
        features = [
            float(self.price_changed),
            self.price_change_pct or 0.0,
            float(self.title_changed),
            self.title_similarity or 0.0,
            float(self.content_degraded),
            float(self.change_persisted_runs),
            self.cross_competitor_agreement or 0.0,
            self.price_anomaly_score or 0.0,
            float(self.is_price_anomaly),
            self.run_health_score or 0.0,
            float(self.run_is_healthy) if self.run_is_healthy is not None else 1.0,
        ]
        return np.array(features, dtype=float)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            "competitor_product_id": self.competitor_product_id,
            "competitor": self.competitor,
            "country": self.country,
            "price_changed": self.price_changed,
            "price_change_pct": self.price_change_pct,
            "title_changed": self.title_changed,
            "title_similarity": self.title_similarity,
            "content_degraded": self.content_degraded,
            "change_persisted_runs": self.change_persisted_runs,
            "cross_competitor_agreement": self.cross_competitor_agreement,
            "price_anomaly_score": self.price_anomaly_score,
            "is_price_anomaly": self.is_price_anomaly,
            "run_health_score": self.run_health_score,
            "run_is_healthy": self.run_is_healthy,
        }


@dataclass
class PreviousObservation:
    """Previous observation for a product (for comparison)."""

    price: float
    title: str | None
    scraped_at: str
    run_id: str | None


class CoherenceFeatureExtractor:
    """Extract coherence features for scrape-bug classification.

    Usage:
        extractor = CoherenceFeatureExtractor()

        # Extract features for a product
        features = extractor.extract_product_features(
            record=current_record,
            previous=previous_observation,
            run_health=current_run_health,
        )

        # Use features in classifier
        if features.content_degraded and features.price_changed:
            # Likely scrape bug
            pass
    """

    def __init__(
        self,
        price_change_threshold: float = 0.01,  # 1% to consider as "changed"
        title_similarity_threshold: float = 0.8,  # Below this = "changed"
        title_length_degradation_ratio: float = 0.5,  # Below this = "degraded"
    ):
        """Initialize the extractor.

        Args:
            price_change_threshold: Minimum % change to consider price as "changed".
            title_similarity_threshold: Below this, title is considered "changed".
            title_length_degradation_ratio: Below this ratio, title is "degraded".
        """
        self.price_change_threshold = price_change_threshold
        self.title_similarity_threshold = title_similarity_threshold
        self.title_length_degradation_ratio = title_length_degradation_ratio

    def extract_product_features(
        self,
        record: ProductRecord,
        previous: PreviousObservation | None = None,
        run_health: RunHealth | None = None,
        anomaly_score: float | None = None,
        is_anomaly: bool = False,
        persistence_runs: int = 0,
        cross_competitor_pct: float | None = None,
    ) -> ProductLevelFeatures:
        """Extract features for a single product.

        Args:
            record: Current product record.
            previous: Previous observation for comparison.
            run_health: Run health metrics.
            anomaly_score: Anomaly score from detector.
            is_anomaly: Whether anomaly was flagged.
            persistence_runs: Number of runs this change has persisted.
            cross_competitor_pct: % of other competitors with similar change.

        Returns:
            ProductLevelFeatures with extracted features.
        """
        # Price change features
        price_changed = False
        price_change_pct = None
        price_change_direction = None

        if previous is not None and record.price and previous.price:
            price_change_pct = (record.price - previous.price) / previous.price
            price_changed = abs(price_change_pct) > self.price_change_threshold
            if price_changed:
                price_change_direction = "increase" if price_change_pct > 0 else "decrease"

        # Title change features
        title_changed = False
        title_similarity = None
        content_degraded = False

        if previous is not None and previous.title and record.product_name:
            title_similarity = self._calculate_title_similarity(record.product_name, previous.title)
            title_changed = title_similarity < self.title_similarity_threshold

            # Check for degradation (title got shorter)
            if len(record.product_name) < len(previous.title) * self.title_length_degradation_ratio:
                content_degraded = True
        elif previous is not None and previous.title and not record.product_name:
            # Title disappeared entirely
            title_changed = True
            content_degraded = True
            title_similarity = 0.0

        # Additional content degradation checks
        if record.raw_data:
            # Check if key fields are missing that were previously present
            if not record.raw_data.get("images") and not record.raw_data.get("image_url"):
                # Could indicate degradation if images were present before
                # (would need historical comparison)
                pass

        return ProductLevelFeatures(
            competitor_product_id=record.competitor_product_id,
            competitor=record.competitor,
            country=record.country,
            price_changed=price_changed,
            price_change_pct=price_change_pct,
            price_change_direction=price_change_direction,
            title_changed=title_changed,
            title_similarity=title_similarity,
            content_degraded=content_degraded,
            change_persisted_runs=persistence_runs,
            first_seen_at=previous.scraped_at if previous else None,
            cross_competitor_agreement=cross_competitor_pct,
            price_anomaly_score=anomaly_score,
            is_price_anomaly=is_anomaly,
            run_health_score=run_health.health_score if run_health else None,
            run_is_healthy=run_health.is_healthy if run_health else None,
        )

    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two titles.

        Uses character-level Jaccard similarity for simplicity.
        Could be replaced with edit distance or embedding similarity.
        """
        if not title1 or not title2:
            return 0.0

        # Normalize
        t1 = title1.lower().strip()
        t2 = title2.lower().strip()

        if t1 == t2:
            return 1.0

        # Character n-gram similarity (n=3)
        def ngrams(s: str, n: int = 3) -> set[str]:
            return set(s[i : i + n] for i in range(len(s) - n + 1))

        ng1 = ngrams(t1)
        ng2 = ngrams(t2)

        if not ng1 or not ng2:
            # Very short strings - use character set
            ng1 = set(t1)
            ng2 = set(t2)

        intersection = len(ng1 & ng2)
        union = len(ng1 | ng2)

        return intersection / union if union > 0 else 0.0

    def extract_batch(
        self,
        records: list[ProductRecord],
        previous_map: dict[str, PreviousObservation],
        run_health: RunHealth | None = None,
        anomaly_scores: dict[str, float] | None = None,
        anomaly_flags: dict[str, bool] | None = None,
    ) -> list[ProductLevelFeatures]:
        """Extract features for a batch of products.

        Args:
            records: List of current product records.
            previous_map: Map of competitor_product_id -> previous observation.
            run_health: Run health metrics.
            anomaly_scores: Map of competitor_product_id -> anomaly score.
            anomaly_flags: Map of competitor_product_id -> is_anomaly.

        Returns:
            List of ProductLevelFeatures.
        """
        features = []
        for record in records:
            prev = previous_map.get(record.competitor_product_id)
            score = anomaly_scores.get(record.competitor_product_id) if anomaly_scores else None
            flag = (
                anomaly_flags.get(record.competitor_product_id, False) if anomaly_flags else False
            )

            feat = self.extract_product_features(
                record=record,
                previous=prev,
                run_health=run_health,
                anomaly_score=score,
                is_anomaly=flag,
            )
            features.append(feat)

        return features


def calculate_cross_competitor_agreement(
    product_changes: dict[str, dict[str, float]],
    ean: str,
    exclude_competitor: str,
) -> float | None:
    """Calculate what fraction of competitors show similar price changes for a product.

    Args:
        product_changes: Map of EAN -> {competitor -> price_change_pct}.
        ean: EAN of the product to check.
        exclude_competitor: Competitor to exclude from calculation.

    Returns:
        Fraction of other competitors with similar (>5%) price change, or None if no data.
    """
    if not ean or ean not in product_changes:
        return None

    changes = product_changes[ean]
    other_competitors = [c for c in changes.keys() if c != exclude_competitor]

    if not other_competitors:
        return None

    # Count competitors with similar price change direction and magnitude
    similar_count = 0
    for comp in other_competitors:
        change_pct = changes.get(comp, 0)
        target_change = changes.get(exclude_competitor, 0)

        # Similar if same direction and within 2x magnitude
        if target_change * change_pct > 0:  # Same sign
            ratio = min(abs(target_change), abs(change_pct)) / max(
                abs(target_change), abs(change_pct), 0.001
            )
            if ratio > 0.5:  # Within 2x
                similar_count += 1

    return similar_count / len(other_competitors)
