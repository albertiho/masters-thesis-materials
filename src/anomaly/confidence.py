"""Confidence Aggregator - Compute confidence from multi-detector votes.

This module aggregates votes from multiple anomaly detectors into a single
confidence score. Higher agreement between detectors = higher confidence.

Usage:
    aggregator = ConfidenceAggregator()
    
    aggregator.add_vote("zscore", layer="price_anomaly", is_flagged=True, score=0.8)
    aggregator.add_vote("iqr", layer="price_anomaly", is_flagged=True, score=0.7)
    aggregator.add_vote("isolation_forest", layer="price_anomaly", is_flagged=False, score=0.3)
    
    confidence = aggregator.get_confidence("price_anomaly")
    # Returns 0.67 (2/3 detectors flagged)

Note:
    Anomaly decisions (based on confidence) are stored in anomaly_alerts.is_anomaly
    (True=excluded, False=normal). Reason codes explaining anomalies are in reason_codes.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DetectorVote:
    """A single detector's vote.
    
    Attributes:
        detector_name: Name of the detector.
        layer: Detection layer (scrape_bug, price_anomaly).
        is_flagged: Whether detector flagged as anomaly.
        score: Detector's internal score (0-1).
    """
    
    detector_name: str
    layer: str
    is_flagged: bool
    score: float | None = None


@dataclass
class AggregatedConfidence:
    """Aggregated confidence for a record.
    
    Attributes:
        raw_record_id: FK to raw_scrape_records.
        product_id: FK to products table.
        competitor_id: Competitor identifier.
        layer: Detection layer.
        total_detectors: Number of detectors that voted.
        flagged_count: Number of detectors that flagged.
        confidence: Ratio of flagged/total (0-1).
        detector_scores: Individual detector scores.
    """
    
    raw_record_id: int
    product_id: int
    competitor_id: str
    layer: str
    total_detectors: int
    flagged_count: int
    confidence: float
    detector_scores: dict[str, float] = field(default_factory=dict)
    
    @property
    def is_anomaly(self) -> bool:
        """Whether majority of detectors flagged as anomaly."""
        return self.confidence > 0.5


class ConfidenceAggregator:
    """Aggregate detector votes into confidence scores.
    
    In-memory aggregation: Add votes one by one, get confidence.
    Anomaly decisions are stored in anomaly_alerts.is_anomaly and reason_codes.
    """
    
    def __init__(self) -> None:
        """Initialize the aggregator."""
        # In-memory votes: key = (raw_record_id, product_id, layer)
        self._votes: dict[tuple[int, int, str], list[DetectorVote]] = {}
        
    def add_vote(
        self,
        detector_name: str,
        layer: str,
        is_flagged: bool,
        score: float | None = None,
        raw_record_id: int = 0,
        product_id: int = 0,
    ) -> None:
        """Add a detector vote for a record.
        
        Args:
            detector_name: Name of the detector.
            layer: Detection layer.
            is_flagged: Whether detector flagged as anomaly.
            score: Detector's internal score.
            raw_record_id: Record ID (default 0 for single-record aggregation).
            product_id: Product ID (default 0 for single-record aggregation).
        """
        key = (raw_record_id, product_id, layer)
        if key not in self._votes:
            self._votes[key] = []
        
        self._votes[key].append(DetectorVote(
            detector_name=detector_name,
            layer=layer,
            is_flagged=is_flagged,
            score=score,
        ))
    
    def get_confidence(
        self,
        layer: str,
        raw_record_id: int = 0,
        product_id: int = 0,
    ) -> float:
        """Get confidence for a specific layer.
        
        Args:
            layer: Detection layer to get confidence for.
            raw_record_id: Record ID.
            product_id: Product ID.
            
        Returns:
            Confidence score (0-1).
        """
        key = (raw_record_id, product_id, layer)
        votes = self._votes.get(key, [])
        
        if not votes:
            return 0.0
        
        flagged_count = sum(1 for v in votes if v.is_flagged)
        return flagged_count / len(votes)
    
    def get_aggregated(
        self,
        raw_record_id: int,
        product_id: int,
        competitor_id: str,
        layer: str,
    ) -> AggregatedConfidence:
        """Get full aggregated confidence for a record/layer.
        
        Args:
            raw_record_id: Record ID.
            product_id: Product ID.
            competitor_id: Competitor identifier.
            layer: Detection layer.
            
        Returns:
            AggregatedConfidence object.
        """
        key = (raw_record_id, product_id, layer)
        votes = self._votes.get(key, [])
        
        if not votes:
            return AggregatedConfidence(
                raw_record_id=raw_record_id,
                product_id=product_id,
                competitor_id=competitor_id,
                layer=layer,
                total_detectors=0,
                flagged_count=0,
                confidence=0.0,
            )
        
        flagged_count = sum(1 for v in votes if v.is_flagged)
        detector_scores = {
            v.detector_name: v.score for v in votes if v.score is not None
        }
        
        return AggregatedConfidence(
            raw_record_id=raw_record_id,
            product_id=product_id,
            competitor_id=competitor_id,
            layer=layer,
            total_detectors=len(votes),
            flagged_count=flagged_count,
            confidence=flagged_count / len(votes),
            detector_scores=detector_scores,
        )
    
    def get_all_aggregated(self, competitor_id: str) -> list[AggregatedConfidence]:
        """Get all aggregated confidences.
        
        Args:
            competitor_id: Competitor identifier (required for output).
            
        Returns:
            List of AggregatedConfidence objects.
        """
        results = []
        for (raw_record_id, product_id, layer), votes in self._votes.items():
            flagged_count = sum(1 for v in votes if v.is_flagged)
            detector_scores = {
                v.detector_name: v.score for v in votes if v.score is not None
            }
            
            results.append(AggregatedConfidence(
                raw_record_id=raw_record_id,
                product_id=product_id,
                competitor_id=competitor_id,
                layer=layer,
                total_detectors=len(votes),
                flagged_count=flagged_count,
                confidence=flagged_count / len(votes),
                detector_scores=detector_scores,
            ))
        
        return results
    
    def clear(self) -> None:
        """Clear all in-memory votes."""
        self._votes.clear()


def compute_weighted_confidence(
    votes: list[DetectorVote],
    weights: dict[str, float] | None = None,
) -> float:
    """Compute weighted confidence from votes.
    
    Allows giving more weight to certain detectors (e.g., ML detectors
    might be more reliable than simple statistical ones).
    
    Args:
        votes: List of detector votes.
        weights: Optional dict mapping detector_name to weight. Default weight is 1.0.
        
    Returns:
        Weighted confidence score (0-1).
    """
    if not votes:
        return 0.0
    
    weights = weights or {}
    
    total_weight = 0.0
    flagged_weight = 0.0
    
    for vote in votes:
        weight = weights.get(vote.detector_name, 1.0)
        total_weight += weight
        if vote.is_flagged:
            flagged_weight += weight
    
    if total_weight == 0:
        return 0.0
    
    return flagged_weight / total_weight
