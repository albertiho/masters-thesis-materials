"""Anomaly Detection Module - ML/Statistical Detection.

Purpose: Detect both data quality issues and genuine market anomalies.
Supports multiple algorithm families for comparison.

Key Components:
- statistical.py: Classical methods (z-score, IQR, thresholds, sanity checks)
- ml/: ML methods (isolation forest, autoencoders, embeddings)
- persistence.py: Save/load trained models on the local filesystem
- confidence.py: Multi-detector confidence aggregation
- combined.py: Composable detector pipelines with history-based routing

Layered Detection Architecture:
    Layer 0: Data Quality Gate (deterministic, before ML)
    Layer 1: Scrape Bug Detection (content-focused features)
    Layer 2: Price Anomaly Detection (price-focused features)

CombinedDetector Framework:
    BaseCombinedDetector enables composable detector pipelines:
    - DetectorLayer: Pluggable layers with short-circuit support
    - DetectionContext: Shared state across layers
    - History-based routing (cold start vs warm vs full)
    - Configurable detector weights and order

Multi-Detector Confidence:
    ConfidenceAggregator computes confidence in-memory = flagged/total detectors.
    Anomaly decisions are stored in anomaly_alerts.is_anomaly (True=excluded, False=normal) and reason_codes.

Algorithm Families:
1. Statistical: Z-score, IQR, percentage thresholds, sanity checks
2. ML Tree-based: Isolation Forest
3. ML Neural: Autoencoders
4. Embedding-based: Text embedding drift detection

Dependencies: numpy, scikit-learn (optional: torch, sentence-transformers)
Consumed by: local training and evaluation scripts

Module TODOs:
    - [ ] Create detector.py with unified DetectorInterface base class
    - [ ] Implement embedding_drift.py for content change detection
    - [x] Add model persistence (save/load trained models)
    - [x] Create confidence.py for multi-detector voting
    - [x] Create combined.py for composable detector pipelines
"""

from src.anomaly.base import BaseDetector
from src.anomaly.batch_processor import BatchRoundProcessor, RowContext
from src.anomaly.change_tracker import (
    ChangePersistenceTracker,
    PersistenceInfo,
    PriceChange,
)
from src.anomaly.classifier import (
    ClassificationResult,
    ClassifierConfig,
    ScrapeIssueClassification,
    ScrapeIssueClassifier,
)
from src.anomaly.combined import (
    BaseCombinedDetector,
    CombinedDetector,
    CombinedDetectorConfig,
    DetectionContext,
    DetectorLayer,
)
from src.anomaly.combined_variants import (
    DefaultCombinedDetector,
    MinimalCombinedDetector,
    ProductionCombinedDetector,
    StatisticalOnlyCombinedDetector,
    create_combined_detector,
)
from src.anomaly.confidence import (
    AggregatedConfidence,
    ConfidenceAggregator,
    DetectorVote,
    compute_weighted_confidence,
)
from src.anomaly.persistence import (
    ModelMetadata,
    ModelPersistence,
    StatisticalConfig,
)
from src.anomaly.statistical import (
    AnomalyResult,
    AnomalySeverity,
    AnomalyType,
    HybridAvgZScoreDetector,
    HybridMaxZScoreDetector,
    HybridWeightedZScoreDetector,
    IQRDetector,
    InvariantContext,
    InvariantDetector,
    ModifiedMADDetector,
    ModifiedSNDetector,
    SanityCheckDetector,
    StatisticalEnsemble,
    ThresholdDetector,
    ZScoreDetector,
)
__all__ = [
    # Base class
    "BaseDetector",
    # Batch processing
    "BatchRoundProcessor",
    "RowContext",
    # Core types
    "AnomalyResult",
    "AnomalySeverity",
    "AnomalyType",
    # Statistical detectors
    "ZScoreDetector",
    "ModifiedMADDetector",
    "ModifiedSNDetector",
    "HybridWeightedZScoreDetector",
    "HybridMaxZScoreDetector",
    "HybridAvgZScoreDetector",
    "IQRDetector",
    "ThresholdDetector",
    "SanityCheckDetector",
    "StatisticalEnsemble",
    # Tier 0 invariant detector
    "InvariantDetector",
    "InvariantContext",
    # Scrape issue classifier
    "ScrapeIssueClassifier",
    "ScrapeIssueClassification",
    "ClassificationResult",
    "ClassifierConfig",
    # Combined detector framework
    "BaseCombinedDetector",
    "CombinedDetector",
    "CombinedDetectorConfig",
    "DetectionContext",
    "DetectorLayer",
    # Combined detector variants
    "ProductionCombinedDetector",
    "DefaultCombinedDetector",
    "StatisticalOnlyCombinedDetector",
    "MinimalCombinedDetector",
    "create_combined_detector",
    # Multi-detector confidence
    "ConfidenceAggregator",
    "AggregatedConfidence",
    "DetectorVote",
    "compute_weighted_confidence",
    # Change persistence tracking
    "ChangePersistenceTracker",
    "PersistenceInfo",
    "PriceChange",
    # Model persistence
    "ModelMetadata",
    "ModelPersistence",
    "StatisticalConfig",
]
