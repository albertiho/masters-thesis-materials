"""Feature Engineering Module for ML Anomaly Detection.

Purpose:
    Extract ML-ready features from raw product records for anomaly detection.
    Supports numeric, temporal, categorical, text embedding, and coherence features.

Key Components:
    - base.py: Common FeatureExtractor interface and FeatureVector
    - price_features.py: Layer 2 price anomaly features (22 features including robust stats)
    - scrape_features.py: Layer 1 scrape bug detection features (~13 features)
    - numeric.py: Price and numeric feature extraction
    - temporal.py: TemporalCacheManager with local on-demand cache initialization and robust statistics
    - coherence.py: Features for scrape-bug vs real-move classification
    - embeddings.py: Text embeddings for titles, descriptions, specs

Feature Layers:
    Layer 1 (Scrape Bug Detection): Content stability features
        - Title/description changes, identifier stability, run-level metrics
        - Use ScrapeFeatureExtractor for consistent extraction
        
    Layer 2 (Price Anomaly Detection): Price-focused features
        - Price ratios, temporal patterns, contextual signals
        - Use PriceFeatureExtractor for consistent extraction

Dependencies:
    - src/ingestion/parser.py (ProductRecord)
    - numpy, pandas for feature computation
    - sentence-transformers for text embeddings (optional)

Consumed by:
    - src/anomaly/ (all detection methods)
    - src/evaluation/ (method comparison)

Module TODOs:
    - [ ] Create categorical.py for competitor/category encoding
    - [ ] Create pipeline.py to orchestrate all feature extraction
    - [x] Integrate TemporalFeatureStore with deterministic local cache initialization
    - [x] Create base.py with FeatureExtractor interface
    - [x] Create price_features.py for Layer 2 detection
    - [x] Create scrape_features.py for Layer 1 detection
    - [x] Implement TemporalCacheManager with robust stats (2026-01-10)
    - [x] Add robust statistics (median, MAD, percentiles) to PriceFeatureExtractor
"""

from src.features.base import (
    FeatureExtractor,
    FeatureVector,
    safe_divide,
    safe_pct_change,
    safe_ratio,
)
from src.features.price_features import (
    PriceFeatureExtractor,
    PriceData,
    TemporalData,
    ContextData,
)
from src.features.scrape_features import (
    ScrapeFeatureExtractor,
    ContentData,
    RunData,
)
from src.features.numeric import (
    extract_numeric_features,
    extract_numeric_features_batch,
    NumericFeatures,
)
from src.features.temporal import (
    ProductTemporalCache,
    TemporalCacheManager,
    TemporalFeatures,
    TemporalFeatureStore,
    compute_rolling_statistics,
    recompute_stats,
)
from src.features.embeddings import (
    ContentDriftDetector,
    ContentDriftResult,
    RunContentDriftSummary,
    TextEmbeddings,
    TextEmbeddingExtractor,
    compute_embedding_drift,
)
from src.features.coherence import (
    CoherenceFeatureExtractor,
    ProductLevelFeatures,
    RunLevelFeatures,
    PreviousObservation,
    calculate_cross_competitor_agreement,
)

__all__ = [
    # Base feature classes
    "FeatureExtractor",
    "FeatureVector",
    "safe_divide",
    "safe_pct_change",
    "safe_ratio",
    # Price features (Layer 2)
    "PriceFeatureExtractor",
    "PriceData",
    "TemporalData",
    "ContextData",
    # Scrape features (Layer 1)
    "ScrapeFeatureExtractor",
    "ContentData",
    "RunData",
    # Numeric features
    "extract_numeric_features",
    "extract_numeric_features_batch",
    "NumericFeatures",
    # Temporal features
    "ProductTemporalCache",
    "TemporalCacheManager",
    "TemporalFeatures",
    "TemporalFeatureStore",
    "compute_rolling_statistics",
    "recompute_stats",
    # Text embeddings
    "TextEmbeddings",
    "TextEmbeddingExtractor",
    "compute_embedding_drift",
    # Content drift detection
    "ContentDriftDetector",
    "ContentDriftResult",
    "RunContentDriftSummary",
    # Coherence features
    "CoherenceFeatureExtractor",
    "ProductLevelFeatures",
    "RunLevelFeatures",
    "PreviousObservation",
    "calculate_cross_competitor_agreement",
]
