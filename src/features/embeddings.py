"""Text Embedding Feature Extraction for Content Drift Detection.

Extracts dense vector embeddings from text fields (title, description, specs)
using pre-trained language models. These embeddings enable:
    - Content drift detection (compare old vs new embeddings)
    - Semantic similarity for product matching
    - Clustering similar products/anomalies

Uses sentence-transformers library with pre-trained models.
Falls back gracefully if the library is not installed.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.ingestion.parser import ProductRecord

logger = logging.getLogger(__name__)

# Lazy import to avoid errors if not installed
_sentence_transformers_available = None
_model_cache: dict[str, Any] = {}


def _check_sentence_transformers():
    """Check if sentence-transformers is available."""
    global _sentence_transformers_available
    if _sentence_transformers_available is None:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401

            _sentence_transformers_available = True
        except ImportError:
            _sentence_transformers_available = False
    return _sentence_transformers_available


# Default model - multilingual for Nordic language support (DK, SE, NO, FI, EN)
DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # 384 dimensions, 50+ languages

# Alternative models for different use cases
AVAILABLE_MODELS = {
    "fast": "all-MiniLM-L6-v2",  # 384 dims, fastest, English-only
    "balanced": "all-mpnet-base-v2",  # 768 dims, good quality, English-only
    "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",  # 384 dims, 50+ languages (default)
}


@dataclass
class TextEmbeddings:
    """Container for text embeddings from a product record.

    Attributes:
        title_embedding: Embedding of the product title
        description_embedding: Embedding of the description (if available)
        combined_embedding: Combined embedding of all text fields
        embedding_dim: Dimension of the embedding vectors
        has_title: Whether title was available
        has_description: Whether description was available
        model_name: Name of the model used for embedding
    """

    title_embedding: np.ndarray | None
    description_embedding: np.ndarray | None
    combined_embedding: np.ndarray | None
    embedding_dim: int
    has_title: bool
    has_description: bool
    model_name: str

    # Identifiers
    competitor_product_id: str
    competitor: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (embeddings as lists for serialization)."""
        return {
            "competitor_product_id": self.competitor_product_id,
            "competitor": self.competitor,
            "title_embedding": (
                self.title_embedding.tolist() if self.title_embedding is not None else None
            ),
            "description_embedding": (
                self.description_embedding.tolist()
                if self.description_embedding is not None
                else None
            ),
            "combined_embedding": (
                self.combined_embedding.tolist() if self.combined_embedding is not None else None
            ),
            "embedding_dim": self.embedding_dim,
            "has_title": self.has_title,
            "has_description": self.has_description,
            "model_name": self.model_name,
        }


class TextEmbeddingExtractor:
    """Extract text embeddings using sentence-transformers.

    Memory optimization: Model is loaded lazily on first use, not at initialization.
    This saves ~500-800MB of memory if embeddings are never used.

    Usage:
        extractor = TextEmbeddingExtractor()

        # Single record
        embeddings = extractor.extract(record)

        # Batch (more efficient)
        embeddings_list = extractor.extract_batch(records)

        # Compare embeddings
        similarity = extractor.cosine_similarity(emb1, emb2)
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, lazy_load: bool = True):
        """Initialize the embedding extractor.

        Args:
            model_name: Name of the sentence-transformers model to use.
            lazy_load: If True (default), delay loading model until first use.
                       This saves memory if embeddings are never used.
        """
        self.model_name = model_name
        self._model = None
        self._embedding_dim = 0
        self._lazy_load = lazy_load
        self._load_attempted = False

        # Only load immediately if lazy_load is False
        if not lazy_load:
            self._ensure_model_loaded()

    def _ensure_model_loaded(self) -> bool:
        """Ensure the model is loaded. Returns True if model is available.

        This is called lazily on first use of the model.
        """
        if self._model is not None:
            return True

        if self._load_attempted:
            return False  # Already tried and failed

        self._load_attempted = True

        if not _check_sentence_transformers():
            logger.warning(
                "sentence_transformers_not_available",
                extra={
                    "message": "Text embeddings will not be available. "
                    "Install with: pip install sentence-transformers"
                },
            )
            return False

        self._model = self._load_model(self.model_name)
        if self._model is not None:
            self._embedding_dim = self._model.get_sentence_embedding_dimension()
            return True

        return False

    def _load_model(self, model_name: str):
        """Load or retrieve cached model."""
        global _model_cache

        if model_name in _model_cache:
            logger.info(
                "embedding_model_cache_hit",
                extra={"model_name": model_name},
            )
            return _model_cache[model_name]

        from sentence_transformers import SentenceTransformer

        logger.info(
            "loading_embedding_model",
            extra={"model_name": model_name, "lazy_load": self._lazy_load},
        )

        model = SentenceTransformer(model_name)
        _model_cache[model_name] = model

        logger.info(
            "embedding_model_loaded",
            extra={
                "model_name": model_name,
                "embedding_dim": model.get_sentence_embedding_dimension(),
            },
        )

        return model

    @property
    def is_available(self) -> bool:
        """Check if embedding extraction is available.

        Note: This triggers lazy loading if not already loaded.
        """
        return self._ensure_model_loaded()

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension.

        Note: This triggers lazy loading if not already loaded.
        """
        self._ensure_model_loaded()
        return self._embedding_dim

    def extract(self, record: ProductRecord) -> TextEmbeddings:
        """Extract text embeddings from a product record.

        Args:
            record: ProductRecord with text fields.

        Returns:
            TextEmbeddings with embedded text fields.
        """
        if not self.is_available:
            return self._empty_embeddings(record)

        # Prepare text fields
        title = record.product_name or ""
        has_title = bool(title.strip())

        # We don't have description in current schema, but prepare for it
        description = ""  # record.description if hasattr(record, 'description') else ""
        has_description = bool(description.strip())

        # Embed title
        title_embedding = None
        if has_title:
            title_embedding = self._model.encode(title, convert_to_numpy=True)

        # Embed description
        description_embedding = None
        if has_description:
            description_embedding = self._model.encode(description, convert_to_numpy=True)

        # Combined embedding (average of available embeddings)
        combined_embedding = None
        embeddings_to_combine = []
        if title_embedding is not None:
            embeddings_to_combine.append(title_embedding)
        if description_embedding is not None:
            embeddings_to_combine.append(description_embedding)

        if embeddings_to_combine:
            combined_embedding = np.mean(embeddings_to_combine, axis=0)

        return TextEmbeddings(
            title_embedding=title_embedding,
            description_embedding=description_embedding,
            combined_embedding=combined_embedding,
            embedding_dim=self._embedding_dim,
            has_title=has_title,
            has_description=has_description,
            model_name=self.model_name,
            competitor_product_id=record.competitor_product_id,
            competitor=record.competitor,
        )

    def extract_batch(self, records: list[ProductRecord]) -> list[TextEmbeddings]:
        """Extract embeddings for a batch of records.

        More efficient than calling extract() repeatedly due to
        batched inference.

        Args:
            records: List of ProductRecords.

        Returns:
            List of TextEmbeddings, one per record.
        """
        if not self.is_available:
            return [self._empty_embeddings(r) for r in records]

        # Collect all texts
        titles = [r.product_name or "" for r in records]

        # Batch encode titles
        title_embeddings = self._model.encode(
            titles, convert_to_numpy=True, show_progress_bar=False
        )

        # Build results
        results = []
        for i, record in enumerate(records):
            has_title = bool(titles[i].strip())
            title_emb = title_embeddings[i] if has_title else None

            results.append(
                TextEmbeddings(
                    title_embedding=title_emb,
                    description_embedding=None,  # Not available in current schema
                    combined_embedding=title_emb,  # Just title for now
                    embedding_dim=self._embedding_dim,
                    has_title=has_title,
                    has_description=False,
                    model_name=self.model_name,
                    competitor_product_id=record.competitor_product_id,
                    competitor=record.competitor,
                )
            )

        logger.info(
            "batch_embeddings_extracted",
            extra={
                "batch_size": len(records),
                "records_with_title": sum(1 for r in results if r.has_title),
            },
        )

        return results

    def _empty_embeddings(self, record: ProductRecord) -> TextEmbeddings:
        """Create empty embeddings when model is not available."""
        return TextEmbeddings(
            title_embedding=None,
            description_embedding=None,
            combined_embedding=None,
            embedding_dim=0,
            has_title=False,
            has_description=False,
            model_name="none",
            competitor_product_id=record.competitor_product_id,
            competitor=record.competitor,
        )

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings.

        Args:
            emb1: First embedding vector.
            emb2: Second embedding vector.

        Returns:
            Cosine similarity in range [-1, 1].
        """
        if emb1 is None or emb2 is None:
            return 0.0

        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    @staticmethod
    def euclidean_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate Euclidean distance between two embeddings.

        Args:
            emb1: First embedding vector.
            emb2: Second embedding vector.

        Returns:
            Euclidean distance (0 = identical).
        """
        if emb1 is None or emb2 is None:
            return float("inf")

        return float(np.linalg.norm(emb1 - emb2))


def compute_embedding_drift(
    old_embedding: np.ndarray | None,
    new_embedding: np.ndarray | None,
    threshold: float = 0.1,
) -> dict[str, Any]:
    """Compute drift between old and new embeddings.

    Used to detect content changes in product titles/descriptions.

    Args:
        old_embedding: Previous embedding.
        new_embedding: Current embedding.
        threshold: Cosine distance threshold for flagging drift.

    Returns:
        Dictionary with drift metrics.
    """
    if old_embedding is None or new_embedding is None:
        return {
            "has_drift_data": False,
            "is_drift": False,
            "cosine_similarity": None,
            "cosine_distance": None,
            "euclidean_distance": None,
        }

    cos_sim = TextEmbeddingExtractor.cosine_similarity(old_embedding, new_embedding)
    cos_dist = 1.0 - cos_sim
    euc_dist = TextEmbeddingExtractor.euclidean_distance(old_embedding, new_embedding)

    is_drift = cos_dist > threshold

    return {
        "has_drift_data": True,
        "is_drift": is_drift,
        "cosine_similarity": cos_sim,
        "cosine_distance": cos_dist,
        "euclidean_distance": euc_dist,
        "threshold": threshold,
    }


@dataclass
class ContentDriftResult:
    """Result of content drift detection for a single product."""

    competitor_product_id: str
    competitor: str

    # Drift detection
    has_previous_embedding: bool
    title_similarity: float | None  # 0-1, 1 = identical
    title_drift_detected: bool
    description_similarity: float | None
    description_drift_detected: bool

    # Content quality indicators
    title_length_ratio: float | None  # current / previous
    content_degraded: bool  # Title shortened significantly

    # Details
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "competitor_product_id": self.competitor_product_id,
            "competitor": self.competitor,
            "has_previous_embedding": self.has_previous_embedding,
            "title_similarity": self.title_similarity,
            "title_drift_detected": self.title_drift_detected,
            "description_similarity": self.description_similarity,
            "description_drift_detected": self.description_drift_detected,
            "title_length_ratio": self.title_length_ratio,
            "content_degraded": self.content_degraded,
            **self.details,
        }


@dataclass
class RunContentDriftSummary:
    """Aggregated content drift metrics for a scrape run."""

    # Counts
    total_products: int
    products_with_previous: int
    products_with_title_drift: int
    products_with_content_degradation: int

    # Rates
    title_drift_rate: float
    content_degradation_rate: float

    # Distribution
    mean_title_similarity: float | None
    min_title_similarity: float | None

    # Products of concern
    degraded_product_ids: list[str]
    drifted_product_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_products": self.total_products,
            "products_with_previous": self.products_with_previous,
            "products_with_title_drift": self.products_with_title_drift,
            "products_with_content_degradation": self.products_with_content_degradation,
            "title_drift_rate": self.title_drift_rate,
            "content_degradation_rate": self.content_degradation_rate,
            "mean_title_similarity": self.mean_title_similarity,
            "min_title_similarity": self.min_title_similarity,
            "degraded_product_ids": self.degraded_product_ids[:10],  # Limit for logging
            "drifted_product_ids": self.drifted_product_ids[:10],
        }


class ContentDriftDetector:
    """Detect content drift using text embeddings.

    Compares current product content to previous observations to detect:
    - Title drift (semantic change in title)
    - Content degradation (title shortened, fields missing)
    - Run-level drift patterns (many products drifting = scraper issue)

    Usage:
        detector = ContentDriftDetector()

        # Detect drift for a single product
        result = detector.detect_drift(
            current_record=record,
            previous_embedding=old_embedding,
            previous_title=old_title,
        )

        # Aggregate drift for a run
        summary = detector.aggregate_run_drift(results)
    """

    def __init__(
        self,
        drift_threshold: float = 0.2,  # Cosine distance > this = drift
        degradation_threshold: float = 0.5,  # Length ratio < this = degraded
        model_name: str = DEFAULT_MODEL,
    ):
        """Initialize the detector.

        Args:
            drift_threshold: Cosine distance threshold for drift detection.
            degradation_threshold: Title length ratio threshold for degradation.
            model_name: Embedding model to use.
        """
        self.drift_threshold = drift_threshold
        self.degradation_threshold = degradation_threshold
        self._extractor = TextEmbeddingExtractor(model_name)

    @property
    def is_available(self) -> bool:
        """Check if embedding model is available."""
        return self._extractor.is_available

    def detect_drift(
        self,
        current_record: ProductRecord,
        previous_embedding: np.ndarray | None = None,
        previous_title: str | None = None,
    ) -> ContentDriftResult:
        """Detect content drift for a single product.

        Args:
            current_record: Current product record.
            previous_embedding: Previous title embedding (if available).
            previous_title: Previous title text (for length comparison).

        Returns:
            ContentDriftResult with drift detection results.
        """
        details: dict[str, Any] = {}

        # Check if we have previous data
        has_previous = previous_embedding is not None
        current_title = current_record.product_name or ""

        # Calculate title length ratio
        title_length_ratio = None
        if previous_title and current_title:
            prev_len = len(previous_title)
            if prev_len > 0:
                title_length_ratio = len(current_title) / prev_len
                details["previous_title_length"] = prev_len
                details["current_title_length"] = len(current_title)

        # Check for content degradation (title shortened significantly)
        content_degraded = False
        if title_length_ratio is not None and title_length_ratio < self.degradation_threshold:
            content_degraded = True
            details["degradation_reason"] = "title_shortened"

        # If title is now empty but was present before
        if previous_title and not current_title:
            content_degraded = True
            details["degradation_reason"] = "title_missing"

        # Calculate embedding similarity
        title_similarity = None
        title_drift_detected = False

        if has_previous and self.is_available and current_title:
            # Get current embedding
            current_embeddings = self._extractor.extract(current_record)
            current_embedding = current_embeddings.title_embedding

            if current_embedding is not None:
                title_similarity = TextEmbeddingExtractor.cosine_similarity(
                    previous_embedding, current_embedding
                )
                details["cosine_distance"] = 1.0 - title_similarity

                # Detect drift
                if (1.0 - title_similarity) > self.drift_threshold:
                    title_drift_detected = True

        return ContentDriftResult(
            competitor_product_id=current_record.competitor_product_id,
            competitor=current_record.competitor,
            has_previous_embedding=has_previous,
            title_similarity=title_similarity,
            title_drift_detected=title_drift_detected,
            description_similarity=None,  # Not implemented yet
            description_drift_detected=False,
            title_length_ratio=title_length_ratio,
            content_degraded=content_degraded,
            details=details,
        )

    def detect_drift_batch(
        self,
        records: list[ProductRecord],
        previous_embeddings: dict[str, np.ndarray],
        previous_titles: dict[str, str],
    ) -> list[ContentDriftResult]:
        """Detect drift for a batch of products.

        Args:
            records: List of current product records.
            previous_embeddings: Map of competitor_product_id -> embedding.
            previous_titles: Map of competitor_product_id -> title.

        Returns:
            List of ContentDriftResult.
        """
        results = []
        for record in records:
            prev_emb = previous_embeddings.get(record.competitor_product_id)
            prev_title = previous_titles.get(record.competitor_product_id)

            result = self.detect_drift(
                current_record=record,
                previous_embedding=prev_emb,
                previous_title=prev_title,
            )
            results.append(result)

        return results

    def aggregate_run_drift(
        self, drift_results: list[ContentDriftResult]
    ) -> RunContentDriftSummary:
        """Aggregate drift results for a scrape run.

        Args:
            drift_results: List of ContentDriftResult from detect_drift_batch.

        Returns:
            RunContentDriftSummary with aggregated metrics.
        """
        total = len(drift_results)
        if total == 0:
            return RunContentDriftSummary(
                total_products=0,
                products_with_previous=0,
                products_with_title_drift=0,
                products_with_content_degradation=0,
                title_drift_rate=0.0,
                content_degradation_rate=0.0,
                mean_title_similarity=None,
                min_title_similarity=None,
                degraded_product_ids=[],
                drifted_product_ids=[],
            )

        with_previous = sum(1 for r in drift_results if r.has_previous_embedding)
        with_drift = sum(1 for r in drift_results if r.title_drift_detected)
        with_degradation = sum(1 for r in drift_results if r.content_degraded)

        # Calculate rates (relative to products with previous data)
        drift_rate = with_drift / with_previous if with_previous > 0 else 0.0
        degradation_rate = with_degradation / with_previous if with_previous > 0 else 0.0

        # Calculate similarity statistics
        similarities = [r.title_similarity for r in drift_results if r.title_similarity is not None]
        mean_sim = float(np.mean(similarities)) if similarities else None
        min_sim = float(min(similarities)) if similarities else None

        # Collect problematic product IDs
        degraded_ids = [r.competitor_product_id for r in drift_results if r.content_degraded]
        drifted_ids = [r.competitor_product_id for r in drift_results if r.title_drift_detected]

        summary = RunContentDriftSummary(
            total_products=total,
            products_with_previous=with_previous,
            products_with_title_drift=with_drift,
            products_with_content_degradation=with_degradation,
            title_drift_rate=drift_rate,
            content_degradation_rate=degradation_rate,
            mean_title_similarity=mean_sim,
            min_title_similarity=min_sim,
            degraded_product_ids=degraded_ids,
            drifted_product_ids=drifted_ids,
        )

        logger.info(
            "run_content_drift_aggregated",
            extra={
                "total_products": total,
                "products_with_previous": with_previous,
                "title_drift_rate": round(drift_rate, 3),
                "content_degradation_rate": round(degradation_rate, 3),
                "mean_title_similarity": round(mean_sim, 3) if mean_sim else None,
            },
        )

        return summary
