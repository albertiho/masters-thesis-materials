"""Scrape Bug vs Real Move Classifier.

Classifies anomalies as either:
- Scrape issue: Data artifact from scraper failure
- Real market event: Genuine price/content change

This is the "operational heart" of the trusted signal system.
The classifier routes anomalies:
- Scrape issue -> Suppress downstream price updates, alert ops
- Real event -> Allow to flow to pricing rules (with optional persistence check)

Implementation:
- Phase A: Rule-based baseline (interpretable, high precision)
- Phase B: ML classifier (optional enhancement, trained on weak labels)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from src.features.coherence import ProductLevelFeatures

logger = logging.getLogger(__name__)


class ClassificationResult(str, Enum):
    """Classification result for an anomaly."""

    SCRAPE_ISSUE = "scrape_issue"
    REAL_EVENT = "real_event"
    UNCERTAIN = "uncertain"


@dataclass
class ScrapeIssueClassification:
    """Result of scrape issue classification."""

    # Classification
    classification: ClassificationResult
    scrape_issue_probability: float  # 0-1, probability of scrape issue

    # Confidence
    confidence: float  # 0-1, confidence in the classification

    # Explanation
    reason_codes: list[str]  # Reasons for classification
    contributing_factors: dict[str, float]  # Factor name -> contribution

    # Recommendations
    suppress_downstream: bool  # Should this be suppressed from pricing rules?
    requires_persistence_check: bool  # Should we wait for persistence?
    requires_manual_review: bool  # Should this be flagged for human review?

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "classification": self.classification.value,
            "scrape_issue_probability": self.scrape_issue_probability,
            "confidence": self.confidence,
            "reason_codes": self.reason_codes,
            "contributing_factors": self.contributing_factors,
            "suppress_downstream": self.suppress_downstream,
            "requires_persistence_check": self.requires_persistence_check,
            "requires_manual_review": self.requires_manual_review,
        }


@dataclass
class ClassifierConfig:
    """Configuration for the classifier."""

    # Thresholds
    scrape_issue_threshold: float = 0.5  # Above this = scrape issue
    uncertain_band: float = 0.15  # Within this of threshold = uncertain

    # Factor weights for rule-based classifier
    weight_content_broke_price_changed: float = 0.40
    weight_run_unhealthy: float = 0.25
    weight_not_persisted: float = 0.15
    weight_no_cross_competitor: float = 0.10
    weight_high_anomaly_score: float = 0.10

    # Thresholds for individual factors
    run_health_threshold: float = 0.6  # Below this = unhealthy
    persistence_threshold: int = 2  # Runs required for persistence
    cross_competitor_threshold: float = 0.3  # Above this = corroborated
    anomaly_score_threshold: float = 0.7  # Above this = high anomaly


class ScrapeIssueClassifier:
    """Classify anomalies as scrape issues vs real market events.

    Uses a rule-based approach for interpretability and high precision.

    Usage:
        classifier = ScrapeIssueClassifier()

        result = classifier.classify(product_features)

        if result.suppress_downstream:
            # Don't propagate to pricing rules
            pass
        elif result.requires_persistence_check:
            # Wait for change to persist
            pass
        else:
            # Allow through to pricing rules
            pass
    """

    def __init__(self, config: ClassifierConfig | None = None):
        """Initialize classifier.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or ClassifierConfig()
        self.name = "scrape_issue_classifier"

    def classify(self, features: ProductLevelFeatures) -> ScrapeIssueClassification:
        """Classify a product's anomaly as scrape issue or real event.

        Args:
            features: ProductLevelFeatures extracted for the product.

        Returns:
            ScrapeIssueClassification with result and explanation.
        """
        cfg = self.config
        contributing_factors: dict[str, float] = {}
        reason_codes: list[str] = []
        total_score = 0.0

        # Factor 1: Content broke AND price changed (strongest signal)
        if features.content_degraded and features.price_changed:
            score = cfg.weight_content_broke_price_changed
            total_score += score
            contributing_factors["content_broke_price_changed"] = score
            reason_codes.append("CONTENT_DEGRADED_WITH_PRICE_CHANGE")

        # Factor 2: Run is unhealthy
        if features.run_is_healthy is False or (
            features.run_health_score is not None
            and features.run_health_score < cfg.run_health_threshold
        ):
            score = cfg.weight_run_unhealthy
            total_score += score
            contributing_factors["run_unhealthy"] = score
            reason_codes.append("RUN_UNHEALTHY")

        # Factor 3: Change hasn't persisted
        if features.change_persisted_runs < cfg.persistence_threshold:
            score = cfg.weight_not_persisted
            total_score += score
            contributing_factors["not_persisted"] = score
            reason_codes.append("NOT_PERSISTED")

        # Factor 4: No cross-competitor corroboration
        if (
            features.cross_competitor_agreement is not None
            and features.cross_competitor_agreement < cfg.cross_competitor_threshold
        ):
            score = cfg.weight_no_cross_competitor
            total_score += score
            contributing_factors["no_cross_competitor"] = score
            reason_codes.append("NO_CROSS_COMPETITOR_AGREEMENT")

        # Factor 5: High anomaly score (from other detectors)
        if (
            features.price_anomaly_score is not None
            and features.price_anomaly_score > cfg.anomaly_score_threshold
        ):
            score = cfg.weight_high_anomaly_score
            total_score += score
            contributing_factors["high_anomaly_score"] = score
            reason_codes.append("HIGH_ANOMALY_SCORE")

        # Negative factors (reduce scrape issue probability)

        # Factor: Change persisted multiple runs (strong signal of real event)
        if features.change_persisted_runs >= cfg.persistence_threshold * 2:
            reduction = 0.2
            total_score = max(0, total_score - reduction)
            contributing_factors["well_persisted"] = -reduction
            reason_codes.append("CHANGE_WELL_PERSISTED")

        # Factor: Cross-competitor agreement (strong signal of real event)
        if (
            features.cross_competitor_agreement is not None
            and features.cross_competitor_agreement > 0.5
        ):
            reduction = 0.15
            total_score = max(0, total_score - reduction)
            contributing_factors["cross_competitor_agree"] = -reduction
            reason_codes.append("CROSS_COMPETITOR_CORROBORATED")

        # Factor: Content intact (if price changed but content fine, more likely real)
        if features.price_changed and not features.content_degraded and not features.title_changed:
            reduction = 0.1
            total_score = max(0, total_score - reduction)
            contributing_factors["content_intact"] = -reduction
            reason_codes.append("CONTENT_INTACT")

        # Normalize score to 0-1
        scrape_issue_probability = min(max(total_score, 0.0), 1.0)

        # Determine classification
        if scrape_issue_probability > cfg.scrape_issue_threshold + cfg.uncertain_band:
            classification = ClassificationResult.SCRAPE_ISSUE
            confidence = min(
                (scrape_issue_probability - cfg.scrape_issue_threshold)
                / (1 - cfg.scrape_issue_threshold),
                1.0,
            )
        elif scrape_issue_probability < cfg.scrape_issue_threshold - cfg.uncertain_band:
            classification = ClassificationResult.REAL_EVENT
            confidence = min(
                (cfg.scrape_issue_threshold - scrape_issue_probability)
                / cfg.scrape_issue_threshold,
                1.0,
            )
        else:
            classification = ClassificationResult.UNCERTAIN
            confidence = 0.5

        # Determine recommendations
        suppress_downstream = classification == ClassificationResult.SCRAPE_ISSUE
        requires_persistence_check = (
            classification in (ClassificationResult.UNCERTAIN, ClassificationResult.REAL_EVENT)
            and features.change_persisted_runs < cfg.persistence_threshold
        )
        requires_manual_review = classification == ClassificationResult.UNCERTAIN

        result = ScrapeIssueClassification(
            classification=classification,
            scrape_issue_probability=scrape_issue_probability,
            confidence=confidence,
            reason_codes=reason_codes,
            contributing_factors=contributing_factors,
            suppress_downstream=suppress_downstream,
            requires_persistence_check=requires_persistence_check,
            requires_manual_review=requires_manual_review,
        )

        # Log classification
        logger.info(
            "anomaly_classified",
            extra={
                "classifier": self.name,
                "competitor_product_id": features.competitor_product_id,
                "competitor": features.competitor,
                "classification": classification.value,
                "scrape_issue_probability": round(scrape_issue_probability, 3),
                "confidence": round(confidence, 3),
                "reason_codes": reason_codes,
                "suppress_downstream": suppress_downstream,
            },
        )

        return result

    def classify_batch(
        self, features_list: list[ProductLevelFeatures]
    ) -> list[ScrapeIssueClassification]:
        """Classify a batch of products.

        Args:
            features_list: List of ProductLevelFeatures.

        Returns:
            List of ScrapeIssueClassification results.
        """
        return [self.classify(f) for f in features_list]


class MLScrapeIssueClassifier:
    """ML-based classifier for scrape issue detection.

    Trained on weak labels from the rule-based classifier.
    Provides calibrated probability outputs.

    Note: This is an optional enhancement. The rule-based classifier
    is sufficient for most use cases and is more interpretable.
    """

    def __init__(
        self,
        model: Any | None = None,
        threshold: float = 0.5,
    ):
        """Initialize ML classifier.

        Args:
            model: Trained sklearn model (e.g., LogisticRegression, GradientBoosting).
            threshold: Threshold for classification.
        """
        self.model = model
        self.threshold = threshold
        self.name = "ml_scrape_issue_classifier"
        self._is_fitted = model is not None

    def fit(
        self,
        features: list[ProductLevelFeatures],
        labels: list[bool],
    ) -> "MLScrapeIssueClassifier":
        """Train the classifier.

        Args:
            features: List of ProductLevelFeatures.
            labels: List of labels (True = scrape issue, False = real event).

        Returns:
            self for chaining.
        """
        from sklearn.linear_model import LogisticRegression

        # Convert to feature vectors
        X = np.array([f.to_feature_vector() for f in features])
        y = np.array(labels, dtype=int)

        # Train logistic regression (well-calibrated probabilities)
        self.model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )
        self.model.fit(X, y)
        self._is_fitted = True

        logger.info(
            "ml_classifier_trained",
            extra={
                "classifier": self.name,
                "num_samples": len(features),
                "positive_rate": float(np.mean(y)),
            },
        )

        return self

    def predict_proba(self, features: ProductLevelFeatures) -> float:
        """Predict probability of scrape issue.

        Args:
            features: ProductLevelFeatures.

        Returns:
            Probability of scrape issue (0-1).
        """
        if not self._is_fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")

        X = features.to_feature_vector().reshape(1, -1)
        proba = self.model.predict_proba(X)[0, 1]
        return float(proba)

    def classify(self, features: ProductLevelFeatures) -> ScrapeIssueClassification:
        """Classify using ML model.

        Args:
            features: ProductLevelFeatures.

        Returns:
            ScrapeIssueClassification.
        """
        proba = self.predict_proba(features)

        if proba > self.threshold + 0.15:
            classification = ClassificationResult.SCRAPE_ISSUE
        elif proba < self.threshold - 0.15:
            classification = ClassificationResult.REAL_EVENT
        else:
            classification = ClassificationResult.UNCERTAIN

        return ScrapeIssueClassification(
            classification=classification,
            scrape_issue_probability=proba,
            confidence=abs(proba - 0.5) * 2,  # Scale to 0-1
            reason_codes=["ML_CLASSIFIER"],
            contributing_factors={"ml_prediction": proba},
            suppress_downstream=classification == ClassificationResult.SCRAPE_ISSUE,
            requires_persistence_check=classification != ClassificationResult.SCRAPE_ISSUE
            and features.change_persisted_runs < 2,
            requires_manual_review=classification == ClassificationResult.UNCERTAIN,
        )
