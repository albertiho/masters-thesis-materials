"""Run Health Calculator - Compute health metrics for scrape runs.

Calculates per-run metrics to determine if a scrape run is healthy enough
for downstream consumption. Unhealthy runs can be suppressed to prevent
bad data from affecting pricing decisions.

Health factors:
- Parse error rate
- Missing field rates (price, title, images, EAN)
- Row count drift from expected
- Price distribution drift from recent history
"""

from __future__ import annotations

import json
import logging
import statistics
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from src.ingestion.parser import ParseResult, ProductRecord

logger = logging.getLogger(__name__)


class WarningFlag(str, Enum):
    """Warning flags for run health issues."""

    HIGH_PARSE_ERRORS = "high_parse_errors"
    HIGH_MISSING_PRICE = "high_missing_price"
    HIGH_MISSING_TITLE = "high_missing_title"
    HIGH_MISSING_IMAGE = "high_missing_image"
    HIGH_MISSING_EAN = "high_missing_ean"
    ROW_COUNT_DROP = "row_count_drop"
    ROW_COUNT_SPIKE = "row_count_spike"
    PRICE_DISTRIBUTION_DRIFT = "price_distribution_drift"
    LOW_UNIQUE_PRODUCTS = "low_unique_products"
    EMPTY_RUN = "empty_run"


@dataclass
class RunHealth:
    """Health metrics for a single scrape run."""

    # Identifiers
    run_id: str
    competitor: str
    country: str
    channel: str | None
    scraped_at: datetime
    processed_at: datetime
    source_path: str | None

    # Record counts
    total_records: int
    parsed_records: int
    parse_error_count: int
    parse_error_rate: float

    # Missing field metrics
    missing_price_count: int
    missing_price_rate: float
    missing_title_count: int
    missing_title_rate: float
    missing_image_count: int
    missing_image_rate: float
    missing_ean_count: int
    missing_ean_rate: float
    unique_products: int

    # Price distribution metrics
    price_mean: float | None
    price_std: float | None
    price_median: float | None
    price_min: float | None
    price_max: float | None

    # Drift metrics (populated when historical data available)
    row_count_expected: int | None = None
    row_count_drift_pct: float | None = None
    row_count_drift_severity: float | None = None
    price_distribution_drift: float | None = None
    price_distribution_drift_severity: float | None = None
    content_drift_rate: float | None = None

    # Health score
    health_score: float = 0.0
    is_healthy: bool = False
    health_threshold: float = 0.7
    health_factors: dict[str, float] = field(default_factory=dict)
    warning_flags: list[WarningFlag] = field(default_factory=list)

    def to_row_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable row dictionary."""
        return {
            "run_id": self.run_id,
            "competitor": self.competitor,
            "country": self.country,
            "channel": self.channel,
            "scraped_at": self.scraped_at.isoformat(),
            "processed_at": self.processed_at.isoformat(),
            "source_path": self.source_path,
            "total_records": self.total_records,
            "parsed_records": self.parsed_records,
            "parse_error_count": self.parse_error_count,
            "parse_error_rate": self.parse_error_rate,
            "missing_price_count": self.missing_price_count,
            "missing_price_rate": self.missing_price_rate,
            "missing_title_count": self.missing_title_count,
            "missing_title_rate": self.missing_title_rate,
            "missing_image_count": self.missing_image_count,
            "missing_image_rate": self.missing_image_rate,
            "missing_ean_count": self.missing_ean_count,
            "missing_ean_rate": self.missing_ean_rate,
            "unique_products": self.unique_products,
            "price_mean": self.price_mean,
            "price_std": self.price_std,
            "price_median": self.price_median,
            "price_min": self.price_min,
            "price_max": self.price_max,
            "row_count_expected": self.row_count_expected,
            "row_count_drift_pct": self.row_count_drift_pct,
            "row_count_drift_severity": self.row_count_drift_severity,
            "price_distribution_drift": self.price_distribution_drift,
            "price_distribution_drift_severity": self.price_distribution_drift_severity,
            "content_drift_rate": self.content_drift_rate,
            "health_score": self.health_score,
            "is_healthy": self.is_healthy,
            "health_threshold": self.health_threshold,
            "health_factors": json.dumps(self.health_factors),
            "warning_flags": [f.value for f in self.warning_flags],
        }


@dataclass
class RunHealthConfig:
    """Configuration for health score calculation."""

    # Health threshold
    health_threshold: float = 0.7

    # Weight factors for health score
    weight_parse_error: float = 0.30
    weight_missing_price: float = 0.20
    weight_missing_title: float = 0.20
    weight_row_count_drift: float = 0.15
    weight_price_distribution_drift: float = 0.15

    # Warning thresholds
    parse_error_warning_threshold: float = 0.05  # 5%
    missing_price_warning_threshold: float = 0.10  # 10%
    missing_title_warning_threshold: float = 0.20  # 20%
    missing_image_warning_threshold: float = 0.30  # 30%
    missing_ean_warning_threshold: float = 0.50  # 50% (EAN often missing)
    row_count_drift_warning_threshold: float = 0.20  # 20% drop/spike
    price_drift_warning_threshold: float = 0.30  # 30% distribution change


class RunHealthCalculator:
    """Calculate health metrics for scrape runs.

    Usage:
        calculator = RunHealthCalculator()
        health = calculator.calculate(
            records=parsed_records,
            parse_result=parse_result,
            competitor="PROSHOP_DK",
            country="DK",
            source_path="data/training/source/example.jsonl",
        )

        if not health.is_healthy:
            # Suppress downstream processing
            logger.warning("Unhealthy run", extra={"flags": health.warning_flags})
    """

    def __init__(self, config: RunHealthConfig | None = None) -> None:
        """Initialize the calculator.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or RunHealthConfig()

    def calculate(
        self,
        records: list[ProductRecord],
        parse_result: ParseResult,
        competitor: str,
        country: str,
        channel: str | None = None,
        source_path: str | None = None,
        run_id: str | None = None,
        expected_row_count: int | None = None,
        historical_price_stats: dict[str, float] | None = None,
    ) -> RunHealth:
        """Calculate health metrics for a scrape run.

        Args:
            records: List of successfully parsed ProductRecords.
            parse_result: ParseResult with total/failed counts.
            competitor: Competitor identifier.
            country: Country code.
            channel: Sales channel (b2c, b2b).
            source_path: Local path of source files.
            run_id: Optional run ID (generated if not provided).
            expected_row_count: Expected row count from historical data.
            historical_price_stats: Historical price stats for drift detection.

        Returns:
            RunHealth with calculated metrics and health score.
        """
        now = datetime.now(timezone.utc)

        # Generate run_id if not provided
        if not run_id:
            run_id = (
                f"{competitor}_{country}_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            )

        # Get earliest scraped_at from records
        scraped_at = now
        if records:
            scraped_times = [r.scraped_at for r in records if r.scraped_at]
            if scraped_times:
                scraped_at = min(scraped_times)

        # Calculate missing field counts
        missing_price = sum(1 for r in records if r.price is None or r.price <= 0)
        missing_title = sum(1 for r in records if not r.product_name)
        missing_image = sum(
            1 for r in records if not r.raw_data.get("images") and not r.raw_data.get("image_url")
        )
        missing_ean = sum(1 for r in records if not r.ean)

        # Calculate rates
        total = parse_result.total_lines
        parsed = len(records)
        parse_error_rate = parse_result.failed / total if total > 0 else 0.0
        missing_price_rate = missing_price / parsed if parsed > 0 else 0.0
        missing_title_rate = missing_title / parsed if parsed > 0 else 0.0
        missing_image_rate = missing_image / parsed if parsed > 0 else 0.0
        missing_ean_rate = missing_ean / parsed if parsed > 0 else 0.0

        # Calculate unique products
        unique_products = len(set(r.competitor_product_id for r in records))

        # Calculate price distribution
        prices = [r.price for r in records if r.price and r.price > 0]
        price_mean = statistics.mean(prices) if prices else None
        price_std = statistics.stdev(prices) if len(prices) > 1 else None
        price_median = statistics.median(prices) if prices else None
        price_min = min(prices) if prices else None
        price_max = max(prices) if prices else None

        # Calculate drift metrics
        row_count_drift_pct = None
        row_count_drift_severity = None
        if expected_row_count and expected_row_count > 0:
            row_count_drift_pct = (parsed - expected_row_count) / expected_row_count
            # Convert drift to severity (0-1 scale)
            row_count_drift_severity = min(abs(row_count_drift_pct) / 0.5, 1.0)

        price_distribution_drift = None
        price_distribution_drift_severity = None
        if historical_price_stats and price_mean is not None:
            hist_mean = historical_price_stats.get("mean")
            hist_std = historical_price_stats.get("std")
            if hist_mean and hist_std and hist_std > 0:
                # Z-score of current mean vs historical
                price_distribution_drift = abs(price_mean - hist_mean) / hist_std
                price_distribution_drift_severity = min(price_distribution_drift / 3.0, 1.0)

        # Calculate health score
        health_factors = self._calculate_health_factors(
            parse_error_rate=parse_error_rate,
            missing_price_rate=missing_price_rate,
            missing_title_rate=missing_title_rate,
            row_count_drift_severity=row_count_drift_severity,
            price_distribution_drift_severity=price_distribution_drift_severity,
        )

        health_score = sum(health_factors.values())
        is_healthy = health_score >= self.config.health_threshold

        # Determine warning flags
        warning_flags = self._determine_warning_flags(
            parse_error_rate=parse_error_rate,
            missing_price_rate=missing_price_rate,
            missing_title_rate=missing_title_rate,
            missing_image_rate=missing_image_rate,
            missing_ean_rate=missing_ean_rate,
            row_count_drift_pct=row_count_drift_pct,
            price_distribution_drift_severity=price_distribution_drift_severity,
            unique_products=unique_products,
            total_records=total,
        )

        health = RunHealth(
            run_id=run_id,
            competitor=competitor,
            country=country,
            channel=channel,
            scraped_at=scraped_at,
            processed_at=now,
            source_path=source_path,
            total_records=total,
            parsed_records=parsed,
            parse_error_count=parse_result.failed,
            parse_error_rate=parse_error_rate,
            missing_price_count=missing_price,
            missing_price_rate=missing_price_rate,
            missing_title_count=missing_title,
            missing_title_rate=missing_title_rate,
            missing_image_count=missing_image,
            missing_image_rate=missing_image_rate,
            missing_ean_count=missing_ean,
            missing_ean_rate=missing_ean_rate,
            unique_products=unique_products,
            price_mean=price_mean,
            price_std=price_std,
            price_median=price_median,
            price_min=price_min,
            price_max=price_max,
            row_count_expected=expected_row_count,
            row_count_drift_pct=row_count_drift_pct,
            row_count_drift_severity=row_count_drift_severity,
            price_distribution_drift=price_distribution_drift,
            price_distribution_drift_severity=price_distribution_drift_severity,
            health_score=health_score,
            is_healthy=is_healthy,
            health_threshold=self.config.health_threshold,
            health_factors=health_factors,
            warning_flags=warning_flags,
        )

        # Log health metrics
        logger.info(
            "run_health_calculated",
            extra={
                "run_id": run_id,
                "competitor": competitor,
                "country": country,
                "health_score": round(health_score, 3),
                "is_healthy": is_healthy,
                "total_records": total,
                "parsed_records": parsed,
                "parse_error_rate": round(parse_error_rate, 3),
                "warning_flags": [f.value for f in warning_flags],
            },
        )

        return health

    def _calculate_health_factors(
        self,
        parse_error_rate: float,
        missing_price_rate: float,
        missing_title_rate: float,
        row_count_drift_severity: float | None,
        price_distribution_drift_severity: float | None,
    ) -> dict[str, float]:
        """Calculate individual health factors.

        Returns:
            Dict mapping factor name to weighted score contribution.
        """
        cfg = self.config

        # Each factor contributes (1 - rate) * weight to the score
        factors = {
            "parse_success": (1 - parse_error_rate) * cfg.weight_parse_error,
            "has_price": (1 - missing_price_rate) * cfg.weight_missing_price,
            "has_title": (1 - missing_title_rate) * cfg.weight_missing_title,
        }

        # Drift factors (use 1.0 if no historical data available)
        if row_count_drift_severity is not None:
            factors["row_count_stable"] = (
                1 - row_count_drift_severity
            ) * cfg.weight_row_count_drift
        else:
            factors["row_count_stable"] = cfg.weight_row_count_drift  # Full credit if no baseline

        if price_distribution_drift_severity is not None:
            factors["price_stable"] = (
                1 - price_distribution_drift_severity
            ) * cfg.weight_price_distribution_drift
        else:
            factors["price_stable"] = (
                cfg.weight_price_distribution_drift
            )  # Full credit if no baseline

        return factors

    def _determine_warning_flags(
        self,
        parse_error_rate: float,
        missing_price_rate: float,
        missing_title_rate: float,
        missing_image_rate: float,
        missing_ean_rate: float,
        row_count_drift_pct: float | None,
        price_distribution_drift_severity: float | None,
        unique_products: int,
        total_records: int,
    ) -> list[WarningFlag]:
        """Determine which warning flags apply to this run."""
        cfg = self.config
        flags = []

        if total_records == 0:
            flags.append(WarningFlag.EMPTY_RUN)
            return flags

        if parse_error_rate > cfg.parse_error_warning_threshold:
            flags.append(WarningFlag.HIGH_PARSE_ERRORS)

        if missing_price_rate > cfg.missing_price_warning_threshold:
            flags.append(WarningFlag.HIGH_MISSING_PRICE)

        if missing_title_rate > cfg.missing_title_warning_threshold:
            flags.append(WarningFlag.HIGH_MISSING_TITLE)

        if missing_image_rate > cfg.missing_image_warning_threshold:
            flags.append(WarningFlag.HIGH_MISSING_IMAGE)

        if missing_ean_rate > cfg.missing_ean_warning_threshold:
            flags.append(WarningFlag.HIGH_MISSING_EAN)

        if row_count_drift_pct is not None:
            if row_count_drift_pct < -cfg.row_count_drift_warning_threshold:
                flags.append(WarningFlag.ROW_COUNT_DROP)
            elif row_count_drift_pct > cfg.row_count_drift_warning_threshold:
                flags.append(WarningFlag.ROW_COUNT_SPIKE)

        if (
            price_distribution_drift_severity is not None
            and price_distribution_drift_severity > cfg.price_drift_warning_threshold
        ):
            flags.append(WarningFlag.PRICE_DISTRIBUTION_DRIFT)

        if unique_products < 10 and total_records > 100:
            flags.append(WarningFlag.LOW_UNIQUE_PRODUCTS)

        return flags
