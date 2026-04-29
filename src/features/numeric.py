"""Numeric Feature Extraction for Anomaly Detection.

Extracts price-based and numeric features from ProductRecords.
These features are inputs for all anomaly detection methods.

Features extracted:
    - price: Current price
    - list_price: Original/list price (if available)
    - price_ratio: sale_price / list_price (discount indicator)
    - has_list_price: Boolean flag for list price presence
    - price_per_char: price / len(product_name) - sanity check feature
"""

import logging
import math
from dataclasses import dataclass
from typing import Any

from src.ingestion.parser import ProductRecord

logger = logging.getLogger(__name__)


@dataclass
class NumericFeatures:
    """Container for numeric features extracted from a product record.

    Attributes:
        price: Current sale/offer price
        list_price: Original list price (None if not available)
        price_ratio: sale_price / list_price (1.0 if no list price)
        has_list_price: Whether list_price was present
        price_log: log(price + 1) for scale-invariant comparisons
        is_valid: Whether the record has valid numeric data
        validation_errors: List of validation issues found
    """

    price: float
    list_price: float | None
    price_ratio: float
    has_list_price: bool
    price_log: float
    is_valid: bool
    validation_errors: list[str]

    # Identifiers for joining back to original records
    competitor_product_id: str
    competitor: str
    country: str | None
    currency: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            "competitor_product_id": self.competitor_product_id,
            "competitor": self.competitor,
            "country": self.country,
            "currency": self.currency,
            "price": self.price,
            "list_price": self.list_price,
            "price_ratio": self.price_ratio,
            "has_list_price": self.has_list_price,
            "price_log": self.price_log,
            "is_valid": self.is_valid,
            "validation_errors": (
                ",".join(self.validation_errors) if self.validation_errors else None
            ),
        }


def extract_numeric_features(record: ProductRecord) -> NumericFeatures:
    """Extract numeric features from a ProductRecord.

    Args:
        record: ProductRecord from the parser.

    Returns:
        NumericFeatures with extracted values and validation status.

    Note:
        This function logs validation issues for audit trail compliance.
    """
    validation_errors: list[str] = []
    is_valid = True

    # Validate vital non-price fields.
    product_name = record.product_name.strip() if isinstance(record.product_name, str) else None
    if not product_name:
        validation_errors.append("missing_product_name")

    currency = record.currency.strip() if isinstance(record.currency, str) else None
    if not currency:
        validation_errors.append("missing_currency")
        is_valid = False

    # Extract price
    price = record.price
    if price is None:
        validation_errors.append("missing_price")
        is_valid = False
        price = 0.0
    elif price < 0:
        validation_errors.append(f"negative_price:{price}")
        is_valid = False
    elif price == 0:
        validation_errors.append("zero_price")
        # Zero price might be valid (free item) but flag it

    # Extract list_price
    list_price = record.list_price
    has_list_price = list_price is not None

    if list_price is not None and list_price < 0:
        validation_errors.append(f"negative_list_price:{list_price}")
        list_price = None
        has_list_price = False

    # Calculate price_ratio (discount indicator)
    if has_list_price and list_price and list_price > 0:
        price_ratio = price / list_price
        # Flag suspicious ratios
        if price_ratio > 1.0:
            validation_errors.append(f"price_exceeds_list:{price_ratio:.2f}")
        elif price_ratio < 0.1:
            validation_errors.append(f"extreme_discount:{price_ratio:.2f}")
    else:
        price_ratio = 1.0  # No discount info available

    # Calculate log price for scale-invariant comparisons
    price_log = math.log(price + 1) if price >= 0 else 0.0

    # Log validation issues for audit trail
    if validation_errors:
        logger.warning(
            "numeric_feature_validation",
            extra={
                "competitor_product_id": record.competitor_product_id,
                "competitor": record.competitor,
                "country": record.country,
                "price": record.price,
                "list_price": record.list_price,
                "currency": record.currency,
                "product_name": record.product_name,
                "validation_errors": validation_errors,
                "is_valid": is_valid,
            },
        )

    return NumericFeatures(
        price=price,
        list_price=list_price,
        price_ratio=price_ratio,
        has_list_price=has_list_price,
        price_log=price_log,
        is_valid=is_valid,
        validation_errors=validation_errors,
        competitor_product_id=record.competitor_product_id,
        competitor=record.competitor,
        country=record.country,
        currency=currency,
    )


def extract_numeric_features_batch(
    records: list[ProductRecord],
) -> list[NumericFeatures]:
    """Extract numeric features from a batch of records.

    Args:
        records: List of ProductRecords.

    Returns:
        List of NumericFeatures, one per record.
    """
    features = []
    valid_count = 0
    invalid_count = 0

    for record in records:
        feature = extract_numeric_features(record)
        features.append(feature)
        if feature.is_valid:
            valid_count += 1
        else:
            invalid_count += 1

    logger.info(
        "numeric_feature_extraction_complete",
        extra={
            "total_records": len(records),
            "valid_records": valid_count,
            "invalid_records": invalid_count,
            "validation_rate": valid_count / len(records) if records else 0,
        },
    )

    return features
