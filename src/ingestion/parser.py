"""JSONL Parser - Parse raw crawler output into structured records.

Validates required fields and logs malformed records for audit trail.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.constants import infer_currency_from_competitor

logger = logging.getLogger(__name__)


@dataclass
class ProductRecord:
    """Structured product record from crawler output.

    Required fields for golden master:
    - competitor_product_id
    - competitor
    - price
    - scraped_at
    """

    # Required fields
    competitor_product_id: str
    competitor: str
    price: float
    currency: str | None
    scraped_at: datetime

    # Optional fields (populated if available)
    product_name: str | None = None
    brand: str | None = None
    availability_status: str | None = None
    scrape_run_id: str | None = None
    product_url: str | None = None
    mpn: str | None = None
    ean: str | None = None
    list_price: float | None = None

    # Metadata (set during processing)
    country: str | None = None
    channel: str | None = None
    source: str | None = None

    # Raw data for reference
    raw_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParseResult:
    """Result of parsing a JSONL file."""

    records: list[ProductRecord]
    total_lines: int
    successful: int
    failed: int
    errors: list[dict[str, Any]]


def extract_price(pricing: dict[str, Any] | None) -> tuple[float | None, str | None, float | None]:
    """Extract price, currency, and list_price from pricing object.

    Supports both legacy flat pricing (price/currency/list_price) and the
    nested format observed in item_minimal:
        {"price": {"currency": "DKK", "current": [regular, discounted], ...}}

    Returns:
        Tuple of (price, currency, list_price).
    """
    if not pricing:
        return None, None, None

    raw_price = pricing.get("price")
    currency = pricing.get("currency")
    list_price = pricing.get("list_price")

    # Handle nested price dict with current list/tuple
    if isinstance(raw_price, dict):
        currency = raw_price.get("currency") or currency
        current = raw_price.get("current")
        if isinstance(current, (list, tuple)) and current:
            # Use discounted if present (second element), else first
            try:
                price_val = (
                    float(current[1])
                    if len(current) > 1 and current[1] is not None
                    else float(current[0])
                )
            except (TypeError, ValueError):
                price_val = None
            try:
                list_price = float(current[0]) if current[0] is not None else None
            except (TypeError, ValueError):
                list_price = None
        else:
            price_val = current
        raw_price = price_val

    price = None
    if raw_price is not None:
        try:
            price = float(raw_price)
        except (TypeError, ValueError):
            price = None

    if list_price is not None:
        try:
            list_price = float(list_price)
        except (TypeError, ValueError):
            list_price = None

    return price, currency, list_price


def extract_availability_status(availability: dict[str, Any] | None) -> str | None:
    """Extract availability status from availability object.

    Args:
        availability: Availability object from crawler output.

    Returns:
        Online availability status string.
    """
    if not availability:
        return None

    online = availability.get("online", {})
    return online.get("status")


def parse_scraped_at(scraped_at_str: str | None) -> datetime | None:
    """Parse scraped_at timestamp from ISO format string.

    Args:
        scraped_at_str: ISO format timestamp string.

    Returns:
        Parsed datetime or None if parsing fails.
    """
    if not scraped_at_str:
        return None

    try:
        # Handle ISO format with timezone
        return datetime.fromisoformat(scraped_at_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        logger.warning("failed_to_parse_scraped_at", extra={"value": scraped_at_str})
        return None


def parse_line(line: str, line_number: int) -> tuple[ProductRecord | None, dict[str, Any] | None]:
    """Parse a single JSONL line into a ProductRecord.

    Args:
        line: Raw JSON line.
        line_number: Line number for error reporting.

    Returns:
        Tuple of (ProductRecord, None) on success, or (None, error_dict) on failure.
    """
    try:
        data = json.loads(line)
    except json.JSONDecodeError as e:
        error = {
            "line_number": line_number,
            "error_type": "json_decode_error",
            "error_message": str(e),
            "line_preview": line[:100] if line else "",
        }
        logger.warning(
            "json_decode_error",
            extra=error,
        )
        return None, error

    # Extract required fields
    competitor_product_id = data.get("competitor_product_id")
    competitor = data.get("competitor")

    # Try pricing first, then fall back to api_product.price (item_minimal format)
    price, currency, list_price = extract_price(data.get("pricing"))
    if price is None:
        api_product = data.get("api_product", {})
        api_price = api_product.get("price")
        if isinstance(api_price, dict):
            # api_product.price has same structure as nested pricing.price
            price, currency, list_price = extract_price({"price": api_price})

    scraped_at = parse_scraped_at(data.get("scraped_at"))

    # Validate required fields
    missing_fields = []
    if not competitor_product_id:
        missing_fields.append("competitor_product_id")
    if not competitor:
        missing_fields.append("competitor")
    if price is None:
        missing_fields.append("price")
    if scraped_at is None:
        missing_fields.append("scraped_at")

    if missing_fields:
        error = {
            "line_number": line_number,
            "error_type": "missing_required_fields",
            "missing_fields": missing_fields,
            "competitor_product_id": competitor_product_id,
            "competitor": competitor,
        }
        # Don't log per-line - summary logged in parse_jsonl
        return None, error

    # Extract optional fields, falling back to api_product for item_minimal format
    api_product = data.get("api_product", {})
    product_name = data.get("product_name") or api_product.get("name")
    brand = data.get("brand") or api_product.get("brand")
    inferred_currency = infer_currency_from_competitor(str(competitor) if competitor else None)

    # Create ProductRecord with all available data
    record = ProductRecord(
        competitor_product_id=str(competitor_product_id),
        competitor=str(competitor),
        price=price,
        currency=currency or inferred_currency,
        scraped_at=scraped_at,
        product_name=product_name,
        brand=brand,
        availability_status=extract_availability_status(data.get("availability")),
        scrape_run_id=data.get("scrape_run_id"),
        product_url=data.get("product_url"),
        mpn=data.get("mpn"),
        ean=data.get("ean"),
        list_price=list_price,
        raw_data=data,
    )

    return record, None


def parse_jsonl(content: str, source_path: str | None = None) -> ParseResult:
    """Parse JSONL content into ProductRecords.

    Args:
        content: Raw JSONL content (multiple lines).
        source_path: Optional source path for logging context.

    Returns:
        ParseResult with records, stats, and errors.
    """
    records: list[ProductRecord] = []
    errors: list[dict[str, Any]] = []
    total_lines = 0

    for line_number, line in enumerate(content.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue

        total_lines += 1
        record, error = parse_line(line, line_number)

        if record:
            records.append(record)
        if error:
            errors.append(error)

    result = ParseResult(
        records=records,
        total_lines=total_lines,
        successful=len(records),
        failed=len(errors),
        errors=errors[:100],  # Keep first 100 errors for debugging, don't store all
    )

    # Log summary (not per-line)
    if result.failed > 0:
        # Count error types
        missing_fields_count = sum(
            1 for e in errors if e.get("error_type") == "missing_required_fields"
        )
        json_errors_count = sum(1 for e in errors if e.get("error_type") == "json_decode_error")

        logger.warning(
            "parse_errors_summary",
            extra={
                "source_path": source_path,
                "total_errors": result.failed,
                "missing_required_fields": missing_fields_count,
                "json_decode_errors": json_errors_count,
            },
        )

    logger.info(
        "jsonl_parsed",
        extra={
            "source_path": source_path,
            "total_lines": result.total_lines,
            "successful": result.successful,
            "failed": result.failed,
            "success_rate": (
                f"{(result.successful / result.total_lines * 100):.1f}%"
                if result.total_lines > 0
                else "N/A"
            ),
        },
    )

    return result


async def parse_jsonl_async(content: str, source_path: str | None = None) -> ParseResult:
    """Async wrapper for parse_jsonl (for API consistency).

    Args:
        content: Raw JSONL content.
        source_path: Optional source path for logging.

    Returns:
        ParseResult with parsed records.
    """
    return parse_jsonl(content, source_path)
