"""Shared constants used across parsing and anomaly modules."""

COUNTRY_CURRENCY_MAP: dict[str, str] = {
    "DK": "DKK",
    "SE": "SEK",
    "NO": "NOK",
    "FI": "EUR",
}


def extract_country_code_from_competitor(competitor: str | None) -> str | None:
    """Extract a country code from a competitor identifier.

    Args:
        competitor: Competitor identifier like ``PROSHOP_DK``.

    Returns:
        Two-letter uppercase country code when available, else None.
    """
    if not competitor or "_" not in competitor:
        return None

    parts = competitor.rsplit("_", 1)
    if len(parts) != 2:
        return None

    country_code = parts[1].strip().upper()
    if not country_code:
        return None
    return country_code


def infer_currency_from_competitor(competitor: str | None) -> str | None:
    """Infer currency from competitor suffix country code.

    Args:
        competitor: Competitor identifier like ``PROSHOP_DK``.

    Returns:
        Currency code such as ``DKK``, or None when unknown.
    """
    country_code = extract_country_code_from_competitor(competitor)
    if country_code is None:
        return None
    return COUNTRY_CURRENCY_MAP.get(country_code)
