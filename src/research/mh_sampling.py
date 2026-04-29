"""Shared research-time minimum-history defaults."""

from __future__ import annotations

from typing import Sequence

RESEARCH_MH_VALUES = (5, 10, 15, 20, 25, 30)
RESEARCH_MH_LEVELS = tuple(f"mh{value}" for value in RESEARCH_MH_VALUES)


def normalize_mh_values(values: Sequence[int | str] | None = None) -> list[int]:
    """Normalize mh values while defaulting to the thesis research subset."""
    if not values:
        return list(RESEARCH_MH_VALUES)

    normalized: set[int] = set()
    for value in values:
        token = str(value).strip().lower()
        if not token:
            continue
        if token.startswith("mh"):
            token = token[2:]
        parsed = int(token)
        if parsed < 1:
            raise ValueError(f"Minimum-history values must be positive integers, got {value!r}")
        normalized.add(parsed)

    if not normalized:
        raise ValueError("At least one minimum-history value must be provided")
    return sorted(normalized)


def normalize_mh_levels(values: Sequence[int | str] | None = None) -> list[str]:
    """Normalize mh values into ``mhX`` level strings."""
    return [f"mh{value}" for value in normalize_mh_values(values)]
