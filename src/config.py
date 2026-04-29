"""Local configuration for the standalone thesis reproduction package.

The published `src/` repository is intentionally filesystem-only. This module
keeps only the environment and local runtime toggles that are still useful for
training, evaluation, and cached feature computation.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Application environment."""

    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


DEFAULT_PROCESSING_ENABLED = True

# ---------------------------------------------------------------------------
# Temporal Cache Configuration
# ---------------------------------------------------------------------------
COMPETITOR_CACHE_LIMITS: dict[str, int] = {
    "default": 20_000,
}
TEMPORAL_OBSERVATIONS_PER_PRODUCT: int = 30
DEFAULT_PRICE_PERSIST_HOURS: float = 24.0
DEFAULT_PRICE_PERSIST_THRESHOLD: float = 0.02


def get_price_persist_hours() -> float:
    """Get price persistence threshold from the environment."""
    env_value = os.environ.get("PRICE_PERSIST_HOURS")
    if env_value is not None:
        try:
            return float(env_value)
        except ValueError:
            logger.warning(
                "invalid_price_persist_hours",
                extra={"value": env_value, "using_default": DEFAULT_PRICE_PERSIST_HOURS},
            )
    return DEFAULT_PRICE_PERSIST_HOURS


def get_price_persist_threshold() -> float:
    """Get price persistence tolerance threshold from the environment."""
    env_value = os.environ.get("PRICE_PERSIST_THRESHOLD")
    if env_value is not None:
        try:
            threshold = float(env_value)
        except ValueError:
            threshold = -1.0

        if 0.0 <= threshold <= 1.0:
            return threshold

        logger.warning(
            "invalid_price_persist_threshold",
            extra={"value": env_value, "using_default": DEFAULT_PRICE_PERSIST_THRESHOLD},
        )

    return DEFAULT_PRICE_PERSIST_THRESHOLD


def get_competitor_cache_limit(competitor: str) -> int:
    """Get the cache limit for a specific competitor."""
    return COMPETITOR_CACHE_LIMITS.get(competitor, COMPETITOR_CACHE_LIMITS["default"])


@dataclass(frozen=True)
class Config:
    """Application configuration for local execution."""

    environment: Environment
    processing_enabled: bool

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PROD

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEV

    @property
    def is_sleeping(self) -> bool:
        """Check if processing is disabled."""
        return not self.processing_enabled

    def __str__(self) -> str:
        """Human-readable config summary."""
        sleep_status = " [SLEEPING]" if self.is_sleeping else ""
        return f"Config(env={self.environment.value}{sleep_status})"


def _parse_environment(env_str: str) -> Environment:
    """Parse environment string to enum."""
    try:
        return Environment(env_str.lower())
    except ValueError:
        valid = [e.value for e in Environment]
        raise ValueError(f"Invalid ENVIRONMENT '{env_str}'. Must be one of: {valid}") from None


def _parse_bool(value: str | None, default: bool) -> bool:
    """Parse a boolean from an environment variable string."""
    if not value:
        return default
    return value.lower() in ("true", "1", "yes", "on")


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Get the cached local configuration."""
    environment = _parse_environment(os.environ.get("ENVIRONMENT", "dev"))
    processing_enabled = _parse_bool(
        os.environ.get("PROCESSING_ENABLED"),
        DEFAULT_PROCESSING_ENABLED,
    )

    config = Config(
        environment=environment,
        processing_enabled=processing_enabled,
    )

    logger.info(
        "config_loaded",
        extra={
            "environment": environment.value,
            "processing_enabled": processing_enabled,
        },
    )
    return config


def reset_config() -> None:
    """Reset the cached config."""
    get_config.cache_clear()


config = get_config()
