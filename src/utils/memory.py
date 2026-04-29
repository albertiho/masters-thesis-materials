"""Memory tracking utilities for OOM investigation.

Lightweight wrappers around psutil to log RSS at pipeline checkpoints.
Diagnostic-only — no behavior changes.
"""

import logging
import os

logger = logging.getLogger(__name__)

_last_rss_mb: float = 0.0
_peak_rss_mb: float = 0.0


def get_memory_mb() -> float:
    """Return current process RSS in megabytes."""
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def log_memory(label: str, **extra_fields: object) -> float:
    """Log current RSS, delta from last checkpoint, and peak RSS.

    Args:
        label: Human-readable checkpoint name (e.g. "model_load_complete").
        **extra_fields: Additional fields to include in the structured log.

    Returns:
        Current RSS in MB.
    """
    global _last_rss_mb, _peak_rss_mb

    current = get_memory_mb()
    delta = current - _last_rss_mb
    _peak_rss_mb = max(_peak_rss_mb, current)
    _last_rss_mb = current

    log_data: dict[str, object] = {
        "memory_label": label,
        "rss_mb": round(current, 1),
        "delta_mb": round(delta, 1),
        "peak_rss_mb": round(_peak_rss_mb, 1),
        "pipeline_stage": "memory_tracking",
    }
    log_data.update(extra_fields)

    logger.info("memory_checkpoint", extra=log_data)
    return current
