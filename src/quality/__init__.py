"""Data Quality Module - Run-level health metrics and validation.

Purpose: Monitor the health of scrape runs to enable safety decisions
(e.g., "suppress all price updates from unhealthy runs").

Key Components:
- run_health.py: Calculate run-level health metrics and composite scores

Dependencies: src/ingestion/parser.py (ProductRecord, ParseResult)
Consumed by: src/services/refinery.py, src/signals/trusted_prices.py

Module TODOs:
    - [x] Implement RunHealthCalculator with composite scoring
    - [ ] Add drift detection vs historical runs
    - [ ] Add content drift aggregation (requires embeddings integration)
"""

from src.quality.run_health import (
    RunHealth,
    RunHealthCalculator,
    WarningFlag,
)

__all__ = [
    "RunHealth",
    "RunHealthCalculator",
    "WarningFlag",
]
