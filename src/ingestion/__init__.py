"""Minimal ingestion exports for the standalone training repository.

The training and evaluation code only needs the legacy parser dataclasses and
JSONL parsing helpers. The production-only readers, schema parsers, and
writers are intentionally omitted here to keep the release subset independent
from the full refinery service stack.
"""

from src.ingestion.parser import ParseResult, ProductRecord, parse_jsonl, parse_jsonl_async

__all__ = [
    "ProductRecord",
    "ParseResult",
    "parse_jsonl",
    "parse_jsonl_async",
]
