"""Generate filtered dataset subsets by minimum product history."""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

BY_COMPETITOR_DIR_NAME = "by_competitor"
SPLIT_PATTERN = re.compile(r"_train[_.]|_test[_.]")


@dataclass(frozen=True)
class HistorySubsetArtifact:
    """One generated subset artifact for a source parquet file."""

    min_history: int
    rows: int
    products: int
    relative_output_path: str


@dataclass(frozen=True)
class SourceSubsetSummary:
    """Summary for one source parquet file and its derived subset artifacts."""

    relative_source_path: str
    total_rows: int
    total_products: int
    min_product_history: int
    max_product_history: int
    subsets: list[HistorySubsetArtifact]


def resolve_history_values(
    history_values: Sequence[int] | None,
    *,
    min_history_start: int,
    min_history_end: int,
) -> list[int]:
    """Resolve explicit or ranged minimum-history values."""
    if history_values:
        values = sorted({int(value) for value in history_values})
    else:
        if min_history_start > min_history_end:
            raise ValueError(
                "min_history_start must be less than or equal to min_history_end"
            )
        values = list(range(min_history_start, min_history_end + 1))

    if not values:
        raise ValueError("At least one minimum-history value must be provided")

    invalid_values = [value for value in values if value < 1]
    if invalid_values:
        raise ValueError(
            f"Minimum-history values must be positive integers: {invalid_values!r}"
        )
    return values


def discover_competitor_files(source_root: str | Path) -> list[Path]:
    """Discover canonical bundled competitor parquet files."""
    resolved_source_root = Path(source_root).resolve()
    if not resolved_source_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {resolved_source_root}")

    parquet_files = sorted(
        path
        for path in resolved_source_root.rglob("*.parquet")
        if not SPLIT_PATTERN.search(path.name)
    )
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under: {resolved_source_root}")
    return parquet_files


def _output_path_for_history(
    *,
    source_file: Path,
    source_root: Path,
    output_root: Path,
    min_history: int,
) -> Path:
    relative_source = source_file.relative_to(source_root)
    return output_root / f"mh{min_history}" / BY_COMPETITOR_DIR_NAME / relative_source


def _subset_summary_for_file(
    *,
    source_file: Path,
    source_root: Path,
    output_root: Path,
    history_values: Sequence[int],
    dry_run: bool,
    skip_existing: bool,
) -> tuple[SourceSubsetSummary, int, int]:
    df = pd.read_parquet(source_file)
    if "product_id" not in df.columns:
        raise ValueError(f"Missing product_id column in {source_file}")

    product_counts = df.groupby("product_id").size()
    history_per_row = df["product_id"].map(product_counts).fillna(0).astype("int32")

    total_rows = len(df)
    total_products = int(len(product_counts))
    min_product_history = int(product_counts.min()) if not product_counts.empty else 0
    max_product_history = int(product_counts.max()) if not product_counts.empty else 0
    relative_source_path = (
        Path(BY_COMPETITOR_DIR_NAME) / source_file.relative_to(source_root)
    ).as_posix()

    logger.info(
        "processing_source_file path=%s rows=%s products=%s min_history=%s max_history=%s",
        relative_source_path,
        total_rows,
        total_products,
        min_product_history,
        max_product_history,
    )

    subset_artifacts: list[HistorySubsetArtifact] = []
    created_subset_files = 0
    skipped_subset_files = 0

    for min_history in history_values:
        valid_products = int((product_counts >= min_history).sum())
        subset_df = df.loc[history_per_row >= min_history].copy()
        output_path = _output_path_for_history(
            source_file=source_file,
            source_root=source_root,
            output_root=output_root,
            min_history=min_history,
        )
        relative_output_path = output_path.relative_to(output_root).as_posix()

        if dry_run:
            logger.info(
                "dry_run_subset target=%s rows=%s products=%s",
                relative_output_path,
                len(subset_df),
                valid_products,
            )
        elif skip_existing and output_path.exists():
            skipped_subset_files += 1
            logger.info(
                "skipped_existing_subset target=%s rows=%s products=%s",
                relative_output_path,
                len(subset_df),
                valid_products,
            )
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            subset_df.to_parquet(output_path, index=False, compression="snappy")
            created_subset_files += 1
            logger.info(
                "created_subset target=%s rows=%s products=%s",
                relative_output_path,
                len(subset_df),
                valid_products,
            )

        subset_artifacts.append(
            HistorySubsetArtifact(
                min_history=min_history,
                rows=int(len(subset_df)),
                products=valid_products,
                relative_output_path=relative_output_path,
            )
        )

    return SourceSubsetSummary(
        relative_source_path=relative_source_path,
        total_rows=int(total_rows),
        total_products=total_products,
        min_product_history=min_product_history,
        max_product_history=max_product_history,
        subsets=subset_artifacts,
    ), created_subset_files, skipped_subset_files


def create_history_subsets(
    *,
    source_root: str | Path,
    output_root: str | Path,
    history_values: Sequence[int],
    dry_run: bool = False,
    skip_existing: bool = False,
) -> dict[str, Any]:
    """Create parquet subsets for each requested minimum-history threshold."""
    resolved_source_root = Path(source_root).resolve()
    resolved_output_root = Path(output_root).resolve()
    source_files = discover_competitor_files(resolved_source_root)

    file_summaries: list[SourceSubsetSummary] = []
    created_subset_files = 0
    skipped_subset_files = 0

    for source_file in source_files:
        summary, created_count, skipped_count = _subset_summary_for_file(
            source_file=source_file,
            source_root=resolved_source_root,
            output_root=resolved_output_root,
            history_values=history_values,
            dry_run=dry_run,
            skip_existing=skip_existing,
        )
        file_summaries.append(summary)
        created_subset_files += created_count
        skipped_subset_files += skipped_count

    return {
        "source_root": str(resolved_source_root),
        "output_root": str(resolved_output_root),
        "history_values": list(history_values),
        "dry_run": dry_run,
        "skip_existing": skip_existing,
        "files_processed": len(file_summaries),
        "generated_subset_files": 0 if dry_run else created_subset_files,
        "skipped_subset_files": 0 if dry_run else skipped_subset_files,
        "source_summaries": [asdict(summary) for summary in file_summaries],
    }
