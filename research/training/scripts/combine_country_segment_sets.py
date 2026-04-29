#!/usr/bin/env python3
"""Combine per-competitor training extracts into country+segment datasets.

Reads files from by_competitor/{country}/{segment}/*.parquet and writes
by_country_segment/{COUNTRY}_{SEGMENT}_{date}.parquet for training and tuning.

Usage:
    # Combine all country+segment outputs
    python research/training/scripts/combine_country_segment_sets.py

    # Limit to a single country+segment
    python research/training/scripts/combine_country_segment_sets.py --country DK --segment B2C

    # Custom input/output roots
    python research/training/scripts/combine_country_segment_sets.py \
        --input data/training/by_competitor \
        --output data/training/by_country_segment
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install pyarrow")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

EXPECTED_SEGMENTS = ("B2C", "B2B")
BATCH_SIZE = 50000


def resolve_path(path: str, project_root: Path) -> Path:
    """Resolve a possibly relative local path."""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = project_root / resolved
    return resolved.resolve()


def list_parquet_files(base_path: Path) -> list[Path]:
    """List parquet files under a local directory."""
    if not base_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {base_path}")
    return sorted(base_path.rglob("*.parquet"))


def parse_country_segment(file_path: Path) -> tuple[str, str] | None:
    """Extract country and segment from a by_competitor path."""
    parts = [part.upper() for part in file_path.parts]
    lowered = [part.lower() for part in file_path.parts]

    if "by_competitor" in lowered:
        idx = lowered.index("by_competitor")
        if len(parts) >= idx + 3:
            return parts[idx + 1], parts[idx + 2]

    if len(parts) >= 3:
        return parts[-3], parts[-2]

    return None


def align_table_to_schema(table: pa.Table, schema: pa.Schema) -> pa.Table:
    """Align table columns to the target schema."""
    if table.schema.equals(schema):
        return table

    arrays = []
    for field in schema:
        if field.name in table.column_names:
            column = table[field.name]
            if not column.type.equals(field.type):
                column = column.cast(field.type)
        else:
            column = pa.nulls(table.num_rows, type=field.type)
        arrays.append(column)

    return pa.Table.from_arrays(arrays, schema=schema)


def iter_tables(file_path: Path, batch_size: int) -> Iterable[pa.Table]:
    """Yield tables from a parquet file in batches."""
    parquet_file = pq.ParquetFile(file_path)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        yield pa.Table.from_batches([batch])


def get_row_count(file_path: Path) -> int:
    """Return row count from parquet metadata."""
    parquet_file = pq.ParquetFile(file_path)
    return parquet_file.metadata.num_rows if parquet_file.metadata else 0


def ensure_output_dir(base_path: Path) -> None:
    """Create output directory if needed."""
    base_path.mkdir(parents=True, exist_ok=True)


def combine_group(
    group_key: tuple[str, str],
    files: list[Path],
    output_path: Path,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, object]:
    """Combine parquet files for a single country+segment group."""
    country, segment = group_key
    total_rows = 0
    total_files = len(files)

    if dry_run:
        for file_path in files:
            total_rows += get_row_count(file_path)
        return {
            "country": country,
            "segment": segment,
            "files": total_files,
            "rows": total_rows,
            "output_path": str(output_path),
            "skipped": False,
        }

    if output_path.exists():
        if not overwrite:
            logger.warning("Output exists, skipping: %s", output_path)
            return {
                "country": country,
                "segment": segment,
                "files": total_files,
                "rows": 0,
                "output_path": str(output_path),
                "skipped": True,
            }
        output_path.unlink(missing_ok=True)

    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None

    for file_path in files:
        for table in iter_tables(file_path, BATCH_SIZE):
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(output_path, schema, compression="snappy")
            elif schema is not None:
                table = align_table_to_schema(table, schema)

            if writer is not None:
                writer.write_table(table)
                total_rows += table.num_rows

    if writer is not None:
        writer.close()

    return {
        "country": country,
        "segment": segment,
        "files": total_files,
        "rows": total_rows,
        "output_path": str(output_path),
        "skipped": False,
    }


def validate_date(date_str: str) -> str:
    """Validate date string format."""
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        raise ValueError("Date must be in YYYY-MM-DD format")
    return date_str


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine per-competitor training data into country+segment sets"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/training/by_competitor",
        help="Input directory containing by_competitor files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/training/by_country_segment",
        help="Output directory for by_country_segment files",
    )
    parser.add_argument(
        "--country",
        type=str,
        help="Limit to a single country identifier (default: all discovered countries)",
    )
    parser.add_argument(
        "--segment",
        type=str,
        choices=EXPECTED_SEGMENTS,
        help="Limit to a single segment (default: all)",
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Date string for output filenames (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned outputs without writing files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    input_path = resolve_path(args.input, project_root)
    output_path = resolve_path(args.output, project_root)
    ensure_output_dir(output_path)

    date_str = validate_date(args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    files = list_parquet_files(input_path)
    if not files:
        logger.error("No parquet files found under %s", input_path)
        sys.exit(1)

    country_filter = args.country.upper() if args.country else None
    segment_filter = args.segment.upper() if args.segment else None

    grouped: dict[tuple[str, str], list[Path]] = {}
    skipped = 0

    for file_path in files:
        parsed = parse_country_segment(file_path)
        if not parsed:
            skipped += 1
            logger.warning("Skipping file without country/segment: %s", file_path)
            continue

        country, segment = parsed
        if country_filter and country != country_filter:
            continue
        if segment_filter and segment != segment_filter:
            continue

        grouped.setdefault((country, segment), []).append(file_path)

    if not grouped:
        logger.warning("No files matched the requested filters.")
        sys.exit(0)

    matched_files = sum(len(group) for group in grouped.values())
    logger.info(
        "Matched %d files across %d country+segment groups (scanned %d)",
        matched_files,
        len(grouped),
        len(files),
    )

    results = []
    for (country, segment), group_files in sorted(grouped.items()):
        output_file_path = output_path / f"{country}_{segment}_{date_str}.parquet"
        logger.info(
            "Combining %d files for %s_%s -> %s",
            len(group_files),
            country,
            segment,
            output_file_path,
        )
        result = combine_group(
            group_key=(country, segment),
            files=group_files,
            output_path=output_file_path,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
        results.append(result)

    expected = set(grouped)
    produced = {(r["country"], r["segment"]) for r in results}
    missing = sorted(expected - produced)

    print("\n" + "=" * 70)
    print("COMBINE SUMMARY")
    print("=" * 70)
    print(f"Input base: {input_path}")
    print(f"Output base: {output_path}")
    if skipped:
        print(f"Skipped (unparsed) files: {skipped}")
    print(f"Dry run: {args.dry_run}")
    print()

    for result in results:
        status = "SKIPPED" if result["skipped"] else "OK"
        print(
            f"{result['country']}_{result['segment']}: "
            f"{result['files']} files, {result['rows']:,} rows, {status}"
        )
        print(f"  -> {result['output_path']}")

    if missing:
        print("\nMissing expected combos:")
        for country, segment in missing:
            print(f"  - {country}_{segment}")
    else:
        print("\nAll expected combos produced.")

    print("\nDone!")


if __name__ == "__main__":
    main()
