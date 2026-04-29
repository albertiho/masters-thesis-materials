"""Prepare bundled anonymized parquet data for local thesis reproduction.

This module defines the Phase 1 public data-preparation surface:

    python -m src.research.prepare_data --data-root data/training

The command validates the bundled anonymized source contract under
``data/training/source/`` and deterministically regenerates all derived parquet
artifacts under ``data/training/derived/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

from src.tuning_config import TuningConfig, get_tuning_config

logger = logging.getLogger(__name__)

MANIFEST_VERSION = 1
DATASET_SCHEMA_VERSION = "phase1-v1"
DATASET_MANIFEST_NAME = "dataset_manifest.json"
CHECKSUMS_FILE_NAME = "SHA256SUMS"
SPLIT_MANIFEST_NAME = "split_manifest.json"

SOURCE_DIR_NAME = "source"
DERIVED_DIR_NAME = "derived"
BY_COMPETITOR_DIR = "by_competitor"
BY_COUNTRY_SEGMENT_DIR = "by_country_segment"

DATE_TOKEN_PATTERN = re.compile(r"_(\d{4}-\d{2}-\d{2})(?:_|\.|$)")
TIME_COLUMNS = ("first_seen_at", "scraped_at", "timestamp")


@dataclass(frozen=True)
class SourceFileRecord:
    """Describe one bundled source parquet file."""

    relative_path: str
    country: str
    segment: str
    competitor: str
    date_token: str
    row_count: int
    size_bytes: int
    schema_columns: list[str]
    sha256: str


def _project_root() -> Path:
    """Return the repository root for the standalone training package."""
    return Path(__file__).resolve().parents[2]


def _resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the current working directory."""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    return resolved.resolve()


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic JSON to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    """Compute a SHA256 checksum for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_date_token(filename: str) -> str:
    """Extract the YYYY-MM-DD token from a parquet filename."""
    match = DATE_TOKEN_PATTERN.search(Path(filename).name)
    if not match:
        raise ValueError(f"Missing YYYY-MM-DD token in filename: {filename}")
    return match.group(1)


def _parse_source_record(source_root: Path, parquet_path: Path) -> SourceFileRecord:
    """Create a manifest record for a bundled source parquet file."""
    relative_path = parquet_path.relative_to(source_root)
    if relative_path.parts[0] != BY_COMPETITOR_DIR or len(relative_path.parts) < 4:
        raise ValueError(f"Unexpected source parquet path: {relative_path.as_posix()}")

    country = relative_path.parts[1]
    segment = relative_path.parts[2]
    filename = relative_path.name
    date_token = _extract_date_token(filename)
    competitor = parquet_path.stem.removesuffix(f"_{date_token}")

    metadata = pq.ParquetFile(parquet_path).metadata
    row_count = metadata.num_rows if metadata else 0
    schema_columns = list(pq.read_schema(parquet_path).names)

    return SourceFileRecord(
        relative_path=relative_path.as_posix(),
        country=country,
        segment=segment,
        competitor=competitor,
        date_token=date_token,
        row_count=row_count,
        size_bytes=parquet_path.stat().st_size,
        schema_columns=schema_columns,
        sha256=_sha256(parquet_path),
    )


def discover_source_files(source_root: Path) -> list[SourceFileRecord]:
    """Discover bundled anonymized source parquet files."""
    parquet_paths = sorted((source_root / BY_COMPETITOR_DIR).rglob("*.parquet"))
    return [_parse_source_record(source_root, parquet_path) for parquet_path in parquet_paths]


def _country_segment_output_path(country: str, segment: str, date_token: str) -> str:
    return f"{BY_COUNTRY_SEGMENT_DIR}/{country}_{segment}_{date_token}.parquet"


def _split_output_paths(base_relative_path: str, min_history: int) -> list[str]:
    base = base_relative_path.removesuffix(".parquet")
    suffix = f"_mh{min_history}"
    return [
        f"{base}_train{suffix}.parquet",
        f"{base}_test_new_prices{suffix}.parquet",
        f"{base}_test_new_products{suffix}.parquet",
    ]


def _min_history_variants(config: TuningConfig) -> list[int]:
    """Return the public split variants for Phase 1."""
    return sorted(
        {
            config.minimum_history.autoencoder,
            config.minimum_history.isolation_forest,
        }
    )


def build_expected_derived_outputs(
    records: list[SourceFileRecord], config: TuningConfig
) -> dict[str, list[str]]:
    """Build the deterministic expected derived parquet inventory."""
    by_competitor_outputs = sorted(record.relative_path for record in records)

    by_country_segment_outputs = sorted(
        {
            _country_segment_output_path(record.country, record.segment, record.date_token)
            for record in records
        }
    )

    split_outputs: list[str] = []
    for relative_path in by_competitor_outputs + by_country_segment_outputs:
        for min_history in _min_history_variants(config):
            split_outputs.extend(_split_output_paths(relative_path, min_history))

    return {
        "by_competitor": by_competitor_outputs,
        "by_country_segment": by_country_segment_outputs,
        "splits": sorted(split_outputs),
    }


def build_dataset_manifest_payload(
    source_root: Path,
    records: list[SourceFileRecord],
    config: TuningConfig,
) -> dict[str, Any]:
    """Build the committed source dataset manifest."""
    return {
        "manifest_version": MANIFEST_VERSION,
        "schema_version": DATASET_SCHEMA_VERSION,
        "source_root": f"data/training/{SOURCE_DIR_NAME}",
        "derived_root": f"data/training/{DERIVED_DIR_NAME}",
        "total_files": len(records),
        "total_rows": sum(record.row_count for record in records),
        "total_size_bytes": sum(record.size_bytes for record in records),
        "date_tokens": sorted({record.date_token for record in records}),
        "segments": sorted({record.segment for record in records}),
        "files": [asdict(record) for record in records],
        "expected_derived_outputs": build_expected_derived_outputs(records, config),
    }


def write_dataset_contract_files(
    source_root: str | Path,
    config_path: str | None = None,
) -> tuple[Path, Path]:
    """Write ``dataset_manifest.json`` and ``SHA256SUMS`` for a source tree."""
    resolved_source_root = _resolve_path(source_root)
    tuning_config = get_tuning_config(config_path)
    records = discover_source_files(resolved_source_root)
    payload = build_dataset_manifest_payload(resolved_source_root, records, tuning_config)

    manifest_path = resolved_source_root / DATASET_MANIFEST_NAME
    checksum_path = resolved_source_root / CHECKSUMS_FILE_NAME

    _json_dump(manifest_path, payload)
    checksum_lines = [
        f"{record.sha256}  {record.relative_path}"
        for record in records
    ]
    checksum_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return manifest_path, checksum_path


def _load_dataset_manifest(source_root: Path) -> dict[str, Any]:
    manifest_path = source_root / DATASET_MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _load_checksums(source_root: Path) -> dict[str, str]:
    checksum_path = source_root / CHECKSUMS_FILE_NAME
    if not checksum_path.exists():
        raise FileNotFoundError(f"Missing checksum file: {checksum_path}")

    checksums: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        digest, relative_path = stripped.split("  ", maxsplit=1)
        checksums[relative_path] = digest
    return checksums


def validate_source_contract(source_root: str | Path, config_path: str | None = None) -> list[SourceFileRecord]:
    """Validate the committed manifest and checksums against actual parquet files."""
    resolved_source_root = _resolve_path(source_root)
    tuning_config = get_tuning_config(config_path)
    actual_records = discover_source_files(resolved_source_root)
    actual_manifest = build_dataset_manifest_payload(resolved_source_root, actual_records, tuning_config)
    committed_manifest = _load_dataset_manifest(resolved_source_root)

    if committed_manifest != actual_manifest:
        raise ValueError("dataset_manifest.json does not match the bundled source parquet inventory")

    committed_checksums = _load_checksums(resolved_source_root)
    actual_checksums = {record.relative_path: record.sha256 for record in actual_records}
    if committed_checksums != actual_checksums:
        raise ValueError("SHA256SUMS does not match the bundled source parquet files")

    return actual_records


def _align_table_to_schema(table: pa.Table, schema: pa.Schema) -> pa.Table:
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


def _reset_derived_root(derived_root: Path) -> None:
    if derived_root.exists():
        shutil.rmtree(derived_root)
    derived_root.mkdir(parents=True, exist_ok=True)


def _copy_source_to_derived(
    source_root: Path,
    derived_root: Path,
    records: list[SourceFileRecord],
) -> list[str]:
    copied_relative_paths: list[str] = []
    for record in records:
        source_path = source_root / record.relative_path
        target_path = derived_root / record.relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        copied_relative_paths.append(record.relative_path)
    return sorted(copied_relative_paths)


def _combine_country_segment_sets(
    source_root: Path,
    derived_root: Path,
    records: list[SourceFileRecord],
) -> list[str]:
    grouped_paths: dict[tuple[str, str, str], list[Path]] = {}
    for record in records:
        key = (record.country, record.segment, record.date_token)
        grouped_paths.setdefault(key, []).append(source_root / record.relative_path)

    output_paths: list[str] = []
    country_segment_root = derived_root / BY_COUNTRY_SEGMENT_DIR
    country_segment_root.mkdir(parents=True, exist_ok=True)

    for (country, segment, date_token), input_paths in sorted(grouped_paths.items()):
        output_relative = _country_segment_output_path(country, segment, date_token)
        output_path = derived_root / output_relative

        schema: pa.Schema | None = None
        tables: list[pa.Table] = []
        for input_path in sorted(input_paths):
            table = pq.read_table(input_path)
            if schema is None:
                schema = table.schema
            else:
                table = _align_table_to_schema(table, schema)
            tables.append(table)

        if not tables:
            raise ValueError(f"No input tables found for {country}_{segment}_{date_token}")

        combined = pa.concat_tables(tables, promote_options="none")
        pq.write_table(combined, output_path, compression="snappy")
        output_paths.append(output_relative)

    return sorted(output_paths)


def _time_column(df: pd.DataFrame, filepath: Path) -> str:
    for column in TIME_COLUMNS:
        if column in df.columns:
            return column
    raise ValueError(f"No supported time column found in {filepath.name}")


def _split_one_file(
    input_path: Path,
    derived_root: Path,
    test_size: float,
    last_n: int,
    min_history: int,
    random_state: int,
) -> dict[str, Any]:
    df = pd.read_parquet(input_path)
    total_rows = len(df)
    total_products = int(df["product_id"].nunique())
    time_column = _time_column(df, input_path)

    df = df.sort_values(["product_id", time_column])
    product_counts = df.groupby("product_id").size()
    valid_products = product_counts[product_counts >= min_history].index.tolist()
    df_valid = df[df["product_id"].isin(valid_products)].copy()

    if len(valid_products) < 2:
        raise ValueError(
            f"{input_path.name} has only {len(valid_products)} products with >= {min_history} observations"
        )

    sorted_product_ids = sorted(valid_products, key=str)
    train_product_ids, test_product_ids = train_test_split(
        sorted_product_ids,
        test_size=test_size,
        random_state=random_state,
    )

    test_new_products_df = df_valid[df_valid["product_id"].isin(test_product_ids)].copy()
    train_products_df = df_valid[df_valid["product_id"].isin(train_product_ids)].copy()
    train_products_df["_row_num"] = train_products_df.groupby("product_id").cumcount()
    train_products_df["_group_size"] = train_products_df.groupby("product_id")["product_id"].transform("count")

    new_prices_mask = train_products_df["_row_num"] >= (train_products_df["_group_size"] - last_n)
    train_df = train_products_df[~new_prices_mask].drop(columns=["_row_num", "_group_size"])
    test_new_prices_df = train_products_df[new_prices_mask].drop(columns=["_row_num", "_group_size"])

    base_relative = input_path.relative_to(derived_root).as_posix().removesuffix(".parquet")
    train_relative = f"{base_relative}_train_mh{min_history}.parquet"
    new_prices_relative = f"{base_relative}_test_new_prices_mh{min_history}.parquet"
    new_products_relative = f"{base_relative}_test_new_products_mh{min_history}.parquet"

    train_df.to_parquet(derived_root / train_relative, index=False, compression="snappy")
    test_new_prices_df.to_parquet(derived_root / new_prices_relative, index=False, compression="snappy")
    test_new_products_df.to_parquet(derived_root / new_products_relative, index=False, compression="snappy")

    return {
        "input_file": input_path.relative_to(derived_root).as_posix(),
        "time_column": time_column,
        "min_history": min_history,
        "total_rows": total_rows,
        "total_products": total_products,
        "valid_products": len(valid_products),
        "excluded_products": total_products - len(valid_products),
        "generated_files": {
            "train": {
                "path": train_relative,
                "rows": len(train_df),
                "products": len(train_product_ids),
            },
            "test_new_prices": {
                "path": new_prices_relative,
                "rows": len(test_new_prices_df),
                "products": int(test_new_prices_df["product_id"].nunique()),
            },
            "test_new_products": {
                "path": new_products_relative,
                "rows": len(test_new_products_df),
                "products": len(test_product_ids),
            },
        },
    }


def _build_split_manifest(
    source_records: list[SourceFileRecord],
    copied_files: list[str],
    country_segment_files: list[str],
    split_jobs: list[dict[str, Any]],
    config: TuningConfig,
) -> dict[str, Any]:
    split_outputs = sorted(
        job["generated_files"][artifact]["path"]
        for job in split_jobs
        for artifact in ("train", "test_new_prices", "test_new_products")
    )

    return {
        "manifest_version": MANIFEST_VERSION,
        "schema_version": DATASET_SCHEMA_VERSION,
        "source_root": f"data/training/{SOURCE_DIR_NAME}",
        "derived_root": f"data/training/{DERIVED_DIR_NAME}",
        "source_files": [record.relative_path for record in source_records],
        "generated_files": {
            "by_competitor": copied_files,
            "by_country_segment": country_segment_files,
            "splits": split_outputs,
        },
        "min_history_by_model": {
            "autoencoder": config.minimum_history.autoencoder,
            "isolation_forest": config.minimum_history.isolation_forest,
            "statistical": config.minimum_history.statistical,
        },
        "split_variants": _min_history_variants(config),
        "split_parameters": {
            "test_size": config.data_splitting.test_size,
            "last_n": config.data_splitting.test_split_amount_of_prices,
        },
        "seed_values": {
            "random_state": config.data_splitting.random_state,
        },
        "split_jobs": split_jobs,
    }


def prepare_dataset(data_root: str | Path, config_path: str | None = None) -> dict[str, Any]:
    """Prepare deterministic derived parquet artifacts from bundled source data."""
    resolved_data_root = _resolve_path(data_root)
    source_root = resolved_data_root / SOURCE_DIR_NAME
    derived_root = resolved_data_root / DERIVED_DIR_NAME

    tuning_config = get_tuning_config(config_path)
    source_records = validate_source_contract(source_root, config_path=config_path)

    _reset_derived_root(derived_root)
    copied_files = _copy_source_to_derived(source_root, derived_root, source_records)
    country_segment_files = _combine_country_segment_sets(source_root, derived_root, source_records)

    base_input_files = sorted(
        [
            *(derived_root / relative_path for relative_path in copied_files),
            *(derived_root / relative_path for relative_path in country_segment_files),
        ],
        key=lambda path: path.relative_to(derived_root).as_posix(),
    )

    split_jobs: list[dict[str, Any]] = []
    for input_path in base_input_files:
        for min_history in _min_history_variants(tuning_config):
            split_jobs.append(
                _split_one_file(
                    input_path=input_path,
                    derived_root=derived_root,
                    test_size=tuning_config.data_splitting.test_size,
                    last_n=tuning_config.data_splitting.test_split_amount_of_prices,
                    min_history=min_history,
                    random_state=tuning_config.data_splitting.random_state,
                )
            )

    split_manifest = _build_split_manifest(
        source_records=source_records,
        copied_files=copied_files,
        country_segment_files=country_segment_files,
        split_jobs=split_jobs,
        config=tuning_config,
    )
    _json_dump(derived_root / SPLIT_MANIFEST_NAME, split_manifest)

    return {
        "source_files": len(source_records),
        "copied_files": len(copied_files),
        "country_segment_files": len(country_segment_files),
        "split_jobs": len(split_jobs),
        "split_files": len(split_manifest["generated_files"]["splits"]),
        "derived_root": str(derived_root),
    }


def main() -> None:
    """Command-line entry point for deterministic local data preparation."""
    parser = argparse.ArgumentParser(
        description="Prepare anonymized bundled parquet data for local thesis reproduction",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/training",
        help="Training data root containing source/ and derived/ (default: data/training)",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Optional path to tuning_config.json",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    result = prepare_dataset(data_root=args.data_root, config_path=args.config_path)

    print("=" * 70)
    print("Prepared Local Training Data")
    print("=" * 70)
    print(f"Source parquet files: {result['source_files']}")
    print(f"Copied by_competitor files: {result['copied_files']}")
    print(f"Generated by_country_segment files: {result['country_segment_files']}")
    print(f"Split jobs executed: {result['split_jobs']}")
    print(f"Generated split parquet files: {result['split_files']}")
    print(f"Derived root: {result['derived_root']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
