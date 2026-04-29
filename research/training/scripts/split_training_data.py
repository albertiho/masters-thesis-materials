#!/usr/bin/env python3
"""Run the local dataset-preparation flow with hardcoded workspace paths.

The flow is:

1. Create an ``mh5`` copy-through dataset from ``cleaned-data/training``.
2. Create the thesis-sampled history subsets from ``cleaned-data/training``.
3. Create derived ``by_country``, ``by_country_market``, and ``global`` datasets
   under each processed dataset root.
4. Create suffixless train/test split files for ``cleaned-data/training`` and
   the selected ``src/data-subsets/mh*`` dataset roots.
"""

from __future__ import annotations

import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from src.research.history_subsets import create_history_subsets
from src.research.mh_sampling import RESEARCH_MH_VALUES, normalize_mh_values

try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from sklearn.model_selection import train_test_split
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install pandas pyarrow scikit-learn")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TRAINING_REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = TRAINING_REPO_ROOT.parent
CLEANED_DATA_ROOT = WORKSPACE_ROOT / "cleaned-data" / "training"
DATA_SUBSETS_ROOT = TRAINING_REPO_ROOT / "data-subsets"

SOURCE_DATA_SUBDIR = "by_competitor"
DISCOVERABLE_DATA_SUBDIRS = (
    "by_country",
    "by_country_market",
    "by_country_segment",
    SOURCE_DATA_SUBDIR,
    "global",
)
SPLIT_PATTERN = re.compile(r"_train[_.]|_test[_.]")
DATE_TOKEN_PATTERN = re.compile(r"_(\d{4}-\d{2}-\d{2})(?:_|\.|$)")
TIME_COLUMNS = ("first_seen_at", "scraped_at", "timestamp")
TEST_SIZE = 0.2
LAST_N = 2
RANDOM_STATE = 42
ORIGINAL_DATA_VARIANT_HISTORY = 5


def find_parquet_files(
    data_dir: Path,
    subdirs: tuple[str, ...] = DISCOVERABLE_DATA_SUBDIRS,
) -> list[Path]:
    """Find unsplit Parquet files under the configured dataset subdirectories."""
    files: list[Path] = []

    for subdir_name in subdirs:
        subdir_path = data_dir / subdir_name
        if not subdir_path.exists():
            continue

        for parquet_file in subdir_path.rglob("*.parquet"):
            if SPLIT_PATTERN.search(parquet_file.name):
                continue
            files.append(parquet_file)

    return sorted(files)


def find_competitor_source_files(data_dir: Path) -> list[Path]:
    """Find unsplit source Parquet files under by_competitor only."""
    return find_parquet_files(data_dir, subdirs=(SOURCE_DATA_SUBDIR,))


def discover_dataset_roots(
    cleaned_data_root: Path = CLEANED_DATA_ROOT,
    data_subsets_root: Path = DATA_SUBSETS_ROOT,
    history_values: list[int] | None = None,
) -> list[Path]:
    """Discover the hardcoded dataset roots that should be processed."""
    dataset_roots: list[Path] = []

    if cleaned_data_root.exists():
        dataset_roots.append(cleaned_data_root)
    else:
        logger.warning("Cleaned-data root does not exist: %s", cleaned_data_root)

    if data_subsets_root.exists():
        selected_history_values = (
            default_history_values() if history_values is None else normalize_mh_values(history_values)
        )
        selected_history_values = sorted({ORIGINAL_DATA_VARIANT_HISTORY, *selected_history_values})
        selected_history_levels = {
            f"mh{value}"
            for value in selected_history_values
        }
        subset_roots = sorted(
            [
                path
                for path in data_subsets_root.iterdir()
                if path.is_dir()
                and re.fullmatch(r"mh\d+", path.name)
                and path.name in selected_history_levels
            ],
            key=lambda path: int(path.name[2:]),
        )
        dataset_roots.extend(subset_roots)
    else:
        logger.warning("Data-subsets root does not exist: %s", data_subsets_root)

    return dataset_roots


def default_history_values() -> list[int]:
    """Return the thesis-sampled mh values used in the research."""
    return list(RESEARCH_MH_VALUES)


def resolved_filtered_history_values(history_values: list[int] | None = None) -> list[int]:
    """Resolve filtered history values while reserving mh5 for copied original data."""
    resolved_history_values = normalize_mh_values(history_values) if history_values else default_history_values()
    return [value for value in resolved_history_values if value != ORIGINAL_DATA_VARIANT_HISTORY]


def ensure_original_data_variant(
    cleaned_data_root: Path = CLEANED_DATA_ROOT,
    data_subsets_root: Path = DATA_SUBSETS_ROOT,
    min_history: int = ORIGINAL_DATA_VARIANT_HISTORY,
) -> dict[str, Any]:
    """Copy the original unsplit Parquet files into the mh5 dataset root."""
    output_root = data_subsets_root / f"mh{min_history}"

    if not cleaned_data_root.exists():
        logger.warning("Original data root does not exist: %s", cleaned_data_root)
        return {
            "source_root": str(cleaned_data_root),
            "output_root": str(output_root),
            "variant_history": min_history,
            "files_processed": 0,
            "copied_files": 0,
            "skipped_files": 0,
        }

    source_files = find_competitor_source_files(cleaned_data_root)
    if not source_files:
        logger.info("No original Parquet files found under: %s", cleaned_data_root)
        return {
            "source_root": str(cleaned_data_root),
            "output_root": str(output_root),
            "variant_history": min_history,
            "files_processed": 0,
            "copied_files": 0,
            "skipped_files": 0,
        }

    logger.info("Ensuring original-data variant under: %s", output_root)
    copied_files = 0
    skipped_files = 0

    for source_file in source_files:
        output_path = output_root / source_file.relative_to(cleaned_data_root)
        if output_path.exists():
            skipped_files += 1
            logger.info("  Skipping existing mh%s copy: %s", min_history, output_path.name)
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, output_path)
        copied_files += 1
        logger.info("  Copied mh%s source: %s", min_history, output_path.name)

    return {
        "source_root": str(cleaned_data_root),
        "output_root": str(output_root),
        "variant_history": min_history,
        "files_processed": len(source_files),
        "copied_files": copied_files,
        "skipped_files": skipped_files,
    }


def _extract_date_token(filename: str) -> str:
    """Extract the YYYY-MM-DD token from a parquet filename."""
    match = DATE_TOKEN_PATTERN.search(filename)
    if not match:
        raise ValueError(f"Missing YYYY-MM-DD token in filename: {filename}")
    return match.group(1)


def _parse_competitor_source_file(dataset_root: Path, filepath: Path) -> dict[str, str]:
    """Extract grouping metadata from one by_competitor parquet path."""
    relative_path = filepath.relative_to(dataset_root / SOURCE_DATA_SUBDIR)
    if len(relative_path.parts) < 3:
        raise ValueError(f"Unexpected competitor parquet path: {filepath}")

    country = relative_path.parts[0]
    segment = relative_path.parts[1]
    return {
        "country": country,
        "segment": segment,
        "date_token": _extract_date_token(filepath.name),
    }


def _align_table_to_schema(table: pa.Table, schema: pa.Schema) -> pa.Table:
    """Align a PyArrow table to a target schema."""
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


def _combine_parquet_files(source_files: list[Path], output_path: Path) -> int:
    """Stream-combine parquet files into one output and return written rows."""
    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    total_rows = 0

    try:
        for source_file in sorted(source_files):
            parquet_file = pq.ParquetFile(source_file)
            for batch in parquet_file.iter_batches():
                table = pa.Table.from_batches([batch])
                if writer is None:
                    schema = table.schema
                    writer = pq.ParquetWriter(output_path, schema, compression="snappy")
                elif schema is not None:
                    table = _align_table_to_schema(table, schema)

                if writer is not None:
                    writer.write_table(table)
                    total_rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()

    return total_rows


def ensure_derived_datasets(dataset_root: Path) -> dict[str, Any]:
    """Create missing by_country, by_country_market, and global datasets from by_competitor."""
    source_files = find_competitor_source_files(dataset_root)
    if not source_files:
        logger.info("No by_competitor source files found for derived datasets under: %s", dataset_root)
        return {
            "dataset_root": str(dataset_root),
            "source_files": 0,
            "created_derived_files": 0,
            "skipped_derived_files": 0,
        }

    by_country_groups: dict[tuple[str, str], list[Path]] = {}
    by_country_market_groups: dict[tuple[str, str, str], list[Path]] = {}
    global_groups: dict[str, list[Path]] = {}

    for filepath in source_files:
        parsed = _parse_competitor_source_file(dataset_root, filepath)
        country = parsed["country"]
        segment = parsed["segment"]
        date_token = parsed["date_token"]

        by_country_groups.setdefault((country, date_token), []).append(filepath)
        by_country_market_groups.setdefault((country, segment, date_token), []).append(filepath)
        global_groups.setdefault(date_token, []).append(filepath)

    planned_outputs: list[tuple[Path, list[Path], str]] = []
    for (country, date_token), files in sorted(by_country_groups.items()):
        output_path = dataset_root / "by_country" / country / f"{country}_{date_token}.parquet"
        planned_outputs.append((output_path, files, "by_country"))
    for (country, segment, date_token), files in sorted(by_country_market_groups.items()):
        output_path = (
            dataset_root
            / "by_country_market"
            / country
            / segment
            / f"{country}_{segment}_{date_token}.parquet"
        )
        planned_outputs.append((output_path, files, "by_country_market"))
    for date_token, files in sorted(global_groups.items()):
        output_path = dataset_root / "global" / f"GLOBAL_{date_token}.parquet"
        planned_outputs.append((output_path, files, "global"))

    created_derived_files = 0
    skipped_derived_files = 0
    for output_path, files, scope_name in planned_outputs:
        if output_path.exists():
            skipped_derived_files += 1
            logger.info("  Skipping existing %s dataset: %s", scope_name, output_path.name)
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        total_rows = _combine_parquet_files(files, output_path)
        created_derived_files += 1
        logger.info(
            "  Created %s dataset: %s (%s files, %s rows)",
            scope_name,
            output_path.name,
            len(files),
            total_rows,
        )

    return {
        "dataset_root": str(dataset_root),
        "source_files": len(source_files),
        "created_derived_files": created_derived_files,
        "skipped_derived_files": skipped_derived_files,
    }


def ensure_history_subsets(
    cleaned_data_root: Path = CLEANED_DATA_ROOT,
    data_subsets_root: Path = DATA_SUBSETS_ROOT,
    history_values: list[int] | None = None,
) -> dict[str, Any]:
    """Create missing mh subset files from cleaned-data."""
    source_root = cleaned_data_root / "by_competitor"
    resolved_history_values = resolved_filtered_history_values(history_values)

    if not resolved_history_values:
        logger.info("No filtered history subsets requested after reserving mh5 for copied data.")
        return {
            "source_root": str(source_root),
            "output_root": str(data_subsets_root),
            "history_values": [],
            "dry_run": False,
            "skip_existing": True,
            "files_processed": 0,
            "generated_subset_files": 0,
            "skipped_subset_files": 0,
            "source_summaries": [],
        }

    if not source_root.exists():
        logger.warning("History subset source root does not exist: %s", source_root)
        return {
            "source_root": str(source_root),
            "output_root": str(data_subsets_root),
            "history_values": resolved_history_values,
            "dry_run": False,
            "skip_existing": True,
            "files_processed": 0,
            "generated_subset_files": 0,
            "skipped_subset_files": 0,
            "source_summaries": [],
        }

    logger.info("Ensuring history subsets under: %s", data_subsets_root)
    return create_history_subsets(
        source_root=source_root,
        output_root=data_subsets_root,
        history_values=resolved_history_values,
        skip_existing=True,
    )


def split_output_paths(filepath: Path) -> dict[str, Path]:
    """Return the suffixless output paths for one source file."""
    base_name = filepath.stem
    return {
        "train": filepath.parent / f"{base_name}_train.parquet",
        "test_new_products": filepath.parent / f"{base_name}_test_new_products.parquet",
        "test_new_prices": filepath.parent / f"{base_name}_test_new_prices.parquet",
    }


def split_outputs_exist(filepath: Path) -> bool:
    """Return True when all expected split outputs already exist."""
    return all(path.exists() for path in split_output_paths(filepath).values())


def _resolve_time_column(df: pd.DataFrame, filepath: Path) -> str:
    """Find the time column used for the temporal split."""
    for column_name in TIME_COLUMNS:
        if column_name in df.columns:
            return column_name
    raise ValueError(f"No time column found in {filepath}")


def _split_product_ids(product_ids: list[Any], filepath: Path) -> tuple[list[Any], list[Any]]:
    """Split product ids into train and new-product test groups."""
    if not product_ids:
        return [], []

    if len(product_ids) == 1:
        logger.warning(
            "Only one product found in %s; keeping it in train and leaving test_new_products empty.",
            filepath.name,
        )
        return product_ids, []

    train_product_ids, test_product_ids = train_test_split(
        product_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    return list(train_product_ids), list(test_product_ids)


def split_combined(filepath: Path) -> dict[str, int | str]:
    """Create one train split plus two test splits for a Parquet file."""
    logger.info("Processing: %s", filepath)

    df = pd.read_parquet(filepath)
    if "product_id" not in df.columns:
        raise ValueError(f"Missing product_id column in {filepath}")

    total_rows = len(df)
    total_products = int(df["product_id"].nunique())
    time_col = _resolve_time_column(df, filepath)
    df = df.sort_values(["product_id", time_col])

    product_ids = df["product_id"].drop_duplicates().tolist()
    train_product_ids, test_product_ids = _split_product_ids(product_ids, filepath)

    test_new_products_df = df[df["product_id"].isin(test_product_ids)].copy()
    train_products_df = df[df["product_id"].isin(train_product_ids)].copy()

    if train_products_df.empty:
        train_df = train_products_df.copy()
        test_new_prices_df = train_products_df.copy()
    else:
        train_products_df["_row_num"] = train_products_df.groupby("product_id").cumcount()
        train_products_df["_group_size"] = train_products_df.groupby("product_id")["product_id"].transform("count")

        new_prices_mask = train_products_df["_row_num"] >= (
            train_products_df["_group_size"] - LAST_N
        )

        train_df = train_products_df[~new_prices_mask].drop(columns=["_row_num", "_group_size"])
        test_new_prices_df = train_products_df[new_prices_mask].drop(
            columns=["_row_num", "_group_size"]
        )

    output_paths = split_output_paths(filepath)
    train_df.to_parquet(output_paths["train"], index=False, compression="snappy")
    test_new_products_df.to_parquet(
        output_paths["test_new_products"],
        index=False,
        compression="snappy",
    )
    test_new_prices_df.to_parquet(
        output_paths["test_new_prices"],
        index=False,
        compression="snappy",
    )

    logger.info(
        "  Created: %s (%s rows, %s products)",
        output_paths["train"].name,
        len(train_df),
        len(train_product_ids),
    )
    logger.info(
        "  Created: %s (%s rows, %s products)",
        output_paths["test_new_products"].name,
        len(test_new_products_df),
        len(test_product_ids),
    )
    logger.info(
        "  Created: %s (%s rows)",
        output_paths["test_new_prices"].name,
        len(test_new_prices_df),
    )

    return {
        "source_file": str(filepath),
        "total_rows": total_rows,
        "total_products": total_products,
        "train_rows": len(train_df),
        "train_products": len(train_product_ids),
        "test_new_products_rows": len(test_new_products_df),
        "test_new_products_products": len(test_product_ids),
        "test_new_prices_rows": len(test_new_prices_df),
        "test_new_prices_products": int(test_new_prices_df["product_id"].nunique()),
    }


def process_dataset_root(dataset_root: Path) -> dict[str, Any]:
    """Create missing split files for one dataset root."""
    logger.info("Scanning dataset root: %s", dataset_root)
    derived_result = ensure_derived_datasets(dataset_root)
    source_files = find_parquet_files(dataset_root)

    if not source_files:
        logger.info("  No source parquet files found.")
        return {
            "dataset_root": str(dataset_root),
            "source_files": 0,
            "created_derived_files": int(derived_result["created_derived_files"]),
            "skipped_derived_files": int(derived_result["skipped_derived_files"]),
            "created_source_files": 0,
            "skipped_source_files": 0,
            "created_split_files": 0,
        }

    created_stats: list[dict[str, int | str]] = []
    skipped_source_files = 0

    for filepath in source_files:
        if split_outputs_exist(filepath):
            skipped_source_files += 1
            logger.info("  Skipping already split file: %s", filepath.name)
            continue

        missing_outputs = [
            path.name
            for path in split_output_paths(filepath).values()
            if not path.exists()
        ]
        if missing_outputs and len(missing_outputs) < 3:
            logger.info(
                "  Recreating partial splits for %s: %s",
                filepath.name,
                ", ".join(missing_outputs),
            )

        created_stats.append(split_combined(filepath))

    return {
        "dataset_root": str(dataset_root),
        "source_files": len(source_files),
        "created_derived_files": int(derived_result["created_derived_files"]),
        "skipped_derived_files": int(derived_result["skipped_derived_files"]),
        "created_source_files": len(created_stats),
        "skipped_source_files": skipped_source_files,
        "created_split_files": len(created_stats) * 3,
        "train_rows": sum(int(stat["train_rows"]) for stat in created_stats),
        "test_new_products_rows": sum(
            int(stat["test_new_products_rows"]) for stat in created_stats
        ),
        "test_new_prices_rows": sum(int(stat["test_new_prices_rows"]) for stat in created_stats),
    }


def run_local_data_preparation(
    cleaned_data_root: Path = CLEANED_DATA_ROOT,
    data_subsets_root: Path = DATA_SUBSETS_ROOT,
    history_values: list[int] | None = None,
) -> dict[str, Any]:
    """Run the full local preparation flow with hardcoded dataset roots."""
    mh5_copy_result = ensure_original_data_variant(
        cleaned_data_root=cleaned_data_root,
        data_subsets_root=data_subsets_root,
    )
    subset_result = ensure_history_subsets(
        cleaned_data_root=cleaned_data_root,
        data_subsets_root=data_subsets_root,
        history_values=history_values,
    )
    dataset_roots = discover_dataset_roots(
        cleaned_data_root=cleaned_data_root,
        data_subsets_root=data_subsets_root,
        history_values=history_values,
    )
    dataset_summaries = [process_dataset_root(dataset_root) for dataset_root in dataset_roots]

    return {
        "workspace_root": str(WORKSPACE_ROOT),
        "mh5_copy_result": mh5_copy_result,
        "history_subset_result": subset_result,
        "dataset_roots_processed": len(dataset_summaries),
        "created_derived_files": sum(
            int(summary["created_derived_files"]) for summary in dataset_summaries
        ),
        "skipped_derived_files": sum(
            int(summary["skipped_derived_files"]) for summary in dataset_summaries
        ),
        "source_files": sum(int(summary["source_files"]) for summary in dataset_summaries),
        "created_source_files": sum(
            int(summary["created_source_files"]) for summary in dataset_summaries
        ),
        "skipped_source_files": sum(
            int(summary["skipped_source_files"]) for summary in dataset_summaries
        ),
        "created_split_files": sum(
            int(summary["created_split_files"]) for summary in dataset_summaries
        ),
        "dataset_summaries": dataset_summaries,
    }


def main() -> int:
    """Run the unified local dataset-preparation flow."""
    if len(sys.argv) > 1:
        raise SystemExit("split_training_data.py takes no arguments.")

    result = run_local_data_preparation()
    mh5_copy_result = result["mh5_copy_result"]
    subset_result = result["history_subset_result"]

    print("=" * 70)
    print("Local Dataset Preparation")
    print("=" * 70)
    print(f"Workspace root: {result['workspace_root']}")
    print(f"mh5 source files: {mh5_copy_result['files_processed']}")
    print(f"mh5 files copied: {mh5_copy_result['copied_files']}")
    print(f"mh5 files skipped: {mh5_copy_result['skipped_files']}")
    print(f"History subset source files: {subset_result['files_processed']}")
    print(f"History subset files created: {subset_result['generated_subset_files']}")
    print(f"History subset files skipped: {subset_result['skipped_subset_files']}")
    print(f"Dataset roots processed: {result['dataset_roots_processed']}")
    print(f"Derived parquet files created: {result['created_derived_files']}")
    print(f"Derived parquet files skipped: {result['skipped_derived_files']}")
    print(f"Source files found: {result['source_files']}")
    print(f"Source files split this run: {result['created_source_files']}")
    print(f"Source files skipped: {result['skipped_source_files']}")
    print(f"Split parquet files created: {result['created_split_files']}")
    print()

    for summary in result["dataset_summaries"]:
        print(summary["dataset_root"])
        print(f"  derived created: {summary['created_derived_files']}")
        print(f"  derived skipped: {summary['skipped_derived_files']}")
        print(f"  source files: {summary['source_files']}")
        print(f"  split this run: {summary['created_source_files']}")
        print(f"  skipped: {summary['skipped_source_files']}")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
