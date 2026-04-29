#!/usr/bin/env python3
"""Precompute shared tree-feature caches for the sampled research variants.

The generated ``.iforest_features.npz`` files are reused by Isolation Forest,
EIF, and RRCF training because all three share the same extracted feature set.
By default the script targets the thesis-sampled mh levels:
``mh5,mh10,mh15,mh20,mh25,mh30``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import Sequence

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_script_dir)))
sys.path.insert(0, _project_root)
sys.path.insert(0, _script_dir)

from src.research.mh_sampling import RESEARCH_MH_LEVELS, normalize_mh_levels
from train_isolation_forest import (
    extract_features_vectorized,
    extract_model_name,
    find_parquet_files,
    get_cache_path,
    load_cached_features,
    load_parquet_file,
    save_cached_features,
    select_latest_per_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ALL_GRANULARITIES = ("country_segment", "competitor", "global")


@dataclass(frozen=True)
class CacheJob:
    """One feature-cache build target."""

    mh_level: str
    granularity: str
    model_name: str
    filepath: str
    cache_path: str


def resolve_granularities(granularity: str) -> list[str]:
    """Resolve the CLI granularity selection into concrete values."""
    if granularity == "both":
        return list(ALL_GRANULARITIES)
    if granularity not in ALL_GRANULARITIES:
        raise ValueError(f"Unsupported granularity: {granularity}")
    return [granularity]


def discover_cache_jobs(
    *,
    data_path: str,
    mh_values: Sequence[int | str] | None = None,
    granularity: str = "both",
    model_filter: str | None = None,
) -> list[CacheJob]:
    """Discover the latest train-file variants that need shared tree features."""
    mh_levels = normalize_mh_levels(mh_values)
    granularities = resolve_granularities(granularity)

    jobs: list[CacheJob] = []
    for resolved_granularity in granularities:
        for mh_level in mh_levels:
            file_suffix = f"_train_{mh_level}"
            candidate_files = find_parquet_files(data_path, resolved_granularity, file_suffix)
            if not candidate_files:
                logger.info(
                    "No %s train files found for %s under %s",
                    resolved_granularity,
                    mh_level,
                    data_path,
                )
                continue

            for filepath in select_latest_per_model(candidate_files, resolved_granularity):
                model_name = extract_model_name(filepath)
                if model_filter and model_filter not in model_name:
                    continue

                jobs.append(
                    CacheJob(
                        mh_level=mh_level,
                        granularity=resolved_granularity,
                        model_name=model_name,
                        filepath=filepath,
                        cache_path=get_cache_path(filepath),
                    )
                )

    jobs.sort(key=lambda job: (int(job.mh_level.removeprefix("mh")), job.granularity, job.model_name))
    return jobs


def build_cache_for_job(job: CacheJob, *, rebuild_cache: bool = False) -> dict[str, object]:
    """Build or reuse one cached feature matrix."""
    cache_existed = os.path.exists(job.cache_path)
    if not rebuild_cache:
        cached = load_cached_features(job.cache_path)
        if cached is not None:
            return {
                "status": "reused",
                "mh_level": job.mh_level,
                "granularity": job.granularity,
                "model_name": job.model_name,
                "filepath": job.filepath,
                "cache_path": job.cache_path,
                "rows": int(cached.shape[0]),
                "features": int(cached.shape[1]),
            }

    df = load_parquet_file(job.filepath)
    X = extract_features_vectorized(df)
    save_cached_features(job.cache_path, X)
    return {
        "status": "rebuilt" if cache_existed else "built",
        "mh_level": job.mh_level,
        "granularity": job.granularity,
        "model_name": job.model_name,
        "filepath": job.filepath,
        "cache_path": job.cache_path,
        "rows": int(X.shape[0]),
        "features": int(X.shape[1]),
        "products": int(df["product_id"].nunique()) if "product_id" in df.columns else 0,
    }


def run_precompute(
    *,
    data_path: str,
    mh_values: Sequence[int | str] | None = None,
    granularity: str = "both",
    model_filter: str | None = None,
    rebuild_cache: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    """Run the shared tree-feature cache precompute flow."""
    jobs = discover_cache_jobs(
        data_path=data_path,
        mh_values=mh_values,
        granularity=granularity,
        model_filter=model_filter,
    )
    if dry_run:
        return {
            "jobs": [job.__dict__ for job in jobs],
            "processed": 0,
            "built": 0,
            "rebuilt": 0,
            "reused": 0,
        }

    results = [build_cache_for_job(job, rebuild_cache=rebuild_cache) for job in jobs]
    return {
        "jobs": [job.__dict__ for job in jobs],
        "results": results,
        "processed": len(results),
        "built": sum(1 for result in results if result["status"] == "built"),
        "rebuilt": sum(1 for result in results if result["status"] == "rebuilt"),
        "reused": sum(1 for result in results if result["status"] == "reused"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Precompute shared tree-feature caches for IF, EIF, and RRCF."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/training/derived",
        help="Data directory containing by_competitor/, by_country_segment/, and global/",
    )
    parser.add_argument(
        "--granularity",
        type=str,
        choices=["country_segment", "competitor", "global", "both"],
        default="both",
        help="Granularity to scan (default: both)",
    )
    parser.add_argument(
        "--mh-values",
        nargs="*",
        default=list(RESEARCH_MH_LEVELS),
        help=(
            "mh variants to precompute. Defaults to the thesis research subset "
            f"{', '.join(RESEARCH_MH_LEVELS)}."
        ),
    )
    parser.add_argument(
        "--model-filter",
        type=str,
        default=None,
        help="Only cache models matching this substring",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Force regeneration even when a valid cache already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the planned cache jobs without writing files",
    )
    args = parser.parse_args()

    result = run_precompute(
        data_path=args.data_path,
        mh_values=args.mh_values,
        granularity=args.granularity,
        model_filter=args.model_filter,
        rebuild_cache=args.rebuild_cache,
        dry_run=args.dry_run,
    )

    print("=" * 70)
    print("Tree Feature Cache Precompute")
    print("=" * 70)
    print(f"Data path: {args.data_path}")
    print(f"Granularity: {args.granularity}")
    print(f"mh values: {', '.join(normalize_mh_levels(args.mh_values))}")
    print(f"Model filter: {args.model_filter or '(none)'}")
    print(f"Rebuild cache: {args.rebuild_cache}")
    print(f"Dry run: {args.dry_run}")
    print(f"Jobs discovered: {len(result['jobs'])}")

    for job in result["jobs"]:
        print(
            f"  - {job['mh_level']} | {job['granularity']} | "
            f"{job['model_name']} | {os.path.basename(job['filepath'])}"
        )

    if args.dry_run:
        return 0

    print(f"Processed: {result['processed']}")
    print(f"Built: {result['built']}")
    print(f"Rebuilt: {result['rebuilt']}")
    print(f"Reused: {result['reused']}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
