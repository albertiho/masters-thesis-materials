#!/usr/bin/env python3
"""Quick smoke validation for trained mh5 Isolation Forest models.

The script is intentionally narrow: it loads existing mh5 IF models, picks a
small deterministic subset of one or more train/test pairs, and runs a fast
threshold-tuning pass to validate that the saved models can be loaded, scored,
and tuned on real data. Saved models are left untouched unless
``--write-threshold`` is provided.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_script_dir)))
sys.path.insert(0, _project_root)
sys.path.insert(0, _script_dir)

from tune_isolation_forest import update_model_threshold
from tuning_utils import run_tuning_trials
from src.anomaly.persistence import ModelPersistence
from train_isolation_forest import (
    extract_model_name,
    find_matching_test_file,
    find_parquet_files,
    select_latest_per_model,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandidateModel:
    """Resolved IF smoke-validation candidate."""

    model_name: str
    granularity: str
    train_path: str
    test_path: str


@dataclass
class SmokeValidationResult:
    """Summary for one validated model."""

    model_name: str
    granularity: str
    train_rows: int
    test_rows: int
    train_products: int
    test_products: int
    current_threshold: float | None
    best_threshold: float | None
    current_f1: float | None
    best_f1: float | None
    improvement_pct: float | None
    status: str
    updated_model: bool = False
    message: str = ""


def _detect_time_column(df: pd.DataFrame) -> str | None:
    for column_name in ("first_seen_at", "scraped_at", "timestamp"):
        if column_name in df.columns:
            return column_name
    return None


def _infer_country(df: pd.DataFrame) -> str | None:
    if "country" not in df.columns:
        return None
    non_null = df["country"].dropna()
    if non_null.empty:
        return None
    return str(non_null.iloc[0])


def discover_candidates(
    *,
    data_path: str,
    granularity: str,
    mh_level: str = "mh5",
    split: str = "new_prices",
    model_filter: str | None = None,
) -> list[CandidateModel]:
    """Discover latest mh5 train/test pairs for smoke validation."""
    train_suffix = f"_train_{mh_level}"
    test_suffix = f"_test_{split}_{mh_level}"
    candidate_files = find_parquet_files(data_path, granularity, train_suffix)
    latest_files = select_latest_per_model(candidate_files, granularity)

    candidates: list[CandidateModel] = []
    for train_path in latest_files:
        model_name = extract_model_name(train_path)
        if model_filter and model_filter not in model_name:
            continue

        test_path = find_matching_test_file(train_path, test_suffix, data_path)
        if not test_path:
            logger.warning("Missing %s test file for %s", split, model_name)
            continue

        candidates.append(
            CandidateModel(
                model_name=model_name,
                granularity=granularity,
                train_path=train_path,
                test_path=test_path,
            )
        )

    return candidates


def build_quick_subset(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    max_products: int = 64,
    max_history_per_product: int = 30,
    min_test_rows: int = 128,
    max_test_rows: int = 512,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a deterministic small subset suitable for quick tuning."""
    if train_df.empty or test_df.empty:
        raise ValueError("Train/test data must be non-empty")
    if "product_id" not in train_df.columns or "product_id" not in test_df.columns:
        raise ValueError("Both train and test data must contain product_id")

    test_counts = (
        test_df.groupby("product_id")
        .size()
        .reset_index(name="row_count")
        .sort_values(["row_count", "product_id"], ascending=[False, True], kind="stable")
    )
    selected_products: list[Any] = []
    selected_test_rows = 0

    for row in test_counts.itertuples(index=False):
        if len(selected_products) >= max_products:
            break
        if selected_test_rows >= max_test_rows and selected_test_rows >= min_test_rows:
            break
        selected_products.append(row.product_id)
        selected_test_rows += int(row.row_count)

    if not selected_products:
        raise ValueError("Could not select any products for the test subset")

    test_subset = test_df[test_df["product_id"].isin(selected_products)].copy()
    train_subset = train_df[train_df["product_id"].isin(selected_products)].copy()

    time_col = _detect_time_column(train_subset)
    if time_col is not None:
        train_subset = train_subset.sort_values(["product_id", time_col], kind="stable")
        test_subset = test_subset.sort_values(["product_id", time_col], kind="stable")
    else:
        train_subset = train_subset.sort_values(["product_id"], kind="stable")
        test_subset = test_subset.sort_values(["product_id"], kind="stable")

    train_subset = (
        train_subset.groupby("product_id", group_keys=False, sort=False)
        .tail(max_history_per_product)
        .reset_index(drop=True)
    )
    test_subset = test_subset.reset_index(drop=True)

    if len(test_subset) < min_test_rows:
        raise ValueError(
            f"Subset too small for tuning: need at least {min_test_rows} test rows, got {len(test_subset)}"
        )

    return train_subset, test_subset


def run_smoke_validation(
    *,
    data_path: str = "data/training/derived",
    model_root: str = "artifacts/models",
    granularity: str = "competitor",
    mh_level: str = "mh5",
    split: str = "new_prices",
    model_filter: str | None = None,
    model_limit: int = 2,
    max_products: int = 64,
    max_history_per_product: int = 30,
    min_test_rows: int = 128,
    max_test_rows: int = 512,
    min_threshold: float = 0.2,
    max_threshold: float = 0.8,
    steps: int = 5,
    n_trials: int = 1,
    injection_rate: float = 0.1,
    max_workers: int = 1,
    write_threshold: bool = False,
    output_csv: str | None = None,
) -> list[SmokeValidationResult]:
    """Run a quick IF smoke-validation/tuning pass on real mh5 models."""
    persistence = ModelPersistence(model_root=model_root)
    candidates = discover_candidates(
        data_path=data_path,
        granularity=granularity,
        mh_level=mh_level,
        split=split,
        model_filter=model_filter,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No {granularity} {mh_level} candidates discovered under {data_path}"
        )

    selected_candidates = candidates[:model_limit] if model_limit > 0 else candidates
    thresholds = np.linspace(min_threshold, max_threshold, steps)
    results: list[SmokeValidationResult] = []

    for candidate in selected_candidates:
        logger.info("Validating %s", candidate.model_name)
        if not persistence.model_exists(candidate.model_name, "isolation_forest"):
            results.append(
                SmokeValidationResult(
                    model_name=candidate.model_name,
                    granularity=candidate.granularity,
                    train_rows=0,
                    test_rows=0,
                    train_products=0,
                    test_products=0,
                    current_threshold=None,
                    best_threshold=None,
                    current_f1=None,
                    best_f1=None,
                    improvement_pct=None,
                    status="missing_model",
                    message="Isolation Forest model not found in local persistence",
                )
            )
            continue

        try:
            detector = persistence.load_isolation_forest(candidate.model_name)
            train_df = pd.read_parquet(candidate.train_path)
            test_df = pd.read_parquet(candidate.test_path)
            train_subset, test_subset = build_quick_subset(
                train_df,
                test_df,
                max_products=max_products,
                max_history_per_product=max_history_per_product,
                min_test_rows=min_test_rows,
                max_test_rows=max_test_rows,
            )
            tuning_result = run_tuning_trials(
                detector=detector,
                detector_name=candidate.model_name,
                test_df=test_subset,
                train_df=train_subset,
                thresholds=thresholds,
                current_threshold=detector.config.anomaly_threshold,
                n_trials=n_trials,
                injection_rate=injection_rate,
                country=_infer_country(train_subset) or _infer_country(test_subset),
                max_workers=max_workers,
                target_metric="f1",
                min_precision=0.0,
                drop_range=(0.1, 0.5),
                min_successful_trials=1,
            )
            if tuning_result is None:
                results.append(
                    SmokeValidationResult(
                        model_name=candidate.model_name,
                        granularity=candidate.granularity,
                        train_rows=len(train_subset),
                        test_rows=len(test_subset),
                        train_products=int(train_subset["product_id"].nunique()),
                        test_products=int(test_subset["product_id"].nunique()),
                        current_threshold=float(detector.config.anomaly_threshold),
                        best_threshold=None,
                        current_f1=None,
                        best_f1=None,
                        improvement_pct=None,
                        status="tuning_failed",
                        message="run_tuning_trials returned None",
                    )
                )
                continue

            updated_model = False
            if write_threshold:
                update_model_threshold(
                    persistence,
                    candidate.model_name,
                    float(tuning_result.best_threshold),
                )
                updated_model = True

            results.append(
                SmokeValidationResult(
                    model_name=candidate.model_name,
                    granularity=candidate.granularity,
                    train_rows=len(train_subset),
                    test_rows=len(test_subset),
                    train_products=int(train_subset["product_id"].nunique()),
                    test_products=int(test_subset["product_id"].nunique()),
                    current_threshold=float(tuning_result.current_threshold),
                    best_threshold=float(tuning_result.best_threshold),
                    current_f1=float(tuning_result.current_f1),
                    best_f1=float(tuning_result.best_f1),
                    improvement_pct=float(tuning_result.improvement_pct),
                    status="ok",
                    updated_model=updated_model,
                    message="threshold_updated" if updated_model else "validated_without_model_update",
                )
            )
        except Exception as exc:
            results.append(
                SmokeValidationResult(
                    model_name=candidate.model_name,
                    granularity=candidate.granularity,
                    train_rows=0,
                    test_rows=0,
                    train_products=0,
                    test_products=0,
                    current_threshold=None,
                    best_threshold=None,
                    current_f1=None,
                    best_f1=None,
                    improvement_pct=None,
                    status="error",
                    message=str(exc),
                )
            )

    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([asdict(result) for result in results]).to_csv(output_path, index=False)

    return results


def _default_output_csv() -> str:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return str(
        Path("results") / "validation" / f"iforest_mh5_smoke_tuning_{timestamp}.csv"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quick smoke validation for trained mh5 Isolation Forest models."
    )
    parser.add_argument("--data-path", type=str, default="data/training/derived")
    parser.add_argument("--model-root", type=str, default="artifacts/models")
    parser.add_argument(
        "--granularity",
        type=str,
        choices=["country_segment", "competitor", "global"],
        default="competitor",
    )
    parser.add_argument("--mh-level", type=str, default="mh5")
    parser.add_argument(
        "--split",
        type=str,
        choices=["new_prices", "new_products"],
        default="new_prices",
    )
    parser.add_argument("--model-filter", type=str, default=None)
    parser.add_argument("--model-limit", type=int, default=2)
    parser.add_argument("--max-products", type=int, default=64)
    parser.add_argument("--max-history-per-product", type=int, default=30)
    parser.add_argument("--min-test-rows", type=int, default=128)
    parser.add_argument("--max-test-rows", type=int, default=512)
    parser.add_argument("--min-threshold", type=float, default=0.2)
    parser.add_argument("--max-threshold", type=float, default=0.8)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--n-trials", type=int, default=1)
    parser.add_argument("--injection-rate", type=float, default=0.1)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument(
        "--write-threshold",
        action="store_true",
        help="Persist the tuned threshold back into the saved model",
    )
    parser.add_argument("--output-csv", type=str, default=None)
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    output_csv = args.output_csv or _default_output_csv()
    results = run_smoke_validation(
        data_path=args.data_path,
        model_root=args.model_root,
        granularity=args.granularity,
        mh_level=args.mh_level,
        split=args.split,
        model_filter=args.model_filter,
        model_limit=args.model_limit,
        max_products=args.max_products,
        max_history_per_product=args.max_history_per_product,
        min_test_rows=args.min_test_rows,
        max_test_rows=args.max_test_rows,
        min_threshold=args.min_threshold,
        max_threshold=args.max_threshold,
        steps=args.steps,
        n_trials=args.n_trials,
        injection_rate=args.injection_rate,
        max_workers=args.max_workers,
        write_threshold=args.write_threshold,
        output_csv=output_csv,
    )

    print("=" * 70)
    print("Isolation Forest mh5 Smoke Validation")
    print("=" * 70)
    print(f"Granularity: {args.granularity}")
    print(f"Split: {args.split}")
    print(f"Model root: {args.model_root}")
    print(f"Output CSV: {output_csv}")
    print()

    for result in results:
        print(
            f"{result.model_name}: status={result.status}, "
            f"train_rows={result.train_rows}, test_rows={result.test_rows}, "
            f"current_threshold={result.current_threshold}, best_threshold={result.best_threshold}, "
            f"best_f1={result.best_f1}, updated_model={result.updated_model}"
        )
        if result.message:
            print(f"  {result.message}")

    success_count = sum(1 for result in results if result.status == "ok")
    print()
    print(f"Validated {success_count}/{len(results)} model(s) successfully")
    return 0 if success_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
