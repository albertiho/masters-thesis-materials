#!/usr/bin/env python3
"""Quick smoke validation for trained mh5 EIF models.

The script mirrors the IF smoke-validation flow: it loads existing mh5 EIF
models, picks a small deterministic subset of one or more train/test pairs,
and runs a fast threshold-tuning pass to validate that the saved models can be
loaded, scored, and tuned on real data. Saved models are left untouched unless
``--write-threshold`` is provided.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_script_dir)))
sys.path.insert(0, _project_root)
sys.path.insert(0, _script_dir)

from tuning_utils import run_tuning_trials
from src.anomaly.persistence import ModelPersistence
from validate_iforest_mh5_smoke import (
    SmokeValidationResult,
    _infer_country,
    build_quick_subset,
    discover_candidates,
)

logger = logging.getLogger(__name__)


def update_model_threshold(
    persistence: ModelPersistence,
    model_name: str,
    new_threshold: float,
) -> str:
    """Update the anomaly_threshold in a saved local EIF model."""
    from joblib import dump as joblib_dump
    from joblib import load as joblib_load

    model_path = persistence._get_model_path(model_name, "eif", "model.joblib")
    model_bytes = persistence._download_bytes(model_path)

    buffer = io.BytesIO(model_bytes)
    saved_data = joblib_load(buffer)
    saved_data["config"]["anomaly_threshold"] = new_threshold

    buffer = io.BytesIO()
    joblib_dump(saved_data, buffer)
    return persistence._upload_bytes(buffer.getvalue(), model_path)


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
    """Run a quick{EIF} smoke-validation/tuning pass on real mh5 models."""
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
        if not persistence.model_exists(candidate.model_name, "eif"):
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
                    message="EIF model not found in local persistence",
                )
            )
            continue

        try:
            detector = persistence.load_eif(candidate.model_name)
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
        Path("results") / "validation" / f"eif_mh5_smoke_tuning_{timestamp}.csv"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quick smoke validation for trained mh5 EIF models."
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
    print("EIF mh5 Smoke Validation")
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
