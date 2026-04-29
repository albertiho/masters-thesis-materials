#!/usr/bin/env python3
"""One-off replay of retained country-level IF on failed COUNTRY_4 competitor cases.

This script evaluates the retained country-level Isolation Forest configuration
for COUNTRY_4 against the failed competitor-level case
COUNTRY_4 / B2C / COMPETITOR_3_COUNTRY_4_2026-02-08 across the mh levels that
failed during the competitor-level sweep.

Important:
    - This is intentionally a one-off script. Edit the module-level constants if
      the target case needs to change. No CLI is provided on purpose.
    - The script prefers loading an existing persisted country-level IF model.
      If the artifact is not present locally, it refits the retained
      country-level configuration from the saved forest sweep metadata before
      evaluating it on the failed competitor-level splits.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

from src.anomaly.persistence import ModelPersistence
from train_isolation_forest import extract_features_vectorized, train_from_matrix
from tuning_utils import run_tuning_trials

LOGGER = logging.getLogger(__name__)

TARGET_COUNTRY = "COUNTRY_4"
TARGET_SEGMENT = "B2C"
TARGET_COMPETITOR = "COMPETITOR_3_COUNTRY_4"
TARGET_DATE = "2026-02-08"
FAILED_MH_LEVELS = ("mh10", "mh15", "mh20", "mh25", "mh30")
TARGET_SPLITS = ("new_prices", "new_products")

DATA_SUBSETS_ROOT = _PROJECT_ROOT / "data-subsets"
FOREST_SWEEP_ROOT = _PROJECT_ROOT / "results" / "tuning" / "forests" / "single_config_optimized_mh5_run"
OUTPUT_ROOT = _PROJECT_ROOT / "results" / "analysis" / "country_level_if_country4_competitor3_replay"

INJECTION_RATE = 0.10
N_TRIALS = 10
MAX_WORKERS = 1
SPLIT_WEIGHTS = {"new_prices": 0.7, "new_products": 0.3}


@dataclass(frozen=True)
class CountryModelConfig:
    mh_level: str
    model_name: str
    threshold: float
    n_estimators: int
    max_samples: str | int
    max_features: float
    contamination: str | float
    source_json: str


@dataclass
class SplitReplayResult:
    mh_level: str
    split: str
    status: str
    message: str
    original_competitor_failure_reason: str
    country_model_name: str
    country_model_source: str
    country_model_threshold: float
    country_model_n_estimators: int
    country_model_max_samples: str | int
    country_model_max_features: float
    country_model_contamination: str | float
    competitor_train_rows: int
    competitor_train_products: int
    matched_train_rows: int
    matched_train_products: int
    test_rows: int
    test_products: int
    n_trials: int
    injection_rate: float
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    tnr: float | None = None
    fpr: float | None = None
    fnr: float | None = None
    f1: float | None = None
    g_mean: float | None = None
    accuracy_std: float | None = None
    precision_std: float | None = None
    recall_std: float | None = None
    tnr_std: float | None = None
    fpr_std: float | None = None
    fnr_std: float | None = None
    f1_std: float | None = None
    g_mean_std: float | None = None
    true_positives: float | None = None
    false_positives: float | None = None
    false_negatives: float | None = None
    true_negatives: float | None = None
    n_injected: float | None = None
    n_predicted: float | None = None


def _relative(path: Path) -> str:
    return path.resolve().relative_to(_PROJECT_ROOT.resolve()).as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _country_train_path(mh_level: str) -> Path:
    return DATA_SUBSETS_ROOT / mh_level / "by_country" / TARGET_COUNTRY / f"{TARGET_COUNTRY}_{TARGET_DATE}_train.parquet"


def _competitor_train_path(mh_level: str) -> Path:
    return (
        DATA_SUBSETS_ROOT
        / mh_level
        / "by_competitor"
        / TARGET_COUNTRY
        / TARGET_SEGMENT
        / f"{TARGET_COMPETITOR}_{TARGET_DATE}_train.parquet"
    )


def _competitor_test_path(mh_level: str, split: str) -> Path:
    return (
        DATA_SUBSETS_ROOT
        / mh_level
        / "by_competitor"
        / TARGET_COUNTRY
        / TARGET_SEGMENT
        / f"{TARGET_COMPETITOR}_{TARGET_DATE}_test_{split}.parquet"
    )


def _country_best_configuration_path(mh_level: str) -> Path:
    return (
        FOREST_SWEEP_ROOT
        / mh_level
        / "by_country"
        / f"{TARGET_COUNTRY}_{TARGET_COUNTRY}_{TARGET_DATE}"
        / "if"
        / "best_configuration.json"
    )


def _competitor_failure_summary_path(mh_level: str) -> Path:
    return (
        FOREST_SWEEP_ROOT
        / mh_level
        / "by_competitor"
        / f"{TARGET_COUNTRY}_{TARGET_SEGMENT}_{TARGET_COMPETITOR}_{TARGET_DATE}"
        / "if"
        / "summary.json"
    )


def _load_country_model_config(mh_level: str) -> CountryModelConfig:
    payload = _load_json(_country_best_configuration_path(mh_level))
    best = payload["best_candidate"]
    return CountryModelConfig(
        mh_level=mh_level,
        model_name=str(best["model_name"]),
        threshold=float(best["threshold"]),
        n_estimators=int(best["n_estimators"]),
        max_samples=best["max_samples"],
        max_features=float(best["max_features"]),
        contamination=best["contamination"],
        source_json=_relative(_country_best_configuration_path(mh_level)),
    )


def _load_failure_reason(mh_level: str) -> str:
    payload = _load_json(_competitor_failure_summary_path(mh_level))
    return str(payload.get("error", "")).strip()


def _load_or_refit_country_model(
    mh_level: str,
    config: CountryModelConfig,
    persistence: ModelPersistence,
) -> tuple[Any, str]:
    if persistence.model_exists(config.model_name, "isolation_forest"):
        LOGGER.info("Loading persisted country-level IF model %s", config.model_name)
        return persistence.load_isolation_forest(config.model_name), "loaded_persisted_model"

    LOGGER.info(
        "Persisted model %s was not found locally; refitting retained configuration from %s",
        config.model_name,
        config.source_json,
    )
    country_train_df = pd.read_parquet(_country_train_path(mh_level))
    feature_matrix = extract_features_vectorized(country_train_df)
    detector, _ = train_from_matrix(
        feature_matrix,
        contamination=config.contamination,
        anomaly_threshold=config.threshold,
        n_estimators=config.n_estimators,
        max_samples=config.max_samples,
        max_features=config.max_features,
    )
    return detector, "refit_from_best_country_configuration"


def _mh_has_any_test_rows(mh_level: str) -> bool:
    for split in TARGET_SPLITS:
        test_path = _competitor_test_path(mh_level, split)
        if not test_path.exists():
            continue
        test_df = pd.read_parquet(test_path)
        if not test_df.empty:
            return True
    return False


def _aggregate_split_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    fields = (
        "accuracy",
        "precision",
        "recall",
        "tnr",
        "fpr",
        "fnr",
        "f1",
        "g_mean",
        "accuracy_std",
        "precision_std",
        "recall_std",
        "tnr_std",
        "fpr_std",
        "fnr_std",
        "f1_std",
        "g_mean_std",
        "true_positives",
        "false_positives",
        "false_negatives",
        "true_negatives",
        "n_injected",
        "n_predicted",
    )
    return {field: float(metrics[field]) for field in fields if field in metrics}


def _evaluate_split(
    *,
    mh_level: str,
    split: str,
    detector: Any,
    model_source: str,
    config: CountryModelConfig,
    failure_reason: str,
) -> SplitReplayResult:
    competitor_train_df = pd.read_parquet(_competitor_train_path(mh_level))
    test_df = pd.read_parquet(_competitor_test_path(mh_level, split))

    matched_train_rows = 0
    matched_train_products = 0
    if not test_df.empty and not competitor_train_df.empty:
        test_products = set(test_df["product_id"].unique())
        matched_train_df = competitor_train_df[competitor_train_df["product_id"].isin(test_products)]
        matched_train_rows = int(len(matched_train_df))
        matched_train_products = int(matched_train_df["product_id"].nunique())

    base = SplitReplayResult(
        mh_level=mh_level,
        split=split,
        status="pending",
        message="",
        original_competitor_failure_reason=failure_reason,
        country_model_name=config.model_name,
        country_model_source=model_source,
        country_model_threshold=config.threshold,
        country_model_n_estimators=config.n_estimators,
        country_model_max_samples=config.max_samples,
        country_model_max_features=config.max_features,
        country_model_contamination=config.contamination,
        competitor_train_rows=int(len(competitor_train_df)),
        competitor_train_products=int(competitor_train_df["product_id"].nunique()) if not competitor_train_df.empty else 0,
        matched_train_rows=matched_train_rows,
        matched_train_products=matched_train_products,
        test_rows=int(len(test_df)),
        test_products=int(test_df["product_id"].nunique()) if not test_df.empty else 0,
        n_trials=N_TRIALS,
        injection_rate=INJECTION_RATE,
    )

    if test_df.empty:
        base.status = "empty_test_split"
        base.message = "No competitor-level evaluation rows remain after the minimum-history filter."
        return base

    tuning_result = run_tuning_trials(
        detector=detector,
        detector_name=f"{config.model_name}_on_{TARGET_COMPETITOR}_{mh_level}_{split}",
        test_df=test_df,
        train_df=competitor_train_df,
        thresholds=np.array([config.threshold], dtype=np.float64),
        current_threshold=config.threshold,
        n_trials=N_TRIALS,
        injection_rate=INJECTION_RATE,
        country=TARGET_COUNTRY,
        max_workers=MAX_WORKERS,
        target_metric="f1",
        min_precision=0.0,
        min_successful_trials=1,
    )
    if tuning_result is None or not tuning_result.all_results:
        base.status = "evaluation_failed"
        base.message = "run_tuning_trials returned no aggregate metrics."
        return base

    aggregate = _aggregate_split_metrics(tuning_result.all_results[0])
    base.status = "ok"
    base.message = "evaluated"
    for key, value in aggregate.items():
        setattr(base, key, value)
    return base


def _combine_weighted_metrics(split_rows: list[SplitReplayResult]) -> dict[str, float]:
    split_by_name = {row.split: row for row in split_rows if row.status == "ok"}
    if set(split_by_name) != set(SPLIT_WEIGHTS):
        return {}

    def weighted(field: str) -> float:
        return float(
            sum(SPLIT_WEIGHTS[split] * float(getattr(split_by_name[split], field)) for split in SPLIT_WEIGHTS)
        )

    return {
        "combined_accuracy": weighted("accuracy"),
        "combined_precision": weighted("precision"),
        "combined_recall": weighted("recall"),
        "combined_tnr": weighted("tnr"),
        "combined_fpr": weighted("fpr"),
        "combined_fnr": weighted("fnr"),
        "combined_f1": weighted("f1"),
        "combined_g_mean": weighted("g_mean"),
    }


def _build_mh_summary(split_results: list[SplitReplayResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for mh_level in FAILED_MH_LEVELS:
        mh_rows = [row for row in split_results if row.mh_level == mh_level]
        failure_reason = mh_rows[0].original_competitor_failure_reason if mh_rows else _load_failure_reason(mh_level)
        row: dict[str, Any] = {
            "mh_level": mh_level,
            "original_competitor_failure_reason": failure_reason,
            "country_model_name": mh_rows[0].country_model_name if mh_rows else None,
            "country_model_source": mh_rows[0].country_model_source if mh_rows else None,
            "country_model_threshold": mh_rows[0].country_model_threshold if mh_rows else None,
            "country_model_n_estimators": mh_rows[0].country_model_n_estimators if mh_rows else None,
            "country_model_max_samples": mh_rows[0].country_model_max_samples if mh_rows else None,
            "country_model_max_features": mh_rows[0].country_model_max_features if mh_rows else None,
            "country_model_contamination": mh_rows[0].country_model_contamination if mh_rows else None,
        }
        for split in TARGET_SPLITS:
            split_row = next((candidate for candidate in mh_rows if candidate.split == split), None)
            row[f"{split}_status"] = split_row.status if split_row else "missing"
            row[f"{split}_message"] = split_row.message if split_row else ""
            row[f"{split}_test_rows"] = split_row.test_rows if split_row else 0
            row[f"{split}_test_products"] = split_row.test_products if split_row else 0
            row[f"{split}_matched_train_rows"] = split_row.matched_train_rows if split_row else 0
            row[f"{split}_matched_train_products"] = split_row.matched_train_products if split_row else 0
            row[f"{split}_f1"] = split_row.f1 if split_row else np.nan
            row[f"{split}_g_mean"] = split_row.g_mean if split_row else np.nan
            row[f"{split}_precision"] = split_row.precision if split_row else np.nan
            row[f"{split}_recall"] = split_row.recall if split_row else np.nan

        combined = _combine_weighted_metrics(mh_rows)
        if combined:
            row["status"] = "ok"
            row.update(combined)
        elif any(candidate.status == "ok" for candidate in mh_rows):
            row["status"] = "partial"
            row.update(
                {
                    "combined_accuracy": np.nan,
                    "combined_precision": np.nan,
                    "combined_recall": np.nan,
                    "combined_tnr": np.nan,
                    "combined_fpr": np.nan,
                    "combined_fnr": np.nan,
                    "combined_f1": np.nan,
                    "combined_g_mean": np.nan,
                }
            )
        else:
            row["status"] = "no_eval_rows"
            row.update(
                {
                    "combined_accuracy": np.nan,
                    "combined_precision": np.nan,
                    "combined_recall": np.nan,
                    "combined_tnr": np.nan,
                    "combined_fpr": np.nan,
                    "combined_fnr": np.nan,
                    "combined_f1": np.nan,
                    "combined_g_mean": np.nan,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _write_summary_markdown(split_df: pd.DataFrame, mh_df: pd.DataFrame) -> None:
    output_path = OUTPUT_ROOT / "summary.md"

    evaluable = mh_df[mh_df["status"] == "ok"].copy()
    lines = [
        "# Country-Level IF Replay On COUNTRY_4 / COMPETITOR_3 Failed Cases",
        "",
        "## Purpose",
        "",
        (
            f"- Evaluate the retained country-level IF configuration for `{TARGET_COUNTRY}` on the failed "
            f"competitor-level case `{TARGET_COUNTRY}/{TARGET_SEGMENT}/{TARGET_COMPETITOR}_{TARGET_DATE}` "
            "across the mh levels that could not support a competitor-level IF model."
        ),
        (
            f"- Use the same synthetic-injection evaluation path as the forest sweep, with `{N_TRIALS}` repeated "
            f"trial(s) per split at injection rate `{INJECTION_RATE:.0%}`."
        ),
        (
            "- Prefer loading the persisted country-level IF model; when it is absent locally, refit the retained "
            "country-level configuration from the saved sweep metadata before scoring."
        ),
        "",
        "## Key Findings",
        "",
    ]

    loaded_count = int((split_df["country_model_source"] == "loaded_persisted_model").sum())
    if loaded_count > 0:
        lines.append(f"- Persisted country-level IF artifacts were available for {loaded_count} split evaluation(s).")
    else:
        lines.append(
            "- No persisted country-level IF artifact was available locally for the targeted COUNTRY_4 mh levels; "
            "all evaluated cases were therefore refit from the retained country-level sweep configuration."
        )

    if not evaluable.empty:
        best = evaluable.sort_values("combined_f1", ascending=False).iloc[0]
        lines.append(
            f"- The only failed mh case with non-empty competitor-level evaluation splits was `{best['mh_level']}`, "
            f"where the country-level IF replay reached weighted combined `F1={best['combined_f1']:.4f}` and "
            f"`G={best['combined_g_mean']:.4f}`."
        )
    else:
        lines.append("- None of the failed mh cases contained non-empty competitor-level evaluation splits.")

    empty_cases = mh_df[mh_df["status"] == "no_eval_rows"]["mh_level"].tolist()
    if empty_cases:
        lines.append(
            f"- The later failed cases `{', '.join(empty_cases)}` could not be replayed because both competitor-level "
            "test splits are empty after the minimum-history filter."
        )

    mh10_products = split_df[
        (split_df["mh_level"] == "mh10") & (split_df["split"] == "new_products") & (split_df["status"] == "ok")
    ]
    if not mh10_products.empty:
        products_row = mh10_products.iloc[0]
        lines.append(
            f"- At `mh10`, the `new_products` split had `{int(products_row['matched_train_rows'])}` matched "
            "competitor-history row(s) for the evaluated test products, which indicates that the replay includes an "
            "extremely sparse local-history case."
        )

    lines.extend(
        [
            "",
            "## mh Summary",
            "",
            "| mh | Replay status | Original competitor failure | New prices F1 | New prices G | New products F1 | New products G | Combined F1 | Combined G |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in mh_df.itertuples(index=False):
        def fmt(value: Any) -> str:
            if pd.isna(value):
                return "NA"
            return f"{float(value):.4f}"

        lines.append(
            f"| {row.mh_level} | {row.status} | {row.original_competitor_failure_reason} | "
            f"{fmt(row.new_prices_f1)} | {fmt(row.new_prices_g_mean)} | "
            f"{fmt(row.new_products_f1)} | {fmt(row.new_products_g_mean)} | "
            f"{fmt(row.combined_f1)} | {fmt(row.combined_g_mean)} |"
        )

    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- `split_metrics.csv`: per-split replay metrics for each failed mh case.")
    lines.append("- `mh_summary.csv`: one row per failed mh case, including weighted combined scores when both splits are evaluable.")
    lines.append("- `summary.json`: machine-readable summary of the replay output.")
    lines.append("- `data_references.md`: provenance note for thesis-facing reuse.")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_data_references() -> None:
    content = """# Country-Level IF Replay Data References

## Thesis artifact references

| Thesis item | Source file | Presents data |
| --- | --- | --- |
| Table `country_if_country4_competitor3_replay` | `results/analysis/country_level_if_country4_competitor3_replay/mh_summary.csv` | Weighted combined and split-level replay metrics for the retained country-level \\(\\mathrm{IF}\\) configuration on the failed `COUNTRY_4 / B2C / COMPETITOR_3_COUNTRY_4_2026-02-08` competitor case across `mh10`, `mh15`, `mh20`, `mh25`, and `mh30`. |

## Raw aggregation inputs

| Source path | Contributes data for | Presents data |
| --- | --- | --- |
| `results/tuning/forests/single_config_optimized_mh5_run/` | Retained country-level \\(\\mathrm{IF}\\) configurations and original competitor-level failure reasons | Nested `by_country/.../if/best_configuration.json` payloads define the replayed country-level model configuration for each `mh`, while nested `by_competitor/.../if/summary.json` payloads record the original competitor-level failure reason. |
| `data-subsets/` | Replay datasets | The competitor-level `train.parquet`, `test_new_prices.parquet`, and `test_new_products.parquet` files under each failed `mh` level provide the evaluation target for the replay. |

## Notes

- The replay prefers persisted country-level IF artifacts under `artifacts/models/` when they exist locally.
- When the persisted artifact is absent, the replay refits the retained country-level configuration from the saved sweep metadata before evaluating it on the failed competitor case.
"""
    (OUTPUT_ROOT / "data_references.md").write_text(content, encoding="utf-8")


def _write_summary_json(split_df: pd.DataFrame, mh_df: pd.DataFrame) -> None:
    payload = {
        "target_country": TARGET_COUNTRY,
        "target_segment": TARGET_SEGMENT,
        "target_competitor": TARGET_COMPETITOR,
        "target_date": TARGET_DATE,
        "failed_mh_levels": list(FAILED_MH_LEVELS),
        "n_trials": N_TRIALS,
        "injection_rate": INJECTION_RATE,
        "split_weights": SPLIT_WEIGHTS,
        "split_results": split_df.to_dict(orient="records"),
        "mh_summary": mh_df.to_dict(orient="records"),
    }
    (OUTPUT_ROOT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    persistence = ModelPersistence()
    detector_cache: dict[str, tuple[Any, str, CountryModelConfig]] = {}
    split_results: list[SplitReplayResult] = []

    for mh_level in FAILED_MH_LEVELS:
        failure_reason = _load_failure_reason(mh_level)
        config = _load_country_model_config(mh_level)
        if not _mh_has_any_test_rows(mh_level):
            detector = None
            model_source = "not_loaded_empty_test_split"
        else:
            cached_entry = detector_cache.get(mh_level)
            if cached_entry is None:
                detector = None
                model_source = ""
            else:
                detector, model_source, _ = cached_entry

            if detector is None:
                detector, model_source = _load_or_refit_country_model(mh_level, config, persistence)
                detector_cache[mh_level] = (detector, model_source, config)

        for split in TARGET_SPLITS:
            LOGGER.info("Replaying %s on %s %s", config.model_name, mh_level, split)
            split_results.append(
                _evaluate_split(
                    mh_level=mh_level,
                    split=split,
                    detector=detector,
                    model_source=model_source,
                    config=config,
                    failure_reason=failure_reason,
                )
            )

    split_df = pd.DataFrame([asdict(result) for result in split_results])
    mh_df = _build_mh_summary(split_results)

    split_df.to_csv(OUTPUT_ROOT / "split_metrics.csv", index=False)
    mh_df.to_csv(OUTPUT_ROOT / "mh_summary.csv", index=False)
    _write_summary_markdown(split_df, mh_df)
    _write_summary_json(split_df, mh_df)
    _write_data_references()

    LOGGER.info("Wrote replay outputs to %s", OUTPUT_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
