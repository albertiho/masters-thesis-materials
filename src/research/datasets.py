from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SPLIT_MANIFEST_NAME = "split_manifest.json"
ALLOWED_DATASET_SCOPES = ("global", "country", "country-market", "competitor")
ALLOWED_SPLIT_NAMES = ("train", "test_new_prices", "test_new_products")
_SCOPE_ALIASES = {
    "global": "global",
    "country": "country",
    "country-market": "country-market",
    "country_market": "country-market",
    "country-segment": "country-market",
    "country_segment": "country-market",
    "competitor": "competitor",
}

_COUNTRY_SEGMENT_FILENAME = re.compile(
    r"^(?P<country>COUNTRY_\d+)_(?P<segment>B2[BC])_(?P<date>\d{4}-\d{2}-\d{2})$"
)
_DATE_SUFFIX = re.compile(r"_(\d{4}-\d{2}-\d{2})$")


@dataclass(frozen=True)
class BaseDatasetJob:
    """One split-manifest job resolved to a stable dataset identifier."""

    scope: str
    dataset_id: str
    country: str
    segment: str
    competitor_id: str | None
    date_token: str
    min_history: int
    generated_files: dict[str, str]


@dataclass(frozen=True)
class ResolvedDataset:
    """Resolved training and evaluation files for one manifest dataset."""

    dataset_id: str
    scope: str
    min_history: int
    train_split: str
    evaluation_splits: tuple[str, ...]
    dataset_name: str
    component_dataset_ids: tuple[str, ...]
    countries: tuple[str, ...]
    segments: tuple[str, ...]
    train_files: tuple[Path, ...]
    evaluation_files: dict[str, tuple[Path, ...]]
    source_dataset_paths: tuple[str, ...]


class DatasetResolutionError(ValueError):
    """Raised when a manifest references an unsupported dataset layout."""


def project_root() -> Path:
    """Return the standalone research repository root."""
    return Path(__file__).resolve().parents[3]


def normalize_split_name(value: str) -> str:
    """Normalize split names to the Phase 1 split keys."""
    normalized = value.strip().lower().replace("-", "_")
    mapping = {
        "train": "train",
        "new_prices": "test_new_prices",
        "test_new_prices": "test_new_prices",
        "test_prices": "test_new_prices",
        "new_products": "test_new_products",
        "test_new_products": "test_new_products",
        "test_products": "test_new_products",
    }
    if normalized not in mapping:
        raise DatasetResolutionError(
            f"Unsupported split name {value!r}; expected one of {sorted(mapping)}"
        )
    return mapping[normalized]


def normalize_scope(value: str) -> str:
    """Normalize scope aliases to the canonical Phase 1 scope names."""
    normalized = value.strip().lower().replace(" ", "_")
    if normalized not in _SCOPE_ALIASES:
        raise DatasetResolutionError(
            f"Unsupported dataset scope {value!r}; expected one of {sorted(_SCOPE_ALIASES)}"
        )
    return _SCOPE_ALIASES[normalized]


def load_split_manifest(data_root: str | Path) -> dict[str, Any]:
    """Load the deterministic Phase 1 split manifest from the given data root."""
    resolved_data_root = _resolve_path(data_root)
    manifest_path = resolved_data_root / "derived" / SPLIT_MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing split manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def list_available_dataset_ids(data_root: str | Path) -> dict[str, list[str]]:
    """List all dataset identifiers derived from the Phase 1 split manifest."""
    jobs = _build_base_jobs(load_split_manifest(data_root))
    country_market = sorted(
        {
            job.dataset_id
            for job in jobs
            if job.scope == "country-market"
        }
    )
    competitor = sorted({job.dataset_id for job in jobs if job.scope == "competitor"})
    countries = sorted({job.country for job in jobs if job.scope == "country-market"})
    return {
        "global": ["GLOBAL"],
        "country": countries,
        "country-market": country_market,
        "competitor": competitor,
    }


def resolve_dataset(
    *,
    data_root: str | Path,
    scope: str,
    dataset_id: str,
    min_history: int,
    train_split: str,
    evaluation_splits: list[str] | tuple[str, ...],
) -> ResolvedDataset:
    """Resolve one dataset identifier into concrete split files."""
    normalized_scope = normalize_scope(scope)
    normalized_train = normalize_split_name(train_split)
    normalized_evals = tuple(normalize_split_name(split) for split in evaluation_splits)

    jobs = _build_base_jobs(load_split_manifest(data_root))
    if normalized_scope == "global":
        component_jobs = _resolve_global_jobs(jobs, dataset_id, min_history)
    elif normalized_scope == "country":
        component_jobs = _resolve_country_jobs(jobs, dataset_id, min_history)
    elif normalized_scope == "country-market":
        component_jobs = _resolve_exact_jobs(jobs, normalized_scope, dataset_id, min_history)
    else:
        component_jobs = _resolve_exact_jobs(jobs, normalized_scope, dataset_id, min_history)

    data_root_path = _resolve_path(data_root)
    derived_root = data_root_path / "derived"

    split_to_paths: dict[str, tuple[Path, ...]] = {}
    for split_name in (normalized_train, *normalized_evals):
        resolved_paths = tuple(
            sorted(
                derived_root / job.generated_files[split_name]
                for job in component_jobs
            )
        )
        missing_paths = [path for path in resolved_paths if not path.exists()]
        if missing_paths:
            raise DatasetResolutionError(
                f"Split-manifest resolved missing files for dataset {dataset_id!r}: "
                f"{[path.as_posix() for path in missing_paths]!r}"
            )
        split_to_paths[split_name] = resolved_paths

    source_dataset_paths = tuple(
        _display_path(path)
        for split_name in (normalized_train, *normalized_evals)
        for path in split_to_paths[split_name]
    )
    countries = tuple(sorted({job.country for job in component_jobs}))
    segments = tuple(sorted({job.segment for job in component_jobs}))
    component_dataset_ids = tuple(sorted(job.dataset_id for job in component_jobs))

    return ResolvedDataset(
        dataset_id=dataset_id,
        scope=normalized_scope,
        min_history=min_history,
        train_split=normalized_train,
        evaluation_splits=normalized_evals,
        dataset_name=f"{dataset_id}_mh{min_history}",
        component_dataset_ids=component_dataset_ids,
        countries=countries,
        segments=segments,
        train_files=split_to_paths[normalized_train],
        evaluation_files={split_name: split_to_paths[split_name] for split_name in normalized_evals},
        source_dataset_paths=source_dataset_paths,
    )


def resolve_dataset_by_id(
    *,
    data_root: str | Path,
    dataset_id: str,
    min_history: int,
    train_split: str,
    evaluation_splits: list[str] | tuple[str, ...],
) -> ResolvedDataset:
    """Resolve a dataset id without separately specifying the scope."""
    catalog = list_available_dataset_ids(data_root)
    matches = [scope for scope, dataset_ids in catalog.items() if dataset_id in dataset_ids]
    if len(matches) != 1:
        raise DatasetResolutionError(
            f"Dataset id {dataset_id!r} is ambiguous or unknown across scopes: {matches!r}"
        )
    return resolve_dataset(
        data_root=data_root,
        scope=matches[0],
        dataset_id=dataset_id,
        min_history=min_history,
        train_split=train_split,
        evaluation_splits=evaluation_splits,
    )


def resolve_dataset_group(
    *,
    data_root: str | Path,
    dataset_ids: list[str] | tuple[str, ...],
    min_history: int,
    train_split: str,
    evaluation_splits: list[str] | tuple[str, ...],
    group_id: str,
    scope_label: str = "global",
) -> ResolvedDataset:
    """Resolve and concatenate an explicit list of dataset ids into one bundle."""
    normalized_ids = [str(dataset_id).strip() for dataset_id in dataset_ids if str(dataset_id).strip()]
    if not normalized_ids:
        raise DatasetResolutionError("Dataset groups must declare at least one dataset_id")

    resolved = [
        resolve_dataset_by_id(
            data_root=data_root,
            dataset_id=dataset_id,
            min_history=min_history,
            train_split=train_split,
            evaluation_splits=evaluation_splits,
        )
        for dataset_id in normalized_ids
    ]
    return combine_resolved_datasets(
        resolved,
        dataset_id=group_id,
        scope=scope_label,
        min_history=min_history,
        train_split=train_split,
        evaluation_splits=evaluation_splits,
    )


def combine_resolved_datasets(
    datasets: list[ResolvedDataset] | tuple[ResolvedDataset, ...],
    *,
    dataset_id: str,
    scope: str,
    min_history: int,
    train_split: str,
    evaluation_splits: list[str] | tuple[str, ...],
) -> ResolvedDataset:
    """Combine compatible resolved datasets into one logical bundle."""
    if not datasets:
        raise DatasetResolutionError("Cannot combine an empty dataset list")

    normalized_scope = normalize_scope(scope)
    normalized_train = normalize_split_name(train_split)
    normalized_evals = tuple(normalize_split_name(split) for split in evaluation_splits)

    train_files = tuple(
        sorted(
            path
            for dataset in datasets
            for path in dataset.train_files
        )
    )
    evaluation_files = {
        split_name: tuple(
            sorted(
                path
                for dataset in datasets
                for path in dataset.evaluation_files.get(split_name, ())
            )
        )
        for split_name in normalized_evals
    }
    source_dataset_paths = tuple(
        dict.fromkeys(
            path
            for dataset in datasets
            for path in dataset.source_dataset_paths
        )
    )
    component_dataset_ids = tuple(
        sorted(
            {
                component
                for dataset in datasets
                for component in dataset.component_dataset_ids
            }
        )
    )
    countries = tuple(
        sorted(
            {
                country
                for dataset in datasets
                for country in dataset.countries
            }
        )
    )
    segments = tuple(
        sorted(
            {
                segment
                for dataset in datasets
                for segment in dataset.segments
            }
        )
    )

    return ResolvedDataset(
        dataset_id=dataset_id,
        scope=normalized_scope,
        min_history=min_history,
        train_split=normalized_train,
        evaluation_splits=normalized_evals,
        dataset_name=f"{dataset_id}_mh{min_history}",
        component_dataset_ids=component_dataset_ids,
        countries=countries,
        segments=segments,
        train_files=train_files,
        evaluation_files=evaluation_files,
        source_dataset_paths=source_dataset_paths,
    )


def country_market_dataset_id_for_dataset_id(dataset_id: str) -> str:
    """Derive the country-market dataset id for a dataset id when possible."""
    normalized = str(dataset_id).strip()
    competitor_match = re.match(r"^(COUNTRY_\d+)__(B2[BC])__.+$", normalized)
    if competitor_match:
        return f"{competitor_match.group(1)}_{competitor_match.group(2)}"

    country_market_match = re.match(r"^(COUNTRY_\d+)_(B2[BC])$", normalized)
    if country_market_match:
        return normalized

    raise DatasetResolutionError(
        f"Dataset id {dataset_id!r} cannot be mapped to a country-market dataset"
    )


def competitor_dataset_ids_for_country_market(
    data_root: str | Path,
    country_market_dataset_id: str,
) -> list[str]:
    """List competitor dataset ids contained in one country-market dataset."""
    normalized = str(country_market_dataset_id).strip()
    match = re.match(r"^(COUNTRY_\d+)_(B2[BC])$", normalized)
    if match is None:
        raise DatasetResolutionError(
            f"Country-market dataset id must look like COUNTRY_N_B2B/B2C, got {country_market_dataset_id!r}"
        )
    prefix = f"{match.group(1)}__{match.group(2)}__"
    return [
        dataset_id
        for dataset_id in list_available_dataset_ids(data_root)["competitor"]
        if dataset_id.startswith(prefix)
    ]


def _build_base_jobs(split_manifest: dict[str, Any]) -> list[BaseDatasetJob]:
    jobs: list[BaseDatasetJob] = []
    for raw_job in split_manifest.get("split_jobs", []):
        input_file = str(raw_job["input_file"])
        min_history = int(raw_job["min_history"])
        generated_files = {
            split_name: str(details["path"])
            for split_name, details in raw_job["generated_files"].items()
        }
        parsed = _parse_input_file(input_file)
        jobs.append(
            BaseDatasetJob(
                scope=parsed["scope"],
                dataset_id=parsed["dataset_id"],
                country=parsed["country"],
                segment=parsed["segment"],
                competitor_id=parsed["competitor_id"],
                date_token=parsed["date_token"],
                min_history=min_history,
                generated_files=generated_files,
            )
        )
    return jobs


def _parse_input_file(relative_path: str) -> dict[str, str | None]:
    input_path = Path(relative_path)
    parts = input_path.parts
    if len(parts) < 2:
        raise DatasetResolutionError(f"Unexpected split-manifest input path: {relative_path}")

    if parts[0] == "by_country_segment":
        stem = input_path.stem
        match = _COUNTRY_SEGMENT_FILENAME.match(stem)
        if match is None:
            raise DatasetResolutionError(f"Unexpected country-market file name: {relative_path}")
        country = match.group("country")
        segment = match.group("segment")
        date_token = match.group("date")
        dataset_id = f"{country}_{segment}"
        return {
            "scope": "country-market",
            "dataset_id": dataset_id,
            "country": country,
            "segment": segment,
            "competitor_id": None,
            "date_token": date_token,
        }

    if parts[0] == "by_competitor" and len(parts) >= 4:
        country = parts[1]
        segment = parts[2]
        stem = input_path.stem
        date_match = _DATE_SUFFIX.search(stem)
        if date_match is None:
            raise DatasetResolutionError(f"Missing date token in competitor file name: {relative_path}")
        date_token = date_match.group(1)
        competitor_id = stem[: -(len(date_token) + 1)]
        dataset_id = f"{country}__{segment}__{competitor_id}"
        return {
            "scope": "competitor",
            "dataset_id": dataset_id,
            "country": country,
            "segment": segment,
            "competitor_id": competitor_id,
            "date_token": date_token,
        }

    raise DatasetResolutionError(f"Unsupported split-manifest input path: {relative_path}")


def _resolve_exact_jobs(
    jobs: list[BaseDatasetJob],
    scope: str,
    dataset_id: str,
    min_history: int,
) -> list[BaseDatasetJob]:
    matched = [
        job
        for job in jobs
        if job.scope == scope and job.dataset_id == dataset_id and job.min_history == min_history
    ]
    if not matched:
        raise DatasetResolutionError(
            f"No {scope} dataset {dataset_id!r} with min_history={min_history} in split manifest"
        )
    return matched


def _resolve_country_jobs(
    jobs: list[BaseDatasetJob],
    dataset_id: str,
    min_history: int,
) -> list[BaseDatasetJob]:
    matched = [
        job
        for job in jobs
        if job.scope == "country-market" and job.country == dataset_id and job.min_history == min_history
    ]
    if not matched:
        raise DatasetResolutionError(
            f"No country dataset {dataset_id!r} with min_history={min_history} in split manifest"
        )
    return matched


def _resolve_global_jobs(
    jobs: list[BaseDatasetJob],
    dataset_id: str,
    min_history: int,
) -> list[BaseDatasetJob]:
    if dataset_id != "GLOBAL":
        raise DatasetResolutionError("Global scope requires dataset_id 'GLOBAL'")
    matched = [
        job for job in jobs if job.scope == "country-market" and job.min_history == min_history
    ]
    if not matched:
        raise DatasetResolutionError(
            f"No global dataset components with min_history={min_history} in split manifest"
        )
    return matched


def _normalize_scope(scope: str) -> str:
    return normalize_scope(scope)


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return project_root() / candidate


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root()).as_posix()
    except ValueError:
        return path.resolve().as_posix()
