"""Shared feature helpers for tree-based anomaly detectors.

This module centralizes the tabular feature schema used by the tree-based
research detectors (Isolation Forest, EIF, and RRCF). The goal is to keep
training and inference aligned across all methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.features.numeric import NumericFeatures
from src.features.temporal import DEFAULT_HISTORY_DEPTH, MIN_OBSERVATIONS, TemporalFeatures

TREE_FEATURE_SCHEMA_VERSION = "2026-03-26-v1"

TREE_FEATURE_NAMES = [
    "price",
    "price_log",
    "price_ratio",
    "has_list_price",
    "rolling_mean",
    "rolling_std",
    "rolling_min",
    "rolling_max",
    "price_zscore",
    "price_change_pct",
    "price_vs_mean_ratio",
    "price_range_position",
]


@dataclass
class TreeFeatureVector:
    """Feature vector prepared for tree-based detectors."""

    features: np.ndarray
    feature_names: list[str]
    competitor_product_id: str
    competitor: str
    is_valid: bool
    missing_features: list[str] = field(default_factory=list)


def prepare_tree_feature_vector(
    numeric_features: NumericFeatures,
    temporal_features: TemporalFeatures,
) -> TreeFeatureVector:
    """Prepare the shared tree feature vector for one record."""
    feature_dict: dict[str, float | None] = {
        "price": numeric_features.price,
        "price_log": numeric_features.price_log,
        "price_ratio": numeric_features.price_ratio,
        "has_list_price": 1.0 if numeric_features.has_list_price else 0.0,
    }
    missing_features: list[str] = []

    if temporal_features.has_sufficient_history:
        feature_dict["rolling_mean"] = temporal_features.rolling_mean
        feature_dict["rolling_std"] = temporal_features.rolling_std
        feature_dict["rolling_min"] = temporal_features.rolling_min
        feature_dict["rolling_max"] = temporal_features.rolling_max
        feature_dict["price_zscore"] = temporal_features.price_zscore
        feature_dict["price_change_pct"] = temporal_features.price_change_pct

        if temporal_features.rolling_mean and temporal_features.rolling_mean > 0:
            feature_dict["price_vs_mean_ratio"] = (
                numeric_features.price / temporal_features.rolling_mean
            )
        else:
            feature_dict["price_vs_mean_ratio"] = 1.0

        if (
            temporal_features.rolling_max is not None
            and temporal_features.rolling_min is not None
        ):
            price_range = temporal_features.rolling_max - temporal_features.rolling_min
            if price_range > 0:
                feature_dict["price_range_position"] = (
                    numeric_features.price - temporal_features.rolling_min
                ) / price_range
            else:
                feature_dict["price_range_position"] = 0.5
        else:
            feature_dict["price_range_position"] = 0.5
    else:
        for name in TREE_FEATURE_NAMES[4:]:
            feature_dict[name] = None
            missing_features.append(name)

    features: list[float] = []
    is_valid = True
    for feature_name in TREE_FEATURE_NAMES:
        value = feature_dict[feature_name]
        if value is None:
            features.append(0.0)
            is_valid = False
        elif np.isnan(value) or np.isinf(value):
            features.append(0.0)
            is_valid = False
            missing_features.append(f"{feature_name}_nan")
        else:
            features.append(float(value))

    return TreeFeatureVector(
        features=np.asarray(features, dtype=np.float64),
        feature_names=list(TREE_FEATURE_NAMES),
        competitor_product_id=numeric_features.competitor_product_id,
        competitor=numeric_features.competitor,
        is_valid=is_valid,
        missing_features=missing_features,
    )


def extract_tree_features_vectorized(
    df: pd.DataFrame,
    window_size: int = DEFAULT_HISTORY_DEPTH,
) -> np.ndarray:
    """Extract the shared tree feature matrix from a price DataFrame."""
    if df.empty:
        return np.empty((0, len(TREE_FEATURE_NAMES)), dtype=np.float64)

    n_rows = len(df)
    frame = df.copy()

    if "first_seen_at" in frame.columns:
        frame = frame.sort_values(["product_id", "first_seen_at"])
    else:
        frame = frame.sort_values(["product_id"])

    grouped_price = frame.groupby("product_id")["price"]
    shifted_price = grouped_price.shift(1)

    frame["rolling_mean"] = grouped_price.transform(
        lambda values: values.shift(1).rolling(window=window_size, min_periods=1).mean()
    )
    frame["rolling_std"] = grouped_price.transform(
        lambda values: values.shift(1).rolling(window=window_size, min_periods=1).std()
    )
    frame["rolling_min"] = grouped_price.transform(
        lambda values: values.shift(1).rolling(window=window_size, min_periods=1).min()
    )
    frame["rolling_max"] = grouped_price.transform(
        lambda values: values.shift(1).rolling(window=window_size, min_periods=1).max()
    )
    frame["obs_count"] = shifted_price.groupby(frame["product_id"]).transform(
        lambda values: values.rolling(window=window_size, min_periods=1).count()
    ).fillna(0).astype(np.int32)

    frame["prev_price"] = shifted_price
    frame["price_change_pct"] = np.where(
        (frame["prev_price"].notna()) & (frame["prev_price"] > 0),
        (frame["price"] - frame["prev_price"]) / frame["prev_price"],
        0.0,
    )

    price = pd.to_numeric(frame["price"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    list_price_series = (
        frame["list_price"]
        if "list_price" in frame.columns
        else pd.Series(np.nan, index=frame.index, dtype=np.float64)
    )
    list_price = pd.to_numeric(list_price_series, errors="coerce").to_numpy(dtype=np.float64)
    rolling_mean = pd.to_numeric(frame["rolling_mean"], errors="coerce").to_numpy(dtype=np.float64)
    rolling_std = pd.to_numeric(frame["rolling_std"], errors="coerce").to_numpy(dtype=np.float64)
    rolling_min = pd.to_numeric(frame["rolling_min"], errors="coerce").to_numpy(dtype=np.float64)
    rolling_max = pd.to_numeric(frame["rolling_max"], errors="coerce").to_numpy(dtype=np.float64)
    obs_count = frame["obs_count"].to_numpy(dtype=np.int32)
    has_history = obs_count >= MIN_OBSERVATIONS

    feat_price = price
    feat_price_log = np.log(np.maximum(price, 1e-10) + 1.0)

    valid_list_price = ~np.isnan(list_price) & (list_price > 0)
    feat_price_ratio = np.ones(n_rows, dtype=np.float64)
    feat_price_ratio[valid_list_price] = price[valid_list_price] / list_price[valid_list_price]
    feat_has_list_price = (~np.isnan(list_price)).astype(np.float64)

    feat_rolling_mean = np.zeros(n_rows, dtype=np.float64)
    feat_rolling_mean[has_history] = np.nan_to_num(rolling_mean[has_history], nan=0.0)

    feat_rolling_std = np.zeros(n_rows, dtype=np.float64)
    feat_rolling_std[has_history] = np.nan_to_num(rolling_std[has_history], nan=0.0)

    feat_rolling_min = np.zeros(n_rows, dtype=np.float64)
    feat_rolling_min[has_history] = np.nan_to_num(rolling_min[has_history], nan=0.0)

    feat_rolling_max = np.zeros(n_rows, dtype=np.float64)
    feat_rolling_max[has_history] = np.nan_to_num(rolling_max[has_history], nan=0.0)

    feat_price_zscore = np.zeros(n_rows, dtype=np.float64)
    valid_zscore = has_history & (rolling_std > 0)
    feat_price_zscore[valid_zscore] = (
        price[valid_zscore] - rolling_mean[valid_zscore]
    ) / rolling_std[valid_zscore]

    feat_price_change_pct = np.zeros(n_rows, dtype=np.float64)
    feat_price_change_pct[has_history] = (
        pd.to_numeric(frame["price_change_pct"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float64)[has_history]
    )

    feat_price_vs_mean_ratio = np.zeros(n_rows, dtype=np.float64)
    valid_mean = has_history & (rolling_mean > 0)
    feat_price_vs_mean_ratio[has_history & ~valid_mean] = 1.0
    feat_price_vs_mean_ratio[valid_mean] = price[valid_mean] / rolling_mean[valid_mean]

    feat_price_range_position = np.zeros(n_rows, dtype=np.float64)
    price_range = rolling_max - rolling_min
    valid_range = has_history & (price_range > 0)
    feat_price_range_position[has_history & ~valid_range] = 0.5
    feat_price_range_position[valid_range] = (
        price[valid_range] - rolling_min[valid_range]
    ) / price_range[valid_range]

    matrix = np.column_stack(
        [
            feat_price,
            feat_price_log,
            feat_price_ratio,
            feat_has_list_price,
            feat_rolling_mean,
            feat_rolling_std,
            feat_rolling_min,
            feat_rolling_max,
            feat_price_zscore,
            feat_price_change_pct,
            feat_price_vs_mean_ratio,
            feat_price_range_position,
        ]
    )

    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def infer_tree_training_valid_mask(X: np.ndarray) -> np.ndarray:
    """Infer which rows in a feature matrix are valid training samples."""
    if X.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {X.shape}")
    if X.shape[1] != len(TREE_FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(TREE_FEATURE_NAMES)} tree features, got {X.shape[1]}"
        )

    temporal_slice = X[:, 4:]
    has_temporal_signal = ~np.all(np.isclose(temporal_slice, 0.0), axis=1)
    finite_rows = np.isfinite(X).all(axis=1)
    return finite_rows & has_temporal_signal
