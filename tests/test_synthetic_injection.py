from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from src.research.evaluation.synthetic import SyntheticAnomalyType, inject_anomalies_to_dataframe


def test_currency_swap_updates_currency_column_and_details() -> None:
    frame = pd.DataFrame(
        {
            "price": [100.0],
            "currency": ["CURRENCY_1"],
        }
    )

    injected_frame, labels, details = inject_anomalies_to_dataframe(
        frame,
        injection_rate=1.0,
        seed=42,
        anomaly_types=[SyntheticAnomalyType.CURRENCY_SWAP],
    )

    assert labels.tolist() == [True]
    assert injected_frame.loc[0, "currency"] == "CURRENCY_5"
    assert injected_frame.loc[0, "price"] == 1000.0
    assert details == [
        {
            "original_currency": "CURRENCY_1",
            "new_currency": "CURRENCY_5",
            "currency_swap_factor": 10.0,
            "change_pct": 9.0,
            "index": 0,
            "original_price": 100.0,
            "new_price": 1000.0,
            "anomaly_type": "currency_swap",
            "injection_phase": 2,
        }
    ]


def test_price_spike_scales_list_price_with_price() -> None:
    frame = pd.DataFrame(
        {
            "price": [100.0],
            "list_price": [120.0],
            "currency": ["CURRENCY_1"],
        }
    )

    injected_frame, labels, details = inject_anomalies_to_dataframe(
        frame,
        injection_rate=1.0,
        seed=42,
        anomaly_types=[SyntheticAnomalyType.PRICE_SPIKE],
        spike_range=(2.0, 2.0),
    )

    assert labels.tolist() == [True]
    assert injected_frame.loc[0, "price"] == pytest.approx(200.0)
    assert injected_frame.loc[0, "list_price"] == pytest.approx(240.0)
    assert injected_frame.loc[0, "price"] / injected_frame.loc[0, "list_price"] == pytest.approx(
        frame.loc[0, "price"] / frame.loc[0, "list_price"]
    )
    assert details[0]["list_price_multiplier"] == pytest.approx(2.0)
    assert details[0]["original_list_price"] == pytest.approx(120.0)
    assert details[0]["new_list_price"] == pytest.approx(240.0)


def test_price_drop_clamps_reduced_price_to_one_and_scales_list_price() -> None:
    frame = pd.DataFrame(
        {
            "price": [1.2],
            "list_price": [1.5],
            "currency": ["CURRENCY_1"],
        }
    )

    injected_frame, labels, details = inject_anomalies_to_dataframe(
        frame,
        injection_rate=1.0,
        seed=42,
        anomaly_types=[SyntheticAnomalyType.PRICE_DROP],
        drop_range=(0.1, 0.1),
    )

    assert labels.tolist() == [True]
    assert injected_frame.loc[0, "price"] == pytest.approx(1.0)
    assert injected_frame.loc[0, "list_price"] == pytest.approx(1.25)
    assert injected_frame.loc[0, "price"] / injected_frame.loc[0, "list_price"] == pytest.approx(
        frame.loc[0, "price"] / frame.loc[0, "list_price"]
    )
    assert details[0]["list_price_multiplier"] == pytest.approx(1.0 / 1.2)
    assert details[0]["original_list_price"] == pytest.approx(1.5)
    assert details[0]["new_list_price"] == pytest.approx(1.25)
