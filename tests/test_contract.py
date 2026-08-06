from datetime import date

import pandas as pd
import pytest

from freight.ingest.contract import ContractViolation, SeriesContract, validate_series


def test_valid_contract_constructs():
    c = SeriesContract(
        ticker="C3", description="Baltic C3", native_unit="USD/t",
        frequency="daily", source="Bloomberg", last_valid_date=date(2026, 8, 4),
    )
    assert c.ticker == "C3"


def test_bad_frequency_rejected():
    with pytest.raises(ContractViolation):
        SeriesContract(ticker="C3", description="x", native_unit="USD/t",
                        frequency="fortnightly", source="Bloomberg")


def test_forward_fill_gap_policy_rejected():
    """A gap is information, not noise to smooth — forward-fill is not a legal gap policy."""
    with pytest.raises(ContractViolation):
        SeriesContract(ticker="C3", description="x", native_unit="USD/t",
                        frequency="daily", source="Bloomberg", gap_policy="forward_fill")


def test_stale_series_shown_as_current_is_rejected():
    """Reproduces the exact failure that killed the prior attempt: BSI frozen since 2017
    but displayed without flagging staleness."""
    contract = SeriesContract(
        ticker="BSI", description="Supramax index", native_unit="points",
        frequency="daily", source="Bloomberg", last_valid_date=date(2017, 3, 31),
    )
    df = pd.DataFrame({
        "date": pd.date_range("2026-06-01", periods=3, freq="D"),
        "ticker": "BSI",
        "valeur": [900, 905, 910],
    })
    with pytest.raises(ContractViolation, match="last_valid_date"):
        validate_series(df, contract, today=date(2026, 8, 5))


def test_undeclared_staleness_is_rejected():
    contract = SeriesContract(
        ticker="X", description="x", native_unit="USD/t",
        frequency="daily", source="Bloomberg",  # no last_valid_date set
    )
    df = pd.DataFrame({
        "date": [pd.Timestamp("2020-01-01")],
        "ticker": ["X"],
        "valeur": [1.0],
    })
    with pytest.raises(ContractViolation, match="fill in last_valid_date"):
        validate_series(df, contract, today=date(2026, 8, 5))


def test_series_matching_declared_last_valid_date_passes():
    contract = SeriesContract(
        ticker="X", description="x", native_unit="USD/t",
        frequency="daily", source="Bloomberg", last_valid_date=date(2020, 1, 1),
    )
    df = pd.DataFrame({
        "date": [pd.Timestamp("2020-01-01")],
        "ticker": ["X"],
        "valeur": [1.0],
    })
    validate_series(df, contract, today=date(2026, 8, 5))  # should not raise
