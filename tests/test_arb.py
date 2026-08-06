import pandas as pd
import pytest

from freight.signals.arb import reconstruct_arb, price_component, freight_value_ratio

DATES = pd.date_range("2024-01-01", periods=3, freq="D")


def test_reconstruct_arb_basic():
    dest = pd.Series([100, 102, 98], index=DATES)
    origin = pd.Series([90, 90, 90], index=DATES)
    freight = pd.Series([5, 5, 5], index=DATES)
    arb = reconstruct_arb(dest, origin, freight)
    assert list(arb) == [5, 7, 3]


def test_reconstruct_arb_with_other_costs():
    dest = pd.Series([100], index=DATES[:1])
    origin = pd.Series([90], index=DATES[:1])
    freight = pd.Series([5], index=DATES[:1])
    arb = reconstruct_arb(dest, origin, freight, other_costs=2.0)
    assert arb.iloc[0] == pytest.approx(3.0)


def test_price_component_matches_arb_minus_freight():
    dest = pd.Series([100, 102], index=DATES[:2])
    origin = pd.Series([90, 91], index=DATES[:2])
    freight = pd.Series([5, 6], index=DATES[:2])
    arb = reconstruct_arb(dest, origin, freight)
    pc = price_component(dest, origin)
    assert list(arb) == list(pc - freight)


def test_freight_value_ratio():
    freight = pd.Series([20, 22, 18], index=DATES)
    value = pd.Series([100, 100, 100], index=DATES)
    assert freight_value_ratio(freight, value) == pytest.approx(0.2, rel=1e-6)
