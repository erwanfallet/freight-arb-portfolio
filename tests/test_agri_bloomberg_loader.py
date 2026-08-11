"""Golden tests for the real Bloomberg loader.

These tests read the user's actual file (~/Desktop/Data Bloomberg.xlsx): they are
cleanly skipped if the file is absent, so the suite stays runnable on another machine.
"""
from __future__ import annotations

import pandas as pd
import pytest

from agri.data.bloomberg_loader import (
    DEFAULT_PATH,
    SERIES_SPECS,
    BloombergLoaderError,
    detect_unit_jumps,
    load,
    load_raw_series,
)

pytestmark = pytest.mark.skipif(
    not DEFAULT_PATH.exists(), reason=f"Bloomberg file absent: {DEFAULT_PATH}"
)


def test_all_registered_series_load_without_error():
    for key in SERIES_SPECS:
        series = load(key)
        assert len(series) > 100, f"{key}: too few observations ({len(series)})"
        assert isinstance(series.index, pd.DatetimeIndex)


def test_unknown_key_raises():
    with pytest.raises(BloombergLoaderError, match="unknown key"):
        load("this_ticker_does_not_exist")


def test_dropna_removes_leading_blank_prints():
    """Bloomberg sometimes exports a first dated row with no value — a future not yet
    fixed. `dropna=True` (default) must drop it."""
    with_na = load_raw_series("henry_hub", dropna=False)
    without_na = load_raw_series("henry_hub", dropna=True)
    assert len(without_na) <= len(with_na)
    assert without_na.isna().sum() == 0


def test_detect_unit_jumps_catches_the_known_jet_swap_defect():
    """The check that found the real defect: jet_swap_m1 alternates between USD/gal and
    c/gal several times across its history. This test locks in the detection — if it
    turns green by accident (data fixed on Bloomberg's side), the module needs to be
    re-evaluated, not the test silently deleted."""
    jumps = detect_unit_jumps(load("jet_swap_m1"))
    assert len(jumps) >= 5


def test_detect_unit_jumps_is_clean_on_jet_spot():
    """Direct contrast: the clean series triggers nothing — otherwise the check would
    be a test that always screams, and therefore useless."""
    jumps = detect_unit_jumps(load("jet_spot"))
    assert len(jumps) == 0


def test_detect_unit_jumps_is_clean_on_ulsd_ttf_henry_hub_eurusd():
    for key in ("ulsd", "ttf", "henry_hub", "eurusd"):
        jumps = detect_unit_jumps(load(key))
        assert len(jumps) == 0, f"{key} carries {len(jumps)} unexpected suspicious jump(s)"


def test_eurusd_is_in_the_usd_per_eur_convention():
    """Quoting-direction check (L-H4 of lng_netback): recent values must be on the
    order of 1.0-1.3 (USD per EUR), not 0.7-0.95 (EUR per USD)."""
    recent = load("eurusd").tail(250)
    assert recent.between(0.9, 1.5).mean() > 0.95


# ===========================================================================
# Extension: softs, CBOT grains, DCE, FX (cents/bushel + cruzeiro traps)
# ===========================================================================
def test_detect_unit_jumps_is_clean_on_all_new_series():
    new_keys = [
        "cocoa_ny", "cocoa_london", "coffee_arabica", "coffee_robusta",
        "sugar_no11", "sugar_no5", "cbot_soybean", "cbot_corn", "cbot_wheat",
        "cbot_soymeal", "cbot_soyoil", "dce_soymeal", "dce_soyoil", "usdbrl", "usdcny",
    ]
    for key in new_keys:
        jumps = detect_unit_jumps(load(key))
        assert len(jumps) == 0, f"{key} carries {len(jumps)} unexpected suspicious jump(s)"


def test_cbot_grains_are_converted_from_cents_to_dollars_per_bushel():
    """The trap found while building this extension: CBOT grains are quoted in CENTS
    per bushel in this export (1156.50 = 11.565 USD/bu), not in dollars. Without the
    scale=0.01, any formula expecting USD/bu (board_crush_usd_bu, financing_cost...)
    would come out a hundred times too large."""
    soy = load("cbot_soybean")
    corn = load("cbot_corn")
    wheat = load("cbot_wheat")
    # real 2026 orders of magnitude: soybean ~9-14 USD/bu, corn ~3-6, wheat ~5-9
    assert 8.0 < soy.iloc[-1] < 16.0
    assert 3.0 < corn.iloc[-1] < 7.0
    assert 4.0 < wheat.iloc[-1] < 10.0


def test_cbot_soymeal_and_soyoil_need_no_scaling():
    """Direct contrast with the previous test: these two are already in their native
    economic unit (USD/short ton, c/lb) — a reflexive scale=0.01 would break them."""
    meal = load("cbot_soymeal")
    oil = load("cbot_soyoil")
    assert 100.0 < meal.iloc[-1] < 700.0        # USD/short ton, real order of magnitude
    assert 10.0 < oil.iloc[-1] < 100.0          # c/lb, real order of magnitude


def test_usdbrl_excludes_the_pre_plano_real_era():
    """Before July 1994: pre-monetary-reform cruzeiro/cruzeiro real (Brazilian
    hyperinflation, values ~0.0004). A different currency, not an outlier — excluded
    via valid_from rather than rescaled."""
    usdbrl = load("usdbrl")
    assert usdbrl.index.min() >= pd.Timestamp("1994-07-01")
    assert (usdbrl > 0.3).all()  # no more near-zero values from the cruzeiro era


def test_cocoa_ny_peak_matches_the_real_2024_crisis():
    """External consistency check: cocoa really did peak around 12,000 USD/t in April
    2024 (Barry Callebaut, T1-2 source). If this peak didn't show up here, it would be
    a sign of rescaling or a truncated series."""
    cocoa = load("cocoa_ny")
    peak = cocoa.loc["2024-01-01":"2024-12-31"]
    assert peak.max() > 9000.0


def test_sofr_is_a_decimal_fraction_not_a_percent():
    """Defect found while building T1-2, and fixed here.

    Bloomberg quotes SOFR in PERCENT (5.40 at the peak of the 2023 tightening). Added
    as-is to a spread already expressed in decimal (250 bps -> 0.025) and then used as
    a decimal, it produced an all-in rate of 243% and inflated the financing cost by a
    factor of ~100 — without ever raising an error.

    The loader's contract is now: **every rate comes out as a decimal fraction**, ready
    to multiply an amount.
    """
    sofr = load("sofr")
    assert sofr.max() < 0.15, "SOFR is coming out in percent — the scale=0.01 got dropped"
    assert 0.03 < sofr.max() < 0.08, "the 2023 tightening peak should be around 5.4%"
    assert (sofr >= 0).all()


def test_financing_cost_is_plausible_once_sofr_is_decimal():
    """End-to-end check: the all-in rate applied in the simulation must stay within a
    market range, not at 243%."""
    from agri.chains.hedge_cost import (
        SHORT_HEDGE,
        HedgeParams,
        load_real_hedge_frame,
    )

    params = HedgeParams(side=SHORT_HEDGE, book_size_t=100_000.0, credit_line_usd=250e6)
    simulation = load_real_hedge_frame("cacao_ny", params=params)
    implied_rate = (simulation["financing_usd"] / simulation["cash_usd"] * 360).median()
    assert 0.01 < implied_rate < 0.12, f"implied all-in rate of {implied_rate:.1%}"


def test_dce_series_are_in_cny_thousands_not_usd():
    """Order-of-magnitude check: DCE meal/oil are quoted in CNY/t (thousands), not
    USD/t (hundreds) — a common mix-up between the two markets."""
    meal = load("dce_soymeal")
    oil = load("dce_soyoil")
    assert 1000.0 < meal.iloc[-1] < 8000.0
    assert 3000.0 < oil.iloc[-1] < 20000.0
