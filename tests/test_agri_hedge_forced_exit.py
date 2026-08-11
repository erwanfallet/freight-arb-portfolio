"""Golden tests T1-2 — the price at which the balance sheet forces an exit.

The page's deliverable is a closed form, therefore fully verifiable by hand:

    P* = (B/Q + P0) / (1 + im_rate)

With B = 250 M USD, Q = 100 kt, P0 = 2,572 USD/t (cocoa on 03/01/2023) and an implied
margin rate of 4.4988%:

    B/Q = 2,500
    P*  = (2,500 + 2,572) / 1.044988 = 5,072 / 1.044988 = 4,853.64 USD/t
"""
from __future__ import annotations

import pandas as pd
import pytest

from agri.chains.hedge_cost import (
    SHORT_HEDGE,
    HedgeCostError,
    HedgeParams,
    forced_exit_price,
    forced_exit_schedule,
    hedging_intensity,
    implied_margin_rate,
    load_real_hedge_frame,
)
from agri.data.bloomberg_loader import DEFAULT_PATH, load

pytestmark = pytest.mark.skipif(
    not DEFAULT_PATH.exists(), reason=f"Bloomberg file absent: {DEFAULT_PATH}"
)

BOOK_T = 100_000.0
LINE_USD = 250e6


@pytest.fixture(scope="module")
def cocoa() -> pd.Series:
    return load("cocoa_ny")


@pytest.fixture(scope="module")
def margin_rate() -> float:
    params = HedgeParams(side=SHORT_HEDGE, book_size_t=BOOK_T, credit_line_usd=LINE_USD)
    simulation = load_real_hedge_frame("cacao_ny", params=params)
    return implied_margin_rate(simulation, book_size_t=BOOK_T)


# ===========================================================================
# The closed form
# ===========================================================================
def test_closed_form_hand_computed():
    """On a minimal synthetic series, to isolate the algebra from market noise."""
    price = pd.Series(
        [2572.0, 3000.0, 6000.0], index=pd.to_datetime(["2023-01-03", "2023-06-01", "2024-02-01"])
    )
    result = forced_exit_price(
        price, inception="2023-01-03", book_size_t=BOOK_T,
        credit_line_usd=LINE_USD, im_rate=0.044988,
    )
    assert result.inception_price == pytest.approx(2572.0)
    assert result.exit_price == pytest.approx(4853.64, abs=0.01)
    assert result.crossed_on == pd.Timestamp("2024-02-01")


def test_exit_price_scales_with_the_credit_line():
    """Doubling the line pushes back the B/Q threshold, not the threshold twice over —
    the relationship is affine in B/Q, not proportional to price. That's what lets a
    desk redo the calculation in their head."""
    price = pd.Series([2000.0], index=pd.to_datetime(["2023-01-03"]))
    small = forced_exit_price(price, inception="2023-01-03", book_size_t=BOOK_T,
                              credit_line_usd=100e6, im_rate=0.05)
    large = forced_exit_price(price, inception="2023-01-03", book_size_t=BOOK_T,
                              credit_line_usd=200e6, im_rate=0.05)
    delta = large.exit_price - small.exit_price
    assert delta == pytest.approx((200e6 - 100e6) / BOOK_T / 1.05, abs=1e-6)


def test_bigger_book_lowers_the_exit_price():
    """Same line, a book twice as large: forced out earlier."""
    price = pd.Series([2000.0], index=pd.to_datetime(["2023-01-03"]))
    small_book = forced_exit_price(price, inception="2023-01-03", book_size_t=50_000.0,
                                   credit_line_usd=LINE_USD, im_rate=0.05)
    big_book = forced_exit_price(price, inception="2023-01-03", book_size_t=200_000.0,
                                 credit_line_usd=LINE_USD, im_rate=0.05)
    assert big_book.exit_price < small_book.exit_price


def test_never_crossed_is_reported_as_such():
    price = pd.Series(
        [2000.0, 2100.0], index=pd.to_datetime(["2023-01-03", "2023-06-01"])
    )
    result = forced_exit_price(price, inception="2023-01-03", book_size_t=BOOK_T,
                               credit_line_usd=LINE_USD, im_rate=0.05)
    assert result.crossed_on is None
    assert result.days_of_protection is None
    assert "never got there" in result.headline


def test_invalid_inputs_raise():
    price = pd.Series([2000.0], index=pd.to_datetime(["2023-01-03"]))
    with pytest.raises(HedgeCostError, match="book and credit line"):
        forced_exit_price(price, inception="2023-01-03", book_size_t=0.0,
                          credit_line_usd=LINE_USD, im_rate=0.05)
    with pytest.raises(HedgeCostError, match="no price"):
        forced_exit_price(price, inception="2030-01-01", book_size_t=BOOK_T,
                          credit_line_usd=LINE_USD, im_rate=0.05)


# ===========================================================================
# THE FACT THE PAGE EXPLAINS
# ===========================================================================
def test_hedging_six_years_early_bought_three_months(cocoa, margin_rate):
    """The result that carries the page.

    On real cocoa, a house hedged since January 2018 and a house hedged since January
    2024 are forced to exit **within less than three months of each other**, even
    though six years separate their entry points. The early-2024 move was violent
    enough to crush the dispersion of opening dates — which explains why so many
    houses hit the balance-sheet constraint at the same time rather than each in turn.
    """
    schedule = forced_exit_schedule(
        cocoa, ["2018-01-02", "2022-01-03", "2023-01-03", "2023-09-01", "2024-01-02"],
        book_size_t=BOOK_T, credit_line_usd=LINE_USD, im_rate=margin_rate,
    )
    assert len(schedule) == 5
    assert schedule["crossed on"].notna().all()

    span_days = (schedule["crossed on"].max() - schedule["crossed on"].min()).days
    assert span_days < 100, f"the forced exits span {span_days} days"

    # all fall within the Nov. 2023 - Feb. 2024 window
    assert schedule["crossed on"].min() >= pd.Timestamp("2023-11-01")
    assert schedule["crossed on"].max() <= pd.Timestamp("2024-03-31")


def test_earlier_inception_still_buys_some_protection(cocoa, margin_rate):
    """The dispersion is crushed, not zero: hedging early is still better, the order
    is respected. That's what distinguishes "timing doesn't matter" (false) from
    "timing only bought a few weeks" (true)."""
    schedule = forced_exit_schedule(
        cocoa, ["2018-01-02", "2022-01-03", "2023-01-03", "2023-09-01", "2024-01-02"],
        book_size_t=BOOK_T, credit_line_usd=LINE_USD, im_rate=margin_rate,
    ).sort_values("opened")
    assert schedule["days of protection"].is_monotonic_decreasing
    assert schedule["crossed on"].is_monotonic_increasing


def test_headroom_shrinks_as_the_market_runs(cocoa, margin_rate):
    """The later a position opens in a rising market, the less headroom the line leaves, in %."""
    schedule = forced_exit_schedule(
        cocoa, ["2018-01-02", "2023-01-03", "2024-01-02"],
        book_size_t=BOOK_T, credit_line_usd=LINE_USD, im_rate=margin_rate,
    ).sort_values("opened")
    assert schedule["headroom"].is_monotonic_decreasing


# ===========================================================================
# Hedging intensity — the Montesanto anchor
# ===========================================================================
def test_hedging_intensity_climbs_from_margin_only_to_near_book_value():
    """At first only the initial margin is posted (~4.5% of price); when the market
    runs against the hedge, variation margin piles up until it ties up almost as much
    as the value of the protected stock."""
    params = HedgeParams(side=SHORT_HEDGE, book_size_t=BOOK_T, credit_line_usd=LINE_USD)
    simulation = load_real_hedge_frame("cacao_ny", params=params)
    intensity = hedging_intensity(
        simulation, book_size_t=BOOK_T, calm_window=("2018-04-02", "2022-12-31")
    )
    assert intensity.calm_median < 0.15
    assert intensity.peak_ratio > 0.7
    assert intensity.peak_ratio > intensity.calm_median * 5
    assert "Montesanto" in intensity.headline


def test_hedging_intensity_peak_lands_in_the_crisis_window():
    params = HedgeParams(side=SHORT_HEDGE, book_size_t=BOOK_T, credit_line_usd=LINE_USD)
    simulation = load_real_hedge_frame("cacao_ny", params=params)
    intensity = hedging_intensity(simulation, book_size_t=BOOK_T)
    assert pd.Timestamp("2024-01-01") <= intensity.peak_date <= pd.Timestamp("2025-06-30")


def test_implied_margin_rate_is_plausible():
    """An initial margin rate outside [1%, 15%] of price would signal a poorly
    calibrated proxy rather than a real clearinghouse schedule."""
    params = HedgeParams(side=SHORT_HEDGE, book_size_t=BOOK_T, credit_line_usd=LINE_USD)
    simulation = load_real_hedge_frame("cacao_ny", params=params)
    rate = implied_margin_rate(simulation, book_size_t=BOOK_T)
    assert 0.01 < rate < 0.15
