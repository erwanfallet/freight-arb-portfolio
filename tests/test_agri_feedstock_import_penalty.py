"""Golden tests T3-1 — the inversion into a discount, and the invariance that makes it hard to dispute.

This file's central test is
`test_the_discount_is_invariant_to_diesel_rin_and_plant_costs`: it verifies on the
FULL model (plant-gate value, RIN, opex, ROI) what the closed form claims, namely that
these terms cancel out between the two pathways. That's section S2's argument; without
this test, it's only a claim in a docstring.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agri.chains.feedstock_lcfs import (
    CENTS_PER_USD,
    LCFS_PROGRAM_HIGH_USD_T,
    LCFS_PROGRAM_LOW_USD_T,
    SOYOIL_DOMESTIC,
    UCO_IMPORTED,
    Feedstock,
    FeedstockError,
    crush_from_soyoil_lb,
    discount_burden,
    feedstock_breakeven_usd_lb,
    import_penalty,
    lcfs_breakeven,
    lcfs_neutral_price,
    load_soyoil_usd_lb,
    penalty_bounds,
    structural_exit,
)
from agri.data.bloomberg_loader import DEFAULT_PATH

needs_bloomberg = pytest.mark.skipif(
    not DEFAULT_PATH.exists(), reason=f"Bloomberg file absent: {DEFAULT_PATH}"
)


# ===========================================================================
# S2'S ARGUMENT — what cancels out really does cancel out
# ===========================================================================
@pytest.mark.parametrize("ulsd", [2.00, 3.50, 5.00])
@pytest.mark.parametrize("rin", [0.40, 1.20])
@pytest.mark.parametrize("opex,roi", [(0.30, 0.10), (0.55, 0.25), (1.10, 0.60)])
def test_the_discount_is_invariant_to_diesel_rin_and_plant_costs(ulsd, rin, opex, roi):
    """THE page's test.

    The indifference discount is computed via the full model — plant-gate value,
    RIN, opex, ROI, everything included — and checked against the closed form that
    contains none of them. Eighteen combinations of diesel, RIN and cost structure
    give exactly the same number: that's what lets the page say a reader can't
    dispute it by challenging its diesel forecast.
    """
    kwargs = dict(ulsd_usd_gal=ulsd, rin_d4_usd=rin, opex_usd_gal=opex, roi_usd_gal=roi)
    lcfs = 75.0

    # Indifference discount via the full model: at what price gap do the two
    # pathways deliver the same net advantage?
    breakeven_domestic = feedstock_breakeven_usd_lb(
        SOYOIL_DOMESTIC, lcfs_usd_t=lcfs, **kwargs
    )
    breakeven_imported = feedstock_breakeven_usd_lb(
        UCO_IMPORTED, lcfs_usd_t=lcfs, **kwargs
    )
    discount_from_full_model = breakeven_domestic - breakeven_imported

    assert import_penalty(lcfs).discount_required_usd_lb == pytest.approx(
        discount_from_full_model, rel=1e-12
    )


def test_the_closed_form_matches_the_existing_threshold_at_price_parity():
    """Cross-check against `lcfs_breakeven`, written earlier and independently: at
    feedstock price parity, the neutralising threshold must be the same number."""
    threshold = lcfs_breakeven(
        price_domestic_usd_lb=0.50,
        price_imported_usd_lb=0.50,
        lcfs_current_usd_t=75.0,
    )
    assert lcfs_neutral_price() == pytest.approx(threshold.lcfs_star_usd_t, rel=1e-12)


def test_the_penalty_vanishes_exactly_at_the_neutral_price():
    neutral = lcfs_neutral_price()
    assert import_penalty(neutral).discount_required_usd_lb == pytest.approx(0.0, abs=1e-12)
    assert import_penalty(neutral).imports_win_outright


def test_above_the_neutral_price_imports_can_pay_a_premium():
    """Beyond the threshold, the carbon advantage exceeds 45Z: the "required
    discount" turns negative, which reads as a premium the imported pathway can
    afford."""
    penalty = import_penalty(lcfs_neutral_price() + 100.0)
    assert penalty.discount_required_usd_lb < 0
    assert penalty.imports_win_outright
    assert "premium" in penalty.headline


def test_the_discount_decreases_monotonically_in_the_lcfs_price():
    discounts = [import_penalty(x).discount_required_usd_lb for x in (0.0, 50.0, 100.0, 200.0)]
    assert discounts == sorted(discounts, reverse=True)


def test_the_45z_credit_is_the_whole_penalty_when_the_lcfs_is_worthless():
    """At zero LCFS, the required discount is exactly the 45Z credit restated per
    yield — the problem's upper bound, and a gallon -> pound unit check."""
    penalty = import_penalty(0.0)
    assert penalty.lcfs_offset_usd_gal == 0.0
    assert penalty.discount_required_usd_lb == pytest.approx(0.46 / 7.6, rel=1e-12)


# ===========================================================================
# THE RESULT — the answer is bounded
# ===========================================================================
def test_the_lcfs_has_never_traded_high_enough_to_offset_45z():
    """S3's result: the neutralising price sits outside the range the programme has
    ever realised, so no historical level of the credit offsets 45Z at price parity."""
    bounds = penalty_bounds()
    assert not bounds.reaches_neutral
    assert bounds.lcfs_neutral_usd_t > LCFS_PROGRAM_HIGH_USD_T
    assert bounds.lcfs_neutral_usd_t == pytest.approx(285.07, abs=0.01)


def test_the_whole_lcfs_range_moves_the_answer_by_only_a_few_cents():
    """The "both camps have the wrong variable" argument: across the credit's
    entire realised range, the required discount only moves by three-and-change
    cents."""
    bounds = penalty_bounds()
    assert bounds.span_c_lb == pytest.approx(3.18, abs=0.01)
    assert 0.0 < bounds.span_c_lb < 4.0
    assert bounds.discount_at_low_usd_lb > bounds.discount_at_high_usd_lb


def test_bounds_reject_an_inverted_range():
    with pytest.raises(FeedstockError, match="high bound"):
        penalty_bounds(lcfs_low_usd_t=200.0, lcfs_high_usd_t=50.0)


def test_a_richer_ci_gap_lowers_the_neutral_price():
    """Sign check: the cleaner the import is against the domestic pathway, the less
    LCFS is needed to offset 45Z."""
    clean = Feedstock("very clean UCO", 5.0, north_american=False)
    assert lcfs_neutral_price(imported=clean) < lcfs_neutral_price()


def test_a_non_positive_ci_gap_raises_rather_than_returning_a_sign_flip():
    dirty = Feedstock("dirtier UCO", 40.0, north_american=False)
    with pytest.raises(FeedstockError, match="both counts"):
        import_penalty(75.0, imported=dirty)


# ===========================================================================
# The relative weight — and the unit guardrail
# ===========================================================================
def test_discount_burden_rejects_a_series_quoted_in_cents():
    """The module's unit trap, turned into an error: CBOT soyoil is quoted in cents
    per pound, and a forgotten factor of 100 would produce a plausible, wrong
    percentage."""
    cents = pd.Series([45.0, 50.0, 55.0], index=pd.date_range("2024-01-01", periods=3))
    with pytest.raises(FeedstockError, match="cents per pound"):
        discount_burden(cents, lcfs_usd_t=75.0)


def test_discount_burden_is_countercyclical_to_the_oil_price():
    """Same discount in cents, a relative weight that's heavier the cheaper the oil."""
    prices = pd.Series([0.25, 0.50, 0.90], index=pd.date_range("2024-01-01", periods=3))
    burden = discount_burden(prices, lcfs_usd_t=75.0)
    shares = burden.frame["burden_share"].tolist()
    assert shares == sorted(shares, reverse=True)
    assert burden.burden_max / burden.burden_min == pytest.approx(0.90 / 0.25, rel=1e-9)


def test_discount_burden_rejects_an_empty_series():
    with pytest.raises(FeedstockError, match="empty"):
        discount_burden(pd.Series(dtype=float), lcfs_usd_t=75.0)


# ===========================================================================
# The structural exit
# ===========================================================================
def test_structural_exit_is_the_floor_plus_the_required_discount():
    penalty = import_penalty(75.0)
    result = structural_exit(uco_floor_usd_lb=0.35, lcfs_usd_t=75.0)
    assert result.soyoil_critical_usd_lb == pytest.approx(
        0.35 + penalty.discount_required_usd_lb, rel=1e-12
    )
    assert result.share_below is None  # no series supplied


def test_structural_exit_counts_the_crossings():
    prices = pd.Series(
        [0.30, 0.35, 0.45, 0.60], index=pd.date_range("2024-01-01", periods=4)
    )
    result = structural_exit(prices, uco_floor_usd_lb=0.35, lcfs_usd_t=75.0)
    # threshold ~0.3946: the first two prints are below, the last two above
    assert result.share_below == pytest.approx(0.5)
    assert result.n_obs == 4


def test_structural_exit_rejects_a_non_positive_floor():
    with pytest.raises(FeedstockError, match="floor"):
        structural_exit(uco_floor_usd_lb=0.0, lcfs_usd_t=75.0)


# ===========================================================================
# The crush balance
# ===========================================================================
def test_crush_from_soyoil_lb_arithmetic():
    """3.25 Bn lb of oil / 11 lb per bushel / 365 days = ~809,000 bu/day."""
    balance = crush_from_soyoil_lb(3.25e9, installed_capacity_bu_day=6.8e6)
    assert balance.crush_required_bu_day == pytest.approx(809_464.5, rel=1e-4)
    assert not balance.is_short


def test_crush_from_soyoil_lb_rejects_a_non_positive_capacity():
    with pytest.raises(FeedstockError, match="capacity"):
        crush_from_soyoil_lb(1e9, installed_capacity_bu_day=0.0)


# ===========================================================================
# On real data
# ===========================================================================
@needs_bloomberg
def test_loaded_soyoil_is_in_usd_per_pound_not_cents():
    """Regression guard on the unit trap: a median soyoil above 5 would mean the
    cents -> USD conversion wasn't applied."""
    series = load_soyoil_usd_lb("2015")
    assert 0.10 < series.median() < 1.50
    assert series.max() < 2.0


@needs_bloomberg
def test_the_import_economics_work_today_because_oil_is_expensive():
    """S5's reversal, on real data.

    At the same collection floor and the same LCFS price, soyoil spent a large share
    of 2015-2026 below the critical price, and almost none of it since 2024. What
    changed isn't policy — it's the level of vegetable oil.
    """
    long_run = structural_exit(
        load_soyoil_usd_lb("2015"), uco_floor_usd_lb=0.35, lcfs_usd_t=75.0
    )
    recent = structural_exit(
        load_soyoil_usd_lb("2024"), uco_floor_usd_lb=0.35, lcfs_usd_t=75.0
    )
    assert long_run.share_below > 0.40
    assert recent.share_below < 0.05


@needs_bloomberg
def test_the_soyoil_range_dwarfs_the_lcfs_lever():
    """S3's quantified argument: the oil price travels an order of magnitude more
    than the LCFS can move across its entire history."""
    series = load_soyoil_usd_lb("2015")
    oil_range_c_lb = (series.max() - series.min()) * CENTS_PER_USD
    assert oil_range_c_lb > 10 * penalty_bounds().span_c_lb


@needs_bloomberg
def test_the_burden_swings_by_more_than_a_factor_three_on_real_prices():
    burden = discount_burden(load_soyoil_usd_lb("2015"), lcfs_usd_t=75.0)
    assert burden.burden_max / burden.burden_min > 3.0
    assert 0.03 < burden.burden_min < 0.08


@needs_bloomberg
def test_the_program_bounds_are_documented_not_loaded():
    """Posture guardrail: the LCFS programme's bounds are documented constants, not
    export data. If the CARB series is ever added to the loader, this test must fail
    and force a review of the sections that rely on it."""
    from agri.data import bloomberg_loader

    assert "lcfs" not in bloomberg_loader.SERIES_SPECS
    assert LCFS_PROGRAM_LOW_USD_T < LCFS_PROGRAM_HIGH_USD_T < lcfs_neutral_price()
