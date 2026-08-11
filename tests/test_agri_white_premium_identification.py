"""Golden tests T2-4 — what a price can identify, and what it can't.

The central test is `test_the_level_is_not_identifiable_but_the_variation_is`. That's
the page's thesis, and it's unusual in nature: it asserts both a limit (the rent's
level can't be inferred from prices) and a result (its variation can). Both halves must
be tested, otherwise the page could drift toward the comfortable conclusion — publish a
level — without anything flagging it.
"""
from __future__ import annotations

import pandas as pd
import pytest

from agri.chains.white_premium import (
    POL_PLAUSIBLE_HI,
    POL_PLAUSIBLE_LO,
    WhitePremiumError,
    identification_check,
    implied_pol_adjust,
    implied_refining_cost,
    load_real_richness_frame,
)
from agri.data.bloomberg_loader import DEFAULT_PATH

pytestmark = pytest.mark.skipif(
    not DEFAULT_PATH.exists(), reason=f"Bloomberg file absent: {DEFAULT_PATH}"
)

START = "2015-01-01"


@pytest.fixture(scope="module")
def check():
    return identification_check(start=START)


# ===========================================================================
# THE THESIS
# ===========================================================================
def test_the_level_is_not_identifiable_but_the_variation_is(check):
    """THE page's test, in its two halves.

    Half 1 — the level is not identifiable: the uncertainty a single unobservable
    parameter injects is of the same order as the median richness itself.
    Half 2 — the variation is: the gap between the best and worst year exceeds that
    uncertainty by a factor that leaves no room for doubt.
    """
    median_richness = abs(check.annual["richness_ref"].median())
    assert check.parameter_span_max > median_richness, (
        "the parameter uncertainty has become small next to the median richness: "
        "the page's S3 section claims the opposite and needs a rewrite"
    )
    assert check.ratio > 3.0
    assert check.signal_span > 40.0


def test_the_year_ranking_survives_the_parameter_entirely(check):
    """What licenses reading the gaps between years: the parameter shifts them all in
    the same direction, so it reorders nothing."""
    assert check.rank_correlation == pytest.approx(1.0, abs=1e-9)


def test_only_years_near_zero_can_flip_sign(check):
    """The only non-interpretable years are the ones whose richness is already close
    to zero. If a clearly positive or negative year flipped sign, the page's
    conclusion would no longer hold."""
    for year in check.sign_flipping_years:
        assert abs(check.annual.loc[year, "richness_ref"]) < check.parameter_span_max


def test_the_regime_shift_is_larger_than_the_parameter_can_explain(check):
    """S5's flip: richness is mostly negative through 2021 and clearly positive from
    2023 on, and the gap between the two periods exceeds what the choice of
    pol_adjust can produce."""
    reference = check.annual["richness_ref"]
    before = reference.loc[2017:2021].median()
    after = reference.loc[2023:].median()
    assert before < 0 < after
    assert after - before > 2 * check.parameter_span_max


# ===========================================================================
# The inversion
# ===========================================================================
def test_implied_pol_zeroes_the_median_richness():
    """Solver consistency check: at the pol* it returns, the median richness must be
    zero up to tolerance."""
    implied = implied_pol_adjust(start=START)
    frame = load_real_richness_frame(pol_adjust=implied.pol_star, start=START)
    assert frame["richness"].median() == pytest.approx(0.0, abs=0.05)


def test_implied_pol_sits_just_above_the_plausible_band():
    """S3's result: the adjustment that would zero the rent sits outside the
    plausible range, but only just — which is precisely what keeps both readings open."""
    implied = implied_pol_adjust(start=START)
    assert not implied.within_plausible
    assert POL_PLAUSIBLE_HI < implied.pol_star < POL_PLAUSIBLE_HI + 0.02
    assert "cannot settle it" in implied.headline


def test_implied_pol_stays_inside_the_guarded_range():
    """The solver must never step outside the range `white_premium_usd_t` accepts —
    otherwise it raises instead of returning a result (hit during development)."""
    implied = implied_pol_adjust(start=START)
    assert 1.00 <= implied.pol_star <= 1.20


def test_a_range_with_no_root_raises_rather_than_returning_a_bound():
    """Over a range where the richness keeps the same sign at both ends, there is no
    pol* — and the engine says so instead of returning a bound as if it were a
    solution. Beyond 1.10 the median richness is negative everywhere: no polarisation
    adjustment zeroes it in this window."""
    with pytest.raises(WhitePremiumError, match="no pol_adjust"):
        implied_pol_adjust(start=START, search_lo=1.10, search_hi=1.1999)


# ===========================================================================
# The number for the email
# ===========================================================================
def test_implied_refining_cost_is_the_white_premium_itself():
    """The price paid for refining is the white premium, with no cost assumption.
    That's what makes it the presentable number: it's observed, not modelled."""
    cost = implied_refining_cost(start=START)
    frame = load_real_richness_frame(start=START)
    assert cost.market_usd_t == pytest.approx(frame["white_premium"].median())
    assert cost.gap_usd_t == pytest.approx(cost.market_usd_t - cost.modelled_usd_t)


def test_the_market_pays_a_plausible_order_of_magnitude_for_refining():
    cost = implied_refining_cost(start=START)
    assert 40.0 < cost.market_usd_t < 120.0


# ===========================================================================
# Guardrails
# ===========================================================================
def test_identification_check_rejects_a_reference_outside_the_bounds():
    with pytest.raises(WhitePremiumError, match="strictly between"):
        identification_check(start=START, pol_ref=1.15)


def test_identification_check_rejects_a_window_too_short_to_compare_years():
    with pytest.raises(WhitePremiumError, match="year"):
        identification_check(start="2025-06-01")


def test_the_parameter_moves_every_year_in_the_same_direction(check):
    """The property the whole S4 section rests on: pol_adjust is a multiplier on the
    raw price, so it shifts every year in the same direction. If it didn't, the gaps
    between years wouldn't survive."""
    differences = check.annual["richness_lo"] - check.annual["richness_hi"]
    assert (differences > 0).all()


def test_a_higher_pol_adjust_always_lowers_the_richness():
    """Sign check: the higher the adjustment, the more expensive the converted raw
    sugar, so the lower the richness."""
    medians = [
        load_real_richness_frame(pol_adjust=pol, start=START)["richness"].median()
        for pol in (1.02, 1.06, 1.10, 1.14)
    ]
    assert medians == sorted(medians, reverse=True)


def test_the_plausible_band_brackets_the_default():
    assert POL_PLAUSIBLE_LO < 1.07 < POL_PLAUSIBLE_HI
