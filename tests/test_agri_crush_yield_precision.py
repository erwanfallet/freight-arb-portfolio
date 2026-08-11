"""Golden tests T2-3 — the board crush's implied yield precision, on real CBOT data.

The central test is `test_the_precision_demanded_collapses_when_the_margin_tightens`:
that's the page's result, and it isn't about an average level but about **regime
dependence**. A test that only checked the median would let through a rework that
breaks exactly what makes the page interesting.

`test_the_exposure_is_largest_when_the_margin_is_widest` guards S4's counter-intuitive
result: my initial intuition ("the position grows as the margin tightens, so the two
problems compound") was **wrong**, and the page says so. If a future rework restored
the comfortable intuition, this test would fail.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agri.chains.crush_tracking import (
    CBOT_MEAL_LB_BU,
    CBOT_OIL_LB_BU,
    CrushError,
    hedge_ratio_identity_bias,
    load_real_board_frame,
    required_yield_precision,
    yield_exposure,
)
from agri.data.bloomberg_loader import DEFAULT_PATH

pytestmark = pytest.mark.skipif(
    not DEFAULT_PATH.exists(), reason=f"Bloomberg file absent: {DEFAULT_PATH}"
)


@pytest.fixture(scope="module")
def frame():
    return load_real_board_frame("2015-01-01")


# ===========================================================================
# The board crush itself
# ===========================================================================
def test_board_crush_on_2026_08_07_hand_computed(frame):
    """0.022 x 308.10 + 0.11 x 68.160 - 11.5650 = 6.7782 + 7.4976 - 11.5650 = 2.7108."""
    row = frame.loc[pd.Timestamp("2026-08-07")]
    assert row["meal"] == pytest.approx(308.10)
    assert row["oil"] == pytest.approx(68.160)
    assert row["bean"] == pytest.approx(11.5650)
    assert row["board"] == pytest.approx(2.7108, abs=1e-6)


def test_the_board_crush_never_goes_negative(frame):
    """Same signature as in T2-5: the board carries no opex, so it never dips below
    zero. That's a property of the contract, not of a plant's economics — and it's
    what justifies why the whole page reasons in NET margin."""
    assert frame["board"].min() > 0
    assert (frame["board"] < 0).sum() == 0


def test_a_plausible_opex_pushes_the_margin_below_zero_sometimes(frame):
    """Contrast with the previous test: opex is what turns an always-positive board
    into a margin that genuinely goes underwater."""
    assert ((frame["board"] - 0.70) < 0).mean() > 0.05


# ===========================================================================
# THE DELIVERABLE — the required precision
# ===========================================================================
def test_required_precision_hand_computed(frame):
    """On 07/08/2026, opex 0.55: (2.7108 - 0.55) / (308.10 / 2000) = 2.1608 / 0.15405."""
    precision = required_yield_precision(frame, opex_usd_bu=0.55)
    row = precision.frame.loc[pd.Timestamp("2026-08-07")]
    assert row["net_margin"] == pytest.approx(2.1608, abs=1e-6)
    assert row["position_per_lb"] == pytest.approx(0.15405, abs=1e-9)
    assert row["breakeven_lb"] == pytest.approx(2.1608 / 0.15405, rel=1e-12)


def test_the_precision_demanded_collapses_when_the_margin_tightens(frame):
    """THE page's test.

    The precision demanded isn't a level, it's a function of the regime. Between the
    widest and tightest margin decile, it's divided by more than ten — and in the
    tight decile it falls below 1 pound, i.e. less than 2.5% of the yield the
    contract assumes.
    """
    precision = required_yield_precision(frame, opex_usd_bu=0.55)
    assert precision.tight_decile_lb < 1.0
    assert precision.wide_decile_lb > 10.0
    assert precision.wide_decile_lb / precision.tight_decile_lb > 10.0
    assert precision.tight_decile_pct < 0.025


def test_one_pound_wipes_the_margin_on_a_material_share_of_days(frame):
    """The number for the email: a one-pound gap — 2.3% of the contract's yield —
    wipes out the entire net margin on a non-trivial share of the sample."""
    precision = required_yield_precision(frame, opex_usd_bu=0.55)
    assert precision.share_below(1.0) > 0.05
    assert precision.share_below(2.0) > precision.share_below(1.0)
    assert precision.share_below(20.0) > 0.90


def test_precision_is_monotone_in_opex(frame):
    """The heavier the opex, the thinner the net margin, so the harder the requirement."""
    medians = [
        required_yield_precision(frame, opex_usd_bu=value).median_lb
        for value in (0.30, 0.50, 0.70, 0.90)
    ]
    assert medians == sorted(medians, reverse=True)


def test_required_precision_rejects_a_frame_without_the_board(frame):
    with pytest.raises(CrushError, match="board"):
        required_yield_precision(frame[["bean", "meal"]], opex_usd_bu=0.55)


# ===========================================================================
# The position
# ===========================================================================
def test_yield_exposure_hand_computed(frame):
    """1 lb/bu on 07/08/2026: 308.10 / 2000 = 0.15405 USD/bu."""
    exposure = yield_exposure(frame, meal_lb_gap=1.0, oil_lb_gap=0.0, opex_usd_bu=0.55)
    row = exposure.frame.loc[pd.Timestamp("2026-08-07")]
    assert row["meal_leg"] == pytest.approx(0.15405, abs=1e-9)
    assert row["oil_leg"] == 0.0
    assert row["position_usd_bu"] == pytest.approx(0.15405, abs=1e-9)


def test_the_oil_leg_uses_the_cents_to_dollar_conversion(frame):
    """Unit trap: oil is quoted in cents per pound. 1 lb/bu on 07/08/2026 is worth
    68.160 / 100 = 0.6816 USD/bu, not 68.16."""
    exposure = yield_exposure(frame, meal_lb_gap=0.0, oil_lb_gap=1.0, opex_usd_bu=0.55)
    assert exposure.frame.loc[pd.Timestamp("2026-08-07"), "oil_leg"] == pytest.approx(0.6816)


def test_exposure_is_linear_in_the_gap(frame):
    """This is what licenses calling it a "position": the value is a quantity times a
    price, hence strictly proportional to the quantity."""
    one = yield_exposure(frame, meal_lb_gap=1.0, opex_usd_bu=0.55).position_median
    three = yield_exposure(frame, meal_lb_gap=3.0, opex_usd_bu=0.55).position_median
    assert three == pytest.approx(3.0 * one, rel=1e-12)


def test_a_negative_gap_flips_the_position(frame):
    positive = yield_exposure(frame, meal_lb_gap=1.0, opex_usd_bu=0.55).position_median
    negative = yield_exposure(frame, meal_lb_gap=-1.0, opex_usd_bu=0.55).position_median
    assert negative == pytest.approx(-positive, rel=1e-12)


def test_an_implausible_gap_is_rejected(frame):
    with pytest.raises(CrushError, match="implausible"):
        yield_exposure(frame, meal_lb_gap=60.0, opex_usd_bu=0.55)


def test_exposure_rejects_a_frame_missing_a_leg(frame):
    with pytest.raises(CrushError, match="missing column"):
        yield_exposure(frame[["bean", "board"]], meal_lb_gap=1.0)


# ===========================================================================
# THE COUNTER-INTUITIVE RESULT — a guard against comfort
# ===========================================================================
def test_the_exposure_is_largest_when_the_margin_is_widest(frame):
    """My starting intuition was that the naked position grows as the margin
    tightens, which would compound the two problems. The data says the opposite:
    meal is the crush's main source of revenue, so its price is POSITIVELY
    correlated with the net margin.

    The page states this result instead of hiding it. This test prevents a later
    rework from silently restoring the more comfortable story.
    """
    net_margin = frame["board"] - 0.55
    correlation = net_margin.corr(frame["meal"], method="spearman")
    assert correlation > 0.15, (
        "the margin/meal correlation has turned negative or zero: the page's S4 "
        "section claims the opposite and needs a rewrite"
    )

    tight = frame.loc[net_margin <= net_margin.quantile(0.10), "meal"].median()
    wide = frame.loc[net_margin >= net_margin.quantile(0.90), "meal"].median()
    assert wide > tight


def test_what_degrades_in_the_tight_regime_is_the_ratio_not_the_position(frame):
    """The exact formulation the page uses: in a tight regime the position is small
    in dollars, but the margin is smaller still. A plant watching its exposure in
    absolute dollars sees nothing coming."""
    net_margin = frame["board"] - 0.55
    exposure = yield_exposure(frame, meal_lb_gap=1.0, opex_usd_bu=0.55)
    position = exposure.frame["position_usd_bu"]

    tight = net_margin <= net_margin.quantile(0.10)
    wide = net_margin >= net_margin.quantile(0.90)

    assert position[tight].median() < position[wide].median()          # smaller position
    assert (position / net_margin.clip(lower=0.05))[tight].median() > (
        position / net_margin.clip(lower=0.05)
    )[wide].median()                                                   # ratio much worse


# ===========================================================================
# The accounting-identity trap, inherited from T2-1
# ===========================================================================
def test_the_identity_bias_is_positive_and_scales_with_the_gap(frame):
    betas = [
        hedge_ratio_identity_bias(frame, meal_lb_gap=gap, opex_usd_bu=0.55).beta_naive
        for gap in (0.5, 1.0, 2.0, 4.0)
    ]
    assert all(b > 1.0 for b in betas)
    assert betas == sorted(betas)
    # linearity: the bias doubles when the gap doubles
    biases = [b - 1.0 for b in betas]
    assert biases[2] == pytest.approx(2 * biases[1], rel=1e-6)


def test_the_identity_bias_is_small_and_the_page_says_so(frame):
    """Honesty guardrail. The page explicitly states this contamination is on the
    order of a percent and that it is NOT the argument. If the bias grew large,
    S5's wording would need revisiting — this test would flag it."""
    bias = hedge_ratio_identity_bias(frame, meal_lb_gap=1.0, opex_usd_bu=0.55)
    assert 0.0 < bias.bias < 0.05
    assert "only" in bias.headline
    assert "yield" in bias.headline


def test_a_zero_gap_leaves_no_bias_at_all(frame):
    """Check: with no yield gap, the plant margin IS the board minus a constant, so
    the regression returns exactly 1."""
    bias = hedge_ratio_identity_bias(frame, meal_lb_gap=0.0, oil_lb_gap=0.0, opex_usd_bu=0.55)
    assert bias.beta_naive == pytest.approx(1.0, abs=1e-12)
    assert bias.bias == pytest.approx(0.0, abs=1e-12)


def test_identity_bias_refuses_a_short_sample(frame):
    with pytest.raises(CrushError, match="too short"):
        hedge_ratio_identity_bias(frame.head(10), meal_lb_gap=1.0)


# ===========================================================================
# Loading
# ===========================================================================
def test_load_real_board_frame_has_the_three_legs_and_the_board(frame):
    assert list(frame.columns) == ["bean", "meal", "oil", "board"]
    assert len(frame) > 2_000
    assert frame.index.is_monotonic_increasing
    assert not frame.isna().any().any()


def test_the_board_coefficients_are_the_cbot_yields():
    """The page's core subject: 0.022 and 0.11 are not unit conversions but yields.
    This test states it in code."""
    assert CBOT_MEAL_LB_BU / 2000.0 == pytest.approx(0.022)
    assert CBOT_OIL_LB_BU / 100.0 == pytest.approx(0.11)


def test_an_impossible_start_date_raises():
    with pytest.raises(CrushError, match="no common date"):
        load_real_board_frame("2099-01-01")
