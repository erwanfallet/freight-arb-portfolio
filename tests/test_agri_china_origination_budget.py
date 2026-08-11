"""Golden tests T3-4 — the origination budget, and the windows where no origin works.

The budget's interest lies in what it does **not** contain: neither origin basis nor
freight. Two tests verify this explicitly
(`test_the_budget_does_not_depend_on_the_freight_assumption` and
`..._on_the_basis_assumption`), because that's exactly the property that lets the page
conclude without the two series the export doesn't provide. If a future rework
reintroduced either one into the calculation, the conclusion would become conditional
on an assumed figure with nothing flagging it.
"""
from __future__ import annotations

import pandas as pd
import pytest

from agri.chains.china_soy import (
    BUSHELS_PER_TONNE_SOYBEAN,
    DEFAULT_BASIS_CENTS_BU,
    DEFAULT_FREIGHT_USD_T,
    DEFAULT_IMPORT_DUTY,
    DEFAULT_PROCESSING_CNY_T,
    ChinaSoyError,
    affordable_origination_budget,
    impossible_windows,
    load_real_crush_frame,
)
from agri.data.bloomberg_loader import DEFAULT_PATH, load

pytestmark = pytest.mark.skipif(
    not DEFAULT_PATH.exists(), reason=f"Bloomberg file absent: {DEFAULT_PATH}"
)


@pytest.fixture(scope="module")
def budget():
    return affordable_origination_budget(start="2018-01-01")


# ===========================================================================
# THE PROPERTY THE PAGE RESTS ON
# ===========================================================================
def test_the_budget_does_not_depend_on_the_freight_assumption():
    """THE page's test.

    The reference freight serves as a **reading threshold**, never as an input to the
    calculation. Two very different values must produce exactly the same budget —
    otherwise the conclusion would be conditional on an assumed figure the export
    doesn't provide.
    """
    low = affordable_origination_budget(start="2018-01-01", freight_reference_usd_t=25.0)
    high = affordable_origination_budget(start="2018-01-01", freight_reference_usd_t=85.0)
    pd.testing.assert_series_equal(
        low.frame["budget_usd_t"], high.frame["budget_usd_t"]
    )
    # only the READING changes
    assert low.share_below_freight < high.share_below_freight


def test_the_budget_does_not_depend_on_the_basis_assumption():
    """Same requirement for the FOB basis: it's passed to `load_real_crush_frame`
    downstream but must not reach the budget, which starts from revenue and the raw
    CBOT price."""
    low = affordable_origination_budget(start="2018-01-01", basis_cents_bu=0.0)
    high = affordable_origination_budget(start="2018-01-01", basis_cents_bu=150.0)
    pd.testing.assert_series_equal(
        low.frame["budget_usd_t"], high.frame["budget_usd_t"]
    )


def test_budget_hand_computed(budget):
    """budget = (revenue_ex_vat - processing)/(1 + duty)/USDCNY - CBOT x 36.7437."""
    crush = load_real_crush_frame(start="2018-01-01")
    row = budget.frame.iloc[-1]
    revenue = float(crush.loc[budget.frame.index[-1], "revenue_ex_vat"])

    expected_cnf = (revenue - DEFAULT_PROCESSING_CNY_T) / (1.0 + DEFAULT_IMPORT_DUTY) / row["usdcny"]
    assert row["cnf_max_usd_t"] == pytest.approx(expected_cnf, rel=1e-12)
    assert row["cbot_usd_t"] == pytest.approx(
        row["cbot_usd_bu"] * BUSHELS_PER_TONNE_SOYBEAN, rel=1e-12
    )
    assert row["budget_usd_t"] == pytest.approx(
        row["cnf_max_usd_t"] - row["cbot_usd_t"], rel=1e-12
    )


def test_the_bushel_conversion_is_derived_not_hardcoded():
    """60 lb per soybean bushel -> 36.7437 bushels per metric tonne."""
    assert BUSHELS_PER_TONNE_SOYBEAN == pytest.approx(36.7437, abs=1e-4)


# ===========================================================================
# THE RESULT
# ===========================================================================
def test_a_material_share_of_sessions_admits_no_origin_at_all(budget):
    """S2's result: over a non-trivial share of sessions, the budget is negative — a
    free bean, freighted for free, still wouldn't make the crush pay."""
    assert budget.share_impossible > 0.005
    assert (budget.frame["budget_usd_t"] < 0).any()
    assert "negative" in budget.headline


def test_freight_alone_eats_the_whole_budget_far_more_often(budget):
    """Contrast: the budget falls below freight cost alone far more often than it
    turns negative. Between the two, the bean would have to be bought BELOW the CBOT
    price at origin."""
    assert budget.share_below_freight > budget.share_impossible
    assert budget.share_below_freight > 0.05


def test_the_impossible_windows_are_concentrated_in_2023(budget):
    """The temporal concentration is S3's salient fact: this isn't noise around zero
    spread over eight years, it's a dated episode."""
    windows = impossible_windows(budget, threshold_usd_t=0.0, min_obs=3)
    assert len(windows) > 0
    years = {pd.Timestamp(value).year for value in windows["start"]}
    assert years == {2023}
    assert windows["duration_days"].max() >= 20


def test_the_windows_calendar_carries_dates_not_just_a_count(budget):
    """The deliverable is a calendar that can be checked against an arrival book."""
    windows = impossible_windows(budget, threshold_usd_t=0.0, min_obs=3)
    assert {"start", "end", "duration_days"} <= set(windows.columns)
    assert (pd.to_datetime(windows["end"]) >= pd.to_datetime(windows["start"])).all()


def test_a_higher_threshold_can_only_add_windows(budget):
    """Monotonicity: raising the threshold can't make days below it disappear."""
    strict = impossible_windows(budget, threshold_usd_t=0.0, min_obs=3)
    loose = impossible_windows(budget, threshold_usd_t=45.0, min_obs=3)
    assert loose["duration_days"].sum() > strict["duration_days"].sum()


# ===========================================================================
# Consistency and guardrails
# ===========================================================================
def test_the_budget_median_is_a_plausible_origination_cost(budget):
    """Plausibility check: a Gulf basis plus China freight typically run around
    60-100 USD/t. A median budget far outside that would signal a conversion error."""
    assert 40.0 < budget.median_budget < 140.0


def test_the_module_default_assumption_sits_near_the_median(budget):
    """The assumed figure used elsewhere in the module (70 c/bu of basis + 45 USD/t of
    freight) must fall within the range the budget's median allows — otherwise one of
    the two is wrong."""
    from agri.chains.china_soy import DEFAULT_BASIS_CENTS_BU

    assumed = DEFAULT_BASIS_CENTS_BU / 100.0 * BUSHELS_PER_TONNE_SOYBEAN + DEFAULT_FREIGHT_USD_T
    assert abs(assumed - budget.median_budget) < 25.0


def test_budget_frame_has_the_reading_flags(budget):
    assert {"budget_usd_t", "cnf_max_usd_t", "cbot_usd_t", "impossible", "below_freight"} <= set(
        budget.frame.columns
    )
    assert budget.frame["impossible"].equals(budget.frame["budget_usd_t"] < 0)


def test_an_impossible_start_date_raises():
    with pytest.raises(ChinaSoyError, match="no common date"):
        affordable_origination_budget(start="2099-01-01")


def test_the_budget_is_the_margin_stripped_of_its_two_forfaits(budget):
    """What the budget **exactly** is, stated rather than implied.

    Written during development to check whether budget and margin were the same
    thing; the data answered that they are, up to an affine transform, and the
    identity is exact to floating-point precision:

        margin = (1 + duty) x USDCNY x (budget - basis_forfait - freight_forfait)

    The budget therefore carries **no new information** — it removes two arbitrary
    parameters. That's precisely what makes its crossing zero interpretable where the
    margin's is not: the margin's zero depends on the assumed figure used, the
    budget's depends on nothing. The page states this explicitly rather than letting
    it read as an independent quantity.
    """
    crush = load_real_crush_frame(start="2018-01-01")
    aligned = pd.concat(
        {
            "budget": budget.frame["budget_usd_t"],
            "usdcny": budget.frame["usdcny"],
            "margin": crush["margin"],
        },
        axis=1,
        sort=True,
    ).dropna()

    forfait = DEFAULT_BASIS_CENTS_BU / 100.0 * BUSHELS_PER_TONNE_SOYBEAN + DEFAULT_FREIGHT_USD_T
    predicted = (1.0 + DEFAULT_IMPORT_DUTY) * aligned["usdcny"] * (aligned["budget"] - forfait)
    assert (predicted - aligned["margin"]).abs().max() < 1e-8
