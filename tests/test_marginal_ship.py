"""Golden tests for project G — the fuel-only breakeven multiplier on the P8 route.

Three tests carry this page and none of them may be weakened by a later rework.

`test_the_cost_floor_is_exactly_independent_of_bunker_price` is the algebraic fact the
whole decomposition depends on — if it stops being an identity, the variance split stops
being exact and becomes a claim that needs a residual, which the page does not have.

`test_the_margin_never_approached_one` is the result.

`test_the_decomposition_shares_sum_to_one` is the honesty check on the decomposition
itself: shares that do not sum to one would mean the identity used to derive them is
wrong, silently.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agri.core.voyage import VoyageParams
from agri.data.bloomberg_loader import DEFAULT_PATH
from freight.chains.marginal_ship import (
    MarginalShipError,
    breakeven_multiplier,
    breakeven_series,
    cost_floor_usd_t,
    load_marginal_ship_frame,
    margin_summary,
    variance_decomposition,
)

pytestmark = pytest.mark.skipif(
    not DEFAULT_PATH.exists(), reason=f"Bloomberg file absent: {DEFAULT_PATH}"
)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return load_marginal_ship_frame()


# ===========================================================================
# The algebraic fact the decomposition depends on
# ===========================================================================
def test_the_cost_floor_is_exactly_independent_of_bunker_price():
    """A must not move with vlsfo — a zero-consumption vessel pays no bunker bill no
    matter what bunkers cost. If this ever fails, the decomposition below is no longer
    an identity and must not be reported as one."""
    low = cost_floor_usd_t(ballast_share=1.0)
    high_params = VoyageParams(ballast_share=1.0)
    from dataclasses import replace as _replace

    # cost_floor_usd_t always evaluates at vlsfo=mgo=1.0 internally; the direct check is
    # that voyage_freight_usd_t at k=0 does not move across a wide bunker range.
    from agri.core.voyage import ROUTES, VESSELS, voyage_freight_usd_t

    vessel = VESSELS["panamax"]
    zero_vessel = _replace(
        vessel, consumption_laden_t_day=0.0, consumption_ballast_t_day=0.0, consumption_port_t_day=0.0
    )
    a_cheap = voyage_freight_usd_t(
        0.0, 50.0, 70.0, vessel=zero_vessel, route=ROUTES["santos_qingdao"], params=high_params
    ).freight_usd_t
    a_expensive = voyage_freight_usd_t(
        0.0, 2000.0, 2700.0, vessel=zero_vessel, route=ROUTES["santos_qingdao"], params=high_params
    ).freight_usd_t
    assert a_cheap == pytest.approx(a_expensive, rel=1e-9)
    assert a_cheap == pytest.approx(low, rel=1e-9)


def test_multiplier_scales_correctly_at_two_reference_points():
    """Sanity check on the closed-form solve: at k=1 the breakeven condition should
    reproduce the vessel's own true breakeven freight rate."""
    from agri.core.voyage import ROUTES, VESSELS, voyage_freight_usd_t

    vessel = VESSELS["panamax"]
    true_breakeven_rate = voyage_freight_usd_t(
        0.0, 500.0, 675.0, vessel=vessel, route=ROUTES["santos_qingdao"],
        params=VoyageParams(ballast_share=1.0),
    ).freight_usd_t
    k = breakeven_multiplier(true_breakeven_rate, 500.0, 675.0, ballast_share=1.0)
    assert k == pytest.approx(1.0, abs=1e-6)


# ===========================================================================
# THE RESULT
# ===========================================================================
def test_the_margin_never_approached_one(frame):
    """The page's central claim: even at its tightest point in nearly five years, the
    fuel-only breakeven margin left comfortable room above the reference vessel."""
    summary = margin_summary(frame, ballast_share=1.0)
    assert summary.min > 1.4
    assert summary.max < 4.0
    assert summary.never_approached_one
    assert "well above" in summary.headline


def test_the_tightest_point_is_the_2026_bunker_spike(frame):
    """Ties the result to a dated, checkable market event rather than an unlabelled
    minimum — the tightest point should fall inside the VLSFO spike documented in
    project F, not at some unrelated date."""
    summary = margin_summary(frame, ballast_share=1.0)
    assert summary.min_date >= pd.Timestamp("2026-01-01")


def test_the_zero_ballast_convention_is_always_looser(frame):
    """Charging no ballast to this voyage lowers the assumed cost, which must raise the
    breakeven multiplier at every date — a sign flip here would mean the ballast
    parameter is wired backwards."""
    full = breakeven_series(frame, ballast_share=1.0)
    zero = breakeven_series(frame, ballast_share=0.0)
    assert (zero > full).all()


# ===========================================================================
# THE DECOMPOSITION
# ===========================================================================
def test_the_decomposition_shares_sum_to_one(frame):
    result = variance_decomposition(frame, ballast_share=1.0)
    assert result.share_rate + result.share_bunker == pytest.approx(1.0, abs=1e-9)
    assert 0.0 <= result.share_rate <= 1.0
    assert 0.0 <= result.share_bunker <= 1.0


def test_the_bunker_price_dominates_the_variance(frame):
    """The page's second claim, distinct from the level result: what actually moves the
    margin day to day is more the bunker price than the freight rate."""
    result = variance_decomposition(frame, ballast_share=1.0)
    assert result.share_bunker > result.share_rate
    assert result.share_bunker == pytest.approx(0.62, abs=0.03)
    assert "bunker price" in result.headline


def test_the_log_change_identity_holds_exactly(frame):
    """d(log k*) must equal d(log(rate-A)) - d(log vlsfo) exactly, with no residual —
    this is what makes the variance split an identity rather than a fitted decomposition."""
    a = cost_floor_usd_t(ballast_share=1.0)
    k = breakeven_series(frame, ballast_share=1.0)
    d_log_k = np.log(k).diff().dropna()
    d_log_margin = np.log(frame["rate"] - a).diff().dropna()
    d_log_vlsfo = np.log(frame["vlsfo"]).diff().dropna()
    reconstructed = d_log_margin - d_log_vlsfo
    assert np.allclose(d_log_k.to_numpy(), reconstructed.to_numpy(), atol=1e-9)


def test_a_degenerate_sample_below_the_cost_floor_is_rejected():
    bad = pd.DataFrame(
        {"rate": [0.5, 0.6], "vlsfo": [500.0, 510.0], "mgo": [675.0, 688.5]},
        index=pd.date_range("2020-01-01", periods=2, freq="D"),
    )
    with pytest.raises(MarginalShipError, match="falls below the fixed cost floor"):
        variance_decomposition(bad, ballast_share=1.0)


# ===========================================================================
# Guardrails
# ===========================================================================
def test_a_non_positive_bunker_price_is_rejected():
    with pytest.raises(MarginalShipError, match="bunker prices must be > 0"):
        breakeven_multiplier(50.0, 0.0, 10.0, ballast_share=1.0)


def test_an_impossible_start_date_raises():
    with pytest.raises(MarginalShipError, match="no common dates"):
        load_marginal_ship_frame("2099-01-01")
