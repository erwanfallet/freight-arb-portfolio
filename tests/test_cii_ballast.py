"""Golden tests for project I — the ballast leg the regulator counts like a laden one.

Three tests carry this page and none of them may be weakened by a later rework.

`test_slowing_only_the_ballast_leg_improves_aer_by_31_percent` is the anomaly: a rating
improvement bought with zero additional cargo transported.

`test_loading_more_cargo_cannot_change_aer` is the other half of the same critique,
checked structurally rather than numerically — the function has no cargo-tonnage input
at all, which is the point.

`test_the_rating_gain_costs_real_net_contribution` is what keeps this from reading as a
free lunch: the same slowdown that buys the rating costs real money at real prices.
"""
from __future__ import annotations

import inspect

import pytest

from agri.data.bloomberg_loader import DEFAULT_PATH
from freight.chains.cii_ballast import (
    CARBON_FACTOR_VLSFO,
    CiiError,
    annual_economics,
    attained_aer,
    ballast_share_sweep,
    ballast_speed_sweep,
    market_speed_tradeoff,
    speed_tradeoff,
)


# ===========================================================================
# THE ANOMALY
# ===========================================================================
def test_slowing_only_the_ballast_leg_improves_aer_by_31_percent():
    fast = attained_aer(13.0, ballast_share=1.0)
    slow = attained_aer(8.0, ballast_share=1.0)
    improvement = 1.0 - slow.aer / fast.aer
    assert improvement == pytest.approx(0.31, abs=0.01)


def test_the_laden_leg_is_untouched_by_the_ballast_speed_choice():
    """The cargo-carrying leg's own fuel burn must not move when only the ballast speed
    changes — otherwise the 'zero additional cargo transported' claim would be false."""
    fast = attained_aer(13.0, ballast_share=1.0, laden_speed_kn=12.5)
    slow = attained_aer(8.0, ballast_share=1.0, laden_speed_kn=12.5)
    laden_only_fast = attained_aer(13.0, ballast_share=0.0, laden_speed_kn=12.5)
    laden_only_slow = attained_aer(8.0, ballast_share=0.0, laden_speed_kn=12.5)
    assert laden_only_fast.fuel_t == pytest.approx(laden_only_slow.fuel_t, rel=1e-9)


def test_loading_more_cargo_cannot_change_aer():
    """Structural check: `attained_aer` has no cargo-tonnage parameter. DWT (nameplate
    capacity) is the denominator, not cargo carried — the absence of that parameter IS
    the demonstration that utilization cannot move the rating."""
    params = set(inspect.signature(attained_aer).parameters)
    assert not params & {"cargo_t", "cargo", "payload_t", "utilization"}


def test_more_ballast_distance_charged_to_the_voyage_slightly_improves_aer():
    """The direct 'does more ballast distance reward a better score' check, at fixed
    ballast speed. The effect is real but small at default speeds — the speed lever
    (tested above) is what makes the anomaly large, not the distance lever alone."""
    sweep = ballast_share_sweep()
    assert sweep["aer"].is_monotonic_decreasing
    assert sweep.loc[1.0, "pct_change_vs_no_ballast"] < 0.0
    assert abs(sweep.loc[1.0, "pct_change_vs_no_ballast"]) < 0.01


# ===========================================================================
# NOT A FREE LUNCH
# ===========================================================================
def test_the_rating_gain_costs_real_net_contribution():
    tradeoff = speed_tradeoff(route_rate_usd_t=50.0, vlsfo_usd_t=650.0)
    assert tradeoff.aer_improvement == pytest.approx(0.31, abs=0.01)
    assert tradeoff.net_contribution_cost > 0.0
    assert "AER" in tradeoff.headline
    assert "net contribution" in tradeoff.headline


@pytest.mark.skipif(not DEFAULT_PATH.exists(), reason=f"Bloomberg file absent: {DEFAULT_PATH}")
def test_market_tradeoff_uses_real_prices_not_placeholders():
    """The real-price version must differ from an arbitrarily chosen round-number price
    — otherwise it would not actually be reading the market."""
    market = market_speed_tradeoff()
    assert market.route_rate_usd_t != 50.0
    assert market.vlsfo_usd_t != 650.0
    assert market.route_rate_usd_t > 0
    assert market.vlsfo_usd_t > 0


def test_fewer_trips_per_year_at_slower_ballast_speed():
    fast = annual_economics(13.0, route_rate_usd_t=50.0, vlsfo_usd_t=650.0)
    slow = annual_economics(8.0, route_rate_usd_t=50.0, vlsfo_usd_t=650.0)
    assert slow.trips_per_year < fast.trips_per_year
    assert slow.annual_fuel_cost_usd < fast.annual_fuel_cost_usd


# ===========================================================================
# Method discipline
# ===========================================================================
def test_the_carbon_factor_is_the_stated_imo_reference_value():
    """I-H1 — pinned so a silent edit to this constant cannot pass unnoticed."""
    assert CARBON_FACTOR_VLSFO == pytest.approx(3.114)


def test_sweeps_require_at_least_two_points():
    with pytest.raises(CiiError, match="at least two"):
        ballast_speed_sweep(speeds=(10.0,))
    with pytest.raises(CiiError, match="at least two"):
        ballast_share_sweep(shares=(1.0,))
    with pytest.raises(CiiError, match="at least two"):
        speed_tradeoff(50.0, 650.0, speeds=(10.0,))


# ===========================================================================
# Guardrails
# ===========================================================================
def test_a_non_positive_ballast_speed_is_rejected():
    with pytest.raises(CiiError, match="ballast speed must be > 0"):
        attained_aer(0.0)


def test_a_ballast_share_outside_the_unit_interval_is_rejected():
    with pytest.raises(CiiError, match="ballast_share must be in"):
        attained_aer(10.0, ballast_share=1.5)


def test_a_non_positive_market_price_is_rejected():
    with pytest.raises(CiiError, match="must both be > 0"):
        annual_economics(10.0, route_rate_usd_t=0.0, vlsfo_usd_t=650.0)
