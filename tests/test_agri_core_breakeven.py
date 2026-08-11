"""Golden tests for the tipping-point solver."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agri.core.breakeven import (
    BreakevenError,
    NoBreakevenInRange,
    solve_breakeven,
)


def test_linear_margin_root_and_sensitivity_hand_computed():
    # margin(theta) = 10 - 20*theta  ->  root at theta = 0.5, slope -20 everywhere
    out = solve_breakeven(lambda t: 10.0 - 20.0 * t, 0.0, 1.0)
    assert out.theta_star == pytest.approx(0.5, abs=1e-9)
    assert out.sensitivity == pytest.approx(-20.0, abs=1e-6)


def test_nonlinear_margin():
    # margin(theta) = 4 - theta^2  ->  root at theta = 2, slope = -2*theta = -4
    out = solve_breakeven(lambda t: 4.0 - t**2, 0.0, 5.0)
    assert out.theta_star == pytest.approx(2.0, abs=1e-8)
    assert out.sensitivity == pytest.approx(-4.0, abs=1e-4)


def test_ballast_share_breakeven_is_the_t1_1_deliverable():
    """T1-1's expected sentence: "beyond 40% of charged ballast, the arb closes."

    arb(ballast) = 8 - 20 * ballast, in USD/t: the arb is worth 8 USD/t with no ballast
    charged and closes when the ballast share reaches 0.40. The sensitivity, -20 USD/t
    per unit of ballast, reads "10 more points of ballast cost 2 USD/t of arb."
    """
    out = solve_breakeven(
        lambda b: 8.0 - 20.0 * b,
        0.0,
        1.0,
        theta_label="ballast_share",
        margin_label="arb",
    )
    assert out.theta_star == pytest.approx(0.40, abs=1e-9)
    assert out.sensitivity / 10.0 == pytest.approx(-2.0, abs=1e-6)
    assert "ballast_share* = 0.4" in out.summary


def test_distance_in_sigmas_hand_computed():
    # root at 0.5, current level 0.3, historical std dev 0.1 -> +2.0 sigmas
    history = pd.Series([0.2, 0.3, 0.4])  # sample std dev = 0.1
    out = solve_breakeven(
        lambda t: 10.0 - 20.0 * t, 0.0, 1.0, theta_current=0.3, theta_history=history
    )
    assert out.theta_sigma == pytest.approx(0.1, abs=1e-9)
    assert out.distance_sigmas == pytest.approx(2.0, abs=1e-6)


def test_a_breakeven_two_sigmas_away_is_flagged_out_of_reach():
    """The central safeguard: don't announce a distant threshold as imminent."""
    history = pd.Series([0.2, 0.3, 0.4])
    out = solve_breakeven(
        lambda t: 10.0 - 20.0 * t, 0.0, 1.0, theta_current=0.3, theta_history=history
    )
    assert not out.is_within_reach
    assert "out of reach" in out.summary


def test_a_close_breakeven_is_flagged_within_reach():
    history = pd.Series([0.1, 0.3, 0.5, 0.7])  # std dev ~0.2582
    out = solve_breakeven(
        lambda t: 10.0 - 20.0 * t, 0.0, 1.0, theta_current=0.45, theta_history=history
    )
    assert out.distance_sigmas == pytest.approx(0.1936, abs=1e-3)
    assert out.is_within_reach
    assert "within reach" in out.summary


def test_distance_is_none_without_history_rather_than_a_number_without_scale():
    out = solve_breakeven(lambda t: 10.0 - 20.0 * t, 0.0, 1.0, theta_current=0.3)
    assert out.distance_sigmas is None
    assert not out.is_within_reach
    assert "standard deviation" not in out.summary


def test_no_sign_change_is_a_result_not_a_crash():
    """"Over the entire plausible range, the arb stays open" is a publishable claim."""
    with pytest.raises(NoBreakevenInRange) as excinfo:
        solve_breakeven(lambda b: 8.0 - 2.0 * b, 0.0, 1.0)
    err = excinfo.value
    assert err.margin_lo == pytest.approx(8.0)
    assert err.margin_hi == pytest.approx(6.0)
    assert "stays positive" in str(err)
    assert "without extrapolating" in str(err)


def test_no_sign_change_reports_a_persistently_negative_margin():
    with pytest.raises(NoBreakevenInRange, match="stays negative"):
        solve_breakeven(lambda b: -8.0 - 2.0 * b, 0.0, 1.0)


def test_root_exactly_on_the_lower_bound():
    out = solve_breakeven(lambda t: t, 0.0, 1.0)
    assert out.theta_star == pytest.approx(0.0)
    assert out.sensitivity == pytest.approx(1.0, abs=1e-6)


def test_root_exactly_on_the_upper_bound():
    out = solve_breakeven(lambda t: t - 1.0, 0.0, 1.0)
    assert out.theta_star == pytest.approx(1.0)


def test_sensitivity_stays_inside_the_bracket_at_a_boundary_root():
    """A negative ballast doesn't exist: the derivative must not step outside the bounds."""
    calls: list[float] = []

    def margin(t: float) -> float:
        calls.append(t)
        return t

    solve_breakeven(margin, 0.0, 1.0)
    assert min(calls) >= 0.0
    assert max(calls) <= 1.0


def test_inverted_bounds_raise():
    with pytest.raises(BreakevenError, match="inconsistent bounds"):
        solve_breakeven(lambda t: t, 1.0, 0.0)


def test_non_finite_margin_raises():
    with pytest.raises(BreakevenError, match="non-finite"):
        solve_breakeven(lambda t: np.inf, 0.0, 1.0)


def test_increasing_margin_lcfs_style():
    """T3-1: plant-gate value rises with the LCFS credit.

    value(LCFS) = -30 + 0.25 * LCFS, in USD/t: the threshold is at 120 USD/t CO2e.
    The solver must handle an increasing margin exactly like a decreasing one.
    """
    out = solve_breakeven(
        lambda lcfs: -30.0 + 0.25 * lcfs,
        0.0,
        400.0,
        theta_current=95.0,
        theta_history=pd.Series([60.0, 80.0, 95.0, 110.0, 130.0]),
        theta_label="LCFS",
    )
    assert out.theta_star == pytest.approx(120.0, abs=1e-8)
    assert out.sensitivity == pytest.approx(0.25, abs=1e-6)
    # mean 95, deviations -35/-15/0/+15/+35, sum of squares 2900,
    # sample variance 2900/4 = 725, std dev sqrt(725) = 26.9258
    # distance = (120 - 95) / 26.9258 = 0.9285
    assert out.theta_sigma == pytest.approx(26.9258, abs=1e-3)
    assert out.distance_sigmas == pytest.approx(0.9285, abs=1e-3)
    assert out.is_within_reach
