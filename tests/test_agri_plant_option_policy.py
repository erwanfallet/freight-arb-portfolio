"""Golden tests for the T2-5 policy simulator and its inversion.

The reference margin path, used for every hand-computed calculation:

    margin = [10, -5, -5, -5, -5, 10, 10, 10, 10]      (9 periods)
    costs  : restart 100, shutdown 50, idle 1/period

It's built so the N=4 persistence rule fires exactly once, at the 5th point: that's
the first date where the four preceding values (inclusive) are all negative.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agri.chains.china_soy import load_real_crush_frame
from agri.chains.plant_option import (
    HysteresisBand,
    PlantOptionError,
    calibrate_ou,
    compare_policies,
    implied_switching_cost,
    run_always_on_policy,
    run_band_policy,
    run_heuristic_policy,
    solve_hysteresis,
    switching_cost_sensitivity,
)
from agri.data.bloomberg_loader import DEFAULT_PATH

PATH_VALUES = [10.0, -5.0, -5.0, -5.0, -5.0, 10.0, 10.0, 10.0, 10.0]
COSTS = dict(cost_restart=100.0, cost_shutdown=50.0, cost_idle=1.0)


@pytest.fixture
def path() -> pd.Series:
    return pd.Series(PATH_VALUES, index=pd.date_range("2024-01-31", periods=9, freq="ME"))


# ===========================================================================
# The simulator, hand-computed
# ===========================================================================
def test_heuristic_policy_hand_computed(path):
    """Rule "margin < 0 for 4 periods," symmetric on restart.

    Shutdown at the 5th point (index 4): the first window of 4 all-negative values.
    Restart at the 9th (index 8): the first window of 4 all-positive values.

        operating = 10 - 5 - 5 - 5 - 5           = -10   (5 periods running)
        switch    = 50 (shutdown) + 100 (restart) = 150
        idle      = 4 periods x 1                 =   4
        total     = -10 - 150 - 4                 = -164
    """
    result = run_heuristic_policy(path, threshold=0.0, n_periods=4, **COSTS)
    assert result.n_stops == 1
    assert result.n_starts == 1
    assert result.periods_on == 5
    assert result.periods_off == 4
    assert result.operating_pnl == pytest.approx(-10.0)
    assert result.switching_cost == pytest.approx(150.0)
    assert result.idle_cost == pytest.approx(4.0)
    assert result.total_pnl == pytest.approx(-164.0)


def test_persistence_rule_does_not_fire_before_n_periods(path):
    """The 4th point is negative but the window still contains the initial +10: no shutdown.

    This is exactly what persistence buys — and it's also why the rule stops at a
    margin level lower than its displayed threshold.
    """
    result = run_heuristic_policy(path, threshold=0.0, n_periods=4, **COSTS)
    assert result.stop_margins == [-5.0]
    # the shutdown happens at the 5th period, not the 4th
    assert bool(result.state.iloc[3]) is True
    assert bool(result.state.iloc[4]) is False


def test_band_policy_hand_computed(path):
    """Band [-3, +5]: instant shutdown at the first point below -3, restart at the
    first point above +5.

        operating = 10 - 5 + 10 + 10 + 10 = 35
        switch    = 50 + 100              = 150
        idle      = 4 x 1                 =   4
        total     = 35 - 150 - 4          = -119
    """
    band = HysteresisBand(
        m_off=-3.0, m_on=5.0, grid=np.linspace(-20, 20, 5),
        value_on=np.zeros(5), value_off=np.zeros(5), n_iterations=1, converged=True,
    )
    result = run_band_policy(path, band, **COSTS)
    assert result.n_stops == 1
    assert result.periods_on == 5
    assert result.operating_pnl == pytest.approx(35.0)
    assert result.total_pnl == pytest.approx(-119.0)


def test_always_on_is_the_counterfactual(path):
    """Counterfactual: the path's raw sum, no switching or idle cost at all.

        10 + 4 x (-5) + 4 x 10 = 10 - 20 + 40 = +30

    It beats both rules on this path — a reminder that stopping isn't costlessly
    good, and that the comparison must always include this case.
    """
    result = run_always_on_policy(path, **COSTS)
    assert result.n_stops == 0
    assert result.periods_off == 0
    assert result.total_pnl == pytest.approx(30.0)
    assert result.total_pnl == pytest.approx(sum(PATH_VALUES))


# ===========================================================================
# The degenerate band — refuse rather than substitute
# ===========================================================================
def test_degenerate_band_is_detected():
    band = HysteresisBand(
        m_off=10.0, m_on=-10.0, grid=np.linspace(-20, 20, 5),
        value_on=np.zeros(5), value_off=np.zeros(5), n_iterations=1, converged=True,
    )
    assert band.is_degenerate
    assert "never stop" in band.headline


def test_degenerate_band_refuses_to_run_rather_than_substituting(path):
    """The discipline point: applying M_on < M_off would make the plant oscillate
    every period, and substituting a fallback policy would amount to comparing a rule
    the model never produced."""
    band = HysteresisBand(
        m_off=10.0, m_on=-10.0, grid=np.linspace(-20, 20, 5),
        value_on=np.zeros(5), value_off=np.zeros(5), n_iterations=1, converged=True,
    )
    with pytest.raises(PlantOptionError, match="degenerate"):
        run_band_policy(path, band, **COSTS)


def test_comparison_continues_without_the_band_when_degenerate(path):
    """The comparison doesn't stop: threshold rule against counterfactual stays
    informative, and the headline says why the band is missing."""
    band = HysteresisBand(
        m_off=10.0, m_on=-10.0, grid=np.linspace(-20, 20, 5),
        value_on=np.zeros(5), value_off=np.zeros(5), n_iterations=1, converged=True,
    )
    comparison = compare_policies(path, band, **COSTS)
    assert not comparison.band_is_available
    assert np.isnan(comparison.gap_vs_band)
    assert "No exercise boundary" in comparison.headline
    # the counterfactual is still computable: -164 (heuristic) - 30 (never stop)
    assert comparison.heuristic_flexibility_value == pytest.approx(-194.0)
    assert len(comparison.to_frame()) == 2


# ===========================================================================
# The inversion, on the real Chinese crush margin
# ===========================================================================
@pytest.mark.skipif(not DEFAULT_PATH.exists(), reason=f"Bloomberg file absent: {DEFAULT_PATH}")
def test_chinese_crush_margin_is_the_right_anchor():
    """Why this page anchors on the Chinese margin rather than the US board crush: it
    genuinely goes below zero (41% of the time, down to -865 CNY/t) and it's
    stationary, so the OU calibration is legitimate there and the curtailment
    question is alive there."""
    margin = load_real_crush_frame()["margin"]
    assert (margin < 0).mean() > 0.35
    assert margin.min() < -500
    assert calibrate_ou(margin, strict=False).stationarity.verdict == "stationary"


@pytest.mark.skipif(not DEFAULT_PATH.exists(), reason=f"Bloomberg file absent: {DEFAULT_PATH}")
def test_implied_switching_cost_is_the_deliverable():
    """THE number for the email: the N=4 rule stops and restarts at precise levels,
    so it assumes a precise round-trip cost — here ~143 CNY/t of bean crushed."""
    margin = load_real_crush_frame()["margin"]
    ou = calibrate_ou(margin, strict=False)
    implied = implied_switching_cost(margin, ou, cost_idle=2.0)
    assert implied.converged
    assert implied.n_stops_observed > 10
    assert implied.effective_m_off < 0 < implied.effective_m_on
    assert 100.0 < implied.implied_switching_cost < 200.0
    assert "assumes without saying so" in implied.headline


@pytest.mark.skipif(not DEFAULT_PATH.exists(), reason=f"Bloomberg file absent: {DEFAULT_PATH}")
def test_rule_stops_below_its_own_threshold():
    """Persistence shifts the stopping point: the rule displays a threshold of 0 but
    actually stops well below it. It's this gap that makes it equivalent to a band,
    and therefore translatable into a switching cost."""
    margin = load_real_crush_frame()["margin"]
    ou = calibrate_ou(margin, strict=False)
    implied = implied_switching_cost(margin, ou, cost_idle=2.0)
    assert implied.effective_m_off < -10.0


@pytest.mark.skipif(not DEFAULT_PATH.exists(), reason=f"Bloomberg file absent: {DEFAULT_PATH}")
def test_sensitivity_marks_degenerate_rows_instead_of_negative_widths():
    """Regression test for a defect found while building the page: at a low
    switching cost the band inverts, and a negative width used to read as a narrow
    band."""
    margin = load_real_crush_frame()["margin"]
    ou = calibrate_ou(margin, strict=False)
    table = switching_cost_sensitivity(
        margin, ou, cost_grid=np.array([5.0, 30.0, 143.0, 600.0]), cost_idle=2.0
    )
    assert table["degenerate"].any()
    widths = table.loc[~table["degenerate"], "band_width"]
    assert (widths > 0).all()
    assert table.loc[table["degenerate"], "band_width"].isna().all()


@pytest.mark.skipif(not DEFAULT_PATH.exists(), reason=f"Bloomberg file absent: {DEFAULT_PATH}")
def test_band_width_grows_with_switching_cost():
    """Expected monotonicity — it's what licenses interpolating the inversion."""
    margin = load_real_crush_frame()["margin"]
    ou = calibrate_ou(margin, strict=False)
    table = switching_cost_sensitivity(
        margin, ou, cost_grid=np.array([30.0, 60.0, 143.0, 300.0, 600.0]), cost_idle=2.0
    )
    valid = table[~table["degenerate"]]
    assert valid["band_width"].is_monotonic_increasing


@pytest.mark.skipif(not DEFAULT_PATH.exists(), reason=f"Bloomberg file absent: {DEFAULT_PATH}")
def test_flexibility_is_worth_far_more_than_the_rule_choice():
    """Hierarchy of stakes, to be stated in this order: being able to stop is worth a
    lot (~125,000 CNY/t cumulative against never stopping), picking the right
    stopping rule is worth noticeably less (~3,000). Reversing this order in the
    presentation would make a refinement look like the main subject."""
    margin = load_real_crush_frame()["margin"]
    ou = calibrate_ou(margin, strict=False)
    band = solve_hysteresis(ou, cost_restart=96.0, cost_shutdown=47.0, cost_idle=2.0)
    comparison = compare_policies(margin, band, cost_restart=96.0, cost_shutdown=47.0, cost_idle=2.0)
    assert comparison.band_is_available
    assert comparison.flexibility_value > 50_000
    assert 0 < comparison.gap_vs_band < comparison.flexibility_value / 10
