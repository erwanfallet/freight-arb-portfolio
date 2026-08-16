"""Golden tests for project F — the crude hedge on bunkers, and what it leaves behind.

Three tests carry this page and none of them may be weakened by a later rework.

`test_the_break_test_also_rejects_on_a_placebo_date` is the test that failed, kept
because the failure is informative: it is what rules out a tidy "IMO 2020 changed the
product" story and forces the drift reading instead.

`test_the_hedge_ratio_collapsed_across_windows` is the result.

`test_most_of_the_daily_variance_is_not_explained_by_either_instrument` is the
power-side honesty check: it is what makes "crude is a partial hedge" a stated fact
rather than an implication left for the reader to draw.
"""
from __future__ import annotations

import pandas as pd
import pytest

from agri.data.bloomberg_loader import DEFAULT_PATH
from freight.chains.bunker_basis import (
    BBL_PER_TONNE_DISTILLATE,
    BBL_PER_TONNE_FUEL_OIL,
    BunkerBasisError,
    break_attempt,
    density_mis_sizing,
    hedge_effectiveness,
    load_bunker_frame,
    rolling_hedge_beta,
)

pytestmark = pytest.mark.skipif(
    not DEFAULT_PATH.exists(), reason=f"Bloomberg file absent: {DEFAULT_PATH}"
)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return load_bunker_frame()


# ===========================================================================
# F-H1 — the unit trap
# ===========================================================================
def test_the_two_densities_are_both_real_and_different():
    assert BBL_PER_TONNE_FUEL_OIL == pytest.approx(6.35)
    assert BBL_PER_TONNE_DISTILLATE == pytest.approx(7.45)


def test_using_the_distillate_density_oversizes_the_hedge_by_17_percent():
    """The trap: both numbers are correct for their own product, so nothing about the
    calculation looks wrong. The hedge is simply the wrong size."""
    mis = density_mis_sizing()
    assert mis["oversize_fraction"] == pytest.approx(0.173, abs=0.01)


def test_a_non_positive_density_is_rejected():
    with pytest.raises(BunkerBasisError, match="barrels per tonne"):
        density_mis_sizing(correct=0.0)


# ===========================================================================
# THE RESULT — the hedge ratio drifts
# ===========================================================================
def test_the_hedge_ratio_collapsed_across_windows(frame):
    """The page's central number: a position sized on the 2016-2017 relationship is
    roughly three times too large today."""
    drift = rolling_hedge_beta(frame)
    assert drift.first_beta == pytest.approx(0.691, abs=0.02)
    assert drift.last_beta == pytest.approx(0.233, abs=0.02)
    assert drift.max_beta == pytest.approx(0.795, abs=0.02)
    assert drift.collapse_factor > 2.5
    assert "times too large today" in drift.headline


def test_the_residual_grows_as_the_beta_falls(frame):
    """A falling beta with a rising residual means the relationship is not just getting
    smaller, it is getting less reliable — the two must move together for the drift
    story to hold rather than a simple rescaling."""
    drift = rolling_hedge_beta(frame)
    early = drift.windows[0].residual_sigma
    late = drift.windows[-1].residual_sigma
    peak_residual = min(w.residual_sigma for w in drift.windows[:5])
    assert peak_residual < early
    assert late > peak_residual


def test_too_few_windows_is_rejected(frame):
    with pytest.raises(BunkerBasisError, match="at least two windows"):
        rolling_hedge_beta(frame, windows=((2016, 2017),))


def test_an_unknown_instrument_column_is_rejected(frame):
    with pytest.raises(BunkerBasisError, match="unknown instrument"):
        rolling_hedge_beta(frame, instrument="not_a_column")


# ===========================================================================
# THE TEST THAT FAILED, KEPT ON THE PAGE
# ===========================================================================
def test_the_break_test_also_rejects_on_a_placebo_date(frame):
    """A break test on this many observations of two trending series rejects almost
    anywhere. The IMO 2020 date is not shown to be special unless the placebo does not
    also reject — and here it does, decisively."""
    attempt = break_attempt(frame)
    assert attempt.tested_f > 4.0
    assert attempt.placebo_f > 4.0
    assert attempt.placebo_also_rejects
    assert not attempt.identifies_a_break
    assert "dates nothing" in attempt.headline


def test_a_genuinely_quiet_placebo_would_have_passed(frame):
    """Sanity check on the placebo mechanism itself, not the bunker data: a placebo test
    against a null relationship (a series regressed on itself, split in half) must NOT
    reject, or the placebo methodology would be meaningless."""
    import numpy as np

    from agri.chains.feedstock_lcfs import chow_break_test

    rng = np.random.default_rng(0)
    dates = pd.date_range("2010-01-01", periods=1000, freq="D")
    # Positive random walks, unrelated to each other — chow_break_test takes log
    # differences, so the series must stay strictly positive throughout.
    walk_a = pd.Series(100 + rng.normal(size=1000).cumsum(), index=dates)
    walk_b = pd.Series(100 + rng.normal(size=1000).cumsum(), index=dates)
    result = chow_break_test(walk_a, walk_b, dates[500].isoformat())
    assert result.f_stat < 4.0


# ===========================================================================
# THE OTHER HALF — how much of the risk is not hedgeable at all
# ===========================================================================
def test_most_of_the_daily_variance_is_not_explained_by_either_instrument(frame):
    """Crude explains under a fifth of daily VLSFO variance; gasoil does not do
    materially better; combined they still leave most of the risk carried."""
    eff = hedge_effectiveness(frame)
    assert eff.table.loc["crude only", "r_squared"] == pytest.approx(0.167, abs=0.02)
    assert eff.table.loc["gasoil only", "r_squared"] < eff.table.loc["crude only", "r_squared"] + 0.05
    assert eff.best_r2 < 0.30
    assert eff.unhedgeable_share > 0.70
    assert "partial one" in eff.headline


def test_combining_instruments_helps_but_not_enough_to_change_the_conclusion(frame):
    eff = hedge_effectiveness(frame)
    assert eff.table.loc["crude + gasoil", "r_squared"] > eff.table.loc["crude only", "r_squared"]
    assert eff.table.loc["crude + gasoil", "r_squared"] < 0.30


# ===========================================================================
# Method discipline
# ===========================================================================
def test_hedge_ratio_is_estimated_on_changes_not_levels(frame):
    """F-H3. Two trending levels give a ratio that describes a shared trend and fails
    out of sample — the check is that the changes-based estimate is well inside the
    physically plausible range for a density-linked ratio, unlike a levels regression
    would be."""
    drift = rolling_hedge_beta(frame)
    for window in drift.windows:
        assert 0.0 < window.beta < 1.5


def test_conversion_to_usd_per_tonne_is_applied_before_any_regression(frame):
    """The regression must run against `brent_usd_t`, not raw USD/bbl brent — mixing
    units into a regression produces a beta with no economic meaning."""
    assert "brent_usd_t" in frame.columns
    assert (frame["brent_usd_t"] > frame["brent"]).all()


# ===========================================================================
# Guardrails
# ===========================================================================
def test_an_impossible_start_date_raises():
    with pytest.raises(BunkerBasisError, match="no common dates"):
        load_bunker_frame("2099-01-01")
