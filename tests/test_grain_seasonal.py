"""Golden tests for project D — the harvest that leaves no mark.

Two tests carry the page and must not be weakened by a later rework:

`test_the_harvest_months_are_indistinguishable_from_the_annual_level` is the negative
result. Negative results are fragile because it is always tempting to turn one into a
positive result by moving a window, so it is locked on both harvest definitions.

`test_the_seasonal_grew_while_the_harvest_months_did_not` is the identification. It is a
difference, not a level: it does not require knowing what the harvest months would have
been without Brazil, which is exactly the quantity nobody can observe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agri.data.bloomberg_loader import DEFAULT_PATH
from freight.chains.grain_seasonal import (
    BRAZIL_HARVEST_MONTHS,
    BRAZIL_HARVEST_WIDE,
    NORTHERN_PEAK_MONTHS,
    SeasonalError,
    effective_sample,
    harvest_footprint,
    load_panamax_frame,
    seasonal_profile,
    window_sensitivity,
)

pytestmark = pytest.mark.skipif(
    not DEFAULT_PATH.exists(), reason=f"Bloomberg file absent: {DEFAULT_PATH}"
)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return load_panamax_frame()


# ===========================================================================
# The normalisation — dividing the cycle out without dividing the seasonal out
# ===========================================================================
def test_the_relative_level_is_invariant_to_the_cycle():
    """The point of D-H2, stated as a property rather than as a claim.

    Multiplying an entire calendar year by ten — which is roughly what 2021 did to dry
    bulk — must leave the seasonal shape untouched. If it did not, the monthly profile
    would mostly be a report on which years happened to be strong.
    """
    index = pd.date_range("2010-01-01", "2011-12-31", freq="B")
    base = pd.Series(1_000.0 + 100.0 * np.sin(2 * np.pi * index.month / 12), index=index)
    scaled = base.copy()
    scaled.loc["2011"] *= 10.0

    def relative(series: pd.Series) -> pd.Series:
        year = series.index.year
        return series / series.groupby(year).transform("median")

    pd.testing.assert_series_equal(
        relative(base).loc["2011"], relative(scaled).loc["2011"], rtol=1e-12
    )


def test_the_frame_carries_the_cycle_and_the_seasonal_separately(frame):
    assert {"bpi", "year", "month", "annual_level", "rel"} <= set(frame.columns)
    # the cycle is an order of magnitude larger than the seasonal: without dividing it
    # out, any monthly average is dominated by 2021
    assert frame["annual_level"].max() / frame["annual_level"].min() > 5.0
    assert 0.1 < frame["rel"].median() < 3.0


# ===========================================================================
# RULE C — the sample is years, not days
# ===========================================================================
def test_the_effective_sample_is_years_not_daily_prints(frame):
    """D-H3, enforced in code.

    A month contributes one number per year, not one per trading day. Treating the
    ~6,700 daily prints as independent would shrink every standard error by a factor of
    about sixteen and make every month look significant.
    """
    sample = effective_sample(frame)
    assert sample["n_daily_prints"] > 6_000
    assert sample["n_years"] == pytest.approx(27.0)
    assert sample["n_eff"] == pytest.approx(sample["n_years"], abs=1e-6)
    assert sample["overstatement_factor"] > 10.0


def test_every_month_has_one_observation_per_year(frame):
    profile = seasonal_profile(frame).profile
    assert len(profile) == 12
    assert (profile["n_years"] >= 25).all()
    assert (profile["n_years"] <= 27).all()


def test_the_interval_brackets_the_median(frame):
    profile = seasonal_profile(frame).profile
    assert (profile["lo"] <= profile["median"]).all()
    assert (profile["median"] <= profile["hi"]).all()


# ===========================================================================
# THE NEGATIVE RESULT
# ===========================================================================
def test_the_harvest_months_are_indistinguishable_from_the_annual_level(frame):
    """THE page's result.

    Every Brazilian harvest month's interval contains 1.00 — the year's own median. The
    largest single-origin grain flow in the world does not lift the freight market it
    travels on, to a precision the data can actually support.
    """
    profile = seasonal_profile(frame).profile
    for month in BRAZIL_HARVEST_MONTHS:
        row = profile.loc[month]
        assert row["lo"] <= 1.0 <= row["hi"], (
            f"month {month} became distinguishable from the annual level "
            f"([{row['lo']:.3f}, {row['hi']:.3f}]): the page's central claim needs a rewrite"
        )


def test_february_and_the_autumn_peak_are_distinguishable(frame):
    """The contrast that makes the negative result readable.

    It is not that nothing is seasonal — February and October both exclude the annual
    level. A test that found nothing anywhere would be a test with no power, not a
    finding about the harvest.
    """
    profile = seasonal_profile(frame).profile
    assert profile.loc[2, "hi"] < 1.0, "the February trough stopped being significant"
    assert profile.loc[10, "lo"] > 1.0, "the October peak stopped being significant"


def test_the_trough_is_not_in_the_harvest_window(frame):
    """The naive story predicts a peak in the harvest months. The observed trough sits
    in February and the peak in autumn — both outside the Brazilian window."""
    out = seasonal_profile(frame)
    assert out.trough_month not in BRAZIL_HARVEST_MONTHS
    assert out.peak_month not in BRAZIL_HARVEST_MONTHS
    assert out.peak_month in NORTHERN_PEAK_MONTHS


# ===========================================================================
# THE IDENTIFICATION — growth of the flow against growth of the seasonal
# ===========================================================================
def test_the_seasonal_grew_while_the_harvest_months_did_not(frame):
    """The comparison that does not need the counterfactual.

    Whatever the harvest months would have been without Brazil — unobservable, D-H4 —
    the fact that the amplitude nearly tripled while those months moved by less than one
    point of the annual level does not depend on knowing it.
    """
    footprint = harvest_footprint(frame)
    assert footprint.amplitude_growth > 2.0
    assert footprint.harvest_is_flat
    assert abs(footprint.harvest_drift) < 0.05
    assert "no detectable footprint" in footprint.headline


def test_the_amplitude_growth_is_monotone_enough_to_be_a_trend(frame):
    """Two sub-periods could be an accident; a rising sequence across four is a trend."""
    table = harvest_footprint(frame).table
    amplitudes = table["amplitude"].tolist()
    assert amplitudes[-1] > amplitudes[0]
    assert amplitudes[1] > amplitudes[0]
    assert min(amplitudes) < 0.30 < max(amplitudes)


def test_the_trough_stays_in_the_first_quarter_across_every_period(frame):
    """The seasonal did not merely grow, it grew in the same place — which is what makes
    it a structural pattern rather than four unrelated samples."""
    table = harvest_footprint(frame).table
    assert table["trough_month"].isin([1, 2]).all()


def test_the_result_survives_widening_the_harvest_window(frame):
    """D-H1. A negative result that only holds on a four-month window is a result about
    the window. Feb-Jul is tested alongside Mar-Jun and must give the same verdict."""
    table = window_sensitivity(frame)
    narrow_drift = table["harvest_level_narrow"].iloc[-1] - table["harvest_level_narrow"].iloc[0]
    wide_drift = table["harvest_level_wide"].iloc[-1] - table["harvest_level_wide"].iloc[0]
    assert abs(narrow_drift) < 0.05
    assert abs(wide_drift) < 0.05


def test_the_northern_peak_carries_the_amplitude_instead(frame):
    """Where the growth actually went: the autumn window sits above the annual level in
    every sub-period, and it is the peak in three of the four."""
    table = harvest_footprint(frame).table
    assert (table["northern_level"] > 1.0).all()
    assert (table["northern_level"] > table["harvest_level"]).all()


# ===========================================================================
# Guardrails
# ===========================================================================
def test_a_single_period_is_rejected(frame):
    with pytest.raises(SeasonalError, match="at least two sub-periods"):
        harvest_footprint(frame, periods=((2020, 2026),))


def test_an_impossible_start_date_raises():
    with pytest.raises(SeasonalError, match="no Panamax index observations"):
        load_panamax_frame("2099-01-01")


def test_sparse_years_are_dropped_rather_than_weighted(frame):
    """A year with a handful of prints cannot describe a monthly pattern, and including
    it would silently weight the median toward whichever months were quoted."""
    profile = seasonal_profile(frame)
    # 1998 contributes a single print in the export and must not count as a year
    assert profile.n_years <= frame["year"].nunique() - 1
