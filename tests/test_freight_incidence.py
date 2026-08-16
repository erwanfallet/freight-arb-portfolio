"""Golden tests for project E — who pays the freight inside a delivered price.

Three tests carry this page and none of them may be weakened by a later rework.

`test_the_test_can_see_full_passthrough` is the one that makes a null admissible. Without
it, "no relationship found" is indistinguishable from "no relationship looked for".

`test_full_passthrough_is_rejected_where_the_sample_has_power` is the result.

`test_no_verdict_is_drawn_from_the_low_frequency_estimates` is the honesty guard. The
monthly point estimate is +0.62 and looks like pass-through; on 43 observations it means
nothing, and the code must refuse to say otherwise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agri.core.voyage import VoyageParams
from agri.data.bloomberg_loader import DEFAULT_PATH
from freight.chains.freight_incidence import (
    MIN_OBS_FOR_VERDICT,
    IncidenceError,
    freight_share_of_value,
    incidence_by_frequency,
    lag_scan,
    load_incidence_frame,
    power_benchmark,
    predicted_lag_days,
)

pytestmark = pytest.mark.skipif(
    not DEFAULT_PATH.exists(), reason=f"Bloomberg file absent: {DEFAULT_PATH}"
)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return load_incidence_frame()


# ===========================================================================
# The two facts that get conflated
# ===========================================================================
def test_freight_is_a_tenth_of_the_value_and_a_thirtieth_of_the_variance(frame):
    """Both statements are true and they answer different questions.

    Freight is economically large in the delivered price and statistically almost
    invisible in its daily movement. Quoting the first as if it implied the second is
    the error this page exists to separate.
    """
    shares = freight_share_of_value(frame)
    assert 0.05 < shares["share_of_level"] < 0.20
    assert shares["share_of_variance"] < shares["share_of_level"] / 2
    assert shares["sigma_cfr"] > 4 * shares["sigma_freight"]


# ===========================================================================
# E-H3 — the lag is predicted, not fitted
# ===========================================================================
def test_the_lag_comes_from_the_voyage_not_from_the_data():
    """A lag chosen by scanning for the biggest correlation is a lag the search made.

    This one is computed from distance and speed before any data is touched, so it can
    be wrong — which is what makes testing at it meaningful.
    """
    lag = predicted_lag_days()
    assert lag["distance_nm"] == pytest.approx(3_500.0)
    # 3500 nm / (12.5 kn x 24 h) = 11.67 days at sea, plus 6 port days
    assert lag["sea_days"] == pytest.approx(11.667, abs=1e-2)
    assert lag["calendar_days"] == pytest.approx(17.667, abs=1e-2)


def test_a_slower_ship_predicts_a_longer_lag():
    """Sanity check that the lag is structural: it must move with the voyage."""
    fast = predicted_lag_days(params=VoyageParams(speed_laden_kn=14.0))
    slow = predicted_lag_days(params=VoyageParams(speed_laden_kn=11.0))
    assert slow["calendar_days"] > fast["calendar_days"]


# ===========================================================================
# THE POWER ARGUMENT — what makes a null admissible
# ===========================================================================
def test_the_test_can_see_full_passthrough(frame):
    """THE precondition for reading anything into a null.

    Under the identity CFR = FOB + freight with an independent FOB, full pass-through
    implies a correlation of exactly sigma(dFreight)/sigma(dCFR). That has to clear the
    significance band, or "we found nothing" says nothing.
    """
    power = power_benchmark(frame)
    assert power.can_detect
    assert power.margin > 2.0
    assert power.implied_correlation == pytest.approx(0.18, abs=0.03)
    assert "can see it" in power.headline


def test_the_implied_correlation_is_a_ratio_of_volatilities_not_an_estimate(frame):
    """It follows from the identity, so it must equal the ratio exactly."""
    power = power_benchmark(frame)
    changes = frame.diff().dropna()
    assert power.implied_correlation == pytest.approx(
        float(changes["c5"].std() / changes["cfr62"].std()), rel=1e-12
    )


# ===========================================================================
# THE RESULT
# ===========================================================================
def test_no_lag_carries_a_relationship_including_the_predicted_one(frame):
    """The negative result, locked across the whole ±30-day window.

    If a later rework finds a significant lag, either the data changed or the method
    did, and both deserve a rewrite rather than a silent pass.
    """
    scan = lag_scan(frame)
    assert scan.significant_lags == []
    assert abs(scan.value_at_predicted) < scan.band
    assert "not a timing problem" in scan.headline


def test_the_predicted_lag_is_inside_the_scanned_window(frame):
    """A scan that did not cover the predicted lag would not have tested the hypothesis."""
    scan = lag_scan(frame)
    assert scan.lags.min() <= scan.predicted_lag <= scan.lags.max()
    assert scan.predicted_lag == 18


def test_full_passthrough_is_rejected_where_the_sample_has_power(frame):
    """The page's result: the delivered price does not move with the freight in it."""
    result = incidence_by_frequency(frame)
    powered = result.powered
    assert len(powered) >= 2
    assert result.full_passthrough_rejected_everywhere_with_power
    for estimate in powered:
        lo, hi = estimate.ci
        assert hi < 1.0, f"{estimate.label}: full pass-through stopped being rejected"


def test_absorption_is_not_claimed_to_be_proven(frame):
    """The honest half. Rejecting b=1 does not establish b=0, and the code must not
    pretend otherwise: zero sits inside every interval."""
    result = incidence_by_frequency(frame)
    for estimate in result.estimates:
        assert not estimate.rejects_absorption
        lo, hi = estimate.ci
        assert lo <= 0.0 <= hi


def test_no_verdict_is_drawn_from_the_low_frequency_estimates(frame):
    """The honesty guard.

    The monthly coefficient is positive and looks like the pass-through the daily test
    rejects. On 43 differenced observations its interval spans from full absorption to
    twice full pass-through, and the code must refuse to draw a verdict from it.
    """
    result = incidence_by_frequency(frame)
    table = result.to_frame()
    for label in ("monthly", "quarterly"):
        if label not in table.index:
            continue
        row = table.loc[label]
        assert not row["has_power"]
        assert not row["rejects_full_passthrough"]
        assert not row["rejects_absorption"]
        assert row["n"] < MIN_OBS_FOR_VERDICT


def test_the_headline_states_the_limit_as_well_as_the_result(frame):
    """The page must say where it cannot answer, not only where it can."""
    headline = incidence_by_frequency(frame).headline
    assert "full pass-through is rejected" in headline
    assert "quarterly" in headline
    assert "too few observations" in headline


# ===========================================================================
# Method discipline
# ===========================================================================
def test_the_regression_runs_on_changes_not_levels(frame):
    """E-H4. Two trending levels give a coefficient that describes a shared trend.

    The check is that the estimate differs from the levels regression, which would be
    the mistake — if they agreed, the differencing would not be doing anything.
    """
    from agri.core.stats import hac_ols

    on_levels = hac_ols(frame["cfr62"], frame[["c5"]]).params["c5"]
    on_changes = incidence_by_frequency(frame).estimates[0].beta
    assert abs(on_levels - on_changes) > 0.5


def test_standard_errors_are_hac_not_naive(frame):
    """Both series are autocorrelated; a naive standard error would make the coefficient
    look far better determined than it is, and could flip the verdict on b=1."""
    import statsmodels.api as sm

    changes = frame.diff().dropna()
    naive = sm.OLS(
        changes["cfr62"], sm.add_constant(changes[["c5"]])
    ).fit().bse["c5"]
    hac = incidence_by_frequency(frame).estimates[0].std_error
    assert hac > naive


# ===========================================================================
# Guardrails
# ===========================================================================
def test_an_impossible_start_date_raises():
    with pytest.raises(IncidenceError, match="no common dates"):
        load_incidence_frame("2099-01-01")


def test_a_frequency_set_with_no_usable_horizon_raises(frame):
    with pytest.raises(IncidenceError, match="enough observations to estimate"):
        incidence_by_frequency(frame, frequencies=(("decadal", "10YE"),))
