"""Golden tests T2-6 — the fixed-parity window, and the negative result it produces.

Two things are guarded here, and the second matters as much as the first.

1. `test_the_substitution_band_does_not_exist`: the page's result is **negative** —
   wide gaps do not revert. A negative result is fragile because it's always
   tempting to turn it into a positive one by changing a parameter. This test locks
   it in, and locks it in across several window and quantile settings.

2. `test_splitting_on_the_raw_level_manufactures_the_expected_answer`: shows where
   the artefact actually comes from. My first hypothesis — "comparing against a
   constant rather than a rolling median" — was wrong, and this test established that
   before the page asserted it. The artefact comes from splitting on the spread's
   **absolute level**: since the median is around -83 USD/t, a large |spread| selects
   the 2004-2005 era instead of genuine outliers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agri.chains.oil_substitution import (
    CENTS_LB_TO_USD_T,
    MYR_PEG_END,
    MYR_PEG_RATE,
    MYR_PEG_START,
    SubstitutionError,
    estimate_half_life,
    load_peg_window_spread,
    rolling_deviation,
    structural_drift,
    substitution_verdict,
)
from agri.data.bloomberg_loader import DEFAULT_PATH, load

pytestmark = pytest.mark.skipif(
    not DEFAULT_PATH.exists(), reason=f"Bloomberg file absent: {DEFAULT_PATH}"
)


@pytest.fixture(scope="module")
def frame():
    return load_peg_window_spread()


# ===========================================================================
# The window and its conversion
# ===========================================================================
def test_the_window_is_exactly_the_peg_period(frame):
    assert frame.index.min() >= pd.Timestamp(MYR_PEG_START)
    assert frame.index.max() <= pd.Timestamp(MYR_PEG_END)
    assert len(frame) > 1_500


def test_the_palm_conversion_is_a_division_by_a_constant(frame):
    """The whole point of the window: the currency isn't estimated, it's decreed."""
    pd.testing.assert_series_equal(
        frame["palm_usd"], (frame["palm_myr"] / MYR_PEG_RATE).rename("palm_usd")
    )
    assert MYR_PEG_RATE == pytest.approx(3.80)


def test_the_soy_leg_uses_the_cents_to_tonne_conversion(frame):
    soy_raw = load("cbot_soyoil").reindex(frame.index)
    pd.testing.assert_series_equal(
        frame["soy_usd"], (soy_raw * CENTS_LB_TO_USD_T).rename("soy_usd")
    )


def test_both_legs_land_in_a_plausible_usd_range(frame):
    """A vegetable oil trades between 200 and 900 USD/t over this period. A failed
    conversion would land outside this range by a visible factor."""
    assert 200 < frame["palm_usd"].median() < 900
    assert 200 < frame["soy_usd"].median() < 900


def test_palm_trades_below_soy_most_of_the_time(frame):
    assert (frame["spread"] < 0).mean() > 0.75


# ===========================================================================
# THE NEGATIVE RESULT
# ===========================================================================
@pytest.mark.parametrize("window", [125, 250, 375])
@pytest.mark.parametrize("quantile", [0.65, 0.70, 0.80])
def test_the_substitution_band_does_not_exist(frame, window, quantile):
    """THE page's test, and it's repeated across nine settings.

    The thesis predicts wide gaps revert FASTER than narrow ones. Across every
    combination of rolling window and separating quantile, the opposite comes out. A
    negative result that only holds for one setting isn't one.
    """
    verdict = substitution_verdict(frame["spread"], window=window, quantile=quantile)
    assert not verdict.substitution_band_exists
    assert np.isfinite(verdict.narrow.half_life_days)


def test_narrow_deviations_do_revert_quickly(frame):
    """The contrast that makes the result readable: it isn't that nothing ever
    reverts. Narrow gaps revert fast — that's microstructure, not substitution."""
    verdict = substitution_verdict(frame["spread"])
    assert verdict.narrow.half_life_days < 30
    assert verdict.narrow.pvalue < 0.01


def test_the_wide_regime_coefficient_is_not_merely_small(frame):
    """Distinguishing "underpowered" from "no effect": the wide sample is ample and
    the coefficient isn't just small, it has the wrong sign."""
    verdict = substitution_verdict(frame["spread"])
    assert verdict.wide.n_obs > 300
    assert verdict.wide.pvalue > 0.10
    assert verdict.wide.beta >= -0.005


def test_the_headline_states_the_negative_result_plainly(frame):
    verdict = substitution_verdict(frame["spread"])
    assert "contrary to the thesis" in verdict.headline
    assert "no mean reversion" in verdict.headline


# ===========================================================================
# THE ARTEFACT S4 CALLS OUT
# ===========================================================================
def test_splitting_on_the_raw_level_manufactures_the_expected_answer(frame):
    """The mistake avoided, kept as a test to guard against falling back into it.

    Written first to show that testing "against a constant" created the artefact —
    the data said no, and the real origin had to be found. Here it is: the artefact
    comes from splitting on the spread's **absolute level** rather than on a
    deviation from a centre. Since the median spread is around -83 USD/t, selecting a
    large |spread| doesn't select genuine outliers: it selects the 2004-2005 era,
    when palm was deeply discounted. What gets measured is then an era's dynamics,
    mislabelled substitution.

    Splitting on a deviation — even from a constant — already gives the right
    answer; the rolling median only sharpens it further.
    """
    level = frame["spread"].abs()
    threshold = level.quantile(0.70)
    wide = estimate_half_life(frame["spread"], mask=level >= threshold, label="wide")
    narrow = estimate_half_life(frame["spread"], mask=level < threshold, label="narrow")

    # the naive split on the LEVEL "finds" the band...
    assert wide.half_life_days < narrow.half_life_days

    # ...while splitting on a DEVIATION, even from a constant, doesn't find it
    deviation = frame["spread"] - frame["spread"].median()
    threshold_deviation = deviation.abs().quantile(0.70)
    wide_deviation = estimate_half_life(
        frame["spread"], mask=deviation.abs() >= threshold_deviation, label="wide"
    )
    narrow_deviation = estimate_half_life(
        frame["spread"], mask=deviation.abs() < threshold_deviation, label="narrow"
    )
    assert wide_deviation.half_life_days > narrow_deviation.half_life_days

    # ...and neither does the page's test
    assert not substitution_verdict(frame["spread"]).substitution_band_exists


def test_the_spread_drifts_rather_than_oscillates(frame):
    """The reason for the artefact: over seven years the spread moves by nearly
    200 USD/t, which rules out treating it as stationary around a constant."""
    drift = structural_drift(frame["spread"])
    assert abs(drift.attrs["drift_usd_t"]) > 150
    assert drift.attrs["range_usd_t"] > 200
    assert drift["median_spread"].iloc[0] > 0 > drift["median_spread"].iloc[-1]


def test_the_two_tails_are_separated_in_time(frame):
    """The detail that settles it in S4: palm expensive early on, deeply discounted
    at the end. These aren't two excursions around an equilibrium, they're two eras."""
    deviation = frame["spread"] - frame["spread"].median()
    threshold = deviation.abs().quantile(0.70)
    expensive_years = deviation[deviation > threshold].index.year
    cheap_years = deviation[deviation < -threshold].index.year
    assert expensive_years.min() < cheap_years.min()
    assert pd.Series(expensive_years).median() < pd.Series(cheap_years).median()


# ===========================================================================
# Guardrails
# ===========================================================================
def test_rolling_deviation_refuses_a_degenerate_window(frame):
    with pytest.raises(SubstitutionError, match="too short"):
        rolling_deviation(frame["spread"], window=5)


def test_substitution_verdict_refuses_an_implausible_quantile(frame):
    with pytest.raises(SubstitutionError, match="quantile"):
        substitution_verdict(frame["spread"], quantile=0.30)


def test_the_loader_marks_palm_as_quoted_in_ringgit():
    """Unit guardrail: if someone ever adds a `scale` to palm or redeclares it in
    USD, this test must fail before the page produces wrong spreads."""
    from agri.data.bloomberg_loader import SERIES_SPECS

    spec = SERIES_SPECS["palm_oil_myr"]
    assert spec.unit == "MYR/t"
    assert getattr(spec, "scale", 1.0) in (1.0, None)
    assert "USDMYR" in (spec.note or "")


def test_no_usdmyr_series_exists_in_the_export():
    """The constraint that justifies the page's entire construction. If USDMYR is
    ever added, this test fails and forces reopening the test over thirty years
    instead of seven."""
    from agri.data.bloomberg_loader import SERIES_SPECS

    assert not any("myr" in key.lower() and "palm" not in key.lower() for key in SERIES_SPECS)
