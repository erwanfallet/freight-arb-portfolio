"""Golden tests T1-1 — freight inside the C&F calculation.

Freight on the reference voyage (Panamax, Santos -> Qingdao, TCE 15,000, VLSFO 500,
MGO 700) is **linear in ballast_share**:

    freight(b) = 20.610258 + b x 15.816610   $/t

It's this linearity that makes the tipping point computable by hand, and it's the first
thing verified below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agri.chains.freight_cf import (
    NoBreakevenInRange,
    arb_usd_t,
    ballast_breakeven,
    build_conventions,
    disagreement_episodes,
    financing_cost_usd_t,
    marginal_decision_zone,
    pnl_attribution,
    sensitivity_grid,
    sign_flip_rate,
    spread_distribution,
    spread_seasonality,
)
from agri.core.voyage import ROUTES, VESSELS, VoyageParams
from agri.fixtures.freight_cf import build, build_frame

PANAMAX = VESSELS["panamax"]
SANTOS_QINGDAO = ROUTES["santos_qingdao"]

FREIGHT_AT_ZERO_BALLAST = 20.610258
FREIGHT_SLOPE_PER_BALLAST = 15.816610


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return build_frame()


# ===========================================================================
# The accounting identity
# ===========================================================================
def test_financing_uses_a_360_day_base():
    # (440 + 36) x 5.5% x (78 + 30) / 360 = 476 x 0.055 x 0.3 = 7.854
    assert financing_cost_usd_t(
        440.0, 36.0, annual_rate=0.055, voyage_days=78.0, credit_days=30.0
    ) == pytest.approx(7.854, abs=1e-3)


def test_arb_identity_is_a_plain_subtraction():
    # 500 - 440 - 36 - financing - 0.85
    # financing = (440 + 36) x 0.055 x (78 + 30)/360 = 7.854
    # arb = 500 - 440 - 36 - 7.854 - 0.85 = 15.296
    index = pd.date_range("2024-01-01", periods=3, freq="B")
    out = arb_usd_t(
        pd.Series(500.0, index=index),
        pd.Series(440.0, index=index),
        pd.Series(36.0, index=index),
        annual_rate=0.055,
        voyage_days=78.0,
        credit_days=30.0,
        insurance_usd_t=0.85,
    )
    assert out.iloc[0] == pytest.approx(15.296, abs=1e-3)


def test_negative_financing_rate_is_rejected():
    with pytest.raises(Exception, match="negative"):
        financing_cost_usd_t(440.0, 36.0, annual_rate=-0.01, voyage_days=78.0, credit_days=30.0)


# ===========================================================================
# The three conventions
# ===========================================================================
def test_freight_is_linear_in_ballast_share():
    """The property that makes the tipping point computable by hand."""
    from agri.core.voyage import voyage_freight_usd_t

    for share in (0.0, 0.25, 0.5, 0.75, 1.0):
        expected = FREIGHT_AT_ZERO_BALLAST + share * FREIGHT_SLOPE_PER_BALLAST
        actual = voyage_freight_usd_t(
            15_000.0,
            500.0,
            700.0,
            vessel=PANAMAX,
            route=SANTOS_QINGDAO,
            params=VoyageParams(ballast_share=share),
        ).freight_usd_t
        assert actual == pytest.approx(expected, abs=1e-5)


def test_full_is_always_above_index(frame):
    """The central invariant: ballast can only make it more expensive, never the reverse.

    A persistent `freight_full < freight_index` would be a voyage-model error, not an
    opportunity — it's the first failure mode listed in the spec.
    """
    assert (frame["freight_full"] > frame["freight_index"]).all()
    assert (frame["spread_full_index"] > 0).all()


def test_internal_convention_lags_the_full_one(frame):
    """The internal convention is stable *because* it lags — that's its whole appeal and its whole flaw."""
    assert frame["freight_internal"].std() < frame["freight_full"].std()
    correlation = frame["freight_internal"].corr(frame["freight_full"])
    assert 0.0 < correlation < 0.99


def test_internal_convention_refuses_a_partial_window():
    """An "internal" average over three points is not an internal average."""
    series = build(periods=200, seed=1)
    out = build_conventions(
        series["tce"],
        series["vlsfo"],
        series["mgo"],
        series["cif"],
        series["fob"],
        vessel=PANAMAX,
        route=SANTOS_QINGDAO,
        params=VoyageParams(),
        internal_window_days=90,
    )
    # 200 input days - 89 eaten by the window = 111 usable days
    assert len(out) == 111


def test_arb_is_more_generous_under_the_index_convention(frame):
    """The trading desk systematically sees a more open arb. That's the disagreement."""
    assert (frame["arb_index"] > frame["arb_full"]).all()
    assert frame["arb_index"].mean() - frame["arb_full"].mean() == pytest.approx(
        frame["spread_full_index"].mean(), abs=0.5
    )


def test_disagreement_flag_marks_sign_conflicts(frame):
    signs = np.sign(frame[["arb_index", "arb_full", "arb_internal"]].to_numpy())
    expected = (signs != signs[:, [0]]).any(axis=1)
    assert (frame["disagreement"].to_numpy() == expected).all()
    assert frame["disagreement"].any()


# ===========================================================================
# S4 — the disagreement panel
# ===========================================================================
def test_sign_flip_rate_carries_an_exact_interval(frame):
    out = sign_flip_rate(frame)
    assert 0.0 < out.point < 1.0
    assert out.lo < out.point < out.hi
    assert "95% CI" in out.summary


def test_sign_flip_rate_is_zero_when_conventions_agree():
    """On a widely open arb, the conventions agree — and that has to be said."""
    index = pd.date_range("2024-01-01", periods=120, freq="B")
    wide = pd.DataFrame(
        {"arb_index": pd.Series(80.0, index=index), "arb_full": pd.Series(60.0, index=index)}
    )
    assert sign_flip_rate(wide).point == 0.0


def test_spread_distribution_reports_median_and_iqr_not_just_the_mean(frame):
    """The distribution is bounded on the left by construction: a mean alone lies."""
    stats = spread_distribution(frame)
    assert stats["min"] > 0
    assert stats["q1"] < stats["median"] < stats["q3"]
    assert stats["iqr"] == pytest.approx(stats["q3"] - stats["q1"])


def test_spread_seasonality_covers_twelve_months(frame):
    seasonal = spread_seasonality(frame)
    assert set(seasonal.index) <= set(range(1, 13))
    assert len(seasonal) == 12


def test_disagreement_episodes_are_dated_and_measurable(frame):
    episodes = disagreement_episodes(frame, min_days=3)
    assert len(episodes) > 0
    assert (episodes["n_obs"] >= 3).all()
    assert "duration_days" in episodes.columns


# ===========================================================================
# S5 — the marginal decision zone: the number for the email
# ===========================================================================
def test_marginal_zone_produces_the_headline(frame):
    zone = marginal_decision_zone(frame, band_usd_t=5.0)
    assert zone.n_in_band > 0
    assert 0.0 < zone.share_of_sample <= 1.0
    headline = zone.headline
    assert "5 USD/t of breakeven" in headline
    assert "95% exact CI" in headline
    assert "freight convention" in headline


def test_a_wider_band_captures_more_days(frame):
    narrow = marginal_decision_zone(frame, band_usd_t=2.0)
    wide = marginal_decision_zone(frame, band_usd_t=10.0)
    assert wide.n_in_band > narrow.n_in_band


def test_marginal_zone_is_empty_when_the_arb_is_never_close():
    index = pd.date_range("2024-01-01", periods=50, freq="B")
    wide = pd.DataFrame(
        {"arb_index": pd.Series(80.0, index=index), "arb_full": pd.Series(60.0, index=index)}
    )
    zone = marginal_decision_zone(wide, band_usd_t=5.0)
    assert zone.n_in_band == 0
    assert zone.share_of_sample == 0.0


def test_zero_band_is_rejected():
    frame = pd.DataFrame({"arb_index": [1.0], "arb_full": [1.0]})
    with pytest.raises(Exception, match="band"):
        marginal_decision_zone(frame, band_usd_t=0.0)


# ===========================================================================
# Tipping point — ballast_share*
# ===========================================================================
def test_ballast_breakeven_hand_computed():
    """Without financing or insurance, the arb is affine in ballast and the threshold is exact.

        arb(b) = (CIF - FOB) - 20.610258 - b x 15.816610

    Setting CIF - FOB = 28.518563, the root lands exactly at b* = 0.50:
        (28.518563 - 20.610258) / 15.816610 = 7.908305 / 15.816610 = 0.500000
    """
    out = ballast_breakeven(
        15_000.0,
        500.0,
        700.0,
        cif_usd_t=468.518563,
        fob_usd_t=440.0,
        vessel=PANAMAX,
        route=SANTOS_QINGDAO,
        params=VoyageParams(),
        annual_rate=0.0,
        insurance_usd_t=0.0,
    )
    assert out.theta_star == pytest.approx(0.5, abs=1e-6)
    # the sensitivity is the freight slope, up to sign: -15.8166 $/t per unit of ballast
    assert out.sensitivity == pytest.approx(-FREIGHT_SLOPE_PER_BALLAST, abs=1e-3)


def test_breakeven_summary_is_the_mail_sentence():
    out = ballast_breakeven(
        15_000.0,
        500.0,
        700.0,
        cif_usd_t=468.518563,
        fob_usd_t=440.0,
        vessel=PANAMAX,
        route=SANTOS_QINGDAO,
        params=VoyageParams(),
        annual_rate=0.0,
        insurance_usd_t=0.0,
    )
    assert "ballast_share* = 0.5" in out.summary


def test_financing_pushes_the_breakeven_down():
    """Financing is one more cost: it closes the arb sooner, i.e. at less ballast."""
    kwargs = dict(
        cif_usd_t=468.518563,
        fob_usd_t=440.0,
        vessel=PANAMAX,
        route=SANTOS_QINGDAO,
        params=VoyageParams(),
        insurance_usd_t=0.0,
    )
    free = ballast_breakeven(15_000.0, 500.0, 700.0, annual_rate=0.0, **kwargs)
    costly = ballast_breakeven(15_000.0, 500.0, 700.0, annual_rate=0.055, **kwargs)
    assert costly.theta_star < free.theta_star


def test_a_wide_open_arb_has_no_breakeven_and_that_is_the_result():
    """"Even charging 100% of ballast, the arb stays open" is a publishable claim."""
    with pytest.raises(NoBreakevenInRange) as excinfo:
        ballast_breakeven(
            15_000.0,
            500.0,
            700.0,
            cif_usd_t=560.0,      # 120 $/t gap: well above any plausible freight
            fob_usd_t=440.0,
            vessel=PANAMAX,
            route=SANTOS_QINGDAO,
            params=VoyageParams(),
        )
    assert excinfo.value.margin_lo > 0
    assert "stays positive" in str(excinfo.value)


def test_a_shut_arb_has_no_breakeven_either():
    with pytest.raises(NoBreakevenInRange, match="stays negative"):
        ballast_breakeven(
            15_000.0,
            500.0,
            700.0,
            cif_usd_t=445.0,      # 5 $/t gap: shut even with zero ballast charged
            fob_usd_t=440.0,
            vessel=PANAMAX,
            route=SANTOS_QINGDAO,
            params=VoyageParams(),
        )


def test_breakeven_reports_distance_in_sigmas_when_history_is_given():
    history = pd.Series([0.3, 0.5, 0.7, 0.9])
    out = ballast_breakeven(
        15_000.0,
        500.0,
        700.0,
        cif_usd_t=468.518563,
        fob_usd_t=440.0,
        vessel=PANAMAX,
        route=SANTOS_QINGDAO,
        params=VoyageParams(ballast_share=0.8),
        annual_rate=0.0,
        insurance_usd_t=0.0,
        ballast_history=history,
    )
    assert out.theta_current == 0.8
    assert out.distance_sigmas is not None
    assert out.distance_sigmas < 0      # the threshold is below the level used


# ===========================================================================
# S6 — cross sensitivity
# ===========================================================================
def test_sensitivity_grid_shape_and_monotonicity():
    grid = sensitivity_grid(
        15_000.0,
        500.0,
        700.0,
        cif_usd_t=468.5,
        fob_usd_t=440.0,
        vessel=PANAMAX,
        route=SANTOS_QINGDAO,
        params=VoyageParams(),
        ballast_values=np.array([0.0, 0.5, 1.0]),
        bunker_shifts_pct=np.array([-0.2, 0.0, 0.2]),
    )
    assert len(grid) == 9
    # at a fixed bunker shift, more ballast = firmer (lower) arb
    at_zero_shift = grid[grid["bunker_shift_pct"] == 0.0].sort_values("ballast_share")
    assert at_zero_shift["arb_usd_t"].is_monotonic_decreasing
    # at fixed ballast, pricier bunkers = firmer (lower) arb
    at_full_ballast = grid[grid["ballast_share"] == 1.0].sort_values("bunker_shift_pct")
    assert at_full_ballast["arb_usd_t"].is_monotonic_decreasing


# ===========================================================================
# S7 — P&L attribution
# ===========================================================================
def test_pnl_attribution_sign_convention(frame):
    """Positive sign = the freight department charged above its cost of the day."""
    out = pnl_attribution(frame, cargo_t=66_000.0)
    assert (out["gap_usd_t"] * 66_000.0).equals(out["pnl_shifted_usd"])
    positive = out[out["gap_usd_t"] > 0]
    assert (positive["credited"] == "freight department").all()
    negative = out[out["gap_usd_t"] <= 0]
    assert (negative["credited"] == "trading desk").all()


def test_pnl_attribution_cumulates(frame):
    out = pnl_attribution(frame, cargo_t=66_000.0)
    assert out["pnl_shifted_cum_usd"].iloc[-1] == pytest.approx(out["pnl_shifted_usd"].sum())


def test_zero_cargo_is_rejected(frame):
    with pytest.raises(Exception, match="cargo_t"):
        pnl_attribution(frame, cargo_t=0.0)


# ===========================================================================
# The synthetic dataset does impose the phenomenon
# ===========================================================================
def test_fixture_puts_a_large_share_of_days_in_the_marginal_band(frame):
    """Without this property, the page would have nothing to show — that's the fixture's point."""
    zone = marginal_decision_zone(frame, band_usd_t=5.0)
    assert zone.share_of_sample > 0.3


def test_fixture_produces_genuine_sign_conflicts(frame):
    zone = marginal_decision_zone(frame, band_usd_t=5.0)
    assert zone.n_decided_by_convention > 0


def test_fixture_is_deterministic():
    a = build_frame(seed=7)
    b = build_frame(seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_give_different_data():
    a = build_frame(seed=1)
    b = build_frame(seed=2)
    assert not a["arb_full"].equals(b["arb_full"])
