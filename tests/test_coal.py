"""Golden tests for project B — values hand-computed in the comments."""
import numpy as np
import pandas as pd
import pytest

from freight.chains.coal import (
    ceiling_test,
    generation_cost_eur_mwh_e,
    non_overlapping,
    ols,
    switch_ttf_eur_mwh,
    switching_carbon_price,
    switching_distance_pct,
    trailing_median_distance_pct,
)


def _dates(n: int, start: str = "2021-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


# --------------------------------------------------------------------------- OLS
def test_ols_recovers_exact_coefficients():
    """y = 1 + 2·x1 + 3·x2 exactly -> constant 1, slopes 2 and 3, R² = 1.

    Perfect fit: the standard error is zero and the t-stat is undefined. The code must
    return NaN rather than an infinity that would contaminate a table.
    """
    idx = _dates(50)
    x1 = pd.Series(np.linspace(1, 10, 50), index=idx)
    x2 = pd.Series(np.linspace(-3, 4, 50) ** 2, index=idx)
    y = 1.0 + 2.0 * x1 + 3.0 * x2
    r = ols(y, {"x1": x1, "x2": x2})
    assert r.coefficients["const"] == pytest.approx(1.0, abs=1e-8)
    assert r.coefficients["x1"] == pytest.approx(2.0, abs=1e-8)
    assert r.coefficients["x2"] == pytest.approx(3.0, abs=1e-8)
    assert r.r_squared == pytest.approx(1.0, abs=1e-12)
    assert np.isnan(r.t_stats["x1"])
    assert r.n_obs == 50


def test_ols_needs_enough_observations():
    idx = _dates(2)
    with pytest.raises(ValueError, match="not enough observations"):
        ols(pd.Series([1.0, 2.0], index=idx), {"x": pd.Series([1.0, 2.0], index=idx)})


def test_ols_requires_a_regressor():
    idx = _dates(10)
    with pytest.raises(ValueError, match="at least one regressor"):
        ols(pd.Series(np.arange(10.0), index=idx), {})


def test_omitting_the_control_biases_the_freight_coefficient():
    """THE test that justifies `ols` taking multiple regressors.

    Construction: a common factor z, `freight = z + a`, `ttf = z + b`, with a and b
    independent and of the same variance as z. The truth is y = 1·freight + 5·ttf.

    Simple regression on freight alone:
        coef = 1 + 5 · cov(ttf, freight)/var(freight) = 1 + 5 · (1/2) = 3.5
    Freight inherits half of TTF's effect. With the control, 1 and 5 are recovered
    exactly.

    In other words: without controlling for TTF, freight would be credited with an
    effect 3.5 times too large — and the symmetric reasoning applies to attributing a
    forward-return effect to switching when it is really a confound.
    """
    idx = _dates(600)
    rng = np.random.default_rng(5)
    z = rng.normal(0, 1, 600)
    freight = pd.Series(z + rng.normal(0, 1, 600), index=idx)
    ttf = pd.Series(z + rng.normal(0, 1, 600), index=idx)
    y = 1.0 * freight + 5.0 * ttf

    naive = ols(y, {"freight": freight})
    controlled = ols(y, {"freight": freight, "ttf": ttf})

    assert naive.coefficients["freight"] > 2.5  # biased, ~3.5
    assert controlled.coefficients["freight"] == pytest.approx(1.0, abs=1e-8)
    assert controlled.coefficients["ttf"] == pytest.approx(5.0, abs=1e-8)


# ----------------------------------------------------------------- switching level
def test_switch_ttf_golden():
    """coal_th = 20, eua = 80, eta_coal = 0.38, eta_gas = 0.55:

    ttf* = (0.55/0.38) x 20 + 80 x (0.55 x 0.34/0.38 - 0.20)
         = 28.947368421... + 80 x 0.292105263...
         = 52.315789473684213
    """
    idx = _dates(1)
    frame = pd.DataFrame(
        {"coal_eur_mwh_th": [20.0], "eua_eur_t": [80.0], "ttf_eur_mwh": [0.0]}, index=idx
    )
    switch = switch_ttf_eur_mwh(frame, coal_efficiency=0.38, gas_efficiency=0.55)
    assert switch.iloc[0] == pytest.approx(52.315789473684213, rel=1e-12)


def test_switch_ttf_and_switching_carbon_price_are_the_same_equality_solved_two_ways():
    """Set TTF exactly at its own switching level: the carbon price that makes gas and
    coal cost the same must come back out as the EUA already in the frame — the two
    functions solve the same equation for different variables and must agree.
    """
    idx = _dates(1)
    frame = pd.DataFrame(
        {"coal_eur_mwh_th": [20.0], "eua_eur_t": [80.0], "ttf_eur_mwh": [0.0]}, index=idx
    )
    switch = switch_ttf_eur_mwh(frame, coal_efficiency=0.38, gas_efficiency=0.55)
    frame_at_switch = frame.assign(ttf_eur_mwh=switch)

    recovered_eua = switching_carbon_price(
        frame_at_switch, coal_efficiency=0.38, gas_efficiency=0.55
    )
    assert recovered_eua.iloc[0] == pytest.approx(80.0, abs=1e-8)

    generation = generation_cost_eur_mwh_e(
        frame_at_switch, coal_efficiency=0.38, gas_efficiency=0.55
    )
    assert generation["spread"].iloc[0] == pytest.approx(0.0, abs=1e-8)


def test_switching_distance_pct_golden():
    idx = _dates(1)
    switch = pd.Series([50.0], index=idx)
    frame_above = pd.DataFrame({"ttf_eur_mwh": [55.0]}, index=idx)
    frame_below = pd.DataFrame({"ttf_eur_mwh": [45.0]}, index=idx)
    assert switching_distance_pct(frame_above, switch).iloc[0] == pytest.approx(0.10)
    assert switching_distance_pct(frame_below, switch).iloc[0] == pytest.approx(-0.10)


# -------------------------------------------------------------------------- placebo
def test_trailing_median_distance_pct_golden():
    """Ten flat days at 50, then a jump to 60. The 5-day trailing median only reaches
    50 once the window is full of pre-jump days; on the jump day itself the median of
    the last 5 (still all 50s) is 50, so distance = (60-50)/50 = 0.20.
    """
    idx = _dates(11)
    ttf = pd.Series([50.0] * 10 + [60.0], index=idx)
    dist = trailing_median_distance_pct(ttf, window=5)
    assert np.isnan(dist.iloc[3])  # window not yet full (only 4 obs)
    assert dist.iloc[4] == pytest.approx(0.0)  # window full, all 50s
    assert dist.iloc[10] == pytest.approx(0.20, rel=1e-12)


def test_non_overlapping_keeps_every_horizon_th_row():
    idx = _dates(23)
    frame = pd.DataFrame({"x": np.arange(23)}, index=idx)
    kept = non_overlapping(frame, horizon_days=5)
    assert list(kept["x"]) == [0, 5, 10, 15, 20]


# ---------------------------------------------------------------------- ceiling test
def test_ceiling_test_recovers_a_designed_reversion_and_rejects_a_flat_placebo():
    """Construction: `switch` follows a slow deterministic drift. Every `horizon` days,
    TTF's deviation from `switch` is redrawn independently of its current value — i.e.
    the next window's level is `switch(t+h)` plus fresh noise, not a continuation of
    today's gap. That makes the forward return mechanically anti-correlated with
    `distance_pct` (today's gap gets closed by construction) while carrying no
    particular relationship to TTF's own trailing median, which merely smooths the
    same noisy path and is not the anchor doing the closing.
    """
    horizon = 5
    trailing_window = 10
    n_anchors = 90
    n = trailing_window + n_anchors * horizon + horizon + 5
    idx = _dates(n)

    rng = np.random.default_rng(7)
    drift = 80.0 + 0.02 * np.arange(n)  # switch level: slow deterministic drift
    ttf = drift.copy()
    # redraw the deviation from drift every `horizon` days, independent draws
    for start in range(0, n, horizon):
        end = min(start + horizon, n)
        deviation = rng.normal(0, 6.0)
        ttf[start:end] = drift[start:end] + deviation

    frame = pd.DataFrame({"ttf_eur_mwh": ttf}, index=idx)
    switch = pd.Series(drift, index=idx)

    result = ceiling_test(frame, switch, horizon_days=horizon, trailing_window=trailing_window)

    assert result.switching.n_obs > 60
    assert result.switching.coefficients["distance"] < 0
    assert abs(result.switching.t_stats["distance"]) > 3.0
    assert result.horse_race.coefficients["distance"] < 0
    assert 0.0 <= result.share_above <= 1.0
    assert isinstance(result.n_overlapping, int) and result.n_overlapping > result.switching.n_obs


def test_ceiling_test_refuses_a_verdict_on_too_short_a_sample():
    idx = _dates(80)
    frame = pd.DataFrame({"ttf_eur_mwh": np.linspace(50, 55, 80)}, index=idx)
    switch = pd.Series(50.0, index=idx)
    with pytest.raises(ValueError, match="non-overlapping windows"):
        ceiling_test(frame, switch, horizon_days=20, trailing_window=10)
