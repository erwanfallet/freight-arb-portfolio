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


# ===========================================================================
# THE RIGOROUS LAYER — the four tests that carry project B's argument
# ===========================================================================
# Three of these may not be weakened by a later rework.
#
# `test_the_switching_level_is_affine_in_the_efficiency_ratio` is the algebraic fact the
# whole unifying result rests on. If it stops holding, the invariance below is a
# coincidence of this sample rather than a theorem, and the page must be rewritten.
#
# `test_stambaugh_bias_points_toward_the_finding` is the objection the page exists to
# survive. It must keep failing in the awkward direction, or the correction is decoration.
#
# `test_the_naive_anchor_cannot_be_distinguished` is the honest negative — the finding
# that the elaborate switching arithmetic is not what predicts.
from agri.data.bloomberg_loader import DEFAULT_PATH  # noqa: E402
from freight.chains.coal import (  # noqa: E402
    EF_COAL_T_PER_MWH_TH,
    EF_GAS_T_PER_MWH_TH,
    anchor_encompassing,
    asymmetry_test,
    bootstrap_null,
    efficiency_invariance,
    load_real_switching_frame,
    naive_thermal_anchor,
    phase_robustness,
    predictive_sample,
    stambaugh_diagnostics,
    subperiod_stability,
)

_real = pytest.mark.skipif(
    not DEFAULT_PATH.exists(), reason=f"Bloomberg file absent: {DEFAULT_PATH}"
)


@pytest.fixture(scope="module")
def real_frame() -> pd.DataFrame:
    return load_real_switching_frame()


@pytest.fixture(scope="module")
def real_switch(real_frame) -> pd.Series:
    return switch_ttf_eur_mwh(real_frame)


# --------------------------------------------------------------- the algebraic identity
def test_the_switching_level_is_affine_in_the_efficiency_ratio():
    """ttf* = lambda·(coal_th + EUA·EF_coal) − EUA·EF_gas, with lambda = eta_g/eta_c.

    Hand-derived from the definition in `switch_ttf_eur_mwh`:
        ttf* = (eg/ec)·C + E·(eg·EFc/ec − EFg)
             = (eg/ec)·C + (eg/ec)·E·EFc − E·EFg
             = (eg/ec)·(C + E·EFc) − E·EFg                            [factor lambda]

    The efficiencies enter ONLY through lambda, and only affinely. That is why the
    prediction cannot depend on them: a t-statistic is invariant under an affine map of
    its regressor.
    """
    frame = pd.DataFrame(
        {"coal_eur_mwh_th": [12.0, 15.0, 9.0], "eua_eur_t": [70.0, 85.0, 55.0]},
        index=_dates(3),
    )
    for coal_efficiency, gas_efficiency in ((0.36, 0.50), (0.42, 0.60), (0.38, 0.55)):
        lam = gas_efficiency / coal_efficiency
        expected = (
            lam * (frame["coal_eur_mwh_th"] + frame["eua_eur_t"] * EF_COAL_T_PER_MWH_TH)
            - frame["eua_eur_t"] * EF_GAS_T_PER_MWH_TH
        )
        actual = switch_ttf_eur_mwh(
            frame, coal_efficiency=coal_efficiency, gas_efficiency=gas_efficiency
        )
        np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), rtol=1e-12)


def test_a_t_statistic_is_invariant_under_an_affine_map_of_its_regressor():
    """The theorem the invariance result leans on, verified directly rather than cited."""
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=200), index=_dates(200))
    y = pd.Series(0.4 * x.to_numpy() + rng.normal(size=200), index=x.index)
    base = ols(y, {"x": x})
    mapped = ols(y, {"x": 3.7 * x + 12.0})
    assert mapped.t_stats["x"] == pytest.approx(base.t_stats["x"], rel=1e-9)
    assert mapped.r_squared == pytest.approx(base.r_squared, rel=1e-9)


@_real
def test_the_efficiency_grid_moves_the_level_but_not_the_prediction(real_frame):
    """THE unifying result: same grid, opposite answers to two different questions."""
    inv = efficiency_invariance(real_frame)
    assert inv.identity_holds
    # the level and the diagnosis swing hard
    assert inv.level_swing > 15.0
    assert inv.share_swing > 0.50
    # the prediction does not move at all
    assert inv.t_swing < 0.15
    assert inv.min_pairwise_corr > 0.99
    assert all(t < -2.5 for t in inv.grid["t_stat"])


# ------------------------------------------------------------------- Stambaugh bias
@_real
def test_stambaugh_bias_points_toward_the_finding(real_frame, real_switch):
    """The awkward direction, and the reason this page needs a bootstrap at all.

    TTF sits in the regressor's numerator, so innovations correlate strongly positive;
    Kendall's AR bias is negative; the product is negative — pushing OLS toward the
    negative coefficient the page reports.
    """
    diagnostics = stambaugh_diagnostics(real_frame, real_switch)
    assert diagnostics.rho > 0.6
    assert diagnostics.corr_uv > 0.5
    assert diagnostics.kendall_term < 0
    assert diagnostics.bias < 0
    assert diagnostics.bias_favours_the_finding
    assert 0.05 < diagnostics.bias_share < 0.40
    # correcting it makes the coefficient smaller in magnitude, never larger
    assert abs(diagnostics.beta_corrected) < abs(diagnostics.beta_ols)


@_real
def test_the_bootstrap_measures_the_same_bias_as_the_formula(real_frame, real_switch):
    """Two independent routes to the bias must agree, or one of them is a bug.

    The analytic route is Kendall's first-order term; the bootstrap simulates a null
    world and reads the bias off the mean. Nothing forces them to match.
    """
    analytic = stambaugh_diagnostics(real_frame, real_switch).bias
    simulated = bootstrap_null(real_frame, real_switch, n_boot=4000, seed=1).null_mean
    assert simulated == pytest.approx(analytic, abs=0.01)
    assert simulated < 0


@_real
def test_the_honest_p_value_is_far_worse_than_the_naive_one(real_frame, real_switch):
    """The page's central methodological claim: an order of magnitude of the apparent
    significance was bias, and what remains is significant but only just."""
    from scipy import stats as scipy_stats

    sample = predictive_sample(real_frame, real_switch)
    naive_t = ols(sample["forward_return"], {"distance": sample["distance"]}).t_stats[
        "distance"
    ]
    naive_p = scipy_stats.t.cdf(naive_t, df=len(sample) - 2)

    boot = bootstrap_null(real_frame, real_switch, n_boot=4000, seed=1)
    assert boot.p_value > 5 * naive_p        # materially worse
    assert boot.significant                   # but it does survive
    assert 0.005 < boot.p_value < 0.05        # and only just


# ----------------------------------------------------------------- the asymmetry
@_real
def test_the_effect_is_one_sided_as_the_mechanism_requires(real_frame, real_switch):
    """Burning coal only removes gas demand when gas is the dearer fuel. Below the
    switch nothing pushes TTF back up, so a real switching effect must be one-sided —
    which generic mean reversion cannot reproduce."""
    asymmetry = asymmetry_test(real_frame, real_switch)
    assert asymmetry.is_one_sided
    assert asymmetry.above.coefficients["distance"] < 0
    assert abs(asymmetry.above.t_stats["distance"]) > 1.98
    assert abs(asymmetry.below.t_stats["distance"]) < 1.98


# ------------------------------------------------------------- the honest negative
@_real
def test_the_naive_anchor_cannot_be_distinguished(real_frame, real_switch):
    """Raw thermal parity — no efficiencies, no carbon — predicts as well as the full
    switching level. The elaborate arithmetic is not what is doing the work, and the
    page has to say so."""
    encompassing = anchor_encompassing(real_frame, real_switch)
    assert encompassing.full_adds_nothing
    assert encompassing.increment_p > 0.10
    assert encompassing.regressor_corr > 0.80
    assert encompassing.naive_only.r_squared > 0.8 * encompassing.full_only.r_squared


@_real
def test_the_naive_anchor_contains_no_efficiency_or_carbon(real_frame):
    """Structural guarantee that the competitor really is naive — if it drifted toward
    the switching level the encompassing test above would be comparing like with like."""
    anchor = naive_thermal_anchor(real_frame)
    pd.testing.assert_series_equal(
        anchor, real_frame["coal_eur_mwh_th"].rename("naive_anchor")
    )


# ----------------------------------------------------------------- phase robustness
@_real
def test_the_result_does_not_live_in_one_phase(real_frame, real_switch):
    """`iloc[::20]` keeps one of twenty possible non-overlapping samples. Reporting the
    one that worked would be a researcher degree of freedom worth exactly one result."""
    phases = phase_robustness(real_frame, real_switch)
    assert phases.n_phases == 20
    assert phases.all_agree_on_sign
    assert phases.n_negative == 20
    assert phases.n_significant >= 18


@_real
def test_the_evidence_is_concentrated_in_the_crisis(real_frame, real_switch):
    """Reported because it changes how much weight the headline deserves: the calm
    pre-crisis years alone do not carry the result."""
    table = subperiod_stability(real_frame, real_switch)
    assert abs(table.loc["2018-2020 pre-crisis", "t_stat"]) < 1.98
    assert abs(table.loc["2021-2022 crisis", "t_stat"]) > 1.98


# ----------------------------------------------------------------------- guardrails
@_real
def test_an_out_of_range_phase_is_rejected(real_frame, real_switch):
    with pytest.raises(ValueError, match="phase must be in"):
        predictive_sample(real_frame, real_switch, phase=20)


def test_asymmetry_refuses_a_one_sided_sample():
    """A sample sitting entirely above the switch cannot support the comparison."""
    frame = pd.DataFrame(
        {
            "coal_eur_mwh_th": np.full(400, 10.0),
            "eua_eur_t": np.full(400, 60.0),
            "ttf_eur_mwh": np.full(400, 500.0),   # far above any switching level
        },
        index=_dates(400),
    )
    with pytest.raises(ValueError, match="too few windows"):
        asymmetry_test(frame, switch_ttf_eur_mwh(frame))


# ===========================================================================
# THE TRADE — the carbon hedge inside the switching spread
# ===========================================================================
# `test_the_two_betas_always_have_opposite_signs` is the one that may never be weakened:
# the entire trade rests on the cross term being negative, and if the signs ever agreed
# a positive correlation would AMPLIFY spread volatility instead of damping it.
from freight.chains.coal import (  # noqa: E402
    COAL_EFFICIENCY_RANGE,
    GAS_EFFICIENCY_RANGE,
    dampening_attribution,
    natural_hedge,
    spread_betas,
    switching_depth_profile,
    transmission_test,
)


def test_the_two_betas_always_have_opposite_signs():
    """Structural, across the whole plausible efficiency grid.

        b_ttf = -1/eta_gas                       < 0 always
        b_eua = EF_coal/eta_coal - EF_gas/eta_gas

    b_eua stays positive because coal emits far more per MWh of electricity than gas even
    at coal's best efficiency and gas's worst: 0.34/0.42 = 0.810 > 0.20/0.50 = 0.400.
    """
    for coal_efficiency in (COAL_EFFICIENCY_RANGE[0], 0.38, COAL_EFFICIENCY_RANGE[1]):
        for gas_efficiency in (GAS_EFFICIENCY_RANGE[0], 0.55, GAS_EFFICIENCY_RANGE[1]):
            b_ttf, b_eua = spread_betas(
                coal_efficiency=coal_efficiency, gas_efficiency=gas_efficiency
            )
            assert b_ttf < 0 < b_eua, (coal_efficiency, gas_efficiency)
            assert b_ttf * b_eua < 0


def test_the_betas_are_the_hand_derived_values():
    """At the default pair, differentiating the generation-cost identity by hand:
    b_ttf = -1/0.55 = -1.818 and b_eua = 0.34/0.38 - 0.20/0.55 = 0.895 - 0.364 = 0.531."""
    b_ttf, b_eua = spread_betas(coal_efficiency=0.38, gas_efficiency=0.55)
    assert b_ttf == pytest.approx(-1.8182, abs=1e-3)
    assert b_eua == pytest.approx(+0.5311, abs=1e-3)


def test_spread_betas_reject_an_implausible_efficiency():
    with pytest.raises(ValueError, match="outside the plausible range"):
        spread_betas(gas_efficiency=0.95)


def test_a_positive_correlation_reduces_spread_volatility():
    """The trade's whole mechanism, on synthetic data where the correlation is imposed.

    Two series with identical marginal volatilities, differing only in correlation: the
    positively-correlated pair must produce the LOWER spread volatility.
    """
    rng = np.random.default_rng(0)
    n = 1500
    dates = _dates(n)
    base = rng.normal(size=n)

    def frame_with(rho: float) -> pd.DataFrame:
        other = rho * base + np.sqrt(1 - rho**2) * rng.normal(size=n)
        return pd.DataFrame(
            {
                "ttf_eur_mwh": 40 + np.cumsum(base) * 0.1,
                "eua_eur_t": 70 + np.cumsum(other) * 0.1,
            },
            index=dates,
        )

    hedged = natural_hedge(frame_with(0.8), min_obs=100)
    independent = natural_hedge(frame_with(0.0), min_obs=100)
    assert hedged.years[0].dampening < independent.years[0].dampening
    assert hedged.years[0].dampening < 0          # correlation removed volatility
    assert hedged.years[0].vol_actual < hedged.years[0].vol_if_independent


@_real
def test_the_hedge_worked_for_years_then_stopped(real_frame):
    """The page's central empirical claim."""
    hedge = natural_hedge(real_frame)
    assert len(hedge.hedged_years) >= 6
    assert hedge.typical_dampening < -0.05        # a median of at least 5% removed
    assert min(y.dampening for y in hedge.hedged_years) < -0.20   # up to ~29% in 2024
    # the most recent year lost it
    assert hedge.latest.rho < 0.1
    assert hedge.latest.dampening > -0.05
    assert "the hedge is gone" in hedge.headline


@_real
def test_the_correlation_is_the_smaller_term_and_the_page_says_so(real_frame):
    """The honesty check on the attribution: gas volatility dominates, and the claim is
    only that the correlation term is the one nobody re-marks — not that it is the
    largest."""
    attribution = dampening_attribution(real_frame, year_from=2025, year_to=2026)
    assert attribution.vol_to > attribution.vol_from
    assert 0.0 < attribution.correlation_share < 0.5      # smaller than the vol term
    assert attribution.volatility_part > attribution.correlation_part
    assert 0.05 < attribution.option_value_uplift < 0.25
    assert (
        attribution.volatility_part + attribution.correlation_part
        == pytest.approx(attribution.total_change)
    )


@_real
def test_saturation_is_ruled_out_by_two_years_with_the_same_depth(real_frame):
    """2018 and 2026 sit at nearly identical depth above the switching level and have
    opposite correlations, so 'the coal fleet ran out of room' cannot be the explanation.
    2022 is the only genuinely saturated year and is the sample's other negative one."""
    depth = switching_depth_profile(real_frame)
    assert depth.loc[2018, "share_deep"] < 0.05
    assert depth.loc[2026, "share_deep"] < 0.05
    assert abs(depth.loc[2018, "median_distance"] - depth.loc[2026, "median_distance"]) < 0.05
    assert depth.loc[2018, "rho"] > 0.25
    assert depth.loc[2026, "rho"] < 0.10
    # 2022 is the genuinely saturated year
    assert depth.loc[2022, "share_deep"] > 0.40
    # and a late year in the coal decline has the strongest correlation, killing the
    # 'structural fleet decline' story too
    assert depth.loc[2024, "rho"] > 0.60


@_real
def test_transmission_breaks_only_in_the_two_anomalous_years(real_frame):
    """Non-parametric version: on the largest gas shocks, does carbon respond at all?"""
    transmission = transmission_test(real_frame)
    table = transmission.table
    assert table.loc[2026, "same_sign"] <= 4
    assert table.loc[2022, "same_sign"] <= 4
    normal = table.drop(index=[2022, 2026])
    assert (normal["same_sign"] >= 7).all()
    assert "did not take it" in transmission.headline


def test_transmission_test_needs_enough_observations():
    tiny = pd.DataFrame(
        {"ttf_eur_mwh": [40.0, 41.0, 42.0], "eua_eur_t": [70.0, 71.0, 72.0]},
        index=_dates(3),
    )
    with pytest.raises(ValueError, match="no year has enough observations"):
        transmission_test(tiny)
