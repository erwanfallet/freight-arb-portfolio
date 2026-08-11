"""Golden tests for the six Tier 2 engines.

Posture reminder, checked at the end of the file: every T2 engine rests on an
**inferred** tension, not a citation. The docstrings must say "it seems to me," never
"I read that" — a test checks this, because it's the line that gets torn apart in one
reply if it slips.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from agri.chains import crush_tracking as ct
from agri.chains import oil_substitution as os_
from agri.chains import plant_option as po
from agri.chains import white_premium as wp
from agri.fixtures import tier2


# ===========================================================================
# T2-1 — basis against flat price
# ===========================================================================
def test_plant_crush_hand_computed():
    """Bean 13.00 $/bu, meal 400 $/short ton, oil 55 c/lb,
    yields 43.5 and 10.8 lb/bu, opex 0.42:

        meal : 43.5/2000 x 400 = 8.70
        oil  : 10.8 x 0.55     = 5.94
        crush: 8.70 + 5.94 - 13.00 - 0.42 = 1.22 $/bu
    """
    index = pd.to_datetime(["2024-01-01"])
    out = ct.plant_crush_usd_bu(
        pd.Series([13.00], index=index),
        pd.Series([400.0], index=index),
        pd.Series([55.0], index=index),
    )
    assert out.iloc[0] == pytest.approx(1.22, abs=1e-10)


def test_board_crush_beats_plant_crush_on_cbot_yields():
    """At the same prices, the board uses 44/11 lb and the plant 43.5/10.8: the gap
    is the minimal tracking error, the one that exists even with zero basis.

        board = 0.022 x 400 + 0.11 x 55 - 13.00 = 1.85
        plant = 1.22 (above)
        gap   = 0.63 $/bu
    """
    from agri.core.units import board_crush_usd_bu

    board = board_crush_usd_bu(13.00, 400.0, 55.0)
    assert board == pytest.approx(1.85, abs=1e-10)
    assert board - 1.22 == pytest.approx(0.63, abs=1e-10)


def test_impossible_yields_are_rejected():
    index = pd.to_datetime(["2024-01-01"])
    args = (
        pd.Series([13.0], index=index),
        pd.Series([400.0], index=index),
        pd.Series([55.0], index=index),
    )
    with pytest.raises(ct.CrushError, match="60 lb"):
        ct.plant_crush_usd_bu(*args, yield_meal_lb_bu=55.0, yield_oil_lb_bu=12.0)


def test_negative_yield_is_rejected():
    index = pd.to_datetime(["2024-01-01"])
    with pytest.raises(ct.CrushError, match="physical range"):
        ct.plant_crush_usd_bu(
            pd.Series([13.0], index=index),
            pd.Series([400.0], index=index),
            pd.Series([55.0], index=index),
            yield_meal_lb_bu=-1.0,
        )


@pytest.fixture(scope="module")
def crush_frame() -> pd.DataFrame:
    return ct.build_tracking(**tier2.crush_tracking())


def test_tracking_error_is_the_difference(crush_frame):
    assert (
        crush_frame["tracking_error"] == crush_frame["board_crush"] - crush_frame["plant_crush"]
    ).all()


def test_optimal_hedge_ratio_is_near_one_overall(crush_frame):
    out = ct.optimal_hedge_ratio(crush_frame)
    assert 0.85 < out.h_star < 1.15
    assert out.variance_reduction_at_h_star >= out.variance_reduction_at_one


def test_rolling_hedge_ratio_moves_away_from_one(crush_frame):
    """The page's deliverable: `h*` is not constant, and it drifts from 1 in episodes."""
    rolling = ct.rolling_hedge_ratio(crush_frame, window=120)
    assert rolling["h_star"].min() < 0.92
    assert rolling["h_star"].max() > 1.08
    assert rolling.attrs["n_eff"] == pytest.approx(len(rolling) / 120)


def test_decoupling_episodes_are_found(crush_frame):
    episodes = ct.decoupling_episodes(crush_frame, threshold_usd_bu=0.35)
    assert len(episodes) > 0
    assert (episodes["n_obs"] >= 5).all()


def test_decomposition_is_exact(crush_frame):
    """The identity must close to floating-point precision, on every date.

    This is what distinguishes a decomposition from a regression: nothing is
    estimated, so nothing can be biased.
    """
    components = ct.decompose_tracking_error(crush_frame)
    assert np.allclose(components["total"], crush_frame["tracking_error"], atol=1e-10)


def test_the_yield_mismatch_term_is_not_a_basis(crush_frame):
    """The term that "the decoupling comes from the basis" completely misses.

    The gap between the CBOT yields (44/11 lb) and the real yields (43.5/10.8 lb)
    creates a term proportional to the board's **level**, which exists even when
    every basis is zero. On this dataset, its dispersion exceeds the bean basis's.
    """
    components = ct.decompose_tracking_error(crush_frame)
    assert components["oil_yield"].std() > components["bean_basis"].std()
    assert components["oil_yield"].abs().mean() > 0


def test_meal_basis_dominates_the_variability(crush_frame):
    """The mechanism plant people describe: it's the meal basis that breaks the
    hedge, not the bean."""
    contributions = ct.basis_contributions(crush_frame)
    assert contributions.loc[0, "term"] == "meal_basis"
    assert contributions.loc[0, "share"] > 0.5
    assert contributions["share"].sum() == pytest.approx(1.0)


def test_opex_moves_the_level_but_not_the_variability(crush_frame):
    """A distinction that matters for a hedge: a fixed cost shifts things, it
    doesn't shake them."""
    contributions = ct.basis_contributions(crush_frame).set_index("term")
    assert contributions.loc["opex", "std_usd_bu"] == pytest.approx(0.0)
    assert contributions.loc["opex", "mean_usd_bu"] == pytest.approx(0.42)


def test_a_regression_on_the_basis_alone_is_biased(crush_frame):
    """Why the exact decomposition replaces the regression.

    Regressing the tracking error on the three basis terms omits the two yield
    terms, which are as dispersed as the bean basis. The bean coefficient — whose
    structural value is exactly +1 — comes out biased at ~0.99.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        regression = ct.explain_tracking_error(crush_frame)
    assert regression.params["bean_basis"] != pytest.approx(1.0, abs=1e-3)
    assert regression.params["bean_basis"] == pytest.approx(1.0, abs=0.05)
    # the two high-dispersion basis terms, by contrast, are well identified
    assert regression.params["meal_basis"] == pytest.approx(-0.02175, abs=1e-3)
    assert regression.params["oil_basis"] == pytest.approx(-0.108, abs=1e-3)


# ===========================================================================
# T2-4 — white premium
# ===========================================================================
def test_white_premium_hand_computed():
    """No.5 at 520 $/t, No.11 at 20 c/lb, pol_adjust 1.07:
        20 x 22.0462262 = 440.924524 $/t
        x 1.07          = 471.789241 $/t on a white basis
        premium         = 520 - 471.789241 = 48.210759 $/t
    """
    index = pd.to_datetime(["2024-01-01"])
    out = wp.white_premium_usd_t(
        pd.Series([520.0], index=index), pd.Series([20.0], index=index)
    )
    assert out.iloc[0] == pytest.approx(48.210759, abs=1e-6)


def test_fair_value_refining_hand_computed():
    """No.11 at 20 c/lb = 440.924524 $/t of raw sugar:
        energy           28.000000
        2% loss           8.818490
        labour           12.000000
        freight          18.000000
        financing        440.924524 x 0.055 x 45/360 = 3.031356
        total            69.849846
    """
    index = pd.to_datetime(["2024-01-01"])
    costs = wp.fair_value_refining_usd_t(pd.Series([20.0], index=index))
    assert costs["yield_loss"].iloc[0] == pytest.approx(8.818490, abs=1e-6)
    assert costs["financing"].iloc[0] == pytest.approx(3.031356, abs=1e-6)
    assert costs["total"].iloc[0] == pytest.approx(69.849846, abs=1e-6)


def test_richness_hand_computed():
    # 48.210759 - 69.849846 = -21.639087 -> CHEAP zone
    index = pd.to_datetime(["2024-01-01"])
    frame = wp.build_richness(
        pd.Series([520.0], index=index), pd.Series([20.0], index=index)
    )
    assert frame["richness"].iloc[0] == pytest.approx(-21.639087, abs=1e-6)
    assert frame["zone"].iloc[0] == "CHEAP"


def test_pol_adjust_out_of_range_is_rejected():
    index = pd.to_datetime(["2024-01-01"])
    with pytest.raises(wp.WhitePremiumError, match="pol_adjust"):
        wp.white_premium_usd_t(
            pd.Series([520.0], index=index), pd.Series([20.0], index=index), pol_adjust=1.4
        )


def test_pol_adjust_sensitivity_is_material():
    """W-H1: between 1.06 and 1.08 the share of time in the RICH zone moves enough
    that the parameter can't be fixed. That's what justifies the slider."""
    data = tier2.white_premium()
    table = wp.pol_adjust_sensitivity(
        data["no5_usd_t"], data["no11_cents_lb"], values=np.array([1.06, 1.08])
    )
    low, high = table["share_rich"].tolist()
    assert abs(high - low) > 0.05
    # a higher pol_adjust makes the raw leg more expensive, so it compresses the premium
    assert table["mean_white_premium"].is_monotonic_decreasing


def test_richness_summary_produces_the_headline():
    data = tier2.white_premium()
    frame = wp.build_richness(**data)
    summary = wp.summarise_richness(frame)
    assert 0.0 < summary.share_rich < 1.0
    assert len(summary.rich_episodes) > 0
    assert len(summary.cheap_episodes) > 0
    assert "physical availability" in summary.headline


# ===========================================================================
# T2-4 on real data (Bloomberg export) — No.11/No.5 + Henry Hub
# ===========================================================================
from agri.data.bloomberg_loader import DEFAULT_PATH as _BBG_PATH  # noqa: E402

pytestmark_real_t2_4 = pytest.mark.skipif(
    not _BBG_PATH.exists(), reason=f"Bloomberg file absent: {_BBG_PATH}"
)


@pytest.mark.skipif(not _BBG_PATH.exists(), reason=f"Bloomberg file absent: {_BBG_PATH}")
def test_real_richness_on_2026_08_07_hand_computed():
    """No.11 = 16.45 c/lb, No.5 = 503.4 USD/t, Henry Hub = 2.662 USD/mmBtu:
        raw_usd_t     = 16.45 x 22.0462262           = 362.660421
        white_premium = 503.4 - 362.660421 x 1.07     = 115.353350
        energy        = 2.662 x 8.0                   = 21.296
        yield_loss    = 0.02 x 362.660421              = 7.253208
        financing     = 362.660421 x 0.055 x 45/360    = 2.493290
        fv_refining   = 21.296 + 7.253208 + 12 + 18 + 2.493290 = 61.042499
        richness      = 115.353350 - 61.042499         = 54.310851
    """
    frame = wp.load_real_richness_frame()
    row = frame.loc["2026-08-07"]
    assert row["no11"] == pytest.approx(16.45)
    assert row["no5"] == pytest.approx(503.4)
    assert row["white_premium"] == pytest.approx(115.353350, abs=1e-4)
    assert row["fv_refining"] == pytest.approx(61.042499, abs=1e-4)
    assert row["richness"] == pytest.approx(54.310851, abs=1e-4)
    assert row["zone"] == "RICH"


@pytest.mark.skipif(not _BBG_PATH.exists(), reason=f"Bloomberg file absent: {_BBG_PATH}")
def test_real_energy_cost_tracks_henry_hub_not_a_constant():
    """The energy cost must vary over time (Henry Hub proxy), unlike the
    DEFAULT_ENERGY_USD_T flat rate it replaces — otherwise the "real data" upgrade
    would be cosmetic."""
    frame = wp.load_real_richness_frame()
    from agri.chains.white_premium import fair_value_refining_usd_t
    from agri.data.bloomberg_loader import load as load_bloomberg

    energy_leg = fair_value_refining_usd_t(
        frame["no11"], energy_usd_t=(load_bloomberg("henry_hub") * 8.0).reindex(frame.index)
    )["energy"]
    assert energy_leg.std() > 0.5
    assert energy_leg.nunique() > 100


@pytest.mark.skipif(not _BBG_PATH.exists(), reason=f"Bloomberg file absent: {_BBG_PATH}")
def test_real_richness_summary_and_headline_run_on_real_data():
    frame = wp.load_real_richness_frame()
    summary = wp.summarise_richness(frame)
    assert 0.0 < summary.share_rich < 1.0
    assert "physical availability" in summary.headline
    assert frame.attrs["energy_source"] == "henry_hub_real"


# ===========================================================================
# T2-5 — the plant as an option
# ===========================================================================
@pytest.fixture(scope="module")
def ou_params() -> po.OUParams:
    return po.calibrate_ou(tier2.plant_margin())


def test_ou_calibration_recovers_kappa_and_sigma(ou_params):
    """theta is deliberately tested with a wider tolerance: it's the slowest
    parameter to estimate. With kappa = 0.035 the half-life is 20 periods, so 1,600
    observations are only worth about thirty independent ones for the long-run mean."""
    assert ou_params.kappa == pytest.approx(tier2.TRUE_OU_KAPPA, abs=0.008)
    assert ou_params.sigma == pytest.approx(tier2.TRUE_OU_SIGMA, abs=0.4)
    assert ou_params.theta == pytest.approx(tier2.TRUE_OU_THETA, abs=3.0)


def test_half_life_is_consistent_with_kappa(ou_params):
    assert ou_params.half_life == pytest.approx(np.log(2) / ou_params.kappa)


def test_calibration_refuses_a_random_walk():
    """O-H1: calibrating an OU process on a random walk would give a near-zero
    kappa and an absurd option value, without ever crashing. The refusal is explicit."""
    rng = np.random.default_rng(0)
    walk = pd.Series(
        np.cumsum(rng.normal(size=400)),
        index=pd.date_range("2020-01-01", periods=400, freq="B"),
    )
    with pytest.raises(po.PlantOptionError, match="stationary"):
        po.calibrate_ou(walk)


def test_calibration_can_be_forced_with_a_warning_path():
    rng = np.random.default_rng(0)
    walk = pd.Series(
        np.cumsum(rng.normal(size=400)),
        index=pd.date_range("2020-01-01", periods=400, freq="B"),
    )
    forced = po.calibrate_ou(walk, strict=False)
    assert forced.stationarity.verdict != "stationary"


@pytest.fixture(scope="module")
def band(ou_params) -> po.HysteresisBand:
    return po.solve_hysteresis(
        ou_params, cost_restart=120.0, cost_shutdown=60.0, cost_idle=0.5
    )


def test_value_iteration_converges(band):
    assert band.converged
    assert band.n_iterations < 5_000


def test_the_frontier_is_a_band_not_a_threshold(band):
    """THE page's result: M_off < 0 < M_on, with strictly positive hysteresis.

    A "margin < 0" rule assumes M_off = M_on = 0. The optimal frontier never is one,
    the moment stopping and restarting cost anything at all.
    """
    assert band.m_off < 0.0 < band.m_on
    assert band.width > 0.0
    assert "hysteresis" in band.headline


def test_higher_switching_costs_widen_the_band(ou_params):
    cheap = po.solve_hysteresis(ou_params, cost_restart=20.0, cost_shutdown=10.0, cost_idle=0.5)
    expensive = po.solve_hysteresis(
        ou_params, cost_restart=400.0, cost_shutdown=200.0, cost_idle=0.5
    )
    assert expensive.width > cheap.width


def test_plant_value_increases_with_volatility(ou_params):
    """The counter-intuitive demonstration, and it's what makes the page.

    At an equal average margin, a plant whose margin is **more volatile** is worth
    more, because the option to stop truncates the lower tail. It's a number placed
    on a debate that's usually held in slogans.
    """
    table = po.volatility_sensitivity(
        ou_params, cost_restart=120.0, cost_shutdown=60.0, cost_idle=0.5
    )
    assert table["value_at_theta"].is_monotonic_increasing
    assert table["band_width"].is_monotonic_increasing
    assert table["value_at_theta"].iloc[-1] > table["value_at_theta"].iloc[0]


def test_heuristic_rule_shuts_down_more_often_than_the_frontier(band):
    margin = tier2.plant_margin()
    comparison = po.compare_to_heuristic(margin, band, threshold=0.0, consecutive_periods=4)
    assert comparison.n_shutdowns_heuristic >= comparison.n_shutdowns_optimal
    assert "shutdowns" in comparison.headline


def test_negative_switching_costs_are_rejected(ou_params):
    with pytest.raises(po.PlantOptionError, match="costs"):
        po.solve_hysteresis(ou_params, cost_restart=-1.0, cost_shutdown=60.0, cost_idle=0.5)


# ===========================================================================
# T2-5 on real data — CBOT crush margin, entirely real
# ===========================================================================
@pytest.mark.skipif(not _BBG_PATH.exists(), reason=f"Bloomberg file absent: {_BBG_PATH}")
def test_real_crush_margin_on_2026_08_07_hand_computed():
    """Soybean 11.565 USD/bu, meal 308.1 USD/short ton, oil 68.16 c/lb:
        meal : 44/2000 x 308.1 = 6.7782
        oil  : 11 x 0.6816    = 7.4976
        crush: 6.7782 + 7.4976 - 11.565 = 2.7108 USD/bu
    """
    margin = po.real_board_crush_margin()
    assert margin.loc["2026-08-07"] == pytest.approx(2.7108, abs=1e-4)


@pytest.mark.skipif(not _BBG_PATH.exists(), reason=f"Bloomberg file absent: {_BBG_PATH}")
def test_real_margin_uses_only_real_legs_no_parameters():
    """Unlike T1-2 (roll omitted) or T2-4 (labour/freight parameterised), all three
    legs here are entirely real — no constant term injected."""
    margin = po.real_board_crush_margin(start="2020-01-01")
    assert margin.std() > 0.1
    assert margin.nunique() > 500


@pytest.mark.skipif(not _BBG_PATH.exists(), reason=f"Bloomberg file absent: {_BBG_PATH}")
def test_real_margin_fails_stationarity_and_that_is_the_finding():
    """Result verified in session: no window tested (the full 36 years, nor the
    sub-periods since 2005) passes the joint ADF+KPSS verdict. This test locks in
    that the diagnostic states this clearly rather than masking the failure."""
    margin = po.real_board_crush_margin()
    diagnostic = po.diagnose_real_margin_stationarity(margin)
    assert diagnostic.stationarity.verdict != "stationary"
    assert "does not behave like a homogeneous OU" in diagnostic.headline
    assert diagnostic.n_obs == len(margin)


@pytest.mark.skipif(not _BBG_PATH.exists(), reason=f"Bloomberg file absent: {_BBG_PATH}")
def test_indicative_calibration_still_produces_a_usable_band():
    """The indicative calibration (strict=False) must remain usable — a valid band,
    value-iteration convergence — even when the stationarity verdict is
    unfavourable. That's what makes it displayable as an illustrative result."""
    margin = po.real_board_crush_margin(start="2018-01-01")
    ou = po.calibrate_real_ou_indicative(margin)
    assert ou.stationarity.verdict != "stationary"
    band = po.solve_hysteresis(ou, cost_restart=0.30, cost_shutdown=0.15, cost_idle=0.02)
    assert band.converged
    assert band.m_off < 0 < band.m_on


# ===========================================================================
# T2-6 — inter-oil substitution
# ===========================================================================
@pytest.fixture(scope="module")
def spreads() -> pd.DataFrame:
    return os_.build_spreads(tier2.oil_prices())


def test_spreads_are_built_once_per_pair(spreads):
    assert set(spreads.columns) == {
        "canola_minus_palm",
        "canola_minus_soy",
        "palm_minus_soy",
    }


def test_half_life_formula():
    # b = -0.10 -> half-life = -ln(2)/ln(0.90) = 6.579
    assert -np.log(2) / np.log(0.90) == pytest.approx(6.5788, abs=1e-3)


def test_substitution_bound_finds_the_regime_split(spreads):
    """The dataset imposes a threshold AR process: slow below a 60 $/t gap, fast
    beyond it.

    This doesn't test exact recovery of the half-lives — a threshold model
    estimated by splitting is an approximation (S-H4) — but the **separation**,
    which is what the page claims.
    """
    bound = os_.substitution_bound(
        spreads["palm_minus_soy"],
        pair="palm-soy",
        threshold_usd_t=tier2.SUBSTITUTION_THRESHOLD_USD_T,
    )
    assert bound.narrow.is_mean_reverting
    assert bound.wide.is_mean_reverting
    assert bound.wide.half_life_days < bound.narrow.half_life_days
    assert bound.substitution_kicks_in
    assert "substitution bound" in bound.headline


def test_lags_are_computed_before_the_regime_filter(spreads):
    """The bug this test locks in against.

    Filtering the subsample **before** computing `.diff()` and `.shift()` computes
    gaps between observations that aren't adjacent in time, which manufactures fake
    mean reversion: two points three weeks apart appear to have converged in a
    single step. On this dataset, the error brought a 173-day half-life down to 10.
    """
    series = spreads["palm_minus_soy"]
    mask = (series.shift(1).abs() < 20.0)

    correct = os_.estimate_half_life(series, mask=mask)
    naive = os_.estimate_half_life(series[mask])       # the mistake: filter first
    assert correct.beta != pytest.approx(naive.beta, abs=1e-6)
    assert abs(naive.beta) > abs(correct.beta)          # the mistake exaggerates reversion


def test_screen_flags_non_stationary_pairs(spreads):
    """The check that prevents reading a half-life off a unit root.

    The dataset contains one pair built as a genuine relationship (palm-soy) and two
    pairs that aren't. The screen must tell them apart, otherwise the page would
    publish three "substitution bounds," two of which don't exist.
    """
    table = os_.screen_all_pairs(spreads).set_index("pair")
    assert table.loc["palm_minus_soy", "stationarity"] == "stationary"
    assert table.loc["palm_minus_soy", "substitution_kicks_in"]
    assert table.loc["canola_minus_soy", "stationarity"] != "stationary"
    assert "do not read" in table.loc["canola_minus_soy", "note"]


def test_a_pure_random_walk_has_no_half_life():
    rng = np.random.default_rng(1)
    walk = pd.Series(
        np.cumsum(rng.normal(size=400)),
        index=pd.date_range("2020-01-01", periods=400, freq="B"),
    )
    out = os_.estimate_half_life(walk)
    assert not out.is_mean_reverting
    assert "no detectable mean reversion" in out.summary


def test_too_few_oils_is_rejected():
    index = pd.date_range("2024-01-01", periods=100, freq="B")
    with pytest.raises(os_.SubstitutionError, match="two oils"):
        os_.build_spreads({"palm": pd.Series(900.0, index=index)})


# ===========================================================================
# Tier 2 posture — the absolute rule
# ===========================================================================
# `plant_option` is deliberately outside this list since its rework: the page no
# longer rests on an inferred market tension ("it seems to me desks argue about this")
# but on the critique of a rule that is **actually used** — the
# `consecutive_below(margin, 0, N=4)` used on the zinc and lithium pages — whose
# implication it computes. This isn't a weakening of the guardrail: it's a different
# epistemic status, covered by its own test right below.
#
# `basis_flat` (T2-1) and `grain_carry` (T2-2) left the portfolio on 10/08/2026: the
# Bloomberg export only contains generic front months, so neither a cash series nor a
# calendar spread is available. Their deliverable wasn't computable and would have
# stayed synthetic. Code kept in `_archive/`, not deleted.
INFERRED_TENSION_MODULES = [ct, wp, os_]


@pytest.mark.parametrize("module", INFERRED_TENSION_MODULES)
def test_tier2_modules_frame_the_tension_as_inferred(module):
    """"It seems to me," never "I read that."

    Presenting an inferred tension as a citation gets torn apart in one line,
    because it's false. This test keeps the boundary at the code level rather than
    at the discipline level.
    """
    import re

    doc = module.__doc__ or ""
    lowered = doc.lower()
    assert "INFERRED" in doc, f"{module.__name__} does not mark its tension as inferred"
    assert "it seems to me" in lowered, f"{module.__name__} does not use the cautious phrasing"

    # "I read that" is acceptable only when preceded by "never" — i.e. cited as the
    # phrasing to NOT use. Any other occurrence is a claim presented as sourced when
    # it is not.
    for match in re.finditer(r"i read that", lowered):
        preceding = lowered[max(0, match.start() - 30) : match.start()]
        assert "never" in preceding, (
            f"{module.__name__} uses \"I read that\" outside the disclaimer: "
            "an inferred tension presented as a citation falls apart in one line"
        )


def test_plant_option_rests_on_a_rule_it_can_point_at_not_an_inferred_tension():
    """The guardrail that replaces the one above, for the one reworked module.

    The page claims no market dispute — it targets a rule that can be pointed at,
    and it must say what it does with it: name the rule, announce that it computes
    its implication, and expose the counterfactual that prevents a refinement from
    being presented as the main subject.
    """
    doc = po.__doc__ or ""
    lowered = doc.lower()

    assert "consecutive_below" in lowered, "the targeted rule is not named"
    assert "zinc" in lowered and "lithium" in lowered, "the targeted pages are not named"
    assert "inverting the question" in lowered
    assert "counterfactual" in lowered, "the never-stop counterfactual is not announced"

    # Same rule as for the inferred-tension modules: no claim presented as sourced
    # that is not.
    import re

    for match in re.finditer(r"i read that", lowered):
        preceding = lowered[max(0, match.start() - 30) : match.start()]
        assert "never" in preceding
