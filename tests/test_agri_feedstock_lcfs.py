"""Golden tests T3-1 — the LCFS threshold.

Model constants, used in every hand-computed calculation below:
    CI soyoil = 27, CI UCO = 15            -> CI differential = 12 gCO2e/MJ
    45Z credit on soyoil = (50 - 27)/50    = 0.46 $/gal
    yield                                  = 7.6 lb/gal
    LCFS conversion = 134.47e-6            (134.47 MJ/gal x 1e-6 t/g)

    threshold denominator = 12 x 1.0 x 134.47e-6 = 1.61364e-3
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agri.chains.feedstock_lcfs import (
    SOYOIL_DOMESTIC,
    UCO_IMPORTED,
    Feedstock,
    FeedstockError,
    NoBreakevenInRange,
    calibration_gap_45z,
    chow_break_test,
    crush_balance,
    feedstock_breakeven_usd_lb,
    gate_value,
    lcfs_breakeven,
    lcfs_breakeven_numeric,
    lcfs_value_usd_gal,
    rolling_energy_beta,
    winner_grid,
)
from agri.fixtures.feedstock_lcfs import (
    BETA_AFTER_BREAK,
    BETA_BEFORE_BREAK,
    RVO_BREAK_DATE,
    build,
)

LCFS_DENOMINATOR = 12 * 1.0 * 134.47e-6      # 1.61364e-3


@pytest.fixture(scope="module")
def series() -> dict[str, pd.Series]:
    return build()


# ===========================================================================
# The 45Z credit (L-H4)
# ===========================================================================
def test_45z_credit_on_soyoil():
    # 1.00 x (50 - 27)/50 = 0.46 $/gal
    assert SOYOIL_DOMESTIC.credit_45z_usd_gal() == pytest.approx(0.46, abs=1e-12)


def test_45z_credit_is_zero_for_imported_feedstock_however_clean():
    """The central regulatory point: 45Z is an origin rule, not a CI rule.

    Imported UCO has a CI of 15, far better than soyoil's 27, and yet earns zero.
    That's exactly what the LCFS has to offset — so it's the page's whole subject.
    """
    assert UCO_IMPORTED.carbon_intensity < SOYOIL_DOMESTIC.carbon_intensity
    assert UCO_IMPORTED.credit_45z_usd_gal() == 0.0


def test_45z_credit_floors_at_zero_for_dirty_feedstock():
    dirty = Feedstock("carbon-heavy pathway", 65.0, north_american=True)
    assert dirty.credit_45z_usd_gal() == 0.0


def test_calibration_gap_is_shown_not_absorbed():
    """L-H4: the 3 c/gal gap against the published value is shown, not quietly corrected."""
    gap = calibration_gap_45z()
    assert gap["modelled_usd_gal"] == pytest.approx(0.46)
    assert gap["published_usd_gal"] == pytest.approx(0.49)
    assert gap["gap_usd_gal"] == pytest.approx(0.03, abs=1e-9)
    assert gap["gap_pct"] == pytest.approx(0.061224, abs=1e-5)


# ===========================================================================
# The LCFS leg — the unit trap
# ===========================================================================
def test_lcfs_leg_hand_computed():
    # 200 $/t x (95 - 27) x 1.0 x 134.47e-6 = 200 x 68 x 134.47e-6 = 1.828792 $/gal
    assert lcfs_value_usd_gal(200.0, 27.0, ci_std=95.0) == pytest.approx(1.828792, abs=1e-6)


def test_cleaner_feedstock_earns_more_lcfs():
    clean = lcfs_value_usd_gal(200.0, 15.0, ci_std=95.0)
    dirty = lcfs_value_usd_gal(200.0, 27.0, ci_std=95.0)
    # the gap is worth 200 x 12 x 134.47e-6 = 0.322728 $/gal
    assert clean - dirty == pytest.approx(0.322728, abs=1e-6)


def test_gate_value_stack_sums_to_total():
    value = gate_value(
        SOYOIL_DOMESTIC, ulsd_usd_gal=2.55, rin_d4_usd=0.62, lcfs_usd_t=200.0
    )
    # diesel 2.55 + RIN 0.62 x 1.7 = 1.054 + LCFS 1.828792 + 45Z 0.46 = 5.892792
    assert value.total_usd_gal == pytest.approx(5.892792, abs=1e-6)
    assert sum(value.stack.values()) == pytest.approx(value.total_usd_gal)


def test_feedstock_breakeven_hand_computed():
    # (5.892792 - 0.55 opex - 0.25 roi) / 7.6 = 5.092792 / 7.6 = 0.670104 $/lb
    out = feedstock_breakeven_usd_lb(
        SOYOIL_DOMESTIC, ulsd_usd_gal=2.55, rin_d4_usd=0.62, lcfs_usd_t=200.0
    )
    assert out == pytest.approx(0.670104, abs=1e-6)


def test_zero_yield_is_rejected():
    with pytest.raises(FeedstockError, match="yield_lb_gal"):
        feedstock_breakeven_usd_lb(
            SOYOIL_DOMESTIC, ulsd_usd_gal=2.55, rin_d4_usd=0.62, lcfs_usd_t=200.0, yield_lb_gal=0.0
        )


# ===========================================================================
# THE TIPPING POINT — the deliverable
# ===========================================================================
def test_lcfs_threshold_at_price_parity():
    """At equal feedstock prices, the LCFS alone must offset the entire 45Z.

        LCFS* = 0.46 / 1.61364e-3 = 285.07 $/t CO2e

    That's the disagreement's bare number: with no price advantage, an LCFS credit
    of 285 $/t is needed for imported UCO to match domestic soyoil.
    """
    out = lcfs_breakeven(
        price_domestic_usd_lb=0.50,
        price_imported_usd_lb=0.50,
        lcfs_current_usd_t=95.0,
    )
    assert out.lcfs_star_usd_t == pytest.approx(0.46 / LCFS_DENOMINATOR, abs=1e-6)
    assert out.lcfs_star_usd_t == pytest.approx(285.070, abs=1e-2)
    assert out.ci_gap == 12.0


def test_a_cheaper_import_collapses_the_threshold():
    """UCO discounted by 6 c/lb: the threshold falls from 285 to 2.5 $/t.

        numerator = 0.46 + (-0.06 x 7.6) = 0.46 - 0.456 = 0.004
        LCFS*     = 0.004 / 1.61364e-3   = 2.479 $/t

    This is the result that makes the page useful: the threshold is **extremely
    sensitive** to the feedstock price differential. Six cents a pound is enough to
    flip 46 c/gal of tax credit. That's the number an insider can confirm or demolish.
    """
    out = lcfs_breakeven(
        price_domestic_usd_lb=0.52,
        price_imported_usd_lb=0.46,
        lcfs_current_usd_t=95.0,
    )
    assert out.price_gap_usd_lb == pytest.approx(-0.06)
    assert out.lcfs_star_usd_t == pytest.approx(2.479, abs=1e-3)


def test_a_pricier_import_raises_the_threshold():
    # numerator = 0.46 + 0.02 x 7.6 = 0.612 ; 0.612 / 1.61364e-3 = 379.27 $/t
    out = lcfs_breakeven(
        price_domestic_usd_lb=0.50,
        price_imported_usd_lb=0.52,
        lcfs_current_usd_t=95.0,
    )
    assert out.lcfs_star_usd_t == pytest.approx(379.267, abs=1e-2)


def test_threshold_does_not_depend_on_diesel_rin_or_plant_costs():
    """The argument that makes the threshold robust, and that has to come across in
    the email.

    The diesel, RIN, opex and ROI terms are identical across both pathways and cancel
    out of the difference. The threshold only moves with the CI differential and the
    feedstock price differential — so it survives not knowing a plant's cost
    structure.
    """
    reference = lcfs_breakeven(
        price_domestic_usd_lb=0.52, price_imported_usd_lb=0.46, lcfs_current_usd_t=95.0
    )
    # the analytic threshold doesn't even take these arguments; verify via the
    # numeric form, which does take them, that the result is identical regardless
    cheap_diesel = lcfs_breakeven_numeric(
        price_domestic_usd_lb=0.52,
        price_imported_usd_lb=0.46,
        ulsd_usd_gal=1.80,
        rin_d4_usd=0.30,
        lcfs_current_usd_t=95.0,
    )
    rich_diesel = lcfs_breakeven_numeric(
        price_domestic_usd_lb=0.52,
        price_imported_usd_lb=0.46,
        ulsd_usd_gal=4.20,
        rin_d4_usd=1.40,
        lcfs_current_usd_t=95.0,
    )
    assert cheap_diesel.theta_star == pytest.approx(rich_diesel.theta_star, abs=1e-6)
    assert cheap_diesel.theta_star == pytest.approx(reference.lcfs_star_usd_t, abs=1e-4)


def test_closed_form_and_numeric_solver_agree():
    """Cross-check: if the two diverge, the closed form has an algebra error."""
    analytic = lcfs_breakeven(
        price_domestic_usd_lb=0.50, price_imported_usd_lb=0.50, lcfs_current_usd_t=95.0
    )
    numeric = lcfs_breakeven_numeric(
        price_domestic_usd_lb=0.50,
        price_imported_usd_lb=0.50,
        ulsd_usd_gal=2.55,
        rin_d4_usd=0.62,
        lcfs_current_usd_t=95.0,
        hi=600.0,
    )
    assert analytic.lcfs_star_usd_t == pytest.approx(numeric.theta_star, abs=1e-4)


def test_headline_names_the_threshold_and_the_distance():
    history = pd.Series([60.0, 80.0, 95.0, 110.0, 130.0])
    out = lcfs_breakeven(
        price_domestic_usd_lb=0.52,
        price_imported_usd_lb=0.50,
        lcfs_current_usd_t=95.0,
        lcfs_history=history,
    )
    headline = out.headline
    assert "$/t CO2e" in headline
    assert "standard deviations" in headline
    assert "soy takes the share" in headline


def test_an_import_dirtier_than_the_domestic_has_no_threshold():
    """If it loses on both counts, there is no threshold — and the error must say so."""
    dirty_import = Feedstock("dirty imported UCO", 35.0, north_american=False)
    with pytest.raises(FeedstockError, match="no threshold"):
        lcfs_breakeven(
            imported=dirty_import,
            price_domestic_usd_lb=0.50,
            price_imported_usd_lb=0.50,
            lcfs_current_usd_t=95.0,
        )


def test_numeric_solver_reports_no_crossing_in_range():
    with pytest.raises(NoBreakevenInRange):
        lcfs_breakeven_numeric(
            price_domestic_usd_lb=0.90,      # soyoil priced out: UCO wins everywhere
            price_imported_usd_lb=0.20,
            ulsd_usd_gal=2.55,
            rin_d4_usd=0.62,
            lcfs_current_usd_t=95.0,
        )


# ===========================================================================
# S4 — the heatmap
# ===========================================================================
def test_winner_grid_has_both_zones():
    grid = winner_grid()
    assert set(grid["winner"].unique()) == {"imported UCO", "domestic soyoil"}


def test_imports_win_at_high_lcfs_and_low_ci():
    grid = winner_grid(
        ci_imported_values=np.array([12.0]),
        lcfs_values=np.array([0.0, 400.0]),
        price_domestic_usd_lb=0.50,
        price_imported_usd_lb=0.50,
    )
    low, high = grid.sort_values("lcfs_usd_t")["winner"].tolist()
    assert low == "domestic soyoil"
    assert high == "imported UCO"


def test_advantage_is_monotonic_in_lcfs():
    grid = winner_grid(ci_imported_values=np.array([15.0]))
    ordered = grid.sort_values("lcfs_usd_t")
    assert ordered["advantage_usd_lb"].is_monotonic_increasing


# ===========================================================================
# The crush balance
# ===========================================================================
def test_crush_balance_hand_computed():
    """5 Bn gal of RD, 40% soyoil share, 7.6 lb/gal, 11 lb of oil per bushel.

        soyoil required = 5e9 x 0.40 x 7.6      = 1.52e10 lb
        crush required  = 1.52e10 / 11          = 1,381,818,182 bu
        per day         = / 365                 = 3,785,803 bu/day
        gap             = 3,785,803 - 2,500,000 = 1,285,803 bu/day
    """
    out = crush_balance(
        rvo_gallons=5e9, soyoil_share=0.40, installed_capacity_bu_day=2_500_000.0
    )
    assert out.soyoil_required_lb == pytest.approx(1.52e10)
    assert out.crush_required_bu == pytest.approx(1_381_818_181.8, rel=1e-9)
    assert out.crush_required_bu_day == pytest.approx(3_785_803.24, abs=1e-2)
    assert out.gap_bu_day == pytest.approx(1_285_803.24, abs=1e-2)
    assert out.is_short


def test_crush_balance_when_capacity_is_sufficient():
    out = crush_balance(
        rvo_gallons=1e9, soyoil_share=0.20, installed_capacity_bu_day=2_500_000.0
    )
    assert not out.is_short
    assert "to spare" in out.headline


def test_crush_headline_quantifies_the_shortfall():
    out = crush_balance(
        rvo_gallons=5e9, soyoil_share=0.40, installed_capacity_bu_day=2_500_000.0
    )
    assert "short" in out.headline
    assert "bu/day" in out.headline


def test_soyoil_share_out_of_range_is_rejected():
    with pytest.raises(FeedstockError, match="soyoil_share"):
        crush_balance(rvo_gallons=5e9, soyoil_share=1.4, installed_capacity_bu_day=2.5e6)


# ===========================================================================
# T3-5 — energy beta and the policy break
# ===========================================================================
def test_rolling_beta_recovers_the_two_regimes(series):
    out = rolling_energy_beta(series["soyoil"], series["brent"], window=120)
    # well before the break, the window contains only the low regime
    early = out[out.index < RVO_BREAK_DATE - pd.Timedelta(days=200)]["beta"].mean()
    # well after, it contains only the high regime
    late = out[out.index > RVO_BREAK_DATE + pd.Timedelta(days=200)]["beta"].mean()
    assert early == pytest.approx(BETA_BEFORE_BREAK, abs=0.10)
    assert late == pytest.approx(BETA_AFTER_BREAK, abs=0.10)
    assert late > early


def test_chow_detects_the_policy_break(series):
    out = chow_break_test(series["soyoil"], series["brent"], RVO_BREAK_DATE)
    assert out.rejects_stability
    assert out.beta_after > out.beta_before
    assert "significant break" in out.summary


def test_chow_finds_nothing_within_a_single_regime(series):
    """Negative control, on a sample that contains no break at all.

    The cut lands in the middle of the PRE-break period, both sides in the same
    regime. Without this test, there would be no way to know whether Chow detects
    the policy or detects anything at all.
    """
    pre_break = slice(None, RVO_BREAK_DATE - pd.Timedelta(days=1))
    out = chow_break_test(
        series["soyoil"].loc[pre_break], series["brent"].loc[pre_break], "2024-08-01"
    )
    assert not out.rejects_stability


def test_a_split_that_straddles_the_real_break_also_fires(series):
    """Why the break date must be chosen a priori, never searched for.

    Cutting in June 2024 places the real March 2026 break inside the "after"
    subsample, whose average beta (~0.29) therefore genuinely differs from the
    "before" beta (~0.19). The test rejects — rightly, there is a real difference —
    but the date it names isn't the event's date. Sweeping every date and keeping the
    maximal F would produce a "break point" that is nothing but a search artefact.
    The only legitimate dates here are the regulatory calendar's own.
    """
    out = chow_break_test(series["soyoil"], series["brent"], "2024-06-03")
    assert out.rejects_stability
    at_policy_date = chow_break_test(series["soyoil"], series["brent"], RVO_BREAK_DATE)
    # the real policy date gives a markedly stronger F
    assert at_policy_date.f_stat > out.f_stat


def test_chow_refuses_a_date_too_close_to_the_edge(series):
    with pytest.raises(FeedstockError, match="too short"):
        chow_break_test(series["soyoil"], series["brent"], "2023-01-20")


def test_rolling_beta_refuses_a_window_longer_than_the_sample(series):
    with pytest.raises(FeedstockError, match="not enough"):
        rolling_energy_beta(series["soyoil"].head(50), series["brent"].head(50), window=120)


# ===========================================================================
# The fixture does impose the phenomenon
# ===========================================================================
def test_fixture_lcfs_crosses_the_parity_threshold(series):
    lcfs = series["lcfs"]
    assert lcfs.min() < 285.07 < lcfs.max()


def test_fixture_uco_trades_at_a_discount(series):
    assert (series["uco"] < series["soyoil"]).mean() > 0.95


def test_fixture_is_deterministic():
    a = build(seed=3)["lcfs"]
    b = build(seed=3)["lcfs"]
    pd.testing.assert_series_equal(a, b)
