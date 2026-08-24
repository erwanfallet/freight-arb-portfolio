"""Golden tests for project A — values computed by hand, independently of the code.

Every expected value below is derived by hand in the comment preceding it. If the code
changes and a test breaks, the code is wrong until proven otherwise.
"""
import numpy as np
import pandas as pd
import pytest

from freight.chains.ironore import (
    carry_cost_of_extra_voyage_days,
    decompose_premium,
    explained_variance,
    freight_hedge_effect,
    freight_per_dry_tonne,
    negative_residual_episodes,
)


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="B")


# --------------------------------------------------------------------------- moisture


def test_moisture_conversion_golden():
    """20.00 $/wmt at 9% moisture = 20 / 0.91 = 21.978021978... $/dmt."""
    assert freight_per_dry_tonne(20.0, 0.09) == pytest.approx(21.978021978021978, rel=1e-12)
    # 10.00 $/wmt at 8% = 10 / 0.92 = 10.869565217...
    assert freight_per_dry_tonne(10.0, 0.08) == pytest.approx(10.869565217391305, rel=1e-12)


def test_moisture_zero_is_identity():
    assert freight_per_dry_tonne(17.5, 0.0) == pytest.approx(17.5)


def test_moisture_out_of_range_is_rejected():
    with pytest.raises(ValueError):
        freight_per_dry_tonne(20.0, 1.0)
    with pytest.raises(ValueError):
        freight_per_dry_tonne(20.0, -0.01)


# ---------------------------------------------------------------------- decomposition


def test_decomposition_golden_single_date():
    """Hand-computed case, the only one that really matters.

    P65 = 120, P62 = 100          -> observed premium = 20.00 $/dmt
    C3  = 20 $/wmt, 9% moisture   -> 21.978021978 $/dmt
    C5  = 10 $/wmt, 8% moisture   -> 10.869565217 $/dmt
    fair value freight            = 11.108456761 $/dmt
    residual                      = 20 - 11.108456761 = 8.891543239 $/dmt
    freight share                 = 11.108456761 / 20 = 55.542%

    And the pedagogical point: the raw, uncorrected differential is 10.00, a 50.0%
    freight share. The moisture correction shifts the freight share by 5.5 points.
    """
    idx = _dates(1)
    d = decompose_premium(
        p65=pd.Series([120.0], index=idx),
        p62=pd.Series([100.0], index=idx),
        c3=pd.Series([20.0], index=idx),
        c5=pd.Series([10.0], index=idx),
        moisture_brazil=0.09,
        moisture_australia=0.08,
    )
    row = d.iloc[0]
    assert row["premium_observed"] == pytest.approx(20.0)
    assert row["c3_dmt"] == pytest.approx(21.978021978021978, rel=1e-12)
    assert row["c5_dmt"] == pytest.approx(10.869565217391305, rel=1e-12)
    assert row["freight_fair_value"] == pytest.approx(11.108456760630673, rel=1e-12)
    assert row["residual"] == pytest.approx(8.891543239369327, rel=1e-12)
    assert row["freight_share"] == pytest.approx(0.5554228380315337, rel=1e-12)
    # the naive figure, to measure what the correction changes
    assert row["premium_naive_freight"] == pytest.approx(10.0)
    assert row["premium_naive_freight"] / row["premium_observed"] == pytest.approx(0.50)


def test_moisture_correction_always_increases_freight_share():
    """The moisture correction can't reduce the freight share when Brazilian
    moisture exceeds Australian moisture: that's a property of the identity, not a
    sample accident.
    """
    idx = _dates(50)
    rng = np.random.default_rng(0)
    d = decompose_premium(
        p65=pd.Series(120 + rng.normal(0, 3, 50), index=idx),
        p62=pd.Series(100 + rng.normal(0, 3, 50), index=idx),
        c3=pd.Series(20 + rng.normal(0, 1, 50), index=idx),
        c5=pd.Series(10 + rng.normal(0, 1, 50), index=idx),
        moisture_brazil=0.09,
        moisture_australia=0.08,
    )
    assert (d["freight_fair_value"] >= d["premium_naive_freight"]).all()


def test_negative_premium_yields_nan_share_not_nonsense():
    """A negative premium must not produce an interpretable negative freight share."""
    idx = _dates(1)
    d = decompose_premium(
        p65=pd.Series([98.0], index=idx),
        p62=pd.Series([100.0], index=idx),
        c3=pd.Series([20.0], index=idx),
        c5=pd.Series([10.0], index=idx),
    )
    assert np.isnan(d.iloc[0]["freight_share"])
    # the residual, on the other hand, stays defined and strongly negative: -2 - 11.1 = -13.1
    assert d.iloc[0]["residual"] < -13.0


def test_no_common_dates_raises_rather_than_filling():
    """Data contract rule: a gap is information, never a forward-fill."""
    with pytest.raises(ValueError, match="no common date"):
        decompose_premium(
            p65=pd.Series([120.0], index=pd.DatetimeIndex(["2024-01-01"])),
            p62=pd.Series([100.0], index=pd.DatetimeIndex(["2024-01-02"])),
            c3=pd.Series([20.0], index=pd.DatetimeIndex(["2024-01-03"])),
            c5=pd.Series([10.0], index=pd.DatetimeIndex(["2024-01-04"])),
        )


# ------------------------------------------------------------------------ regression


def test_explained_variance_on_exact_linear_relation():
    """premium = 2 + 1.5 × freight exactly -> slope 1.5, intercept 2, R² = 1."""
    idx = _dates(40)
    fret = pd.Series(np.linspace(8.0, 16.0, 40), index=idx)
    premium = 2.0 + 1.5 * fret
    ev = explained_variance(premium, fret)
    assert ev.slope == pytest.approx(1.5, rel=1e-9)
    assert ev.intercept == pytest.approx(2.0, rel=1e-9)
    assert ev.r_squared == pytest.approx(1.0, rel=1e-9)
    assert ev.n_obs == 40


def test_explained_variance_on_changes_differs_from_levels():
    """Two trending series give a flattering R² in levels and an honest one in
    changes. The gap between the two is the result to show, not to hide.
    """
    idx = _dates(120)
    rng = np.random.default_rng(7)
    trend = np.linspace(0, 10, 120)
    fret = pd.Series(trend + rng.normal(0, 0.5, 120), index=idx)
    premium = pd.Series(trend + rng.normal(0, 3.0, 120), index=idx)
    levels = explained_variance(premium, fret, on_changes=False)
    changes = explained_variance(premium, fret, on_changes=True)
    assert levels.r_squared > changes.r_squared


# --------------------------------------------------------------------------- episodes


def test_negative_residual_episodes_respects_min_days():
    idx = _dates(20)
    residual = np.ones(20) * 5.0
    residual[3:9] = -1.0   # 6 consecutive days -> kept
    residual[15:17] = -2.0  # 2 days -> filtered out
    d = pd.DataFrame({"residual": residual}, index=idx)
    episodes = negative_residual_episodes(d, min_days=5)
    assert len(episodes) == 1
    assert episodes.iloc[0]["n_obs"] == 6
    assert episodes.iloc[0]["residual_min"] == pytest.approx(-1.0)


def test_no_negative_residual_returns_empty_frame():
    idx = _dates(10)
    d = pd.DataFrame({"residual": np.ones(10)}, index=idx)
    assert negative_residual_episodes(d).empty


# ------------------------------------------------------------------------------ hedge


def test_perfect_hedge_removes_all_volatility():
    """If the premium moves exactly 2× the freight share, beta = 2 and the residual
    vol is zero: a 100% reduction.
    """
    idx = _dates(60)
    rng = np.random.default_rng(3)
    fret = pd.Series(np.cumsum(rng.normal(0, 1, 60)) + 12.0, index=idx)
    premium = 2.0 * fret + 5.0
    h = freight_hedge_effect(premium, fret)
    assert h.beta == pytest.approx(2.0, rel=1e-9)
    assert h.vol_hedged == pytest.approx(0.0, abs=1e-9)
    assert h.vol_reduction_pct == pytest.approx(100.0, abs=1e-6)


def test_unit_hedge_can_be_worse_than_optimal_hedge():
    """The naive unit hedge is not the minimum-variance hedge. The gap is itself a
    desk-relevant result.
    """
    idx = _dates(200)
    rng = np.random.default_rng(11)
    fret = pd.Series(np.cumsum(rng.normal(0, 1, 200)) + 12.0, index=idx)
    premium = 0.4 * fret + pd.Series(np.cumsum(rng.normal(0, 1, 200)), index=idx)
    optimal = freight_hedge_effect(premium, fret)
    naive = freight_hedge_effect(premium, fret, beta=1.0)
    assert optimal.vol_hedged <= naive.vol_hedged


def test_zero_variance_freight_is_rejected():
    idx = _dates(30)
    fret = pd.Series(np.ones(30) * 11.0, index=idx)
    premium = pd.Series(np.linspace(1, 5, 30), index=idx)
    with pytest.raises(ValueError, match="doesn't vary"):
        freight_hedge_effect(premium, fret)


# ------------------------------------------------------------------------ carry cost


def test_carry_cost_golden():
    """100 $/dmt, 25 extra days, 6% annual -> 100 × 0.06 × 25/365 = 0.410958904 $/dmt."""
    assert carry_cost_of_extra_voyage_days(100.0, 25.0, 0.06) == pytest.approx(
        0.410958904109589, rel=1e-12
    )


def test_carry_cost_rejects_negative_days():
    with pytest.raises(ValueError):
        carry_cost_of_extra_voyage_days(100.0, -1.0, 0.06)


# ===========================================================================
# THE ATTRIBUTION AND THE THIRD ORIGIN
# ===========================================================================
# Two tests here may not be weakened by a later rework.
#
# `test_the_attribution_reports_the_sign_flip_threshold` is what keeps the 2026 result
# honest: the sign of the quality residual depends on an assumed weight, and the page is
# only allowed to state it alongside the weight at which it reverses.
#
# `test_guinea_is_a_brazil_length_haul_not_a_cheap_one` is the fact the whole Simandou
# reading turns on. If Guinea were a short haul the two-opposing-effects argument would
# collapse into a simple supply story.
from freight.chains.ironore import (  # noqa: E402
    AUSTRALIA_QINGDAO_NM,
    BRAZIL_QINGDAO_NM,
    GUINEA_QINGDAO_NM,
    premium_attribution,
    third_origin_arithmetic,
)


def _attribution_frame() -> pd.DataFrame:
    """Two years, hand-built so the arithmetic below is checkable by eye.

    2025: premium 10, freight_spread 9.10  -> freight_dry = 9.10/0.91 = 10.0
    2026: premium 14, freight_spread 18.20 -> freight_dry = 18.20/0.91 = 20.0
    so d(premium) = +4, d(freight_dry) = +10, and the residual flips sign at w = 0.4.
    """
    index = pd.to_datetime(["2025-06-30", "2025-12-31", "2026-06-30", "2026-12-31"])
    return pd.DataFrame(
        {
            "premium": [10.0, 10.0, 14.0, 14.0],
            "freight_spread": [9.10, 9.10, 18.20, 18.20],
        },
        index=index,
    )


def test_attribution_splits_the_move_exactly():
    """d(premium) = freight_part + residual_part, by construction and to the decimal."""
    result = premium_attribution(
        _attribution_frame(), weight=0.5, year_from=2025, year_to=2026, moisture=0.09
    )
    assert result.premium_change == pytest.approx(4.0)
    assert result.freight_change == pytest.approx(10.0)
    assert result.freight_part == pytest.approx(5.0)      # 0.5 x 10
    assert result.residual_part == pytest.approx(-1.0)    # 4 - 5
    assert result.freight_part + result.residual_part == pytest.approx(result.premium_change)


def test_the_attribution_reports_the_sign_flip_threshold():
    """The honest object is the threshold, not the point estimate.

    d(premium)/d(freight) = 4/10 = 0.4, so any weight above 0.4 makes the quality
    residual negative and any weight below it makes it positive. A page that quoted the
    residual without the threshold would be hiding the assumption that produced its sign.
    """
    frame = _attribution_frame()
    above = premium_attribution(frame, weight=0.5, year_from=2025, year_to=2026, moisture=0.09)
    below = premium_attribution(frame, weight=0.3, year_from=2025, year_to=2026, moisture=0.09)

    assert above.threshold_weight == pytest.approx(0.4)
    assert below.threshold_weight == pytest.approx(0.4)
    assert above.quality_fell
    assert not below.quality_fell
    assert "flips at a weight of" in above.headline


def test_attribution_rejects_an_implausible_weight():
    with pytest.raises(ValueError, match="weight outside the plausible range"):
        premium_attribution(
            _attribution_frame(), weight=2.0, year_from=2025, year_to=2026
        )


def test_attribution_rejects_a_year_with_no_observations():
    with pytest.raises(ValueError, match="no observations for"):
        premium_attribution(
            _attribution_frame(), weight=0.5, year_from=2019, year_to=2026
        )


def test_guinea_is_a_brazil_length_haul_not_a_cheap_one():
    """The fact the Simandou reading depends on.

    Simandou adds high-grade supply AND Capesize tonne-miles, which push the observed
    premium in opposite directions. That argument only holds because Guinea is a
    long haul comparable to Brazil — a short haul would make it a pure supply story.
    """
    arithmetic = third_origin_arithmetic()
    assert arithmetic.is_long_haul_like_brazil
    assert arithmetic.guinea_vs_brazil == pytest.approx(GUINEA_QINGDAO_NM / BRAZIL_QINGDAO_NM)
    assert arithmetic.guinea_vs_australia > 3.0
    assert "not cheap high-grade ore" in arithmetic.headline


def test_the_derived_guinea_distance_agrees_with_published_voyage_time():
    """The distance is derived, not looked up, so the derivation is pinned here.

    Drewry: Guinea-China round voyage beyond 90 days. At 12 kn with 8 round-trip port
    days, (90 - 8) x 12 x 24 / 2 = 11,808 nm per leg. The same arithmetic applied to
    Australia's published 30-35 day rotation must recover its known 3,500 nm — that
    cross-check is what makes the Guinea number usable rather than a guess.
    """
    def one_leg_nm(round_trip_days: float, speed_kn: float = 12.0, port_days: float = 8.0) -> float:
        return (round_trip_days - port_days) * speed_kn * 24.0 / 2.0

    assert one_leg_nm(90.0) == pytest.approx(GUINEA_QINGDAO_NM, rel=0.02)
    assert one_leg_nm(32.5) == pytest.approx(AUSTRALIA_QINGDAO_NM, rel=0.02)


def test_a_non_positive_distance_is_rejected():
    with pytest.raises(ValueError, match="must be > 0 nm"):
        third_origin_arithmetic(guinea_nm=0.0)
