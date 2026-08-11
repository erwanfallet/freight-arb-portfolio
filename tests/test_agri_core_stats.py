"""Golden tests for the statistics toolkit.

Many of these tests verify that the code **refuses to conclude**. That's intentional:
this module's value lies in the bands, the intervals and the "inconclusive" verdicts,
not in the point estimates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agri.core.stats import (
    StatsError,
    adf_kpss,
    block_bootstrap,
    ccf_with_band,
    clopper_pearson,
    hac_ols,
    newey_west_lags,
    regime_runs,
)


# ===========================================================================
# Stationarity
# ===========================================================================
def test_white_noise_is_stationary():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(size=500), index=pd.date_range("2020-01-01", periods=500, freq="D"))
    out = adf_kpss(s)
    assert out.adf_rejects_unit_root
    assert not out.kpss_rejects_stationarity
    assert out.verdict == "stationary"


def test_random_walk_has_a_unit_root():
    rng = np.random.default_rng(1)
    s = pd.Series(
        np.cumsum(rng.normal(size=500)),
        index=pd.date_range("2020-01-01", periods=500, freq="D"),
    )
    out = adf_kpss(s)
    assert not out.adf_rejects_unit_root
    assert out.kpss_rejects_stationarity
    assert out.verdict == "unit_root"


def test_short_series_refuses_the_test():
    s = pd.Series(np.arange(10.0), index=pd.date_range("2024-01-01", periods=10, freq="D"))
    with pytest.raises(StatsError, match="20 observations"):
        adf_kpss(s)


def test_summary_says_do_not_conclude_when_tests_disagree():
    from agri.core.stats import StationarityVerdict

    # both reject: probable structural break, definitely not "somewhat stationary"
    both = StationarityVerdict(
        adf_stat=-4.0, adf_pvalue=0.001, kpss_stat=1.0, kpss_pvalue=0.01, alpha=0.05, n_obs=300
    )
    assert both.verdict == "conflicting"
    assert "do not conclude" in both.summary

    # neither rejects: sample too short
    neither = StationarityVerdict(
        adf_stat=-1.0, adf_pvalue=0.60, kpss_stat=0.1, kpss_pvalue=0.10, alpha=0.05, n_obs=30
    )
    assert neither.verdict == "inconclusive"
    assert "do not conclude" in neither.summary


# ===========================================================================
# Cross-correlation and the Bartlett band
# ===========================================================================
def test_bartlett_band_hand_computed():
    """z / sqrt(n_eff) with z = 1.959964 for 95%.

    At n = 150: 1.959964 / sqrt(150) = 1.959964 / 12.24745 = 0.16003.
    A correlation of -0.10 is therefore WITHIN the band — the spec's example.
    """
    rng = np.random.default_rng(2)
    x = pd.Series(rng.normal(size=150))
    y = pd.Series(rng.normal(size=150))
    out = ccf_with_band(x, y, max_lag=10, n_eff=150)
    assert out.band == pytest.approx(0.16003, abs=1e-4)
    assert abs(-0.10) < out.band


def test_peak_lag_sign_convention_x_leads_y():
    """Convention: the value at lag k is corr(x[t], y[t+k]), so positive lag = x leads y."""
    rng = np.random.default_rng(3)
    base = rng.normal(size=400)
    x = pd.Series(base)
    y = pd.Series(np.concatenate([np.full(3, np.nan), base[:-3]]))  # y[t] = x[t-3]
    out = ccf_with_band(x.iloc[3:].reset_index(drop=True), y.iloc[3:].reset_index(drop=True), max_lag=10)
    lag, value = out.peak()
    assert lag == 3
    assert value > 0.9


def test_effective_n_widens_the_band_and_can_kill_a_signal():
    """The number that changes the reading: the same correlation, two effective n's.

    At n_eff = 150 the band is 0.160; at n_eff = 7 (a backtest with overlapping
    positions) it rises to 1.959964/sqrt(7) = 0.7408 — wider than any possible
    correlation. In other words: at this n, no correlation is publishable.
    """
    rng = np.random.default_rng(4)
    x = pd.Series(rng.normal(size=150))
    y = pd.Series(rng.normal(size=150))
    wide = ccf_with_band(x, y, max_lag=10, n_eff=7)
    assert wide.band == pytest.approx(0.74080, abs=1e-4)
    assert len(wide.significant_lags) == 0


def test_summary_flags_a_peak_inside_the_band():
    rng = np.random.default_rng(5)
    x = pd.Series(rng.normal(size=200))
    y = pd.Series(rng.normal(size=200))
    out = ccf_with_band(x, y, max_lag=5, n_eff=7)
    assert "WITHIN THE BAND" in out.summary


def test_ccf_refuses_when_sample_too_short_for_the_lags():
    x = pd.Series(np.arange(15.0))
    y = pd.Series(np.arange(15.0))
    with pytest.raises(StatsError, match="too short"):
        ccf_with_band(x, y, max_lag=20)


# ===========================================================================
# Newey-West
# ===========================================================================
def test_newey_west_lag_rule_hand_computed():
    # floor(4 * (n/100)^(2/9))
    assert newey_west_lags(100) == 4      # 4 * 1        = 4.000
    assert newey_west_lags(250) == 4      # 4 * 1.22580  = 4.903
    assert newey_west_lags(1000) == 6     # 4 * 1.66810  = 6.672
    assert newey_west_lags(50) == 3       # 4 * 0.85720  = 3.429


def test_hac_ols_recovers_a_known_slope():
    rng = np.random.default_rng(6)
    n = 300
    x = pd.Series(rng.normal(size=n), name="x")
    y = 2.0 + 3.0 * x + pd.Series(rng.normal(scale=0.5, size=n))
    out = hac_ols(y, x)
    assert out.params["x"] == pytest.approx(3.0, abs=0.1)
    assert out.params["const"] == pytest.approx(2.0, abs=0.1)
    assert out.is_significant("x")
    assert out.lags == newey_west_lags(n)


def test_hac_standard_errors_exceed_naive_ones_under_autocorrelation():
    """Why HAC isn't an optional refinement.

    With a strongly autocorrelated residual, the naive standard error is too small and
    the p-value too flattering. HAC corrects in the direction that's inconvenient, which
    is the right direction.
    """
    import statsmodels.api as sm

    rng = np.random.default_rng(7)
    n = 300
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.9 * noise[i - 1] + rng.normal(scale=0.3)
    x = pd.Series(np.cumsum(rng.normal(size=n)), name="x")
    y = 1.0 + 0.5 * x + pd.Series(noise)

    hac = hac_ols(y, x)
    naive = sm.OLS(y, sm.add_constant(x.to_frame())).fit()
    assert hac.std_errors["x"] > naive.bse["x"]


# ===========================================================================
# Block bootstrap
# ===========================================================================
def test_bootstrap_is_deterministic_for_a_given_seed():
    rng = np.random.default_rng(8)
    s = pd.Series(rng.normal(size=200))
    a = block_bootstrap(s, block_len=20, n_iter=500)
    b = block_bootstrap(s, block_len=20, n_iter=500)
    assert (a.lo, a.hi) == (b.lo, b.hi)


def test_bootstrap_on_a_constant_series_is_degenerate():
    s = pd.Series([4.0] * 100)
    out = block_bootstrap(s, block_len=10, n_iter=200)
    assert out.point == pytest.approx(4.0)
    assert out.lo == pytest.approx(4.0)
    assert out.hi == pytest.approx(4.0)
    assert not out.includes_zero


def test_bootstrap_ci_covers_zero_for_zero_mean_noise():
    rng = np.random.default_rng(9)
    s = pd.Series(rng.normal(size=400))
    out = block_bootstrap(s, block_len=20, n_iter=2000)
    assert out.includes_zero
    assert "INCLUDES ZERO" in out.summary


def test_bootstrap_ci_excludes_zero_for_a_strong_mean():
    rng = np.random.default_rng(10)
    s = pd.Series(rng.normal(loc=5.0, scale=1.0, size=400))
    out = block_bootstrap(s, block_len=20, n_iter=2000)
    assert not out.includes_zero
    assert out.lo > 0


def test_longer_blocks_widen_the_interval_on_autocorrelated_data():
    """The whole point of the exercise: ignoring dependence falsely shrinks the CI."""
    rng = np.random.default_rng(11)
    n = 500
    values = np.zeros(n)
    for i in range(1, n):
        values[i] = 0.95 * values[i - 1] + rng.normal()
    s = pd.Series(values)
    short = block_bootstrap(s, block_len=1, n_iter=2000)
    long = block_bootstrap(s, block_len=50, n_iter=2000)
    assert (long.hi - long.lo) > (short.hi - short.lo)


def test_bootstrap_accepts_a_custom_statistic():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
    out = block_bootstrap(s, block_len=5, statistic=np.median, n_iter=300)
    assert out.point == pytest.approx(3.0)


def test_block_longer_than_series_raises():
    s = pd.Series(np.arange(10.0))
    with pytest.raises(StatsError, match="exceeds the series length"):
        block_bootstrap(s, block_len=50, n_iter=100)


# ===========================================================================
# Regimes
# ===========================================================================
@pytest.fixture
def margin_with_two_negative_runs():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    margin = pd.Series([1.0, -2.0, -5.0, -3.0, 1.0, 1.0, -1.0, -4.0, 1.0, 1.0], index=idx)
    return margin


def test_regime_runs_hand_computed(margin_with_two_negative_runs):
    """Margin: [1, -2, -5, -3, 1, 1, -1, -4, 1, 1] over 10 consecutive days.

    Two negative episodes:
        Jan 02 to Jan 04: 3 observations, mean (-2-5-3)/3 = -3.3333, min -5
        Jan 07 to Jan 08: 2 observations, mean (-1-4)/2  = -2.5,    min -4
    """
    margin = margin_with_two_negative_runs
    runs = regime_runs(margin < 0, depth=margin)
    assert len(runs) == 2

    assert runs.loc[0, "start"] == pd.Timestamp("2024-01-02")
    assert runs.loc[0, "end"] == pd.Timestamp("2024-01-04")
    assert runs.loc[0, "n_obs"] == 3
    assert runs.loc[0, "duration_days"] == 3
    assert runs.loc[0, "depth_mean"] == pytest.approx(-3.33333, abs=1e-4)
    assert runs.loc[0, "depth_min"] == pytest.approx(-5.0)

    assert runs.loc[1, "n_obs"] == 2
    assert runs.loc[1, "depth_mean"] == pytest.approx(-2.5)
    assert runs.loc[1, "depth_min"] == pytest.approx(-4.0)


def test_min_obs_filters_short_episodes(margin_with_two_negative_runs):
    margin = margin_with_two_negative_runs
    runs = regime_runs(margin < 0, depth=margin, min_obs=3)
    assert len(runs) == 1
    assert runs.loc[0, "n_obs"] == 3


def test_regime_runs_returns_empty_frame_with_columns_when_never_true():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    runs = regime_runs(pd.Series([False] * 5, index=idx))
    assert len(runs) == 0
    assert "duration_days" in runs.columns
    assert "depth_min" in runs.columns


def test_regime_runs_without_depth_leaves_depth_columns_missing():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    runs = regime_runs(pd.Series([False, True, True, False, False], index=idx))
    assert len(runs) == 1
    assert np.isnan(runs.loc[0, "depth_mean"])


def test_longest_run_is_the_mail_sentence(margin_with_two_negative_runs):
    # "the margin has been negative for N days, the sample's longest run"
    runs = regime_runs(margin_with_two_negative_runs < 0, depth=margin_with_two_negative_runs)
    longest = runs.loc[runs["duration_days"].idxmax()]
    assert longest["duration_days"] == 3
    assert longest["depth_min"] == pytest.approx(-5.0)


# ===========================================================================
# Clopper-Pearson — the T1-M module's number
# ===========================================================================
def test_clopper_pearson_perfect_score_on_18_draws():
    """18 wins out of 18: lower bound = 0.025^(1/18) = 81.5%."""
    out = clopper_pearson(18, 18)
    assert out.point == pytest.approx(1.0)
    assert out.lo == pytest.approx(0.81470, abs=1e-4)
    assert out.hi == pytest.approx(1.0)


def test_clopper_pearson_perfect_score_on_seven_effective_draws():
    """THE number for the email. 7 wins out of 7: lower bound = 0.025^(1/7) = 59.0%.

    This is the quantitative translation of the self-critique sentence: a 100% win rate
    on ~7 independent draws is not distinguishable from a process that succeeds 6 times
    out of 10. The gap with the 81.5% bound announced on n = 18 is exactly what the
    position overlap costs in information.
    """
    out = clopper_pearson(7, 7)
    assert out.lo == pytest.approx(0.59043, abs=1e-4)
    assert out.lo < 0.60


def test_the_overlap_correction_costs_22_points_of_lower_bound():
    naive = clopper_pearson(18, 18)
    honest = clopper_pearson(7, 7)
    assert naive.lo - honest.lo == pytest.approx(0.2243, abs=1e-3)


def test_clopper_pearson_accepts_a_non_integer_effective_n():
    # n_eff = 7.23 -> rounded to 7, conservative so in the right direction
    out = clopper_pearson(7, 7.23)
    assert out.n == pytest.approx(7.23)
    assert out.lo == pytest.approx(0.59043, abs=1e-4)


def test_clopper_pearson_zero_successes():
    # 0 out of 10: upper bound = 1 - 0.025^(1/10) = 30.85%
    out = clopper_pearson(0, 10)
    assert out.lo == 0.0
    assert out.hi == pytest.approx(0.30850, abs=1e-4)


def test_clopper_pearson_is_symmetric_at_one_half():
    out = clopper_pearson(5, 10)
    assert out.point == pytest.approx(0.5)
    assert out.lo == pytest.approx(1.0 - out.hi, abs=1e-9)


def test_summary_reports_the_interval_not_just_the_point():
    assert "exact95%CI" in clopper_pearson(7, 7).summary.replace(" ", "")
