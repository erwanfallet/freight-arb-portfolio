"""Golden tests for the three resampling rules.

Expected values hand-computed in the comment preceding them. This file mostly tests
**refusals**: this module's value is in what it prevents, not in what it computes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agri.core.resample import (
    EffectiveSample,
    ResampleError,
    StaleDataInTest,
    align_for_test,
    average_concurrency,
    downsample,
    effective_n,
    effective_n_from_trades,
    effective_n_rolling,
    ffill_with_provenance,
    for_pnl,
    for_test,
    infer_frequency,
    slowest_frequency,
)


def _daily(start: str, periods: int, start_value: float = 100.0) -> pd.Series:
    idx = pd.date_range(start, periods=periods, freq="B")
    return pd.Series(np.arange(periods, dtype=float) + start_value, index=idx)


def _monthly(start: str, periods: int, start_value: float = 10.0) -> pd.Series:
    idx = pd.date_range(start, periods=periods, freq="MS")  # dated on the 1st of the month
    return pd.Series(np.arange(periods, dtype=float) + start_value, index=idx)


# ===========================================================================
# RULE A
# ===========================================================================
def test_infer_frequency_daily_survives_weekends():
    # business days: median gap of 1 day despite 3-day weekend jumps
    assert infer_frequency(_daily("2024-01-01", 60)) == "daily"


def test_infer_frequency_weekly():
    s = pd.Series(1.0, index=pd.date_range("2024-01-01", periods=30, freq="W"))
    assert infer_frequency(s) == "weekly"


def test_infer_frequency_fortnightly_unica_style():
    # UNICA fortnight: median gap of 15 days
    s = pd.Series(1.0, index=pd.date_range("2024-01-01", periods=30, freq="15D"))
    assert infer_frequency(s) == "fortnightly"


def test_infer_frequency_monthly_despite_uneven_month_lengths():
    # months from 28 to 31 days: the median crosses them, a mode or a min would get caught out
    assert infer_frequency(_monthly("2024-01-01", 24)) == "monthly"


def test_infer_frequency_quarterly():
    s = pd.Series(1.0, index=pd.date_range("2020-01-01", periods=16, freq="QE"))
    assert infer_frequency(s) == "quarterly"


def test_infer_frequency_needs_three_observations():
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-01-01", "2024-02-01"]))
    with pytest.raises(ResampleError, match="3 observations"):
        infer_frequency(s)


def test_slowest_frequency_picks_the_binding_one():
    assert slowest_frequency(
        {"px": _daily("2024-01-01", 60), "grind": _monthly("2024-01-01", 24)}
    ) == "monthly"


def test_downsample_daily_to_monthly_takes_last_print():
    # daily series 100, 101, 102, ... on business days from 2024-01-01.
    # January 2024 has 23 business days -> values 100..122, last print = 122.
    s = _daily("2024-01-01", 60)
    out = downsample(s, "monthly", how="last")
    assert out.index[0] == pd.Timestamp("2024-01-31")
    assert out.iloc[0] == pytest.approx(122.0)


def test_downsample_daily_to_monthly_mean_differs_from_last():
    s = _daily("2024-01-01", 60)
    # mean of 100..122 = (100+122)/2 = 111.0
    assert downsample(s, "monthly", how="mean").iloc[0] == pytest.approx(111.0)


def test_downsample_refuses_to_go_up_rule_a():
    """The core of Rule A: upsampling a slow series must raise, not run."""
    with pytest.raises(ResampleError, match="RULE A violated"):
        downsample(_monthly("2024-01-01", 24), "daily")


def test_downsample_normalises_labels_even_at_same_frequency():
    """The empty-intersection trap.

    A monthly series dated on the 1st and a daily series downsampled to monthly (dated
    at month-end) have the same frequency and **zero common date**. Without
    normalisation, align_for_test would come out empty with nothing to explain why.
    """
    monthly = _monthly("2024-01-01", 6)
    assert monthly.index[0] == pd.Timestamp("2024-01-01")
    out = downsample(monthly, "monthly")
    assert out.index[0] == pd.Timestamp("2024-01-31")
    assert out.iloc[0] == pytest.approx(10.0)  # the value is preserved, only the label moves


def test_downsample_rejects_unknown_aggregation():
    with pytest.raises(ResampleError, match="aggregation"):
        downsample(_daily("2024-01-01", 60), "monthly", how="median")


def test_align_for_test_runs_at_the_slowest_frequency():
    aligned = align_for_test(
        {"px": _daily("2024-01-01", 120), "grind": _monthly("2024-01-01", 6)}
    )
    assert aligned.attrs["test_frequency"] == "monthly"
    # 120 business days from 2024-01-01 cover January through mid-June -> 6 full months
    # on the daily side, and the monthly series supplies 6: the intersection is 6.
    assert len(aligned) == 6
    assert list(aligned.columns) == ["px", "grind"]


def test_align_for_test_intersects_without_filling_gaps():
    # the monthly series starts three months after the daily one: the first three
    # months drop out of the test instead of being filled in
    aligned = align_for_test(
        {"px": _daily("2024-01-01", 250), "grind": _monthly("2024-04-01", 6)}
    )
    assert len(aligned) == 6
    assert aligned.index[0] == pd.Timestamp("2024-04-30")


def test_align_for_test_raises_when_no_common_date():
    aligned_a = _monthly("2020-01-01", 6)
    aligned_b = _monthly("2024-01-01", 6)
    with pytest.raises(ResampleError, match="no common date"):
        align_for_test({"a": aligned_a, "b": aligned_b})


# ===========================================================================
# RULE B
# ===========================================================================
@pytest.fixture
def filled_monthly_on_daily() -> pd.DataFrame:
    """Monthly TC (10 in January, 20 in February) carried onto a daily calendar."""
    source = pd.Series(
        [10.0, 20.0], index=pd.to_datetime(["2024-01-01", "2024-02-01"])
    )
    target = pd.date_range("2024-01-01", "2024-02-05", freq="D")
    return ffill_with_provenance(source, target)


def test_ffill_marks_true_prints(filled_monthly_on_daily):
    df = filled_monthly_on_daily
    assert bool(df.loc["2024-01-01", "is_true_print"]) is True
    assert bool(df.loc["2024-02-01", "is_true_print"]) is True
    assert bool(df.loc["2024-01-15", "is_true_print"]) is False


def test_ffill_carries_the_value_forward(filled_monthly_on_daily):
    df = filled_monthly_on_daily
    assert df.loc["2024-01-15", "value"] == pytest.approx(10.0)
    assert df.loc["2024-02-05", "value"] == pytest.approx(20.0)


def test_staleness_days_hand_computed(filled_monthly_on_daily):
    df = filled_monthly_on_daily
    # a genuine print has zero age
    assert df.loc["2024-01-01", "staleness_days"] == pytest.approx(0.0)
    # 2024-01-15 is 14 days from the January 1st print
    assert df.loc["2024-01-15", "staleness_days"] == pytest.approx(14.0)
    # 2024-02-05 is 4 days from the February 1st print, NOT 35 from the January print
    assert df.loc["2024-02-05", "staleness_days"] == pytest.approx(4.0)


def test_no_value_and_no_age_before_the_first_print():
    source = pd.Series([10.0], index=pd.to_datetime(["2024-01-10"]))
    target = pd.date_range("2024-01-01", "2024-01-15", freq="D")
    df = ffill_with_provenance(source, target)
    assert np.isnan(df.loc["2024-01-05", "value"])
    assert np.isnan(df.loc["2024-01-05", "staleness_days"])
    assert df.loc["2024-01-12", "value"] == pytest.approx(10.0)


def test_max_staleness_kills_dead_data(filled_monthly_on_daily):
    source = pd.Series(
        [10.0, 20.0], index=pd.to_datetime(["2024-01-01", "2024-02-01"])
    )
    target = pd.date_range("2024-01-01", "2024-02-05", freq="D")
    df = ffill_with_provenance(source, target, max_staleness_days=10)
    # at 14 days old, the value is removed: a two-week-old TC is no longer a
    # conservative assumption
    assert np.isnan(df.loc["2024-01-15", "value"])
    # at 4 days, it stays
    assert df.loc["2024-02-05", "value"] == pytest.approx(20.0)


def test_for_pnl_accepts_filled_data(filled_monthly_on_daily):
    """Rule B, allowed side: a smelter does pay a monthly TC every day."""
    values = for_pnl(filled_monthly_on_daily)
    assert len(values) == 36  # 2024-01-01 through 2024-02-05 inclusive
    assert values.loc["2024-01-15"] == pytest.approx(10.0)


def test_for_test_refuses_filled_data(filled_monthly_on_daily):
    """Rule B, blocked side. This is the file's most important test.

    The rule stops being a method guideline: the code path that would feed a
    regression with forward-filled data raises an exception.
    """
    with pytest.raises(StaleDataInTest, match="RULE B violated"):
        for_test(filled_monthly_on_daily)


def test_for_test_error_names_the_fallback(filled_monthly_on_daily):
    # the error must say what to do instead, otherwise it will be worked around
    with pytest.raises(StaleDataInTest, match="align_for_test"):
        for_test(filled_monthly_on_daily)


def test_for_test_accepts_data_that_was_never_filled():
    source = pd.Series([10.0, 11.0, 12.0], index=pd.date_range("2024-01-01", periods=3, freq="D"))
    df = ffill_with_provenance(source, pd.date_range("2024-01-01", periods=3, freq="D"))
    assert for_test(df).tolist() == [10.0, 11.0, 12.0]


def test_provenance_frame_shape_is_enforced():
    with pytest.raises(ResampleError, match="ffill_with_provenance"):
        for_test(pd.DataFrame({"value": [1.0]}))


# ===========================================================================
# RULE C
# ===========================================================================
def test_effective_n_basic():
    # 18 observations, overlap 3 -> n_eff = 6
    out = effective_n(18, 3)
    assert out.n_eff == pytest.approx(6.0)
    assert out.shrinkage == pytest.approx(3.0)


def test_effective_n_rolling_window_is_the_overlap():
    # 250 days in a 30-day rolling window -> n_eff = 8.33
    assert effective_n_rolling(250, 30).n_eff == pytest.approx(8.3333, abs=1e-3)


def test_effective_n_refuses_overlap_below_one():
    with pytest.raises(ResampleError, match="overlap"):
        effective_n(18, 0.5)


def test_average_concurrency_hand_computed():
    """Three entries spaced 10 days apart, held 30 days.

    Day-by-day coverage from the first entry:
        days  0-9  : 1 position open
        days 10-19 : 2
        days 20-29 : 3
        days 30-39 : 2
        days 40-49 : 1
    Sum = 3 trades x 30 days = 90 position-days over 50 days -> average 1.8.
    """
    entries = pd.to_datetime(["2024-01-01", "2024-01-11", "2024-01-21"])
    assert average_concurrency(entries, hold_days=30) == pytest.approx(1.8)


def test_effective_n_from_trades_hand_computed():
    # 3 trades / average overlap 1.8 = 1.667
    entries = pd.to_datetime(["2024-01-01", "2024-01-11", "2024-01-21"])
    assert effective_n_from_trades(entries, hold_days=30).n_eff == pytest.approx(1.6667, abs=1e-3)


def test_non_overlapping_trades_keep_their_full_sample():
    # entries spaced 30 days apart, held 30 days: no overlap, n_eff = n
    entries = pd.to_datetime(["2024-01-01", "2024-01-31", "2024-03-01"])
    out = effective_n_from_trades(entries, hold_days=30)
    assert out.overlap == pytest.approx(1.0)
    assert out.n_eff == pytest.approx(3.0)


def test_copper_backtest_shape_lands_in_the_predicted_range():
    """The T1-M module's number, on the copper backtest's shape.

    18 positions held 30 days, entries every 11 days:
        span     = 17 x 11 + 30 = 217 days covered
        sum      = 18 x 30 = 540 position-days
        average  = 540 / 217 = 2.4885 positions open at once
        n_eff    = 18 / 2.4885 = 7.23

    The spec announced "on the order of 6-8, not 18." That's what changes how a 100%
    win rate reads: on ~7 independent draws, it isn't distinguishable from a process
    that succeeds 70% of the time.
    """
    entries = pd.date_range("2024-01-01", periods=18, freq="11D")
    out = effective_n_from_trades(entries, hold_days=30)
    assert out.overlap == pytest.approx(2.4885, abs=1e-3)
    assert out.n_eff == pytest.approx(7.233, abs=1e-2)
    assert 6.0 <= out.n_eff <= 8.0


def test_effective_sample_summary_is_readable():
    summary = EffectiveSample(n_obs=18, overlap=2.5, n_eff=7.2).summary
    assert "n_eff = 7.2" in summary
    assert "2.5x less" in summary
