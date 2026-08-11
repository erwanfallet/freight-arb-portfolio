"""Golden tests for the T1-M module (regime or skill).

The central test is `test_copper_backtest_self_criticism_sentence`: it produces,
literally, the sentence the spec asks to be written in the email.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agri.core.regime import (
    RegimeError,
    attribute_pnl_to_regime,
    honest_win_rate,
)


# ===========================================================================
# Honest win rate — the number for the email
# ===========================================================================
def test_copper_backtest_self_criticism_sentence():
    """The shape of the copper backtest: 18 positions, 30-day hold, entries every 11 days.

    Average overlap 2.49 -> n_eff = 7.23 -> conservatively rounded to 7 draws.
    18 wins out of 18 becomes 7 wins out of 7, whose exact lower bound is 59.0%.
    """
    entries = pd.date_range("2024-01-01", periods=18, freq="11D")
    out = honest_win_rate(entries, [True] * 18, hold_days=30)

    assert out.n_trades == 18
    assert out.sample.n_eff == pytest.approx(7.233, abs=1e-2)
    assert out.naive.point == pytest.approx(1.0)
    assert out.naive.lo == pytest.approx(0.81470, abs=1e-4)
    assert out.honest.lo == pytest.approx(0.59043, abs=1e-4)
    # ignoring the overlap would overstate the lower bound by 22 points
    assert out.lower_bound_cost == pytest.approx(0.2243, abs=1e-3)

    sentence = out.mail_sentence
    assert "18 positions" in sentence
    assert "7.2 independent draws" in sentence
    assert "6 times out of 10" in sentence
    assert "22 points" in sentence


def test_non_overlapping_trades_cost_nothing():
    # entries spaced exactly one holding period apart: no overlap to correct
    entries = pd.date_range("2024-01-01", periods=10, freq="30D")
    out = honest_win_rate(entries, [True] * 10, hold_days=30)
    assert out.sample.overlap == pytest.approx(1.0)
    assert out.sample.n_eff == pytest.approx(10.0)
    assert out.lower_bound_cost == pytest.approx(0.0, abs=1e-9)


def test_win_share_is_preserved_not_the_raw_count():
    """Overlap reduces the number of observations, not the success rate."""
    entries = pd.date_range("2024-01-01", periods=18, freq="11D")
    wins = [True] * 12 + [False] * 6  # 2/3 success
    out = honest_win_rate(entries, wins, hold_days=30)
    assert out.naive.point == pytest.approx(12 / 18)
    # 2/3 carried over onto 7 draws -> round(0.667 * 7) = 5 wins out of 7
    assert out.honest.successes == 5
    assert out.honest.point == pytest.approx(5 / 7)


def test_shorter_hold_preserves_more_information():
    entries = pd.date_range("2024-01-01", periods=18, freq="11D")
    long_hold = honest_win_rate(entries, [True] * 18, hold_days=30)
    short_hold = honest_win_rate(entries, [True] * 18, hold_days=5)
    assert short_hold.sample.n_eff > long_hold.sample.n_eff
    assert short_hold.honest.lo > long_hold.honest.lo


def test_mismatched_lengths_raise():
    entries = pd.date_range("2024-01-01", periods=5, freq="D")
    with pytest.raises(RegimeError, match="trade for trade"):
        honest_win_rate(entries, [True] * 3, hold_days=10)


# ===========================================================================
# Attributing P&L to the regime
# ===========================================================================
def _synthetic_trades(n: int, *, alpha: float, seed: int, freq: str = "11D") -> pd.DataFrame:
    """P&L built to be EXACTLY the regime plus a known constant.

    pnl = alpha + 2*vol + 1.5*width - 0.5*dispersion + 0.1*duration + noise
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq=freq)
    vol = rng.uniform(0.1, 0.5, n)
    width = rng.uniform(-2.0, 2.0, n)
    dispersion = rng.uniform(0.0, 3.0, n)
    duration = rng.uniform(20.0, 40.0, n)
    pnl = alpha + 2.0 * vol + 1.5 * width - 0.5 * dispersion + 0.1 * duration
    pnl = pnl + rng.normal(scale=0.2, size=n)
    return pd.DataFrame(
        {
            "pnl": pnl,
            "vol_realised": vol,
            "term_structure_width": width,
            "dispersion": dispersion,
            "duration": duration,
        },
        index=idx,
    )


def test_regime_regression_recovers_known_coefficients():
    trades = _synthetic_trades(400, alpha=0.0, seed=20, freq="3D")
    out = attribute_pnl_to_regime(trades, hold_days=3)
    assert out.regression.params["vol_realised"] == pytest.approx(2.0, abs=0.2)
    assert out.regression.params["term_structure_width"] == pytest.approx(1.5, abs=0.1)
    assert out.regression.params["dispersion"] == pytest.approx(-0.5, abs=0.1)
    assert out.r_squared > 0.9


def test_pure_regime_pnl_yields_an_alpha_indistinguishable_from_zero():
    """A P&L entirely manufactured by the regime must not produce an alpha."""
    trades = _synthetic_trades(400, alpha=0.0, seed=21, freq="3D")
    out = attribute_pnl_to_regime(trades, hold_days=3, n_iter=2000)
    assert not out.alpha_is_distinguishable_from_zero
    assert out.verdict == "indistinguishable from the regime"
    assert "can't be distinguished" in out.mail_sentence


def test_genuine_alpha_is_detected():
    trades = _synthetic_trades(400, alpha=5.0, seed=22, freq="3D")
    out = attribute_pnl_to_regime(trades, hold_days=3, n_iter=2000)
    assert out.alpha == pytest.approx(5.0, abs=0.3)
    assert out.alpha_is_distinguishable_from_zero
    assert out.verdict == "presumed alpha"


def test_eighteen_overlapping_trades_are_flagged_uninterpretable():
    """The copper backtest case: 5 parameters for ~7 independent observations.

    R² will be flattering and meaningless. The module must say so itself instead of
    just displaying the number — that's the "the result is shown even when it's
    unfavourable" line in the checklist.
    """
    trades = _synthetic_trades(18, alpha=3.0, seed=23, freq="11D")
    out = attribute_pnl_to_regime(trades, hold_days=30, n_iter=1000)
    assert out.sample.n_eff == pytest.approx(7.233, abs=1e-2)
    assert out.is_overfit
    assert out.verdict == "not interpretable"
    assert "model's flexibility" in out.mail_sentence


def test_missing_regime_column_raises_with_guidance():
    trades = _synthetic_trades(50, alpha=1.0, seed=24, freq="3D").drop(columns="dispersion")
    with pytest.raises(RegimeError, match="regime_columns"):
        attribute_pnl_to_regime(trades, hold_days=3)


def test_custom_regime_columns_are_supported():
    """Each page names its own variables: cross-origin dispersion for a grain."""
    trades = _synthetic_trades(200, alpha=1.0, seed=25, freq="3D").rename(
        columns={"dispersion": "cross_origin_dispersion"}
    )
    out = attribute_pnl_to_regime(
        trades,
        hold_days=3,
        regime_columns=("vol_realised", "term_structure_width", "cross_origin_dispersion"),
        n_iter=1000,
    )
    assert "cross_origin_dispersion" in out.regime_columns
    assert out.r_squared > 0.8


def test_missing_pnl_column_raises():
    trades = _synthetic_trades(50, alpha=1.0, seed=26, freq="3D").rename(columns={"pnl": "profit"})
    with pytest.raises(RegimeError, match="missing P&L column"):
        attribute_pnl_to_regime(trades, hold_days=3)
