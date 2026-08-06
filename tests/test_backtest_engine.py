import pandas as pd
import pytest

from freight.backtest.engine import run_backtest, summarize

DATES = pd.date_range("2024-01-01", periods=5, freq="D")


def test_single_round_trip_trade():
    decision = pd.Series([False, True, True, False, False], index=DATES)
    arb = pd.Series([1.0, 2.0, 3.0, 4.0, 1.0], index=DATES)
    trades = run_backtest(decision, arb, cost_per_unit=0.5)
    assert len(trades) == 1
    t = trades[0]
    assert t.entry_date == DATES[1]
    assert t.exit_date == DATES[3]
    assert t.pnl_per_unit == pytest.approx(4.0 - 2.0)
    assert t.net_pnl_per_unit == pytest.approx(4.0 - 2.0 - 0.5)


def test_summarize_empty():
    summary = summarize([])
    assert summary["n_trades"] == 0
    assert summary["total_pnl"] == 0.0


def test_summarize_hit_rate():
    decision = pd.Series([True, False, False, False], index=DATES[:4])
    arb = pd.Series([1.0, 2.0, 5.0, 4.0], index=DATES[:4])
    trades = run_backtest(decision, arb)
    summary = summarize(trades)
    assert summary["n_trades"] == 1
    assert summary["hit_rate"] == 1.0  # 2.0 - 1.0 > 0


def test_summarize_multiple_round_trips():
    decision = pd.Series([True, False, True, False], index=DATES[:4])
    arb = pd.Series([1.0, 2.0, 5.0, 4.0], index=DATES[:4])
    trades = run_backtest(decision, arb)
    summary = summarize(trades)
    assert summary["n_trades"] == 2
    assert summary["hit_rate"] == 0.5  # trade1: +1 win, trade2: -1 loss
