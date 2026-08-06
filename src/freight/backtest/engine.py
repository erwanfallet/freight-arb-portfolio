"""Minimal backtest engine — a skeleton, not a strategy.

Per PROJECT_NOTES.md / Partie 4.1: the H1 test's primary metric is the switching-share
in signals/switching.py, not a P&L backtest. This engine exists for H2/H3 once those are
reached, and for any downstream "does this actually make money" sanity check — kept
deliberately thin.

Limits declared up front (Partie 8): no credit constraint, no margin, no position limit.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    pnl_per_unit: float
    costs_per_unit: float

    @property
    def net_pnl_per_unit(self) -> float:
        return self.pnl_per_unit - self.costs_per_unit


def run_backtest(decision: pd.Series, arb: pd.Series, cost_per_unit: float = 0.0) -> list[Trade]:
    """decision: boolean series ("ship today"), True->False and False->True transitions
    are treated as entering/exiting a position. arb: the $/t signal being captured.
    Both must share a date index.
    """
    aligned = pd.concat({"decision": decision, "arb": arb}, axis=1).dropna().sort_index()
    trades: list[Trade] = []
    entry_date = None
    entry_arb = None
    for date, row in aligned.iterrows():
        if row["decision"] and entry_date is None:
            entry_date, entry_arb = date, row["arb"]
        elif not row["decision"] and entry_date is not None:
            trades.append(
                Trade(
                    entry_date=entry_date,
                    exit_date=date,
                    pnl_per_unit=row["arb"] - entry_arb,
                    costs_per_unit=cost_per_unit,
                )
            )
            entry_date, entry_arb = None, None
    return trades


def summarize(trades: list[Trade]) -> dict:
    if not trades:
        return {"n_trades": 0, "total_pnl": 0.0, "avg_pnl": 0.0, "hit_rate": float("nan")}
    pnls = [t.net_pnl_per_unit for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    return {
        "n_trades": len(trades),
        "total_pnl": sum(pnls),
        "avg_pnl": sum(pnls) / len(pnls),
        "hit_rate": wins / len(pnls),
    }
