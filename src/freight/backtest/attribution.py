"""P&L attribution: for a set of trades, split net P&L into the freight-move
contribution vs the price-move contribution, using the same held-fixed counterfactual
as signals/switching.py — kept consistent so the H1 narrative and any backtest P&L
attribution tell the same story with the same method.
"""
from __future__ import annotations

import pandas as pd

from freight.backtest.engine import Trade


def attribute_trade(trade: Trade, price_component: pd.Series, freight: pd.Series) -> dict:
    price_entry = price_component.loc[trade.entry_date]
    price_exit = price_component.loc[trade.exit_date]
    freight_entry = freight.loc[trade.entry_date]
    freight_exit = freight.loc[trade.exit_date]

    price_contribution = price_exit - price_entry
    freight_contribution = -(freight_exit - freight_entry)  # freight is subtracted in the arb

    return {
        "entry_date": trade.entry_date,
        "exit_date": trade.exit_date,
        "net_pnl_per_unit": trade.net_pnl_per_unit,
        "price_contribution": price_contribution,
        "freight_contribution": freight_contribution,
    }


def attribute_trades(trades: list[Trade], price_component: pd.Series, freight: pd.Series) -> pd.DataFrame:
    rows = [attribute_trade(t, price_component, freight) for t in trades]
    return pd.DataFrame(rows)
