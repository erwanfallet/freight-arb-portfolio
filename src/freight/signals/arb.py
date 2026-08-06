"""Arb reconstruction — Partie 2.4.

arb = price_destination - price_origin - freight - other_transfer_costs

Not a risk-free arbitrage in the academic sense (3-8 weeks of transit risk) — a
relative-value trade with real execution risk. Nothing here treats it otherwise.
"""
from __future__ import annotations

import pandas as pd


def reconstruct_arb(
    price_destination: pd.Series,
    price_origin: pd.Series,
    freight: pd.Series,
    other_costs: pd.Series | float = 0.0,
) -> pd.Series:
    """All inputs indexed by date (already unit-aligned — that alignment is the
    ingestion layer's job, not this function's). Returns the arb series, same index.
    """
    aligned = pd.concat(
        {"price_destination": price_destination, "price_origin": price_origin, "freight": freight},
        axis=1,
    ).dropna()
    other = other_costs if isinstance(other_costs, pd.Series) else pd.Series(other_costs, index=aligned.index)
    arb = aligned["price_destination"] - aligned["price_origin"] - aligned["freight"] - other
    arb.name = "arb"
    return arb


def price_component(
    price_destination: pd.Series,
    price_origin: pd.Series,
    other_costs: pd.Series | float = 0.0,
) -> pd.Series:
    """The non-freight side of the arb — used by signals/switching.py to decompose
    each sign flip into a price-driven vs freight-driven move.
    """
    other = other_costs if isinstance(other_costs, pd.Series) else pd.Series(other_costs, index=price_destination.index)
    aligned = pd.concat({"pd": price_destination, "po": price_origin, "other": other}, axis=1).dropna()
    component = aligned["pd"] - aligned["po"] - aligned["other"]
    component.name = "price_component"
    return component


def freight_value_ratio(freight: pd.Series, cargo_value: pd.Series) -> float:
    """Mean freight as a share of mean cargo value — the classic (and, per H1, possibly
    misleading) criterion for where freight risk is assumed to matter.
    """
    aligned = pd.concat({"freight": freight, "value": cargo_value}, axis=1).dropna()
    return float(aligned["freight"].mean() / aligned["value"].mean())
