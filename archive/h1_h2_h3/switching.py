"""Sign-flip detection and attribution — the H1 test engine (Partie 4.1 / 5.3 Etape 2).

For each date where the arb's sign flips relative to the prior observation, decompose
the move into a price-driven component and a freight-driven component using a
counterfactual: replay the flip holding one side fixed at its previous value. If only
the freight-held-fixed counterfactual would have avoided the flip, freight caused it,
and vice versa. If either counterfactual alone reproduces the flip, both moved it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def detect_and_attribute_flips(price_component: pd.Series, freight: pd.Series) -> pd.DataFrame:
    """price_component and freight must share a date index (see signals/arb.py).

    Returns one row per sign flip: date, prev_sign, new_sign, attribution in
    {"freight", "price", "both", "ambiguous"}.
    """
    aligned = pd.concat({"price_component": price_component, "freight": freight}, axis=1).dropna()
    aligned = aligned.sort_index()
    arb = aligned["price_component"] - aligned["freight"]

    rows = []
    prev_date = None
    for date, value in arb.items():
        if prev_date is None:
            prev_date = date
            continue
        prev_sign = _sign(arb.loc[prev_date])
        new_sign = _sign(value)
        if new_sign != prev_sign and prev_sign != 0:
            price_t = aligned.loc[date, "price_component"]
            price_prev = aligned.loc[prev_date, "price_component"]
            freight_t = aligned.loc[date, "freight"]
            freight_prev = aligned.loc[prev_date, "freight"]

            freight_only_sign = _sign(price_prev - freight_t)   # freight moves, price held
            price_only_sign = _sign(price_t - freight_prev)     # price moves, freight held

            caused_by_freight = freight_only_sign != prev_sign
            caused_by_price = price_only_sign != prev_sign

            if caused_by_freight and not caused_by_price:
                attribution = "freight"
            elif caused_by_price and not caused_by_freight:
                attribution = "price"
            elif caused_by_freight and caused_by_price:
                attribution = "both"
            else:
                attribution = "ambiguous"  # combined move flips it, neither alone does

            rows.append(
                {
                    "date": date,
                    "prev_sign": prev_sign,
                    "new_sign": new_sign,
                    "attribution": attribution,
                }
            )
        prev_date = date

    return pd.DataFrame(rows, columns=["date", "prev_sign", "new_sign", "attribution"])


def freight_attributable_share(flips: pd.DataFrame, *, count_both_as_half: bool = True) -> float:
    """Share of flips attributable to freight — the H1 metric that replaces the
    freight/cargo-value ratio. "both" flips count as half by default (they're genuinely
    joint causation); set count_both_as_half=False to count them as fully freight-caused
    (upper bound) for a sensitivity check.
    """
    if flips.empty:
        return float("nan")
    weight = {"freight": 1.0, "price": 0.0, "both": 0.5 if count_both_as_half else 1.0, "ambiguous": 0.5}
    return float(np.mean(flips["attribution"].map(weight)))
