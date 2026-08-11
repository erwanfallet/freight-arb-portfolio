"""Converts the canonical long format (date, ticker, valeur) into date-indexed series.

The long format stays the upstream source of truth: every ticker keeps its own
calendar and gaps stay visible. Conversion to series happens as late as possible, right
before the calculation, and alignment is done by explicit intersection in the chain
module — never by a silent `reindex().ffill()`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from freight.ingest.loader import load_raw_directory


class MissingSeries(Exception):
    """An expected ticker is not present in data/raw/."""


def to_series(df_long: pd.DataFrame, ticker: str) -> pd.Series:
    """Extracts one ticker from the long format. Raises if absent — no silent empty return."""
    rows = df_long[df_long["ticker"] == ticker]
    if rows.empty:
        available = sorted(df_long["ticker"].unique())
        raise MissingSeries(
            f"ticker '{ticker}' not found in data/raw/. Available tickers: {available}"
        )
    s = rows.set_index("date")["valeur"].astype(float).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s.name = ticker
    return s


def load_series_map(raw_dir: str | Path, tickers: dict[str, str]) -> dict[str, pd.Series]:
    """Loads data/raw/ and returns {role: series} from a {role: ticker} mapping.

    The role -> ticker mapping lives in the chain module, not here: it's a modelling
    decision ("which ticker plays the role of the 65% Fe price"), not an ingestion
    decision.
    """
    raw = load_raw_directory(raw_dir)
    if raw.empty:
        raise MissingSeries(f"{raw_dir} contains no usable CSV")
    return {role: to_series(raw, ticker) for role, ticker in tickers.items()}


def coverage_report(series_map: dict[str, pd.Series]) -> pd.DataFrame:
    """Coverage table: to display at the top of the dashboard, before any chart.

    A series being present doesn't make it usable. This table makes the first and last
    observation, the point count, and the number of missing business days over the
    range visible — without filling in anything.
    """
    rows = []
    for role, s in series_map.items():
        if s.empty:
            rows.append({"role": role, "ticker": s.name, "n_obs": 0})
            continue
        business_days = pd.bdate_range(s.index.min(), s.index.max())
        rows.append(
            {
                "role": role,
                "ticker": s.name,
                "first_obs": s.index.min().date(),
                "last_obs": s.index.max().date(),
                "n_obs": int(s.notna().sum()),
                "business_days_in_range": len(business_days),
                "business_day_gaps": int(len(business_days) - s.notna().sum()),
            }
        )
    return pd.DataFrame(rows)
