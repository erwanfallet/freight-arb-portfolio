"""Load raw Bloomberg exports into the canonical long format: date, ticker, valeur.

Long format over wide (Partie 6.6): tickers have different calendars (holidays,
quarterly fundamentals vs daily indices), and wide format forces an artificial join
that hides gaps. Long format keeps every gap visible until the audit step decides
what, if anything, to do with it — which for this project is usually "nothing".
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CANONICAL_COLUMNS = ["date", "ticker", "valeur"]


def load_long_csv(path: str | Path) -> pd.DataFrame:
    """Load a file already in (date, ticker, valeur) form. No coercion of missing values."""
    df = pd.read_csv(path)
    rename = {}
    lower_cols = {c.lower(): c for c in df.columns}
    for canonical in CANONICAL_COLUMNS:
        if canonical not in df.columns:
            for alias in _ALIASES[canonical]:
                if alias in lower_cols:
                    rename[lower_cols[alias]] = canonical
                    break
    df = df.rename(columns=rename)
    missing = set(CANONICAL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: cannot map to canonical columns, missing {missing}")
    df = df[CANONICAL_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.strip()
    return df


def load_wide_csv(path: str | Path, date_col: str = "date") -> pd.DataFrame:
    """Melt a wide Bloomberg export (one column per ticker) into canonical long format."""
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"{path}: expected a '{date_col}' column, got {list(df.columns)}")
    long_df = df.melt(id_vars=[date_col], var_name="ticker", value_name="valeur")
    long_df = long_df.rename(columns={date_col: "date"})
    long_df["date"] = pd.to_datetime(long_df["date"])
    long_df["ticker"] = long_df["ticker"].astype(str).str.strip()
    long_df = long_df.dropna(subset=["valeur"])
    return long_df[CANONICAL_COLUMNS]


def load_raw_directory(raw_dir: str | Path) -> pd.DataFrame:
    """Load and concatenate every CSV in data/raw/, wide or long, auto-detected.

    Auto-detection: a file is treated as long-format if it already has ticker/valeur-like
    columns, wide-format otherwise (one column per ticker, first column is the date).
    """
    raw_dir = Path(raw_dir)
    frames = []
    for csv_path in sorted(raw_dir.glob("*.csv")):
        cols_lower = {c.lower() for c in pd.read_csv(csv_path, nrows=0).columns}
        looks_long = bool(cols_lower & set(_ALIASES["ticker"])) and bool(
            cols_lower & set(_ALIASES["valeur"])
        )
        frame = load_long_csv(csv_path) if looks_long else load_wide_csv(csv_path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["date", "ticker"], keep="last")
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def write_interim(df: pd.DataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


_ALIASES = {
    "date": {"date", "dates", "asof", "as_of_date"},
    "ticker": {"ticker", "tickers", "symbol", "field"},
    "valeur": {"valeur", "value", "px_last", "last", "price"},
}
