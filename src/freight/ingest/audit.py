"""Data audit — Partie 5.3 Etape 1.

Not a formality: this step decides which hypotheses are even testable. Produces a
per-ticker table of coverage, gaps, and staleness so failures like the BSI-frozen-since-
2017-shown-as-current case get caught before anything downstream trusts the series.

Deliberately does not fill, smooth, or interpolate anything — it only reports.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from freight.ingest.contract import load_dictionary
from freight.ingest.loader import load_raw_directory

_FREQ_TO_PANDAS = {
    "daily": "B",  # business days — freight/commodity markets don't trade weekends
    "weekly": "W",
    "monthly": "MS",
    "quarterly": "QS",
    "yearly": "YS",
}

# a daily series with no fresh observation in this many days is presumed stale unless
# the dictionary explicitly says so via last_valid_date
STALE_THRESHOLD_DAYS = {
    "daily": 10, "weekly": 21, "monthly": 60, "quarterly": 150, "yearly": 400, "irregular": None,
}


def coverage_report(long_df: pd.DataFrame, dictionary: dict, *, today: date | None = None) -> pd.DataFrame:
    today = today or datetime.utcnow().date()
    rows = []
    seen_tickers = set(long_df["ticker"].unique()) | set(dictionary.keys())

    for ticker in sorted(seen_tickers):
        series = long_df[long_df["ticker"] == ticker]
        contract = dictionary.get(ticker)
        flags: list[str] = []

        if contract is None:
            flags.append("unknown_ticker_not_in_dictionary")
        if series.empty:
            flags.append("no_data_in_raw")
            rows.append(_row(ticker, contract, 0, None, None, None, today, flags))
            continue

        dates = pd.to_datetime(series["date"]).sort_values()
        first_obs, last_obs = dates.iloc[0].date(), dates.iloc[-1].date()
        n_obs = len(dates)

        n_gaps = None
        if contract is not None and contract.frequency in _FREQ_TO_PANDAS:
            expected = pd.date_range(first_obs, last_obs, freq=_FREQ_TO_PANDAS[contract.frequency])
            n_gaps = max(len(expected) - n_obs, 0)

        staleness_days = (today - last_obs).days
        threshold = STALE_THRESHOLD_DAYS.get(contract.frequency) if contract else None
        if threshold is not None and staleness_days > threshold:
            if contract.last_valid_date is None or contract.last_valid_date != last_obs:
                flags.append(
                    f"stale_{staleness_days}d_undeclared"
                    if contract.last_valid_date is None
                    else "stale_but_declared_mismatch"
                )
        if contract is not None and contract.last_valid_date == last_obs and staleness_days > (threshold or 0):
            flags.append("stale_but_declared")  # informational, not an error

        rows.append(_row(ticker, contract, n_obs, first_obs, last_obs, n_gaps, today, flags))

    report = pd.DataFrame(rows)
    return report.sort_values("ticker").reset_index(drop=True)


def _row(ticker, contract, n_obs, first_obs, last_obs, n_gaps, today, flags):
    staleness = (today - last_obs).days if last_obs else np.nan
    return {
        "ticker": ticker,
        "description": contract.description if contract else None,
        "native_unit": contract.native_unit if contract else None,
        "frequency": contract.frequency if contract else None,
        "n_obs": n_obs,
        "first_obs": first_obs,
        "last_obs": last_obs,
        "n_gaps_vs_expected": n_gaps,
        "staleness_days": staleness,
        "flags": ";".join(flags) if flags else "",
        "usable": len(flags) == 0 or all(f == "stale_but_declared" for f in flags),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--dictionary", default="data_dictionary.csv")
    parser.add_argument("--out", default="data/interim/audit_report.csv")
    args = parser.parse_args()

    long_df = load_raw_directory(args.raw)
    dictionary = load_dictionary(args.dictionary)
    report = coverage_report(long_df, dictionary)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_path, index=False)

    n_unusable = (~report["usable"]).sum()
    print(f"Audit written to {out_path} — {len(report)} tickers, {n_unusable} flagged unusable")
    if n_unusable:
        print(report.loc[~report["usable"], ["ticker", "flags"]].to_string(index=False))


if __name__ == "__main__":
    main()
