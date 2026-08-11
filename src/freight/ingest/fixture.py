"""SYNTHETIC DATA GENERATOR — NO ECONOMIC VALUE.

Purpose: prove the pipeline runs and the dashboard's six sections render, before the
real series arrive. Nothing this module produces should leave the repo, appear in an
email, or be read as a result.

Three guardrails:
  1. every ticker is prefixed `SYNTH_`
  2. the function returns a DataFrame with the attribute `.attrs["synthetic"] = True`
  3. the dashboard shows a red banner as long as that attribute is true

The chosen levels are on the order of market magnitude (P62 around 100 $/dmt, C3 around
20 $/wmt, C5 around 10 $/wmt) purely so the chart axes stay readable. They are not
calibrated on anything.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SYNTHETIC_TICKERS = {
    "p62": "SYNTH_IRON62_CFR",
    "p65": "SYNTH_IRON65_CFR",
    "c3": "SYNTH_C3_TUBARAO_QINGDAO",
    "c5": "SYNTH_C5_WA_QINGDAO",
}


def synthetic_ironore(n_days: int = 520, seed: int = 42) -> pd.DataFrame:
    """Four synthetic series in the canonical long format (date, ticker, valeur).

    The structure is deliberately imposed: a common Capesize factor pushes C3 and C5
    together, C3 reacts more strongly (longer haul, more bunker-sensitive), and the
    65-62 premium contains an independent quality component. That's the structure the
    decomposition is supposed to recover — so this dataset tests the plumbing, not the
    thesis. It cannot confirm it.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n_days)

    cape_factor = np.cumsum(rng.normal(0, 0.45, n_days))
    c5 = 10.0 + 0.6 * cape_factor + rng.normal(0, 0.25, n_days)
    c3 = 20.0 + 1.5 * cape_factor + rng.normal(0, 0.55, n_days)
    c3 = np.clip(c3, 6.0, None)
    c5 = np.clip(c5, 3.0, None)

    p62 = 100.0 + np.cumsum(rng.normal(0, 1.1, n_days))
    quality = 6.0 + np.cumsum(rng.normal(0, 0.22, n_days))
    p65 = p62 + quality + (c3 / 0.91 - c5 / 0.92) + rng.normal(0, 0.4, n_days)

    frames = []
    for role, values in (("p62", p62), ("p65", p65), ("c3", c3), ("c5", c5)):
        frames.append(
            pd.DataFrame(
                {"date": idx, "ticker": SYNTHETIC_TICKERS[role], "valeur": values}
            )
        )
    out = pd.concat(frames, ignore_index=True)

    # deliberate gaps: holidays misaligned between freight and prices, so calendar
    # handling gets exercised even in synthetic mode.
    holidays = set(rng.choice(n_days, size=12, replace=False))
    mask = out.apply(
        lambda r: (idx.get_loc(r["date"]) in holidays)
        and r["ticker"].startswith("SYNTH_C"),
        axis=1,
    )
    out = out[~mask].reset_index(drop=True)

    out.attrs["synthetic"] = True
    return out
