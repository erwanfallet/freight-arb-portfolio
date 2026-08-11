"""SYNTHETIC DATA GENERATOR FOR PROJECT B — NO ECONOMIC VALUE.

A stronger warning than for project A, and it needs to be read.

Project A claims that part of the premium comes from freight; synthetic set A imposes a
structure of that kind, so it cannot confirm it. Here it's worse: **the 2022 break is
imposed by hand in the generator**. The dashboard will therefore show a freight
coefficient that collapses after 2022 and an arb that breaks — because it's written
into these fifteen lines of code, not because the market did it.

This dataset exists solely to verify the six sections render and that the break test
with the TTF control runs. Any economic reading of these charts is a mistake.

Tickers prefixed `SYNTH_`, attribute `.attrs["synthetic"] = True`, red banner in the
dashboard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SYNTHETIC_TICKERS = {
    "api2": "SYNTH_API2_CIF_ARA",
    "api4": "SYNTH_API4_FOB_RB",
    "freight": "SYNTH_C4_RB_ROTTERDAM",
    "ttf": "SYNTH_TTF",
    "eua": "SYNTH_EUA",
    "eurusd": "SYNTH_EURUSD",
}

BREAKPOINT = pd.Timestamp("2022-03-01")


def synthetic_coal(seed: int = 11) -> pd.DataFrame:
    """Six synthetic series in the canonical long format (date, ticker, valeur).

    Imposed structure, to know before looking at a single chart:
      - before the break: API2−API4 spread ≈ freight, so the arb sits around zero
      - after: the spread keeps a clean level independent of freight
      - a TTF shock in spring 2022, correlated with the spread's level, so the
        attribution test has something to untangle
      - EUA is zero before 2024 only in the regulatory facts, not in the series: the
        allowance price exists before then, it's the phase-in that neutralises it
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", "2026-06-30")
    n = len(idx)
    post = (idx >= BREAKPOINT).astype(float)

    freight = 12.0 + np.cumsum(rng.normal(0, 0.12, n))
    freight = np.clip(freight, 5.0, 40.0)

    gas_shock = np.exp(-(((np.arange(n) - int(post.argmax()) - 60) / 90.0) ** 2))
    ttf = 20.0 + 90.0 * gas_shock + np.cumsum(rng.normal(0, 0.35, n))
    ttf = np.clip(ttf, 5.0, None)

    api4 = 70.0 + np.cumsum(rng.normal(0, 0.55, n)) + 0.25 * (ttf - 20.0)
    api4 = np.clip(api4, 30.0, None)

    # before: the spread pays for freight. after: it keeps a clean level.
    spread = np.where(
        post > 0,
        0.15 * freight + 18.0 + 0.05 * (ttf - 20.0) + rng.normal(0, 1.2, n),
        1.0 * freight + rng.normal(0, 0.9, n),
    )
    api2 = api4 + spread

    eua = 55.0 + np.cumsum(rng.normal(0, 0.35, n))
    eua = np.clip(eua, 15.0, None)
    eurusd = 1.10 + np.cumsum(rng.normal(0, 0.0018, n))
    eurusd = np.clip(eurusd, 0.95, 1.30)

    frames = []
    for role, values in (
        ("api2", api2), ("api4", api4), ("freight", freight),
        ("ttf", ttf), ("eua", eua), ("eurusd", eurusd),
    ):
        frames.append(
            pd.DataFrame({"date": idx, "ticker": SYNTHETIC_TICKERS[role], "valeur": values})
        )
    out = pd.concat(frames, ignore_index=True)
    out.attrs["synthetic"] = True
    return out
