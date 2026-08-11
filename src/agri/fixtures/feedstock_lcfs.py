"""T3-1 synthetic dataset.

Two properties are **imposed**, and the golden tests verify the engine recovers them:

1. the LCFS price crosses the `LCFS*` threshold partway through the sample, so the
   winning pathway changes — otherwise the heatmap would have only one zone and the
   page would have nothing to say;
2. soyoil's energy beta to Brent **doubles** on the RVO finalisation date (March 2026),
   so the Chow test has something to detect and it can be checked that it doesn't
   detect anything elsewhere.

Tickers prefixed `SYNTH_`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_START = "2023-01-02"
# ~4.4 years of business days, through mid-2027. Sized to leave more than 300 business
# days AFTER the March 2026 break: without that, no 120-day rolling window sits
# entirely in the post-break regime, and the "after" beta can't be read.
DEFAULT_PERIODS = 1150

# Policy date: EPA finalisation of the 2026-27 RVOs.
RVO_BREAK_DATE = pd.Timestamp("2026-03-16")

BETA_BEFORE_BREAK = 0.20       # soyoil loosely tracked crude
BETA_AFTER_BREAK = 0.45        # the biofuel pull strengthens


def build(
    *, start: str = DEFAULT_START, periods: int = DEFAULT_PERIODS, seed: int = 0
) -> dict[str, pd.Series]:
    """T3-1's input series: ULSD, RIN D4, LCFS credit, soyoil, Brent."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=periods, freq="B")
    n = len(index)

    # --- Brent: random walk with a geopolitical escalation in early 2026 ---
    brent_returns = rng.normal(scale=0.014, size=n)
    escalation = (index >= pd.Timestamp("2026-01-15")) & (index <= pd.Timestamp("2026-02-28"))
    brent_returns[escalation] += 0.004
    brent = pd.Series(78.0 * np.exp(np.cumsum(brent_returns)), index=index, name="SYNTH_BRENT")

    # --- Soyoil: beta to Brent that DOUBLES on the RVO date (for the Chow test) ---
    betas = np.where(index >= RVO_BREAK_DATE, BETA_AFTER_BREAK, BETA_BEFORE_BREAK)
    soyoil_returns = betas * brent_returns + rng.normal(scale=0.010, size=n)
    soyoil = pd.Series(
        0.52 * np.exp(np.cumsum(soyoil_returns)), index=index, name="SYNTH_SOYOIL_USD_LB"
    )

    # --- LCFS: crosses the threshold partway through the sample ---
    # ramp from 60 to 340 $/t with noise: the price-parity threshold is at ~285 $/t,
    # so the winning pathway flips somewhere in the middle of the sample.
    ramp = np.linspace(60.0, 340.0, n)
    lcfs = pd.Series(
        np.clip(ramp + rng.normal(scale=12.0, size=n), 5.0, None),
        index=index,
        name="SYNTH_LCFS_USD_T",
    )

    # --- ULSD and RIN D4: plausible levels, lightly noised ---
    ulsd = pd.Series(
        2.55 * np.exp(np.cumsum(0.6 * brent_returns + rng.normal(scale=0.006, size=n))),
        index=index,
        name="SYNTH_ULSD_USD_GAL",
    )
    rin_d4 = pd.Series(
        np.clip(0.62 + np.cumsum(rng.normal(scale=0.004, size=n)), 0.15, 2.0),
        index=index,
        name="SYNTH_RIN_D4_USD",
    )

    # --- Imported UCO: discounted against soyoil, noisy spread ---
    uco = pd.Series(
        soyoil.to_numpy() - np.clip(0.055 + rng.normal(scale=0.012, size=n), 0.0, None),
        index=index,
        name="SYNTH_UCO_USD_LB",
    )

    return {
        "ulsd": ulsd,
        "rin_d4": rin_d4,
        "lcfs": lcfs,
        "soyoil": soyoil,
        "uco": uco,
        "brent": brent,
    }
