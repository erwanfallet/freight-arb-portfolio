"""SYNTHETIC DATA GENERATOR FOR PROJECT C — NO ECONOMIC VALUE.

As with project B, the structure is imposed: the January 1st flat-rate jumps are
hand-written into `SYNTHETIC_FLAT_RATES`. The dashboard will therefore show a cost jump
on January 1st because it was put there, not because the Worldscale Association made
one that year.

This dataset exists to verify the market/reset/cross decomposition, the open-days
illusion, and the TCE inversion all run. Nothing else.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from freight.signals.worldscale import FlatRateTable

SYNTHETIC_TICKERS = {
    "p_ara": "SYNTH_GASOIL_ARA_USD_T",
    "p_usgc": "SYNTH_ULSD_USGC_USD_GAL",
    "ws": "SYNTH_TC14_WS_POINTS",
    "bunker": "SYNTH_VLSFO_USD_T",
    "tce_mr": "SYNTH_MR_TCE_USD_DAY",
}

ROUTE = "TC14"

# Imposed annual steps. The 2022 -> 2023 jump is deliberately large, to reproduce the
# effect of a bunker environment that spiked the year before.
SYNTHETIC_FLAT_RATES = FlatRateTable(
    rates={ROUTE: {2022: 18.0, 2023: 24.5, 2024: 25.8, 2025: 25.1, 2026: 27.3}}
)


def synthetic_products(seed: int = 23) -> pd.DataFrame:
    """Five synthetic series in the canonical long format (date, ticker, valeur)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", "2026-06-30")
    n = len(idx)

    # US leg in $/gal, around 2.40
    p_usgc = 2.40 + np.cumsum(rng.normal(0, 0.012, n))
    p_usgc = np.clip(p_usgc, 1.20, 4.50)

    # European leg in $/t: the converted US leg, plus a seasonal spread
    seasonal = 12.0 * np.cos(2 * np.pi * (idx.dayofyear.to_numpy() - 15) / 365.25)
    p_ara = p_usgc * 42.0 * 7.45 + 18.0 + seasonal + np.cumsum(rng.normal(0, 1.6, n))

    ws = 150.0 + np.cumsum(rng.normal(0, 1.6, n))
    ws = np.clip(ws, 70.0, 350.0)

    bunker = 560.0 + np.cumsum(rng.normal(0, 3.2, n))
    bunker = np.clip(bunker, 300.0, 900.0)

    tce_mr = 20_000.0 + np.cumsum(rng.normal(0, 320.0, n))
    tce_mr = np.clip(tce_mr, 3_000.0, 70_000.0)

    frames = []
    for role, values in (
        ("p_ara", p_ara), ("p_usgc", p_usgc), ("ws", ws),
        ("bunker", bunker), ("tce_mr", tce_mr),
    ):
        frames.append(
            pd.DataFrame({"date": idx, "ticker": SYNTHETIC_TICKERS[role], "valeur": values})
        )
    out = pd.concat(frames, ignore_index=True)
    out.attrs["synthetic"] = True
    return out
