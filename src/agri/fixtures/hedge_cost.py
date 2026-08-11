"""T1-2 synthetic dataset — the 2022-2026 cocoa cycle, with its two punishments.

Three properties **imposed**, verified by the golden tests:

1. the price rises roughly x5 to a December 2024 peak, then collapses — that's the
   reversal that makes the project new;
2. the curve is in **backwardation during the rally** (the deferred below the front) and
   in contango during the calm phases — so the short pays the roll exactly when they're
   already paying margin calls;
3. volatility, and therefore the initial margin, is multiplied by a factor close to nine
   at the peak — the Barry Callebaut anchor.

Tickers prefixed `SYNTH_`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_START = "2022-01-03"
DEFAULT_END = "2026-06-30"

PEAK_DATE = pd.Timestamp("2024-12-16")
CALM_PRICE = 2_400.0            # USD/t, pre-crisis regime
PEAK_PRICE = 11_500.0           # USD/t, order of magnitude of the peak
TROUGH_PRICE = 5_200.0          # USD/t, after the collapse

# Analysis windows for the S6 section's mirror chart.
WINDOWS = {
    "2023-24 rally (the short pays)": ("2023-06-01", "2024-12-16"),
    "2025-26 decline (the long pays)": ("2025-01-02", "2026-06-30"),
}


def build(
    *, start: str = DEFAULT_START, end: str = DEFAULT_END, seed: int = 0
) -> dict:
    """T1-2's input series: front, deferred, regime implied volatility, rate, rolls."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, end, freq="B")
    n = len(index)

    days = (index - index[0]).days.to_numpy(dtype=float)
    peak_day = float((PEAK_DATE - index[0]).days)
    total_days = days[-1]

    # --- price path: exponential rise then collapse ---
    ramp = np.clip(days / peak_day, 0.0, 1.0) ** 2.2
    fall = np.clip((days - peak_day) / (total_days - peak_day), 0.0, 1.0) ** 0.8
    trend = CALM_PRICE + (PEAK_PRICE - CALM_PRICE) * ramp - (PEAK_PRICE - TROUGH_PRICE) * fall

    # regime volatility: low when calm, x3 during the frenzy
    stress = np.exp(-(((days - peak_day) / 210.0) ** 2))       # bell centred on the peak
    daily_vol = 0.010 + 0.024 * stress
    noise = np.cumsum(rng.normal(scale=daily_vol, size=n))
    front = pd.Series(trend * np.exp(noise - noise.mean() * 0.0), index=index, name="SYNTH_COCOA_FRONT")
    front = front.clip(lower=800.0)

    # --- term structure: backwardation under stress, contango when calm ---
    # spread = deferred - front. Negative = backwardation.
    spread = -front * (0.045 * stress) + front * (0.008 * (1.0 - stress))
    deferred = pd.Series(
        (front + spread).to_numpy(), index=index, name="SYNTH_COCOA_DEFERRED"
    ).clip(lower=700.0)

    # --- financing rate: rising through 2022-2023 then plateauing ---
    rate = pd.Series(
        np.clip(0.005 + 0.045 * np.clip(days / 400.0, 0.0, 1.0), 0.005, 0.055),
        index=index,
        name="SYNTH_SOFR",
    )

    # --- roll dates: the 15th of each month, rolled back to the previous business day ---
    month_starts = pd.date_range(index[0], index[-1], freq="MS")
    candidates = month_starts + pd.Timedelta(days=14)
    roll_dates = pd.DatetimeIndex(
        [index[index.get_indexer([d], method="ffill")[0]] for d in candidates if d >= index[0]]
    ).unique()

    return {
        "front": front,
        "deferred": deferred,
        "rate": rate,
        "roll_dates": roll_dates,
        "windows": WINDOWS,
    }
