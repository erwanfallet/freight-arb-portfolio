"""Smoke test for project B on a SYNTHETIC switching frame with an imposed reversion.

Proves nothing about the coal or gas market. Proves that the frame -> switching level ->
distance -> non-overlapping regression -> horse race chain runs end to end and produces
finite numbers, on data specifically constructed so the switching regressor should win
and a flat placebo shouldn't confuse it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from freight.chains.coal import (
    DEFAULT_COAL_EFFICIENCY,
    DEFAULT_GAS_EFFICIENCY,
    ceiling_test,
    efficiency_identification,
    generation_cost_eur_mwh_e,
    switch_ttf_eur_mwh,
    switching_carbon_price,
    switching_distance_pct,
)

HORIZON_DAYS = 20
TRAILING_WINDOW = 60
N_ANCHORS = 120


def synthetic_switching_frame(seed: int = 3) -> pd.DataFrame:
    """coal, EUA and EURUSD wander; TTF is built to revert to its own switching level
    every `HORIZON_DAYS`, by construction — the same shape the real ceiling test looks
    for, imposed by hand so the pipeline has something to find.
    """
    rng = np.random.default_rng(seed)
    n = TRAILING_WINDOW + N_ANCHORS * HORIZON_DAYS + HORIZON_DAYS + 10
    idx = pd.bdate_range("2019-01-01", periods=n)

    coal_th = 25.0 + np.cumsum(rng.normal(0, 0.08, n))
    coal_th = np.clip(coal_th, 8.0, None)
    eua = 55.0 + np.cumsum(rng.normal(0, 0.30, n))
    eua = np.clip(eua, 10.0, None)
    eurusd = 1.10 + np.cumsum(rng.normal(0, 0.0015, n))

    frame = pd.DataFrame(
        {"coal_eur_mwh_th": coal_th, "eua_eur_t": eua, "eurusd": eurusd}, index=idx
    )
    switch = switch_ttf_eur_mwh(frame.assign(ttf_eur_mwh=0.0))

    ttf = switch.to_numpy().copy()
    for start in range(0, n, HORIZON_DAYS):
        end = min(start + HORIZON_DAYS, n)
        ttf[start:end] = switch.to_numpy()[start:end] + rng.normal(0, 5.0)
    frame["ttf_eur_mwh"] = ttf
    frame.attrs["synthetic"] = True
    return frame


def main() -> None:
    print("=" * 78)
    print("SYNTHETIC DATA — the reversion-to-switching-level is IMPOSED by hand.")
    print("No economic reading of these numbers is valid.")
    print("=" * 78)

    frame = synthetic_switching_frame()
    assert frame.attrs.get("synthetic") is True

    switch = switch_ttf_eur_mwh(frame)
    distance = switching_distance_pct(frame, switch)
    print(f"\nswitching level: last {switch.iloc[-1]:.2f} EUR/MWh, TTF last {frame['ttf_eur_mwh'].iloc[-1]:.2f}")
    print(f"share of sample above the ceiling: {100 * (distance > 0).mean():.1f}%")

    gen = generation_cost_eur_mwh_e(frame)
    print(f"generation cost spread (coal - gas), mean: {gen['spread'].mean():.2f} EUR/MWh")

    switch_eua = switching_carbon_price(frame)
    print(f"switching EUA, mean: {switch_eua.mean():.1f} EUR/t")

    identification = efficiency_identification(frame)
    print(f"\n{identification.headline}")

    print("\n-- ceiling test (non-overlapping) ---------------------------------------")
    test = ceiling_test(
        frame, switch, horizon_days=HORIZON_DAYS, trailing_window=TRAILING_WINDOW
    )
    print(f"switching  | {test.switching.summary()}")
    print(f"placebo    | {test.placebo.summary()}")
    print(f"horse race | {test.horse_race.summary()}")

    assert test.switching.coefficients["distance"] < 0, "the imposed reversion should show up"
    assert pd.notna(distance).all()
    print("\nOK — project B's ceiling-test pipeline runs end to end.")


if __name__ == "__main__":
    main()
