"""Jeu synthétique T1-2 — le cycle cacao 2022-2026, avec ses deux punitions.

Trois propriétés **imposées**, vérifiées par les golden tests :

1. le prix monte d'environ x5 jusqu'à un pic en décembre 2024, puis s'effondre — c'est le
   retournement qui rend le projet neuf ;
2. la courbe est en **backwardation pendant la hausse** (le déféré sous le front) et en
   contango dans les phases calmes — donc le short paie le roll exactement quand il paie
   déjà les appels de marge ;
3. la volatilité, donc la marge initiale, est multipliée par un facteur proche de neuf au
   pic — l'ancrage Barry Callebaut.

Tickers préfixés `SYNTH_`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_START = "2022-01-03"
DEFAULT_END = "2026-06-30"

PEAK_DATE = pd.Timestamp("2024-12-16")
CALM_PRICE = 2_400.0            # USD/t, régime d'avant-crise
PEAK_PRICE = 11_500.0           # USD/t, ordre de grandeur du pic
TROUGH_PRICE = 5_200.0          # USD/t, après l'effondrement

# Fenêtres d'analyse pour le graphe en miroir de la section S6.
WINDOWS = {
    "hausse 2023-24 (le short paie)": ("2023-06-01", "2024-12-16"),
    "baisse 2025-26 (le long paie)": ("2025-01-02", "2026-06-30"),
}


def build(
    *, start: str = DEFAULT_START, end: str = DEFAULT_END, seed: int = 0
) -> dict:
    """Séries d'entrée de T1-2 : front, déféré, volatilité implicite du régime, taux, rolls."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, end, freq="B")
    n = len(index)

    days = (index - index[0]).days.to_numpy(dtype=float)
    peak_day = float((PEAK_DATE - index[0]).days)
    total_days = days[-1]

    # --- trajectoire de prix : montée exponentielle puis effondrement ---
    ramp = np.clip(days / peak_day, 0.0, 1.0) ** 2.2
    fall = np.clip((days - peak_day) / (total_days - peak_day), 0.0, 1.0) ** 0.8
    trend = CALM_PRICE + (PEAK_PRICE - CALM_PRICE) * ramp - (PEAK_PRICE - TROUGH_PRICE) * fall

    # volatilité du régime : faible au calme, x3 sur l'emballement
    stress = np.exp(-(((days - peak_day) / 210.0) ** 2))       # cloche centrée sur le pic
    daily_vol = 0.010 + 0.024 * stress
    noise = np.cumsum(rng.normal(scale=daily_vol, size=n))
    front = pd.Series(trend * np.exp(noise - noise.mean() * 0.0), index=index, name="SYNTH_COCOA_FRONT")
    front = front.clip(lower=800.0)

    # --- structure par terme : backwardation pendant le stress, contango au calme ---
    # spread = deferred - front. Negatif = backwardation.
    spread = -front * (0.045 * stress) + front * (0.008 * (1.0 - stress))
    deferred = pd.Series(
        (front + spread).to_numpy(), index=index, name="SYNTH_COCOA_DEFERRED"
    ).clip(lower=700.0)

    # --- taux de financement : remontée 2022-2023 puis plateau ---
    rate = pd.Series(
        np.clip(0.005 + 0.045 * np.clip(days / 400.0, 0.0, 1.0), 0.005, 0.055),
        index=index,
        name="SYNTH_SOFR",
    )

    # --- dates de roll : le 15 de chaque mois, ramené au jour ouvré précédent ---
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
