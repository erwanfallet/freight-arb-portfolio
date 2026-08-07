"""GÉNÉRATEUR DE DONNÉES SYNTHÉTIQUES POUR LE PROJET C — AUCUNE VALEUR ÉCONOMIQUE.

Comme pour le projet B, la structure est imposée : les marches de flat rate au 1er janvier
sont écrites à la main dans `SYNTHETIC_FLAT_RATES`. Le dashboard montrera donc un saut de
coût au 1er janvier parce que je l'ai mis là, pas parce que la Worldscale Association l'a
fait cette année-là.

Ce jeu sert à vérifier que la décomposition marché / réglage / croisé, l'illusion des jours
ouverts et l'inversion TCE tournent. Rien d'autre.
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

# Marches annuelles imposées. La marche 2022 -> 2023 est volontairement grosse, pour
# reproduire l'effet d'un environnement de soutes qui a bondi l'année précédente.
SYNTHETIC_FLAT_RATES = FlatRateTable(
    rates={ROUTE: {2022: 18.0, 2023: 24.5, 2024: 25.8, 2025: 25.1, 2026: 27.3}}
)


def synthetic_products(seed: int = 23) -> pd.DataFrame:
    """Cinq séries synthétiques en format long canonique (date, ticker, valeur)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", "2026-06-30")
    n = len(idx)

    # jambe US en $/gal, autour de 2,40
    p_usgc = 2.40 + np.cumsum(rng.normal(0, 0.012, n))
    p_usgc = np.clip(p_usgc, 1.20, 4.50)

    # jambe européenne en $/t : la jambe US convertie, plus un spread saisonnier
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
