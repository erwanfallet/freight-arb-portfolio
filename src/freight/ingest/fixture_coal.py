"""GÉNÉRATEUR DE DONNÉES SYNTHÉTIQUES POUR LE PROJET B — AUCUNE VALEUR ÉCONOMIQUE.

Avertissement plus fort que pour le projet A, et il faut le lire.

Le projet A affirme qu'une part du premium vient du fret ; le jeu synthétique A impose une
structure de ce type, donc il ne peut pas la confirmer. Ici c'est pire : **la rupture de
2022 est imposée à la main dans le générateur**. Le dashboard affichera donc un coefficient
de fret qui s'effondre après 2022 et un arb qui décroche — parce que c'est écrit dans ces
quinze lignes de code, pas parce que le marché l'a fait.

Ce jeu de données sert exclusivement à vérifier que les six sections s'affichent et que le
test de rupture avec contrôle TTF tourne. Toute lecture économique de ces graphiques est
une erreur.

Tickers préfixés `SYNTH_`, attribut `.attrs["synthetic"] = True`, bannière rouge dans le
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
    """Six séries synthétiques en format long canonique (date, ticker, valeur).

    Structure imposée, à connaître avant de regarder un seul graphique :
      - avant la rupture : spread API2−API4 ≈ fret, donc arb autour de zéro
      - après : le spread garde un niveau propre indépendant du fret
      - un choc TTF au printemps 2022, corrélé au niveau du spread, pour que le test
        d'attribution ait quelque chose à démêler
      - EUA nul avant 2024 uniquement dans les faits réglementaires, pas dans la série :
        le prix du quota existe avant, c'est la montée en charge qui le neutralise
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

    # avant : le spread paie le fret. après : il garde un niveau propre.
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
