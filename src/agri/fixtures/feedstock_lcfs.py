"""Jeu synthétique T3-1.

Deux propriétés sont **imposées**, et les golden tests vérifient que le moteur les retrouve :

1. le prix LCFS traverse le seuil `LCFS*` en cours d'échantillon, donc la filière gagnante
   change — sans quoi la heatmap n'aurait qu'une seule zone et la page n'aurait rien à dire ;
2. le bêta énergie du soyoil au Brent **double** à la date de finalisation des RVO
   (mars 2026), pour que le test de Chow ait quelque chose à détecter et qu'on puisse
   vérifier qu'il ne le détecte pas ailleurs.

Tickers préfixés `SYNTH_`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_START = "2023-01-02"
# ~4,4 ans de jours ouvrés, jusqu'a mi-2027. Dimensionné pour laisser plus de 300 jours
# ouvrés APRÈS la rupture de mars 2026 : sans ça, aucune fenêtre glissante de 120 jours
# n'est entièrement dans le régime post-rupture, et le bêta d'après ne peut pas être lu.
DEFAULT_PERIODS = 1150

# Date de politique : finalisation des RVO 2026-27 par l'EPA.
RVO_BREAK_DATE = pd.Timestamp("2026-03-16")

BETA_BEFORE_BREAK = 0.20       # le soyoil suivait mollement le brut
BETA_AFTER_BREAK = 0.45        # le biofuel pull se renforce


def build(
    *, start: str = DEFAULT_START, periods: int = DEFAULT_PERIODS, seed: int = 0
) -> dict[str, pd.Series]:
    """Séries d'entrée de T3-1 : ULSD, RIN D4, crédit LCFS, soyoil, Brent."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=periods, freq="B")
    n = len(index)

    # --- Brent : marche aléatoire avec une escalade géopolitique début 2026 ---
    brent_returns = rng.normal(scale=0.014, size=n)
    escalation = (index >= pd.Timestamp("2026-01-15")) & (index <= pd.Timestamp("2026-02-28"))
    brent_returns[escalation] += 0.004
    brent = pd.Series(78.0 * np.exp(np.cumsum(brent_returns)), index=index, name="SYNTH_BRENT")

    # --- Soyoil : bêta au Brent qui DOUBLE à la date des RVO (pour le test de Chow) ---
    betas = np.where(index >= RVO_BREAK_DATE, BETA_AFTER_BREAK, BETA_BEFORE_BREAK)
    soyoil_returns = betas * brent_returns + rng.normal(scale=0.010, size=n)
    soyoil = pd.Series(
        0.52 * np.exp(np.cumsum(soyoil_returns)), index=index, name="SYNTH_SOYOIL_USD_LB"
    )

    # --- LCFS : traverse le seuil en cours d'échantillon ---
    # rampe de 60 vers 340 $/t avec du bruit : le seuil a parité de prix est a ~285 $/t,
    # donc la filiere gagnante bascule quelque part au milieu de l'echantillon.
    ramp = np.linspace(60.0, 340.0, n)
    lcfs = pd.Series(
        np.clip(ramp + rng.normal(scale=12.0, size=n), 5.0, None),
        index=index,
        name="SYNTH_LCFS_USD_T",
    )

    # --- ULSD et RIN D4 : niveaux plausibles, faiblement bruités ---
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

    # --- UCO importé : décoté par rapport au soyoil, spread bruité ---
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
