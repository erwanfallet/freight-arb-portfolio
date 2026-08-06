"""GÉNÉRATEUR DE DONNÉES SYNTHÉTIQUES — AUCUNE VALEUR ÉCONOMIQUE.

Raison d'être : prouver que le pipeline tourne et que les six sections du dashboard
s'affichent, avant que les vraies séries n'arrivent. Rien de ce que produit ce module ne
doit sortir du repo, apparaître dans un email, ou être lu comme un résultat.

Trois garde-fous :
  1. tous les tickers sont préfixés `SYNTH_`
  2. la fonction renvoie un DataFrame avec l'attribut `.attrs["synthetic"] = True`
  3. le dashboard affiche une bannière rouge tant que cet attribut est vrai

Les niveaux choisis sont de l'ordre de grandeur du marché (P62 autour de 100 $/dmt, C3
autour de 20 $/wmt, C5 autour de 10 $/wmt) uniquement pour que les axes des graphiques
soient lisibles. Ils ne sont calibrés sur rien.
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
    """Quatre séries synthétiques en format long canonique (date, ticker, valeur).

    La structure imposée volontairement : un facteur Capesize commun pousse C3 et C5
    ensemble, C3 réagit plus fort (long-courrier, plus sensible aux soutes), et le
    premium 65-62 contient une composante qualité indépendante. C'est la structure que
    la décomposition est censée retrouver — donc ce jeu de données teste la plomberie,
    pas la thèse. Il ne peut pas la confirmer.
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

    # trous volontaires : jours fériés désalignés entre le fret et les prix, pour que la
    # gestion des calendriers soit exercée dès le mode synthétique.
    holidays = set(rng.choice(n_days, size=12, replace=False))
    mask = out.apply(
        lambda r: (idx.get_loc(r["date"]) in holidays)
        and r["ticker"].startswith("SYNTH_C"),
        axis=1,
    )
    out = out[~mask].reset_index(drop=True)

    out.attrs["synthetic"] = True
    return out
