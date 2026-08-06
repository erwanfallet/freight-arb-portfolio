"""Passage du format long canonique (date, ticker, valeur) à des séries indexées date.

Le format long reste la vérité en amont : chaque ticker garde son propre calendrier et
les trous restent visibles. La conversion en séries se fait le plus tard possible, juste
avant le calcul, et l'alignement est fait par intersection explicite dans le module de
chaîne — jamais par un `reindex().ffill()` silencieux.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from freight.ingest.loader import load_raw_directory


class MissingSeries(Exception):
    """Un ticker attendu n'est pas présent dans data/raw/."""


def to_series(df_long: pd.DataFrame, ticker: str) -> pd.Series:
    """Extrait un ticker du format long. Lève si absent — pas de retour vide silencieux."""
    rows = df_long[df_long["ticker"] == ticker]
    if rows.empty:
        available = sorted(df_long["ticker"].unique())
        raise MissingSeries(
            f"ticker '{ticker}' absent de data/raw/. Tickers présents : {available}"
        )
    s = rows.set_index("date")["valeur"].astype(float).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s.name = ticker
    return s


def load_series_map(raw_dir: str | Path, tickers: dict[str, str]) -> dict[str, pd.Series]:
    """Charge data/raw/ et renvoie {rôle: série} d'après un mapping {rôle: ticker}.

    Le mapping rôle -> ticker vit dans le module de chaîne, pas ici : c'est une décision
    de modélisation (« quel ticker joue le rôle du prix 65 % Fe »), pas une décision
    d'ingestion.
    """
    raw = load_raw_directory(raw_dir)
    if raw.empty:
        raise MissingSeries(f"{raw_dir} ne contient aucun CSV exploitable")
    return {role: to_series(raw, ticker) for role, ticker in tickers.items()}


def coverage_report(series_map: dict[str, pd.Series]) -> pd.DataFrame:
    """Tableau de couverture : à afficher en tête de dashboard, avant tout graphique.

    Une série présente n'est pas une série utilisable. Ce tableau rend visibles la
    première et la dernière observation, le nombre de points et le nombre de jours
    ouvrés manquants sur la plage — sans rien combler.
    """
    rows = []
    for role, s in series_map.items():
        if s.empty:
            rows.append({"rôle": role, "ticker": s.name, "n_obs": 0})
            continue
        business_days = pd.bdate_range(s.index.min(), s.index.max())
        rows.append(
            {
                "rôle": role,
                "ticker": s.name,
                "première_obs": s.index.min().date(),
                "dernière_obs": s.index.max().date(),
                "n_obs": int(s.notna().sum()),
                "jours_ouvrés_sur_la_plage": len(business_days),
                "trous_jours_ouvrés": int(len(business_days) - s.notna().sum()),
            }
        )
    return pd.DataFrame(rows)
