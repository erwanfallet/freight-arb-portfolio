"""Jeu synthétique T1-1 — fabriqué pour **imposer** le phénomène que la thèse prédit.

Ce n'est pas du bruit plausible. La construction force trois propriétés, et les golden
tests vérifient que le moteur les retrouve :

1. `freight_full > freight_index` **toujours** — le ballast ne peut que renchérir ;
2. l'arb sous la convention full oscille autour de zéro avec un écart-type de quelques
   dollars, donc une fraction substantielle de jours tombe dans la bande frontière ;
3. sur ces jours-là, les deux conventions donnent des signes opposés — c'est le sujet.

La propriété 2 est obtenue en construisant le CIF **à partir** du fret full plus un bruit
centré, plutôt qu'en tirant CIF et FOB indépendamment et en espérant que l'arb tombe au
bon endroit. C'est assumé : un jeu synthétique sert à tester le moteur, pas à simuler un
marché.

Tickers préfixés `SYNTH_`. Rien ici ne doit jamais être lu comme un chiffre de marché.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from agri.chains.freight_cf import financing_cost_usd_t
from agri.core.voyage import ROUTES, VESSELS, VoyageParams, voyage_freight_series

DEFAULT_START = "2023-01-02"
DEFAULT_PERIODS = 600          # ~2,4 ans de jours ouvrés, dont 90 mangés par le lissage

# Niveaux de référence, ordres de grandeur crédibles pour un Panamax grains.
TCE_MEAN_USD_DAY = 15_000.0
VLSFO_MEAN_USD_T = 560.0
MGO_MEAN_USD_T = 780.0
FOB_MEAN_USD_T = 440.0         # Santos soja, ordre de grandeur

# Écart-type du bruit ajouté au CIF : c'est lui qui décide de la largeur de la bande
# frontière, donc de la fraction de jours où la convention tranche.
CIF_WOBBLE_STD_USD_T = 6.0


def build(
    *,
    start: str = DEFAULT_START,
    periods: int = DEFAULT_PERIODS,
    seed: int = 0,
    vessel_key: str = "panamax",
    route_key: str = "santos_qingdao",
    annual_rate: float = 0.055,
    credit_days: float = 30.0,
    insurance_usd_t: float = 0.85,
) -> dict[str, pd.Series]:
    """Renvoie les cinq séries d'entrée de T1-1, en dictionnaire prêt pour `build_conventions`."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=periods, freq="B")
    n = len(index)
    day_of_year = index.dayofyear.to_numpy()

    # --- TCE : moyenne-réversion lente + saisonnalité (creux de l'hiver nord) ---
    seasonal_tce = 1.0 + 0.18 * np.sin(2 * np.pi * (day_of_year - 60) / 365.25)
    shocks = rng.normal(scale=0.02, size=n)
    drift = np.zeros(n)
    for i in range(1, n):
        drift[i] = 0.97 * drift[i - 1] + shocks[i]
    tce = pd.Series(
        TCE_MEAN_USD_DAY * seasonal_tce * np.exp(drift), index=index, name="SYNTH_TCE_PANAMAX"
    )

    # --- Soutes : tendance lente commune, MGO au-dessus du VLSFO avec un spread bruité ---
    bunker_drift = np.zeros(n)
    bunker_shocks = rng.normal(scale=0.012, size=n)
    for i in range(1, n):
        bunker_drift[i] = 0.985 * bunker_drift[i - 1] + bunker_shocks[i]
    vlsfo = pd.Series(
        VLSFO_MEAN_USD_T * np.exp(bunker_drift), index=index, name="SYNTH_VLSFO_SIN"
    )
    mgo = pd.Series(
        MGO_MEAN_USD_T * np.exp(bunker_drift) + rng.normal(scale=8.0, size=n),
        index=index,
        name="SYNTH_MGO_SIN",
    )

    # --- FOB origine : marche aléatoire douce autour du niveau de référence ---
    fob_drift = np.zeros(n)
    fob_shocks = rng.normal(scale=0.006, size=n)
    for i in range(1, n):
        fob_drift[i] = 0.995 * fob_drift[i - 1] + fob_shocks[i]
    fob = pd.Series(
        FOB_MEAN_USD_T * np.exp(fob_drift), index=index, name="SYNTH_FOB_SANTOS"
    )

    # --- CIF destination : construit POUR que l'arb full oscille autour de zéro ---
    vessel = VESSELS[vessel_key]
    route = ROUTES[route_key]
    params = VoyageParams()
    freight_full = voyage_freight_series(
        tce, vlsfo, mgo, vessel=vessel, route=route, params=params.with_ballast(1.0)
    )
    voyage_days = _reference_days(vessel, route, params)
    financing = financing_cost_usd_t(
        fob,
        freight_full,
        annual_rate=annual_rate,
        voyage_days=voyage_days,
        credit_days=credit_days,
    )
    wobble = pd.Series(rng.normal(scale=CIF_WOBBLE_STD_USD_T, size=n), index=index)
    cif = pd.Series(
        fob + freight_full + financing + insurance_usd_t + wobble,
        index=index,
        name="SYNTH_CIF_QINGDAO",
    )

    return {"tce": tce, "vlsfo": vlsfo, "mgo": mgo, "cif": cif, "fob": fob}


def _reference_days(vessel, route, params: VoyageParams) -> float:
    from agri.core.voyage import HOURS_PER_DAY

    laden = route.distance_laden_nm / (params.speed_laden_kn * HOURS_PER_DAY)
    ballast = route.distance_ballast_nm / (params.speed_ballast_kn * HOURS_PER_DAY)
    return laden + ballast + params.port_days + params.wait_days


def build_frame(**kwargs) -> pd.DataFrame:
    """Raccourci : les séries synthétiques déjà passées dans `build_conventions`."""
    from agri.chains.freight_cf import build_conventions

    vessel_key = kwargs.pop("vessel_key", "panamax")
    route_key = kwargs.pop("route_key", "santos_qingdao")
    series = build(vessel_key=vessel_key, route_key=route_key, **kwargs)
    return build_conventions(
        series["tce"],
        series["vlsfo"],
        series["mgo"],
        series["cif"],
        series["fob"],
        vessel=VESSELS[vessel_key],
        route=ROUTES[route_key],
        params=VoyageParams(),
    )
