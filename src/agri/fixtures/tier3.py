"""Jeux synthétiques des trois derniers projets Tier 3.

Chaque générateur impose le phénomène de sa thèse, paramètres vrais exposés en constantes.
Tickers préfixés `SYNTH_`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# T3-2 — sucre : le mix ne suit pas la parité
# ---------------------------------------------------------------------------
# Élasticité vraie du mix à la parité, et sa conditionnalité au hedge d'entrée.
TRUE_BETA_PARITY = 0.0020          # points de mix par cent/lb, à hedge nul
TRUE_BETA_INTERACTION = -0.0028    # l'effet s'annule puis s'inverse quand on est couvert

# hedge_ratio et dist_port sont deliberement DECORRELES ici (contrairement a une version
# initiale ou les deux etaient monotones region par region, correlation -0,98). Sans ca,
# le terme d'interaction parity x hedge et le terme parity x dist_port sont quasi
# colineaires, et beta2 (l'objet d'interet de la page) sort indetermine (~0 au lieu de la
# vraie valeur) sans jamais lever d'erreur — un piege silencieux, pas une absence de
# signal. Six regions plutot que quatre, pour plus de degres de liberte sur l'interaction.
REGIONS = {
    "Sao Paulo centre": {"hedge": 0.65, "dist_port": 0.20, "cap": 0.52, "presold": 0.44},
    "Sao Paulo ouest": {"hedge": 0.60, "dist_port": 0.85, "cap": 0.50, "presold": 0.42},
    "Minas Gerais": {"hedge": 0.30, "dist_port": 0.35, "cap": 0.48, "presold": 0.38},
    "Goias": {"hedge": 0.25, "dist_port": 1.00, "cap": 0.46, "presold": 0.34},
    "Mato Grosso do Sul": {"hedge": 0.50, "dist_port": 0.65, "cap": 0.49, "presold": 0.40},
    "Parana": {"hedge": 0.40, "dist_port": 0.15, "cap": 0.51, "presold": 0.43},
}


def sugar_prices(*, periods: int = 900, seed: int = 10) -> dict[str, pd.Series]:
    """NY11, éthanol hydraté ex-usine et USDBRL, avec une parité qui change de signe."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2023-01-02", periods=periods, freq="B")

    ny11 = pd.Series(
        19.0 * np.exp(np.cumsum(rng.normal(scale=0.012, size=periods))),
        index=index,
        name="SYNTH_NY11_C_LB",
    )
    usdbrl = pd.Series(
        5.10 * np.exp(np.cumsum(rng.normal(scale=0.005, size=periods))),
        index=index,
        name="SYNTH_USDBRL",
    )
    # hydraté calé pour que la parité oscille autour de zéro sur l'échantillon
    cycle = 0.09 * np.sin(2 * np.pi * np.arange(periods) / 260.0)
    hydrous = pd.Series(
        (2.95 + cycle) * np.exp(np.cumsum(rng.normal(scale=0.006, size=periods))),
        index=index,
        name="SYNTH_HYDROUS_BRL_L",
    )
    return {"ny11": ny11, "hydrous": hydrous, "usdbrl": usdbrl}


def sugar_panel(*, seed: int = 11, fortnights: int = 60) -> pd.DataFrame:
    """Panel quinzaine x région, avec l'élasticité conditionnelle **imposée**.

        d_mix = b1 x parity + b2 x (parity x hedge) + effets fixes + bruit

    C'est `b2` que `estimate_mix_elasticity` doit retrouver — l'objet du désaccord entre
    Hedgepoint et Czarnikow.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-04-01", periods=fortnights, freq="SME")

    rows = []
    for region, spec in REGIONS.items():
        region_effect = rng.normal(scale=0.0015)
        for date in dates:
            parity = rng.normal(loc=0.4, scale=1.8)          # cents/lb
            cap_utilisation = np.clip(rng.normal(0.86, 0.06), 0.5, 1.0)
            d_mix = (
                region_effect
                + TRUE_BETA_PARITY * parity
                + TRUE_BETA_INTERACTION * parity * spec["hedge"]
                + 0.004 * (cap_utilisation - 0.86)
                + rng.normal(scale=0.0016)
            )
            rows.append(
                {
                    "date": date,
                    "region": region,
                    "d_mix": d_mix,
                    "parity_gap_lag": parity,
                    "hedge_ratio_entry": spec["hedge"],
                    "cap_utilisation": cap_utilisation,
                    "dist_port": spec["dist_port"],
                }
            )
    return pd.DataFrame(rows).set_index("date")


# ---------------------------------------------------------------------------
# T3-4 — Chine soja
# ---------------------------------------------------------------------------
# Signature imposée : NÉGATIVE, donc politique. Les achats se concentrent quand la marge
# de crush est basse — ce que le commercial ne peut pas faire.
TRUE_POLITICAL_SIGNATURE = True


def china_soy(*, periods: int = 220, seed: int = 14) -> dict:
    """Séries mensuelles : CBOT, basis, fret, DCE tourteau/huile, USDCNY, imports, crush.

    Les achats de réserve sont générés par une probabilité **décroissante** avec la marge
    de crush : c'est la signature politique que `signature_test` doit détecter.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2018-01-31", periods=periods, freq="ME")

    cbot = pd.Series(
        12.0 * np.exp(np.cumsum(rng.normal(scale=0.035, size=periods))),
        index=index,
        name="SYNTH_CBOT_USD_BU",
    )
    basis = pd.Series(
        70.0 + np.cumsum(rng.normal(scale=6.0, size=periods)),
        index=index,
        name="SYNTH_BASIS_C_BU",
    )
    freight = pd.Series(
        45.0 + np.cumsum(rng.normal(scale=2.0, size=periods)),
        index=index,
        name="SYNTH_FREIGHT_USD_T",
    ).clip(lower=15.0)
    usdcny = pd.Series(
        6.85 * np.exp(np.cumsum(rng.normal(scale=0.010, size=periods))),
        index=index,
        name="SYNTH_USDCNY",
    )
    meal = pd.Series(
        3_300.0 * np.exp(np.cumsum(rng.normal(scale=0.035, size=periods))),
        index=index,
        name="SYNTH_DCE_MEAL_CNY_T",
    )
    oil = pd.Series(
        7_900.0 * np.exp(np.cumsum(rng.normal(scale=0.032, size=periods))),
        index=index,
        name="SYNTH_DCE_OIL_CNY_T",
    )

    from agri.chains.china_soy import bean_cnf_usd_t, crush_margin_cny_t

    bean = bean_cnf_usd_t(cbot, basis, freight)
    margin = crush_margin_cny_t(meal, oil, bean, usdcny)["margin"]

    # probabilité d'achat DÉCROISSANTE avec la marge retardée -> signature politique
    standardised = (margin - margin.mean()) / margin.std()
    logit = -0.4 - 1.6 * standardised.shift(1)
    probability = 1.0 / (1.0 + np.exp(-logit))
    purchases = pd.Series(
        (rng.random(periods) < probability.fillna(0.3)).astype(int),
        index=index,
        name="SYNTH_RESERVE_PURCHASE",
    )

    crush_observed = pd.Series(
        8_200_000 + rng.normal(scale=350_000, size=periods),
        index=index,
        name="SYNTH_CRUSH_T",
    )
    imports = pd.Series(
        (crush_observed + 400_000 * purchases + rng.normal(scale=250_000, size=periods)).to_numpy(),
        index=index,
        name="SYNTH_IMPORTS_T",
    )
    stock_days = pd.Series(
        np.clip(32.0 + np.cumsum(rng.normal(scale=1.2, size=periods)), 8.0, 70.0),
        index=index,
        name="SYNTH_STOCK_DAYS",
    )

    return {
        "cbot": cbot,
        "basis": basis,
        "freight": freight,
        "usdcny": usdcny,
        "meal": meal,
        "oil": oil,
        "margin": margin,
        "purchases": purchases,
        "imports": imports,
        "crush_observed": crush_observed,
        "stock_days": stock_days,
    }
