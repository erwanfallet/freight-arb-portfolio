"""Jeux synthétiques des six projets Tier 2.

Chaque générateur **impose** le phénomène que sa thèse prédit, avec les paramètres vrais
exposés en constantes pour que les golden tests vérifient que le moteur les retrouve.
Tickers préfixés `SYNTH_`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# T2-3 — board crush contre crush usine
# ---------------------------------------------------------------------------
DECOUPLING_WINDOWS = (("2022-09-01", "2022-11-15"), ("2024-03-01", "2024-05-01"))


def crush_tracking(*, periods: int = 1_400, seed: int = 2) -> dict[str, pd.Series]:
    """Prix board et cash, avec deux épisodes de décrochage imposés.

    Hors épisodes, le basis local est petit et stable ; dedans, le basis tourteau explose
    — c'est le mécanisme que les gens d'usine décrivent et que le board ne voit pas.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2021-01-04", periods=periods, freq="B")
    n = len(index)

    bean_board = pd.Series(
        13.20 * np.exp(np.cumsum(rng.normal(scale=0.010, size=n))),
        index=index,
        name="SYNTH_CBOT_BEAN_USD_BU",
    )
    meal_board = pd.Series(
        390.0 * np.exp(np.cumsum(rng.normal(scale=0.011, size=n))),
        index=index,
        name="SYNTH_CBOT_MEAL_USD_STON",
    )
    oil_board = pd.Series(
        58.0 * np.exp(np.cumsum(rng.normal(scale=0.013, size=n))),
        index=index,
        name="SYNTH_CBOT_OIL_C_LB",
    )

    stress = np.zeros(n)
    for start, end in DECOUPLING_WINDOWS:
        stress[(index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))] = 1.0

    # Pendant le décrochage, le basis tourteau ne se contente pas de se décaler : il
    # acquiert sa PROPRE dynamique, indépendante du board. C'est ce qui fait bouger la
    # covariance, donc le ratio de couverture optimal. Un simple décalage de niveau
    # laisserait h* collé à 1 et la page n'aurait rien à montrer.
    independent_walk = np.cumsum(rng.normal(scale=9.0, size=n) * stress)
    bean_cash = bean_board + (-0.25 + rng.normal(scale=0.04, size=n))
    meal_cash = meal_board + (
        8.0 + 55.0 * stress + independent_walk + rng.normal(scale=3.0, size=n)
    )
    oil_cash = oil_board + (-0.6 + rng.normal(scale=0.25, size=n))

    return {
        "bean_board": bean_board,
        "meal_board": meal_board,
        "oil_board": oil_board,
        "bean_cash": pd.Series(bean_cash.to_numpy(), index=index, name="SYNTH_CASH_BEAN"),
        "meal_cash": pd.Series(meal_cash.to_numpy(), index=index, name="SYNTH_CASH_MEAL"),
        "oil_cash": pd.Series(oil_cash.to_numpy(), index=index, name="SYNTH_CASH_OIL"),
    }


# ---------------------------------------------------------------------------
# T2-4 — white premium
# ---------------------------------------------------------------------------
def white_premium(*, periods: int = 1_300, seed: int = 3) -> dict[str, pd.Series]:
    """No.11 et No.5, avec une `richness` qui bascule entre RICH et CHEAP.

    Le No.5 est construit **à partir** du coût de raffinage reconstruit plus un résidu
    cyclique : sans ça, la page n'aurait pas de zones à montrer.
    """
    from agri.chains.white_premium import (
        DEFAULT_POL_ADJUST,
        fair_value_refining_usd_t,
    )
    from agri.core.units import cents_lb_to_usd_t

    rng = np.random.default_rng(seed)
    index = pd.date_range("2021-01-04", periods=periods, freq="B")
    n = len(index)

    no11 = pd.Series(
        19.5 * np.exp(np.cumsum(rng.normal(scale=0.011, size=n))),
        index=index,
        name="SYNTH_NY11_C_LB",
    )
    costs = fair_value_refining_usd_t(no11)
    richness = 34.0 * np.sin(2 * np.pi * np.arange(n) / 300.0) + rng.normal(scale=6.0, size=n)
    no5 = pd.Series(
        (cents_lb_to_usd_t(no11) * DEFAULT_POL_ADJUST + costs["total"] + richness).to_numpy(),
        index=index,
        name="SYNTH_NO5_USD_T",
    )
    # clés nommées comme les paramètres de `build_richness`, pour que `**fixture` marche
    return {"no5_usd_t": no5, "no11_cents_lb": no11}


# ---------------------------------------------------------------------------
# T2-5 — l'usine comme option
# ---------------------------------------------------------------------------
TRUE_OU_KAPPA = 0.035
TRUE_OU_THETA = 4.0
TRUE_OU_SIGMA = 3.2


def plant_margin(*, periods: int = 1_600, seed: int = 4) -> pd.Series:
    """Marge de trituration en Ornstein-Uhlenbeck, paramètres vrais connus.

    Simulation exacte du processus discret :
        M_{t+1} = theta + (M_t - theta) e^{-kappa dt} + eps,
        eps ~ N(0, sigma sqrt((1 - e^{-2 kappa dt}) / (2 kappa)))
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2019-01-02", periods=periods, freq="B")
    decay = np.exp(-TRUE_OU_KAPPA)
    innovation_std = TRUE_OU_SIGMA * np.sqrt((1.0 - decay**2) / (2.0 * TRUE_OU_KAPPA))

    values = np.empty(periods)
    values[0] = TRUE_OU_THETA
    shocks = rng.normal(scale=innovation_std, size=periods)
    for i in range(1, periods):
        values[i] = TRUE_OU_THETA + (values[i - 1] - TRUE_OU_THETA) * decay + shocks[i]
    return pd.Series(values, index=index, name="SYNTH_CRUSH_MARGIN_USD_BU")


# ---------------------------------------------------------------------------
# T2-6 — substitution inter-huiles
# ---------------------------------------------------------------------------
# Demi-vies vraies : -ln(2) / ln(1 + beta).
# Le régime étroit doit rester assez réversif pour que le processus **oscille** autour de
# zéro dans l'échantillon. Avec une demi-vie de 173 jours sur 1 600 jours, une seule longue
# excursion dominait la trajectoire, la médiane réalisée tombait à -88 au lieu de 0, et la
# classification des régimes s'inversait. Un jeu de test doit être stationnaire à l'échelle
# de l'échantillon, pas seulement en théorie.
TRUE_BETA_NARROW = -0.015      # spread étroit : retour lent    (demi-vie ~45,9 j)
TRUE_BETA_WIDE = -0.120        # spread large  : retour rapide  (demi-vie ~5,4 j)
# Seuil calé pour que le régime large contienne assez d'observations pour être estimable.
# À 90 $/t contre un écart-type réalisé de ~41, le régime large ne représentait que 2,8 %
# de l'échantillon (n=60) : le bêta sortait juste (-0,125 pour un vrai -0,120) mais avec
# p=0,30, donc inutilisable. À 60 $/t il pèse ~14 %.
SUBSTITUTION_THRESHOLD_USD_T = 60.0
SUBSTITUTION_SHOCK_STD = 10.0


def oil_prices(*, periods: int = 2_500, seed: int = 5) -> dict[str, pd.Series]:
    """Palme, soja et colza en USD/t, avec un spread palme-soja à seuil.

    Le spread suit un AR(1) **à seuil** : lent tant qu'il reste dans une bande de
    +/- 90 $/t autour de sa moyenne, rapide au-delà. C'est la borne de substitution que
    `substitution_bound` doit retrouver.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2019-01-02", periods=periods, freq="B")

    soy = pd.Series(
        980.0 * np.exp(np.cumsum(rng.normal(scale=0.010, size=periods))),
        index=index,
        name="SYNTH_SOYOIL_USD_T",
    )

    spread = np.empty(periods)
    spread[0] = 0.0
    shocks = rng.normal(scale=SUBSTITUTION_SHOCK_STD, size=periods)
    for i in range(1, periods):
        beta = TRUE_BETA_WIDE if abs(spread[i - 1]) > SUBSTITUTION_THRESHOLD_USD_T else TRUE_BETA_NARROW
        spread[i] = spread[i - 1] + beta * spread[i - 1] + shocks[i]

    # Pas de décote structurelle palme-soja ici, volontairement : le régime du processus
    # bascule sur |spread| > seuil autour de ZÉRO, et `build_spreads` renvoie
    # `palme - soja`. Ajouter un décalage de -120 centrerait la série observée sur -120
    # alors que le régime resterait défini autour de 0 : les deux centres divergeraient et
    # la classification des régimes serait fausse sans que rien ne le signale.
    palm = pd.Series((soy + spread).to_numpy(), index=index, name="SYNTH_PALM_USD_T")
    rape = pd.Series(
        (soy + 60.0 + np.cumsum(rng.normal(scale=3.0, size=periods))).to_numpy(),
        index=index,
        name="SYNTH_RAPEOIL_USD_T",
    )
    return {"palm": palm, "soy": soy, "canola": rape}
