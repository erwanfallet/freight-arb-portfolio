"""Conversions d'unités — première cause de page fausse en agri.

Le principe qui gouverne ce module : **une seule constante physique**, `LB_PER_TONNE`,
dont tout le reste dérive par le calcul. Écrire 36.7437 et 22.0462 côte à côte comme deux
nombres indépendants, c'est se garantir qu'un jour l'un des deux sera corrigé et pas
l'autre. Ici, corriger la tonne corrige tout.

LE PIÈGE, ET IL EST LE SUJET DE PLUSIEURS PAGES
-----------------------------------------------
Le board crush CBOT mélange trois unités dans une seule formule :

    fève   -> USD/boisseau   (boisseau soja = 60 lb)
    tourteau -> USD/short ton (2000 lb, PAS une tonne métrique)
    huile  -> cents/lb

C'est la même famille de piège que dmt/wmt (minerai de fer) ou gallon/tonne (distillat) :
l'unité de cotation n'est pas l'unité économique. Voir `docs/NOUVELLE_CHAINE.md`.

TROIS POINTS VÉRIFIÉS, À NE PAS RE-DÉRIVER À CHAQUE FOIS
--------------------------------------------------------
U-H1  Le boisseau n'est pas une unité de volume utilisable ici : c'est un **poids-test**
      qui dépend de la graine (soja et blé 60 lb, maïs et sorgho 56 lb, avoine 32 lb,
      orge 48 lb). Un facteur boisseau->tonne sans nommer la graine est un bug.
U-H2  Le sucre NY No.11 est en c/lb **base 96° pol**, le blanc No.5 en USD/t. Les
      comparer exige la conversion ET un ajustement de polarisation — `pol_adjust` est
      un paramètre, jamais une constante figée (cf. T2-4).
U-H3  La TVA chinoise sur les oléagineux importés n'est pas celle des produits
      transformés, et les taux bougent. `strip_vat` prend le taux ET oblige l'appelant
      à savoir de quel produit et de quelle date il parle.
"""
from __future__ import annotations

from typing import TypeVar

import pandas as pd

Numeric = TypeVar("Numeric", float, pd.Series)

# ---------------------------------------------------------------------------
# La seule constante physique du module. Tout le reste en dérive.
# ---------------------------------------------------------------------------
LB_PER_TONNE = 2204.62262      # livres avoirdupois dans une tonne métrique
LB_PER_SHORT_TON = 2000.0
KG_PER_LB = 0.45359237         # définition légale exacte de la livre
LITRES_PER_US_GALLON = 3.785411784   # définition exacte du gallon US

# ---------------------------------------------------------------------------
# Poids-test par graine, en livres par boisseau (U-H1).
# Source : standards USDA. Ce sont des définitions réglementaires, pas des mesures.
# ---------------------------------------------------------------------------
BUSHEL_WEIGHT_LB: dict[str, float] = {
    "soybean": 60.0,
    "wheat": 60.0,
    "corn": 56.0,
    "sorghum": 56.0,
    "rye": 56.0,
    "barley": 48.0,
    "oats": 32.0,
}


class UnitError(ValueError):
    """Conversion impossible ou mal spécifiée — toujours une erreur d'appelant."""


# ---------------------------------------------------------------------------
# Grains : boisseau <-> tonne métrique
# ---------------------------------------------------------------------------
def bushels_per_tonne(commodity: str) -> float:
    """Nombre de boisseaux dans une tonne métrique, pour une graine nommée.

    >>> round(bushels_per_tonne("soybean"), 4)
    36.7437
    >>> round(bushels_per_tonne("corn"), 4)
    39.3683
    """
    key = commodity.strip().lower()
    if key not in BUSHEL_WEIGHT_LB:
        raise UnitError(
            f"poids-test inconnu pour '{commodity}' — graines connues : "
            f"{sorted(BUSHEL_WEIGHT_LB)}. Ne pas deviner : le facteur dépend de la graine."
        )
    return LB_PER_TONNE / BUSHEL_WEIGHT_LB[key]


def usd_bu_to_usd_t(price: Numeric, commodity: str) -> Numeric:
    """USD/boisseau -> USD/tonne métrique. La graine est obligatoire (U-H1)."""
    return price * bushels_per_tonne(commodity)


def usd_t_to_usd_bu(price: Numeric, commodity: str) -> Numeric:
    """USD/tonne métrique -> USD/boisseau."""
    return price / bushels_per_tonne(commodity)


# ---------------------------------------------------------------------------
# Cotations en cents et en short ton
# ---------------------------------------------------------------------------
def cents_lb_to_usd_t(price: Numeric) -> Numeric:
    """c/lb -> USD/t. Sucre, coton, café arabica, huile de soja.

    >>> round(cents_lb_to_usd_t(1.0), 4)
    22.0462
    """
    return price * (LB_PER_TONNE / 100.0)


def usd_t_to_cents_lb(price: Numeric) -> Numeric:
    """USD/t -> c/lb."""
    return price / (LB_PER_TONNE / 100.0)


def usd_short_ton_to_usd_t(price: Numeric) -> Numeric:
    """USD/short ton -> USD/tonne métrique. **Le tourteau CBOT est en short ton.**

    >>> round(usd_short_ton_to_usd_t(1.0), 5)
    1.10231
    """
    return price * (LB_PER_TONNE / LB_PER_SHORT_TON)


def usd_t_to_usd_short_ton(price: Numeric) -> Numeric:
    """USD/tonne métrique -> USD/short ton."""
    return price / (LB_PER_TONNE / LB_PER_SHORT_TON)


# ---------------------------------------------------------------------------
# Masse et volume
# ---------------------------------------------------------------------------
def lb_to_kg(mass: Numeric) -> Numeric:
    return mass * KG_PER_LB


def kg_to_lb(mass: Numeric) -> Numeric:
    return mass / KG_PER_LB


def gallons_to_litres(volume: Numeric) -> Numeric:
    return volume * LITRES_PER_US_GALLON


def litres_to_gallons(volume: Numeric) -> Numeric:
    return volume / LITRES_PER_US_GALLON


# ---------------------------------------------------------------------------
# Change et taxe
# ---------------------------------------------------------------------------
def local_to_usd_per_t(price_local: Numeric, fx_local_per_usd: Numeric) -> Numeric:
    """MYR/t, CNY/t, BRL/t -> USD/t.

    `fx_local_per_usd` est la cotation **locale par dollar** (USDBRL = 5.4 signifie
    5,4 BRL pour 1 USD), donc on divise. C'est le sens de cotation usuel pour BRL, CNY et
    MYR ; l'inverser passe silencieusement un prix par un facteur 25 sans lever d'erreur,
    d'où le contrôle de plausibilité ci-dessous.
    """
    if isinstance(fx_local_per_usd, float) and fx_local_per_usd <= 0:
        raise UnitError(f"taux de change doit être > 0, reçu {fx_local_per_usd}")
    return price_local / fx_local_per_usd


def strip_vat(price_incl: Numeric, vat_rate: float) -> Numeric:
    """Prix TTC -> prix HT. `vat_rate` en fraction (0.09 = 9 %).

    Chine : ~9 % sur l'agricole, ~13 % sur l'industriel — mais vérifier le taux
    applicable **au produit et à la date** (U-H3). Ce module ne choisit pas pour toi.
    """
    if not 0.0 <= vat_rate < 1.0:
        raise UnitError(f"taux de TVA doit être dans [0, 1), reçu {vat_rate}")
    return price_incl / (1.0 + vat_rate)


def add_vat(price_excl: Numeric, vat_rate: float) -> Numeric:
    """Prix HT -> prix TTC."""
    if not 0.0 <= vat_rate < 1.0:
        raise UnitError(f"taux de TVA doit être dans [0, 1), reçu {vat_rate}")
    return price_excl * (1.0 + vat_rate)


# ---------------------------------------------------------------------------
# Sucre : la comparaison No.11 / No.5 exige une polarisation (U-H2)
# ---------------------------------------------------------------------------
def raw_sugar_to_white_basis(
    ny11_cents_lb: Numeric, *, pol_adjust: float = 1.07
) -> Numeric:
    """NY No.11 (c/lb, 96° pol) -> USD/t sur base blanc raffiné.

    `pol_adjust` est le facteur de rendement brut->blanc, autour de 1,06-1,08 selon la
    spécification retenue. **Paramètre, pas constante** : le figer fausserait tout le
    white premium de T2-4, qui est précisément le résidu qu'on cherche à mesurer.
    """
    if not 1.0 <= pol_adjust <= 1.20:
        raise UnitError(
            f"pol_adjust hors plage plausible [1.00, 1.20], reçu {pol_adjust} — "
            "au-delà, ce n'est plus une correction de polarisation"
        )
    return cents_lb_to_usd_t(ny11_cents_lb) * pol_adjust


# ---------------------------------------------------------------------------
# Board crush CBOT — l'exemple canonique des trois unités dans une formule
# ---------------------------------------------------------------------------
# Rendements implicites du contrat CBOT, par boisseau de soja triture :
MEAL_LB_PER_BUSHEL = 44.0
OIL_LB_PER_BUSHEL = 11.0


def board_crush_usd_bu(
    bean_usd_bu: Numeric,
    meal_usd_short_ton: Numeric,
    oil_cents_lb: Numeric,
) -> Numeric:
    """Board crush CBOT en USD/boisseau, avec les trois unités traitées explicitement.

    Coefficients dérivés, jamais écrits en dur :
        tourteau : 44 lb/bu / 2000 lb/short ton = 0.022 short ton par boisseau
        huile    : 11 lb/bu, prix en c/lb      -> /100 pour passer en USD

    Le dollar par boisseau est conservé comme unité de sortie parce que c'est celle dans
    laquelle un desk board parle. `usd_bu_to_usd_t(..., "soybean")` pour comparer à un
    crush usine coté à la tonne.
    """
    meal_short_tons_per_bu = MEAL_LB_PER_BUSHEL / LB_PER_SHORT_TON
    oil_usd_per_lb = oil_cents_lb / 100.0
    return meal_short_tons_per_bu * meal_usd_short_ton + OIL_LB_PER_BUSHEL * oil_usd_per_lb - bean_usd_bu
