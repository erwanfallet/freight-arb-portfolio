"""Estimation de voyage : d'un index de fret à un coût par tonne sur *ta* route.

LE PIÈGE D'UNITÉ DU PORTEFEUILLE, APPLIQUÉ AU FRET
---------------------------------------------------
Un index de fret dry bulk se cote en **USD/jour** (timecharter equivalent) ou en USD/t sur
une route de référence qui n'est pas la tienne. L'unité économique dont tu as besoin dans
un calcul C&F est **USD/tonne sur ta route à ta date**. Le passage de l'une à l'autre
n'est pas une conversion : c'est une **estimation de voyage**, et chacun de ses paramètres
est un point de désaccord légitime entre un desk de trading et un département fret.

C'est la même famille que tonne sèche/tonne humide ou gallon/tonne : l'unité de cotation
n'est pas l'unité économique. Sauf qu'ici le facteur de conversion n'est pas une constante
physique — c'est un modèle, avec des hypothèses contestables. D'où la page.

MODÈLE
------
    freight_usd_t =
        [ TCE_usd_day × (D_laden + D_ballast + D_port + D_wait)
        + P_vlsfo × conso_mer × (D_laden + D_ballast)
        + P_mgo   × conso_port × D_port
        + port_costs + canal_dues ]
        / ( cargo_t × (1 − broker_comm) )

    D_laden   = distance_laden_nm  / (speed_laden_kn   × 24)
    D_ballast = ballast_share × distance_ballast_nm / (speed_ballast_kn × 24)

HYPOTHÈSES
----------
V-H1  `ballast_share` — la fraction du repositionnement à vide imputée à ce voyage.
      **C'est le désaccord central de T1-1.** Le desk trading raisonne à 0 (« le navire
      était déjà là »), le département fret à 1 (« quelqu'un paie le ballast »).
      Aucune des deux positions n'est absurde ; le chiffre qui compte est le seuil où
      elles cessent de donner le même signal d'arb.
V-H2  La consommation varie au cube de la vitesse : conso = conso_ref × (v/v_ref)³.
      Approximation d'ingénierie standard, valable dans une plage étroite autour de la
      vitesse de référence. Au-delà de ±3 nœuds elle devient douteuse — d'où les bornes
      des sliders.
V-H3  Les soutes sont valorisées à une date de choix (fixture, moyenne de voyage, spot).
      **C'est le second désaccord de T1-1** : un voyage de 40 jours sur un marché
      soutes volatil peut porter plusieurs dollars par tonne d'écart selon la convention.
V-H4  Les frais de port et les droits de canal sont forfaitaires par route. Dans la
      réalité ils dépendent du terminal et du tirant d'eau. Paramétrés, jamais figés.
V-H5  Le TCE fourni est celui d'une classe de navire cohérente avec la cargaison. Charger
      un TCE Capesize dans un voyage Panamax produit un fret absurde sans lever d'erreur —
      d'où le contrôle de plausibilité en sortie.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

HOURS_PER_DAY = 24.0


class VoyageError(ValueError):
    """Voyage mal spécifié — toujours une erreur d'appelant."""


# ---------------------------------------------------------------------------
# Classes de navire (V-H2, V-H5)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VesselClass:
    """Consommations de référence à la vitesse de référence, en tonnes par jour."""

    name: str
    cargo_t: float
    consumption_laden_t_day: float
    consumption_ballast_t_day: float
    consumption_port_t_day: float      # MGO au port, pas VLSFO
    reference_speed_kn: float = 12.5

    def sea_consumption(self, speed_kn: float, *, laden: bool) -> float:
        """Consommation journalière à la vitesse donnée, loi cubique (V-H2)."""
        if speed_kn <= 0:
            raise VoyageError(f"la vitesse doit être > 0, reçu {speed_kn}")
        base = self.consumption_laden_t_day if laden else self.consumption_ballast_t_day
        return base * (speed_kn / self.reference_speed_kn) ** 3


VESSELS: dict[str, VesselClass] = {
    "supramax": VesselClass("Supramax 55", 55_000, 22.0, 20.0, 2.5),
    "panamax": VesselClass("Panamax 66", 66_000, 26.0, 24.0, 3.0),
    "kamsarmax": VesselClass("Kamsarmax 82", 82_000, 30.0, 27.5, 3.2),
}


# ---------------------------------------------------------------------------
# Routes (V-H4). Distances en milles nautiques, ordres de grandeur à revérifier
# avant tout usage sur données réelles — elles sont ici pour faire tourner le modèle.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Route:
    name: str
    origin: str
    destination: str
    distance_laden_nm: float
    distance_ballast_nm: float
    port_costs_usd: float = 150_000.0
    canal_dues_usd: float = 0.0


ROUTES: dict[str, Route] = {
    "santos_qingdao": Route(
        "Santos -> Qingdao", "Santos", "Qingdao", 11_000, 11_000, 180_000, 0.0
    ),
    "usgulf_qingdao": Route(
        "US Gulf -> Qingdao (Panama)", "US Gulf", "Qingdao", 9_800, 9_800, 190_000, 420_000
    ),
    "pnw_qingdao": Route(
        "PNW -> Qingdao", "PNW", "Qingdao", 5_100, 5_100, 160_000, 0.0
    ),
    "novo_alexandria": Route(
        "Novorossiysk -> Alexandrie", "Novorossiysk", "Alexandrie", 1_000, 1_000, 90_000, 0.0
    ),
    "rouen_alexandria": Route(
        "Rouen -> Alexandrie", "Rouen", "Alexandrie", 2_700, 2_700, 110_000, 0.0
    ),
}


# ---------------------------------------------------------------------------
# Paramètres de voyage
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VoyageParams:
    """Les paramètres contestables. `ballast_share` est le sujet de T1-1."""

    ballast_share: float = 1.0
    speed_laden_kn: float = 12.5
    speed_ballast_kn: float = 13.0
    port_days: float = 6.0
    wait_days: float = 0.0
    broker_commission: float = 0.0375

    def __post_init__(self) -> None:
        if not 0.0 <= self.ballast_share <= 1.0:
            raise VoyageError(
                f"ballast_share doit être dans [0, 1], reçu {self.ballast_share} — "
                "au-delà de 1 on facturerait plus de ballast qu'il n'y a de route"
            )
        if not 0.0 <= self.broker_commission < 0.25:
            raise VoyageError(f"commission hors plage plausible : {self.broker_commission}")
        if self.port_days < 0 or self.wait_days < 0:
            raise VoyageError("les jours de port et d'attente doivent être >= 0")

    def with_ballast(self, share: float) -> "VoyageParams":
        """Copie avec un autre ballast — c'est le balayage de la page T1-1."""
        return replace(self, ballast_share=share)


@dataclass(frozen=True)
class VoyageBreakdown:
    """Décomposition terme à terme. Chaque champ est une ligne du waterfall du dashboard."""

    laden_days: float
    ballast_days: float
    port_days: float
    wait_days: float
    total_days: float
    hire_usd: float
    bunker_sea_usd: float
    bunker_port_usd: float
    port_costs_usd: float
    canal_dues_usd: float
    total_cost_usd: float
    cargo_t: float
    freight_usd_t: float

    @property
    def waterfall(self) -> dict[str, float]:
        """Les postes en USD/t, pour la section « d'où vient le taux »."""
        payable = self.cargo_t
        return {
            "affrètement": self.hire_usd / payable,
            "soutes mer": self.bunker_sea_usd / payable,
            "soutes port": self.bunker_port_usd / payable,
            "frais de port": self.port_costs_usd / payable,
            "droits de canal": self.canal_dues_usd / payable,
        }


def voyage_freight_usd_t(
    tce_usd_day: float,
    vlsfo_usd_t: float,
    mgo_usd_t: float,
    *,
    vessel: VesselClass,
    route: Route,
    params: VoyageParams,
) -> VoyageBreakdown:
    """Convertit un TCE en USD/jour en un fret en USD/tonne sur une route donnée.

    C'est l'opération que le desk trading croit être une lecture d'index et que le
    département fret sait être une estimation. Tous les termes sont renvoyés séparément
    pour que la page puisse montrer lequel porte le désaccord.
    """
    if tce_usd_day < 0:
        raise VoyageError(
            f"TCE négatif ({tce_usd_day}) — possible en marché déprimé mais suspect : "
            "vérifier l'unité de l'index avant de continuer"
        )
    if vlsfo_usd_t <= 0 or mgo_usd_t <= 0:
        raise VoyageError("les prix de soutes doivent être > 0")

    laden_days = route.distance_laden_nm / (params.speed_laden_kn * HOURS_PER_DAY)
    ballast_days = (
        params.ballast_share
        * route.distance_ballast_nm
        / (params.speed_ballast_kn * HOURS_PER_DAY)
    )
    total_days = laden_days + ballast_days + params.port_days + params.wait_days

    conso_laden = vessel.sea_consumption(params.speed_laden_kn, laden=True)
    conso_ballast = vessel.sea_consumption(params.speed_ballast_kn, laden=False)
    bunker_sea = vlsfo_usd_t * (conso_laden * laden_days + conso_ballast * ballast_days)
    bunker_port = mgo_usd_t * vessel.consumption_port_t_day * params.port_days

    hire = tce_usd_day * total_days
    total_cost = hire + bunker_sea + bunker_port + route.port_costs_usd + route.canal_dues_usd

    payable_cargo = vessel.cargo_t * (1.0 - params.broker_commission)
    freight = total_cost / payable_cargo

    return VoyageBreakdown(
        laden_days=laden_days,
        ballast_days=ballast_days,
        port_days=params.port_days,
        wait_days=params.wait_days,
        total_days=total_days,
        hire_usd=hire,
        bunker_sea_usd=bunker_sea,
        bunker_port_usd=bunker_port,
        port_costs_usd=route.port_costs_usd,
        canal_dues_usd=route.canal_dues_usd,
        total_cost_usd=total_cost,
        cargo_t=payable_cargo,
        freight_usd_t=freight,
    )


def voyage_freight_series(
    tce: pd.Series,
    vlsfo: pd.Series,
    mgo: pd.Series,
    *,
    vessel: VesselClass,
    route: Route,
    params: VoyageParams,
) -> pd.Series:
    """`voyage_freight_usd_t` appliqué jour par jour, sur les dates communes aux trois séries.

    Intersection stricte, aucun forward-fill : le prix des soutes à la date de fixture est
    précisément l'objet du désaccord V-H3, le combler le ferait disparaître.
    """
    aligned = pd.concat({"tce": tce, "vlsfo": vlsfo, "mgo": mgo}, axis=1).dropna()
    if aligned.empty:
        raise VoyageError(
            "aucune date commune au TCE et aux deux soutes — vérifier les calendriers"
        )
    values = [
        voyage_freight_usd_t(
            row.tce, row.vlsfo, row.mgo, vessel=vessel, route=route, params=params
        ).freight_usd_t
        for row in aligned.itertuples()
    ]
    return pd.Series(values, index=aligned.index, name="freight_usd_t")


def implied_tce_from_freight(
    freight_usd_t: float,
    vlsfo_usd_t: float,
    mgo_usd_t: float,
    *,
    vessel: VesselClass,
    route: Route,
    params: VoyageParams,
) -> float:
    """L'inverse : d'un taux de route publié en USD/t vers le TCE qu'il implique.

    Sert au repli du gate T1-1. Quand les routes Baltic en USD/t sont accessibles mais pas
    le TCE, ou l'inverse, on passe de l'un à l'autre — et l'écart entre le TCE impliqué par
    un taux de route publié et le TCE coté est **lui-même** une mesure du désaccord.
    """
    reference = voyage_freight_usd_t(
        0.0, vlsfo_usd_t, mgo_usd_t, vessel=vessel, route=route, params=params
    )
    costs_only = reference.freight_usd_t          # fret à TCE nul = coûts seuls, par tonne
    if reference.total_days <= 0:
        raise VoyageError("durée de voyage nulle — route ou vitesses incohérentes")
    return (freight_usd_t - costs_only) * reference.cargo_t / reference.total_days


def plausibility_warnings(breakdown: VoyageBreakdown) -> list[str]:
    """Contrôles d'invariants (V-H5). Un print aberrant est un diagnostic de données.

    Rien ici n'est un signal de marché : ce sont les cas où le résultat ne peut pas être
    lu du tout, et où la page doit afficher un avertissement plutôt qu'un chiffre.
    """
    warnings: list[str] = []
    if breakdown.freight_usd_t <= 0:
        warnings.append(
            f"fret négatif ou nul ({breakdown.freight_usd_t:.2f} USD/t) — impossible : "
            "erreur d'unité sur le TCE ou sur les soutes"
        )
    if breakdown.total_days > 120:
        warnings.append(
            f"voyage de {breakdown.total_days:.0f} jours — vérifier la distance et la vitesse"
        )
    bunker_share = (
        (breakdown.bunker_sea_usd + breakdown.bunker_port_usd) / breakdown.total_cost_usd
        if breakdown.total_cost_usd > 0
        else np.nan
    )
    if np.isfinite(bunker_share) and bunker_share > 0.75:
        warnings.append(
            f"les soutes pèsent {bunker_share:.0%} du coût total — plausible en pic de "
            "bunker, mais vérifier la classe de navire avant de conclure"
        )
    return warnings
