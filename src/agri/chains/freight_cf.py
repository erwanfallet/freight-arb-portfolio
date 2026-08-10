"""T1-1 — Le fret dans le calcul C&F.

THÈSE
-----
Sur la bande frontière — les jours où l'arb est à quelques dollars du breakeven — le fret
ne *bruite* pas l'arb : il le **détermine**. La dispute documentée chez Louis Dreyfus entre
desks de trading et département fret n'était pas un conflit d'ego, c'était un conflit
d'attribution de P&L, et il est arithmétiquement mesurable.

    arb_usd_t = CIF_dest − FOB_orig − freight_usd_t − financing − insurance

Tout le désaccord porte sur un seul terme : `freight_usd_t`. Les deux camps ont raison.

ORIGINE DU DÉSACCORD (sourcé, citable)
---------------------------------------
Interview Mat Halsall, Commodity Conversations, 25 nov. 2024 : chez Louis Dreyfus,
nombreuses disputes entre les desks de trading et le département fret ; les traders
contestaient le taux de fret sans comprendre ce qu'est le fret ni ses composantes. Halsall
a proposé d'apprendre le métier des traders pour leur enseigner le sien, et a passé
beaucoup de temps en réunions S&D à reverse-engineerer le business freight physique.

    Desk trading  : « votre taux n'est pas le marché, je fixe moins cher au spot. »
    Dépt fret     : « vous regardez un index, pas un coût. Il manque le positionnement du
                      navire, le ballast, le bunker à la date de fixture, le laytime, la
                      démurrage, le temps de rotation, la valeur des options de cargo. »

LE PIÈGE D'UNITÉ
----------------
L'index est coté en **USD/jour** (TCE) ou en USD/t sur une route de référence ; l'unité
économique du calcul C&F est **USD/tonne sur ta route à ta date**. Le passage n'est pas
une conversion mais une estimation de voyage — voir `core/voyage.py`. Le facteur de
conversion n'est pas une constante physique : c'est un modèle avec des hypothèses
contestables, et deux d'entre elles portent tout le désaccord.

LES TROIS CONVENTIONS
---------------------
    freight_index     ballast = 0, ce que défend le desk trading
    freight_full      ballast = 1, soutes au jour de fixture, ce que défend le dépt fret
    freight_internal  moyenne mobile 90j de freight_full — ce qu'un département fret vend
                      en interne, parce qu'il doit vendre un taux *stable*

HYPOTHÈSES
----------
F-H1  `ballast_share` par défaut à 1.0 pour `freight_full`, 0.0 pour `freight_index`.
      C'est le désaccord central, et la page le balaie plutôt que de trancher.
F-H2  La convention interne est une moyenne mobile 90 jours. Un vrai département fret
      lisse autrement (budget, taux négocié annuel) ; la MM90 est le proxy le plus simple
      qui reproduise la propriété qui compte : la stabilité, donc le décalage.
F-H3  L'assurance est un forfait en USD/t, pas un pourcentage de la valeur. Faux en
      toute rigueur, négligeable devant le fret, et paramétré.
F-H4  Le financement court sur la durée de voyage plus les jours de crédit. Le taux est
      exogène. Le terme est petit mais pas nul sur la bande frontière — donc affiché.
F-H5  Aucun coût de démurrage n'est modélisé. C'est un des termes que le département fret
      cite, et son omission **sous-estime** `freight_full` : le biais va contre la thèse,
      donc dans le bon sens.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from agri.data.snapshot import cached

from agri.core.breakeven import Breakeven, NoBreakevenInRange, solve_breakeven
from agri.core.stats import ProportionCI, clopper_pearson, regime_runs
from agri.core.voyage import (
    Route,
    VesselClass,
    VoyageParams,
    voyage_freight_series,
)

# F-H2
DEFAULT_INTERNAL_WINDOW_DAYS = 90
# Largeur par défaut de la zone de décision marginale, en USD/t
DEFAULT_MARGINAL_BAND_USD_T = 5.0

CONVENTIONS = ("index", "full", "internal")


class FreightCfError(ValueError):
    """Décomposition mal spécifiée."""


# ===========================================================================
# L'identité comptable
# ===========================================================================
def financing_cost_usd_t(
    fob_usd_t: pd.Series | float,
    freight_usd_t: pd.Series | float,
    *,
    annual_rate: float,
    voyage_days: float,
    credit_days: float,
) -> pd.Series | float:
    """Financement de la cargaison sur la durée de voyage plus le crédit (F-H4).

    Base 360 jours, convention de marché monétaire — pas 365. L'écart est de 1,4 % sur le
    terme, invisible sur un arb large et pas sur la bande frontière.
    """
    if annual_rate < 0:
        raise FreightCfError(f"taux de financement négatif : {annual_rate}")
    if voyage_days < 0 or credit_days < 0:
        raise FreightCfError("les durées doivent être >= 0")
    return (fob_usd_t + freight_usd_t) * annual_rate * (voyage_days + credit_days) / 360.0


def arb_usd_t(
    cif_usd_t: pd.Series,
    fob_usd_t: pd.Series,
    freight_usd_t: pd.Series,
    *,
    annual_rate: float,
    voyage_days: float,
    credit_days: float = 30.0,
    insurance_usd_t: float = 0.85,
) -> pd.Series:
    """L'identité complète. Un seul terme est contesté : `freight_usd_t`."""
    financing = financing_cost_usd_t(
        fob_usd_t,
        freight_usd_t,
        annual_rate=annual_rate,
        voyage_days=voyage_days,
        credit_days=credit_days,
    )
    return cif_usd_t - fob_usd_t - freight_usd_t - financing - insurance_usd_t


# ===========================================================================
# Les trois conventions, côte à côte
# ===========================================================================
def build_conventions(
    tce: pd.Series,
    vlsfo: pd.Series,
    mgo: pd.Series,
    cif: pd.Series,
    fob: pd.Series,
    *,
    vessel: VesselClass,
    route: Route,
    params: VoyageParams,
    annual_rate: float = 0.055,
    credit_days: float = 30.0,
    insurance_usd_t: float = 0.85,
    internal_window_days: int = DEFAULT_INTERNAL_WINDOW_DAYS,
) -> pd.DataFrame:
    """Le fret et l'arb sous les trois conventions, sur les dates communes à tout.

    Colonnes : freight_index, freight_full, freight_internal, arb_index, arb_full,
    arb_internal, spread_full_index, et `disagreement` — vrai quand les trois conventions
    ne s'accordent pas sur le signe de l'arb.

    Aucun forward-fill nulle part : le prix des soutes à la date de fixture est
    précisément l'objet du désaccord, le combler le ferait disparaître.
    """
    freight_index = voyage_freight_series(
        tce, vlsfo, mgo, vessel=vessel, route=route, params=params.with_ballast(0.0)
    )
    freight_full = voyage_freight_series(
        tce, vlsfo, mgo, vessel=vessel, route=route, params=params.with_ballast(1.0)
    )
    # F-H2 : min_periods complet, on refuse une moyenne « interne » sur trois points
    freight_internal = freight_full.rolling(
        internal_window_days, min_periods=internal_window_days
    ).mean()

    frame = pd.concat(
        {
            "cif": cif,
            "fob": fob,
            "freight_index": freight_index,
            "freight_full": freight_full,
            "freight_internal": freight_internal,
        },
        axis=1,
    ).dropna()
    if frame.empty:
        raise FreightCfError(
            "aucune date commune aux prix et aux frets après la fenêtre de lissage — "
            f"il faut au moins {internal_window_days} jours d'historique de fret avant "
            "la première date d'arb exploitable"
        )

    # la durée de voyage sert au financement : on prend celle de la convention full,
    # qui est la seule à représenter un cycle complet
    reference_days = _reference_voyage_days(vessel, route, params)

    for convention in CONVENTIONS:
        frame[f"arb_{convention}"] = arb_usd_t(
            frame["cif"],
            frame["fob"],
            frame[f"freight_{convention}"],
            annual_rate=annual_rate,
            voyage_days=reference_days,
            credit_days=credit_days,
            insurance_usd_t=insurance_usd_t,
        )

    frame["spread_full_index"] = frame["freight_full"] - frame["freight_index"]
    signs = np.sign(frame[[f"arb_{c}" for c in CONVENTIONS]].to_numpy())
    frame["disagreement"] = (signs != signs[:, [0]]).any(axis=1)
    frame.attrs["reference_voyage_days"] = reference_days
    frame.attrs["internal_window_days"] = internal_window_days
    return frame


def _reference_voyage_days(vessel: VesselClass, route: Route, params: VoyageParams) -> float:
    from agri.core.voyage import HOURS_PER_DAY

    laden = route.distance_laden_nm / (params.speed_laden_kn * HOURS_PER_DAY)
    ballast = route.distance_ballast_nm / (params.speed_ballast_kn * HOURS_PER_DAY)
    return laden + ballast + params.port_days + params.wait_days


# ===========================================================================
# S4 — le panneau de désaccord
# ===========================================================================
def sign_flip_rate(
    frame: pd.DataFrame, *, left: str = "index", right: str = "full"
) -> ProportionCI:
    """Part des jours où deux conventions ne donnent pas le même signe d'arb.

    IC binomial exact (Clopper-Pearson) plutôt qu'approximation normale : sur un
    échantillon d'un an de jours ouvrés et un taux proche de 0 ou de 1, l'approximation
    normale produit des bornes hors de [0, 1].
    """
    a = np.sign(frame[f"arb_{left}"].to_numpy())
    b = np.sign(frame[f"arb_{right}"].to_numpy())
    flips = int((a != b).sum())
    return clopper_pearson(flips, len(frame))


def spread_distribution(frame: pd.DataFrame) -> dict[str, float]:
    """Médiane et IQR de l'écart `freight_full − freight_index`, pas seulement la moyenne.

    La distribution est asymétrique **par construction** : le ballast ne peut que
    renchérir. Reporter une moyenne seule sur une distribution bornée à gauche est le
    genre de raccourci qu'un praticien repère immédiatement.
    """
    spread = frame["spread_full_index"].dropna()
    q1, q3 = spread.quantile([0.25, 0.75])
    return {
        "median": float(spread.median()),
        "mean": float(spread.mean()),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "min": float(spread.min()),
        "max": float(spread.max()),
    }


def spread_seasonality(frame: pd.DataFrame) -> pd.Series:
    """Écart moyen `full − index` par mois calendaire. Le ballast a une saison."""
    spread = frame["spread_full_index"].dropna()
    return spread.groupby(spread.index.month).mean()


# ===========================================================================
# S5 — la zone de décision marginale : LE chiffre du mail
# ===========================================================================
@dataclass(frozen=True)
class MarginalZone:
    """Ce qui se passe sur la bande frontière, là où le fret cesse d'être un détail."""

    band_usd_t: float
    n_total: int
    n_in_band: int
    n_decided_by_convention: int
    share_of_sample: float
    decision_rate: ProportionCI

    @property
    def headline(self) -> str:
        """La phrase chiffrée, générée depuis les données."""
        return (
            f"Sur les jours où l'arb est à moins de {self.band_usd_t:g} $/t du breakeven — "
            f"{self.share_of_sample:.0%} de l'échantillon — c'est la convention de fret, "
            f"et non le marché, qui détermine si l'arb est ouvert dans "
            f"{self.decision_rate.point:.0%} des cas "
            f"(IC 95 % exact : {self.decision_rate.lo:.0%}–{self.decision_rate.hi:.0%})."
        )


def marginal_decision_zone(
    frame: pd.DataFrame, *, band_usd_t: float = DEFAULT_MARGINAL_BAND_USD_T
) -> MarginalZone:
    """Restreint aux jours de bande frontière, et mesure la part décidée par la convention.

    « Décidée par la convention » a un sens précis : parmi les jours où l'arb sous la
    convention de référence est à moins de `band_usd_t` de zéro, ceux où deux conventions
    donnent des signes opposés. Ce n'est pas « le fret compte » — c'est « le fret tranche ».
    """
    if band_usd_t <= 0:
        raise FreightCfError(f"la bande doit être > 0, reçu {band_usd_t}")

    in_band = frame[frame["arb_full"].abs() < band_usd_t]
    n_in_band = len(in_band)
    if n_in_band == 0:
        return MarginalZone(
            band_usd_t=band_usd_t,
            n_total=len(frame),
            n_in_band=0,
            n_decided_by_convention=0,
            share_of_sample=0.0,
            decision_rate=clopper_pearson(0, 1),
        )

    decided = int(
        (np.sign(in_band["arb_index"]) != np.sign(in_band["arb_full"])).sum()
    )
    return MarginalZone(
        band_usd_t=band_usd_t,
        n_total=len(frame),
        n_in_band=n_in_band,
        n_decided_by_convention=decided,
        share_of_sample=n_in_band / len(frame),
        decision_rate=clopper_pearson(decided, n_in_band),
    )


# ===========================================================================
# Point de bascule — ballast_share*
# ===========================================================================
def ballast_breakeven(
    tce_usd_day: float,
    vlsfo_usd_t: float,
    mgo_usd_t: float,
    cif_usd_t: float,
    fob_usd_t: float,
    *,
    vessel: VesselClass,
    route: Route,
    params: VoyageParams,
    annual_rate: float = 0.055,
    credit_days: float = 30.0,
    insurance_usd_t: float = 0.85,
    ballast_history: pd.Series | None = None,
) -> Breakeven:
    """`ballast_share*` : la part de ballast facturé à laquelle l'arb change de signe.

    C'est le livrable de la page. Réponse type : « au-delà de 40 % de ballast facturé,
    l'arb Santos->Qingdao se ferme aux prix d'aujourd'hui » — un seuil qu'un praticien
    conteste ou confirme en dix secondes.

    Lève `NoBreakevenInRange` quand l'arb garde le même signe sur toute la plage [0, 1].
    Ce n'est pas un échec : « même en facturant 100 % du ballast, l'arb reste ouvert » est
    une affirmation publiable, et souvent la plus forte de la journée.
    """
    from agri.core.voyage import voyage_freight_usd_t

    def margin(share: float) -> float:
        local = params.with_ballast(float(np.clip(share, 0.0, 1.0)))
        breakdown = voyage_freight_usd_t(
            tce_usd_day, vlsfo_usd_t, mgo_usd_t, vessel=vessel, route=route, params=local
        )
        financing = financing_cost_usd_t(
            fob_usd_t,
            breakdown.freight_usd_t,
            annual_rate=annual_rate,
            voyage_days=breakdown.total_days,
            credit_days=credit_days,
        )
        return cif_usd_t - fob_usd_t - breakdown.freight_usd_t - financing - insurance_usd_t

    return solve_breakeven(
        margin,
        0.0,
        1.0,
        theta_current=params.ballast_share,
        theta_history=ballast_history,
        theta_label="ballast_share",
        margin_label="arb",
    )


def sensitivity_grid(
    tce_usd_day: float,
    vlsfo_usd_t: float,
    mgo_usd_t: float,
    cif_usd_t: float,
    fob_usd_t: float,
    *,
    vessel: VesselClass,
    route: Route,
    params: VoyageParams,
    ballast_values: np.ndarray | None = None,
    bunker_shifts_pct: np.ndarray | None = None,
    annual_rate: float = 0.055,
) -> pd.DataFrame:
    """S6 — heatmap `ballast_share × décalage du prix des soutes -> arb`.

    Les deux paramètres qui portent le désaccord, croisés. Le décalage de soutes est en
    pourcentage plutôt qu'en jours : un décalage de date n'a d'effet que par le prix qu'il
    implique, et raisonner en prix rend la sensibilité lisible sans série de soutes longue.
    """
    from agri.core.voyage import voyage_freight_usd_t

    ballast_values = (
        np.linspace(0.0, 1.0, 11) if ballast_values is None else np.asarray(ballast_values)
    )
    bunker_shifts_pct = (
        np.linspace(-0.30, 0.30, 13)
        if bunker_shifts_pct is None
        else np.asarray(bunker_shifts_pct)
    )

    rows = []
    for share in ballast_values:
        for shift in bunker_shifts_pct:
            local = params.with_ballast(float(share))
            breakdown = voyage_freight_usd_t(
                tce_usd_day,
                vlsfo_usd_t * (1.0 + shift),
                mgo_usd_t * (1.0 + shift),
                vessel=vessel,
                route=route,
                params=local,
            )
            financing = financing_cost_usd_t(
                fob_usd_t,
                breakdown.freight_usd_t,
                annual_rate=annual_rate,
                voyage_days=breakdown.total_days,
                credit_days=30.0,
            )
            arb = cif_usd_t - fob_usd_t - breakdown.freight_usd_t - financing - 0.85
            rows.append(
                {
                    "ballast_share": float(share),
                    "bunker_shift_pct": float(shift),
                    "freight_usd_t": breakdown.freight_usd_t,
                    "arb_usd_t": arb,
                    "is_open": arb > 0,
                }
            )
    return pd.DataFrame(rows)


# ===========================================================================
# S7 — attribution de P&L entre les deux desks
# ===========================================================================
def pnl_attribution(frame: pd.DataFrame, *, cargo_t: float) -> pd.DataFrame:
    """Le P&L déplacé d'un desk à l'autre par la convention interne.

        P&L_déplacé = (freight_internal − freight_full) × cargo_t

    Signe positif : le département fret a facturé au-dessus de son coût du jour, donc le
    desk trading a été débité et le département crédité. Négatif : l'inverse. Cumulé sur
    l'échantillon, c'est le montant que la dispute LDC portait réellement — et il a un
    propriétaire identifiable des deux côtés.
    """
    if cargo_t <= 0:
        raise FreightCfError(f"cargo_t doit être > 0, reçu {cargo_t}")
    out = pd.DataFrame(index=frame.index)
    out["gap_usd_t"] = frame["freight_internal"] - frame["freight_full"]
    out["pnl_shifted_usd"] = out["gap_usd_t"] * cargo_t
    out["pnl_shifted_cum_usd"] = out["pnl_shifted_usd"].cumsum()
    out["credited"] = np.where(
        out["gap_usd_t"] > 0, "département fret", "desk trading"
    )
    return out


# ===========================================================================
# SUR DONNÉES RÉELLES — ce que le taux publié implique, selon la convention
# ===========================================================================
@dataclass(frozen=True)
class ImpliedTceSpread:
    """Le même taux de route publié, traduit en TCE sous les deux conventions.

    C'est le désaccord exprimé dans l'unité où le département fret pense réellement.
    Un desk trading lit « 55 $/t » et conclut que le marché paie 55 $/t. Un département
    fret lit le même print et demande : sur combien de jours ? Si le voyage inclut le
    repositionnement à vide, le même revenu doit couvrir presque deux fois plus de jours,
    donc le TCE que ce taux dégage est presque deux fois plus faible.
    """

    route_rate_usd_t: pd.Series
    tce_no_ballast: pd.Series
    tce_full_ballast: pd.Series

    @property
    def spread(self) -> pd.Series:
        return (self.tce_no_ballast - self.tce_full_ballast).rename("spread_usd_day")

    @property
    def headline(self) -> str:
        latest_rate = float(self.route_rate_usd_t.iloc[-1])
        no_ballast = float(self.tce_no_ballast.iloc[-1])
        full = float(self.tce_full_ballast.iloc[-1])
        return (
            f"Le taux de route publié de {latest_rate:,.1f} $/t dégage un TCE de "
            f"{no_ballast:,.0f} $/jour si l'on ne facture aucun ballast, et de "
            f"{full:,.0f} $/jour si on le facture entièrement — soit un écart de "
            f"{no_ballast - full:,.0f} $/jour sur le même print de marché. C'est le "
            "nombre sur lequel les deux desks se disputent, dans l'unité du département fret."
        )


def implied_tce_by_convention(
    route_rate_usd_t: pd.Series,
    vlsfo_usd_t: pd.Series,
    mgo_usd_t: pd.Series,
    *,
    vessel: VesselClass,
    route: Route,
    params: VoyageParams,
) -> ImpliedTceSpread:
    """Inverse le modèle de voyage sur un taux de route **réellement publié**.

    Le taux publié est une donnée ; le TCE qu'il dégage est une conclusion, et elle
    dépend entièrement de la convention de ballast. On ne choisit pas la bonne convention
    — on montre l'écart que le choix produit.
    """
    from agri.core.voyage import implied_tce_from_freight

    aligned = pd.concat(
        {"rate": route_rate_usd_t, "vlsfo": vlsfo_usd_t, "mgo": mgo_usd_t}, axis=1, sort=True
    ).dropna()
    if aligned.empty:
        raise FreightCfError("aucune date commune au taux de route et aux soutes")

    def invert(share: float) -> pd.Series:
        local = params.with_ballast(share)
        values = [
            implied_tce_from_freight(
                row.rate, row.vlsfo, row.mgo, vessel=vessel, route=route, params=local
            )
            for row in aligned.itertuples()
        ]
        return pd.Series(values, index=aligned.index)

    return ImpliedTceSpread(
        route_rate_usd_t=aligned["rate"],
        tce_no_ballast=invert(0.0).rename("tce_ballast_0"),
        tce_full_ballast=invert(1.0).rename("tce_ballast_1"),
    )


@dataclass(frozen=True)
class MarketImpliedBallast:
    """La part de ballast que le marché price, une fois un TCE de référence connu.

    LE LIVRABLE DE LA PAGE. Le désaccord n'est pas tranchable en théorie — mais si l'on
    connaît le TCE auquel les armateurs affrètent réellement, alors le taux de route
    publié révèle la part de ballast que le marché a déjà intégrée. La question au desk
    cesse d'être « qui a raison » et devient « le marché price X %, est-ce ce que vous
    facturez en interne ? ».
    """

    reference_tce_usd_day: float
    implied_share: float | None
    route_rate_usd_t: float
    bracket: tuple[float, float]
    reason: str = ""

    @property
    def headline(self) -> str:
        if self.implied_share is None:
            return (
                f"À un TCE de référence de {self.reference_tce_usd_day:,.0f} $/jour, le "
                f"taux publié de {self.route_rate_usd_t:,.1f} $/t n'est compatible avec "
                f"aucune part de ballast dans [{self.bracket[0]:.0%}, {self.bracket[1]:.0%}] "
                f"— {self.reason}"
            )
        return (
            f"À un TCE de référence de {self.reference_tce_usd_day:,.0f} $/jour, le taux "
            f"publié de {self.route_rate_usd_t:,.1f} $/t implique que le marché facture "
            f"**{self.implied_share:.0%} du repositionnement à vide**. C'est le chiffre "
            "que votre taux interne doit égaler pour être neutre."
        )


def market_implied_ballast_share(
    route_rate_usd_t: float,
    reference_tce_usd_day: float,
    vlsfo_usd_t: float,
    mgo_usd_t: float,
    *,
    vessel: VesselClass,
    route: Route,
    params: VoyageParams,
) -> MarketImpliedBallast:
    """Résout la part de ballast telle que le modèle reproduise le taux publié.

    Le fret est **affine croissant** en part de ballast (chaque point de ballast ajoute
    des jours et des soutes), donc la racine est unique quand elle existe. Quand elle
    n'existe pas, c'est un résultat : le taux publié est hors de la plage que le modèle
    peut produire, ce qui signale une hypothèse de voyage à revoir — vitesse, frais de
    port, ou classe de navire — plutôt qu'une part de ballast aberrante.
    """
    from agri.core.voyage import voyage_freight_usd_t

    def gap(share: float) -> float:
        local = params.with_ballast(float(np.clip(share, 0.0, 1.0)))
        modelled = voyage_freight_usd_t(
            reference_tce_usd_day, vlsfo_usd_t, mgo_usd_t,
            vessel=vessel, route=route, params=local,
        ).freight_usd_t
        return route_rate_usd_t - modelled

    try:
        solution = solve_breakeven(
            gap, 0.0, 1.0, theta_label="ballast_share", margin_label="écart au taux publié"
        )
        return MarketImpliedBallast(
            reference_tce_usd_day=reference_tce_usd_day,
            implied_share=solution.theta_star,
            route_rate_usd_t=route_rate_usd_t,
            bracket=(0.0, 1.0),
        )
    except NoBreakevenInRange as error:
        # `gap` est DÉCROISSANT en part de ballast (plus de ballast -> fret modélisé plus
        # cher -> écart plus petit). Le message doit donc nommer la borne CONTRAIGNANTE,
        # c'est-à-dire celle où le modèle s'approche le plus du taux publié : ballast plein
        # quand l'écart reste positif partout, ballast nul quand il reste négatif partout.
        # Nommer l'autre borne serait vrai mais inutile — elle est la plus éloignée.
        if error.margin_lo > 0:
            reason = (
                "le taux publié dépasse ce que le modèle produit même en facturant 100 % "
                "du repositionnement à vide : le TCE de référence retenu est trop bas pour "
                "cette route, ou les hypothèses de voyage (vitesse, jours de port, frais "
                "portuaires) sous-estiment le coût du cycle"
            )
        else:
            reason = (
                "le taux publié reste sous ce que le modèle produit même sans facturer "
                "aucun ballast : le TCE de référence retenu est trop élevé pour cette route"
            )
        return MarketImpliedBallast(
            reference_tce_usd_day=reference_tce_usd_day,
            implied_share=None,
            route_rate_usd_t=route_rate_usd_t,
            bracket=(0.0, 1.0),
            reason=reason,
        )


@cached('t1_1_route', from_frame=lambda f: ImpliedTceSpread(f["route_rate_usd_t"], f["tce_no_ballast"], f["tce_full_ballast"]))
def load_real_route_frame(
    *,
    vessel_key: str = "panamax",
    route_key: str = "santos_qingdao",
    params: VoyageParams | None = None,
    mgo_premium: float = 1.35,
) -> ImpliedTceSpread:
    """Charge le taux de route P8 réel et les soutes réelles, puis inverse le modèle.

    `mgo_premium` reconstruit le MGO à partir du VLSFO : l'export ne contient pas de série
    MGO séparée, et le MGO cote historiquement 30 à 40 % au-dessus du VLSFO. Le terme est
    petit — les soutes de port pèsent quelques pour cent du coût de voyage — mais le
    paramètre est exposé plutôt que caché dans une constante.
    """
    from agri.core.voyage import ROUTES, VESSELS
    from agri.data.bloomberg_loader import load as load_bloomberg

    rate = load_bloomberg("p8_route_usd_t")
    vlsfo = load_bloomberg("vlsfo_singapore")
    mgo = (vlsfo * mgo_premium).rename("mgo_reconstruit")

    return implied_tce_by_convention(
        rate, vlsfo, mgo,
        vessel=VESSELS[vessel_key], route=ROUTES[route_key],
        params=params or VoyageParams(),
    )


def disagreement_episodes(frame: pd.DataFrame, *, min_days: int = 3) -> pd.DataFrame:
    """Épisodes consécutifs où les trois conventions ne s'accordent pas sur le signe.

    « Les conventions ont divergé pendant 11 jours ouvrés en mars » est une phrase datée
    et vérifiable ; « les conventions divergent parfois » n'en est pas une.
    """
    return regime_runs(
        frame["disagreement"], depth=frame["spread_full_index"], min_obs=min_days
    )


__all__ = [
    "CONVENTIONS",
    "FreightCfError",
    "ImpliedTceSpread",
    "MarginalZone",
    "MarketImpliedBallast",
    "NoBreakevenInRange",
    "arb_usd_t",
    "ballast_breakeven",
    "build_conventions",
    "disagreement_episodes",
    "financing_cost_usd_t",
    "implied_tce_by_convention",
    "load_real_route_frame",
    "marginal_decision_zone",
    "market_implied_ballast_share",
    "pnl_attribution",
    "sensitivity_grid",
    "sign_flip_rate",
    "spread_distribution",
    "spread_seasonality",
]
