"""T2-3 — Le board crush n'est pas un prix, c'est un rendement déguisé en prix.

L'HISTOIRE
----------
Le board crush s'écrit `0,022 x tourteau + 0,11 x huile - fève`. Ces deux coefficients
ressemblent à des paramètres de conversion anodins. Ce sont en réalité des **rendements** :
44 livres de tourteau et 11 livres d'huile par boisseau. Le CBOT les a figés une fois pour
toutes, parce qu'un contrat a besoin d'une définition stable.

Une usine, elle, n'a pas de rendement stable. Le tourteau qu'elle sort dépend de la teneur
en protéines des fèves, qui dépend de l'origine, de la saison et du lot. Deux points de
protéine déplacent le rendement de plusieurs livres au boisseau.

Donc quand un triturateur se couvre au board crush, il ne se contente pas de couvrir : il
**accepte silencieusement 44/11 comme son propre rendement**, et garde la différence en
position nue. Cette position n'a jamais été décidée par personne — elle est le résidu d'une
convention de contrat — et sa taille en dollars est fixée par le prix du tourteau, donc par
le marché.

LE LIVRABLE — L'INVERSION
--------------------------
On ne mesure pas le tracking error (il faudrait des prix cash, absents de l'export). On
demande l'inverse : **à quelle précision de rendement le board vous contraint-il ?**

    seuil_lb = (board_crush - opex) / (prix_tourteau / 2000)

C'est l'écart de rendement qui consomme toute la marge nette, en livres par boisseau —
l'unité qu'un exploitant manie tous les jours. Le résultat n'est pas un niveau moyen mais
sa **dépendance au régime** : l'exigence s'effondre quand la marge se resserre, c'est-à-dire
exactement quand la couverture devait servir.

TENSION — INFÉRÉE, PAS SOURCÉE
-------------------------------
**Il me semble que** le crush CBOT est traité comme un proxy hedgeable de l'économie
réelle d'une usine, alors que le basis tourteau intérieur, le rendement réel et la
logistique cassent le hedge exactement quand il faut. Aucune preuve documentaire qu'un
desk précis s'engueule là-dessus : « il me semble », jamais « j'ai lu que ».

LE PIÈGE D'UNITÉ, ET IL EST LE CŒUR DU SUJET
---------------------------------------------
Le board crush mélange **trois unités** dans une seule formule :
    la fève en USD/boisseau, le tourteau en USD/**short ton**, l'huile en cents/lb.
Traiter la short ton comme une tonne métrique fausse le tourteau de 10 % — soit, sur un
crush typique, la moitié du crush lui-même. Voir `core/units.board_crush_usd_bu`, qui
dérive les coefficients au lieu de les coder en dur.

IDENTITÉ
--------
    board_crush = 0,022 x meal_usd_short_ton + 0,11 x oil_c_lb - bean_usd_bu
    plant_crush = y_meal x meal_cash + y_oil x oil_cash - bean_cash - opex
    tracking_error = board_crush - plant_crush

POINT DE BASCULE
----------------
Le ratio de couverture de variance minimale `h* = cov(dplant, dboard) / var(dboard)`,
et les régimes où il s'éloigne de 1. Au-delà d'un certain décrochage, couvrir au board
**est une position en soi**, pas une couverture.

HYPOTHÈSES
----------
X-H1  Rendements CBOT implicites : 44 lb de tourteau et 11 lb d'huile par boisseau. Les
      rendements réels d'une usine en diffèrent — d'où `y_meal` et `y_oil` en sliders.
X-H2  L'opex de trituration est un forfait par boisseau, exogène. Varie fortement avec le
      prix de l'énergie ; paramétré.
X-H3  Aucun délai entre l'achat de la fève et la vente des produits : le crush est calculé
      à la même date pour les trois jambes. Une usine réelle porte un décalage de
      plusieurs semaines, qui **ajoute** du tracking error — biais dans le bon sens.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from agri.core.stats import HacRegression, hac_ols, regime_runs
from agri.core.units import board_crush_usd_bu

# X-H1 : rendements réels par défaut, légèrement sous les rendements CBOT
DEFAULT_YIELD_MEAL_LB_BU = 43.5
DEFAULT_YIELD_OIL_LB_BU = 10.8
DEFAULT_OPEX_USD_BU = 0.42

CBOT_MEAL_LB_BU = 44.0
CBOT_OIL_LB_BU = 11.0
LB_PER_SHORT_TON = 2000.0


class CrushError(ValueError):
    """Modèle mal spécifié."""


def plant_crush_usd_bu(
    bean_cash_usd_bu: pd.Series,
    meal_cash_usd_short_ton: pd.Series,
    oil_cash_cents_lb: pd.Series,
    *,
    yield_meal_lb_bu: float = DEFAULT_YIELD_MEAL_LB_BU,
    yield_oil_lb_bu: float = DEFAULT_YIELD_OIL_LB_BU,
    opex_usd_bu: float = DEFAULT_OPEX_USD_BU,
) -> pd.Series:
    """Crush d'une usine réelle, aux prix cash locaux et aux rendements réels (X-H1).

    Mêmes unités que le board — short ton pour le tourteau, cents/lb pour l'huile — pour
    que la différence entre les deux soit un vrai tracking error et pas une erreur de
    conversion déguisée.
    """
    if not 0 < yield_meal_lb_bu < 60 or not 0 < yield_oil_lb_bu < 20:
        raise CrushError(
            f"rendements hors plage physique : {yield_meal_lb_bu} lb tourteau et "
            f"{yield_oil_lb_bu} lb huile par boisseau (un boisseau de soja pèse 60 lb)"
        )
    if yield_meal_lb_bu + yield_oil_lb_bu > 60.0:
        raise CrushError(
            f"les rendements somment à {yield_meal_lb_bu + yield_oil_lb_bu:.1f} lb pour un "
            "boisseau de 60 lb — impossible avant même de compter les pertes"
        )
    meal_leg = (yield_meal_lb_bu / LB_PER_SHORT_TON) * meal_cash_usd_short_ton
    oil_leg = yield_oil_lb_bu * (oil_cash_cents_lb / 100.0)
    return meal_leg + oil_leg - bean_cash_usd_bu - opex_usd_bu


def build_tracking(
    bean_board: pd.Series,
    meal_board: pd.Series,
    oil_board: pd.Series,
    bean_cash: pd.Series,
    meal_cash: pd.Series,
    oil_cash: pd.Series,
    *,
    yield_meal_lb_bu: float = DEFAULT_YIELD_MEAL_LB_BU,
    yield_oil_lb_bu: float = DEFAULT_YIELD_OIL_LB_BU,
    opex_usd_bu: float = DEFAULT_OPEX_USD_BU,
) -> pd.DataFrame:
    """Board crush, crush usine et leur écart, sur les dates communes.

    Colonnes : board_crush, plant_crush, tracking_error, meal_basis, oil_basis, bean_basis.
    Les trois basis sont exposés parce que le tracking error vient d'eux — c'est la
    décomposition qui rend le décrochage explicable au lieu d'être constaté.
    """
    frame = pd.concat(
        {
            "bean_board": bean_board,
            "meal_board": meal_board,
            "oil_board": oil_board,
            "bean_cash": bean_cash,
            "meal_cash": meal_cash,
            "oil_cash": oil_cash,
        },
        axis=1,
    ).dropna()
    if frame.empty:
        raise CrushError("aucune date commune aux six séries")

    frame["board_crush"] = board_crush_usd_bu(
        frame["bean_board"], frame["meal_board"], frame["oil_board"]
    )
    frame["plant_crush"] = plant_crush_usd_bu(
        frame["bean_cash"],
        frame["meal_cash"],
        frame["oil_cash"],
        yield_meal_lb_bu=yield_meal_lb_bu,
        yield_oil_lb_bu=yield_oil_lb_bu,
        opex_usd_bu=opex_usd_bu,
    )
    frame["tracking_error"] = frame["board_crush"] - frame["plant_crush"]
    frame["bean_basis"] = frame["bean_cash"] - frame["bean_board"]
    frame["meal_basis"] = frame["meal_cash"] - frame["meal_board"]
    frame["oil_basis"] = frame["oil_cash"] - frame["oil_board"]
    frame.attrs["yield_meal_lb_bu"] = yield_meal_lb_bu
    frame.attrs["yield_oil_lb_bu"] = yield_oil_lb_bu
    frame.attrs["opex_usd_bu"] = opex_usd_bu
    return frame


def decompose_tracking_error(frame: pd.DataFrame) -> pd.DataFrame:
    """Décomposition **exacte** du tracking error. Pas une régression : une identité.

        tracking = (0,022 - y_meal/2000) x meal_board      <- écart de rendement tourteau
                 + (0,11  - y_oil/100)   x oil_board       <- écart de rendement huile
                 - (y_meal/2000) x meal_basis
                 - (y_oil/100)   x oil_basis
                 + bean_basis
                 + opex

    Les deux premiers termes sont ce qu'une lecture « le décrochage vient du basis » rate
    complètement : ils sont proportionnels au **niveau** du board, pas à un basis, et ils
    existent même quand tous les basis sont nuls. Sur le jeu de test, le terme de rendement
    huile a un écart-type supérieur à celui du basis fève lui-même.

    Une régression sur les trois basis les traite en variables omises et estime le
    coefficient de la fève avec un biais visible (0,988 au lieu de 1,000). L'identité
    n'a pas ce problème — quand le calcul est exact, on ne l'estime pas.
    """
    required = {"yield_meal_lb_bu", "yield_oil_lb_bu", "opex_usd_bu"}
    if not required.issubset(frame.attrs):
        raise CrushError(
            "frame sans métadonnées de rendement — utiliser build_tracking(), qui les pose"
        )
    y_meal = frame.attrs["yield_meal_lb_bu"] / LB_PER_SHORT_TON
    y_oil = frame.attrs["yield_oil_lb_bu"] / 100.0
    board_meal = CBOT_MEAL_LB_BU / LB_PER_SHORT_TON
    board_oil = CBOT_OIL_LB_BU / 100.0

    out = pd.DataFrame(index=frame.index)
    out["rendement_tourteau"] = (board_meal - y_meal) * frame["meal_board"]
    out["rendement_huile"] = (board_oil - y_oil) * frame["oil_board"]
    out["basis_tourteau"] = -y_meal * frame["meal_basis"]
    out["basis_huile"] = -y_oil * frame["oil_basis"]
    out["basis_feve"] = frame["bean_basis"]
    out["opex"] = frame.attrs["opex_usd_bu"]
    out["total"] = out.sum(axis=1)
    return out


@dataclass(frozen=True)
class OptimalHedge:
    """Ratio de couverture de variance minimale, et ce qu'il coûte de l'ignorer."""

    h_star: float
    variance_reduction_at_h_star: float
    variance_reduction_at_one: float
    n_obs: int

    @property
    def naive_hedge_adds_risk(self) -> bool:
        """Une couverture 1:1 augmente-t-elle la variance au lieu de la réduire ?"""
        return self.variance_reduction_at_one < 0.0

    @property
    def headline(self) -> str:
        if self.naive_hedge_adds_risk:
            return (
                f"Le ratio de couverture optimal board/usine tombe à {self.h_star:.2f} : "
                f"sur ces fenêtres, un hedge à 1:1 **ajoute** "
                f"{-self.variance_reduction_at_one:.0%} de variance au lieu d'en retirer."
            )
        return (
            f"Le ratio de couverture optimal est {self.h_star:.2f}. Couvrir à 1:1 retire "
            f"{self.variance_reduction_at_one:.0%} de variance contre "
            f"{self.variance_reduction_at_h_star:.0%} au ratio optimal."
        )


def optimal_hedge_ratio(frame: pd.DataFrame) -> OptimalHedge:
    """`h* = cov(dplant, dboard) / var(dboard)`, sur variations et non sur niveaux.

    Sur niveaux, deux séries non stationnaires donnent un ratio flatteur qui ne veut rien
    dire. La couverture s'exécute sur des variations : c'est là qu'il faut la mesurer.
    """
    changes = frame[["plant_crush", "board_crush"]].diff().dropna()
    if len(changes) < 10:
        raise CrushError(f"pas assez d'observations : n={len(changes)}")

    var_board = float(changes["board_crush"].var())
    if var_board == 0:
        raise CrushError("le board crush ne varie pas — ratio indéfini")
    covariance = float(changes["plant_crush"].cov(changes["board_crush"]))
    h_star = covariance / var_board

    var_plant = float(changes["plant_crush"].var())

    def _reduction(h: float) -> float:
        residual = changes["plant_crush"] - h * changes["board_crush"]
        return 1.0 - float(residual.var()) / var_plant

    return OptimalHedge(
        h_star=h_star,
        variance_reduction_at_h_star=_reduction(h_star),
        variance_reduction_at_one=_reduction(1.0),
        n_obs=len(changes),
    )


def rolling_hedge_ratio(frame: pd.DataFrame, *, window: int = 120) -> pd.DataFrame:
    """`h*` en fenêtre glissante — le graphe qui montre qu'il n'est pas constant.

    `n_eff` d'une fenêtre glissante vaut n_obs/window (Règle C) : à afficher avec la
    courbe, sinon le ratio paraît bien mieux estimé qu'il ne l'est.
    """
    changes = frame[["plant_crush", "board_crush"]].diff().dropna()
    if len(changes) < window + 2:
        raise CrushError(f"pas assez d'observations pour une fenêtre de {window}")
    covariance = changes["plant_crush"].rolling(window).cov(changes["board_crush"])
    variance = changes["board_crush"].rolling(window).var()
    out = pd.DataFrame({"h_star": covariance / variance}).dropna()
    out.attrs["window"] = window
    out.attrs["n_eff"] = len(out) / window
    return out


def decoupling_episodes(
    frame: pd.DataFrame, *, threshold_usd_bu: float = 0.35, min_obs: int = 5
) -> pd.DataFrame:
    """Épisodes où le tracking error dépasse un seuil en valeur absolue.

    « Le board et l'usine ont décroché de plus de 35 c/bu pendant six semaines en
    septembre » est une phrase datée qu'un triturateur peut confirmer ou démentir.
    """
    return regime_runs(
        frame["tracking_error"].abs() > threshold_usd_bu,
        depth=frame["tracking_error"],
        min_obs=min_obs,
    )


def explain_tracking_error(frame: pd.DataFrame) -> HacRegression:
    """Régression du tracking error sur les trois basis, erreurs HAC.

    Répond à « d'où vient le décrochage » plutôt qu'à « y a-t-il un décrochage ».

    Les coefficients sont **prévisibles** : le tracking error est une fonction linéaire
    exacte des trois basis, de coefficients `-y_meal/2000`, `-y_oil/100` et `+1`. Retrouver
    ces valeurs est un contrôle de cohérence du moteur, pas une découverte. Ce qui est
    informatif, ce sont les **contributions** — voir `basis_contributions`.
    """
    return hac_ols(
        frame["tracking_error"], frame[["bean_basis", "meal_basis", "oil_basis"]]
    )


def basis_contributions(frame: pd.DataFrame) -> pd.DataFrame:
    """Poids de chaque terme du tracking error, mesuré par son **écart-type en USD/bu**.

    POURQUOI PAS LES COEFFICIENTS NUS. Les trois basis sont dans trois unités différentes
    — tourteau en USD/short ton, huile en cents/lb, fève en USD/bu. Comparer leurs
    coefficients revient à comparer des dollars par short ton à des cents par livre : le
    classement obtenu ne veut rien dire. La décomposition exacte ramène chaque terme en
    USD/bu, et c'est la dispersion de ces termes-là qui se compare.

    Le poste `opex` est constant, donc de dispersion nulle : il déplace le niveau du
    tracking error, jamais sa variabilité. C'est visible dans la table, et c'est une
    distinction qui compte pour un hedge.
    """
    components = decompose_tracking_error(frame).drop(columns="total")
    rows = [
        {
            "terme": column,
            "mean_usd_bu": float(components[column].mean()),
            "std_usd_bu": float(components[column].std()),
        }
        for column in components.columns
    ]
    out = pd.DataFrame(rows).sort_values("std_usd_bu", ascending=False)
    total = out["std_usd_bu"].sum()
    out["share"] = out["std_usd_bu"] / total if total > 0 else np.nan
    return out.reset_index(drop=True)


# ===========================================================================
# L'INVERSION — ce que le board crush exige de vous sans le dire
# ===========================================================================
# Les coefficients du board crush ne sont pas des prix : ce sont des **rendements**. Écrire
# 0,022 x tourteau + 0,11 x huile - fève, c'est poser 44 lb de tourteau et 11 lb d'huile par
# boisseau. Une usine qui se couvre au board accepte donc silencieusement ces rendements
# comme les siens, et garde la différence en position nue. La page mesure cette position et
# l'inverse en une exigence : à quelle précision de rendement le board vous contraint-il.
def load_real_board_frame(start: str | None = "2015-01-01") -> pd.DataFrame:
    """Le board crush CBOT sur données réelles — les trois jambes de l'export.

    Colonnes : bean (USD/bu), meal (USD/short ton), oil (c/lb), board (USD/bu).
    Aucun prix cash n'entre ici : l'export n'en contient pas, et le livrable de la page est
    justement construit pour ne pas en avoir besoin.
    """
    from agri.data.bloomberg_loader import load

    frame = pd.concat(
        {"bean": load("cbot_soybean"), "meal": load("cbot_soymeal"), "oil": load("cbot_soyoil")},
        axis=1,
        sort=True,
    ).dropna()
    if start is not None:
        frame = frame[frame.index >= pd.Timestamp(start)]
    if frame.empty:
        raise CrushError(f"aucune date commune aux trois jambes du crush après {start}")
    frame["board"] = board_crush_usd_bu(frame["bean"], frame["meal"], frame["oil"])
    return frame


@dataclass(frozen=True)
class YieldExposure:
    """La position non couverte qu'un écart de rendement crée, jour par jour.

    Un écart de rendement n'est pas une erreur de niveau qu'on rattraperait par une
    constante : c'est une **position dans les produits**, dont la taille en dollars est
    fixée par le prix du tourteau et de l'huile — donc par le marché, pas par l'usine.
    """

    frame: pd.DataFrame
    meal_lb_gap: float
    oil_lb_gap: float
    opex_usd_bu: float

    @property
    def position_median(self) -> float:
        return float(self.frame["position_usd_bu"].median())

    @property
    def share_median(self) -> float:
        return float(self.frame["share_of_margin"].median())

    @property
    def share_exceeding_margin(self) -> float:
        """Part des séances où la position nue dépasse la marge nette entière."""
        return float((self.frame["position_usd_bu"].abs() > self.frame["net_margin"]).mean())

    @property
    def headline(self) -> str:
        return (
            f"Un écart de {self.meal_lb_gap:+.1f} lb de tourteau et {self.oil_lb_gap:+.1f} lb "
            f"d'huile par boisseau crée une position nue de "
            f"{self.position_median:+.3f} USD/bu en médiane, soit {self.share_median:.0%} de la "
            f"marge nette. Elle dépasse la marge entière "
            f"{self.share_exceeding_margin:.0%} des séances."
        )


def yield_exposure(
    frame: pd.DataFrame,
    *,
    meal_lb_gap: float = 1.0,
    oil_lb_gap: float = 0.0,
    opex_usd_bu: float = DEFAULT_OPEX_USD_BU,
) -> YieldExposure:
    """La position que laisse un écart de rendement, sur le board crush réel.

        position = (Δlb_tourteau / 2000) x prix_tourteau + Δlb_huile x prix_huile / 100

    Les deux termes ont la forme d'un prix multiplié par une quantité : c'est bien une
    position, pas un résidu de modèle. Rapportée à la marge nette `board - opex`, elle dit
    quelle fraction du résultat de l'usine n'est pas couverte alors qu'elle se croit couverte.
    """
    for column in ("meal", "oil", "board"):
        if column not in frame.columns:
            raise CrushError(f"colonne manquante dans le frame board : {column!r}")
    if abs(meal_lb_gap) > CBOT_MEAL_LB_BU or abs(oil_lb_gap) > CBOT_OIL_LB_BU:
        raise CrushError(
            f"écart de rendement invraisemblable ({meal_lb_gap:+.1f} lb tourteau, "
            f"{oil_lb_gap:+.1f} lb huile) : il dépasse le rendement du board lui-même "
            f"({CBOT_MEAL_LB_BU:.0f} et {CBOT_OIL_LB_BU:.0f} lb/bu)."
        )

    meal_leg = (meal_lb_gap / LB_PER_SHORT_TON) * frame["meal"]
    oil_leg = oil_lb_gap * (frame["oil"] / 100.0)
    position = meal_leg + oil_leg
    net_margin = frame["board"] - opex_usd_bu

    out = pd.DataFrame(
        {
            "meal_leg": meal_leg,
            "oil_leg": oil_leg,
            "position_usd_bu": position,
            "net_margin": net_margin,
            # Marge nette bornée par le bas : au voisinage de zéro le ratio explose et
            # rendrait la médiane illisible. Le plancher est signalé, pas caché.
            "share_of_margin": position / net_margin.clip(lower=0.05),
        }
    )
    return YieldExposure(
        frame=out,
        meal_lb_gap=float(meal_lb_gap),
        oil_lb_gap=float(oil_lb_gap),
        opex_usd_bu=float(opex_usd_bu),
    )


@dataclass(frozen=True)
class RequiredPrecision:
    """LE livrable : la précision de rendement que le board exige, en lb par boisseau.

    On inverse `yield_exposure` : au lieu de demander ce que coûte un écart donné, on
    demande quel écart consomme **toute** la marge nette. Le nombre qui sort est dans
    l'unité qu'un exploitant de trituration manie tous les jours.
    """

    frame: pd.DataFrame
    opex_usd_bu: float
    board_meal_lb: float

    @property
    def median_lb(self) -> float:
        return float(self.frame["breakeven_lb"].median())

    @property
    def tight_decile_lb(self) -> float:
        """La précision exigée dans le décile de marge le plus tendu."""
        tight = self.frame[self.frame["net_margin"] <= self.frame["net_margin"].quantile(0.10)]
        return float(tight["breakeven_lb"].median())

    @property
    def wide_decile_lb(self) -> float:
        wide = self.frame[self.frame["net_margin"] >= self.frame["net_margin"].quantile(0.90)]
        return float(wide["breakeven_lb"].median())

    @property
    def tight_decile_pct(self) -> float:
        return self.tight_decile_lb / self.board_meal_lb

    def share_below(self, lb: float) -> float:
        """Part des séances où un écart de `lb` livres suffit à effacer la marge nette."""
        return float((self.frame["breakeven_lb"] <= lb).mean())

    @property
    def headline(self) -> str:
        return (
            f"Le board crush exige en médiane une précision de {self.median_lb:.1f} lb de "
            f"tourteau par boisseau. Mais dans le décile de marge le plus tendu, l'exigence "
            f"tombe à {self.tight_decile_lb:.2f} lb — soit {self.tight_decile_pct:.1%} des "
            f"{self.board_meal_lb:.0f} lb du board. Une seule livre d'écart efface la marge "
            f"nette entière {self.share_below(1.0):.0%} des séances."
        )


def required_yield_precision(
    frame: pd.DataFrame,
    *,
    opex_usd_bu: float = DEFAULT_OPEX_USD_BU,
) -> RequiredPrecision:
    """L'écart de rendement tourteau qui annule exactement la marge nette, jour par jour.

        seuil_lb = (board - opex) / (prix_tourteau / 2000)

    Le seuil est décroissant en marge : il se resserre précisément quand la marge se
    resserre, c'est-à-dire dans le régime où la décision d'arrêt de l'usine devient vive
    (cf. T2-5). C'est là que la couverture au board cesse d'en être une.
    """
    if "board" not in frame.columns or "meal" not in frame.columns:
        raise CrushError("le frame doit contenir les colonnes 'board' et 'meal'")

    net_margin = frame["board"] - opex_usd_bu
    position_per_lb = frame["meal"] / LB_PER_SHORT_TON
    if (position_per_lb <= 0).any():
        raise CrushError("prix du tourteau nul ou négatif : le seuil n'est pas défini")

    out = pd.DataFrame(
        {
            "net_margin": net_margin,
            "position_per_lb": position_per_lb,
            "breakeven_lb": net_margin / position_per_lb,
        }
    )
    return RequiredPrecision(
        frame=out, opex_usd_bu=float(opex_usd_bu), board_meal_lb=CBOT_MEAL_LB_BU
    )


@dataclass(frozen=True)
class IdentityBias:
    """Pourquoi la page calcule la position directement au lieu de la régresser.

    Hérité de T2-1, retiré du portefeuille faute de séries cash. Le résultat s'applique ici
    tel quel : régresser une grandeur sur une autre qui partage ses composantes ne mesure
    pas une relation économique, elle mesure une identité comptable.

    HONNÊTETÉ SUR L'ORDRE DE GRANDEUR : la contamination est **petite** — environ +1 % pour
    un écart d'une livre, et proportionnelle à l'écart. Ce n'est pas un piège qui fait
    exploser les chiffres, et le présenter comme tel serait malhonnête. Ce qui compte n'est
    pas sa taille, c'est ce qu'un praticien ferait du coefficient : voir `headline`.
    """

    beta_naive: float
    beta_structural: float
    bias: float
    meal_lb_gap: float
    oil_lb_gap: float

    @property
    def headline(self) -> str:
        return (
            f"Régresser la marge de l'usine sur le board crush donne {self.beta_naive:.3f} "
            f"là où la réponse structurelle est {self.beta_structural:.3f} — un écart de "
            f"{self.bias:+.3f} seulement, mais entièrement mécanique : les deux grandeurs "
            "partagent le tourteau, l'huile et la fève. Le danger n'est pas la taille du "
            "biais, c'est ce qu'on en fait : appliquer ce coefficient revient à couvrir son "
            "écart de rendement avec **davantage de board crush**, alors que l'écart est une "
            "position en tourteau et en huile pris séparément. On ne couvre pas un écart de "
            "rendement avec l'instrument dont l'hypothèse de rendement l'a créé."
        )


def hedge_ratio_identity_bias(
    frame: pd.DataFrame,
    *,
    meal_lb_gap: float = 1.0,
    oil_lb_gap: float = 0.0,
    opex_usd_bu: float = DEFAULT_OPEX_USD_BU,
) -> IdentityBias:
    """Le piège d'identité comptable, démontré sur la donnée réelle.

    La marge d'usine s'écrit exactement `board + écart_rendement - opex`. Régresser sa
    variation sur celle du board donne donc

        beta = 1 + cov(Δécart, Δboard) / var(Δboard)

    et le second terme n'est pas nul, puisque l'écart de rendement est lui-même une
    combinaison de tourteau et d'huile — les deux jambes qui composent le board. Le
    coefficient estimé n'est pas un ratio de couverture : c'est 1 plus une contamination.

    La réponse structurelle est 1 : à rendements identiques, un boisseau couvert au board
    est couvert un pour un. Tout écart à 1 mesuré par cette régression est mécanique.

    LA CONCLUSION UTILE n'est pas « attention, votre beta est biaisé » — le biais est petit.
    C'est que la bonne couverture d'un écart de rendement n'est pas un ajustement du ratio
    de board crush, mais des jambes tourteau et huile dimensionnées séparément. Le board
    crush est un panier 44/11 figé : il ne peut pas, par construction, couvrir un écart à
    44/11.
    """
    exposure = yield_exposure(
        frame, meal_lb_gap=meal_lb_gap, oil_lb_gap=oil_lb_gap, opex_usd_bu=opex_usd_bu
    )
    plant_margin = frame["board"] + exposure.frame["position_usd_bu"] - opex_usd_bu

    changes = pd.concat(
        {"plant": plant_margin.diff(), "board": frame["board"].diff()}, axis=1, sort=True
    ).dropna()
    if len(changes) < 30:
        raise CrushError(f"échantillon trop court pour la démonstration : n={len(changes)}")

    variance = float(changes["board"].var())
    if variance <= 0:
        raise CrushError("variance nulle du board crush : la régression n'est pas définie")
    beta_naive = float(changes["plant"].cov(changes["board"]) / variance)

    return IdentityBias(
        beta_naive=beta_naive,
        beta_structural=1.0,
        bias=beta_naive - 1.0,
        meal_lb_gap=float(meal_lb_gap),
        oil_lb_gap=float(oil_lb_gap),
    )


__all__ = [
    "CBOT_MEAL_LB_BU",
    "CBOT_OIL_LB_BU",
    "CrushError",
    "IdentityBias",
    "OptimalHedge",
    "RequiredPrecision",
    "YieldExposure",
    "basis_contributions",
    "build_tracking",
    "decoupling_episodes",
    "explain_tracking_error",
    "hedge_ratio_identity_bias",
    "load_real_board_frame",
    "optimal_hedge_ratio",
    "plant_crush_usd_bu",
    "required_yield_precision",
    "rolling_hedge_ratio",
    "yield_exposure",
]
