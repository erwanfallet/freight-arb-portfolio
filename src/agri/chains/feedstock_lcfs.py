"""T3-1 — Deux subventions qui se contredisent, et l'usine prise entre les deux.

L'HISTOIRE
----------
Environ trois quarts de la capacité américaine de renewable diesel a été construite sur ou
près des côtes du Golfe et de Californie. Ce n'est pas un hasard de géographie : ces usines
ont été **sitées pour tourner sur du feedstock importé** — UCO d'Asie, suif — débarqué à
quai. Le choix d'implantation encode un pari sur l'origine de la matière première, et il a
été fait avant que la règle fiscale n'existe.

Puis deux administrations écrivent deux subventions qui ne poursuivent pas le même objectif.
Le Congrès écrit 45Z pour récompenser le feedstock **nord-américain**, et exclut du crédit
tout ce qui ne l'est pas. La Californie, elle, continue de payer la **faible intensité
carbone** par son LCFS, sans regarder l'origine — et l'UCO importé est justement moins
carboné que le soyoil domestique. Une politique pénalise ce que l'autre récompense, sur le
même gallon, dans la même usine.

D'où le désaccord :

    Camp A — les usines côtières : la prime LCFS suffit, les imports tiennent.
    Camp B — le complexe soja     : elle ne suffit pas, le soyoil rafle la part.

CE QUE LA PAGE MONTRE — ET POURQUOI LES DEUX CAMPS SE TROMPENT DE VARIABLE
--------------------------------------------------------------------------
Les deux camps argumentent sur le **prix du crédit LCFS**. Or ce prix ne peut pas trancher :
sur toute la plage que le programme a réalisée depuis sa création, il ne déplace la réponse
que de quelques cents par livre. Ce qui tranche, c'est le **spread de prix entre l'UCO et le
soyoil** — donc un prix de matière première fixé par la collecte en Asie et par le fret, pas
une décision réglementaire à Sacramento.

Le livrable n'est donc pas un pronostic mais une **décote** : combien de cents sous le
soyoil l'UCO importé doit-il se vendre, uniquement pour compenser un crédit d'impôt qu'il
n'a pas le droit de réclamer. Un acheteur de feedstock confirme ou dément ce nombre en dix
secondes sur son propre book.

POSTURE — À TENIR ABSOLUMENT
----------------------------
On ne prend pas parti sur le côté où ça tombe. On quantifie la bascule ; l'insider a la
donnée interne. Un candidat qui prédit se fait corriger ; un candidat qui donne un seuil
falsifiable se fait répondre.

ÉTAT DES LIEUX (sourcé)
------------------------
- RVO 2026-27 finalisés par l'EPA en mars 2026 : hausse d'environ 60 % de la production
  biodiesel + renewable diesel contre 2025, plus haut niveau du programme
  (Fastmarkets, 16 juin 2026).
- ILUC retiré : le soyoil US porte un CI de 27 et un crédit 45Z de ~49 c/gal, quasi au
  niveau du suif et de l'UCO. Usage soyoil projeté de 14,55 à ~17,8 Md lb (WASDE juin 2026).
- farmdoc daily (25 juin 2026) : incertitude à trois étages — production domestique vs
  carburant importé, feedstock domestique vs importé sous la pénalité 45Z, distribution du
  mix dans chaque canal. Le feedstock importé « pourrait se retirer brutalement, ou se
  révéler plus résilient que prévu si la prime LCFS reste assez large ». Environ trois
  quarts de la capacité RD domestique est sur ou près des côtes du Golfe et de Californie,
  construite pour tourner sur du feedstock importé.

LE PIÈGE D'UNITÉ
----------------
Trois unités s'empilent dans une seule marge, et deux d'entre elles ne sont pas des unités
de prix :
    le carburant se vend au **gallon** ;
    le feedstock s'achète à la **livre** (ou à la tonne), via un rendement ~7,6 lb/gal ;
    le crédit LCFS se cote en **USD par tonne de CO2e**, et n'entre dans la marge qu'après
    passage par une intensité carbone en gCO2e/MJ et un contenu énergétique de
    134,47 MJ/gal.
Le facteur 134,47e-6 n'est pas un détail de conversion : c'est lui qui fixe la pente du
seuil LCFS, donc la réponse.

MODÈLE
------
    gate_value(f) = P_ulsd + RIN_D4 x rin_per_gal
                  + LCFS x (CI_std - CI_f) x EER x 134,47e-6
                  + credit_45Z(f)

    credit_45Z(f) = 1,00 x max(0, (50 - CI_f)/50)   si f est éligible Amérique du Nord
                  = 0                                sinon

    feedstock_breakeven(f) = (gate_value(f) - opex - roi) / yield_lb_gal

HYPOTHÈSES
----------
L-H1  `yield_lb_gal` = 7,6 lb de feedstock par gallon de RD. Varie avec la filière et
      l'usine ; balayé en sensibilité.
L-H2  EER = 1,0 pour le renewable diesel. Le LCFS applique des EER > 1 à l'électricité et
      à l'hydrogène, pas au RD.
L-H3  `CI_std` est le barème CARB de l'année, qui **décroît chaque année par
      construction**. Le figer fausserait tout l'historique — c'est un paramètre daté.
L-H4  Le crédit 45Z suit la formule linéaire (50 - CI)/50 plafonnée à 1 $/gal. Contrôle de
      calibration : CI = 27 donne 0,46 $/gal contre ~0,49 publié. L'écart tient à la
      définition exacte du CI retenue et n'est pas résorbé ici : il est affiché.
L-H5  Le crédit 45Z n'est **pas une constante** — il dépend d'un CI qui dépend d'une
      méthodologie en cours de finalisation. Traité comme un slider avec une plage,
      jamais comme un nombre.
L-H6  Les exemptions de petites raffineries (SRE) réallouent rétroactivement les volumes.
      Toute conclusion sur les RVO porte un avertissement SRE.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from agri.data.snapshot import cached

from agri.core.breakeven import Breakeven, NoBreakevenInRange, solve_breakeven

# --- constantes physiques et réglementaires ------------------------------
MJ_PER_GALLON_ULSD = 134.47          # contenu énergétique du diesel de référence
GRAMS_PER_TONNE = 1_000_000.0
LCFS_CONVERSION = MJ_PER_GALLON_ULSD / GRAMS_PER_TONNE   # 134,47e-6

CREDIT_45Z_MAX_USD_GAL = 1.00        # plafond de la formule
CREDIT_45Z_CI_REFERENCE = 50.0       # le CI à partir duquel le crédit s'annule

OIL_LB_PER_BUSHEL = 11.0             # livres d'huile par boisseau de soja trituré

# --- valeurs par défaut (L-H1 à L-H4) ------------------------------------
DEFAULT_YIELD_LB_GAL = 7.6
DEFAULT_EER = 1.0
DEFAULT_RIN_PER_GAL_RD = 1.7         # le biodiesel génère 1,5
DEFAULT_CI_STD = 95.0                # barème CARB, à dater (L-H3)
DEFAULT_CI_SOY = 27.0
DEFAULT_CI_UCO = 15.0
DEFAULT_OPEX_USD_GAL = 0.55
DEFAULT_ROI_USD_GAL = 0.25


class FeedstockError(ValueError):
    """Modèle mal spécifié."""


@dataclass(frozen=True)
class Feedstock:
    """Une filière : son intensité carbone et son éligibilité au 45Z."""

    name: str
    carbon_intensity: float          # gCO2e/MJ
    north_american: bool             # éligible 45Z

    def credit_45z_usd_gal(self) -> float:
        """Crédit 45Z (L-H4). Zéro si non nord-américain, quelle que soit l'intensité."""
        if not self.north_american:
            return 0.0
        return CREDIT_45Z_MAX_USD_GAL * max(
            0.0, (CREDIT_45Z_CI_REFERENCE - self.carbon_intensity) / CREDIT_45Z_CI_REFERENCE
        )


SOYOIL_DOMESTIC = Feedstock("soyoil domestique", DEFAULT_CI_SOY, north_american=True)
UCO_IMPORTED = Feedstock("UCO importé", DEFAULT_CI_UCO, north_american=False)
TALLOW_DOMESTIC = Feedstock("suif domestique", 18.0, north_american=True)
DCO_DOMESTIC = Feedstock("DCO domestique", 20.0, north_american=True)


def calibration_gap_45z() -> dict[str, float]:
    """Contrôle de calibration L-H4, affiché dans le panneau de diagnostics.

    La formule donne 0,46 $/gal sur un CI de 27 ; la valeur publiée tourne autour de
    0,49 $/gal. L'écart de 3 c/gal n'est pas résorbé — il est montré, parce que le
    résorber par un facteur d'ajustement inventé transformerait le résultat en artefact.
    """
    modelled = SOYOIL_DOMESTIC.credit_45z_usd_gal()
    published = 0.49
    return {
        "modelled_usd_gal": modelled,
        "published_usd_gal": published,
        "gap_usd_gal": published - modelled,
        "gap_pct": (published - modelled) / published,
    }


# ===========================================================================
# Valeur en sortie d'usine
# ===========================================================================
def lcfs_value_usd_gal(
    lcfs_usd_t: float | pd.Series,
    carbon_intensity: float,
    *,
    ci_std: float = DEFAULT_CI_STD,
    eer: float = DEFAULT_EER,
) -> float | pd.Series:
    """La jambe LCFS, en USD/gallon.

    C'est ici que le piège d'unité se referme : un crédit coté à la tonne de CO2e devient
    des cents par gallon via une intensité carbone et un contenu énergétique.
    """
    return lcfs_usd_t * (ci_std - carbon_intensity) * eer * LCFS_CONVERSION


@dataclass(frozen=True)
class GateValue:
    """Valeur en sortie d'usine, décomposée. Chaque champ est une barre du graphe S1."""

    feedstock: str
    diesel: float
    rin: float
    lcfs: float
    credit_45z: float

    @property
    def total_usd_gal(self) -> float:
        return self.diesel + self.rin + self.lcfs + self.credit_45z

    @property
    def stack(self) -> dict[str, float]:
        return {
            "diesel": self.diesel,
            "RIN D4": self.rin,
            "LCFS": self.lcfs,
            "45Z": self.credit_45z,
        }


def gate_value(
    feedstock: Feedstock,
    *,
    ulsd_usd_gal: float,
    rin_d4_usd: float,
    lcfs_usd_t: float,
    ci_std: float = DEFAULT_CI_STD,
    eer: float = DEFAULT_EER,
    rin_per_gal: float = DEFAULT_RIN_PER_GAL_RD,
) -> GateValue:
    """Valeur d'un gallon de renewable diesel en sortie d'usine, par filière."""
    return GateValue(
        feedstock=feedstock.name,
        diesel=ulsd_usd_gal,
        rin=rin_d4_usd * rin_per_gal,
        lcfs=float(
            lcfs_value_usd_gal(
                lcfs_usd_t, feedstock.carbon_intensity, ci_std=ci_std, eer=eer
            )
        ),
        credit_45z=feedstock.credit_45z_usd_gal(),
    )


def feedstock_breakeven_usd_lb(
    feedstock: Feedstock,
    *,
    ulsd_usd_gal: float,
    rin_d4_usd: float,
    lcfs_usd_t: float,
    opex_usd_gal: float = DEFAULT_OPEX_USD_GAL,
    roi_usd_gal: float = DEFAULT_ROI_USD_GAL,
    yield_lb_gal: float = DEFAULT_YIELD_LB_GAL,
    ci_std: float = DEFAULT_CI_STD,
    eer: float = DEFAULT_EER,
    rin_per_gal: float = DEFAULT_RIN_PER_GAL_RD,
) -> float:
    """Prix de revient maximal du feedstock, en USD/lb — ce qu'une usine peut payer."""
    if yield_lb_gal <= 0:
        raise FeedstockError(f"yield_lb_gal doit être > 0, reçu {yield_lb_gal}")
    value = gate_value(
        feedstock,
        ulsd_usd_gal=ulsd_usd_gal,
        rin_d4_usd=rin_d4_usd,
        lcfs_usd_t=lcfs_usd_t,
        ci_std=ci_std,
        eer=eer,
        rin_per_gal=rin_per_gal,
    )
    return (value.total_usd_gal - opex_usd_gal - roi_usd_gal) / yield_lb_gal


# ===========================================================================
# LE POINT DE BASCULE — le livrable
# ===========================================================================
@dataclass(frozen=True)
class LcfsThreshold:
    """Le seuil LCFS, sa distance au marché, et la phrase du mail."""

    lcfs_star_usd_t: float
    lcfs_current_usd_t: float
    distance_sigmas: float | None
    ci_gap: float
    price_gap_usd_lb: float
    imports_win_above: bool

    @property
    def headline(self) -> str:
        side = "reste devant" if self.imports_win_above else "repasse derrière"
        distance = (
            f", soit {self.distance_sigmas:+.2f} écart-type de l'historique"
            if self.distance_sigmas is not None
            else ""
        )
        return (
            f"Au-delà de {self.lcfs_star_usd_t:.0f} $/t CO2e sur le crédit LCFS, l'UCO "
            f"importé sans 45Z {side} le soyoil domestique avec 45Z. Le LCFS cote "
            f"aujourd'hui {self.lcfs_current_usd_t:.0f} $/t{distance} : la thèse « le soja "
            f"rafle la part » tient à {abs(self.lcfs_star_usd_t - self.lcfs_current_usd_t):.0f} $ près."
        )


def lcfs_breakeven(
    *,
    domestic: Feedstock = SOYOIL_DOMESTIC,
    imported: Feedstock = UCO_IMPORTED,
    price_domestic_usd_lb: float,
    price_imported_usd_lb: float,
    lcfs_current_usd_t: float,
    yield_lb_gal: float = DEFAULT_YIELD_LB_GAL,
    eer: float = DEFAULT_EER,
    lcfs_history: pd.Series | None = None,
) -> LcfsThreshold:
    """Résout `LCFS*` en forme fermée.

        LCFS* = [ credit_45Z(dom) + (P_imp - P_dom) x yield ]
                / [ (CI_dom - CI_imp) x EER x 134,47e-6 ]

    Dérivation : on égalise l'avantage net des deux filières,
    `breakeven(f) - P_f`. Les termes diesel, RIN, opex et ROI sont **identiques** pour les
    deux et disparaissent — c'est pourquoi le seuil ne dépend ni du prix du diesel, ni du
    RIN, ni de la structure de coûts de l'usine. C'est ce qui le rend robuste, et c'est
    l'argument à faire passer : le seuil ne bouge que par le différentiel de CI et le
    différentiel de prix feedstock.

    REPLI DU GATE : si les prix UCO sont inaccessibles, poser
    `price_imported_usd_lb = price_domestic_usd_lb` donne le seuil à parité de prix,
    c'est-à-dire l'écart de valeur implicite que l'UCO devrait porter. Même livrable,
    sans jamais avoir besoin du prix.
    """
    ci_gap = domestic.carbon_intensity - imported.carbon_intensity
    if ci_gap <= 0:
        raise FeedstockError(
            f"le différentiel de CI doit être > 0 pour que le LCFS puisse compenser le 45Z "
            f"(CI_dom = {domestic.carbon_intensity}, CI_imp = {imported.carbon_intensity}). "
            "Si l'importé est plus carboné que le domestique, il n'y a pas de seuil : "
            "il perd sur les deux tableaux."
        )

    price_gap = price_imported_usd_lb - price_domestic_usd_lb
    numerator = domestic.credit_45z_usd_gal() + price_gap * yield_lb_gal
    denominator = ci_gap * eer * LCFS_CONVERSION
    lcfs_star = numerator / denominator

    sigma = None
    if lcfs_history is not None:
        history = pd.Series(lcfs_history).dropna().astype(float)
        if len(history) >= 2:
            std = float(history.std())
            sigma = (lcfs_star - lcfs_current_usd_t) / std if std > 0 else float("inf")

    return LcfsThreshold(
        lcfs_star_usd_t=lcfs_star,
        lcfs_current_usd_t=lcfs_current_usd_t,
        distance_sigmas=sigma,
        ci_gap=ci_gap,
        price_gap_usd_lb=price_gap,
        imports_win_above=True,
    )


def lcfs_breakeven_numeric(
    *,
    domestic: Feedstock = SOYOIL_DOMESTIC,
    imported: Feedstock = UCO_IMPORTED,
    price_domestic_usd_lb: float,
    price_imported_usd_lb: float,
    ulsd_usd_gal: float,
    rin_d4_usd: float,
    lcfs_current_usd_t: float,
    lo: float = 0.0,
    hi: float = 800.0,
    **kwargs,
) -> Breakeven:
    """Le même seuil, résolu numériquement — sert de contrôle croisé de la forme fermée.

    Si les deux divergent, la forme fermée a une erreur d'algèbre. C'est un test, pas une
    alternative : la forme fermée est celle qu'on affiche, parce qu'elle montre *de quoi*
    le seuil dépend.
    """

    def advantage(lcfs: float) -> float:
        imported_edge = (
            feedstock_breakeven_usd_lb(
                imported,
                ulsd_usd_gal=ulsd_usd_gal,
                rin_d4_usd=rin_d4_usd,
                lcfs_usd_t=lcfs,
                **kwargs,
            )
            - price_imported_usd_lb
        )
        domestic_edge = (
            feedstock_breakeven_usd_lb(
                domestic,
                ulsd_usd_gal=ulsd_usd_gal,
                rin_d4_usd=rin_d4_usd,
                lcfs_usd_t=lcfs,
                **kwargs,
            )
            - price_domestic_usd_lb
        )
        return imported_edge - domestic_edge

    return solve_breakeven(
        advantage,
        lo,
        hi,
        theta_current=lcfs_current_usd_t,
        theta_label="LCFS",
        margin_label="avantage UCO",
    )


# ===========================================================================
# L'INVERSION — le livrable réel de la page
# ===========================================================================
# Le seuil LCFS* ci-dessus est algébriquement correct mais illisible pour un desk : personne
# n'a d'intuition sur « 285 $/t CO2e ». Un acheteur de feedstock, lui, cote toute la journée
# des DÉCOTES en cents par livre. On inverse donc la question : au lieu de demander quel
# prix du LCFS sauverait l'importé, on demande **quelle décote l'importé doit tenir** face
# au domestique, à un prix du LCFS donné. Même algèbre, mais le nombre qui sort est celui
# que l'interlocuteur peut confirmer ou démentir en dix secondes sur son propre book.
CENTS_PER_USD = 100.0

# Bornes réalisées du programme LCFS californien, en $/t CO2e. Ce ne sont PAS des données
# de l'export : le prix du crédit LCFS est publié gratuitement par CARB (rapport mensuel de
# transferts de crédits) et n'a pas été téléchargé. Les deux bornes servent uniquement à
# encadrer un résultat qui est, lui, calculé comme une fonction du prix LCFS — un lecteur
# qui dispose de la série substitue ses propres valeurs sans rien changer au raisonnement.
LCFS_PROGRAM_LOW_USD_T = 50.0    # creux 2023-2024
LCFS_PROGRAM_HIGH_USD_T = 200.0  # plus haut historique, 2019-2020


@dataclass(frozen=True)
class ImportPenalty:
    """Ce que l'exclusion 45Z coûte à un importateur, exprimé en prix de feedstock.

    `discount_required_usd_lb` est le livrable : la décote que l'UCO importé doit tenir sur
    le soyoil domestique pour qu'une usine soit indifférente entre les deux. Positif = le
    domestique garde l'avantage et l'importé doit payer la différence en décote.
    """

    lcfs_usd_t: float
    credit_45z_usd_gal: float
    lcfs_offset_usd_gal: float
    residual_usd_gal: float
    discount_required_usd_lb: float

    @property
    def discount_required_c_lb(self) -> float:
        return self.discount_required_usd_lb * CENTS_PER_USD

    @property
    def imports_win_outright(self) -> bool:
        """Au-delà du LCFS neutre, l'avantage CI dépasse le 45Z : l'importé peut même se
        payer une PRIME et rester devant."""
        return self.residual_usd_gal <= 0

    @property
    def headline(self) -> str:
        if self.imports_win_outright:
            return (
                f"À {self.lcfs_usd_t:.0f} $/t CO2e, l'avantage carbone de l'UCO "
                f"({self.lcfs_offset_usd_gal:.2f} $/gal) dépasse le 45Z que touche le soyoil "
                f"({self.credit_45z_usd_gal:.2f} $/gal) : l'importé tient même à parité de "
                f"prix, et jusqu'à {-self.discount_required_c_lb:.2f} c/lb de prime."
            )
        return (
            f"À {self.lcfs_usd_t:.0f} $/t CO2e, l'UCO importé doit se vendre "
            f"{self.discount_required_c_lb:.2f} c/lb sous le soyoil domestique pour qu'une "
            f"usine soit indifférente — uniquement pour compenser un crédit d'impôt qu'il "
            f"n'a pas le droit de réclamer."
        )


def import_penalty(
    lcfs_usd_t: float,
    *,
    domestic: Feedstock = SOYOIL_DOMESTIC,
    imported: Feedstock = UCO_IMPORTED,
    yield_lb_gal: float = DEFAULT_YIELD_LB_GAL,
    eer: float = DEFAULT_EER,
) -> ImportPenalty:
    """La décote que l'importé doit tenir, à un prix du LCFS donné.

        décote = [ crédit_45Z(dom) - LCFS x (CI_dom - CI_imp) x EER x 134,47e-6 ] / rendement

    Les termes diesel, RIN, opex et ROI ont disparu de cette expression : les deux filières
    produisent le MÊME gallon de renewable diesel dans la MÊME usine, donc ils s'annulent au
    lieu de s'ajouter. C'est ce qui rend le nombre difficile à contester — on ne peut pas le
    récuser en contestant une prévision de diesel ou une hypothèse d'opex, puisqu'aucune des
    deux n'y figure.
    """
    if yield_lb_gal <= 0:
        raise FeedstockError(f"yield_lb_gal doit être > 0, reçu {yield_lb_gal}")
    ci_gap = domestic.carbon_intensity - imported.carbon_intensity
    if ci_gap <= 0:
        raise FeedstockError(
            f"le différentiel de CI doit être > 0 (CI_dom = {domestic.carbon_intensity}, "
            f"CI_imp = {imported.carbon_intensity}) : si l'importé est plus carboné, il perd "
            "sur les deux tableaux et la notion de décote compensatrice n'a pas de sens."
        )

    credit = domestic.credit_45z_usd_gal()
    offset = float(lcfs_usd_t * ci_gap * eer * LCFS_CONVERSION)
    residual = credit - offset
    return ImportPenalty(
        lcfs_usd_t=float(lcfs_usd_t),
        credit_45z_usd_gal=credit,
        lcfs_offset_usd_gal=offset,
        residual_usd_gal=residual,
        discount_required_usd_lb=residual / yield_lb_gal,
    )


def lcfs_neutral_price(
    *,
    domestic: Feedstock = SOYOIL_DOMESTIC,
    imported: Feedstock = UCO_IMPORTED,
    eer: float = DEFAULT_EER,
) -> float:
    """Le prix du LCFS qui annule exactement le 45Z, à parité de prix feedstock.

    C'est `lcfs_breakeven` avec `price_gap = 0`, isolé parce qu'il sert de repère absolu :
    au-dessus, la Californie sur-compense ce que le Congrès a retiré ; en dessous, elle ne
    fait que l'amortir.
    """
    ci_gap = domestic.carbon_intensity - imported.carbon_intensity
    if ci_gap <= 0:
        raise FeedstockError("le différentiel de CI doit être > 0")
    return domestic.credit_45z_usd_gal() / (ci_gap * eer * LCFS_CONVERSION)


@dataclass(frozen=True)
class PenaltyBounds:
    """Le résultat central : la décote requise est BORNÉE sur toute la plage du LCFS.

    Le débat public pose la question comme « la prime LCFS suffit-elle à sauver les
    imports ? », donc comme une question sur le prix du crédit. Si l'amplitude totale que le
    LCFS peut imprimer sur la réponse est étroite devant les mouvements du prix du
    feedstock, la question est mal posée à la source.
    """

    lcfs_low_usd_t: float
    lcfs_high_usd_t: float
    discount_at_low_usd_lb: float
    discount_at_high_usd_lb: float
    lcfs_neutral_usd_t: float

    @property
    def span_usd_lb(self) -> float:
        """L'amplitude totale que le LCFS peut imprimer sur la décote requise."""
        return self.discount_at_low_usd_lb - self.discount_at_high_usd_lb

    @property
    def span_c_lb(self) -> float:
        return self.span_usd_lb * CENTS_PER_USD

    @property
    def reaches_neutral(self) -> bool:
        """Le LCFS a-t-il jamais coté assez haut pour annuler le 45Z à parité de prix ?"""
        return self.lcfs_high_usd_t >= self.lcfs_neutral_usd_t

    @property
    def headline(self) -> str:
        verdict = (
            "il y est parvenu"
            if self.reaches_neutral
            else (
                f"il n'y est jamais parvenu — son plus haut historique "
                f"({self.lcfs_high_usd_t:.0f} $/t) reste "
                f"{self.lcfs_neutral_usd_t - self.lcfs_high_usd_t:.0f} $/t sous le compte"
            )
        )
        return (
            f"Annuler le 45Z à parité de prix demanderait {self.lcfs_neutral_usd_t:.0f} $/t "
            f"CO2e sur le crédit LCFS ; sur toute l'histoire du programme, {verdict}. Entre "
            f"son creux et son plus haut, le LCFS ne déplace la décote requise que de "
            f"{self.span_c_lb:.2f} c/lb — de "
            f"{self.discount_at_low_usd_lb * CENTS_PER_USD:.2f} à "
            f"{self.discount_at_high_usd_lb * CENTS_PER_USD:.2f} c/lb."
        )


def penalty_bounds(
    *,
    lcfs_low_usd_t: float = LCFS_PROGRAM_LOW_USD_T,
    lcfs_high_usd_t: float = LCFS_PROGRAM_HIGH_USD_T,
    **kwargs,
) -> PenaltyBounds:
    """Encadre la décote requise sur la plage réalisée du prix du crédit LCFS."""
    if lcfs_high_usd_t <= lcfs_low_usd_t:
        raise FeedstockError(
            f"la borne haute doit dépasser la borne basse : {lcfs_high_usd_t} <= {lcfs_low_usd_t}"
        )
    low = import_penalty(lcfs_low_usd_t, **kwargs)
    high = import_penalty(lcfs_high_usd_t, **kwargs)
    return PenaltyBounds(
        lcfs_low_usd_t=float(lcfs_low_usd_t),
        lcfs_high_usd_t=float(lcfs_high_usd_t),
        discount_at_low_usd_lb=low.discount_required_usd_lb,
        discount_at_high_usd_lb=high.discount_required_usd_lb,
        lcfs_neutral_usd_t=lcfs_neutral_price(
            domestic=kwargs.get("domestic", SOYOIL_DOMESTIC),
            imported=kwargs.get("imported", UCO_IMPORTED),
            eer=kwargs.get("eer", DEFAULT_EER),
        ),
    )


# ===========================================================================
# La décote requise confrontée au prix réel du soyoil
# ===========================================================================
@dataclass(frozen=True)
class DiscountBurden:
    """La même décote en cents, rapportée au prix du feedstock qu'elle grève.

    Une décote de 4,5 c/lb ne veut pas dire la même chose selon que le soyoil cote 90 c/lb
    ou 25 : dans un cas c'est 5 % du prix, dans l'autre 18 %. Le 45Z est écrit en dollars
    par gallon, donc son poids relatif est **contracyclique** au prix de l'huile végétale —
    il mord le plus fort quand l'huile est bon marché, c'est-à-dire quand les marges de
    trituration sont déjà minces.
    """

    frame: pd.DataFrame
    lcfs_usd_t: float
    discount_required_usd_lb: float

    @property
    def burden_min(self) -> float:
        return float(self.frame["burden_share"].min())

    @property
    def burden_max(self) -> float:
        return float(self.frame["burden_share"].max())

    @property
    def burden_last(self) -> float:
        return float(self.frame["burden_share"].iloc[-1])

    @property
    def headline(self) -> str:
        return (
            f"La même décote de {self.discount_required_usd_lb * CENTS_PER_USD:.2f} c/lb pèse "
            f"{self.burden_min:.1%} du prix du soyoil à son plus haut et {self.burden_max:.1%} "
            f"à son plus bas sur la période — un facteur "
            f"{self.burden_max / self.burden_min:.1f}. Elle en pèse {self.burden_last:.1%} "
            f"au dernier cours."
        )


def discount_burden(
    domestic_price_usd_lb: pd.Series,
    *,
    lcfs_usd_t: float,
    **kwargs,
) -> DiscountBurden:
    """Le poids relatif de la décote requise, sur une série réelle de prix du domestique.

    ATTENTION UNITÉ : le soyoil CBOT cote en **cents par livre** ; cette fonction attend des
    **USD par livre**. Le facteur 100 entre les deux est exactement du même ordre que le
    résultat cherché, donc l'oublier ne produit pas une erreur visible — il produit un
    chiffre plausible et faux.
    """
    prices = pd.Series(domestic_price_usd_lb).dropna().astype(float)
    if prices.empty:
        raise FeedstockError("série de prix domestique vide")
    if (prices <= 0).any():
        raise FeedstockError("prix domestique nul ou négatif dans la série")
    if prices.median() > 5.0:
        raise FeedstockError(
            f"prix médian de {prices.median():.1f} : la série semble être en cents par livre "
            "alors que des USD par livre sont attendus (diviser par 100)."
        )

    penalty = import_penalty(lcfs_usd_t, **kwargs)
    frame = pd.DataFrame(
        {
            "domestic_price_usd_lb": prices,
            "discount_required_usd_lb": penalty.discount_required_usd_lb,
            "burden_share": penalty.discount_required_usd_lb / prices,
        }
    )
    return DiscountBurden(
        frame=frame,
        lcfs_usd_t=float(lcfs_usd_t),
        discount_required_usd_lb=penalty.discount_required_usd_lb,
    )


@dataclass(frozen=True)
class StructuralExit:
    """Le prix du soyoil sous lequel l'import devient impossible, pas seulement désavantagé.

    L'UCO a un coût de collecte et un fret : il existe un prix plancher sous lequel il n'y a
    tout simplement pas d'offre à l'export. Si la décote exigée par le 45Z pousse l'UCO sous
    ce plancher, la filière ne se contracte pas — elle s'arrête, quel que soit le LCFS.
    """

    soyoil_critical_usd_lb: float
    uco_floor_usd_lb: float
    discount_required_usd_lb: float
    lcfs_usd_t: float
    share_below: float | None = None
    n_obs: int | None = None

    @property
    def headline(self) -> str:
        base = (
            f"Avec un plancher de collecte UCO à {self.uco_floor_usd_lb * CENTS_PER_USD:.0f} c/lb "
            f"rendu USGC, l'import cesse d'être finançable dès que le soyoil passe sous "
            f"{self.soyoil_critical_usd_lb * CENTS_PER_USD:.1f} c/lb."
        )
        if self.share_below is None:
            return base
        return (
            f"{base} Le soyoil a coté sous ce seuil {self.share_below:.0%} du temps sur "
            f"l'échantillon ({self.n_obs:,} séances)."
        )


def structural_exit(
    domestic_price_usd_lb: pd.Series | None = None,
    *,
    uco_floor_usd_lb: float,
    lcfs_usd_t: float,
    **kwargs,
) -> StructuralExit:
    """Le prix critique du soyoil, et la fréquence à laquelle le marché l'a franchi.

        soyoil* = plancher_UCO + décote_requise(LCFS)

    Le plancher de collecte est le seul paramètre que la page ne peut pas observer : c'est
    délibérément celui qu'on demande à l'interlocuteur. Le reste du calcul est fermé.
    """
    if uco_floor_usd_lb <= 0:
        raise FeedstockError("le plancher de collecte UCO doit être > 0")

    penalty = import_penalty(lcfs_usd_t, **kwargs)
    critical = uco_floor_usd_lb + penalty.discount_required_usd_lb

    share = n_obs = None
    if domestic_price_usd_lb is not None:
        prices = pd.Series(domestic_price_usd_lb).dropna().astype(float)
        if not prices.empty:
            share = float((prices < critical).mean())
            n_obs = int(len(prices))

    return StructuralExit(
        soyoil_critical_usd_lb=critical,
        uco_floor_usd_lb=float(uco_floor_usd_lb),
        discount_required_usd_lb=penalty.discount_required_usd_lb,
        lcfs_usd_t=float(lcfs_usd_t),
        share_below=share,
        n_obs=n_obs,
    )


@cached('t3_1_soyoil', from_frame=lambda f: f.iloc[:, 0].rename("soyoil_usd_lb"))
def load_soyoil_usd_lb(start: str | None = None) -> pd.Series:
    """Le soyoil CBOT de l'export réel, converti de cents par livre en USD par livre.

    La conversion est isolée dans une fonction plutôt que répétée sur les pages : c'est le
    piège d'unité de ce module, et un facteur 100 dispersé dans le code finit toujours par
    être appliqué deux fois quelque part, ou zéro.
    """
    from agri.data.bloomberg_loader import load

    series = load("cbot_soyoil") / CENTS_PER_USD
    if start is not None:
        series = series[series.index >= pd.Timestamp(start)]
    return series.rename("soyoil_usd_lb")


def winner_grid(
    *,
    ci_imported_values: np.ndarray | None = None,
    lcfs_values: np.ndarray | None = None,
    domestic: Feedstock = SOYOIL_DOMESTIC,
    price_domestic_usd_lb: float = 0.52,
    price_imported_usd_lb: float = 0.46,
    yield_lb_gal: float = DEFAULT_YIELD_LB_GAL,
    eer: float = DEFAULT_EER,
) -> pd.DataFrame:
    """S4 — heatmap `CI de l'importé × prix LCFS -> filière gagnante`.

    Le graphe qui explique le désaccord sans une ligne de texte : deux zones séparées par
    une frontière, et le point de marché courant posé dessus.
    """
    ci_imported_values = (
        np.arange(10.0, 31.0, 1.0) if ci_imported_values is None else np.asarray(ci_imported_values)
    )
    lcfs_values = (
        np.arange(0.0, 401.0, 10.0) if lcfs_values is None else np.asarray(lcfs_values)
    )

    rows = []
    for ci in ci_imported_values:
        imported = Feedstock("UCO importé", float(ci), north_american=False)
        for lcfs in lcfs_values:
            imported_edge = lcfs_value_usd_gal(lcfs, ci, eer=eer) / yield_lb_gal - price_imported_usd_lb
            domestic_edge = (
                lcfs_value_usd_gal(lcfs, domestic.carbon_intensity, eer=eer)
                + domestic.credit_45z_usd_gal()
            ) / yield_lb_gal - price_domestic_usd_lb
            rows.append(
                {
                    "ci_imported": float(ci),
                    "lcfs_usd_t": float(lcfs),
                    "advantage_usd_lb": imported_edge - domestic_edge,
                    "winner": "UCO importé" if imported_edge > domestic_edge else "soyoil domestique",
                }
            )
    return pd.DataFrame(rows)


# ===========================================================================
# Second module — le bilan de trituration
# ===========================================================================
@dataclass(frozen=True)
class CrushBalance:
    """La traduction chiffrée de « la trituration peut-elle suivre ? »."""

    soyoil_required_lb: float
    crush_required_bu: float
    crush_required_bu_day: float
    installed_capacity_bu_day: float
    gap_bu_day: float

    @property
    def is_short(self) -> bool:
        return self.gap_bu_day > 0

    @property
    def headline(self) -> str:
        if self.is_short:
            return (
                f"Livrer cette part du mandat demande {self.crush_required_bu_day:,.0f} bu/jour "
                f"de trituration contre {self.installed_capacity_bu_day:,.0f} annoncés : "
                f"il manque {self.gap_bu_day:,.0f} bu/jour, soit "
                f"{self.gap_bu_day / self.installed_capacity_bu_day:.0%} de la capacité."
            )
        return (
            f"La capacité annoncée ({self.installed_capacity_bu_day:,.0f} bu/jour) couvre le "
            f"besoin ({self.crush_required_bu_day:,.0f} bu/jour) avec "
            f"{-self.gap_bu_day:,.0f} bu/jour de marge."
        )


def crush_from_soyoil_lb(
    soyoil_lb: float,
    *,
    installed_capacity_bu_day: float,
    oil_lb_per_bushel: float = OIL_LB_PER_BUSHEL,
) -> CrushBalance:
    """Trituration requise pour un volume d'huile donné, sans passer par un mandat.

    Variante de `crush_balance` quand la contrainte connue est un **volume d'huile** — par
    exemple l'incrément de consommation projeté par le WASDE — plutôt qu'un mandat en
    gallons dont il faudrait supposer la part soyoil.
    """
    if soyoil_lb < 0:
        raise FeedstockError("le volume d'huile doit être >= 0")
    if installed_capacity_bu_day <= 0:
        raise FeedstockError("la capacité installée doit être > 0")

    crush_bu = soyoil_lb / oil_lb_per_bushel
    crush_bu_day = crush_bu / 365.0
    return CrushBalance(
        soyoil_required_lb=float(soyoil_lb),
        crush_required_bu=crush_bu,
        crush_required_bu_day=crush_bu_day,
        installed_capacity_bu_day=float(installed_capacity_bu_day),
        gap_bu_day=crush_bu_day - installed_capacity_bu_day,
    )


def crush_balance(
    *,
    rvo_gallons: float,
    soyoil_share: float,
    installed_capacity_bu_day: float,
    yield_lb_gal: float = DEFAULT_YIELD_LB_GAL,
    oil_lb_per_bushel: float = OIL_LB_PER_BUSHEL,
) -> CrushBalance:
    """Capacité de trituration requise contre capacité annoncée.

        soyoil_requis_lb     = RVO_gal x part_soyoil x yield_lb_gal
        crush_requis_bu      = soyoil_requis_lb / 11
        gap_bu_jour          = crush_requis_bu / 365 - capacité_installée
    """
    if not 0.0 <= soyoil_share <= 1.0:
        raise FeedstockError(f"part_soyoil doit être dans [0, 1], reçu {soyoil_share}")
    if installed_capacity_bu_day <= 0:
        raise FeedstockError("la capacité installée doit être > 0")

    soyoil_lb = rvo_gallons * soyoil_share * yield_lb_gal
    crush_bu = soyoil_lb / oil_lb_per_bushel
    crush_bu_day = crush_bu / 365.0
    return CrushBalance(
        soyoil_required_lb=soyoil_lb,
        crush_required_bu=crush_bu,
        crush_required_bu_day=crush_bu_day,
        installed_capacity_bu_day=installed_capacity_bu_day,
        gap_bu_day=crush_bu_day - installed_capacity_bu_day,
    )


# ===========================================================================
# T3-5 — bêta énergie du complexe agri (section, pas page autonome)
# ===========================================================================
def rolling_energy_beta(
    ag_price: pd.Series, brent: pd.Series, *, window: int = 120
) -> pd.DataFrame:
    """Bêta glissant de Δln(prix agri) sur Δln(Brent).

    Le « biofuel pull » : quand le brut monte, la marge biocarburant tire l'huile
    végétale. La question de T3-5 est de savoir si le bêta est structurellement plus élevé
    depuis 2026 ou si c'était un épisode.

    Renvoie beta, r_squared et n_obs par fenêtre. `n_eff` d'une fenêtre glissante vaut
    n_obs/window (Règle C) : à afficher avec le graphe, sinon le bêta paraît bien mieux
    estimé qu'il ne l'est.
    """
    aligned = pd.concat({"ag": ag_price, "brent": brent}, axis=1).dropna()
    if len(aligned) < window + 2:
        raise FeedstockError(
            f"pas assez d'observations pour une fenêtre de {window} : n={len(aligned)}"
        )
    returns = np.log(aligned).diff().dropna()

    beta = (
        returns["ag"].rolling(window).cov(returns["brent"])
        / returns["brent"].rolling(window).var()
    )
    correlation = returns["ag"].rolling(window).corr(returns["brent"])
    out = pd.DataFrame(
        {"beta": beta, "r_squared": correlation**2, "n_obs": window}
    ).dropna()
    out.attrs["n_eff_per_window"] = 1.0
    out.attrs["window"] = window
    return out


@dataclass(frozen=True)
class ChowTest:
    """Test de rupture à une date connue — les dates de politique, pas des dates cherchées."""

    break_date: pd.Timestamp
    f_stat: float
    p_value: float
    beta_before: float
    beta_after: float
    n_before: int
    n_after: int

    @property
    def rejects_stability(self) -> bool:
        return self.p_value < 0.05

    @property
    def summary(self) -> str:
        verdict = (
            "rupture significative" if self.rejects_stability else "pas de rupture détectable"
        )
        return (
            f"bêta {self.beta_before:.3f} -> {self.beta_after:.3f} au {self.break_date:%d/%m/%Y} | "
            f"F = {self.f_stat:.2f}, p = {self.p_value:.4f} | {verdict}"
        )


def chow_break_test(
    ag_price: pd.Series, brent: pd.Series, break_date: str | pd.Timestamp
) -> ChowTest:
    """Test de Chow sur le bêta énergie, à une date de politique **choisie a priori**.

    Le point de rupture est donné, jamais cherché dans les données : chercher la rupture
    qui maximise le F et rapporter sa p-value nominale est un des moyens les plus rapides
    de produire un résultat qui ne se reproduira pas. Les dates légitimes ici sont
    celles du calendrier réglementaire — finalisation des RVO en mars 2026, entrée en
    vigueur du 45Z.
    """
    from scipy import stats as scipy_stats

    aligned = pd.concat({"ag": ag_price, "brent": brent}, axis=1).dropna()
    returns = np.log(aligned).diff().dropna()
    cut = pd.Timestamp(break_date)

    before = returns[returns.index < cut]
    after = returns[returns.index >= cut]
    if len(before) < 30 or len(after) < 30:
        raise FeedstockError(
            f"sous-échantillons trop courts autour du {cut:%d/%m/%Y} : "
            f"{len(before)} avant, {len(after)} après (30 minimum de chaque côté)"
        )

    def _fit(sample: pd.DataFrame) -> tuple[float, float]:
        x = np.column_stack([np.ones(len(sample)), sample["brent"].to_numpy()])
        y = sample["ag"].to_numpy()
        coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
        residuals = y - x @ coefficients
        return float(coefficients[1]), float(residuals @ residuals)

    beta_pooled, rss_pooled = _fit(returns)
    beta_before, rss_before = _fit(before)
    beta_after, rss_after = _fit(after)

    k = 2                                   # constante + pente
    n = len(returns)
    numerator = (rss_pooled - (rss_before + rss_after)) / k
    denominator = (rss_before + rss_after) / (n - 2 * k)
    f_stat = numerator / denominator if denominator > 0 else float("inf")
    p_value = float(scipy_stats.f.sf(f_stat, k, n - 2 * k))

    return ChowTest(
        break_date=cut,
        f_stat=float(f_stat),
        p_value=p_value,
        beta_before=beta_before,
        beta_after=beta_after,
        n_before=len(before),
        n_after=len(after),
    )


SRE_WARNING = (
    "Les exemptions de petites raffineries (SRE) réallouent rétroactivement les volumes "
    "RVO. Toute conclusion tirée d'un mandat affiché porte cette incertitude, et elle "
    "n'est pas modélisée ici (L-H6)."
)

__all__ = [
    "CENTS_PER_USD",
    "ChowTest",
    "CrushBalance",
    "DCO_DOMESTIC",
    "DiscountBurden",
    "Feedstock",
    "FeedstockError",
    "GateValue",
    "ImportPenalty",
    "LCFS_PROGRAM_HIGH_USD_T",
    "LCFS_PROGRAM_LOW_USD_T",
    "LcfsThreshold",
    "NoBreakevenInRange",
    "PenaltyBounds",
    "SOYOIL_DOMESTIC",
    "SRE_WARNING",
    "StructuralExit",
    "TALLOW_DOMESTIC",
    "UCO_IMPORTED",
    "calibration_gap_45z",
    "chow_break_test",
    "crush_balance",
    "crush_from_soyoil_lb",
    "discount_burden",
    "feedstock_breakeven_usd_lb",
    "gate_value",
    "import_penalty",
    "lcfs_breakeven",
    "lcfs_breakeven_numeric",
    "lcfs_neutral_price",
    "lcfs_value_usd_gal",
    "load_soyoil_usd_lb",
    "penalty_bounds",
    "rolling_energy_beta",
    "structural_exit",
    "winner_grid",
]
