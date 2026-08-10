"""T3-4 — Chine soja : politique ou commercial ?

THÈSE
-----
Les achats de réserve d'État se concentrent là où le commercial **ne peut pas** acheter.
Si les achats se groupent dans les quintiles de marge de crush les plus bas, ce sont des
achats que l'économie du crush interdit — signature politique. S'ils se groupent dans les
quintiles hauts, c'est de la rotation de stock opportuniste — signature commerciale.

Le test est binaire, et le signe d'un seul coefficient le tranche.

LE DÉSACCORD (ouvert, sourcé)
-------------------------------
Sinograin a vendu environ la moitié des 504 000 t de soja importé proposées à sa plus
grosse enchère depuis janvier ; des traders cités par Reuters soutiennent que ces enchères
servent à faire de la place pour l'arrivée de nouvelles cargaisons américaines
(août 2026). En face, ADM a relevé ses perspectives 2026 pour la deuxième fois, invoquant
un environnement biocarburants constructif et l'attente que la Chine continue d'acheter du
soja US.

LE PIÈGE D'UNITÉ, ET IL Y EN A TROIS EMPILÉS
----------------------------------------------
1. Le CBOT cote en **USD/boisseau** (soja, 60 lb), la DCE en **CNY/tonne**.
2. Les prix chinois sont **TTC** ; la marge de crush se calcule HT. Et la TVA sur les
   oléagineux importés n'est pas celle des produits transformés — vérifier le taux
   applicable au produit **et à la date**, ils bougent.
3. Les droits de douane s'appliquent à la valeur CNF, pas au prix FOB.

Une seule de ces trois erreurs déplace la marge de plusieurs dizaines de CNY/t et inverse
le signe du test.

MODÈLE
------
    bean_cnf_usd_t = (CBOT_usd_bu + basis_c_bu/100) x 36,7437 + freight_usd_t
    crush_margin   = (0,785 x meal_dce + 0,185 x oil_dce) / (1 + TVA)
                     - bean_cnf_cny_t x (1 + droit) - processing

    reserve_flow = imports_douanes - crush_observe - usage_direct

Test discriminant, en logit :

    logit(1{achat_reserve}) = a + b1 crush_margin_{t-1} + b2 stock_days_{t-1}
                                + b3 price_level_{t-1}

    b1 < 0 significatif  ->  signature POLITIQUE
    b1 > 0 significatif  ->  signature COMMERCIALE

HYPOTHÈSES
----------
N-H1  Rendements de trituration chinois : 0,785 t de tourteau et 0,185 t d'huile par tonne
      de fève. Paramétrés.
N-H2  La marge est retardée d'une période dans le logit. Un achat décidé aujourd'hui
      répond à la marge d'hier, pas à celle qu'il contribue à créer — sans ce décalage, le
      test souffre d'une simultanéité qui peut inverser le signe.
N-H3  L'usage direct (alimentation animale non triturée, semences) est un forfait
      saisonnier. C'est le terme le plus faible du résidu de réserve, et il est affiché
      comme tel.
N-H4  Les séries de réserve d'État ne sont pas publiées. Le repli — et c'est le mode
      normal de fonctionnement — reconstruit le flux par résidu douanes moins crush
      implicite, ce qui accumule les erreurs de mesure des deux séries.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from agri.core.fmt import fr, fr_pct
from agri.core.stats import regime_runs
from agri.core.units import bushels_per_tonne, strip_vat

# N-H1
DEFAULT_MEAL_YIELD = 0.785
DEFAULT_OIL_YIELD = 0.185
DEFAULT_PROCESSING_CNY_T = 120.0

# Taux à vérifier à la date d'usage — ils bougent (piège d'unité n°2)
DEFAULT_VAT_PROCESSED = 0.09
DEFAULT_IMPORT_DUTY = 0.03

BUSHELS_PER_TONNE_SOYBEAN = bushels_per_tonne("soybean")   # 36,7437


class ChinaSoyError(ValueError):
    """Modèle mal spécifié."""


# ===========================================================================
# La marge de crush
# ===========================================================================
def bean_cnf_usd_t(
    cbot_usd_bu: pd.Series, basis_cents_bu: pd.Series, freight_usd_t: pd.Series
) -> pd.Series:
    """Coût rendu Chine, en USD/t. Le boisseau devient tonne **avant** l'ajout du fret.

    Ajouter un fret en USD/t à un prix en USD/boisseau est l'erreur silencieuse classique :
    le résultat reste un nombre plausible et se trompe d'un facteur 36.
    """
    frame = pd.concat(
        {"cbot": cbot_usd_bu, "basis": basis_cents_bu, "freight": freight_usd_t}, axis=1, sort=True
    ).dropna()
    if frame.empty:
        raise ChinaSoyError("aucune date commune au CBOT, au basis et au fret")
    fob_usd_bu = frame["cbot"] + frame["basis"] / 100.0
    return (fob_usd_bu * BUSHELS_PER_TONNE_SOYBEAN + frame["freight"]).rename("bean_cnf_usd_t")


def crush_margin_cny_t(
    meal_dce_cny_t: pd.Series,
    oil_dce_cny_t: pd.Series,
    bean_cnf_usd_t_series: pd.Series,
    usdcny: pd.Series,
    *,
    meal_yield: float = DEFAULT_MEAL_YIELD,
    oil_yield: float = DEFAULT_OIL_YIELD,
    vat_rate: float = DEFAULT_VAT_PROCESSED,
    import_duty: float = DEFAULT_IMPORT_DUTY,
    processing_cny_t: float = DEFAULT_PROCESSING_CNY_T,
) -> pd.DataFrame:
    """Marge de trituration chinoise, en CNY/t de fève, TVA et droits traités séparément.

    Colonnes : revenue_gross, revenue_ex_vat, bean_cost, margin.
    """
    if meal_yield + oil_yield > 1.0:
        raise ChinaSoyError(
            f"les rendements somment à {meal_yield + oil_yield:.3f} : plus de produits que "
            "de fève entrante"
        )
    frame = pd.concat(
        {
            "meal": meal_dce_cny_t,
            "oil": oil_dce_cny_t,
            "bean_usd": bean_cnf_usd_t_series,
            "fx": usdcny,
        },
        axis=1,
        sort=True,
    ).dropna()
    if frame.empty:
        raise ChinaSoyError("aucune date commune aux quatre séries")
    if (frame["fx"] <= 0).any():
        raise ChinaSoyError("USDCNY doit être > 0 — vérifier le sens de cotation")

    out = pd.DataFrame(index=frame.index)
    out["revenue_gross"] = meal_yield * frame["meal"] + oil_yield * frame["oil"]
    out["revenue_ex_vat"] = strip_vat(out["revenue_gross"], vat_rate)
    out["bean_cost"] = frame["bean_usd"] * frame["fx"] * (1.0 + import_duty)
    out["margin"] = out["revenue_ex_vat"] - out["bean_cost"] - processing_cny_t
    out.attrs["vat_rate"] = vat_rate
    out.attrs["import_duty"] = import_duty
    return out


# ===========================================================================
# Le flux de réserve, par résidu (N-H4)
# ===========================================================================
def reserve_flow(
    imports_t: pd.Series,
    crush_observed_t: pd.Series,
    *,
    direct_use_t: pd.Series | float = 0.0,
) -> pd.DataFrame:
    """`réserve = imports - crush - usage direct`, avec l'accumulation d'erreur assumée.

    Les deux séries d'entrée portent chacune leur erreur de mesure, et le résidu les
    additionne. Un résidu de faible amplitude n'est donc pas interprétable ; seuls les
    mouvements larges le sont. La colonne `is_large` marque le seuil retenu.
    """
    frame = pd.concat({"imports": imports_t, "crush": crush_observed_t}, axis=1, sort=True).dropna()
    if frame.empty:
        raise ChinaSoyError("aucune date commune aux imports et au crush")
    direct = (
        pd.Series(float(direct_use_t), index=frame.index)
        if isinstance(direct_use_t, (int, float))
        else pd.Series(direct_use_t).reindex(frame.index).fillna(0.0)
    )
    frame["direct_use"] = direct
    frame["reserve_flow"] = frame["imports"] - frame["crush"] - frame["direct_use"]
    threshold = float(frame["reserve_flow"].abs().quantile(0.60))
    frame["is_large"] = frame["reserve_flow"].abs() > threshold
    frame["is_purchase"] = (frame["reserve_flow"] > threshold).astype(int)
    frame.attrs["large_threshold_t"] = threshold
    return frame


# ===========================================================================
# LE TEST DISCRIMINANT
# ===========================================================================
@dataclass(frozen=True)
class SignatureTest:
    """Le signe de b1 tranche entre politique et commercial."""

    beta_margin: float
    pvalue_margin: float
    beta_stock: float
    beta_price: float
    pseudo_r2: float
    n_obs: int
    n_purchases: int

    @property
    def is_significant(self) -> bool:
        return self.pvalue_margin < 0.05

    @property
    def signature(self) -> str:
        if not self.is_significant:
            return "indéterminée"
        return "politique" if self.beta_margin < 0 else "commerciale"

    @property
    def headline(self) -> str:
        if not self.is_significant:
            return (
                f"Le lien entre marge de crush et achat de réserve n'est pas significatif "
                f"(b1 = {self.beta_margin:+.5f}, p = {self.pvalue_margin:.3f}, "
                f"n = {self.n_obs} dont {self.n_purchases} achats). Sur cet échantillon, "
                "je ne peux pas distinguer une signature politique d'une rotation de stock."
            )
        if self.beta_margin < 0:
            return (
                f"Les achats de réserve se concentrent dans les quintiles de marge de crush "
                f"les plus bas (b1 = {self.beta_margin:+.5f}, p = {self.pvalue_margin:.3f}) : "
                "ce sont des achats que le commercial ne peut pas faire, pas des achats "
                "qu'il refuse."
            )
        return (
            f"Les achats de réserve suivent la marge de crush (b1 = {self.beta_margin:+.5f}, "
            f"p = {self.pvalue_margin:.3f}) : la signature est commerciale, pas politique. "
            "L'État achète quand c'est économique, comme tout le monde."
        )


def signature_test(
    purchases: pd.Series,
    margin: pd.Series,
    stock_days: pd.Series,
    price_level: pd.Series,
    *,
    lag: int = 1,
) -> SignatureTest:
    """Logit de l'achat de réserve sur la marge de crush retardée (N-H2).

    Les trois régresseurs sont **standardisés** avant estimation. Sans ça, la marge en
    CNY/t (des centaines) et les jours de stock (des dizaines) produisent des coefficients
    d'ordres de grandeur incomparables, et le logit converge mal.
    """
    import statsmodels.api as sm

    frame = pd.concat(
        {
            "y": purchases,
            "margin": margin.shift(lag),
            "stock": stock_days.shift(lag),
            "price": price_level.shift(lag),
        },
        axis=1,
        sort=True,
    ).dropna()
    if len(frame) < 40:
        raise ChinaSoyError(f"échantillon trop court pour un logit : n={len(frame)}")
    n_purchases = int(frame["y"].sum())
    if n_purchases < 5 or n_purchases > len(frame) - 5:
        raise ChinaSoyError(
            f"trop peu de variation à expliquer : {n_purchases} achats sur {len(frame)} "
            "observations. Un logit ne dit rien sur une variable quasi constante."
        )

    design = frame[["margin", "stock", "price"]]
    standardised = (design - design.mean()) / design.std()
    standardised = sm.add_constant(standardised)

    model = sm.Logit(frame["y"], standardised).fit(disp=0)
    return SignatureTest(
        beta_margin=float(model.params["margin"]),
        pvalue_margin=float(model.pvalues["margin"]),
        beta_stock=float(model.params["stock"]),
        beta_price=float(model.params["price"]),
        pseudo_r2=float(model.prsquared),
        n_obs=len(frame),
        n_purchases=n_purchases,
    )


def purchases_by_margin_quintile(
    purchases: pd.Series, margin: pd.Series, *, lag: int = 1
) -> pd.DataFrame:
    """Taux d'achat par quintile de marge — la lecture qui précède le logit.

    Un tableau à cinq lignes convainc un desk plus vite qu'un coefficient, et il montre
    tout de suite si la relation est monotone ou concentrée dans une seule queue.
    """
    frame = pd.concat({"y": purchases, "margin": margin.shift(lag)}, axis=1, sort=True).dropna()
    if len(frame) < 20:
        raise ChinaSoyError(f"échantillon trop court pour des quintiles : n={len(frame)}")
    frame["quintile"] = pd.qcut(frame["margin"], 5, labels=[1, 2, 3, 4, 5])
    grouped = frame.groupby("quintile", observed=True).agg(
        n_obs=("y", "size"),
        n_purchases=("y", "sum"),
        purchase_rate=("y", "mean"),
        mean_margin=("margin", "mean"),
    )
    return grouped.reset_index()


# Ordres de grandeur documentes pour la basis et le fret US Gulf -> Chine (constantes
# parametrees, PAS des donnees reelles) : ni l'un ni l'autre n'est dans l'export
# Bloomberg. Meme traitement que le roll omis de T1-2 ou l'energie constante de T2-4 —
# affiche comme limite, pas cache.
DEFAULT_BASIS_CENTS_BU = 70.0
DEFAULT_FREIGHT_USD_T = 45.0


def load_real_crush_frame(
    *,
    start: str = "2018-01-01",
    basis_cents_bu: float = DEFAULT_BASIS_CENTS_BU,
    freight_usd_t: float = DEFAULT_FREIGHT_USD_T,
    **margin_kwargs,
) -> pd.DataFrame:
    """Marge de crush chinoise sur CBOT soja, DCE tourteau/huile et USDCNY **réels**.

    LIMITE DE DONNÉE, DOCUMENTÉE : le basis FOB US Gulf et le fret Chine ne sont pas dans
    l'export Bloomberg — ils restent des forfaits paramétrés (mêmes valeurs que le repli
    du fixture synthétique), appliqués sur un prix CBOT réel. Trois jambes sur quatre
    (soja CBOT, tourteau DCE, huile DCE, change USDCNY) sont entièrement réelles ; seule
    la conversion FOB->CNF porte un terme constant.

    Renvoie le DataFrame complet de `crush_margin_cny_t` (revenue_gross, revenue_ex_vat,
    bean_cost, margin) — pas seulement la marge — pour que la page puisse tracer le
    waterfall terme à terme. Ne calcule pas le test de signature politique/commerciale,
    qui exige un signal d'achat de réserve (`purchases`) qu'aucune source publique
    gratuite ne fournit en série temporelle : ce volet reste illustratif sur synthétique.
    """
    from agri.data.bloomberg_loader import load as load_bloomberg

    cbot = load_bloomberg("cbot_soybean").loc[start:]
    meal = load_bloomberg("dce_soymeal")
    oil = load_bloomberg("dce_soyoil")
    fx = load_bloomberg("usdcny")

    basis = pd.Series(basis_cents_bu, index=cbot.index)
    freight = pd.Series(freight_usd_t, index=cbot.index)
    bean_cnf = bean_cnf_usd_t(cbot, basis, freight)

    margin = crush_margin_cny_t(meal, oil, bean_cnf, fx, **margin_kwargs)
    margin.attrs["basis_cents_bu"] = basis_cents_bu
    margin.attrs["freight_usd_t"] = freight_usd_t
    margin.attrs["real_legs"] = ["cbot_soybean", "dce_soymeal", "dce_soyoil", "usdcny"]
    margin.attrs["parametrized_legs"] = ["basis_cents_bu", "freight_usd_t"]
    return margin


# ===========================================================================
# LE TEST QUI NE DEMANDE PAS LES DONNÉES D'ENCHÈRES
# ===========================================================================
# Le test de signature ci-dessus est binaire et propre, mais il exige une série d'achats de
# réserve que l'export ne contient pas — et que Sinograin ne publie pas en série temporelle.
# On le remplace par un argument qui ne demande aucune donnée de flux, seulement des prix.
#
# L'idée : la marge de crush chinoise borne par le haut ce qu'un triturateur peut payer pour
# une tonne de fève rendue. Retranchée du CBOT converti à la tonne, cette borne devient un
# **budget pour le basis d'origine et le fret** — ce dont dispose un originateur pour aller
# chercher la fève, où que ce soit. Quand ce budget passe sous zéro, aucune origine ne
# fonctionne : même une fève gratuite, transportée gratuitement, ne rend pas le crush
# rentable. Des cargaisons qui arrivent dans ces fenêtres ne sont pas commerciales **par
# construction arithmétique**, sans qu'aucun test statistique ait à être passé.
@dataclass(frozen=True)
class OriginationBudget:
    """Ce qu'un originateur peut dépenser en basis + fret, et quand il ne peut plus rien.

    `frame` porte le budget jour par jour ; les propriétés en donnent les régimes. Le
    livrable est le **calendrier** des fenêtres impossibles, pas un niveau moyen : c'est un
    objet qu'un insider confronte à ses propres arrivées.
    """

    frame: pd.DataFrame
    freight_reference_usd_t: float

    @property
    def median_budget(self) -> float:
        return float(self.frame["budget_usd_t"].median())

    @property
    def last_budget(self) -> float:
        return float(self.frame["budget_usd_t"].iloc[-1])

    @property
    def share_impossible(self) -> float:
        """Part des séances où AUCUNE origine ne fonctionne, fret et basis nuls compris."""
        return float((self.frame["budget_usd_t"] < 0).mean())

    @property
    def share_below_freight(self) -> float:
        """Part des séances où le fret seul consomme tout le budget, ne laissant rien pour
        le basis d'origine — il faudrait acheter la fève SOUS le CBOT."""
        return float((self.frame["budget_usd_t"] < self.freight_reference_usd_t).mean())

    @property
    def headline(self) -> str:
        return (
            f"Le crush chinois peut financer {fr(self.median_budget, 0)} USD/t de basis et "
            f"de fret en médiane, et {fr(self.last_budget, 0)} au dernier cours. Mais "
            f"{fr_pct(self.share_impossible, 1)} des séances affichent un budget "
            f"**négatif** — une fève gratuite, transportée gratuitement, ne rendrait pas le "
            f"crush rentable — et {fr_pct(self.share_below_freight)} le placent sous le "
            f"seul coût du fret ({fr(self.freight_reference_usd_t, 0)} USD/t)."
        )


def affordable_origination_budget(
    *,
    start: str = "2018-01-01",
    freight_reference_usd_t: float = DEFAULT_FREIGHT_USD_T,
    processing_cny_t: float = DEFAULT_PROCESSING_CNY_T,
    import_duty: float = DEFAULT_IMPORT_DUTY,
    **margin_kwargs,
) -> OriginationBudget:
    """Le budget basis + fret que la marge de crush chinoise autorise, en USD/tonne.

        recette_HT     = (0,785 x tourteau_DCE + 0,185 x huile_DCE) / (1 + TVA)
        CNF_max_CNY_t  = (recette_HT - transformation) / (1 + droit)
        CNF_max_USD_t  = CNF_max_CNY_t / USDCNY
        budget         = CNF_max_USD_t - CBOT_usd_bu x 36,7437

    Le budget est ce qui reste pour aller chercher la fève. Il ne dépend d'**aucune**
    hypothèse de basis ni de fret — c'est précisément ce qui le rend utile : les deux termes
    que l'export ne contient pas sortent du calcul au lieu d'y entrer.
    """
    from agri.data.bloomberg_loader import load

    crush = load_real_crush_frame(start=start, **margin_kwargs)
    aligned = pd.concat(
        {
            "revenue_ex_vat": crush["revenue_ex_vat"],
            "cbot_usd_bu": load("cbot_soybean"),
            "usdcny": load("usdcny"),
        },
        axis=1,
        sort=True,
    ).dropna()
    aligned = aligned[aligned.index >= pd.Timestamp(start)]
    if aligned.empty:
        raise ChinaSoyError(f"aucune date commune aux jambes du crush chinois après {start}")
    if (aligned["usdcny"] <= 0).any():
        raise ChinaSoyError("USDCNY nul ou négatif — vérifier le sens de cotation")

    cnf_max_cny = (aligned["revenue_ex_vat"] - processing_cny_t) / (1.0 + import_duty)
    aligned["cnf_max_usd_t"] = cnf_max_cny / aligned["usdcny"]
    aligned["cbot_usd_t"] = aligned["cbot_usd_bu"] * BUSHELS_PER_TONNE_SOYBEAN
    aligned["budget_usd_t"] = aligned["cnf_max_usd_t"] - aligned["cbot_usd_t"]
    aligned["impossible"] = aligned["budget_usd_t"] < 0
    aligned["below_freight"] = aligned["budget_usd_t"] < freight_reference_usd_t

    return OriginationBudget(
        frame=aligned, freight_reference_usd_t=float(freight_reference_usd_t)
    )


def impossible_windows(
    budget: OriginationBudget, *, threshold_usd_t: float = 0.0, min_obs: int = 3
) -> pd.DataFrame:
    """Le calendrier des fenêtres où le budget d'origination passe sous un seuil.

    C'est le livrable de la page : des dates, pas un coefficient. Un originateur les
    confronte à ses propres arrivées, et la question « avez-vous chargé pendant celles-là ? »
    se répond par oui ou par non.
    """
    return regime_runs(
        budget.frame["budget_usd_t"] < threshold_usd_t,
        depth=budget.frame["budget_usd_t"],
        min_obs=min_obs,
    )


__all__ = [
    "BUSHELS_PER_TONNE_SOYBEAN",
    "ChinaSoyError",
    "DEFAULT_BASIS_CENTS_BU",
    "DEFAULT_FREIGHT_USD_T",
    "DEFAULT_IMPORT_DUTY",
    "DEFAULT_PROCESSING_CNY_T",
    "OriginationBudget",
    "SignatureTest",
    "affordable_origination_budget",
    "bean_cnf_usd_t",
    "crush_margin_cny_t",
    "impossible_windows",
    "load_real_crush_frame",
    "purchases_by_margin_quintile",
    "reserve_flow",
    "signature_test",
]
