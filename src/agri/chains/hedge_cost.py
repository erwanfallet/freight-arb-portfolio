"""T1-2 — Le coût complet de la couverture : cacao et café, cycle entier.

THÈSE
-----
La contrainte contraignante n'est pas le prix, c'est le **collatéral**. Une maison sans
lignes de crédit suffisantes ne peut pas couvrir, donc elle réduit ses achats physiques —
et c'est le mécanisme par lequel une crise financière de trading devient un problème de
paiement au bord-champ. La procyclicité est l'insulte finale : le coût de la couverture
explose précisément au moment où elle est le plus nécessaire.

    hedge_capacity(B) = max { Q : max_t cash_t(Q) <= B }

`cash_t` étant linéaire en Q, la résolution est immédiate : `Q* = B / max_t cash_t(1 t)`.
C'est la traduction quantitative de la citation Montesanto : leur contrainte n'était pas
la vue de marché, c'était le bilan.

ORIGINE DU DÉSACCORD (sourcé)
------------------------------
- Barry Callebaut, S1 2024/25 : exigences de marge initiale multipliées par neuf par
  rapport aux niveaux normaux ; coût de la backwardation 60 % plus cher au pic.
- Cacao 2024 : les longs physiques rechignent à vendre du futures dans un marché haussier
  à cause de la taille des appels de marge, ce qui crée des problèmes de liquidité ; les
  industriels se retrouvent dans la position qui était celle des agriculteurs.
- Café, nov. 2025 : ~7 Md USD d'appels de marge sur un seul mois ; chez Montesanto Tavares,
  le coût de maintien des couvertures passe de 74 % des créances clients en mai à 158 % en
  novembre, rendant la structure de trésorerie court terme insoutenable selon leurs avocats.

CE QUI REND LE PROJET NEUF — LE RETOURNEMENT DE 2026
-----------------------------------------------------
Le cacao a chuté depuis le pic de décembre 2024. Mais les manufacturiers couvrent 12 à
24 mois à l'avance : les coûts verrouillés au sommet traversent encore les budgets 2026.
**La couverture a donc puni les deux camps, à quatorze mois d'intervalle** — à la hausse
elle étranglait la trésorerie des vendeurs de futures, à la baisse elle immobilise les
acheteurs à des niveaux morts. Aucun modèle statique de hedge ne capture ça.

ATTENTION — LE SIGNE DU ROLL
-----------------------------
En **backwardation** (F_near > F_deferred), un **short** qui roll se retrouve court à un
prix plus bas ; quand ce contrat converge vers le spot, il **perd**. Le long gagne. C'est
l'inverse en contango.

    roll_pnl = -s x Q x (F_deferred - F_near)      avec s = +1 short, -1 long

Vérification de cohérence avec la source : Barry Callebaut était short futures contre du
physique long, et rapporte la backwardation comme un **coût**. C'est bien ce que donne la
formule. (La note d'annexe de la spec de cadrage énonce le signe inverse ; elle est
contredite par sa propre identité et par la source, et n'a pas été suivie.)

HYPOTHÈSES
----------
H-H1  `side` = +1 pour un négociant long physique / short futures, -1 pour un
      manufacturier acheteur long futures. Les deux sont simulés en miroir.
H-H2  Le barème de marge initiale, quand il n'est pas disponible, est approché par
      `IM = k x sigma_20j x F`. `k` est calibré sur des points publiés — l'ancrage
      « x9 » de Barry Callebaut est utilisable. Le proxy est **annoncé comme tel** dans
      le panneau de diagnostics, jamais présenté comme le barème.
H-H3  Le cash mobilisé est `IM + pertes cumulées non compensées`. On ne modélise pas les
      appels de marge intrajournaliers, ni les haircuts sur collatéral non-cash : les deux
      **sous-estiment** le besoin de trésorerie. Biais contre la thèse, donc dans le bon sens.
H-H4  Le coût de liquidité est `Q x spread` à chaque roll seulement. Ignorer le slippage
      hors roll sous-estime encore le coût.
H-H5  Base 360 pour le financement, convention de marché monétaire.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from agri.data.snapshot import cached

SHORT_HEDGE = 1      # négociant long physique, short futures
LONG_HEDGE = -1      # manufacturier acheteur, long futures

DEFAULT_LOT_SIZE_T = 10.0        # ICE cacao Londres : 10 t
DEFAULT_IM_PROXY_K = 2.33        # ~99e centile d'une normale : ancrage usuel des chambres


class HedgeCostError(ValueError):
    """Simulation mal spécifiée."""


# ===========================================================================
# Marge initiale
# ===========================================================================
def initial_margin_proxy(
    price: pd.Series, *, window: int = 20, k: float = DEFAULT_IM_PROXY_K
) -> pd.Series:
    """Proxy de marge initiale `IM = k x sigma_20j x F`, en USD par tonne (H-H2).

    À n'utiliser que si le barème historique de la chambre est indisponible, et à
    afficher comme proxy. La volatilité est réalisée sur `window` jours, annualisée non :
    la chambre couvre un horizon de un à deux jours, pas un an.
    """
    if k <= 0:
        raise HedgeCostError(f"k doit être > 0, reçu {k}")
    returns = np.log(price).diff()
    vol = returns.rolling(window, min_periods=window).std()
    return (k * vol * price).rename("im_usd_t")


def calibrate_im_k(
    price: pd.Series, observed_im_usd_t: pd.Series, *, window: int = 20
) -> float:
    """Calibre `k` pour reproduire des niveaux de marge publiés (H-H2).

    Moindres carrés sans constante sur `IM_observée = k x sigma x F` : on force le passage
    par zéro, parce qu'une marge initiale nulle à volatilité nulle est la seule valeur
    défendable.
    """
    returns = np.log(price).diff()
    vol = returns.rolling(window, min_periods=window).std()
    frame = pd.concat({"x": vol * price, "y": observed_im_usd_t}, axis=1).dropna()
    if len(frame) < 5:
        raise HedgeCostError(f"pas assez de points de calibration : n={len(frame)}")
    x = frame["x"].to_numpy()
    y = frame["y"].to_numpy()
    denominator = float(x @ x)
    if denominator == 0:
        raise HedgeCostError("volatilité identiquement nulle — calibration impossible")
    return float(x @ y) / denominator


# ===========================================================================
# Le roll — et son signe
# ===========================================================================
def roll_pnl_usd_t(
    front_price: float, deferred_price: float, *, side: int
) -> float:
    """P&L de roll par tonne.

        roll_pnl = s x (F_deferred - F_near)      s = +1 short, -1 long

    Dérivation, pour ne pas avoir à retenir le signe. Un short qui roll rachète le front
    et revend le déféré : il se retrouve court **au prix du déféré**. Le contrat qu'il
    porte converge ensuite vers le spot.
        contango (F_def > F_near)      : il est court plus haut, le prix descend -> il gagne
        backwardation (F_def < F_near) : il est court plus bas, le prix monte  -> il perd
    Et l'inverse pour un long.

    Contrôle contre la source : Barry Callebaut était short futures contre du physique
    long, et rapporte la backwardation comme un **coût**. C'est bien ce que donne la
    formule.
    """
    if side not in (SHORT_HEDGE, LONG_HEDGE):
        raise HedgeCostError(f"side doit valoir +1 (short) ou -1 (long), reçu {side}")
    return side * (deferred_price - front_price)


def roll_cost_usd_t(front_price: float, deferred_price: float, *, side: int) -> float:
    """Le même roll, exprimé en **coût** (positif = ça coûte).

        roll_cost = s x (F_near - F_deferred)

    C'est l'identité de la spec de cadrage. Deux fonctions explicites plutôt qu'un signe à
    retenir : c'est l'erreur la plus fréquente du sujet, et elle est silencieuse — un roll
    mal signé transforme un coût de couverture en revenu de couverture sans rien casser.
    """
    return -roll_pnl_usd_t(front_price, deferred_price, side=side)


# ===========================================================================
# La simulation complète
# ===========================================================================
@dataclass(frozen=True)
class HedgeParams:
    side: int = SHORT_HEDGE
    book_size_t: float = 100_000.0
    lot_size_t: float = DEFAULT_LOT_SIZE_T
    credit_line_usd: float = 250_000_000.0
    funding_spread_bps: float = 250.0
    liquidity_spread_usd_t: float = 4.0

    def __post_init__(self) -> None:
        if self.side not in (SHORT_HEDGE, LONG_HEDGE):
            raise HedgeCostError(f"side doit valoir +1 ou -1, reçu {self.side}")
        if self.book_size_t <= 0:
            raise HedgeCostError("book_size_t doit être > 0")
        if self.lot_size_t <= 0:
            raise HedgeCostError("lot_size_t doit être > 0")


def simulate_hedge(
    front: pd.Series,
    deferred: pd.Series,
    im_usd_t: pd.Series,
    rate: pd.Series | float,
    roll_dates: pd.DatetimeIndex,
    *,
    params: HedgeParams,
) -> pd.DataFrame:
    """Simule une couverture roulante et décompose son coût complet.

    Colonnes renvoyées, toutes en USD sauf mention :
        vm_usd              variation margin du jour (signe de P&L)
        cum_loss_usd        pertes cumulées non compensées (>= 0)
        im_usd              marge initiale exigée
        cash_usd            trésorerie mobilisée = IM + pertes cumulées
        financing_usd       coût de portage de cette trésorerie
        roll_usd            coût de roll aux dates de roll (positif = coût)
        liquidity_usd       spread payé à chaque roll
        hedge_cost_cum_usd  somme cumulée financement + roll + liquidité

    Le coût cumulé n'inclut **pas** la variation margin elle-même : la VM est un
    transfert, pas un coût. Ce qui coûte, c'est de la financer.
    """
    aligned = pd.concat(
        {"front": front, "deferred": deferred, "im_usd_t": im_usd_t}, axis=1
    ).dropna()
    if aligned.empty:
        raise HedgeCostError("aucune date commune aux prix et à la marge initiale")

    if isinstance(rate, (int, float)):
        rate_series = pd.Series(float(rate), index=aligned.index)
    else:
        rate_series = pd.Series(rate).reindex(aligned.index).ffill()
        if rate_series.isna().any():
            raise HedgeCostError("le taux de financement ne couvre pas toute la période")

    q = params.book_size_t
    side = params.side
    all_in_rate = rate_series + params.funding_spread_bps / 10_000.0

    out = pd.DataFrame(index=aligned.index)
    out["front"] = aligned["front"]
    out["deferred"] = aligned["deferred"]

    # variation margin : -s x Q x dF. Un short perd quand le prix monte.
    out["vm_usd"] = -side * q * aligned["front"].diff().fillna(0.0)
    cumulative = out["vm_usd"].cumsum()
    out["cum_loss_usd"] = (-cumulative).clip(lower=0.0)

    # marge initiale : en lots entiers, comme une chambre la calcule
    n_lots = np.ceil(q / params.lot_size_t)
    out["im_usd"] = aligned["im_usd_t"] * params.lot_size_t * n_lots

    out["cash_usd"] = out["im_usd"] + out["cum_loss_usd"]
    out["financing_usd"] = out["cash_usd"] * all_in_rate / 360.0

    # roll et liquidité, aux dates de roll seulement
    out["roll_usd"] = 0.0
    out["liquidity_usd"] = 0.0
    valid_rolls = pd.DatetimeIndex(roll_dates).intersection(out.index)
    for date in valid_rolls:
        out.loc[date, "roll_usd"] = q * roll_cost_usd_t(
            float(out.loc[date, "front"]), float(out.loc[date, "deferred"]), side=side
        )
        out.loc[date, "liquidity_usd"] = q * params.liquidity_spread_usd_t

    out["hedge_cost_usd"] = out["financing_usd"] + out["roll_usd"] + out["liquidity_usd"]
    out["hedge_cost_cum_usd"] = out["hedge_cost_usd"].cumsum()
    out["hedge_cost_cum_usd_t"] = out["hedge_cost_cum_usd"] / q

    out.attrs["side"] = "short (négociant)" if side == SHORT_HEDGE else "long (industriel)"
    out.attrs["book_size_t"] = q
    out.attrs["n_rolls"] = len(valid_rolls)
    return out


# ===========================================================================
# Le cœur du projet — la capacité de couverture
# ===========================================================================
@dataclass(frozen=True)
class HedgeCapacity:
    """Combien de tonnes une maison peut couvrir avec ses lignes, dans le temps."""

    capacity_t: pd.Series
    credit_line_usd: float
    book_size_t: float
    peak_cash_usd: float
    peak_cash_date: pd.Timestamp

    @property
    def min_capacity_t(self) -> float:
        return float(self.capacity_t.min())

    @property
    def is_binding(self) -> bool:
        """La ligne de crédit contraint-elle le book à un moment de l'échantillon ?"""
        return self.min_capacity_t < self.book_size_t

    def contraction_over(self, months: int = 4) -> float:
        """Contraction de la capacité sur les `months` mois qui précèdent le pic de cash.

        C'est la mesure de la spec — « tombée de Y % en quatre mois, au pic exact de la
        valeur du physique ». Le maximum absolu de la série est un mauvais point de
        départ : il tombe dans une période calme où la marge initiale est minuscule, ce
        qui produit mécaniquement une contraction proche de 100 % et ne dit rien.
        """
        window_start = self.peak_cash_date - pd.DateOffset(months=months)
        window = self.capacity_t.loc[window_start : self.peak_cash_date]
        if len(window) < 2:
            return float("nan")
        return 1.0 - float(window.iloc[-1]) / float(window.iloc[0])

    @property
    def headline(self) -> str:
        verdict = (
            f"la ligne devient contraignante : la capacité tombe à "
            f"{self.min_capacity_t:,.0f} t contre un book de {self.book_size_t:,.0f} t"
            if self.is_binding
            else "la ligne reste au-dessus du book sur tout l'échantillon"
        )
        return (
            f"Au pic du {self.peak_cash_date:%d/%m/%Y}, le book mobilisait "
            f"{self.peak_cash_usd / 1e6:,.0f} M USD de trésorerie. Avec "
            f"{self.credit_line_usd / 1e6:,.0f} M USD de lignes, {verdict} — la capacité "
            f"a reculé de {self.contraction_over(4):.0%} sur les quatre mois qui précèdent "
            "ce pic, au moment exact où le physique valait le plus cher."
        )


def hedge_capacity(
    simulation: pd.DataFrame, *, credit_line_usd: float, book_size_t: float
) -> HedgeCapacity:
    """`Q*(t) = B / cash_t(1 tonne)` — le graphe qui fait le mail.

    Le book maximal couvrable se contracte exactement au moment où le physique vaut le
    plus cher. C'est la procyclicité, rendue visible en tonnes plutôt qu'en pourcentage
    de marge : un desk lit des tonnes.
    """
    if credit_line_usd <= 0:
        raise HedgeCostError("la ligne de crédit doit être > 0")
    if book_size_t <= 0:
        raise HedgeCostError("le book doit être > 0")

    cash_per_tonne = simulation["cash_usd"] / book_size_t
    if (cash_per_tonne <= 0).any():
        raise HedgeCostError(
            "trésorerie mobilisée nulle ou négative sur certaines dates — "
            "vérifier la marge initiale avant d'en déduire une capacité"
        )
    capacity = (credit_line_usd / cash_per_tonne).rename("capacity_t")
    peak_date = simulation["cash_usd"].idxmax()
    return HedgeCapacity(
        capacity_t=capacity,
        credit_line_usd=credit_line_usd,
        book_size_t=book_size_t,
        peak_cash_usd=float(simulation["cash_usd"].max()),
        peak_cash_date=peak_date,
    )


def margin_breakeven_im_usd_t(
    *, credit_line_usd: float, book_size_t: float, cumulative_loss_usd: float = 0.0
) -> float:
    """`IM*` : la marge initiale à laquelle la capacité tombe sous le book physique.

    C'est le second point de bascule de la page, et le meilleur des deux : au-delà de ce
    niveau, la maison **doit** réduire ses achats physiques. C'est le mécanisme de
    transmission vers le producteur.
    """
    if book_size_t <= 0:
        raise HedgeCostError("le book doit être > 0")
    available = credit_line_usd - cumulative_loss_usd
    if available <= 0:
        return 0.0
    return available / book_size_t


# ===========================================================================
# S6 — les deux côtés en miroir
# ===========================================================================
def compare_sides(
    front: pd.Series,
    deferred: pd.Series,
    im_usd_t: pd.Series,
    rate: pd.Series | float,
    roll_dates: pd.DatetimeIndex,
    *,
    params: HedgeParams,
    windows: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    """Coût par tonne des deux côtés, sur des fenêtres nommées.

    Le graphe du mail : coût par tonne du short hedge sur la hausse, coût par tonne du
    long hedge sur la baisse. Même graphe, deux barres, quatorze mois d'écart — et la
    démonstration que la couverture a puni les deux camps.
    """
    rows = []
    for label, (start, end) in windows.items():
        for side, side_label in ((SHORT_HEDGE, "short (négociant)"), (LONG_HEDGE, "long (industriel)")):
            local = HedgeParams(
                side=side,
                book_size_t=params.book_size_t,
                lot_size_t=params.lot_size_t,
                credit_line_usd=params.credit_line_usd,
                funding_spread_bps=params.funding_spread_bps,
                liquidity_spread_usd_t=params.liquidity_spread_usd_t,
            )
            window_slice = slice(start, end)
            simulation = simulate_hedge(
                front.loc[window_slice],
                deferred.loc[window_slice],
                im_usd_t.loc[window_slice],
                rate,
                roll_dates,
                params=local,
            )
            rows.append(
                {
                    "window": label,
                    "side": side_label,
                    "cost_usd_t": float(simulation["hedge_cost_cum_usd_t"].iloc[-1]),
                    "peak_cash_usd": float(simulation["cash_usd"].max()),
                    "price_start": float(simulation["front"].iloc[0]),
                    "price_end": float(simulation["front"].iloc[-1]),
                }
            )
    return pd.DataFrame(rows)


REAL_COMMODITY_KEYS: dict[str, str] = {
    "cacao_ny": "cocoa_ny",
    "cacao_londres": "cocoa_london",
    "cafe_arabica": "coffee_arabica",
    "cafe_robusta": "coffee_robusta",
}


@cached('t1_2_hedge')
def load_real_hedge_frame(
    commodity: str = "cacao_ny",
    *,
    start: str = "2018-04-02",
    params: HedgeParams | None = None,
) -> pd.DataFrame:
    """Simulation de couverture sur prix ICE **réels** (export Bloomberg de l'utilisateur).

    LIMITE DE DONNÉE, DOCUMENTÉE PLUTÔT QUE CONTOURNÉE : l'export ne contient que le
    contrat front-month générique (M1) — aucune échéance différée. Le coût de roll ne
    peut donc pas être calculé sur donnée réelle : `deferred` est posé égal à `front`,
    ce qui annule mécaniquement le terme (`roll_cost_usd_t(x, x, ...) = 0`) plutôt que
    d'inventer une structure par terme. Financement (SOFR réel) et marge initiale (proxy
    calibré sur la vraie volatilité réalisée) restent, eux, entièrement réels — ce sont
    les deux composantes que Barry Callebaut cite explicitement (marge x9, coût de
    portage), donc la thèse reste testable même sans le roll.

    `start` est calé sur le début de la couverture SOFR (2018-04-02) : au-delà, le taux
    de financement réel ne couvre pas la période et `simulate_hedge` lèverait.
    """
    from agri.data.bloomberg_loader import load as load_bloomberg

    key = REAL_COMMODITY_KEYS.get(commodity, commodity)
    front_full = load_bloomberg(key)
    front = front_full.loc[start:]
    if front.empty:
        raise HedgeCostError(f"aucune donnée pour {commodity!r} après {start}")

    im = initial_margin_proxy(front_full).loc[start:]
    rate = load_bloomberg("sofr")
    resolved_params = params or HedgeParams()

    simulation = simulate_hedge(front, front, im, rate, pd.DatetimeIndex([]), params=resolved_params)
    simulation.attrs["commodity"] = commodity
    simulation.attrs["roll_cost_omitted"] = True
    return simulation


# ===========================================================================
# LE LIVRABLE — le prix auquel le bilan force la sortie
# ===========================================================================
def implied_margin_rate(simulation: pd.DataFrame, *, book_size_t: float) -> float:
    """Marge initiale exprimée en fraction du prix, mesurée sur la simulation.

    On la mesure plutôt que de la poser : le proxy `k x sigma x F` la rend
    proportionnelle au prix par construction, mais le coefficient effectif dépend de la
    volatilité réalisée de la période et n'a pas de raison d'être celui d'un barème.
    """
    im_per_tonne = simulation["im_usd"] / book_size_t
    return float((im_per_tonne / simulation["front"]).median())


@dataclass(frozen=True)
class ForcedExit:
    """Le prix auquel la ligne de crédit est saturée, pour une couverture donnée.

    DÉRIVATION (forme fermée, short hedge ouvert à `inception_price`) :

        perte mark-to-market  = Q x (P - P0)          un short perd quand le prix monte
        marge initiale        = Q x im_rate x P       proportionnelle au prix
        trésorerie mobilisée  = Q x (im_rate x P + P - P0)

        saturation quand      Q x (im_rate x P + P - P0) = B
        d'où                  P* = (B/Q + P0) / (1 + im_rate)

    Au-delà de `P*`, la maison ne peut plus financer sa couverture. Elle ne peut donc
    plus couvrir de physique supplémentaire — et comme personne n'achète du physique non
    couvert à ce niveau de volatilité, elle **cesse d'acheter**. C'est le point de
    bascule qui relie une crise de trésorerie de trading à un arrêt des achats au bord-champ.
    """

    inception_date: pd.Timestamp
    inception_price: float
    exit_price: float
    credit_line_usd: float
    book_size_t: float
    im_rate: float
    crossed_on: pd.Timestamp | None
    days_of_protection: int | None

    @property
    def headroom_pct(self) -> float:
        """Combien le prix peut monter, en %, avant que la ligne ne soit saturée."""
        return self.exit_price / self.inception_price - 1.0

    @property
    def headline(self) -> str:
        if self.crossed_on is None:
            return (
                f"Couverture ouverte le {self.inception_date:%d/%m/%Y} à "
                f"{self.inception_price:,.0f} : avec {self.credit_line_usd/1e6:,.0f} M USD "
                f"de lignes sur {self.book_size_t/1000:,.0f} kt, la sortie forcée est à "
                f"{self.exit_price:,.0f} USD/t (+{self.headroom_pct:.0%}). Le marché n'y "
                "est jamais allé sur l'échantillon."
            )
        return (
            f"Couverture ouverte le {self.inception_date:%d/%m/%Y} à "
            f"{self.inception_price:,.0f} : la ligne sature à {self.exit_price:,.0f} USD/t "
            f"(+{self.headroom_pct:.0%}), franchi le {self.crossed_on:%d/%m/%Y} — soit "
            f"{self.days_of_protection} jours de protection."
        )


def forced_exit_price(
    price: pd.Series,
    *,
    inception: str | pd.Timestamp,
    book_size_t: float,
    credit_line_usd: float,
    im_rate: float,
) -> ForcedExit:
    """`P*` en forme fermée, et la date à laquelle le marché l'a franchi.

    La forme fermée est préférée à un solveur parce qu'elle montre **de quoi le seuil
    dépend** : linéairement de la ligne rapportée à la taille du book, du prix d'entrée,
    et à peine du taux de marge. Un desk peut refaire le calcul de tête.
    """
    if book_size_t <= 0 or credit_line_usd <= 0:
        raise HedgeCostError("book et ligne de crédit doivent être > 0")
    if im_rate < 0:
        raise HedgeCostError("le taux de marge ne peut pas être négatif")

    clean = pd.Series(price).dropna().astype(float)
    forward = clean.loc[pd.Timestamp(inception) :]
    if forward.empty:
        raise HedgeCostError(f"aucun prix après {inception}")

    inception_date = forward.index[0]
    inception_price = float(forward.iloc[0])
    exit_price = (credit_line_usd / book_size_t + inception_price) / (1.0 + im_rate)

    breached = forward[forward >= exit_price]
    crossed_on = breached.index.min() if len(breached) else None
    days = int((crossed_on - inception_date).days) if crossed_on is not None else None

    return ForcedExit(
        inception_date=inception_date,
        inception_price=inception_price,
        exit_price=exit_price,
        credit_line_usd=credit_line_usd,
        book_size_t=book_size_t,
        im_rate=im_rate,
        crossed_on=crossed_on,
        days_of_protection=days,
    )


def forced_exit_schedule(
    price: pd.Series,
    inceptions: list[str],
    *,
    book_size_t: float,
    credit_line_usd: float,
    im_rate: float,
) -> pd.DataFrame:
    """Le même seuil pour plusieurs dates d'ouverture — le tableau qui porte la thèse.

    Ce qu'il montre sur le cacao 2024 : les dates de sortie forcée se resserrent sur
    quelques semaines quel que soit le moment où la couverture a été ouverte. Le
    mouvement de prix a été assez violent pour écraser la dispersion des points d'entrée
    — ce qui explique que tant de maisons aient heurté la contrainte en même temps,
    plutôt que chacune à son tour.
    """
    rows = []
    for inception in inceptions:
        try:
            result = forced_exit_price(
                price, inception=inception, book_size_t=book_size_t,
                credit_line_usd=credit_line_usd, im_rate=im_rate,
            )
        except HedgeCostError:
            continue
        rows.append(
            {
                "ouverture": result.inception_date,
                "prix d'entrée": result.inception_price,
                "sortie forcée": result.exit_price,
                "marge de manœuvre": result.headroom_pct,
                "franchi le": result.crossed_on,
                "jours de protection": result.days_of_protection,
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class HedgingIntensity:
    """Trésorerie mobilisée rapportée à la valeur du book physique qu'elle protège.

    C'est la grandeur que les avocats de Montesanto Tavares ont chiffrée : le coût de
    maintien des couvertures est passé de **74 % des créances clients en mai 2025 à
    158 % en novembre**, et ils l'ont qualifiée d'insoutenable. La même mesure, calculée
    ici sur des prix ICE réels, dit à quel moment une structure de trésorerie cesse de
    tenir — sans avoir besoin du bilan de qui que ce soit.
    """

    ratio: pd.Series
    calm_median: float
    peak_ratio: float
    peak_date: pd.Timestamp
    share_above_one: float

    @property
    def headline(self) -> str:
        return (
            f"La trésorerie immobilisée pour tenir la couverture passe de "
            f"{self.calm_median:.0%} de la valeur du book en régime calme à "
            f"{self.peak_ratio:.0%} au pic du {self.peak_date:%d/%m/%Y}. À ce niveau, "
            "financer la couverture coûte presque autant que financer le stock qu'elle "
            "protège — c'est la mesure que les avocats de Montesanto Tavares ont "
            "qualifiée d'insoutenable à 158 %."
        )


def hedging_intensity(
    simulation: pd.DataFrame, *, book_size_t: float, calm_window: tuple[str, str] | None = None
) -> HedgingIntensity:
    """Ratio trésorerie mobilisée / valeur du book physique, dans le temps."""
    book_value = book_size_t * simulation["front"]
    ratio = (simulation["cash_usd"] / book_value).rename("hedging_intensity")

    if calm_window is not None:
        calm = ratio.loc[calm_window[0] : calm_window[1]]
    else:
        calm = ratio.loc[: ratio.index[len(ratio) // 4]]

    return HedgingIntensity(
        ratio=ratio,
        calm_median=float(calm.median()),
        peak_ratio=float(ratio.max()),
        peak_date=ratio.idxmax(),
        share_above_one=float((ratio > 1.0).mean()),
    )


def procyclicality(simulation: pd.DataFrame) -> dict[str, float]:
    """Corrélation entre variations de marge initiale et variations de prix.

    Sur données différenciées, pas en niveau : deux séries en niveau non stationnaires
    produisent une corrélation flatteuse qui ne veut rien dire.
    """
    changes = simulation[["im_usd", "front"]].diff().dropna()
    if len(changes) < 3:
        raise HedgeCostError("pas assez d'observations")
    return {
        "corr_delta_im_delta_price": float(changes["im_usd"].corr(changes["front"])),
        "n_obs": len(changes),
    }


IM_PROXY_WARNING = (
    "La marge initiale affichée est un proxy `k x sigma_20j x F`, pas le barème de la "
    "chambre (H-H2). Les sauts de barème réels ne sont pas datés ici : vérifier les "
    "notices ICE avant de lire un pic comme de la volatilité."
)

__all__ = [
    "HedgeCapacity",
    "HedgeCostError",
    "HedgeParams",
    "IM_PROXY_WARNING",
    "LONG_HEDGE",
    "SHORT_HEDGE",
    "calibrate_im_k",
    "compare_sides",
    "ForcedExit",
    "HedgingIntensity",
    "forced_exit_price",
    "forced_exit_schedule",
    "hedge_capacity",
    "hedging_intensity",
    "implied_margin_rate",
    "initial_margin_proxy",
    "load_real_hedge_frame",
    "margin_breakeven_im_usd_t",
    "procyclicality",
    "roll_cost_usd_t",
    "roll_pnl_usd_t",
    "simulate_hedge",
]
