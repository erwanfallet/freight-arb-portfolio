"""Le point de bascule — le livrable de chaque page du portefeuille.

Un seuil chiffré qu'un praticien peut contester en dix secondes vaut mieux qu'une analyse
juste. « Le breakeven est à 52 % de tarif » appelle une réponse ; « voici mon analyse »
n'en appelle aucune. Ce module transforme une fonction de marge en ce seuil-là.

Trois sorties, et la troisième est celle qui protège :
    theta_star        le niveau où le signe change
    sensitivity       d(marge)/d(theta) au point de bascule, en unités lisibles
    distance_sigmas   l'écart entre le niveau courant et theta_star, en écarts-types
                      historiques de theta

Sans la troisième, on annonce un breakeven à trois sigmas comme s'il était imminent —
c'est la façon la plus rapide de perdre un lecteur qui connaît son marché.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import brentq


class BreakevenError(ValueError):
    """Problème mal posé — bornes incohérentes, fonction non numérique."""


class NoBreakevenInRange(Exception):
    """Aucun changement de signe sur l'intervalle exploré.

    **Ce n'est pas un échec de calcul, c'est un résultat**, et souvent le plus fort de la
    page : « sur toute la plage plausible de ballast facturé, l'arb reste ouvert » est une
    affirmation falsifiable. Les valeurs aux deux bornes sont attachées à l'exception pour
    que la page puisse l'afficher au lieu de tomber en erreur.
    """

    def __init__(self, lo: float, hi: float, margin_lo: float, margin_hi: float):
        self.lo = lo
        self.hi = hi
        self.margin_lo = margin_lo
        self.margin_hi = margin_hi
        sign = "positive" if margin_lo > 0 else "négative"
        super().__init__(
            f"aucun changement de signe sur [{lo:g}, {hi:g}] : la marge reste {sign} "
            f"({margin_lo:+.4f} en {lo:g}, {margin_hi:+.4f} en {hi:g}). "
            "Ce n'est pas une erreur — c'est le résultat à afficher, sans extrapoler "
            "hors de la plage plausible."
        )


@dataclass(frozen=True)
class Breakeven:
    """Point de bascule, sa sensibilité locale, et sa distance au niveau courant."""

    theta_star: float
    sensitivity: float
    bracket: tuple[float, float]
    theta_current: float | None = None
    theta_sigma: float | None = None
    theta_label: str = "theta"
    margin_label: str = "marge"

    @property
    def distance_sigmas(self) -> float | None:
        """Distance du niveau courant au point de bascule, en écarts-types de theta.

        None quand l'historique n'a pas été fourni : mieux vaut ne rien afficher que
        d'afficher une distance sans échelle.
        """
        if self.theta_current is None or self.theta_sigma is None:
            return None
        if self.theta_sigma == 0:
            return float("inf")
        return (self.theta_star - self.theta_current) / self.theta_sigma

    @property
    def is_within_reach(self) -> bool:
        """Le point de bascule est-il à moins d'un écart-type ? Sinon, dire qu'il est loin."""
        d = self.distance_sigmas
        return d is not None and abs(d) <= 1.0

    @property
    def summary(self) -> str:
        head = (
            f"{self.theta_label}* = {self.theta_star:.4g} "
            f"(d{self.margin_label}/d{self.theta_label} = {self.sensitivity:+.4g} au seuil)"
        )
        d = self.distance_sigmas
        if d is None:
            return head
        reach = "à portée" if self.is_within_reach else "hors de portée sur l'historique"
        return (
            f"{head} | niveau courant {self.theta_current:.4g}, "
            f"soit {d:+.2f} écart-type — {reach}"
        )


def solve_breakeven(
    margin_fn: Callable[[float], float],
    lo: float,
    hi: float,
    *,
    theta_current: float | None = None,
    theta_history: pd.Series | np.ndarray | None = None,
    h: float | None = None,
    theta_label: str = "theta",
    margin_label: str = "marge",
    xtol: float = 1e-10,
) -> Breakeven:
    """Résout `margin_fn(theta) = 0` sur [lo, hi] et mesure la sensibilité au seuil.

    `margin_fn` doit être monotone en theta sur l'intervalle — c'est le cas de toutes les
    marges de ce portefeuille (un ballast plus élevé ne peut que renchérir le fret, un
    crédit LCFS plus élevé ne peut qu'améliorer la valeur en sortie d'usine). La monotonie
    n'est pas vérifiée numériquement : elle doit être argumentée dans la page.

    `theta_history` sert à exprimer la distance en écarts-types. Le fournir est
    fortement recommandé : c'est ce qui distingue un seuil imminent d'un seuil théorique.
    """
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise BreakevenError(f"bornes non finies : [{lo}, {hi}]")
    if lo >= hi:
        raise BreakevenError(f"bornes incohérentes : lo={lo} doit être < hi={hi}")

    margin_lo = float(margin_fn(lo))
    margin_hi = float(margin_fn(hi))
    if not np.isfinite(margin_lo) or not np.isfinite(margin_hi):
        raise BreakevenError(
            f"marge non finie aux bornes : f({lo})={margin_lo}, f({hi})={margin_hi}"
        )
    if margin_lo == 0.0:
        theta_star = lo
    elif margin_hi == 0.0:
        theta_star = hi
    elif np.sign(margin_lo) == np.sign(margin_hi):
        raise NoBreakevenInRange(lo, hi, margin_lo, margin_hi)
    else:
        theta_star = float(brentq(lambda t: float(margin_fn(t)), lo, hi, xtol=xtol))

    step = h if h is not None else (hi - lo) * 1e-5
    if step <= 0:
        raise BreakevenError(f"le pas de dérivation doit être > 0, reçu {step}")
    # on garde les deux points d'évaluation à l'intérieur du bracket : hors bornes, la
    # fonction de marge peut être indéfinie (un ballast négatif n'existe pas)
    left = max(theta_star - step, lo)
    right = min(theta_star + step, hi)
    sensitivity = (float(margin_fn(right)) - float(margin_fn(left))) / (right - left)

    sigma = None
    if theta_history is not None:
        history = pd.Series(theta_history).dropna().astype(float)
        if len(history) >= 2:
            sigma = float(history.std())

    return Breakeven(
        theta_star=theta_star,
        sensitivity=sensitivity,
        bracket=(lo, hi),
        theta_current=theta_current,
        theta_sigma=sigma,
        theta_label=theta_label,
        margin_label=margin_label,
    )
