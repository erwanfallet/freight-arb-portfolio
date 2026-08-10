"""Module T1-M — régime ou skill.

Origine du désaccord : Halsall, Commodity Conversations, 25 nov. 2024. À propos des
maisons qui battaient des records sur le cacao, il demande si c'est parce qu'on trade
mieux ou à cause des conditions de marché, et note qu'on juge le succès au chiffre absolu
plutôt qu'à la performance relative aux conditions.

Ce n'est pas une page autonome : c'est une section à insérer dans **chaque** page, et la
spec impose qu'elle tourne par défaut, pas en option — y compris quand le résultat est
défavorable. C'est précisément quand il est défavorable qu'il vaut quelque chose.

DEUX SORTIES, ET LA SECONDE EST CELLE QUI VA DANS LE MAIL
---------------------------------------------------------
1. `attribute_pnl_to_regime` : régresse le P&L par trade sur des variables de régime. Le
   fit donne le « P&L attendu du régime », le résidu est l'alpha présumé, et l'IC sur
   l'alpha dit si cet alpha est distinguable de zéro.
2. `honest_win_rate` : corrige le nombre de trades du chevauchement des positions, puis
   calcule l'IC exact du win rate sur ce n effectif.

La seconde est la plus rentable. Écrire soi-même « mon win rate de 100 % sur 18 trades
n'est pas distinguable d'un processus à 60 % » avant que le destinataire ne le pense est
le geste de crédibilité le moins cher disponible — et c'est exactement ce qu'il allait
penser en silence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from agri.core.resample import EffectiveSample, effective_n_from_trades
from agri.core.stats import (
    BootstrapCI,
    HacRegression,
    ProportionCI,
    clopper_pearson,
    hac_ols,
    stationary_bootstrap_indices,
)

# Variables de régime attendues par défaut. Chaque page les nomme dans ses propres
# termes (dispersion inter-origines pour un grain, inter-régions pour du sucre), mais la
# structure ne change pas.
DEFAULT_REGIME_COLUMNS = (
    "vol_realised",
    "term_structure_width",
    "dispersion",
    "duration",
)


class RegimeError(ValueError):
    """Attribution mal spécifiée — colonnes manquantes ou échantillon insuffisant."""


# ===========================================================================
# 1. Attribution du P&L au régime
# ===========================================================================
@dataclass(frozen=True)
class RegimeAttribution:
    """Ce que le régime explique, ce qu'il reste, et si ce reste est distinguable de zéro."""

    regression: HacRegression
    alpha_ci: BootstrapCI
    sample: EffectiveSample
    regime_columns: tuple[str, ...]

    @property
    def r_squared(self) -> float:
        """Part du P&L expliquée par les conditions de marché seules."""
        return self.regression.r_squared

    @property
    def alpha(self) -> float:
        """P&L moyen par trade non expliqué par le régime."""
        return float(self.regression.params["const"])

    @property
    def alpha_is_distinguishable_from_zero(self) -> bool:
        return not self.alpha_ci.includes_zero

    @property
    def is_overfit(self) -> bool:
        """Trop de paramètres pour le n effectif.

        Seuil retenu : moins de 3 observations indépendantes par paramètre estimé. En
        dessous, le R² mesure la souplesse du modèle, pas une régularité du marché — et
        il faut le dire au lieu d'afficher un R² flatteur.
        """
        n_params = len(self.regime_columns) + 1
        return self.sample.n_eff < 3 * n_params

    @property
    def verdict(self) -> str:
        if self.is_overfit:
            return "non interprétable"
        if self.alpha_is_distinguishable_from_zero:
            return "alpha présumé"
        return "indistinguable du régime"

    @property
    def mail_sentence(self) -> str:
        """La phrase chiffrée, générée depuis les données — jamais recopiée à la main."""
        if self.is_overfit:
            n_params = len(self.regime_columns) + 1
            return (
                f"Le régime explique {self.r_squared:.0%} de la variance du P&L, mais sur "
                f"{self.sample.n_eff:.1f} observations indépendantes pour {n_params} "
                "paramètres, ce chiffre mesure la souplesse du modèle et pas une "
                "régularité du marché. Je ne le lis pas."
            )
        if self.alpha_is_distinguishable_from_zero:
            return (
                f"Les conditions de marché expliquent {self.r_squared:.0%} du P&L. "
                f"Le résidu vaut {self.alpha:+.2f} par trade, "
                f"IC {self.alpha_ci.confidence:.0%} [{self.alpha_ci.lo:+.2f}, "
                f"{self.alpha_ci.hi:+.2f}] : il ne contient pas zéro, "
                f"sur n_eff = {self.sample.n_eff:.1f}."
            )
        return (
            f"Les conditions de marché expliquent {self.r_squared:.0%} du P&L. Ce qui "
            f"reste — {self.alpha:+.2f} par trade — a un IC "
            f"[{self.alpha_ci.lo:+.2f}, {self.alpha_ci.hi:+.2f}] qui contient zéro : "
            f"sur n_eff = {self.sample.n_eff:.1f}, je ne peux pas distinguer ce résultat "
            "du régime de marché."
        )


def attribute_pnl_to_regime(
    trades: pd.DataFrame,
    *,
    hold_days: int,
    pnl_column: str = "pnl",
    regime_columns: tuple[str, ...] | None = None,
    n_iter: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> RegimeAttribution:
    """Régresse le P&L par trade sur les variables de régime.

    `trades` est indexé par **date d'entrée** (c'est ce qui permet de mesurer le
    chevauchement réel) et contient au minimum `pnl_column` et les colonnes de régime.

    La longueur de bloc du bootstrap est fixée à la période de hold : c'est la durée sur
    laquelle deux trades restent dépendants.
    """
    columns = tuple(regime_columns) if regime_columns is not None else DEFAULT_REGIME_COLUMNS
    if pnl_column not in trades.columns:
        raise RegimeError(f"colonne P&L absente : {pnl_column!r}")
    missing = [c for c in columns if c not in trades.columns]
    if missing:
        raise RegimeError(
            f"colonnes de régime absentes : {missing}. Chaque page nomme ses propres "
            "variables de régime — les passer via regime_columns."
        )

    clean = trades[[pnl_column, *columns]].dropna()
    if len(clean) < len(columns) + 3:
        raise RegimeError(
            f"pas assez de trades : n={len(clean)} pour {len(columns)} variables de régime"
        )

    regression = hac_ols(clean[pnl_column], clean[list(columns)])
    sample = effective_n_from_trades(pd.DatetimeIndex(clean.index), hold_days)

    # IC sur l'alpha : le résidu moyen, rééchantillonné en blocs de la longueur du hold.
    block_len = min(float(hold_days), float(len(clean)))
    alpha_ci = _bootstrap_intercept(
        y=clean[pnl_column].to_numpy(dtype=float),
        regressors=clean[list(columns)].to_numpy(dtype=float),
        point=float(regression.params["const"]),
        block_len=block_len,
        n_iter=n_iter,
        confidence=confidence,
        seed=seed,
    )

    return RegimeAttribution(
        regression=regression,
        alpha_ci=alpha_ci,
        sample=sample,
        regime_columns=columns,
    )


def _bootstrap_intercept(
    *,
    y: np.ndarray,
    regressors: np.ndarray,
    point: float,
    block_len: float,
    n_iter: int,
    confidence: float,
    seed: int,
    chunk: int = 500,
) -> BootstrapCI:
    """IC bootstrap par blocs sur l'intercept, **avec refit complet** de la régression.

    Pourquoi le refit plutôt que le rééchantillonnage des résidus, qui serait dix fois
    moins cher : l'intercept vaut `ȳ − Σ β̂ⱼ x̄ⱼ`. Bootstrapper les résidus à β̂ figé
    traite les pentes comme connues et ignore la façon dont leur erreur d'estimation se
    propage dans l'intercept, amplifiée par la moyenne de chaque régresseur. Sur une
    variable de régime centrée loin de zéro — une durée de hold moyenne de 30 jours, par
    exemple — une erreur de 0,003 sur la pente déplace l'alpha de 0,09. L'IC sortirait
    beaucoup trop étroit, et conclurait à un alpha là où il n'y a que du bruit
    d'estimation : exactement le travers que ce module existe pour empêcher.

    Le calcul est fait par paquets pour borner la mémoire : (n_iter, n, k) en un bloc
    dépasserait le gigaoctet sur un backtest de taille normale.
    """
    n, k = regressors.shape
    design = np.column_stack([np.ones(n), regressors])   # l'intercept est la colonne 0
    rng = np.random.default_rng(seed)
    indices = stationary_bootstrap_indices(n, block_len, n_iter, rng)

    intercepts = np.empty(n_iter, dtype=float)
    for start in range(0, n_iter, chunk):
        idx = indices[start : start + chunk]
        x_b = design[idx]                                 # (c, n, k+1)
        y_b = y[idx]                                      # (c, n)
        xtx = np.einsum("cni,cnj->cij", x_b, x_b)
        xty = np.einsum("cni,cn->ci", x_b, y_b)
        try:
            # xty porte un axe supplémentaire : sous numpy 2, un second opérande en 2-D
            # est lu comme une matrice (m, n) et non comme un lot de vecteurs
            beta = np.linalg.solve(xtx, xty[..., None])[..., 0]
        except np.linalg.LinAlgError:
            # un rééchantillon peut être dégénéré (un bloc unique répété) : la
            # pseudo-inverse le traite sans faire tomber tout le bootstrap
            beta = np.einsum("cij,cj->ci", np.linalg.pinv(xtx), xty)
        intercepts[start : start + chunk] = beta[:, 0]

    tail = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(intercepts, [tail, 1.0 - tail])
    return BootstrapCI(
        point=point,
        lo=float(lo),
        hi=float(hi),
        confidence=confidence,
        block_len=float(block_len),
        n_iter=n_iter,
    )


# ===========================================================================
# 2. Le win rate honnête — la phrase du mail
# ===========================================================================
@dataclass(frozen=True)
class WinRateHonesty:
    """Win rate affiché contre win rate défendable, et l'écart entre les deux."""

    n_trades: int
    n_wins: int
    hold_days: int
    sample: EffectiveSample
    naive: ProportionCI
    honest: ProportionCI

    @property
    def lower_bound_cost(self) -> float:
        """Points de borne basse perdus en corrigeant du chevauchement."""
        return self.naive.lo - self.honest.lo

    @property
    def mail_sentence(self) -> str:
        """La phrase d'auto-critique préemptive, chiffrée depuis le backtest lui-même."""
        equivalent = self.honest.lo
        return (
            f"Sur {self.n_trades} positions tenues {self.hold_days} jours, le "
            f"chevauchement ({self.sample.overlap:.1f} positions ouvertes en moyenne) "
            f"ramène l'échantillon à {self.sample.n_eff:.1f} tirages indépendants. "
            f"Le win rate de {self.naive.point:.0%} a alors un IC exact de "
            f"[{equivalent:.0%}, {self.honest.hi:.0%}] : il n'est pas distinguable d'un "
            f"processus qui réussit {equivalent * 10:.0f} fois sur 10. "
            f"Annoncer la borne à {self.naive.lo:.0%} en ignorant le chevauchement "
            f"surestimerait ce que ce backtest démontre de "
            f"{self.lower_bound_cost * 100:.0f} points."
        )


def honest_win_rate(
    entry_dates: pd.DatetimeIndex,
    wins: pd.Series | np.ndarray | list[bool],
    *,
    hold_days: int,
    confidence: float = 0.95,
) -> WinRateHonesty:
    """Win rate avec IC exact calculé sur le **n effectif**, pas sur le nombre de trades.

    C'est le calcul à passer sur le backtest cuivre avant d'envoyer quoi que ce soit.
    18 positions tenues 30 jours avec plusieurs trades simultanés ne valent pas 18
    tirages indépendants, et un win rate de 100 % sur ~7 tirages a une borne basse aux
    alentours de 59 % — très loin de ce que « 100 % sur 18 trades » laisse entendre.
    """
    entries = pd.DatetimeIndex(entry_dates)
    outcomes = np.asarray(pd.Series(list(wins)).astype(bool))
    if len(outcomes) != len(entries):
        raise RegimeError(
            f"{len(entries)} dates d'entrée pour {len(outcomes)} résultats — "
            "les deux doivent correspondre trade à trade"
        )
    if len(entries) == 0:
        raise RegimeError("aucun trade")

    sample = effective_n_from_trades(entries, hold_days)
    n_wins = int(outcomes.sum())
    win_share = n_wins / len(outcomes)

    # On conserve la PROPORTION observée en la reportant sur le n effectif, plutôt que de
    # garder le compte brut de succès : c'est le nombre d'observations qui est réduit par
    # le chevauchement, pas le taux de réussite.
    effective_wins = int(round(win_share * np.floor(sample.n_eff)))

    return WinRateHonesty(
        n_trades=len(outcomes),
        n_wins=n_wins,
        hold_days=hold_days,
        sample=sample,
        naive=clopper_pearson(n_wins, len(outcomes), confidence=confidence),
        honest=clopper_pearson(effective_wins, sample.n_eff, confidence=confidence),
    )
