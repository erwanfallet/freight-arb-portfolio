"""Boîte à outils statistique — conçue pour produire des refus de conclure.

Le fil directeur : chaque fonction renvoie sa mesure **avec ce qui permet de la
disqualifier**. Une corrélation sans sa bande de significativité, un R² sans son n
effectif, un win rate sans son intervalle de confiance sont des chiffres qui survivent
mal à un desk. C'est précisément là que se joue la crédibilité d'une page.

Tout ce qui prend un `n_eff` le prend explicitement : ces fonctions ne savent pas si
l'échantillon qu'on leur donne provient de fenêtres chevauchantes. C'est à l'appelant de
le dire, via `core.resample.effective_n*`.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

__all__ = [
    "StationarityVerdict",
    "adf_kpss",
    "CrossCorrelation",
    "ccf_with_band",
    "HacRegression",
    "hac_ols",
    "newey_west_lags",
    "BootstrapCI",
    "block_bootstrap",
    "stationary_bootstrap_indices",
    "regime_runs",
    "clopper_pearson",
    "ProportionCI",
]


class StatsError(ValueError):
    """Test mal spécifié ou échantillon insuffisant — toujours une erreur d'appelant."""


# ===========================================================================
# Stationnarité : les deux tests, toujours
# ===========================================================================
@dataclass(frozen=True)
class StationarityVerdict:
    """Résultat conjoint ADF + KPSS.

    Les deux tests ont des hypothèses nulles **opposées** :
        ADF  H0 = racine unitaire      -> p petit = stationnaire
        KPSS H0 = stationnarité        -> p petit = non stationnaire

    Les lancer tous les deux est le seul moyen de distinguer « stationnaire » de
    « je n'ai pas assez de données pour le dire ». Quand ils se contredisent, on ne
    conclut pas : `verdict` vaut alors "conflicting" ou "inconclusive", et la page doit
    l'afficher plutôt que de choisir le test qui l'arrange.
    """

    adf_stat: float
    adf_pvalue: float
    kpss_stat: float
    kpss_pvalue: float
    alpha: float
    n_obs: int

    @property
    def adf_rejects_unit_root(self) -> bool:
        return self.adf_pvalue < self.alpha

    @property
    def kpss_rejects_stationarity(self) -> bool:
        return self.kpss_pvalue < self.alpha

    @property
    def verdict(self) -> str:
        adf_ok = self.adf_rejects_unit_root
        kpss_bad = self.kpss_rejects_stationarity
        if adf_ok and not kpss_bad:
            return "stationary"
        if not adf_ok and kpss_bad:
            return "unit_root"
        if adf_ok and kpss_bad:
            # les deux rejettent : typiquement une rupture structurelle ou de
            # l'hétéroscédasticité, pas une série « moyennement stationnaire »
            return "conflicting"
        return "inconclusive"

    @property
    def summary(self) -> str:
        labels = {
            "stationary": "stationnaire (les deux tests concordent)",
            "unit_root": "racine unitaire (les deux tests concordent)",
            "conflicting": "CONTRADICTOIRE — les deux rejettent : suspecter une rupture "
                           "structurelle ou de l'hétéroscédasticité, ne pas conclure",
            "inconclusive": "NON CONCLUANT — aucun des deux ne rejette : échantillon "
                            "probablement trop court, ne pas conclure",
        }
        return (
            f"{labels[self.verdict]} | ADF p={self.adf_pvalue:.3f}, "
            f"KPSS p={self.kpss_pvalue:.3f}, n={self.n_obs}"
        )


def adf_kpss(
    series: pd.Series, *, alpha: float = 0.05, regression: str = "c"
) -> StationarityVerdict:
    """ADF et KPSS sur la même série, avec verdict conjoint.

    `regression='c'` teste autour d'une constante (le cas usuel pour un spread ou une
    marge) ; `'ct'` ajoute une tendance déterministe.
    """
    from statsmodels.tsa.stattools import adfuller, kpss

    clean = pd.Series(series).dropna().astype(float)
    if len(clean) < 20:
        raise StatsError(
            f"au moins 20 observations sont nécessaires pour un test de stationnarité "
            f"lisible, reçu {len(clean)}"
        )

    adf_stat, adf_p, *_ = adfuller(clean.to_numpy(), regression=regression, autolag="AIC")

    with warnings.catch_warnings():
        # KPSS borne sa p-value dans [0.01, 0.10] et prévient quand elle sature ; c'est
        # une information portée par le verdict, pas un problème à faire remonter.
        warnings.simplefilter("ignore")
        kpss_stat, kpss_p, *_ = kpss(clean.to_numpy(), regression=regression, nlags="auto")

    return StationarityVerdict(
        adf_stat=float(adf_stat),
        adf_pvalue=float(adf_p),
        kpss_stat=float(kpss_stat),
        kpss_pvalue=float(kpss_p),
        alpha=alpha,
        n_obs=len(clean),
    )


# ===========================================================================
# Cross-corrélation avec bande de significativité
# ===========================================================================
@dataclass(frozen=True)
class CrossCorrelation:
    """CCF et sa bande de Bartlett. La bande est le produit, pas la corrélation."""

    lags: np.ndarray
    values: np.ndarray
    band: float
    n_eff: float

    @property
    def significant_lags(self) -> np.ndarray:
        """Les seuls lags dont on a le droit de parler."""
        return self.lags[np.abs(self.values) > self.band]

    def peak(self) -> tuple[int, float]:
        """Lag de corrélation absolue maximale, et sa valeur — significative ou non."""
        i = int(np.nanargmax(np.abs(self.values)))
        return int(self.lags[i]), float(self.values[i])

    @property
    def summary(self) -> str:
        lag, value = self.peak()
        verdict = (
            "SIGNIFICATIF" if abs(value) > self.band
            else f"DANS LA BANDE (±{self.band:.3f}) — indistinguable de zéro"
        )
        return (
            f"pic à lag {lag:+d} : rho = {value:+.3f} | {verdict} | "
            f"n_eff = {self.n_eff:.1f}"
        )


def ccf_with_band(
    x: pd.Series,
    y: pd.Series,
    *,
    max_lag: int = 20,
    n_eff: float | None = None,
    confidence: float = 0.95,
) -> CrossCorrelation:
    """Cross-corrélation entre x et y, avec la bande de Bartlett ±z/sqrt(n_eff).

    CONVENTION DE SIGNE, à afficher dans la page parce qu'elle s'inverse d'un logiciel à
    l'autre : la valeur au lag k est `corr(x[t], y[t+k])`. Un pic à **lag positif**
    signifie donc que **x précède y**.

    `n_eff` est la taille d'échantillon effective (Règle C). Le laisser à None utilise le
    nombre d'observations brut — correct seulement si les observations ne se chevauchent
    pas. C'est le paramètre qui décide de la largeur de la bande, donc de ce qu'on a le
    droit d'appeler un signal : sur n=150, une corrélation de -0.10 est dans la bande.
    """
    aligned = pd.concat({"x": x, "y": y}, axis=1).dropna()
    n = len(aligned)
    if n < max_lag + 10:
        raise StatsError(
            f"échantillon trop court pour {max_lag} lags : n={n}. "
            "Réduire max_lag ou renoncer au test."
        )

    xs = aligned["x"].to_numpy(dtype=float)
    ys = aligned["y"].to_numpy(dtype=float)
    xs = xs - xs.mean()
    ys = ys - ys.mean()
    denom = np.sqrt(np.sum(xs**2) * np.sum(ys**2))
    if denom == 0:
        raise StatsError("une des deux séries est constante — corrélation indéfinie")

    lags = np.arange(-max_lag, max_lag + 1)
    values = np.empty(len(lags), dtype=float)
    for i, k in enumerate(lags):
        if k >= 0:
            values[i] = np.sum(xs[: n - k] * ys[k:]) / denom
        else:
            values[i] = np.sum(xs[-k:] * ys[: n + k]) / denom

    effective = float(n_eff) if n_eff is not None else float(n)
    if effective < 2:
        raise StatsError(f"n_eff doit être >= 2, reçu {effective}")
    z = float(scipy_stats.norm.ppf(0.5 + confidence / 2.0))
    return CrossCorrelation(
        lags=lags, values=values, band=z / np.sqrt(effective), n_eff=effective
    )


# ===========================================================================
# Régression à erreurs HAC (Newey-West)
# ===========================================================================
def newey_west_lags(n_obs: int) -> int:
    """Règle de sélection automatique : floor(4 * (n/100)^(2/9))."""
    if n_obs < 1:
        raise StatsError(f"n_obs doit être >= 1, reçu {n_obs}")
    return int(np.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0)))


@dataclass(frozen=True)
class HacRegression:
    """MCO à erreurs Newey-West. Les t-stats naïves d'une série temporelle mentent."""

    params: pd.Series
    std_errors: pd.Series
    tvalues: pd.Series
    pvalues: pd.Series
    r_squared: float
    n_obs: int
    lags: int

    def is_significant(self, name: str, *, alpha: float = 0.05) -> bool:
        return bool(self.pvalues[name] < alpha)

    def describe(self, name: str) -> str:
        return (
            f"{name} = {self.params[name]:+.4f} "
            f"(HAC se = {self.std_errors[name]:.4f}, t = {self.tvalues[name]:+.2f}, "
            f"p = {self.pvalues[name]:.3f})"
        )


def hac_ols(
    y: pd.Series,
    X: pd.DataFrame | pd.Series,
    *,
    lags: int | None = None,
    add_constant: bool = True,
) -> HacRegression:
    """MCO avec covariance HAC (Newey-West).

    `lags=None` applique `newey_west_lags(n)`. Toutes les régressions de ce portefeuille
    portent sur des séries temporelles autocorrélées : la covariance HAC n'est pas un
    raffinement, c'est la condition pour que la p-value affichée veuille dire quelque chose.
    """
    import statsmodels.api as sm

    if isinstance(X, pd.Series):
        X = X.to_frame()
    aligned = pd.concat([pd.Series(y).rename("__y__"), X], axis=1).dropna()
    if len(aligned) < X.shape[1] + 3:
        raise StatsError(
            f"pas assez d'observations : n={len(aligned)} pour {X.shape[1]} régresseurs"
        )

    y_clean = aligned["__y__"].astype(float)
    X_clean = aligned.drop(columns="__y__").astype(float)
    if add_constant:
        X_clean = sm.add_constant(X_clean, has_constant="add")

    n = len(aligned)
    chosen = newey_west_lags(n) if lags is None else int(lags)
    fitted = sm.OLS(y_clean, X_clean).fit(
        cov_type="HAC", cov_kwds={"maxlags": chosen, "use_correction": True}
    )
    return HacRegression(
        params=fitted.params,
        std_errors=fitted.bse,
        tvalues=fitted.tvalues,
        pvalues=fitted.pvalues,
        r_squared=float(fitted.rsquared),
        n_obs=n,
        lags=chosen,
    )


# ===========================================================================
# Bootstrap par blocs (Politis-Romano stationnaire)
# ===========================================================================
@dataclass(frozen=True)
class BootstrapCI:
    """Intervalle de confiance qui respecte la dépendance temporelle."""

    point: float
    lo: float
    hi: float
    confidence: float
    block_len: float
    n_iter: int

    @property
    def includes_zero(self) -> bool:
        return self.lo <= 0.0 <= self.hi

    @property
    def summary(self) -> str:
        verdict = " — INCLUT ZÉRO" if self.includes_zero else ""
        return (
            f"{self.point:+.4f} [IC {self.confidence:.0%} : {self.lo:+.4f}, "
            f"{self.hi:+.4f}]{verdict}"
        )


def stationary_bootstrap_indices(
    n: int, block_len: float, n_iter: int, rng: np.random.Generator
) -> np.ndarray:
    """Indices de rééchantillonnage Politis-Romano, entièrement vectorisés.

    Les blocs ont une longueur géométrique de moyenne `block_len` et bouclent en fin de
    série. Longueur aléatoire plutôt que fixe : c'est ce qui rend le bootstrap
    stationnaire, un découpage en blocs de taille fixe introduit une périodicité.
    """
    p = 1.0 / block_len
    starts = rng.random((n_iter, n)) < p
    starts[:, 0] = True
    anchors = rng.integers(0, n, size=(n_iter, n))

    positions = np.arange(n)
    last_start = np.maximum.accumulate(np.where(starts, positions, -1), axis=1)
    block_anchor = np.take_along_axis(anchors, last_start, axis=1)
    return (block_anchor + (positions - last_start)) % n


def block_bootstrap(
    values: pd.Series | np.ndarray,
    *,
    block_len: float,
    statistic=None,
    n_iter: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapCI:
    """IC par bootstrap stationnaire — jamais une p-value naïve sur un P&L.

    `block_len` doit être de l'ordre de la **période de hold** du backtest : c'est la
    longueur sur laquelle les observations restent dépendantes. Un bloc de 1 revient à
    supposer l'indépendance, ce qui est exactement l'hypothèse qu'on cherche à éviter.

    `statistic=None` prend la moyenne (chemin vectorisé). Toute autre statistique est un
    callable appliqué à chaque rééchantillon.

    `seed` est fixé par défaut : un IC qui bouge d'un rafraîchissement de page à l'autre
    détruit la confiance dans le dashboard bien plus vite qu'il ne renseigne sur l'aléa.
    """
    array = np.asarray(pd.Series(values).dropna(), dtype=float)
    n = len(array)
    if n < 5:
        raise StatsError(f"au moins 5 observations sont nécessaires, reçu {n}")
    if block_len < 1:
        raise StatsError(f"block_len doit être >= 1, reçu {block_len}")
    if block_len > n:
        raise StatsError(
            f"block_len ({block_len}) dépasse la longueur de la série ({n}) — "
            "le rééchantillonnage ne ferait que recopier la série"
        )

    rng = np.random.default_rng(seed)
    indices = stationary_bootstrap_indices(n, float(block_len), n_iter, rng)
    resampled = array[indices]

    if statistic is None:
        draws = resampled.mean(axis=1)
        point = float(array.mean())
    else:
        draws = np.apply_along_axis(statistic, 1, resampled).astype(float)
        point = float(statistic(array))

    tail = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(draws, [tail, 1.0 - tail])
    return BootstrapCI(
        point=point,
        lo=float(lo),
        hi=float(hi),
        confidence=confidence,
        block_len=float(block_len),
        n_iter=n_iter,
    )


# ===========================================================================
# Régimes : durée et profondeur
# ===========================================================================
def regime_runs(
    mask: pd.Series, *, depth: pd.Series | None = None, min_obs: int = 1
) -> pd.DataFrame:
    """Épisodes consécutifs où `mask` est vrai.

    Colonnes : start, end, n_obs, duration_days, depth_mean, depth_min, depth_max.

    C'est la fonction qui produit les phrases de mail. « La marge est négative » est une
    observation ; « la marge est négative depuis 7 mois, la plus longue série depuis
    2015 » est un fait daté que le destinataire peut contester — donc auquel il peut
    répondre. La différence tient entièrement dans la durée et la profondeur.
    """
    flag = pd.Series(mask).astype("boolean").fillna(False).astype(bool).sort_index()
    columns = ["start", "end", "n_obs", "duration_days", "depth_mean", "depth_min", "depth_max"]
    if not flag.any():
        return pd.DataFrame(columns=columns)

    if depth is not None:
        depth = pd.Series(depth).reindex(flag.index)

    group_id = (flag != flag.shift()).cumsum()
    rows = []
    for _, chunk in flag[flag].groupby(group_id[flag]):
        if len(chunk) < min_obs:
            continue
        start, end = chunk.index.min(), chunk.index.max()
        row = {
            "start": start,
            "end": end,
            "n_obs": len(chunk),
            "duration_days": (end - start).days + 1 if hasattr(end - start, "days") else np.nan,
            "depth_mean": np.nan,
            "depth_min": np.nan,
            "depth_max": np.nan,
        }
        if depth is not None:
            window = depth.loc[chunk.index].dropna()
            if len(window):
                row["depth_mean"] = float(window.mean())
                row["depth_min"] = float(window.min())
                row["depth_max"] = float(window.max())
        rows.append(row)

    return pd.DataFrame(rows, columns=columns)


# ===========================================================================
# Proportion : intervalle exact de Clopper-Pearson
# ===========================================================================
@dataclass(frozen=True)
class ProportionCI:
    """IC binomial exact. Le seul honnête quand n est petit ou la proportion extrême."""

    successes: int
    n: float
    point: float
    lo: float
    hi: float
    confidence: float

    @property
    def summary(self) -> str:
        return (
            f"{self.point:.1%} [IC {self.confidence:.0%} exact : {self.lo:.1%}, "
            f"{self.hi:.1%}] sur n = {self.n:g}"
        )


def clopper_pearson(
    successes: int, n: float, *, confidence: float = 0.95
) -> ProportionCI:
    """IC exact d'une proportion — sans approximation normale.

    Deux usages dans ce portefeuille, et le second est le plus important :
    le `sign_flip_rate` de T1-1, et **le win rate d'un backtest évalué sur n_eff**.
    Sur un n effectif de 7, un win rate de 100 % a une borne basse très éloignée de
    100 % : c'est ce chiffre-là qui va dans le mail.

    `n` accepte un flottant parce qu'un n effectif n'est pas entier. L'IC est alors
    approché en arrondissant à l'entier inférieur — conservateur, donc dans le bon sens.
    """
    n_int = int(np.floor(n))
    if n_int < 1:
        raise StatsError(f"n doit être >= 1, reçu {n}")
    successes = int(min(successes, n_int))
    if successes < 0:
        raise StatsError(f"successes doit être >= 0, reçu {successes}")

    alpha = 1.0 - confidence
    lo = 0.0 if successes == 0 else float(scipy_stats.beta.ppf(alpha / 2, successes, n_int - successes + 1))
    hi = 1.0 if successes == n_int else float(scipy_stats.beta.ppf(1 - alpha / 2, successes + 1, n_int - successes))
    return ProportionCI(
        successes=successes,
        n=float(n),
        point=successes / n_int,
        lo=lo,
        hi=hi,
        confidence=confidence,
    )
