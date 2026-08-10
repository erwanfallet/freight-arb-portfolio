"""T2-6 — Substitution inter-huiles.

TENSION — INFÉRÉE, PAS SOURCÉE
-------------------------------
**Il me semble que** les triturateurs tiennent l'élasticité palme/soja/colza/tournesol pour
forte, et les formulateurs pour collante — reformuler une recette prend des mois et repasse
par des validations. « Il me semble », jamais « j'ai lu que ».

L'IDÉE
------
Si la substitution était instantanée, les spreads inter-huiles reviendraient vite à la
moyenne dès qu'ils s'écartent. Si elle était impossible, ils dériveraient librement. La
réalité est entre les deux, **et elle dépend du niveau** : c'est mesurable.

    spread_ij = P_i - P_j            (toutes les huiles ramenées en USD/t)
    dspread_t = a + b spread_{t-1} + e
    demi-vie  = -ln(2) / ln(1 + b)

**Estimer la demi-vie par régime** (spread large / spread étroit) donne les bornes
implicites de substitution : les niveaux au-delà desquels la demi-vie s'effondre, c'est-à-
dire au-delà desquels quelqu'un bascule vraiment.

POINT DE BASCULE
----------------
Le niveau de spread où la demi-vie chute — la borne de substitution. C'est un chiffre en
USD/t qu'un triturateur confirme ou dément immédiatement, parce que c'est le niveau auquel
son téléphone sonne.

HYPOTHÈSES
----------
S-H1  Toutes les huiles sont ramenées en USD/t avant tout calcul. Palme en MYR/t, soja en
      cents/lb : mélanger les unités ici produirait des spreads dénués de sens.
S-H2  Le seuil séparant « spread large » de « spread étroit » est un quantile de la
      distribution historique, pas un niveau absolu — les niveaux dérivent avec l'inflation
      et le niveau général des prix.
S-H3  Aucun coût de transport ni de qualité n'est modélisé dans le spread. Ils décalent le
      niveau d'équilibre mais pas la vitesse de retour, qui est l'objet du test.
S-H4  La demi-vie est estimée sur un AR(1) simple. Un modèle à seuil (TAR) serait plus
      juste ; l'estimation par régime en est une approximation lisible et robuste.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from agri.core.stats import adf_kpss, hac_ols

DEFAULT_WIDE_QUANTILE = 0.75          # S-H2


class SubstitutionError(ValueError):
    """Modèle mal spécifié."""


@dataclass(frozen=True)
class HalfLife:
    """Vitesse de retour à la moyenne d'un spread, et sa lisibilité."""

    beta: float
    half_life_days: float
    pvalue: float
    n_obs: int
    label: str = ""

    @property
    def is_mean_reverting(self) -> bool:
        """Retour à la moyenne significatif : beta doit être négatif ET significatif."""
        return self.beta < 0 and self.pvalue < 0.05

    @property
    def summary(self) -> str:
        if not self.is_mean_reverting:
            return (
                f"{self.label}: pas de retour à la moyenne détectable "
                f"(beta = {self.beta:+.4f}, p = {self.pvalue:.3f}, n = {self.n_obs})"
            )
        return (
            f"{self.label}: demi-vie {self.half_life_days:.0f} jours "
            f"(beta = {self.beta:+.4f}, p = {self.pvalue:.3f}, n = {self.n_obs})"
        )


def estimate_half_life(
    spread: pd.Series, *, mask: pd.Series | None = None, label: str = ""
) -> HalfLife:
    """Demi-vie de retour à la moyenne, via `dspread = a + b spread_{t-1} + e`, erreurs HAC.

        demi-vie = -ln(2) / ln(1 + b)

    Un `b` positif ou nul signifie qu'il n'y a pas de retour à la moyenne : la demi-vie est
    alors infinie, et on renvoie l'infini plutôt qu'un nombre trompeur.

    `mask` restreint l'estimation à un régime. **Les décalages sont calculés sur la série
    complète, puis filtrés** — jamais l'inverse. Filtrer d'abord produirait des
    différences entre observations non adjacentes dans le temps, ce qui fabrique une
    fausse moyenne-réversion : deux points distants de trois semaines paraissent avoir
    « convergé » en un pas. Sur le jeu de test, l'erreur ramenait une demi-vie de 173
    jours à 10.
    """
    clean = pd.Series(spread).dropna().astype(float)
    if len(clean) < 60:
        raise SubstitutionError(f"au moins 60 observations sont nécessaires, reçu {len(clean)}")

    frame = pd.concat({"d": clean.diff(), "lag": clean.shift(1)}, axis=1).dropna()
    if mask is not None:
        frame = frame[pd.Series(mask).reindex(frame.index).fillna(False).astype(bool)]
    if len(frame) < 40:
        raise SubstitutionError(
            f"régime trop court pour estimer une demi-vie : n={len(frame)}"
        )
    regression = hac_ols(frame["d"], frame[["lag"]])
    beta = float(regression.params["lag"])

    if beta >= 0 or (1.0 + beta) <= 0:
        half_life = float("inf")
    else:
        half_life = -np.log(2.0) / np.log(1.0 + beta)

    return HalfLife(
        beta=beta,
        half_life_days=half_life,
        pvalue=float(regression.pvalues["lag"]),
        n_obs=len(frame),
        label=label,
    )


def build_spreads(prices_usd_t: dict[str, pd.Series]) -> pd.DataFrame:
    """Tous les spreads deux à deux, toutes les huiles déjà en USD/t (S-H1).

    Les colonnes sont nommées `huile_a_moins_huile_b`, dans l'ordre alphabétique des
    couples pour qu'un spread n'apparaisse jamais deux fois avec des signes opposés.
    """
    if len(prices_usd_t) < 2:
        raise SubstitutionError("au moins deux huiles sont nécessaires")
    frame = pd.concat(prices_usd_t, axis=1).dropna()
    if frame.empty:
        raise SubstitutionError("aucune date commune aux séries d'huiles")

    names = sorted(prices_usd_t)
    out = pd.DataFrame(index=frame.index)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            out[f"{first}_moins_{second}"] = frame[first] - frame[second]
    return out


@dataclass(frozen=True)
class SubstitutionBound:
    """La borne implicite de substitution, pour un couple d'huiles."""

    pair: str
    threshold_usd_t: float
    narrow: HalfLife
    wide: HalfLife

    @property
    def substitution_kicks_in(self) -> bool:
        """La demi-vie s'effondre-t-elle quand le spread s'écarte ?"""
        return (
            self.wide.is_mean_reverting
            and self.wide.half_life_days < self.narrow.half_life_days
        )

    @property
    def headline(self) -> str:
        if not self.substitution_kicks_in:
            return (
                f"{self.pair} : au-delà de {self.threshold_usd_t:.0f} $/t d'écart, le spread "
                "ne revient pas plus vite. Sur cet échantillon, je ne vois pas de borne de "
                "substitution — la thèse « collante » des formulateurs tient."
            )
        return (
            f"{self.pair} : sous {self.threshold_usd_t:.0f} $/t d'écart, le spread met "
            f"{self.narrow.half_life_days:.0f} jours à se résorber ; au-delà, "
            f"{self.wide.half_life_days:.0f} jours. La borne de substitution est là — "
            "c'est le niveau auquel quelqu'un bascule vraiment."
        )


def substitution_bound(
    spread: pd.Series,
    *,
    pair: str = "",
    wide_quantile: float = DEFAULT_WIDE_QUANTILE,
    threshold_usd_t: float | None = None,
) -> SubstitutionBound:
    """Demi-vie du spread par régime, et le seuil qui les sépare (S-H2, S-H4).

    Le régime est défini sur l'**écart absolu** au niveau médian, pas sur le niveau brut :
    un spread peut être large dans les deux sens, et la substitution joue dans les deux.

    Le régime est évalué sur la valeur **retardée** du spread, jamais sur la valeur
    courante : classer une observation d'après le niveau qu'elle atteint après le
    mouvement qu'on cherche à expliquer serait circulaire.

    `threshold_usd_t` force un seuil au lieu de le déduire d'un quantile — utile quand un
    praticien en propose un, ce qui est précisément la conversation qu'on cherche.
    """
    clean = pd.Series(spread).dropna().astype(float)
    if not 0.5 < wide_quantile < 1.0:
        raise SubstitutionError(f"wide_quantile doit être dans (0,5 ; 1), reçu {wide_quantile}")

    deviation = (clean.shift(1) - clean.median()).abs()
    threshold = (
        float(deviation.quantile(wide_quantile))
        if threshold_usd_t is None
        else float(threshold_usd_t)
    )
    is_wide = deviation > threshold

    return SubstitutionBound(
        pair=pair or (spread.name or "spread"),
        threshold_usd_t=threshold,
        narrow=estimate_half_life(clean, mask=~is_wide, label="spread étroit"),
        wide=estimate_half_life(clean, mask=is_wide, label="spread large"),
    )


def screen_all_pairs(
    spreads: pd.DataFrame, *, wide_quantile: float = DEFAULT_WIDE_QUANTILE
) -> pd.DataFrame:
    """Table de synthèse : demi-vie par régime pour chaque couple.

    Les couples dont le spread n'est pas stationnaire sont marqués et **exclus de la
    lecture** : une demi-vie estimée sur une série à racine unitaire est un nombre sans
    contenu, pas une mesure lente.
    """
    rows = []
    for column in spreads.columns:
        series = spreads[column].dropna()
        try:
            verdict = adf_kpss(series).verdict
        except Exception:
            verdict = "non testable"
        try:
            bound = substitution_bound(series, pair=column, wide_quantile=wide_quantile)
        except SubstitutionError as error:
            rows.append(
                {
                    "pair": column,
                    "stationarity": verdict,
                    "threshold_usd_t": np.nan,
                    "half_life_narrow": np.nan,
                    "half_life_wide": np.nan,
                    "substitution_kicks_in": False,
                    "note": str(error),
                }
            )
            continue
        rows.append(
            {
                "pair": column,
                "stationarity": verdict,
                "threshold_usd_t": bound.threshold_usd_t,
                "half_life_narrow": bound.narrow.half_life_days,
                "half_life_wide": bound.wide.half_life_days,
                "substitution_kicks_in": bound.substitution_kicks_in,
                "note": "" if verdict == "stationary" else "spread non stationnaire — ne pas lire la demi-vie",
            }
        )
    return pd.DataFrame(rows)


# ===========================================================================
# LA FENÊTRE DE PARITÉ FIXE — le seul test propre que la donnée autorise
# ===========================================================================
# L'export contient la palme (KO1) en **ringgits par tonne** et le soja (BO1) en cents par
# livre, mais **aucune série USDMYR**. Calculer un spread palme-soja reviendrait donc à
# soustraire deux devises — précisément l'erreur que ce portefeuille traque ailleurs. On
# refuse de le faire.
#
# Sauf sur une fenêtre : Bank Negara a arrimé le ringgit à 3,80 MYR/USD du 2 septembre 1998
# au 21 juillet 2005. Pendant ces sept ans, la série manquante est une **constante connue
# par décret**, et le spread se calcule exactement, sans aucune hypothèse de change. C'est
# une expérience naturelle : tout mouvement du spread y est de l'économie de substitution
# pure, non contaminée par le change.
MYR_PEG_RATE = 3.80
MYR_PEG_START = "1998-09-02"
MYR_PEG_END = "2005-07-21"
CENTS_LB_TO_USD_T = 22.0462


def load_peg_window_spread() -> pd.DataFrame:
    """Spread palme-soja en USD/tonne sur la fenêtre de parité fixe du ringgit.

    Colonnes : palm_myr, palm_usd, soy_usd, spread.

    La conversion de la palme est une **division par une constante réglementaire**, pas par
    une série de marché — c'est ce qui rend ce spread lisible sans hypothèse.
    """
    from agri.data.bloomberg_loader import load

    frame = pd.concat(
        {"palm_myr": load("palm_oil_myr"), "soy_c_lb": load("cbot_soyoil")},
        axis=1,
        sort=True,
    ).dropna()
    frame = frame[
        (frame.index >= pd.Timestamp(MYR_PEG_START)) & (frame.index <= pd.Timestamp(MYR_PEG_END))
    ]
    if frame.empty:
        raise SubstitutionError(
            "aucune donnée commune palme/soja sur la fenêtre de parité fixe "
            f"{MYR_PEG_START} — {MYR_PEG_END}"
        )
    frame["palm_usd"] = frame["palm_myr"] / MYR_PEG_RATE
    frame["soy_usd"] = frame["soy_c_lb"] * CENTS_LB_TO_USD_T
    frame["spread"] = frame["palm_usd"] - frame["soy_usd"]
    return frame[["palm_myr", "palm_usd", "soy_usd", "spread"]]


def rolling_deviation(spread: pd.Series, *, window: int = 250) -> pd.Series:
    """Écart du spread à sa médiane glissante.

    Indispensable ici : sur la fenêtre de parité fixe, le spread ne tourne pas autour d'un
    niveau stable — il **dérive** d'une prime de 67 USD/t à une décote de 170. Tester le
    retour à la moyenne contre une constante reviendrait à mesurer cette dérive et à
    l'appeler substitution. La médiane glissante isole les écarts au niveau courant de ce
    déplacement du niveau lui-même.
    """
    if window < 20:
        raise SubstitutionError(f"fenêtre glissante trop courte : {window}")
    reference = spread.rolling(window, min_periods=window // 2).median()
    return (spread - reference).dropna().rename("deviation")


@dataclass(frozen=True)
class SubstitutionVerdict:
    """Le test de substitution sur la fenêtre propre — et son résultat négatif.

    L'hypothèse prédisait : écart large -> quelqu'un bascule -> retour rapide. La donnée dit
    l'inverse. Les écarts **étroits** reviennent vite, les **larges** ne reviennent pas du
    tout. Lecture : les petits écarts sont du bruit autour d'un équilibre qui bouge
    lentement ; les grands écarts ne sont pas des dislocations, ce sont des déplacements de
    l'équilibre lui-même.
    """

    narrow: HalfLife
    wide: HalfLife
    threshold_usd_t: float
    window: int
    n_obs: int

    @property
    def substitution_band_exists(self) -> bool:
        """Vrai seulement si les écarts larges reviennent PLUS vite que les étroits, ce que
        la thèse prédit et que la donnée ne montre pas."""
        return (
            np.isfinite(self.wide.half_life_days)
            and np.isfinite(self.narrow.half_life_days)
            and self.wide.half_life_days < self.narrow.half_life_days
        )

    @property
    def headline(self) -> str:
        if self.substitution_band_exists:
            return (
                f"Au-delà de {self.threshold_usd_t:.0f} USD/t d'écart, le spread revient en "
                f"{self.wide.half_life_days:.0f} jours contre "
                f"{self.narrow.half_life_days:.0f} en régime étroit : une borne de "
                "substitution existe, et elle est à ce niveau."
            )
        return (
            f"Résultat contraire à la thèse. Les écarts étroits reviennent en "
            f"{self.narrow.half_life_days:.0f} jours, mais au-delà de "
            f"{self.threshold_usd_t:.0f} USD/t **aucun retour à la moyenne n'est détectable** "
            "— les grands écarts ne se referment pas, ils déplacent le niveau. Fader un "
            "spread palme-soja large n'a aucun support dans la seule fenêtre où le test est "
            "propre."
        )


def substitution_verdict(
    spread: pd.Series, *, window: int = 250, quantile: float = 0.70
) -> SubstitutionVerdict:
    """Compare le retour à la moyenne des écarts larges et étroits, référence mobile.

    PIÈGE ÉVITÉ, ET IL EST CELUI DE CE MODULE : les décalages sont calculés sur la série
    **complète** puis masqués, jamais l'inverse. Filtrer d'abord un sous-échantillon non
    contigu puis appliquer `.diff()` fabriquerait du retour à la moyenne à partir des
    ruptures de calendrier — c'est ce que fait `estimate_half_life` via son paramètre `mask`.

    BIAIS RÉSIDUEL, DANS LE BON SENS : conditionner sur |écart| grand suréchantillonne le
    bruit de mesure, qui revient mécaniquement. Ce biais pousse donc vers la DÉTECTION d'un
    retour à la moyenne. Ne pas en trouver malgré lui rend le résultat négatif plus solide,
    pas moins.
    """
    if not 0.5 < quantile < 1.0:
        raise SubstitutionError(f"quantile de séparation implausible : {quantile}")

    deviation = rolling_deviation(spread, window=window)
    threshold = float(deviation.abs().quantile(quantile))
    return SubstitutionVerdict(
        narrow=estimate_half_life(
            deviation, mask=deviation.abs() < threshold, label="écart étroit"
        ),
        wide=estimate_half_life(
            deviation, mask=deviation.abs() >= threshold, label="écart large"
        ),
        threshold_usd_t=threshold,
        window=window,
        n_obs=len(deviation),
    )


def structural_drift(spread: pd.Series) -> pd.DataFrame:
    """La dérive annuelle du spread — ce qui interdit de le traiter comme stationnaire.

    Renvoie la médiane par année, plus l'amplitude totale. Sur la fenêtre de parité fixe, la
    palme passe d'une prime au soja à une décote de plusieurs dizaines de dollars : ce n'est
    pas une oscillation autour d'un équilibre, c'est une repricing structurelle.
    """
    annual = spread.groupby(spread.index.year).median().rename("median_spread")
    frame = annual.to_frame()
    frame["n_obs"] = spread.groupby(spread.index.year).size()
    frame.attrs["drift_usd_t"] = float(annual.iloc[-1] - annual.iloc[0])
    frame.attrs["range_usd_t"] = float(annual.max() - annual.min())
    return frame


__all__ = [
    "CENTS_LB_TO_USD_T",
    "HalfLife",
    "MYR_PEG_END",
    "MYR_PEG_RATE",
    "MYR_PEG_START",
    "SubstitutionBound",
    "SubstitutionError",
    "SubstitutionVerdict",
    "build_spreads",
    "estimate_half_life",
    "load_peg_window_spread",
    "rolling_deviation",
    "screen_all_pairs",
    "structural_drift",
    "substitution_bound",
    "substitution_verdict",
]
