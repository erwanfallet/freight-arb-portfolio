"""Les trois règles qui décident de la validité d'un test.

Ce module ne calcule presque rien. Il rend **exécutables** trois règles qui, écrites en
prose dans une note de méthode, sont systématiquement oubliées au bout de trois semaines.

RÈGLE A — on descend, jamais on ne monte
----------------------------------------
Pour tout test statistique (corrélation, CCF, régression), si la série la plus lente est
mensuelle, le test entier est mensuel. La série rapide est agrégée ; l'inverse — étirer
la lente sur le calendrier de la rapide — fabrique des observations qui n'existent pas et
biaise toute autocorrélation. C'est le diagnostic déjà posé sur la page cuivre : le
`.ffill()` du Yangshan et des stocks SHFE avait détruit la cross-corrélation.

RÈGLE B — le forward-fill est légitime en P&L, jamais en test
-------------------------------------------------------------
Un smelter paie bien un TC mensuel tous les jours : reconstruire un P&L quotidien à
partir d'une série mensuelle est correct. Mais le remplissage doit être **visible**, d'où
les colonnes jumelles `is_true_print` et `staleness_days`.

Cette règle est ici rendue impossible à violer par accident : `ffill_with_provenance`
renvoie un DataFrame à trois colonnes qu'on ne peut pas passer tel quel à un test. Pour
en extraire des valeurs, il faut choisir explicitement `for_pnl()` — qui accepte — ou
`for_test()` — qui **lève** dès qu'une seule valeur est remplie. La règle n'est plus une
consigne, c'est une exception.

Note sur `ingest/contract.py` : ce module-là interdit toute politique de trou autre que
`none`, parce qu'il garde la frontière d'**ingestion** (une série arrive telle quelle, un
trou reste un trou, c'est la leçon du BSI périmé). Le forward-fill de la Règle B est une
opération de **modélisation**, en aval du contrat, sur une série déjà validée. Les deux
règles ne se contredisent pas : elles gardent deux frontières différentes.

RÈGLE C — n effectif
--------------------
Une fenêtre glissante de 30 jours sur données quotidiennes ne donne pas n observations
indépendantes. Toute statistique calculée sur des fenêtres qui se chevauchent doit
reporter `n_eff`, et c'est `n_eff` — pas `n_obs` — qui entre dans une bande de
significativité ou un intervalle de confiance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Fréquences : ordonnées de la plus rapide à la plus lente
# ---------------------------------------------------------------------------
FREQ_ORDER: dict[str, int] = {
    "daily": 0,
    "weekly": 1,
    "fortnightly": 2,   # quinzaine UNICA, CEPEA
    "monthly": 3,
    "quarterly": 4,
    "yearly": 5,
}

# Alias pandas 3.x correspondants (les alias 'M'/'Q'/'A' ont été retirés)
_PANDAS_RULE: dict[str, str] = {
    "daily": "D",
    "weekly": "W",
    "fortnightly": "SME",
    "monthly": "ME",
    "quarterly": "QE",
    "yearly": "YE",
}

VALID_AGGREGATIONS = ("last", "mean")


class StaleDataInTest(Exception):
    """Une série forward-fillée a tenté d'entrer dans un test statistique (Règle B)."""


class ResampleError(ValueError):
    """Rééchantillonnage mal spécifié — toujours une erreur d'appelant."""


# ---------------------------------------------------------------------------
# RÈGLE A
# ---------------------------------------------------------------------------
def infer_frequency(series: pd.Series) -> str:
    """Fréquence observée d'une série, déduite de l'écart médian entre observations.

    On prend la **médiane** et pas le mode ni le minimum : une série quotidienne a des
    trous de week-end et de fériés, une série mensuelle a des mois à 28 et 31 jours. La
    médiane traverse les deux sans se laisser attraper.
    """
    index = pd.DatetimeIndex(series.dropna().index)
    if len(index) < 3:
        raise ResampleError(
            f"au moins 3 observations sont nécessaires pour inférer une fréquence, "
            f"reçu {len(index)}"
        )
    # Arithmétique en secondes, jamais sur les entiers bruts de l'index : pandas 3 date
    # en microsecondes par défaut (datetime64[us]) là où pandas 2 datait en nanosecondes.
    # Diviser `.asi8` par une constante nanoseconde donne un écart 1000x trop petit,
    # classe toutes les séries en quotidien, et désarme silencieusement la Règle A.
    gaps_days = index.to_series().diff().dropna().dt.total_seconds() / 86_400.0
    median_gap = float(gaps_days.median())

    if median_gap <= 4:
        return "daily"
    if median_gap <= 10:
        return "weekly"
    if median_gap <= 20:
        return "fortnightly"
    if median_gap <= 45:
        return "monthly"
    if median_gap <= 135:
        return "quarterly"
    return "yearly"


def slowest_frequency(series_map: dict[str, pd.Series]) -> str:
    """La fréquence la plus lente du lot — celle qui commande le test entier."""
    if not series_map:
        raise ResampleError("aucune série fournie")
    return max(
        (infer_frequency(s) for s in series_map.values()),
        key=lambda f: FREQ_ORDER[f],
    )


def downsample(series: pd.Series, target_freq: str, *, how: str = "last") -> pd.Series:
    """Agrège une série vers une fréquence **plus lente ou égale**. Jamais l'inverse.

    `how='last'` prend le dernier print de la période (ce qu'un trader regarde),
    `how='mean'` la moyenne (ce qu'un économètre préfère pour un flux). Le choix est
    imposé à l'appelant parce qu'il change le résultat et n'a pas de bon défaut
    universel — il doit être documenté dans la page qui l'appelle.
    """
    if target_freq not in FREQ_ORDER:
        raise ResampleError(f"fréquence cible inconnue : {target_freq!r}")
    if how not in VALID_AGGREGATIONS:
        raise ResampleError(f"agrégation doit être dans {VALID_AGGREGATIONS}, reçu {how!r}")

    source_freq = infer_frequency(series)
    if FREQ_ORDER[source_freq] > FREQ_ORDER[target_freq]:
        raise ResampleError(
            f"RÈGLE A violée : on ne remonte pas une série {source_freq} vers du "
            f"{target_freq}. Descendre le test à la fréquence de la série la plus lente, "
            "ou renoncer au test."
        )
    # On rééchantillonne même quand la fréquence est déjà la bonne : c'est ce qui
    # normalise les étiquettes. Une série mensuelle datée au 1er et une série quotidienne
    # descendue en mensuel (datée en fin de mois) ont la même fréquence et zéro date
    # commune — l'intersection sortirait vide sans que rien ne le signale.
    resampled = series.dropna().sort_index().resample(_PANDAS_RULE[target_freq])
    aggregated = resampled.last() if how == "last" else resampled.mean()
    return aggregated.dropna()


def align_for_test(
    series_map: dict[str, pd.Series], *, how: str = "last"
) -> pd.DataFrame:
    """Aligne plusieurs séries pour un test statistique, en respectant la Règle A.

    Toutes les séries descendent à la fréquence de la plus lente, puis on intersecte les
    dates. Aucun trou n'est comblé : une date où une série manque disparaît du test.

    Le DataFrame renvoyé porte la fréquence retenue dans `.attrs["test_frequency"]`, à
    afficher dans la page — « la corrélation est calculée en mensuel parce que le grind
    est trimestriel » est une phrase qui doit apparaître à l'écran, pas rester dans le code.
    """
    target = slowest_frequency(series_map)
    aligned = pd.concat(
        {name: downsample(s, target, how=how) for name, s in series_map.items()},
        axis=1,
        sort=True,
    ).dropna()
    if aligned.empty:
        raise ResampleError(
            f"aucune date commune aux {len(series_map)} séries après passage en {target} — "
            "vérifier les calendriers avant d'aller plus loin, ne pas combler les trous"
        )
    aligned.attrs["test_frequency"] = target
    return aligned


# ---------------------------------------------------------------------------
# RÈGLE B
# ---------------------------------------------------------------------------
def ffill_with_provenance(
    series: pd.Series,
    target_index: pd.DatetimeIndex,
    *,
    max_staleness_days: int | None = None,
) -> pd.DataFrame:
    """Forward-fill traçable, pour reconstruction de P&L uniquement (Règle B).

    Renvoie trois colonnes :
        value             la valeur, remplie
        is_true_print     True si la date porte une observation réelle
        staleness_days    âge de l'observation portée, 0 sur un vrai print

    `max_staleness_days` remet en NaN au-delà d'un âge donné : un TC mensuel de 400 jours
    n'est plus une hypothèse conservatrice, c'est une donnée morte. Correspond à l'option
    « filtrer les jours périmés » de la sidebar.
    """
    target_index = pd.DatetimeIndex(target_index).sort_values()
    clean = series.dropna().sort_index()
    if clean.empty:
        raise ResampleError("série vide — rien à reporter")

    reindexed = clean.reindex(target_index)
    is_true_print = reindexed.notna()
    filled = reindexed.ffill()

    # âge : nombre de jours depuis le dernier vrai print à gauche de chaque date.
    # Avant le premier print, last_print reste NaT et l'âge sort en NaN — c'est le
    # comportement voulu : il n'y a rien à reporter, pas une valeur d'âge nulle.
    own_date = pd.Series(target_index, index=target_index)
    last_print = own_date.where(is_true_print.to_numpy()).ffill()
    staleness = (own_date - last_print).dt.days.astype(float)

    out = pd.DataFrame(
        {
            "value": filled,
            "is_true_print": is_true_print,
            "staleness_days": staleness,
        }
    )

    if max_staleness_days is not None:
        too_old = out["staleness_days"] > max_staleness_days
        out.loc[too_old, "value"] = np.nan

    out.attrs["is_ffilled"] = True
    return out


def for_pnl(filled: pd.DataFrame) -> pd.Series:
    """Extrait les valeurs d'un `ffill_with_provenance` **pour un calcul de P&L**.

    Autorisé par la Règle B. Le nom de la fonction est la documentation : si tu écris
    `for_pnl` dans un chemin de code qui alimente une régression, tu l'as fait exprès.
    """
    _check_provenance_frame(filled)
    return filled["value"]


def for_test(filled: pd.DataFrame) -> pd.Series:
    """Extrait les valeurs **pour un test statistique** — lève si quoi que ce soit est rempli.

    C'est le point où la Règle B cesse d'être une consigne. Le repli correct n'est pas de
    forcer : c'est `align_for_test`, qui descend le test à la fréquence de la série lente.
    """
    _check_provenance_frame(filled)
    live = filled[filled["value"].notna()]
    n_filled = int((~live["is_true_print"]).sum())
    if n_filled:
        share = 100.0 * n_filled / len(live)
        raise StaleDataInTest(
            f"RÈGLE B violée : {n_filled} des {len(live)} valeurs ({share:.0f} %) sont "
            "des forward-fill et ne peuvent pas entrer dans un test statistique. "
            "Utiliser align_for_test() pour descendre le test à la fréquence de la série "
            "la plus lente."
        )
    return live["value"]


def _check_provenance_frame(filled: pd.DataFrame) -> None:
    required = {"value", "is_true_print", "staleness_days"}
    missing = required - set(filled.columns)
    if missing:
        raise ResampleError(
            f"attendu un DataFrame produit par ffill_with_provenance (colonnes "
            f"{sorted(required)}), colonnes manquantes : {sorted(missing)}"
        )


# ---------------------------------------------------------------------------
# RÈGLE C
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EffectiveSample:
    """Taille d'échantillon corrigée du chevauchement, et ce qu'elle change."""

    n_obs: int
    overlap: float
    n_eff: float

    @property
    def shrinkage(self) -> float:
        """Facteur de réduction : 3.0 signifie « trois fois moins d'information »."""
        return self.n_obs / self.n_eff if self.n_eff > 0 else float("inf")

    @property
    def summary(self) -> str:
        return (
            f"n = {self.n_obs} observations, chevauchement moyen {self.overlap:.2f} "
            f"-> n_eff = {self.n_eff:.1f} (l'échantillon vaut {self.shrinkage:.1f}x moins "
            "que sa taille apparente)"
        )


def effective_n(n_obs: int, overlap: float) -> EffectiveSample:
    """n_eff = n_obs / chevauchement. La formule est triviale, l'oublier ne l'est pas."""
    if n_obs < 1:
        raise ResampleError(f"n_obs doit être >= 1, reçu {n_obs}")
    if overlap < 1:
        raise ResampleError(
            f"le chevauchement doit être >= 1 (1 = observations indépendantes), reçu {overlap}"
        )
    return EffectiveSample(n_obs=n_obs, overlap=float(overlap), n_eff=n_obs / overlap)


def effective_n_rolling(n_obs: int, window: int) -> EffectiveSample:
    """Cas d'une statistique en fenêtre glissante : le chevauchement est la fenêtre."""
    return effective_n(n_obs, window)


def average_concurrency(entry_dates: pd.DatetimeIndex, hold_days: int) -> float:
    """Nombre moyen de positions ouvertes en même temps, sur les jours où il y en a au moins une.

    Mesuré sur les vraies dates d'entrée, pas déduit d'un plafond théorique : un backtest
    à « 3 positions concurrentes maximum » tourne rarement à 3 en permanence, et prendre
    le maximum sous-estimerait `n_eff`, donc surestimerait la significativité — le biais
    est dans le mauvais sens, il faut le mesurer.
    """
    entries = pd.DatetimeIndex(entry_dates).sort_values()
    if len(entries) == 0:
        raise ResampleError("aucune date d'entrée")
    if hold_days < 1:
        raise ResampleError(f"hold_days doit être >= 1, reçu {hold_days}")

    span = pd.date_range(entries.min(), entries.max() + pd.Timedelta(days=hold_days - 1), freq="D")
    open_count = np.zeros(len(span), dtype=int)
    positions = span.get_indexer(entries)
    for start in positions:
        open_count[start : start + hold_days] += 1

    active = open_count[open_count > 0]
    return float(active.mean())


def effective_n_from_trades(
    entry_dates: pd.DatetimeIndex, hold_days: int
) -> EffectiveSample:
    """`n_eff` d'un backtest à positions chevauchantes — le cas du module T1-M.

    C'est ce calcul qu'il faut passer sur le backtest cuivre avant d'envoyer quoi que ce
    soit : 18 trades tenus 30 jours avec plusieurs positions simultanées ne valent pas
    18 tirages indépendants, et l'intervalle de confiance sur un win rate de 100 % s'en
    trouve changé du tout au tout.
    """
    overlap = average_concurrency(entry_dates, hold_days)
    return effective_n(len(pd.DatetimeIndex(entry_dates)), overlap)
