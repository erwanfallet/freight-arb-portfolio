"""Projet A — décomposition du premium 65-62 % Fe en une part fret et un résidu.

THÈSE
-----
Les indices minerai de fer 62 % Fe et 65 % Fe sont tous deux cotés **CFR Chine**, donc le
fret est déjà à l'intérieur des deux prix. Le 65 % est essentiellement brésilien
(Tubarão -> Qingdao, ~11 000 nm, route Baltic C3), le 62 % essentiellement australien
(Port Hedland -> Qingdao, ~1 600 nm, route C5). Le premium observé contient donc
mécaniquement le différentiel de fret C3 - C5.

    premium_observé = P65_CFR - P62_CFR
    fair_value_fret = C3_par_dmt - C5_par_dmt
    résidu          = premium_observé - fair_value_fret

Le résidu n'est PAS « de la tension physique ». C'est « qualité + valeur-en-usage +
tension + base FFA-vs-index ». On refuse de le baptiser autrement : aucune donnée
publique ne permet de séparer ces quatre termes, et un modèle de valeur-en-usage inventé
transformerait le résultat en artefact.

LE PIÈGE D'UNITÉ, QUI EST LE CŒUR TECHNIQUE
-------------------------------------------
Les indices minerai sont cotés en USD par tonne métrique **sèche** (dmt). Le fret se paie
sur le poids **embarqué**, c'est-à-dire humide (wmt, poids du connaissement). Donc :

    fret_par_dmt = fret_par_wmt / (1 - humidité)

Les fines brésiliennes sortent autour de 9 % d'humidité, le Pilbara Blend autour de 8 %.
Ignorer la correction sous-estime systématiquement la part fret du premium.

C'est le même mouvement que `TC_par_t_zinc = TC_par_dmt_conc / (grade × recovery)` :
l'unité de cotation n'est pas l'unité économique.

HYPOTHÈSES (toutes explicites, toutes paramétrées)
--------------------------------------------------
A-H1  Le 65 % CFR est essentiellement brésilien, le 62 % essentiellement australien.
      Le 62 % contient en réalité aussi du brésilien et de l'indien -> la part fret est
      SOUS-estimée. Biais conservateur, donc dans le bon sens.
A-H2  Humidité : 9,0 % Brésil, 8,0 % Australie. Paramétrées, balayées en sensibilité.
A-H2b Le fret C3/C5 est payé sur le poids humide. Vrai pour un affrètement au voyage
      standard sur poids embarqué ; à revérifier si une contrepartie paie sur base sèche.
A-H3  Le FFA front-month est un proxy de la route spot. Il existe une base. Elle est
      affichée comme avertissement de données, jamais masquée.
A-H4  Aucune prime de valeur-en-usage modélisée (posée à zéro). Le résidu l'absorbe.
A-H5  Pas de coût de portage sur l'écart de ~25 jours de temps de voyage. Chiffré à part
      dans la section sensibilité, pas dans l'identité principale.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Valeurs par défaut des hypothèses A-H2. Documentées, jamais codées en dur ailleurs.
DEFAULT_MOISTURE_BRAZIL = 0.090
DEFAULT_MOISTURE_AUSTRALIA = 0.080


def freight_per_dry_tonne(freight_per_wet_tonne: pd.Series | float, moisture: float) -> pd.Series | float:
    """Convertit un fret payé sur tonne humide en coût par tonne sèche (A-H2b).

    moisture est une fraction (0.09 = 9 %), pas un pourcentage.
    """
    if not 0.0 <= moisture < 1.0:
        raise ValueError(f"moisture doit être dans [0, 1), reçu {moisture}")
    return freight_per_wet_tonne / (1.0 - moisture)


def decompose_premium(
    p65: pd.Series,
    p62: pd.Series,
    c3: pd.Series,
    c5: pd.Series,
    *,
    moisture_brazil: float = DEFAULT_MOISTURE_BRAZIL,
    moisture_australia: float = DEFAULT_MOISTURE_AUSTRALIA,
) -> pd.DataFrame:
    """Décompose le premium 65-62 en part fret et résidu.

    Toutes les séries sont indexées par date. Les prix sont en USD/dmt, les frets en
    USD/wmt. L'alignement de calendrier est fait ici par intersection : aucun
    forward-fill, un trou reste un trou (règle du contrat de données).

    Retourne un DataFrame avec, par date :
        p65, p62, premium_observed,
        c3_wmt, c5_wmt, c3_dmt, c5_dmt,
        freight_fair_value, residual, freight_share,
        premium_naive_freight  (le différentiel non corrigé de l'humidité, pour montrer
                                exactement ce que la correction change)
    """
    aligned = pd.concat(
        {"p65": p65, "p62": p62, "c3_wmt": c3, "c5_wmt": c5}, axis=1
    ).dropna()
    if aligned.empty:
        raise ValueError(
            "aucune date commune aux quatre séries — vérifier les calendriers avant "
            "d'aller plus loin, ne pas combler les trous"
        )
    aligned = aligned.sort_index()

    out = aligned.copy()
    out["c3_dmt"] = freight_per_dry_tonne(aligned["c3_wmt"], moisture_brazil)
    out["c5_dmt"] = freight_per_dry_tonne(aligned["c5_wmt"], moisture_australia)
    out["premium_observed"] = aligned["p65"] - aligned["p62"]
    out["freight_fair_value"] = out["c3_dmt"] - out["c5_dmt"]
    out["residual"] = out["premium_observed"] - out["freight_fair_value"]
    out["premium_naive_freight"] = aligned["c3_wmt"] - aligned["c5_wmt"]

    # part fret du premium : non définie quand le premium est nul, et trompeuse quand il
    # est négatif -> on la laisse en NaN plutôt que de produire un pourcentage absurde.
    share = out["freight_fair_value"] / out["premium_observed"]
    share[out["premium_observed"] <= 0] = np.nan
    out["freight_share"] = share

    return out


@dataclass(frozen=True)
class ExplainedVariance:
    """Résultat de la régression premium ~ a + b * fair_value_fret."""

    slope: float
    intercept: float
    r_squared: float
    correlation: float
    n_obs: int

    @property
    def summary(self) -> str:
        return (
            f"premium = {self.intercept:.2f} + {self.slope:.2f} × fret  "
            f"(R² = {self.r_squared:.3f}, rho = {self.correlation:.3f}, n = {self.n_obs})"
        )


def explained_variance(
    premium: pd.Series, freight_fair_value: pd.Series, *, on_changes: bool = False
) -> ExplainedVariance:
    """Part de la variance du premium expliquée par la part fret.

    on_changes=True régresse les variations plutôt que les niveaux. À privilégier pour
    la lecture statistique : deux séries en niveau non stationnaires produisent un R²
    flatteur qui ne veut rien dire. Les deux sont exposés parce que le niveau parle au
    trader et la variation parle à l'économètre — et l'écart entre les deux est
    lui-même une information à afficher.
    """
    aligned = pd.concat({"y": premium, "x": freight_fair_value}, axis=1).dropna()
    if on_changes:
        aligned = aligned.diff().dropna()
    if len(aligned) < 3:
        raise ValueError(f"pas assez d'observations pour régresser (n={len(aligned)})")

    x = aligned["x"].to_numpy(dtype=float)
    y = aligned["y"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    corr = float(np.corrcoef(x, y)[0, 1])
    return ExplainedVariance(
        slope=float(slope),
        intercept=float(intercept),
        r_squared=r_squared,
        correlation=corr,
        n_obs=len(aligned),
    )


def negative_residual_episodes(
    decomposition: pd.DataFrame, *, min_days: int = 5
) -> pd.DataFrame:
    """Épisodes où le résidu est négatif : le premium haute teneur est inférieur au seul
    surcoût de distance. Autrement dit, la qualité est vendue gratuitement ou à perte.

    C'est l'anomalie qui porte la conversation. min_days filtre le bruit d'un jour.

    Retourne un DataFrame : start, end, n_obs, residual_min, residual_mean.
    """
    flag = decomposition["residual"] < 0
    if not flag.any():
        return pd.DataFrame(
            columns=["start", "end", "n_obs", "residual_min", "residual_mean"]
        )

    # regroupe les True consécutifs
    group_id = (flag != flag.shift()).cumsum()
    rows = []
    for _, chunk in decomposition[flag].groupby(group_id[flag]):
        if len(chunk) < min_days:
            continue
        rows.append(
            {
                "start": chunk.index.min(),
                "end": chunk.index.max(),
                "n_obs": len(chunk),
                "residual_min": float(chunk["residual"].min()),
                "residual_mean": float(chunk["residual"].mean()),
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class HedgeResult:
    """Effet d'une couverture de la jambe fret sur la volatilité du trade de premium."""

    beta: float
    vol_unhedged: float
    vol_hedged: float
    n_obs: int

    @property
    def vol_reduction_pct(self) -> float:
        if self.vol_unhedged == 0:
            return float("nan")
        return 100.0 * (1.0 - self.vol_hedged / self.vol_unhedged)


def freight_hedge_effect(
    premium: pd.Series,
    freight_fair_value: pd.Series,
    *,
    beta: float | None = None,
) -> HedgeResult:
    """Volatilité des variations quotidiennes du premium, avant et après couverture de la
    part fret par des FFA C3/C5.

    beta=None estime le ratio de couverture par MCO sur les variations (le ratio de
    variance minimale). beta=1.0 force la couverture unitaire, qui est ce qu'un desk
    ferait naïvement — l'écart entre les deux est en soi un résultat.

    Ce que ce calcul NE dit PAS : que le hedge est exécutable. Les FFA C3/C5 sont des
    contrats mensuels sur moyenne de route ; couvrir une exposition quotidienne avec ça
    laisse une base non triviale. À afficher comme caveat.
    """
    aligned = pd.concat({"prem": premium, "fret": freight_fair_value}, axis=1).dropna()
    changes = aligned.diff().dropna()
    if len(changes) < 3:
        raise ValueError(f"pas assez d'observations (n={len(changes)})")

    if beta is None:
        var_f = float(changes["fret"].var())
        if var_f == 0:
            raise ValueError("la part fret ne varie pas — beta indéterminé")
        beta = float(changes[["prem", "fret"]].cov().iloc[0, 1] / var_f)

    residual_changes = changes["prem"] - beta * changes["fret"]
    return HedgeResult(
        beta=float(beta),
        vol_unhedged=float(changes["prem"].std()),
        vol_hedged=float(residual_changes.std()),
        n_obs=len(changes),
    )


def carry_cost_of_extra_voyage_days(
    cargo_value_per_dmt: pd.Series | float,
    extra_days: float,
    annual_rate: float,
) -> pd.Series | float:
    """A-H5 chiffrée : coût de portage des jours de voyage supplémentaires du Brésil.

    Tubarão -> Qingdao contre Port Hedland -> Qingdao, c'est de l'ordre de 25 jours de
    mer en plus. Sur une cargaison à 100 $/dmt et 6 % annuel, ça fait ~0,41 $/dmt : petit
    devant un différentiel de fret de 10-14 $/dmt, mais pas nul devant un résidu de
    2-3 $/dmt. Le montrer, c'est refuser de faire semblant que le terme n'existe pas.
    """
    if extra_days < 0:
        raise ValueError("extra_days doit être positif")
    return cargo_value_per_dmt * annual_rate * (extra_days / 365.0)
