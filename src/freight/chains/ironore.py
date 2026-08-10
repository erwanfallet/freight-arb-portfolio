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


# ===========================================================================
# REAL DATA — the shorthand tested, and found arithmetically impossible
# ===========================================================================
# The whole freight-spread story rests on a shorthand: 65 % Fe is Brazilian and travels on
# route C3, 62 % Fe is Australian and travels on C5, so the C3 - C5 differential sits inside
# the CFR premium. The export now carries all four series. Testing the shorthand at full
# strength turns out to break it — and the moisture correction, which is this project's
# signature unit trap, makes the contradiction worse rather than resolving it.
IODEX_62_SHEET = "SGX IODEX (61%) Iron Ore Future"
MB_65_SHEET = "SGX MB IronOre 65 Sep26 Comdty"
C3_SHEET = "SGX Baltic C3 Futures FSP Index"
C5_SHEET = "C5 FFA USD MT M1 Index"


def _read_sheet(sheet: str) -> pd.Series:
    """Read one raw sheet of the Bloomberg export.

    These four series are not in `agri.data.bloomberg_loader` because they belong to the
    freight side of the portfolio, and because two of them carry defects the loader's
    contract would have to describe one by one — see `load_real_premium_frame`.
    """
    from agri.data.bloomberg_loader import DEFAULT_PATH

    raw = pd.read_excel(DEFAULT_PATH, sheet_name=sheet, header=None)
    values = pd.to_numeric(raw.iloc[:, 1], errors="coerce")
    dates = pd.to_datetime(raw.iloc[:, 0], errors="coerce", format="mixed")
    return pd.Series(values.values, index=dates).dropna().sort_index()


def load_real_premium_frame() -> pd.DataFrame:
    """The four legs on a common **monthly** grid, and the reason it has to be monthly.

    Columns: io65, io62, c3, c5, premium, freight_spread.

    FREQUENCY, AND WHY IT IS NOT A DETAIL. C5 is daily with 3 130 observations; the C3 route
    in this export has **64 points at a monthly step**. Forward-filling C3 onto a daily grid
    would create 600 observations out of 31 real ones and make every standard error in the
    page meaningless — the portfolio's own resampling rules forbid exactly that. So both are
    taken down to the monthly grid the coarser series actually supports. Thirty-one months
    is a small sample and the page says so rather than manufacturing a large one.
    """
    monthly = {
        name: _read_sheet(sheet).resample("ME").median()
        for name, sheet in (
            ("io65", MB_65_SHEET),
            ("io62", IODEX_62_SHEET),
            ("c3", C3_SHEET),
            ("c5", C5_SHEET),
        )
    }
    frame = pd.concat(monthly, axis=1, sort=True).dropna()
    if frame.empty:
        raise ValueError("no common month across the four iron ore legs")
    frame["premium"] = frame["io65"] - frame["io62"]
    frame["freight_spread"] = frame["c3"] - frame["c5"]
    return frame


@dataclass(frozen=True)
class ShorthandTest:
    """What the origin shorthand implies about the FOB quality differential.

    Both indices are quoted CFR China, so

        premium_CFR = (FOB_65 - FOB_62) + (C3 - C5)

    Taken at full strength, the shorthand therefore pins the FOB quality differential to
    `premium - freight_spread`. If that comes out negative, high-grade ore would be cheaper
    at the loadport than low-grade ore — which nobody believes, and which is the point.
    """

    frame: pd.DataFrame
    moisture: float

    @property
    def implied_quality_median(self) -> float:
        return float(self.frame["implied_quality"].median())

    @property
    def share_negative(self) -> float:
        return float((self.frame["implied_quality"] < 0).mean())

    @property
    def freight_share_of_premium(self) -> float:
        return float(
            self.frame["freight_dry"].median() / self.frame["premium"].median()
        )

    @property
    def shorthand_survives(self) -> bool:
        """The shorthand only survives if it implies a non-negative quality differential
        most of the time."""
        return self.share_negative < 0.5

    @property
    def headline(self) -> str:
        return (
            f"At {self.moisture:.0%} moisture, the freight differential covers "
            f"{self.freight_share_of_premium:.0%} of the 65-62 premium, which leaves an "
            f"implied FOB quality differential of {self.implied_quality_median:+.2f} USD/t "
            f"— negative in {self.share_negative:.0%} of months. Taken at full strength the "
            "origin shorthand would make high-grade ore cheaper at the loadport than "
            "low-grade ore. It cannot be right as stated."
        )


def evaluate_origin_shorthand(
    frame: pd.DataFrame, *, moisture: float = DEFAULT_MOISTURE_BRAZIL
) -> ShorthandTest:
    """Apply the shorthand at full strength and look at what it implies.

    The moisture correction is applied to the freight leg and **only** to the freight leg:
    freight is paid on the wet tonne shipped, the index is quoted on the dry tonne
    delivered, so one dry tonne costs `freight / (1 - moisture)` to move. This is the
    correction the project exists to make — and here it deepens the contradiction instead of
    resolving it, which is worth more than a correction that had tidied things up.
    """
    if not 0.0 <= moisture < 0.30:
        raise ValueError(f"moisture outside the plausible range [0, 0.30): {moisture}")

    out = frame.copy()
    out["freight_dry"] = out["freight_spread"] / (1.0 - moisture)
    out["implied_quality"] = out["premium"] - out["freight_dry"]
    return ShorthandTest(frame=out, moisture=float(moisture))


@dataclass(frozen=True)
class ImpliedOriginWeight:
    """The inversion: how much of the freight spread the premium can actually carry.

    Rather than asserting the shorthand and finding a contradiction, assume the FOB quality
    differential a practitioner believes in and solve for the weight at which the freight
    spread enters:

        premium = quality_FOB + w x freight_dry     =>     w = (premium - quality) / freight_dry

    `w = 1` is the full-strength shorthand. `w = 0` is a pure quality premium with no freight
    content. What comes out is neither — and an iron ore desk reads its own answer straight
    off the curve, because it knows its FOB differential.
    """

    curve: pd.DataFrame
    moisture: float

    def weight_at(self, quality_usd_t: float) -> float:
        row = self.curve.iloc[(self.curve["quality_usd_t"] - quality_usd_t).abs().argmin()]
        return float(row["implied_weight"])

    @property
    def headline(self) -> str:
        return (
            f"Assuming a 6 USD/t FOB quality differential, the freight spread enters the "
            f"premium at a weight of {self.weight_at(6.0):.2f} rather than 1 — the two "
            "indices are not single-origin, and the shorthand overstates the freight content "
            "by roughly a factor of two."
        )


def implied_origin_weight(
    frame: pd.DataFrame,
    *,
    moisture: float = DEFAULT_MOISTURE_BRAZIL,
    quality_grid: np.ndarray | None = None,
) -> ImpliedOriginWeight:
    """The weight at which the freight spread enters, as a function of assumed FOB quality."""
    tested = evaluate_origin_shorthand(frame, moisture=moisture)
    grid = np.arange(0.0, 14.1, 0.5) if quality_grid is None else np.asarray(quality_grid)

    rows = []
    for quality in grid:
        weight = (tested.frame["premium"] - quality) / tested.frame["freight_dry"]
        rows.append(
            {
                "quality_usd_t": float(quality),
                "implied_weight": float(weight.median()),
                "weight_low": float(weight.quantile(0.10)),
                "weight_high": float(weight.quantile(0.90)),
            }
        )
    return ImpliedOriginWeight(curve=pd.DataFrame(rows), moisture=float(moisture))
