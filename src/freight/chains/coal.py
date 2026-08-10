"""Projet B — l'arb charbon Atlantique a perdu sa contrainte contraignante en 2022.

THÈSE
-----
L'arb classique de Richards Bay vers l'ARA :

    arb_ARA = API2 (CIF ARA) − API4 (FOB Richards Bay) − fret(C4) − financement − ETS

Le manuel dit que cet arb ne peut pas rester grand ouvert : la concurrence et le fret le
ramènent vers zéro. Depuis 2022 le charbon sud-africain part vers l'Inde plutôt que vers
l'Europe. **La cargaison marginale de Richards Bay n'est plus cotée sur Rotterdam.**
L'équation a donc perdu son terme contraignant, et il en manque un : le netback vers la
destination alternative.

Le prix CFR Inde est sous licence, donc on ne peut pas prouver l'équation en prix. On la
montre en **flux** : quand l'arb ARA se ferme, la part indienne des exports de Richards
Bay monte. C'est un résultat plus faible qu'une égalité de prix, et il faut le dire tel
quel plutôt que d'inventer une série.

LE RISQUE INTELLECTUEL, TRAITÉ DE FRONT
---------------------------------------
2022 est aussi l'année du choc gazier européen, qui a fait exploser la demande de charbon
en Europe. Un décrochage post-2022 attribué à l'Inde alors qu'il vient du gaz est une
erreur que la première personne compétente qui lit l'email verra. **Le TTF est donc un
contrôle obligatoire, pas un raffinement.** C'est la raison d'être de `ols()` avec
plusieurs régresseurs dans ce module : sans contrôle, le test ne vaut rien.

LES DEUX COUCHES TECHNIQUES
---------------------------
1. **Le pouvoir calorifique.** API2 et API4 sont tous deux des références 6 000 kcal/kg
   NAR : l'arb de référence est donc cohérent avec lui-même et neutre en CV **par
   construction**. Mais le charbon réellement chargé à Richards Bay a dérivé vers
   ~5 700-5 800 kcal. Or le fret se paie **à la tonne** et le charbon se vend **au
   kcal** : une cargaison à 5 700 kcal paie le même fret par tonne mais livre 5 % moins
   d'énergie. Le fret par tonne-équivalent-6 000 est donc plus cher de ~5 %.

   C'est le même mouvement que l'humidité sur le minerai de fer : l'unité de cotation
   n'est pas l'unité économique. Ici c'est une **sensibilité**, pas l'identité
   principale — l'arb de référence reste juste, il a simplement cessé de décrire la
   cargaison physique.

2. **L'ETS maritime européen.** Depuis 2024, un navire de plus de 5 000 GT doit restituer
   des quotas sur ses émissions. Pour un voyage dont une extrémité est hors UE — ce qui
   est le cas de Richards Bay → Rotterdam — la couverture est de 50 % des émissions du
   voyage, appliquée à un facteur de montée en charge de 40 % en 2024, 70 % en 2025 et
   100 % à partir de 2026. Soit une couverture effective de 20 %, 35 % puis 50 %.

   Conséquence : le fret vers l'Europe devient structurellement plus cher que le fret vers
   l'Inde **à distance égale**. Terme récent, chiffrable, et absent des modèles d'arb
   charbon publics.

   Piège d'unité supplémentaire : le quota est coté en **EUR**, l'arb est en **USD**. Il
   faut une série de change, et c'est un terme réel, pas un détail.

HYPOTHÈSES
----------
B-H1  API2 et API4 partagent la même référence 6 000 kcal/kg NAR. Solide.
B-H2  Le CV réellement exporté à Richards Bay est inférieur à la référence. Traité en
      sensibilité, valeur à sourcer dans les rapports annuels des producteurs.
B-H3  C4 (Capesize) est le fret pertinent RB → ARA. Une partie du flux voyage en
      Panamax/Supramax, à coût différent. Paramétré.
B-H4  Financement au taux annuel appliqué à la valeur FOB sur la durée de voyage.
B-H5  Facteur d'émission du combustible : 3,114 tCO2 par tonne de fuel (ordre de grandeur
      du facteur IMO pour le fuel lourd). **Paramétré, à confirmer** selon le combustible
      réellement soutéé — VLSFO et MGO ont des facteurs différents.
B-H6  Les paramètres réglementaires ETS (50 % de portée, montée en charge 40/70/100) sont
      **paramétrés et non codés en dur ailleurs**. À reconfirmer sur le texte avant
      publication.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from agri.data.snapshot import cached

BENCHMARK_CV_KCAL_PER_KG = 6000.0

# B-H6 — montée en charge de l'ETS maritime. Avant 2024 : aucune obligation.
# À partir de 2026 : 100 %. Paramétrable pour que le texte réglementaire reste la
# référence et le code une transcription.
EU_ETS_PHASE_IN: dict[int, float] = {2024: 0.40, 2025: 0.70}
EU_ETS_PHASE_IN_FROM_2026 = 1.00
# Portée pour un voyage dont une extrémité est hors UE.
EXTRA_EU_SCOPE_FACTOR = 0.50
# B-H5 — facteur d'émission, tCO2 par tonne de combustible.
DEFAULT_EMISSION_FACTOR = 3.114


# --------------------------------------------------------------- pouvoir calorifique
def to_energy_basis(
    value_per_tonne: pd.Series | float,
    cv_kcal_per_kg: float,
    cv_benchmark: float = BENCHMARK_CV_KCAL_PER_KG,
) -> pd.Series | float:
    """Exprime un montant par tonne sur la base énergétique de référence.

    Une cargaison à 5 700 kcal/kg livre 5 700/6 000 = 95 % de l'énergie d'une tonne de
    référence. Donc tout coût payé à la tonne — le fret au premier chef — vaut
    `× 6000/5700 = 1,0526` par tonne-équivalent-6 000.

    Fonctionne dans les deux sens : appliqué à un prix, il ramène le prix sur la base de
    référence ; appliqué à un fret, il donne le fret par unité d'énergie livrée.
    """
    if cv_kcal_per_kg <= 0:
        raise ValueError(f"cv_kcal_per_kg doit être > 0, reçu {cv_kcal_per_kg}")
    if cv_benchmark <= 0:
        raise ValueError(f"cv_benchmark doit être > 0, reçu {cv_benchmark}")
    return value_per_tonne * (cv_benchmark / cv_kcal_per_kg)


# ------------------------------------------------------------------------------- ETS
def phase_in_factor(year: int) -> float:
    """Facteur de montée en charge de l'ETS maritime pour une année civile (B-H6)."""
    if year < min(EU_ETS_PHASE_IN):
        return 0.0
    return EU_ETS_PHASE_IN.get(year, EU_ETS_PHASE_IN_FROM_2026)


def phase_in_series(index: pd.DatetimeIndex) -> pd.Series:
    """Le facteur de montée en charge, aligné sur un index de dates."""
    return pd.Series([phase_in_factor(d.year) for d in index], index=index, name="phase_in")


def voyage_emissions_t_co2(
    bunker_consumed_t: float, emission_factor: float = DEFAULT_EMISSION_FACTOR
) -> float:
    """Émissions d'un voyage, en tonnes de CO2 (B-H5)."""
    if bunker_consumed_t < 0:
        raise ValueError("bunker_consumed_t doit être positif")
    return bunker_consumed_t * emission_factor


def ets_cost_per_cargo_tonne(
    eua_price_eur: pd.Series | float,
    eurusd: pd.Series | float,
    *,
    emissions_t_co2: float,
    cargo_t: float,
    phase_in: pd.Series | float,
    scope_factor: float = EXTRA_EU_SCOPE_FACTOR,
) -> pd.Series | float:
    """Coût ETS du voyage, ramené à la tonne de cargaison, en USD.

        coût = émissions × portée × montée_en_charge × prix_EUA × EURUSD / cargaison

    Le prix du quota est en EUR et l'arb en USD : la conversion de change est un terme du
    calcul, pas un ajustement cosmétique.
    """
    if cargo_t <= 0:
        raise ValueError("cargo_t doit être > 0")
    if not 0.0 <= scope_factor <= 1.0:
        raise ValueError(f"scope_factor doit être dans [0, 1], reçu {scope_factor}")
    quotas_due = emissions_t_co2 * scope_factor * phase_in
    return quotas_due * eua_price_eur * eurusd / cargo_t


# ------------------------------------------------------------------------------- arb
def financing_cost(
    cargo_value_per_tonne: pd.Series | float, voyage_days: float, annual_rate: float
) -> pd.Series | float:
    """Portage de la valeur FOB pendant le voyage (B-H4)."""
    if voyage_days < 0:
        raise ValueError("voyage_days doit être positif")
    return cargo_value_per_tonne * annual_rate * (voyage_days / 365.0)


def reconstruct_ara_arb(
    api2: pd.Series,
    api4: pd.Series,
    freight: pd.Series,
    *,
    voyage_days: float = 20.0,
    annual_rate: float = 0.06,
    ets_cost: pd.Series | float = 0.0,
) -> pd.DataFrame:
    """Arb Richards Bay -> ARA, terme par terme.

    Alignement par intersection des calendriers, aucun forward-fill. Retourne un
    DataFrame avec api2, api4, spread, freight, financing, ets, arb, is_open.
    """
    aligned = pd.concat({"api2": api2, "api4": api4, "freight": freight}, axis=1).dropna()
    if aligned.empty:
        raise ValueError(
            "aucune date commune aux trois séries — vérifier les calendriers, "
            "ne pas combler les trous"
        )
    aligned = aligned.sort_index()

    out = aligned.copy()
    out["spread"] = aligned["api2"] - aligned["api4"]
    out["financing"] = financing_cost(aligned["api4"], voyage_days, annual_rate)
    if isinstance(ets_cost, pd.Series):
        out["ets"] = ets_cost.reindex(out.index)
        if out["ets"].isna().any():
            raise ValueError(
                "la série de coût ETS ne couvre pas toutes les dates de l'arb — "
                "l'aligner explicitement plutôt que de la laisser trouer le calcul"
            )
    else:
        out["ets"] = float(ets_cost)
    out["arb"] = out["spread"] - out["freight"] - out["financing"] - out["ets"]
    out["is_open"] = out["arb"] > 0
    return out


# ------------------------------------------------------------------------- régression
@dataclass(frozen=True)
class OLSResult:
    """Régression avec constante. `coefficients` inclut 'const'."""

    coefficients: dict[str, float]
    std_errors: dict[str, float]
    t_stats: dict[str, float]
    r_squared: float
    n_obs: int
    regressors: tuple[str, ...] = field(default=())

    def summary(self) -> str:
        parts = [
            f"{name} = {self.coefficients[name]:+.3f} (t = {self.t_stats[name]:.2f})"
            for name in ["const", *self.regressors]
        ]
        return " | ".join(parts) + f" | R² = {self.r_squared:.3f} | n = {self.n_obs}"


def ols(y: pd.Series, regressors: dict[str, pd.Series]) -> OLSResult:
    """MCO avec constante et plusieurs régresseurs.

    Existe pour une raison précise : tester l'effet du fret sur le spread **en
    contrôlant par le TTF**. Une régression simple sur le fret seul attribuerait au fret
    ou à l'Inde ce qui appartient au choc gazier de 2022.

    Écarts-types classiques, non robustes à l'autocorrélation. Sur des séries
    quotidiennes de prix, les t de Student sont donc optimistes. C'est un caveat à
    afficher, pas à corriger en silence avec une formule dont on ne montrerait pas
    l'hypothèse.
    """
    if not regressors:
        raise ValueError("il faut au moins un régresseur")
    frame = pd.concat({"__y__": y, **regressors}, axis=1).dropna()
    names = tuple(regressors.keys())
    n = len(frame)
    k = len(names) + 1
    if n <= k:
        raise ValueError(f"pas assez d'observations : n={n} pour k={k} paramètres")

    y_vec = frame["__y__"].to_numpy(dtype=float)
    x_mat = np.column_stack(
        [np.ones(n)] + [frame[name].to_numpy(dtype=float) for name in names]
    )
    beta, *_ = np.linalg.lstsq(x_mat, y_vec, rcond=None)
    fitted = x_mat @ beta
    residuals = y_vec - fitted
    ss_res = float(residuals @ residuals)
    ss_tot = float(((y_vec - y_vec.mean()) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Ajustement parfait, ou numériquement indiscernable d'un ajustement parfait : le t
    # de Student n'est pas défini. Le seuil est relatif à la variance de y, sinon un
    # résidu de l'ordre de 1e-28 produit un t de 1e15 — un nombre qui a l'air d'une
    # certitude écrasante alors qu'il ne mesure que du bruit d'arrondi.
    perfect_fit = ss_res <= 1e-12 * max(ss_tot, 1.0)
    sigma2 = ss_res / (n - k)
    if perfect_fit or sigma2 <= 0:
        se = np.zeros(k)
    else:
        xtx_inv = np.linalg.pinv(x_mat.T @ x_mat)
        se = np.sqrt(np.maximum(np.diag(sigma2 * xtx_inv), 0.0))

    labels = ["const", *names]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = [
            float(b / s) if s > 0 else float("nan") for b, s in zip(beta, se)
        ]
    return OLSResult(
        coefficients=dict(zip(labels, (float(b) for b in beta))),
        std_errors=dict(zip(labels, (float(s) for s in se))),
        t_stats=dict(zip(labels, t_stats)),
        r_squared=r_squared,
        n_obs=n,
        regressors=names,
    )


# ---------------------------------------------------------------------------- régimes
@dataclass(frozen=True)
class RegimeStats:
    label: str
    n_obs: int
    arb_mean: float
    arb_std: float
    share_open: float
    longest_open_run: int


def _longest_true_run(flags: pd.Series) -> int:
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def regime_stats(arb_frame: pd.DataFrame, breakpoint: str | pd.Timestamp) -> list[RegimeStats]:
    """Statistiques de l'arb avant et après une date de rupture.

    Le test de la thèse : si le fret était la contrainte contraignante, l'arb devrait
    osciller autour de zéro sans persistance. Un arb dont la moyenne s'éloigne de zéro
    **et** dont les séquences d'ouverture s'allongent est un arb qui n'est plus contraint.
    """
    bp = pd.Timestamp(breakpoint)
    out = []
    for label, chunk in (
        (f"avant {bp.date()}", arb_frame[arb_frame.index < bp]),
        (f"depuis {bp.date()}", arb_frame[arb_frame.index >= bp]),
    ):
        if chunk.empty:
            out.append(RegimeStats(label, 0, float("nan"), float("nan"), float("nan"), 0))
            continue
        out.append(
            RegimeStats(
                label=label,
                n_obs=len(chunk),
                arb_mean=float(chunk["arb"].mean()),
                arb_std=float(chunk["arb"].std()),
                share_open=float(chunk["is_open"].mean()),
                longest_open_run=_longest_true_run(chunk["is_open"]),
            )
        )
    return out


def freight_binding_test(
    spread: pd.Series,
    freight: pd.Series,
    breakpoint: str | pd.Timestamp,
    *,
    controls: dict[str, pd.Series] | None = None,
) -> dict[str, OLSResult]:
    """Régresse le spread API2−API4 sur le fret, avant et après la rupture.

    Si le fret est la contrainte, le coefficient sur le fret doit être proche de 1 : un
    dollar de fret en plus, un dollar de spread en plus. Un coefficient qui s'effondre
    après 2022 dit que le spread ne se règle plus sur le fret.

    `controls` est là pour le TTF. Ne pas s'en passer.
    """
    controls = controls or {}
    bp = pd.Timestamp(breakpoint)
    results = {}
    for label, mask in (
        (f"avant {bp.date()}", spread.index < bp),
        (f"depuis {bp.date()}", spread.index >= bp),
    ):
        y = spread[mask]
        regs = {"freight": freight, **controls}
        regs = {name: s[s.index.isin(y.index)] for name, s in regs.items()}
        results[label] = ols(y, regs)
    return results


# ===========================================================================
# REAL DATA — coal-to-gas switching, and the parameter that decides it
# ===========================================================================
# The API2 - API4 arb this module was built around is not computable from the export: API4
# Richards Bay is absent. API2, TTF, EUA and EURUSD are all present, and they support a
# better question — the one a European generator actually faces every morning.
#
# A generator does not choose between coal and gas on fuel price. It chooses on fuel plus
# carbon, per MWh of ELECTRICITY, at plant efficiencies that differ by roughly a factor of
# one and a half. Three units and two currencies have to be reconciled before the comparison
# even exists, and the answer turns on two efficiencies that no exchange quotes.
API2_SHEET = "XA1 Comdty"
EUA_SHEET = "MO1 Comdty"

KCAL_TO_MWH = 1.163e-6
API2_KCAL_PER_KG = 6000.0          # API2 is assessed at 6 000 kcal/kg NAR
MWH_TH_PER_TONNE_COAL = API2_KCAL_PER_KG * 1000.0 * KCAL_TO_MWH   # ~6,978

# Emission factors, tonnes of CO2 per MWh of THERMAL input. Standard combustion figures,
# not plant-specific — a real unit varies with coal rank and with gas composition.
EF_COAL_T_PER_MWH_TH = 0.34
EF_GAS_T_PER_MWH_TH = 0.20

# Plant efficiencies. THE parameter of this page: not market data, not published, and the
# thing the whole answer turns on. Ranges span an old subcritical coal unit to a supercritical
# one, and an early CCGT to a modern H-class.
COAL_EFFICIENCY_RANGE = (0.36, 0.42)
GAS_EFFICIENCY_RANGE = (0.50, 0.60)
DEFAULT_COAL_EFFICIENCY = 0.38
DEFAULT_GAS_EFFICIENCY = 0.55


@cached('b_switching')
def load_real_switching_frame(start: str | None = "2018-01-01") -> pd.DataFrame:
    """API2, TTF, EUA and EURUSD on their common calendar, plus coal restated per MWh.

    Columns: api2_usd_t, ttf_eur_mwh, eua_eur_t, eurusd, coal_eur_mwh_th.

    The coal leg is the one that needs work: it arrives as a **price per tonne in dollars**
    and has to become a **price per thermal MWh in euros**, which takes a calorific value and
    a currency. Neither conversion is reversible by eye, which is why both are done here once
    and never again downstream.
    """
    from agri.data.bloomberg_loader import DEFAULT_PATH, load

    def read_sheet(sheet: str) -> pd.Series:
        raw = pd.read_excel(DEFAULT_PATH, sheet_name=sheet, header=None)
        values = pd.to_numeric(raw.iloc[:, 1], errors="coerce")
        dates = pd.to_datetime(raw.iloc[:, 0], errors="coerce", format="mixed")
        return pd.Series(values.values, index=dates).dropna().sort_index()

    frame = pd.concat(
        {
            "api2_usd_t": read_sheet(API2_SHEET),
            "eua_eur_t": read_sheet(EUA_SHEET),
            "ttf_eur_mwh": load("ttf"),
            "eurusd": load("eurusd"),
        },
        axis=1,
        sort=True,
    ).dropna()
    if start is not None:
        frame = frame[frame.index >= pd.Timestamp(start)]
    if frame.empty:
        raise ValueError(f"no common dates across the four switching legs after {start}")
    if not (0.6 < frame["eurusd"].median() < 1.8):
        raise ValueError(
            "EURUSD does not look like USD per EUR — check the quoting direction before "
            "any of this means anything"
        )

    frame["coal_eur_mwh_th"] = (
        frame["api2_usd_t"] / frame["eurusd"] / MWH_TH_PER_TONNE_COAL
    )
    return frame


def generation_cost_eur_mwh_e(
    frame: pd.DataFrame,
    *,
    coal_efficiency: float = DEFAULT_COAL_EFFICIENCY,
    gas_efficiency: float = DEFAULT_GAS_EFFICIENCY,
) -> pd.DataFrame:
    """Short-run marginal cost of each plant, in euros per MWh of electricity.

        coal = (coal_eur_mwh_th + EUA x EF_coal) / eta_coal
        gas  = (ttf_eur_mwh     + EUA x EF_gas ) / eta_gas

    Dividing by the efficiency is what turns a fuel price into a generation cost, and it is
    also what makes the carbon term bigger than it looks: a coal unit pays for its CO2 **and**
    burns more fuel per unit of output, so the efficiency divides the carbon cost too.
    """
    for efficiency, label in ((coal_efficiency, "coal"), (gas_efficiency, "gas")):
        if not 0.20 < efficiency < 0.70:
            raise ValueError(f"{label} efficiency outside the plausible range: {efficiency}")

    out = pd.DataFrame(index=frame.index)
    out["coal"] = (
        frame["coal_eur_mwh_th"] + frame["eua_eur_t"] * EF_COAL_T_PER_MWH_TH
    ) / coal_efficiency
    out["gas"] = (
        frame["ttf_eur_mwh"] + frame["eua_eur_t"] * EF_GAS_T_PER_MWH_TH
    ) / gas_efficiency
    out["spread"] = out["coal"] - out["gas"]
    out["gas_cheaper"] = out["spread"] > 0
    return out


def switching_carbon_price(
    frame: pd.DataFrame,
    *,
    coal_efficiency: float = DEFAULT_COAL_EFFICIENCY,
    gas_efficiency: float = DEFAULT_GAS_EFFICIENCY,
) -> pd.Series:
    """The EUA price at which the two plants cost the same, in closed form.

        EUA* = (ttf / eta_gas - coal_th / eta_coal) / (EF_coal / eta_coal - EF_gas / eta_gas)

    The denominator is the whole story. It contains **no price at all** — only two emission
    factors and two efficiencies. So the sensitivity of the switching price to the efficiency
    assumption is not a second-order correction: the efficiencies sit in the denominator of
    the answer.
    """
    denominator = (
        EF_COAL_T_PER_MWH_TH / coal_efficiency - EF_GAS_T_PER_MWH_TH / gas_efficiency
    )
    if denominator <= 0:
        raise ValueError(
            "the coal plant does not emit more CO2 per MWh of electricity than the gas "
            "plant at these efficiencies — no carbon price can make gas competitive, and "
            "the switching price is undefined rather than large"
        )
    numerator = (
        frame["ttf_eur_mwh"] / gas_efficiency - frame["coal_eur_mwh_th"] / coal_efficiency
    )
    return (numerator / denominator).rename("eua_switch_eur_t")


@dataclass(frozen=True)
class EfficiencyIdentification:
    """What the unpublished parameter does to the published answer.

    The coal-to-gas switching price is quoted in market commentary as if it were a property
    of the two fuels. It is not. It is a property of two plant efficiencies, and this class
    measures how much of the answer they account for.
    """

    grid: pd.DataFrame
    swing_eur_t: float
    eua_std_eur_t: float
    share_above_low: float
    share_above_high: float

    @property
    def ratio(self) -> float:
        return self.swing_eur_t / self.eua_std_eur_t

    @property
    def headline(self) -> str:
        return (
            f"The efficiency pair alone moves the switching price by {self.swing_eur_t:.0f} "
            f"EUR/t — {self.ratio:.1f} times the standard deviation of the carbon price "
            f"itself ({self.eua_std_eur_t:.0f} EUR/t). Depending on which plants one assumes "
            f"are at the margin, carbon has been high enough to displace coal anywhere "
            f"between {self.share_above_high:.0%} and {self.share_above_low:.0%} of the "
            "sample. Same fuel prices, same carbon price, opposite conclusions."
        )


def efficiency_identification(
    frame: pd.DataFrame,
    *,
    coal_efficiencies: tuple[float, ...] = (0.36, 0.38, 0.42),
    gas_efficiencies: tuple[float, ...] = (0.50, 0.55, 0.60),
) -> EfficiencyIdentification:
    """Sweep the plausible efficiency pairs and compare the swing to the EUA's own variability."""
    rows = []
    for coal_efficiency in coal_efficiencies:
        for gas_efficiency in gas_efficiencies:
            switch = switching_carbon_price(
                frame, coal_efficiency=coal_efficiency, gas_efficiency=gas_efficiency
            )
            rows.append(
                {
                    "coal_efficiency": coal_efficiency,
                    "gas_efficiency": gas_efficiency,
                    "switch_median_eur_t": float(switch.median()),
                    "share_eua_above": float((frame["eua_eur_t"] > switch).mean()),
                }
            )
    grid = pd.DataFrame(rows)
    return EfficiencyIdentification(
        grid=grid,
        swing_eur_t=float(grid["switch_median_eur_t"].max() - grid["switch_median_eur_t"].min()),
        eua_std_eur_t=float(frame["eua_eur_t"].std()),
        share_above_low=float(grid["share_eua_above"].max()),
        share_above_high=float(grid["share_eua_above"].min()),
    )
