"""Project B — the Atlantic coal arb lost its binding constraint in 2022.

THESIS
------
The classic Richards Bay to ARA arb:

    arb_ARA = API2 (CIF ARA) − API4 (FOB Richards Bay) − freight(C4) − financing − ETS

The textbook says this arb can't stay wide open: competition and freight pull it back
toward zero. Since 2022 South African coal has been going to India rather than Europe.
**The marginal Richards Bay cargo is no longer priced off Rotterdam.** The equation
therefore lost its binding term, and it's missing one: the netback to the alternative
destination.

The CFR India price is licensed data, so the equation can't be proven in price. It's
shown in **flow**: when the ARA arb closes, the Indian share of Richards Bay exports
rises. That's a weaker result than a price equality, and it should be stated as such
rather than inventing a series.

THE INTELLECTUAL RISK, ADDRESSED HEAD-ON
-------------------------------------------
2022 is also the year of the European gas shock, which sent European coal demand
soaring. A post-2022 break attributed to India when it actually comes from gas is a
mistake the first competent reader of the email will catch. **TTF is therefore a
mandatory control, not a refinement.** That's the reason `ols()` in this module takes
multiple regressors: without the control, the test is worthless.

THE TWO TECHNICAL LAYERS
---------------------------
1. **Calorific value.** API2 and API4 are both 6,000 kcal/kg NAR reference grades: the
   reference arb is therefore internally consistent and CV-neutral **by construction**.
   But the coal actually loaded at Richards Bay has drifted to ~5,700-5,800 kcal.
   Freight is paid **per tonne** and coal is sold **per kcal**: a 5,700 kcal cargo pays
   the same freight per tonne but delivers 5% less energy. Freight per
   tonne-equivalent-6,000 is therefore ~5% more expensive.

   It's the same move as moisture on iron ore: the quoted unit is not the economic
   unit. Here it's a **sensitivity**, not the main identity — the reference arb stays
   correct, it has simply stopped describing the physical cargo.

2. **The European maritime ETS.** Since 2024, a ship over 5,000 GT must surrender
   allowances on its emissions. For a voyage with one end outside the EU — the case for
   Richards Bay → Rotterdam — coverage is 50% of the voyage's emissions, applied to a
   phase-in factor of 40% in 2024, 70% in 2025 and 100% from 2026. That's an effective
   coverage of 20%, 35% then 50%.

   Consequence: freight to Europe becomes structurally more expensive than freight to
   India **at equal distance**. A recent, quantifiable term, absent from public coal
   arb models.

   An additional unit trap: the allowance is quoted in **EUR**, the arb is in **USD**.
   An FX series is needed, and it's a real term, not a detail.

ASSUMPTIONS
-----------
B-H1  API2 and API4 share the same 6,000 kcal/kg NAR reference. Solid.
B-H2  The CV actually exported at Richards Bay is below the reference. Handled in
      sensitivity, value to be sourced from producers' annual reports.
B-H3  C4 (Capesize) is the relevant RB → ARA freight. Part of the flow travels on
      Panamax/Supramax, at a different cost. Parameterised.
B-H4  Financing at the annual rate applied to the FOB value over the voyage duration.
B-H5  Fuel emission factor: 3.114 tCO2 per tonne of fuel (order of magnitude of the IMO
      factor for heavy fuel oil). **Parameterised, to confirm** depending on the fuel
      actually bunkered — VLSFO and MGO have different factors.
B-H6  The ETS regulatory parameters (50% scope, 40/70/100 phase-in) are
      **parameterised and never hard-coded elsewhere**. To reconfirm against the text
      before publication.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from agri.data.snapshot import cached

BENCHMARK_CV_KCAL_PER_KG = 6000.0

# B-H6 — maritime ETS phase-in. Before 2024: no obligation.
# From 2026: 100%. Parameterised so the regulatory text stays the reference and the
# code a transcription of it.
EU_ETS_PHASE_IN: dict[int, float] = {2024: 0.40, 2025: 0.70}
EU_ETS_PHASE_IN_FROM_2026 = 1.00
# Scope for a voyage with one end outside the EU.
EXTRA_EU_SCOPE_FACTOR = 0.50
# B-H5 — emission factor, tCO2 per tonne of fuel.
DEFAULT_EMISSION_FACTOR = 3.114


# --------------------------------------------------------------- calorific value
def to_energy_basis(
    value_per_tonne: pd.Series | float,
    cv_kcal_per_kg: float,
    cv_benchmark: float = BENCHMARK_CV_KCAL_PER_KG,
) -> pd.Series | float:
    """Expresses an amount per tonne on the reference energy basis.

    A 5,700 kcal/kg cargo delivers 5,700/6,000 = 95% of a reference tonne's energy. So
    any cost paid per tonne — freight foremost — is worth `× 6000/5700 = 1.0526` per
    tonne-equivalent-6,000.

    Works both ways: applied to a price, it restates the price on the reference basis;
    applied to a freight rate, it gives the freight per unit of delivered energy.
    """
    if cv_kcal_per_kg <= 0:
        raise ValueError(f"cv_kcal_per_kg must be > 0, got {cv_kcal_per_kg}")
    if cv_benchmark <= 0:
        raise ValueError(f"cv_benchmark must be > 0, got {cv_benchmark}")
    return value_per_tonne * (cv_benchmark / cv_kcal_per_kg)


# ------------------------------------------------------------------------------- ETS
def phase_in_factor(year: int) -> float:
    """Maritime ETS phase-in factor for a calendar year (B-H6)."""
    if year < min(EU_ETS_PHASE_IN):
        return 0.0
    return EU_ETS_PHASE_IN.get(year, EU_ETS_PHASE_IN_FROM_2026)


def phase_in_series(index: pd.DatetimeIndex) -> pd.Series:
    """The phase-in factor, aligned to a date index."""
    return pd.Series([phase_in_factor(d.year) for d in index], index=index, name="phase_in")


def voyage_emissions_t_co2(
    bunker_consumed_t: float, emission_factor: float = DEFAULT_EMISSION_FACTOR
) -> float:
    """A voyage's emissions, in tonnes of CO2 (B-H5)."""
    if bunker_consumed_t < 0:
        raise ValueError("bunker_consumed_t must be positive")
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
    """The voyage's ETS cost, restated per tonne of cargo, in USD.

        cost = emissions × scope × phase_in × EUA_price × EURUSD / cargo

    The allowance price is in EUR and the arb in USD: the FX conversion is a term in
    the calculation, not a cosmetic adjustment.
    """
    if cargo_t <= 0:
        raise ValueError("cargo_t must be > 0")
    if not 0.0 <= scope_factor <= 1.0:
        raise ValueError(f"scope_factor must be in [0, 1], got {scope_factor}")
    quotas_due = emissions_t_co2 * scope_factor * phase_in
    return quotas_due * eua_price_eur * eurusd / cargo_t


# ------------------------------------------------------------------------------- arb
def financing_cost(
    cargo_value_per_tonne: pd.Series | float, voyage_days: float, annual_rate: float
) -> pd.Series | float:
    """Carry cost of the FOB value during the voyage (B-H4)."""
    if voyage_days < 0:
        raise ValueError("voyage_days must be positive")
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
    """Richards Bay -> ARA arb, term by term.

    Alignment by calendar intersection, no forward-fill. Returns a DataFrame with
    api2, api4, spread, freight, financing, ets, arb, is_open.
    """
    aligned = pd.concat({"api2": api2, "api4": api4, "freight": freight}, axis=1).dropna()
    if aligned.empty:
        raise ValueError(
            "no common date across the three series — check the calendars, "
            "don't fill the gaps"
        )
    aligned = aligned.sort_index()

    out = aligned.copy()
    out["spread"] = aligned["api2"] - aligned["api4"]
    out["financing"] = financing_cost(aligned["api4"], voyage_days, annual_rate)
    if isinstance(ets_cost, pd.Series):
        out["ets"] = ets_cost.reindex(out.index)
        if out["ets"].isna().any():
            raise ValueError(
                "the ETS cost series doesn't cover every date in the arb — "
                "align it explicitly rather than letting it punch holes in the calculation"
            )
    else:
        out["ets"] = float(ets_cost)
    out["arb"] = out["spread"] - out["freight"] - out["financing"] - out["ets"]
    out["is_open"] = out["arb"] > 0
    return out


# ------------------------------------------------------------------------- regression
@dataclass(frozen=True)
class OLSResult:
    """Regression with a constant. `coefficients` includes 'const'."""

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
    """OLS with a constant and several regressors.

    Exists for a specific reason: testing freight's effect on the spread **while
    controlling for TTF**. A simple regression on freight alone would attribute to
    freight or to India what belongs to the 2022 gas shock.

    Classic standard errors, not robust to autocorrelation. On daily price series, the
    t-stats are therefore optimistic. That's a caveat to display, not to silently
    correct with a formula whose assumption wouldn't be shown.
    """
    if not regressors:
        raise ValueError("at least one regressor is required")
    frame = pd.concat({"__y__": y, **regressors}, axis=1).dropna()
    names = tuple(regressors.keys())
    n = len(frame)
    k = len(names) + 1
    if n <= k:
        raise ValueError(f"not enough observations: n={n} for k={k} parameters")

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

    # Perfect fit, or numerically indistinguishable from a perfect fit: the t-stat is
    # undefined. The threshold is relative to y's variance, otherwise a residual on the
    # order of 1e-28 produces a t of 1e15 — a number that looks like overwhelming
    # certainty while it only measures rounding noise.
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


# ---------------------------------------------------------------------------- regimes
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
    """Arb statistics before and after a break date.

    The thesis's test: if freight were the binding constraint, the arb should
    oscillate around zero with no persistence. An arb whose mean drifts from zero
    **and** whose open runs grow longer is an arb that's no longer constrained.
    """
    bp = pd.Timestamp(breakpoint)
    out = []
    for label, chunk in (
        (f"before {bp.date()}", arb_frame[arb_frame.index < bp]),
        (f"since {bp.date()}", arb_frame[arb_frame.index >= bp]),
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
    """Regresses the API2−API4 spread on freight, before and after the break.

    If freight is the constraint, the coefficient on freight should be close to 1: one
    more dollar of freight, one more dollar of spread. A coefficient that collapses
    after 2022 says the spread no longer settles against freight.

    `controls` is there for TTF. Don't skip it.
    """
    controls = controls or {}
    bp = pd.Timestamp(breakpoint)
    results = {}
    for label, mask in (
        (f"before {bp.date()}", spread.index < bp),
        (f"since {bp.date()}", spread.index >= bp),
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
