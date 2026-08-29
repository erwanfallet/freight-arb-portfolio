"""Project B — the carbon hedge inside the switching spread, and its disappearance.

THE ASSET
---------
A generator holding both a coal unit and a CCGT owns an option to switch between them.
An option is priced off VOLATILITY, not level — so what values that asset is the
volatility of the coal-minus-gas generation spread. Written out, the two market legs
enter that spread with **opposite signs**:

    spread = coal_cost - gas_cost
           = const - (1/eta_gas) x TTF + (EF_coal/eta_coal - EF_gas/eta_gas) x EUA

    Var(spread) = (b_ttf x sigma_ttf)^2 + (b_eua x sigma_eua)^2
                  + 2 x b_ttf x b_eua x rho x sigma_ttf x sigma_eua

with `b_ttf < 0 < b_eua`, so a **positive correlation SUBTRACTS from the variance**.
Carbon is a natural hedge for gas inside this particular spread. That is algebra rather
than an empirical regularity: across the whole plausible efficiency grid `b_ttf` runs
-2.000 to -1.667 and `b_eua` runs +0.410 to +0.611, and the signs never cross.

THE TRADE
---------
From 2018 to 2025 the gas-carbon correlation ran +0.30 to +0.70 and removed a median
**12%** of the spread's volatility, up to 29% in 2024. In 2026 it collapsed to zero — a
95% interval spanning zero, against +0.48 the year before, a break significant at
p < 0.00001 — and the dampening went with it.

**The part worth trading is not the biggest part.** Spread volatility rose 175% into
2026, and 84% of that is TTF's own volatility doubling, which is in every market report.
Only 16% is the lost correlation hedge. But a risk model carrying rho at its historical
+0.4 misses exactly that 16%: it understates spread volatility by ~20% and an
at-the-money switching option by **~11%**, with nothing anywhere flagging an error.
`natural_hedge()`, `dampening_attribution()`.

WHY THE OBVIOUS EXPLANATION IS WRONG
------------------------------------
The reflex answer is saturation: if every coal unit is already running, dearer gas cannot
start another one and the transmission channel is exhausted. `switching_depth_profile()`
rules it out. 2018 sat above the switching level 87% of the time at a median depth of
+7%, with **zero** days beyond +40% — and carbon tracked gas at rho = +0.38. 2026 sits
above 63% of the time at a median depth of +9%, with 1% of days beyond +40%. Nearly
identical depth, opposite correlation. Only 2022 was genuinely saturated (61% of days
beyond +40%), and it is the sample's other negative year.

Nor is it a structural decline in the European coal fleet: 2024 has the **strongest**
correlation in the whole sample at +0.70.

What is left is a transmission failure rather than an exhausted mechanism.
`transmission_test()` puts it non-parametrically: on each year's ten largest gas moves,
carbon moved the same way 7 to 9 times in seven of nine years, and **3 times in 2026**.
A crisis-sized gas shock happened and the carbon market did not take it. Why that is —
Q1 LNG and weather shocks read as transient, forward hedging that decouples spot gas
from near-term generation, or a carbon market driven since January by CBAM's definitive
regime and cap reform — is not answerable from price data alone, and is the question the
page ends on.

THE CEILING TEST THAT CAME FIRST
--------------------------------
The sections below predate the trade above and are kept because they constrain it. The
switching LEVEL is not identified — it depends on two plant efficiencies no exchange
quotes, and across the plausible pair the level swings ~30 to ~47 EUR/MWh while the share
of days above it swings 16% to 75%. Distance from that level does predict TTF's next 20
days, but the t-statistic is invariant across the entire efficiency grid (-3.02 to -3.05)
because the efficiencies enter only affinely and a t-statistic is invariant under an
affine map; and a naive thermal-parity anchor with no carbon price in it predicts just as
well (F = 1.17, p = 0.28). The predictive claim also survives its own Stambaugh bias only
at p = 0.018 against ~0.001 read naively.

That matters for the trade: **the level is unidentified, the sensitivities are not.**
`b_ttf` and `b_eua` are definitional, so the hedge result does not inherit the level's
identification problem.

WHAT THIS REPLACES
-------------------
This module was originally built around the Richards Bay → ARA arb (API2 CIF ARA minus
API4 FOB Richards Bay, minus freight and a maritime-ETS layer), on the thesis that
Indian demand had pulled the marginal RB cargo off the European netback since 2022.
**API4 Richards Bay is absent from the export**, so that spread was never computable
from data actually available here, and the thesis was abandoned before publication
rather than faked with a proxy. API2, TTF, EUA and EURUSD remain, and support the
switching question above instead.

ASSUMPTIONS
-----------
B-H1  API2 is assessed at 6,000 kcal/kg NAR; `MWH_TH_PER_TONNE_COAL` converts on that
      reference. The coal actually burned at a given plant may differ — a sensitivity,
      not the identity.
B-H2  Emission factors (`EF_COAL_T_PER_MWH_TH`, `EF_GAS_T_PER_MWH_TH`) are standard
      combustion figures, not plant-specific.
B-H3  Plant efficiencies are the parameter the switching LEVEL turns on, and
      `efficiency_invariance()` shows they are irrelevant to the PREDICTION. Both halves
      are reported; neither is allowed to stand in for the other.
B-H4  The 20-trading-day horizon and the 250-day trailing-median window are
      parameterised, not tuned — changing them is a robustness check, not a re-fit.
B-H5  The bootstrap resamples (u_t, v_{t+1}) in pairs to preserve their contemporaneous
      correlation, and rebuilds the regressor through its own fitted AR(1). It assumes
      an AR(1) is an adequate description of the regressor's persistence — a stronger
      assumption than the rest of the page, and the reason the analytic Kendall
      correction is reported alongside it as an independent check.
B-H6  Physical coal capacity is finite, so the ceiling cannot bind without limit. This
      page cannot test the saturation point: EU coal generation and plant availability
      are not in this export, and the sample has 43 above-switch windows in total.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from agri.data.snapshot import cached

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


# ===========================================================================
# THE CEILING TEST — does switching-distance predict TTF's forward return?
# ===========================================================================
# The idea that fuel switching caps gas is not new to anyone who trades power. What is
# untested is whether the level implied by plant efficiencies actually predicts what TTF
# does next, or whether "switching" is just a name attached after the fact to ordinary
# mean reversion. This section answers that, on non-overlapping windows so the t-stat
# is not manufactured, against a placebo that contains no switching economics at all.

FORWARD_HORIZON_DAYS = 20
TRAILING_MEDIAN_WINDOW = 250
MIN_OBS_FOR_VERDICT = 60


def switch_ttf_eur_mwh(
    frame: pd.DataFrame,
    *,
    coal_efficiency: float = DEFAULT_COAL_EFFICIENCY,
    gas_efficiency: float = DEFAULT_GAS_EFFICIENCY,
) -> pd.Series:
    """The TTF level at which gas costs the same to burn as coal, in EUR/MWh.

    The same equality as `switching_carbon_price`, rearranged for TTF instead of EUA:

        ttf* = (eta_gas/eta_coal) x coal_th + eua x (eta_gas x EF_coal/eta_coal - EF_gas)

    Above this level, a generator holding both units running should burn coal, not gas —
    the physical substitution this section tests for.
    """
    return (
        (gas_efficiency / coal_efficiency) * frame["coal_eur_mwh_th"]
        + frame["eua_eur_t"]
        * (gas_efficiency * EF_COAL_T_PER_MWH_TH / coal_efficiency - EF_GAS_T_PER_MWH_TH)
    ).rename("ttf_switch_eur_mwh")


def switching_distance_pct(frame: pd.DataFrame, switch_ttf: pd.Series) -> pd.Series:
    """How far actual TTF sits above (positive) or below (negative) its switching level."""
    return ((frame["ttf_eur_mwh"] - switch_ttf) / switch_ttf).rename("distance_pct")


def trailing_median_distance_pct(
    ttf: pd.Series, window: int = TRAILING_MEDIAN_WINDOW
) -> pd.Series:
    """Placebo regressor: distance from TTF's own trailing median — no switching economics.

    Exists to answer one question: is the ceiling effect specific to switching, or would
    any "TTF is far from its recent level" measure predict the same reversion? If both
    regressors predicted forward returns equally, switching would explain nothing beyond
    what the market already prices into an ordinary pullback.
    """
    trailing_median = ttf.rolling(window, min_periods=window).median()
    return ((ttf - trailing_median) / trailing_median).rename("placebo_distance_pct")


def non_overlapping(frame: pd.DataFrame, horizon_days: int = FORWARD_HORIZON_DAYS) -> pd.DataFrame:
    """Every `horizon_days`-th row, so a forward-return regression is not inflated by overlap.

    The daily version of this regression reuses the same 20-day-ahead outcome on twenty
    consecutive rows; the t-stat it produces measures the overlap, not the evidence.
    """
    return frame.iloc[::horizon_days].dropna()


@dataclass(frozen=True)
class CeilingTest:
    """Does distance from the switching level predict TTF's forward return — or is it
    mean reversion wearing a fuel-switching costume?

    `switching` regresses the 20-day forward log return on distance from the switching
    level; `placebo` uses distance from TTF's own trailing median instead, a null with no
    switching economics in it. `horse_race` runs both together: if switching added
    nothing beyond generic reversion, its coefficient would collapse once the placebo
    enters the equation. All three are fit on the **non-overlapping** sample.
    """

    switching: OLSResult
    placebo: OLSResult
    horse_race: OLSResult
    n_overlapping: int
    share_above: float


def ceiling_test(
    frame: pd.DataFrame,
    switch_ttf: pd.Series,
    *,
    horizon_days: int = FORWARD_HORIZON_DAYS,
    trailing_window: int = TRAILING_MEDIAN_WINDOW,
) -> CeilingTest:
    """Run the switching-vs-placebo forward-return horse race on non-overlapping windows.

    Raises rather than returns a verdict below `MIN_OBS_FOR_VERDICT` non-overlapping
    windows — a t-stat computed on too few independent observations is not a t-stat
    worth reading, however it happens to come out.
    """
    distance = switching_distance_pct(frame, switch_ttf)
    placebo_distance = trailing_median_distance_pct(frame["ttf_eur_mwh"], trailing_window)
    forward_return = np.log(
        frame["ttf_eur_mwh"].shift(-horizon_days) / frame["ttf_eur_mwh"]
    ).rename("forward_return")

    joined = pd.concat(
        {"forward_return": forward_return, "distance": distance, "placebo": placebo_distance},
        axis=1,
    ).dropna()
    n_overlapping = len(joined)
    honest = non_overlapping(joined, horizon_days)
    if len(honest) <= MIN_OBS_FOR_VERDICT:
        raise ValueError(
            f"only {len(honest)} non-overlapping windows available, below the "
            f"{MIN_OBS_FOR_VERDICT} needed to trust a t-stat here"
        )

    return CeilingTest(
        switching=ols(honest["forward_return"], {"distance": honest["distance"]}),
        placebo=ols(honest["forward_return"], {"placebo": honest["placebo"]}),
        horse_race=ols(
            honest["forward_return"],
            {"distance": honest["distance"], "placebo": honest["placebo"]},
        ),
        n_overlapping=n_overlapping,
        share_above=float((distance > 0).mean()),
    )


# ===========================================================================
# The predictive-regression sample, built once
# ===========================================================================
def predictive_sample(
    frame: pd.DataFrame,
    switch_ttf: pd.Series,
    *,
    horizon_days: int = FORWARD_HORIZON_DAYS,
    phase: int = 0,
) -> pd.DataFrame:
    """Forward return and switching distance, on one non-overlapping phase.

    `phase` selects which of the `horizon_days` possible non-overlapping subsamples is
    taken. It exists to be swept, not chosen: `phase_robustness()` runs all of them,
    because picking one and reporting it would be a researcher degree of freedom worth
    exactly one significant result.
    """
    if not 0 <= phase < horizon_days:
        raise ValueError(f"phase must be in [0, {horizon_days}), got {phase}")
    distance = switching_distance_pct(frame, switch_ttf)
    forward_return = np.log(
        frame["ttf_eur_mwh"].shift(-horizon_days) / frame["ttf_eur_mwh"]
    )
    joined = pd.concat(
        {"forward_return": forward_return, "distance": distance}, axis=1
    ).dropna()
    return joined.iloc[phase::horizon_days]


# ===========================================================================
# ROBUSTNESS 1 — the phase of the non-overlapping sample
# ===========================================================================
@dataclass(frozen=True)
class PhaseRobustness:
    """The same regression on all `horizon_days` non-overlapping subsamples.

    `iloc[::20]` throws away nineteen twentieths of the data and keeps one arbitrary
    phase. There are twenty such samples and no reason to prefer any one; if the result
    lived in a single phase it would be a starting-point artefact.
    """

    betas: tuple[float, ...]
    t_stats: tuple[float, ...]

    @property
    def n_phases(self) -> int:
        return len(self.betas)

    @property
    def n_negative(self) -> int:
        return sum(1 for b in self.betas if b < 0)

    @property
    def n_significant(self) -> int:
        return sum(1 for t in self.t_stats if abs(t) > 1.98)

    @property
    def all_agree_on_sign(self) -> bool:
        return self.n_negative in (0, self.n_phases)

    @property
    def headline(self) -> str:
        return (
            f"Across all {self.n_phases} non-overlapping phases the coefficient is "
            f"negative {self.n_negative} times and significant {self.n_significant} "
            f"times, ranging {min(self.betas):+.3f} to {max(self.betas):+.3f}. The "
            "result does not live in the starting point."
        )


def phase_robustness(
    frame: pd.DataFrame,
    switch_ttf: pd.Series,
    *,
    horizon_days: int = FORWARD_HORIZON_DAYS,
) -> PhaseRobustness:
    betas, t_stats = [], []
    for phase in range(horizon_days):
        sample = predictive_sample(frame, switch_ttf, horizon_days=horizon_days, phase=phase)
        result = ols(sample["forward_return"], {"distance": sample["distance"]})
        betas.append(result.coefficients["distance"])
        t_stats.append(result.t_stats["distance"])
    return PhaseRobustness(betas=tuple(betas), t_stats=tuple(t_stats))


# ===========================================================================
# ROBUSTNESS 2 — Stambaugh bias, the objection that points the same way
# ===========================================================================
@dataclass(frozen=True)
class StambaughDiagnostics:
    """Why this particular regression is biased toward its own conclusion.

    In a predictive regression `r_t = a + b·x_t + u_t` where the regressor follows
    `x_{t+1} = c + rho·x_t + v_{t+1}`, OLS is biased whenever `u_t` and `v_{t+1}`
    correlate, because the AR coefficient is itself biased downward in finite samples
    (Kendall). The bias is `(sigma_uv / sigma_vv) · E[rho_hat − rho]`.

    Here the regressor is `(TTF − switch)/switch`, so TTF sits in its numerator and the
    correlation is large and positive by construction — not by accident of this sample.
    Kendall's term is negative, so the product is negative: **OLS is pushed toward the
    negative coefficient this page reports.**
    """

    rho: float
    corr_uv: float
    kendall_term: float
    bias: float
    beta_ols: float
    n_obs: int

    @property
    def beta_corrected(self) -> float:
        return self.beta_ols - self.bias

    @property
    def bias_share(self) -> float:
        return abs(self.bias / self.beta_ols) if self.beta_ols else float("nan")

    @property
    def bias_favours_the_finding(self) -> bool:
        """True when the bias has the same sign as the estimate — the awkward case."""
        return self.bias * self.beta_ols > 0

    @property
    def headline(self) -> str:
        direction = "toward" if self.bias_favours_the_finding else "against"
        return (
            f"The regressor is persistent (rho = {self.rho:.2f}) and its innovations "
            f"correlate {self.corr_uv:+.2f} with the return being predicted, because TTF "
            f"sits inside it. That biases OLS {direction} the result reported here, by "
            f"{self.bias:+.4f} — {self.bias_share:.0%} of the coefficient. Bias-corrected, "
            f"beta is {self.beta_corrected:+.3f} rather than {self.beta_ols:+.3f}."
        )


def stambaugh_diagnostics(
    frame: pd.DataFrame,
    switch_ttf: pd.Series,
    *,
    horizon_days: int = FORWARD_HORIZON_DAYS,
    phase: int = 0,
) -> StambaughDiagnostics:
    """Analytic first-order bias for the predictive regression (Stambaugh 1999).

    Reported next to the bootstrap rather than instead of it: the two rest on different
    assumptions, and their agreement is the check that neither is a coding accident.
    """
    sample = predictive_sample(frame, switch_ttf, horizon_days=horizon_days, phase=phase)
    y = sample["forward_return"].to_numpy(dtype=float)
    x = sample["distance"].to_numpy(dtype=float)
    n = len(y)
    if n < 20:
        raise ValueError(f"only {n} windows — too few for a bias estimate worth reporting")

    design = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    u_all = y - design @ beta

    ar_design = np.column_stack([np.ones(n - 1), x[:-1]])
    ar_beta = np.linalg.lstsq(ar_design, x[1:], rcond=None)[0]
    rho = float(ar_beta[1])
    v_next = x[1:] - ar_design @ ar_beta      # v_{t+1}
    u = u_all[:-1]                             # u_t, same [t, t+1] interval

    sigma_uv = float(np.cov(u, v_next, ddof=2)[0, 1])
    sigma_vv = float(v_next.var(ddof=2))
    kendall = -(1.0 + 3.0 * rho) / n
    return StambaughDiagnostics(
        rho=rho,
        corr_uv=float(np.corrcoef(u, v_next)[0, 1]),
        kendall_term=float(kendall),
        bias=float((sigma_uv / sigma_vv) * kendall) if sigma_vv > 0 else float("nan"),
        beta_ols=float(beta[1]),
        n_obs=n,
    )


@dataclass(frozen=True)
class BootstrapTest:
    """Nelson–Kim bootstrap: where the estimate falls in a world with no predictability.

    The null is simulated rather than assumed: the regressor is rebuilt through its own
    fitted AR(1) and the innovation pairs `(u_t, v_{t+1})` are resampled **together**, so
    the simulated world keeps both the persistence and the correlation that cause the
    bias — and has no predictability in it at all. `null_mean` is therefore a direct
    measurement of the bias, obtained without the Kendall formula.
    """

    beta_obs: float
    null_mean: float
    null_p05: float
    p_value: float
    n_boot: int
    n_obs: int

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    @property
    def headline(self) -> str:
        verdict = "survives" if self.significant else "does not survive"
        return (
            f"Under a simulated null with the same persistence and the same innovation "
            f"correlation but no predictability, the coefficient averages "
            f"{self.null_mean:+.4f} — the bias, measured without a formula. The observed "
            f"{self.beta_obs:+.3f} sits at p = {self.p_value:.3f}, so the ceiling "
            f"{verdict} the honest test."
        )


def bootstrap_null(
    frame: pd.DataFrame,
    switch_ttf: pd.Series,
    *,
    horizon_days: int = FORWARD_HORIZON_DAYS,
    phase: int = 0,
    n_boot: int = 5_000,
    seed: int = 0,
) -> BootstrapTest:
    """Empirical p-value for the predictive coefficient under H0: no predictability (B-H5)."""
    sample = predictive_sample(frame, switch_ttf, horizon_days=horizon_days, phase=phase)
    y = sample["forward_return"].to_numpy(dtype=float)
    x = sample["distance"].to_numpy(dtype=float)
    n = len(y)
    if n < 20:
        raise ValueError(f"only {n} windows — too few to bootstrap anything meaningful")

    def slope(y_vec: np.ndarray, x_vec: np.ndarray) -> float:
        design = np.column_stack([np.ones(len(x_vec)), x_vec])
        return float(np.linalg.lstsq(design, y_vec, rcond=None)[0][1])

    design = np.column_stack([np.ones(n), x])
    fitted = np.linalg.lstsq(design, y, rcond=None)[0]
    beta_obs = float(fitted[1])
    u_all = y - design @ fitted

    ar_design = np.column_stack([np.ones(n - 1), x[:-1]])
    ar_beta = np.linalg.lstsq(ar_design, x[1:], rcond=None)[0]
    intercept, rho = float(ar_beta[0]), float(ar_beta[1])
    pairs = np.column_stack([u_all[:-1], x[1:] - ar_design @ ar_beta])

    rng = np.random.default_rng(seed)
    y_mean = float(y.mean())
    betas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(pairs), size=n)
        u_sim, v_sim = pairs[idx, 0], pairs[idx, 1]
        x_sim = np.empty(n)
        x_sim[0] = x[rng.integers(0, n)]
        for t in range(1, n):
            x_sim[t] = intercept + rho * x_sim[t - 1] + v_sim[t - 1]
        betas[i] = slope(y_mean + u_sim, x_sim)

    return BootstrapTest(
        beta_obs=beta_obs,
        null_mean=float(betas.mean()),
        null_p05=float(np.percentile(betas, 5)),
        p_value=float((betas <= beta_obs).mean()),
        n_boot=n_boot,
        n_obs=n,
    )


# ===========================================================================
# ROBUSTNESS 3 — the asymmetry only the mechanism predicts
# ===========================================================================
@dataclass(frozen=True)
class AsymmetryTest:
    """Above the switch there is a mechanism; below it there is none.

    Burning coal instead of gas removes gas demand and pulls TTF down — that only
    operates when gas is the dearer fuel. Below the switching level gas is already the
    cheap one and nothing pushes TTF back up. So the physical story predicts a *one-sided*
    effect, while generic mean reversion pulls symmetrically from both sides. This is the
    sharpest available discriminator between them, and unlike the trailing-median placebo
    it uses no second series.
    """

    above: OLSResult
    below: OLSResult
    kinked: OLSResult

    @property
    def is_one_sided(self) -> bool:
        return abs(self.above.t_stats["distance"]) > 1.98 and abs(
            self.below.t_stats["distance"]
        ) < 1.98

    @property
    def headline(self) -> str:
        above_t = self.above.t_stats["distance"]
        below_t = self.below.t_stats["distance"]
        if self.is_one_sided:
            return (
                f"Above the switching level the coefficient is "
                f"{self.above.coefficients['distance']:+.3f} (t = {above_t:.2f}, "
                f"n = {self.above.n_obs}); below it there is nothing (t = {below_t:.2f}, "
                f"n = {self.below.n_obs}). Mean reversion would pull from both sides — "
                "this is one-sided in the direction the physical mechanism requires."
            )
        return (
            f"The effect is not one-sided: above t = {above_t:.2f}, below t = "
            f"{below_t:.2f}. That is the signature of mean reversion rather than of "
            "switching, and it weakens the mechanism reading."
        )


def asymmetry_test(
    frame: pd.DataFrame,
    switch_ttf: pd.Series,
    *,
    horizon_days: int = FORWARD_HORIZON_DAYS,
    phase: int = 0,
) -> AsymmetryTest:
    sample = predictive_sample(frame, switch_ttf, horizon_days=horizon_days, phase=phase)
    above = sample[sample["distance"] > 0]
    below = sample[sample["distance"] <= 0]
    if min(len(above), len(below)) < 15:
        raise ValueError(
            f"one side has too few windows to regress ({len(above)} above, {len(below)} below)"
        )
    return AsymmetryTest(
        above=ols(above["forward_return"], {"distance": above["distance"]}),
        below=ols(below["forward_return"], {"distance": below["distance"]}),
        kinked=ols(
            sample["forward_return"],
            {
                "slope_above": sample["distance"].clip(lower=0),
                "slope_below": sample["distance"].clip(upper=0),
            },
        ),
    )


# ===========================================================================
# THE UNIFYING RESULT — the level moves, the prediction cannot
# ===========================================================================
def naive_thermal_anchor(frame: pd.DataFrame) -> pd.Series:
    """Raw thermal parity: coal's own per-MWh price, no efficiencies, no carbon.

    The honest competitor. If distance from *this* predicts as well as distance from the
    carefully-constructed switching level, then the switching arithmetic is decoration
    on "gas is expensive relative to coal".
    """
    return frame["coal_eur_mwh_th"].rename("naive_anchor")


@dataclass(frozen=True)
class EfficiencyInvariance:
    """The two halves of project B, measured on one grid.

    `level_swing` and `share_swing` are the original thesis: the unpublished efficiency
    pair decides where the line sits, and the diagnosis flips with it. `t_range` and
    `max_pairwise_corr` are the new one: it cannot decide what the line predicts, because
    the distance measures it generates are the same variable up to an affine map, and an
    OLS t-statistic does not move under one.
    """

    grid: pd.DataFrame
    level_low: float
    level_high: float
    share_low: float
    share_high: float
    t_low: float
    t_high: float
    max_pairwise_corr: float
    min_pairwise_corr: float
    affine_residual: float

    @property
    def level_swing(self) -> float:
        return self.level_high - self.level_low

    @property
    def share_swing(self) -> float:
        return self.share_high - self.share_low

    @property
    def t_swing(self) -> float:
        return abs(self.t_high - self.t_low)

    @property
    def identity_holds(self) -> bool:
        """ttf* = lambda·(coal_th + EUA·EF_coal) − EUA·EF_gas, to machine precision."""
        return self.affine_residual < 1e-9

    @property
    def headline(self) -> str:
        return (
            f"The efficiency pair moves the switching level from {self.level_low:.0f} to "
            f"{self.level_high:.0f} EUR/MWh and the share of days above it from "
            f"{self.share_low:.0%} to {self.share_high:.0%} — the same prices, the "
            f"opposite diagnosis. It moves the t-statistic of the prediction by "
            f"{self.t_swing:.2f}. The nine distance measures correlate "
            f"{self.min_pairwise_corr:.3f} or better: they are one variable under an "
            "affine map, and a t-statistic is invariant under one. The efficiencies "
            "decide where the line is and nothing about what it predicts."
        )


def efficiency_invariance(
    frame: pd.DataFrame,
    *,
    coal_efficiencies: tuple[float, ...] = (0.36, 0.38, 0.42),
    gas_efficiencies: tuple[float, ...] = (0.50, 0.55, 0.60),
    horizon_days: int = FORWARD_HORIZON_DAYS,
    phase: int = 0,
) -> EfficiencyInvariance:
    """Sweep the efficiency grid and measure the level, the diagnosis and the prediction.

    Also verifies the algebraic identity that explains the result, rather than asserting
    it: `ttf* = lambda·(coal_th + EUA·EF_coal) − EUA·EF_gas` with `lambda = eta_g/eta_c`.
    """
    rows, distances, affine_residual = [], {}, 0.0
    for coal_efficiency in coal_efficiencies:
        for gas_efficiency in gas_efficiencies:
            switch = switch_ttf_eur_mwh(
                frame, coal_efficiency=coal_efficiency, gas_efficiency=gas_efficiency
            )
            lam = gas_efficiency / coal_efficiency
            closed_form = (
                lam * (frame["coal_eur_mwh_th"] + frame["eua_eur_t"] * EF_COAL_T_PER_MWH_TH)
                - frame["eua_eur_t"] * EF_GAS_T_PER_MWH_TH
            )
            affine_residual = max(affine_residual, float((switch - closed_form).abs().max()))

            distance = switching_distance_pct(frame, switch)
            sample = predictive_sample(frame, switch, horizon_days=horizon_days, phase=phase)
            result = ols(sample["forward_return"], {"distance": sample["distance"]})
            key = (coal_efficiency, gas_efficiency)
            distances[key] = distance
            rows.append(
                {
                    "coal_efficiency": coal_efficiency,
                    "gas_efficiency": gas_efficiency,
                    "switch_median_eur_mwh": float(switch.median()),
                    "share_above": float((distance > 0).mean()),
                    "beta": result.coefficients["distance"],
                    "t_stat": result.t_stats["distance"],
                }
            )

    grid = pd.DataFrame(rows)
    corr = pd.DataFrame(distances).dropna().corr().to_numpy()
    off_diagonal = corr[np.triu_indices_from(corr, 1)]
    return EfficiencyInvariance(
        grid=grid,
        level_low=float(grid["switch_median_eur_mwh"].min()),
        level_high=float(grid["switch_median_eur_mwh"].max()),
        share_low=float(grid["share_above"].min()),
        share_high=float(grid["share_above"].max()),
        t_low=float(grid["t_stat"].min()),
        t_high=float(grid["t_stat"].max()),
        max_pairwise_corr=float(off_diagonal.max()),
        min_pairwise_corr=float(off_diagonal.min()),
        affine_residual=affine_residual,
    )


@dataclass(frozen=True)
class AnchorEncompassing:
    """Does the switching arithmetic beat raw thermal parity? The honest negative.

    `increment_f` is an F-test on what the full switching distance adds once the naive
    anchor is already in the regression. A large p-value is the finding, not a failure:
    it says the sample cannot tell the elaborate calculation from the crude one.
    """

    full_only: OLSResult
    naive_only: OLSResult
    both: OLSResult
    increment_f: float
    increment_p: float
    regressor_corr: float

    @property
    def full_adds_nothing(self) -> bool:
        return self.increment_p > 0.05

    @property
    def headline(self) -> str:
        if self.full_adds_nothing:
            return (
                f"Raw thermal parity — no efficiencies, no carbon price — reaches "
                f"R² = {self.naive_only.r_squared:.3f} against {self.full_only.r_squared:.3f} "
                f"for the full switching level. Adding the full level on top of the naive "
                f"anchor gives F = {self.increment_f:.2f} (p = {self.increment_p:.2f}): the "
                f"sample cannot distinguish them, and the two regressors correlate "
                f"{self.regressor_corr:.2f}. The switching arithmetic is not what is doing "
                "the predicting."
            )
        return (
            f"The full switching level adds significantly over raw thermal parity "
            f"(F = {self.increment_f:.2f}, p = {self.increment_p:.3f}) — the arithmetic "
            "earns its place."
        )


def anchor_encompassing(
    frame: pd.DataFrame,
    switch_ttf: pd.Series,
    *,
    horizon_days: int = FORWARD_HORIZON_DAYS,
    phase: int = 0,
) -> AnchorEncompassing:
    """Race the switching level against an anchor with none of its economics in it."""
    from scipy import stats as scipy_stats

    naive = naive_thermal_anchor(frame)
    forward_return = np.log(
        frame["ttf_eur_mwh"].shift(-horizon_days) / frame["ttf_eur_mwh"]
    )
    sample = pd.concat(
        {
            "forward_return": forward_return,
            "full": switching_distance_pct(frame, switch_ttf),
            "naive": (frame["ttf_eur_mwh"] - naive) / naive,
        },
        axis=1,
    ).dropna().iloc[phase::horizon_days]

    full_only = ols(sample["forward_return"], {"distance": sample["full"]})
    naive_only = ols(sample["forward_return"], {"distance": sample["naive"]})
    both = ols(
        sample["forward_return"], {"full": sample["full"], "naive": sample["naive"]}
    )
    n = both.n_obs
    increment_f = ((both.r_squared - naive_only.r_squared) / 1.0) / (
        (1.0 - both.r_squared) / (n - 3)
    )
    return AnchorEncompassing(
        full_only=full_only,
        naive_only=naive_only,
        both=both,
        increment_f=float(increment_f),
        increment_p=float(1.0 - scipy_stats.f.cdf(increment_f, 1, n - 3)),
        regressor_corr=float(sample["full"].corr(sample["naive"])),
    )


# ===========================================================================
# Sub-period stability — where the evidence actually comes from
# ===========================================================================
DEFAULT_SUBPERIODS: tuple[tuple[str, str, str], ...] = (
    ("2018-2020 pre-crisis", "2018", "2020"),
    ("2021-2022 crisis", "2021", "2022"),
    ("2023-2026 after", "2023", "2026"),
)


def subperiod_stability(
    frame: pd.DataFrame,
    switch_ttf: pd.Series,
    *,
    periods: tuple[tuple[str, str, str], ...] = DEFAULT_SUBPERIODS,
    horizon_days: int = FORWARD_HORIZON_DAYS,
    phase: int = 0,
    min_obs: int = 12,
) -> pd.DataFrame:
    """The same regression period by period.

    On 110 windows this is descriptive rather than a formal stability test — with 26 to
    45 observations per period, differences of this size are not separately identified.
    It is reported because knowing the evidence is concentrated in the crisis changes how
    much weight the headline deserves.
    """
    sample = predictive_sample(frame, switch_ttf, horizon_days=horizon_days, phase=phase)
    rows = []
    for label, start, end in periods:
        window = sample.loc[start:end]
        if len(window) < min_obs:
            continue
        result = ols(window["forward_return"], {"distance": window["distance"]})
        rows.append(
            {
                "period": label,
                "n": result.n_obs,
                "beta": result.coefficients["distance"],
                "t_stat": result.t_stats["distance"],
                "r_squared": result.r_squared,
            }
        )
    if not rows:
        raise ValueError("no sub-period has enough non-overlapping windows to regress")
    return pd.DataFrame(rows).set_index("period")


# ===========================================================================
# THE TRADE — what a dual-fuel generator owns, and the hedge that vanished
# ===========================================================================
# A generator holding both a coal unit and a CCGT owns an option to switch between them.
# An option is worth its VOLATILITY, not its level — so what prices that asset is the
# volatility of the coal-minus-gas generation spread, and that volatility has a term in it
# that almost nobody re-estimates.
#
# Writing the spread out, the two market legs enter with OPPOSITE SIGNS:
#
#     spread = coal_cost - gas_cost
#            = const - (1/eta_gas) x TTF + (EF_coal/eta_coal - EF_gas/eta_gas) x EUA
#
# so
#
#     Var(spread) = (b_ttf x sigma_ttf)^2 + (b_eua x sigma_eua)^2
#                   + 2 x b_ttf x b_eua x rho x sigma_ttf x sigma_eua
#                                          ^^^ b_ttf < 0 < b_eua, so a POSITIVE rho
#                                              SUBTRACTS from the variance
#
# Carbon is a natural hedge for gas inside this particular spread. That is algebra, not an
# empirical regularity: across the whole plausible efficiency grid b_ttf runs -2.000 to
# -1.667 and b_eua runs +0.410 to +0.611, and the signs never cross.
#
# WHAT THE HEDGE WAS WORTH, AND WHAT HAPPENED TO IT. From 2018 to 2025 the gas-carbon
# correlation ran +0.30 to +0.70 and removed 9 % to 29 % of the spread's volatility. In
# 2026 it collapsed to zero (95 % interval spanning zero) and the dampening went with it.
#
# THE PART WORTH TRADING IS NOT THE BIGGEST PART. Spread volatility rose 175 % into 2026,
# and 84 % of that is simply TTF's own volatility doubling — which is in every market
# report. Only 16 % is the lost correlation hedge. But a risk model carrying rho at its
# historical +0.4 misses exactly that 16 %: it understates spread volatility by ~20 % and
# an at-the-money switching option by ~11 %, with nothing anywhere flagging an error.

SHOCK_COUNT_FOR_TRANSMISSION = 10


def spread_betas(
    *,
    coal_efficiency: float = DEFAULT_COAL_EFFICIENCY,
    gas_efficiency: float = DEFAULT_GAS_EFFICIENCY,
) -> tuple[float, float]:
    """(b_ttf, b_eua): the spread's exact sensitivities to the two market legs.

    Derived by differentiating `generation_cost_eur_mwh_e`'s spread, so these are
    definitional rather than estimated — there is no residual and no standard error.
    """
    for efficiency, label in ((coal_efficiency, "coal"), (gas_efficiency, "gas")):
        if not 0.20 < efficiency < 0.70:
            raise ValueError(f"{label} efficiency outside the plausible range: {efficiency}")
    b_ttf = -1.0 / gas_efficiency
    b_eua = EF_COAL_T_PER_MWH_TH / coal_efficiency - EF_GAS_T_PER_MWH_TH / gas_efficiency
    return b_ttf, b_eua


@dataclass(frozen=True)
class DampeningYear:
    """One year of the natural hedge: the spread's volatility with and without rho."""

    year: int
    rho: float
    vol_actual: float
    vol_if_independent: float
    n_obs: int

    @property
    def dampening(self) -> float:
        """Negative means the correlation REMOVED volatility from the spread."""
        if self.vol_if_independent <= 0:
            return float("nan")
        return self.vol_actual / self.vol_if_independent - 1.0


@dataclass(frozen=True)
class NaturalHedge:
    """The carbon-hedges-gas effect, measured year by year on the real spread."""

    years: tuple[DampeningYear, ...]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "year": y.year,
                    "rho": y.rho,
                    "vol_actual": y.vol_actual,
                    "vol_if_independent": y.vol_if_independent,
                    "dampening": y.dampening,
                    "n": y.n_obs,
                }
                for y in self.years
            ]
        ).set_index("year")

    @property
    def hedged_years(self) -> tuple[DampeningYear, ...]:
        """Years where the correlation was meaningfully positive — the hedge working."""
        return tuple(y for y in self.years if y.rho > 0.2)

    @property
    def typical_dampening(self) -> float:
        hedged = self.hedged_years
        if not hedged:
            return float("nan")
        return float(np.median([y.dampening for y in hedged]))

    @property
    def latest(self) -> DampeningYear:
        return self.years[-1]

    @property
    def headline(self) -> str:
        hedged = self.hedged_years
        worst = min(y.dampening for y in hedged)
        return (
            f"In the {len(hedged)} years where the gas-carbon correlation was positive it "
            f"removed a median {abs(self.typical_dampening):.0%} of the switching spread's "
            f"volatility, and as much as {abs(worst):.0%}. In "
            f"{self.latest.year} the correlation is {self.latest.rho:+.2f} and the "
            f"dampening is {self.latest.dampening:+.0%} — the hedge is gone."
        )


def natural_hedge(
    frame: pd.DataFrame,
    *,
    coal_efficiency: float = DEFAULT_COAL_EFFICIENCY,
    gas_efficiency: float = DEFAULT_GAS_EFFICIENCY,
    min_obs: int = 50,
    trading_days: int = 252,
) -> NaturalHedge:
    """Year-by-year volatility of the switching spread, with and without the correlation.

    `vol_if_independent` is the counterfactual where the two legs move with the same
    individual volatilities but no correlation at all — so the gap between the two columns
    is exactly what the carbon leg was contributing as a hedge.
    """
    b_ttf, b_eua = spread_betas(
        coal_efficiency=coal_efficiency, gas_efficiency=gas_efficiency
    )
    changes = frame[["ttf_eur_mwh", "eua_eur_t"]].diff().dropna()
    scale = float(np.sqrt(trading_days))

    years = []
    for year, group in changes.groupby(changes.index.year):
        if len(group) < min_obs:
            continue
        sigma_t = float(group["ttf_eur_mwh"].std())
        sigma_e = float(group["eua_eur_t"].std())
        rho = float(group["ttf_eur_mwh"].corr(group["eua_eur_t"]))
        var_independent = (b_ttf * sigma_t) ** 2 + (b_eua * sigma_e) ** 2
        var_actual = var_independent + 2 * b_ttf * b_eua * rho * sigma_t * sigma_e
        years.append(
            DampeningYear(
                year=int(year),
                rho=rho,
                vol_actual=float(np.sqrt(max(var_actual, 0.0)) * scale),
                vol_if_independent=float(np.sqrt(var_independent) * scale),
                n_obs=int(len(group)),
            )
        )
    if len(years) < 2:
        raise ValueError("need at least two years to measure the hedge")
    return NaturalHedge(years=tuple(years))


@dataclass(frozen=True)
class DampeningAttribution:
    """Split a change in spread volatility into individual vols and the correlation.

    The counterfactual holds the destination year's individual volatilities and swaps in
    the origin year's correlation, so `correlation_part` is what the lost hedge cost on
    its own — the term a risk model carrying a historical rho would silently miss.
    """

    year_from: int
    year_to: int
    vol_from: float
    vol_to: float
    vol_counterfactual: float
    rho_from: float
    rho_to: float

    @property
    def total_change(self) -> float:
        return self.vol_to - self.vol_from

    @property
    def volatility_part(self) -> float:
        return self.vol_counterfactual - self.vol_from

    @property
    def correlation_part(self) -> float:
        return self.vol_to - self.vol_counterfactual

    @property
    def correlation_share(self) -> float:
        return self.correlation_part / self.total_change if self.total_change else float("nan")

    @property
    def option_value_uplift(self) -> float:
        """An ATM option is near-linear in volatility, so this is the mispricing a stale
        correlation produces on a switching option."""
        if self.vol_counterfactual <= 0:
            return float("nan")
        return self.vol_to / self.vol_counterfactual - 1.0

    @property
    def headline(self) -> str:
        return (
            f"Spread volatility went from {self.vol_from:.0f} to {self.vol_to:.0f} EUR/MWh "
            f"between {self.year_from} and {self.year_to}, a {self.vol_to / self.vol_from - 1:+.0%} "
            f"move. Individual volatilities account for {self.volatility_part:+.0f} of it "
            f"({1 - self.correlation_share:.0%}) and the collapsed correlation for "
            f"{self.correlation_part:+.0f} ({self.correlation_share:.0%}). The second term "
            f"is the smaller one and the only one a stale rho hides: it lifts an ATM "
            f"switching option by {self.option_value_uplift:+.0%} with nothing flagging it."
        )


def dampening_attribution(
    frame: pd.DataFrame,
    *,
    year_from: int,
    year_to: int,
    coal_efficiency: float = DEFAULT_COAL_EFFICIENCY,
    gas_efficiency: float = DEFAULT_GAS_EFFICIENCY,
    trading_days: int = 252,
) -> DampeningAttribution:
    """How much of a change in spread volatility is the correlation rather than the legs."""
    b_ttf, b_eua = spread_betas(
        coal_efficiency=coal_efficiency, gas_efficiency=gas_efficiency
    )
    changes = frame[["ttf_eur_mwh", "eua_eur_t"]].diff().dropna()
    scale = float(np.sqrt(trading_days))

    def stats(year: int) -> tuple[float, float, float]:
        group = changes[changes.index.year == year]
        if len(group) < 30:
            raise ValueError(f"only {len(group)} observations in {year}")
        return (
            float(group["ttf_eur_mwh"].std()),
            float(group["eua_eur_t"].std()),
            float(group["ttf_eur_mwh"].corr(group["eua_eur_t"])),
        )

    def vol(sigma_t: float, sigma_e: float, rho: float) -> float:
        variance = (
            (b_ttf * sigma_t) ** 2
            + (b_eua * sigma_e) ** 2
            + 2 * b_ttf * b_eua * rho * sigma_t * sigma_e
        )
        return float(np.sqrt(max(variance, 0.0)) * scale)

    sigma_t_from, sigma_e_from, rho_from = stats(year_from)
    sigma_t_to, sigma_e_to, rho_to = stats(year_to)
    return DampeningAttribution(
        year_from=year_from,
        year_to=year_to,
        vol_from=vol(sigma_t_from, sigma_e_from, rho_from),
        vol_to=vol(sigma_t_to, sigma_e_to, rho_to),
        vol_counterfactual=vol(sigma_t_to, sigma_e_to, rho_from),
        rho_from=rho_from,
        rho_to=rho_to,
    )


@dataclass(frozen=True)
class TransmissionTest:
    """Do the biggest gas shocks of each year reach the carbon price at all?

    Deliberately non-parametric: take each year's largest moves in TTF and count how often
    EUA moved the same way that day. A correlation can be dragged around by quiet days;
    this asks only about the days the switching channel would have to be working.
    """

    table: pd.DataFrame  # index=year, columns=same_sign, n_shocks, mean_abs_ttf
    n_shocks: int

    @property
    def latest_year(self) -> int:
        return int(self.table.index.max())

    @property
    def latest_same_sign(self) -> int:
        return int(self.table.loc[self.latest_year, "same_sign"])

    @property
    def normal_years(self) -> pd.DataFrame:
        return self.table[self.table["same_sign"] >= 7]

    @property
    def headline(self) -> str:
        normal = self.normal_years
        return (
            f"On each year's {self.n_shocks} largest gas moves, carbon moved the same way "
            f"{normal['same_sign'].min():.0f} to {normal['same_sign'].max():.0f} times in "
            f"{len(normal)} of the {len(self.table)} years. In {self.latest_year} it moved "
            f"the same way {self.latest_same_sign} times. A crisis-sized gas shock happened "
            "and the carbon market did not take it."
        )


def transmission_test(
    frame: pd.DataFrame,
    *,
    n_shocks: int = SHOCK_COUNT_FOR_TRANSMISSION,
    min_obs: int = 50,
) -> TransmissionTest:
    """Count same-direction responses on each year's largest gas shocks."""
    returns = np.log(frame[["ttf_eur_mwh", "eua_eur_t"]]).diff().dropna()
    rows = []
    for year, group in returns.groupby(returns.index.year):
        if len(group) < min_obs:
            continue
        largest = group.reindex(
            group["ttf_eur_mwh"].abs().sort_values(ascending=False).index
        ).head(n_shocks)
        rows.append(
            {
                "year": int(year),
                "same_sign": int(
                    ((largest["ttf_eur_mwh"] * largest["eua_eur_t"]) > 0).sum()
                ),
                "n_shocks": int(len(largest)),
                "mean_abs_ttf": float(largest["ttf_eur_mwh"].abs().mean()),
            }
        )
    if not rows:
        raise ValueError("no year has enough observations for the transmission test")
    return TransmissionTest(
        table=pd.DataFrame(rows).set_index("year"), n_shocks=n_shocks
    )


def switching_depth_profile(
    frame: pd.DataFrame,
    *,
    coal_efficiency: float = DEFAULT_COAL_EFFICIENCY,
    gas_efficiency: float = DEFAULT_GAS_EFFICIENCY,
    deep_threshold: float = 0.40,
    min_obs: int = 50,
) -> pd.DataFrame:
    """How far above the switching level each year sat — the saturation story's own test.

    Saturation is the obvious explanation for a broken gas-carbon link: if every coal unit
    is already running, more expensive gas cannot start another one. This table is what
    rules it out. A year can only be saturated if it sat DEEP above the switching level,
    and comparing years with the same depth but different correlations settles it.
    """
    switch = switch_ttf_eur_mwh(
        frame, coal_efficiency=coal_efficiency, gas_efficiency=gas_efficiency
    )
    distance = switching_distance_pct(frame, switch)
    changes = frame[["ttf_eur_mwh", "eua_eur_t"]].diff()
    joined = pd.concat(
        {"distance": distance, "dttf": changes["ttf_eur_mwh"], "deua": changes["eua_eur_t"]},
        axis=1,
    ).dropna()

    rows = []
    for year, group in joined.groupby(joined.index.year):
        if len(group) < min_obs:
            continue
        rows.append(
            {
                "year": int(year),
                "share_above": float((group["distance"] > 0).mean()),
                "median_distance": float(group["distance"].median()),
                "share_deep": float((group["distance"] > deep_threshold).mean()),
                "rho": float(group["dttf"].corr(group["deua"])),
                "n": int(len(group)),
            }
        )
    return pd.DataFrame(rows).set_index("year")
