"""Project B — the switching ceiling predicts, and the efficiencies it is built on do not.

THESIS
------
Every European power desk knows the shape of the idea: when gas gets expensive enough
relative to coal-plus-carbon, generators burn coal instead, gas demand eases, and TTF's
upside is capped. The belief is old. Two things about it are not:

1. **The level is not identified.** The switching price is a property of two plant
   efficiencies no exchange quotes. Across the plausible pair, the level swings from
   ~30 to ~47 EUR/MWh and the share of days sitting "above the switch" swings from 16%
   to 75% — the same fuel prices, the same carbon price, and the opposite diagnosis.

2. **The prediction is identified, and it does not need the level.** Distance from the
   switching level does predict TTF's next 20 days. But the t-statistic is invariant
   across that entire efficiency grid (−3.02 to −3.05), and the elaborate calculation
   adds nothing measurable over a naive thermal-parity anchor that has no efficiencies
   and no carbon price in it at all (F = 1.17, p = 0.28).

**Those two facts are the same fact, and the reason is algebraic rather than empirical.**
Writing λ = η_gas/η_coal, the switching level collapses to

    ttf* = λ·(coal_th + EUA·EF_coal) − EUA·EF_gas

which is *affine in λ* — verified to machine precision in `efficiency_invariance()`. The
nine distance measures across the grid correlate 0.997 to 1.000, and an OLS t-statistic
is invariant under affine rescaling of its regressor. So the efficiencies decide
everything about **where the line sits** and nothing about **what the line predicts**.
That is not luck, and it cannot be fixed by better efficiency estimates.

WHAT SURVIVES AS A RESULT, AND AT WHAT STRENGTH
------------------------------------------------
The predictive claim is fragile in exactly the way predictive regressions usually are,
and the page reports the corrections rather than the raw t-stat:

* **Stambaugh bias is material and points the same way as the finding.** The regressor
  contains TTF in its numerator, so its innovations correlate +0.78 with the return
  being predicted, and it is persistent (ρ = 0.79). That biases OLS *downward* — toward
  the negative coefficient this page reports. `stambaugh_diagnostics()` measures it at
  13% of the coefficient; a Nelson–Kim bootstrap under H0 puts the honest p-value at
  **0.018**, against ~0.001 read naively off the t-stat. An order of magnitude of the
  apparent significance was bias.
* **It is not an artefact of where the non-overlapping sample starts.** All twenty
  phases of `iloc[::20]` give a negative coefficient and nineteen are significant.
* **It is asymmetric in the direction the mechanism requires** (`asymmetry_test()`):
  above the switch, β = −0.185 (t = −2.62); below it, nothing (t = −0.65). Generic mean
  reversion pulls symmetrically from both sides and cannot produce that.
* **It is concentrated in the crisis and after** — 2018-2020 alone gives t = −1.13. On
  110 non-overlapping windows, a large part of the evidence is the 2022 episode.

So: the ceiling is real, weaker than it first looks, mechanism-shaped in its asymmetry,
and **not evidence for the specific switching arithmetic** — only for gas being dear
against coal in raw thermal terms.

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
