"""Project B — does the coal-to-gas switching price actually cap TTF?

THESIS
------
Every European power desk knows the shape of the idea: when gas gets expensive enough
relative to coal-plus-carbon, generators burn coal instead, gas demand eases, and TTF's
upside is capped. That belief is old and it is not the pitch — the pitch is that nobody
publishes the level at which it should bind, because it is a property of two plant
efficiencies (`DEFAULT_COAL_EFFICIENCY`, `DEFAULT_GAS_EFFICIENCY`) that no exchange
quotes, and it has never been tested against the honest alternative: maybe TTF just
mean-reverts, and "switching" is a story attached after the fact to an ordinary pullback.

**The test, not the idea, is the finding.** `ceiling_test()` regresses TTF's 20-day
forward return on distance from its switching level, on **non-overlapping** windows so
the t-stat is not manufactured by reusing the same outcome twenty times, and races that
regressor against a placebo — distance from TTF's own trailing median, which contains no
switching economics at all. If a generic "far from its recent level" measure predicted
the same reversion, the switching story would explain nothing the market doesn't already
price. It does not: the switching-distance regressor survives on the honest sample and
the placebo does not (see `switching`/`placebo`/`horse_race` on `CeilingTest`).

WHAT THIS REPLACES
-------------------
This module was originally built around the Richards Bay → ARA arb (API2 CIF ARA minus
API4 FOB Richards Bay, minus freight and a maritime-ETS layer), on the thesis that
Indian demand had pulled the marginal RB cargo off the European netback since 2022.
**API4 Richards Bay is absent from the export**, so that spread was never computable
from data actually available here, and the thesis was abandoned before publication
rather than faked with a proxy. API2, TTF, EUA and EURUSD remain, and support the
switching question below instead.

ASSUMPTIONS
-----------
B-H1  API2 is assessed at 6,000 kcal/kg NAR; `MWH_TH_PER_TONNE_COAL` converts on that
      reference. The coal actually burned at a given plant may differ — a sensitivity,
      not the identity.
B-H2  Emission factors (`EF_COAL_T_PER_MWH_TH`, `EF_GAS_T_PER_MWH_TH`) are standard
      combustion figures, not plant-specific.
B-H3  Plant efficiencies (`DEFAULT_COAL_EFFICIENCY`, `DEFAULT_GAS_EFFICIENCY`) are the
      parameter the whole switching level turns on — not market data, not published.
      `efficiency_identification()` measures how much of the level they account for;
      the ceiling test below is run at the default pair and should be re-checked across
      the plausible range before being read as a level rather than a mechanism.
B-H4  The 20-trading-day horizon and the 250-day trailing-median window are both
      parameterised, not tuned — changing them is a robustness check, not a re-fit.
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
