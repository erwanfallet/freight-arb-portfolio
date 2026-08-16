"""Project E — the buyer does not pay the freight, and the index says so.

THESIS
------
The 62% Fe index is quoted **CFR China**: it is a delivered price, so it already
contains the freight the seller paid. That makes the incidence question a regression
coefficient rather than a modelling exercise:

    CFR = FOB + freight        =>        d(CFR)/d(freight) IS the incidence

A coefficient of 1 means the Chinese mill pays every dollar of freight. A coefficient of
0 means the miner absorbs it into its netback and the delivered price never moves.

WHAT THE DATA SAYS
------------------
At daily and weekly horizons, where this sample has power, **full pass-through is
rejected**:

    daily    b = -0.06  [-0.48 ; +0.37]   n = 882   b=1 rejected
    weekly   b = -0.26  [-0.97 ; +0.46]   n = 186   b=1 rejected
    monthly  b = +0.62  [-0.86 ; +2.10]   n =  43   nothing rejected
    quarterly b = +0.55 [-1.34 ; +2.44]   n =  14   nothing rejected

The point estimate rises with the horizon, which is what a contracting story would
predict — and the sample is far too short to say whether that is real. Three and a half
years of IODEX leaves fourteen quarterly observations.

WHY A NULL IS A RESULT HERE AND NOT A WEAK TEST
-----------------------------------------------
The objection to any null is that the test could not have seen the effect. That is
answerable rather than arguable. If the CFR were the FOB plus a freight term, and the
FOB moved independently of freight, the correlation between their daily changes would
be exactly the ratio of their volatilities:

    corr(dCFR, dFreight) = sigma(dFreight) / sigma(dCFR) = 0.296 / 1.643 = 0.18

against a Bartlett band of 0.066. The test can see full pass-through at better than two
and a half times its own noise floor. Observing nothing is therefore information.

THE SEPARATE FACT THAT GETS CONFLATED WITH IT
---------------------------------------------
Freight is about 10% of the delivered value of iron ore and about 3% of its variance.
Those are different statements and only the first is usually made. Even at 100%
incidence, freight would explain almost none of what the CFR does day to day. "Who pays
the freight" and "what moves the price" are different questions, and a freight term that
is economically large can be statistically invisible.

ASSUMPTIONS
-----------
E-H1  C5 (W Australia -> Qingdao) is the freight route matching the 62% Fe index, which
      is predominantly Australian. The 65% index is predominantly Brazilian and belongs
      with C3 — which is monthly in this export and cannot support a daily test.
E-H2  The C5 series is a **front-month FFA**, not the spot route assessment. A basis
      exists between them (inherited from project A's A-H3) and it attenuates any
      measured relationship toward zero. The bias therefore runs toward the null found
      here, which is the wrong direction for the conclusion and is stated rather than
      buried.
E-H3  The lag between fixing freight and the cargo being assessed at destination is
      **predicted from the voyage model**, not fitted: Port Hedland -> Qingdao at the
      reference speed plus port days. Scanning for the lag that maximises the
      correlation would manufacture one.
E-H4  Prices are differenced before any regression. Two non-stationary levels produce a
      flattering coefficient that describes a shared trend, not an incidence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from agri.core.stats import ccf_with_band, hac_ols
from agri.core.voyage import HOURS_PER_DAY, VoyageParams
from agri.data.snapshot import cached

# The reader and the sheet names live in project A's module: this page uses the same two
# raw sheets, and duplicating the reader would let the two drift apart.
from freight.chains.ironore import C5_SHEET, IODEX_62_SHEET, _read_sheet
from freight.voyage.distances import route_distance_nm

# E-H1 — the route that serves the index being tested.
LOADPORT = "PORT_HEDLAND"
DISCHARGE = "QINGDAO"

FREQUENCIES: tuple[tuple[str, str | None], ...] = (
    ("daily", None),
    ("weekly", "W"),
    ("monthly", "ME"),
    ("quarterly", "QE"),
)

# Below this many differenced observations, a coefficient is reported but no verdict is
# drawn from it — fourteen quarters cannot reject anything.
MIN_OBS_FOR_VERDICT = 60


class IncidenceError(ValueError):
    """Mis-specified incidence test — always a caller error."""


# ===========================================================================
# Data
# ===========================================================================
@cached("e_freight_incidence")
def load_incidence_frame(start: str | None = None) -> pd.DataFrame:
    """The delivered price and the freight that is inside it, on their common calendar.

    Columns: cfr62 (USD/dmt, CFR China), c5 (USD/wmt, W Australia -> Qingdao).

    Deliberately left in their native units. The moisture conversion between wet and dry
    tonnes matters for a decomposition of the *level* — project A does it — but the
    incidence test runs on changes, where a constant conversion factor scales the
    coefficient by 1/(1-moisture) and changes no verdict. Applying it here would imply a
    precision the test does not have.
    """
    cfr = _read_sheet(IODEX_62_SHEET).rename("cfr62")
    c5 = _read_sheet(C5_SHEET).rename("c5")
    frame = pd.concat([cfr, c5], axis=1, sort=True).dropna()
    if start is not None:
        frame = frame[frame.index >= pd.Timestamp(start)]
    if frame.empty:
        raise IncidenceError(f"no common dates between the CFR index and C5 after {start}")
    return frame


def freight_share_of_value(frame: pd.DataFrame) -> dict[str, float]:
    """The two statements that get conflated: share of the level, share of the variance.

    Freight is a tenth of what a delivered tonne costs and a thirtieth of why its price
    moved today. Both are true and they answer different questions.
    """
    changes = frame.diff().dropna()
    return {
        "share_of_level": float(frame["c5"].median() / frame["cfr62"].median()),
        "sigma_freight": float(changes["c5"].std()),
        "sigma_cfr": float(changes["cfr62"].std()),
        "share_of_variance": float(changes["c5"].var() / changes["cfr62"].var()),
    }


# ===========================================================================
# E-H3 — the lag, predicted rather than fitted
# ===========================================================================
def predicted_lag_days(*, params: VoyageParams | None = None) -> dict[str, float]:
    """Days between fixing the freight and the cargo being assessed at destination.

    Computed from the voyage, not chosen from the data. A lag picked by scanning for the
    largest correlation is a lag the search produced; this one is falsifiable before the
    test is run.
    """
    params = params or VoyageParams()
    distance = route_distance_nm(LOADPORT, DISCHARGE)
    sea_days = distance / (params.speed_laden_kn * HOURS_PER_DAY)
    total = sea_days + params.port_days
    return {
        "distance_nm": float(distance),
        "sea_days": float(sea_days),
        "port_days": float(params.port_days),
        "calendar_days": float(total),
        "business_days": float(round(total * 5.0 / 7.0)),
    }


@dataclass(frozen=True)
class LagScan:
    """Cross-correlation of freight changes against delivered-price changes."""

    lags: np.ndarray
    values: np.ndarray
    band: float
    n_eff: float
    predicted_lag: int

    @property
    def significant_lags(self) -> list[int]:
        return [int(k) for k, v in zip(self.lags, self.values) if abs(v) > self.band]

    @property
    def value_at_predicted(self) -> float:
        idx = int(np.argmin(np.abs(self.lags - self.predicted_lag)))
        return float(self.values[idx])

    @property
    def headline(self) -> str:
        if not self.significant_lags:
            return (
                f"No lag between -{int(self.lags.max())} and +{int(self.lags.max())} days "
                f"carries a correlation outside the ±{self.band:.3f} band, including the "
                f"{self.predicted_lag}-day lag the voyage predicts "
                f"({self.value_at_predicted:+.3f}). The absence is not a timing problem."
            )
        return (
            f"Significant lags: {self.significant_lags}. The voyage predicts "
            f"{self.predicted_lag} days, where the correlation is "
            f"{self.value_at_predicted:+.3f}."
        )


def lag_scan(frame: pd.DataFrame, *, max_lag: int = 30) -> LagScan:
    """Does freight lead the delivered price at the horizon the voyage implies?

    Sign convention, stated because it inverts between packages: the value at lag k is
    corr(freight[t], cfr[t+k]), so a positive lag means **freight leads**.
    """
    changes = frame.diff().dropna()
    out = ccf_with_band(changes["c5"], changes["cfr62"], max_lag=max_lag)
    return LagScan(
        lags=out.lags,
        values=out.values,
        band=out.band,
        n_eff=out.n_eff,
        predicted_lag=int(round(predicted_lag_days()["calendar_days"])),
    )


# ===========================================================================
# The power benchmark — what full pass-through would have looked like
# ===========================================================================
@dataclass(frozen=True)
class PowerBenchmark:
    """The correlation a coefficient of 1 would produce, against the noise floor."""

    implied_correlation: float
    band: float

    @property
    def can_detect(self) -> bool:
        return self.implied_correlation > self.band

    @property
    def margin(self) -> float:
        return self.implied_correlation / self.band if self.band > 0 else float("nan")

    @property
    def headline(self) -> str:
        verdict = "can" if self.can_detect else "cannot"
        return (
            f"Full pass-through would show up as a correlation of "
            f"{self.implied_correlation:.3f}, against a significance band of "
            f"{self.band:.3f}: the test {verdict} see it, at {self.margin:.1f}x the noise "
            "floor. A null is therefore a finding rather than a weak test."
        )


def power_benchmark(frame: pd.DataFrame, *, max_lag: int = 30) -> PowerBenchmark:
    """If the CFR were FOB plus freight, how correlated would their changes be?

    Under `dCFR = dFOB + dFreight` with `dFOB` independent of freight, the correlation is
    exactly `sigma(dFreight) / sigma(dCFR)`. No estimation is involved: it follows from
    the identity, which is what makes it a legitimate benchmark rather than a guess.
    """
    changes = frame.diff().dropna()
    implied = float(changes["c5"].std() / changes["cfr62"].std())
    band = ccf_with_band(changes["c5"], changes["cfr62"], max_lag=max_lag).band
    return PowerBenchmark(implied_correlation=implied, band=float(band))


# ===========================================================================
# THE TEST
# ===========================================================================
@dataclass(frozen=True)
class IncidenceEstimate:
    """One frequency's answer to "who pays the freight"."""

    label: str
    beta: float
    std_error: float
    n_obs: int
    implied_correlation: float

    @property
    def ci(self) -> tuple[float, float]:
        return (self.beta - 1.96 * self.std_error, self.beta + 1.96 * self.std_error)

    @property
    def has_power(self) -> bool:
        return self.n_obs >= MIN_OBS_FOR_VERDICT

    @property
    def rejects_full_passthrough(self) -> bool:
        lo, hi = self.ci
        return self.has_power and not (lo <= 1.0 <= hi)

    @property
    def rejects_absorption(self) -> bool:
        lo, hi = self.ci
        return self.has_power and not (lo <= 0.0 <= hi)


@dataclass(frozen=True)
class IncidenceResult:
    """The frequency ladder, and what survives it."""

    estimates: tuple[IncidenceEstimate, ...]

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for e in self.estimates:
            lo, hi = e.ci
            rows.append(
                {
                    "frequency": e.label,
                    "n": e.n_obs,
                    "beta": e.beta,
                    "std_error": e.std_error,
                    "ci_low": lo,
                    "ci_high": hi,
                    "rejects_full_passthrough": e.rejects_full_passthrough,
                    "rejects_absorption": e.rejects_absorption,
                    "has_power": e.has_power,
                }
            )
        return pd.DataFrame(rows).set_index("frequency")

    @property
    def powered(self) -> tuple[IncidenceEstimate, ...]:
        return tuple(e for e in self.estimates if e.has_power)

    @property
    def full_passthrough_rejected_everywhere_with_power(self) -> bool:
        powered = self.powered
        return bool(powered) and all(e.rejects_full_passthrough for e in powered)

    @property
    def headline(self) -> str:
        powered = self.powered
        if not powered:
            return "No frequency in this sample carries enough observations for a verdict."
        labels = " and ".join(e.label for e in powered)
        beta_range = ", ".join(f"{e.beta:+.2f}" for e in powered)
        if self.full_passthrough_rejected_everywhere_with_power:
            return (
                f"At {labels} frequency the coefficient is {beta_range}, and full "
                "pass-through is rejected at every horizon where this sample has power. "
                "The delivered price does not move with the freight inside it. At the "
                "monthly and quarterly horizons where term contracts actually live, "
                "three and a half years of index leave too few observations to reject "
                "anything at all."
            )
        return (
            f"At {labels} frequency the coefficient is {beta_range}; full pass-through is "
            "not rejected everywhere."
        )


def incidence_by_frequency(
    frame: pd.DataFrame,
    *,
    frequencies: tuple[tuple[str, str | None], ...] = FREQUENCIES,
) -> IncidenceResult:
    """Regress the change in the delivered price on the change in freight, per horizon.

    Differenced (E-H4) and with HAC errors, because both series are autocorrelated and
    the naive standard error would make the coefficient look far more precisely
    estimated than it is.
    """
    estimates = []
    for label, rule in frequencies:
        sampled = frame if rule is None else frame.resample(rule).last().dropna()
        changes = sampled.diff().dropna()
        if len(changes) < 12:
            continue
        regression = hac_ols(changes["cfr62"], changes[["c5"]])
        estimates.append(
            IncidenceEstimate(
                label=label,
                beta=float(regression.params["c5"]),
                std_error=float(regression.std_errors["c5"]),
                n_obs=int(regression.n_obs),
                implied_correlation=float(changes["c5"].std() / changes["cfr62"].std()),
            )
        )
    if not estimates:
        raise IncidenceError("no frequency carries enough observations to estimate anything")
    return IncidenceResult(estimates=tuple(estimates))
