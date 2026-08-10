"""T2-6 — Inter-oil substitution.

TENSION — INFERRED, NOT SOURCED
----------------------------------
**It seems to me** crushers hold palm/soy/rapeseed/sunflower elasticity to be strong, and
formulators hold it to be sticky — reformulating a recipe takes months and goes back
through validations. "It seems to me", never "I read that".

THE IDEA
--------
If substitution were instantaneous, inter-oil spreads would revert quickly to the mean as
soon as they widen. If it were impossible, they would drift freely. Reality sits between
the two, **and it depends on the level**: this is measurable.

    spread_ij = P_i - P_j            (all oils brought to USD/t)
    dspread_t = a + b spread_{t-1} + e
    half_life  = -ln(2) / ln(1 + b)

**Estimating the half-life by regime** (wide spread / narrow spread) gives the implied
substitution bounds: the levels beyond which the half-life collapses, i.e. beyond which
someone genuinely switches.

TIPPING POINT
-------------
The spread level where the half-life drops — the substitution bound. This is a number in
USD/t that a crusher confirms or denies immediately, because it is the level at which their
phone rings.

ASSUMPTIONS
-----------
S-H1  Every oil is brought to USD/t before any calculation. Palm in MYR/t, soy in
      cents/lb: mixing units here would produce meaningless spreads.
S-H2  The threshold separating "wide spread" from "narrow spread" is a quantile of the
      historical distribution, not an absolute level — levels drift with inflation and
      the general price level.
S-H3  No transport or quality cost is modelled in the spread. They shift the
      equilibrium level but not the speed of reversion, which is what the test targets.
S-H4  The half-life is estimated on a simple AR(1). A threshold model (TAR) would be
      more accurate; the regime-based estimate is a readable, robust approximation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from agri.data.snapshot import cached

from agri.core.stats import adf_kpss, hac_ols

DEFAULT_WIDE_QUANTILE = 0.75          # S-H2


class SubstitutionError(ValueError):
    """Mis-specified model."""


@dataclass(frozen=True)
class HalfLife:
    """Speed of mean reversion of a spread, and its readability."""

    beta: float
    half_life_days: float
    pvalue: float
    n_obs: int
    label: str = ""

    @property
    def is_mean_reverting(self) -> bool:
        """Significant mean reversion: beta must be negative AND significant."""
        return self.beta < 0 and self.pvalue < 0.05

    @property
    def summary(self) -> str:
        if not self.is_mean_reverting:
            return (
                f"{self.label}: no detectable mean reversion "
                f"(beta = {self.beta:+.4f}, p = {self.pvalue:.3f}, n = {self.n_obs})"
            )
        return (
            f"{self.label}: half-life {self.half_life_days:.0f} days "
            f"(beta = {self.beta:+.4f}, p = {self.pvalue:.3f}, n = {self.n_obs})"
        )


def estimate_half_life(
    spread: pd.Series, *, mask: pd.Series | None = None, label: str = ""
) -> HalfLife:
    """Mean-reversion half-life, via `dspread = a + b spread_{t-1} + e`, HAC errors.

        half_life = -ln(2) / ln(1 + b)

    A positive or zero `b` means there is no mean reversion: the half-life is then
    infinite, and infinity is returned rather than a misleading number.

    `mask` restricts the estimate to a regime. **Lags are computed on the full
    series, then filtered** — never the reverse. Filtering first would produce
    differences between observations that are not time-adjacent, which fabricates
    false mean reversion: two points three weeks apart would look like they
    "converged" in one step. On the test set, this mistake turned a 173-day
    half-life into 10.
    """
    clean = pd.Series(spread).dropna().astype(float)
    if len(clean) < 60:
        raise SubstitutionError(f"at least 60 observations are needed, got {len(clean)}")

    frame = pd.concat({"d": clean.diff(), "lag": clean.shift(1)}, axis=1).dropna()
    if mask is not None:
        frame = frame[pd.Series(mask).reindex(frame.index).fillna(False).astype(bool)]
    if len(frame) < 40:
        raise SubstitutionError(
            f"regime too short to estimate a half-life: n={len(frame)}"
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
    """Every pairwise spread, all oils already in USD/t (S-H1).

    Columns are named `oil_a_minus_oil_b`, in alphabetical order of the pair, so
    that a spread never appears twice with opposite signs.
    """
    if len(prices_usd_t) < 2:
        raise SubstitutionError("at least two oils are needed")
    frame = pd.concat(prices_usd_t, axis=1).dropna()
    if frame.empty:
        raise SubstitutionError("no common date across the oil series")

    names = sorted(prices_usd_t)
    out = pd.DataFrame(index=frame.index)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            out[f"{first}_minus_{second}"] = frame[first] - frame[second]
    return out


@dataclass(frozen=True)
class SubstitutionBound:
    """The implied substitution bound, for a pair of oils."""

    pair: str
    threshold_usd_t: float
    narrow: HalfLife
    wide: HalfLife

    @property
    def substitution_kicks_in(self) -> bool:
        """Does the half-life collapse when the spread widens?"""
        return (
            self.wide.is_mean_reverting
            and self.wide.half_life_days < self.narrow.half_life_days
        )

    @property
    def headline(self) -> str:
        if not self.substitution_kicks_in:
            return (
                f"{self.pair}: beyond {self.threshold_usd_t:.0f} USD/t of gap, the "
                "spread does not revert faster. On this sample, no substitution "
                "bound is visible — the formulators' \"sticky\" thesis holds."
            )
        return (
            f"{self.pair}: below {self.threshold_usd_t:.0f} USD/t of gap, the "
            f"spread takes {self.narrow.half_life_days:.0f} days to close; beyond "
            f"it, {self.wide.half_life_days:.0f} days. The substitution bound is "
            "there — the level at which someone genuinely switches."
        )


def substitution_bound(
    spread: pd.Series,
    *,
    pair: str = "",
    wide_quantile: float = DEFAULT_WIDE_QUANTILE,
    threshold_usd_t: float | None = None,
) -> SubstitutionBound:
    """Half-life of the spread by regime, and the threshold separating them (S-H2, S-H4).

    The regime is defined on the **absolute deviation** from the median level, not
    on the raw level: a spread can be wide in either direction, and substitution
    plays out both ways.

    The regime is evaluated on the **lagged** value of the spread, never the current
    one: classifying an observation by the level it reaches after the move being
    explained would be circular.

    `threshold_usd_t` forces a threshold instead of deriving it from a quantile —
    useful when a practitioner proposes one, which is exactly the conversation
    being sought.
    """
    clean = pd.Series(spread).dropna().astype(float)
    if not 0.5 < wide_quantile < 1.0:
        raise SubstitutionError(f"wide_quantile must be in (0.5, 1), got {wide_quantile}")

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
        narrow=estimate_half_life(clean, mask=~is_wide, label="narrow spread"),
        wide=estimate_half_life(clean, mask=is_wide, label="wide spread"),
    )


def screen_all_pairs(
    spreads: pd.DataFrame, *, wide_quantile: float = DEFAULT_WIDE_QUANTILE
) -> pd.DataFrame:
    """Summary table: half-life by regime for every pair.

    Pairs whose spread is not stationary are flagged and **excluded from the
    reading**: a half-life estimated on a unit-root series is a number with no
    content, not a slow measurement.
    """
    rows = []
    for column in spreads.columns:
        series = spreads[column].dropna()
        try:
            verdict = adf_kpss(series).verdict
        except Exception:
            verdict = "not testable"
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
                "note": "" if verdict == "stationary" else "non-stationary spread — do not read the half-life",
            }
        )
    return pd.DataFrame(rows)


# ===========================================================================
# THE FIXED-PARITY WINDOW — the only clean test the data allows
# ===========================================================================
# The export contains palm (KO1) in **ringgit per tonne** and soy (BO1) in cents per pound,
# but **no USDMYR series**. Computing a palm-soy spread would therefore mean subtracting two
# currencies — exactly the mistake this portfolio tracks elsewhere. It is refused.
#
# Except over one window: Bank Negara pegged the ringgit at 3.80 MYR/USD from 2 September
# 1998 to 21 July 2005. Over those seven years, the missing series is a **constant fixed by
# decree**, and the spread computes exactly, with no exchange-rate assumption at all. It is
# a natural experiment: every move in the spread there is pure substitution economics,
# uncontaminated by FX.
MYR_PEG_RATE = 3.80
MYR_PEG_START = "1998-09-02"
MYR_PEG_END = "2005-07-21"
CENTS_LB_TO_USD_T = 22.0462


@cached('t2_6_peg')
def load_peg_window_spread() -> pd.DataFrame:
    """Palm-soy spread in USD/tonne, over the ringgit's fixed-parity window.

    Columns: palm_myr, palm_usd, soy_usd, spread.

    The palm conversion is a **division by a regulatory constant**, not by a market
    series — this is what makes this spread readable with no assumption.
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
            "no common palm/soy data over the fixed-parity window "
            f"{MYR_PEG_START} — {MYR_PEG_END}"
        )
    frame["palm_usd"] = frame["palm_myr"] / MYR_PEG_RATE
    frame["soy_usd"] = frame["soy_c_lb"] * CENTS_LB_TO_USD_T
    frame["spread"] = frame["palm_usd"] - frame["soy_usd"]
    return frame[["palm_myr", "palm_usd", "soy_usd", "spread"]]


def rolling_deviation(spread: pd.Series, *, window: int = 250) -> pd.Series:
    """Deviation of the spread from its rolling median.

    Essential here: over the fixed-parity window, the spread does not orbit a
    stable level — it **drifts** from a 67 USD/t premium to a 170 USD/t discount.
    Testing mean reversion against a constant would just measure this drift and
    call it substitution. The rolling median isolates deviations from the current
    level from this shift of the level itself.
    """
    if window < 20:
        raise SubstitutionError(f"rolling window too short: {window}")
    reference = spread.rolling(window, min_periods=window // 2).median()
    return (spread - reference).dropna().rename("deviation")


@dataclass(frozen=True)
class SubstitutionVerdict:
    """The substitution test on the clean window — and its negative result.

    The hypothesis predicted: wide gap -> someone switches -> fast reversion. The
    data says the opposite. **Narrow** gaps revert fast, **wide** ones do not revert
    at all. Reading: small gaps are noise around a slowly moving equilibrium; large
    gaps are not dislocations, they are shifts of the equilibrium itself.
    """

    narrow: HalfLife
    wide: HalfLife
    threshold_usd_t: float
    window: int
    n_obs: int

    @property
    def substitution_band_exists(self) -> bool:
        """True only if wide gaps revert FASTER than narrow ones, which the thesis
        predicts and the data does not show."""
        return (
            np.isfinite(self.wide.half_life_days)
            and np.isfinite(self.narrow.half_life_days)
            and self.wide.half_life_days < self.narrow.half_life_days
        )

    @property
    def headline(self) -> str:
        if self.substitution_band_exists:
            return (
                f"Beyond {self.threshold_usd_t:.0f} USD/t of gap, the spread "
                f"reverts in {self.wide.half_life_days:.0f} days against "
                f"{self.narrow.half_life_days:.0f} in the narrow regime: a "
                "substitution bound exists, and it sits at this level."
            )
        return (
            f"Result contrary to the thesis. Narrow gaps revert in "
            f"{self.narrow.half_life_days:.0f} days, but beyond "
            f"{self.threshold_usd_t:.0f} USD/t **no mean reversion is detectable** "
            "— large gaps do not close, they shift the level. Fading a wide "
            "palm-soy spread has no support in the only window where the test is clean."
        )


def substitution_verdict(
    spread: pd.Series, *, window: int = 250, quantile: float = 0.70
) -> SubstitutionVerdict:
    """Compares mean reversion of wide and narrow gaps, against a rolling reference.

    THE TRAP AVOIDED, AND IT IS THIS MODULE'S OWN: lags are computed on the **full**
    series and then masked, never the reverse. Filtering a non-contiguous
    subsample first and then applying `.diff()` would fabricate mean reversion out
    of calendar breaks — which is what `estimate_half_life` avoids via its `mask`
    parameter.

    RESIDUAL BIAS, IN THE RIGHT DIRECTION: conditioning on a large |gap|
    over-samples measurement noise, which reverts mechanically. This bias
    therefore pushes toward DETECTING mean reversion. Not finding one despite it
    makes the negative result more solid, not less.
    """
    if not 0.5 < quantile < 1.0:
        raise SubstitutionError(f"implausible separation quantile: {quantile}")

    deviation = rolling_deviation(spread, window=window)
    threshold = float(deviation.abs().quantile(quantile))
    return SubstitutionVerdict(
        narrow=estimate_half_life(
            deviation, mask=deviation.abs() < threshold, label="narrow gap"
        ),
        wide=estimate_half_life(
            deviation, mask=deviation.abs() >= threshold, label="wide gap"
        ),
        threshold_usd_t=threshold,
        window=window,
        n_obs=len(deviation),
    )


def structural_drift(spread: pd.Series) -> pd.DataFrame:
    """The spread's annual drift — why it cannot be treated as stationary.

    Returns the median by year, plus the total range. Over the fixed-parity
    window, palm goes from a premium over soy to a discount of several tens of
    dollars: this is not an oscillation around an equilibrium, it is a structural
    repricing.
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
