"""Project F — you hedge bunkers with crude, and here is what you keep.

THESIS
------
Bunkers are the largest variable cost of a voyage and the only one with a liquid
futures market attached to it. So they get hedged, usually with crude, because crude is
the cheapest and deepest instrument available. This page measures what that hedge
actually does, and the answer is: much less than it looks, and less every year.

    crude explains 16.7% of VLSFO's daily variance
    adding gasoil takes it to 21.3%
    -> roughly four fifths of the bunker price risk is not hedgeable with either

And the ratio itself has collapsed. A hedge sized on the 2010s is three times too large
today:

    2016-2017  beta = 0.795   residual sigma =  4.5 USD/t
    2018-2019  beta = 0.776   residual sigma =  6.8 USD/t
    2020-2021  beta = 0.527   residual sigma = 13.8 USD/t
    2022-2023  beta = 0.278   residual sigma = 21.4 USD/t
    2024-2025  beta = 0.233   residual sigma = 10.2 USD/t

THE TEST THAT FAILED, AND WHY IT IS REPORTED
--------------------------------------------
The obvious story is IMO 2020: the sulphur cap created VLSFO as a distinct product in
January 2020, so the relationship to crude should break there. A Chow test on that date
rejects stability decisively (F = 36.9).

It also rejects on a placebo date in the middle of a quiet period (F = 13.9). On 4,484
daily observations of two trending series, a break test rejects almost everywhere, so a
single rejection identifies nothing. The honest reading is that the relationship
**drifts continuously** rather than breaking once — which is worse for a hedger than a
break would be, because a break can be dated and drift cannot.

That failed test is kept on the page rather than replaced by the version that worked.

THE UNIT TRAP
-------------
Crude is quoted in USD per **barrel** and bunkers in USD per **tonne**, so any hedge
ratio has a density inside it. Fuel oil runs about 6.35 barrels per tonne and middle
distillate about 7.45. Sizing a bunker hedge with the distillate figure — the one most
people know, because it is the diesel number — oversizes the position by 17%, silently,
and in the direction that looks like a working hedge most of the time.

ASSUMPTIONS
-----------
F-H1  6.35 bbl/t for fuel oil, 7.45 for distillate. Both vary with the specific blend;
      they are parameters here, and the sensitivity is the point rather than a caveat.
F-H2  The series is labelled VLSFO back to 2009, which **predates the product**: the
      0.5% sulphur grade did not exist as a bunker before 2019-2020. What the early
      history describes is not established here, and the drift measured below is
      reported as a property of the series rather than as a fact about the fuel.
F-H3  All estimation is on daily changes with HAC errors. Two trending levels give a
      hedge ratio that describes a shared trend and fails out of sample.
F-H4  Hedge effectiveness is measured in-sample. A ratio fitted on the same window it is
      scored on flatters itself, so the drift table is the honest statement about what a
      hedger would actually have carried.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from agri.core.stats import hac_ols
from agri.data.snapshot import cached

# F-H1 — the two densities, and the gap between them is the trap.
BBL_PER_TONNE_FUEL_OIL = 6.35
BBL_PER_TONNE_DISTILLATE = 7.45

IMO_2020 = "2020-01-01"
# A date chosen inside a quiet stretch, to show what the break test does when there is
# nothing to find. Picked before running it, not after.
PLACEBO_BREAK = "2016-06-01"

DEFAULT_WINDOWS: tuple[tuple[int, int], ...] = (
    (2010, 2011), (2012, 2013), (2014, 2015), (2016, 2017),
    (2018, 2019), (2020, 2021), (2022, 2023), (2024, 2025),
)


class BunkerBasisError(ValueError):
    """Mis-specified bunker hedge test — always a caller error."""


# ===========================================================================
# Data
# ===========================================================================
@cached("f_bunker_basis")
def load_bunker_frame(start: str | None = None) -> pd.DataFrame:
    """VLSFO Singapore against the two instruments anyone would hedge it with.

    Columns: vlsfo (USD/t), brent (USD/bbl), gasoil (USD/t), brent_usd_t.

    `brent_usd_t` is the conversion the hedge ratio depends on, applied once here so that
    no downstream function can quietly use a different density.
    """
    from agri.data.bloomberg_loader import load

    frame = pd.concat(
        {
            "vlsfo": load("vlsfo_singapore"),
            "brent": load("brent"),
            "gasoil": load("ice_gasoil"),
        },
        axis=1,
        sort=True,
    ).dropna()
    if start is not None:
        frame = frame[frame.index >= pd.Timestamp(start)]
    if frame.empty:
        raise BunkerBasisError(f"no common dates across the three series after {start}")
    frame["brent_usd_t"] = frame["brent"] * BBL_PER_TONNE_FUEL_OIL
    return frame


# ===========================================================================
# The unit trap, quantified
# ===========================================================================
def density_mis_sizing(
    *,
    correct: float = BBL_PER_TONNE_FUEL_OIL,
    mistaken: float = BBL_PER_TONNE_DISTILLATE,
) -> dict[str, float]:
    """What using the distillate density on a fuel oil hedge does to the position.

    Both numbers are correct for their own product, which is what makes the error easy:
    nothing about the calculation looks wrong, the hedge is simply the wrong size.
    """
    if correct <= 0 or mistaken <= 0:
        raise BunkerBasisError("densities must be > 0 barrels per tonne")
    return {
        "correct_bbl_per_t": float(correct),
        "mistaken_bbl_per_t": float(mistaken),
        "oversize_fraction": float(mistaken / correct - 1.0),
    }


# ===========================================================================
# The hedge ratio, and its drift
# ===========================================================================
@dataclass(frozen=True)
class HedgeWindow:
    """One window's hedge ratio and what it leaves behind."""

    label: str
    beta: float
    residual_sigma: float
    r_squared: float
    n_obs: int


@dataclass(frozen=True)
class BetaDrift:
    """The hedge ratio over time — the page's central object."""

    windows: tuple[HedgeWindow, ...]
    full_sample_beta: float
    full_sample_r2: float

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "window": w.label,
                    "beta": w.beta,
                    "residual_sigma": w.residual_sigma,
                    "r_squared": w.r_squared,
                    "n": w.n_obs,
                }
                for w in self.windows
            ]
        ).set_index("window")

    @property
    def first_beta(self) -> float:
        return float(self.windows[0].beta)

    @property
    def last_beta(self) -> float:
        return float(self.windows[-1].beta)

    @property
    def collapse_factor(self) -> float:
        """How many times too large a hedge sized on the earliest window would be today."""
        return self.first_beta / self.last_beta if self.last_beta else float("nan")

    @property
    def max_beta(self) -> float:
        return float(max(w.beta for w in self.windows))

    @property
    def headline(self) -> str:
        return (
            f"The hedge ratio to crude fell from {self.max_beta:.2f} at its peak to "
            f"{self.last_beta:.2f} in the most recent window — a position sized on the "
            f"earlier relationship is {self.collapse_factor:.1f} times too large today. "
            f"Over the full sample the ratio averages {self.full_sample_beta:.2f} and "
            f"crude explains {self.full_sample_r2:.1%} of the daily variance."
        )


def rolling_hedge_beta(
    frame: pd.DataFrame,
    *,
    windows: tuple[tuple[int, int], ...] = DEFAULT_WINDOWS,
    instrument: str = "brent_usd_t",
    min_obs: int = 100,
) -> BetaDrift:
    """The minimum-variance hedge ratio, estimated separately on each window.

    Windows rather than a rolling regression, because the point is not to smooth the
    ratio but to show what a hedger estimating on any two-year history would have
    carried into the next two years.
    """
    changes = frame.diff().dropna()
    if instrument not in changes.columns:
        raise BunkerBasisError(f"unknown instrument column: {instrument!r}")

    out = []
    for start_year, end_year in windows:
        window = changes.loc[str(start_year):str(end_year)]
        if len(window) < min_obs:
            continue
        regression = hac_ols(window["vlsfo"], window[[instrument]])
        beta = float(regression.params[instrument])
        residual = window["vlsfo"] - beta * window[instrument]
        out.append(
            HedgeWindow(
                label=f"{start_year}-{end_year}",
                beta=beta,
                residual_sigma=float(residual.std()),
                r_squared=float(regression.r_squared),
                n_obs=int(len(window)),
            )
        )
    if len(out) < 2:
        raise BunkerBasisError("at least two windows are needed to show drift")

    full = hac_ols(changes["vlsfo"], changes[[instrument]])
    return BetaDrift(
        windows=tuple(out),
        full_sample_beta=float(full.params[instrument]),
        full_sample_r2=float(full.r_squared),
    )


# ===========================================================================
# The break test that does not identify anything
# ===========================================================================
@dataclass(frozen=True)
class BreakAttempt:
    """A dated break test and its placebo, reported together or not at all."""

    tested_date: str
    tested_f: float
    placebo_date: str
    placebo_f: float

    @property
    def placebo_also_rejects(self) -> bool:
        return self.placebo_f > 4.0

    @property
    def identifies_a_break(self) -> bool:
        """A rejection only means something if the placebo does not also reject."""
        return not self.placebo_also_rejects

    @property
    def headline(self) -> str:
        if self.identifies_a_break:
            return (
                f"The relationship breaks at {self.tested_date} (F = {self.tested_f:.1f}) "
                f"and not at the placebo date (F = {self.placebo_f:.1f})."
            )
        return (
            f"A break test rejects at {self.tested_date} (F = {self.tested_f:.1f}) — and "
            f"also at a placebo date chosen inside a quiet stretch (F = {self.placebo_f:.1f}). "
            "On this many observations of two trending series it rejects almost anywhere, "
            "so the rejection dates nothing. The relationship drifts rather than breaking, "
            "which is the harder case for a hedger: a break can be dated, drift cannot."
        )


def break_attempt(
    frame: pd.DataFrame,
    *,
    tested_date: str = IMO_2020,
    placebo_date: str = PLACEBO_BREAK,
) -> BreakAttempt:
    """Try to date the change, and check the test against a date where nothing happened.

    Kept in the module because the failure is informative: it is what rules out the
    tidy "IMO 2020 changed the product" story and forces the drift reading.
    """
    from agri.chains.feedstock_lcfs import chow_break_test

    tested = chow_break_test(frame["vlsfo"], frame["brent"], tested_date)
    placebo = chow_break_test(frame["vlsfo"], frame["brent"], placebo_date)
    return BreakAttempt(
        tested_date=tested_date,
        tested_f=float(tested.f_stat),
        placebo_date=placebo_date,
        placebo_f=float(placebo.f_stat),
    )


# ===========================================================================
# What any of it actually hedges
# ===========================================================================
@dataclass(frozen=True)
class HedgeEffectiveness:
    """How much of the bunker price risk each instrument set removes."""

    table: pd.DataFrame        # index=instrument set, columns=r_squared, residual_sigma

    @property
    def best_r2(self) -> float:
        return float(self.table["r_squared"].max())

    @property
    def unhedgeable_share(self) -> float:
        return 1.0 - self.best_r2

    @property
    def headline(self) -> str:
        best = self.table["r_squared"].idxmax()
        return (
            f"The best available combination ({best}) removes {self.best_r2:.1%} of the "
            f"daily variance of the bunker price, leaving {self.unhedgeable_share:.1%} "
            f"carried, at {self.table.loc[best, 'residual_sigma']:.1f} USD/t of residual "
            "standard deviation. Crude is not a bunker hedge, it is a partial one."
        )


def hedge_effectiveness(frame: pd.DataFrame) -> HedgeEffectiveness:
    """In-sample variance reduction for crude, gasoil, and both together (F-H4).

    In-sample and therefore generous: the ratios are fitted on the very window they are
    scored on. A real hedger estimates on the past and carries into the future, which is
    strictly worse — so these numbers are an upper bound on what the instruments do.
    """
    changes = frame.diff().dropna()
    rows = []
    for label, columns in (
        ("crude only", ["brent_usd_t"]),
        ("gasoil only", ["gasoil"]),
        ("crude + gasoil", ["brent_usd_t", "gasoil"]),
    ):
        regression = hac_ols(changes["vlsfo"], changes[columns])
        fitted = regression.params["const"] + sum(
            regression.params[c] * changes[c] for c in columns
        )
        rows.append(
            {
                "instruments": label,
                "r_squared": float(regression.r_squared),
                "residual_sigma": float((changes["vlsfo"] - fitted).std()),
            }
        )
    return HedgeEffectiveness(table=pd.DataFrame(rows).set_index("instruments"))
