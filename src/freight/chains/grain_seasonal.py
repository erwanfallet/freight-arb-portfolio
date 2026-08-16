"""Project D — the largest cargo flow in the market leaves no mark on freight.

THESIS
------
Brazilian soybean exports grew roughly sixfold between 1999 and 2026. Over the same
period the Panamax market's seasonal amplitude nearly tripled. If freight seasonality
were a demand seasonality, those two facts would be the same fact.

They are not. The months that carry the Brazilian export peak — March through June —
sit at **the year's own median in every sub-period tested**, with no trend. The
amplitude that did grow is elsewhere: a January-February collapse that deepened from
0.95 to 0.71 of the annual level, and an autumn peak that strengthened.

    seasonal amplitude   1999-2005: 21.5%   ->   2020-2026: 59.1%
    harvest months       1999-2005:  1.008  ->   2020-2026:  1.008

So the biggest and most predictable growth in cargo demand in this market's history
produced no seasonal freight signature at all.

WHY THIS IS A RESULT AND NOT A NULL
-----------------------------------
A freight seasonal is not a demand seasonal. It is a **positioning** seasonal. Owners
know the harvest is coming — the calendar is agronomic, published years in advance, and
identical every year. A shock that everyone can see is arbitraged away by ballasting in
advance; what moves a rate is the part nobody positioned for.

That reading is testable rather than rhetorical, and the test is the growth comparison
above: if anticipation absorbs the flow, multiplying the flow by six changes nothing,
which is what the data shows. If the market simply cleared demand against a fixed fleet,
the harvest months would have risen with the volume, which they did not.

THE UNIT TRAP
-------------
The BPI is quoted in **index points** — a weighted average of four timecharter routes,
each in USD/day, divided by a fixed constant. It is not a price and cannot be converted
into a cost per tonne without the 5TC in USD/day, which this export does not contain.
Everything here is therefore computed as a **ratio to the index's own annual level**,
which is unit-free and survives the missing divisor. Turning the seasonal into dollars
requires a separate calibration step, and its fragility is stated rather than hidden
(see `implied_ballast_distance`).

ASSUMPTIONS
-----------
D-H1  The Brazilian soybean export peak runs March-June. Loading starts in February and
      tails into July; the four-month core is used, and the sensitivity to widening it
      to Feb-Jul is reported rather than assumed away.
D-H2  Normalising by the **annual median** removes the freight cycle, which is an order
      of magnitude larger than the seasonal, without removing the seasonal itself. The
      median rather than the mean, because dry bulk years are violently skewed (2021).
D-H3  **The effective sample is years, not days.** A month contributes ~21 daily prints
      that are one continuous autocorrelated stretch, not 21 draws. Every confidence
      interval here is computed on the count of years, per the portfolio's Rule C.
      Using the daily count would make every month significant and the page worthless.
D-H4  The BPI is a global Panamax index covering both basins. A single-basin flow is
      therefore diluted in it. This bounds what the test can prove and is the reason the
      page ends on a question rather than a conclusion.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from agri.core.resample import effective_n
from agri.core.stats import block_bootstrap
from agri.data.snapshot import cached

# D-H1 — the Brazilian export peak, and the wider window used as a sensitivity.
BRAZIL_HARVEST_MONTHS: tuple[int, ...] = (3, 4, 5, 6)
BRAZIL_HARVEST_WIDE: tuple[int, ...] = (2, 3, 4, 5, 6, 7)
# The northern-hemisphere harvest and pre-winter restocking window, for contrast.
NORTHERN_PEAK_MONTHS: tuple[int, ...] = (9, 10, 11, 12)

# Sub-periods of roughly equal length, cut on calendar convenience rather than on
# anything the data suggested — a break chosen after seeing the series would make the
# growth result circular.
DEFAULT_PERIODS: tuple[tuple[int, int], ...] = (
    (1999, 2005),
    (2006, 2012),
    (2013, 2019),
    (2020, 2026),
)


class SeasonalError(ValueError):
    """Mis-specified seasonal test — always a caller error."""


# ===========================================================================
# Loading and normalisation
# ===========================================================================
@cached("d_panamax_seasonal")
def load_panamax_frame(start: str | None = None) -> pd.DataFrame:
    """The Panamax index with its annual level divided out.

    Columns: bpi, year, month, annual_level, rel.

    `rel` is the index divided by the **median of its own calendar year** (D-H2). The
    dry bulk cycle moves the index by a factor of ten between 2016 and 2021; the
    seasonal moves it by tens of percent. Without dividing the cycle out, the seasonal
    is invisible and any monthly average is really a report on which years happened to
    be strong.
    """
    from agri.data.bloomberg_loader import load

    bpi = load("bpi")
    if start is not None:
        bpi = bpi.loc[pd.Timestamp(start):]
    if bpi.empty:
        raise SeasonalError(f"no Panamax index observations after {start}")

    frame = bpi.to_frame("bpi")
    frame["year"] = frame.index.year
    frame["month"] = frame.index.month
    frame["annual_level"] = frame.groupby("year")["bpi"].transform("median")
    frame["rel"] = frame["bpi"] / frame["annual_level"]
    return frame


def _complete_years(frame: pd.DataFrame, min_obs: int = 200) -> pd.DataFrame:
    """Drop calendar years too sparse to carry a seasonal.

    A year with 40 prints cannot say anything about a monthly pattern, and including it
    silently weights the median toward whichever months happened to be quoted.
    """
    counts = frame.groupby("year").size()
    keep = counts[counts >= min_obs].index
    return frame[frame["year"].isin(keep)]


# ===========================================================================
# The seasonal profile, with intervals computed on years
# ===========================================================================
@dataclass(frozen=True)
class SeasonalProfile:
    """Monthly profile of the index relative to its annual level.

    `n_years` is the honest sample size (D-H3): one independent observation per calendar
    year per month, not one per trading day.
    """

    profile: pd.DataFrame          # index=month, columns=median, lo, hi, n_years
    n_years: int
    label: str

    @property
    def amplitude(self) -> float:
        """Peak-to-trough spread, as a fraction of the annual level."""
        return float(self.profile["median"].max() / self.profile["median"].min() - 1.0)

    @property
    def peak_month(self) -> int:
        return int(self.profile["median"].idxmax())

    @property
    def trough_month(self) -> int:
        return int(self.profile["median"].idxmin())

    def level(self, months: tuple[int, ...]) -> float:
        """Mean relative level over a set of calendar months."""
        return float(self.profile.loc[list(months), "median"].mean())

    @property
    def headline(self) -> str:
        return (
            f"{self.label}: the index peaks in month {self.peak_month} "
            f"({self.profile.loc[self.peak_month, 'median']:.3f} of the annual level) and "
            f"troughs in month {self.trough_month} "
            f"({self.profile.loc[self.trough_month, 'median']:.3f}), an amplitude of "
            f"{self.amplitude:.1%} on {self.n_years} years."
        )


def seasonal_profile(
    frame: pd.DataFrame,
    *,
    label: str = "1999-2026",
    confidence: float = 0.95,
    n_iter: int = 2_000,
    seed: int = 0,
) -> SeasonalProfile:
    """Median relative level by calendar month, with an interval built on **years**.

    The interval is bootstrapped over the per-year monthly medians, not over daily
    prints (D-H3). One month of daily index values is a single autocorrelated stretch:
    treating its 21 observations as 21 draws divides the standard error by nearly five
    and turns noise into significance.
    """
    usable = _complete_years(frame)
    if usable.empty:
        raise SeasonalError("no calendar year carries enough observations for a seasonal")

    # one number per (year, month): the year's median level in that month
    per_year = usable.groupby(["year", "month"])["rel"].median().unstack()
    rows = []
    for month in range(1, 13):
        if month not in per_year.columns:
            continue
        yearly = per_year[month].dropna()
        if len(yearly) < 3:
            continue
        ci = block_bootstrap(
            yearly,
            block_len=1.0,          # years are the independent unit here
            statistic=np.median,
            n_iter=n_iter,
            confidence=confidence,
            seed=seed,
        )
        rows.append(
            {
                "month": month,
                "median": float(yearly.median()),
                "lo": ci.lo,
                "hi": ci.hi,
                "n_years": int(len(yearly)),
            }
        )

    profile = pd.DataFrame(rows).set_index("month")
    return SeasonalProfile(
        profile=profile,
        n_years=int(usable["year"].nunique()),
        label=label,
    )


def effective_sample(frame: pd.DataFrame) -> dict[str, float]:
    """What the daily count would have claimed, against what the data actually supports.

    Kept as an explicit output rather than a comment, because the gap between the two is
    the single most common way a seasonal chart lies.
    """
    usable = _complete_years(frame)
    n_days = int(len(usable))
    n_years = int(usable["year"].nunique())
    per_month_days = n_days / 12.0
    sample = effective_n(n_days, overlap=max(n_days / max(n_years, 1), 1.0))
    return {
        "n_daily_prints": float(n_days),
        "n_years": float(n_years),
        "days_per_month_slot": float(per_month_days),
        "n_eff": float(sample.n_eff),
        "overstatement_factor": float(np.sqrt(max(n_days / max(n_years, 1), 1.0))),
    }


# ===========================================================================
# THE TEST — did the flow's growth show up in the freight seasonal?
# ===========================================================================
@dataclass(frozen=True)
class HarvestFootprint:
    """Whether the Brazilian export peak left a mark that grew with the flow."""

    table: pd.DataFrame            # one row per sub-period
    harvest_months: tuple[int, ...]

    @property
    def amplitude_growth(self) -> float:
        first = float(self.table["amplitude"].iloc[0])
        last = float(self.table["amplitude"].iloc[-1])
        return last / first if first > 0 else float("nan")

    @property
    def harvest_drift(self) -> float:
        """Change in the harvest months' relative level, first period to last."""
        return float(self.table["harvest_level"].iloc[-1] - self.table["harvest_level"].iloc[0])

    @property
    def harvest_is_flat(self) -> bool:
        """The harvest months moved by less than 5% of the annual level across 27 years."""
        return abs(self.harvest_drift) < 0.05

    @property
    def headline(self) -> str:
        verdict = (
            "no detectable footprint"
            if self.harvest_is_flat
            else f"a drift of {self.harvest_drift:+.3f}"
        )
        return (
            f"The Panamax seasonal amplitude grew {self.amplitude_growth:.1f}x between "
            f"{int(self.table.index[0][0])}-{int(self.table.index[0][1])} and "
            f"{int(self.table.index[-1][0])}-{int(self.table.index[-1][1])}, while the "
            f"Brazilian harvest months moved from "
            f"{self.table['harvest_level'].iloc[0]:.3f} to "
            f"{self.table['harvest_level'].iloc[-1]:.3f} of the annual level — "
            f"{verdict}. The flow grew roughly sixfold over the same period."
        )


def harvest_footprint(
    frame: pd.DataFrame,
    *,
    periods: tuple[tuple[int, int], ...] = DEFAULT_PERIODS,
    harvest_months: tuple[int, ...] = BRAZIL_HARVEST_MONTHS,
) -> HarvestFootprint:
    """Compare the seasonal's growth against the harvest months' stability.

    This is the page's test, and it is a difference rather than a level: whatever the
    counterfactual level of the harvest months would have been without Brazil (which is
    not observable — D-H4), the comparison between "the amplitude tripled" and "the
    harvest months did not move" does not depend on knowing it.
    """
    rows = []
    for start_year, end_year in periods:
        window = frame[(frame["year"] >= start_year) & (frame["year"] <= end_year)]
        window = _complete_years(window)
        if window.empty:
            continue
        monthly = window.groupby("month")["rel"].median()
        rows.append(
            {
                "period": (start_year, end_year),
                "n_years": int(window["year"].nunique()),
                "amplitude": float(monthly.max() / monthly.min() - 1.0),
                "harvest_level": float(monthly.loc[list(harvest_months)].mean()),
                "northern_level": float(
                    monthly.loc[list(NORTHERN_PEAK_MONTHS)].mean()
                ),
                "trough_month": int(monthly.idxmin()),
                "peak_month": int(monthly.idxmax()),
            }
        )

    if len(rows) < 2:
        raise SeasonalError(
            "at least two sub-periods are needed to test whether the seasonal grew"
        )
    table = pd.DataFrame(rows).set_index("period")
    return HarvestFootprint(table=table, harvest_months=tuple(harvest_months))


def window_sensitivity(
    frame: pd.DataFrame,
    *,
    periods: tuple[tuple[int, int], ...] = DEFAULT_PERIODS,
) -> pd.DataFrame:
    """The same test on the narrow and the wide harvest window (D-H1).

    A result that only holds for one definition of "the harvest" is a result about the
    definition. Both windows are reported side by side rather than the convenient one
    being kept.
    """
    narrow = harvest_footprint(frame, periods=periods, harvest_months=BRAZIL_HARVEST_MONTHS)
    wide = harvest_footprint(frame, periods=periods, harvest_months=BRAZIL_HARVEST_WIDE)
    return pd.DataFrame(
        {
            "harvest_level_narrow": narrow.table["harvest_level"],
            "harvest_level_wide": wide.table["harvest_level"],
            "amplitude": narrow.table["amplitude"],
        }
    )
