"""The three rules that decide whether a test is valid.

This module computes almost nothing. It makes **executable** three rules that, written
as prose in a method note, are reliably forgotten within three weeks.

RULE A — you go down, never up
--------------------------------
For any statistical test (correlation, CCF, regression), if the slowest series is
monthly, the entire test is monthly. The fast series gets aggregated; the reverse —
stretching the slow one onto the fast one's calendar — fabricates observations that
don't exist and biases every autocorrelation. This is the diagnosis already made on the
copper page: `.ffill()` on Yangshan and SHFE stocks had destroyed the cross-correlation.

RULE B — forward-fill is legitimate for P&L, never for a test
-----------------------------------------------------------------
A smelter does pay a monthly TC every day: reconstructing a daily P&L from a monthly
series is correct. But the fill has to be **visible**, hence the twin columns
`is_true_print` and `staleness_days`.

This rule is made impossible to break by accident here: `ffill_with_provenance` returns
a three-column DataFrame that can't be passed as-is into a test. Extracting values from
it requires explicitly choosing `for_pnl()` — which allows it — or `for_test()` — which
**raises** the moment a single value is filled. The rule is no longer a guideline, it's
an exception.

Note on `ingest/contract.py`: that module forbids any gap policy other than `none`,
because it guards the **ingestion** boundary (a series arrives as-is, a gap stays a gap,
that's the lesson from the stale BSI). Rule B's forward-fill is a **modelling**
operation, downstream of the contract, on a series that's already been validated. The
two rules don't contradict each other: they guard two different boundaries.

RULE C — effective n
----------------------
A 30-day rolling window on daily data does not give n independent observations. Any
statistic computed on overlapping windows must report `n_eff`, and it's `n_eff` — not
`n_obs` — that goes into a significance band or a confidence interval.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Frequencies: ordered from fastest to slowest
# ---------------------------------------------------------------------------
FREQ_ORDER: dict[str, int] = {
    "daily": 0,
    "weekly": 1,
    "fortnightly": 2,   # UNICA, CEPEA fortnight
    "monthly": 3,
    "quarterly": 4,
    "yearly": 5,
}

# Corresponding pandas 3.x aliases (the 'M'/'Q'/'A' aliases were removed)
_PANDAS_RULE: dict[str, str] = {
    "daily": "D",
    "weekly": "W",
    "fortnightly": "SME",
    "monthly": "ME",
    "quarterly": "QE",
    "yearly": "YE",
}

VALID_AGGREGATIONS = ("last", "mean")


class StaleDataInTest(Exception):
    """A forward-filled series tried to enter a statistical test (Rule B)."""


class ResampleError(ValueError):
    """Mis-specified resampling — always a caller error."""


# ---------------------------------------------------------------------------
# RULE A
# ---------------------------------------------------------------------------
def infer_frequency(series: pd.Series) -> str:
    """Observed frequency of a series, inferred from the median gap between observations.

    The **median** is used, not the mode or the minimum: a daily series has weekend and
    holiday gaps, a monthly series has 28- and 31-day months. The median crosses both
    without getting caught out.
    """
    index = pd.DatetimeIndex(series.dropna().index)
    if len(index) < 3:
        raise ResampleError(
            f"at least 3 observations are needed to infer a frequency, "
            f"got {len(index)}"
        )
    # Arithmetic in seconds, never on the index's raw integers: pandas 3 dates in
    # microseconds by default (datetime64[us]) where pandas 2 dated in nanoseconds.
    # Dividing `.asi8` by a nanosecond constant gives a gap 1000x too small, classifies
    # every series as daily, and silently disarms Rule A.
    gaps_days = index.to_series().diff().dropna().dt.total_seconds() / 86_400.0
    median_gap = float(gaps_days.median())

    if median_gap <= 4:
        return "daily"
    if median_gap <= 10:
        return "weekly"
    if median_gap <= 20:
        return "fortnightly"
    if median_gap <= 45:
        return "monthly"
    if median_gap <= 135:
        return "quarterly"
    return "yearly"


def slowest_frequency(series_map: dict[str, pd.Series]) -> str:
    """The slowest frequency in the set — the one that governs the whole test."""
    if not series_map:
        raise ResampleError("no series supplied")
    return max(
        (infer_frequency(s) for s in series_map.values()),
        key=lambda f: FREQ_ORDER[f],
    )


def downsample(series: pd.Series, target_freq: str, *, how: str = "last") -> pd.Series:
    """Aggregates a series to a **slower or equal** frequency. Never the reverse.

    `how='last'` takes the period's last print (what a trader looks at),
    `how='mean'` the average (what an econometrician prefers for a flow). The choice is
    forced on the caller because it changes the result and has no good universal
    default — it must be documented in the page that calls it.
    """
    if target_freq not in FREQ_ORDER:
        raise ResampleError(f"unknown target frequency: {target_freq!r}")
    if how not in VALID_AGGREGATIONS:
        raise ResampleError(f"aggregation must be in {VALID_AGGREGATIONS}, got {how!r}")

    source_freq = infer_frequency(series)
    if FREQ_ORDER[source_freq] > FREQ_ORDER[target_freq]:
        raise ResampleError(
            f"RULE A violated: a {source_freq} series cannot be upsampled to "
            f"{target_freq}. Either lower the test to the slowest series' frequency, "
            "or drop the test."
        )
    # Resampling happens even when the frequency is already right: that's what
    # normalises the labels. A monthly series dated on the 1st and a daily series
    # downsampled to monthly (dated at month-end) have the same frequency and zero
    # common date — the intersection would come out empty with nothing to explain why.
    resampled = series.dropna().sort_index().resample(_PANDAS_RULE[target_freq])
    aggregated = resampled.last() if how == "last" else resampled.mean()
    return aggregated.dropna()


def align_for_test(
    series_map: dict[str, pd.Series], *, how: str = "last"
) -> pd.DataFrame:
    """Aligns several series for a statistical test, honouring Rule A.

    Every series is downsampled to the slowest one's frequency, then the dates are
    intersected. No gap is filled: a date where a series is missing drops out of the
    test.

    The returned DataFrame carries the frequency used in `.attrs["test_frequency"]`, to
    be shown on the page — "the correlation is computed monthly because the grind data
    is quarterly" is a sentence that must appear on screen, not stay in the code.
    """
    target = slowest_frequency(series_map)
    aligned = pd.concat(
        {name: downsample(s, target, how=how) for name, s in series_map.items()},
        axis=1,
        sort=True,
    ).dropna()
    if aligned.empty:
        raise ResampleError(
            f"no common date across the {len(series_map)} series after downsampling to "
            f"{target} — check the calendars before going further, don't fill the gaps"
        )
    aligned.attrs["test_frequency"] = target
    return aligned


# ---------------------------------------------------------------------------
# RULE B
# ---------------------------------------------------------------------------
def ffill_with_provenance(
    series: pd.Series,
    target_index: pd.DatetimeIndex,
    *,
    max_staleness_days: int | None = None,
) -> pd.DataFrame:
    """Traceable forward-fill, for P&L reconstruction only (Rule B).

    Returns three columns:
        value             the filled value
        is_true_print     True if the date carries a genuine observation
        staleness_days    age of the carried observation, 0 on a genuine print

    `max_staleness_days` resets to NaN beyond a given age: a 400-day-old monthly TC is
    no longer a conservative assumption, it's dead data. Corresponds to the "filter
    stale days" option in the sidebar.
    """
    target_index = pd.DatetimeIndex(target_index).sort_values()
    clean = series.dropna().sort_index()
    if clean.empty:
        raise ResampleError("empty series — nothing to carry forward")

    reindexed = clean.reindex(target_index)
    is_true_print = reindexed.notna()
    filled = reindexed.ffill()

    # age: number of days since the last genuine print to the left of each date. Before
    # the first print, last_print stays NaT and the age comes out as NaN — that's the
    # intended behaviour: there is nothing to carry forward, not a zero age value.
    own_date = pd.Series(target_index, index=target_index)
    last_print = own_date.where(is_true_print.to_numpy()).ffill()
    staleness = (own_date - last_print).dt.days.astype(float)

    out = pd.DataFrame(
        {
            "value": filled,
            "is_true_print": is_true_print,
            "staleness_days": staleness,
        }
    )

    if max_staleness_days is not None:
        too_old = out["staleness_days"] > max_staleness_days
        out.loc[too_old, "value"] = np.nan

    out.attrs["is_ffilled"] = True
    return out


def for_pnl(filled: pd.DataFrame) -> pd.Series:
    """Extracts values from a `ffill_with_provenance` frame **for a P&L computation**.

    Allowed under Rule B. The function's name is the documentation: writing `for_pnl` in
    a code path that feeds a regression means you did it on purpose.
    """
    _check_provenance_frame(filled)
    return filled["value"]


def for_test(filled: pd.DataFrame) -> pd.Series:
    """Extracts values **for a statistical test** — raises if anything at all is filled.

    This is the point where Rule B stops being a guideline. The correct fallback isn't
    to force it through: it's `align_for_test`, which lowers the test to the slow
    series' frequency.
    """
    _check_provenance_frame(filled)
    live = filled[filled["value"].notna()]
    n_filled = int((~live["is_true_print"]).sum())
    if n_filled:
        share = 100.0 * n_filled / len(live)
        raise StaleDataInTest(
            f"RULE B violated: {n_filled} of {len(live)} values ({share:.0f}%) are "
            "forward-filled and cannot enter a statistical test. "
            "Use align_for_test() to lower the test to the slowest series' frequency."
        )
    return live["value"]


def _check_provenance_frame(filled: pd.DataFrame) -> None:
    required = {"value", "is_true_print", "staleness_days"}
    missing = required - set(filled.columns)
    if missing:
        raise ResampleError(
            f"expected a DataFrame produced by ffill_with_provenance (columns "
            f"{sorted(required)}), missing columns: {sorted(missing)}"
        )


# ---------------------------------------------------------------------------
# RULE C
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EffectiveSample:
    """Sample size corrected for overlap, and what that changes."""

    n_obs: int
    overlap: float
    n_eff: float

    @property
    def shrinkage(self) -> float:
        """Reduction factor: 3.0 means 'three times less information.'"""
        return self.n_obs / self.n_eff if self.n_eff > 0 else float("inf")

    @property
    def summary(self) -> str:
        return (
            f"n = {self.n_obs} observations, average overlap {self.overlap:.2f} "
            f"-> n_eff = {self.n_eff:.1f} (the sample is worth {self.shrinkage:.1f}x less "
            "than its apparent size)"
        )


def effective_n(n_obs: int, overlap: float) -> EffectiveSample:
    """n_eff = n_obs / overlap. The formula is trivial, forgetting it isn't."""
    if n_obs < 1:
        raise ResampleError(f"n_obs must be >= 1, got {n_obs}")
    if overlap < 1:
        raise ResampleError(
            f"overlap must be >= 1 (1 = independent observations), got {overlap}"
        )
    return EffectiveSample(n_obs=n_obs, overlap=float(overlap), n_eff=n_obs / overlap)


def effective_n_rolling(n_obs: int, window: int) -> EffectiveSample:
    """The rolling-window statistic case: the overlap is the window."""
    return effective_n(n_obs, window)


def average_concurrency(entry_dates: pd.DatetimeIndex, hold_days: int) -> float:
    """Average number of positions open at once, over the days where at least one is.

    Measured on the real entry dates, not inferred from a theoretical cap: a backtest
    capped at "3 concurrent positions max" rarely runs at 3 all the time, and taking the
    maximum would underestimate `n_eff`, thus overstating significance — the bias runs
    the wrong way, so it has to be measured.
    """
    entries = pd.DatetimeIndex(entry_dates).sort_values()
    if len(entries) == 0:
        raise ResampleError("no entry dates")
    if hold_days < 1:
        raise ResampleError(f"hold_days must be >= 1, got {hold_days}")

    span = pd.date_range(entries.min(), entries.max() + pd.Timedelta(days=hold_days - 1), freq="D")
    open_count = np.zeros(len(span), dtype=int)
    positions = span.get_indexer(entries)
    for start in positions:
        open_count[start : start + hold_days] += 1

    active = open_count[open_count > 0]
    return float(active.mean())


def effective_n_from_trades(
    entry_dates: pd.DatetimeIndex, hold_days: int
) -> EffectiveSample:
    """`n_eff` of a backtest with overlapping positions — the T1-M module's case.

    This is the computation to run on the copper backtest before sending anything: 18
    trades held 30 days with several simultaneous positions are not 18 independent
    draws, and the confidence interval on a 100% win rate changes completely as a
    result.
    """
    overlap = average_concurrency(entry_dates, hold_days)
    return effective_n(len(pd.DatetimeIndex(entry_dates)), overlap)
