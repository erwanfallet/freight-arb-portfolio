"""Statistics toolkit — built to produce refusals to conclude.

The throughline: every function returns its measurement **together with what allows it
to be disqualified**. A correlation without its significance band, an R² without its
effective n, a win rate without its confidence interval are numbers that don't survive a
desk well. That's precisely where a page's credibility is decided.

Everything that takes an `n_eff` takes it explicitly: these functions don't know whether
the sample they're given comes from overlapping windows. It's up to the caller to say
so, via `core.resample.effective_n*`.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

__all__ = [
    "StationarityVerdict",
    "adf_kpss",
    "CrossCorrelation",
    "ccf_with_band",
    "HacRegression",
    "hac_ols",
    "newey_west_lags",
    "BootstrapCI",
    "block_bootstrap",
    "stationary_bootstrap_indices",
    "regime_runs",
    "clopper_pearson",
    "ProportionCI",
]


class StatsError(ValueError):
    """Mis-specified test or insufficient sample — always a caller error."""


# ===========================================================================
# Stationarity: both tests, always
# ===========================================================================
@dataclass(frozen=True)
class StationarityVerdict:
    """Joint ADF + KPSS result.

    The two tests have **opposite** null hypotheses:
        ADF  H0 = unit root        -> small p = stationary
        KPSS H0 = stationarity     -> small p = non-stationary

    Running both is the only way to tell "stationary" apart from "I don't have enough
    data to say." When they disagree, no conclusion is drawn: `verdict` is then
    "conflicting" or "inconclusive," and the page must show that rather than picking
    whichever test is convenient.
    """

    adf_stat: float
    adf_pvalue: float
    kpss_stat: float
    kpss_pvalue: float
    alpha: float
    n_obs: int

    @property
    def adf_rejects_unit_root(self) -> bool:
        return self.adf_pvalue < self.alpha

    @property
    def kpss_rejects_stationarity(self) -> bool:
        return self.kpss_pvalue < self.alpha

    @property
    def verdict(self) -> str:
        adf_ok = self.adf_rejects_unit_root
        kpss_bad = self.kpss_rejects_stationarity
        if adf_ok and not kpss_bad:
            return "stationary"
        if not adf_ok and kpss_bad:
            return "unit_root"
        if adf_ok and kpss_bad:
            # both reject: typically a structural break or heteroscedasticity, not a
            # "moderately stationary" series
            return "conflicting"
        return "inconclusive"

    @property
    def summary(self) -> str:
        labels = {
            "stationary": "stationary (both tests agree)",
            "unit_root": "unit root (both tests agree)",
            "conflicting": "CONFLICTING — both reject: suspect a structural break or "
                           "heteroscedasticity, do not conclude",
            "inconclusive": "INCONCLUSIVE — neither rejects: sample probably too "
                            "short, do not conclude",
        }
        return (
            f"{labels[self.verdict]} | ADF p={self.adf_pvalue:.3f}, "
            f"KPSS p={self.kpss_pvalue:.3f}, n={self.n_obs}"
        )


def adf_kpss(
    series: pd.Series, *, alpha: float = 0.05, regression: str = "c"
) -> StationarityVerdict:
    """ADF and KPSS on the same series, with a joint verdict.

    `regression='c'` tests around a constant (the usual case for a spread or a margin);
    `'ct'` adds a deterministic trend.
    """
    from statsmodels.tsa.stattools import adfuller, kpss

    clean = pd.Series(series).dropna().astype(float)
    if len(clean) < 20:
        raise StatsError(
            f"at least 20 observations are needed for a readable stationarity test, "
            f"got {len(clean)}"
        )

    adf_stat, adf_p, *_ = adfuller(clean.to_numpy(), regression=regression, autolag="AIC")

    with warnings.catch_warnings():
        # KPSS clamps its p-value into [0.01, 0.10] and warns when it saturates; that's
        # information carried by the verdict, not a problem to surface.
        warnings.simplefilter("ignore")
        kpss_stat, kpss_p, *_ = kpss(clean.to_numpy(), regression=regression, nlags="auto")

    return StationarityVerdict(
        adf_stat=float(adf_stat),
        adf_pvalue=float(adf_p),
        kpss_stat=float(kpss_stat),
        kpss_pvalue=float(kpss_p),
        alpha=alpha,
        n_obs=len(clean),
    )


# ===========================================================================
# Cross-correlation with a significance band
# ===========================================================================
@dataclass(frozen=True)
class CrossCorrelation:
    """CCF and its Bartlett band. The band is the deliverable, not the correlation."""

    lags: np.ndarray
    values: np.ndarray
    band: float
    n_eff: float

    @property
    def significant_lags(self) -> np.ndarray:
        """The only lags there's a right to talk about."""
        return self.lags[np.abs(self.values) > self.band]

    def peak(self) -> tuple[int, float]:
        """Lag of maximum absolute correlation, and its value — significant or not."""
        i = int(np.nanargmax(np.abs(self.values)))
        return int(self.lags[i]), float(self.values[i])

    @property
    def summary(self) -> str:
        lag, value = self.peak()
        verdict = (
            "SIGNIFICANT" if abs(value) > self.band
            else f"WITHIN THE BAND (±{self.band:.3f}) — indistinguishable from zero"
        )
        return (
            f"peak at lag {lag:+d}: rho = {value:+.3f} | {verdict} | "
            f"n_eff = {self.n_eff:.1f}"
        )


def ccf_with_band(
    x: pd.Series,
    y: pd.Series,
    *,
    max_lag: int = 20,
    n_eff: float | None = None,
    confidence: float = 0.95,
) -> CrossCorrelation:
    """Cross-correlation between x and y, with the Bartlett band ±z/sqrt(n_eff).

    SIGN CONVENTION, worth displaying on the page because it flips between software
    packages: the value at lag k is `corr(x[t], y[t+k])`. A peak at **positive lag**
    therefore means **x leads y**.

    `n_eff` is the effective sample size (Rule C). Leaving it at None uses the raw
    observation count — correct only if the observations don't overlap. It's the
    parameter that decides the band's width, and therefore what's allowed to be called
    a signal: at n=150, a correlation of -0.10 is within the band.
    """
    aligned = pd.concat({"x": x, "y": y}, axis=1).dropna()
    n = len(aligned)
    if n < max_lag + 10:
        raise StatsError(
            f"sample too short for {max_lag} lags: n={n}. "
            "Reduce max_lag or drop the test."
        )

    xs = aligned["x"].to_numpy(dtype=float)
    ys = aligned["y"].to_numpy(dtype=float)
    xs = xs - xs.mean()
    ys = ys - ys.mean()
    denom = np.sqrt(np.sum(xs**2) * np.sum(ys**2))
    if denom == 0:
        raise StatsError("one of the two series is constant — correlation undefined")

    lags = np.arange(-max_lag, max_lag + 1)
    values = np.empty(len(lags), dtype=float)
    for i, k in enumerate(lags):
        if k >= 0:
            values[i] = np.sum(xs[: n - k] * ys[k:]) / denom
        else:
            values[i] = np.sum(xs[-k:] * ys[: n + k]) / denom

    effective = float(n_eff) if n_eff is not None else float(n)
    if effective < 2:
        raise StatsError(f"n_eff must be >= 2, got {effective}")
    z = float(scipy_stats.norm.ppf(0.5 + confidence / 2.0))
    return CrossCorrelation(
        lags=lags, values=values, band=z / np.sqrt(effective), n_eff=effective
    )


# ===========================================================================
# HAC-error regression (Newey-West)
# ===========================================================================
def newey_west_lags(n_obs: int) -> int:
    """Automatic selection rule: floor(4 * (n/100)^(2/9))."""
    if n_obs < 1:
        raise StatsError(f"n_obs must be >= 1, got {n_obs}")
    return int(np.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0)))


@dataclass(frozen=True)
class HacRegression:
    """OLS with Newey-West errors. Naive t-stats on a time series lie."""

    params: pd.Series
    std_errors: pd.Series
    tvalues: pd.Series
    pvalues: pd.Series
    r_squared: float
    n_obs: int
    lags: int

    def is_significant(self, name: str, *, alpha: float = 0.05) -> bool:
        return bool(self.pvalues[name] < alpha)

    def describe(self, name: str) -> str:
        return (
            f"{name} = {self.params[name]:+.4f} "
            f"(HAC se = {self.std_errors[name]:.4f}, t = {self.tvalues[name]:+.2f}, "
            f"p = {self.pvalues[name]:.3f})"
        )


def hac_ols(
    y: pd.Series,
    X: pd.DataFrame | pd.Series,
    *,
    lags: int | None = None,
    add_constant: bool = True,
) -> HacRegression:
    """OLS with HAC (Newey-West) covariance.

    `lags=None` applies `newey_west_lags(n)`. Every regression in this portfolio runs on
    autocorrelated time series: HAC covariance isn't a refinement, it's the condition
    for the displayed p-value to mean anything.
    """
    import statsmodels.api as sm

    if isinstance(X, pd.Series):
        X = X.to_frame()
    aligned = pd.concat([pd.Series(y).rename("__y__"), X], axis=1).dropna()
    if len(aligned) < X.shape[1] + 3:
        raise StatsError(
            f"not enough observations: n={len(aligned)} for {X.shape[1]} regressors"
        )

    y_clean = aligned["__y__"].astype(float)
    X_clean = aligned.drop(columns="__y__").astype(float)
    if add_constant:
        X_clean = sm.add_constant(X_clean, has_constant="add")

    n = len(aligned)
    chosen = newey_west_lags(n) if lags is None else int(lags)
    fitted = sm.OLS(y_clean, X_clean).fit(
        cov_type="HAC", cov_kwds={"maxlags": chosen, "use_correction": True}
    )
    return HacRegression(
        params=fitted.params,
        std_errors=fitted.bse,
        tvalues=fitted.tvalues,
        pvalues=fitted.pvalues,
        r_squared=float(fitted.rsquared),
        n_obs=n,
        lags=chosen,
    )


# ===========================================================================
# Block bootstrap (stationary Politis-Romano)
# ===========================================================================
@dataclass(frozen=True)
class BootstrapCI:
    """Confidence interval that respects time dependence."""

    point: float
    lo: float
    hi: float
    confidence: float
    block_len: float
    n_iter: int

    @property
    def includes_zero(self) -> bool:
        return self.lo <= 0.0 <= self.hi

    @property
    def summary(self) -> str:
        verdict = " — INCLUDES ZERO" if self.includes_zero else ""
        return (
            f"{self.point:+.4f} [{self.confidence:.0%} CI: {self.lo:+.4f}, "
            f"{self.hi:+.4f}]{verdict}"
        )


def stationary_bootstrap_indices(
    n: int, block_len: float, n_iter: int, rng: np.random.Generator
) -> np.ndarray:
    """Politis-Romano resampling indices, fully vectorised.

    Blocks have a geometric length averaging `block_len` and wrap around at the end of
    the series. Random rather than fixed length: that's what makes the bootstrap
    stationary — a fixed-size block split introduces periodicity.
    """
    p = 1.0 / block_len
    starts = rng.random((n_iter, n)) < p
    starts[:, 0] = True
    anchors = rng.integers(0, n, size=(n_iter, n))

    positions = np.arange(n)
    last_start = np.maximum.accumulate(np.where(starts, positions, -1), axis=1)
    block_anchor = np.take_along_axis(anchors, last_start, axis=1)
    return (block_anchor + (positions - last_start)) % n


def block_bootstrap(
    values: pd.Series | np.ndarray,
    *,
    block_len: float,
    statistic=None,
    n_iter: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapCI:
    """Stationary-bootstrap CI — never a naive p-value on a P&L.

    `block_len` should be on the order of the backtest's **holding period**: that's the
    length over which observations stay dependent. A block of 1 amounts to assuming
    independence, which is exactly the assumption this is meant to avoid.

    `statistic=None` takes the mean (vectorised path). Any other statistic is a callable
    applied to each resample.

    `seed` is fixed by default: a CI that shifts from one page refresh to the next
    destroys trust in the dashboard far faster than it informs about the randomness.
    """
    array = np.asarray(pd.Series(values).dropna(), dtype=float)
    n = len(array)
    if n < 5:
        raise StatsError(f"at least 5 observations are needed, got {n}")
    if block_len < 1:
        raise StatsError(f"block_len must be >= 1, got {block_len}")
    if block_len > n:
        raise StatsError(
            f"block_len ({block_len}) exceeds the series length ({n}) — "
            "resampling would just copy the series"
        )

    rng = np.random.default_rng(seed)
    indices = stationary_bootstrap_indices(n, float(block_len), n_iter, rng)
    resampled = array[indices]

    if statistic is None:
        draws = resampled.mean(axis=1)
        point = float(array.mean())
    else:
        draws = np.apply_along_axis(statistic, 1, resampled).astype(float)
        point = float(statistic(array))

    tail = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(draws, [tail, 1.0 - tail])
    return BootstrapCI(
        point=point,
        lo=float(lo),
        hi=float(hi),
        confidence=confidence,
        block_len=float(block_len),
        n_iter=n_iter,
    )


# ===========================================================================
# Regimes: duration and depth
# ===========================================================================
def regime_runs(
    mask: pd.Series, *, depth: pd.Series | None = None, min_obs: int = 1
) -> pd.DataFrame:
    """Consecutive episodes where `mask` is true.

    Columns: start, end, n_obs, duration_days, depth_mean, depth_min, depth_max.

    This is the function that produces the email sentences. "The margin is negative" is
    an observation; "the margin has been negative for 7 months, the longest run since
    2015" is a dated fact the recipient can contest — and therefore can respond to. The
    difference lies entirely in the duration and the depth.
    """
    flag = pd.Series(mask).astype("boolean").fillna(False).astype(bool).sort_index()
    columns = ["start", "end", "n_obs", "duration_days", "depth_mean", "depth_min", "depth_max"]
    if not flag.any():
        return pd.DataFrame(columns=columns)

    if depth is not None:
        depth = pd.Series(depth).reindex(flag.index)

    group_id = (flag != flag.shift()).cumsum()
    rows = []
    for _, chunk in flag[flag].groupby(group_id[flag]):
        if len(chunk) < min_obs:
            continue
        start, end = chunk.index.min(), chunk.index.max()
        row = {
            "start": start,
            "end": end,
            "n_obs": len(chunk),
            "duration_days": (end - start).days + 1 if hasattr(end - start, "days") else np.nan,
            "depth_mean": np.nan,
            "depth_min": np.nan,
            "depth_max": np.nan,
        }
        if depth is not None:
            window = depth.loc[chunk.index].dropna()
            if len(window):
                row["depth_mean"] = float(window.mean())
                row["depth_min"] = float(window.min())
                row["depth_max"] = float(window.max())
        rows.append(row)

    return pd.DataFrame(rows, columns=columns)


# ===========================================================================
# Proportion: exact Clopper-Pearson interval
# ===========================================================================
@dataclass(frozen=True)
class ProportionCI:
    """Exact binomial CI. The only honest one when n is small or the proportion extreme."""

    successes: int
    n: float
    point: float
    lo: float
    hi: float
    confidence: float

    @property
    def summary(self) -> str:
        return (
            f"{self.point:.1%} [exact {self.confidence:.0%} CI: {self.lo:.1%}, "
            f"{self.hi:.1%}] on n = {self.n:g}"
        )


def clopper_pearson(
    successes: int, n: float, *, confidence: float = 0.95
) -> ProportionCI:
    """Exact CI of a proportion — no normal approximation.

    Two uses in this portfolio, and the second is the more important one: T1-1's
    `sign_flip_rate`, and **a backtest's win rate evaluated on n_eff**. At an effective n
    of 7, a 100% win rate has a lower bound far from 100% — that's the number that goes
    in the email.

    `n` accepts a float because an effective n isn't an integer. The CI is then
    approximated by rounding down to the nearest integer — conservative, so in the right
    direction.
    """
    n_int = int(np.floor(n))
    if n_int < 1:
        raise StatsError(f"n must be >= 1, got {n}")
    successes = int(min(successes, n_int))
    if successes < 0:
        raise StatsError(f"successes must be >= 0, got {successes}")

    alpha = 1.0 - confidence
    lo = 0.0 if successes == 0 else float(scipy_stats.beta.ppf(alpha / 2, successes, n_int - successes + 1))
    hi = 1.0 if successes == n_int else float(scipy_stats.beta.ppf(1 - alpha / 2, successes + 1, n_int - successes))
    return ProportionCI(
        successes=successes,
        n=float(n),
        point=successes / n_int,
        lo=lo,
        hi=hi,
        confidence=confidence,
    )
