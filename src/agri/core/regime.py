"""Module T1-M — regime or skill.

Origin of the disagreement: Halsall, Commodity Conversations, 25 Nov 2024. About the
houses posting record cocoa numbers, he asks whether it's because they trade better or
because of market conditions, and notes that success gets judged on the absolute number
rather than on performance relative to conditions.

This isn't a standalone page: it's a section to insert into **every** page, and the spec
requires it to run by default, not as an option — including when the result is
unfavourable. It's precisely when it's unfavourable that it's worth something.

TWO OUTPUTS, AND THE SECOND IS THE ONE THAT GOES IN THE EMAIL
----------------------------------------------------------------
1. `attribute_pnl_to_regime`: regresses per-trade P&L on regime variables. The fit gives
   the "P&L expected from the regime," the residual is the presumed alpha, and the CI on
   the alpha says whether that alpha is distinguishable from zero.
2. `honest_win_rate`: corrects the trade count for position overlap, then computes the
   exact CI of the win rate on that effective n.

The second is the more valuable one. Writing "my 100% win rate on 18 trades is not
distinguishable from a 60% process" yourself before the recipient thinks it is the
cheapest credibility move available — and it's exactly what they were going to think in
silence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from agri.core.resample import EffectiveSample, effective_n_from_trades
from agri.core.stats import (
    BootstrapCI,
    HacRegression,
    ProportionCI,
    clopper_pearson,
    hac_ols,
    stationary_bootstrap_indices,
)

# Default regime variables expected. Every page names them in its own terms (cross-origin
# dispersion for a grain, cross-region for sugar), but the structure doesn't change.
DEFAULT_REGIME_COLUMNS = (
    "vol_realised",
    "term_structure_width",
    "dispersion",
    "duration",
)


class RegimeError(ValueError):
    """Mis-specified attribution — missing columns or insufficient sample."""


# ===========================================================================
# 1. Attributing P&L to the regime
# ===========================================================================
@dataclass(frozen=True)
class RegimeAttribution:
    """What the regime explains, what's left, and whether that remainder is distinguishable from zero."""

    regression: HacRegression
    alpha_ci: BootstrapCI
    sample: EffectiveSample
    regime_columns: tuple[str, ...]

    @property
    def r_squared(self) -> float:
        """Share of P&L explained by market conditions alone."""
        return self.regression.r_squared

    @property
    def alpha(self) -> float:
        """Average per-trade P&L not explained by the regime."""
        return float(self.regression.params["const"])

    @property
    def alpha_is_distinguishable_from_zero(self) -> bool:
        return not self.alpha_ci.includes_zero

    @property
    def is_overfit(self) -> bool:
        """Too many parameters for the effective n.

        Threshold used: fewer than 3 independent observations per estimated parameter.
        Below that, R² measures the model's flexibility, not a market regularity — and
        that has to be said instead of showing off a flattering R².
        """
        n_params = len(self.regime_columns) + 1
        return self.sample.n_eff < 3 * n_params

    @property
    def verdict(self) -> str:
        if self.is_overfit:
            return "not interpretable"
        if self.alpha_is_distinguishable_from_zero:
            return "presumed alpha"
        return "indistinguishable from the regime"

    @property
    def mail_sentence(self) -> str:
        """The numeric sentence, generated from the data — never copied by hand."""
        if self.is_overfit:
            n_params = len(self.regime_columns) + 1
            return (
                f"The regime explains {self.r_squared:.0%} of the P&L variance, but on "
                f"{self.sample.n_eff:.1f} independent observations for {n_params} "
                "parameters, this number measures the model's flexibility, not a "
                "market regularity. I don't read into it."
            )
        if self.alpha_is_distinguishable_from_zero:
            return (
                f"Market conditions explain {self.r_squared:.0%} of the P&L. "
                f"The residual is worth {self.alpha:+.2f} per trade, "
                f"{self.alpha_ci.confidence:.0%} CI [{self.alpha_ci.lo:+.2f}, "
                f"{self.alpha_ci.hi:+.2f}]: it does not contain zero, "
                f"on n_eff = {self.sample.n_eff:.1f}."
            )
        return (
            f"Market conditions explain {self.r_squared:.0%} of the P&L. What's left — "
            f"{self.alpha:+.2f} per trade — has a CI "
            f"[{self.alpha_ci.lo:+.2f}, {self.alpha_ci.hi:+.2f}] that contains zero: "
            f"on n_eff = {self.sample.n_eff:.1f}, this result can't be distinguished "
            "from the market regime."
        )


def attribute_pnl_to_regime(
    trades: pd.DataFrame,
    *,
    hold_days: int,
    pnl_column: str = "pnl",
    regime_columns: tuple[str, ...] | None = None,
    n_iter: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> RegimeAttribution:
    """Regresses per-trade P&L on the regime variables.

    `trades` is indexed by **entry date** (that's what allows the real overlap to be
    measured) and contains at least `pnl_column` and the regime columns.

    The bootstrap block length is set to the holding period: that's the duration over
    which two trades stay dependent.
    """
    columns = tuple(regime_columns) if regime_columns is not None else DEFAULT_REGIME_COLUMNS
    if pnl_column not in trades.columns:
        raise RegimeError(f"missing P&L column: {pnl_column!r}")
    missing = [c for c in columns if c not in trades.columns]
    if missing:
        raise RegimeError(
            f"missing regime columns: {missing}. Each page names its own regime "
            "variables — pass them via regime_columns."
        )

    clean = trades[[pnl_column, *columns]].dropna()
    if len(clean) < len(columns) + 3:
        raise RegimeError(
            f"not enough trades: n={len(clean)} for {len(columns)} regime variables"
        )

    regression = hac_ols(clean[pnl_column], clean[list(columns)])
    sample = effective_n_from_trades(pd.DatetimeIndex(clean.index), hold_days)

    # CI on the alpha: the average residual, block-bootstrapped at the holding-period length.
    block_len = min(float(hold_days), float(len(clean)))
    alpha_ci = _bootstrap_intercept(
        y=clean[pnl_column].to_numpy(dtype=float),
        regressors=clean[list(columns)].to_numpy(dtype=float),
        point=float(regression.params["const"]),
        block_len=block_len,
        n_iter=n_iter,
        confidence=confidence,
        seed=seed,
    )

    return RegimeAttribution(
        regression=regression,
        alpha_ci=alpha_ci,
        sample=sample,
        regime_columns=columns,
    )


def _bootstrap_intercept(
    *,
    y: np.ndarray,
    regressors: np.ndarray,
    point: float,
    block_len: float,
    n_iter: int,
    confidence: float,
    seed: int,
    chunk: int = 500,
) -> BootstrapCI:
    """Block-bootstrap CI on the intercept, **with a full refit** of the regression.

    Why refit rather than resampling the residuals, which would be ten times cheaper:
    the intercept is `ȳ − Σ β̂ⱼ x̄ⱼ`. Bootstrapping the residuals with β̂ held fixed
    treats the slopes as known and ignores how their estimation error propagates into
    the intercept, amplified by each regressor's mean. On a regime variable centred far
    from zero — an average 30-day holding period, say — a 0.003 error on the slope
    shifts alpha by 0.09. The CI would come out far too narrow, and would conclude
    there's an alpha where there is only estimation noise: exactly the failure mode
    this module exists to prevent.

    The computation is chunked to bound memory: (n_iter, n, k) as a single block would
    exceed a gigabyte on a normally sized backtest.
    """
    n, k = regressors.shape
    design = np.column_stack([np.ones(n), regressors])   # the intercept is column 0
    rng = np.random.default_rng(seed)
    indices = stationary_bootstrap_indices(n, block_len, n_iter, rng)

    intercepts = np.empty(n_iter, dtype=float)
    for start in range(0, n_iter, chunk):
        idx = indices[start : start + chunk]
        x_b = design[idx]                                 # (c, n, k+1)
        y_b = y[idx]                                      # (c, n)
        xtx = np.einsum("cni,cnj->cij", x_b, x_b)
        xty = np.einsum("cni,cn->ci", x_b, y_b)
        try:
            # xty carries an extra axis: under numpy 2, a 2-D second operand is read as
            # a single (m, n) matrix rather than as a batch of vectors
            beta = np.linalg.solve(xtx, xty[..., None])[..., 0]
        except np.linalg.LinAlgError:
            # a resample can be degenerate (a single block repeated): the pseudo-inverse
            # handles it without taking down the whole bootstrap
            beta = np.einsum("cij,cj->ci", np.linalg.pinv(xtx), xty)
        intercepts[start : start + chunk] = beta[:, 0]

    tail = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(intercepts, [tail, 1.0 - tail])
    return BootstrapCI(
        point=point,
        lo=float(lo),
        hi=float(hi),
        confidence=confidence,
        block_len=float(block_len),
        n_iter=n_iter,
    )


# ===========================================================================
# 2. The honest win rate — the sentence for the email
# ===========================================================================
@dataclass(frozen=True)
class WinRateHonesty:
    """Displayed win rate against defensible win rate, and the gap between the two."""

    n_trades: int
    n_wins: int
    hold_days: int
    sample: EffectiveSample
    naive: ProportionCI
    honest: ProportionCI

    @property
    def lower_bound_cost(self) -> float:
        """Points of lower bound lost by correcting for overlap."""
        return self.naive.lo - self.honest.lo

    @property
    def mail_sentence(self) -> str:
        """The pre-emptive self-critique sentence, computed straight from the backtest."""
        equivalent = self.honest.lo
        return (
            f"Over {self.n_trades} positions held {self.hold_days} days, the overlap "
            f"({self.sample.overlap:.1f} positions open on average) reduces the sample "
            f"to {self.sample.n_eff:.1f} independent draws. The {self.naive.point:.0%} "
            f"win rate then has an exact CI of [{equivalent:.0%}, {self.honest.hi:.0%}]: "
            f"it is not distinguishable from a process that succeeds "
            f"{equivalent * 10:.0f} times out of 10. Announcing the "
            f"{self.naive.lo:.0%} bound while ignoring the overlap would overstate what "
            f"this backtest demonstrates by {self.lower_bound_cost * 100:.0f} points."
        )


def honest_win_rate(
    entry_dates: pd.DatetimeIndex,
    wins: pd.Series | np.ndarray | list[bool],
    *,
    hold_days: int,
    confidence: float = 0.95,
) -> WinRateHonesty:
    """Win rate with an exact CI computed on the **effective n**, not the trade count.

    This is the computation to run on the copper backtest before sending anything. 18
    positions held 30 days with several simultaneous trades are not 18 independent
    draws, and a 100% win rate on ~7 draws has a lower bound around 59% — far from what
    "100% on 18 trades" implies.
    """
    entries = pd.DatetimeIndex(entry_dates)
    outcomes = np.asarray(pd.Series(list(wins)).astype(bool))
    if len(outcomes) != len(entries):
        raise RegimeError(
            f"{len(entries)} entry dates for {len(outcomes)} outcomes — "
            "the two must correspond trade for trade"
        )
    if len(entries) == 0:
        raise RegimeError("no trades")

    sample = effective_n_from_trades(entries, hold_days)
    n_wins = int(outcomes.sum())
    win_share = n_wins / len(outcomes)

    # The observed PROPORTION is carried over onto the effective n, rather than keeping
    # the raw win count: it's the number of observations that overlap reduces, not the
    # success rate.
    effective_wins = int(round(win_share * np.floor(sample.n_eff)))

    return WinRateHonesty(
        n_trades=len(outcomes),
        n_wins=n_wins,
        hold_days=hold_days,
        sample=sample,
        naive=clopper_pearson(n_wins, len(outcomes), confidence=confidence),
        honest=clopper_pearson(effective_wins, sample.n_eff, confidence=confidence),
    )
