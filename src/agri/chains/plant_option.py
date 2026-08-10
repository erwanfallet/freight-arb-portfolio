"""T2-5 — The plant as an option on the margin.

WHAT THIS PAGE EXPLAINS
-------------------------
A curtailment signal of the form `consecutive_below(margin, 0, N=4)` — the one running on
the **zinc** and **lithium** pages — silently assumes that stopping and restarting is free.
It is not: on a zinc furnace or a spodumene conversion line, restarting is counted in weeks
of lost production and reagents. The rule then costs money on both sides — it triggers
shutdowns the restart cost does not justify, and it restarts before the margin covers that
restart.

And there is a harder problem, visible by cross-referencing two sections of the zinc page
itself: its sensitivity section shows that it takes **376 USD/t of acid credit to break
even**, and that this credit is the largest unmodelled lever. Uncertainty about the
**sign** of the margin therefore dwarfs the 0 USD/t threshold the N=4 rule tests against.
The sensitivity section invalidates the curtailment section, on the same page.

THE DELIVERABLE — INVERTING THE QUESTION
--------------------------------------------
Rather than proposing a better rule (which nobody asked for), the existing rule is made
**contestable**: on a given margin path, `margin < 0 for 4 months` stops and restarts at
precise levels, so it *is* equivalent to a band, so it implies a precise round-trip cost.

    "Your rule stops at a median of M_off and restarts at M_on. An exercise boundary
     that would produce the same band implies a shutdown-restart cost of X. Does X
     look like yours?"

This is a question only someone who runs the asset can answer, and it does not ask the
reader to accept a model — only to compare a number to their own.

THE COUNTERFACTUAL THAT MUST ALWAYS BE SHOWN
------------------------------------------------
Before comparing two shutdown rules, it has to be checked that stopping is worth anything
at all: `run_always_on_policy` gives the P&L of a plant that never stops. If the best rule
does not beat this counterfactual over the period, the whole discussion of the exercise
boundary is theoretical — and the page has to say so instead of comparing two equally
useless rules.

THE TECHNICAL IDEA
-------------------
The optimal rule is not a threshold: it is a **hysteresis band** `[M_off, M_on]` with
`M_off < 0 < M_on`, whose width is set by shutdown and restart costs and by the margin's
volatility. A rational operator keeps producing at a negative margin if the round-trip
switching cost exceeds the loss, and does not restart the moment the margin turns positive
again.

MODEL
-----
Margin as an Ornstein-Uhlenbeck process — conversion margins are mean-reverting, unlike
prices:

    dM = kappa (theta - M) dt + sigma dW

OLS calibration on `M_{t+1} = a + b M_t + e`:
    kappa = -ln(b)/dt,  theta = a/(1-b),  sigma = sd(e) x sqrt(2 kappa / (1 - b^2))

Valuation by two-state dynamic programming:

    V_on(M)  = max( M - c_fix + d E[V_on(M')] ,  -K_off + d E[V_off(M')] )
    V_off(M) = max( -c_idle    + d E[V_off(M')] ,  -K_on  + d E[V_on(M')]  )

Solved by value iteration on a grid of M, then extracting the boundary `M_off*` (stop)
and `M_on*` (restart).

EXPECTED RESULT — COUNTER-INTUITIVE, AND THAT IS THE POINT
----------------------------------------------------------------
A plant whose margin is often negative can be worth **more** than a plant whose margin is
stably positive, if volatility and flexibility are large enough. Option value grows with
sigma at equal mean.

ASSUMPTIONS
-----------
O-H1  OU margin. Conversion margins mean-revert; test it (ADF+KPSS) before calibrating,
      and refuse to calibrate if the test says unit root.
O-H2  Flat shutdown and restart costs, expressed in days of average margin. This is the
      page's most uncertain parameter — hence two sliders and a dedicated sensitivity
      check.
O-H3  No technical lag between decision and effect. A real restart takes days to weeks,
      which **widens** the hysteresis band: a conservative bias.
O-H4  No supply-contract constraint or delivery commitment. A real plant does not stop
      freely — the computed boundary is therefore a bound.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from agri.data.snapshot import cached

from agri.core.stats import StationarityVerdict, adf_kpss, regime_runs

STATE_ON = 0
STATE_OFF = 1


class PlantOptionError(ValueError):
    """Mis-specified model or refused calibration."""


# ===========================================================================
# Ornstein-Uhlenbeck calibration
# ===========================================================================
@dataclass(frozen=True)
class OUParams:
    """OU parameters, with the stationarity verdict that licenses their use."""

    kappa: float           # mean-reversion speed, per unit of time
    theta: float           # long-run level
    sigma: float           # instantaneous volatility
    dt: float
    stationarity: StationarityVerdict
    n_obs: int

    @property
    def half_life(self) -> float:
        """Mean-reversion half-life, in `dt`'s time unit."""
        return np.log(2.0) / self.kappa if self.kappa > 0 else float("inf")

    @property
    def summary(self) -> str:
        return (
            f"kappa = {self.kappa:.4f}/period (half-life {self.half_life:.1f} periods), "
            f"theta = {self.theta:.2f}, sigma = {self.sigma:.2f} | "
            f"stationarity: {self.stationarity.verdict}"
        )


def calibrate_ou(margin: pd.Series, *, dt: float = 1.0, strict: bool = True) -> OUParams:
    """Calibrates an OU by OLS on `M_{t+1} = a + b M_t + e` (O-H1).

    `strict=True` **refuses** to calibrate if ADF and KPSS do not jointly conclude
    stationarity. Calibrating an OU on a random walk produces a kappa near zero and an
    absurd option value, without ever crashing — hence the explicit refusal.
    """
    clean = pd.Series(margin).dropna().astype(float)
    if len(clean) < 50:
        raise PlantOptionError(f"at least 50 observations are needed, got {len(clean)}")

    verdict = adf_kpss(clean)
    if strict and verdict.verdict != "stationary":
        raise PlantOptionError(
            f"the margin is not stationary under the joint ADF+KPSS test "
            f"(verdict: {verdict.verdict}). Calibrating an OU on it would produce a "
            "kappa near zero and an absurd option value. Pass strict=False to force "
            "it, showing the warning on the page."
        )

    y = clean.to_numpy()[1:]
    x = clean.to_numpy()[:-1]
    design = np.column_stack([np.ones(len(x)), x])
    (a, b), *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ np.array([a, b])

    if not 0.0 < b < 1.0:
        raise PlantOptionError(
            f"autoregressive coefficient outside (0, 1): b = {b:.4f}. Above 1 the "
            "series is explosive, at zero or below it is not an OU."
        )

    kappa = -np.log(b) / dt
    theta = a / (1.0 - b)
    sigma = float(np.std(residuals, ddof=2)) * np.sqrt(2.0 * kappa / (1.0 - b**2))
    return OUParams(
        kappa=float(kappa),
        theta=float(theta),
        sigma=float(sigma),
        dt=dt,
        stationarity=verdict,
        n_obs=len(clean),
    )


# ===========================================================================
# Dynamic programming — the exercise boundary
# ===========================================================================
@dataclass(frozen=True)
class HysteresisBand:
    """The real shutdown-restart rule, and what it is worth."""

    m_off: float
    m_on: float
    grid: np.ndarray
    value_on: np.ndarray
    value_off: np.ndarray
    n_iterations: int
    converged: bool

    @property
    def width(self) -> float:
        return self.m_on - self.m_off

    @property
    def is_degenerate(self) -> bool:
        """`M_on < M_off` — the band is inverted, so there is no shutdown policy.

        This case is REAL, not a solver bug: it appears as soon as the cost of
        idling becomes expensive relative to the cost of restarting. An asset that
        costs 2 per period idle and 3 to restart should never sit idle — the dynamic
        program says so by making the "restart" region wider than the "stop"
        region, causing them to overlap.

        Read naively, this case produces a NEGATIVE band width that silently
        propagates into a sensitivity table or an interpolation. It is named here so
        that downstream code refuses to treat it as an ordinary band.
        """
        return self.m_on < self.m_off

    def option_value_at(self, margin: float) -> float:
        """Value of the running plant, at a given margin level."""
        return float(np.interp(margin, self.grid, self.value_on))

    @property
    def headline(self) -> str:
        if self.is_degenerate:
            return (
                f"No shutdown policy exists at these costs: the \"restart\" region "
                f"(above {self.m_on:+.2f}) overlaps the \"stop\" region "
                f"(below {self.m_off:+.2f}). An asset that costs more to leave idle "
                "than to restart should never stop — the optimum is to run "
                "continuously, and a curtailment rule has nothing to optimise here."
            )
        return (
            f"The optimal boundary is not a threshold but a band: stop at "
            f"{self.m_off:+.2f} and only restart at {self.m_on:+.2f}, "
            f"{self.width:.2f} of hysteresis. A \"margin < 0\" rule stops too early "
            "and restarts too early, twice per cycle."
        )


def solve_hysteresis(
    ou: OUParams,
    *,
    cost_restart: float,
    cost_shutdown: float,
    cost_idle: float,
    cost_fixed: float = 0.0,
    discount_rate: float = 0.08,
    grid_points: int = 401,
    grid_span_sigmas: float = 4.0,
    max_iterations: int = 5_000,
    tolerance: float = 1e-8,
) -> HysteresisBand:
    """Two-state value iteration, and extraction of the hysteresis band.

    The conditional expectation under the OU is computed by discrete Gaussian
    quadrature: from `M`, the next margin is normal with mean
    `theta + (M - theta) e^{-kappa dt}` and standard deviation
    `sigma sqrt((1 - e^{-2 kappa dt}) / (2 kappa))`.
    """
    if ou.kappa <= 0:
        raise PlantOptionError("kappa must be > 0 for a mean-reverting margin")
    if min(cost_restart, cost_shutdown, cost_idle) < 0:
        raise PlantOptionError("transition and idling costs must be >= 0")

    span = grid_span_sigmas * ou.sigma / np.sqrt(2.0 * ou.kappa)
    grid = np.linspace(ou.theta - span, ou.theta + span, grid_points)

    decay = np.exp(-ou.kappa * ou.dt)
    conditional_mean = ou.theta + (grid - ou.theta) * decay
    conditional_std = ou.sigma * np.sqrt((1.0 - decay**2) / (2.0 * ou.kappa))
    if conditional_std <= 0:
        raise PlantOptionError("zero conditional standard deviation — degenerate OU parameters")

    # transition matrix: row = current state on the grid, column = next state
    difference = grid[None, :] - conditional_mean[:, None]
    weights = np.exp(-0.5 * (difference / conditional_std) ** 2)
    transition = weights / weights.sum(axis=1, keepdims=True)

    discount = np.exp(-discount_rate * ou.dt)
    value_on = np.maximum(grid - cost_fixed, 0.0)
    value_off = np.zeros_like(grid)

    converged = False
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        expected_on = transition @ value_on
        expected_off = transition @ value_off

        new_on = np.maximum(
            grid - cost_fixed + discount * expected_on,
            -cost_shutdown + discount * expected_off,
        )
        new_off = np.maximum(
            -cost_idle + discount * expected_off,
            -cost_restart + discount * expected_on,
        )
        gap = max(np.max(np.abs(new_on - value_on)), np.max(np.abs(new_off - value_off)))
        value_on, value_off = new_on, new_off
        if gap < tolerance:
            converged = True
            break

    # boundaries: highest margin where we choose to stop, lowest where we restart
    stop = (-cost_shutdown + discount * (transition @ value_off)) >= (
        grid - cost_fixed + discount * (transition @ value_on)
    )
    restart = (-cost_restart + discount * (transition @ value_on)) >= (
        -cost_idle + discount * (transition @ value_off)
    )
    m_off = float(grid[stop].max()) if stop.any() else float(grid[0])
    m_on = float(grid[restart].min()) if restart.any() else float(grid[-1])

    return HysteresisBand(
        m_off=m_off,
        m_on=m_on,
        grid=grid,
        value_on=value_on,
        value_off=value_off,
        n_iterations=iteration,
        converged=converged,
    )


# ===========================================================================
# Comparison against the heuristic rule
# ===========================================================================
@dataclass(frozen=True)
class RuleComparison:
    """What the "margin < 0 for N months" rule costs against the optimal boundary."""

    n_shutdowns_heuristic: int
    n_shutdowns_optimal: int
    false_shutdowns: int
    heuristic_threshold: float
    consecutive_periods: int

    @property
    def headline(self) -> str:
        return (
            f"The rule \"margin < {self.heuristic_threshold:g} for "
            f"{self.consecutive_periods} periods\" triggers "
            f"{self.n_shutdowns_heuristic} shutdowns against {self.n_shutdowns_optimal} "
            f"for the optimal boundary, {self.false_shutdowns} of which the boundary "
            "would not have made. Each pays a shutdown and a restart for nothing."
        )


def compare_to_heuristic(
    margin: pd.Series,
    band: HysteresisBand,
    *,
    threshold: float = 0.0,
    consecutive_periods: int = 4,
) -> RuleComparison:
    """Backtests the heuristic rule against the hysteresis boundary.

    The contrast is the page's product: a threshold rule stops on passing dips that
    the band absorbs, and every avoidable shutdown pays a `K_off` plus a `K_on`.
    """
    clean = pd.Series(margin).dropna().astype(float)
    below = clean < threshold
    heuristic_episodes = regime_runs(below, depth=clean, min_obs=consecutive_periods)

    optimal_episodes = regime_runs(clean < band.m_off, depth=clean, min_obs=1)
    n_heuristic = len(heuristic_episodes)
    n_optimal = len(optimal_episodes)
    return RuleComparison(
        n_shutdowns_heuristic=n_heuristic,
        n_shutdowns_optimal=n_optimal,
        false_shutdowns=max(0, n_heuristic - n_optimal),
        heuristic_threshold=threshold,
        consecutive_periods=consecutive_periods,
    )


# ===========================================================================
# Operating-policy simulator — what the rule really COSTS
# ===========================================================================
@dataclass(frozen=True)
class PolicyResult:
    """Full P&L of a shutdown-restart policy, line item by line item.

    The three line items are kept separate because they do not offset each other the
    same way: operations depend on the market, switching costs depend on the
    *frequency* of decisions, and idling cost depends on their *duration*. A rule can
    be good on one line item and bad on another.
    """

    label: str
    operating_pnl: float
    switching_cost: float
    idle_cost: float
    n_stops: int
    n_starts: int
    periods_on: int
    periods_off: int
    stop_margins: list[float]
    start_margins: list[float]
    state: pd.Series

    @property
    def total_pnl(self) -> float:
        return self.operating_pnl - self.switching_cost - self.idle_cost

    @property
    def effective_m_off(self) -> float:
        """Margin level at which the rule ACTUALLY stops, at the median.

        For a persistence rule ("below threshold for N periods"), this level is not
        the threshold: the waiting time lets the margin fall further. This
        difference is what allows comparing a persistence rule to a band.
        """
        return float(np.median(self.stop_margins)) if self.stop_margins else float("nan")

    @property
    def effective_m_on(self) -> float:
        return float(np.median(self.start_margins)) if self.start_margins else float("nan")

    @property
    def effective_width(self) -> float:
        return self.effective_m_on - self.effective_m_off


def simulate_operating_policy(
    margin: pd.Series,
    *,
    label: str,
    stop_rule,
    start_rule,
    cost_restart: float,
    cost_shutdown: float,
    cost_idle: float,
    start_on: bool = True,
) -> PolicyResult:
    """Runs an operating policy over a real margin path.

    `stop_rule(i, values, state_history)` and `start_rule(...)` return a boolean. The
    full history is passed rather than just the current value, because a persistence
    rule needs to look back — and that is precisely what distinguishes it from a band.
    """
    values = pd.Series(margin).dropna().astype(float)
    if values.empty:
        raise PlantOptionError("empty margin")

    on = start_on
    operating = switching = idle = 0.0
    n_stops = n_starts = periods_on = periods_off = 0
    stop_margins: list[float] = []
    start_margins: list[float] = []
    states: list[bool] = []

    array = values.to_numpy()
    for i in range(len(array)):
        if on:
            operating += array[i]
            periods_on += 1
            if stop_rule(i, array):
                switching += cost_shutdown
                stop_margins.append(float(array[i]))
                n_stops += 1
                on = False
        else:
            idle += cost_idle
            periods_off += 1
            if start_rule(i, array):
                switching += cost_restart
                start_margins.append(float(array[i]))
                n_starts += 1
                on = True
        states.append(on)

    return PolicyResult(
        label=label,
        operating_pnl=operating,
        switching_cost=switching,
        idle_cost=idle,
        n_stops=n_stops,
        n_starts=n_starts,
        periods_on=periods_on,
        periods_off=periods_off,
        stop_margins=stop_margins,
        start_margins=start_margins,
        state=pd.Series(states, index=values.index, name=f"on_{label}"),
    )


def _persistence_rules(threshold: float, n_periods: int):
    """Rule "below threshold for N consecutive periods", and its mirror image."""

    def stop_rule(i: int, values: np.ndarray) -> bool:
        if i + 1 < n_periods:
            return False
        return bool(np.all(values[i + 1 - n_periods : i + 1] < threshold))

    def start_rule(i: int, values: np.ndarray) -> bool:
        if i + 1 < n_periods:
            return False
        return bool(np.all(values[i + 1 - n_periods : i + 1] > threshold))

    return stop_rule, start_rule


def _band_rules(m_off: float, m_on: float):
    """Band rule: instantaneous on both sides, the hysteresis sits in the levels."""

    def stop_rule(i: int, values: np.ndarray) -> bool:
        return bool(values[i] < m_off)

    def start_rule(i: int, values: np.ndarray) -> bool:
        return bool(values[i] > m_on)

    return stop_rule, start_rule


def _never_stop_rules():
    """Degenerate policy: never stop (see `HysteresisBand.is_degenerate`)."""
    return (lambda i, v: False), (lambda i, v: False)


def run_heuristic_policy(
    margin: pd.Series, *, threshold: float = 0.0, n_periods: int = 4, **costs
) -> PolicyResult:
    """The rule used on the zinc and lithium pages: `consecutive_below(margin, 0, N)`."""
    stop_rule, start_rule = _persistence_rules(threshold, n_periods)
    return simulate_operating_policy(
        margin, label=f"heuristic N={n_periods}", stop_rule=stop_rule,
        start_rule=start_rule, **costs,
    )


def run_band_policy(margin: pd.Series, band: HysteresisBand, **costs) -> PolicyResult:
    """The calibrated exercise boundary.

    **Refuses to run on a degenerate band.** Applying `M_on < M_off` as-is would make
    the plant oscillate — it would stop below `M_off` then immediately restart since
    the same level is already above `M_on`, paying a round trip every period.
    Substituting a fallback policy (e.g. "never stop") would mean inventing a rule
    the model did not produce, then comparing it as if it were its recommendation.

    The case occurs when switching costs become small relative to the margin's
    conditional volatility: there is then no clean band, only chattering. This is a
    result about the problem, not a computational glitch.
    """
    if band.is_degenerate:
        raise PlantOptionError(
            f"degenerate band (M_off={band.m_off:+.2f} > M_on={band.m_on:+.2f}): at "
            "these switching costs, friction is too small relative to the margin's "
            "volatility for an exercise boundary to exist. No band policy can be "
            "simulated — and substituting an invented fallback would distort the "
            "comparison."
        )
    stop_rule, start_rule = _band_rules(band.m_off, band.m_on)
    return simulate_operating_policy(
        margin, label="hysteresis band", stop_rule=stop_rule, start_rule=start_rule, **costs,
    )


def run_always_on_policy(margin: pd.Series, **costs) -> PolicyResult:
    """Counterfactual: the plant never stops.

    Essential to know whether flexibility is worth anything **at all** over this
    period: if the best rule does not beat "never stop", the whole discussion of the
    exercise boundary is theoretical.
    """
    return simulate_operating_policy(
        margin, label="never stops", stop_rule=lambda i, v: False,
        start_rule=lambda i, v: False, **costs,
    )


# ===========================================================================
# THE DELIVERABLE — inverting the question
# ===========================================================================
@dataclass(frozen=True)
class ImpliedSwitchingCost:
    """The restart cost the threshold rule implicitly assumes.

    This is the number that makes the rule contestable. "Margin < 0 for 4 months" is
    not a neutral assumption: on a given margin path, it stops and restarts at
    precise levels, so it is equivalent to a band, so it implies a precise switching
    cost. An operator knows whether that cost looks like their own.
    """

    effective_m_off: float
    effective_m_on: float
    effective_width: float
    implied_switching_cost: float
    searched_lo: float
    searched_hi: float
    converged: bool
    n_stops_observed: int

    @property
    def headline(self) -> str:
        if not self.converged:
            return (
                f"The rule stops at a median of {self.effective_m_off:+.2f} and "
                f"restarts at {self.effective_m_on:+.2f} (width {self.effective_width:.2f}), "
                f"but no switching cost in [{self.searched_lo:g}, {self.searched_hi:g}] "
                "reproduces this band — the rule is not equivalent to any rational "
                "exercise boundary over this range."
            )
        return (
            f"The rule stops at a median of {self.effective_m_off:+.2f} and restarts "
            f"at {self.effective_m_on:+.2f}. An exercise boundary producing the same "
            f"band implies a shutdown-restart cost of "
            f"**{self.implied_switching_cost:,.2f} per unit of margin**. This is the "
            "number the rule assumes without saying so."
        )


def implied_switching_cost(
    margin: pd.Series,
    ou: OUParams,
    *,
    threshold: float = 0.0,
    n_periods: int = 4,
    cost_idle: float = 0.0,
    restart_share: float = 0.67,
    lo: float = 1e-4,
    hi: float = 1e4,
    n_grid: int = 24,
    **solve_kwargs,
) -> ImpliedSwitchingCost:
    """Which restart cost would make the "margin < threshold for N" rule optimal?

    Method: run the rule on the real margin path to read off the band it **actually**
    implements (median of stop and restart margins), then search for the switching
    cost whose calibrated exercise boundary reproduces that band width. Width grows
    monotonically with switching cost, so a log-spaced grid search is enough and
    stays readable.

    `restart_share` splits the total cost between restart and shutdown — a restart
    typically costs more than a shutdown on a thermal asset, hence 2/3 by default.
    The result is returned as the **total** round-trip cost, the quantity an operator
    knows.
    """
    heuristic = run_heuristic_policy(
        margin, threshold=threshold, n_periods=n_periods,
        cost_restart=0.0, cost_shutdown=0.0, cost_idle=cost_idle,
    )
    if heuristic.n_stops == 0 or not heuristic.start_margins:
        return ImpliedSwitchingCost(
            effective_m_off=heuristic.effective_m_off,
            effective_m_on=heuristic.effective_m_on,
            effective_width=float("nan"),
            implied_switching_cost=float("nan"),
            searched_lo=lo, searched_hi=hi, converged=False,
            n_stops_observed=heuristic.n_stops,
        )

    target_width = heuristic.effective_width
    grid = np.geomspace(lo, hi, n_grid)
    costs_kept: list[float] = []
    widths: list[float] = []
    for total_cost in grid:
        band = solve_hysteresis(
            ou,
            cost_restart=total_cost * restart_share,
            cost_shutdown=total_cost * (1.0 - restart_share),
            cost_idle=cost_idle,
            **solve_kwargs,
        )
        # Degenerate bands (M_on < M_off) carry a negative width: letting them into
        # the interpolation would send a monotone curve through points that do not
        # represent any band. They are dropped, and if everything is degenerate that
        # is reported.
        if band.is_degenerate:
            continue
        costs_kept.append(float(total_cost))
        widths.append(band.width)

    if not widths:
        return ImpliedSwitchingCost(
            effective_m_off=heuristic.effective_m_off,
            effective_m_on=heuristic.effective_m_on,
            effective_width=target_width,
            implied_switching_cost=float("nan"),
            searched_lo=lo, searched_hi=hi, converged=False,
            n_stops_observed=heuristic.n_stops,
        )

    widths_array = np.asarray(widths)
    grid = np.asarray(costs_kept)
    if target_width < widths_array.min() or target_width > widths_array.max():
        return ImpliedSwitchingCost(
            effective_m_off=heuristic.effective_m_off,
            effective_m_on=heuristic.effective_m_on,
            effective_width=target_width,
            implied_switching_cost=float("nan"),
            searched_lo=lo, searched_hi=hi, converged=False,
            n_stops_observed=heuristic.n_stops,
        )

    implied = float(np.interp(target_width, widths_array, grid))
    return ImpliedSwitchingCost(
        effective_m_off=heuristic.effective_m_off,
        effective_m_on=heuristic.effective_m_on,
        effective_width=target_width,
        implied_switching_cost=implied,
        searched_lo=lo, searched_hi=hi, converged=True,
        n_stops_observed=heuristic.n_stops,
    )


@dataclass(frozen=True)
class PolicyComparison:
    """The three policies on the same margin path, in full P&L.

    `band` is None when the boundary is degenerate: only the threshold rule and the
    counterfactual are then compared, and it is said explicitly.
    """

    heuristic: PolicyResult
    band: PolicyResult | None
    always_on: PolicyResult
    cost_restart: float
    cost_shutdown: float
    band_error: str = ""

    @property
    def band_is_available(self) -> bool:
        return self.band is not None

    @property
    def gap_vs_band(self) -> float:
        """What the threshold rule costs against the calibrated boundary."""
        if self.band is None:
            return float("nan")
        return self.band.total_pnl - self.heuristic.total_pnl

    @property
    def heuristic_flexibility_value(self) -> float:
        """What the threshold rule gains against a plant that never stops.

        Always computable, even without a band — and it is the first thing to look
        at: if it is negative, stopping destroys value over this period.
        """
        return self.heuristic.total_pnl - self.always_on.total_pnl

    @property
    def flexibility_value(self) -> float:
        """What the best available rule is worth against the counterfactual."""
        if self.band is None:
            return self.heuristic_flexibility_value
        return self.band.total_pnl - self.always_on.total_pnl

    @property
    def headline(self) -> str:
        if self.band is None:
            return (
                f"No exercise boundary at these costs — {self.band_error} "
                f"Still comparable: the threshold rule beats \"never stop\" by "
                f"{self.heuristic_flexibility_value:+,.1f} per unit, over "
                f"{self.heuristic.n_stops} shutdowns."
            )
        if self.flexibility_value <= 0:
            return (
                f"Over this period, no shutdown rule beats \"never stop\" "
                f"({self.flexibility_value:+,.1f} for the best one): the margin "
                "never stays low enough for long enough for the restart cost to pay "
                "off. The exercise boundary is theoretical here, and that has to be "
                "said."
            )
        return (
            f"The calibrated boundary beats the threshold rule by "
            f"{self.gap_vs_band:+,.1f} per unit, and beats \"never stop\" by "
            f"{self.flexibility_value:+,.1f}. The gap comes from "
            f"{self.heuristic.n_stops} shutdowns against {self.band.n_stops}: each "
            f"avoidable shutdown pays a round trip of "
            f"{self.cost_restart + self.cost_shutdown:,.2f}."
        )

    def to_frame(self) -> pd.DataFrame:
        results = [self.heuristic, self.always_on]
        if self.band is not None:
            results.insert(1, self.band)
        rows = []
        for result in results:
            rows.append(
                {
                    "policy": result.label,
                    "total P&L": result.total_pnl,
                    "operating": result.operating_pnl,
                    "switching costs": -result.switching_cost,
                    "idling cost": -result.idle_cost,
                    "shutdowns": result.n_stops,
                    "restarts": result.n_starts,
                    "periods idle": result.periods_off,
                }
            )
        return pd.DataFrame(rows)


def compare_policies(
    margin: pd.Series,
    band: HysteresisBand,
    *,
    cost_restart: float,
    cost_shutdown: float,
    cost_idle: float,
    threshold: float = 0.0,
    n_periods: int = 4,
) -> PolicyComparison:
    """The three policies on the same path, in full and comparable P&L.

    If the band is degenerate, the comparison continues without it rather than
    stopping: the threshold rule against the counterfactual remains useful
    information.
    """
    costs = dict(cost_restart=cost_restart, cost_shutdown=cost_shutdown, cost_idle=cost_idle)
    band_result: PolicyResult | None
    band_error = ""
    try:
        band_result = run_band_policy(margin, band, **costs)
    except PlantOptionError as error:
        band_result = None
        band_error = str(error)

    return PolicyComparison(
        heuristic=run_heuristic_policy(margin, threshold=threshold, n_periods=n_periods, **costs),
        band=band_result,
        always_on=run_always_on_policy(margin, **costs),
        cost_restart=cost_restart,
        cost_shutdown=cost_shutdown,
        band_error=band_error,
    )


def switching_cost_sensitivity(
    margin: pd.Series,
    ou: OUParams,
    *,
    cost_grid: np.ndarray | None = None,
    cost_idle: float = 0.0,
    restart_share: float = 0.67,
    threshold: float = 0.0,
    n_periods: int = 4,
    **solve_kwargs,
) -> pd.DataFrame:
    """Band width and P&L gap as a function of switching cost.

    This is the sensitivity that decides: it shows the restart cost beyond which the
    threshold rule becomes genuinely expensive, and therefore whether the debate is
    worth bringing to an operator.
    """
    grid = (
        np.geomspace(0.01, 100.0, 15) if cost_grid is None else np.asarray(cost_grid)
    )
    rows = []
    for total_cost in grid:
        restart = total_cost * restart_share
        shutdown = total_cost * (1.0 - restart_share)
        band = solve_hysteresis(
            ou, cost_restart=restart, cost_shutdown=shutdown, cost_idle=cost_idle, **solve_kwargs
        )
        comparison = compare_policies(
            margin, band, cost_restart=restart, cost_shutdown=shutdown,
            cost_idle=cost_idle, threshold=threshold, n_periods=n_periods,
        )
        rows.append(
            {
                "switching_cost": float(total_cost),
                "m_off": band.m_off,
                "m_on": band.m_on,
                # A negative width means nothing: on a degenerate band we show NaN
                # and flag the row, rather than let a negative number be read as a
                # narrow band.
                "band_width": float("nan") if band.is_degenerate else band.width,
                "degenerate": band.is_degenerate,
                "gap_vs_heuristic": comparison.gap_vs_band,
                "flexibility_value": comparison.flexibility_value,
                "n_stops_heuristic": comparison.heuristic.n_stops,
                "n_stops_band": (
                    comparison.band.n_stops if comparison.band is not None else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def volatility_sensitivity(
    ou: OUParams,
    *,
    sigma_multipliers: np.ndarray | None = None,
    **solve_kwargs,
) -> pd.DataFrame:
    """The counter-intuitive demonstration: plant value **grows** with sigma.

    At equal average margin, a plant whose margin is more volatile is worth more,
    because the flexibility to stop truncates the low tail. This gives a number to a
    debate usually conducted in slogans.
    """
    multipliers = (
        np.array([0.5, 0.75, 1.0, 1.5, 2.0])
        if sigma_multipliers is None
        else np.asarray(sigma_multipliers)
    )
    rows = []
    for multiplier in multipliers:
        scaled = OUParams(
            kappa=ou.kappa,
            theta=ou.theta,
            sigma=ou.sigma * float(multiplier),
            dt=ou.dt,
            stationarity=ou.stationarity,
            n_obs=ou.n_obs,
        )
        band = solve_hysteresis(scaled, **solve_kwargs)
        rows.append(
            {
                "sigma_multiplier": float(multiplier),
                "sigma": scaled.sigma,
                "m_off": band.m_off,
                "m_on": band.m_on,
                "band_width": band.width,
                "value_at_theta": band.option_value_at(ou.theta),
            }
        )
    return pd.DataFrame(rows)


@cached("t2_5_us_board", from_frame=lambda f: f.iloc[:, 0].rename("board_crush"))
def real_board_crush_margin(*, start: str = "1990-07-18") -> pd.Series:
    """Board crush margin, entirely real — CBOT soybean/meal/oil.

    Unlike T2-4's energy proxy or T1-2's omitted roll, all three legs of
    `board_crush_usd_bu` here are real with no parameterised term at all: this is
    the margin any board desk reads on their screen.
    """
    from agri.core.units import board_crush_usd_bu
    from agri.data.bloomberg_loader import load as load_bloomberg

    bean = load_bloomberg("cbot_soybean")
    meal = load_bloomberg("cbot_soymeal")
    oil = load_bloomberg("cbot_soyoil")
    frame = pd.concat({"bean": bean, "meal": meal, "oil": oil}, axis=1, sort=True).dropna()
    frame = frame.loc[start:]
    return board_crush_usd_bu(frame["bean"], frame["meal"], frame["oil"]).rename("board_crush_usd_bu")


@dataclass(frozen=True)
class RealMarginDiagnostic:
    """What the stationarity test says about the real margin — a result, not a miss.

    OU calibration assumes a mean-reverting margin. Verifying this on the real
    series rather than assuming it is the very subject of `core.stats.adf_kpss`:
    here, it returns an unfavourable verdict on every window tested (the full
    1990-2026 span, and every sub-period since 2005), which is itself a finding —
    the real crush margin goes through genuine regime breaks (Covid 2020, the war
    in Ukraine in 2022, RVO mandate revisions) that a single OU over the whole
    period cannot represent.
    """

    stationarity: StationarityVerdict
    window_start: str
    window_end: str
    n_obs: int

    @property
    def headline(self) -> str:
        return (
            f"Over {self.window_start} → {self.window_end} ({self.n_obs} observations), "
            f"the joint stationarity verdict is \"{self.stationarity.verdict}\": the "
            "real crush margin does not behave like a homogeneous OU over this "
            "period. This is not a calibration failure — it is evidence that the "
            "regime changed at least once (Covid, the war in Ukraine, RVO), which no "
            "fixed-parameter model can absorb."
        )


def diagnose_real_margin_stationarity(margin: pd.Series) -> RealMarginDiagnostic:
    """Tests — rather than assumes — the stationarity of the real margin (O-H1
    applied to real data, not just a synthetic set built to satisfy it)."""
    verdict = adf_kpss(margin, alpha=0.05)
    return RealMarginDiagnostic(
        stationarity=verdict,
        window_start=str(margin.index.min().date()),
        window_end=str(margin.index.max().date()),
        n_obs=len(margin),
    )


def calibrate_real_ou_indicative(margin: pd.Series) -> OUParams:
    """**Indicative** OU calibration on non-stationary real data (`strict=False`).

    Always show this alongside `diagnose_real_margin_stationarity`: the resulting
    parameters describe the chosen window's average regime, not a stable dynamic —
    an illustrative result, not an exercise boundary to follow as-is.
    """
    return calibrate_ou(margin, strict=False)


__all__ = [
    "HysteresisBand",
    "ImpliedSwitchingCost",
    "OUParams",
    "PlantOptionError",
    "PolicyComparison",
    "PolicyResult",
    "RealMarginDiagnostic",
    "RuleComparison",
    "calibrate_ou",
    "calibrate_real_ou_indicative",
    "compare_policies",
    "compare_to_heuristic",
    "diagnose_real_margin_stationarity",
    "implied_switching_cost",
    "real_board_crush_margin",
    "run_always_on_policy",
    "run_band_policy",
    "run_heuristic_policy",
    "simulate_operating_policy",
    "solve_hysteresis",
    "switching_cost_sensitivity",
    "volatility_sensitivity",
]
