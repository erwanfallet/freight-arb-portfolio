"""The tipping point — every page in the portfolio's deliverable.

A numeric threshold a practitioner can contest in ten seconds beats a sound analysis.
"The breakeven is at 52% of the tariff" invites a response; "here is my analysis" invites
none. This module turns a margin function into that threshold.

Three outputs, and the third is the one that protects:
    theta_star        the level where the sign flips
    sensitivity       d(margin)/d(theta) at the tipping point, in readable units
    distance_sigmas   the gap between the current level and theta_star, in historical
                      standard deviations of theta

Without the third, a three-sigma-away breakeven gets announced as if it were imminent —
that's the fastest way to lose a reader who knows their market.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import brentq


class BreakevenError(ValueError):
    """Malformed problem — inconsistent bounds, non-numeric function."""


class NoBreakevenInRange(Exception):
    """No sign change over the explored interval.

    **This is not a computation failure, it's a result**, and often the page's strongest
    one: "over the entire plausible range of charged ballast, the arb stays open" is a
    falsifiable claim. The values at both bounds are attached to the exception so the
    page can display it instead of erroring out.
    """

    def __init__(self, lo: float, hi: float, margin_lo: float, margin_hi: float):
        self.lo = lo
        self.hi = hi
        self.margin_lo = margin_lo
        self.margin_hi = margin_hi
        sign = "positive" if margin_lo > 0 else "negative"
        super().__init__(
            f"no sign change over [{lo:g}, {hi:g}]: the margin stays {sign} "
            f"({margin_lo:+.4f} at {lo:g}, {margin_hi:+.4f} at {hi:g}). "
            "This is not an error — it's the result to display, without extrapolating "
            "beyond the plausible range."
        )


@dataclass(frozen=True)
class Breakeven:
    """Tipping point, its local sensitivity, and its distance to the current level."""

    theta_star: float
    sensitivity: float
    bracket: tuple[float, float]
    theta_current: float | None = None
    theta_sigma: float | None = None
    theta_label: str = "theta"
    margin_label: str = "margin"

    @property
    def distance_sigmas(self) -> float | None:
        """Distance from the current level to the tipping point, in standard deviations of theta.

        None when the history wasn't supplied: better to show nothing than to show a
        distance with no scale.
        """
        if self.theta_current is None or self.theta_sigma is None:
            return None
        if self.theta_sigma == 0:
            return float("inf")
        return (self.theta_star - self.theta_current) / self.theta_sigma

    @property
    def is_within_reach(self) -> bool:
        """Is the tipping point within one standard deviation? If not, say it's far off."""
        d = self.distance_sigmas
        return d is not None and abs(d) <= 1.0

    @property
    def summary(self) -> str:
        head = (
            f"{self.theta_label}* = {self.theta_star:.4g} "
            f"(d{self.margin_label}/d{self.theta_label} = {self.sensitivity:+.4g} at threshold)"
        )
        d = self.distance_sigmas
        if d is None:
            return head
        reach = "within reach" if self.is_within_reach else "out of reach historically"
        return (
            f"{head} | current level {self.theta_current:.4g}, "
            f"i.e. {d:+.2f} standard deviations — {reach}"
        )


def solve_breakeven(
    margin_fn: Callable[[float], float],
    lo: float,
    hi: float,
    *,
    theta_current: float | None = None,
    theta_history: pd.Series | np.ndarray | None = None,
    h: float | None = None,
    theta_label: str = "theta",
    margin_label: str = "margin",
    xtol: float = 1e-10,
) -> Breakeven:
    """Solves `margin_fn(theta) = 0` over [lo, hi] and measures the sensitivity at the threshold.

    `margin_fn` must be monotone in theta over the interval — true of every margin in
    this portfolio (higher ballast can only raise freight cost, a higher LCFS credit can
    only improve the value out of the plant gate). Monotonicity isn't checked
    numerically: it must be argued for in the page.

    `theta_history` is used to express the distance in standard deviations. Supplying it
    is strongly recommended: it's what distinguishes an imminent threshold from a
    theoretical one.
    """
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise BreakevenError(f"non-finite bounds: [{lo}, {hi}]")
    if lo >= hi:
        raise BreakevenError(f"inconsistent bounds: lo={lo} must be < hi={hi}")

    margin_lo = float(margin_fn(lo))
    margin_hi = float(margin_fn(hi))
    if not np.isfinite(margin_lo) or not np.isfinite(margin_hi):
        raise BreakevenError(
            f"non-finite margin at the bounds: f({lo})={margin_lo}, f({hi})={margin_hi}"
        )
    if margin_lo == 0.0:
        theta_star = lo
    elif margin_hi == 0.0:
        theta_star = hi
    elif np.sign(margin_lo) == np.sign(margin_hi):
        raise NoBreakevenInRange(lo, hi, margin_lo, margin_hi)
    else:
        theta_star = float(brentq(lambda t: float(margin_fn(t)), lo, hi, xtol=xtol))

    step = h if h is not None else (hi - lo) * 1e-5
    if step <= 0:
        raise BreakevenError(f"the derivation step must be > 0, got {step}")
    # keep both evaluation points inside the bracket: outside the bounds, the margin
    # function may be undefined (e.g. a negative ballast doesn't exist)
    left = max(theta_star - step, lo)
    right = min(theta_star + step, hi)
    sensitivity = (float(margin_fn(right)) - float(margin_fn(left))) / (right - left)

    sigma = None
    if theta_history is not None:
        history = pd.Series(theta_history).dropna().astype(float)
        if len(history) >= 2:
            sigma = float(history.std())

    return Breakeven(
        theta_star=theta_star,
        sensitivity=sensitivity,
        bracket=(lo, hi),
        theta_current=theta_current,
        theta_sigma=sigma,
        theta_label=theta_label,
        margin_label=margin_label,
    )
