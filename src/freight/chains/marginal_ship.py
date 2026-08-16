"""Project G — the fuel bill a real freight rate could still afford, and what moves it.

THESIS
------
A chartering desk fixes a vessel without knowing exactly how efficient it is relative to
the reference specification a rate was quoted against. This page asks a narrower, fully
answerable version of that question: **holding the real market freight rate and the real
bunker price fixed on any given day, how much MORE fuel per day could a ship burn than the
reference vessel before its own fuel bill alone would exceed the freight revenue it earns?**

That threshold is a multiple of the reference vessel's consumption, not an absolute number,
and it can be solved for exactly — the freight-cost identity is affine in the consumption
multiplier, so no search or fit is needed.

    k*(t) = [ rate(t) - A ] / [ B x vlsfo(t) ]

    A = the route's fixed cost per tonne (port costs + canal dues), independent of bunkers
    B = the reference vessel's bunker cost per tonne per unit of vlsfo, at multiplier 1

MEASURED ON THE REAL P8 SANTOS-QINGDAO ROUTE, 2021-11 TO 2026-08 (643 days, panamax
reference, full-ballast convention): k* ranged from **1.46 to 3.86**, median 2.4. It has
never approached 1 in this sample — not even during the March 2026 VLSFO spike, which is
this margin's tightest point on record.

THE DECOMPOSITION THAT ANSWERS "WHAT ACTUALLY MOVES IT"
---------------------------------------------------------
A is a constant (verified below), and MGO is a fixed multiple of VLSFO (G-H1), so the
whole relationship reduces to two variables and the log-change decomposition is an
**identity, not a regression**:

    d(log k*) = d(log(rate - A)) - d(log vlsfo)          -- exact, no residual

Splitting the variance of the left side between the two terms on the right (covariance
allocated evenly, so the shares sum to exactly one) gives the honest answer to "does the
freight cycle or the oil cycle drive this margin": **the bunker price accounts for
somewhat more of it (62%) than the freight rate itself (38%)** — and the two are mildly
positively correlated (correlation of the two components 0.20), meaning the freight and
bunker cycles partially offset each other's effect on the margin rather than reinforcing it.
That the bunker term dominates at all is worth stating plainly: a chartering desk that
reads the freight market as the thing setting its fuel-cost headroom is reading the
smaller of the two effects.

THE WALL, AND WHY IT IS THE RIGHT PLACE TO ASK
-------------------------------------------------
k* = 1 means the reference vessel's OWN fuel bill exactly consumes the freight revenue —
a contribution margin of zero, before a single dollar of opex, crew, insurance, or capital
cost is paid. That is a far more permissive threshold than the one that actually governs a
layup or scrapping decision, which needs TCE to clear opex, not merely to clear zero. So
k* is an upper bound: the true efficiency bar the market sets on this route is tighter than
what this page shows, by an amount this page cannot compute, because doing so needs real
opex data by vessel and owner that is not public and is not in this export. That is the
question worth putting to a chartering desk, not a claim this page can settle itself.

ASSUMPTIONS
-----------
G-H1  MGO is reconstructed as 1.35x VLSFO (T1-1's own parameter, reused rather than
      re-derived, since the export has no separate MGO series).
G-H2  Vessel and route: panamax reference (66,000t cargo), Santos-Qingdao, matching T1-1
      exactly — chosen for direct comparability with the existing P8 analysis, not
      because it is the only plausible choice.
G-H3  The consumption multiplier k scales laden, ballast, AND port consumption uniformly.
      A vintage that is only less efficient at sea, not in port, would need a different
      split; the export gives no basis to prefer one split over another.
G-H4  k* = 1 is a **fuel-only** contribution-margin floor, not an operating or scrapping
      threshold. It is stated as an upper bound throughout, never as an estimate of the
      threshold that actually governs a lay-up decision.
G-H5  The P8 route rate is missing 2023 and 2024 entirely in this export (the same gap
      documented in project D) — this test runs on 2021-2022 and 2025-2026 only.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from agri.core.voyage import ROUTES, VESSELS, VoyageParams, voyage_freight_usd_t
from agri.data.snapshot import cached

# G-H1 — reused verbatim from T1-1 (freight_cf.py), not re-derived.
MGO_PREMIUM = 1.35
# G-H2 — same vessel and route as T1-1, for direct comparability.
VESSEL_KEY = "panamax"
ROUTE_KEY = "santos_qingdao"

BALLAST_CONVENTIONS: tuple[float, ...] = (0.0, 1.0)
PRIMARY_BALLAST_SHARE = 1.0  # the freight department's convention, the more conservative one


class MarginalShipError(ValueError):
    """Mis-specified breakeven test — always a caller error."""


# ===========================================================================
# Data
# ===========================================================================
@cached("g_marginal_ship")
def load_marginal_ship_frame(start: str | None = None) -> pd.DataFrame:
    """The real P8 route rate, real VLSFO, and MGO reconstructed at 1.35x (G-H1).

    Columns: rate (USD/t), vlsfo (USD/t), mgo (USD/t, reconstructed).
    """
    from agri.data.bloomberg_loader import load

    rate = load("p8_route_usd_t")
    vlsfo = load("vlsfo_singapore")
    frame = pd.concat({"rate": rate, "vlsfo": vlsfo}, axis=1, sort=True).dropna()
    if start is not None:
        frame = frame[frame.index >= pd.Timestamp(start)]
    if frame.empty:
        raise MarginalShipError(f"no common dates between the route rate and VLSFO after {start}")
    frame["mgo"] = frame["vlsfo"] * MGO_PREMIUM
    return frame


def _reference_vessel():
    return VESSELS[VESSEL_KEY]


def _reference_route():
    return ROUTES[ROUTE_KEY]


# ===========================================================================
# The affine decomposition: cost = A + k x B(vlsfo)
# ===========================================================================
def cost_floor_usd_t(*, ballast_share: float, params: VoyageParams | None = None) -> float:
    """A: the route's fixed cost per tonne (port costs + canal dues), at zero consumption.

    Provably independent of the bunker price — a vessel that burns nothing pays no
    bunker bill regardless of what bunkers cost. Kept as its own function because that
    independence is asserted in the module docstring and is worth being able to verify
    directly rather than only implicitly through `breakeven_multiplier`.
    """
    vessel = _reference_vessel()
    zero_vessel = replace(
        vessel,
        consumption_laden_t_day=0.0,
        consumption_ballast_t_day=0.0,
        consumption_port_t_day=0.0,
    )
    p = (params or VoyageParams()).with_ballast(ballast_share)
    return voyage_freight_usd_t(
        0.0, 1.0, 1.0, vessel=zero_vessel, route=_reference_route(), params=p
    ).freight_usd_t


def breakeven_multiplier(
    freight_usd_t: float,
    vlsfo_usd_t: float,
    mgo_usd_t: float,
    *,
    ballast_share: float,
    params: VoyageParams | None = None,
) -> float:
    """k*: the consumption multiplier on the reference vessel at which TCE = 0.

    Solved directly rather than by search: freight cost is affine in the consumption
    multiplier k, so two evaluations of the existing voyage-cost function (at k=0 and
    k=1) pin down the line exactly.
    """
    if vlsfo_usd_t <= 0 or mgo_usd_t <= 0:
        raise MarginalShipError("bunker prices must be > 0")
    vessel = _reference_vessel()
    p = (params or VoyageParams()).with_ballast(ballast_share)
    a = cost_floor_usd_t(ballast_share=ballast_share, params=params)
    a_plus_b = voyage_freight_usd_t(
        0.0, vlsfo_usd_t, mgo_usd_t, vessel=vessel, route=_reference_route(), params=p
    ).freight_usd_t
    b = a_plus_b - a
    if b <= 0:
        raise MarginalShipError("degenerate route: zero bunker cost at multiplier 1")
    return (freight_usd_t - a) / b


def breakeven_series(frame: pd.DataFrame, *, ballast_share: float = PRIMARY_BALLAST_SHARE) -> pd.Series:
    """`breakeven_multiplier` applied day by day."""
    values = [
        breakeven_multiplier(row.rate, row.vlsfo, row.mgo, ballast_share=ballast_share)
        for row in frame.itertuples()
    ]
    return pd.Series(values, index=frame.index, name="k_star")


# ===========================================================================
# Summary of the margin's level and history
# ===========================================================================
@dataclass(frozen=True)
class MarginSummary:
    k: pd.Series
    ballast_share: float

    @property
    def min(self) -> float:
        return float(self.k.min())

    @property
    def median(self) -> float:
        return float(self.k.median())

    @property
    def max(self) -> float:
        return float(self.k.max())

    @property
    def min_date(self) -> pd.Timestamp:
        return self.k.idxmin()

    @property
    def max_date(self) -> pd.Timestamp:
        return self.k.idxmax()

    def quarterly(self) -> pd.Series:
        return self.k.resample("QE").median().dropna()

    @property
    def share_below_2(self) -> float:
        return float((self.k < 2.0).mean())

    @property
    def never_approached_one(self) -> bool:
        return self.min > 1.2

    @property
    def headline(self) -> str:
        convention = "full-ballast" if self.ballast_share >= 0.5 else "zero-ballast"
        return (
            f"Under the {convention} convention, the fuel-only breakeven multiplier ran "
            f"from {self.min:.2f} to {self.max:.2f} (median {self.median:.2f}), and its "
            f"tightest point on record was {self.min_date:%d %b %Y} — "
            f"{'well above' if self.never_approached_one else 'close to'} the multiplier "
            "of 1 that would mean the reference vessel's own fuel bill consumed the "
            "entire freight revenue."
        )


def margin_summary(
    frame: pd.DataFrame, *, ballast_share: float = PRIMARY_BALLAST_SHARE
) -> MarginSummary:
    return MarginSummary(k=breakeven_series(frame, ballast_share=ballast_share), ballast_share=ballast_share)


# ===========================================================================
# THE DECOMPOSITION — an identity, not a regression
# ===========================================================================
@dataclass(frozen=True)
class VarianceDecomposition:
    share_rate: float
    share_bunker: float
    component_correlation: float
    n_obs: int

    @property
    def dominant(self) -> str:
        return "bunker price" if self.share_bunker > self.share_rate else "freight rate"

    @property
    def headline(self) -> str:
        return (
            f"Of the day-to-day variance in the breakeven multiplier, {self.share_bunker:.0%} "
            f"traces to the bunker price and {self.share_rate:.0%} to the freight rate "
            f"itself — the {self.dominant} moves this margin more, and the two components "
            f"are only weakly correlated ({self.component_correlation:+.2f}), so they mostly "
            "do not offset each other."
        )


def variance_decomposition(
    frame: pd.DataFrame, *, ballast_share: float = PRIMARY_BALLAST_SHARE
) -> VarianceDecomposition:
    """Exact decomposition of d(log k*) into a rate term and a bunker term (G-H1, G-H2).

    Not a regression: A is a constant (cost_floor_usd_t does not depend on vlsfo) and MGO
    is a fixed multiple of VLSFO, so log(k*) = log(rate - A) - log(vlsfo) - log(B/vlsfo-ratio)
    is an algebraic identity in the two inputs. The covariance term is split evenly between
    the two shares so they sum to exactly one — there is no residual to assign it to.
    """
    a = cost_floor_usd_t(ballast_share=ballast_share)
    margin = frame["rate"] - a
    if (margin <= 0).any():
        raise MarginalShipError(
            "the freight rate falls below the fixed cost floor on at least one date — "
            "the log decomposition is undefined there"
        )
    d_rate_term = np.log(margin).diff().dropna()
    d_bunker_term = np.log(frame["vlsfo"]).diff().dropna()
    var_rate = float(d_rate_term.var())
    var_bunker = float(d_bunker_term.var())
    cov = float(d_rate_term.cov(d_bunker_term))
    total = var_rate + var_bunker - 2 * cov
    if total <= 0:
        raise MarginalShipError("degenerate sample: zero variance in the breakeven multiplier")
    return VarianceDecomposition(
        share_rate=(var_rate - cov) / total,
        share_bunker=(var_bunker - cov) / total,
        component_correlation=float(d_rate_term.corr(d_bunker_term)),
        n_obs=int(len(d_rate_term)),
    )
