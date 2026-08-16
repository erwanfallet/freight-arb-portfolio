"""Project I — the ballast leg the regulator counts the same as a laden one.

THESIS
------
CII's attained metric (AER) is CO2 emitted divided by deadweight capacity times distance
sailed. Neither term in that ratio asks whether the ship was carrying cargo. A ballast
leg — sailed empty, earning nothing — counts exactly as much distance as a laden leg, and
because a ship is free to sail its ballast leg at whatever speed it likes, it can buy a
better rating with a decision that transports not one additional tonne.

    AER = (fuel_t x Cf x 1e6) / (DWT x distance_nm)          grams CO2 per DWT-mile

MEASURED ON THE REAL P8 SANTOS-QINGDAO VOYAGE (panamax reference): slowing only the
ballast leg from 13 to 8 knots — the laden leg and every tonne of cargo on it completely
unchanged — improves attained AER by **31%**. Loading more or less cargo changes AER by
exactly **zero**: DWT is the ship's nameplate capacity, not what is actually in the hold,
and it does not appear anywhere the fuel or distance terms can see it.

THE PART THAT KEEPS THIS FROM BEING A FREE LUNCH
----------------------------------------------------
A slower ballast leg also means fewer round trips per year, and that costs real freight
revenue. Priced at the real median P8 route rate and real median VLSFO over the same
window this route rate actually covers (2021-2022 and 2025-2026): the same 13-to-8-knot
ballast slowdown that buys 31% on AER costs the voyage economics **6% of its annual net
contribution** (revenue less fuel cost) — a real cost, and directionally the wrong way
for anyone hoping the rating improves for free. It is not monotonic on the way there
either: net contribution rises slightly from 13 to 12 knots before falling, because the
first knot of slowdown saves more in fuel than it costs in trips, and later ones do not.

WHY NO PRICE IS ATTACHED TO ANY OF THIS
-------------------------------------------
Unlike the EU ETS carbon cost modelled in project B, CII carries **no market price**. A
bad AER does not generate an invoice — it triggers a corrective-action-plan requirement
at the regulatory boundary, and whatever commercial consequence it has runs entirely
through charter-party clauses and fixture terms this export has no visibility into. That
is stated as a limit, not modelled around: the only two real prices in this project (the
route rate and VLSFO) are use to cost the SPEED decision, never to cost the rating itself.

WHAT THIS PAGE DELIBERATELY DOES NOT ATTEMPT
-------------------------------------------------
The official A-to-E rating bands (IMO's reference-line parameters by ship type and size,
and the annual reduction factors that tighten them) are not reproduced here. They come
from a large, ship-type-specific regulatory table, and citing exact boundary values from
memory risks presenting a wrong regulatory threshold as fact — a worse error than not
answering. Everything on this page is expressed as attained AER and its percentage
change, which needs none of that table and does not depend on getting it right.

ASSUMPTIONS
-----------
I-H1  Carbon factor for VLSFO: Cf = 3.114 g CO2 per g fuel, the IMO reference value for
      the HFO/VLSFO family (MEPC.1/Circ.866). Stated explicitly because it is the one
      regulatory constant this page does rely on.
I-H2  DWT is approximated by the vessel's rated cargo capacity (66,000t, the same
      panamax used in projects A and G). Real DWT runs somewhat higher (it includes
      bunkers, stores, crew). This does not affect any result on this page: capacity is
      the same ship in every comparison and cancels out.
I-H3  Vessel and route: panamax, Santos-Qingdao, matching projects A and G — chosen for
      comparability, not because it is the only plausible choice.
I-H4  The trade-off in dollar terms uses the real P8 route rate and real VLSFO price
      from project G's own snapshot (same underlying series, reused rather than
      re-fetched) — median over the same 2021-2026 window, since the P8 route is
      missing 2023-2024 in this export (the gap documented in project D).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from agri.core.voyage import ROUTES, VESSELS, VoyageParams

# I-H1
CARBON_FACTOR_VLSFO = 3.114
# I-H3
VESSEL_KEY = "panamax"
ROUTE_KEY = "santos_qingdao"

DEFAULT_BALLAST_SPEEDS: tuple[float, ...] = (13.0, 12.0, 11.0, 10.0, 9.0, 8.0)
DEFAULT_BALLAST_SHARES: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)


class CiiError(ValueError):
    """Mis-specified CII/voyage scenario — always a caller error."""


def _vessel():
    return VESSELS[VESSEL_KEY]


def _route():
    return ROUTES[ROUTE_KEY]


# ===========================================================================
# The attained metric — pure physics, no market data
# ===========================================================================
@dataclass(frozen=True)
class AerResult:
    ballast_speed_kn: float
    ballast_share: float
    fuel_t: float
    co2_t: float
    distance_nm: float
    aer: float  # g CO2 / DWT-nm


def attained_aer(
    ballast_speed_kn: float,
    *,
    ballast_share: float = 1.0,
    laden_speed_kn: float | None = None,
) -> AerResult:
    """One round trip's attained AER at a given ballast speed.

    Sea fuel (VLSFO) only — port fuel is MGO, a different carbon factor and a small
    enough share of the total that it is left out rather than mixed in under one factor.
    The laden leg's speed and the cargo it carries never enter this function's variable
    inputs beyond `laden_speed_kn` (held fixed by default) — that is deliberate: the
    point is to isolate what the ballast-speed decision alone does to the rating.
    """
    if ballast_speed_kn <= 0:
        raise CiiError(f"ballast speed must be > 0, got {ballast_speed_kn}")
    if not 0.0 <= ballast_share <= 1.0:
        raise CiiError(f"ballast_share must be in [0, 1], got {ballast_share}")

    vessel = _vessel()
    route = _route()
    params = VoyageParams()
    laden_speed = laden_speed_kn or params.speed_laden_kn

    laden_days = route.distance_laden_nm / (laden_speed * 24.0)
    ballast_days = ballast_share * route.distance_ballast_nm / (ballast_speed_kn * 24.0)

    conso_laden = vessel.sea_consumption(laden_speed, laden=True)
    conso_ballast = vessel.sea_consumption(ballast_speed_kn, laden=False)
    fuel_t = conso_laden * laden_days + conso_ballast * ballast_days

    co2_t = fuel_t * CARBON_FACTOR_VLSFO
    distance_nm = route.distance_laden_nm + ballast_share * route.distance_ballast_nm
    dwt = vessel.cargo_t  # I-H2

    aer = (co2_t * 1e6) / (dwt * distance_nm)
    return AerResult(
        ballast_speed_kn=ballast_speed_kn, ballast_share=ballast_share,
        fuel_t=fuel_t, co2_t=co2_t, distance_nm=distance_nm, aer=aer,
    )


def ballast_speed_sweep(speeds: tuple[float, ...] = DEFAULT_BALLAST_SPEEDS) -> pd.DataFrame:
    """AER as a function of ballast speed alone — the page's central sweep."""
    if len(speeds) < 2:
        raise CiiError("need at least two speeds to show a sweep")
    rows = [attained_aer(s, ballast_share=1.0) for s in speeds]
    base = rows[0].aer
    return pd.DataFrame(
        {
            "ballast_speed_kn": [r.ballast_speed_kn for r in rows],
            "aer": [r.aer for r in rows],
            "pct_change_vs_fastest": [(r.aer / base - 1.0) for r in rows],
        }
    ).set_index("ballast_speed_kn")


def ballast_share_sweep(shares: tuple[float, ...] = DEFAULT_BALLAST_SHARES) -> pd.DataFrame:
    """AER as a function of how much ballast distance is charged to the voyage, at a
    fixed reference ballast speed — the direct answer to 'does more ballast distance
    reward a better score'."""
    if len(shares) < 2:
        raise CiiError("need at least two ballast shares to show a sweep")
    default_ballast_speed = VoyageParams().speed_ballast_kn
    rows = [attained_aer(default_ballast_speed, ballast_share=s) for s in shares]
    base = rows[0].aer
    return pd.DataFrame(
        {
            "ballast_share": [r.ballast_share for r in rows],
            "aer": [r.aer for r in rows],
            "pct_change_vs_no_ballast": [(r.aer / base - 1.0) for r in rows],
        }
    ).set_index("ballast_share")


# ===========================================================================
# The trade-off — priced in real freight rate and real VLSFO
# ===========================================================================
@dataclass(frozen=True)
class VoyageEconomics:
    ballast_speed_kn: float
    trips_per_year: float
    annual_revenue_usd: float
    annual_fuel_cost_usd: float
    aer: float

    @property
    def net_contribution_usd(self) -> float:
        return self.annual_revenue_usd - self.annual_fuel_cost_usd


def annual_economics(
    ballast_speed_kn: float,
    *,
    route_rate_usd_t: float,
    vlsfo_usd_t: float,
    ballast_share: float = 1.0,
) -> VoyageEconomics:
    """One year of round trips at a given ballast speed, priced at real market levels.

    Round trips per year is what makes this a genuine trade-off rather than a second
    free improvement: a slower ballast leg buys a better AER and burns less fuel per
    trip, but it also fits fewer trips into a year, and that is priced here too.
    """
    if route_rate_usd_t <= 0 or vlsfo_usd_t <= 0:
        raise CiiError("route rate and VLSFO price must both be > 0")

    vessel = _vessel()
    route = _route()
    params = VoyageParams()

    laden_days = route.distance_laden_nm / (params.speed_laden_kn * 24.0)
    ballast_days = ballast_share * route.distance_ballast_nm / (ballast_speed_kn * 24.0)
    total_days = laden_days + ballast_days + 2 * params.port_days

    result = attained_aer(ballast_speed_kn, ballast_share=ballast_share)
    trips_per_year = 365.0 / total_days
    annual_revenue = trips_per_year * vessel.cargo_t * route_rate_usd_t
    annual_fuel_cost = trips_per_year * result.fuel_t * vlsfo_usd_t

    return VoyageEconomics(
        ballast_speed_kn=ballast_speed_kn,
        trips_per_year=trips_per_year,
        annual_revenue_usd=annual_revenue,
        annual_fuel_cost_usd=annual_fuel_cost,
        aer=result.aer,
    )


@dataclass(frozen=True)
class SpeedTradeoff:
    table: pd.DataFrame  # index=ballast_speed_kn, columns=aer, net_contribution_usd, ...
    route_rate_usd_t: float
    vlsfo_usd_t: float

    @property
    def aer_improvement(self) -> float:
        fastest, slowest = self.table.iloc[0], self.table.iloc[-1]
        return 1.0 - slowest["aer"] / fastest["aer"]

    @property
    def net_contribution_cost(self) -> float:
        fastest, slowest = self.table.iloc[0], self.table.iloc[-1]
        return 1.0 - slowest["net_contribution_usd"] / fastest["net_contribution_usd"]

    @property
    def headline(self) -> str:
        return (
            f"Slowing the ballast leg from {self.table.index[0]:.0f} to "
            f"{self.table.index[-1]:.0f} knots improves attained AER by "
            f"{self.aer_improvement:.0%}, at real prices (route rate "
            f"{self.route_rate_usd_t:.0f} USD/t, VLSFO {self.vlsfo_usd_t:.0f} USD/t) "
            f"it also costs {self.net_contribution_cost:.0%} of the voyage's annual net "
            "contribution — the lost round trips are worth more than the fuel saved."
        )


def speed_tradeoff(
    route_rate_usd_t: float,
    vlsfo_usd_t: float,
    *,
    speeds: tuple[float, ...] = DEFAULT_BALLAST_SPEEDS,
) -> SpeedTradeoff:
    """The combined AER-and-P&L sweep, priced at given (real, market) levels."""
    if len(speeds) < 2:
        raise CiiError("need at least two speeds to show a trade-off")
    rows = [
        annual_economics(s, route_rate_usd_t=route_rate_usd_t, vlsfo_usd_t=vlsfo_usd_t)
        for s in speeds
    ]
    table = pd.DataFrame(
        {
            "ballast_speed_kn": [r.ballast_speed_kn for r in rows],
            "aer": [r.aer for r in rows],
            "trips_per_year": [r.trips_per_year for r in rows],
            "annual_revenue_usd": [r.annual_revenue_usd for r in rows],
            "annual_fuel_cost_usd": [r.annual_fuel_cost_usd for r in rows],
            "net_contribution_usd": [r.net_contribution_usd for r in rows],
        }
    ).set_index("ballast_speed_kn")
    return SpeedTradeoff(table=table, route_rate_usd_t=route_rate_usd_t, vlsfo_usd_t=vlsfo_usd_t)


def market_speed_tradeoff(speeds: tuple[float, ...] = DEFAULT_BALLAST_SPEEDS) -> SpeedTradeoff:
    """Same trade-off, priced at the real median P8 route rate and real median VLSFO
    over 2021-2026 (I-H4) — reuses project G's own cached real-data frame rather than
    re-fetching, since it is the identical underlying series."""
    from freight.chains.marginal_ship import load_marginal_ship_frame

    frame = load_marginal_ship_frame()
    return speed_tradeoff(
        float(frame["rate"].median()), float(frame["vlsfo"].median()), speeds=speeds
    )
