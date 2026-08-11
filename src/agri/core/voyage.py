"""Voyage estimation: from a freight index to a cost per tonne on *your* route.

THE PORTFOLIO'S UNIT TRAP, APPLIED TO FREIGHT
-------------------------------------------------
A dry bulk freight index is quoted in **USD/day** (timecharter equivalent) or in USD/t on
a reference route that isn't yours. The economic unit needed in a C&F calculation is
**USD/tonne on your route on your date**. Getting from one to the other isn't a
conversion: it's a **voyage estimate**, and every one of its parameters is a legitimate
point of disagreement between a trading desk and a freight department.

This is the same family as dry tonne/wet tonne or gallon/tonne: the quoted unit is not
the economic unit. Except here the conversion factor isn't a physical constant — it's a
model, with contestable assumptions. Hence the page.

MODEL
-----
    freight_usd_t =
        [ TCE_usd_day × (D_laden + D_ballast + D_port + D_wait)
        + P_vlsfo × sea_consumption × (D_laden + D_ballast)
        + P_mgo   × port_consumption × D_port
        + port_costs + canal_dues ]
        / ( cargo_t × (1 − broker_comm) )

    D_laden   = distance_laden_nm  / (speed_laden_kn   × 24)
    D_ballast = ballast_share × distance_ballast_nm / (speed_ballast_kn × 24)

ASSUMPTIONS
-----------
V-H1  `ballast_share` — the fraction of empty repositioning charged to this voyage.
      **This is T1-1's central disagreement.** The trading desk reasons at 0 ("the ship
      was already there"), the freight department at 1 ("someone pays for the
      ballast"). Neither position is absurd; the number that matters is the threshold
      where they stop giving the same arb signal.
V-H2  Consumption varies with the cube of speed: consumption = ref_consumption ×
      (v/v_ref)³. Standard engineering approximation, valid in a narrow band around the
      reference speed. Beyond ±3 knots it becomes doubtful — hence the slider bounds.
V-H3  Bunkers are valued at a chosen date (fixture, voyage average, spot). **This is
      T1-1's second disagreement**: a 40-day voyage in a volatile bunker market can carry
      several dollars per tonne of difference depending on the convention.
V-H4  Port costs and canal dues are flat per route. In reality they depend on the
      terminal and the draft. Parameterised, never fixed.
V-H5  The supplied TCE is for a vessel class consistent with the cargo. Loading a
      Capesize TCE into a Panamax voyage produces an absurd freight without raising any
      error — hence the plausibility check on the output.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

HOURS_PER_DAY = 24.0


class VoyageError(ValueError):
    """Mis-specified voyage — always a caller error."""


# ---------------------------------------------------------------------------
# Vessel classes (V-H2, V-H5)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VesselClass:
    """Reference consumptions at the reference speed, in tonnes per day."""

    name: str
    cargo_t: float
    consumption_laden_t_day: float
    consumption_ballast_t_day: float
    consumption_port_t_day: float      # MGO in port, not VLSFO
    reference_speed_kn: float = 12.5

    def sea_consumption(self, speed_kn: float, *, laden: bool) -> float:
        """Daily consumption at the given speed, cubic law (V-H2)."""
        if speed_kn <= 0:
            raise VoyageError(f"speed must be > 0, got {speed_kn}")
        base = self.consumption_laden_t_day if laden else self.consumption_ballast_t_day
        return base * (speed_kn / self.reference_speed_kn) ** 3


VESSELS: dict[str, VesselClass] = {
    "supramax": VesselClass("Supramax 55", 55_000, 22.0, 20.0, 2.5),
    "panamax": VesselClass("Panamax 66", 66_000, 26.0, 24.0, 3.0),
    "kamsarmax": VesselClass("Kamsarmax 82", 82_000, 30.0, 27.5, 3.2),
}


# ---------------------------------------------------------------------------
# Routes (V-H4). Distances in nautical miles, orders of magnitude to double-check
# before any use on real data — they're here to make the model runnable.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Route:
    name: str
    origin: str
    destination: str
    distance_laden_nm: float
    distance_ballast_nm: float
    port_costs_usd: float = 150_000.0
    canal_dues_usd: float = 0.0


ROUTES: dict[str, Route] = {
    "santos_qingdao": Route(
        "Santos -> Qingdao", "Santos", "Qingdao", 11_000, 11_000, 180_000, 0.0
    ),
    "usgulf_qingdao": Route(
        "US Gulf -> Qingdao (Panama)", "US Gulf", "Qingdao", 9_800, 9_800, 190_000, 420_000
    ),
    "pnw_qingdao": Route(
        "PNW -> Qingdao", "PNW", "Qingdao", 5_100, 5_100, 160_000, 0.0
    ),
    "novo_alexandria": Route(
        "Novorossiysk -> Alexandria", "Novorossiysk", "Alexandria", 1_000, 1_000, 90_000, 0.0
    ),
    "rouen_alexandria": Route(
        "Rouen -> Alexandria", "Rouen", "Alexandria", 2_700, 2_700, 110_000, 0.0
    ),
}


# ---------------------------------------------------------------------------
# Voyage parameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VoyageParams:
    """The contestable parameters. `ballast_share` is T1-1's subject."""

    ballast_share: float = 1.0
    speed_laden_kn: float = 12.5
    speed_ballast_kn: float = 13.0
    port_days: float = 6.0
    wait_days: float = 0.0
    broker_commission: float = 0.0375

    def __post_init__(self) -> None:
        if not 0.0 <= self.ballast_share <= 1.0:
            raise VoyageError(
                f"ballast_share must be in [0, 1], got {self.ballast_share} — "
                "beyond 1 more ballast would be charged than the route has"
            )
        if not 0.0 <= self.broker_commission < 0.25:
            raise VoyageError(f"commission outside the plausible range: {self.broker_commission}")
        if self.port_days < 0 or self.wait_days < 0:
            raise VoyageError("port days and wait days must be >= 0")

    def with_ballast(self, share: float) -> "VoyageParams":
        """Copy with a different ballast — this is the T1-1 page's sweep."""
        return replace(self, ballast_share=share)


@dataclass(frozen=True)
class VoyageBreakdown:
    """Term-by-term decomposition. Every field is a line in the dashboard's waterfall."""

    laden_days: float
    ballast_days: float
    port_days: float
    wait_days: float
    total_days: float
    hire_usd: float
    bunker_sea_usd: float
    bunker_port_usd: float
    port_costs_usd: float
    canal_dues_usd: float
    total_cost_usd: float
    cargo_t: float
    freight_usd_t: float

    @property
    def waterfall(self) -> dict[str, float]:
        """The line items in USD/t, for the "where the rate comes from" section."""
        payable = self.cargo_t
        return {
            "hire": self.hire_usd / payable,
            "sea bunkers": self.bunker_sea_usd / payable,
            "port bunkers": self.bunker_port_usd / payable,
            "port costs": self.port_costs_usd / payable,
            "canal dues": self.canal_dues_usd / payable,
        }


def voyage_freight_usd_t(
    tce_usd_day: float,
    vlsfo_usd_t: float,
    mgo_usd_t: float,
    *,
    vessel: VesselClass,
    route: Route,
    params: VoyageParams,
) -> VoyageBreakdown:
    """Converts a TCE in USD/day into a freight in USD/tonne on a given route.

    This is the operation the trading desk believes is reading an index and that the
    freight department knows is an estimate. Every term is returned separately so the
    page can show which one carries the disagreement.
    """
    if tce_usd_day < 0:
        raise VoyageError(
            f"negative TCE ({tce_usd_day}) — possible in a depressed market but "
            "suspicious: check the index's unit before continuing"
        )
    if vlsfo_usd_t <= 0 or mgo_usd_t <= 0:
        raise VoyageError("bunker prices must be > 0")

    laden_days = route.distance_laden_nm / (params.speed_laden_kn * HOURS_PER_DAY)
    ballast_days = (
        params.ballast_share
        * route.distance_ballast_nm
        / (params.speed_ballast_kn * HOURS_PER_DAY)
    )
    total_days = laden_days + ballast_days + params.port_days + params.wait_days

    conso_laden = vessel.sea_consumption(params.speed_laden_kn, laden=True)
    conso_ballast = vessel.sea_consumption(params.speed_ballast_kn, laden=False)
    bunker_sea = vlsfo_usd_t * (conso_laden * laden_days + conso_ballast * ballast_days)
    bunker_port = mgo_usd_t * vessel.consumption_port_t_day * params.port_days

    hire = tce_usd_day * total_days
    total_cost = hire + bunker_sea + bunker_port + route.port_costs_usd + route.canal_dues_usd

    payable_cargo = vessel.cargo_t * (1.0 - params.broker_commission)
    freight = total_cost / payable_cargo

    return VoyageBreakdown(
        laden_days=laden_days,
        ballast_days=ballast_days,
        port_days=params.port_days,
        wait_days=params.wait_days,
        total_days=total_days,
        hire_usd=hire,
        bunker_sea_usd=bunker_sea,
        bunker_port_usd=bunker_port,
        port_costs_usd=route.port_costs_usd,
        canal_dues_usd=route.canal_dues_usd,
        total_cost_usd=total_cost,
        cargo_t=payable_cargo,
        freight_usd_t=freight,
    )


def voyage_freight_series(
    tce: pd.Series,
    vlsfo: pd.Series,
    mgo: pd.Series,
    *,
    vessel: VesselClass,
    route: Route,
    params: VoyageParams,
) -> pd.Series:
    """`voyage_freight_usd_t` applied day by day, on the dates common to all three series.

    Strict intersection, no forward-fill: the bunker price on the fixture date is
    precisely what V-H3's disagreement is about, filling it in would make it disappear.
    """
    aligned = pd.concat({"tce": tce, "vlsfo": vlsfo, "mgo": mgo}, axis=1).dropna()
    if aligned.empty:
        raise VoyageError(
            "no common date across the TCE and the two bunker series — check the calendars"
        )
    values = [
        voyage_freight_usd_t(
            row.tce, row.vlsfo, row.mgo, vessel=vessel, route=route, params=params
        ).freight_usd_t
        for row in aligned.itertuples()
    ]
    return pd.Series(values, index=aligned.index, name="freight_usd_t")


def implied_tce_from_freight(
    freight_usd_t: float,
    vlsfo_usd_t: float,
    mgo_usd_t: float,
    *,
    vessel: VesselClass,
    route: Route,
    params: VoyageParams,
) -> float:
    """The inverse: from a published route rate in USD/t to the TCE it implies.

    Used as the T1-1 gate's fallback. When the Baltic routes are accessible in USD/t but
    not the TCE, or the reverse, one is converted to the other — and the gap between the
    TCE implied by a published route rate and the quoted TCE is **itself** a measure of
    the disagreement.
    """
    reference = voyage_freight_usd_t(
        0.0, vlsfo_usd_t, mgo_usd_t, vessel=vessel, route=route, params=params
    )
    costs_only = reference.freight_usd_t          # freight at zero TCE = costs alone, per tonne
    if reference.total_days <= 0:
        raise VoyageError("zero voyage duration — inconsistent route or speeds")
    return (freight_usd_t - costs_only) * reference.cargo_t / reference.total_days


def plausibility_warnings(breakdown: VoyageBreakdown) -> list[str]:
    """Invariant checks (V-H5). An outlandish print is a data diagnosis.

    Nothing here is a market signal: these are the cases where the result can't be read
    at all, and where the page must show a warning instead of a number.
    """
    warnings: list[str] = []
    if breakdown.freight_usd_t <= 0:
        warnings.append(
            f"negative or zero freight ({breakdown.freight_usd_t:.2f} USD/t) — "
            "impossible: unit error on the TCE or the bunkers"
        )
    if breakdown.total_days > 120:
        warnings.append(
            f"{breakdown.total_days:.0f}-day voyage — check the distance and the speed"
        )
    bunker_share = (
        (breakdown.bunker_sea_usd + breakdown.bunker_port_usd) / breakdown.total_cost_usd
        if breakdown.total_cost_usd > 0
        else np.nan
    )
    if np.isfinite(bunker_share) and bunker_share > 0.75:
        warnings.append(
            f"bunkers weigh {bunker_share:.0%} of the total cost — plausible at a bunker "
            "price peak, but check the vessel class before concluding"
        )
    return warnings
