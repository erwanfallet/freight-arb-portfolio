"""Voyage -> Time Charter Equivalent (Partie 2.2).

The four points that matter, carried into this implementation directly:
1. The ballast leg is included — omitting it overstates TCE by ~40-50% on a long route.
2. TCE nets voyage costs only, not OPEX/capital costs — it is a contribution margin,
   never reported or compared to anything as if it were profit.
3. Bunkers and freight are coupled: bunker price is a required input, never assumed away.
4. Speed is a decision variable, not a given — see voyage/consumption.py.
"""
from __future__ import annotations

from dataclasses import dataclass

from freight.voyage.config import VoyageParams
from freight.voyage.consumption import leg_bunker_consumption_t, sea_days


@dataclass(frozen=True)
class VoyageInputs:
    cargo_t: float
    freight_rate_usd_per_t: float
    distance_laden_nm: float
    distance_ballast_nm: float
    bunker_price_usd_per_t: float
    port_days: float
    params: VoyageParams


@dataclass(frozen=True)
class VoyageResult:
    revenue_gross_usd: float
    commission_usd: float
    revenue_net_usd: float
    laden_days: float
    ballast_days: float
    port_days: float
    total_days: float
    bunker_consumption_t: float
    bunker_cost_usd: float
    port_costs_usd: float
    total_voyage_costs_usd: float
    tce_usd_per_day: float


def compute_tce(inputs: VoyageInputs) -> VoyageResult:
    p = inputs.params

    revenue_gross = inputs.cargo_t * inputs.freight_rate_usd_per_t
    commission = revenue_gross * p.brokerage_commission
    revenue_net = revenue_gross - commission

    laden_days = sea_days(inputs.distance_laden_nm, p.laden_speed_kn)
    ballast_days = sea_days(inputs.distance_ballast_nm, p.ballast_speed_kn)
    total_days = laden_days + ballast_days + inputs.port_days

    laden_consumption = leg_bunker_consumption_t(inputs.distance_laden_nm, p.laden_speed_kn, p)
    ballast_consumption = leg_bunker_consumption_t(inputs.distance_ballast_nm, p.ballast_speed_kn, p)
    port_consumption = inputs.port_days * p.port_consumption_t_per_day
    bunker_consumption = laden_consumption + ballast_consumption + port_consumption
    bunker_cost = bunker_consumption * inputs.bunker_price_usd_per_t

    total_voyage_costs = bunker_cost + p.port_costs_usd
    tce = (revenue_net - total_voyage_costs) / total_days

    return VoyageResult(
        revenue_gross_usd=revenue_gross,
        commission_usd=commission,
        revenue_net_usd=revenue_net,
        laden_days=laden_days,
        ballast_days=ballast_days,
        port_days=inputs.port_days,
        total_days=total_days,
        bunker_consumption_t=bunker_consumption,
        bunker_cost_usd=bunker_cost,
        port_costs_usd=p.port_costs_usd,
        total_voyage_costs_usd=total_voyage_costs,
        tce_usd_per_day=tce,
    )
