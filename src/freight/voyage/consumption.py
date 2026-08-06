"""Bunker consumption — the cubic propeller law (H3, Partie 2.10 / 3.5).

consumption(v) = reference_consumption * (v / reference_speed) ** exponent

This is also the mechanism behind the skewness-of-freight-rates fallback result
(Partie 3.5): consumption rises steeply with speed, so slowing down buys large bunker
savings cheaply while speeding up runs into a wall — supply of transport capacity
(vessels x speed) is elastic downward and inelastic upward.
"""
from __future__ import annotations

from freight.voyage.config import VoyageParams


def consumption_t_per_day(speed_kn: float, params: VoyageParams) -> float:
    if speed_kn <= 0:
        raise ValueError("speed_kn must be positive")
    ratio = speed_kn / params.reference_speed_kn
    return params.reference_consumption_t_per_day * ratio**params.consumption_exponent


def sea_days(distance_nm: float, speed_kn: float) -> float:
    if speed_kn <= 0:
        raise ValueError("speed_kn must be positive")
    return distance_nm / (speed_kn * 24.0)


def leg_bunker_consumption_t(distance_nm: float, speed_kn: float, params: VoyageParams) -> float:
    """Total bunker consumption (t) for one leg at a given speed."""
    return sea_days(distance_nm, speed_kn) * consumption_t_per_day(speed_kn, params)
