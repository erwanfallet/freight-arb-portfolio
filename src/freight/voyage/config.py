"""Modeling hypotheses H1-H7 (Partie 7). All parameterized, none hardcoded downstream —
every function in voyage/ takes these as explicit arguments so a sensitivity sweep
(backtest/sensitivity.py) can perturb any of them without touching the model code.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VoyageParams:
    # H1 — cargo Capesize: DWT 180k - ~8k soutes/consumables
    cargo_t: float = 170_000.0
    # H2 — laden/ballast speed (kn): ballast faster, reduced draft
    laden_speed_kn: float = 12.0
    ballast_speed_kn: float = 13.0
    # H3 — consumption ∝ v^3 (propeller law), calibrated on published Capesize specs
    consumption_exponent: float = 3.0
    reference_speed_kn: float = 12.5
    reference_consumption_t_per_day: float = 40.0
    # H4 — port days, from published loading/discharge rates
    port_days_brazil: float = 6.0
    port_days_australia: float = 4.0
    # H6 — brokerage commission, standard voyage-charter market rate
    brokerage_commission: float = 0.0375
    # port + maneuvering costs at anchor, flat placeholder until real fixture data arrives
    port_costs_usd: float = 180_000.0
    # consumption while in port (loading/discharge/maneuvering) — set 0 to match the
    # doc's illustrative example, which only bunkers the sea legs
    port_consumption_t_per_day: float = 0.0

    def with_overrides(self, **kwargs) -> "VoyageParams":
        """Return a copy with the given fields overridden — used by the sensitivity sweep."""
        from dataclasses import replace

        return replace(self, **kwargs)


DEFAULT_PARAMS = VoyageParams()
