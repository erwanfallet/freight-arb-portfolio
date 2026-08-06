"""The armateur-indifference condition and the C3/C5 dislocation signal (Partie 2.3, 3.4).

An owner is indifferent between two employments of the same ship class when their TCEs
are equal. Setting TCE(C3) = TCE(C5) and solving for C3 gives the fair-value C3 implied
by the currently observed C5 (or vice versa). The gap between observed and fair-value
freight is the dislocation the "Ballast Constraint" iteration wanted to trade.

Derivation (kept explicit — the brainstorm doc's own formula transcription has a sign
ambiguity in B3/B5 that this derivation resolves from first principles):

    TCE_3 = TCE_5
    (cargo * C3 * (1-c) - Costs3) / D3 = (cargo * C5 * (1-c) - Costs5) / D5
    => C3* = (D3/D5) * C5 + [Costs3 - (D3/D5) * Costs5] / (cargo * (1-c))

where D3, D5 are total cycle days (sea + port) for each employment and Costs3, Costs5
are the corresponding total voyage costs (bunkers + port costs), matching the B3/B5
vocabulary used in Partie 3.4.
"""
from __future__ import annotations

from dataclasses import dataclass

from freight.voyage.tce import VoyageResult


def fair_value_c3(
    c5_usd_per_t: float,
    d3_days: float,
    d5_days: float,
    costs3_usd: float,
    costs5_usd: float,
    cargo_t: float,
    commission: float,
) -> float:
    """C3* implied by C5 under the indifference condition. See module docstring."""
    ratio = d3_days / d5_days
    bracket = costs3_usd - ratio * costs5_usd
    return ratio * c5_usd_per_t + bracket / (cargo_t * (1 - commission))


@dataclass(frozen=True)
class DislocationResult:
    c3_observed: float
    c3_fair_value: float
    dislocation_usd_per_t: float


def dislocation_from_voyages(
    c3_observed_usd_per_t: float,
    voyage_c3: VoyageResult,
    voyage_c5: VoyageResult,
    cargo_t: float,
    commission: float,
) -> DislocationResult:
    """Convenience wrapper: compute the dislocation directly from two VoyageResult objects
    (i.e. the same voyages priced through voyage/tce.py), using observed C5's rate.
    """
    c5_rate = voyage_c5.revenue_gross_usd / cargo_t
    c3_star = fair_value_c3(
        c5_usd_per_t=c5_rate,
        d3_days=voyage_c3.total_days,
        d5_days=voyage_c5.total_days,
        costs3_usd=voyage_c3.total_voyage_costs_usd,
        costs5_usd=voyage_c5.total_voyage_costs_usd,
        cargo_t=cargo_t,
        commission=commission,
    )
    return DislocationResult(
        c3_observed=c3_observed_usd_per_t,
        c3_fair_value=c3_star,
        dislocation_usd_per_t=c3_observed_usd_per_t - c3_star,
    )
