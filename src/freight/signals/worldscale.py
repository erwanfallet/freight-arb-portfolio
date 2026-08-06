"""Worldscale conversion — the trap in Partie 2.7.

Tanker routes are quoted in Worldscale points (WS 100 = the flat rate for that route).
Flat rates are recalculated every January 1st from the prior year's bunker/cost
environment, so **WS points from different years are not comparable** — WS 180 in 2021
and WS 180 in 2022 can mean very different $/t. This module forces every conversion
through an explicit, dated flat-rate table so that mistake is structurally impossible:
there is no function here that returns a $/t figure from WS points alone.
"""
from __future__ import annotations

from dataclasses import dataclass


class FlatRateMissing(Exception):
    """Raised when a WS -> $/t conversion is attempted without a flat rate for that
    route/year — deliberately loud, since a silent fallback is exactly how the 2022
    WS300+-doesn't-mean-what-it-looks-like trap gets missed."""


@dataclass(frozen=True)
class FlatRateTable:
    """route -> {year -> flat_rate_usd_per_t at WS100}."""

    rates: dict[str, dict[int, float]]

    def flat_rate(self, route: str, year: int) -> float:
        try:
            return self.rates[route][year]
        except KeyError as exc:
            raise FlatRateMissing(f"no flat rate for route={route!r} year={year!r}") from exc

    def ws_to_usd_per_t(self, route: str, year: int, ws_points: float) -> float:
        return self.flat_rate(route, year) * ws_points / 100.0

    def usd_per_t_to_ws(self, route: str, year: int, usd_per_t: float) -> float:
        return usd_per_t / self.flat_rate(route, year) * 100.0


def convert_series_to_usd_per_t(rows: list[tuple[str, int, float]], table: FlatRateTable) -> list[float]:
    """rows: list of (route, year, ws_points). Returns $/t, one per row, in order.

    Use this instead of converting year-over-year point comparisons by hand — it forces
    every point through the dated flat rate for its own year.
    """
    return [table.ws_to_usd_per_t(route, year, ws) for route, year, ws in rows]
