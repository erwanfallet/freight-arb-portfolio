"""Port-to-port distances.

H5 (Partie 7): distances must be calculated, not assumed, and cross-checked against
published figures. This module has two layers:

1. `great_circle_nm` — a real (if simplistic) computation, usable today, that stands in
   for the "routing sur graphe" mentioned in Partie 6.5 as a source of edge. A proper
   sea-routing graph (respecting coastlines, canals, chokepoints) is future work — the
   function signature here is the seam where it plugs in later.
2. `NAMED_ROUTES` — the specific distances used in the brainstorm's worked examples
   (Partie 2.2, 3.4), kept as an explicit override table so the golden tests reproduce
   the doc's numbers exactly. `route_distance_nm` prefers a named route and falls back
   to great-circle, so unnamed routes still return a defensible number instead of
   raising.
"""
from __future__ import annotations

import math

EARTH_RADIUS_NM = 3440.065  # nautical miles

# approximate port coordinates (lat, lon) — good enough for a cross-check, not for routing
PORT_COORDS: dict[str, tuple[float, float]] = {
    "TUBARAO": (-20.29, -40.29),
    "QINGDAO": (36.07, 120.38),
    "PORT_HEDLAND": (-20.31, 118.57),
    "ROTTERDAM": (51.95, 4.14),
    "RICHARDS_BAY": (-28.78, 32.09),
    "SANTOS": (-23.96, -46.33),
    "NEW_YORK": (40.67, -74.0),
    "RAS_TANURA": (26.64, 50.16),
    "NINGBO": (29.87, 121.55),
}

# distances as used in the brainstorm's worked examples (Partie 2.2 §Exemple chiffré,
# Partie 3.4 §Ballast Constraint) — kept literal so golden tests match the doc.
NAMED_ROUTES: dict[tuple[str, str], float] = {
    ("TUBARAO", "QINGDAO"): 11_000.0,      # C3 laden leg
    ("QINGDAO", "TUBARAO"): 10_500.0,      # C3 ballast leg
    ("PORT_HEDLAND", "QINGDAO"): 3_500.0,  # C5 laden leg
    ("QINGDAO", "PORT_HEDLAND"): 3_500.0,  # C5 ballast leg
}


def great_circle_nm(origin: str, destination: str) -> float:
    """Great-circle distance in nautical miles between two named ports.

    A lower bound on actual sailed distance (ignores coastlines, canals, routing
    restrictions) — use for cross-checking NAMED_ROUTES, not as a primary source once
    a real routing graph is wired in.
    """
    lat1, lon1 = PORT_COORDS[origin]
    lat2, lon2 = PORT_COORDS[destination]
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_NM * c


def route_distance_nm(origin: str, destination: str) -> float:
    """Distance for a voyage leg, preferring the documented figure over great-circle."""
    if (origin, destination) in NAMED_ROUTES:
        return NAMED_ROUTES[(origin, destination)]
    if origin in PORT_COORDS and destination in PORT_COORDS:
        return great_circle_nm(origin, destination)
    raise KeyError(
        f"no distance known for {origin!r} -> {destination!r}: add to NAMED_ROUTES or PORT_COORDS"
    )
