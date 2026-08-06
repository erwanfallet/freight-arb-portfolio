"""Post-mortem episode registry — Partie 11.

Each episode follows the same template: reconstruct the ex-ante state, the decision the
market actually made, the ex-post optimal decision, the measured gap, and the
transposition to today. This module only holds the registry (dates, angle, the specific
question each episode is meant to answer); the actual reconstruction code is
chain/data-specific and gets added once real series are available.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Episode:
    name: str
    start: date
    end: date | None
    angle: str  # the specific question this episode is meant to answer


EPISODES: list[Episode] = [
    Episode(
        "valemax", date(2011, 1, 1), date(2015, 12, 31),
        "Was the Valemax investment rational ex ante at 2011 forward curves? "
        "Vale bought scale by selling flexibility (China port bans, Teluk Rubiah transshipment).",
    ),
    Episode(
        "brumadinho", date(2019, 1, 25), None,
        "Repricing delay of C3 vs IODEX after ~90Mt of export capacity was destroyed — "
        "did freight react before, after, or not at all, and over what execution window?",
    ),
    Episode(
        "imo2020", date(2020, 1, 1), None,
        "Did the Dec-2019 FFA forward curve price in the HSFO->VLSFO bunker discontinuity?",
    ),
    Episode(
        "floating_storage_2020", date(2020, 4, 1), date(2020, 6, 30),
        "Self-limiting arb: VLCC storage removes tonnage from the transport market and "
        "spikes rates — quantify the self-limiting mechanism.",
    ),
    Episode(
        "suez_2021", date(2021, 3, 23), date(2021, 3, 29),
        "Pure exogenous supply shock — implicit ton-mile created, cross-segment contagion.",
    ),
    Episode(
        "capesize_boom_2021", date(2021, 10, 1), date(2021, 10, 31),
        "Was the dislocation at a historical extreme? Would the signal have given the "
        "trade, with how much lead time?",
    ),
    Episode(
        "russian_crude_reroute_2022", date(2022, 2, 24), None,
        "Massive ton-mile shift, dark fleet emergence, route decorrelation.",
    ),
    Episode(
        "panama_drought", date(2023, 10, 1), date(2024, 6, 30),
        "Transit/draft restrictions and LPG/LNG US reorientation via Cape/Suez.",
    ),
    Episode(
        "red_sea", date(2023, 12, 1), None,
        "Cape rerouting, +9-10 days Asia-Europe — did the shock transmit to dry bulk, "
        "through which channel?",
    ),
]


def episodes_overlapping(start: date, end: date) -> list[Episode]:
    return [
        e for e in EPISODES
        if e.start <= end and (e.end is None or e.end >= start)
    ]
