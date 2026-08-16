"""Compute every page's frames from the Bloomberg export and write them to the snapshot.

Run this on the machine that has the export, before pushing:

    python scripts/build_snapshot.py

What lands in `data/snapshot/` is the **output of the analysis** — margins, spreads,
breakeven curves, implied TCEs — never the market data it was computed from. The export
itself stays out of the repository, which is public and has no right to redistribute it.

Everything is written at the loaders' **default parameters**. A deployed page therefore
shows one parameterisation rather than an interactive model, and says so; run the app
locally with the export in place and the sliders come back.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from agri.data.bloomberg_loader import DEFAULT_PATH, load  # noqa: E402
from agri.data.snapshot import SNAPSHOT_DIR, save_frame, to_snapshot_frame  # noqa: E402


def _series_frame(**series: pd.Series) -> pd.DataFrame:
    return pd.concat(series, axis=1, sort=True).dropna(how="all")


def build() -> dict[str, tuple[int, int]]:
    """Compute and save every snapshot frame. Returns {name: (rows, columns)}."""
    from agri.chains.china_soy import affordable_origination_budget, load_real_crush_frame
    from agri.chains.crush_tracking import load_real_board_frame
    from agri.chains.plant_option import real_board_crush_margin
    from agri.chains.feedstock_lcfs import load_soyoil_usd_lb
    from agri.chains.freight_cf import load_real_route_frame
    from agri.chains.hedge_cost import load_real_hedge_frame
    from agri.chains.oil_substitution import load_peg_window_spread
    from agri.chains.sugar_mix import load_real_parity_frame
    from agri.chains.white_premium import load_real_richness_frame
    from freight.chains.coal import load_real_switching_frame
    from freight.chains.freight_incidence import load_incidence_frame
    from freight.chains.grain_seasonal import load_panamax_frame
    from freight.chains.ironore import load_real_premium_frame
    from freight.chains.products import load_real_transatlantic_frame

    # `__wrapped__` reaches the undecorated function, so building never reads its own
    # snapshot back — a subtle way to freeze stale results if it were skipped.
    def live(function):
        return getattr(function, "__wrapped__", function)

    route = live(load_real_route_frame)()
    budget = live(affordable_origination_budget)()

    frames: dict[str, pd.DataFrame] = {
        "t1_1_route": _series_frame(
            route_rate_usd_t=route.route_rate_usd_t,
            tce_no_ballast=route.tce_no_ballast,
            tce_full_ballast=route.tce_full_ballast,
        ),
        # The 2021 USD/day segment is a separate series, and the T1-1 page needs it as the
        # plausibility ceiling that adjudicates the whole disagreement.
        "t1_1_tce_2021": to_snapshot_frame(
            load("p8_route_tce_2021").rename("tce_usd_day").rename_axis("date")
        ),
        # VLSFO: the T1-1 page quotes the latest bunker level in its prose.
        "t1_1_vlsfo": to_snapshot_frame(
            load("vlsfo_singapore").rename("vlsfo_usd_t").rename_axis("date")
        ),
        "t1_2_hedge": live(load_real_hedge_frame)(),
        "t2_3_board": live(load_real_board_frame)(),
        "t2_4_richness": live(load_real_richness_frame)(),
        "t2_5_china": live(load_real_crush_frame)(),
        # US board crush: T2-5 uses it as the counter-example, the margin that never went
        # negative and therefore never posed the shutdown question at all.
        "t2_5_us_board": to_snapshot_frame(
            live(real_board_crush_margin)(start="2018-01-01").rename("board_crush").rename_axis("date")
        ),
        "t2_6_peg": live(load_peg_window_spread)(),
        "t3_1_soyoil": to_snapshot_frame(
            live(load_soyoil_usd_lb)("2015").rename_axis("date")
        ),
        "t3_2_parity": live(load_real_parity_frame)(),
        "t3_4_budget": budget.frame,
        "a_iron_ore": live(load_real_premium_frame)(),
        "b_switching": live(load_real_switching_frame)(),
        "c_products": live(load_real_transatlantic_frame)(),
        "d_panamax_seasonal": live(load_panamax_frame)(),
        "e_freight_incidence": live(load_incidence_frame)(),
    }

    written: dict[str, tuple[int, int]] = {}
    for name, frame in frames.items():
        save_frame(name, frame)
        written[name] = frame.shape
    return written


def main() -> int:
    if not DEFAULT_PATH.exists():
        print(f"Bloomberg export not found at {DEFAULT_PATH}", file=sys.stderr)
        print("This script must run on a machine that has it.", file=sys.stderr)
        return 1

    written = build()
    total_bytes = sum(path.stat().st_size for path in SNAPSHOT_DIR.glob("*.parquet"))
    print(f"snapshot written to {SNAPSHOT_DIR}")
    for name, (rows, columns) in written.items():
        print(f"  {name:<18} {rows:>6} rows x {columns} cols")
    print(f"\n{len(written)} frames, {total_bytes / 1024:.0f} kB total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
