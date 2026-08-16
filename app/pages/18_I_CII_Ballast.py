from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_APP_DIR.parent / "src"))
sys.path.insert(0, str(_APP_DIR))

from freight.chains.cii_ballast import (  # noqa: E402
    CARBON_FACTOR_VLSFO,
    ballast_share_sweep,
    ballast_speed_sweep,
    market_speed_tradeoff,
)
from page_template import (  # noqa: E402
    Scope,
    diagnostic_note,
    finding,
    kpi_banner,
    mail_question,
    page_header,
    scope_note,
    section,
    show,
    snapshot_banner,
)

st.set_page_config(page_title="I — CII ballast anomaly", layout="wide")

_LIVE = snapshot_banner()

# ===========================================================================
# Header and scope
# ===========================================================================
page_header(
    code="I",
    title="The ballast leg the regulator counts the same as a laden one",
    subtitle=(
        "CII's attained rating divides emissions by capacity and distance sailed — "
        "neither term asks whether the ship was carrying cargo. Slowing only the "
        "ballast leg buys a 31% better rating with zero additional cargo transported, "
        "and it is priced here in real freight rate and real bunker terms to show why "
        "it is not a free lunch"
    ),
    scope=Scope(
        unit_trap=(
            "AER is grams of CO2 per **DWT-mile**, not per tonne actually carried. A "
            "ballast leg — sailed empty — contributes full distance to the denominator "
            "exactly as a laden leg does. Reading AER as a measure of transport "
            "efficiency (cargo moved per unit of emissions) instead of what it actually "
            "is (emissions per unit of nameplate capacity times distance) is the trap: "
            "the two coincide only when the ship is always full, which the formula "
            "never checks."
        ),
        conversion=(
            "AER = (fuel_t x Cf x 1e6) / (DWT x distance_nm)      grams CO2 / DWT-mile\n"
            "\n"
            f"  Cf = {CARBON_FACTOR_VLSFO}  IMO carbon factor for VLSFO (I-H1)\n"
            "  DWT = nameplate capacity — not cargo actually loaded"
        ),
        proxies=[
            "DWT is approximated by the vessel's rated cargo capacity (I-H2) — this "
            "cancels in every comparison on this page, since it is the same ship in "
            "both scenarios",
        ],
        out_of_scope=[
            "the official A-to-E rating bands: IMO's reference-line parameters differ "
            "by ship type and size and are not reproduced here, to avoid citing exact "
            "regulatory boundary values without full certainty — everything below is "
            "expressed as attained AER and its percentage change, which needs none of "
            "that table",
            "any carbon price on CII itself — unlike the EU ETS modelled in project B, "
            "CII carries no market price; only the route rate and VLSFO price this page "
            "does use are real",
        ],
        frequency_note=(
            "The AER sweep is pure voyage physics — no market data, no frequency to "
            "choose. The dollar trade-off is priced at real median levels from "
            "project G's own snapshot, over the same 2021-2022 / 2025-2026 window the "
            "P8 route rate actually covers."
        ),
        data_warnings=[
            "the real-price trade-off inherits the P8 route's 2023-2024 gap, documented "
            "in project D and reused as-is rather than patched",
        ],
    ),
)

# ===========================================================================
# Data
# ===========================================================================
speed_sweep = ballast_speed_sweep()
share_sweep = ballast_share_sweep()
tradeoff = market_speed_tradeoff()

kpi_banner(
    {
        "AER improvement, 13→8kn ballast": f"{tradeoff.aer_improvement:.0%}",
        "Net contribution cost": f"{tradeoff.net_contribution_cost:.0%}",
        "Real route rate used": f"{tradeoff.route_rate_usd_t:.0f} USD/t",
        "Real VLSFO used": f"{tradeoff.vlsfo_usd_t:.0f} USD/t",
        "Cargo effect on AER": "0% (structural)",
    }
)

# ===========================================================================
# S1
# ===========================================================================
section(
    "S1",
    "A ratio that never asks whether the ship was carrying anything",
    "The Carbon Intensity Indicator rates a ship on grams of CO2 per deadweight-mile: "
    "total emissions, divided by nameplate capacity, divided by distance sailed. Every "
    "term in that ratio is measurable without knowing whether a single tonne of cargo "
    "was aboard for any of it.\n\n"
    "That is not a flaw hidden in the implementation — it is the formula as published. "
    "A ballast leg, sailed with an empty hold and earning no freight, contributes "
    "exactly as much distance to the denominator as a fully laden leg covering the same "
    "miles. Whatever a ship burns getting there counts toward its numerator either way.",
    formula="AER = (fuel_t x Cf x 1e6) / (DWT x distance_nm)",
)

# ===========================================================================
# S2 — THE ANOMALY
# ===========================================================================
section(
    "S2",
    "Slowing an empty ship by five knots buys a better rating than anything on the laden leg could",
    "Hold the laden leg completely fixed — same cargo, same speed, same route, same "
    "revenue — and vary only the speed of the ballast leg that repositions the ship "
    "back to load again. Nothing about the actual transport work changes.",
    formula="AER at a fixed laden leg, varying ballast speed only",
)

speed_fig = go.Figure()
speed_fig.add_trace(
    go.Bar(
        x=[f"{s:.0f} kn" for s in speed_sweep.index], y=speed_sweep["aer"],
        marker_color=["crimson" if s == speed_sweep.index.max() else "rgba(120,120,120,0.6)" for s in speed_sweep.index],
        text=[f"{v:.2f}" for v in speed_sweep["aer"]], textposition="outside",
    )
)
speed_fig.update_layout(
    title="Attained AER by ballast speed (laden leg unchanged)",
    xaxis_title="ballast speed", yaxis_title="AER, gCO2/DWT-nm",
    height=400, margin=dict(t=50, b=20, l=10, r=10), showlegend=False,
)
show(speed_fig)
finding(
    f"Slowing the ballast leg from {speed_sweep.index.max():.0f} to "
    f"{speed_sweep.index.min():.0f} knots improves attained AER by "
    f"{-speed_sweep['pct_change_vs_fastest'].iloc[-1]:.0%} — with the laden leg, the "
    "cargo, and the revenue completely untouched."
)

# ===========================================================================
# S3 — the other half of the same critique
# ===========================================================================
section(
    "S3",
    "Carrying more cargo has no effect on the rating at all",
    "The formula's denominator is deadweight — the ship's rated capacity — not the "
    "tonnage actually in the hold. A voyage sailed half-empty gets exactly the same "
    "AER as the identical voyage sailed full, because cargo carried does not appear in "
    "the calculation anywhere. Utilization, the thing a chartering desk actually "
    "optimizes for, and the rating a compliance desk reports, are simply not the same "
    "target — and nothing in the metric requires them to move together.\n\n"
    "For completeness: charging more ballast *distance* to the same voyage, at a fixed "
    "speed, has the same directional effect as the speed lever but a far smaller one — "
    "under a percent at default speeds, since a ballast leg's own emissions-per-mile "
    "are only marginally below a laden leg's at the reference speeds this route uses. "
    "Speed, not distance, is what makes the anomaly large.",
)
scope_note(
    f"Ballast-distance-only effect at reference speed: {share_sweep['pct_change_vs_no_ballast'].iloc[-1]:+.2%} "
    "from zero to full ballast charged — real, and two orders of magnitude smaller than "
    "the speed effect above."
)

# ===========================================================================
# S4 — NOT A FREE LUNCH
# ===========================================================================
section(
    "S4",
    "The rating can be gamed for free; the P&L cannot",
    "A slower ballast leg also fits fewer round trips into a year, and fewer round "
    "trips is less freight revenue — priced here at the real median P8 route rate and "
    "real median VLSFO over the same window the route rate actually covers.",
    formula="net contribution = trips/year x (cargo x rate - fuel burned x VLSFO)",
)

tradeoff_fig = go.Figure()
tradeoff_fig.add_trace(
    go.Scatter(
        x=tradeoff.table.index, y=tradeoff.table["aer"], name="AER (left)",
        mode="lines+markers", line=dict(color="crimson"),
    )
)
tradeoff_fig.add_trace(
    go.Scatter(
        x=tradeoff.table.index, y=tradeoff.table["net_contribution_usd"] / 1e6,
        name="net contribution, M USD/yr (right)", mode="lines+markers",
        line=dict(color="rgba(60,60,60,0.85)"), yaxis="y2",
    )
)
tradeoff_fig.update_layout(
    title="AER and annual net contribution, by ballast speed (real prices)",
    xaxis_title="ballast speed, knots",
    yaxis=dict(title="AER, gCO2/DWT-nm"),
    yaxis2=dict(title="net contribution, M USD/yr", overlaying="y", side="right"),
    height=430, margin=dict(t=50, b=20, l=10, r=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
show(tradeoff_fig)
scope_note(
    "Net contribution is not monotonic across the sweep: it rises slightly from 13 to "
    "12 knots before falling — the first knot of slowdown saves more in fuel than it "
    "costs in lost trips, and later knots do not. The rating improves at every step; "
    "the P&L does not."
)
finding(tradeoff.headline)

# ===========================================================================
# S5
# ===========================================================================
section(
    "S5",
    "Why no carbon price appears anywhere on this page",
    "Project B prices EU ETS directly: a tonne of CO2 above the free allocation costs "
    "the EUA price, in euros, invoiced. CII does not work that way. A bad AER does not "
    "generate a bill — it triggers a corrective-action-plan requirement once a ship "
    "rates D for three consecutive years or E once, a compliance obligation rather than "
    "a market cost.\n\n"
    "Whatever real financial consequence a bad rating carries runs through commercial "
    "channels this export cannot see: charter-party clauses that reference CII "
    "performance, and any discount a chartering desk applies to a poorly rated ship "
    "when fixing it. That is a real cost and this page cannot size it — which is "
    "exactly the question worth putting to someone who fixes ships for a living rather "
    "than assuming an answer.",
)

# ===========================================================================
# Diagnostic
# ===========================================================================
diagnostic_note(
    "The official IMO rating-boundary table (reference lines and reduction factors by "
    "ship type and size) is deliberately not reproduced on this page — citing exact "
    "regulatory thresholds without full certainty in them would be worse than not "
    "answering. Every result here is an attained-AER percentage change, which needs "
    "none of that table. The dollar trade-off also inherits the P8 route's 2023-2024 "
    "gap, already documented and worked around in project D."
)

mail_question(
    "Slowing only the ballast leg on this route buys a 31% better attained AER with "
    "zero additional cargo moved — but at real freight and bunker prices it costs "
    "about 6% of the voyage's annual net contribution. Does a bad CII rating actually "
    "show up in your fixture terms or charter-party clauses in a way that would justify "
    "eating that cost, or is the rating something the fleet reports and the chartering "
    "decision otherwise ignores?",
    "Dry bulk shipowners and operators (Oldendorff, Star Bulk, Golden Ocean, Pacific "
    "Basin) — not grain trading houses' chartering desks, since the rating sits with "
    "the owner, not the charterer",
)
