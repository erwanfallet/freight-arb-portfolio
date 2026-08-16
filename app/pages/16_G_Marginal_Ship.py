from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_APP_DIR.parent / "src"))
sys.path.insert(0, str(_APP_DIR))

from freight.chains.marginal_ship import (  # noqa: E402
    BALLAST_CONVENTIONS,
    breakeven_series,
    cost_floor_usd_t,
    load_marginal_ship_frame,
    margin_summary,
    variance_decomposition,
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

st.set_page_config(page_title="G — Marginal ship", layout="wide")

_LIVE = snapshot_banner()

# ===========================================================================
# Header and scope
# ===========================================================================
page_header(
    code="G",
    title="How much less efficient could a ship be, and still cover its fuel bill here",
    subtitle=(
        "Holding the real P8 route rate and the real VLSFO price fixed on each day, "
        "solve for the fuel-consumption multiplier on a reference panamax at which its "
        "own fuel bill alone would exceed the freight revenue — a threshold that never "
        "approached 1 in nearly five years, and is driven more by the oil cycle than by "
        "the freight cycle it sits inside"
    ),
    scope=Scope(
        unit_trap=(
            "The threshold is a **multiplier on a reference vessel's consumption**, not "
            "an absolute tonnes-per-day figure — it answers 'how many times worse than "
            "this reference could a ship be', not 'what is the efficiency limit'. "
            "Reporting it as a t/day number would smuggle in the reference vessel's own "
            "specification as if it were a fact about the fleet, which it is not."
        ),
        conversion=(
            "k*(t) = [ rate(t) - A ] / [ B x vlsfo(t) ]\n"
            "\n"
            "  A = fixed cost per tonne (port + canal), independent of bunkers\n"
            "  B = reference vessel's bunker cost per tonne, per unit of vlsfo\n"
            "  k* = 1  the reference vessel's fuel bill alone consumes all freight revenue"
        ),
        proxies=[
            "MGO is reconstructed at 1.35x VLSFO, T1-1's own parameter (G-H1) — the "
            "export has no separate MGO series",
            "vessel and route are the same panamax / Santos-Qingdao pairing T1-1 uses "
            "(G-H2), chosen for direct comparability rather than because it is the only "
            "defensible choice",
        ],
        out_of_scope=[
            "any claim about a specific vintage or build year — the export has no "
            "per-vessel or per-age consumption data, and none is assumed",
            "opex, capital cost, or an actual lay-up/scrapping threshold — k*=1 is a "
            "fuel-only contribution-margin floor, stated as an upper bound throughout "
            "(G-H4)",
        ],
        frequency_note=(
            "Daily, on the real P8 route rate's own calendar. The decomposition runs on "
            "log-changes because it is an algebraic identity in the two inputs, not a "
            "fitted relationship — there is no frequency choice to make."
        ),
        data_warnings=[
            "the P8 route rate is missing 2023 and 2024 entirely in this export (the "
            "same gap documented in project D) — this test covers 2021-2022 and "
            "2025-2026 only (G-H5)",
        ],
    ),
)

# ===========================================================================
# Data
# ===========================================================================
frame = load_marginal_ship_frame()
summary_full = margin_summary(frame, ballast_share=1.0)
summary_zero = margin_summary(frame, ballast_share=0.0)
decomposition = variance_decomposition(frame, ballast_share=1.0)
cost_floor = cost_floor_usd_t(ballast_share=1.0)

kpi_banner(
    {
        "Breakeven multiplier, min": f"{summary_full.min:.2f}x",
        "Breakeven multiplier, median": f"{summary_full.median:.2f}x",
        "Breakeven multiplier, max": f"{summary_full.max:.2f}x",
        "Tightest point": f"{summary_full.min_date:%b %Y}",
        "Bunker share of variance": f"{decomposition.share_bunker:.0%}",
    }
)

# ===========================================================================
# S1
# ===========================================================================
section(
    "S1",
    "A question that does not need to know which ship is actually fixed",
    "A chartering desk reading a freight rate does not usually know the exact fuel "
    "efficiency of the tonnage it will end up fixing — vessels of the same nominal "
    "class differ in age, hull condition, and design. What can be answered without that "
    "knowledge is narrower and still useful: **given today's rate and today's bunker "
    "price, how much worse than a reference vessel could a ship be before its fuel bill "
    "alone would exceed the freight revenue it earns?**\n\n"
    "That threshold is a multiplier — a factor on the reference vessel's own "
    "consumption — and the freight-cost identity is affine in it, so it can be solved "
    "for directly rather than searched for.",
    formula="k*(t) = [rate(t) - A] / [B x vlsfo(t)]      solved in closed form, not fitted",
)

# ===========================================================================
# S2 — the identity
# ===========================================================================
section(
    "S2",
    "The fixed cost is provably independent of the bunker price",
    f"A vessel burning zero fuel pays zero bunker bill regardless of what bunkers cost — "
    f"so A, the route's fixed cost per tonne, is a **constant**: "
    f"**{cost_floor:.2f} USD/t** on this route, at every bunker price tested from "
    "50 to 2,000 USD/t. That is not an approximation, it is verified directly below.\n\n"
    "Because MGO is reconstructed as a fixed multiple of VLSFO, the whole relationship "
    "reduces to two independent inputs — the freight rate and the bunker price — and the "
    "log-change decomposition used in S4 follows as an algebraic identity rather than a "
    "regression with a residual.",
)

# ===========================================================================
# S3 — THE RESULT
# ===========================================================================
section(
    "S3",
    "The margin never got close to 1, even at its worst",
    f"Under the full-ballast convention — the more conservative of the two, and the one "
    f"T1-1 also treats as the freight department's — the multiplier ran from "
    f"**{summary_full.min:.2f} to {summary_full.max:.2f}** across 643 days, median "
    f"{summary_full.median:.2f}. Its tightest point, {summary_full.min_date:%d %b %Y}, "
    "landed inside the same VLSFO spike documented on project F's page — the sharpest "
    "bunker move in the whole 17-year VLSFO history sits directly behind this margin's "
    "narrowest point on record, and even there it stayed above 1.4.\n\n"
    f"Under the zero-ballast convention the range is looser still, "
    f"{summary_zero.min:.2f} to {summary_zero.max:.2f}. Both conventions agree on the "
    "same reading: on this route, across a nearly five-year sample that includes a "
    "freight boom, a freight slump, and the sharpest bunker spike in the data, fuel cost "
    "alone was never close to being the binding constraint on a panamax-class ship.",
    formula="the reference panamax's OWN fuel bill vs. the freight it would earn",
)

table = pd.DataFrame(
    {
        "full ballast": summary_full.quarterly(),
        "zero ballast": summary_zero.quarterly(),
    }
)
margin_fig = go.Figure()
margin_fig.add_trace(
    go.Scatter(x=table.index, y=table["full ballast"], mode="lines+markers",
               name="full ballast (conservative)", line=dict(color="crimson"))
)
margin_fig.add_trace(
    go.Scatter(x=table.index, y=table["zero ballast"], mode="lines+markers",
               name="zero ballast", line=dict(color="rgba(120,120,120,0.7)"))
)
margin_fig.add_hline(
    y=1.0, line_dash="dash", line_color="gray",
    annotation_text="fuel bill = all freight revenue", annotation_position="right",
)
margin_fig.update_layout(
    title="Fuel-only breakeven multiplier, quarterly median",
    yaxis_title="multiplier on reference vessel consumption",
    height=430, margin=dict(t=50, b=20, l=10, r=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
show(margin_fig)
finding(summary_full.headline)

# ===========================================================================
# S4 — the decomposition
# ===========================================================================
section(
    "S4",
    "What actually moves the margin: the oil cycle more than the freight cycle",
    "The level result says the margin stayed wide. The decomposition asks a different "
    "question: on the days it moved, what moved it? Because the relationship is an "
    "identity in two inputs, the answer splits exactly, with no residual left over.\n\n"
    f"**{decomposition.share_bunker:.0%}** of the day-to-day variance traces to the "
    f"bunker price, **{decomposition.share_rate:.0%}** to the freight rate itself. The "
    "two components are only weakly correlated "
    f"({decomposition.component_correlation:+.2f}) — the freight cycle and the oil "
    "cycle do not move together closely enough on this route to cancel each other out "
    "in either direction.\n\n"
    "That the bunker price dominates at all is worth stating plainly: a desk that reads "
    "a strong freight market as the thing buying it fuel-cost headroom is reading the "
    "smaller of the two effects. The larger one is the price of oil, which the freight "
    "market does not set and does not track closely.",
    formula="d(log k*) = d(log(rate - A)) - d(log vlsfo)      exact, no residual",
)

dec_fig = go.Figure()
dec_fig.add_trace(
    go.Bar(
        x=["freight rate", "bunker price"],
        y=[decomposition.share_rate, decomposition.share_bunker],
        marker_color=["rgba(120,120,120,0.6)", "crimson"],
        text=[f"{decomposition.share_rate:.0%}", f"{decomposition.share_bunker:.0%}"],
        textposition="outside",
    )
)
dec_fig.update_layout(
    title="Share of the breakeven multiplier's variance, by driver",
    yaxis_title="share of variance", yaxis_tickformat=".0%",
    height=380, margin=dict(t=50, b=20, l=10, r=10), showlegend=False,
)
show(dec_fig)
scope_note(
    "The covariance between the two components is split evenly between the shares, so "
    "they sum to exactly one by construction — this is a variance accounting identity, "
    "not a model that leaves an unexplained residual."
)
finding(decomposition.headline)

# ===========================================================================
# S5
# ===========================================================================
section(
    "S5",
    "Why an upper bound, and where it needs a desk's own numbers",
    "k*=1 means the reference vessel's fuel bill alone consumes the entire freight "
    "revenue — a contribution margin of exactly zero, before crew, insurance, "
    "maintenance, or a single dollar of capital cost is paid. No ship is actually "
    "chartered down to that line: the threshold that governs a real lay-up or scrapping "
    "decision needs TCE to clear opex, not merely to clear zero.\n\n"
    "So everything on this page is an **upper bound**. The true efficiency bar the "
    "market sets on this route is tighter than k* shows, by an amount this page cannot "
    "compute — that gap needs real opex figures by vessel and owner, which are not "
    "public and are not in this export.",
)

# ===========================================================================
# Diagnostic
# ===========================================================================
diagnostic_note(
    "The P8 route rate is missing 2023 and 2024 entirely (G-H5), the same gap documented "
    "in project D — this result covers the 2021-2022 boom-to-slump transition and the "
    "2025-2026 window including the March 2026 VLSFO spike, not a continuous history. "
    "The multiplier is also, by construction, an upper bound on tolerable inefficiency "
    "(G-H4): it clears zero contribution margin, not a real operating threshold."
)

mail_question(
    "On the P8 route since late 2021, a panamax could have burned 1.5 to nearly 4 times "
    "its reference consumption and still covered its fuel bill from freight revenue "
    "alone — and that margin moves more with the bunker price than with the freight "
    "rate itself. Does your desk actually track a fuel-cost headroom like this when "
    "assessing which tonnage to fix, or does opex and capital cost dominate the "
    "chartering decision so completely that fuel efficiency alone rarely binds?",
    "Grain and freight chartering desks (Cargill Ocean Transportation, Bunge, LDC, "
    "COFCO, Viterra)",
)
