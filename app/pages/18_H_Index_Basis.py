from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_APP_DIR.parent / "src"))
sys.path.insert(0, str(_APP_DIR))

from freight.chains.index_basis import (  # noqa: E402
    MIN_OBS_FOR_VERDICT,
    absolute_risk_scaling,
    load_index_basis_frame,
    tail_decomposition,
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

st.set_page_config(page_title="H — Index basis", layout="wide")

_LIVE = snapshot_banner()

# ===========================================================================
# Header and scope
# ===========================================================================
page_header(
    code="H",
    title="The index tracks the route worst on the days that matter least",
    subtitle=(
        "A route-specific hedger fears the index decouples exactly when the hedge is "
        "needed most. On the real BPI and the real P8 route, that fear is rejected — "
        "the index tracks the route BETTER on its biggest-move days — but the "
        "unexplained dollars grow just as fast as the move itself, so the better "
        "statistical fit buys no more absolute protection"
    ),
    scope=Scope(
        unit_trap=(
            "R² measures a **share** of variance; a hedger carries **dollars** of "
            "unexplained move. A rising R² in the tail sounds like the hedge working "
            "better exactly when it matters — the trap is stopping there instead of "
            "checking what happens to the residual in the units the book is marked in."
        ),
        conversion=(
            "d(P8) = a + b x d(BPI) + residual        HAC errors, split by |d(P8)|\n"
            "\n"
            "  R² = share of the route's variance the index explains\n"
            "  residual std = the route's move, in USD/t, the index leaves unhedged"
        ),
        proxies=[
            "BPI is a global four-route Panamax average (project D's own limit, "
            "restated here as the mechanism under test rather than a caveat)",
        ],
        out_of_scope=[
            "any specific FFA contract's actual basis — this tests the index and the "
            "route rate directly, not a tradable derivative's marked basis",
        ],
        frequency_note=(
            "Daily, on the 624 dates both series share. Tested on changes throughout "
            "(H-H1) — a levels regression on two trending series would describe their "
            "shared trend, not their co-movement."
        ),
        data_warnings=[
            "the P8 route rate is missing 2023 and 2024 entirely (the same gap "
            "documented in project D) — this covers 2021-2022 and 2025-2026 only "
            "(H-H4)",
            f"a verdict requires at least {MIN_OBS_FOR_VERDICT} observations in a "
            "bucket (H-H3) — a finer slice than the top decile exists in the data and "
            "is deliberately not reported as a verdict",
        ],
    ),
)

# ===========================================================================
# Data
# ===========================================================================
frame = load_index_basis_frame()
decomposition = tail_decomposition(frame)
scaling = absolute_risk_scaling(frame)

kpi_banner(
    {
        "Full-sample R²": f"{decomposition.full.r_squared:.1%}",
        "Calm-day R²": f"{decomposition.calm.r_squared:.1%}",
        "Extreme-day R²": f"{decomposition.top_decile.r_squared:.1%}",
        "Residual growth, calm→extreme": f"{scaling.resid_scaling:.0f}x",
        "Raw move growth, calm→extreme": f"{scaling.raw_scaling:.0f}x",
    }
)

# ===========================================================================
# S1
# ===========================================================================
section(
    "S1",
    "A hedger's fear that is answerable rather than arguable",
    "Someone hedging the specific Santos-Qingdao route with a BPI-linked instrument is "
    "accepting a basis: the index and the route are not the same thing, and the "
    "question is how much that gap costs, and when. The natural fear is that the gap "
    "widens exactly on the days the route moves hardest — the days the hedge is "
    "actually protecting something.\n\n"
    "That is a testable claim, not an assumption to reason about in the abstract: "
    "regress the route's daily change on the index's daily change, split by how large "
    "the route's own move was that day, and read off what happens to the fit.",
    formula="d(P8) = a + b x d(BPI) + residual, split by |d(P8)| into calm / top 50% / top 25% / top 10%",
)

# ===========================================================================
# S2 — THE FIRST RESULT
# ===========================================================================
section(
    "S2",
    "The fear is rejected — the index tracks the route better in the tails",
    f"On a typical day, the index explains almost nothing: full-sample R² is "
    f"**{decomposition.full.r_squared:.1%}**, and the relationship is not "
    f"statistically significant (p={decomposition.full.pvalue:.2f}). Splitting by the "
    "size of the route's own move, the fit rises at every step — "
    f"{decomposition.calm.r_squared:.1%} on calm days, "
    f"{decomposition.top_half.r_squared:.1%} on the top half, "
    f"{decomposition.top_quarter.r_squared:.1%} on the top quarter, "
    f"**{decomposition.top_decile.r_squared:.1%}** on the top decile — where it also "
    f"turns significant (p={decomposition.top_decile.pvalue:.3f}).\n\n"
    "This is the opposite of the naive fear: the index does not decouple on the route's "
    "biggest days, it tracks it more closely than on a quiet one.",
)

bucket_labels = ["calm\n(bottom 50%)", "top 50%", "top 25%", "top 10%"]
bucket_r2 = [decomposition.calm.r_squared, decomposition.top_half.r_squared,
             decomposition.top_quarter.r_squared, decomposition.top_decile.r_squared]
bucket_sig = [decomposition.calm.significant, decomposition.top_half.significant,
              decomposition.top_quarter.significant, decomposition.top_decile.significant]
r2_fig = go.Figure()
r2_fig.add_trace(
    go.Bar(
        x=bucket_labels, y=bucket_r2,
        marker_color=["crimson" if s else "rgba(120,120,120,0.55)" for s in bucket_sig],
        text=[f"{v:.1%}" for v in bucket_r2], textposition="outside",
    )
)
r2_fig.update_layout(
    title="R² of the route explained by the index, by size of the route's own move",
    yaxis_title="R²", yaxis_tickformat=".0%",
    height=400, margin=dict(t=50, b=20, l=10, r=10), showlegend=False,
)
show(r2_fig)
scope_note("Bars in red are statistically significant (p<0.05); grey bars are not.")
finding(decomposition.headline)

# ===========================================================================
# S3 — THE DEEPER RESULT
# ===========================================================================
section(
    "S3",
    "A better R² does not mean better protection, in the units a book is marked in",
    "R² is a share of variance. What a hedger carries on a P&L is dollars. The "
    "residual — the part of the route's move the index does not explain — is "
    f"**{scaling.calm_resid_std:.2f} USD/t on calm days** and "
    f"**{scaling.extreme_resid_std:.2f} USD/t on the biggest-move decile**, a "
    f"**{scaling.resid_scaling:.0f}x** increase. The route's own raw move grows "
    f"**{scaling.raw_scaling:.0f}x** over the same split.\n\n"
    "Those two numbers are close enough that the index's improved statistical fit buys "
    "almost no improvement in absolute protection: the unexplained dollars scale up "
    "essentially in line with the move itself, whether or not the R² looks better that "
    "day. The tail is not more dangerous *relative to the hedge* than a calm day is — "
    "it is simply larger, and the hedge is neither more nor less complete against it.",
    formula="resid_std(extreme) / resid_std(calm)  vs.  raw_std(extreme) / raw_std(calm)",
)

scale_fig = go.Figure()
scale_fig.add_trace(
    go.Bar(
        x=["raw move (P8)", "unexplained residual"],
        y=[scaling.raw_scaling, scaling.resid_scaling],
        marker_color=["rgba(120,120,120,0.6)", "crimson"],
        text=[f"{scaling.raw_scaling:.0f}x", f"{scaling.resid_scaling:.0f}x"],
        textposition="outside",
    )
)
scale_fig.update_layout(
    title="Growth from calm days to the top decile of route moves",
    yaxis_title="multiple, calm to extreme",
    height=380, margin=dict(t=50, b=20, l=10, r=10), showlegend=False,
)
show(scale_fig)
finding(scaling.headline)

# ===========================================================================
# S4
# ===========================================================================
section(
    "S4",
    "Where the resolution runs out",
    f"Slicing finer than the top decile is possible in the data — the 90th-to-95th "
    f"percentile of route moves alone carries a striking R², computed and then "
    f"discarded from this page because it rests on fewer than "
    f"{MIN_OBS_FOR_VERDICT} observations. A bucket that small can be moved by two or "
    "three dates, and reporting it as a finding would be mistaking noise for structure "
    "— so the top decile, at 62 observations, is the finest slice this page reports a "
    "verdict on.",
)

# ===========================================================================
# Diagnostic
# ===========================================================================
diagnostic_note(
    "The P8 route rate is missing 2023 and 2024 entirely (H-H4), the same gap "
    "documented in project D — this result covers the 2021-2022 boom-to-slump "
    "transition and the 2025-2026 window, not a continuous history. It also cannot "
    "distinguish a genuinely structural tail relationship from a property of this "
    "particular window; a longer route history would be needed to confirm it repeats."
)

mail_question(
    "On the days the P8 route moves hardest, the BPI index actually tracks it more "
    "closely than on a quiet day — but the dollars left unhedged grow just as fast as "
    "the move itself, so the better statistical fit does not translate into better "
    "protection. On your largest route-specific moves, is there usually an "
    "identifiable driver — a large single fixture, port congestion, a itinerary shift "
    "— that a desk would see operationally but that never shows up in the index?",
    "Freight derivatives and FFA risk desks (Freight Investor Services, SSY Futures, "
    "Clarksons Securities) — the paper-hedging side of freight risk, distinct from the "
    "physical chartering desks this portfolio's other projects target",
)
