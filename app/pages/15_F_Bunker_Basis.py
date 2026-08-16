from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_APP_DIR.parent / "src"))
sys.path.insert(0, str(_APP_DIR))

from freight.chains.bunker_basis import (  # noqa: E402
    BBL_PER_TONNE_DISTILLATE,
    BBL_PER_TONNE_FUEL_OIL,
    break_attempt,
    density_mis_sizing,
    hedge_effectiveness,
    load_bunker_frame,
    rolling_hedge_beta,
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

st.set_page_config(page_title="F — Bunker basis", layout="wide")

_LIVE = snapshot_banner()

# ===========================================================================
# Header and scope
# ===========================================================================
page_header(
    code="F",
    title="You hedge bunkers with crude, and here is what you keep",
    subtitle=(
        "Crude is the instrument bunkers actually get hedged with — it is the cheapest "
        "and deepest one available. The hedge ratio to it has fallen by a factor of "
        "three since 2016, and even in the best window crude only ever explained "
        "about half the daily variance it was hedging"
    ),
    scope=Scope(
        unit_trap=(
            "Crude is quoted in **USD per barrel** and bunkers in **USD per tonne**, so "
            "every hedge ratio has a density folded into it. Fuel oil runs about "
            f"**{BBL_PER_TONNE_FUEL_OIL} barrels per tonne**; middle distillate, the "
            f"figure most people actually know because it is the diesel number, runs "
            f"about **{BBL_PER_TONNE_DISTILLATE}**. Using the distillate density on a "
            "fuel oil hedge oversizes the position by "
            f"**{density_mis_sizing()['oversize_fraction']:.0%}** — silently, because "
            "both numbers are correct for their own product and nothing about the "
            "calculation looks wrong."
        ),
        conversion=(
            "d(VLSFO) = beta x d(Brent x bbl/t) + residual\n"
            "\n"
            "  beta = the crude tonnage that hedges one tonne of bunkers\n"
            "  residual = the risk crude cannot see, whatever beta is set to"
        ),
        proxies=[
            "the series is labelled VLSFO back to 2009, which predates the product — "
            "the 0.5% sulphur grade did not exist as a bunker fuel before the 2020 "
            "sulphur cap. What the pre-2020 history describes is not established here; "
            "the drift below is reported as a property of the quoted series",
        ],
        out_of_scope=[
            "sizing a hedge against a specific bunker delivery port or grade — the "
            "series is the Singapore benchmark, the largest bunkering hub, not a "
            "specific stem",
            "the cost of the hedge itself (roll, basis, margin) — this page measures "
            "only how much price risk the instrument removes, not what removing it costs",
        ],
        frequency_note=(
            "All estimation runs on daily changes, never on levels — two trending "
            "series in levels would hand back a hedge ratio that describes their shared "
            "trend and fails the moment the trend does."
        ),
        data_warnings=[
            "a Chow break test at the IMO 2020 date is attempted and shown even though "
            "it fails — it rejects at a placebo date too, and that failure is the "
            "reason the page reports drift instead of a break",
        ],
    ),
)

# ===========================================================================
# Data
# ===========================================================================
frame = load_bunker_frame()
mis = density_mis_sizing()
drift = rolling_hedge_beta(frame)
attempt = break_attempt(frame)
eff = hedge_effectiveness(frame)

kpi_banner(
    {
        "Hedge ratio, peak window": f"{drift.max_beta:.2f}",
        "Hedge ratio, latest window": f"{drift.last_beta:.2f}",
        "Collapse factor": f"{drift.collapse_factor:.1f}x",
        "Best variance explained": f"{eff.best_r2:.0%}",
        "Density mis-sizing": f"+{mis['oversize_fraction']:.0%}",
    }
)

# ===========================================================================
# S1
# ===========================================================================
section(
    "S1",
    "The largest variable cost with the only liquid hedge attached to it",
    "Bunkers are the biggest variable cost of a voyage, and crude oil futures are the "
    "cheapest, deepest instrument anyone can put against them. So they get hedged with "
    "crude — not because crude is a precise match, but because it is the instrument "
    "that is actually available in size.\n\n"
    "That makes the hedge ratio a real number rather than a convention, and it can be "
    "measured directly: regress the change in the bunker price on the change in crude, "
    "converted into the same unit, and the slope is the tonnes of crude exposure that "
    "offset one tonne of bunker exposure.\n\n"
    "The question this page asks is not whether the hedge helps — it does — but how "
    "much of the risk it actually removes, and whether the ratio it removes at is "
    "stable enough to size a position on.",
    formula="beta = cov(d VLSFO, d Brent_usd_t) / var(d Brent_usd_t)      HAC errors",
)

# ===========================================================================
# S2 — the unit trap, quantified
# ===========================================================================
section(
    "S2",
    "Both densities are correct, which is what makes the wrong one dangerous",
    f"Fuel oil converts at **{mis['correct_bbl_per_t']} barrels per tonne**. Middle "
    f"distillate — diesel, gasoil — converts at **{mis['mistaken_bbl_per_t']}**. Both "
    "numbers are standard reference values for their own product, and a desk that "
    "mainly thinks in diesel terms can reach for the wrong one without anything in the "
    "calculation flagging it.\n\n"
    f"The effect is a hedge sized **{mis['oversize_fraction']:.0%} too large** — every "
    "dollar of intended crude exposure becomes "
    f"{1 + mis['oversize_fraction']:.2f} dollars of actual exposure. In a falling "
    "market that overhedge looks like a bonus; in a rising one it is an unpriced short "
    "crude position riding along with the bunker hedge.",
)

# ===========================================================================
# S3 — the test that failed, and why it is shown
# ===========================================================================
section(
    "S3",
    "The obvious story does not survive a placebo",
    "The sulphur cap that created VLSFO as a distinct grade took effect on 1 January "
    "2020, so the relationship to crude should break there if the product itself "
    "changed. A Chow test on that date rejects stability decisively "
    f"(**F = {attempt.tested_f:.1f}**).\n\n"
    "It also rejects at a date chosen inside a quiet stretch of the 2010s, where "
    f"nothing structural happened (**F = {attempt.placebo_f:.1f}**). On "
    f"{len(frame):,} daily observations of two trending series, a break test rejects "
    "almost anywhere — so a single rejection does not identify a break, and the IMO "
    "2020 date is not shown to be special by this test.\n\n"
    "That failure is the finding, not a dead end. It rules out the tidy story and "
    "points at the alternative: the relationship does not break once, it drifts "
    "continuously — which is the harder case for a hedger, because a break can be "
    "dated and a position resized on the day, and drift cannot.",
)
finding(attempt.headline)

# ===========================================================================
# S4 — THE RESULT
# ===========================================================================
section(
    "S4",
    "The hedge ratio that would have been sized in 2017 is three times too large today",
    "Estimated separately on each two-year window rather than smoothed, so the "
    "numbers show what a hedger estimating on any two-year history would actually "
    "have carried into the years after it — not a rolling average that blends the "
    "regimes together.\n\n"
    f"The ratio rose through the 2010s to a peak of **{drift.max_beta:.2f}** in "
    "2016-2017, and has fallen in every window since, to "
    f"**{drift.last_beta:.2f}** most recently. The residual risk left behind moves the "
    "other way: it was smallest exactly when the ratio peaked, and has since more "
    "than doubled.",
    formula="hedge ratio, and the risk it leaves behind, by two-year window",
)

table = drift.to_frame()
beta_fig = go.Figure()
beta_fig.add_trace(
    go.Bar(x=table.index, y=table["beta"], name="hedge ratio (beta)", marker_color="crimson")
)
beta_fig.add_trace(
    go.Scatter(
        x=table.index, y=table["residual_sigma"], name="residual sigma (USD/t)",
        mode="lines+markers", yaxis="y2", line=dict(color="rgba(120,120,120,0.85)"),
    )
)
beta_fig.update_layout(
    title="Crude hedge ratio and residual risk, by window",
    yaxis=dict(title="hedge ratio (beta)"),
    yaxis2=dict(title="residual sigma, USD/t", overlaying="y", side="right"),
    height=430, margin=dict(t=50, b=20, l=10, r=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
show(beta_fig)
st.dataframe(
    table.rename(columns={
        "beta": "hedge ratio", "residual_sigma": "residual sigma (USD/t)",
        "r_squared": "R²", "n": "n obs",
    }).style.format({
        "hedge ratio": "{:+.3f}", "residual sigma (USD/t)": "{:.2f}",
        "R²": "{:.1%}", "n obs": "{:.0f}",
    }),
    width="stretch",
)
finding(drift.headline)

# ===========================================================================
# S5 — what any of it actually hedges
# ===========================================================================
section(
    "S5",
    "Even at its best, crude only ever explained half the daily risk",
    "The ratio above is the best-fit slope in each window — but a slope can be well "
    "estimated and still not explain much of the variance it is fitted to. The R² "
    "answers a different question from the beta: not what ratio to hedge at, but how "
    "much of the risk hedging at any ratio actually removes.\n\n"
    f"Across the full sample, crude alone explains **{eff.table.loc['crude only','r_squared']:.1%}** "
    "of the daily variance in VLSFO. Adding gasoil — the middle-distillate benchmark, "
    "and in principle a closer cousin to a refined bunker fuel — takes it to "
    f"**{eff.best_r2:.1%}**. Neither instrument, alone or combined, gets past a third.",
    formula="R² of d(VLSFO) explained by each instrument set (in-sample, F-H4)",
)

eff_fig = go.Figure()
eff_fig.add_trace(
    go.Bar(
        x=list(eff.table.index), y=eff.table["r_squared"],
        marker_color=["rgba(120,120,120,0.6)", "rgba(120,120,120,0.6)", "crimson"],
        text=[f"{v:.1%}" for v in eff.table["r_squared"]], textposition="outside",
    )
)
eff_fig.update_layout(
    title="Share of daily bunker variance explained, by instrument set",
    yaxis_title="R²", yaxis_tickformat=".0%",
    height=380, margin=dict(t=50, b=20, l=10, r=10), showlegend=False,
)
show(eff_fig)
scope_note(
    "These R² are in-sample: the ratio is fitted on the same window it is scored on, "
    "which is the generous case. A hedger estimates on the past and carries the ratio "
    "into the future, which is strictly worse — so this section is an upper bound on "
    "what the instruments do, not an estimate of what a live hedge achieves."
)
finding(eff.headline)

# ===========================================================================
# S6
# ===========================================================================
section(
    "S6",
    "Why the ratio would drift rather than hold",
    "A single crude benchmark prices one barrel of a globally fungible commodity. "
    "VLSFO is a blend specification, met by refiners combining whatever streams clear "
    "that spec most cheaply in a given month — low-sulphur straight-run residue, "
    "cracked components, blending cutter stock. The mix refineries use to hit the "
    "specification is not fixed, and it does not have to move with crude.\n\n"
    "That gives the ratio a structural reason to drift rather than break: as the "
    "marginal blending component shifts — refinery-by-refinery, region-by-region, "
    "and with whatever regulation is currently tightening the spec — the fraction of "
    "VLSFO's price that crude actually explains shifts with it, gradually and without "
    "a single date attached to the change.\n\n"
    "A more complete hedge would need the actual blending components, not the crude "
    "price at the top of the barrel — and identifying which ones are marginal, "
    "region by region, is exactly the kind of thing this page cannot infer from price "
    "series alone.",
)

# ===========================================================================
# Diagnostic
# ===========================================================================
diagnostic_note(
    "The series is labelled VLSFO for its full history, but the 0.5% sulphur grade it "
    "names did not exist as a bunker fuel before the January 2020 cap (F-H2). What the "
    "pre-2020 window actually tracks — a predecessor grade, a proxy, a relabelled "
    "series — is not established here, and the drift measured across it is reported as "
    "a property of the quoted series rather than as a claim about the fuel itself."
)

mail_question(
    "The crude hedge ratio on VLSFO Singapore has fallen from about 0.8 in 2016-2017 to "
    "about 0.2 now, and even in the best window crude only explained roughly half the "
    "daily variance it was hedging. How do you size a bunker hedge today — do you still "
    "use crude as the primary instrument, or has the desk moved toward gasoil, a "
    "blended proxy, or something else entirely?",
    "Bunker purchasing and fuel risk desks (Cargill Fuel & Freight Risk, trading-house "
    "bunker desks at Vitol, Glencore, Trafigura; ship-operator bunker procurement)",
)
