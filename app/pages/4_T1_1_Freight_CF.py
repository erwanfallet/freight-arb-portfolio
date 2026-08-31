from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_APP_DIR.parent / "src"))
sys.path.insert(0, str(_APP_DIR))

from agri.chains.freight_cf import (  # noqa: E402
    ballast_lower_bound,
    ballast_value_by_route,
    ballast_value_usd_t,
    smoothing_bias,
    load_real_route_frame,
    market_implied_ballast_share,
)
from agri.core.voyage import ROUTES, VESSELS, VoyageParams, voyage_freight_usd_t  # noqa: E402
from agri.data.bloomberg_loader import DEFAULT_PATH  # noqa: E402
from agri.data.snapshot import series_or_live  # noqa: E402
from page_template import (
    snapshot_banner,  # noqa: E402
    Scope,
    diagnostic_note,
    finding,
    kpi_banner,
    mail_question,
    page_header,
    regime_chart,
    scope_note,
    section,
    show,
)

st.set_page_config(page_title="T1-1 — Freight in the C&F", layout="wide")

_LIVE = snapshot_banner()

st.sidebar.markdown("### Voyage assumptions")
speed_laden = st.sidebar.slider("Laden speed (knots)", 10.0, 15.0, 12.5, 0.5)
port_days = st.sidebar.slider("Port days", 2.0, 12.0, 6.0, 0.5)
mgo_premium = st.sidebar.slider("MGO premium over VLSFO", 1.0, 1.8, 1.35, 0.05)

params = VoyageParams(speed_laden_kn=speed_laden, port_days=port_days)
spread = load_real_route_frame(params=params, mgo_premium=mgo_premium)
tce_2021 = series_or_live("t1_1_tce_2021", "p8_route_tce_2021")
boom_peak = float(tce_2021.max())

# --- the trade -----------------------------------------------------------------
_vessel = VESSELS["panamax"]
_route = ROUTES["santos_qingdao"]
_vlsfo = series_or_live("t1_1_vlsfo", "vlsfo_singapore")
ballast_usd_t = ballast_value_usd_t(
    _vlsfo,
    reference_tce_usd_day=float(spread.tce_full_ballast.median()),
    vessel=_vessel, route=_route, params=params, mgo_premium=mgo_premium,
)
bound = ballast_lower_bound(
    spread.route_rate_usd_t, _vlsfo,
    vessel=_vessel, route=_route, params=params,
    ceiling_usd_day=boom_peak, mgo_premium=mgo_premium,
)
_cargo_t = _vessel.cargo_t
_by_route = ballast_value_by_route(
    float(_vlsfo.median()),
    reference_tce_usd_day=float(spread.tce_full_ballast.median()),
    vessel=_vessel,
    routes={k: ROUTES[k] for k in ("pnw_qingdao", "usgulf_qingdao", "santos_qingdao")},
    params=params, mgo_premium=mgo_premium,
)
_bias = smoothing_bias(spread.route_rate_usd_t)

page_header(
    code="T1-1",
    title="Freight inside the C&F calculation",
    subtitle=(
        "The same published route rate implies two TCEs 36,000 USD/day apart depending "
        "on whether ballast repositioning is charged — and the data settles it"
    ),
    scope=Scope(
        unit_trap=(
            "A freight rate is quoted either in **USD/day** (time-charter equivalent) or "
            "in **USD/tonne** on a route. Getting from one to the other is not a "
            "conversion but a **voyage estimate** — the factor depends on distance, "
            "speed, bunkers, and above all the share of ballast repositioning charged to "
            "the voyage. That factor is not a physical constant: it is the disagreement "
            "itself."
        ),
        conversion=(
            "freight_usd_t = [ TCE x (D_laden + ballast_share x D_ballast + D_port)\n"
            "                  + sea_bunkers + port_bunkers + port_costs + canal ]\n"
            "                / ( cargo_t x (1 - commission) )"
        ),
        proxies=[
            "MGO reconstructed from Singapore VLSFO (parameterised premium) — the export "
            "has no separate MGO series; the item weighs a few percent of the voyage",
            "distances and port costs: coded orders of magnitude, to re-verify before "
            "any binding use",
        ],
        out_of_scope=[
            "the arb's price legs (FOB Santos, CIF China): absent from the export, so "
            "the page covers the freight term alone, not the full arb",
            "demurrage and laytime — omitting them **understates** the full cost, so the "
            "bias runs against the thesis",
        ],
        frequency_note=(
            "P8 route rate and Singapore VLSFO are both daily, strict intersection: no "
            "forward-fill, since the bunker price on the fixture date is precisely what "
            "the disagreement is about."
        ),
        data_warnings=[
            "The export's P8 cell contains TWO unit regimes — see S2. Only the USD/tonne "
            "segment feeds the calculations; the USD/day segment is used as a test bed "
            "in S4.",
            "A freight print of zero on 30/04/2022 is excluded: physically impossible.",
        ],
    ),
)

kpi_banner(
    {
        "Published rate (USD/t)": f"{spread.route_rate_usd_t.iloc[-1]:,.1f}",
        "TCE if ballast = 0": f"{spread.tce_no_ballast.iloc[-1]:,.0f} USD/d",
        "TCE if ballast = 1": f"{spread.tce_full_ballast.iloc[-1]:,.0f} USD/d",
        "Gap": f"{spread.spread.iloc[-1]:,.0f} USD/d",
        "Real 2021 TCE peak": f"{boom_peak:,.0f} USD/d",
    }
)

# ===========================================================================
section(
    "S2",
    "The data defect — and it is the page's subject",
    "The export's P8 cell contains **two unit regimes in the same series**. From July to "
    "October 2021 it quotes between 24,500 and 38,000: these are **USD/day**, a Panamax "
    "TCE at the peak of the dry bulk boom. From 18 November 2021 onward it quotes between "
    "36 and 85: these are **USD/tonne**, a voyage rate. The two segments share no common "
    "date and are separated by a 19-day gap, which rules out calibrating the conversion "
    "factor on the junction — the market moved between the two. The fact that the source "
    "itself mixes USD/day and USD/tonne is not a convenient coincidence: it is the "
    "confusion this page measures, made visible in the data before any calculation even "
    "starts.",
)
diagnostic_note(
    f"TCE segment: {tce_2021.index.min():%d %b %Y} → {tce_2021.index.max():%d %b %Y}, "
    f"{len(tce_2021)} prints, {tce_2021.min():,.0f}–{tce_2021.max():,.0f} USD/day. "
    f"Voyage-rate segment: {spread.route_rate_usd_t.index.min():%d %b %Y} → "
    f"{spread.route_rate_usd_t.index.max():%d %b %Y}, {len(spread.route_rate_usd_t)} prints, "
    f"{spread.route_rate_usd_t.min():,.1f}–{spread.route_rate_usd_t.max():,.1f} USD/t. "
    "Each segment is coherent within its own unit; it is their juxtaposition that is not."
)
show(
    regime_chart(
        spread.route_rate_usd_t.to_frame("published rate"), "published rate",
        title="P8 Santos → Qingdao, USD/tonne segment only",
        y_title="USD/t", zero_line=False,
    )
)

# ===========================================================================
section(
    "S3",
    "The same print, two readings",
    "A trading desk reads \"55 USD/t\" and concludes the market pays 55 USD/t. A freight "
    "department reads the same print and asks: over how many days? If the voyage includes "
    "ballast repositioning, the same revenue has to cover nearly twice as many days, so "
    "the TCE it implies is nearly half as large. This is not a disagreement about the "
    "market — both are reading the **same** number — it is a disagreement about what that "
    "number pays for.",
    formula="TCE = ( freight_usd_t x paying_cargo − voyage costs ) / cycle_days",
)
finding(spread.headline)
show(
    regime_chart(
        pd.DataFrame(
            {"TCE ballast = 0": spread.tce_no_ballast, "TCE ballast = 1": spread.tce_full_ballast}
        ).assign(dummy=False),
        "TCE ballast = 0", title="", y_title="", zero_line=False,
    )
    .add_scatter(
        x=spread.tce_full_ballast.index, y=spread.tce_full_ballast.values,
        name="TCE ballast = 1", mode="lines",
    )
    .update_layout(
        title="TCE implied by the published rate, by ballast convention",
        yaxis_title="USD/day", showlegend=True,
    )
    .add_hline(
        y=boom_peak, line_dash="dash", line_color="crimson",
        annotation_text=f"real 2021 TCE peak: {boom_peak:,.0f} USD/d",
        annotation_position="top left",
    )
)

# ===========================================================================
section(
    "S4",
    "What the data settles",
    "The disagreement looks undecidable in theory. It is not here, because the 2021 "
    "segment — the one that had to be isolated as a data defect in S2 — gives the TCE "
    "**actually quoted on this route**, at the peak of the dry bulk boom. It therefore "
    "supplies a plausibility ceiling, and it is enough to settle between the two readings.",
)
share_above_no_ballast = float((spread.tce_no_ballast > boom_peak).mean())
share_above_full = float((spread.tce_full_ballast > boom_peak).mean())

c1, c2, c3 = st.columns(3)
c1.metric("Real TCE peak (Jul-Oct 2021)", f"{boom_peak:,.0f} USD/d")
c2.metric("No-ballast reading above the peak", f"{share_above_no_ballast:.0%} of the time")
c3.metric("Full-ballast reading above the peak", f"{share_above_full:.0%} of the time")

finding(
    f"Reading the published rate **without charging ballast** implies a TCE above the "
    f"peak of the dry bulk boom {share_above_no_ballast:.0%} of the time over five years "
    "— which would mean the market spent five years above its own all-time high. The same "
    f"inversion **charging ballast** exceeds that peak only {share_above_full:.0%} of the "
    "time. The trading desk's reading is not merely debatable: it is arithmetically "
    "untenable."
)
scope_note(
    "This is not an argument from authority in favour of the freight department: it is a "
    "plausibility bound drawn from a price actually quoted on the same route, and it says "
    "nothing about the exact ballast share — only that it is not zero."
)

# ===========================================================================
section(
    "S5",
    "The ballast share the market is pricing",
    "Once it is accepted that ballast is charged, the question is how much. Positing a "
    "reference TCE — the level at which owners are actually chartering — the published "
    "route rate reveals the share of repositioning the market has already priced in. The "
    "question to the desk then stops being \"who is right\" and becomes \"the market "
    "prices X%, is that what you charge internally?\".",
    formula="solve for ballast_share such that  model(reference_TCE, ballast_share) = published_rate",
)
reference_tce = st.slider(
    "Reference TCE (USD/day)", 8_000, 40_000, 18_000, 500,
    help="The level at which a Panamax owner is actually chartering, to compare against "
         f"the 2021 peak of {boom_peak:,.0f}",
)
latest_rate = float(spread.route_rate_usd_t.iloc[-1])
latest_vlsfo = float(series_or_live("t1_1_vlsfo", "vlsfo_singapore").reindex(spread.route_rate_usd_t.index).ffill().iloc[-1])

implied = market_implied_ballast_share(
    latest_rate, float(reference_tce), latest_vlsfo, latest_vlsfo * mgo_premium,
    vessel=VESSELS["panamax"], route=ROUTES["santos_qingdao"], params=params,
)
finding(implied.headline)

grid = pd.DataFrame(
    {
        "ballast_share": [i / 20 for i in range(21)],
    }
)
grid["modelled freight"] = [
    voyage_freight_usd_t(
        float(reference_tce), latest_vlsfo, latest_vlsfo * mgo_premium,
        vessel=VESSELS["panamax"], route=ROUTES["santos_qingdao"],
        params=params.with_ballast(share),
    ).freight_usd_t
    for share in grid["ballast_share"]
]
show(
    regime_chart(
        grid.set_index("ballast_share"), "modelled freight",
        title=f"Modelled freight by ballast share, at a TCE of {reference_tce:,.0f} USD/day",
        y_title="USD/t", zero_line=False,
        reference_lines={f"published rate {latest_rate:,.1f} USD/t": latest_rate},
    )
)

# ===========================================================================
# T1 — WHAT THE ARGUMENT COSTS, IN THE UNIT THE DECISION IS TAKEN IN
# ===========================================================================
section(
    "T1",
    "The dispute is conducted in USD per day; the cargo is decided in USD per tonne",
    "The gap above is real but unusable: a grain desk does not mark an arb in USD per "
    "day. Restating the same disagreement per tonne of cargo is what turns an "
    "inter-departmental argument into a number that changes whether a ship loads.\n\n"
    f"On Santos-Qingdao the ballast leg is worth a median **{ballast_usd_t.median():.1f} "
    f"USD per tonne of grain** — the difference between charging none of the repositioning "
    f"and charging all of it. On a {int(_cargo_t):,} tonne Panamax that is "
    f"**{ballast_usd_t.median() * _cargo_t / 1e6:.2f} million USD per voyage**.\n\n"
    "And it barely moves. Across seventeen years of bunker prices the figure ranges "
    f"{ballast_usd_t.min():.1f} to {ballast_usd_t.max():.1f} USD/t, with annual medians "
    "in a narrow band — because ballast is mostly **time**, not fuel. A desk hoping the "
    "argument will resolve itself when bunkers fall is waiting for something that does "
    "not happen: this is a structural cost, not a cyclical one.",
    formula="ballast value = freight(ballast = 1) - freight(ballast = 0),  at the same TCE",
)
_v1, _v2, _v3 = st.columns(3)
_v1.metric("Ballast, per tonne of grain", f"{ballast_usd_t.median():.1f} USD/t")
_v2.metric("Per Panamax voyage", f"{ballast_usd_t.median() * _cargo_t / 1e6:.2f} M USD")
_v3.metric(
    "Range over 17 years",
    f"{ballast_usd_t.min():.0f} - {ballast_usd_t.max():.0f} USD/t",
    delta="structural, not cyclical", delta_color="off",
)
_bal_annual = ballast_usd_t.groupby(ballast_usd_t.index.year).median()
_bal_fig = go.Figure()
_bal_fig.add_trace(
    go.Bar(x=_bal_annual.index.astype(str), y=_bal_annual.values,
           marker_color="rgba(120,120,120,0.65)")
)
_bal_fig.update_layout(
    title="What the ballast leg is worth, per tonne of grain (annual median)",
    yaxis_title="USD per tonne", height=360,
    margin=dict(t=50, b=20, l=10, r=10), showlegend=False,
)
show(_bal_fig)

# ===========================================================================
# T2 — WHAT THE MARKET SETTLES, AND WHAT IT LEAVES OPEN
# ===========================================================================
section(
    "T2",
    "The market refutes one desk and does not vindicate the other",
    "The plausibility ceiling above rules out reading the rate at zero ballast. But it is "
    "a one-sided test, and running it properly means asking not *whether* zero works but "
    "**how little ballast would be enough** to keep the implied TCE plausible — date by "
    "date, on the real published rate and real bunkers.\n\n"
    f"Zero is tenable on **{bound.share_where_zero_works:.0%} of days**. That settles the "
    "trading desk's position: it is not a defensible convention, it is an arithmetic "
    f"impossibility. But the binding lower bound has a median of only "
    f"**{bound.median_bound:.2f}**. At least half the ballast is required on "
    f"{bound.share_needing_at_least(0.5):.0%} of days, at least eighty percent on "
    f"{bound.share_needing_at_least(0.8):.0%}.\n\n"
    "So the freight department's full-cost convention is far stronger than anything the "
    "market imposes. **Between roughly a quarter and all of it, no price in this export "
    "decides.** That range is not a measurement problem to be narrowed with better data — "
    "it is the space in which an internal transfer rate is negotiated, and it is genuinely "
    "empty of market information.",
    formula="min ballast share such that implied TCE <= the route's own recorded peak",
)
_b1, _b2, _b3 = st.columns(3)
_b1.metric("Days where zero ballast works", f"{bound.share_where_zero_works:.0%}",
           delta="trading desk refuted", delta_color="off")
_b2.metric("Median required ballast", f"{bound.median_bound:.2f}")
_b3.metric("Days needing 80% or more", f"{bound.share_needing_at_least(0.8):.0%}")

_bound_fig = go.Figure()
_bound_fig.add_trace(
    go.Scatter(x=bound.shares.index, y=bound.shares.values, mode="markers",
               marker=dict(size=3, color="rgba(120,120,120,0.55)"), name="daily minimum")
)
_bound_fig.add_hline(y=bound.median_bound, line_dash="dash", line_color="crimson",
                     annotation_text=f"median {bound.median_bound:.2f}", annotation_position="right")
_bound_fig.add_hline(y=1.0, line_dash="dot", line_color="gray",
                     annotation_text="freight department claims this", annotation_position="right")
_bound_fig.update_layout(
    title="Minimum ballast share the published rate is consistent with",
    yaxis_title="ballast share", yaxis_range=[-0.05, 1.1], height=400,
    margin=dict(t=50, b=20, l=10, r=10), showlegend=False,
)
show(_bound_fig)
finding(bound.headline)
scope_note(
    f"On {bound.n_impossible} days out of {bound.n_obs + bound.n_impossible} the rate is "
    "so high that even charging every mile of repositioning still implies a TCE above the "
    "route's own recorded peak. Those are not errors — they are the days the market was "
    "paying more than any voyage economics can rationalise, and they are excluded rather "
    "than forced into the bound."
)

# ===========================================================================
# T3 — WHY THIS IS A VOLUME POLICY, NOT AN ACCOUNTING CHOICE
# ===========================================================================
section(
    "T3",
    "The internal rate decides how much grain moves, and nobody owns that",
    "Put the two numbers together. The ballast is worth about "
    f"{ballast_usd_t.median():.0f} USD/t, and the market leaves its allocation "
    "undetermined across most of the range. So the internal rate is set by negotiation "
    "between two profit centres — and whichever number they land on does not merely "
    "split the P&L.\n\n"
    "**It decides which cargoes exist.** On the marginal band — the days an arb sits a few "
    f"dollars from breakeven — a {ballast_usd_t.median():.0f} USD/t swing in the freight "
    "term is the difference between a cargo that clears and one that does not. Charge the "
    "full ballast and the trading desk sees fewer open arbs and moves less grain. Charge "
    "the index and it moves more, with the repositioning cost landing in the freight "
    "book.\n\n"
    "That is a volume policy expressed as an accounting convention. The trading desk owns "
    "the tonnage target, the freight department owns the vessel cost, and the parameter "
    "that reconciles them sits between the two mandates — which is precisely why the "
    "argument recurs rather than resolving.",
)

# ===========================================================================
# T4 — THE SAME CONVENTION IS NOT WORTH THE SAME THING EVERYWHERE
# ===========================================================================
section(
    "T4",
    "One ballast policy, three different taxes",
    "The ballast is a **time** cost, so what it is worth per tonne scales with the length "
    "of the haul. Applying one internal convention across a whole programme therefore "
    "applies a different effective charge to each origin — and the spread is not "
    "marginal.\n\n"
    f"Out of the Pacific Northwest the ballast is worth "
    f"**{_by_route.iloc[0]['ballast_usd_t']:.1f} USD/t**. Out of Santos it is "
    f"**{_by_route.iloc[-1]['ballast_usd_t']:.1f}** — "
    f"{_by_route.iloc[-1]['ballast_usd_t'] / _by_route.iloc[0]['ballast_usd_t']:.1f} times "
    "more, on the same policy, for the same house.\n\n"
    "So a full-ballast convention does not merely make freight dearer; it makes "
    "**long-haul origination dearer relative to short-haul**, by a margin large enough to "
    "move an origin decision on the marginal band. That is an origin-selection distortion "
    "produced by an accounting choice, and it is invisible to whoever picks the origin.",
    formula="ballast value scales with laden distance, because it is hire, not fuel",
)
st.dataframe(
    _by_route.rename(columns={"laden_nm": "laden nm", "ballast_usd_t": "ballast (USD/t)"})
    .style.format({"laden nm": "{:,.0f}", "ballast (USD/t)": "{:.1f}"}),
    width="stretch",
)

# ===========================================================================
# T5 — THE SLEEPER: A STABLE RATE IS A BIASED RATE
# ===========================================================================
section(
    "T5",
    "A freight department has to quote a stable rate, and that puts it on one side of the market for months",
    "This is the term nobody argues about, and it is larger than the one they do.\n\n"
    "A trading desk cannot budget against a number that moves every day, so the internal "
    f"rate is smoothed — here a {_bias.window}-session average. That is treated as a "
    "presentational convenience. It is not: a smoothed series does not sit around the "
    "market, it sits **on one side of it**, and it stays there while the market trends.\n\n"
    f"Measured on the real published rate, within contiguous segments: the internal number "
    f"is a median **{_bias.median_abs_error:.1f} USD/t** from spot and "
    f"**{_bias.p90_abs_error:.1f}** at the 90th percentile. It holds the same side for up "
    f"to **{_bias.longest_episode} consecutive sessions**, and "
    f"**{_bias.share_in_episode:.0%}** of the sample sits inside such an episode.\n\n"
    "The direction is what makes it a trading problem rather than a reporting one. While "
    "the market rallies, the smoothed rate lags below it: the desk is quoted freight that "
    "is too cheap, every arb looks better than it is, and it over-commits. While the market "
    "falls, the rate lags above: freight looks dear and the desk under-commits — precisely "
    "when physical tonnage is cheapest and most available. **The convention systematically "
    "buys at the wrong point of the cycle**, and it does so for months at a time.",
    formula="internal rate = rolling mean of spot, computed inside contiguous segments only",
)
_e1, _e2, _e3 = st.columns(3)
_e1.metric("Median distance from spot", f"{_bias.median_abs_error:.1f} USD/t")
_e2.metric("Longest one-sided run", f"{_bias.longest_episode} sessions")
_e3.metric("Sample inside an episode", f"{_bias.share_in_episode:.0%}")

_ep = _bias.episodes.copy()
_ep["start"] = _ep["start"].dt.date.astype(str)
_ep["end"] = _ep["end"].dt.date.astype(str)
st.dataframe(
    _ep.rename(columns={
        "n_obs": "sessions", "mean_error": "mean error (USD/t)", "direction": "direction",
    }).style.format({"mean error (USD/t)": "{:+.1f}"}),
    width="stretch", hide_index=True,
)
_bias_fig = go.Figure()
_bias_fig.add_trace(
    go.Scatter(x=_bias.error.index, y=_bias.error.values, mode="lines",
               line=dict(color="crimson", width=1), name="internal minus spot")
)
_bias_fig.add_hline(y=0.0, line_dash="dash", line_color="gray")
_bias_fig.update_layout(
    title="Internal smoothed rate minus spot (below zero = freight quoted too cheap)",
    yaxis_title="USD per tonne", height=380,
    margin=dict(t=50, b=20, l=10, r=10), showlegend=False,
)
show(_bias_fig)
finding(_bias.headline)
scope_note(
    "Computed inside contiguous segments only. This export's P8 series has a 782-day hole "
    "between November 2022 and January 2025, and a rolling mean run straight through it "
    "would average 2022 prices into a 2025 rate — manufacturing a single 221-session "
    "episode that never happened. Segments shorter than the smoothing window are dropped "
    "rather than padded."
)

mail_question(
    "Three numbers from the real P8 rate that I think belong together.\n\n"
    f"First, the ballast is worth about {ballast_usd_t.median():.0f} USD per tonne of "
    "grain and it is stable across seventeen years, because it is time rather than fuel. "
    f"Second, testing both conventions against the route's own recorded TCE peak, reading "
    f"the rate at zero ballast is tenable on only {bound.share_where_zero_works:.0%} of "
    f"days — but the binding lower bound is a median {bound.median_bound:.2f}, not 1. The "
    "market refutes the trading desk and does not vindicate the freight department; "
    "everything between is negotiated, not priced.\n\n"
    "Third, and this is the one I did not expect to be the largest: the internal rate has "
    f"to be smoothed to be usable, and a {_bias.window}-session average sits on ONE side of "
    f"spot for up to {_bias.longest_episode} consecutive sessions, "
    f"{_bias.share_in_episode:.0%} of the time. In a rally the desk is quoted freight that "
    "is too cheap and over-commits; in a fall it is quoted freight that is too dear and "
    "under-commits, exactly when tonnage is most available.\n\n"
    "And the ballast is worth "
    f"{_by_route.iloc[-1]['ballast_usd_t'] / _by_route.iloc[0]['ballast_usd_t']:.1f} times "
    "more out of Santos than out of the PNW, so a single fleet-wide policy quietly taxes "
    "long-haul origination.\n\n"
    "Is any of that owned explicitly on your side — is the smoothing window set against a "
    "volume or utilisation target, and is the origin distortion something the chartering "
    "desk corrects for — or does it all sit between the two mandates as P&L attribution?",
    "Freight desks (Cargill Ocean Transportation, Bunge, LDC, COFCO, Viterra) and grain/oilseed traders",
)
