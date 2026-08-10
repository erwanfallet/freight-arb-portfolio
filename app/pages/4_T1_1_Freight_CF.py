from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_APP_DIR.parent / "src"))
sys.path.insert(0, str(_APP_DIR))

from agri.chains.freight_cf import (  # noqa: E402
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

mail_question(
    f"On Santos → Qingdao, the published rate of {latest_rate:,.1f} USD/t implies a TCE "
    f"of {spread.tce_no_ballast.iloc[-1]:,.0f} USD/day if no ballast is charged at all, "
    f"against {spread.tce_full_ballast.iloc[-1]:,.0f} if it is charged in full — and the "
    f"first reading would place the market above its 2021 peak "
    f"{share_above_no_ballast:.0%} of the time. How much repositioning does your internal "
    "rate actually charge, and is it negotiated with the trading desk or imposed by the "
    "freight department?",
    "Freight desks (Cargill Ocean Transportation, Bunge, LDC, COFCO, Viterra) and grain/oilseed traders",
)
