from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_APP_DIR.parent / "src"))
sys.path.insert(0, str(_APP_DIR))

from freight.chains.grain_seasonal import (  # noqa: E402
    BRAZIL_HARVEST_MONTHS,
    BRAZIL_HARVEST_WIDE,
    NORTHERN_PEAK_MONTHS,
    effective_sample,
    harvest_footprint,
    load_panamax_frame,
    seasonal_profile,
    window_sensitivity,
)
from page_template import (  # noqa: E402
    ALT_COLOR,
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

st.set_page_config(page_title="D — Grain seasonality", layout="wide")

_LIVE = snapshot_banner()

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ===========================================================================
# Header and scope
# ===========================================================================
page_header(
    code="D",
    title="The largest cargo flow in the market leaves no mark on freight",
    subtitle=(
        "Brazilian soybean exports grew roughly sixfold in 25 years and the Panamax "
        "seasonal nearly tripled — but not in the harvest months, which sit at the "
        "year's own median in every period tested"
    ),
    scope=Scope(
        unit_trap=(
            "The Baltic Panamax Index is quoted in **index points**, not in money. It is "
            "a weighted average of four timecharter routes, each in USD/day, divided by "
            "a fixed constant — and converting it back into a cost per tonne requires the "
            "5TC in USD/day, which this export does not contain. Every number on this "
            "page is therefore a **ratio to the index's own annual level**, which is "
            "unit-free and survives the missing divisor. A page that quoted this seasonal "
            "in dollars would be inventing the conversion."
        ),
        conversion=(
            "rel_t      = BPI_t / median(BPI over its calendar year)\n"
            "seasonal_m = median over years of rel, for calendar month m\n"
            "amplitude  = max(seasonal) / min(seasonal) - 1\n"
            "\n"
            "  the annual median divides out the freight cycle, which is an order of\n"
            "  magnitude larger than the seasonal being measured"
        ),
        proxies=[
            "Brazilian export volumes: **absent from the export**, so the flow's growth "
            "is taken from the public record (roughly sixfold since 1999) and used as "
            "context, never as a regressor — nothing on this page is fitted to it",
            "the BPI as a stand-in for the Atlantic grain trade: it is a global Panamax "
            "index covering both basins, which dilutes a single-basin flow (D-H4)",
        ],
        out_of_scope=[
            "the P8 Santos-Qingdao route itself: the export is **missing 2023 and 2024 "
            "entirely**, leaving three seasons of which one is 2022 — too few, and too "
            "distorted by the war, to carry a seasonal (see the diagnostic below)",
            "harvest timing year by year: planting and weather shift the window by weeks, "
            "and no calendar of actual loading dates is available here",
        ],
        frequency_note=(
            "The index is daily, but **the seasonal's sample is annual**: one observation "
            "per calendar month per year, 27 years. Every interval below is built on that "
            "count, not on the 6,738 daily prints."
        ),
        data_warnings=[
            "The P8 route series has a two-year hole (no 2023, no 2024) in the Bloomberg "
            "export. It is why this test runs on the index rather than on the route.",
        ],
    ),
)

# ===========================================================================
# Data
# ===========================================================================
frame = load_panamax_frame()
profile = seasonal_profile(frame)
footprint = harvest_footprint(frame)
sample = effective_sample(frame)
sensitivity = window_sensitivity(frame)

harvest_level = profile.level(BRAZIL_HARVEST_MONTHS)

kpi_banner(
    {
        "Years in sample": f"{profile.n_years}",
        "Seasonal amplitude": f"{profile.amplitude:.0%}",
        "Trough / peak": f"{MONTH_LABELS[profile.trough_month - 1]} / {MONTH_LABELS[profile.peak_month - 1]}",
        "Harvest months": f"{harvest_level:.3f} of annual",
        "Amplitude growth": f"{footprint.amplitude_growth:.1f}x",
    }
)

# ===========================================================================
# S1
# ===========================================================================
section(
    "S1",
    "The obvious story, and why it is the one worth testing",
    "Brazil ships roughly a hundred million tonnes of soybeans a year, most of it to "
    "China, most of it between March and June, and almost all of it on Panamaxes. It is "
    "the largest single-origin grain flow in the world and its calendar is agronomic — "
    "published years in advance, and near-identical every season. If any cargo flow "
    "anywhere ought to leave a seasonal footprint on a freight index, it is this one.\n\n"
    "That expectation is also how the trade gets described: the harvest arrives, tonnage "
    "gets absorbed, rates firm. The claim is specific enough to check, and the index has "
    "twenty-seven complete years — long enough that the answer is not a sample accident. "
    "What follows is the check, and it does not come out the way the story predicts.",
)

# ===========================================================================
# S2
# ===========================================================================
section(
    "S2",
    "Two things have to be fixed before the question can be asked at all",
    "**The index is not a price.** The BPI is published in points: a weighted average of "
    "four timecharter routes in USD/day, divided by a constant fixed at the index's "
    "inception. Without the 5TC in USD/day — absent here — no amount of arithmetic turns "
    "a point into a dollar. Everything below is therefore a ratio to the index's own "
    "annual median, which needs no divisor and is unaffected by the missing one.\n\n"
    "**And the cycle is bigger than the seasonal.** The annual level of this index moves "
    f"by a factor of {frame['annual_level'].max() / frame['annual_level'].min():.0f} "
    "between its weakest and strongest year. A raw monthly average across 27 years would "
    "mostly be a report on which years happened to be strong, not on which months are. "
    "Dividing by the annual median removes the cycle and leaves the shape.\n\n"
    "**The sample is years, not days.** A calendar month contributes about twenty-one "
    "daily prints, but they are one continuous autocorrelated stretch, not twenty-one "
    "independent draws. The honest count is one observation per month per year — 27, not "
    "6,738. Every interval on this page is built on that count.",
    formula="rel_t = BPI_t / median(BPI over its calendar year)",
)
scope_note(
    f"Using the daily count would divide every standard error by "
    f"{sample['overstatement_factor']:.1f}. That is the difference between a chart where "
    "every month is significant and a chart that says something. It is applied here, not "
    "mentioned."
)

# ===========================================================================
# S3 — THE RESULT
# ===========================================================================
section(
    "S3",
    "The harvest months are not weak — they are invisible",
    "The profile below is the index's median level in each calendar month, relative to "
    "its own year, with intervals built on the 27 annual observations. Two months are "
    "genuinely distinguishable from the annual level: **February sits below it and "
    "October above it**, and neither interval touches 1.00.\n\n"
    "The four months that carry the Brazilian export peak — March, April, May, June — "
    "all contain 1.00 inside their interval. They are not a trough and not a peak. To the "
    "precision twenty-seven years can support, the largest grain flow in the world does "
    "not move the freight market it travels on.\n\n"
    "The peak is in autumn and the trough in February, both outside the Brazilian "
    "window. Whatever this index's seasonal is made of, it is not this harvest.",
)

fig = go.Figure()
colours = [
    ALT_COLOR.replace("0.18", "0.85") if m in BRAZIL_HARVEST_MONTHS else "rgba(120,120,120,0.55)"
    for m in range(1, 13)
]
fig.add_trace(
    go.Bar(
        x=MONTH_LABELS,
        y=profile.profile["median"],
        marker_color=colours,
        error_y=dict(
            type="data",
            symmetric=False,
            array=(profile.profile["hi"] - profile.profile["median"]).to_numpy(),
            arrayminus=(profile.profile["median"] - profile.profile["lo"]).to_numpy(),
            color="rgba(200,200,200,0.8)",
        ),
        name="median level",
    )
)
fig.add_hline(
    y=1.0, line_dash="dash", line_color="gray",
    annotation_text="the year's own median", annotation_position="right",
)
fig.update_layout(
    title="Panamax index by calendar month, relative to its own annual level (1999-2026)",
    yaxis_title="ratio to the annual median",
    height=430,
    margin=dict(t=50, b=20, l=10, r=10),
    showlegend=False,
    yaxis=dict(range=[0.6, 1.35]),
)
show(fig)
scope_note(
    "Coloured bars are the Brazilian harvest months. Error bars are the 95% interval on "
    "27 annual observations — not on daily prints. A bar whose interval crosses the "
    "dashed line is indistinguishable from an ordinary month."
)
finding(profile.headline)

# ===========================================================================
# S4 — THE IDENTIFICATION
# ===========================================================================
section(
    "S4",
    "The flow grew sixfold. The seasonal grew too — somewhere else",
    "A single profile could hide a lot: perhaps the harvest lifts the market but the "
    "counterfactual month would have been weaker still. That objection is real and it "
    "cannot be settled by looking at levels, because the counterfactual is not "
    "observable.\n\n"
    "It can be settled by looking at **change**. Brazilian exports grew roughly sixfold "
    "between 1999 and today. If the freight seasonal were a demand seasonal, the harvest "
    "months would have risen as the flow grew. The table below splits the 27 years into "
    "four periods and asks exactly that.\n\n"
    f"The amplitude nearly tripled — {footprint.amplitude_growth:.1f}x. The harvest months "
    f"moved by {footprint.harvest_drift:+.3f} of the annual level, which is nothing. "
    "Whatever tripled the seasonal, it was not the cargo that grew sixfold. This "
    "comparison is a difference, so it survives not knowing the counterfactual.",
    formula="amplitude(period) = max(seasonal) / min(seasonal) - 1     harvest = mean(seasonal over Mar-Jun)",
)

display = footprint.table.copy()
display.index = [f"{a}-{b}" for a, b in display.index]
display = display.rename(
    columns={
        "n_years": "years",
        "amplitude": "amplitude",
        "harvest_level": "harvest (Mar-Jun)",
        "northern_level": "autumn (Sep-Dec)",
        "trough_month": "trough",
        "peak_month": "peak",
    }
)
display["amplitude"] = display["amplitude"].map(lambda v: f"{v:.1%}")
display["harvest (Mar-Jun)"] = display["harvest (Mar-Jun)"].map(lambda v: f"{v:.3f}")
display["autumn (Sep-Dec)"] = display["autumn (Sep-Dec)"].map(lambda v: f"{v:.3f}")
display["trough"] = display["trough"].map(lambda m: MONTH_LABELS[int(m) - 1])
display["peak"] = display["peak"].map(lambda m: MONTH_LABELS[int(m) - 1])
st.dataframe(display, width="stretch")

growth = go.Figure()
periods = [f"{a}-{b}" for a, b in footprint.table.index]
growth.add_trace(
    go.Bar(
        x=periods, y=footprint.table["amplitude"],
        name="seasonal amplitude", marker_color="rgba(120,120,120,0.55)",
    )
)
growth.add_trace(
    go.Scatter(
        x=periods, y=footprint.table["harvest_level"] - 1.0,
        name="harvest months, deviation from annual level",
        mode="lines+markers", line=dict(color="crimson", width=3),
    )
)
growth.add_hline(y=0, line_dash="dash", line_color="gray")
growth.update_layout(
    title="The amplitude tripled; the harvest months did not move",
    yaxis_title="fraction of the annual level",
    height=400, margin=dict(t=50, b=20, l=10, r=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
show(growth)
finding(footprint.headline)

# ===========================================================================
# S5 — robustness
# ===========================================================================
section(
    "S5",
    "The result does not depend on where the harvest window is drawn",
    "A negative result that holds only for one definition of \"the harvest\" is a result "
    "about the definition. Loading starts in February and tails into July, so the test is "
    "re-run on the wide window as well as the four-month core. Both give the same verdict, "
    "and both stay flat across the four periods.",
)
wide_display = sensitivity.copy()
wide_display.index = [f"{a}-{b}" for a, b in wide_display.index]
wide_display = wide_display.rename(
    columns={
        "harvest_level_narrow": "Mar-Jun",
        "harvest_level_wide": "Feb-Jul",
        "amplitude": "amplitude",
    }
)
wide_display["amplitude"] = wide_display["amplitude"].map(lambda v: f"{v:.1%}")
wide_display["Mar-Jun"] = wide_display["Mar-Jun"].map(lambda v: f"{v:.3f}")
wide_display["Feb-Jul"] = wide_display["Feb-Jul"].map(lambda v: f"{v:.3f}")
st.dataframe(wide_display, width="stretch")

# ===========================================================================
# S6 — the reading
# ===========================================================================
section(
    "S6",
    "A freight seasonal is a positioning seasonal, not a demand one",
    "The reading that fits both facts is that a freight rate does not respond to demand, "
    "it responds to **unanticipated** demand. The Brazilian calendar is the most "
    "predictable large flow in dry bulk: everyone knows when it starts, roughly how big it "
    "is, and where it goes. Ships ballast toward the Atlantic ahead of it. A shock that "
    "the whole market positions for in advance has already been priced by the time the "
    "first cargo loads.\n\n"
    "That also explains the shape that *is* there. February is Chinese New Year on top of "
    "the northern winter — a demand hole nobody can position into, because there is "
    "nothing to position toward. Autumn stacks the North American harvest against "
    "pre-winter restocking, two flows whose size is genuinely uncertain until late in the "
    "season.\n\n"
    "If that is right, the growth of a flow tells you nothing about its effect on freight. "
    "What matters is how much of it was a surprise — and Brazil, by now, is never a "
    "surprise.",
)
finding(
    "The testable version of that reading: multiplying an anticipated flow by six should "
    f"change nothing, and it changed {footprint.harvest_drift:+.3f} of the annual level. "
    "The alternative — that supply simply grew in step — would need the Panamax fleet to "
    "have grown sixfold too, which it did not."
)

# ===========================================================================
# Diagnostic
# ===========================================================================
diagnostic_note(
    "Data defect that redirected this page. The P8 Santos-Qingdao route — the obvious "
    "series for a Brazilian grain seasonal, and the one this project started on — is "
    "**missing 2023 and 2024 entirely** in the export: 37 prints in 2021, 269 in 2022, "
    "nothing for two years, then 283 in 2025 and 186 in 2026. That leaves three seasons, "
    "one of which is the war year. The seasonal was moved onto the index, which has 27 "
    "complete years, at the cost of basin dilution. Recovering the route-level test needs "
    "one thing: the missing two years of P8."
)

mail_question(
    "Brazilian exports went up roughly sixfold since 1999 and I cannot find any footprint "
    "in the Panamax seasonal — the harvest months sit at the year's own median in every "
    "sub-period I test, while the amplitude itself nearly tripled somewhere else. Is that "
    "because you pre-position the fleet months ahead of the programme, or because the BPI "
    "is simply too global to show a single-basin flow? And when the harvest runs early or "
    "late, does that move your freight, or only your laycans?",
    "Grain chartering desks (Cargill Ocean Transportation, Bunge, LDC, COFCO, Viterra), Panamax owners and operators",
)
