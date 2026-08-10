from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_APP_DIR.parent / "src"))
sys.path.insert(0, str(_APP_DIR))

from agri.chains.sugar_mix import (  # noqa: E402
    CENTS_LB_TO_USD_T,
    CZARNIKOW_COST_BRL_T,
    DEFAULT_POL_FACTOR,
    floor_variance_decomposition,
    indifference_hydrous_brl_l,
    load_real_parity_frame,
    moving_floor,
    production_cost_check,
)
from agri.core.fmt import fmt_num, fmt_pct  # noqa: E402
from agri.data.bloomberg_loader import DEFAULT_PATH  # noqa: E402
from page_template import (
    snapshot_banner,  # noqa: E402
    SHUT_COLOR,
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

st.set_page_config(page_title="T3-2 — Sugar: the moving floor", layout="wide")

_LIVE = snapshot_banner()

# ===========================================================================
# Header and scope
# ===========================================================================
page_header(
    code="T3-2",
    title="The \"Brazilian cost floor\" is an exchange-rate series",
    subtitle=(
        "A production cost is denominated in reais; translated into cents per "
        "pound for a New York reader, it starts moving by twenty cents with no "
        "Brazilian cost having changed"
    ),
    scope=Scope(
        unit_trap=(
            "NY11 quotes in **cents per pound**, in USD. A Brazilian production "
            "cost is denominated in **BRL per tonne**, because that is the "
            "currency costs are actually paid in. Getting from one to the other "
            "needs two conversions — pound to tonne, then USD to BRL — and the "
            "second is not a constant: it is a market price that moves every day. "
            "A \"structural level\" expressed in the wrong currency stops being "
            "structural."
        ),
        conversion=(
            "sugar_BRL_t   = NY11_c_lb x 22.0462 x USDBRL\n"
            "floor_c_lb    = cost_BRL_t / (22.0462 x USDBRL)\n"
            "hydrous*_BRL_l = NY11_c_lb x pol_factor x USDBRL x 2.20462 x (ATR_h / ATR_s) / 100"
        ),
        proxies=[
            "cost of production: the BRL 2,000/t figure comes from Czarnikow "
            "(Jun 2026), it is **sourced and dated** — but regional and "
            "time-varying, hence the slider",
            "pol_factor: adjusts NY11 (96° pol) toward the VHP quality actually produced",
        ],
        out_of_scope=[
            "the mix's conditional elasticity, the project's original thesis: it "
            "needs the UNICA mix by fortnight and region plus CEPEA ethanol, "
            "**neither of which is in the export** — see S5, where the "
            "specification is left as-is rather than simulated",
            "inland transport costs and distance to port, part of the mill's "
            "programme but not of the price identity tested here",
        ],
        frequency_note=(
            "NY11 and USDBRL are both daily. UNICA publishes by fortnight — which "
            "is why the mix portion of S5 would need a frequency change, not just a data one."
        ),
        data_warnings=[
            "The Consecana coefficients are **revised every season** (G-H1). "
            "Freezing them would corrupt the whole history, and the error would be "
            "silent since parity would stay plausible.",
        ],
    ),
)

# ===========================================================================
# Parameters
# ===========================================================================
st.sidebar.markdown("### Parameters")
cost_brl_t = st.sidebar.slider(
    "Cost of production (BRL/tonne)", 1200.0, 3000.0, CZARNIKOW_COST_BRL_T, 50.0,
    help="Czarnikow, June 2026: pricing opportunities stayed below BRL 2,000/t.",
)
pol_factor = st.sidebar.slider("pol_factor (NY11 → VHP)", 0.94, 1.00, DEFAULT_POL_FACTOR, 0.005)
window_start = st.sidebar.selectbox(
    "Window", ["2015-01-01", "2010-01-01", "2000-01-01"], index=0
)

frame = load_real_parity_frame(window_start)
cost_check = production_cost_check(frame, cost_brl_t=cost_brl_t)
floor = moving_floor(frame, cost_brl_t=cost_brl_t)
decomposition = floor_variance_decomposition(frame)
hydrous = indifference_hydrous_brl_l(frame["ny11"], frame["usdbrl"], pol_factor=pol_factor)

kpi_banner(
    {
        "NY11 (last)": f"{fmt_num(frame['ny11'].iloc[-1], 2)} c/lb",
        "USDBRL": fmt_num(frame["usdbrl"].iloc[-1], 3),
        "Sugar in BRL": f"{fmt_num(cost_check.last_brl_t, 0)} BRL/t",
        "Implied floor": f"{fmt_num(floor.floor_last, 1)} c/lb",
        "Floor range": f"{fmt_num(floor.floor_range, 1)} c/lb",
    }
)

# ===========================================================================
# S1
# ===========================================================================
section(
    "S1",
    "Two houses, a dated disagreement, and what they agree on",
    "**Hedgepoint** (February 2026) sees a center-south crop of 630 Mt and a mix "
    "that should fall towards 46% to materially cut the surplus — but will not get "
    "there, because **mill limits and sugar already sold forward** prevent it. "
    "**Czarnikow** (June 2026) observes the opposite on the constraint side: mills "
    "enter the season **far less hedged** than the previous four, and 2026/27 "
    "pricing opportunities stayed **below BRL 2,000/t, i.e. below the cost of "
    "production**.\n\n"
    "Hedgepoint points to constraints; Czarnikow observes that the unlock comes "
    "precisely from an unusually low hedge level. But both houses agree on one "
    "point: **the degree of prior hedging is the discriminating variable**. It is "
    "observable, and it is rarely modelled.\n\n"
    "This page does not settle between them. It does two more useful things: it "
    "checks Czarnikow's quantified claim against real prices, and it shows that "
    "the usual way this number gets translated for a New York reader is misleading.",
)

# ===========================================================================
# S2
# ===========================================================================
section(
    "S2",
    "Czarnikow's claim, checked against prices",
    "\"Pricing stayed below the cost of production\" is a verifiable claim, and "
    "checking it beats quoting it. NY11 and USDBRL are enough: sugar in reais per "
    "tonne is the product of the two, up to the pound-tonne conversion.",
    formula="sugar_BRL_t = NY11_c_lb x 22.0462 x USDBRL",
)
finding(cost_check.headline)
show(
    regime_chart(
        floor.frame.assign(below_cost=floor.frame["sugar_brl_t"] < cost_brl_t),
        "sugar_brl_t",
        regime_col="below_cost",
        regime_color=SHUT_COLOR,
        title="Sugar expressed in reais per tonne",
        y_title="BRL/tonne",
        zero_line=False,
        reference_lines={f"cost {fmt_num(cost_brl_t, 0)} BRL/t": cost_brl_t},
        annotations={"2026-01-01": "Czarnikow window"},
    )
)
scope_note(
    "Shaded zones: the price is below the chosen cost of production. Czarnikow's "
    "claim concerns the 2026/27 campaign; the chart shows it sits within a wider "
    "regime, not a one-off accident."
)

# ===========================================================================
# S3 — THE RESULT
# ===========================================================================
section(
    "S3",
    "The same cost, translated into cents per pound, stops being a level",
    "A New York desk does not think in reais. The Brazilian cost is therefore "
    "reported to them in cents per pound — \"there's support near 18 cents, "
    "that's the Brazilian cost\" — and heard as a structural level, a floor the "
    "market would struggle to break durably.\n\n"
    "But this translation divides by the exchange rate. The cost in reais can "
    "stay **strictly constant**: its expression in cents moves by whatever "
    "USDBRL moves by. Over the period, the same cost of production produces a "
    f"floor ranging from {fmt_num(floor.floor_min, 1)} to {fmt_num(floor.floor_max, 1)} "
    "c/lb. This is not a floor that shifted because Brazilian costs changed — "
    "**no cost changed in this calculation, it is fixed by construction**. It is "
    "a floor that shifted because the real depreciated.",
    formula="floor_c_lb = cost_BRL_t / (22.0462 x USDBRL)",
)
finding(floor.headline)
show(
    regime_chart(
        floor.frame,
        "floor_c_lb",
        regime_col="below_floor",
        regime_color=SHUT_COLOR,
        title="NY11 floor implied by a CONSTANT cost of production in reais",
        y_title="cents per pound",
        zero_line=False,
        reference_lines={f"NY11 last: {fmt_num(frame['ny11'].iloc[-1], 1)}": float(frame["ny11"].iloc[-1])},
    )
)
scope_note(
    "Shaded zones: periods where NY11 was actually below the floor. The curve "
    "itself is **nothing but** the inverse of USDBRL, rescaled — that is exactly the point."
)

c1, c2, c3 = st.columns(3)
c1.metric("Lowest floor", f"{fmt_num(floor.floor_min, 1)} c/lb")
c2.metric("Highest floor", f"{fmt_num(floor.floor_max, 1)} c/lb")
c3.metric("Range", f"{fmt_num(floor.floor_range, 1)} c/lb", delta="FX only", delta_color="off")

# ===========================================================================
# S4
# ===========================================================================
section(
    "S4",
    "What FX does to the price itself, and why it is not symmetric",
    "One could object that FX also moves the sugar price in reais, so the two "
    "effects offset. They do not offset the same way, and the variance "
    "decomposition says so.\n\n"
    f"The movement of sugar in reais comes {fmt_pct(decomposition['share_sugar'])} "
    f"from the dollar price and {fmt_pct(decomposition['share_fx'])} from FX, with "
    f"a covariance of {fmt_pct(decomposition['share_covariance'])} — the "
    f"correlation between the two is {fmt_num(decomposition['correlation'], 3)}, "
    "hence **negative**: when the real depreciates, sugar in dollars tends to "
    "fall. FX therefore partially cushions the price the mill receives.\n\n"
    "But the **floor** benefits from no cushioning at all: it is exactly "
    "proportional to the inverse of FX, by construction. That is where the "
    "asymmetry lies — the price received is partly hedged by the correlation, "
    "the threshold it is compared to is not hedged at all.",
)
st.dataframe(
    pd.DataFrame(
        {
            "component": ["sugar price (USD)", "USDBRL exchange rate", "covariance (×2)"],
            "share of variance": [
                fmt_pct(decomposition["share_sugar"], 1),
                fmt_pct(decomposition["share_fx"], 1),
                fmt_pct(decomposition["share_covariance"], 1),
            ],
        }
    ),
    width="stretch", hide_index=True,
)

# ===========================================================================
# S5
# ===========================================================================
section(
    "S5",
    "The number a mill actually watches",
    "A Brazilian mill does not arbitrate against a cost of production: it "
    "arbitrates between making sugar and making ethanol, from the same cane. The "
    "quantity that drives this decision is the ethanol price at which it becomes "
    "indifferent — and it is derived from NY11 and FX alone, via the Consecana "
    "schedule, with no ethanol series needed.\n\n"
    "This is the inversion of the usual conversion chain: instead of starting "
    "from an ethanol price to compare it to sugar, it starts from sugar and asks "
    "at what ethanol price the mill stops making it.",
    formula="hydrous* = NY11 x pol_factor x USDBRL x 2.20462 x (ATR_hydrous / ATR_sugar) / 100",
)
finding(
    f"At the last print, a mill is indifferent between sugar and hydrous ethanol "
    f"at {fmt_num(hydrous.iloc[-1], 2)} BRL per litre. Over the period, this "
    f"threshold has ranged between {fmt_num(hydrous.min(), 2)} and "
    f"{fmt_num(hydrous.max(), 2)} BRL/l."
)
show(
    regime_chart(
        hydrous.to_frame("hydrous_brl_l"),
        "hydrous_brl_l",
        title="Hydrous ethanol price that makes the mill indifferent",
        y_title="BRL per litre",
        zero_line=False,
    )
)
scope_note(
    "Comparing it to the actual CEPEA price would give the parity gap, hence the "
    "mix signal. CEPEA is published free — the highest-value series to fetch for "
    "this project, and the only one missing to close the original thesis."
)

# ===========================================================================
# S6
# ===========================================================================
section(
    "S6",
    "What this page does not do, and what it would take to do it",
    "The project's original thesis is more ambitious than the above: it concerns "
    "the **mix's elasticity to parity, conditional on the entry-of-season hedge "
    "ratio** — precisely the variable Hedgepoint and Czarnikow converge on. It "
    "needs a panel estimate by UNICA fortnight and region.\n\n"
    "Two series are missing, and neither is paid: the **UNICA mix** by fortnight "
    "and region, and **CEPEA hydrous ethanol**. The specification is left below "
    "exactly as it would be estimated, rather than simulated on fabricated data "
    "— a coefficient estimated on a synthetic set teaches nobody anything, it "
    "only verifies that the code recovers what was put into it.",
    formula=(
        "dmix = a_r + b1 parity_gap_{t-1}\n"
        "            + b2 (parity_gap_{t-1} x entry_hedge_ratio_r)\n"
        "            + b3 capacity_utilisation\n"
        "            + b4 (port_distance_r x parity_gap_{t-1}) + e\n"
        "\n"
        "The object of interest is b2, never b1."
    ),
)
diagnostic_note(
    "The panel elasticity engine exists and is tested (`estimate_mix_elasticity`), "
    "but it runs on a synthetic set. It is deliberately not shown here: a page "
    "showing a fabricated b2 next to results measured on real data would invite "
    "reading both the same way."
)

mail_question(
    f"Translating a production cost of {fmt_num(cost_brl_t, 0)} BRL/t into cents "
    f"per pound gives a floor that has ranged from {fmt_num(floor.floor_min, 1)} "
    f"to {fmt_num(floor.floor_max, 1)} c/lb since {window_start[:4]} — "
    f"{fmt_num(floor.floor_range, 1)} cents of amplitude produced by USDBRL "
    "alone. Does your team reason on a floor in cents, or does it recompute it "
    "on every move in the real? And for 2026/27, at what entry-of-season hedge "
    "ratio did your mills actually start the season?",
    "Sugar desks at Sucden, Czarnikow, Alvean, Wilmar, ED&F Man; trading and origination at CS sugar groups",
)
