from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_APP_DIR.parent / "src"))
sys.path.insert(0, str(_APP_DIR))

from agri.data.bloomberg_loader import DEFAULT_PATH  # noqa: E402
from freight.chains.coal import (  # noqa: E402
    DEFAULT_COAL_EFFICIENCY,
    DEFAULT_GAS_EFFICIENCY,
    EF_COAL_T_PER_MWH_TH,
    EF_GAS_T_PER_MWH_TH,
    MWH_TH_PER_TONNE_COAL,
    efficiency_identification,
    generation_cost_eur_mwh_e,
    load_real_switching_frame,
    switching_carbon_price,
)
from page_template import (
    snapshot_banner,  # noqa: E402
    ALT_COLOR,
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

st.set_page_config(page_title="B — Coal-to-gas switching", layout="wide")

_LIVE = snapshot_banner()

# ===========================================================================
# Header and scope
# ===========================================================================
page_header(
    code="B",
    title="The coal-to-gas switching price is not a property of the fuels",
    subtitle=(
        "It is a property of two plant efficiencies that no exchange quotes — and they move "
        "the answer twice as much as the carbon price itself"
    ),
    scope=Scope(
        unit_trap=(
            "**Three units and two currencies, before any comparison exists.** API2 coal is "
            "quoted in **USD per tonne** — a mass, which only becomes energy through a "
            "calorific value of 6 000 kcal/kg. TTF gas is quoted in **EUR per MWh** — "
            "already energy. EUA carbon is quoted in **EUR per tonne of CO2** — which only "
            "enters through an emission factor. And the two fuels are burned in plants whose "
            "efficiencies differ by roughly a factor of one and a half, so nothing is "
            "comparable until everything is expressed per MWh of **electricity**."
        ),
        conversion=(
            f"coal_eur_mwh_th = API2_usd_t / EURUSD / {MWH_TH_PER_TONNE_COAL:.3f}\n"
            f"coal_eur_mwh_e  = (coal_eur_mwh_th + EUA x {EF_COAL_T_PER_MWH_TH}) / eta_coal\n"
            f"gas_eur_mwh_e   = (ttf_eur_mwh     + EUA x {EF_GAS_T_PER_MWH_TH}) / eta_gas\n"
            "EUA*            = (ttf/eta_gas - coal_th/eta_coal) / (EF_c/eta_c - EF_g/eta_g)"
        ),
        proxies=[
            "emission factors: standard combustion figures, not plant-specific — a real unit "
            "varies with coal rank and gas composition",
            "plant efficiencies: **the parameter this page is about**. Not market data, not "
            "published, and the thing the whole answer turns on",
        ],
        out_of_scope=[
            "start-up costs, minimum stable generation and ramp constraints, which decide "
            "whether a plant that is theoretically in the money actually runs",
            "the API2 − API4 arb this engine was originally built for: **API4 Richards Bay "
            "is absent from the export**, so that spread is simply not computable and is not "
            "faked with a proxy",
        ],
        frequency_note="All four legs are daily, 2 214 common sessions since January 2018.",
        data_warnings=[
            "The EURUSD quoting direction is checked at load time rather than assumed — a "
            "reversed FX leg would produce a coal price roughly 25 % off with no visible "
            "error, which is exactly the failure mode this portfolio is built to catch.",
        ],
    ),
)

# ===========================================================================
# Parameters
# ===========================================================================
st.sidebar.markdown("### Parameters")
coal_efficiency = st.sidebar.slider(
    "Coal plant efficiency", 0.34, 0.45, DEFAULT_COAL_EFFICIENCY, 0.005,
    help="Old subcritical around 0.36, supercritical around 0.42. Not published anywhere.",
)
gas_efficiency = st.sidebar.slider(
    "CCGT efficiency", 0.45, 0.63, DEFAULT_GAS_EFFICIENCY, 0.005,
    help="Early CCGT around 0.50, modern H-class around 0.60.",
)

frame = load_real_switching_frame()
costs = generation_cost_eur_mwh_e(
    frame, coal_efficiency=coal_efficiency, gas_efficiency=gas_efficiency
)
switch = switching_carbon_price(
    frame, coal_efficiency=coal_efficiency, gas_efficiency=gas_efficiency
)
identification = efficiency_identification(frame)

kpi_banner(
    {
        "API2 (last)": f"{frame['api2_usd_t'].iloc[-1]:.1f} USD/t",
        "TTF (last)": f"{frame['ttf_eur_mwh'].iloc[-1]:.1f} EUR/MWh",
        "EUA (last)": f"{frame['eua_eur_t'].iloc[-1]:.1f} EUR/t",
        "Switching EUA": f"{switch.iloc[-1]:.1f} EUR/t",
        "Efficiency swing": f"{identification.swing_eur_t:.0f} EUR/t",
    }
)

# ===========================================================================
# S1
# ===========================================================================
section(
    "S1",
    "Nothing here is comparable until everything is electricity",
    "A European generator with both a coal unit and a CCGT does not compare API2 to TTF. The "
    "two numbers are not commensurable, and not for a subtle reason: one is a price per "
    "**tonne of a solid**, the other a price per **MWh of energy**. Getting from the first "
    "to the second requires a calorific value — API2 is assessed at 6 000 kcal per kg, which "
    f"makes a tonne worth about {MWH_TH_PER_TONNE_COAL:.2f} thermal MWh — and a currency, "
    "because coal prices in dollars and everything else in this calculation prices in "
    "euros.\n\n"
    "Then carbon enters, quoted per **tonne of CO2**, which only becomes a cost per MWh "
    "through an emission factor. And finally both fuel and carbon have to be divided by a "
    "plant efficiency, because what the generator sells is electricity, not heat.\n\n"
    "That last division is where the interesting part hides. Dividing by efficiency scales "
    "**both** the fuel cost and the carbon cost — so a less efficient plant pays more for "
    "its CO2 per unit sold, not just more for its fuel. The carbon term is therefore larger "
    "than it looks, and it is larger by an amount that depends on a number nobody publishes.",
)

# ===========================================================================
# S2
# ===========================================================================
section(
    "S2",
    "The switching price, and what its denominator is made of",
    "Set the two generation costs equal and solve for the carbon price. The result is worth "
    "looking at closely, because the **denominator contains no price at all** — only two "
    "emission factors and two efficiencies.\n\n"
    "That is not a presentational curiosity. It means the sensitivity of the switching price "
    "to the efficiency assumption is not a second-order correction sitting somewhere in the "
    "error term: the efficiencies are in the denominator of the answer. Change them and the "
    "switching price moves, with no fuel price and no carbon price having moved at all.",
    formula="EUA* = (ttf / eta_gas - coal_th / eta_coal) / (EF_coal / eta_coal - EF_gas / eta_gas)",
)
show(
    regime_chart(
        pd.DataFrame({"eua_switch": switch, "eua_actual": frame["eua_eur_t"]}).assign(
            above=frame["eua_eur_t"] > switch
        ),
        "eua_switch",
        regime_col="above",
        regime_color=ALT_COLOR,
        title=f"Switching carbon price at {coal_efficiency:.0%} coal / {gas_efficiency:.0%} gas",
        y_title="EUR per tonne of CO2",
        reference_lines={"EUA today": float(frame["eua_eur_t"].iloc[-1])},
        annotations={"2021-10-01": "gas crisis", "2022-02-24": "Ukraine"},
    )
)
scope_note(
    f"Shaded where the actual carbon price sits **above** the switching level — the regime "
    f"in which carbon is doing the work of displacing coal. At these efficiencies that is "
    f"{(frame['eua_eur_t'] > switch).mean():.0%} of the sample. Hold that number: the next "
    "section moves it by a factor of three without touching a single price."
)

# ===========================================================================
# S3 — THE RESULT
# ===========================================================================
section(
    "S3",
    "The unpublished parameter is twice the size of the published one",
    "Coal plants in Europe run at anywhere from about 36 % efficiency for an old subcritical "
    "unit to about 42 % for a supercritical one. CCGTs run from about 50 % for an early "
    "machine to about 60 % for a modern H-class. Both ranges are ordinary, and which pair is "
    "**at the margin** on a given day is a question about the merit order, not about the "
    "fuels.\n\n"
    "Sweeping those pairs moves the switching price across a range wider than the carbon "
    "price's own variability — and it flips the qualitative conclusion entirely. Assume old "
    "coal against modern gas and carbon looks like it has been doing its job for most of the "
    "past eight years. Assume modern coal against old gas and it looks like it has barely "
    "worked at all.\n\n"
    "**Same fuel prices, same carbon price, opposite conclusions about whether the carbon "
    "market functions.**",
)
finding(identification.headline)

grid = identification.grid.copy()
grid.columns = [
    "coal efficiency", "gas efficiency", "switching EUA (EUR/t)", "share EUA above",
]
st.dataframe(
    grid.style.format(
        {
            "coal efficiency": "{:.0%}", "gas efficiency": "{:.0%}",
            "switching EUA (EUR/t)": "{:.1f}", "share EUA above": "{:.0%}",
        }
    ),
    width="stretch", hide_index=True,
)
diagnostic_note(
    "This is the same shape as several other projects in this portfolio, and the recurrence "
    "is the point: a parameter that is not a market price, is not published, and is quietly "
    "assumed, turns out to move the answer more than the market data does. Here the ratio is "
    f"{identification.ratio:.1f} to one."
)

# ===========================================================================
# S4
# ===========================================================================
section(
    "S4",
    "What the two costs actually did",
    "Underneath the parameter argument there is a real history, and it survives any "
    "efficiency pair in the plausible range because the pair shifts both curves rather than "
    "crossing them. Gas was structurally cheaper to burn than coal through the late 2010s, "
    "the 2021–22 gas crisis inverted that violently, and the relationship has been "
    "renegotiating itself since.\n\n"
    "What the page adds to that familiar story is the caveat that the **level** of the "
    "switching threshold is not knowable from prices — only its movements are.",
)
show(
    regime_chart(
        costs,
        "spread",
        regime_col="gas_cheaper",
        regime_color=SHUT_COLOR,
        title="Coal minus gas generation cost — shaded where gas is the cheaper plant",
        y_title="EUR per MWh of electricity",
        annotations={"2021-10-01": "gas crisis", "2022-02-24": "Ukraine"},
    )
)
c1, c2, c3 = st.columns(3)
c1.metric("Coal cost (median)", f"{costs['coal'].median():.1f} EUR/MWh")
c2.metric("Gas cost (median)", f"{costs['gas'].median():.1f} EUR/MWh")
c3.metric("Gas cheaper", f"{costs['gas_cheaper'].mean():.0%} of sessions")

# ===========================================================================
# S5
# ===========================================================================
section(
    "S5",
    "What this page is not, and the project it replaces",
    "This engine was originally built around the API2 − API4 arb — Rotterdam against "
    "Richards Bay — and the thesis that the marginal South African tonne stopped pricing off "
    "Europe after 2022. **API4 is not in the export**, so that spread is not computable. "
    "Rather than substitute a proxy and keep the original headline, the question was changed "
    "to one the available data can actually answer.\n\n"
    "What is deliberately missing here is the plant-level reality: start-up costs, minimum "
    "stable generation, ramp rates and must-run constraints all decide whether a unit that "
    "is theoretically in the money actually generates. A switching price is a necessary "
    "condition for a fuel switch, never a sufficient one — and no amount of price data fixes "
    "that.",
)

mail_question(
    "Computing the coal-to-gas switching carbon price from API2, TTF and EUA, I find that "
    f"the efficiency pair alone moves it by about {identification.swing_eur_t:.0f} EUR/t — "
    f"roughly {identification.ratio:.1f} times the standard deviation of the carbon price "
    f"itself — and that it changes the share of the sample where carbon displaces coal from "
    f"{identification.share_above_high:.0%} to {identification.share_above_low:.0%}. Which "
    "pair of efficiencies does your switching calculation actually use, and how often is it "
    "re-estimated against the units that are really at the margin rather than a fleet "
    "average?",
    "European power and gas desks (Uniper, RWE, EDF Trading, Vitol, Glencore coal), "
    "utility fuel procurement, carbon desks",
)
