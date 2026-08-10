from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_APP_DIR.parent / "src"))
sys.path.insert(0, str(_APP_DIR))

from agri.chains.white_premium import (  # noqa: E402
    POL_PLAUSIBLE_HI,
    POL_PLAUSIBLE_LO,
    WhitePremiumError,
    identification_check,
    implied_pol_adjust,
    implied_refining_cost,
    load_real_richness_frame,
    pol_adjust_sensitivity,
    summarise_richness,
)
from agri.data.bloomberg_loader import DEFAULT_PATH  # noqa: E402
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

st.set_page_config(page_title="T2-4 — White premium", layout="wide")

_LIVE = snapshot_banner()

# ===========================================================================
# Header and scope
# ===========================================================================
page_header(
    code="T2-4",
    title="The white premium, or what a price can and cannot tell you",
    subtitle=(
        "The level of the refining rent is not identifiable from prices — its "
        "variation fully is, and it shifted regime by about 60 USD/t"
    ),
    scope=Scope(
        unit_trap=(
            "No.11 quotes in **cents per pound, 96° pol basis**; No.5 in **USD per "
            "tonne** of refined sugar. Comparing them needs the c/lb → USD/t "
            "conversion **and** a polarisation adjustment, because it takes more than "
            "one tonne of raw at 96° to make one tonne of white. This second factor is "
            "**published by nobody**: it depends on the contract specification used "
            f"and runs between {POL_PLAUSIBLE_LO:.2f} and {POL_PLAUSIBLE_HI:.2f}. The "
            "whole page is about establishing what can be concluded despite this "
            "ignorance."
        ),
        conversion=(
            "white_premium = No5_usd_t - No11_c_lb x 22.0462 x pol_adjust\n"
            "richness      = white_premium - refining_cost\n"
            "                (> 0: white pays more than it costs to produce)"
        ),
        proxies=[
            "energy cost: real Henry Hub x a parameterised energy intensity "
            "(~8 mmBtu/t) — the price is real, the intensity is not",
            "labour, freight and yield loss: flat rates, no public refinery cost "
            "accounting exists as a time series",
        ],
        out_of_scope=[
            "the refining asset's cost of capital: richness is a contribution margin, "
            "never a profit — do not compare it to a ROIC (W-H4)",
            "quality premia and delivery constraints on No.5, which are part of the "
            "residual the page does not claim to decompose",
        ],
        frequency_note="No.11, No.5 and Henry Hub are all daily; so is richness.",
        data_warnings=[
            "The polarisation adjustment is the parameter that decides the sign. The "
            "page does not fix it: it first measures what its uncertainty costs (S3), "
            "and only concludes on what survives that (S4).",
        ],
    ),
)

# ===========================================================================
# Parameters
# ===========================================================================
st.sidebar.markdown("### Parameters")
pol_adjust = st.sidebar.slider(
    "Polarisation adjustment", 1.00, 1.20, 1.07, 0.005,
    help="Unobservable. Plausible range 1.06–1.08 depending on contract specification.",
)
energy_intensity = st.sidebar.slider(
    "Refining energy intensity (mmBtu/t)", 3.0, 15.0, 8.0, 0.5
)
window_start = st.sidebar.selectbox(
    "Window", ["2015-01-01", "2010-01-01", "1990-07-18"], index=0
)

common = dict(energy_intensity_mmbtu_t=energy_intensity)
frame = load_real_richness_frame(pol_adjust=pol_adjust, start=window_start, **common)
summary = summarise_richness(frame)
check = identification_check(start=window_start, pol_ref=pol_adjust, **common)
cost = implied_refining_cost(start=window_start, pol_adjust=pol_adjust, **common)

kpi_banner(
    {
        "White premium (median)": f"{frame['white_premium'].median():.0f} USD/t",
        "Modelled cost": f"{cost.modelled_usd_t:.0f} USD/t",
        "Median richness": f"{frame['richness'].median():+.1f} USD/t",
        "Parameter span": f"{check.parameter_span_max:.1f} USD/t",
        "Signal span": f"{check.signal_span:.1f} USD/t",
    }
)

# ===========================================================================
# S1
# ===========================================================================
section(
    "S1",
    "\"The white premium is the refining margin\" is a claim, not a definition",
    "The gap between white sugar No.5 and raw sugar No.11 is commonly called the "
    "refining margin, and treated as if it were one by construction. It is not. It is "
    "a gap between two futures contracts that differ by **four things at once**: "
    "degree of refining, quoting unit, polarisation basis, and delivery location and "
    "form. Calling this gap \"the refining margin\" amounts to claiming the first "
    "term dominates the other three.\n\n"
    "The claim may well be true. But it is testable, and the test immediately runs "
    "into an obstacle that is not economic: just to **write down** the gap requires a "
    "conversion factor nobody publishes.",
)

# ===========================================================================
# S2
# ===========================================================================
section(
    "S2",
    "The factor nobody publishes",
    "No.11 is delivered at 96° polarisation, No.5 is refined sugar. It therefore "
    "takes more than one tonne of raw to make one tonne of white, and the ratio "
    "depends on the exact contract specification — not on a physical law. The values "
    f"in circulation run from {POL_PLAUSIBLE_LO:.2f} to {POL_PLAUSIBLE_HI:.2f}.\n\n"
    "The gap between these two bounds looks harmless: two percentage points. But it "
    "multiplies the raw price once converted to USD/t, a quantity worth several "
    "hundred dollars. On the real sample, switching bound moves richness by about "
    f"{check.parameter_span_max:.0f} USD/t — against a median richness of "
    f"{frame['richness'].median():+.1f} USD/t. **The unobservable parameter is the "
    "same order of magnitude as the answer being sought.**",
    formula=f"pol_adjust ∈ [{POL_PLAUSIBLE_LO:.2f} ; {POL_PLAUSIBLE_HI:.2f}]  →  richness ± {check.parameter_span_max / 2:.1f} USD/t",
)
show(
    regime_chart(
        frame.assign(below_zero=frame["richness"] <= 0),
        "richness",
        regime_col="below_zero",
        regime_color=SHUT_COLOR,
        title=f"Refining richness at pol_adjust = {pol_adjust:.3f}",
        y_title="USD/tonne",
        annotations={"2020-03-01": "Covid", "2023-01-01": "?"},
    )
)
st.caption(summary.headline)

# ===========================================================================
# S3 — the problem, stated plainly
# ===========================================================================
section(
    "S3",
    "The level is not identifiable — and the inversion shows it cleanly",
    "Rather than picking a value and concluding, the question is turned around. "
    "Suppose refining is **competitive**: a mature industry does not sustain a "
    "durable median rent. Which polarisation adjustment does the market then price? "
    "That is a one-unknown equation, and its solution compares directly to the "
    "specification a refiner knows by heart.",
    formula="pol* such that median(richness) = 0",
)
try:
    implied = implied_pol_adjust(start=window_start, **common)
    finding(implied.headline)
    c1, c2, c3 = st.columns(3)
    c1.metric("Implied pol*", f"{implied.pol_star:.4f}")
    c2.metric("High plausible bound", f"{POL_PLAUSIBLE_HI:.2f}")
    c3.metric(
        "Gap",
        f"{implied.pol_star - POL_PLAUSIBLE_HI:+.4f}",
        delta="within range" if implied.within_plausible else "outside range",
        delta_color="off",
    )
    scope_note(
        "Two readings remain open and **price alone cannot settle between them**: "
        "either the plausible range is too narrow and refiners work with a higher "
        "adjustment, or refining genuinely captures a rent. This is exactly the kind "
        "of question an insider settles in one sentence and no amount of price data "
        "will."
    )
except WhitePremiumError as error:
    diagnostic_note(f"Inversion not possible on this window: {error}")

# ===========================================================================
# S4 — THE RESULT
# ===========================================================================
section(
    "S4",
    "But the variation is fully identifiable",
    "An unknown parameter does not make a page useless — it makes the conclusions "
    "that depend on it unusable, and only those. The polarisation adjustment "
    "multiplies the raw price, so it shifts every year's richness **in the same "
    "direction and by a comparable amount**. The gaps between years survive it.\n\n"
    "The table below gives median richness by year at both bounds of the plausible "
    "range and at the chosen value. Three checks decide whether reading the gaps is "
    "warranted: the parameter's amplitude compared to the signal's, the stability of "
    "the year ranking, and the number of years whose **sign** depends on the "
    "parameter. Those, and only those, remain non-interpretable.",
)
finding(check.headline)

annual = check.annual.copy()
annual.columns = [
    f"pol {check.pol_lo:.2f}", f"pol {check.pol_ref:.3f}", f"pol {check.pol_hi:.2f}"
]
annual["parameter span"] = (annual.iloc[:, 0] - annual.iloc[:, 2]).round(1)
st.dataframe(annual.round(1), width="stretch")

flipping = check.sign_flipping_years
if flipping:
    diagnostic_note(
        f"Non-interpretable year(s): {', '.join(map(str, flipping))} — their sign "
        "depends on the choice of pol_adjust, so nothing can be said about them. They "
        "are excluded from the reading rather than resolved by a convenient parameter "
        "choice."
    )
else:
    scope_note("No year changes sign across the plausible range.")

# ===========================================================================
# S5 — what the variation reveals
# ===========================================================================
reference = check.annual["richness_ref"]
worst_year, best_year = int(reference.idxmin()), int(reference.idxmax())
section(
    "S5",
    "What the variation reveals: a regime change, not an oscillation",
    "Once it is established that the gaps can be read, the chart is stark. Richness "
    "is **persistently negative from 2017 to 2021** — refining destroys value at the "
    f"quoted price, bottoming at {reference.min():+.0f} USD/t in {worst_year} — then "
    f"turns decisively positive from 2023 onward, reaching {reference.max():+.0f} "
    f"USD/t in {best_year}, and has stayed there.\n\n"
    f"The size of this swing, about {check.signal_span:.0f} USD/t, is **{check.ratio:.1f} "
    "times** what parameter uncertainty alone can produce. This is therefore not a "
    "convention artefact: something changed in the economics of refining between 2021 "
    "and 2023, and it stuck. The page measures this swing; it does not claim to "
    "explain it — export restrictions among major exporters, capacity closures, flow "
    "reallocation toward destination refiners are all hypotheses that price data does "
    "not adjudicate between.",
)
show(
    regime_chart(
        check.annual.assign(positive=check.annual["richness_ref"] > 0),
        "richness_ref",
        regime_col="positive",
        regime_color=ALT_COLOR,
        title="Median richness by year — the swing survives the choice of parameter",
        y_title="USD/tonne",
    )
)

# ===========================================================================
# S6 — the number for the email
# ===========================================================================
section(
    "S6",
    "The number to put in an email",
    "All the discussion so far concerns **richness**, which needs a cost model. The "
    "white premium itself needs none: it is the price the market puts on turning one "
    "tonne of raw into one tonne of white, observed, with no opex or energy "
    "assumption. It is therefore this number that gets shown to a refiner, not "
    "richness — they know their own cost and will do the subtraction themselves.",
)
finding(cost.headline)

st.markdown("**Sensitivity to the parameter, in one table**")
st.dataframe(
    pol_adjust_sensitivity(frame["no5"], frame["no11"]).round(2),
    width="stretch", hide_index=True,
)
scope_note(
    "Energy is the only cost component genuinely observed (Henry Hub); the "
    f"{energy_intensity:.1f} mmBtu/t intensity that converts it to USD/t is itself a "
    "parameter. Labour, freight and yield loss are flat rates — no public source "
    "gives them as a time series."
)

mail_question(
    f"On ICE No.5 against No.11 since {window_start[:4]}, I find the market pays "
    f"around {cost.market_usd_t:.0f} USD/t for the act of refining, and, more "
    f"notably, that this price shifted regime by about {check.signal_span:.0f} USD/t "
    f"between {worst_year} and {best_year} — a swing {check.ratio:.0f} times larger "
    "than the uncertainty on the polarisation adjustment, so not a convention "
    "artefact. Is your all-in refining cost of that order? And what changed on your "
    "side between those two dates?",
    "Destination refiners (Al Khaleej, ASR, Tereos, Südzucker), sugar desks at Sucden, Czarnikow, ED&F Man",
)
