from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_APP_DIR.parent / "src"))
sys.path.insert(0, str(_APP_DIR))

from agri.data.bloomberg_loader import DEFAULT_PATH  # noqa: E402
from freight.chains.products import (  # noqa: E402
    DENSITY_HEAVY,
    DENSITY_LIGHT,
    DENSITY_TYPICAL,
    breakeven_density,
    breakeven_density_series,
    density_identification,
    gallons_per_tonne,
    load_real_transatlantic_frame,
    transatlantic_spread,
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

st.set_page_config(page_title="C — Transatlantic distillate", layout="wide")

_LIVE = snapshot_banner()

# ===========================================================================
# Header and scope
# ===========================================================================
page_header(
    code="C",
    title="The transatlantic distillate arb is a unit conversion before it is a trade",
    subtitle=(
        "A density nobody quotes accounts for 91 % of the variability of the arb it is used "
        "to compute — and more than half the time, it alone decides whether the cargo works"
    ),
    scope=Scope(
        unit_trap=(
            "The US leg is quoted in **cents per gallon** — a **volume**. The European leg "
            "is quoted in **USD per tonne** — a **mass**. The only thing standing between "
            "them is a density, and no exchange publishes it next to the price. It is not a "
            "constant of nature either: it depends on the grade, the additive package and "
            "the reference temperature. Middle distillates run roughly "
            f"{DENSITY_LIGHT:.3f} to {DENSITY_HEAVY:.3f} kg/l, and that band is the whole "
            "subject of this page."
        ),
        conversion=(
            "gallons_per_tonne(rho) = 1000 / rho / 3.785412\n"
            "spread_usd_t = ULSD_c_gal / 100 x gallons_per_tonne(rho) - gasoil_usd_t\n"
            "\n"
            f"rho = {DENSITY_LIGHT:.3f}  ->  {gallons_per_tonne(DENSITY_LIGHT):.1f} gal/t\n"
            f"rho = {DENSITY_HEAVY:.3f}  ->  {gallons_per_tonne(DENSITY_HEAVY):.1f} gal/t"
        ),
        proxies=[
            "freight: a parameter, not a series. The Baltic clean tanker legs in the export "
            "are not identified with enough confidence to price a specific route",
            "no spec bridge, no financing cost: both are real terms of a physical arb and "
            "both are left out rather than guessed",
        ],
        out_of_scope=[
            "Worldscale flat-rate mechanics, which are modelled elsewhere in this engine on "
            "synthetic data — the January reset is real but it is not what decides this page",
            "quality differentials between USGC diesel and ARA gasoil specifications, which "
            "would shift the level and are deliberately not invented",
        ],
        frequency_note="Both legs are daily front-month futures, 2 917 common sessions since 2015.",
        data_warnings=[
            "Both legs are front-month futures, not physical assessments. A physical arb is "
            "struck on cargo differentials against these screens; the page measures the "
            "screen spread and says so.",
        ],
    ),
)

# ===========================================================================
# Parameters
# ===========================================================================
st.sidebar.markdown("### Parameters")
freight = st.sidebar.slider("Freight, USGC to ARA (USD/t)", 0.0, 80.0, 25.0, 1.0)
density = st.sidebar.slider(
    "Density assumption (kg/l)", DENSITY_LIGHT, DENSITY_HEAVY, DENSITY_TYPICAL, 0.001,
    help="The parameter that carries the sign. No exchange publishes it.",
)
window_start = st.sidebar.selectbox("Window", ["2015-01-01", "2020-01-01", "2022-01-01"], index=0)

frame = load_real_transatlantic_frame(window_start)
spread = transatlantic_spread(frame, density_kg_l=density)
identification = density_identification(frame)
breakeven = breakeven_density(frame, freight_usd_t=freight)
breakeven_path = breakeven_density_series(frame, freight_usd_t=freight)

kpi_banner(
    {
        "ULSD (last)": f"{frame['ulsd_c_gal'].iloc[-1]:.1f} c/gal",
        "ICE gasoil (last)": f"{frame['gasoil_usd_t'].iloc[-1]:.1f} USD/t",
        "Spread at your density": f"{spread.iloc[-1]:+.1f} USD/t",
        "Density swing": f"{identification.swing_usd_t:.1f} USD/t",
        "Share of arb variability": f"{identification.ratio:.0%}",
    }
)

# ===========================================================================
# S1
# ===========================================================================
section(
    "S1",
    "Two prices that cannot be subtracted",
    "The transatlantic diesel arb is discussed as a single spread — buy the US barrel, ship "
    "it to Rotterdam, sell it against ARA gasoil — as though the two legs were "
    "commensurable. They are not. NYMEX ULSD is quoted in cents per **gallon**, which is a "
    "volume. ICE gasoil is quoted in dollars per **tonne**, which is a mass. Subtracting "
    "one from the other requires a density.\n\n"
    "That density is not published anywhere near the price, and it is not a constant. It "
    "depends on the grade, on the additive package, and on the temperature the volume is "
    "referenced to. Real middle distillates occupy a band of roughly "
    f"{DENSITY_LIGHT:.3f} to {DENSITY_HEAVY:.3f} kg per litre — a spread of about five per "
    "cent in volume terms.\n\n"
    "Five per cent sounds like a rounding detail. On a leg worth several hundred dollars a "
    "tonne, it is not.",
)

# ===========================================================================
# S2 — the measurement
# ===========================================================================
section(
    "S2",
    "The conversion factor is as large as the thing it measures",
    "The question is not whether the density swing is big in absolute terms. It is whether "
    "it is big **next to the variability of the arb it is used to compute** — because that "
    "ratio decides whether the level of the arb means anything at all.\n\n"
    "On eleven years of real prices, moving the density across its plausible band moves the "
    f"arb by {identification.swing_usd_t:.1f} USD per tonne. The arb's own standard "
    f"deviation over the same window is {identification.spread_std_usd_t:.1f} USD per tonne. "
    "The two are the same size.\n\n"
    "The consequence is the same one that recurs across this portfolio: **the level is not "
    "identifiable, the sign and the changes are.** Anyone quoting a transatlantic arb of "
    "twenty-seven dollars a tonne without stating their density is quoting a number that "
    "another desk, using an equally defensible density, would put at forty-four or at "
    "eighteen.",
    formula="swing = spread(rho_light) - spread(rho_heavy)     compared with     std(spread)",
)
finding(identification.headline)

comparison = pd.DataFrame(
    {
        "density (kg/l)": [DENSITY_LIGHT, DENSITY_TYPICAL, DENSITY_HEAVY],
        "gallons per tonne": [
            gallons_per_tonne(DENSITY_LIGHT),
            gallons_per_tonne(DENSITY_TYPICAL),
            gallons_per_tonne(DENSITY_HEAVY),
        ],
        "median spread (USD/t)": [
            float(transatlantic_spread(frame, density_kg_l=d).median())
            for d in (DENSITY_LIGHT, DENSITY_TYPICAL, DENSITY_HEAVY)
        ],
        "share of days above freight": [
            float((transatlantic_spread(frame, density_kg_l=d) > freight).mean())
            for d in (DENSITY_LIGHT, DENSITY_TYPICAL, DENSITY_HEAVY)
        ],
    }
)
st.dataframe(
    comparison.style.format(
        {
            "density (kg/l)": "{:.3f}", "gallons per tonne": "{:.1f}",
            "median spread (USD/t)": "{:+.1f}", "share of days above freight": "{:.0%}",
        }
    ),
    width="stretch", hide_index=True,
)
scope_note(
    f"Read the last column: at a freight cost of {freight:.0f} USD/t, the share of sessions "
    "on which this trade looks economic moves by tens of percentage points depending on a "
    "parameter that is not a market price."
)

# ===========================================================================
# S3 — THE DELIVERABLE
# ===========================================================================
section(
    "S3",
    "The inversion: which density makes the cargo marginal",
    "Rather than picking a density and announcing whether the arb is open, ask the question "
    "the other way round — **at what density does this cargo exactly break even against "
    "freight?** That has a closed form, and the form is informative in itself: the breakeven "
    "density is inversely proportional to the European price plus freight, which is why it "
    "tightens precisely when freight is expensive.\n\n"
    "The test then writes itself. If the breakeven density falls **inside** the plausible "
    "band, the decision to load turns on the grade of the specific parcel rather than on the "
    "market. If it falls outside, the arb is unambiguous and the conversion argument, while "
    "still true about the magnitude, does not change the decision.",
    formula="rho* = ULSD_c_gal / 100 x 1000 / 3.785412 / (gasoil_usd_t + freight)",
)
finding(breakeven.headline)

c1, c2, c3 = st.columns(3)
c1.metric("Breakeven density", f"{breakeven.density_star:.4f} kg/l")
c2.metric("Plausible band", f"{DENSITY_LIGHT:.3f} – {DENSITY_HEAVY:.3f}")
c3.metric(
    "Inside the band",
    "yes" if breakeven.inside_plausible_band else "no",
    delta="the grade decides" if breakeven.inside_plausible_band else "the market decides",
    delta_color="off",
)

show(
    regime_chart(
        breakeven_path,
        "density_star",
        regime_col="inside_band",
        regime_color=ALT_COLOR,
        title=f"Breakeven density at {freight:.0f} USD/t of freight — shaded where the grade decides",
        y_title="kg per litre",
        zero_line=False,
        reference_lines={
            f"heavy {DENSITY_HEAVY:.3f}": DENSITY_HEAVY,
            f"light {DENSITY_LIGHT:.3f}": DENSITY_LIGHT,
        },
        annotations={"2022-02-24": "Russian diesel"},
    )
)
finding(
    f"The breakeven density sits inside the plausible band "
    f"{breakeven_path['inside_band'].mean():.0%} of the sample and above it "
    f"{breakeven_path['above_band'].mean():.0%}. On more than half the sessions, whether "
    "this trade is economic is settled by which grade is in the tank rather than by either "
    "screen price."
)

# ===========================================================================
# S4
# ===========================================================================
section(
    "S4",
    "What survives the uncertainty, and what does not",
    "It would be too easy to conclude that nothing can be said. Two things survive.\n\n"
    "**The sign, most of the time.** When the breakeven density sits above the heavy end of "
    "the band, no defensible grade closes the arb — it is open regardless of what one "
    "assumes. That is the case on a substantial share of the sample, and those are the "
    "periods a trader can act on without arguing about specifications.\n\n"
    "**The changes, always.** The density enters multiplicatively and moves every date in "
    "the same direction, so comparisons across time are robust even when levels are not. "
    "The widening after February 2022 is real on any density in the band.\n\n"
    "What does not survive is the number itself. A transatlantic arb quoted to the dollar, "
    "with no density stated, is a number two desks will disagree about while both being "
    "right.",
)
show(
    regime_chart(
        spread.to_frame("spread").assign(covers_freight=spread > freight),
        "spread",
        regime_col="covers_freight",
        regime_color=SHUT_COLOR,
        title=f"Transatlantic spread at rho = {density:.3f} kg/l — shaded where it covers freight",
        y_title="USD per tonne",
        reference_lines={f"freight {freight:.0f}": freight},
        annotations={"2022-02-24": "Russian diesel"},
    )
)

# ===========================================================================
# S5
# ===========================================================================
section(
    "S5",
    "What this page does not claim",
    "The two legs are **front-month futures**, not physical assessments. A real cargo is "
    "struck on differentials against these screens, and those differentials carry the "
    "quality bridge between USGC diesel and ARA gasoil specifications — a term this page "
    "deliberately does not invent.\n\n"
    "Freight is a slider, not a series. The Baltic clean tanker legs present in the export "
    "are not identified with enough confidence to price a specific USGC–ARA route, and "
    "attaching a number to the wrong route would be worse than exposing the assumption.\n\n"
    "Neither omission touches the result. The density argument is about the **conversion "
    "between the two legs**, and it holds whatever the level of the differentials or the "
    "freight — those terms shift the breakeven, they do not narrow the band.",
)
diagnostic_note(
    "One thing worth checking before using any of this on a live cargo: the ICE gasoil "
    "contract specification changed over the history of this series, and the export gives "
    "no contract metadata. The page treats the series as homogeneous, which is an "
    "assumption it cannot verify from the data it has."
)

mail_question(
    f"Converting NYMEX ULSD to a tonne basis, I find that moving the density across the "
    f"plausible distillate band changes the transatlantic arb by "
    f"{identification.swing_usd_t:.0f} USD/t — about the same as the arb's own standard "
    f"deviation — and that at {freight:.0f} USD/t of freight the breakeven density lands "
    f"inside that band on {breakeven_path['inside_band'].mean():.0%} of sessions. Which "
    "density does your desk use, and is it a house convention or re-derived per cargo "
    "grade? I would expect the answer to change the arb by more than most people assume.",
    "Refined product desks (Vitol, Trafigura, Gunvor, Mercuria, BP, Shell Trading), USGC "
    "and NWE distillate traders",
)
