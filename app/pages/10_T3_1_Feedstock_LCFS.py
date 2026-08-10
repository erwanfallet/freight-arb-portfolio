from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_APP_DIR.parent / "src"))
sys.path.insert(0, str(_APP_DIR))

from agri.chains.feedstock_lcfs import (  # noqa: E402
    CENTS_PER_USD,
    LCFS_PROGRAM_HIGH_USD_T,
    LCFS_PROGRAM_LOW_USD_T,
    SOYOIL_DOMESTIC,
    SRE_WARNING,
    Feedstock,
    calibration_gap_45z,
    crush_from_soyoil_lb,
    discount_burden,
    import_penalty,
    lcfs_neutral_price,
    load_soyoil_usd_lb,
    penalty_bounds,
    structural_exit,
    winner_grid,
)
from agri.data.bloomberg_loader import DEFAULT_PATH  # noqa: E402
from page_template import (
    snapshot_banner,  # noqa: E402
    ALT_COLOR,
    Scope,
    diagnostic_note,
    finding,
    kpi_banner,
    mail_question,
    page_header,
    regime_chart,
    scope_note,
    section,
    sensitivity_heatmap,
    show,
)

st.set_page_config(page_title="T3-1 — Feedstock LCFS", layout="wide")

_LIVE = snapshot_banner()

# ===========================================================================
# Header and scope
# ===========================================================================
page_header(
    code="T3-1",
    title="Two subsidies that contradict each other",
    subtitle=(
        "How many cents under soyoil does imported UCO have to sell for, purely "
        "to offset a tax credit it has no right to claim — and why the LCFS "
        "credit price cannot settle the debate"
    ),
    scope=Scope(
        unit_trap=(
            "Three units stack into a single margin, and **two are not price "
            "units**: fuel sells by the gallon, feedstock is bought by the pound "
            "via a yield of roughly 7.6 lb/gal, and the LCFS credit is quoted in "
            "USD per **tonne of CO2e** — it only becomes cents per gallon after "
            "passing through a carbon intensity in gCO2e/MJ and an energy content "
            "of 134.47 MJ/gal. A second, sneakier trap: CBOT soyoil quotes in "
            "**cents** per pound while the whole calculation runs in **USD** per "
            "pound. The factor of 100 is the same order as the result being "
            "sought, so forgetting it breaks nothing visibly — it produces a "
            "plausible, wrong number."
        ),
        conversion=(
            "LCFS_value_usd_gal = LCFS_usd_t x (CI_std - CI_f) x EER x 134.47e-6\n"
            "credit_45Z_usd_gal = 1.00 x max(0, (50 - CI_f)/50)   if North American, else 0\n"
            "required_discount_usd_lb = [credit_45Z(dom) - LCFS_value(CI gap)] / yield"
        ),
        proxies=[
            "LCFS credit price: **absent from the Bloomberg export**, published "
            "free by CARB — treated here as an axis, never as a series (see S3)",
            "imported UCO price: unreachable without a Platts/PGA subscription — "
            "the page is built to never need it (see S2)",
            "UCO collection floor delivered USGC: the only unobservable "
            "parameter, and deliberately the one asked of the counterparty (S5)",
        ],
        out_of_scope=[
            "diesel price, RIN D4 price, plant opex and ROI: they cancel out "
            "between the two pathways and enter no result on the page — "
            "demonstrated in S2, not assumed",
            "terminal logistics constraints and existing supply contracts",
        ],
        frequency_note=(
            "CBOT soyoil is daily. The LCFS credit is published monthly by CARB, "
            "45Z is a regulatory constant: results are therefore shown as "
            "functions of these two parameters, not as time series."
        ),
        data_warnings=[
            "The 45Z calibration gap (0.46 modelled against ~0.49 published) is "
            "not resolved — it is shown at the bottom of the page, because "
            "closing it with an invented adjustment factor would turn the result "
            "into an artefact.",
            SRE_WARNING,
        ],
    ),
)

# ===========================================================================
# Parameters
# ===========================================================================
st.sidebar.markdown("### Parameters")
lcfs_price = st.sidebar.slider(
    "LCFS credit price (USD/t CO2e)", 0.0, 400.0, 75.0, 5.0,
    help="Published monthly by CARB. Absent from the export — an axis, not data.",
)
ci_imported = st.sidebar.slider("Imported UCO's CI (gCO2e/MJ)", 5.0, 30.0, 15.0, 1.0)
yield_lb_gal = st.sidebar.slider("Yield (lb of feedstock per gallon)", 6.5, 9.0, 7.6, 0.1)
uco_floor = st.sidebar.slider(
    "UCO collection floor delivered USGC (c/lb)", 15.0, 60.0, 35.0, 1.0,
    help="Collection + freight. Below it, there is no export supply.",
)
window_start = st.sidebar.selectbox("Soyoil window", ["2015", "2020", "2024"], index=0)

imported = Feedstock("imported UCO", ci_imported, north_american=False)
common = dict(imported=imported, yield_lb_gal=yield_lb_gal)

penalty = import_penalty(lcfs_price, **common)
bounds = penalty_bounds(**common)
neutral = lcfs_neutral_price(imported=imported)
soyoil = load_soyoil_usd_lb(window_start)
burden = discount_burden(soyoil, lcfs_usd_t=lcfs_price, **common)
exit_point = structural_exit(
    soyoil, uco_floor_usd_lb=uco_floor / CENTS_PER_USD, lcfs_usd_t=lcfs_price, **common
)

kpi_banner(
    {
        "45Z soyoil earns": f"{penalty.credit_45z_usd_gal:.2f} $/gal",
        "Neutral LCFS": f"{neutral:.0f} $/t",
        "Required discount": f"{penalty.discount_required_c_lb:.2f} c/lb",
        "Weight on soyoil": f"{burden.burden_last:.1%}",
        "Soyoil (last)": f"{soyoil.iloc[-1] * CENTS_PER_USD:.1f} c/lb",
    }
)

# ===========================================================================
# S1 — the story
# ===========================================================================
section(
    "S1",
    "Plants sited for imports, then a rule that excludes imports",
    "Roughly three quarters of US renewable diesel capacity was built on or near "
    "the Gulf and California coasts. This is not a geographic accident: these "
    "plants were sited to run on feedstock **landed at the dock** — UCO from "
    "Asia, tallow — and the siting choice encodes this bet on the origin of the "
    "raw material. It was made before the tax rule existed, and it is not "
    "reversible: a plant does not move to the Corn Belt.\n\n"
    "Then two administrations write two subsidies that do not pursue the same "
    "goal. Congress writes 45Z to reward **North American** feedstock, and "
    "excludes anything that is not from the credit. California keeps paying for "
    "**low carbon intensity** through its LCFS, regardless of origin — and "
    "imported UCO happens to be less carbon-intensive than domestic soyoil. One "
    "policy therefore penalises exactly what the other rewards, on the same "
    "gallon, in the same plant, the same day. The operator is not arbitraging "
    "between two markets: they are arbitraging between two administrations.",
)
scope_note(
    "The public debate is framed this way: coastal plants argue the LCFS "
    "premium is enough to keep imports going, the soy complex argues it is not "
    "enough and soyoil takes the share. The page takes no side — it shows both "
    "camps are arguing about a variable that cannot settle it."
)

# ===========================================================================
# S2 — why the number is hard to dispute
# ===========================================================================
section(
    "S2",
    "What cancels out — and why this number is hard to dispute",
    "Both pathways produce the **same gallon of renewable diesel**, in the "
    "**same plant**, sold at the **same price**, with the **same RINs**, the "
    "**same opex** and the **same ROI**. Writing the value gap between the two, "
    "all these terms appear on both sides and vanish. What remains is what "
    "genuinely distinguishes them: the carbon intensity gap, valued by the "
    "LCFS, and the 45Z credit one pockets and the other does not.\n\n"
    "This is not a presentational nicety, it is what makes the result solid. A "
    "counterparty cannot dispute it by challenging a diesel forecast, a RIN "
    "level or an opex assumption — **none of the three appears in it**. They "
    "can only dispute it on two numbers, both observable: the CI gap between "
    "the two pathways and the LCFS credit price. This is also what allows "
    "dispensing entirely with the UCO price, which is precisely the data that "
    "is missing.",
    formula=(
        "imported_advantage - domestic_advantage\n"
        "  = [LCFS x (CI_dom - CI_imp) x EER x 134.47e-6 - credit_45Z(dom)] / yield\n"
        "\n"
        "  ↑ no P_diesel, no RIN, no opex, no ROI: they cancelled out"
    ),
)
finding(penalty.headline)

# ===========================================================================
# S3 — THE RESULT
# ===========================================================================
section(
    "S3",
    "The LCFS cannot settle the debate — and here is by how much",
    "Both camps' reasoning turns on the LCFS credit price: will it be enough, "
    "will it not. This can be answered exactly, because there exists an LCFS "
    "price that precisely offsets 45Z at feedstock price parity — "
    f"**{neutral:.0f} $/t CO2e**. But the California programme has never quoted "
    f"at this level: its historical high, reached in 2019-2020, sits around "
    f"{LCFS_PROGRAM_HIGH_USD_T:.0f} $/t, and it spent 2023-2024 below "
    f"{LCFS_PROGRAM_LOW_USD_T:.0f} $/t.\n\n"
    "The consequence is stronger than a simple \"it isn't enough\". Between its "
    "historical trough and peak — i.e. across **the entire range the programme "
    f"has ever realised** — the LCFS moves the required discount by only "
    f"{bounds.span_c_lb:.2f} c/lb. Over the same period, soyoil itself has "
    f"ranged by {(soyoil.max() - soyoil.min()) * CENTS_PER_USD:.0f} c/lb. The "
    "regulatory lever is therefore an order of magnitude smaller than the "
    "feedstock-price lever: the question is not poorly answered, it is "
    "**poorly posed**.",
    formula="required_discount(LCFS) = [0.46 - LCFS x CI_gap x 134.47e-6] / yield",
)
finding(bounds.headline)

lcfs_axis = np.arange(0.0, 401.0, 5.0)
curve = pd.DataFrame(
    {
        "discount_c_lb": [
            import_penalty(x, **common).discount_required_c_lb for x in lcfs_axis
        ],
        "realised_range": (lcfs_axis >= LCFS_PROGRAM_LOW_USD_T)
        & (lcfs_axis <= LCFS_PROGRAM_HIGH_USD_T),
    },
    index=pd.Index(lcfs_axis, name="LCFS (USD/t CO2e)"),
)
curve_fig = regime_chart(
    curve,
    "discount_c_lb",
    regime_col="realised_range",
    # Deliberately NOT the template's green: here the shaded band does not mean
    # "the trade is open", it only marks the range the parameter has actually
    # occupied. Reusing green would encode a judgement the chart does not make.
    regime_color=ALT_COLOR,
    title="Discount imported UCO must hold, by LCFS credit price",
    y_title="c/lb under soyoil",
    reference_lines={"discount at chosen price": penalty.discount_required_c_lb},
)
curve_fig.add_vline(
    x=neutral,
    line_dash="dot",
    line_color="crimson",
    annotation_text=f"{neutral:.0f} $/t — neutral",
    annotation_position="top left",
)
show(curve_fig)
scope_note(
    f"Shaded zone: the range the LCFS credit has actually occupied since the "
    f"programme's inception ({LCFS_PROGRAM_LOW_USD_T:.0f}–{LCFS_PROGRAM_HIGH_USD_T:.0f} "
    f"$/t). The curve only crosses zero at {neutral:.0f} $/t, outside this "
    "range — hence the result. Both bounds are documented and adjustable: a "
    "reader with the CARB series substitutes their own without changing the "
    "reasoning at all."
)

# ===========================================================================
# S4 — the real weight
# ===========================================================================
section(
    "S4",
    "The same discount does not carry the same weight depending on the oil price",
    "45Z is written in **dollars per gallon**, so the discount it imposes is a "
    "fixed number of cents. But a buyer does not think in absolute cents: they "
    "think in percent of the price they pay. And soyoil has quoted between "
    f"{soyoil.min() * CENTS_PER_USD:.0f} and {soyoil.max() * CENTS_PER_USD:.0f} "
    f"c/lb since {window_start}. The same "
    f"{penalty.discount_required_c_lb:.2f} c/lb discount therefore represents "
    f"{burden.burden_min:.1%} of the price when oil is dear and "
    f"{burden.burden_max:.1%} when it is cheap.\n\n"
    "The penalty's weight is thus **countercyclical to the vegetable oil "
    "price**: it bites hardest exactly when crushing margins are already thin "
    "and buyers are most price-sensitive. This is a property of how the text "
    "was written — a credit denominated in dollars per gallon rather than as a "
    "percentage of value — not a market effect.",
)
finding(burden.headline)
show(
    regime_chart(
        burden.frame.assign(burden_pct=burden.frame["burden_share"] * 100.0),
        "burden_pct",
        title="Weight of the required discount in the soyoil price",
        y_title="% of the soyoil price",
        zero_line=False,
        annotations={"2025-01-01": "45Z in effect"},
    )
)

# ===========================================================================
# S5 — the structural exit
# ===========================================================================
section(
    "S5",
    "The point where imports stop contracting and simply stop",
    "So far imports are disadvantaged, not prevented: they only need to accept "
    "the discount. But UCO has a collection cost and freight, hence a **floor "
    "price** below which there is simply no export supply — nobody collects "
    "used cooking oil in Asia and ships it at a loss. As soon as the required "
    "discount pushes UCO below this floor, the pathway does not contract "
    "gradually: it stops, and no LCFS price within its realised range brings it back.\n\n"
    "This floor is the only parameter the page cannot observe, and it is "
    "deliberately the one asked of the counterparty: the rest of the "
    "calculation is closed. What follows is therefore *their* answer "
    "translated into a critical soyoil price, then into a historical frequency.",
    formula="soyoil* = UCO_collection_floor + required_discount(LCFS)",
)
finding(exit_point.headline)

recent = structural_exit(
    load_soyoil_usd_lb("2024"),
    uco_floor_usd_lb=uco_floor / CENTS_PER_USD,
    lcfs_usd_t=lcfs_price,
    **common,
)
c1, c2, c3 = st.columns(3)
c1.metric("Critical soyoil price", f"{exit_point.soyoil_critical_usd_lb * CENTS_PER_USD:.1f} c/lb")
c2.metric(f"Below threshold since {window_start}", f"{exit_point.share_below:.0%}")
c3.metric("Below threshold since 2024", f"{recent.share_below:.0%}")

show(
    regime_chart(
        (soyoil * CENTS_PER_USD).to_frame("soyoil_c_lb").assign(
            import_impossible=soyoil < exit_point.soyoil_critical_usd_lb
        ),
        "soyoil_c_lb",
        regime_col="import_impossible",
        regime_color="rgba(248, 113, 113, 0.20)",
        title="CBOT soyoil and the critical price below which imports stop being fundable",
        y_title="c/lb",
        zero_line=False,
        reference_lines={
            "critical price": exit_point.soyoil_critical_usd_lb * CENTS_PER_USD
        },
        annotations={"2025-01-01": "45Z in effect"},
    )
)

if exit_point.share_below is not None and recent.share_below is not None:
    finding(
        f"This is the reversal that gives the page its meaning: over "
        f"{window_start}-2026, soyoil spent {exit_point.share_below:.0%} of the "
        f"time below the critical price; since 2024, {recent.share_below:.0%}. "
        "**Imports work today because oil is expensive, not because policy is "
        "generous.** A return of soyoil to its late-2010s levels would stop "
        "them, regardless of the LCFS credit price."
    )

# ===========================================================================
# S6 — the map
# ===========================================================================
section(
    "S6",
    "Where the boundary flips, as a function of the only two parameters left",
    "Since everything else cancelled out in S2, the answer fits on a two-axis "
    "map: the carbon intensity assigned to imported UCO, and the LCFS credit "
    "price. The black line is the boundary where the two pathways are "
    "equivalent. What matters is not the market point's position — it is the "
    "**slope**: nearly the whole width of the map in LCFS is needed to offset a "
    "few points of CI. Same conclusion as S3, seen differently.",
)
grid = winner_grid(
    price_domestic_usd_lb=float(soyoil.iloc[-1]),
    price_imported_usd_lb=float(soyoil.iloc[-1]),
    yield_lb_gal=yield_lb_gal,
)
show(
    sensitivity_heatmap(
        grid,
        x_col="lcfs_usd_t",
        y_col="ci_imported",
        z_col="advantage_usd_lb",
        title="Imported UCO's advantage at price parity (blue = imports win)",
        x_title="LCFS credit price (USD/t CO2e)",
        y_title="Imported UCO's CI (gCO2e/MJ)",
        breakeven_note="Black line: equivalence boundary between the two pathways",
    )
)
scope_note(
    "Map drawn **at feedstock price parity**: it isolates the effect of the two "
    "regulatory parameters by neutralising the price spread. This is precisely "
    "what makes it readable — and what shows that the price spread, absent "
    "from this map, is the term that actually decides."
)

# ===========================================================================
# S7 — the consequence on the other side of the threshold
# ===========================================================================
section(
    "S7",
    "If imports stop, domestic crushing has to absorb the volume",
    "The S5 threshold is not the end of the story: on the other side, the "
    "volume does not disappear, it has to come from domestic soyoil. The June "
    "2026 WASDE projects soybean oil consumption for biofuel rising from 14.55 "
    "to roughly 17.8 billion pounds. This increment, converted at the crushing "
    "yield of 11 pounds of oil per bushel, gives the extra crushing capacity "
    "the complex has to deliver — a number an operator compares directly to "
    "their investment plan.",
    formula="required_crush_bu_day = (Δoil_lb / 11) / 365",
)
capacity = st.slider(
    "Installed crushing capacity (M bu/day)", 4.0, 10.0, 6.8, 0.1
)
increment_lb = st.slider(
    "Oil consumption increment (bn lb/yr)", 0.5, 6.0, 3.25, 0.05,
    help="June 2026 WASDE: 14.55 → ~17.8 bn lb.",
)
balance = crush_from_soyoil_lb(
    increment_lb * 1e9, installed_capacity_bu_day=capacity * 1e6
)
c1, c2 = st.columns(2)
c1.metric("Crushing required by the increment", f"{balance.crush_required_bu_day:,.0f} bu/day")
c2.metric(
    "As a share of installed capacity",
    f"{balance.crush_required_bu_day / balance.installed_capacity_bu_day:.1%}",
)
scope_note(
    "Reading: the increment is an **additional** volume to deliver, not a total "
    "need — it compares to the expansion plan, not to existing capacity. "
    "Installed capacity is a slider because no public source gives it "
    "unambiguously in scope (announced, permitted, in service)."
)

# ===========================================================================
# Diagnostic
# ===========================================================================
calibration = calibration_gap_45z()
diagnostic_note(
    f"45Z calibration check: the linear formula (50 − CI)/50 gives "
    f"{calibration['modelled_usd_gal']:.2f} $/gal on a CI of "
    f"{SOYOIL_DOMESTIC.carbon_intensity:.0f}, against {calibration['published_usd_gal']:.2f} "
    f"$/gal published — a gap of {calibration['gap_usd_gal']:.3f} $/gal, "
    f"{calibration['gap_pct']:.0%}. It comes down to the exact CI definition "
    "used and **is not resolved**: closing it with an invented adjustment "
    "factor would make every number on the page unverifiable. Carried through "
    f"to the required discount, this gap is worth about "
    f"{calibration['gap_usd_gal'] / yield_lb_gal * CENTS_PER_USD:.2f} c/lb — "
    f"against the {bounds.span_c_lb:.2f} c/lb the LCFS moves across its entire history."
)

mail_question(
    f"At {lcfs_price:.0f} $/t CO2e on the LCFS credit, I find imported UCO must "
    f"sell around {penalty.discount_required_c_lb:.1f} c/lb below domestic "
    "soyoil purely to offset the 45Z it cannot claim — and that across the "
    f"LCFS programme's entire history, this number could only have varied by "
    f"{bounds.span_c_lb:.1f} c/lb. Is the discount you actually see on UCO "
    "delivered USGC of that order? And below what collected price do you "
    "simply stop loading?",
    "Feedstock buyers, coastal renewable diesel operators, used cooking oil origination",
)
