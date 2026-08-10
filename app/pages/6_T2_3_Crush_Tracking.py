from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_APP_DIR.parent / "src"))
sys.path.insert(0, str(_APP_DIR))

from agri.chains.crush_tracking import (  # noqa: E402
    CBOT_MEAL_LB_BU,
    CBOT_OIL_LB_BU,
    hedge_ratio_identity_bias,
    load_real_board_frame,
    required_yield_precision,
    yield_exposure,
)
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

st.set_page_config(page_title="T2-3 — Board crush", layout="wide")

_LIVE = snapshot_banner()

# ===========================================================================
# Header and scope
# ===========================================================================
page_header(
    code="T2-3",
    title="The board crush is not a price, it is a yield in disguise",
    subtitle=(
        "What yield precision does the board crush silently demand of a plant — and "
        "why that requirement tightens exactly when the margin tightens"
    ),
    scope=Scope(
        unit_trap=(
            "The board crush stacks **three units into a single formula**: the bean in "
            "USD/**bushel**, the meal in USD/**short ton**, the oil in **cents/lb**. "
            "Treating the short ton as a metric tonne misprices the meal leg by 10% — "
            "roughly half the crush itself. But this page's real trap is elsewhere: "
            "the coefficients 0.022 and 0.11 are **not unit conversions**, they are "
            "yields (44 lb of meal and 11 lb of oil per bushel) disguised as formula "
            "constants."
        ),
        conversion=(
            f"board = ({CBOT_MEAL_LB_BU:.0f} lb / 2000 lb per short ton) x meal\n"
            f"        + {CBOT_OIL_LB_BU:.0f} lb x oil / 100  -  bean\n"
            "      = 0.022 x meal + 0.11 x oil - bean      [USD/bushel]"
        ),
        proxies=[
            "crushing opex: a parameterised flat rate, no reliable public source — this "
            "is the slider that moves the conclusion the most",
        ],
        out_of_scope=[
            "local cash prices and domestic meal basis: absent from the export, so no "
            "tracking error is measured here — the page is built not to need one",
            "the lag between buying the bean and selling the products, which would add "
            "tracking error: omitting it biases **against** the page's thesis",
        ],
        frequency_note="All three legs are daily; so is the board crush.",
        data_warnings=[
            "The CBOT board crush never went negative over the sample — the same "
            "signature as in T2-5. That is a property of the board, not of a plant's "
            "economics: the board carries no opex.",
        ],
    ),
)

# ===========================================================================
# Parameters
# ===========================================================================
st.sidebar.markdown("### Parameters")
opex = st.sidebar.slider(
    "Crushing opex (USD/bushel)", 0.20, 1.00, 0.55, 0.01,
    help="The parameter that carries the sign: it sets the net margin, hence the precision requirement.",
)
meal_gap = st.sidebar.slider("Meal yield gap (lb/bu)", -4.0, 4.0, 1.0, 0.1)
oil_gap = st.sidebar.slider("Oil yield gap (lb/bu)", -2.0, 2.0, 0.0, 0.1)
window_start = st.sidebar.selectbox("Window", ["2015-01-01", "2020-01-01", "2024-01-01"], index=0)

frame = load_real_board_frame(window_start)
precision = required_yield_precision(frame, opex_usd_bu=opex)
exposure = yield_exposure(frame, meal_lb_gap=meal_gap, oil_lb_gap=oil_gap, opex_usd_bu=opex)
net_margin = frame["board"] - opex

kpi_banner(
    {
        "Board crush (last)": f"{frame['board'].iloc[-1]:+.2f} USD/bu",
        "Median net margin": f"{net_margin.median():+.2f} USD/bu",
        "Required precision (median)": f"{precision.median_lb:.1f} lb/bu",
        "…in the tight decile": f"{precision.tight_decile_lb:.2f} lb/bu",
        "1 lb erases the margin": f"{precision.share_below(1.0):.0%} of sessions",
    }
)

# ===========================================================================
# S1
# ===========================================================================
section(
    "S1",
    "Two coefficients that look like conversions and are not",
    "The board crush is written `0.022 x meal + 0.11 x oil - bean`, and these two "
    "coefficients pass for harmless conversion factors. They are not: 0.022 short "
    f"tons per bushel is {CBOT_MEAL_LB_BU:.0f} pounds of meal, and 0.11 is "
    f"{CBOT_OIL_LB_BU:.0f} pounds of oil. These are **yields**. The CBOT froze them "
    "once and for all, because a futures contract needs a definition that does not "
    "move.\n\n"
    "A plant, on the other hand, has no fixed yield. The meal it produces depends on "
    "the protein content of the beans, which depends on origin, season and lot. Two "
    "points of protein shift the yield by several pounds per bushel. A crusher who "
    "hedges on the board is therefore not just hedging: it **silently accepts "
    f"{CBOT_MEAL_LB_BU:.0f}/{CBOT_OIL_LB_BU:.0f} as its own yield**, and keeps the "
    "difference as a naked position.\n\n"
    "Nobody ever decided this position. It is the residue of a contract convention, "
    "and its dollar size is set not by the plant but by the meal price.",
)
show(
    regime_chart(
        frame.assign(net_margin=net_margin, below_zero=net_margin < 0),
        "net_margin",
        regime_col="below_zero",
        regime_color=SHUT_COLOR,
        title=f"Board crush net margin, opex at {opex:.2f} USD/bu",
        y_title="USD/bushel",
        annotations={"2020-03-01": "Covid", "2022-02-24": "Ukraine"},
    )
)
scope_note(
    f"The raw board never went negative (minimum {frame['board'].min():+.2f} USD/bu) "
    f"— once opex is subtracted, the net margin does so {(net_margin < 0).mean():.0%} "
    "of the time. The whole difference comes down to a parameter the board does not "
    "carry."
)

# ===========================================================================
# S2 — the deliverable
# ===========================================================================
section(
    "S2",
    "The inversion: the yield precision the board demands of you",
    "Measuring the tracking error would need local cash prices, which the export does "
    "not have. So the question is flipped, and it becomes more useful: **what yield "
    "gap is enough to consume the entire net margin?** The number that comes out is "
    "in pounds per bushel — the unit a crush operator thinks in, and can answer "
    "immediately on.\n\n"
    "The answer does not read off its average. The threshold is proportional to the "
    "net margin and inversely proportional to the meal price: it **collapses when the "
    "margin tightens**, exactly the regime where the hedge was supposed to matter.",
    formula="threshold_lb = (board_crush - opex) / (meal_price / 2000)",
)
finding(precision.headline)

c1, c2, c3 = st.columns(3)
c1.metric("Tightest margin decile", f"{precision.tight_decile_lb:.2f} lb/bu",
          delta=f"{precision.tight_decile_pct:.1%} of the board's {CBOT_MEAL_LB_BU:.0f} lb",
          delta_color="off")
c2.metric("Median", f"{precision.median_lb:.1f} lb/bu")
c3.metric("Widest decile", f"{precision.wide_decile_lb:.1f} lb/bu")

show(
    regime_chart(
        precision.frame.assign(under_2lb=precision.frame["breakeven_lb"] <= 2.0),
        "breakeven_lb",
        regime_col="under_2lb",
        regime_color=SHUT_COLOR,
        title="Meal yield gap that consumes the entire net margin",
        y_title="lb per bushel",
        reference_lines={"1 lb/bu": 1.0, "2 lb/bu": 2.0},
    )
)
scope_note(
    f"Shaded zones: sessions where **2 pounds are enough** — {precision.share_below(2.0):.0%} "
    f"of the sample. One pound is enough {precision.share_below(1.0):.0%} of the time. On "
    f"a base of {CBOT_MEAL_LB_BU:.0f} lb, one pound is a gap of "
    f"{1.0 / CBOT_MEAL_LB_BU:.1%} — nobody knows their meal yield to that precision on "
    "beans they have not crushed yet."
)

# ===========================================================================
# S3 — the position, seen as a position
# ===========================================================================
section(
    "S3",
    "What the gap really is: a position, not an imprecision",
    "A yield gap is not fixed by a constant, because it is not one. It is a product "
    "`quantity x price` — so an **open position in meal and oil**, carried permanently "
    "by a plant that believes itself hedged. Its size moves with product prices, "
    "without anyone having sized it.",
    formula="position = (Δlb_meal / 2000) x meal_price + Δlb_oil x oil_price / 100",
)
finding(exposure.headline)
show(
    regime_chart(
        exposure.frame.assign(
            exceeds=exposure.frame["position_usd_bu"].abs() > exposure.frame["net_margin"]
        ),
        "position_usd_bu",
        regime_col="exceeds",
        regime_color=SHUT_COLOR,
        title=f"Naked position left by {meal_gap:+.1f} lb of meal and {oil_gap:+.1f} lb of oil",
        y_title="USD/bushel",
    )
)
scope_note(
    "Shaded zones: sessions where this position alone exceeds the entire net margin. "
    "The ratio to margin is computed with a 0.05 USD/bu floor on the denominator — "
    "without it, near-zero-margin days would produce arbitrarily large ratios that "
    "would swamp the reading."
)

# ===========================================================================
# S4 — the counter-intuitive part
# ===========================================================================
correlation = float(net_margin.corr(frame["meal"], method="spearman"))
section(
    "S4",
    "The result that contradicts intuition — and is kept as such",
    "The natural intuition is that the uncovered position grows as the margin "
    "tightens, compounding both problems. **That is wrong, and it has to be said.** "
    f"The rank correlation between the net margin and the meal price is "
    f"{correlation:+.2f}: meal is the crush's main source of revenue, so a dear meal "
    "goes with a wide margin. The naked position is therefore largest when the margin "
    "is most comfortable — the misalignment partly cushions itself.\n\n"
    "What remains true, and is the real result, is subtler: in the tight regime the "
    "position is small in dollars, but the margin is smaller still. It is the "
    "**ratio** that degrades, not the position. A plant that watches its exposure in "
    "absolute dollars will never see the problem coming.",
)
comparison = pd.DataFrame(
    {
        "regime": ["tight decile", "median", "wide decile"],
        "net margin (USD/bu)": [
            net_margin.quantile(0.10), net_margin.median(), net_margin.quantile(0.90),
        ],
        "meal price (USD/st)": [
            frame.loc[net_margin <= net_margin.quantile(0.10), "meal"].median(),
            frame["meal"].median(),
            frame.loc[net_margin >= net_margin.quantile(0.90), "meal"].median(),
        ],
        "required precision (lb/bu)": [
            precision.tight_decile_lb, precision.median_lb, precision.wide_decile_lb,
        ],
    }
)
st.dataframe(comparison.round(2), width="stretch", hide_index=True)

# ===========================================================================
# S5 — why not a regression
# ===========================================================================
bias = hedge_ratio_identity_bias(
    frame, meal_lb_gap=meal_gap if meal_gap else 1.0, oil_lb_gap=oil_gap, opex_usd_bu=opex
)
section(
    "S5",
    "Why this page does not regress anything",
    "The instinct would be to estimate a hedge ratio by regressing the change in the "
    "plant's margin on the change in the board crush. That is a bad idea, and not for "
    "the reason usually assumed. The plant's margin is written **exactly** as "
    "`board + yield_gap - opex`: regressing one on the other means regressing a "
    "quantity on one of its own components. The resulting coefficient is "
    "`1 + cov(Δgap, Δboard) / var(Δboard)`, and the second term is not zero since the "
    "gap is itself made of meal and oil.\n\n"
    "Honesty requires saying this contamination is **small** — on the order of a "
    "percent. The danger is not its size: it is what a practitioner would do with the "
    "coefficient. Applying it means hedging the yield gap with **more board crush**, "
    f"when the board is a fixed {CBOT_MEAL_LB_BU:.0f}/{CBOT_OIL_LB_BU:.0f} basket. You "
    "do not hedge a gap away from 44/11 with the instrument whose yield assumption "
    "created it. The right hedge is meal and oil legs sized separately — which the "
    "page computes directly in S3, with no regression.",
    formula="beta = 1 + cov(Δyield_gap, Δboard) / var(Δboard)",
)
diagnostic_note(bias.headline)
scope_note(
    "This paragraph is inherited from T2-1 (basis vs. flat price), dropped from the "
    "portfolio for lack of cash series in the export. The result itself remained "
    "valid and applies here unchanged."
)

# ===========================================================================
# S6 — sensitivity
# ===========================================================================
section(
    "S6",
    "The parameter that carries the sign",
    "The whole page depends on a number no public source gives cleanly: crushing "
    "opex. It does not shift the conclusion at the margin, it commands it — because "
    "it sets the net margin, to which the precision threshold is directly "
    "proportional. So here is the full grid rather than one chosen figure.",
)
grid = pd.DataFrame(
    [
        {
            "opex (USD/bu)": value,
            "median net margin": float((frame["board"] - value).median()),
            "share below zero": float(((frame["board"] - value) < 0).mean()),
            "median precision (lb/bu)": required_yield_precision(frame, opex_usd_bu=value).median_lb,
            "tight decile (lb/bu)": required_yield_precision(frame, opex_usd_bu=value).tight_decile_lb,
            "1 lb is enough (share)": required_yield_precision(frame, opex_usd_bu=value).share_below(1.0),
        }
        for value in np.arange(0.30, 0.91, 0.10)
    ]
)
st.dataframe(
    grid.style.format(
        {
            "opex (USD/bu)": "{:.2f}", "median net margin": "{:+.2f}",
            "share below zero": "{:.0%}", "median precision (lb/bu)": "{:.2f}",
            "tight decile (lb/bu)": "{:.2f}", "1 lb is enough (share)": "{:.0%}",
        }
    ),
    width="stretch", hide_index=True,
)

mail_question(
    f"On the CBOT board crush since {window_start[:4]}, with opex at {opex:.2f} USD/bu, "
    f"I find that a gap of {precision.tight_decile_lb:.1f} lb of meal per bushel is "
    "enough to erase the entire net margin in the tightest margin decile — "
    f"{precision.tight_decile_pct:.1%} of the {CBOT_MEAL_LB_BU:.0f} lb the contract "
    "assumes. How far does your real meal yield drift from the board's 44 lb over a "
    "campaign, and does anyone at your shop hedge that gap separately — or does it "
    "just sit in the result?",
    "Crushing operators (ADM, Bunge, Cargill, LDC, CHS), oilseed risk managers",
)
