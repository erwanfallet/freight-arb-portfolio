from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_APP_DIR.parent / "src"))
sys.path.insert(0, str(_APP_DIR))

from agri.chains.china_soy import (  # noqa: E402
    BUSHELS_PER_TONNE_SOYBEAN,
    DEFAULT_FREIGHT_USD_T,
    affordable_origination_budget,
    impossible_windows,
    load_real_crush_frame,
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
    waterfall_chart,
)

st.set_page_config(page_title="T3-4 — China soy", layout="wide")

_LIVE = snapshot_banner()

# ===========================================================================
# Header and scope
# ===========================================================================
page_header(
    code="T3-4",
    title="The windows where no origin works",
    subtitle=(
        "Rather than testing whether Chinese purchases carry a political signature — "
        "which needs auction data nobody publishes — the periods where a commercial "
        "purchase was arithmetically impossible are dated"
    ),
    scope=Scope(
        unit_trap=(
            "**Three traps stacked into a single calculation.** CBOT quotes in USD "
            "per **bushel** (60 lb), DCE in CNY per **tonne**. Chinese prices are "
            "**VAT-inclusive** while the margin is computed ex-VAT — and the VAT on "
            "imported oilseeds is not the same as on processed products, it depends "
            "on the product **and** the date. Finally, import duty applies to the "
            "**CNF** value, not FOB. Any single one of these three mistakes shifts "
            "the margin by tens of CNY/t and flips the conclusion."
        ),
        conversion=(
            "revenue_ex_vat = (0.785 x meal_DCE + 0.185 x oil_DCE) / (1 + VAT)\n"
            "cnf_max_usd_t  = [(revenue_ex_vat - processing) / (1 + duty)] / USDCNY\n"
            "budget         = cnf_max_usd_t - CBOT_usd_bu x 36.7437"
        ),
        proxies=[
            "Chinese crushing yields (0.785 meal / 0.185 oil): a standard schedule, "
            "not a plant measurement",
            "processing cost and VAT rate: parameterised, to re-verify by product "
            "and by date",
        ],
        out_of_scope=[
            "the state reserve purchases themselves: Sinograin does not publish an "
            "auction time series, and the export contains none — precisely why the "
            "page reasons on prices alone (see S2)",
            "origin FOB basis and freight: they **drop out** of the calculation "
            "instead of entering it, which is the page's trick rather than a limitation",
        ],
        frequency_note="CBOT, DCE and USDCNY are all daily; so is the budget.",
        data_warnings=[
            "The Chinese crush margin also underlies T2-5. Both pages use the same "
            "series but draw a different quantity from it: T2-5 treats it as a "
            "process to be stopped, this one as a budget constraint.",
        ],
    ),
)

# ===========================================================================
# Parameters
# ===========================================================================
st.sidebar.markdown("### Parameters")
freight_reference = st.sidebar.slider(
    "Reference China freight (USD/t)", 20.0, 90.0, DEFAULT_FREIGHT_USD_T, 5.0,
    help="Used as a reading threshold, not a calculation input: the budget does not assume it.",
)
threshold = st.sidebar.slider("Calendar threshold (USD/t)", -20.0, 60.0, 0.0, 5.0)
window_start = st.sidebar.selectbox("Window", ["2018-01-01", "2022-01-01"], index=0)

budget = affordable_origination_budget(
    start=window_start, freight_reference_usd_t=freight_reference
)
windows = impossible_windows(budget, threshold_usd_t=threshold)
crush = load_real_crush_frame(start=window_start)

kpi_banner(
    {
        "Median budget": f"{fmt_num(budget.median_budget, 0)} USD/t",
        "Budget at last print": f"{fmt_num(budget.last_budget, 0)} USD/t",
        "Negative budget": fmt_pct(budget.share_impossible, 1),
        "Below freight alone": fmt_pct(budget.share_below_freight),
        "Dated windows": f"{len(windows)}",
    }
)

# ===========================================================================
# S1
# ===========================================================================
section(
    "S1",
    "The clean test that cannot be run",
    "The original question is good, and the test that answers it is binary: if "
    "state reserve purchases concentrate in the **lowest** crush-margin quintiles, "
    "these are purchases the crush economics rule out — a political signature. If "
    "they concentrate in the high quintiles, it is opportunistic stock rotation. "
    "The sign of a single coefficient settles it.\n\n"
    "One thing is missing: the purchase series. Sinograin does not publish its "
    "auctions as a time series, and the export contains none. The test cannot be "
    "run, and simulating it on fabricated data would only prove the code's ability "
    "to recover what was put into it.\n\n"
    "So the angle changes without changing the question. Instead of asking *where* "
    "purchases sit in the margin distribution, the periods where **no commercial "
    "purchase was possible, from any origin** are identified. In those windows, the "
    "signature no longer needs to be estimated: any cargo that arrives is "
    "non-commercial by construction.",
)

# ===========================================================================
# S2 — the mechanism
# ===========================================================================
section(
    "S2",
    "The origination budget, and why it assumes neither basis nor freight",
    "The Chinese crush margin bounds from above what a crusher can pay for a tonne "
    "of bean **delivered to port**. Subtracting the CBOT converted to a tonne basis "
    "leaves the budget an originator has available to go source the bean: origin "
    "basis plus freight.\n\n"
    "The important point is what this calculation does **not** contain. FOB basis "
    "and freight are the two terms the export does not supply, and which used to "
    "force flat-rate assumptions. Here they drop out of the calculation instead of "
    "entering it — the budget is what is left *for* them, not a quantity that "
    "assumes them. The page's conclusion therefore depends on no freight assumption "
    "at all.",
    formula=(
        "budget = cnf_max_usd_t - CBOT_usd_bu x 36.7437\n"
        "         ↑ no basis or freight assumption enters here"
    ),
)
finding(budget.headline)
scope_note(
    "Worth saying before a reader notices it: the budget **is** the crush margin, "
    "redenominated in USD/t and stripped of the two flat rates. The identity is "
    "exact — `margin = (1 + duty) × USDCNY × (budget − flat_rate)` — and verified "
    "to floating-point precision by the tests. The budget therefore adds no new "
    "information: it removes two arbitrary parameters. That is exactly what makes "
    "its crossing of zero interpretable where the margin's is not — the margin's "
    "zero depends on the flat rate chosen, the budget's depends on nothing."
)
show(
    regime_chart(
        budget.frame,
        "budget_usd_t",
        regime_col="impossible",
        regime_color=SHUT_COLOR,
        title="Budget available for origin basis and freight",
        y_title="USD per tonne",
        reference_lines={f"freight alone: {fmt_num(freight_reference, 0)}": freight_reference},
        annotations={"2022-02-24": "Ukraine", "2023-06-01": "2023 trough"},
    )
)
scope_note(
    "Shaded zones: negative budget. The dotted line is the reference freight — "
    "below it, freight consumes the whole budget and the bean would have to be "
    "bought **below** the CBOT price at origin, which no exporter does durably."
)

# ===========================================================================
# S3 — THE DELIVERABLE
# ===========================================================================
section(
    "S3",
    "The calendar — dates, not a coefficient",
    "This is the page's deliverable, and it has a property a coefficient does not: "
    "it checks directly against an originator's own book. The question becomes "
    "\"did you load during those windows?\", answered yes or no.\n\n"
    "Time concentration is the salient fact. Every window of strictly negative "
    "budget falls in **2023**, the longest running from 7 June to 5 July. This is "
    "not noise scattered around zero: it is an episode, with a start date and an "
    "end date.",
)
if len(windows):
    display = windows[
        [c for c in ("start", "end", "duration_days", "min_depth", "mean_depth") if c in windows.columns]
    ].copy()
    display.columns = [
        {"start": "start", "end": "end", "duration_days": "duration (days)",
         "min_depth": "lowest budget", "mean_depth": "average budget"}.get(c, c)
        for c in display.columns
    ]
    # Rounding the whole frame warns on the date columns; target the numeric ones.
    numeric = display.select_dtypes("number").columns
    display[numeric] = display[numeric].round(1)
    st.dataframe(display, width="stretch", hide_index=True)
    finding(
        f"{len(windows)} window(s) below {fmt_num(threshold, 0)} USD/t, for a total "
        f"of {fmt_num(float(windows['duration_days'].sum()), 0)} days. The longest "
        f"lasts {fmt_num(float(windows['duration_days'].max()), 0)} days."
    )
else:
    diagnostic_note(
        f"No window below {fmt_num(threshold, 0)} USD/t over this period — the "
        "threshold may be too low, or the window too short."
    )

# ===========================================================================
# S4 — where the budget comes from
# ===========================================================================
last = budget.frame.iloc[-1]
crush_last = crush.iloc[-1]
section(
    "S4",
    "Where the number comes from, line by line",
    "A budget that emerges from a chain of three conversions deserves to be opened "
    "up. The waterfall below starts from the crusher's ex-tax revenue at the last "
    "print and works down to the available budget. A practitioner disputing the "
    "result can point to the line whose level they dispute, rather than the total.",
)
show(
    waterfall_chart(
        {
            "revenue ex-VAT (meal + oil)": float(crush_last["revenue_ex_vat"] / last["usdcny"]),
            "processing": -120.0 / float(last["usdcny"]),
            "import duty": float(
                last["cnf_max_usd_t"] - (crush_last["revenue_ex_vat"] - 120.0) / last["usdcny"]
            ),
            "CBOT converted to a tonne": -float(last["cbot_usd_t"]),
        },
        total_label="basis + freight budget",
        title=f"Breakdown on {budget.frame.index[-1]:%d %b %Y}",
        y_title="USD per tonne",
    )
)
c1, c2, c3 = st.columns(3)
c1.metric("Maximum fundable CNF", f"{fmt_num(last['cnf_max_usd_t'], 0)} USD/t")
c2.metric("CBOT per tonne", f"{fmt_num(last['cbot_usd_t'], 0)} USD/t")
c3.metric("Remaining budget", f"{fmt_num(last['budget_usd_t'], 0)} USD/t")
scope_note(
    f"Bushel → tonne conversion: {fmt_num(BUSHELS_PER_TONNE_SOYBEAN, 4)} bushels per "
    "tonne of soybean, derived from the regulatory weight of 60 lb per bushel "
    "rather than hardcoded."
)

# ===========================================================================
# S5
# ===========================================================================
section(
    "S5",
    "What this page does not claim to show",
    "The budget says what a crusher **can** pay, not what was paid. It therefore "
    "does not demonstrate that a political cargo arrived in the dated windows — it "
    "establishes that if a cargo did arrive there, it was not explained by crush "
    "economics. That is a shifted burden of proof, not a proof.\n\n"
    "Two further caveats, both running **against** the thesis and so worth stating. "
    "First, an integrated crusher may accept a zero crush margin if it earns "
    "elsewhere in the chain — the budget would then be understated. Second, cargoes "
    "are fixed weeks before arrival: a window that is impossible on arrival could "
    "have been fundable at fixing time. The calendar should therefore be checked "
    "against **fixing dates**, not unloading dates.",
)
diagnostic_note(
    "The fixing/arrival lag is this page's most serious limitation. It cannot be "
    "corrected here — it would need a fixing calendar only an originator holds — "
    "and that is exactly what makes the question in the email a real question."
)

mail_question(
    f"Computing what the Chinese crush margin leaves for origin basis and freight, "
    f"I find {fmt_pct(budget.share_impossible, 1)} of sessions where this budget is "
    f"negative — meaning even a free bean and free freight would not make the "
    f"crush pay — all concentrated in 2023, including a window of "
    f"{fmt_num(float(windows['duration_days'].max()), 0) if len(windows) else '—'} "
    "days. Did you fix China cargoes during those windows? And if so, was the "
    "crush margin really the constraint, or was another link in the chain carrying "
    "the result?",
    "Oilseed origination (COFCO, Sinograin, Bunge, LDC, Cargill), China soy desks",
)
