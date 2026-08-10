from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_APP_DIR.parent / "src"))
sys.path.insert(0, str(_APP_DIR))

from agri.chains.china_soy import load_real_crush_frame  # noqa: E402
from agri.chains.plant_option import (  # noqa: E402
    PlantOptionError,
    calibrate_ou,
    compare_policies,
    diagnose_real_margin_stationarity,
    implied_switching_cost,
    real_board_crush_margin,
    run_heuristic_policy,
    solve_hysteresis,
    switching_cost_sensitivity,
    volatility_sensitivity,
)
from agri.data.bloomberg_loader import DEFAULT_PATH  # noqa: E402
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

st.set_page_config(page_title="T2-5 — The plant as an option", layout="wide")

_LIVE = snapshot_banner()

# ===========================================================================
# Header and scope
# ===========================================================================
page_header(
    code="T2-5",
    title="The plant as an option on the margin",
    subtitle=(
        "What a curtailment signal \"margin < 0 for N periods\" silently assumes — and "
        "the shutdown-restart cost it implies, on the Chinese crush margin"
    ),
    scope=Scope(
        unit_trap=(
            "The Chinese crush margin mixes **three tax regimes and one currency**: "
            "CBOT quotes in cents per bushel, DCE in CNY/t **VAT included**, and "
            "import duty applies to the CNF value rather than the FOB one. The margin "
            "is computed ex-VAT, so DCE revenue is divided by (1+VAT) before any "
            "comparison."
        ),
        conversion=(
            "bean_cnf_usd_t = (CBOT_usd_bu + basis_c_bu/100) x 36.7437 + freight_usd_t\n"
            "margin_cny_t   = (0.785 x meal_dce + 0.185 x oil_dce)/(1+VAT)\n"
            "                 - bean_cnf_usd_t x USDCNY x (1+duty) - processing"
        ),
        proxies=[
            "US Gulf FOB basis and China freight: parameterised flat rates, absent "
            "from the Bloomberg export",
            "shutdown / restart / idling costs: none are public, they are the page's sliders",
        ],
        out_of_scope=[
            "US crushing (board crush): its margin never went negative in 8 years, so "
            "the curtailment question does not arise there — see S5",
            "supply contracts and delivery commitments, which prevent stopping freely",
        ],
        frequency_note=(
            "CBOT, DCE and USDCNY are all daily; the margin is therefore daily, and a "
            "\"period\" in the model is one business day, not a month."
        ),
        data_warnings=[
            "DCE meal/oil are converted from VAT-inclusive CNY/t — the applicable VAT "
            "rate depends on the product AND the date, to re-verify before any binding use.",
        ],
    ),
)

# ===========================================================================
# Data and calibration
# ===========================================================================
st.sidebar.markdown("### Operating costs (CNY/t of bean)")
cost_total = st.sidebar.slider("Shutdown-restart cost (round trip)", 10.0, 600.0, 143.0, 1.0)
restart_share = st.sidebar.slider("Restart's share of that total", 0.5, 0.9, 0.67, 0.01)
cost_idle = st.sidebar.slider("Idling cost (per day)", 0.0, 10.0, 2.0, 0.5)
n_periods = st.sidebar.slider("N of the heuristic rule (days)", 2, 30, 4, 1)

cost_restart = cost_total * restart_share
cost_shutdown = cost_total * (1.0 - restart_share)

margin = load_real_crush_frame()["margin"]
ou = calibrate_ou(margin, strict=False)
diagnostic = diagnose_real_margin_stationarity(margin)

kpi_banner(
    {
        "Margin (last)": f"{margin.iloc[-1]:+,.0f}",
        "Median": f"{margin.median():+,.0f}",
        "Share below zero": f"{(margin < 0).mean():.0%}",
        "Worst print": f"{margin.min():+,.0f}",
        "Stationarity": diagnostic.stationarity.verdict,
    }
)

# ===========================================================================
# S2
# ===========================================================================
section(
    "S2",
    "The margin that actually poses the question",
    # Prose numbers are computed, never hardcoded: a negative share written in stone
    # drifts as soon as the series grows, and a reader comparing the sentence to the
    # KPI banner right above sees the contradiction before seeing the argument.
    f"The Chinese crush margin goes negative **{(margin < 0).mean():.0%} of the "
    f"time** and falls as low as {margin.min():,.0f} CNY/t: this is an asset whose "
    "shutdown is a real, repeated decision, not a textbook hypothesis. It is also "
    f"**{diagnostic.stationarity.verdict} under the joint ADF+KPSS test**, which "
    "licenses the Ornstein-Uhlenbeck calibration that follows — a condition the US "
    "crushing margin, in contrast, fails to meet on every window tested (see S5).",
    formula="dM = kappa (theta - M) dt + sigma dW",
)
show(
    regime_chart(
        margin.to_frame("margin").assign(below_zero=margin < 0),
        "margin", regime_col="below_zero", title="Chinese crush margin (CNY/t of bean)",
        y_title="CNY/t", annotations={"2020-03-01": "Covid", "2022-02-24": "Ukraine"},
    )
)
st.caption(f"Calibration: {ou.summary}")

# ===========================================================================
# S3 — the deliverable
# ===========================================================================
section(
    "S3",
    "What the rule \"margin < 0 for N days\" silently assumes",
    "A threshold curtailment signal does not just say *when* to stop: because it "
    "waits for N periods of confirmation, it actually stops **well below its "
    "displayed threshold**, and restarts well above it. It therefore implements a "
    "band, whether it means to or not — and a band corresponds to a precise "
    "round-trip cost. The rule's actual stop and restart levels are read off the "
    "real path, then the switching cost whose calibrated exercise boundary "
    "reproduces that width is searched for. The reader does not have to accept the "
    "model: comparing one number to their own is enough.",
    formula="M_off_effective, M_on_effective  →  which K reproduces this band width?",
)

implied = implied_switching_cost(margin, ou, n_periods=n_periods, cost_idle=cost_idle)
finding(implied.headline)

heuristic_preview = run_heuristic_policy(
    margin, threshold=0.0, n_periods=n_periods,
    cost_restart=cost_restart, cost_shutdown=cost_shutdown, cost_idle=cost_idle,
)
c1, c2, c3 = st.columns(3)
c1.metric("Rule's displayed threshold", "0 CNY/t")
c2.metric("Effective shutdown (median)", f"{implied.effective_m_off:+,.1f}")
c3.metric("Effective restart (median)", f"{implied.effective_m_on:+,.1f}")
scope_note(
    f"{heuristic_preview.n_stops} shutdowns triggered over the sample. The gap "
    "between the displayed threshold and the effective shutdown level is exactly "
    "what persistence buys."
)

# ===========================================================================
# S4
# ===========================================================================
section(
    "S4",
    "What the rule costs, and the order of magnitude of the stakes",
    "Three policies on the same path, in full P&L: the threshold rule, the "
    "calibrated boundary, and the counterfactual that never stops. **The order of "
    "the stakes matters more than the ranking**: being able to stop at all is worth "
    "a lot; picking the right stopping rule is worth an order of magnitude less. "
    "Presenting the refinement before the first term would make a detail look like "
    "the subject.",
)

band = solve_hysteresis(ou, cost_restart=cost_restart, cost_shutdown=cost_shutdown, cost_idle=cost_idle)
comparison = compare_policies(
    margin, band, cost_restart=cost_restart, cost_shutdown=cost_shutdown,
    cost_idle=cost_idle, n_periods=n_periods,
)
finding(comparison.headline)

if not comparison.band_is_available:
    diagnostic_note(
        "At these costs, friction is too small relative to the margin's conditional "
        "volatility (standard deviation of roughly 62 CNY/t per day) for an exercise "
        "boundary to exist: the \"stop\" and \"restart\" regions overlap, and the "
        "plant would chatter. The solver refuses to produce a band rather than "
        "fabricate an inverted one — raise the round-trip cost to get back to a "
        "well-posed case."
    )
else:
    st.caption(band.headline)

st.dataframe(
    comparison.to_frame().round(0), width="stretch", hide_index=True,
)

# ===========================================================================
# S5
# ===========================================================================
section(
    "S5",
    "Why this page is not anchored on US crushing",
    "The same calculation on the CBOT board crush gives a result that invalidates "
    "the question: the US crushing margin **never went negative** over the past "
    "eight years, so no shutdown rule ever triggers and the option to close is "
    "worth nothing. It also fails the joint ADF+KPSS test on every window tested — "
    "the full 36 years as well as every sub-period since 2005 — which rules out a "
    "homogeneous OU calibration. This is a result, not a calibration failure: the "
    "margin goes through genuine regime breaks that a fixed-parameter model cannot "
    "absorb.",
)
us_margin = real_board_crush_margin(start="2018-01-01")
us_diagnostic = diagnose_real_margin_stationarity(us_margin)
c1, c2, c3 = st.columns(3)
c1.metric("US board crush — minimum", f"{us_margin.min():+.2f} USD/bu")
c2.metric("Share below zero", f"{(us_margin < 0).mean():.0%}")
c3.metric("Stationarity", us_diagnostic.stationarity.verdict)
diagnostic_note(us_diagnostic.headline)

# ===========================================================================
# S6
# ===========================================================================
section(
    "S6",
    "Sensitivity to switching cost — and where the band stops existing",
    "Band width grows with the round-trip cost, which is what makes the inversion "
    "in S3 possible. But below a certain cost the boundary **stops existing**: when "
    "friction becomes small relative to the margin's conditional volatility, the "
    "stop and restart regions overlap and the problem no longer admits a band. "
    "These rows are flagged rather than filled with a negative width that would "
    "read as a narrow band.",
)
sensitivity = switching_cost_sensitivity(
    margin, ou,
    cost_grid=np.geomspace(5.0, 600.0, 10),
    cost_idle=cost_idle, n_periods=n_periods,
)
st.dataframe(
    sensitivity[
        ["switching_cost", "m_off", "m_on", "band_width", "degenerate",
         "gap_vs_heuristic", "n_stops_heuristic", "n_stops_band"]
    ].round(1),
    width="stretch", hide_index=True,
)
n_degenerate = int(sensitivity["degenerate"].sum())
if n_degenerate:
    scope_note(
        f"{n_degenerate} of the {len(sensitivity)} levels tested produce no valid "
        "band — this is the lower bound below which the question \"which shutdown "
        "rule\" has no answer."
    )

# ===========================================================================
# S7
# ===========================================================================
section(
    "S7",
    "The counter-intuitive demonstration",
    "At an unchanged average margin, **a more volatile margin makes the plant more "
    "valuable**: the flexibility to stop truncates the low tail, so option value "
    "grows with sigma. This gives a number to the asset-heavy / asset-light debate, "
    "usually conducted in slogans — an asset whose margin swings violently can be "
    "worth more than one whose margin is stably positive.",
)
try:
    vol_sensitivity = volatility_sensitivity(
        ou, cost_restart=cost_restart, cost_shutdown=cost_shutdown, cost_idle=cost_idle
    )
    show(
        regime_chart(
            vol_sensitivity.set_index("sigma_multiplier"), "value_at_theta",
            title="Plant value at the average margin, by volatility",
            y_title="value", zero_line=False,
        )
    )
    st.dataframe(vol_sensitivity.round(2), width="stretch", hide_index=True)
except PlantOptionError as error:
    diagnostic_note(f"Sensitivity not computable at these costs: {error}")

mail_question(
    "Is your shutdown rule a margin threshold, or does the restart cost move it "
    f"explicitly? On the Chinese crush margin, a rule \"margin < 0 for {n_periods} "
    f"days\" actually stops at {implied.effective_m_off:+,.0f} CNY/t and restarts at "
    f"{implied.effective_m_on:+,.0f} — implying a round trip of roughly "
    f"{implied.implied_switching_cost:,.0f} CNY/t. Does that order of magnitude look "
    "like yours?",
    "Desk management, corporate development, crushing and smelting operators",
)
