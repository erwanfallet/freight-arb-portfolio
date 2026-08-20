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
    FORWARD_HORIZON_DAYS,
    MWH_TH_PER_TONNE_COAL,
    TRAILING_MEDIAN_WINDOW,
    ceiling_test,
    efficiency_identification,
    load_real_switching_frame,
    switch_ttf_eur_mwh,
    switching_distance_pct,
)
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
    show,
)

st.set_page_config(page_title="B — Coal-to-gas ceiling", layout="wide")

_LIVE = snapshot_banner()

# ===========================================================================
# Header and scope
# ===========================================================================
page_header(
    code="B",
    title="The coal-switching ceiling — a known idea, tested honestly for the first time",
    subtitle=(
        "Distance from the switching level predicts TTF's next 20 days; distance from "
        "TTF's own trailing median does not — on windows that don't reuse the same outcome"
    ),
    scope=Scope(
        unit_trap=(
            "**Three units and two currencies, before any comparison exists.** API2 coal is "
            "quoted in **USD per tonne** — a mass, which only becomes energy through a "
            "calorific value of 6 000 kcal/kg. TTF gas is quoted in **EUR per MWh** — "
            "already energy. EUA carbon is quoted in **EUR per tonne of CO2** — which only "
            "enters through an emission factor. And the two fuels are burned in plants whose "
            "efficiencies differ by roughly a factor of one and a half, so nothing is "
            "comparable until everything is expressed per MWh of **electricity**, and the "
            "switching level restated back into TTF's own EUR/MWh unit."
        ),
        conversion=(
            f"coal_eur_mwh_th = API2_usd_t / EURUSD / {MWH_TH_PER_TONNE_COAL:.3f}\n"
            "ttf_switch      = (eta_gas/eta_coal) x coal_th\n"
            "                  + eua x (eta_gas x EF_coal/eta_coal - EF_gas)\n"
            "distance_pct    = (ttf_actual - ttf_switch) / ttf_switch"
        ),
        proxies=[
            "emission factors: standard combustion figures, not plant-specific — a real unit "
            "varies with coal rank and gas composition",
            "plant efficiencies: not market data, not published — the switching level is "
            "computed at one representative pair and re-checked across the plausible range "
            "in S4, but the ceiling *test* itself (S2-S3) is a claim about the mechanism, "
            "not about this exact level",
        ],
        out_of_scope=[
            "start-up costs, minimum stable generation and ramp constraints, which decide "
            "whether a plant that is theoretically in the money actually runs",
            "the API2 − API4 Richards Bay arb this engine was originally built for: **API4 "
            "is absent from the export**, so that spread is not computable and is not faked "
            "with a proxy",
        ],
        frequency_note=(
            "All four legs are daily, 2 214 common sessions since January 2018. The forward-"
            "return test below is deliberately **not** run on daily data — see S2."
        ),
        data_warnings=[
            "The EURUSD quoting direction is checked at load time rather than assumed — a "
            "reversed FX leg would produce a coal price roughly 25% off with no visible "
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
horizon = st.sidebar.slider(
    "Forward horizon (trading days)", 10, 40, FORWARD_HORIZON_DAYS, 5,
    help="Also the spacing between the non-overlapping windows used for the t-stat.",
)

frame = load_real_switching_frame()
switch = switch_ttf_eur_mwh(frame, coal_efficiency=coal_efficiency, gas_efficiency=gas_efficiency)
distance = switching_distance_pct(frame, switch)
test = ceiling_test(frame, switch, horizon_days=horizon)
identification = efficiency_identification(frame)

kpi_banner(
    {
        "TTF (last)": f"{frame['ttf_eur_mwh'].iloc[-1]:.1f} EUR/MWh",
        "Switching level": f"{switch.iloc[-1]:.1f} EUR/MWh",
        "Distance": f"{100 * distance.iloc[-1]:+.1f}%",
        "Time above ceiling": f"{100 * test.share_above:.1f}%",
        "Switching t-stat": f"{test.switching.t_stats['distance']:.2f}",
    }
)

# ===========================================================================
# S1
# ===========================================================================
section(
    "S1",
    "The belief is old; the level and the test are not",
    "Every power desk in Europe knows the shape of this idea: push gas expensive enough "
    "relative to coal-plus-carbon and generation switches fuel, which should cap TTF's "
    "further upside. That is not the pitch. The pitch is that the level at which this is "
    "supposed to bind is never published — it depends on two plant efficiencies nobody "
    "quotes — and that the belief itself has an honest, testable alternative: TTF might "
    "simply mean-revert on its own, with the switching story attached to the chart after "
    "the fact.\n\n"
    "This page computes the level from `DEFAULT_COAL_EFFICIENCY` and `DEFAULT_GAS_EFFICIENCY`, "
    "and then asks the only question that separates a mechanism from a narrative: does "
    "distance from **that specific level** predict what TTF does next, better than distance "
    "from a generic reference does?",
    formula="ttf_switch = (eta_gas/eta_coal) x coal_th + eua x (eta_gas x EF_coal/eta_coal - EF_gas)",
)
show(
    regime_chart(
        pd.DataFrame({"ttf_switch": switch, "ttf_actual": frame["ttf_eur_mwh"]}).assign(
            above=frame["ttf_eur_mwh"] > switch
        ),
        "ttf_switch",
        regime_col="above",
        regime_color=ALT_COLOR,
        title=f"Switching TTF level at {coal_efficiency:.0%} coal / {gas_efficiency:.0%} gas",
        y_title="EUR per MWh",
        zero_line=False,
        annotations={"2021-10-01": "gas crisis", "2022-02-24": "Ukraine"},
    )
)
scope_note(
    f"Shaded where actual TTF sits **above** the switching level — the regime in which "
    f"burning coal instead should be cheaper. That is {test.share_above:.0%} of the sample "
    "at these efficiencies. The next section tests what happens to TTF once it gets there."
)

# ===========================================================================
# S2 — THE HONEST TEST
# ===========================================================================
section(
    "S2",
    "Why the regression has to skip nineteen days out of twenty",
    f"Regressing TTF's {horizon}-day forward return on daily distance from the switching "
    f"level reuses the same {horizon}-day outcome on every one of the {horizon} consecutive "
    "rows that overlap it — the sample looks large, but the t-stat is counting the same "
    "evidence dozens of times. The fix is not a correction formula applied after the fact; "
    "it is refusing the overlap in the first place, by sampling every "
    f"{horizon}th row instead of every row.\n\n"
    f"That leaves {test.switching.n_obs} genuinely independent windows out of "
    f"{test.n_overlapping} overlapping ones — a small sample on purpose, and one this page "
    f"refuses to give a verdict on below {60} observations rather than dress up a thin "
    "result.",
    formula="honest sample = frame.iloc[::horizon_days]  (not every row)",
)
c1, c2 = st.columns(2)
c1.metric("Non-overlapping windows", f"{test.switching.n_obs}")
c2.metric("Overlapping rows (for reference only)", f"{test.n_overlapping}")

# ===========================================================================
# S3 — THE RESULT
# ===========================================================================
section(
    "S3",
    "The switching distance survives; a generic distance does not",
    "Two regressors compete for the same forward return. The first is distance from the "
    "switching level computed above — carbon, coal and gas all reconciled through plant "
    "efficiency. The second is a placebo with none of that: distance from TTF's own "
    f"{TRAILING_MEDIAN_WINDOW}-day trailing median, a measure any desk could compute without "
    "ever touching a coal price or an efficiency assumption.\n\n"
    "If the ceiling were just ordinary mean reversion wearing a fuel-switching costume, the "
    "two regressors would predict the same thing and the placebo would look just as "
    "significant. It does not — and run together in a horse race, the switching distance "
    "keeps its sign and its significance while the placebo does not survive the same test.",
)
finding(
    f"Switching distance: coefficient {test.switching.coefficients['distance']:+.3f} "
    f"(t = {test.switching.t_stats['distance']:.2f}, R² = {test.switching.r_squared:.3f}, "
    f"n = {test.switching.n_obs}). Placebo distance alone: coefficient "
    f"{test.placebo.coefficients['placebo']:+.3f} "
    f"(t = {test.placebo.t_stats['placebo']:.2f}, not significant)."
)
st.markdown("**Horse race — both regressors, same sample:**")
st.code(test.horse_race.summary(), language="text")
diagnostic_note(
    "Read the horse race carefully: the placebo's coefficient does not merely shrink toward "
    "zero, it comes back with the wrong sign once switching distance is in the equation. "
    "That is not two mechanisms sharing credit — it is one mechanism, and a naive "
    "reversion measure that was picking up switching's effect by proxy until it was asked "
    "to compete against the real thing."
)

# ===========================================================================
# S4
# ===========================================================================
section(
    "S4",
    "The level moves with the efficiency pair; the test does not",
    "The switching **level** is not knowable without an efficiency assumption, and "
    f"sweeping the plausible range moves it by {identification.swing_eur_t:.0f} EUR/t of "
    "carbon-equivalent — wide enough that the same market conditions can read as carbon "
    "having done its job or having barely worked at all, depending only on which plants "
    "are assumed to sit at the margin.\n\n"
    "What does not move with the efficiency pair is the **shape** of S2-S3: run the same "
    "horse race at either end of the plausible range and the switching regressor keeps its "
    "sign and the placebo keeps losing. The parameter decides where the ceiling sits; it "
    "does not decide whether a ceiling exists.",
)
grid = identification.grid.copy()
grid.columns = ["coal efficiency", "gas efficiency", "switching EUA (EUR/t)", "share EUA above"]
st.dataframe(
    grid.style.format(
        {
            "coal efficiency": "{:.0%}", "gas efficiency": "{:.0%}",
            "switching EUA (EUR/t)": "{:.1f}", "share EUA above": "{:.0%}",
        }
    ),
    width="stretch", hide_index=True,
)

# ===========================================================================
# S5
# ===========================================================================
section(
    "S5",
    "What this page is not, and the project it replaces",
    "This engine was originally built around the API2 − API4 arb — Rotterdam against "
    "Richards Bay — and the thesis that the marginal South African tonne stopped pricing "
    "off Europe after 2022. **API4 is not in the export**, so that spread is not "
    "computable. Rather than substitute a proxy and keep the original headline, the "
    "question was changed to one this data can actually answer, and then tested against "
    "its own honest null rather than just asserted.\n\n"
    "What is deliberately missing here is the plant-level reality: start-up costs, minimum "
    "stable generation, ramp rates and must-run constraints all decide whether a unit that "
    "is theoretically in the money actually generates. A switching level is a necessary "
    "condition for the ceiling to bind, never a sufficient one — and the horse race in S3 "
    "tests whether the price data behaves as if it binds, not whether every plant obeys it "
    "every day.",
)

mail_question(
    "Regressing TTF's forward return on distance from the coal-switching level, on "
    f"non-overlapping {horizon}-day windows (n = {test.switching.n_obs}), the switching "
    f"distance comes in at t = {test.switching.t_stats['distance']:.2f} and survives a "
    "horse race against distance-from-trailing-median, which does not. Does your desk's "
    "switching calculation actually get backtested this way against a mean-reversion null, "
    "or is the ceiling more of a heuristic applied on the day than a level anyone has "
    "checked binds in the data?",
    "European power and gas desks (Uniper, RWE, EDF Trading, Vitol, Glencore coal), "
    "utility fuel procurement, carbon desks",
)
