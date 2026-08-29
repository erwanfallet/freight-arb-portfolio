from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
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
    anchor_encompassing,
    asymmetry_test,
    dampening_attribution,
    natural_hedge,
    spread_betas,
    switching_depth_profile,
    transmission_test,
    bootstrap_null,
    ceiling_test,
    efficiency_invariance,
    phase_robustness,
    stambaugh_diagnostics,
    subperiod_stability,
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
    title="The carbon hedge inside the switching spread, and its disappearance",
    subtitle=(
        "A dual-fuel generator owns an option to switch, and an option is priced off "
        "volatility. Inside that spread gas and carbon enter with opposite signs, so a "
        "positive correlation removes volatility — a natural hedge worth up to 29% of it "
        "for eight straight years. In 2026 the correlation collapsed to zero and the hedge "
        "went with it, repricing the option by ~11% with nothing flagging an error"
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
            "plant efficiencies: not market data, not published. S11 shows they move the "
            "*level* enormously and the *prediction* not at all — and that this is an "
            "algebraic invariance rather than a property of this sample, so no better "
            "efficiency estimate would change it",
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
            "return test below is deliberately **not** run on daily data — see S8."
        ),
        data_warnings=[
            "The EURUSD quoting direction is checked at load time rather than assumed — a "
            "reversed FX leg would produce a coal price roughly 25% off with no visible "
            "error, which is exactly the failure mode this portfolio is built to catch.",
            "This regressor contains TTF in its own numerator, so the predictive "
            "regression is biased toward the result reported here. The bias is measured "
            "two independent ways in S9 and the p-value quoted anywhere on this page is "
            "the bias-corrected one, never the raw t-statistic.",
            "The evidence is concentrated in the 2021-2026 window; the calm pre-crisis "
            "years alone do not carry the result (S13).",
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
try:
    test = ceiling_test(frame, switch, horizon_days=horizon)
except ValueError as error:
    section(
        "Refused",
        "The sample is too short at this horizon, and the page will not pretend otherwise",
        f"{error}.\n\n"
        "This is the engine's own guardrail rather than a failure: lengthening the "
        "forward horizon also widens the spacing between non-overlapping windows, so the "
        "independent sample shrinks in proportion. A t-statistic computed on what is left "
        "would be arithmetically available and worth nothing.\n\n"
        "Shorten the horizon in the sidebar to bring the window count back above the "
        "threshold. The refusal is the honest answer at this setting — it is what the "
        "rest of the page is protecting.",
    )
    st.stop()

phases = phase_robustness(frame, switch, horizon_days=horizon)
bias = stambaugh_diagnostics(frame, switch, horizon_days=horizon)
boot = bootstrap_null(frame, switch, horizon_days=horizon)
try:
    asymmetry = asymmetry_test(frame, switch, horizon_days=horizon)
except ValueError as error:
    asymmetry = None
    asymmetry_reason = str(error)
invariance = efficiency_invariance(frame, horizon_days=horizon)
encompassing = anchor_encompassing(frame, switch, horizon_days=horizon)

# --- the trade ---------------------------------------------------------------
b_ttf, b_eua = spread_betas(
    coal_efficiency=coal_efficiency, gas_efficiency=gas_efficiency
)
hedge = natural_hedge(
    frame, coal_efficiency=coal_efficiency, gas_efficiency=gas_efficiency
)
attribution = dampening_attribution(
    frame, year_from=2025, year_to=2026,
    coal_efficiency=coal_efficiency, gas_efficiency=gas_efficiency,
)
transmission = transmission_test(frame)
depth = switching_depth_profile(
    frame, coal_efficiency=coal_efficiency, gas_efficiency=gas_efficiency
)

kpi_banner(
    {
        "TTF (last)": f"{frame['ttf_eur_mwh'].iloc[-1]:.1f} EUR/MWh",
        "Switching level": f"{switch.iloc[-1]:.1f} EUR/MWh",
        "Distance": f"{100 * distance.iloc[-1]:+.1f}%",
        "Hedge, typical year": f"{hedge.typical_dampening:.0%} of vol",
        "Hedge, 2026": f"{hedge.latest.dampening:+.0%}",
        "Option mispriced by": f"{attribution.option_value_uplift:+.0%}",
    }
)

# ===========================================================================
# S1 — THE ASSET
# ===========================================================================
section(
    "S1",
    "What a dual-fuel generator actually owns",
    "A utility holding both a coal unit and a CCGT does not own a view on fuel prices. It "
    "owns an **option to switch between them** — it runs whichever is cheaper and leaves "
    "the other idle. Like any option, what prices it is not the level of the underlying "
    "but its **volatility**: a switching right is worth nothing if the coal-gas spread "
    "never moves, and a great deal if it swings.\n\n"
    "So the asset to value is the volatility of the coal-minus-gas generation spread. And "
    "that volatility contains one term almost nobody re-estimates.",
    formula="spread = coal_cost - gas_cost,  in EUR per MWh of electricity",
)

# ===========================================================================
# S2 — THE ALGEBRA, AND THE HEDGE IT CREATES
# ===========================================================================
section(
    "S2",
    "Gas and carbon enter the spread with opposite signs, so their correlation is a hedge",
    "Writing the spread out in its two market legs, the sensitivities are definitional "
    "rather than estimated — differentiate the generation-cost identity and read them off. "
    f"At the efficiencies set in the sidebar: **{b_ttf:+.3f} on TTF** and "
    f"**{b_eua:+.3f} on EUA**.\n\n"
    "Opposite signs. That single fact drives everything below, because the cross term in "
    "the variance carries the product of the two betas — and with `b_ttf < 0 < b_eua`, a "
    "**positive correlation subtracts from the variance**. Carbon is a natural hedge for "
    "gas inside this particular spread: when gas rallies and drags carbon up with it, the "
    "two effects on the spread partly cancel.\n\n"
    "This is not an empirical regularity that might reverse. Across the entire plausible "
    "efficiency grid `b_ttf` runs -2.000 to -1.667 and `b_eua` runs +0.410 to +0.611 — the "
    "signs never cross. Whatever else is uncertain about the switching level, the "
    "existence of this hedge is not.",
    formula=(
        "Var(spread) = (b_ttf x sigma_TTF)^2 + (b_eua x sigma_EUA)^2\n"
        "              + 2 x b_ttf x b_eua x rho x sigma_TTF x sigma_EUA\n"
        "                    ^^^^^^^^^^^^^^^ negative, so rho > 0 REDUCES the variance"
    ),
)
_b1, _b2 = st.columns(2)
_b1.metric("Sensitivity to TTF", f"{b_ttf:+.3f}", delta="per EUR/MWh of gas", delta_color="off")
_b2.metric("Sensitivity to EUA", f"{b_eua:+.3f}", delta="per EUR/t of carbon", delta_color="off")

# ===========================================================================
# S3 — WHAT THE HEDGE WAS WORTH
# ===========================================================================
section(
    "S3",
    "Eight years of a working hedge, then 2026",
    "The table below computes the spread's volatility twice each year: once as it actually "
    "was, and once in a counterfactual where the two legs keep their own volatilities but "
    "move independently. The gap between them is exactly what the carbon leg contributed "
    "as a hedge.\n\n"
    f"For eight years the answer is stable and material — a median "
    f"**{abs(hedge.typical_dampening):.0%} of the spread's volatility removed**, reaching "
    f"29% in 2024. Then {hedge.latest.year}: correlation {hedge.latest.rho:+.2f}, "
    f"dampening {hedge.latest.dampening:+.0%}. The hedge is simply gone.",
    formula="dampening = vol(actual) / vol(if independent) - 1",
)
_hedge_table = hedge.to_frame()
_hedge_fig = go.Figure()
_hedge_fig.add_trace(
    go.Bar(
        x=_hedge_table.index.astype(str), y=_hedge_table["dampening"],
        marker_color=[
            "crimson" if v >= -0.02 else "rgba(120,120,120,0.6)"
            for v in _hedge_table["dampening"]
        ],
        text=[f"{v:.0%}" for v in _hedge_table["dampening"]], textposition="outside",
    )
)
_hedge_fig.update_layout(
    title="Share of switching-spread volatility removed by the gas-carbon correlation",
    yaxis_title="dampening (negative = hedge working)", yaxis_tickformat=".0%",
    height=400, margin=dict(t=50, b=20, l=10, r=10), showlegend=False,
)
show(_hedge_fig)
st.dataframe(
    _hedge_table.rename(columns={
        "rho": "correlation", "vol_actual": "vol actual (EUR/MWh)",
        "vol_if_independent": "vol if independent",
    }).style.format({
        "correlation": "{:+.3f}", "vol actual (EUR/MWh)": "{:.1f}",
        "vol if independent": "{:.1f}", "dampening": "{:+.0%}", "n": "{:.0f}",
    }),
    width="stretch",
)
finding(hedge.headline)

# ===========================================================================
# S4 — THE ATTRIBUTION
# ===========================================================================
section(
    "S4",
    "The part worth trading is not the biggest part",
    "Spread volatility rose sharply into 2026, and most of that is not news: TTF's own "
    "volatility roughly doubled, which is in every market report written this year. "
    "Holding 2026's individual volatilities and swapping in 2025's correlation isolates "
    "what the lost hedge cost on its own.\n\n"
    f"**{1 - attribution.correlation_share:.0%} of the increase is individual "
    f"volatilities. {attribution.correlation_share:.0%} is the collapsed correlation.** "
    "The second number is the smaller one, and it is the only one a risk model carrying a "
    "historical rho will miss — because nothing in that model changes when a correlation "
    "silently goes to zero.\n\n"
    f"On an at-the-money switching option, near-linear in volatility, that omission is "
    f"worth **{attribution.option_value_uplift:+.0%}**. A dual-fuel fleet valued on a "
    "historical correlation, a tolling agreement priced off one, or any book carrying gas "
    "and EUA together under a stale covariance matrix — all three are wrong by roughly "
    "that much, in the same direction, with no error anywhere on screen.",
    formula="counterfactual = vol(2026 sigmas, 2025 rho)",
)
_t1, _t2, _t3 = st.columns(3)
_t1.metric(f"Spread vol {attribution.year_from}", f"{attribution.vol_from:.0f} EUR/MWh")
_t2.metric(
    f"{attribution.year_to} at {attribution.year_from} correlation",
    f"{attribution.vol_counterfactual:.0f}",
    delta="volatilities only", delta_color="off",
)
_t3.metric(
    f"Spread vol {attribution.year_to} actual", f"{attribution.vol_to:.0f}",
    delta=f"{attribution.correlation_part:+.0f} from the lost hedge",
)
finding(attribution.headline)
diagnostic_note(
    "**Read the split honestly.** This page is not claiming the correlation is the main "
    "driver of 2026 spread volatility — it plainly is not, gas is. The claim is narrower "
    "and, for a book, more useful: the correlation term is the one that moves without "
    "anyone noticing. A doubling of gas volatility gets re-marked immediately; a "
    "correlation parameter set from history does not get re-marked at all."
)

# ===========================================================================
# S5 — RULING OUT THE OBVIOUS CAUSE
# ===========================================================================
section(
    "S5",
    "It is not that the coal fleet ran out of room",
    "The reflex explanation for a broken gas-carbon link is saturation: if every available "
    "coal unit is already running, dearer gas cannot start another one, so the "
    "transmission channel is exhausted. It is the right hypothesis to test first, and the "
    "data rejects it.\n\n"
    "Saturation requires sitting **deep** above the switching level, not merely above it. "
    "The two years with nearly identical depth profiles have opposite correlations — and "
    "2022, the sample's other negative year, is the only genuinely saturated one.\n\n"
    "Nor is it a structural decline in the European coal fleet: **2024 has the strongest "
    "correlation in the whole sample**, at +0.70, which is late in that decline. Whatever "
    "changed in 2026 did not change gradually.",
)
st.dataframe(
    depth.rename(columns={
        "share_above": "share above switch", "median_distance": "median depth",
        "share_deep": "share beyond +40%", "rho": "correlation",
    }).style.format({
        "share above switch": "{:.0%}", "median depth": "{:+.0%}",
        "share beyond +40%": "{:.0%}", "correlation": "{:+.3f}", "n": "{:.0f}",
    }),
    width="stretch",
)
scope_note(
    "2018 sat above the switching level 87% of the time at a median depth of +7%, with "
    "**zero** days beyond +40% — and carbon tracked gas at +0.38. 2026 sits above 63% of "
    "the time at a median depth of +9%, with 1% of days beyond +40% — and the correlation "
    "is -0.09. Same room to switch, opposite transmission. Saturation is not the answer."
)

# ===========================================================================
# S6 — THE TRANSMISSION FAILURE
# ===========================================================================
section(
    "S6",
    "A crisis-sized gas shock happened and carbon did not take it",
    "A correlation can be dragged around by quiet days, so the cleanest version of the "
    "question is non-parametric: take each year's largest moves in gas and count how often "
    "carbon moved the same way that day. These are precisely the days on which the "
    "switching channel would have to be working.",
)
_trans = transmission.table
_trans_fig = go.Figure()
_trans_fig.add_trace(
    go.Bar(
        x=_trans.index.astype(str), y=_trans["same_sign"],
        marker_color=[
            "crimson" if v < 5 else "rgba(120,120,120,0.6)" for v in _trans["same_sign"]
        ],
        text=[f"{int(v)}/{int(n)}" for v, n in zip(_trans["same_sign"], _trans["n_shocks"])],
        textposition="outside",
    )
)
_trans_fig.update_layout(
    title=f"Carbon moving the same way as gas, on each year's {transmission.n_shocks} largest gas shocks",
    yaxis_title="same-direction days", yaxis_range=[0, 11],
    height=380, margin=dict(t=50, b=20, l=10, r=10), showlegend=False,
)
show(_trans_fig)
finding(transmission.headline)

st.divider()
st.markdown(
    "### The ceiling test this page was built on\n\n"
    "The sections that follow predate the trade above and are kept because they "
    "**constrain** it. They establish that the switching *level* is not identified — it "
    "depends on two plant efficiencies no exchange quotes — while the *sensitivities* used "
    "in S2 are definitional. That distinction is what lets the hedge result stand on "
    "ground the level result cannot."
)

# ===========================================================================
# S7
# ===========================================================================
section(
    "S7",
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
    "S8",
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

scope_note(
    f"Sampling every {horizon}th row also throws away {horizon - 1} of every {horizon} "
    f"possible starting points, and there is no reason to prefer the one it keeps. "
    f"Running all {phases.n_phases} phases: the coefficient is negative in "
    f"{phases.n_negative} of them and significant in {phases.n_significant}, ranging "
    f"{min(phases.betas):+.3f} to {max(phases.betas):+.3f}. The result is not an "
    "artefact of where the sample starts — which is the first thing that would have "
    "made it one."
)

# ===========================================================================
# S3 — THE RESULT
# ===========================================================================
section(
    "S9",
    "The regression is biased toward its own conclusion, and by how much",
    "Before the coefficient can be read, one objection has to be settled, and it is not a "
    "generic one — it is specific to this regressor and it points the same way as the "
    "finding.\n\n"
    "The regressor is `(TTF − switch)/switch`. **TTF sits in its numerator.** So when TTF "
    "rises over a window, the realised return and the regressor's own innovation move "
    f"together: they correlate {bias.corr_uv:+.2f} here. The regressor is also persistent "
    f"(rho = {bias.rho:.2f}). Those two facts together are the Stambaugh problem: the AR "
    "coefficient of a persistent series is itself biased downward in a finite sample "
    "(Kendall), and that bias transmits into the predictive coefficient in proportion to "
    "the innovation correlation.\n\n"
    f"Worked out on this sample the bias is **{bias.bias:+.4f}** — "
    f"**{bias.bias_share:.0%} of the coefficient**, and negative, meaning OLS is pushed "
    "toward exactly the negative number this page wants to report. That is the awkward "
    "case, and it is why a t-statistic alone would not be evidence here.",
    formula="E[beta_hat - beta] = (sigma_uv / sigma_vv) x E[rho_hat - rho],   E[rho_hat - rho] ~ -(1+3·rho)/n",
)

st.markdown("**The honest test — simulate a world with no predictability and look where the estimate falls:**")
st.markdown(
    "The bootstrap rebuilds the regressor through its own fitted AR(1) and resamples the "
    "innovation pairs *together*, so the simulated world keeps the persistence and the "
    "correlation that cause the bias — and contains no predictability whatsoever. Whatever "
    "coefficient shows up in that world is bias by construction."
)
b1, b2, b3 = st.columns(3)
b1.metric("Coefficient under the null", f"{boot.null_mean:+.4f}", help="Pure bias — nothing to predict in this world")
b2.metric("Bias from the formula", f"{bias.bias:+.4f}", help="Kendall's first-order term, computed independently")
b3.metric("Observed coefficient", f"{boot.beta_obs:+.3f}")
finding(boot.headline)
diagnostic_note(
    f"The two routes to the bias agree to {abs(boot.null_mean - bias.bias):.4f} — an "
    "analytic first-order formula and a simulation that never uses it. That agreement is "
    "the check that neither is a coding accident.\n\n"
    f"What it costs the headline: read naively off the t-statistic the result would be "
    f"significant at roughly p = 0.001. The honest p-value is **{boot.p_value:.3f}**. An "
    "order of magnitude of the apparent significance was bias, and what survives is real "
    "but only just."
)

# ===========================================================================
# S4 — the mechanism-specific prediction
# ===========================================================================
section(
    "S10",
    "The test a mean-reversion story cannot pass",
    "Surviving a bias correction says the relationship is there; it does not say it is "
    "*switching*. The trailing-median placebo in S3's predecessor is a weak discriminator, "
    "because it needs a second series and any two persistent series can be made to look "
    "different. There is a sharper test available, and it needs no second series at all.\n\n"
    "**The physics is one-sided.** Above the switching level, gas is the dearer fuel, "
    "generators burn coal instead, gas demand falls and TTF is pulled down. Below it, gas "
    "is already the cheap fuel — that is the normal state of the world, and nothing pushes "
    "TTF back *up* toward the line. So a genuine switching effect must appear on one side "
    "only. Ordinary mean reversion pulls symmetrically from both sides and cannot produce "
    "that pattern.",
    formula="forward_return = a + b_above x max(distance, 0) + b_below x min(distance, 0)",
)
if asymmetry is None:
    scope_note(
        f"**The asymmetry test cannot run at these settings.** {asymmetry_reason}.\n\n"
        "This is the efficiency-identification problem from S11 biting operationally: at "
        "an extreme efficiency pair the switching level moves far enough that almost "
        "every day lands on one side of it, and a test that compares the two sides has "
        "nothing to compare. Move the efficiency sliders back toward the middle of their "
        "range, or shorten the horizon to keep more windows, and it returns."
    )
else:
    a1, a2 = st.columns(2)
    a1.metric(
        f"Above the switch (n = {asymmetry.above.n_obs})",
        f"t = {asymmetry.above.t_stats['distance']:.2f}",
        delta=f"beta {asymmetry.above.coefficients['distance']:+.3f}",
    )
    a2.metric(
        f"Below the switch (n = {asymmetry.below.n_obs})",
        f"t = {asymmetry.below.t_stats['distance']:.2f}",
        delta=f"beta {asymmetry.below.coefficients['distance']:+.3f}",
        delta_color="off",
    )
    st.markdown("**Both slopes in one regression, kinked at the switching level:**")
    st.code(asymmetry.kinked.summary(), language="text")
    finding(asymmetry.headline)

# ===========================================================================
# S5 — the unifying result
# ===========================================================================
section(
    "S11",
    "The efficiencies decide where the line is, and nothing about what it predicts",
    "This project began as a complaint that the switching level is unknowable — it depends "
    "on two plant efficiencies no exchange quotes. That complaint is entirely correct, and "
    "the grid below shows how correct: across the plausible pair the level moves from "
    f"**{invariance.level_low:.0f} to {invariance.level_high:.0f} EUR/MWh**, and the share "
    f"of days sitting above it moves from **{invariance.share_low:.0%} to "
    f"{invariance.share_high:.0%}**. Same prices, same carbon, opposite diagnosis.\n\n"
    f"And yet the t-statistic of the prediction moves by **{invariance.t_swing:.2f}** across "
    "that entire grid. That is not a lucky cancellation. Collecting the switching formula "
    "on `lambda = eta_gas/eta_coal` — the rearrangement shown at the foot of this section — "
    "the efficiencies enter **only through lambda, and only affinely**, verified to "
    f"machine precision ({invariance.affine_residual:.1e}). The nine resulting distance "
    f"measures correlate {invariance.min_pairwise_corr:.3f} or better; they are one "
    "variable under an affine map. And an OLS t-statistic is invariant under an affine map "
    "of its regressor.\n\n"
    "So the two halves of this project are not in tension — they are two different "
    "functionals of the same object. **The level is location-dependent and therefore "
    "unidentified. The prediction is affine-invariant and therefore immune.** No better "
    "efficiency estimate would change either fact.",
    formula="ttf* = lambda x (coal_th + EUA x EF_coal) - EUA x EF_gas,    lambda = eta_gas / eta_coal",
)
grid = invariance.grid.copy()
grid.columns = ["coal eff.", "gas eff.", "switch level (EUR/MWh)", "share above", "beta", "t-stat"]
st.dataframe(
    grid.style.format(
        {
            "coal eff.": "{:.0%}", "gas eff.": "{:.0%}",
            "switch level (EUR/MWh)": "{:.1f}", "share above": "{:.0%}",
            "beta": "{:+.3f}", "t-stat": "{:.2f}",
        }
    ),
    width="stretch", hide_index=True,
)
finding(invariance.headline)

# ===========================================================================
# S6 — the honest negative
# ===========================================================================
section(
    "S12",
    "And the switching arithmetic is not what does the predicting",
    "If the efficiencies cannot matter, the natural next question is whether any of the "
    "construction matters. The competitor is deliberately crude: distance from **raw "
    "thermal parity** — coal's own price per thermal MWh, with no efficiency ratio and no "
    "carbon price anywhere in it. A measure that says nothing more than *gas is expensive "
    "relative to coal*.\n\n"
    f"It reaches R² = {encompassing.naive_only.r_squared:.3f} against "
    f"{encompassing.full_only.r_squared:.3f} for the full switching level. Put the full "
    "level into a regression that already contains the naive anchor and it adds "
    f"F = {encompassing.increment_f:.2f} (p = {encompassing.increment_p:.2f}) — nothing "
    f"the sample can distinguish from zero. The two regressors correlate "
    f"{encompassing.regressor_corr:.2f}.\n\n"
    "This is the result that should be reported loudest, because it is the one that cuts "
    "against the page's own construction. The ceiling is real and one-sided; **the "
    "evidence that it is specifically a *switching* ceiling, rather than a coal-gas "
    "relative-value ceiling, is not in this data.**",
)
st.markdown("**Encompassing regression — both anchors, same windows:**")
st.code(encompassing.both.summary(), language="text")
finding(encompassing.headline)

# ===========================================================================
# S7 — where the evidence actually comes from
# ===========================================================================
section(
    "S13",
    "Where the evidence comes from, and what would overturn it",
    "Split by period, the result is not evenly supported. The calm pre-crisis years do not "
    "carry it on their own; the crisis and its aftermath do. On a sample this small that "
    "is a limit to state, not a finding to interpret — with 26 to 45 windows per period, "
    "differences of this size are not separately identified.",
)
stability = subperiod_stability(frame, switch, horizon_days=horizon)
st.dataframe(
    stability.style.format({"beta": "{:+.3f}", "t_stat": "{:.2f}", "r_squared": "{:.3f}"}),
    width="stretch",
)
diagnostic_note(
    "**Three things would overturn what is on this page.** A longer calm-period sample "
    "showing the effect is confined to crises would reduce it to a crisis artefact. EU "
    "coal generation data — absent from this export — would allow the saturation test this "
    "page cannot run: coal capacity is finite, so the ceiling must stop binding once every "
    "available unit is already running, and the 2022 observations sit as far as +194% above "
    "the switching level. And a sample with enough above-switch windows to separate the "
    "full level from raw thermal parity would settle S6 one way or the other; 43 is not "
    "enough.\n\n"
    "Also deliberately missing: start-up costs, minimum stable generation, ramp rates and "
    "must-run constraints. A switching level is a necessary condition for the ceiling to "
    "bind, never a sufficient one."
)

# ===========================================================================
# S14
# ===========================================================================
section(
    "S14",
    "The project this replaces",
    "This engine was originally built around the API2 − API4 arb — Rotterdam against "
    "Richards Bay — and the thesis that the marginal South African tonne stopped pricing "
    "off Europe after 2022. **API4 is not in the export**, so that spread is not "
    "computable. Rather than substitute a proxy and keep the original headline, the "
    "question was changed to one this data can actually answer — and then tested until it "
    "gave up most of its apparent significance.",
)

mail_question(
    "Inside the coal-gas switching spread, gas and carbon enter with opposite signs, so a "
    "positive correlation is a natural hedge on the spread's volatility. From 2018 to 2025 "
    "it removed a median 12% of that volatility and as much as 29%. In 2026 the "
    "correlation collapsed to zero and the dampening went with it — on an ATM switching "
    "option that is roughly 11% of value, and it is the one input a risk model carrying a "
    "historical rho never re-marks.\n\n"
    "I tested the obvious cause and I do not think it holds. Saturation would need the "
    "fleet sitting deep above the switching level, and 2026 sits at a median +9% with 1% "
    "of days beyond +40% — almost exactly 2018's profile, when carbon tracked gas at "
    "+0.38. A structural coal decline does not fit either, since 2024 is the strongest "
    "correlation in the sample. On the ten largest gas shocks of 2026 carbon moved the "
    "same way 3 times, against 7 to 9 in every normal year.\n\n"
    "So a crisis-sized gas shock happened and the carbon market did not take it. Is that "
    "the Q1 LNG and weather moves being read as transient, so no implication for annual "
    "emissions? Is the generation stack hedged far enough forward that spot gas no longer "
    "changes near-term burn? Or has carbon simply been trading on its own calendar since "
    "January — CBAM's definitive regime, the cap and MSR reform? And separately: has the "
    "spread-option market repriced for this, or is correlation still being marked off "
    "history?",
    "European power and gas desks (Uniper, RWE, EDF Trading, Vitol, Glencore coal), "
    "utility fuel procurement, carbon desks",
)
