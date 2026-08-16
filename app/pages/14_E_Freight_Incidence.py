from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_APP_DIR.parent / "src"))
sys.path.insert(0, str(_APP_DIR))

from freight.chains.freight_incidence import (  # noqa: E402
    freight_share_of_value,
    incidence_by_frequency,
    lag_scan,
    load_incidence_frame,
    power_benchmark,
    predicted_lag_days,
)
from page_template import (  # noqa: E402
    Scope,
    diagnostic_note,
    finding,
    kpi_banner,
    mail_question,
    page_header,
    scope_note,
    section,
    show,
    snapshot_banner,
)

st.set_page_config(page_title="E — Freight incidence", layout="wide")

_LIVE = snapshot_banner()

# ===========================================================================
# Header and scope
# ===========================================================================
page_header(
    code="E",
    title="The buyer does not pay the freight, and the index says so",
    subtitle=(
        "The 62% Fe index is quoted CFR China, so the freight is already inside it — "
        "which turns the incidence question into a regression coefficient, and the "
        "answer is not the one the delivered-price convention implies"
    ),
    scope=Scope(
        unit_trap=(
            "The index is quoted per **dry** metric tonne and the freight per **wet** "
            "tonne — the same trap project A decomposes. It is deliberately **not** "
            "corrected here: this test runs on changes, where a constant moisture factor "
            "scales the coefficient by 1/(1−h) and moves no verdict. Applying it would "
            "imply a precision this test does not have. The trap that does bite is "
            "different and it is arithmetic: freight is a tenth of what a delivered tonne "
            "**costs** and a thirtieth of why its price **moved today**. Those are two "
            "statements, and only the first is usually made."
        ),
        conversion=(
            "CFR = FOB + freight        =>        d(CFR)/d(freight) IS the incidence\n"
            "\n"
            "  b = 1  the mill pays every dollar of freight\n"
            "  b = 0  the miner absorbs it into its netback, the delivered price is unmoved"
        ),
        proxies=[
            "C5 is a **front-month FFA**, not the spot route assessment (inherited from "
            "project A's A-H3). The basis between them attenuates any measured "
            "relationship toward zero — so it biases toward the null found here, which "
            "is the wrong direction for the conclusion and is stated rather than buried",
        ],
        out_of_scope=[
            "the 65% Fe index and route C3: predominantly Brazilian, and C3 is monthly "
            "in this export with 64 points — it cannot carry a daily test",
            "the FOB/CFR sales mix by miner, which is what the incidence ultimately "
            "depends on and is not public in the detail this would need",
        ],
        frequency_note=(
            "Both series are daily and the test runs at four horizons. The horizon is not "
            "a nuisance parameter here: it is the economics, because term contracts settle "
            "quarterly and the spot index does not."
        ),
        data_warnings=[
            "The IODEX series starts in January 2023, which leaves 14 quarterly "
            "observations. The quarterly horizon is reported and no verdict is drawn "
            "from it.",
        ],
    ),
)

# ===========================================================================
# Data
# ===========================================================================
frame = load_incidence_frame()
shares = freight_share_of_value(frame)
power = power_benchmark(frame)
scan = lag_scan(frame)
result = incidence_by_frequency(frame)
lag = predicted_lag_days()
daily = result.estimates[0]

kpi_banner(
    {
        "Freight, share of value": f"{shares['share_of_level']:.1%}",
        "Freight, share of variance": f"{shares['share_of_variance']:.1%}",
        "Daily coefficient": f"{daily.beta:+.2f}",
        "95% interval": f"[{daily.ci[0]:+.2f} ; {daily.ci[1]:+.2f}]",
        "Full pass-through": "rejected" if daily.rejects_full_passthrough else "not rejected",
    }
)

# ===========================================================================
# S1
# ===========================================================================
section(
    "S1",
    "A delivered price contains a freight bill, so the question has an arithmetic answer",
    "The 62% Fe index is assessed **CFR China**. That is not a detail of quotation: it "
    "means the number already includes the cost of moving the ore, and the miner has "
    "already paid it. So when freight rises, exactly one of two things happens. Either "
    "the delivered price rises with it and the Chinese mill funds the increase, or it "
    "does not and the miner's netback absorbs it.\n\n"
    "That makes the incidence a coefficient rather than a matter of opinion. Regress the "
    "change in the delivered price on the change in freight: a coefficient of one is the "
    "buyer paying, a coefficient of zero is the seller absorbing. No model of the ore "
    "market is required to ask the question — only the identity.\n\n"
    "The reason it is worth asking is that the convention answers it implicitly. Quoting "
    "a price CFR presents freight as something the buyer is charged for. Whether the "
    "market actually behaves that way is a separate question, and it is testable.",
    formula="CFR = FOB + freight     =>     d(CFR)/d(freight) is the incidence",
)

# ===========================================================================
# S2 — the two facts that get conflated
# ===========================================================================
section(
    "S2",
    "Freight is a tenth of the value and a thirtieth of the movement",
    f"Freight is **{shares['share_of_level']:.1%} of the delivered value** of a tonne of "
    "iron ore — economically large, and the figure usually quoted. But its daily changes "
    f"have a standard deviation of {shares['sigma_freight']:.2f} USD/t against "
    f"{shares['sigma_cfr']:.2f} for the delivered price itself. In variance terms freight "
    f"accounts for **{shares['share_of_variance']:.1%}** of what the CFR does.\n\n"
    "Both numbers are true and they answer different questions. A cost can be a large "
    "share of a price and a negligible share of that price's movement, and conflating the "
    "two produces the intuition that freight must be visible in the index because it is "
    "big inside it. It is big and it is quiet.\n\n"
    "This matters before any test, because it sets what a test can hope to see — which is "
    "the subject of the next section.",
)

# ===========================================================================
# S3 — the power argument
# ===========================================================================
section(
    "S3",
    "What full pass-through would have looked like, computed before looking",
    "The objection to any null result is that the test could not have seen the effect. "
    "That is answerable rather than arguable, and the answer follows from the identity "
    "rather than from an estimate.\n\n"
    "If the delivered price were the FOB plus a freight term, and the FOB moved "
    "independently of freight, then the correlation between their daily changes would be "
    "exactly the ratio of their volatilities — nothing is fitted, it is arithmetic. That "
    f"ratio is **{power.implied_correlation:.3f}**, against a significance band of "
    f"**{power.band:.3f}** on this sample.\n\n"
    f"So a coefficient of one would show up at {power.margin:.1f} times the noise floor. "
    "Whatever this test finds, it is not finding it because it was too weak to look.",
    formula="corr(dCFR, dFreight) under b=1  =  sigma(dFreight) / sigma(dCFR)",
)
finding(power.headline)

# ===========================================================================
# S4 — the lag, predicted not fitted
# ===========================================================================
section(
    "S4",
    "The timing objection, settled with the voyage rather than with a search",
    "The second objection is timing: freight is fixed when the ship is chartered, and the "
    "cargo is assessed at destination weeks later, so a same-day comparison would miss "
    "it. That is a real objection and it has a structural answer.\n\n"
    f"Port Hedland to Qingdao is {lag['distance_nm']:,.0f} nautical miles — "
    f"{lag['sea_days']:.1f} days at the reference speed, plus {lag['port_days']:.0f} port "
    f"days, so about **{lag['calendar_days']:.0f} calendar days** between fixing the "
    "freight and the cargo being assessed. That number comes from the voyage model, "
    "before any data is touched. Scanning the correlation for whichever lag looks best "
    "would manufacture the answer; predicting it first makes it falsifiable.\n\n"
    "Across the whole ±30-day window, no lag clears the band — including the predicted "
    "one, which reads "
    f"{scan.value_at_predicted:+.3f}. The absence is not a timing problem.",
)

lag_fig = go.Figure()
lag_fig.add_trace(
    go.Bar(
        x=scan.lags, y=scan.values,
        marker_color=[
            "crimson" if int(k) == scan.predicted_lag else "rgba(120,120,120,0.55)"
            for k in scan.lags
        ],
        name="correlation",
    )
)
for sign in (1, -1):
    lag_fig.add_hline(
        y=sign * scan.band, line_dash="dot", line_color="gray",
        annotation_text="95% band" if sign > 0 else "",
        annotation_position="right",
    )
lag_fig.add_vline(
    x=scan.predicted_lag, line_dash="dash", line_color="crimson",
    annotation_text=f"voyage predicts {scan.predicted_lag}d", annotation_position="top",
)
lag_fig.update_layout(
    title="Correlation of freight changes with delivered-price changes, by lag",
    xaxis_title="lag in days  (positive = freight leads)",
    yaxis_title="correlation",
    height=400, margin=dict(t=50, b=20, l=10, r=10), showlegend=False,
)
show(lag_fig)
finding(scan.headline)

# ===========================================================================
# S5 — THE RESULT
# ===========================================================================
section(
    "S5",
    "Where the sample has power, full pass-through is rejected",
    "With the timing objection settled and the power established, the coefficient can be "
    "read. At daily and weekly horizons — 882 and 186 differenced observations — it sits "
    "near zero and its interval excludes one. **The delivered price does not move with "
    "the freight inside it.**\n\n"
    "What it does not establish is the opposite. Zero sits comfortably inside every "
    "interval, but so does a third of a dollar per dollar. The honest statement is that "
    "full pass-through is ruled out and anything from full absorption to roughly a third "
    "is not.\n\n"
    "At monthly and quarterly frequency the point estimate turns positive and looks like "
    "the pass-through the daily test rejects. It should not be read: on 43 and 14 "
    "observations the intervals span from full absorption to twice full pass-through. "
    "That is the horizon where term contracts actually settle, and this sample cannot "
    "reach it.",
    formula="d(CFR) = a + b · d(C5)     HAC errors, differenced (E-H4)",
)

table = result.to_frame()
display = pd.DataFrame(
    {
        "n": table["n"],
        "coefficient": table["beta"].map(lambda v: f"{v:+.3f}"),
        "95% interval": [
            f"[{lo:+.2f} ; {hi:+.2f}]" for lo, hi in zip(table["ci_low"], table["ci_high"])
        ],
        "rejects b=1": table["rejects_full_passthrough"].map({True: "yes", False: "—"}),
        "rejects b=0": table["rejects_absorption"].map({True: "yes", False: "—"}),
        "verdict drawn": table["has_power"].map({True: "yes", False: "no — too few obs"}),
    }
)
st.dataframe(display, width="stretch")

coef_fig = go.Figure()
coef_fig.add_trace(
    go.Scatter(
        x=list(table.index), y=table["beta"],
        error_y=dict(
            type="data", symmetric=False,
            array=(table["ci_high"] - table["beta"]).to_numpy(),
            arrayminus=(table["beta"] - table["ci_low"]).to_numpy(),
            color="rgba(200,200,200,0.8)",
        ),
        mode="markers", marker=dict(size=12, color="crimson"), name="coefficient",
    )
)
coef_fig.add_hline(
    y=1.0, line_dash="dash", line_color="gray",
    annotation_text="the mill pays it all", annotation_position="right",
)
coef_fig.add_hline(
    y=0.0, line_dash="dash", line_color="gray",
    annotation_text="the miner absorbs it all", annotation_position="right",
)
coef_fig.update_layout(
    title="Freight incidence in the delivered price, by horizon",
    yaxis_title="coefficient on the freight change",
    height=420, margin=dict(t=50, b=20, l=10, r=10), showlegend=False,
)
show(coef_fig)
scope_note(
    "The interval widens by an order of magnitude as the horizon lengthens — not because "
    "the relationship changes but because three and a half years of index contains 882 "
    "days and 14 quarters. The right-hand points are drawn to show that they say nothing, "
    "not to be read."
)
finding(result.headline)

# ===========================================================================
# S6
# ===========================================================================
section(
    "S6",
    "Why the null is plausible rather than surprising",
    "Australian ore into China is the short haul of the seaborne trade — around "
    f"{lag['distance_nm']:,.0f} miles against roughly 11,000 from Brazil. Its freight is "
    "the smallest freight in the market it competes in, and a mill choosing between "
    "origins is comparing delivered prices in which the Australian freight term is both "
    "small and stable.\n\n"
    "In that setting a freight move does not shift the delivered price because it does "
    "not shift who the marginal supplier is. It moves the netback of a producer whose "
    "cash cost sits far below the price, and a producer that far inside the curve absorbs "
    "a cost change rather than passing it on — passing it on would cost volume, and "
    "volume is what an inframarginal producer is protecting.\n\n"
    "The corollary is testable and this sample cannot test it: the same exercise on the "
    "Brazilian leg, where freight is several times larger and the producer is closer to "
    "the margin, should show a different answer. That needs a daily C3, which this export "
    "does not have.",
)

# ===========================================================================
# Diagnostic
# ===========================================================================
diagnostic_note(
    "Two limits, both of which push against the conclusion rather than for it. C5 is a "
    "front-month FFA rather than the spot route (E-H2): the basis between them attenuates "
    "any true relationship toward zero, so it makes a null easier to find — the bias runs "
    "the wrong way for the result and is worth stating plainly. And the IODEX series "
    "begins in January 2023, so the quarterly horizon has 14 observations. Recovering the "
    "contract-horizon test needs a longer index, not a better method."
)

mail_question(
    "The 62% CFR index shows no freight pass-through at daily or weekly horizons — full "
    "pass-through is statistically rejected, and the test is strong enough to have seen "
    "it. Is that your experience: is freight a cost you absorb into the netback, or does "
    "it get renegotiated into the price on term business? And does the answer differ "
    "between your FOB book and your CFR book?",
    "Iron ore desks and marketing at Vale, Rio Tinto, BHP; Capesize freight desks at Glencore, Trafigura, Cargill",
)
