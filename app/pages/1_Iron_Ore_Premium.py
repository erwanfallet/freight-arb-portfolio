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
from freight.chains.ironore import (  # noqa: E402
    DEFAULT_MOISTURE_AUSTRALIA,
    DEFAULT_MOISTURE_BRAZIL,
    implied_origin_weight,
    load_real_premium_frame,
    evaluate_origin_shorthand,
    premium_attribution,
    third_origin_arithmetic,
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

st.set_page_config(page_title="A — Iron ore 65-62 premium", layout="wide")

_LIVE = snapshot_banner()

# ===========================================================================
# Header and scope
# ===========================================================================
page_header(
    code="A",
    title="The 65–62 premium contains a freight spread — but at half weight",
    subtitle=(
        "Taken at full strength the origin shorthand implies that high-grade ore is cheaper "
        "at the loadport than low-grade ore, and the moisture correction makes that worse "
        "rather than better"
    ),
    scope=Scope(
        unit_trap=(
            "**Freight is paid on the wet tonne shipped; the index is quoted on the dry "
            "tonne delivered.** Iron ore leaves Brazil at around 9 % moisture and Australia "
            "at around 8 %, so moving one dry tonne costs `freight / (1 - moisture)`, not "
            "`freight`. Everyone knows this correction exists; the interesting part is that "
            "applying it here **deepens** the contradiction the page finds instead of "
            "resolving it — which is worth more than a correction that had tidied things up."
        ),
        conversion=(
            "freight_per_dry_tonne = freight_per_wet_tonne / (1 - moisture)\n"
            "premium_CFR           = (FOB_65 - FOB_62) + (C3 - C5)\n"
            "implied_quality_FOB   = premium - freight_dry        [the shorthand at w = 1]\n"
            "implied_weight        = (premium - quality) / freight_dry"
        ),
        proxies=[
            "the origin mapping itself — 65 % Fe treated as Brazilian on C3 and 62 % as "
            "Australian on C5 — is a shorthand, and testing it is the subject of the page "
            "rather than an assumption of it",
        ],
        out_of_scope=[
            "lump and pellet premia, blast-furnace productivity and coke rate economics: all "
            "real drivers of the quality differential, none of them separable with price "
            "series alone",
            "Chinese port stocks and mill margins, which would explain the level of the "
            "premium but not its freight content",
        ],
        frequency_note=(
            "**The binding constraint of this page.** C5 is daily with 3 130 observations; "
            "the C3 route in this export has 64 points at a monthly step. Forward-filling C3 "
            "onto a daily grid would turn 31 real observations into 600 and make every "
            "standard error meaningless. Both legs are therefore taken down to the monthly "
            "grid the coarser series actually supports — 31 months. That is a small sample, "
            "and it is stated rather than disguised."
        ),
        data_warnings=[
            "The 65 % Fe series is labelled as a dated September 2026 contract in the "
            "export, yet its ratio to the 62 % index stays inside 1,11–1,19 with comparable "
            "daily volatility, which is spot-tracking behaviour rather than a converging "
            "forward. Treated as an index here, with the caveat left visible.",
        ],
    ),
)

# ===========================================================================
# Parameters
# ===========================================================================
st.sidebar.markdown("### Parameters")
moisture = st.sidebar.slider(
    "Moisture on the freight leg", 0.0, 0.15, DEFAULT_MOISTURE_BRAZIL, 0.005,
    help="Freight is paid on the wet tonne, the index is quoted on the dry tonne.",
)
quality_assumed = st.sidebar.slider(
    "Assumed FOB quality differential (USD/t)", 0.0, 14.0, 6.0, 0.5,
    help="What three extra points of Fe are worth at the loadport, before any freight.",
)

frame = load_real_premium_frame()
tested = evaluate_origin_shorthand(frame, moisture=moisture)
weights = implied_origin_weight(frame, moisture=moisture)
# The attribution runs at the weight the user's own quality assumption implies, so the
# slider drives the 2026 answer rather than a constant buried in the page.
attribution = premium_attribution(
    frame,
    weight=max(0.0, min(1.5, weights.weight_at(quality_assumed))),
    year_from=2025,
    year_to=2026,
    moisture=moisture,
)
third_origin = third_origin_arithmetic()

kpi_banner(
    {
        "Months available": f"{len(frame)}",
        "65–62 premium": f"{frame['premium'].median():.2f} USD/t",
        "C3 − C5 (dry)": f"{tested.frame['freight_dry'].median():.2f} USD/t",
        "Implied FOB quality": f"{tested.implied_quality_median:+.2f} USD/t",
        "Months negative": f"{tested.share_negative:.0%}",
    }
)

# ===========================================================================
# S1
# ===========================================================================
section(
    "S1",
    "A shorthand that everyone uses and nobody checks",
    "The 65–62 Fe premium is read on steel desks as a quality signal: when mill margins are "
    "good, mills pay up for high-grade ore because it lifts blast-furnace productivity and "
    "cuts coke rate. Freight desks make a different observation. High-grade supply is "
    "largely Brazilian and travels on route C3; the 62 % benchmark is largely Australian and "
    "travels on C5. Brazil to China is roughly three times the distance of Western Australia "
    "to China.\n\n"
    "Both indices are quoted **CFR China** — delivered. So the freight differential is "
    "already inside the premium before any quality argument begins:\n\n"
    "That gives a clean identity, and an obvious test. Take the shorthand at full strength, "
    "subtract the freight differential from the premium, and whatever remains has to be what "
    "three extra points of iron are worth at the loadport. That number cannot be negative.",
    formula="premium_CFR = (FOB_65 - FOB_62) + (C3 - C5)",
)

# ===========================================================================
# S2 — the test fails
# ===========================================================================
section(
    "S2",
    "The test, and it fails",
    "On the thirty-one months where all four series overlap, the freight differential is not "
    "a fraction of the premium — it is **larger than the whole premium**. Subtracting it "
    "leaves an implied FOB quality differential that is negative in most months.\n\n"
    "Read literally, that says Brazilian high-grade ore is cheaper at the loadport than "
    "Australian low-grade ore. No one in the market believes that, and no producer would "
    "sell on those terms. So the shorthand is wrong somewhere — and the useful work is "
    "finding out where, not discarding the result.",
)
finding(tested.headline)
show(
    regime_chart(
        tested.frame.assign(negative=tested.frame["implied_quality"] < 0),
        "implied_quality",
        regime_col="negative",
        regime_color=SHUT_COLOR,
        title="Implied FOB quality differential once the freight spread is removed at full weight",
        y_title="USD per dry tonne",
    )
)
scope_note(
    "Shaded months are the ones where the shorthand implies a negative quality differential. "
    "The series is monthly because C3 is monthly — see the frequency note in the scope box."
)

# ===========================================================================
# S3 — the correction makes it worse
# ===========================================================================
section(
    "S3",
    "The moisture correction deepens the problem instead of fixing it",
    "The obvious suspect is the unit trap this project was built around. Freight is quoted "
    "and paid per **wet** tonne loaded; the index is assessed per **dry** tonne of contained "
    "ore. Delivering one dry tonne therefore means paying freight on rather more than one "
    "tonne, and the cost per dry tonne is `freight / (1 - moisture)`.\n\n"
    "Applying it correctly makes the freight leg **larger**, so it makes the implied quality "
    "differential **more negative**. At zero moisture the shorthand already fails in 61 % of "
    "months; at a Brazilian 9 % it fails in 77 %.\n\n"
    "That matters more than a correction that had rescued the story. It rules out the "
    "explanation a reader would reach for first, and it forces the problem back onto the "
    "assumption nobody was examining — the origin mapping itself.",
    formula="freight_per_dry_tonne = freight_per_wet_tonne / (1 - moisture)",
)
sensitivity = pd.DataFrame(
    [
        {
            "moisture": value,
            "freight per dry tonne (USD/t)": float(
                evaluate_origin_shorthand(frame, moisture=value).frame["freight_dry"].median()
            ),
            "implied FOB quality (USD/t)": evaluate_origin_shorthand(
                frame, moisture=value
            ).implied_quality_median,
            "months negative": evaluate_origin_shorthand(frame, moisture=value).share_negative,
        }
        for value in (0.0, 0.04, DEFAULT_MOISTURE_AUSTRALIA, DEFAULT_MOISTURE_BRAZIL, 0.12)
    ]
)
st.dataframe(
    sensitivity.style.format(
        {
            "moisture": "{:.1%}", "freight per dry tonne (USD/t)": "{:.2f}",
            "implied FOB quality (USD/t)": "{:+.2f}", "months negative": "{:.0%}",
        }
    ),
    width="stretch", hide_index=True,
)
diagnostic_note(
    "A correction that worsens the anomaly it was supposed to explain is a result, not a "
    "failure. It eliminates the moisture convention as the culprit and leaves exactly one "
    "candidate standing."
)

# ===========================================================================
# S4 — THE DELIVERABLE
# ===========================================================================
section(
    "S4",
    "The inversion: how much freight the premium can actually carry",
    "The shorthand assumes the freight spread enters the premium at **full weight** — every "
    "tonne of 65 % Fe pays C3, every tonne of 62 % pays C5. Neither index is single-origin, "
    "so that cannot be right. Rather than guess the origin mix, assume the FOB quality "
    "differential a practitioner actually believes in and solve for the weight:\n\n"
    "An iron ore desk knows what three points of Fe are worth at the loadport. It reads its "
    "own answer straight off the curve below, and the answer is the honest version of this "
    "project's thesis: **the freight spread is inside the premium, at roughly half weight, "
    "not at full weight.**",
    formula="w = (premium - quality_FOB) / freight_dry        w = 1 is the full shorthand",
)
finding(weights.headline)

show(
    regime_chart(
        weights.curve.set_index("quality_usd_t"),
        "implied_weight",
        title="Weight at which the freight spread enters the premium, by assumed FOB quality",
        y_title="implied weight (1 = full shorthand)",
        zero_line=True,
        reference_lines={"full shorthand w = 1": 1.0, "half weight": 0.5},
    )
)
c1, c2, c3 = st.columns(3)
c1.metric("At your assumption", f"w = {weights.weight_at(quality_assumed):.2f}")
c2.metric("Full shorthand implies", "w = 1.00")
c3.metric(
    "Overstatement factor",
    f"{1.0 / max(weights.weight_at(quality_assumed), 0.01):.1f}×",
    delta="how much the shorthand exaggerates the freight content",
    delta_color="off",
)
scope_note(
    "The curve is mechanical, not estimated: it is the identity solved for w, month by "
    "month, with the median plotted. It carries no standard error because there is nothing "
    "to estimate — the uncertainty lives entirely in the quality differential you assume, "
    "which is why that is the slider."
)

# ===========================================================================
# S5 — applying the weight instead of admiring it
# ===========================================================================
section(
    "S5",
    "2026: the premium rose, and the quality differential did not",
    "The weight is an input, not a finding. What it is *for* is attribution — splitting a "
    "move in the premium into the part freight explains and the part it does not, which is "
    "the only way to tell a quality story from a shipping story when both are moving.\n\n"
    f"Between 2025 and 2026 the premium rose **{attribution.premium_change:+.2f} USD/t**, "
    "which reads on its own as high-grade ore getting more valuable. Over the same period "
    f"the moisture-corrected freight spread rose **{attribution.freight_change:+.2f}**. At a "
    f"weight of {attribution.weight:.2f}, freight alone accounts for "
    f"**{attribution.freight_part:+.2f}** — more than the entire move. The residual, which "
    "is what the underlying FOB quality differential must have done, is "
    f"**{attribution.residual_part:+.2f}**.\n\n"
    "So on this reading the premium went up for the wrong reason: the market paid more to "
    "deliver high-grade ore, not more for the ore itself. A desk marking the headline "
    "premium would see a strengthening number over a flat-to-softening one.",
    formula="d(premium) = d(quality_FOB) + w x d(freight_dry)",
)
a1, a2, a3 = st.columns(3)
a1.metric("Premium move, 2025 to 2026", f"{attribution.premium_change:+.2f} USD/t")
a2.metric(
    f"Explained by freight (w = {attribution.weight:.2f})",
    f"{attribution.freight_part:+.2f} USD/t",
)
a3.metric(
    "Residual — the quality differential",
    f"{attribution.residual_part:+.2f} USD/t",
    delta="negative means high-grade got relatively cheaper",
    delta_color="off",
)
diagnostic_note(
    f"**This sits on a knife edge and the page will not hide it.** The residual changes "
    f"sign at a weight of {attribution.threshold_weight:.3f}, and the weight implied by a "
    f"6 USD/t quality assumption is {weights.weight_at(6.0):.2f} — barely above it. Read "
    "strictly, the honest statement is not \"quality fell\" but **\"essentially all of the "
    "2026 premium rise was freight, and the quality differential was flat at best\"**. "
    "Anyone assuming a materially lower quality differential than 6 USD/t gets the opposite "
    "sign, which is precisely why the assumption is a slider rather than a constant."
)

# ===========================================================================
# S6 — the third origin
# ===========================================================================
section(
    "S6",
    "The two-origin world this page is built on ended in January 2026",
    "Everything above assumes ore reaches China from two places: Brazil on C3 and Australia "
    "on C5. The first Simandou cargo landed in China in January 2026, and Guinea is neither "
    "of those.\n\n"
    "There is no Guinea-China Capesize benchmark to look up — which is itself the point — so "
    "the distance is derived from two published facts and checked against the two routes "
    "this portfolio already uses. Drewry puts the Guinea round voyage beyond 90 days against "
    "30-35 for Australia, and the distance at roughly three times the Western Australian "
    "one. At 12 knots with 8 round-trip port days those give ~11,800 nm and ~10,500 nm "
    "respectively, and the same arithmetic reproduces Australia's 3,500 nm from its own "
    "voyage time. Two independent statements agree, and one of them recovers a number "
    "already in the repository.\n\n"
    "**The consequence is not the obvious one.** Simandou is not cheap high-grade ore. At a "
    "Brazil-length haul it arrives carrying a Brazil-sized freight bill, not a smaller one. "
    "What it changes is the fleet: roughly 180 Capesizes to move 120 Mtpa from Guinea "
    "against 64 for the same tonnage out of Australia.",
    formula="Guinea ~11,800 nm    Brazil 11,000 nm    Australia 3,500 nm",
)
d1, d2, d3 = st.columns(3)
d1.metric("Guinea to Qingdao", f"{third_origin.guinea_nm:,.0f} nm", delta="derived, not published", delta_color="off")
d2.metric("vs the Australian haul", f"{third_origin.guinea_vs_australia:.1f}x")
d3.metric("vs the Brazilian haul", f"{third_origin.guinea_vs_brazil:.2f}x")
finding(third_origin.headline)

section(
    "S7",
    "Two effects, opposite signs, one observed number",
    "Simandou pushes the 65-62 premium in both directions at once, and the decomposition on "
    "this page is what separates them.\n\n"
    "**Down, through supply.** More 65 % Fe ore competing for the same buyers compresses the "
    "FOB quality differential — the residual term in S5. Fastmarkets' own 2026 outlook puts "
    "it plainly: Simandou gives a cap to the 65 % premium.\n\n"
    "**Up, through freight.** Serving Guinea consumes roughly three times the fleet capacity "
    "per tonne that Australia does. That tightens Capesize, widens C3 - C5, and at any "
    "weight above zero widens the observed premium — with no improvement whatsoever in what "
    "a miner earns for grade.\n\n"
    "A desk watching the headline number sees only the net. The 2026 attribution in S5 is "
    "what that looks like when the freight leg is winning: a premium that rises while the "
    "thing it is supposed to measure does not.",
)
scope_note(
    "The index methodology is moving underneath this too. Fastmarkets launched a separate "
    "61 % Fe index in 2025 to track the drift in Australian mid-grade fines, and has an open "
    "consultation on the 65 % Fe specification. Whether 65 % Fe stays one blended price or "
    "fragments by origin the way the mid-grades just did is being decided now — and it "
    "decides whether the two-origin decomposition on this page still has a slot for every "
    "tonne in the index."
)

# ===========================================================================
# S8
# ===========================================================================
section(
    "S8",
    "What price data cannot settle",
    "Two readings of the half-weight result remain, and no price series separates them. "
    "Either the **origin mix** is the answer — both indices draw on both origins, diluting "
    "the effective freight differential — or the **routes are wrong** for the physical flow "
    "that actually sets each index, in which case C3 and C5 are the wrong benchmarks rather "
    "than the wrong weights.\n\n"
    "Both produce identical arithmetic. What separates them is cargo-level origin data for "
    "the tonnes that actually price each index, which an iron ore desk has and an outsider "
    "does not. Simandou makes that gap wider rather than narrower: a third origin with no "
    "freight benchmark of its own, entering an index whose specification is under review.",
)
diagnostic_note(
    f"Sample size is the honest limit: {len(frame)} months, because the C3 series in this "
    "export is monthly. Every number here should be read as an order of magnitude rather "
    "than a precise estimate. The 2026 attribution rests on seven months and a weight that "
    "cannot be measured, only bounded — it is a framework applied to a live episode, not a "
    "backtested signal, and nothing on this page is presented as one."
)

mail_question(
    "Taking the 65-62 CFR premium and subtracting C3 - C5 at full weight gives an implied "
    f"FOB quality differential of {tested.implied_quality_median:+.1f} USD/t, negative in "
    f"{tested.share_negative:.0%} of months, so the shorthand must overstate the freight "
    f"content; a 6 USD/t quality assumption puts the effective weight near "
    f"{weights.weight_at(6.0):.2f}. Applying that to 2026, essentially the whole rise in the "
    "premium is freight rather than grade.\n\n"
    "What I cannot work out from prices is where Simandou leaves this. It is a "
    "Brazil-length haul, so it adds high-grade supply and Capesize tonne-miles at the same "
    "time — one pushing the premium down, the other up. Do you expect the 65 % Fe index to "
    "fragment by origin the way the mid-grades just did with the new 61 %, or to stay a "
    "blended price that quietly absorbs a Guinea-Brazil freight gap? And is there any "
    "price discovery on Guinea-China freight yet, or is it still moving on relet and COA "
    "terms nobody outside sees?",
    "Iron ore desks at Vale, Rio Tinto, BHP, Glencore, Trafigura; Capesize freight desks; "
    "steel raw materials procurement",
)
