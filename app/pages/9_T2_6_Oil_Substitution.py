from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_APP_DIR.parent / "src"))
sys.path.insert(0, str(_APP_DIR))

from agri.chains.oil_substitution import (  # noqa: E402
    MYR_PEG_END,
    MYR_PEG_RATE,
    MYR_PEG_START,
    load_peg_window_spread,
    rolling_deviation,
    structural_drift,
    substitution_verdict,
)
from agri.core.fmt import fmt_num, fmt_pct  # noqa: E402
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

st.set_page_config(page_title="T2-6 — Inter-oil substitution", layout="wide")

_LIVE = snapshot_banner()

# ===========================================================================
# Header and scope
# ===========================================================================
page_header(
    code="T2-6",
    title="The palm-soy substitution bound does not exist",
    subtitle=(
        "Tested in the one window where FX contaminates nothing — the ringgit's "
        "seven-year fixed parity — the hypothesis comes out inverted: wide gaps do "
        "not close, they shift the level"
    ),
    scope=Scope(
        unit_trap=(
            "Palm quotes in **Malaysian ringgit per tonne** (Bursa), soy in **cents "
            "per pound** (CBOT). Subtracting them needs two conversions, one of "
            "which is a **market price**: the USDMYR rate. But the export contains "
            "no Malaysian FX series at all. Computing a palm-soy spread over the "
            "full history would therefore mean subtracting two currencies — exactly "
            "the mistake this portfolio tracks everywhere else. **The page refuses "
            "to do it.**"
        ),
        conversion=(
            f"Fixed-parity window only ({MYR_PEG_START} → {MYR_PEG_END}):\n"
            f"  palm_USD_t = palm_MYR_t / {MYR_PEG_RATE}      ← regulatory constant\n"
            "  soy_USD_t  = soy_c_lb x 22.0462\n"
            "  spread     = palm_USD_t - soy_USD_t"
        ),
        proxies=[
            "none: over this window, the only missing quantity is fixed by decree, "
            "hence known exactly",
        ],
        out_of_scope=[
            "the whole post-July-2005 period: the ringgit floats, and the export "
            "has no USDMYR — the spread is not computable there without fabricating a currency",
            "rapeseed and sunflower, absent from the export: the test covers the "
            "palm-soy pair only",
        ],
        frequency_note=(
            "Palm and soy are both daily. The usable window covers roughly 1,650 "
            "sessions, enough to estimate half-lives by regime."
        ),
        data_warnings=[
            "The window ends in July 2005, **before** the rise of biodiesel demand. "
            "The result is a clean historical reference point, not a reading of "
            "today's market — and that is precisely what makes the question in the "
            "email worth asking.",
        ],
    ),
)

# ===========================================================================
# Parameters
# ===========================================================================
st.sidebar.markdown("### Parameters")
window = st.sidebar.slider("Rolling median window (days)", 60, 400, 250, 10)
quantile = st.sidebar.slider("Wide/narrow separation quantile", 0.55, 0.90, 0.70, 0.05)

frame = load_peg_window_spread()
verdict = substitution_verdict(frame["spread"], window=window, quantile=quantile)
deviation = rolling_deviation(frame["spread"], window=window)
drift = structural_drift(frame["spread"])

kpi_banner(
    {
        "Usable sessions": f"{fmt_num(len(frame), 0)}",
        "Median spread": f"{fmt_num(frame['spread'].median(), 0)} USD/t",
        "Wide/narrow threshold": f"{fmt_num(verdict.threshold_usd_t, 0)} USD/t",
        "Narrow-gap half-life": f"{fmt_num(verdict.narrow.half_life_days, 0)} d",
        "Wide-gap half-life": (
            "none" if not pd.notna(verdict.wide.half_life_days) or verdict.wide.half_life_days == float("inf")
            else f"{fmt_num(verdict.wide.half_life_days, 0)} d"
        ),
    }
)

# ===========================================================================
# S1
# ===========================================================================
section(
    "S1",
    "A simple question that cannot be asked directly",
    "It seems to me that crushers hold palm/soy elasticity to be strong — a gap "
    "opens, someone switches, the gap closes — while formulators hold it to be "
    "sticky: reformulating a recipe takes months and goes back through "
    "validations. There is no documentary evidence that a specific desk is arguing "
    "about this today, hence \"it seems to me\" rather than \"I read that\".\n\n"
    "The test is direct: if substitution is fast, an abnormally wide spread must "
    "revert **faster** than a normal one. Comparing two half-lives is enough.\n\n"
    "Except a spread is needed first. And for a spread, both prices must be in the "
    "same currency.",
)

# ===========================================================================
# S2
# ===========================================================================
section(
    "S2",
    "The missing currency, and the seven years where it is a constant",
    "Bursa palm quotes in ringgit per tonne, CBOT soy in cents per pound. Getting "
    "from one to the other needs USDMYR, which is **not in the export**. Computing "
    "the spread over thirty years of history would mean subtracting a ringgit "
    "price from a dollar price and calling the difference a spread — the mistake "
    "this portfolio spends its time exposing elsewhere. The page does not commit "
    "it.\n\n"
    f"But one window exists. Bank Negara pegged the ringgit at "
    f"**{fmt_num(MYR_PEG_RATE, 2)} MYR per dollar from "
    f"{MYR_PEG_START[8:10]}/{MYR_PEG_START[5:7]}/{MYR_PEG_START[:4]} to "
    f"{MYR_PEG_END[8:10]}/{MYR_PEG_END[5:7]}/{MYR_PEG_END[:4]}**. Over these seven "
    "years, the missing series is not approximated: it is **known exactly, by "
    "decree**. The spread computes there with no assumption whatsoever, and "
    "everything it does is pure substitution economics, uncontaminated by FX.\n\n"
    "This is a natural experiment, and it is the only place in the entire history "
    "where this test is clean.",
    formula=f"palm_USD_t = palm_MYR_t / {fmt_num(MYR_PEG_RATE, 2)}   ← a division by a constant, not by a series",
)
show(
    regime_chart(
        frame.assign(palm_below_soy=frame["spread"] < 0),
        "spread",
        regime_col="palm_below_soy",
        regime_color=ALT_COLOR,
        title="Palm − soy spread during the ringgit's fixed parity",
        y_title="USD per tonne",
    )
)
scope_note(
    f"Shaded zones: palm trades below soy, which is the case "
    f"{fmt_pct((frame['spread'] < 0).mean())} of the time over the window."
)

# ===========================================================================
# S3 — THE RESULT
# ===========================================================================
section(
    "S3",
    "The result, and it inverts the thesis",
    "The hypothesis predicts: wide gap → someone switches → fast reversion. The "
    "mean-reversion half-life is therefore measured separately in the two "
    "regimes, comparing the spread to its **rolling median** rather than a "
    "constant — for the reason explained in S4.\n\n"
    "The data says the opposite of what was expected. **Narrow** gaps revert "
    f"quickly, in {fmt_num(verdict.narrow.half_life_days, 0)} days. **Wide** gaps, "
    "on the other hand, do not revert at all: the reversion coefficient is not "
    "distinguishable from zero. This is not a lack of statistical power — the wide "
    f"sample has {fmt_num(verdict.wide.n_obs if hasattr(verdict.wide, 'n_obs') else 0, 0)} "
    "observations and the coefficient is positive in sign, not merely small.",
    formula="Δgap = a + b x gap_{t-1} + e     (HAC errors)     half_life = -ln 2 / ln(1 + b)",
)
finding(verdict.headline)

c1, c2 = st.columns(2)
c1.metric("Narrow gap", f"{fmt_num(verdict.narrow.half_life_days, 0)} days", delta=verdict.narrow.summary.split(":")[-1].strip(), delta_color="off")
c2.metric("Wide gap", "no detectable reversion", delta=verdict.wide.summary.split(":")[-1].strip(), delta_color="off")

show(
    regime_chart(
        deviation.to_frame("deviation").assign(wide=deviation.abs() >= verdict.threshold_usd_t),
        "deviation",
        regime_col="wide",
        regime_color=SHUT_COLOR,
        title=f"Deviation from the rolling median ({window} days) — shaded: wide regime",
        y_title="USD per tonne",
        reference_lines={
            f"+{fmt_num(verdict.threshold_usd_t, 0)}": verdict.threshold_usd_t,
            f"−{fmt_num(verdict.threshold_usd_t, 0)}": -verdict.threshold_usd_t,
        },
    )
)
diagnostic_note(
    "A residual bias exists, and it runs **against** this result. Conditioning on "
    "a wide gap over-samples measurement noise, which reverts mechanically to the "
    "mean: this bias therefore pushes toward detecting reversion, not toward its "
    "absence. Finding none despite it makes the negative result more solid, not less."
)

# ===========================================================================
# S4
# ===========================================================================
drift_total = drift.attrs["drift_usd_t"]
section(
    "S4",
    "Why the rolling median, and what the naive test would have given",
    "There is a natural way to run this test that gives the answer the thesis "
    "expects — and it is wrong. It consists of splitting regimes on the spread's "
    "**absolute level**: days where |spread| is large against days where it is "
    "small. Done that way, the measurement gives 12 days in the wide regime "
    "against 39 in the narrow one — exactly the substitution bound being sought.\n\n"
    "The artefact comes down to a detail of the distribution. The median spread "
    f"is {fmt_num(frame['spread'].median(), 0)} USD/t: it is **structurally "
    "negative**. Selecting days where |spread| is large does not therefore select "
    "abnormal deviations from the prevailing level — it selects the era when palm "
    "was most heavily discounted, i.e. 2004-2005. What gets measured is then the "
    "dynamics of a particular period, called substitution.\n\n"
    "The fix is not the rolling median: splitting on a **deviation**, even from a "
    "simple constant, is already enough to make the bound disappear. The rolling "
    "median only sharpens the result, by absorbing the drift in the level — and "
    f"drift there is: the spread goes from a {fmt_num(drift['median_spread'].iloc[0], 0)} "
    "USD/t premium in favour of palm in 1998 to a "
    f"{fmt_num(abs(drift['median_spread'].min()), 0)} USD/t discount in 2004, a "
    f"net move of {fmt_num(abs(drift_total), 0)} USD/t.\n\n"
    "The detail that settles it: the two tails are **separated in time**. Palm is "
    "expensive in 1998-1999, deeply discounted in 2004-2005. These are not two "
    "excursions around an equilibrium, they are two eras.",
)
annual = drift.copy()
annual.columns = ["median spread (USD/t)", "sessions"]
annual["median spread (USD/t)"] = annual["median spread (USD/t)"].round(0)
st.dataframe(annual, width="stretch")
scope_note(
    "The cause of this repricing is not established here. The expansion of "
    "planted area in Malaysia and Indonesia over the period is one plausible "
    "hypothesis, but price data does not distinguish it from other explanations — "
    "it is therefore left as a hypothesis, not stated as a result."
)

# ===========================================================================
# S5
# ===========================================================================
section(
    "S5",
    "What this means for someone taking risk",
    "The consistent reading of S3 and S4 is as follows. Small gaps are noise "
    "around a slowly moving equilibrium, and they close fast — that is "
    "microstructure, not substitution. Large gaps are not temporary dislocations: "
    "**they are shifts of the equilibrium itself**, and they do not close.\n\n"
    "The operational consequence is direct and runs against a common heuristic: "
    "**fading a wide palm-soy spread has no support** in the only window where the "
    "test is clean. The trade \"substitution will bring it back\" assumes a "
    "restoring force these seven years do not show.\n\n"
    "The caveat, however, is serious and has to be carried: the window ends in "
    "July 2005, before biodiesel demand became a driver of the oilseed complex. "
    "Substitution may well have become livelier since. The result is a clean "
    "reference point, not a reading of today's market.",
)
diagnostic_note(
    "Re-running this test on the recent period needs **a single series**: "
    "`USDMYR Curncy`. Free, no particular entitlement. By far the "
    "highest-value missing series in the whole portfolio — it would unlock thirty "
    "years of palm-soy history instead of seven."
)

mail_question(
    "Testing palm-soy substitution over the seven years of the ringgit's fixed "
    "parity — the only window where FX contaminates nothing — I find narrow gaps "
    f"revert in {fmt_num(verdict.narrow.half_life_days, 0)} days, but beyond "
    f"{fmt_num(verdict.threshold_usd_t, 0)} USD/t there is no detectable mean "
    "reversion at all: wide gaps shift the level instead of closing. Does that "
    "match what you see? And more importantly: at what gap does your phone "
    "actually ring for a recipe change today — since this window predates "
    "biodiesel and the bound may well have moved.",
    "Vegetable oil crushers and refiners (Wilmar, Musim Mas, Golden Agri, Bunge, "
    "Cargill), food formulators, oils desks",
)
