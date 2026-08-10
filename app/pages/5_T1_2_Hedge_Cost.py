from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_APP_DIR.parent / "src"))
sys.path.insert(0, str(_APP_DIR))

from agri.chains.hedge_cost import (  # noqa: E402
    SHORT_HEDGE,
    HedgeParams,
    forced_exit_price,
    forced_exit_schedule,
    hedge_capacity,
    hedging_intensity,
    implied_margin_rate,
    load_real_hedge_frame,
    procyclicality,
)
from agri.data.bloomberg_loader import DEFAULT_PATH, load  # noqa: E402
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

st.set_page_config(page_title="T1-2 — Cost of the hedge", layout="wide")

_LIVE = snapshot_banner()

COMMODITIES = {
    "cacao_ny": ("ICE New York Cocoa", "cocoa_ny", "USD/t"),
    "cacao_londres": ("ICE London Cocoa", "cocoa_london", "GBP/t"),
    "cafe_arabica": ("ICE Arabica Coffee", "coffee_arabica", "c/lb"),
    "cafe_robusta": ("ICE Robusta Coffee", "coffee_robusta", "USD/t"),
}

st.sidebar.markdown("### Book and balance sheet")
commodity = st.sidebar.selectbox(
    "Commodity", list(COMMODITIES), format_func=lambda k: COMMODITIES[k][0]
)
book_kt = st.sidebar.slider("Physical book size (kt)", 10, 500, 100, 10)
line_musd = st.sidebar.slider("Credit lines (M USD)", 50, 1500, 250, 25)
book_t = book_kt * 1000.0
line_usd = line_musd * 1e6

label, series_key, unit = COMMODITIES[commodity]
params = HedgeParams(side=SHORT_HEDGE, book_size_t=book_t, credit_line_usd=line_usd)
simulation = load_real_hedge_frame(commodity, params=params)
# The price shown is the one the simulation ACTUALLY used, not a separate reload: the
# two coincide, but only the first survives snapshot mode.
price = simulation["front"].rename(series_key)
margin_rate = implied_margin_rate(simulation, book_size_t=book_t)

page_header(
    code="T1-2",
    title="The full cost of hedging — cocoa and coffee",
    subtitle=(
        "The price at which a house's balance sheet forces it to stop hedging, and "
        "therefore to stop buying physical — and why so many houses hit this wall at "
        "the same time"
    ),
    scope=Scope(
        unit_trap=(
            "**Variation margin is not a cost, it is a transfer** — what costs money is "
            "financing it. The cumulative hedging cost therefore excludes VM itself and "
            "keeps only its financing. Conflating the two turns a recoverable "
            "mark-to-market loss into a destruction of value."
        ),
        conversion=(
            "cash_t   = IM_t + max(0, uncompensated cumulative losses)\n"
            "cost_t   = cash_t x (rate + spread) / 360      <- 360-day basis, money-market convention\n"
            "P*       = (B/Q + P0) / (1 + margin_rate)      <- the forced-exit price"
        ),
        proxies=[
            "ICE initial-margin schedule unavailable → k·σ·F proxy calibrated on realised "
            f"volatility, an implied rate of {margin_rate:.1%} of price",
        ],
        out_of_scope=[
            "roll cost: the export only contains the generic front-month contract, no "
            "deferred maturity — the term is neutralised, not estimated",
            "intraday margin calls and haircuts on non-cash collateral, both of which "
            "**understate** the real cash requirement",
        ],
        frequency_note=(
            "ICE prices and SOFR are both daily. The sample starts on 2018-04-02, the "
            "start of SOFR coverage — before that, the real financing rate does not exist."
        ),
        data_warnings=[
            "London cocoa quotes in GBP/t and arabica in cents/lb: the thresholds "
            "computed are in the selected contract's native unit, not converted to USD/t.",
        ],
    ),
)

capacity = hedge_capacity(simulation, credit_line_usd=line_usd, book_size_t=book_t)
intensity = hedging_intensity(simulation, book_size_t=book_t)

kpi_banner(
    {
        f"Price ({unit})": f"{simulation['front'].iloc[-1]:,.0f}",
        "Historical peak": f"{simulation['front'].max():,.0f}",
        "Cash at peak": f"{simulation['cash_usd'].max()/1e6:,.0f} M USD",
        "Intensity at peak": f"{intensity.peak_ratio:.0%}",
        "Current capacity": f"{capacity.capacity_t.iloc[-1]/1000:,.0f} kt",
    }
)

# ===========================================================================
section(
    "S2",
    "The price at which the balance sheet forces an exit",
    "A house long physical and short futures loses cash when the price **rises** — "
    "exactly when its stock is gaining value. The threshold is computed in closed form "
    "rather than with a solver, because the closed form shows what it depends on: "
    "linearly on the line relative to book size, on the entry price, and barely on the "
    "margin rate. Beyond this price the house can no longer finance its hedge; it can "
    "therefore no longer hedge additional physical, and since nobody buys unhedged "
    "physical at this level of volatility, **it stops buying**. This is the point where "
    "a trading liquidity crisis becomes a stop in farm-gate buying.",
    formula="P* = (B/Q + P0) / (1 + margin_rate)",
)

inception_choices = ["2018-04-02", "2022-01-03", "2023-01-03", "2023-09-01", "2024-01-02"]
schedule = forced_exit_schedule(
    price, inception_choices, book_size_t=book_t, credit_line_usd=line_usd, im_rate=margin_rate
)

span_days: int | None = None
median_exit: float | None = None

if not schedule.empty:
    median_exit = float(schedule["forced exit"].median())

if not schedule.empty and schedule["crossed on"].notna().any():
    crossed = schedule["crossed on"].dropna()
    span_days = int((crossed.max() - crossed.min()).days)
    earliest = schedule.loc[schedule["opened"].idxmin()]
    latest = schedule.loc[schedule["opened"].idxmax()]
    gap_years = (latest["opened"] - earliest["opened"]).days / 365.25
    finding(
        f"A house hedged since {earliest['opened']:%b %Y} and one hedged since "
        f"{latest['opened']:%b %Y} — {gap_years:.0f} years apart at entry — are forced "
        f"to exit **{span_days} days apart**. The move was violent enough to crush the "
        "spread of opening dates: that is why so many houses hit the balance-sheet "
        "constraint at the same time rather than each in their own turn."
    )
else:
    finding(
        "On this sample and this parameterisation, the market never reached the "
        "forced-exit price — the line was never the binding constraint."
    )

# `.round()` targets numeric columns only: applying it to the whole frame warns on the
# date columns and leaves them untouched.
display = schedule.copy()
display["headroom"] = display["headroom"].map(lambda v: f"{v:+.0%}")
numeric = display.select_dtypes("number").columns
display[numeric] = display[numeric].round(0)
st.dataframe(display, width="stretch", hide_index=True)
scope_note(
    "\"Days of protection\" = time between opening the hedge and crossing the threshold. "
    "The ordering holds — hedging early is still better — but the gap is measured in "
    "weeks, not years."
)

# ===========================================================================
section(
    "S3",
    "Cash tied up, relative to the stock it protects",
    "This is the quantity Montesanto Tavares's lawyers put a number on in November "
    "2025: the cost of carrying hedges had gone from **74% of trade receivables in May "
    "to 158% in November**, and they called it unsustainable. The same measure is "
    "rebuilt here without needing anyone's balance sheet — it starts by posting only "
    "the initial margin, then variation margin stacks up until it ties up almost as "
    "much cash as the stock itself is worth.",
    formula="intensity_t = cash mobilised_t / (Q x price_t)",
)
finding(intensity.headline)
show(
    regime_chart(
        intensity.ratio.to_frame("intensity").assign(above=intensity.ratio > 0.5),
        "intensity", regime_col="above",
        title="Cash tied up / physical book value",
        y_title="ratio", zero_line=False, reference_lines={"100%": 1.0},
    )
)

# ===========================================================================
section(
    "S4",
    "Procyclicality — the cost spikes exactly when the hedge is needed most",
    "Initial margin tracks the price: it rises when the market runs, which is exactly "
    "when a house long physical needs the hedge most. The correlation is measured on "
    "**differenced** series — two non-stationary level series would produce a "
    "flattering correlation that means nothing.",
)
stats = procyclicality(simulation)
c1, c2 = st.columns(2)
c1.metric("Corr Δ(initial margin) / Δ(price)", f"{stats['corr_delta_im_delta_price']:+.2f}")
c2.metric("Observations", f"{stats['n_obs']:,}")
show(
    regime_chart(
        capacity.capacity_t.to_frame("capacity") / 1000,
        "capacity",
        title="Maximum hedgeable book with available lines (kt)",
        y_title="kt", zero_line=False, reference_lines={f"book {book_kt} kt": float(book_kt)},
    )
)
if capacity.is_binding:
    diagnostic_note(
        f"The line becomes binding: capacity falls to "
        f"{capacity.min_capacity_t/1000:,.0f} kt against a book of {book_kt} kt, down "
        f"{capacity.contraction_over(4):.0%} over the four months before the peak on "
        f"{capacity.peak_cash_date:%d %b %Y}."
    )
else:
    scope_note("On this parameterisation, the line stays above the book across the whole sample.")

# ===========================================================================
section(
    "S5",
    "Breaking down the hedging cost",
    "Three components, kept separate because they do not offset each other the same "
    "way: financing depends on the level of cash tied up, the roll on the term "
    "structure, liquidity on how often the position rolls. Here only financing is "
    "real — the export has no deferred maturity, so the roll and liquidity terms are "
    "neutralised rather than estimated on an invented curve assumption.",
)
components = pd.DataFrame(
    {
        "financing": simulation["financing_usd"].cumsum() / book_t,
        "roll": simulation["roll_usd"].cumsum() / book_t,
        "liquidity": simulation["liquidity_usd"].cumsum() / book_t,
    }
)
show(
    regime_chart(
        components.assign(total=components.sum(axis=1)), "total",
        title=f"Cumulative hedging cost ({unit} of book)", y_title=unit, zero_line=True,
    )
)
neutralised = [
    name for name in ("roll", "liquidity") if components[name].abs().max() == 0
]
if neutralised:
    scope_note(
        f"Terms neutralised for lack of real data: {', '.join(neutralised)}. "
        "Omitting them **understates** the total cost — the bias runs against the "
        "thesis, which is the right direction."
    )

exit_text = f"around {median_exit:,.0f} {unit}" if median_exit is not None else "not reached over the sample"
span_text = (
    f" — and, notably, crossing dates bunched within {span_days} days regardless of "
    "that opening date"
    if span_days is not None
    else ""
)
mail_question(
    f"On {label}, with {line_musd} M USD of lines and a {book_kt} kt book, I find a "
    f"forced-exit price {exit_text} depending on when the hedge was opened{span_text}. "
    "Does your limit trigger around that level, and is it formalised, or does it get "
    "discovered along the way?",
    "ofi/Olam, ECOM, Volcafe, Sucden Coffee, Touton, Barry Callebaut, Cargill Cocoa, Freepoint softs",
)
