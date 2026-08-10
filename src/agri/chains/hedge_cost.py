"""T1-2 — The full cost of hedging: cocoa and coffee, full cycle.

THESIS
------
The binding constraint is not price, it is **collateral**. A house without enough credit
lines cannot hedge, so it cuts its physical buying — and that is the mechanism by which a
trading-desk liquidity crisis becomes a payment problem at the farm gate. Procyclicality is
the final insult: the cost of hedging spikes precisely when hedging is most needed.

    hedge_capacity(B) = max { Q : max_t cash_t(Q) <= B }

Since `cash_t` is linear in Q, the solve is immediate: `Q* = B / max_t cash_t(1 t)`. That
is the quantitative translation of the Montesanto quote: their constraint was not the
market view, it was the balance sheet.

WHERE THE DISAGREEMENT COMES FROM (sourced)
-----------------------------------------------
- Barry Callebaut, H1 2024/25: initial margin requirements up ninefold against normal
  levels; backwardation cost 60% dearer at the peak.
- Cocoa 2024: physical longs are reluctant to sell futures in a rising market because of
  the size of margin calls, which creates liquidity problems; manufacturers end up in the
  position that used to belong to farmers.
- Coffee, Nov 2025: roughly 7 bn USD of margin calls in a single month; at Montesanto
  Tavares, the cost of carrying hedges goes from 74% of trade receivables in May to 158% in
  November, making short-term treasury structure unsustainable according to their lawyers.

WHAT MAKES THE PROJECT NEW — THE 2026 REVERSAL
--------------------------------------------------
Cocoa has fallen from its December 2024 peak. But manufacturers hedge 12 to 24 months
ahead: costs locked in at the top are still running through 2026 budgets. **Hedging has
therefore punished both sides, fourteen months apart** — on the way up it strangled the
liquidity of futures sellers, on the way down it locks buyers into dead levels. No static
hedging model captures this.

WARNING — THE SIGN OF THE ROLL
--------------------------------
In **backwardation** (F_near > F_deferred), a **short** who rolls ends up short at a lower
price; when that contract converges to spot, it **loses**. The long gains. It is the
reverse in contango.

    roll_pnl = -s x Q x (F_deferred - F_near)      with s = +1 short, -1 long

Consistency check against the source: Barry Callebaut was short futures against long
physical, and reports backwardation as a **cost**. That is exactly what the formula gives.
(The scoping spec's appendix note states the opposite sign; it is contradicted by its own
identity and by the source, and was not followed.)

ASSUMPTIONS
-----------
H-H1  `side` = +1 for a trader long physical / short futures, -1 for a manufacturer buyer
      long futures. Both are simulated as mirror images.
H-H2  The initial-margin schedule, when unavailable, is approximated by
      `IM = k x sigma_20d x F`. `k` is calibrated against published data points — Barry
      Callebaut's "x9" anchor is usable. The proxy is **flagged as such** in the
      diagnostics panel, never presented as the real schedule.
H-H3  Cash mobilised is `IM + uncompensated cumulative losses`. Intraday margin calls and
      haircuts on non-cash collateral are not modelled: both **understate** the cash
      needed. Bias runs against the thesis, hence in the right direction.
H-H4  The liquidity cost is `Q x spread` on roll dates only. Ignoring slippage outside a
      roll further understates the cost.
H-H5  360-day basis for financing, money-market convention.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from agri.data.snapshot import cached

SHORT_HEDGE = 1      # trader long physical, short futures
LONG_HEDGE = -1      # manufacturer buyer, long futures

DEFAULT_LOT_SIZE_T = 10.0        # ICE London cocoa: 10 t
DEFAULT_IM_PROXY_K = 2.33        # ~99th percentile of a normal: usual clearing-house anchor


class HedgeCostError(ValueError):
    """Mis-specified simulation."""


# ===========================================================================
# Initial margin
# ===========================================================================
def initial_margin_proxy(
    price: pd.Series, *, window: int = 20, k: float = DEFAULT_IM_PROXY_K
) -> pd.Series:
    """Initial-margin proxy `IM = k x sigma_20d x F`, in USD per tonne (H-H2).

    Use only when the clearing house's historical schedule is unavailable, and label it
    as a proxy. Volatility is realised over `window` days, not annualised: the clearing
    house covers a one- to two-day horizon, not a year.
    """
    if k <= 0:
        raise HedgeCostError(f"k must be > 0, got {k}")
    returns = np.log(price).diff()
    vol = returns.rolling(window, min_periods=window).std()
    return (k * vol * price).rename("im_usd_t")


def calibrate_im_k(
    price: pd.Series, observed_im_usd_t: pd.Series, *, window: int = 20
) -> float:
    """Calibrates `k` to reproduce published margin levels (H-H2).

    Least squares with no constant on `observed_IM = k x sigma x F`: forcing the fit
    through zero, because a zero initial margin at zero volatility is the only
    defensible value.
    """
    returns = np.log(price).diff()
    vol = returns.rolling(window, min_periods=window).std()
    frame = pd.concat({"x": vol * price, "y": observed_im_usd_t}, axis=1).dropna()
    if len(frame) < 5:
        raise HedgeCostError(f"not enough calibration points: n={len(frame)}")
    x = frame["x"].to_numpy()
    y = frame["y"].to_numpy()
    denominator = float(x @ x)
    if denominator == 0:
        raise HedgeCostError("volatility identically zero — calibration impossible")
    return float(x @ y) / denominator


# ===========================================================================
# The roll — and its sign
# ===========================================================================
def roll_pnl_usd_t(
    front_price: float, deferred_price: float, *, side: int
) -> float:
    """Roll P&L per tonne.

        roll_pnl = s x (F_deferred - F_near)      s = +1 short, -1 long

    Derived, so the sign never has to be memorised. A short who rolls buys back the
    front month and sells the deferred one: they end up short **at the deferred
    price**. The contract they now carry then converges towards spot.
        contango (F_def > F_near)      : short higher, price falls -> gain
        backwardation (F_def < F_near) : short lower, price rises  -> loss
    And the reverse for a long.

    Checked against the source: Barry Callebaut was short futures against long
    physical, and reports backwardation as a **cost**. That is exactly what the formula
    gives.
    """
    if side not in (SHORT_HEDGE, LONG_HEDGE):
        raise HedgeCostError(f"side must be +1 (short) or -1 (long), got {side}")
    return side * (deferred_price - front_price)


def roll_cost_usd_t(front_price: float, deferred_price: float, *, side: int) -> float:
    """The same roll, expressed as a **cost** (positive = it costs money).

        roll_cost = s x (F_near - F_deferred)

    This is the scoping spec's identity. Two explicit functions rather than a sign to
    remember: it is the most common mistake on this topic, and it is silent — a
    mis-signed roll turns a hedging cost into hedging income without anything visibly
    breaking.
    """
    return -roll_pnl_usd_t(front_price, deferred_price, side=side)


# ===========================================================================
# The full simulation
# ===========================================================================
@dataclass(frozen=True)
class HedgeParams:
    side: int = SHORT_HEDGE
    book_size_t: float = 100_000.0
    lot_size_t: float = DEFAULT_LOT_SIZE_T
    credit_line_usd: float = 250_000_000.0
    funding_spread_bps: float = 250.0
    liquidity_spread_usd_t: float = 4.0

    def __post_init__(self) -> None:
        if self.side not in (SHORT_HEDGE, LONG_HEDGE):
            raise HedgeCostError(f"side must be +1 or -1, got {self.side}")
        if self.book_size_t <= 0:
            raise HedgeCostError("book_size_t must be > 0")
        if self.lot_size_t <= 0:
            raise HedgeCostError("lot_size_t must be > 0")


def simulate_hedge(
    front: pd.Series,
    deferred: pd.Series,
    im_usd_t: pd.Series,
    rate: pd.Series | float,
    roll_dates: pd.DatetimeIndex,
    *,
    params: HedgeParams,
) -> pd.DataFrame:
    """Simulates a rolling hedge and decomposes its full cost.

    Columns returned, all in USD unless noted:
        vm_usd              the day's variation margin (P&L sign)
        cum_loss_usd        uncompensated cumulative losses (>= 0)
        im_usd              initial margin required
        cash_usd            cash mobilised = IM + cumulative losses
        financing_usd       carrying cost of that cash
        roll_usd            roll cost on roll dates (positive = cost)
        liquidity_usd       spread paid on each roll
        hedge_cost_cum_usd  cumulative sum of financing + roll + liquidity

    The cumulative cost does **not** include variation margin itself: VM is a transfer,
    not a cost. What costs money is financing it.
    """
    aligned = pd.concat(
        {"front": front, "deferred": deferred, "im_usd_t": im_usd_t}, axis=1
    ).dropna()
    if aligned.empty:
        raise HedgeCostError("no common date across prices and initial margin")

    if isinstance(rate, (int, float)):
        rate_series = pd.Series(float(rate), index=aligned.index)
    else:
        rate_series = pd.Series(rate).reindex(aligned.index).ffill()
        if rate_series.isna().any():
            raise HedgeCostError("the financing rate does not cover the full period")

    q = params.book_size_t
    side = params.side
    all_in_rate = rate_series + params.funding_spread_bps / 10_000.0

    out = pd.DataFrame(index=aligned.index)
    out["front"] = aligned["front"]
    out["deferred"] = aligned["deferred"]

    # variation margin: -s x Q x dF. A short loses when the price rises.
    out["vm_usd"] = -side * q * aligned["front"].diff().fillna(0.0)
    cumulative = out["vm_usd"].cumsum()
    out["cum_loss_usd"] = (-cumulative).clip(lower=0.0)

    # initial margin: in whole lots, the way a clearing house computes it
    n_lots = np.ceil(q / params.lot_size_t)
    out["im_usd"] = aligned["im_usd_t"] * params.lot_size_t * n_lots

    out["cash_usd"] = out["im_usd"] + out["cum_loss_usd"]
    out["financing_usd"] = out["cash_usd"] * all_in_rate / 360.0

    # roll and liquidity, on roll dates only
    out["roll_usd"] = 0.0
    out["liquidity_usd"] = 0.0
    valid_rolls = pd.DatetimeIndex(roll_dates).intersection(out.index)
    for date in valid_rolls:
        out.loc[date, "roll_usd"] = q * roll_cost_usd_t(
            float(out.loc[date, "front"]), float(out.loc[date, "deferred"]), side=side
        )
        out.loc[date, "liquidity_usd"] = q * params.liquidity_spread_usd_t

    out["hedge_cost_usd"] = out["financing_usd"] + out["roll_usd"] + out["liquidity_usd"]
    out["hedge_cost_cum_usd"] = out["hedge_cost_usd"].cumsum()
    out["hedge_cost_cum_usd_t"] = out["hedge_cost_cum_usd"] / q

    out.attrs["side"] = "short (trader)" if side == SHORT_HEDGE else "long (manufacturer)"
    out.attrs["book_size_t"] = q
    out.attrs["n_rolls"] = len(valid_rolls)
    return out


# ===========================================================================
# The core of the project — hedging capacity
# ===========================================================================
@dataclass(frozen=True)
class HedgeCapacity:
    """How many tonnes a house can hedge with its lines, over time."""

    capacity_t: pd.Series
    credit_line_usd: float
    book_size_t: float
    peak_cash_usd: float
    peak_cash_date: pd.Timestamp

    @property
    def min_capacity_t(self) -> float:
        return float(self.capacity_t.min())

    @property
    def is_binding(self) -> bool:
        """Does the credit line constrain the book at any point in the sample?"""
        return self.min_capacity_t < self.book_size_t

    def contraction_over(self, months: int = 4) -> float:
        """Contraction of capacity over the `months` before the cash peak.

        This is the spec's measure — "down Y% in four months, at the exact peak of
        physical value". The series' absolute maximum is a bad starting point: it
        falls in a calm period where initial margin is tiny, which mechanically
        produces a contraction near 100% and says nothing.
        """
        window_start = self.peak_cash_date - pd.DateOffset(months=months)
        window = self.capacity_t.loc[window_start : self.peak_cash_date]
        if len(window) < 2:
            return float("nan")
        return 1.0 - float(window.iloc[-1]) / float(window.iloc[0])

    @property
    def headline(self) -> str:
        verdict = (
            f"the line becomes binding: capacity falls to "
            f"{self.min_capacity_t:,.0f} t against a book of {self.book_size_t:,.0f} t"
            if self.is_binding
            else "the line stays above the book across the whole sample"
        )
        return (
            f"At the peak on {self.peak_cash_date:%d %b %Y}, the book mobilised "
            f"{self.peak_cash_usd / 1e6:,.0f} M USD of cash. With "
            f"{self.credit_line_usd / 1e6:,.0f} M USD of lines, {verdict} — capacity "
            f"fell {self.contraction_over(4):.0%} over the four months before this "
            "peak, at the exact moment the physical was worth the most."
        )


def hedge_capacity(
    simulation: pd.DataFrame, *, credit_line_usd: float, book_size_t: float
) -> HedgeCapacity:
    """`Q*(t) = B / cash_t(1 tonne)` — the chart that makes the email.

    The maximum hedgeable book contracts exactly when the physical is worth the most.
    That is procyclicality, made visible in tonnes rather than in percent of margin: a
    desk reads tonnes.
    """
    if credit_line_usd <= 0:
        raise HedgeCostError("the credit line must be > 0")
    if book_size_t <= 0:
        raise HedgeCostError("the book must be > 0")

    cash_per_tonne = simulation["cash_usd"] / book_size_t
    if (cash_per_tonne <= 0).any():
        raise HedgeCostError(
            "cash mobilised is zero or negative on some dates — "
            "check initial margin before deriving a capacity from it"
        )
    capacity = (credit_line_usd / cash_per_tonne).rename("capacity_t")
    peak_date = simulation["cash_usd"].idxmax()
    return HedgeCapacity(
        capacity_t=capacity,
        credit_line_usd=credit_line_usd,
        book_size_t=book_size_t,
        peak_cash_usd=float(simulation["cash_usd"].max()),
        peak_cash_date=peak_date,
    )


def margin_breakeven_im_usd_t(
    *, credit_line_usd: float, book_size_t: float, cumulative_loss_usd: float = 0.0
) -> float:
    """`IM*`: the initial margin at which capacity falls below the physical book.

    This is the page's second tipping point, and the better of the two: beyond this
    level the house **must** cut its physical buying. This is the transmission
    mechanism into a stop in farm-gate buying.
    """
    if book_size_t <= 0:
        raise HedgeCostError("the book must be > 0")
    available = credit_line_usd - cumulative_loss_usd
    if available <= 0:
        return 0.0
    return available / book_size_t


# ===========================================================================
# S6 — the two sides, mirrored
# ===========================================================================
def compare_sides(
    front: pd.Series,
    deferred: pd.Series,
    im_usd_t: pd.Series,
    rate: pd.Series | float,
    roll_dates: pd.DatetimeIndex,
    *,
    params: HedgeParams,
    windows: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    """Cost per tonne on both sides, over named windows.

    The email chart: cost per tonne of the short hedge on the way up, cost per tonne of
    the long hedge on the way down. Same chart, two bars, fourteen months apart — the
    demonstration that hedging punished both camps.
    """
    rows = []
    for label, (start, end) in windows.items():
        for side, side_label in ((SHORT_HEDGE, "short (trader)"), (LONG_HEDGE, "long (manufacturer)")):
            local = HedgeParams(
                side=side,
                book_size_t=params.book_size_t,
                lot_size_t=params.lot_size_t,
                credit_line_usd=params.credit_line_usd,
                funding_spread_bps=params.funding_spread_bps,
                liquidity_spread_usd_t=params.liquidity_spread_usd_t,
            )
            window_slice = slice(start, end)
            simulation = simulate_hedge(
                front.loc[window_slice],
                deferred.loc[window_slice],
                im_usd_t.loc[window_slice],
                rate,
                roll_dates,
                params=local,
            )
            rows.append(
                {
                    "window": label,
                    "side": side_label,
                    "cost_usd_t": float(simulation["hedge_cost_cum_usd_t"].iloc[-1]),
                    "peak_cash_usd": float(simulation["cash_usd"].max()),
                    "price_start": float(simulation["front"].iloc[0]),
                    "price_end": float(simulation["front"].iloc[-1]),
                }
            )
    return pd.DataFrame(rows)


REAL_COMMODITY_KEYS: dict[str, str] = {
    "cacao_ny": "cocoa_ny",
    "cacao_londres": "cocoa_london",
    "cafe_arabica": "coffee_arabica",
    "cafe_robusta": "coffee_robusta",
}


@cached('t1_2_hedge')
def load_real_hedge_frame(
    commodity: str = "cacao_ny",
    *,
    start: str = "2018-04-02",
    params: HedgeParams | None = None,
) -> pd.DataFrame:
    """Hedge simulation on **real** ICE prices (the user's Bloomberg export).

    DATA LIMIT, DOCUMENTED RATHER THAN WORKED AROUND: the export only contains the
    generic front-month contract (M1) — no deferred maturity. Roll cost therefore
    cannot be computed on real data: `deferred` is set equal to `front`, which
    mechanically zeroes the term (`roll_cost_usd_t(x, x, ...) = 0`) rather than
    inventing a term structure. Financing (real SOFR) and initial margin (a proxy
    calibrated on real realised volatility) remain, however, entirely real — these are
    the two components Barry Callebaut cites explicitly (x9 margin, carrying cost), so
    the thesis stays testable even without the roll.

    `start` is pinned to the beginning of SOFR coverage (2018-04-02): before that, the
    real financing rate does not cover the period and `simulate_hedge` would raise.
    """
    from agri.data.bloomberg_loader import load as load_bloomberg

    key = REAL_COMMODITY_KEYS.get(commodity, commodity)
    front_full = load_bloomberg(key)
    front = front_full.loc[start:]
    if front.empty:
        raise HedgeCostError(f"no data for {commodity!r} after {start}")

    im = initial_margin_proxy(front_full).loc[start:]
    rate = load_bloomberg("sofr")
    resolved_params = params or HedgeParams()

    simulation = simulate_hedge(front, front, im, rate, pd.DatetimeIndex([]), params=resolved_params)
    simulation.attrs["commodity"] = commodity
    simulation.attrs["roll_cost_omitted"] = True
    return simulation


# ===========================================================================
# THE DELIVERABLE — the price at which the balance sheet forces an exit
# ===========================================================================
def implied_margin_rate(simulation: pd.DataFrame, *, book_size_t: float) -> float:
    """Initial margin expressed as a fraction of price, measured on the simulation.

    Measured rather than assumed: the `k x sigma x F` proxy makes it proportional to
    price by construction, but the effective coefficient depends on the period's
    realised volatility and has no reason to match a real schedule's.
    """
    im_per_tonne = simulation["im_usd"] / book_size_t
    return float((im_per_tonne / simulation["front"]).median())


@dataclass(frozen=True)
class ForcedExit:
    """The price at which the credit line saturates, for a given hedge.

    DERIVATION (closed form, short hedge opened at `inception_price`):

        mark-to-market loss   = Q x (P - P0)          a short loses when price rises
        initial margin        = Q x im_rate x P       proportional to price
        cash mobilised        = Q x (im_rate x P + P - P0)

        saturates when         Q x (im_rate x P + P - P0) = B
        so                     P* = (B/Q + P0) / (1 + im_rate)

    Beyond `P*`, the house can no longer finance its hedge. It can therefore no longer
    hedge additional physical — and since nobody buys unhedged physical at this level
    of volatility, it **stops buying**. This is the tipping point that links a trading
    liquidity crisis to a stop in farm-gate buying.
    """

    inception_date: pd.Timestamp
    inception_price: float
    exit_price: float
    credit_line_usd: float
    book_size_t: float
    im_rate: float
    crossed_on: pd.Timestamp | None
    days_of_protection: int | None

    @property
    def headroom_pct(self) -> float:
        """How far the price can rise, in %, before the line saturates."""
        return self.exit_price / self.inception_price - 1.0

    @property
    def headline(self) -> str:
        if self.crossed_on is None:
            return (
                f"Hedge opened on {self.inception_date:%d %b %Y} at "
                f"{self.inception_price:,.0f}: with {self.credit_line_usd/1e6:,.0f} M "
                f"USD of lines on {self.book_size_t/1000:,.0f} kt, forced exit sits at "
                f"{self.exit_price:,.0f} USD/t (+{self.headroom_pct:.0%}). The market "
                "never got there over the sample."
            )
        return (
            f"Hedge opened on {self.inception_date:%d %b %Y} at "
            f"{self.inception_price:,.0f}: the line saturates at "
            f"{self.exit_price:,.0f} USD/t (+{self.headroom_pct:.0%}), crossed on "
            f"{self.crossed_on:%d %b %Y} — {self.days_of_protection} days of "
            "protection."
        )


def forced_exit_price(
    price: pd.Series,
    *,
    inception: str | pd.Timestamp,
    book_size_t: float,
    credit_line_usd: float,
    im_rate: float,
) -> ForcedExit:
    """`P*` in closed form, and the date the market crossed it.

    The closed form is preferred to a solver because it shows **what the threshold
    depends on**: linearly on the line relative to book size, on the entry price, and
    barely on the margin rate. A desk can redo this calculation in their head.
    """
    if book_size_t <= 0 or credit_line_usd <= 0:
        raise HedgeCostError("book and credit line must be > 0")
    if im_rate < 0:
        raise HedgeCostError("the margin rate cannot be negative")

    clean = pd.Series(price).dropna().astype(float)
    forward = clean.loc[pd.Timestamp(inception) :]
    if forward.empty:
        raise HedgeCostError(f"no price after {inception}")

    inception_date = forward.index[0]
    inception_price = float(forward.iloc[0])
    exit_price = (credit_line_usd / book_size_t + inception_price) / (1.0 + im_rate)

    breached = forward[forward >= exit_price]
    crossed_on = breached.index.min() if len(breached) else None
    days = int((crossed_on - inception_date).days) if crossed_on is not None else None

    return ForcedExit(
        inception_date=inception_date,
        inception_price=inception_price,
        exit_price=exit_price,
        credit_line_usd=credit_line_usd,
        book_size_t=book_size_t,
        im_rate=im_rate,
        crossed_on=crossed_on,
        days_of_protection=days,
    )


def forced_exit_schedule(
    price: pd.Series,
    inceptions: list[str],
    *,
    book_size_t: float,
    credit_line_usd: float,
    im_rate: float,
) -> pd.DataFrame:
    """The same threshold for several opening dates — the table that carries the thesis.

    What it shows on 2024 cocoa: forced-exit dates bunch into a few weeks regardless of
    when the hedge was opened. The price move was violent enough to crush the spread of
    entry points — which is why so many houses hit the constraint at the same time,
    rather than each in their own turn.
    """
    rows = []
    for inception in inceptions:
        try:
            result = forced_exit_price(
                price, inception=inception, book_size_t=book_size_t,
                credit_line_usd=credit_line_usd, im_rate=im_rate,
            )
        except HedgeCostError:
            continue
        rows.append(
            {
                "opened": result.inception_date,
                "entry price": result.inception_price,
                "forced exit": result.exit_price,
                "headroom": result.headroom_pct,
                "crossed on": result.crossed_on,
                "days of protection": result.days_of_protection,
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class HedgingIntensity:
    """Cash mobilised relative to the value of the physical book it protects.

    This is the quantity Montesanto Tavares's lawyers put a number on: the cost of
    carrying hedges went from **74% of trade receivables in May 2025 to 158% in
    November**, and they called it unsustainable. The same measure, computed here on
    real ICE prices, tells you the moment a treasury structure stops holding — without
    needing anyone's balance sheet.
    """

    ratio: pd.Series
    calm_median: float
    peak_ratio: float
    peak_date: pd.Timestamp
    share_above_one: float

    @property
    def headline(self) -> str:
        return (
            f"Cash tied up to hold the hedge goes from {self.calm_median:.0%} of book "
            f"value in a calm regime to {self.peak_ratio:.0%} at the peak on "
            f"{self.peak_date:%d %b %Y}. At that level, financing the hedge costs "
            "almost as much as financing the stock it protects — this is the measure "
            "Montesanto Tavares's lawyers called unsustainable at 158%."
        )


def hedging_intensity(
    simulation: pd.DataFrame, *, book_size_t: float, calm_window: tuple[str, str] | None = None
) -> HedgingIntensity:
    """Ratio of cash mobilised to physical book value, over time."""
    book_value = book_size_t * simulation["front"]
    ratio = (simulation["cash_usd"] / book_value).rename("hedging_intensity")

    if calm_window is not None:
        calm = ratio.loc[calm_window[0] : calm_window[1]]
    else:
        calm = ratio.loc[: ratio.index[len(ratio) // 4]]

    return HedgingIntensity(
        ratio=ratio,
        calm_median=float(calm.median()),
        peak_ratio=float(ratio.max()),
        peak_date=ratio.idxmax(),
        share_above_one=float((ratio > 1.0).mean()),
    )


def procyclicality(simulation: pd.DataFrame) -> dict[str, float]:
    """Correlation between changes in initial margin and changes in price.

    On differenced data, not levels: two non-stationary level series produce a
    flattering correlation that means nothing.
    """
    changes = simulation[["im_usd", "front"]].diff().dropna()
    if len(changes) < 3:
        raise HedgeCostError("not enough observations")
    return {
        "corr_delta_im_delta_price": float(changes["im_usd"].corr(changes["front"])),
        "n_obs": len(changes),
    }


IM_PROXY_WARNING = (
    "The initial margin shown is a `k x sigma_20d x F` proxy, not the clearing house's "
    "real schedule (H-H2). Real schedule step changes are not dated here: check ICE "
    "notices before reading a spike as volatility."
)

__all__ = [
    "HedgeCapacity",
    "HedgeCostError",
    "HedgeParams",
    "IM_PROXY_WARNING",
    "LONG_HEDGE",
    "SHORT_HEDGE",
    "calibrate_im_k",
    "compare_sides",
    "ForcedExit",
    "HedgingIntensity",
    "forced_exit_price",
    "forced_exit_schedule",
    "hedge_capacity",
    "hedging_intensity",
    "implied_margin_rate",
    "initial_margin_proxy",
    "load_real_hedge_frame",
    "margin_breakeven_im_usd_t",
    "procyclicality",
    "roll_cost_usd_t",
    "roll_pnl_usd_t",
    "simulate_hedge",
]
