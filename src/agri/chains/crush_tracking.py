"""T2-3 — The board crush is not a price, it is a yield in disguise.

THE STORY
---------
The board crush is written as `0.022 x meal + 0.11 x oil - bean`. Those two coefficients
look like harmless conversion parameters. They are in fact **yields**: 44 pounds of meal
and 11 pounds of oil per bushel. The CBOT froze them once and for all, because a contract
needs a stable definition.

A plant, on the other hand, has no stable yield. The meal it produces depends on the
protein content of the beans, which depends on origin, season and lot. Two points of
protein shift the yield by several pounds per bushel.

So when a crusher hedges on the board crush, it is not just hedging: it **silently accepts
44/11 as its own yield**, and keeps the difference as a naked position. Nobody ever decided
this position — it is the residue of a contract convention — and its size in dollars is set
by the meal price, i.e. by the market.

THE DELIVERABLE — THE INVERSION
----------------------------------
We do not measure the tracking error (that would need cash prices, absent from the
export). We ask the reverse question: **what yield precision does the board silently
demand of you?**

    threshold_lb = (board_crush - opex) / (meal_price / 2000)

This is the yield gap that consumes the entire net margin, in pounds per bushel — the unit
a crush operator works in every day. The result is not an average level but its
**regime-dependence**: the requirement collapses exactly when the margin tightens, which
is exactly when the hedge was supposed to matter.

TENSION — INFERRED, NOT SOURCED
----------------------------------
**It seems to me** the CBOT crush is treated as a hedgeable proxy for a plant's real
economics, while domestic meal basis, real yields and logistics break the hedge exactly
when it matters. No documentary evidence that a specific desk argues about this: "it seems
to me", never "I read that".

THE UNIT TRAP, AND IT IS THE HEART OF THE SUBJECT
-----------------------------------------------------
The board crush mixes **three units** into a single formula:
    the bean in USD/bushel, the meal in USD/**short ton**, the oil in cents/lb.
Treating the short ton as a metric tonne misprices meal by 10% — on a typical crush,
half the crush itself. See `core/units.board_crush_usd_bu`, which derives the
coefficients instead of hardcoding them.

IDENTITY
--------
    board_crush = 0.022 x meal_usd_short_ton + 0.11 x oil_c_lb - bean_usd_bu
    plant_crush = y_meal x meal_cash + y_oil x oil_cash - bean_cash - opex
    tracking_error = board_crush - plant_crush

TIPPING POINT
-------------
The minimum-variance hedge ratio `h* = cov(dplant, dboard) / var(dboard)`, and the
regimes where it drifts from 1. Beyond a certain decoupling, hedging on the board **is a
position in its own right**, not a hedge.

ASSUMPTIONS
-----------
X-H1  Implicit CBOT yields: 44 lb of meal and 11 lb of oil per bushel. A real plant's
      yields differ — hence `y_meal` and `y_oil` as sliders.
X-H2  Crushing opex is a flat rate per bushel, exogenous. Varies strongly with energy
      prices; parameterised.
X-H3  No lag between buying the bean and selling the products: the crush is computed on
      the same date for all three legs. A real plant carries a lag of several weeks,
      which **adds** tracking error — a bias in the right direction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from agri.data.snapshot import cached

from agri.core.stats import HacRegression, hac_ols, regime_runs
from agri.core.units import board_crush_usd_bu

# X-H1: real default yields, slightly below CBOT yields
DEFAULT_YIELD_MEAL_LB_BU = 43.5
DEFAULT_YIELD_OIL_LB_BU = 10.8
DEFAULT_OPEX_USD_BU = 0.42

CBOT_MEAL_LB_BU = 44.0
CBOT_OIL_LB_BU = 11.0
LB_PER_SHORT_TON = 2000.0


class CrushError(ValueError):
    """Mis-specified model."""


def plant_crush_usd_bu(
    bean_cash_usd_bu: pd.Series,
    meal_cash_usd_short_ton: pd.Series,
    oil_cash_cents_lb: pd.Series,
    *,
    yield_meal_lb_bu: float = DEFAULT_YIELD_MEAL_LB_BU,
    yield_oil_lb_bu: float = DEFAULT_YIELD_OIL_LB_BU,
    opex_usd_bu: float = DEFAULT_OPEX_USD_BU,
) -> pd.Series:
    """A real plant's crush, at local cash prices and real yields (X-H1).

    Same units as the board — short ton for meal, cents/lb for oil — so that the
    difference between the two is a genuine tracking error and not a disguised
    conversion error.
    """
    if not 0 < yield_meal_lb_bu < 60 or not 0 < yield_oil_lb_bu < 20:
        raise CrushError(
            f"yields outside the physical range: {yield_meal_lb_bu} lb meal and "
            f"{yield_oil_lb_bu} lb oil per bushel (a bushel of soybeans weighs 60 lb)"
        )
    if yield_meal_lb_bu + yield_oil_lb_bu > 60.0:
        raise CrushError(
            f"yields sum to {yield_meal_lb_bu + yield_oil_lb_bu:.1f} lb for a 60 lb "
            "bushel — impossible before even counting losses"
        )
    meal_leg = (yield_meal_lb_bu / LB_PER_SHORT_TON) * meal_cash_usd_short_ton
    oil_leg = yield_oil_lb_bu * (oil_cash_cents_lb / 100.0)
    return meal_leg + oil_leg - bean_cash_usd_bu - opex_usd_bu


def build_tracking(
    bean_board: pd.Series,
    meal_board: pd.Series,
    oil_board: pd.Series,
    bean_cash: pd.Series,
    meal_cash: pd.Series,
    oil_cash: pd.Series,
    *,
    yield_meal_lb_bu: float = DEFAULT_YIELD_MEAL_LB_BU,
    yield_oil_lb_bu: float = DEFAULT_YIELD_OIL_LB_BU,
    opex_usd_bu: float = DEFAULT_OPEX_USD_BU,
) -> pd.DataFrame:
    """Board crush, plant crush and their gap, on common dates.

    Columns: board_crush, plant_crush, tracking_error, meal_basis, oil_basis, bean_basis.
    The three bases are exposed because the tracking error comes from them — this is the
    decomposition that makes decoupling explainable instead of merely observed.
    """
    frame = pd.concat(
        {
            "bean_board": bean_board,
            "meal_board": meal_board,
            "oil_board": oil_board,
            "bean_cash": bean_cash,
            "meal_cash": meal_cash,
            "oil_cash": oil_cash,
        },
        axis=1,
    ).dropna()
    if frame.empty:
        raise CrushError("no common date across the six series")

    frame["board_crush"] = board_crush_usd_bu(
        frame["bean_board"], frame["meal_board"], frame["oil_board"]
    )
    frame["plant_crush"] = plant_crush_usd_bu(
        frame["bean_cash"],
        frame["meal_cash"],
        frame["oil_cash"],
        yield_meal_lb_bu=yield_meal_lb_bu,
        yield_oil_lb_bu=yield_oil_lb_bu,
        opex_usd_bu=opex_usd_bu,
    )
    frame["tracking_error"] = frame["board_crush"] - frame["plant_crush"]
    frame["bean_basis"] = frame["bean_cash"] - frame["bean_board"]
    frame["meal_basis"] = frame["meal_cash"] - frame["meal_board"]
    frame["oil_basis"] = frame["oil_cash"] - frame["oil_board"]
    frame.attrs["yield_meal_lb_bu"] = yield_meal_lb_bu
    frame.attrs["yield_oil_lb_bu"] = yield_oil_lb_bu
    frame.attrs["opex_usd_bu"] = opex_usd_bu
    return frame


def decompose_tracking_error(frame: pd.DataFrame) -> pd.DataFrame:
    """**Exact** decomposition of the tracking error. Not a regression: an identity.

        tracking = (0.022 - y_meal/2000) x meal_board      <- meal yield gap
                 + (0.11  - y_oil/100)   x oil_board        <- oil yield gap
                 - (y_meal/2000) x meal_basis
                 - (y_oil/100)   x oil_basis
                 + bean_basis
                 + opex

    The first two terms are exactly what a "the decoupling comes from basis" reading
    misses completely: they are proportional to the **level** of the board, not to a
    basis, and they exist even when every basis is zero. On the test set, the oil-yield
    term has a larger standard deviation than the bean basis term itself.

    A regression on the three bases treats these as omitted variables and estimates the
    bean coefficient with a visible bias (0.988 instead of 1.000). The identity does not
    have this problem — when the calculation is exact, it is not estimated.
    """
    required = {"yield_meal_lb_bu", "yield_oil_lb_bu", "opex_usd_bu"}
    if not required.issubset(frame.attrs):
        raise CrushError(
            "frame with no yield metadata — use build_tracking(), which sets them"
        )
    y_meal = frame.attrs["yield_meal_lb_bu"] / LB_PER_SHORT_TON
    y_oil = frame.attrs["yield_oil_lb_bu"] / 100.0
    board_meal = CBOT_MEAL_LB_BU / LB_PER_SHORT_TON
    board_oil = CBOT_OIL_LB_BU / 100.0

    out = pd.DataFrame(index=frame.index)
    out["meal_yield"] = (board_meal - y_meal) * frame["meal_board"]
    out["oil_yield"] = (board_oil - y_oil) * frame["oil_board"]
    out["meal_basis"] = -y_meal * frame["meal_basis"]
    out["oil_basis"] = -y_oil * frame["oil_basis"]
    out["bean_basis"] = frame["bean_basis"]
    out["opex"] = frame.attrs["opex_usd_bu"]
    out["total"] = out.sum(axis=1)
    return out


@dataclass(frozen=True)
class OptimalHedge:
    """Minimum-variance hedge ratio, and what ignoring it costs."""

    h_star: float
    variance_reduction_at_h_star: float
    variance_reduction_at_one: float
    n_obs: int

    @property
    def naive_hedge_adds_risk(self) -> bool:
        """Does a 1:1 hedge increase variance instead of reducing it?"""
        return self.variance_reduction_at_one < 0.0

    @property
    def headline(self) -> str:
        if self.naive_hedge_adds_risk:
            return (
                f"The optimal board/plant hedge ratio falls to {self.h_star:.2f}: over "
                f"this sample, a 1:1 hedge **adds** "
                f"{-self.variance_reduction_at_one:.0%} of variance instead of removing it."
            )
        return (
            f"The optimal hedge ratio is {self.h_star:.2f}. Hedging 1:1 removes "
            f"{self.variance_reduction_at_one:.0%} of variance against "
            f"{self.variance_reduction_at_h_star:.0%} at the optimal ratio."
        )


def optimal_hedge_ratio(frame: pd.DataFrame) -> OptimalHedge:
    """`h* = cov(dplant, dboard) / var(dboard)`, on changes rather than levels.

    On levels, two non-stationary series give a flattering ratio that means nothing.
    The hedge is executed on changes: that is where it has to be measured.
    """
    changes = frame[["plant_crush", "board_crush"]].diff().dropna()
    if len(changes) < 10:
        raise CrushError(f"not enough observations: n={len(changes)}")

    var_board = float(changes["board_crush"].var())
    if var_board == 0:
        raise CrushError("the board crush does not vary — ratio undefined")
    covariance = float(changes["plant_crush"].cov(changes["board_crush"]))
    h_star = covariance / var_board

    var_plant = float(changes["plant_crush"].var())

    def _reduction(h: float) -> float:
        residual = changes["plant_crush"] - h * changes["board_crush"]
        return 1.0 - float(residual.var()) / var_plant

    return OptimalHedge(
        h_star=h_star,
        variance_reduction_at_h_star=_reduction(h_star),
        variance_reduction_at_one=_reduction(1.0),
        n_obs=len(changes),
    )


def rolling_hedge_ratio(frame: pd.DataFrame, *, window: int = 120) -> pd.DataFrame:
    """`h*` on a rolling window — the chart that shows it is not constant.

    A rolling window's `n_eff` equals n_obs/window (Rule C): show it alongside the
    curve, or the ratio looks far better estimated than it is.
    """
    changes = frame[["plant_crush", "board_crush"]].diff().dropna()
    if len(changes) < window + 2:
        raise CrushError(f"not enough observations for a window of {window}")
    covariance = changes["plant_crush"].rolling(window).cov(changes["board_crush"])
    variance = changes["board_crush"].rolling(window).var()
    out = pd.DataFrame({"h_star": covariance / variance}).dropna()
    out.attrs["window"] = window
    out.attrs["n_eff"] = len(out) / window
    return out


def decoupling_episodes(
    frame: pd.DataFrame, *, threshold_usd_bu: float = 0.35, min_obs: int = 5
) -> pd.DataFrame:
    """Episodes where the tracking error exceeds a threshold in absolute value.

    "The board and the plant decoupled by more than 35 c/bu for six weeks in
    September" is a dated sentence a crusher can confirm or deny.
    """
    return regime_runs(
        frame["tracking_error"].abs() > threshold_usd_bu,
        depth=frame["tracking_error"],
        min_obs=min_obs,
    )


def explain_tracking_error(frame: pd.DataFrame) -> HacRegression:
    """Regression of the tracking error on the three bases, HAC errors.

    Answers "where does the decoupling come from" rather than "is there a decoupling".

    The coefficients are **predictable**: the tracking error is an exact linear
    function of the three bases, with coefficients `-y_meal/2000`, `-y_oil/100` and
    `+1`. Recovering these values is a consistency check on the engine, not a
    discovery. What is informative is the **contribution** of each — see
    `basis_contributions`.
    """
    return hac_ols(
        frame["tracking_error"], frame[["bean_basis", "meal_basis", "oil_basis"]]
    )


def basis_contributions(frame: pd.DataFrame) -> pd.DataFrame:
    """Weight of each term of the tracking error, measured by its **std dev in USD/bu**.

    WHY NOT THE RAW COEFFICIENTS. The three bases are in three different units — meal
    in USD/short ton, oil in cents/lb, bean in USD/bu. Comparing their coefficients
    means comparing dollars per short ton to cents per pound: the resulting ranking
    means nothing. The exact decomposition brings every term back to USD/bu, and it is
    the spread of those terms that gets compared.

    The `opex` line is constant, hence zero dispersion: it shifts the level of the
    tracking error, never its variability. That is visible in the table, and it is a
    distinction that matters for a hedge.
    """
    components = decompose_tracking_error(frame).drop(columns="total")
    rows = [
        {
            "term": column,
            "mean_usd_bu": float(components[column].mean()),
            "std_usd_bu": float(components[column].std()),
        }
        for column in components.columns
    ]
    out = pd.DataFrame(rows).sort_values("std_usd_bu", ascending=False)
    total = out["std_usd_bu"].sum()
    out["share"] = out["std_usd_bu"] / total if total > 0 else np.nan
    return out.reset_index(drop=True)


# ===========================================================================
# THE INVERSION — what the board crush demands of you without saying so
# ===========================================================================
# The board crush's coefficients are not prices: they are **yields**. Writing
# 0.022 x meal + 0.11 x oil - bean means positing 44 lb of meal and 11 lb of oil per
# bushel. A plant that hedges on the board therefore silently accepts these yields as
# its own, and keeps the difference as a naked position. The page measures this
# position and inverts it into a requirement: what yield precision does the board
# demand of you.
@cached('t2_3_board')
def load_real_board_frame(start: str | None = "2015-01-01") -> pd.DataFrame:
    """The CBOT board crush on real data — the three legs from the export.

    Columns: bean (USD/bu), meal (USD/short ton), oil (c/lb), board (USD/bu).
    No cash price enters here: the export has none, and the page's deliverable is
    built precisely not to need one.
    """
    from agri.data.bloomberg_loader import load

    frame = pd.concat(
        {"bean": load("cbot_soybean"), "meal": load("cbot_soymeal"), "oil": load("cbot_soyoil")},
        axis=1,
        sort=True,
    ).dropna()
    if start is not None:
        frame = frame[frame.index >= pd.Timestamp(start)]
    if frame.empty:
        raise CrushError(f"no common date across the three crush legs after {start}")
    frame["board"] = board_crush_usd_bu(frame["bean"], frame["meal"], frame["oil"])
    return frame


@dataclass(frozen=True)
class YieldExposure:
    """The uncovered position a yield gap creates, day by day.

    A yield gap is not a level error that a constant would fix: it is a **position in
    the products**, whose dollar size is set by the meal and oil prices — i.e. by the
    market, not by the plant.
    """

    frame: pd.DataFrame
    meal_lb_gap: float
    oil_lb_gap: float
    opex_usd_bu: float

    @property
    def position_median(self) -> float:
        return float(self.frame["position_usd_bu"].median())

    @property
    def share_median(self) -> float:
        return float(self.frame["share_of_margin"].median())

    @property
    def share_exceeding_margin(self) -> float:
        """Share of sessions where the naked position exceeds the whole net margin."""
        return float((self.frame["position_usd_bu"].abs() > self.frame["net_margin"]).mean())

    @property
    def headline(self) -> str:
        return (
            f"A gap of {self.meal_lb_gap:+.1f} lb of meal and {self.oil_lb_gap:+.1f} lb "
            f"of oil per bushel creates a naked position of "
            f"{self.position_median:+.3f} USD/bu at the median, {self.share_median:.0%} "
            f"of net margin. It exceeds the entire margin on "
            f"{self.share_exceeding_margin:.0%} of sessions."
        )


def yield_exposure(
    frame: pd.DataFrame,
    *,
    meal_lb_gap: float = 1.0,
    oil_lb_gap: float = 0.0,
    opex_usd_bu: float = DEFAULT_OPEX_USD_BU,
) -> YieldExposure:
    """The position a yield gap leaves, on the real board crush.

        position = (Δlb_meal / 2000) x meal_price + Δlb_oil x oil_price / 100

    Both terms take the form of a price times a quantity: this really is a position,
    not a model residual. Relative to the net margin `board - opex`, it says what
    fraction of the plant's result is uncovered while believing itself hedged.
    """
    for column in ("meal", "oil", "board"):
        if column not in frame.columns:
            raise CrushError(f"missing column in the board frame: {column!r}")
    if abs(meal_lb_gap) > CBOT_MEAL_LB_BU or abs(oil_lb_gap) > CBOT_OIL_LB_BU:
        raise CrushError(
            f"implausible yield gap ({meal_lb_gap:+.1f} lb meal, "
            f"{oil_lb_gap:+.1f} lb oil): it exceeds the board's own yield "
            f"({CBOT_MEAL_LB_BU:.0f} and {CBOT_OIL_LB_BU:.0f} lb/bu)."
        )

    meal_leg = (meal_lb_gap / LB_PER_SHORT_TON) * frame["meal"]
    oil_leg = oil_lb_gap * (frame["oil"] / 100.0)
    position = meal_leg + oil_leg
    net_margin = frame["board"] - opex_usd_bu

    out = pd.DataFrame(
        {
            "meal_leg": meal_leg,
            "oil_leg": oil_leg,
            "position_usd_bu": position,
            "net_margin": net_margin,
            # Net margin floored: near zero the ratio blows up and would make the
            # median unreadable. The floor is flagged, not hidden.
            "share_of_margin": position / net_margin.clip(lower=0.05),
        }
    )
    return YieldExposure(
        frame=out,
        meal_lb_gap=float(meal_lb_gap),
        oil_lb_gap=float(oil_lb_gap),
        opex_usd_bu=float(opex_usd_bu),
    )


@dataclass(frozen=True)
class RequiredPrecision:
    """THE deliverable: the yield precision the board demands, in lb per bushel.

    Inverts `yield_exposure`: instead of asking what a given gap costs, we ask what
    gap consumes **all** of the net margin. The number that comes out is in the unit a
    crush operator handles every day.
    """

    frame: pd.DataFrame
    opex_usd_bu: float
    board_meal_lb: float

    @property
    def median_lb(self) -> float:
        return float(self.frame["breakeven_lb"].median())

    @property
    def tight_decile_lb(self) -> float:
        """The precision required in the tightest margin decile."""
        tight = self.frame[self.frame["net_margin"] <= self.frame["net_margin"].quantile(0.10)]
        return float(tight["breakeven_lb"].median())

    @property
    def wide_decile_lb(self) -> float:
        wide = self.frame[self.frame["net_margin"] >= self.frame["net_margin"].quantile(0.90)]
        return float(wide["breakeven_lb"].median())

    @property
    def tight_decile_pct(self) -> float:
        return self.tight_decile_lb / self.board_meal_lb

    def share_below(self, lb: float) -> float:
        """Share of sessions where a gap of `lb` pounds is enough to erase net margin."""
        return float((self.frame["breakeven_lb"] <= lb).mean())

    @property
    def headline(self) -> str:
        return (
            f"The board crush demands a median precision of {self.median_lb:.1f} lb of "
            f"meal per bushel. But in the tightest margin decile, the requirement "
            f"falls to {self.tight_decile_lb:.2f} lb — {self.tight_decile_pct:.1%} of "
            f"the board's {self.board_meal_lb:.0f} lb. A single pound of gap erases "
            f"the entire net margin on {self.share_below(1.0):.0%} of sessions."
        )


def required_yield_precision(
    frame: pd.DataFrame,
    *,
    opex_usd_bu: float = DEFAULT_OPEX_USD_BU,
) -> RequiredPrecision:
    """The meal yield gap that exactly zeroes the net margin, day by day.

        threshold_lb = (board - opex) / (meal_price / 2000)

    The threshold decreases with the margin: it tightens exactly when the margin
    tightens, i.e. in the regime where the plant's shutdown decision becomes live
    (cf. T2-5). That is where hedging on the board stops being a hedge.
    """
    if "board" not in frame.columns or "meal" not in frame.columns:
        raise CrushError("the frame must contain 'board' and 'meal' columns")

    net_margin = frame["board"] - opex_usd_bu
    position_per_lb = frame["meal"] / LB_PER_SHORT_TON
    if (position_per_lb <= 0).any():
        raise CrushError("meal price zero or negative: the threshold is undefined")

    out = pd.DataFrame(
        {
            "net_margin": net_margin,
            "position_per_lb": position_per_lb,
            "breakeven_lb": net_margin / position_per_lb,
        }
    )
    return RequiredPrecision(
        frame=out, opex_usd_bu=float(opex_usd_bu), board_meal_lb=CBOT_MEAL_LB_BU
    )


@dataclass(frozen=True)
class IdentityBias:
    """Why the page computes the position directly instead of regressing it.

    Inherited from T2-1, dropped from the portfolio for lack of cash series. The
    result carries over unchanged: regressing one quantity on another that shares its
    components does not measure an economic relationship, it measures an accounting
    identity.

    HONESTY ABOUT THE ORDER OF MAGNITUDE: the contamination is **small** — roughly +1%
    for a one-pound gap, and proportional to the gap. This is not a trap that blows
    numbers up, and presenting it as such would be dishonest. What matters is not its
    size, it is what a practitioner would do with the coefficient: see `headline`.
    """

    beta_naive: float
    beta_structural: float
    bias: float
    meal_lb_gap: float
    oil_lb_gap: float

    @property
    def headline(self) -> str:
        return (
            f"Regressing the plant's margin on the board crush gives {self.beta_naive:.3f} "
            f"where the structural answer is {self.beta_structural:.3f} — a gap of only "
            f"{self.bias:+.3f}, but entirely mechanical: the two quantities share meal, "
            "oil and bean. The danger is not the size of the bias, it is what one does "
            "with it: applying this coefficient means hedging your yield gap with "
            "**more board crush**, when the gap is a position in meal and oil taken "
            "separately. You do not hedge a yield gap with the instrument whose yield "
            "assumption created it."
        )


def hedge_ratio_identity_bias(
    frame: pd.DataFrame,
    *,
    meal_lb_gap: float = 1.0,
    oil_lb_gap: float = 0.0,
    opex_usd_bu: float = DEFAULT_OPEX_USD_BU,
) -> IdentityBias:
    """The accounting-identity trap, demonstrated on real data.

    The plant's margin is written exactly as `board + yield_gap - opex`. Regressing
    its change on the board's change therefore gives

        beta = 1 + cov(Δgap, Δboard) / var(Δboard)

    and the second term is not zero, since the yield gap is itself a combination of
    meal and oil — the two legs that make up the board. The estimated coefficient is
    not a hedge ratio: it is 1 plus contamination.

    The structural answer is 1: at identical yields, a bushel hedged on the board is
    hedged one for one. Any departure from 1 measured by this regression is mechanical.

    THE USEFUL CONCLUSION is not "careful, your beta is biased" — the bias is small. It
    is that the right way to hedge a yield gap is not adjusting the board crush ratio,
    but sizing the meal and oil legs separately. The board crush is a fixed 44/11
    basket: by construction it cannot hedge a gap away from 44/11.
    """
    exposure = yield_exposure(
        frame, meal_lb_gap=meal_lb_gap, oil_lb_gap=oil_lb_gap, opex_usd_bu=opex_usd_bu
    )
    plant_margin = frame["board"] + exposure.frame["position_usd_bu"] - opex_usd_bu

    changes = pd.concat(
        {"plant": plant_margin.diff(), "board": frame["board"].diff()}, axis=1, sort=True
    ).dropna()
    if len(changes) < 30:
        raise CrushError(f"sample too short for the demonstration: n={len(changes)}")

    variance = float(changes["board"].var())
    if variance <= 0:
        raise CrushError("zero variance in the board crush: the regression is undefined")
    beta_naive = float(changes["plant"].cov(changes["board"]) / variance)

    return IdentityBias(
        beta_naive=beta_naive,
        beta_structural=1.0,
        bias=beta_naive - 1.0,
        meal_lb_gap=float(meal_lb_gap),
        oil_lb_gap=float(oil_lb_gap),
    )


__all__ = [
    "CBOT_MEAL_LB_BU",
    "CBOT_OIL_LB_BU",
    "CrushError",
    "IdentityBias",
    "OptimalHedge",
    "RequiredPrecision",
    "YieldExposure",
    "basis_contributions",
    "build_tracking",
    "decoupling_episodes",
    "explain_tracking_error",
    "hedge_ratio_identity_bias",
    "load_real_board_frame",
    "optimal_hedge_ratio",
    "plant_crush_usd_bu",
    "required_yield_precision",
    "rolling_hedge_ratio",
    "yield_exposure",
]
