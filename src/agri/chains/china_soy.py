"""T3-4 — China soy: political or commercial?

THESIS
------
State reserve purchases concentrate where commercial buyers **cannot** buy. If purchases
cluster in the lowest crush-margin quintiles, these are purchases the crush economics rule
out — a political signature. If they cluster in the high quintiles, it is opportunistic
stock rotation — a commercial signature.

The test is binary, and the sign of a single coefficient settles it.

THE DISAGREEMENT (open, sourced)
------------------------------------
Sinograin sold roughly half of the 504,000 t of imported soybeans offered at its largest
auction since January; traders quoted by Reuters argue these auctions are making room for
incoming US cargoes (August 2026). On the other side, ADM raised its 2026 outlook for the
second time, citing a constructive biofuels environment and the expectation that China
keeps buying US soybeans.

THE UNIT TRAP, AND THERE ARE THREE STACKED
-----------------------------------------------
1. CBOT quotes in **USD/bushel** (soybean, 60 lb), DCE in **CNY/tonne**.
2. Chinese prices are **VAT-inclusive**; the crush margin is computed ex-VAT. And the
   VAT on imported oilseeds is not the same as on processed products — check the rate
   applicable to the product **and** the date, they move.
3. Import duty applies to the CNF value, not the FOB price.

Any single one of these three mistakes shifts the margin by tens of CNY/t and flips the
sign of the test.

MODEL
-----
    bean_cnf_usd_t = (CBOT_usd_bu + basis_c_bu/100) x 36.7437 + freight_usd_t
    crush_margin   = (0.785 x meal_dce + 0.185 x oil_dce) / (1 + VAT)
                     - bean_cnf_cny_t x (1 + duty) - processing

    reserve_flow = customs_imports - observed_crush - direct_use

Discriminating test, in logit form:

    logit(1{reserve_purchase}) = a + b1 crush_margin_{t-1} + b2 stock_days_{t-1}
                                    + b3 price_level_{t-1}

    b1 < 0 significant  ->  POLITICAL signature
    b1 > 0 significant  ->  COMMERCIAL signature

ASSUMPTIONS
-----------
N-H1  Chinese crushing yields: 0.785 t of meal and 0.185 t of oil per tonne of bean.
      Parameterised.
N-H2  The margin is lagged by one period in the logit. A purchase decided today
      responds to yesterday's margin, not the one it helps create — without this lag,
      the test suffers from a simultaneity that can flip the sign.
N-H3  Direct use (unprocessed animal feed, seed) is a seasonal flat rate. It is the
      weakest term of the reserve residual, and is shown as such.
N-H4  State reserve series are not published. The fallback — and this is the normal
      mode of operation — reconstructs the flow as the residual of customs minus
      implied crush, which accumulates the measurement error of both series.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from agri.data.snapshot import cached

from agri.core.fmt import fmt_num, fmt_pct
from agri.core.stats import regime_runs
from agri.core.units import bushels_per_tonne, strip_vat

# N-H1
DEFAULT_MEAL_YIELD = 0.785
DEFAULT_OIL_YIELD = 0.185
DEFAULT_PROCESSING_CNY_T = 120.0

# Rates to re-verify at time of use — they move (unit trap #2)
DEFAULT_VAT_PROCESSED = 0.09
DEFAULT_IMPORT_DUTY = 0.03

BUSHELS_PER_TONNE_SOYBEAN = bushels_per_tonne("soybean")   # 36.7437


class ChinaSoyError(ValueError):
    """Mis-specified model."""


# ===========================================================================
# The crush margin
# ===========================================================================
def bean_cnf_usd_t(
    cbot_usd_bu: pd.Series, basis_cents_bu: pd.Series, freight_usd_t: pd.Series
) -> pd.Series:
    """Cost delivered to China, in USD/t. The bushel becomes a tonne **before** freight is added.

    Adding a USD/t freight rate to a USD/bushel price is the classic silent mistake:
    the result stays a plausible number and is wrong by a factor of 36.
    """
    frame = pd.concat(
        {"cbot": cbot_usd_bu, "basis": basis_cents_bu, "freight": freight_usd_t}, axis=1, sort=True
    ).dropna()
    if frame.empty:
        raise ChinaSoyError("no common date across CBOT, basis and freight")
    fob_usd_bu = frame["cbot"] + frame["basis"] / 100.0
    return (fob_usd_bu * BUSHELS_PER_TONNE_SOYBEAN + frame["freight"]).rename("bean_cnf_usd_t")


def crush_margin_cny_t(
    meal_dce_cny_t: pd.Series,
    oil_dce_cny_t: pd.Series,
    bean_cnf_usd_t_series: pd.Series,
    usdcny: pd.Series,
    *,
    meal_yield: float = DEFAULT_MEAL_YIELD,
    oil_yield: float = DEFAULT_OIL_YIELD,
    vat_rate: float = DEFAULT_VAT_PROCESSED,
    import_duty: float = DEFAULT_IMPORT_DUTY,
    processing_cny_t: float = DEFAULT_PROCESSING_CNY_T,
) -> pd.DataFrame:
    """Chinese crushing margin, in CNY/t of bean, VAT and duty handled separately.

    Columns: revenue_gross, revenue_ex_vat, bean_cost, margin.
    """
    if meal_yield + oil_yield > 1.0:
        raise ChinaSoyError(
            f"yields sum to {meal_yield + oil_yield:.3f}: more product than incoming bean"
        )
    frame = pd.concat(
        {
            "meal": meal_dce_cny_t,
            "oil": oil_dce_cny_t,
            "bean_usd": bean_cnf_usd_t_series,
            "fx": usdcny,
        },
        axis=1,
        sort=True,
    ).dropna()
    if frame.empty:
        raise ChinaSoyError("no common date across the four series")
    if (frame["fx"] <= 0).any():
        raise ChinaSoyError("USDCNY must be > 0 — check the quoting direction")

    out = pd.DataFrame(index=frame.index)
    out["revenue_gross"] = meal_yield * frame["meal"] + oil_yield * frame["oil"]
    out["revenue_ex_vat"] = strip_vat(out["revenue_gross"], vat_rate)
    out["bean_cost"] = frame["bean_usd"] * frame["fx"] * (1.0 + import_duty)
    out["margin"] = out["revenue_ex_vat"] - out["bean_cost"] - processing_cny_t
    out.attrs["vat_rate"] = vat_rate
    out.attrs["import_duty"] = import_duty
    return out


# ===========================================================================
# The reserve flow, by residual (N-H4)
# ===========================================================================
def reserve_flow(
    imports_t: pd.Series,
    crush_observed_t: pd.Series,
    *,
    direct_use_t: pd.Series | float = 0.0,
) -> pd.DataFrame:
    """`reserve = imports - crush - direct use`, with the error accumulation acknowledged.

    Both input series carry their own measurement error, and the residual adds them
    up. A small-amplitude residual is therefore not interpretable; only large moves
    are. The `is_large` column flags the chosen threshold.
    """
    frame = pd.concat({"imports": imports_t, "crush": crush_observed_t}, axis=1, sort=True).dropna()
    if frame.empty:
        raise ChinaSoyError("no common date across imports and crush")
    direct = (
        pd.Series(float(direct_use_t), index=frame.index)
        if isinstance(direct_use_t, (int, float))
        else pd.Series(direct_use_t).reindex(frame.index).fillna(0.0)
    )
    frame["direct_use"] = direct
    frame["reserve_flow"] = frame["imports"] - frame["crush"] - frame["direct_use"]
    threshold = float(frame["reserve_flow"].abs().quantile(0.60))
    frame["is_large"] = frame["reserve_flow"].abs() > threshold
    frame["is_purchase"] = (frame["reserve_flow"] > threshold).astype(int)
    frame.attrs["large_threshold_t"] = threshold
    return frame


# ===========================================================================
# THE DISCRIMINATING TEST
# ===========================================================================
@dataclass(frozen=True)
class SignatureTest:
    """The sign of b1 decides between political and commercial."""

    beta_margin: float
    pvalue_margin: float
    beta_stock: float
    beta_price: float
    pseudo_r2: float
    n_obs: int
    n_purchases: int

    @property
    def is_significant(self) -> bool:
        return self.pvalue_margin < 0.05

    @property
    def signature(self) -> str:
        if not self.is_significant:
            return "undetermined"
        return "political" if self.beta_margin < 0 else "commercial"

    @property
    def headline(self) -> str:
        if not self.is_significant:
            return (
                f"The link between crush margin and reserve purchases is not "
                f"significant (b1 = {self.beta_margin:+.5f}, p = {self.pvalue_margin:.3f}, "
                f"n = {self.n_obs} of which {self.n_purchases} purchases). On this "
                "sample, a political signature cannot be distinguished from stock "
                "rotation."
            )
        if self.beta_margin < 0:
            return (
                f"Reserve purchases concentrate in the lowest crush-margin quintiles "
                f"(b1 = {self.beta_margin:+.5f}, p = {self.pvalue_margin:.3f}): these "
                "are purchases the commercial sector cannot make, not purchases it "
                "refuses to make."
            )
        return (
            f"Reserve purchases track the crush margin (b1 = {self.beta_margin:+.5f}, "
            f"p = {self.pvalue_margin:.3f}): the signature is commercial, not "
            "political. The state buys when it is economic, like everyone else."
        )


def signature_test(
    purchases: pd.Series,
    margin: pd.Series,
    stock_days: pd.Series,
    price_level: pd.Series,
    *,
    lag: int = 1,
) -> SignatureTest:
    """Logit of the reserve purchase on the lagged crush margin (N-H2).

    The three regressors are **standardised** before estimation. Without that, the
    margin in CNY/t (hundreds) and stock days (tens) produce coefficients of
    incomparable orders of magnitude, and the logit converges poorly.
    """
    import statsmodels.api as sm

    frame = pd.concat(
        {
            "y": purchases,
            "margin": margin.shift(lag),
            "stock": stock_days.shift(lag),
            "price": price_level.shift(lag),
        },
        axis=1,
        sort=True,
    ).dropna()
    if len(frame) < 40:
        raise ChinaSoyError(f"sample too short for a logit: n={len(frame)}")
    n_purchases = int(frame["y"].sum())
    if n_purchases < 5 or n_purchases > len(frame) - 5:
        raise ChinaSoyError(
            f"too little variation to explain: {n_purchases} purchases out of "
            f"{len(frame)} observations. A logit says nothing about a near-constant "
            "variable."
        )

    design = frame[["margin", "stock", "price"]]
    standardised = (design - design.mean()) / design.std()
    standardised = sm.add_constant(standardised)

    model = sm.Logit(frame["y"], standardised).fit(disp=0)
    return SignatureTest(
        beta_margin=float(model.params["margin"]),
        pvalue_margin=float(model.pvalues["margin"]),
        beta_stock=float(model.params["stock"]),
        beta_price=float(model.params["price"]),
        pseudo_r2=float(model.prsquared),
        n_obs=len(frame),
        n_purchases=n_purchases,
    )


def purchases_by_margin_quintile(
    purchases: pd.Series, margin: pd.Series, *, lag: int = 1
) -> pd.DataFrame:
    """Purchase rate by margin quintile — the reading that precedes the logit.

    A five-row table convinces a desk faster than a coefficient, and it immediately
    shows whether the relationship is monotone or concentrated in a single tail.
    """
    frame = pd.concat({"y": purchases, "margin": margin.shift(lag)}, axis=1, sort=True).dropna()
    if len(frame) < 20:
        raise ChinaSoyError(f"sample too short for quintiles: n={len(frame)}")
    frame["quintile"] = pd.qcut(frame["margin"], 5, labels=[1, 2, 3, 4, 5])
    grouped = frame.groupby("quintile", observed=True).agg(
        n_obs=("y", "size"),
        n_purchases=("y", "sum"),
        purchase_rate=("y", "mean"),
        mean_margin=("margin", "mean"),
    )
    return grouped.reset_index()


# Documented orders of magnitude for US Gulf -> China basis and freight (parameterised
# constants, NOT real data): neither is in the Bloomberg export. Same treatment as the
# omitted roll in T1-2 or the constant energy rate in T2-4 — shown as a limitation, not
# hidden.
DEFAULT_BASIS_CENTS_BU = 70.0
DEFAULT_FREIGHT_USD_T = 45.0


@cached('t2_5_china')
def load_real_crush_frame(
    *,
    start: str = "2018-01-01",
    basis_cents_bu: float = DEFAULT_BASIS_CENTS_BU,
    freight_usd_t: float = DEFAULT_FREIGHT_USD_T,
    **margin_kwargs,
) -> pd.DataFrame:
    """Chinese crush margin on **real** CBOT soybean, DCE meal/oil and USDCNY.

    DATA LIMIT, DOCUMENTED: US Gulf FOB basis and China freight are not in the
    Bloomberg export — they remain parameterised flat rates (same values as the
    synthetic fixture's fallback), applied on a real CBOT price. Three of four legs
    (CBOT soybean, DCE meal, DCE oil, USDCNY) are entirely real; only the FOB->CNF
    conversion carries a constant term.

    Returns the full `crush_margin_cny_t` DataFrame (revenue_gross, revenue_ex_vat,
    bean_cost, margin) — not just the margin — so the page can chart the waterfall
    term by term. Does not compute the political/commercial signature test, which
    needs a reserve-purchase signal (`purchases`) that no free public source
    provides as a time series: that part remains illustrative on synthetic data.
    """
    from agri.data.bloomberg_loader import load as load_bloomberg

    cbot = load_bloomberg("cbot_soybean").loc[start:]
    meal = load_bloomberg("dce_soymeal")
    oil = load_bloomberg("dce_soyoil")
    fx = load_bloomberg("usdcny")

    basis = pd.Series(basis_cents_bu, index=cbot.index)
    freight = pd.Series(freight_usd_t, index=cbot.index)
    bean_cnf = bean_cnf_usd_t(cbot, basis, freight)

    margin = crush_margin_cny_t(meal, oil, bean_cnf, fx, **margin_kwargs)
    margin.attrs["basis_cents_bu"] = basis_cents_bu
    margin.attrs["freight_usd_t"] = freight_usd_t
    margin.attrs["real_legs"] = ["cbot_soybean", "dce_soymeal", "dce_soyoil", "usdcny"]
    margin.attrs["parametrized_legs"] = ["basis_cents_bu", "freight_usd_t"]
    return margin


# ===========================================================================
# THE TEST THAT DOES NOT NEED AUCTION DATA
# ===========================================================================
# The signature test above is binary and clean, but it needs a reserve-purchase series the
# export does not contain — and that Sinograin does not publish as a time series. It is
# replaced by an argument that needs no flow data at all, only prices.
#
# The idea: the Chinese crush margin bounds from above what a crusher can pay for a tonne
# of bean delivered. Subtracting the CBOT converted to a tonne basis, this bound becomes a
# **budget for origin basis and freight** — what an originator has available to go source
# the bean, wherever from. When this budget falls below zero, no origin works at all: even
# a free bean, shipped for free, would not make the crush pay. Cargoes arriving in those
# windows are non-commercial **by arithmetic construction**, with no statistical test
# needed.
@dataclass(frozen=True)
class OriginationBudget:
    """What an originator can spend on basis + freight, and when they can spend nothing.

    `frame` carries the budget day by day; the properties give its regimes. The
    deliverable is the **calendar** of impossible windows, not an average level: an
    object an insider checks against their own arrivals.
    """

    frame: pd.DataFrame
    freight_reference_usd_t: float

    @property
    def median_budget(self) -> float:
        return float(self.frame["budget_usd_t"].median())

    @property
    def last_budget(self) -> float:
        return float(self.frame["budget_usd_t"].iloc[-1])

    @property
    def share_impossible(self) -> float:
        """Share of sessions where NO origin works, zero freight and basis included."""
        return float((self.frame["budget_usd_t"] < 0).mean())

    @property
    def share_below_freight(self) -> float:
        """Share of sessions where freight alone consumes the whole budget, leaving
        nothing for origin basis — the bean would have to be bought BELOW the CBOT price."""
        return float((self.frame["budget_usd_t"] < self.freight_reference_usd_t).mean())

    @property
    def headline(self) -> str:
        return (
            f"The Chinese crush can fund {fmt_num(self.median_budget, 0)} USD/t of "
            f"basis and freight at the median, and {fmt_num(self.last_budget, 0)} at "
            f"the last print. But {fmt_pct(self.share_impossible, 1)} of sessions "
            f"show a **negative** budget — a free bean, shipped for free, would not "
            f"make the crush pay — and {fmt_pct(self.share_below_freight)} put it "
            f"below the cost of freight alone "
            f"({fmt_num(self.freight_reference_usd_t, 0)} USD/t)."
        )


@cached('t3_4_budget', from_frame=lambda f: OriginationBudget(frame=f, freight_reference_usd_t=DEFAULT_FREIGHT_USD_T))
def affordable_origination_budget(
    *,
    start: str = "2018-01-01",
    freight_reference_usd_t: float = DEFAULT_FREIGHT_USD_T,
    processing_cny_t: float = DEFAULT_PROCESSING_CNY_T,
    import_duty: float = DEFAULT_IMPORT_DUTY,
    **margin_kwargs,
) -> OriginationBudget:
    """The basis + freight budget the Chinese crush margin allows, in USD/tonne.

        revenue_ex_vat = (0.785 x meal_DCE + 0.185 x oil_DCE) / (1 + VAT)
        cnf_max_cny_t  = (revenue_ex_vat - processing) / (1 + duty)
        cnf_max_usd_t  = cnf_max_cny_t / USDCNY
        budget         = cnf_max_usd_t - CBOT_usd_bu x 36.7437

    The budget is what is left over to go source the bean. It depends on **no**
    assumption of basis or freight — which is precisely what makes it useful: the
    two terms the export does not contain drop out of the calculation instead of
    entering it.
    """
    from agri.data.bloomberg_loader import load

    crush = load_real_crush_frame(start=start, **margin_kwargs)
    aligned = pd.concat(
        {
            "revenue_ex_vat": crush["revenue_ex_vat"],
            "cbot_usd_bu": load("cbot_soybean"),
            "usdcny": load("usdcny"),
        },
        axis=1,
        sort=True,
    ).dropna()
    aligned = aligned[aligned.index >= pd.Timestamp(start)]
    if aligned.empty:
        raise ChinaSoyError(f"no common date across the Chinese crush legs after {start}")
    if (aligned["usdcny"] <= 0).any():
        raise ChinaSoyError("USDCNY zero or negative — check the quoting direction")

    cnf_max_cny = (aligned["revenue_ex_vat"] - processing_cny_t) / (1.0 + import_duty)
    aligned["cnf_max_usd_t"] = cnf_max_cny / aligned["usdcny"]
    aligned["cbot_usd_t"] = aligned["cbot_usd_bu"] * BUSHELS_PER_TONNE_SOYBEAN
    aligned["budget_usd_t"] = aligned["cnf_max_usd_t"] - aligned["cbot_usd_t"]
    aligned["impossible"] = aligned["budget_usd_t"] < 0
    aligned["below_freight"] = aligned["budget_usd_t"] < freight_reference_usd_t

    return OriginationBudget(
        frame=aligned, freight_reference_usd_t=float(freight_reference_usd_t)
    )


def impossible_windows(
    budget: OriginationBudget, *, threshold_usd_t: float = 0.0, min_obs: int = 3
) -> pd.DataFrame:
    """The calendar of windows where the origination budget falls below a threshold.

    This is the page's deliverable: dates, not a coefficient. An originator checks
    them against their own arrivals, and the question "did you load during those?"
    gets answered yes or no.
    """
    return regime_runs(
        budget.frame["budget_usd_t"] < threshold_usd_t,
        depth=budget.frame["budget_usd_t"],
        min_obs=min_obs,
    )


__all__ = [
    "BUSHELS_PER_TONNE_SOYBEAN",
    "ChinaSoyError",
    "DEFAULT_BASIS_CENTS_BU",
    "DEFAULT_FREIGHT_USD_T",
    "DEFAULT_IMPORT_DUTY",
    "DEFAULT_PROCESSING_CNY_T",
    "OriginationBudget",
    "SignatureTest",
    "affordable_origination_budget",
    "bean_cnf_usd_t",
    "crush_margin_cny_t",
    "impossible_windows",
    "load_real_crush_frame",
    "purchases_by_margin_quintile",
    "reserve_flow",
    "signature_test",
]
