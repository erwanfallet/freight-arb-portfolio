"""T1-1 — Freight inside the C&F calculation.

THESIS
------
On the marginal band — the days when the arb sits a few dollars from breakeven — freight
does not *add noise* to the arb: it **determines** it. The documented dispute at Louis
Dreyfus between trading desks and the freight department was not an ego conflict, it was a
P&L attribution conflict, and it is arithmetically measurable.

    arb_usd_t = CIF_dest − FOB_orig − freight_usd_t − financing − insurance

The entire disagreement sits on one term: `freight_usd_t`. Both sides are right.

WHERE THE DISAGREEMENT COMES FROM (sourced, quotable)
--------------------------------------------------------
Mat Halsall interview, Commodity Conversations, 25 Nov 2024: at Louis Dreyfus, frequent
disputes between trading desks and the freight department; traders disputed the freight
rate without understanding what freight is or what it is made of. Halsall offered to learn
the traders' job to teach them his, and spent a lot of time in S&D meetings
reverse-engineering the physical freight business.

    Trading desk       : "your rate is not the market, I can fix cheaper spot."
    Freight department  : "you're looking at an index, not a cost. It's missing vessel
                            positioning, ballast, bunkers at the fixture date, laytime,
                            demurrage, round-trip time, the value of cargo options."

THE UNIT TRAP
-------------
The index is quoted in **USD/day** (TCE) or in USD/t on a reference route; the economic
unit of the C&F calculation is **USD/tonne on your route, on your date**. Getting from one
to the other is not a conversion, it is a voyage estimate — see `core/voyage.py`. The
conversion factor is not a physical constant: it is a model with contestable assumptions,
and two of them carry the whole disagreement.

THE THREE CONVENTIONS
----------------------
    freight_index     ballast = 0, what the trading desk argues for
    freight_full      ballast = 1, bunkers as of fixture date, what the freight dept argues for
    freight_internal  90-day rolling mean of freight_full — what a freight department
                      sells internally, because it has to sell a *stable* rate

ASSUMPTIONS
-----------
F-H1  `ballast_share` defaults to 1.0 for `freight_full`, 0.0 for `freight_index`. This is
      the central disagreement, and the page sweeps it rather than settling it.
F-H2  The internal convention is a 90-day rolling mean. A real freight department smooths
      differently (budget, annually negotiated rate); the 90d MA is the simplest proxy
      that reproduces the property that matters: stability, hence the lag.
F-H3  Insurance is a flat USD/t rate, not a percentage of value. Strictly wrong,
      negligible next to freight, and parameterised.
F-H4  Financing runs over the voyage duration plus credit days. The rate is exogenous.
      The term is small but not zero on the marginal band — so it is shown.
F-H5  No demurrage cost is modelled. It is one of the terms the freight department cites,
      and omitting it **understates** `freight_full`: the bias runs against the thesis,
      which is the right direction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from agri.data.snapshot import cached

from agri.core.breakeven import Breakeven, NoBreakevenInRange, solve_breakeven
from agri.core.stats import ProportionCI, clopper_pearson, regime_runs
from agri.core.voyage import (
    Route,
    VesselClass,
    VoyageParams,
    voyage_freight_series,
)

# F-H2
DEFAULT_INTERNAL_WINDOW_DAYS = 90
# Default width of the marginal decision zone, in USD/t
DEFAULT_MARGINAL_BAND_USD_T = 5.0

CONVENTIONS = ("index", "full", "internal")


class FreightCfError(ValueError):
    """Mis-specified decomposition."""


# ===========================================================================
# The accounting identity
# ===========================================================================
def financing_cost_usd_t(
    fob_usd_t: pd.Series | float,
    freight_usd_t: pd.Series | float,
    *,
    annual_rate: float,
    voyage_days: float,
    credit_days: float,
) -> pd.Series | float:
    """Financing the cargo over the voyage duration plus credit (F-H4).

    360-day base, money-market convention — not 365. The gap is 1.4% on the term,
    invisible on a wide arb and not on the marginal band.
    """
    if annual_rate < 0:
        raise FreightCfError(f"negative financing rate: {annual_rate}")
    if voyage_days < 0 or credit_days < 0:
        raise FreightCfError("durations must be >= 0")
    return (fob_usd_t + freight_usd_t) * annual_rate * (voyage_days + credit_days) / 360.0


def arb_usd_t(
    cif_usd_t: pd.Series,
    fob_usd_t: pd.Series,
    freight_usd_t: pd.Series,
    *,
    annual_rate: float,
    voyage_days: float,
    credit_days: float = 30.0,
    insurance_usd_t: float = 0.85,
) -> pd.Series:
    """The full identity. Only one term is disputed: `freight_usd_t`."""
    financing = financing_cost_usd_t(
        fob_usd_t,
        freight_usd_t,
        annual_rate=annual_rate,
        voyage_days=voyage_days,
        credit_days=credit_days,
    )
    return cif_usd_t - fob_usd_t - freight_usd_t - financing - insurance_usd_t


# ===========================================================================
# The three conventions, side by side
# ===========================================================================
def build_conventions(
    tce: pd.Series,
    vlsfo: pd.Series,
    mgo: pd.Series,
    cif: pd.Series,
    fob: pd.Series,
    *,
    vessel: VesselClass,
    route: Route,
    params: VoyageParams,
    annual_rate: float = 0.055,
    credit_days: float = 30.0,
    insurance_usd_t: float = 0.85,
    internal_window_days: int = DEFAULT_INTERNAL_WINDOW_DAYS,
) -> pd.DataFrame:
    """Freight and the arb under the three conventions, on dates common to all of them.

    Columns: freight_index, freight_full, freight_internal, arb_index, arb_full,
    arb_internal, spread_full_index, and `disagreement` — true when the three
    conventions do not agree on the sign of the arb.

    No forward-fill anywhere: the bunker price on the fixture date is precisely what
    the disagreement is about, filling it in would make it disappear.
    """
    freight_index = voyage_freight_series(
        tce, vlsfo, mgo, vessel=vessel, route=route, params=params.with_ballast(0.0)
    )
    freight_full = voyage_freight_series(
        tce, vlsfo, mgo, vessel=vessel, route=route, params=params.with_ballast(1.0)
    )
    # F-H2: full min_periods — refuse an "internal" average built on three points
    freight_internal = freight_full.rolling(
        internal_window_days, min_periods=internal_window_days
    ).mean()

    frame = pd.concat(
        {
            "cif": cif,
            "fob": fob,
            "freight_index": freight_index,
            "freight_full": freight_full,
            "freight_internal": freight_internal,
        },
        axis=1,
    ).dropna()
    if frame.empty:
        raise FreightCfError(
            "no common date across prices and freight after the smoothing window — "
            f"at least {internal_window_days} days of freight history are needed before "
            "the first usable arb date"
        )

    # voyage duration feeds financing: use the full convention's, the only one that
    # represents a complete cycle
    reference_days = _reference_voyage_days(vessel, route, params)

    for convention in CONVENTIONS:
        frame[f"arb_{convention}"] = arb_usd_t(
            frame["cif"],
            frame["fob"],
            frame[f"freight_{convention}"],
            annual_rate=annual_rate,
            voyage_days=reference_days,
            credit_days=credit_days,
            insurance_usd_t=insurance_usd_t,
        )

    frame["spread_full_index"] = frame["freight_full"] - frame["freight_index"]
    signs = np.sign(frame[[f"arb_{c}" for c in CONVENTIONS]].to_numpy())
    frame["disagreement"] = (signs != signs[:, [0]]).any(axis=1)
    frame.attrs["reference_voyage_days"] = reference_days
    frame.attrs["internal_window_days"] = internal_window_days
    return frame


def _reference_voyage_days(vessel: VesselClass, route: Route, params: VoyageParams) -> float:
    from agri.core.voyage import HOURS_PER_DAY

    laden = route.distance_laden_nm / (params.speed_laden_kn * HOURS_PER_DAY)
    ballast = route.distance_ballast_nm / (params.speed_ballast_kn * HOURS_PER_DAY)
    return laden + ballast + params.port_days + params.wait_days


# ===========================================================================
# S4 — the disagreement panel
# ===========================================================================
def sign_flip_rate(
    frame: pd.DataFrame, *, left: str = "index", right: str = "full"
) -> ProportionCI:
    """Share of days on which two conventions do not give the arb the same sign.

    Exact binomial CI (Clopper-Pearson) rather than a normal approximation: on a
    one-year sample of business days with a rate near 0 or 1, the normal approximation
    produces bounds outside [0, 1].
    """
    a = np.sign(frame[f"arb_{left}"].to_numpy())
    b = np.sign(frame[f"arb_{right}"].to_numpy())
    flips = int((a != b).sum())
    return clopper_pearson(flips, len(frame))


def spread_distribution(frame: pd.DataFrame) -> dict[str, float]:
    """Median and IQR of the `freight_full − freight_index` gap, not just the mean.

    The distribution is asymmetric **by construction**: ballast can only add cost.
    Reporting a mean alone on a left-bounded distribution is the kind of shortcut a
    practitioner spots immediately.
    """
    spread = frame["spread_full_index"].dropna()
    q1, q3 = spread.quantile([0.25, 0.75])
    return {
        "median": float(spread.median()),
        "mean": float(spread.mean()),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "min": float(spread.min()),
        "max": float(spread.max()),
    }


def spread_seasonality(frame: pd.DataFrame) -> pd.Series:
    """Average `full − index` gap by calendar month. Ballast has a season."""
    spread = frame["spread_full_index"].dropna()
    return spread.groupby(spread.index.month).mean()


# ===========================================================================
# S5 — the marginal decision zone: THE number for the email
# ===========================================================================
@dataclass(frozen=True)
class MarginalZone:
    """What happens on the marginal band, where freight stops being a detail."""

    band_usd_t: float
    n_total: int
    n_in_band: int
    n_decided_by_convention: int
    share_of_sample: float
    decision_rate: ProportionCI

    @property
    def headline(self) -> str:
        """The number-bearing sentence, generated from the data."""
        return (
            f"On days when the arb sits within {self.band_usd_t:g} USD/t of breakeven — "
            f"{self.share_of_sample:.0%} of the sample — it is the freight convention, "
            f"not the market, that decides whether the arb is open, in "
            f"{self.decision_rate.point:.0%} of cases "
            f"(95% exact CI: {self.decision_rate.lo:.0%}–{self.decision_rate.hi:.0%})."
        )


def marginal_decision_zone(
    frame: pd.DataFrame, *, band_usd_t: float = DEFAULT_MARGINAL_BAND_USD_T
) -> MarginalZone:
    """Restricts to marginal-band days, and measures the share decided by the convention.

    "Decided by the convention" has a precise meaning: among the days where the arb
    under the reference convention sits within `band_usd_t` of zero, the ones where two
    conventions give opposite signs. This is not "freight matters" — it is "freight
    decides".
    """
    if band_usd_t <= 0:
        raise FreightCfError(f"the band must be > 0, got {band_usd_t}")

    in_band = frame[frame["arb_full"].abs() < band_usd_t]
    n_in_band = len(in_band)
    if n_in_band == 0:
        return MarginalZone(
            band_usd_t=band_usd_t,
            n_total=len(frame),
            n_in_band=0,
            n_decided_by_convention=0,
            share_of_sample=0.0,
            decision_rate=clopper_pearson(0, 1),
        )

    decided = int(
        (np.sign(in_band["arb_index"]) != np.sign(in_band["arb_full"])).sum()
    )
    return MarginalZone(
        band_usd_t=band_usd_t,
        n_total=len(frame),
        n_in_band=n_in_band,
        n_decided_by_convention=decided,
        share_of_sample=n_in_band / len(frame),
        decision_rate=clopper_pearson(decided, n_in_band),
    )


# ===========================================================================
# Tipping point — ballast_share*
# ===========================================================================
def ballast_breakeven(
    tce_usd_day: float,
    vlsfo_usd_t: float,
    mgo_usd_t: float,
    cif_usd_t: float,
    fob_usd_t: float,
    *,
    vessel: VesselClass,
    route: Route,
    params: VoyageParams,
    annual_rate: float = 0.055,
    credit_days: float = 30.0,
    insurance_usd_t: float = 0.85,
    ballast_history: pd.Series | None = None,
) -> Breakeven:
    """`ballast_share*`: the share of ballast charged at which the arb flips sign.

    This is the page's deliverable. Typical answer: "beyond 40% of ballast charged, the
    Santos->Qingdao arb closes at today's prices" — a threshold a practitioner disputes
    or confirms in ten seconds.

    Raises `NoBreakevenInRange` when the arb keeps the same sign across the whole [0, 1]
    range. That is not a failure: "even charging 100% of ballast, the arb stays open" is
    a publishable claim, and often the strongest one of the day.
    """
    from agri.core.voyage import voyage_freight_usd_t

    def margin(share: float) -> float:
        local = params.with_ballast(float(np.clip(share, 0.0, 1.0)))
        breakdown = voyage_freight_usd_t(
            tce_usd_day, vlsfo_usd_t, mgo_usd_t, vessel=vessel, route=route, params=local
        )
        financing = financing_cost_usd_t(
            fob_usd_t,
            breakdown.freight_usd_t,
            annual_rate=annual_rate,
            voyage_days=breakdown.total_days,
            credit_days=credit_days,
        )
        return cif_usd_t - fob_usd_t - breakdown.freight_usd_t - financing - insurance_usd_t

    return solve_breakeven(
        margin,
        0.0,
        1.0,
        theta_current=params.ballast_share,
        theta_history=ballast_history,
        theta_label="ballast_share",
        margin_label="arb",
    )


def sensitivity_grid(
    tce_usd_day: float,
    vlsfo_usd_t: float,
    mgo_usd_t: float,
    cif_usd_t: float,
    fob_usd_t: float,
    *,
    vessel: VesselClass,
    route: Route,
    params: VoyageParams,
    ballast_values: np.ndarray | None = None,
    bunker_shifts_pct: np.ndarray | None = None,
    annual_rate: float = 0.055,
) -> pd.DataFrame:
    """S6 — heatmap `ballast_share × bunker-price shift -> arb`.

    The two parameters that carry the disagreement, crossed. The bunker shift is in
    percentage rather than days: a date shift only matters through the price it
    implies, and reasoning in price keeps the sensitivity readable without a long
    bunker series.
    """
    from agri.core.voyage import voyage_freight_usd_t

    ballast_values = (
        np.linspace(0.0, 1.0, 11) if ballast_values is None else np.asarray(ballast_values)
    )
    bunker_shifts_pct = (
        np.linspace(-0.30, 0.30, 13)
        if bunker_shifts_pct is None
        else np.asarray(bunker_shifts_pct)
    )

    rows = []
    for share in ballast_values:
        for shift in bunker_shifts_pct:
            local = params.with_ballast(float(share))
            breakdown = voyage_freight_usd_t(
                tce_usd_day,
                vlsfo_usd_t * (1.0 + shift),
                mgo_usd_t * (1.0 + shift),
                vessel=vessel,
                route=route,
                params=local,
            )
            financing = financing_cost_usd_t(
                fob_usd_t,
                breakdown.freight_usd_t,
                annual_rate=annual_rate,
                voyage_days=breakdown.total_days,
                credit_days=30.0,
            )
            arb = cif_usd_t - fob_usd_t - breakdown.freight_usd_t - financing - 0.85
            rows.append(
                {
                    "ballast_share": float(share),
                    "bunker_shift_pct": float(shift),
                    "freight_usd_t": breakdown.freight_usd_t,
                    "arb_usd_t": arb,
                    "is_open": arb > 0,
                }
            )
    return pd.DataFrame(rows)


# ===========================================================================
# S7 — P&L attribution between the two desks
# ===========================================================================
def pnl_attribution(frame: pd.DataFrame, *, cargo_t: float) -> pd.DataFrame:
    """The P&L shifted from one desk to the other by the internal convention.

        pnl_shifted = (freight_internal − freight_full) × cargo_t

    Positive sign: the freight department charged above its cost of the day, so the
    trading desk was debited and the department credited. Negative: the reverse.
    Cumulated over the sample, this is the amount the LDC dispute was really about —
    and it has an identifiable owner on both sides.
    """
    if cargo_t <= 0:
        raise FreightCfError(f"cargo_t must be > 0, got {cargo_t}")
    out = pd.DataFrame(index=frame.index)
    out["gap_usd_t"] = frame["freight_internal"] - frame["freight_full"]
    out["pnl_shifted_usd"] = out["gap_usd_t"] * cargo_t
    out["pnl_shifted_cum_usd"] = out["pnl_shifted_usd"].cumsum()
    out["credited"] = np.where(
        out["gap_usd_t"] > 0, "freight department", "trading desk"
    )
    return out


# ===========================================================================
# ON REAL DATA — what the published rate implies, by convention
# ===========================================================================
@dataclass(frozen=True)
class ImpliedTceSpread:
    """The same published route rate, translated into a TCE under both conventions.

    This is the disagreement expressed in the unit the freight department actually
    thinks in. A trading desk reads "55 USD/t" and concludes the market pays 55 USD/t.
    A freight department reads the same print and asks: over how many days? If the
    voyage includes ballast repositioning, the same revenue has to cover nearly twice
    as many days, so the TCE this rate implies is nearly half as large.
    """

    route_rate_usd_t: pd.Series
    tce_no_ballast: pd.Series
    tce_full_ballast: pd.Series

    @property
    def spread(self) -> pd.Series:
        return (self.tce_no_ballast - self.tce_full_ballast).rename("spread_usd_day")

    @property
    def headline(self) -> str:
        latest_rate = float(self.route_rate_usd_t.iloc[-1])
        no_ballast = float(self.tce_no_ballast.iloc[-1])
        full = float(self.tce_full_ballast.iloc[-1])
        return (
            f"The published route rate of {latest_rate:,.1f} USD/t implies a TCE of "
            f"{no_ballast:,.0f} USD/day if no ballast is charged at all, and "
            f"{full:,.0f} USD/day if it is charged in full — a gap of "
            f"{no_ballast - full:,.0f} USD/day on the same market print. This is the "
            "number the two desks are arguing about, in the freight department's unit."
        )


def implied_tce_by_convention(
    route_rate_usd_t: pd.Series,
    vlsfo_usd_t: pd.Series,
    mgo_usd_t: pd.Series,
    *,
    vessel: VesselClass,
    route: Route,
    params: VoyageParams,
) -> ImpliedTceSpread:
    """Inverts the voyage model on a **genuinely published** route rate.

    The published rate is data; the TCE it implies is a conclusion, and it depends
    entirely on the ballast convention. We do not pick the right convention — we show
    the gap the choice produces.
    """
    from agri.core.voyage import implied_tce_from_freight

    aligned = pd.concat(
        {"rate": route_rate_usd_t, "vlsfo": vlsfo_usd_t, "mgo": mgo_usd_t}, axis=1, sort=True
    ).dropna()
    if aligned.empty:
        raise FreightCfError("no common date between the route rate and bunkers")

    def invert(share: float) -> pd.Series:
        local = params.with_ballast(share)
        values = [
            implied_tce_from_freight(
                row.rate, row.vlsfo, row.mgo, vessel=vessel, route=route, params=local
            )
            for row in aligned.itertuples()
        ]
        return pd.Series(values, index=aligned.index)

    return ImpliedTceSpread(
        route_rate_usd_t=aligned["rate"],
        tce_no_ballast=invert(0.0).rename("tce_ballast_0"),
        tce_full_ballast=invert(1.0).rename("tce_ballast_1"),
    )


@dataclass(frozen=True)
class MarketImpliedBallast:
    """The share of ballast the market prices, once a reference TCE is known.

    THE PAGE'S DELIVERABLE. The disagreement is not settleable in theory — but if the
    TCE at which owners are actually chartering is known, the published route rate
    reveals the share of ballast the market has already priced in. The question to the
    desk stops being "who is right" and becomes "the market prices X%, is that what
    you charge internally?".
    """

    reference_tce_usd_day: float
    implied_share: float | None
    route_rate_usd_t: float
    bracket: tuple[float, float]
    reason: str = ""

    @property
    def headline(self) -> str:
        if self.implied_share is None:
            return (
                f"At a reference TCE of {self.reference_tce_usd_day:,.0f} USD/day, the "
                f"published rate of {self.route_rate_usd_t:,.1f} USD/t is not consistent "
                f"with any ballast share in [{self.bracket[0]:.0%}, {self.bracket[1]:.0%}] "
                f"— {self.reason}"
            )
        return (
            f"At a reference TCE of {self.reference_tce_usd_day:,.0f} USD/day, the "
            f"published rate of {self.route_rate_usd_t:,.1f} USD/t implies the market is "
            f"charging **{self.implied_share:.0%} of ballast repositioning**. That is the "
            "figure your internal rate has to match to be neutral."
        )


def market_implied_ballast_share(
    route_rate_usd_t: float,
    reference_tce_usd_day: float,
    vlsfo_usd_t: float,
    mgo_usd_t: float,
    *,
    vessel: VesselClass,
    route: Route,
    params: VoyageParams,
) -> MarketImpliedBallast:
    """Solves for the ballast share such that the model reproduces the published rate.

    Freight is **affine and increasing** in ballast share (each point of ballast adds
    days and bunkers), so the root is unique when it exists. When it does not, that is
    a result: the published rate is outside the range the model can produce, which
    flags a voyage assumption to revisit — speed, port costs, or vessel class — rather
    than an implausible ballast share.
    """
    from agri.core.voyage import voyage_freight_usd_t

    def gap(share: float) -> float:
        local = params.with_ballast(float(np.clip(share, 0.0, 1.0)))
        modelled = voyage_freight_usd_t(
            reference_tce_usd_day, vlsfo_usd_t, mgo_usd_t,
            vessel=vessel, route=route, params=local,
        ).freight_usd_t
        return route_rate_usd_t - modelled

    try:
        solution = solve_breakeven(
            gap, 0.0, 1.0, theta_label="ballast_share", margin_label="gap to published rate"
        )
        return MarketImpliedBallast(
            reference_tce_usd_day=reference_tce_usd_day,
            implied_share=solution.theta_star,
            route_rate_usd_t=route_rate_usd_t,
            bracket=(0.0, 1.0),
        )
    except NoBreakevenInRange as error:
        # `gap` is DECREASING in ballast share (more ballast -> pricier modelled freight
        # -> smaller gap). The message must therefore name the BINDING bound, i.e. the
        # one where the model comes closest to the published rate: full ballast when the
        # gap stays positive everywhere, zero ballast when it stays negative everywhere.
        # Naming the other bound would be true but useless — it is the farther one.
        if error.margin_lo > 0:
            reason = (
                "the published rate exceeds what the model produces even charging 100% "
                "of ballast repositioning: the chosen reference TCE is too low for this "
                "route, or the voyage assumptions (speed, port days, port costs) "
                "understate the cost of the cycle"
            )
        else:
            reason = (
                "the published rate stays below what the model produces even charging "
                "no ballast at all: the chosen reference TCE is too high for this route"
            )
        return MarketImpliedBallast(
            reference_tce_usd_day=reference_tce_usd_day,
            implied_share=None,
            route_rate_usd_t=route_rate_usd_t,
            bracket=(0.0, 1.0),
            reason=reason,
        )


@cached('t1_1_route', from_frame=lambda f: ImpliedTceSpread(f["route_rate_usd_t"], f["tce_no_ballast"], f["tce_full_ballast"]))
def load_real_route_frame(
    *,
    vessel_key: str = "panamax",
    route_key: str = "santos_qingdao",
    params: VoyageParams | None = None,
    mgo_premium: float = 1.35,
) -> ImpliedTceSpread:
    """Loads the real P8 route rate and real bunkers, then inverts the model.

    `mgo_premium` reconstructs MGO from VLSFO: the export has no separate MGO series,
    and MGO has historically quoted 30 to 40% above VLSFO. The term is small — port
    bunkers weigh a few percent of voyage cost — but the parameter is exposed rather
    than buried in a constant.
    """
    from agri.core.voyage import ROUTES, VESSELS
    from agri.data.bloomberg_loader import load as load_bloomberg

    rate = load_bloomberg("p8_route_usd_t")
    vlsfo = load_bloomberg("vlsfo_singapore")
    mgo = (vlsfo * mgo_premium).rename("mgo_reconstructed")

    return implied_tce_by_convention(
        rate, vlsfo, mgo,
        vessel=VESSELS[vessel_key], route=ROUTES[route_key],
        params=params or VoyageParams(),
    )


def disagreement_episodes(frame: pd.DataFrame, *, min_days: int = 3) -> pd.DataFrame:
    """Consecutive episodes where the three conventions do not agree on sign.

    "The conventions diverged for 11 business days in March" is a dated, verifiable
    sentence; "the conventions sometimes diverge" is not.
    """
    return regime_runs(
        frame["disagreement"], depth=frame["spread_full_index"], min_obs=min_days
    )


__all__ = [
    "CONVENTIONS",
    "FreightCfError",
    "ImpliedTceSpread",
    "MarginalZone",
    "MarketImpliedBallast",
    "NoBreakevenInRange",
    "arb_usd_t",
    "ballast_breakeven",
    "build_conventions",
    "disagreement_episodes",
    "financing_cost_usd_t",
    "implied_tce_by_convention",
    "load_real_route_frame",
    "marginal_decision_zone",
    "market_implied_ballast_share",
    "pnl_attribution",
    "sensitivity_grid",
    "sign_flip_rate",
    "spread_distribution",
    "spread_seasonality",
]
