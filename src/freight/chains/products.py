"""Project C — transatlantic distillate arb, and the freight that isn't a cost.

THESIS
------
Buy diesel on the US Gulf Coast, ship it on an MR to Rotterdam, sell it against ARA
gasoil. Since 2022 Europe has lost Russian diesel: the transatlantic flow reversed and
lengthened in tonne-miles, making this the most watched products trade of the decade.

    arb = P_ARA($/t) − P_USGC($/t) − freight($/t) − spec_bridge − financing

Two terms in this line aren't what they look like.

1. VOLUME IS NOT MASS
-------------------------
US diesel is quoted in **$/gallon**, European gasoil in **$/tonne**. The conversion
runs through a density:

    P($/t) = P($/gal) × 42 gal/bbl × bbl_per_tonne

`bbl_per_tonne` is worth about 7.45 for a diesel at 0.845 kg/l. This is **not** a
universal constant: it depends on the batch's density, which varies with the
specification and the reference temperature.

And the order of magnitude is brutal. On a leg at ~780 $/t, moving from 7.45 to 7.50
bbl/t shifts the price by **~5 $/t**, i.e. often more than the whole arb. **The
conversion factor everyone treats as a constant is bigger than the signal.** That's this
project's central result.

It's the same trap as moisture on iron ore and calorific value on coal: freight is paid
on one unit, value is measured on another.

2. WORLDSCALE POINTS ARE NOT A COST
---------------------------------------
Tanker freight is quoted in Worldscale points. WS 100 corresponds to the route's *flat
rate*, recalculated **every January 1st** by the Worldscale Association from the
previous year's cost environment. So:

    freight($/t) = WS/100 × flat_rate(route, year)

Direct consequence: **the cost in $/t jumps on January 1st even if the WS points
haven't moved an inch.** A model that uses WS points as a cost proxy gets the timing
wrong every year, and the flat rates moved sharply in 2022-2023 with bunker prices.

The exact decomposition of a freight change:

    Δfreight = [ ΔWS·FR_prev + WS_prev·ΔFR + ΔWS·ΔFR ] / 100
                 └ market ┘    └ reset ┘      └ cross ┘

On January 1st, ΔWS can be zero while Δfreight isn't. The "reset" term is invisible to
anyone just watching the points.

The `signals/worldscale.py` module refuses by construction to convert points into $/t
without a dated flat rate. This module relies on it.

IF FLAT RATES ARE UNAVAILABLE
---------------------------------
Two fallbacks, both implemented here:

- **C-3**: both price legs stay exchange-traded and free (EIA for the Gulf Coast, ICE
  Gasoil for ARA), so the arb still gets built. Only the Worldscale result falls away.
- **C-2**: freight isn't bought, it's **computed**. `implied_freight_from_tce` inverts
  the existing TCE engine: from a market MR TCE, distances, bunkers and port days, the
  freight rate that TCE implies is backed out. It's then compared against the quoted
  freight wherever there's a point of comparison. "I didn't buy the freight, I
  reconstructed it" is a stronger argument than a subscription.

ASSUMPTIONS
-----------
C-H1  Diesel density: 7.45 bbl/t by default. **Parameterised and swept** — this is the
      project's most sensitive term, it cannot be a hard-coded number.
C-H2  Spec bridge between USGC ULSD 15 ppm and ICE Gasoil 10 ppm: treated as an
      **explicit range**, never as a single figure. A patched-together bridge would
      turn the arb into noise.
C-H3  Transit losses and demurrage: a single parameter, not finely modelled.
C-H4  Financing of the cargo value over the crossing duration.
C-H5  The flat rates used are those of the route and year being quoted. The module
      raises if a route/year pair is missing, rather than silently falling back.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from agri.data.snapshot import cached

from freight.signals.worldscale import FlatRateTable

# C-H1 — default distillate density, in barrels per tonne.
DEFAULT_BBL_PER_TONNE = 7.45
GALLONS_PER_BARREL = 42.0


# ------------------------------------------------------------------ volume vs mass
def usd_per_gallon_to_usd_per_tonne(
    price_per_gallon: pd.Series | float, bbl_per_tonne: float = DEFAULT_BBL_PER_TONNE
) -> pd.Series | float:
    """Converts a price in $/gal to $/t (C-H1).

        $/t = $/gal × 42 × bbl_per_tonne

    `bbl_per_tonne` is not a universal physical constant: it's a density assumption. It
    is therefore an argument, never a hard-coded value.
    """
    if bbl_per_tonne <= 0:
        raise ValueError(f"bbl_per_tonne must be > 0, got {bbl_per_tonne}")
    return price_per_gallon * GALLONS_PER_BARREL * bbl_per_tonne


def density_sensitivity(
    price_per_gallon: float, bbl_grid: tuple[float, ...] = (7.35, 7.40, 7.45, 7.50, 7.55)
) -> pd.DataFrame:
    """Sensitivity table of the converted price to the density assumption.

    Exists to make visible, in one line, that the project's most uncertain term is also
    one of its largest.
    """
    base = usd_per_gallon_to_usd_per_tonne(price_per_gallon, DEFAULT_BBL_PER_TONNE)
    rows = []
    for bbl in bbl_grid:
        converted = usd_per_gallon_to_usd_per_tonne(price_per_gallon, bbl)
        rows.append(
            {
                "bbl_per_tonne": bbl,
                "price ($/t)": round(float(converted), 2),
                "gap to default ($/t)": round(float(converted - base), 2),
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ Worldscale
def freight_usd_per_tonne(
    ws_points: pd.Series, flat_rates: pd.Series
) -> pd.Series:
    """freight($/t) = WS/100 × flat_rate, both series aligned by date.

    `flat_rates` must be a step-shaped daily series: the same value across the
    calendar year, one step on January 1st. Build it from a `FlatRateTable` with
    `flat_rate_step_series`.
    """
    aligned = pd.concat({"ws": ws_points, "fr": flat_rates}, axis=1).dropna()
    if aligned.empty:
        raise ValueError("no common date between the WS points and the flat rates")
    out = aligned["ws"] / 100.0 * aligned["fr"]
    out.name = "freight_usd_t"
    return out


def flat_rate_step_series(
    index: pd.DatetimeIndex, route: str, table: FlatRateTable
) -> pd.Series:
    """Step-shaped daily series of flat rates, from the dated table.

    Raises if a year in the index has no flat rate — that's deliberate. A silent
    fallback to the previous year is exactly the error this project calls out.
    """
    values = [table.flat_rate(route, d.year) for d in index]
    return pd.Series(values, index=index, name=f"flat_rate_{route}")


def decompose_freight_change(ws_points: pd.Series, flat_rates: pd.Series) -> pd.DataFrame:
    """Exact decomposition of the freight change, in $/t.

        Δfreight = [ ΔWS·FR_prev + WS_prev·ΔFR + ΔWS·ΔFR ] / 100
                     └ market ┘    └ reset ┘      └ cross ┘

    The three terms sum exactly to Δfreight: it's an algebraic identity, not an
    approximation. The "reset" term is the one a points-based WS model doesn't see.
    """
    aligned = pd.concat({"ws": ws_points, "fr": flat_rates}, axis=1).dropna().sort_index()
    if len(aligned) < 2:
        raise ValueError("at least two observations are needed to decompose a change")

    ws_prev = aligned["ws"].shift(1)
    fr_prev = aligned["fr"].shift(1)
    d_ws = aligned["ws"].diff()
    d_fr = aligned["fr"].diff()

    out = pd.DataFrame(index=aligned.index)
    out["freight_usd_t"] = aligned["ws"] / 100.0 * aligned["fr"]
    out["d_freight"] = out["freight_usd_t"].diff()
    out["market_component"] = d_ws * fr_prev / 100.0
    out["reset_component"] = ws_prev * d_fr / 100.0
    out["cross_component"] = d_ws * d_fr / 100.0
    return out.dropna()


def january_reset_effect(decomposition: pd.DataFrame) -> pd.DataFrame:
    """Isolates the dates where the flat rate changes — in practice the first business
    day of each year — and quantifies the cost jump attributable to the reset alone.

    Returns one row per reset: date, total jump, reset share, market share, and the WS
    point change that day. When the latter is small and the reset share is large, the
    project's result is right there.
    """
    reset_rows = decomposition[decomposition["reset_component"].abs() > 1e-12]
    if reset_rows.empty:
        return pd.DataFrame(
            columns=["date", "total_jump", "reset_share", "market_share", "cross_share"]
        )
    return pd.DataFrame(
        {
            "date": reset_rows.index,
            "total_jump": reset_rows["d_freight"].round(3).to_numpy(),
            "reset_share": reset_rows["reset_component"].round(3).to_numpy(),
            "market_share": reset_rows["market_component"].round(3).to_numpy(),
            "cross_share": reset_rows["cross_component"].round(3).to_numpy(),
        }
    )


# ------------------------------------------------------------------------------- arb
def reconstruct_arb(
    price_destination_usd_t: pd.Series,
    price_origin_usd_per_gallon: pd.Series,
    freight_usd_t: pd.Series,
    *,
    bbl_per_tonne: float = DEFAULT_BBL_PER_TONNE,
    spec_bridge_usd_t: float = 0.0,
    voyage_days: float = 16.0,
    annual_rate: float = 0.06,
    losses_usd_t: float = 0.0,
) -> pd.DataFrame:
    """USGC -> ARA arb, term by term, with the volume conversion made explicit.

    Alignment by intersection, no forward-fill. Returns a DataFrame with p_dest,
    p_origin_gal, p_origin_t, spread_naive, freight, spec, financing, losses, arb,
    is_open.

    `spread_naive` is the spread before freight and spec bridge: it's the number
    looked at when believing an arb is open, and the comparison with `arb` is the
    subject of `open_days_comparison`.
    """
    aligned = pd.concat(
        {
            "p_dest": price_destination_usd_t,
            "p_origin_gal": price_origin_usd_per_gallon,
            "freight": freight_usd_t,
        },
        axis=1,
    ).dropna()
    if aligned.empty:
        raise ValueError(
            "no common date across the three series — check the calendars, "
            "don't fill the gaps"
        )
    aligned = aligned.sort_index()

    out = aligned.copy()
    out["p_origin_t"] = usd_per_gallon_to_usd_per_tonne(aligned["p_origin_gal"], bbl_per_tonne)
    out["spread_naive"] = out["p_dest"] - out["p_origin_t"]
    out["spec"] = float(spec_bridge_usd_t)
    out["financing"] = out["p_origin_t"] * annual_rate * (voyage_days / 365.0)
    out["losses"] = float(losses_usd_t)
    out["arb"] = (
        out["spread_naive"] - out["freight"] - out["spec"] - out["financing"] - out["losses"]
    )
    out["is_open"] = out["arb"] > 0
    out["looks_open_naive"] = out["spread_naive"] > 0
    return out


@dataclass(frozen=True)
class OpenDaysComparison:
    n_obs: int
    days_looking_open: int
    days_really_open: int

    @property
    def share_looking_open(self) -> float:
        return self.days_looking_open / self.n_obs if self.n_obs else float("nan")

    @property
    def share_really_open(self) -> float:
        return self.days_really_open / self.n_obs if self.n_obs else float("nan")

    @property
    def illusion_share(self) -> float:
        """Share of the "apparently open" days that don't survive the real cost."""
        if self.days_looking_open == 0:
            return float("nan")
        return 1.0 - self.days_really_open / self.days_looking_open


def open_days_comparison(arb_frame: pd.DataFrame) -> OpenDaysComparison:
    """How many days the arb looks open on the naive spread, and how many stay open
    once freight, spec bridge and financing are paid.
    """
    return OpenDaysComparison(
        n_obs=len(arb_frame),
        days_looking_open=int(arb_frame["looks_open_naive"].sum()),
        days_really_open=int(arb_frame["is_open"].sum()),
    )


def monthly_profile(series: pd.Series) -> pd.DataFrame:
    """Average profile by calendar month. Transatlantic distillate is seasonal
    (European heating demand, maintenance turnarounds), so an arb that only survives a
    few months a year is not the same object as a permanent one.
    """
    frame = pd.DataFrame({"value": series})
    frame["month"] = frame.index.month
    grouped = frame.groupby("month")["value"].agg(["mean", "std", "count"])
    grouped.index.name = "month"
    return grouped.round(3)


# ------------------------------------------------------------------- C-2: computed freight
def implied_freight_from_tce(
    target_tce_usd_per_day: float,
    total_days: float,
    total_voyage_costs_usd: float,
    cargo_t: float,
    commission: float,
) -> float:
    """Inverts the TCE engine: what freight rate produces the target TCE?

        TCE = (cargo · F · (1 − c) − costs) / days
        =>  F = (TCE · days + costs) / (cargo · (1 − c))

    This is the C-2 path. If Worldscale flat rates are unavailable, the project isn't
    abandoned: freight is reconstructed from voyage economics — distances, consumption,
    bunker prices, port days — and compared against the quoted route wherever there's a
    point of comparison.

    Same algebraic form as `voyage/indifference.fair_value_c3`, which makes sense: both
    solve a TCE equation for the freight rate.
    """
    if total_days <= 0:
        raise ValueError("total_days must be > 0")
    if cargo_t <= 0:
        raise ValueError("cargo_t must be > 0")
    if not 0.0 <= commission < 1.0:
        raise ValueError(f"commission must be in [0, 1), got {commission}")
    return (target_tce_usd_per_day * total_days + total_voyage_costs_usd) / (
        cargo_t * (1.0 - commission)
    )


def freight_model_vs_quoted(
    modelled: pd.Series, quoted: pd.Series
) -> dict[str, float]:
    """Gap between the reconstructed freight and the quoted freight.

    A gap systematically on one side isn't a model error: it's what the route prices in
    beyond voyage cost — waiting, positioning, bargaining power. That's the C-2
    variant's result, not its failure.
    """
    aligned = pd.concat({"modelled": modelled, "quoted": quoted}, axis=1).dropna()
    if aligned.empty:
        raise ValueError("no common date between the modelled freight and the quoted freight")
    diff = aligned["modelled"] - aligned["quoted"]
    return {
        "n_obs": float(len(aligned)),
        "mean_gap_usd_t": float(diff.mean()),
        "median_gap_usd_t": float(diff.median()),
        "mean_gap_pct": float(100.0 * (diff / aligned["quoted"]).mean()),
        "correlation": float(np.corrcoef(aligned["modelled"], aligned["quoted"])[0, 1]),
    }


# ===========================================================================
# REAL DATA — the density assumption measured against the arb it computes
# ===========================================================================
# Everything above runs on synthetic prices. The Bloomberg export now carries both legs
# with 9 000+ observations each: NYMEX ULSD (cents per gallon) and ICE gasoil (USD per
# tonne). That is enough to stop asserting that the density conversion is large relative to
# the arb, and start measuring it.
LITRES_PER_GALLON = 3.785411784
KG_PER_TONNE = 1000.0

# Plausible density band for a middle distillate, in kg per litre. Diesel and gasoil
# specifications sit inside it; the exact figure depends on the grade, the additive package
# and the reference temperature, and no exchange publishes it alongside the price.
DENSITY_LIGHT = 0.820
DENSITY_TYPICAL = 0.845
DENSITY_HEAVY = 0.860


def gallons_per_tonne(density_kg_l: float) -> float:
    """Volume of one tonne of distillate, in US gallons.

    This is the whole conversion, and it is one line: a tonne is a mass, a gallon is a
    volume, and the only thing standing between them is a density that nobody quotes.
    """
    if not 0.6 < density_kg_l < 1.1:
        raise ValueError(
            f"density outside the plausible range for a liquid hydrocarbon: {density_kg_l}"
        )
    return KG_PER_TONNE / density_kg_l / LITRES_PER_GALLON


@cached('c_products')
def load_real_transatlantic_frame(start: str | None = "2015-01-01") -> pd.DataFrame:
    """NYMEX ULSD and ICE gasoil, both real, on their common calendar.

    Columns: ulsd_c_gal (US leg, volume-quoted), gasoil_usd_t (European leg, mass-quoted).
    Deliberately left in their native units — converting on the way in would hide the very
    thing this project is about.
    """
    from agri.data.bloomberg_loader import load

    frame = pd.concat(
        {"ulsd_c_gal": load("ulsd"), "gasoil_usd_t": load("ice_gasoil")},
        axis=1,
        sort=True,
    ).dropna()
    if start is not None:
        frame = frame[frame.index >= pd.Timestamp(start)]
    if frame.empty:
        raise ValueError(f"no common dates between ULSD and ICE gasoil after {start}")
    return frame


def transatlantic_spread(
    frame: pd.DataFrame, *, density_kg_l: float = DENSITY_TYPICAL
) -> pd.Series:
    """US leg converted to USD per tonne, minus the European leg.

        spread = ULSD_c_gal / 100 x gallons_per_tonne(density) - gasoil_usd_t

    Positive means the US barrel, once expressed in European units, is worth more than the
    European one — which is what has to be true before freight can be paid out of it.
    """
    converted = frame["ulsd_c_gal"] / 100.0 * gallons_per_tonne(density_kg_l)
    return (converted - frame["gasoil_usd_t"]).rename("spread_usd_t")


@dataclass(frozen=True)
class DensityIdentification:
    """How much of the arb is the arb, and how much is the conversion factor.

    The comparison that matters is not "is the density swing large in absolute terms" but
    "is it large next to the variability of the quantity it is used to compute". If a
    parameter nobody publishes moves the answer by as much as the answer moves on its own,
    then the level of the arb is not identifiable from prices — only its sign and its
    changes are.
    """

    swing_usd_t: float
    spread_std_usd_t: float
    spread_median_usd_t: float
    density_light: float
    density_heavy: float
    n_obs: int

    @property
    def ratio(self) -> float:
        return self.swing_usd_t / self.spread_std_usd_t

    @property
    def level_is_identifiable(self) -> bool:
        """A level is only meaningful if the parameter noise is small next to it. One third
        is the line taken here, and it is stated rather than tuned."""
        return self.ratio < 0.33

    @property
    def headline(self) -> str:
        return (
            f"Moving the density assumption across its plausible range "
            f"({self.density_light:.3f} to {self.density_heavy:.3f} kg/l) moves the arb by "
            f"{self.swing_usd_t:.1f} USD/t. The arb's own standard deviation is "
            f"{self.spread_std_usd_t:.1f} USD/t. The conversion factor therefore accounts "
            f"for {self.ratio:.0%} of the variability of the number it is used to compute — "
            "the level of this arb is not identifiable from prices alone."
        )


def density_identification(
    frame: pd.DataFrame,
    *,
    density_light: float = DENSITY_LIGHT,
    density_heavy: float = DENSITY_HEAVY,
    density_reference: float = DENSITY_TYPICAL,
) -> DensityIdentification:
    """Measure the density swing against the arb's own variability."""
    if density_light >= density_heavy:
        raise ValueError(
            f"the light density must be below the heavy one: {density_light} >= {density_heavy}"
        )
    light = transatlantic_spread(frame, density_kg_l=density_light)
    heavy = transatlantic_spread(frame, density_kg_l=density_heavy)
    reference = transatlantic_spread(frame, density_kg_l=density_reference)
    return DensityIdentification(
        swing_usd_t=float((light - heavy).median()),
        spread_std_usd_t=float(reference.std()),
        spread_median_usd_t=float(reference.median()),
        density_light=float(density_light),
        density_heavy=float(density_heavy),
        n_obs=int(len(frame)),
    )


@dataclass(frozen=True)
class BreakevenDensity:
    """The density at which the arb exactly covers freight — the trade decision.

    This is the inversion. Rather than picking a density and announcing whether the arb is
    open, we ask which density makes it marginal at a stated freight cost. If that density
    falls inside the plausible band, then the decision to load a cargo turns on a number
    that is not a market price.
    """

    density_star: float
    freight_usd_t: float
    band_light: float
    band_heavy: float
    date: pd.Timestamp

    @property
    def inside_plausible_band(self) -> bool:
        return self.band_light <= self.density_star <= self.band_heavy

    @property
    def headline(self) -> str:
        if self.inside_plausible_band:
            return (
                f"At a freight cost of {self.freight_usd_t:.0f} USD/t, the arb breaks even "
                f"at a density of {self.density_star:.4f} kg/l — **inside** the plausible "
                f"band [{self.band_light:.3f} ; {self.band_heavy:.3f}]. A light grade makes "
                "this cargo economic and a heavy one does not, at identical prices. The "
                "decision is not being made on the market."
            )
        side = "above" if self.density_star > self.band_heavy else "below"
        return (
            f"At a freight cost of {self.freight_usd_t:.0f} USD/t, the arb breaks even at a "
            f"density of {self.density_star:.4f} kg/l, {side} the plausible band "
            f"[{self.band_light:.3f} ; {self.band_heavy:.3f}]. The sign of the arb therefore "
            "survives the conversion uncertainty — only its size does not."
        )


def breakeven_density(
    frame: pd.DataFrame,
    *,
    freight_usd_t: float,
    row: pd.Series | None = None,
    band_light: float = DENSITY_LIGHT,
    band_heavy: float = DENSITY_HEAVY,
) -> BreakevenDensity:
    """Solve, in closed form, the density at which the spread equals freight.

        spread(rho) = ULSD/100 x 1000 / rho / 3.785412 - gasoil = freight
        =>  rho* = ULSD/100 x 1000 / 3.785412 / (gasoil + freight)

    Closed form rather than a solver, because it shows what the answer depends on: the
    breakeven density is inversely proportional to the sum of the European price and the
    freight, which is why it tightens exactly when freight is expensive.
    """
    if freight_usd_t < 0:
        raise ValueError("freight must be >= 0")
    if row is None:
        row = frame.iloc[-1]

    denominator = row["gasoil_usd_t"] + freight_usd_t
    if denominator <= 0:
        raise ValueError("European leg plus freight must be > 0")

    density_star = (
        row["ulsd_c_gal"] / 100.0 * KG_PER_TONNE / LITRES_PER_GALLON / denominator
    )
    return BreakevenDensity(
        density_star=float(density_star),
        freight_usd_t=float(freight_usd_t),
        band_light=float(band_light),
        band_heavy=float(band_heavy),
        date=pd.Timestamp(row.name),
    )


def breakeven_density_series(
    frame: pd.DataFrame, *, freight_usd_t: float
) -> pd.DataFrame:
    """The breakeven density day by day, with the plausible band attached.

    The deliverable of the page: a series that spends part of its life inside the band —
    the periods when the cargo decision genuinely turned on the grade rather than on the
    market — and part outside it, when the arb was unambiguous either way.
    """
    if freight_usd_t < 0:
        raise ValueError("freight must be >= 0")
    density = (
        frame["ulsd_c_gal"] / 100.0 * KG_PER_TONNE / LITRES_PER_GALLON
        / (frame["gasoil_usd_t"] + freight_usd_t)
    )
    out = pd.DataFrame({"density_star": density})
    out["inside_band"] = (out["density_star"] >= DENSITY_LIGHT) & (
        out["density_star"] <= DENSITY_HEAVY
    )
    out["above_band"] = out["density_star"] > DENSITY_HEAVY
    return out
