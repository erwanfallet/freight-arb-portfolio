"""T3-1 — Two subsidies that contradict each other, and the plant caught between them.

THE STORY
---------
Roughly three quarters of US renewable diesel capacity was built on or near the Gulf and
California coasts. This is not a geographic accident: these plants were **sited to run on
imported feedstock** — UCO from Asia, tallow — landed at the dock. The siting choice encodes
a bet on the origin of the feedstock, and it was made before the tax rule existed.

Then two administrations write two subsidies that do not pursue the same goal. Congress
writes 45Z to reward **North American** feedstock, and excludes anything that is not from
the credit. California, meanwhile, keeps paying for **low carbon intensity** through its
LCFS, regardless of origin — and imported UCO happens to be less carbon-intensive than
domestic soyoil. One policy penalises exactly what the other rewards, on the same gallon, in
the same plant.

Hence the disagreement:

    Camp A — coastal plants: the LCFS premium is enough, imports hold.
    Camp B — the soy complex: it is not enough, soyoil takes the share.

WHAT THE PAGE SHOWS — AND WHY BOTH CAMPS ARE ARGUING ABOUT THE WRONG VARIABLE
----------------------------------------------------------------------------------
Both camps argue about the **LCFS credit price**. But this price cannot settle it: across
the entire range the programme has realised since inception, it moves the answer by only a
few cents per pound. What settles it is the **UCO-soyoil price spread** — a feedstock price
set by collection in Asia and by freight, not a regulatory decision in Sacramento.

The deliverable is therefore not a forecast but a **discount**: how many cents under soyoil
imported UCO has to sell for, purely to offset a tax credit it has no right to claim. A
feedstock buyer confirms or denies this number in ten seconds against their own book.

STANCE — TO BE HELD ABSOLUTELY
----------------------------------
No side is taken on which way it falls. The tipping point is quantified; the insider has
the internal data. A candidate who predicts gets corrected; a candidate who gives a
falsifiable threshold gets a reply.

STATE OF PLAY (sourced)
------------------------
- 2026-27 RVOs finalised by the EPA in March 2026: roughly a 60% increase in
  biodiesel + renewable diesel production versus 2025, the programme's highest level yet
  (Fastmarkets, 16 June 2026).
- ILUC removed: US soyoil carries a CI of 27 and a 45Z credit of ~49 c/gal, nearly on par
  with tallow and UCO. Soyoil use projected from 14.55 to ~17.8 bn lb (WASDE, June 2026).
- farmdoc daily (25 June 2026): three-layer uncertainty — domestic production versus
  imported fuel, domestic versus imported feedstock under the 45Z penalty, mix
  distribution within each channel. Imported feedstock "could pull back sharply, or prove
  more resilient than expected if the LCFS premium stays wide enough". Roughly three
  quarters of domestic RD capacity sits on or near the Gulf and California coasts, built
  to run on imported feedstock.

THE UNIT TRAP
-------------
Three units stack into a single margin, and two of them are not price units:
    fuel sells by the **gallon**;
    feedstock is bought by the **pound** (or tonne), via a yield of ~7.6 lb/gal;
    the LCFS credit is quoted in **USD per tonne of CO2e**, and only enters the margin
    after passing through a carbon intensity in gCO2e/MJ and an energy content of
    134.47 MJ/gal.
The factor 134.47e-6 is not a conversion detail: it is what sets the slope of the LCFS
threshold, hence the answer.

MODEL
-----
    gate_value(f) = P_ulsd + RIN_D4 x rin_per_gal
                  + LCFS x (CI_std - CI_f) x EER x 134.47e-6
                  + credit_45Z(f)

    credit_45Z(f) = 1.00 x max(0, (50 - CI_f)/50)   if f is North-American eligible
                  = 0                                otherwise

    feedstock_breakeven(f) = (gate_value(f) - opex - roi) / yield_lb_gal

ASSUMPTIONS
-----------
L-H1  `yield_lb_gal` = 7.6 lb of feedstock per gallon of RD. Varies by pathway and
      plant; swept in sensitivity.
L-H2  EER = 1.0 for renewable diesel. The LCFS applies EER > 1 to electricity and
      hydrogen, not to RD.
L-H3  `CI_std` is the year's CARB schedule, which **decreases every year by
      construction**. Freezing it would corrupt the whole history — it is a dated
      parameter.
L-H4  The 45Z credit follows the linear formula (50 - CI)/50 capped at 1 $/gal.
      Calibration check: CI = 27 gives 0.46 $/gal against ~0.49 published. The gap comes
      down to the exact CI definition used and is not resolved here: it is shown.
L-H5  The 45Z credit is **not a constant** — it depends on a CI that depends on a
      methodology still being finalised. Treated as a slider with a range, never as a
      single number.
L-H6  Small refinery exemptions (SRE) retroactively reallocate volumes. Any conclusion
      about RVOs carries an SRE warning.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from agri.data.snapshot import cached

from agri.core.breakeven import Breakeven, NoBreakevenInRange, solve_breakeven

# --- physical and regulatory constants ------------------------------
MJ_PER_GALLON_ULSD = 134.47          # energy content of the reference diesel
GRAMS_PER_TONNE = 1_000_000.0
LCFS_CONVERSION = MJ_PER_GALLON_ULSD / GRAMS_PER_TONNE   # 134.47e-6

CREDIT_45Z_MAX_USD_GAL = 1.00        # formula cap
CREDIT_45Z_CI_REFERENCE = 50.0       # the CI at which the credit hits zero

OIL_LB_PER_BUSHEL = 11.0             # pounds of oil per bushel of soybean crushed

# --- default values (L-H1 to L-H4) ------------------------------
DEFAULT_YIELD_LB_GAL = 7.6
DEFAULT_EER = 1.0
DEFAULT_RIN_PER_GAL_RD = 1.7         # biodiesel generates 1.5
DEFAULT_CI_STD = 95.0                # CARB schedule, needs a date (L-H3)
DEFAULT_CI_SOY = 27.0
DEFAULT_CI_UCO = 15.0
DEFAULT_OPEX_USD_GAL = 0.55
DEFAULT_ROI_USD_GAL = 0.25


class FeedstockError(ValueError):
    """Mis-specified model."""


@dataclass(frozen=True)
class Feedstock:
    """A pathway: its carbon intensity and its 45Z eligibility."""

    name: str
    carbon_intensity: float          # gCO2e/MJ
    north_american: bool             # 45Z-eligible

    def credit_45z_usd_gal(self) -> float:
        """45Z credit (L-H4). Zero if not North American, whatever the intensity."""
        if not self.north_american:
            return 0.0
        return CREDIT_45Z_MAX_USD_GAL * max(
            0.0, (CREDIT_45Z_CI_REFERENCE - self.carbon_intensity) / CREDIT_45Z_CI_REFERENCE
        )


SOYOIL_DOMESTIC = Feedstock("domestic soyoil", DEFAULT_CI_SOY, north_american=True)
UCO_IMPORTED = Feedstock("imported UCO", DEFAULT_CI_UCO, north_american=False)
TALLOW_DOMESTIC = Feedstock("domestic tallow", 18.0, north_american=True)
DCO_DOMESTIC = Feedstock("domestic DCO", 20.0, north_american=True)


def calibration_gap_45z() -> dict[str, float]:
    """L-H4 calibration check, shown in the diagnostics panel.

    The formula gives 0.46 $/gal on a CI of 27; the published value is around
    0.49 $/gal. The 3 c/gal gap is not resolved — it is shown, because resolving
    it with an invented adjustment factor would turn the result into an artefact.
    """
    modelled = SOYOIL_DOMESTIC.credit_45z_usd_gal()
    published = 0.49
    return {
        "modelled_usd_gal": modelled,
        "published_usd_gal": published,
        "gap_usd_gal": published - modelled,
        "gap_pct": (published - modelled) / published,
    }


# ===========================================================================
# Gate value
# ===========================================================================
def lcfs_value_usd_gal(
    lcfs_usd_t: float | pd.Series,
    carbon_intensity: float,
    *,
    ci_std: float = DEFAULT_CI_STD,
    eer: float = DEFAULT_EER,
) -> float | pd.Series:
    """The LCFS leg, in USD/gallon.

    This is where the unit trap closes: a credit quoted per tonne of CO2e becomes
    cents per gallon via a carbon intensity and an energy content.
    """
    return lcfs_usd_t * (ci_std - carbon_intensity) * eer * LCFS_CONVERSION


@dataclass(frozen=True)
class GateValue:
    """Gate value, decomposed. Each field is a bar in the S1 chart."""

    feedstock: str
    diesel: float
    rin: float
    lcfs: float
    credit_45z: float

    @property
    def total_usd_gal(self) -> float:
        return self.diesel + self.rin + self.lcfs + self.credit_45z

    @property
    def stack(self) -> dict[str, float]:
        return {
            "diesel": self.diesel,
            "RIN D4": self.rin,
            "LCFS": self.lcfs,
            "45Z": self.credit_45z,
        }


def gate_value(
    feedstock: Feedstock,
    *,
    ulsd_usd_gal: float,
    rin_d4_usd: float,
    lcfs_usd_t: float,
    ci_std: float = DEFAULT_CI_STD,
    eer: float = DEFAULT_EER,
    rin_per_gal: float = DEFAULT_RIN_PER_GAL_RD,
) -> GateValue:
    """Value of a gallon of renewable diesel at the gate, by pathway."""
    return GateValue(
        feedstock=feedstock.name,
        diesel=ulsd_usd_gal,
        rin=rin_d4_usd * rin_per_gal,
        lcfs=float(
            lcfs_value_usd_gal(
                lcfs_usd_t, feedstock.carbon_intensity, ci_std=ci_std, eer=eer
            )
        ),
        credit_45z=feedstock.credit_45z_usd_gal(),
    )


def feedstock_breakeven_usd_lb(
    feedstock: Feedstock,
    *,
    ulsd_usd_gal: float,
    rin_d4_usd: float,
    lcfs_usd_t: float,
    opex_usd_gal: float = DEFAULT_OPEX_USD_GAL,
    roi_usd_gal: float = DEFAULT_ROI_USD_GAL,
    yield_lb_gal: float = DEFAULT_YIELD_LB_GAL,
    ci_std: float = DEFAULT_CI_STD,
    eer: float = DEFAULT_EER,
    rin_per_gal: float = DEFAULT_RIN_PER_GAL_RD,
) -> float:
    """Maximum feedstock cost, in USD/lb — what a plant can afford to pay."""
    if yield_lb_gal <= 0:
        raise FeedstockError(f"yield_lb_gal must be > 0, got {yield_lb_gal}")
    value = gate_value(
        feedstock,
        ulsd_usd_gal=ulsd_usd_gal,
        rin_d4_usd=rin_d4_usd,
        lcfs_usd_t=lcfs_usd_t,
        ci_std=ci_std,
        eer=eer,
        rin_per_gal=rin_per_gal,
    )
    return (value.total_usd_gal - opex_usd_gal - roi_usd_gal) / yield_lb_gal


# ===========================================================================
# THE TIPPING POINT — the deliverable
# ===========================================================================
@dataclass(frozen=True)
class LcfsThreshold:
    """The LCFS threshold, its distance to the market, and the email sentence."""

    lcfs_star_usd_t: float
    lcfs_current_usd_t: float
    distance_sigmas: float | None
    ci_gap: float
    price_gap_usd_lb: float
    imports_win_above: bool

    @property
    def headline(self) -> str:
        side = "stays ahead of" if self.imports_win_above else "falls back behind"
        distance = (
            f", {self.distance_sigmas:+.2f} standard deviations of the history"
            if self.distance_sigmas is not None
            else ""
        )
        return (
            f"Beyond {self.lcfs_star_usd_t:.0f} $/t CO2e on the LCFS credit, "
            f"imported UCO without 45Z {side} domestic soyoil with 45Z. The LCFS "
            f"quotes {self.lcfs_current_usd_t:.0f} $/t today{distance}: the "
            "\"soy takes the share\" thesis holds within "
            f"{abs(self.lcfs_star_usd_t - self.lcfs_current_usd_t):.0f} $ either way."
        )


def lcfs_breakeven(
    *,
    domestic: Feedstock = SOYOIL_DOMESTIC,
    imported: Feedstock = UCO_IMPORTED,
    price_domestic_usd_lb: float,
    price_imported_usd_lb: float,
    lcfs_current_usd_t: float,
    yield_lb_gal: float = DEFAULT_YIELD_LB_GAL,
    eer: float = DEFAULT_EER,
    lcfs_history: pd.Series | None = None,
) -> LcfsThreshold:
    """Solves `LCFS*` in closed form.

        LCFS* = [ credit_45Z(dom) + (P_imp - P_dom) x yield ]
                / [ (CI_dom - CI_imp) x EER x 134.47e-6 ]

    Derivation: equalise the two pathways' net advantage, `breakeven(f) - P_f`.
    The diesel, RIN, opex and ROI terms are **identical** for both and cancel out
    — which is why the threshold depends on neither the diesel price, nor the
    RIN, nor the plant's cost structure. This is what makes it robust, and it is
    the argument to convey: the threshold only moves through the CI differential
    and the feedstock price differential.

    GATE FALLBACK: if UCO prices are unavailable, setting
    `price_imported_usd_lb = price_domestic_usd_lb` gives the threshold at price
    parity, i.e. the implicit value gap UCO would need to carry. Same
    deliverable, never needing the price.
    """
    ci_gap = domestic.carbon_intensity - imported.carbon_intensity
    if ci_gap <= 0:
        raise FeedstockError(
            f"the CI differential must be > 0 for the LCFS to be able to offset "
            f"45Z (CI_dom = {domestic.carbon_intensity}, CI_imp = "
            f"{imported.carbon_intensity}). If the imported pathway is more "
            "carbon-intensive than the domestic one, there is no threshold: it "
            "loses on both counts."
        )

    price_gap = price_imported_usd_lb - price_domestic_usd_lb
    numerator = domestic.credit_45z_usd_gal() + price_gap * yield_lb_gal
    denominator = ci_gap * eer * LCFS_CONVERSION
    lcfs_star = numerator / denominator

    sigma = None
    if lcfs_history is not None:
        history = pd.Series(lcfs_history).dropna().astype(float)
        if len(history) >= 2:
            std = float(history.std())
            sigma = (lcfs_star - lcfs_current_usd_t) / std if std > 0 else float("inf")

    return LcfsThreshold(
        lcfs_star_usd_t=lcfs_star,
        lcfs_current_usd_t=lcfs_current_usd_t,
        distance_sigmas=sigma,
        ci_gap=ci_gap,
        price_gap_usd_lb=price_gap,
        imports_win_above=True,
    )


def lcfs_breakeven_numeric(
    *,
    domestic: Feedstock = SOYOIL_DOMESTIC,
    imported: Feedstock = UCO_IMPORTED,
    price_domestic_usd_lb: float,
    price_imported_usd_lb: float,
    ulsd_usd_gal: float,
    rin_d4_usd: float,
    lcfs_current_usd_t: float,
    lo: float = 0.0,
    hi: float = 800.0,
    **kwargs,
) -> Breakeven:
    """The same threshold, solved numerically — a cross-check on the closed form.

    If the two diverge, the closed form has an algebra error. This is a test,
    not an alternative: the closed form is the one shown, because it shows
    *what* the threshold depends on.
    """

    def advantage(lcfs: float) -> float:
        imported_edge = (
            feedstock_breakeven_usd_lb(
                imported,
                ulsd_usd_gal=ulsd_usd_gal,
                rin_d4_usd=rin_d4_usd,
                lcfs_usd_t=lcfs,
                **kwargs,
            )
            - price_imported_usd_lb
        )
        domestic_edge = (
            feedstock_breakeven_usd_lb(
                domestic,
                ulsd_usd_gal=ulsd_usd_gal,
                rin_d4_usd=rin_d4_usd,
                lcfs_usd_t=lcfs,
                **kwargs,
            )
            - price_domestic_usd_lb
        )
        return imported_edge - domestic_edge

    return solve_breakeven(
        advantage,
        lo,
        hi,
        theta_current=lcfs_current_usd_t,
        theta_label="LCFS",
        margin_label="UCO advantage",
    )


# ===========================================================================
# THE INVERSION — the page's real deliverable
# ===========================================================================
# The LCFS* threshold above is algebraically correct but unreadable for a desk: nobody has
# intuition for "285 $/t CO2e". A feedstock buyer, on the other hand, quotes DISCOUNTS in
# cents per pound all day long. The question is therefore inverted: instead of asking which
# LCFS price would save the imported pathway, we ask **what discount the imported pathway
# has to hold** against the domestic one, at a given LCFS price. Same algebra, but the
# number that comes out is one the counterparty can confirm or deny in ten seconds against
# their own book.
CENTS_PER_USD = 100.0

# Realised bounds of the California LCFS programme, in $/t CO2e. These are NOT export
# data: the LCFS credit price is published free by CARB (monthly credit transfer report)
# and was not downloaded. The two bounds only serve to bracket a result that is itself
# computed as a function of the LCFS price — a reader with the series substitutes their
# own values without changing the reasoning at all.
LCFS_PROGRAM_LOW_USD_T = 50.0    # 2023-2024 trough
LCFS_PROGRAM_HIGH_USD_T = 200.0  # historical high, 2019-2020


@dataclass(frozen=True)
class ImportPenalty:
    """What 45Z exclusion costs an importer, expressed as a feedstock price.

    `discount_required_usd_lb` is the deliverable: the discount imported UCO has
    to hold against domestic soyoil for a plant to be indifferent between the
    two. Positive = domestic keeps the advantage and the imported pathway must
    pay the difference as a discount.
    """

    lcfs_usd_t: float
    credit_45z_usd_gal: float
    lcfs_offset_usd_gal: float
    residual_usd_gal: float
    discount_required_usd_lb: float

    @property
    def discount_required_c_lb(self) -> float:
        return self.discount_required_usd_lb * CENTS_PER_USD

    @property
    def imports_win_outright(self) -> bool:
        """Beyond the neutral LCFS, the CI advantage exceeds 45Z: the imported
        pathway can even command a PREMIUM and stay ahead."""
        return self.residual_usd_gal <= 0

    @property
    def headline(self) -> str:
        if self.imports_win_outright:
            return (
                f"At {self.lcfs_usd_t:.0f} $/t CO2e, imported UCO's carbon "
                f"advantage ({self.lcfs_offset_usd_gal:.2f} $/gal) exceeds the "
                f"45Z domestic soyoil earns ({self.credit_45z_usd_gal:.2f} $/gal): "
                "the imported pathway holds even at price parity, and up to "
                f"{-self.discount_required_c_lb:.2f} c/lb of premium."
            )
        return (
            f"At {self.lcfs_usd_t:.0f} $/t CO2e, imported UCO must sell "
            f"{self.discount_required_c_lb:.2f} c/lb below domestic soyoil for a "
            "plant to be indifferent — purely to offset a tax credit it has no "
            "right to claim."
        )


def import_penalty(
    lcfs_usd_t: float,
    *,
    domestic: Feedstock = SOYOIL_DOMESTIC,
    imported: Feedstock = UCO_IMPORTED,
    yield_lb_gal: float = DEFAULT_YIELD_LB_GAL,
    eer: float = DEFAULT_EER,
) -> ImportPenalty:
    """The discount the imported pathway must hold, at a given LCFS price.

        discount = [ credit_45Z(dom) - LCFS x (CI_dom - CI_imp) x EER x 134.47e-6 ] / yield

    The diesel, RIN, opex and ROI terms have vanished from this expression: both
    pathways produce the SAME gallon of renewable diesel in the SAME plant, so
    they cancel instead of adding up. This is what makes the number hard to
    dispute — it cannot be disputed by challenging a diesel forecast or an opex
    assumption, since neither appears in it.
    """
    if yield_lb_gal <= 0:
        raise FeedstockError(f"yield_lb_gal must be > 0, got {yield_lb_gal}")
    ci_gap = domestic.carbon_intensity - imported.carbon_intensity
    if ci_gap <= 0:
        raise FeedstockError(
            f"the CI differential must be > 0 (CI_dom = {domestic.carbon_intensity}, "
            f"CI_imp = {imported.carbon_intensity}): if the imported pathway is "
            "more carbon-intensive, it loses on both counts and the notion of a "
            "compensating discount has no meaning."
        )

    credit = domestic.credit_45z_usd_gal()
    offset = float(lcfs_usd_t * ci_gap * eer * LCFS_CONVERSION)
    residual = credit - offset
    return ImportPenalty(
        lcfs_usd_t=float(lcfs_usd_t),
        credit_45z_usd_gal=credit,
        lcfs_offset_usd_gal=offset,
        residual_usd_gal=residual,
        discount_required_usd_lb=residual / yield_lb_gal,
    )


def lcfs_neutral_price(
    *,
    domestic: Feedstock = SOYOIL_DOMESTIC,
    imported: Feedstock = UCO_IMPORTED,
    eer: float = DEFAULT_EER,
) -> float:
    """The LCFS price that exactly offsets 45Z, at feedstock price parity.

    This is `lcfs_breakeven` with `price_gap = 0`, isolated because it serves as
    an absolute reference point: above it, California over-compensates what
    Congress withdrew; below it, it merely cushions it.
    """
    ci_gap = domestic.carbon_intensity - imported.carbon_intensity
    if ci_gap <= 0:
        raise FeedstockError("the CI differential must be > 0")
    return domestic.credit_45z_usd_gal() / (ci_gap * eer * LCFS_CONVERSION)


@dataclass(frozen=True)
class PenaltyBounds:
    """The central result: the required discount is BOUNDED across the LCFS range.

    The public debate poses the question as "is the LCFS premium enough to save
    imports?", i.e. as a question about the credit price. If the total amplitude
    the LCFS can print onto the answer is narrow next to the moves in the
    feedstock price, the question is mis-posed at the source.
    """

    lcfs_low_usd_t: float
    lcfs_high_usd_t: float
    discount_at_low_usd_lb: float
    discount_at_high_usd_lb: float
    lcfs_neutral_usd_t: float

    @property
    def span_usd_lb(self) -> float:
        """The total amplitude the LCFS can print onto the required discount."""
        return self.discount_at_low_usd_lb - self.discount_at_high_usd_lb

    @property
    def span_c_lb(self) -> float:
        return self.span_usd_lb * CENTS_PER_USD

    @property
    def reaches_neutral(self) -> bool:
        """Has the LCFS ever quoted high enough to offset 45Z at price parity?"""
        return self.lcfs_high_usd_t >= self.lcfs_neutral_usd_t

    @property
    def headline(self) -> str:
        verdict = (
            "it got there"
            if self.reaches_neutral
            else (
                f"it never got there — its historical high "
                f"({self.lcfs_high_usd_t:.0f} $/t) remains "
                f"{self.lcfs_neutral_usd_t - self.lcfs_high_usd_t:.0f} $/t short"
            )
        )
        return (
            f"Offsetting 45Z at price parity would take {self.lcfs_neutral_usd_t:.0f} "
            f"$/t CO2e on the LCFS credit; across the programme's entire history, "
            f"{verdict}. Between its trough and its peak, the LCFS moves the "
            f"required discount by only {self.span_c_lb:.2f} c/lb — from "
            f"{self.discount_at_low_usd_lb * CENTS_PER_USD:.2f} to "
            f"{self.discount_at_high_usd_lb * CENTS_PER_USD:.2f} c/lb."
        )


def penalty_bounds(
    *,
    lcfs_low_usd_t: float = LCFS_PROGRAM_LOW_USD_T,
    lcfs_high_usd_t: float = LCFS_PROGRAM_HIGH_USD_T,
    **kwargs,
) -> PenaltyBounds:
    """Brackets the required discount across the LCFS credit price's realised range."""
    if lcfs_high_usd_t <= lcfs_low_usd_t:
        raise FeedstockError(
            f"the high bound must exceed the low bound: {lcfs_high_usd_t} <= {lcfs_low_usd_t}"
        )
    low = import_penalty(lcfs_low_usd_t, **kwargs)
    high = import_penalty(lcfs_high_usd_t, **kwargs)
    return PenaltyBounds(
        lcfs_low_usd_t=float(lcfs_low_usd_t),
        lcfs_high_usd_t=float(lcfs_high_usd_t),
        discount_at_low_usd_lb=low.discount_required_usd_lb,
        discount_at_high_usd_lb=high.discount_required_usd_lb,
        lcfs_neutral_usd_t=lcfs_neutral_price(
            domestic=kwargs.get("domestic", SOYOIL_DOMESTIC),
            imported=kwargs.get("imported", UCO_IMPORTED),
            eer=kwargs.get("eer", DEFAULT_EER),
        ),
    )


# ===========================================================================
# The required discount against the real soyoil price
# ===========================================================================
@dataclass(frozen=True)
class DiscountBurden:
    """The same discount in cents, relative to the feedstock price it weighs on.

    A 4.5 c/lb discount does not mean the same thing whether soyoil quotes at
    90 c/lb or 25: in one case it is 5% of the price, in the other 18%. 45Z is
    written in dollars per gallon, so its relative weight is **countercyclical**
    to the vegetable oil price — it bites hardest when oil is cheap, i.e. when
    crushing margins are already thin.
    """

    frame: pd.DataFrame
    lcfs_usd_t: float
    discount_required_usd_lb: float

    @property
    def burden_min(self) -> float:
        return float(self.frame["burden_share"].min())

    @property
    def burden_max(self) -> float:
        return float(self.frame["burden_share"].max())

    @property
    def burden_last(self) -> float:
        return float(self.frame["burden_share"].iloc[-1])

    @property
    def headline(self) -> str:
        return (
            f"The same {self.discount_required_usd_lb * CENTS_PER_USD:.2f} c/lb "
            f"discount weighs {self.burden_min:.1%} of the soyoil price at its "
            f"high and {self.burden_max:.1%} at its low over the period — a "
            f"factor of {self.burden_max / self.burden_min:.1f}. It weighs "
            f"{self.burden_last:.1%} at the last print."
        )


def discount_burden(
    domestic_price_usd_lb: pd.Series,
    *,
    lcfs_usd_t: float,
    **kwargs,
) -> DiscountBurden:
    """The relative weight of the required discount, on a real domestic price series.

    UNIT WARNING: CBOT soyoil quotes in **cents per pound**; this function
    expects **USD per pound**. The factor of 100 between the two is exactly the
    same order as the result being sought, so forgetting it does not produce a
    visible error — it produces a plausible, wrong number.
    """
    prices = pd.Series(domestic_price_usd_lb).dropna().astype(float)
    if prices.empty:
        raise FeedstockError("empty domestic price series")
    if (prices <= 0).any():
        raise FeedstockError("zero or negative domestic price in the series")
    if prices.median() > 5.0:
        raise FeedstockError(
            f"median price of {prices.median():.1f}: the series looks like it is "
            "in cents per pound while USD per pound is expected (divide by 100)."
        )

    penalty = import_penalty(lcfs_usd_t, **kwargs)
    frame = pd.DataFrame(
        {
            "domestic_price_usd_lb": prices,
            "discount_required_usd_lb": penalty.discount_required_usd_lb,
            "burden_share": penalty.discount_required_usd_lb / prices,
        }
    )
    return DiscountBurden(
        frame=frame,
        lcfs_usd_t=float(lcfs_usd_t),
        discount_required_usd_lb=penalty.discount_required_usd_lb,
    )


@dataclass(frozen=True)
class StructuralExit:
    """The soyoil price below which imports become impossible, not merely disadvantaged.

    UCO has a collection cost and freight: there is a floor price below which
    there is simply no export supply. If the discount 45Z requires pushes UCO
    below that floor, the pathway does not contract — it stops, regardless of the LCFS.
    """

    soyoil_critical_usd_lb: float
    uco_floor_usd_lb: float
    discount_required_usd_lb: float
    lcfs_usd_t: float
    share_below: float | None = None
    n_obs: int | None = None

    @property
    def headline(self) -> str:
        base = (
            f"With a UCO collection floor of {self.uco_floor_usd_lb * CENTS_PER_USD:.0f} "
            f"c/lb delivered USGC, imports stop being fundable as soon as soyoil "
            f"falls below {self.soyoil_critical_usd_lb * CENTS_PER_USD:.1f} c/lb."
        )
        if self.share_below is None:
            return base
        return (
            f"{base} Soyoil has quoted below this threshold {self.share_below:.0%} "
            f"of the time over the sample ({self.n_obs:,} sessions)."
        )


def structural_exit(
    domestic_price_usd_lb: pd.Series | None = None,
    *,
    uco_floor_usd_lb: float,
    lcfs_usd_t: float,
    **kwargs,
) -> StructuralExit:
    """The critical soyoil price, and how often the market has crossed it.

        soyoil* = UCO_floor + required_discount(LCFS)

    The collection floor is the only parameter the page cannot observe: it is
    deliberately the one asked of the counterparty. The rest of the calculation is closed.
    """
    if uco_floor_usd_lb <= 0:
        raise FeedstockError("the UCO collection floor must be > 0")

    penalty = import_penalty(lcfs_usd_t, **kwargs)
    critical = uco_floor_usd_lb + penalty.discount_required_usd_lb

    share = n_obs = None
    if domestic_price_usd_lb is not None:
        prices = pd.Series(domestic_price_usd_lb).dropna().astype(float)
        if not prices.empty:
            share = float((prices < critical).mean())
            n_obs = int(len(prices))

    return StructuralExit(
        soyoil_critical_usd_lb=critical,
        uco_floor_usd_lb=float(uco_floor_usd_lb),
        discount_required_usd_lb=penalty.discount_required_usd_lb,
        lcfs_usd_t=float(lcfs_usd_t),
        share_below=share,
        n_obs=n_obs,
    )


@cached('t3_1_soyoil', from_frame=lambda f: f.iloc[:, 0].rename("soyoil_usd_lb"))
def load_soyoil_usd_lb(start: str | None = None) -> pd.Series:
    """CBOT soyoil from the real export, converted from cents/lb to USD/lb.

    The conversion is isolated in a function rather than repeated across pages:
    this is the module's unit trap, and a factor of 100 scattered through the
    code always ends up applied twice somewhere, or zero times.
    """
    from agri.data.bloomberg_loader import load

    series = load("cbot_soyoil") / CENTS_PER_USD
    if start is not None:
        series = series[series.index >= pd.Timestamp(start)]
    return series.rename("soyoil_usd_lb")


def winner_grid(
    *,
    ci_imported_values: np.ndarray | None = None,
    lcfs_values: np.ndarray | None = None,
    domestic: Feedstock = SOYOIL_DOMESTIC,
    price_domestic_usd_lb: float = 0.52,
    price_imported_usd_lb: float = 0.46,
    yield_lb_gal: float = DEFAULT_YIELD_LB_GAL,
    eer: float = DEFAULT_EER,
) -> pd.DataFrame:
    """S4 — heatmap `imported CI × LCFS price -> winning pathway`.

    The chart that explains the disagreement with no line of text: two zones
    separated by a boundary, with the current market point plotted on it.
    """
    ci_imported_values = (
        np.arange(10.0, 31.0, 1.0) if ci_imported_values is None else np.asarray(ci_imported_values)
    )
    lcfs_values = (
        np.arange(0.0, 401.0, 10.0) if lcfs_values is None else np.asarray(lcfs_values)
    )

    rows = []
    for ci in ci_imported_values:
        imported = Feedstock("imported UCO", float(ci), north_american=False)
        for lcfs in lcfs_values:
            imported_edge = lcfs_value_usd_gal(lcfs, ci, eer=eer) / yield_lb_gal - price_imported_usd_lb
            domestic_edge = (
                lcfs_value_usd_gal(lcfs, domestic.carbon_intensity, eer=eer)
                + domestic.credit_45z_usd_gal()
            ) / yield_lb_gal - price_domestic_usd_lb
            rows.append(
                {
                    "ci_imported": float(ci),
                    "lcfs_usd_t": float(lcfs),
                    "advantage_usd_lb": imported_edge - domestic_edge,
                    "winner": "imported UCO" if imported_edge > domestic_edge else "domestic soyoil",
                }
            )
    return pd.DataFrame(rows)


# ===========================================================================
# Second module — the crushing balance
# ===========================================================================
@dataclass(frozen=True)
class CrushBalance:
    """The quantified translation of "can crushing keep up?"."""

    soyoil_required_lb: float
    crush_required_bu: float
    crush_required_bu_day: float
    installed_capacity_bu_day: float
    gap_bu_day: float

    @property
    def is_short(self) -> bool:
        return self.gap_bu_day > 0

    @property
    def headline(self) -> str:
        if self.is_short:
            return (
                f"Delivering this share of the mandate needs "
                f"{self.crush_required_bu_day:,.0f} bu/day of crushing against "
                f"{self.installed_capacity_bu_day:,.0f} announced: "
                f"{self.gap_bu_day:,.0f} bu/day short, "
                f"{self.gap_bu_day / self.installed_capacity_bu_day:.0%} of capacity."
            )
        return (
            f"Announced capacity ({self.installed_capacity_bu_day:,.0f} bu/day) "
            f"covers the need ({self.crush_required_bu_day:,.0f} bu/day) with "
            f"{-self.gap_bu_day:,.0f} bu/day to spare."
        )


def crush_from_soyoil_lb(
    soyoil_lb: float,
    *,
    installed_capacity_bu_day: float,
    oil_lb_per_bushel: float = OIL_LB_PER_BUSHEL,
) -> CrushBalance:
    """Crushing required for a given oil volume, without going through a mandate.

    A variant of `crush_balance` when the known constraint is an **oil volume**
    — e.g. the consumption increment projected by WASDE — rather than a mandate
    in gallons whose soyoil share would have to be assumed.
    """
    if soyoil_lb < 0:
        raise FeedstockError("the oil volume must be >= 0")
    if installed_capacity_bu_day <= 0:
        raise FeedstockError("installed capacity must be > 0")

    crush_bu = soyoil_lb / oil_lb_per_bushel
    crush_bu_day = crush_bu / 365.0
    return CrushBalance(
        soyoil_required_lb=float(soyoil_lb),
        crush_required_bu=crush_bu,
        crush_required_bu_day=crush_bu_day,
        installed_capacity_bu_day=float(installed_capacity_bu_day),
        gap_bu_day=crush_bu_day - installed_capacity_bu_day,
    )


def crush_balance(
    *,
    rvo_gallons: float,
    soyoil_share: float,
    installed_capacity_bu_day: float,
    yield_lb_gal: float = DEFAULT_YIELD_LB_GAL,
    oil_lb_per_bushel: float = OIL_LB_PER_BUSHEL,
) -> CrushBalance:
    """Crushing capacity required against announced capacity.

        soyoil_required_lb  = RVO_gal x soyoil_share x yield_lb_gal
        crush_required_bu   = soyoil_required_lb / 11
        gap_bu_day          = crush_required_bu / 365 - installed_capacity
    """
    if not 0.0 <= soyoil_share <= 1.0:
        raise FeedstockError(f"soyoil_share must be in [0, 1], got {soyoil_share}")
    if installed_capacity_bu_day <= 0:
        raise FeedstockError("installed capacity must be > 0")

    soyoil_lb = rvo_gallons * soyoil_share * yield_lb_gal
    crush_bu = soyoil_lb / oil_lb_per_bushel
    crush_bu_day = crush_bu / 365.0
    return CrushBalance(
        soyoil_required_lb=soyoil_lb,
        crush_required_bu=crush_bu,
        crush_required_bu_day=crush_bu_day,
        installed_capacity_bu_day=installed_capacity_bu_day,
        gap_bu_day=crush_bu_day - installed_capacity_bu_day,
    )


# ===========================================================================
# T3-5 — energy beta of the agri complex (a section, not a standalone page)
# ===========================================================================
def rolling_energy_beta(
    ag_price: pd.Series, brent: pd.Series, *, window: int = 120
) -> pd.DataFrame:
    """Rolling beta of Δln(agri price) on Δln(Brent).

    The "biofuel pull": when crude rises, the biofuel margin pulls vegetable
    oil. T3-5's question is whether beta has been structurally higher since
    2026, or whether that was an episode.

    Returns beta, r_squared and n_obs per window. A rolling window's `n_eff`
    equals n_obs/window (Rule C): show it alongside the chart, or beta looks far
    better estimated than it is.
    """
    aligned = pd.concat({"ag": ag_price, "brent": brent}, axis=1).dropna()
    if len(aligned) < window + 2:
        raise FeedstockError(
            f"not enough observations for a window of {window}: n={len(aligned)}"
        )
    returns = np.log(aligned).diff().dropna()

    beta = (
        returns["ag"].rolling(window).cov(returns["brent"])
        / returns["brent"].rolling(window).var()
    )
    correlation = returns["ag"].rolling(window).corr(returns["brent"])
    out = pd.DataFrame(
        {"beta": beta, "r_squared": correlation**2, "n_obs": window}
    ).dropna()
    out.attrs["n_eff_per_window"] = 1.0
    out.attrs["window"] = window
    return out


@dataclass(frozen=True)
class ChowTest:
    """A break test at a known date — policy dates, never searched-for dates."""

    break_date: pd.Timestamp
    f_stat: float
    p_value: float
    beta_before: float
    beta_after: float
    n_before: int
    n_after: int

    @property
    def rejects_stability(self) -> bool:
        return self.p_value < 0.05

    @property
    def summary(self) -> str:
        verdict = (
            "significant break" if self.rejects_stability else "no detectable break"
        )
        return (
            f"beta {self.beta_before:.3f} -> {self.beta_after:.3f} at {self.break_date:%d %b %Y} | "
            f"F = {self.f_stat:.2f}, p = {self.p_value:.4f} | {verdict}"
        )


def chow_break_test(
    ag_price: pd.Series, brent: pd.Series, break_date: str | pd.Timestamp
) -> ChowTest:
    """Chow test on the energy beta, at a policy date **chosen a priori**.

    The break point is given, never searched for in the data: hunting for the
    break that maximises F and reporting its nominal p-value is one of the
    fastest ways to produce a result that will not replicate. The legitimate
    dates here are those of the regulatory calendar — RVO finalisation in March
    2026, 45Z taking effect.
    """
    from scipy import stats as scipy_stats

    aligned = pd.concat({"ag": ag_price, "brent": brent}, axis=1).dropna()
    returns = np.log(aligned).diff().dropna()
    cut = pd.Timestamp(break_date)

    before = returns[returns.index < cut]
    after = returns[returns.index >= cut]
    if len(before) < 30 or len(after) < 30:
        raise FeedstockError(
            f"sub-samples too short around {cut:%d %b %Y}: "
            f"{len(before)} before, {len(after)} after (30 minimum each side)"
        )

    def _fit(sample: pd.DataFrame) -> tuple[float, float]:
        x = np.column_stack([np.ones(len(sample)), sample["brent"].to_numpy()])
        y = sample["ag"].to_numpy()
        coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
        residuals = y - x @ coefficients
        return float(coefficients[1]), float(residuals @ residuals)

    beta_pooled, rss_pooled = _fit(returns)
    beta_before, rss_before = _fit(before)
    beta_after, rss_after = _fit(after)

    k = 2                                   # constant + slope
    n = len(returns)
    numerator = (rss_pooled - (rss_before + rss_after)) / k
    denominator = (rss_before + rss_after) / (n - 2 * k)
    f_stat = numerator / denominator if denominator > 0 else float("inf")
    p_value = float(scipy_stats.f.sf(f_stat, k, n - 2 * k))

    return ChowTest(
        break_date=cut,
        f_stat=float(f_stat),
        p_value=p_value,
        beta_before=beta_before,
        beta_after=beta_after,
        n_before=len(before),
        n_after=len(after),
    )


SRE_WARNING = (
    "Small refinery exemptions (SRE) retroactively reallocate RVO volumes. Any "
    "conclusion drawn from a displayed mandate carries this uncertainty, and it "
    "is not modelled here (L-H6)."
)

__all__ = [
    "CENTS_PER_USD",
    "ChowTest",
    "CrushBalance",
    "DCO_DOMESTIC",
    "DiscountBurden",
    "Feedstock",
    "FeedstockError",
    "GateValue",
    "ImportPenalty",
    "LCFS_PROGRAM_HIGH_USD_T",
    "LCFS_PROGRAM_LOW_USD_T",
    "LcfsThreshold",
    "NoBreakevenInRange",
    "PenaltyBounds",
    "SOYOIL_DOMESTIC",
    "SRE_WARNING",
    "StructuralExit",
    "TALLOW_DOMESTIC",
    "UCO_IMPORTED",
    "calibration_gap_45z",
    "chow_break_test",
    "crush_balance",
    "crush_from_soyoil_lb",
    "discount_burden",
    "feedstock_breakeven_usd_lb",
    "gate_value",
    "import_penalty",
    "lcfs_breakeven",
    "lcfs_breakeven_numeric",
    "lcfs_neutral_price",
    "lcfs_value_usd_gal",
    "load_soyoil_usd_lb",
    "penalty_bounds",
    "rolling_energy_beta",
    "structural_exit",
    "winner_grid",
]
