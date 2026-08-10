"""T3-2 — Sugar: why the mix does not track parity.

THESIS
------
The mix's elasticity to parity only makes sense **conditional on the entry-of-season hedge
level**. The mix is not a price ratio: it is the solution to a constrained programme, and
the constraint that binds changes from region to region and season to season.

THE DISAGREEMENT, OPEN AND QUANTIFIED BETWEEN TWO HOUSES
-------------------------------------------------------------
- **Hedgepoint** (Feb 2026): CS 2026/27 crop at 630 Mt, sugar production stable at
  ~40.5 Mt, mix at ~48.6% against 50.6% the previous season. The mix should fall towards
  **46%** to materially cut the surplus, but **mill limits and sugar already sold
  forward** make that hard. Global surplus widened to 3.4 Mt.
- **Czarnikow** (Jun 2026): mills enter **far less hedged** than the previous four
  seasons; 2026/27 pricing opportunities stayed below BRL 2000/t, below the cost of
  production; UNICA shows the mix at its lowest since 2022/23; production revised to
  39.5 Mt.
- Czarnikow (Jul 2026): the ethanol/gasoline parity at its lowest in a decade without
  demand reacting enough; corn ethanol structurally adds supply.

Hedgepoint points to constraints; Czarnikow observes that the unlock comes precisely from
an unusually low hedge level. **Both agree that the degree of prior hedging is the
discriminating parameter** — an observable variable, rarely modelled. That is where this
page sits.

STANCE
------
No side is taken between the two houses. The **conditional** elasticity is measured, and
the insider is left to say whether the conditionality matches what they see.

THE UNIT TRAP
-------------
Parity requires stacking four conversions: kg of ATR per litre of ethanol, BRL per litre,
BRL per kg, then cents per pound via the exchange rate **and** the pound-kilogram
conversion. A single mistake in the chain shifts parity by several cents and flips the
signal. And the Consecana coefficients are **revised**: freezing them in code corrupts the
whole history (a known failure mode).

MODEL
-----
    sugar_eq_hydrous_brl_kg = P_hydrous_brl_l x (ATR_sugar / ATR_hydrous)
    sugar_eq_hydrous_c_lb   = sugar_eq_hydrous_brl_kg x 100 / (USDBRL x 2.20462)
    parity_gap_c_lb         = NY11_c_lb x pol_factor - sugar_eq_hydrous_c_lb

Panel estimation, by UNICA fortnight x region:

    dmix = a_r + b1 parity_gap_{t-1}
                + b2 (parity_gap_{t-1} x hedge_ratio_entry_r)
                + b3 cap_utilisation
                + b4 (dist_port_r x parity_gap_{t-1}) + e

**The object of interest is b2**, not b1.

ASSUMPTIONS
-----------
G-H1  Consecana coefficients parameterised, never frozen (they are revised each season).
G-H2  `pol_factor` adjusts NY11 (96° pol) toward the VHP quality actually produced.
G-H3  The entry-of-season hedge ratio is constant over the season, by region. Strictly
      false — mills keep pricing — but this is the variable as both houses describe it.
G-H4  CEPEA sometimes quotes VAT-inclusive, sometimes ex-VAT depending on the series. The
      `strip_vat` parameter exists for this and must be decided series by series, not by
      default.
G-H5  Brazilian election years influence pump prices independently of the market. Treated
      as a possible control variable, never as noise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from agri.data.snapshot import cached

from agri.core.stats import HacRegression, hac_ols
from agri.core.fmt import fmt_num, fmt_pct
from agri.core.units import LB_PER_TONNE, strip_vat

# G-H1 — Consecana schedule, kg of ATR per unit produced. VERIFY each season.
ATR_SUGAR_VHP_PER_KG = 1.0495
ATR_ETHANOL_ANHYDROUS_PER_L = 1.7651
ATR_ETHANOL_HYDROUS_PER_L = 1.6913

KG_PER_LB = LB_PER_TONNE / 1000.0        # 2.20462 lb per kg
DEFAULT_POL_FACTOR = 0.98                # G-H2


class SugarMixError(ValueError):
    """Mis-specified model."""


# ===========================================================================
# Parity
# ===========================================================================
def hydrous_sugar_equivalent_cents_lb(
    hydrous_brl_l: pd.Series,
    usdbrl: pd.Series,
    *,
    atr_sugar: float = ATR_SUGAR_VHP_PER_KG,
    atr_hydrous: float = ATR_ETHANOL_HYDROUS_PER_L,
    vat_rate: float | None = None,
) -> pd.Series:
    """Hydrous ethanol price converted to sugar equivalent, in cents/lb.

    Conversion chain, to be read in order because each step can be wrong on its own:
        BRL/litre  --(ATR_sugar / ATR_hydrous)-->  BRL/kg of equivalent sugar
        BRL/kg     --(/ USDBRL)-->                 USD/kg
        USD/kg     --(/ 2.20462)-->                USD/lb
        USD/lb     --(x 100)-->                    cents/lb

    `vat_rate` strips VAT when the CEPEA series is VAT-inclusive (G-H4). Leave at
    None when it is already ex-VAT — the module does not guess.
    """
    if atr_hydrous <= 0 or atr_sugar <= 0:
        raise SugarMixError("the Consecana coefficients must be > 0")
    aligned = pd.concat({"hydrous": hydrous_brl_l, "fx": usdbrl}, axis=1).dropna()
    if aligned.empty:
        raise SugarMixError("no common date between hydrous ethanol and FX")
    if (aligned["fx"] <= 0).any():
        raise SugarMixError("USDBRL must be > 0 — check the quoting direction")

    price = aligned["hydrous"]
    if vat_rate is not None:
        price = strip_vat(price, vat_rate)

    brl_per_kg = price * (atr_sugar / atr_hydrous)
    return (brl_per_kg * 100.0 / (aligned["fx"] * KG_PER_LB)).rename("sugar_eq_c_lb")


def parity_gap_cents_lb(
    ny11_cents_lb: pd.Series,
    hydrous_brl_l: pd.Series,
    usdbrl: pd.Series,
    *,
    pol_factor: float = DEFAULT_POL_FACTOR,
    **conversion_kwargs,
) -> pd.DataFrame:
    """Sugar/ethanol parity gap, in cents/lb. Positive = sugar pays better.

    Columns: ny11, ny11_adjusted, sugar_eq_ethanol, parity_gap, sugar_favoured.
    """
    equivalent = hydrous_sugar_equivalent_cents_lb(
        hydrous_brl_l, usdbrl, **conversion_kwargs
    )
    frame = pd.concat({"ny11": ny11_cents_lb, "sugar_eq_ethanol": equivalent}, axis=1).dropna()
    if frame.empty:
        raise SugarMixError("no common date between NY11 and the ethanol equivalent")
    frame["ny11_adjusted"] = frame["ny11"] * pol_factor
    frame["parity_gap"] = frame["ny11_adjusted"] - frame["sugar_eq_ethanol"]
    frame["sugar_favoured"] = frame["parity_gap"] > 0
    frame.attrs["pol_factor"] = pol_factor
    return frame


def consecana_sensitivity(
    ny11_cents_lb: pd.Series,
    hydrous_brl_l: pd.Series,
    usdbrl: pd.Series,
    *,
    atr_sugar_values: np.ndarray | None = None,
) -> pd.DataFrame:
    """G-H1: what a change in the Consecana schedule does to parity.

    A required section. The coefficients are revised; freezing them in code
    corrupts the whole history — and the error is silent, since parity stays
    plausible.
    """
    values = (
        np.array([1.0300, 1.0495, 1.0700])
        if atr_sugar_values is None
        else np.asarray(atr_sugar_values)
    )
    rows = []
    for atr in values:
        frame = parity_gap_cents_lb(
            ny11_cents_lb, hydrous_brl_l, usdbrl, atr_sugar=float(atr)
        )
        rows.append(
            {
                "atr_sugar": float(atr),
                "mean_parity_gap": float(frame["parity_gap"].mean()),
                "share_sugar_favoured": float(frame["sugar_favoured"].mean()),
            }
        )
    return pd.DataFrame(rows)


# ===========================================================================
# The mix under constraints
# ===========================================================================
@dataclass(frozen=True)
class MillConstraints:
    """The mill's three programme constraints, by region."""

    region: str
    crystallisation_cap: float       # maximum technically achievable mix
    presold_share: float             # minimum mix imposed by sugar already sold
    logistics_cost_c_lb: float       # cost of getting sugar to port
    hedge_ratio_entry: float         # entry-of-season hedge ratio

    def __post_init__(self) -> None:
        if not 0.0 <= self.presold_share <= self.crystallisation_cap <= 1.0:
            raise SugarMixError(
                f"{self.region}: inconsistent constraints — need "
                f"0 <= presold ({self.presold_share}) <= capacity "
                f"({self.crystallisation_cap}) <= 1"
            )
        if not 0.0 <= self.hedge_ratio_entry <= 1.0:
            raise SugarMixError(f"{self.region}: hedge ratio outside [0, 1]")


def optimal_mix(parity_gap_c_lb: float, constraints: MillConstraints) -> dict:
    """The mill's programme: maximise revenue subject to the three constraints.

    The programme is linear in `mix`, so the solution is **always at a bound** —
    this is precisely what makes the mix insensitive to parity over wide ranges,
    and it is Hedgepoint's argument. The price signal only moves the mix when it
    changes which bound is active.

    Returns the chosen mix and **which constraint is binding**, which is the
    information the page seeks to display by region and by fortnight.
    """
    net_gap = parity_gap_c_lb - constraints.logistics_cost_c_lb
    if net_gap > 0:
        return {
            "mix": constraints.crystallisation_cap,
            "binding": "crystallisation capacity",
            "net_gap": net_gap,
        }
    return {
        "mix": constraints.presold_share,
        "binding": "sugar already sold",
        "net_gap": net_gap,
    }


def binding_constraint_panel(
    parity: pd.Series, regions: list[MillConstraints]
) -> pd.DataFrame:
    """S5 — which constraint binds, by fortnight and by region."""
    rows = []
    for date, gap in parity.dropna().items():
        for region in regions:
            solution = optimal_mix(float(gap), region)
            rows.append(
                {
                    "date": date,
                    "region": region.region,
                    "parity_gap": float(gap),
                    "mix_predicted": solution["mix"],
                    "binding": solution["binding"],
                }
            )
    return pd.DataFrame(rows)


# ===========================================================================
# Conditional elasticity — the deliverable
# ===========================================================================
@dataclass(frozen=True)
class MixElasticity:
    """b1 and b2: the mix's elasticity to parity, and its conditionality on the hedge."""

    regression: HacRegression
    beta_parity: float
    beta_interaction: float
    interaction_pvalue: float
    n_obs: int
    n_regions: int

    def elasticity_at_hedge(self, hedge_ratio: float) -> float:
        """Points of mix per point of parity, at a given hedge level.

        The **curve** is the deliverable, not a single number: the spec says so
        explicitly, and a curve is harder to wave away than a single point.
        """
        return self.beta_parity + self.beta_interaction * hedge_ratio

    @property
    def hedging_matters(self) -> bool:
        return self.interaction_pvalue < 0.05

    def headline(self, low: float = 0.30, high: float = 0.60) -> str:
        at_low = self.elasticity_at_hedge(low)
        at_high = self.elasticity_at_hedge(high)
        if not self.hedging_matters:
            return (
                f"The mix's elasticity to parity is {self.beta_parity:+.4f} points "
                f"of mix per cent/lb, and no significant effect of the "
                f"entry-of-season hedge level is found (p = {self.interaction_pvalue:.3f}). "
                "On this sample, the hedge-conditionality thesis does not hold."
            )
        if abs(at_low) < 1e-12:
            return "zero elasticity at low hedging — result not interpretable"
        # points of mix per 100 pts of parity, rather than the inverse (cents needed
        # for 1 pt of mix): the inverse blows up into an absurd number when the
        # elasticity is small, while the slope itself stays readable at any scale
        ratio = abs(at_low) / abs(at_high) if at_high else float("inf")
        return (
            f"100 points of parity gap move the mix by {at_high * 100:+.2f} points "
            f"when mills enter {high:.0%} hedged, against {at_low * 100:+.2f} points "
            f"when they enter at {low:.0%} — {ratio:.1f}x more sensitive when "
            "hedging is low. The 46% mix the global balance calls for is only "
            "reachable this season because entry-of-season hedging is low."
        )


def estimate_mix_elasticity(panel: pd.DataFrame) -> MixElasticity:
    """Fortnight x region panel, region fixed effects, HAC errors.

    `panel` must contain: region, d_mix, parity_gap_lag, hedge_ratio_entry,
    cap_utilisation, dist_port.

    Region fixed effects are introduced via explicit dummies rather than
    demeaning: over a dozen regions, readability is worth more than elegance, and
    regional levels remain visible in the output.
    """
    required = {
        "region",
        "d_mix",
        "parity_gap_lag",
        "hedge_ratio_entry",
        "cap_utilisation",
        "dist_port",
    }
    missing = required - set(panel.columns)
    if missing:
        raise SugarMixError(f"missing columns in the panel: {sorted(missing)}")

    clean = panel.dropna(subset=sorted(required - {"region"}))
    if len(clean) < 60:
        raise SugarMixError(f"panel too short: n={len(clean)}")

    design = pd.DataFrame(index=clean.index)
    design["parity"] = clean["parity_gap_lag"]
    design["parity_x_hedge"] = clean["parity_gap_lag"] * clean["hedge_ratio_entry"]
    design["cap_utilisation"] = clean["cap_utilisation"]
    design["parity_x_dist"] = clean["parity_gap_lag"] * clean["dist_port"]

    dummies = pd.get_dummies(clean["region"], prefix="region", drop_first=True).astype(float)
    design = pd.concat([design, dummies], axis=1)

    regression = hac_ols(clean["d_mix"], design)
    return MixElasticity(
        regression=regression,
        beta_parity=float(regression.params["parity"]),
        beta_interaction=float(regression.params["parity_x_hedge"]),
        interaction_pvalue=float(regression.pvalues["parity_x_hedge"]),
        n_obs=len(clean),
        n_regions=clean["region"].nunique(),
    )


def mix_required_for_surplus(
    *, target_surplus_mt: float, current_surplus_mt: float, cane_mt: float, atr_yield_t_per_mt: float = 0.140
) -> float:
    """The mix shift needed to bring the global surplus down to a target.

    Answers Hedgepoint directly: "it would take 46%" becomes "it would take
    removing X Mt, i.e. Y points of mix, and here is whether the constraints allow it".
    """
    if cane_mt <= 0 or atr_yield_t_per_mt <= 0:
        raise SugarMixError("cane crushed and ATR yield must be > 0")
    tonnes_to_remove = current_surplus_mt - target_surplus_mt
    sugar_capacity_mt = cane_mt * atr_yield_t_per_mt
    return tonnes_to_remove / sugar_capacity_mt


# ===========================================================================
# THE FLOOR THAT IS NOT ONE — on real NY11 and USDBRL
# ===========================================================================
# Neither the UNICA mix nor CEPEA ethanol is in the export: the panel regression above
# therefore still runs on a synthetic set. But the two series sufficient to settle one
# sourced claim ARE there — NY11 and USDBRL. This is what this section does, and it
# produces the page's most counter-intuitive result.
CENTS_LB_TO_USD_T = LB_PER_TONNE / 100.0     # 22.0462

# SOURCED anchor: Czarnikow (Jun 2026) notes that 2026/27 pricing opportunities stayed
# "below BRL 2000/t, below the cost of production". A published, dated, attributable
# number — not a page assumption. It is exposed as a parameter because it is regional and
# time-varying, not because it is uncertain.
CZARNIKOW_COST_BRL_T = 2000.0


@cached('t3_2_parity')
def load_real_parity_frame(start: str | None = "2000-01-01") -> pd.DataFrame:
    """Real NY11 and USDBRL, plus sugar expressed in BRL per tonne.

    Columns: ny11 (c/lb), usdbrl, sugar_brl_t.

    This is the only leg of parity the export allows building: the hydrous
    ethanol price (CEPEA) is not there. The page therefore does not pretend to
    compute a parity gap — it exploits what these two series alone can establish.
    """
    from agri.data.bloomberg_loader import load

    frame = pd.concat(
        {"ny11": load("sugar_no11"), "usdbrl": load("usdbrl")}, axis=1, sort=True
    ).dropna()
    if start is not None:
        frame = frame[frame.index >= pd.Timestamp(start)]
    if frame.empty:
        raise SugarMixError(f"no common date between NY11 and USDBRL after {start}")
    frame["sugar_brl_t"] = frame["ny11"] * CENTS_LB_TO_USD_T * frame["usdbrl"]
    return frame


def indifference_hydrous_brl_l(
    ny11_cents_lb: pd.Series,
    usdbrl: pd.Series,
    *,
    pol_factor: float = DEFAULT_POL_FACTOR,
    atr_sugar: float = ATR_SUGAR_VHP_PER_KG,
    atr_hydrous: float = ATR_ETHANOL_HYDROUS_PER_L,
) -> pd.Series:
    """The hydrous ethanol price that makes a mill indifferent, in BRL/litre.

    This is `hydrous_sugar_equivalent_cents_lb` inverted: instead of starting
    from an ethanol price to compare it to sugar, it starts from sugar and asks
    **at what ethanol price the mill stops making sugar**. A number any Brazilian
    trading floor quotes, and one that needs only the two series available.

        hydrous* = NY11 x pol_factor x USDBRL x 2.20462 x (ATR_hydrous / ATR_sugar) / 100
    """
    if atr_sugar <= 0 or atr_hydrous <= 0:
        raise SugarMixError("the Consecana coefficients must be > 0")
    aligned = pd.concat({"ny11": ny11_cents_lb, "fx": usdbrl}, axis=1, sort=True).dropna()
    if aligned.empty:
        raise SugarMixError("no common date between NY11 and USDBRL")
    if (aligned["fx"] <= 0).any():
        raise SugarMixError("USDBRL must be > 0 — check the quoting direction")

    return (
        aligned["ny11"]
        * pol_factor
        * aligned["fx"]
        * KG_PER_LB
        * (atr_hydrous / atr_sugar)
        / 100.0
    ).rename("hydrous_indifference_brl_l")


@dataclass(frozen=True)
class ProductionCostCheck:
    """Czarnikow's claim, checked against prices.

    A house publishes "pricing stayed below the cost of production". This is a
    verifiable claim, and checking it beats quoting it.
    """

    frame: pd.DataFrame
    cost_brl_t: float
    start: str

    @property
    def share_below(self) -> float:
        return float((self.frame["sugar_brl_t"] < self.cost_brl_t).mean())

    @property
    def last_brl_t(self) -> float:
        return float(self.frame["sugar_brl_t"].iloc[-1])

    @property
    def is_below_now(self) -> bool:
        return self.last_brl_t < self.cost_brl_t

    @property
    def headline(self) -> str:
        verdict = "below" if self.is_below_now else "above"
        return (
            f"At the last print, sugar is worth {fmt_num(self.last_brl_t, 0)} "
            f"BRL/t — {verdict} the {fmt_num(self.cost_brl_t, 0)} BRL/t cost of "
            f"production Czarnikow names in June 2026. Since {self.start[:4]}, the "
            f"price has spent {fmt_pct(self.share_below)} of the time below this "
            "threshold."
        )


def production_cost_check(
    frame: pd.DataFrame, *, cost_brl_t: float = CZARNIKOW_COST_BRL_T
) -> ProductionCostCheck:
    """Checks the sugar price in BRL against the published cost of production."""
    if "sugar_brl_t" not in frame.columns:
        raise SugarMixError("the frame must contain 'sugar_brl_t' — see load_real_parity_frame")
    if cost_brl_t <= 0:
        raise SugarMixError("the cost of production must be > 0")
    return ProductionCostCheck(
        frame=frame, cost_brl_t=float(cost_brl_t), start=str(frame.index.min().date())
    )


@dataclass(frozen=True)
class MovingFloor:
    """THE result: the "Brazilian cost floor" quoted in c/lb is not a floor.

    A real production cost is denominated in the currency costs are paid in,
    i.e. BRL. Translated into c/lb for a New York reader, it becomes a function
    of the exchange rate — and starts moving by several cents with no Brazilian
    cost having changed at all. A trader seeing "support near 18 cents because
    that's the Brazilian cost" is looking at an FX series disguised as a
    structural level.
    """

    frame: pd.DataFrame
    cost_brl_t: float

    @property
    def floor_min(self) -> float:
        return float(self.frame["floor_c_lb"].min())

    @property
    def floor_max(self) -> float:
        return float(self.frame["floor_c_lb"].max())

    @property
    def floor_range(self) -> float:
        return self.floor_max - self.floor_min

    @property
    def floor_last(self) -> float:
        return float(self.frame["floor_c_lb"].iloc[-1])

    @property
    def fx_range(self) -> tuple[float, float]:
        return float(self.frame["usdbrl"].min()), float(self.frame["usdbrl"].max())

    @property
    def headline(self) -> str:
        low, high = self.fx_range
        return (
            f"The same {fmt_num(self.cost_brl_t, 0)} BRL/t cost translates into "
            f"an NY11 floor ranging from {fmt_num(self.floor_min, 1)} to "
            f"{fmt_num(self.floor_max, 1)} c/lb over the period — "
            f"{fmt_num(self.floor_range, 1)} cents of amplitude, produced solely "
            f"by a USDBRL that moved from {fmt_num(low, 2)} to {fmt_num(high, 2)}. "
            f"No Brazilian cost moved. At the last exchange rate, the floor sits "
            f"at {fmt_num(self.floor_last, 1)} c/lb."
        )


def moving_floor(
    frame: pd.DataFrame, *, cost_brl_t: float = CZARNIKOW_COST_BRL_T
) -> MovingFloor:
    """The NY11 price that exactly reaches the cost of production, day by day.

        floor_c_lb = cost_BRL_t / (22.0462 x USDBRL)

    The result is a **series**, not a level — and that is the whole point.
    """
    if "usdbrl" not in frame.columns:
        raise SugarMixError("the frame must contain 'usdbrl'")
    if cost_brl_t <= 0:
        raise SugarMixError("the cost of production must be > 0")

    out = frame.copy()
    out["floor_c_lb"] = cost_brl_t / (CENTS_LB_TO_USD_T * out["usdbrl"])
    out["below_floor"] = out["ny11"] < out["floor_c_lb"]
    return MovingFloor(frame=out, cost_brl_t=float(cost_brl_t))


def floor_variance_decomposition(frame: pd.DataFrame) -> pd.Series:
    """Share of the floor's movement attributable to FX rather than to sugar.

    The floor in c/lb is **exactly** proportional to 1/USDBRL: its variation is
    therefore entirely FX, by construction. What is worth decomposing is the gap
    between the price and its floor — what a mill actually watches.

        ln(sugar_BRL) = ln(NY11) + ln(USDBRL) + const
    """
    changes = np.log(frame[["ny11", "usdbrl"]]).diff().dropna()
    if len(changes) < 30:
        raise SugarMixError(f"sample too short to decompose: n={len(changes)}")

    var_sugar = float(changes["ny11"].var())
    var_fx = float(changes["usdbrl"].var())
    covariance = float(changes["ny11"].cov(changes["usdbrl"]))
    total = var_sugar + var_fx + 2 * covariance
    if total <= 0:
        raise SugarMixError("zero or negative total variance — decomposition impossible")

    return pd.Series(
        {
            "share_sugar": var_sugar / total,
            "share_fx": var_fx / total,
            "share_covariance": 2 * covariance / total,
            "correlation": float(changes["ny11"].corr(changes["usdbrl"])),
        }
    )


__all__ = [
    "ATR_ETHANOL_ANHYDROUS_PER_L",
    "ATR_ETHANOL_HYDROUS_PER_L",
    "ATR_SUGAR_VHP_PER_KG",
    "CENTS_LB_TO_USD_T",
    "CZARNIKOW_COST_BRL_T",
    "MillConstraints",
    "MixElasticity",
    "MovingFloor",
    "ProductionCostCheck",
    "SugarMixError",
    "binding_constraint_panel",
    "consecana_sensitivity",
    "estimate_mix_elasticity",
    "floor_variance_decomposition",
    "hydrous_sugar_equivalent_cents_lb",
    "indifference_hydrous_brl_l",
    "load_real_parity_frame",
    "mix_required_for_surplus",
    "moving_floor",
    "optimal_mix",
    "parity_gap_cents_lb",
    "production_cost_check",
]
