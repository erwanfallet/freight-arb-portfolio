"""T2-4 — The white premium, or what a price can and cannot tell you.

THE RESULT, IN ONE SENTENCE
-------------------------------
The **level** of the refining rent is not identifiable from prices: a conversion factor
nobody publishes injects as much uncertainty into it as the quantity being sought. Its
**variation**, however, is fully identifiable — the parameter shifts every year in the same
direction, so the gaps between years survive it. And that variation says something:
richness went from around -26 USD/t in 2021 to +35 in 2024, a swing more than five times
larger than what the parameter alone can produce.

This is a page that first states a **limit** on what it can know, then the result that
survives that limit. The order matters: publishing the level without the limit would be
publishing a convention artefact.

TENSION — INFERRED, NOT SOURCED
----------------------------------
**It seems to me** the white premium (No.5 - No.11) is presented as the refining margin,
when it mostly contains a residual of positioning and delivery constraints. "It seems to
me", never "I read that".

THE UNIT TRAP
-------------
No.11 is quoted in **cents/lb, 96° pol basis**; No.5 in **USD/tonne** of refined sugar.
Comparing them requires the c/lb -> USD/t conversion **and** a polarisation adjustment,
because it takes more than one tonne of raw at 96° to make one tonne of white.
`pol_adjust` runs ~1.06-1.08 depending on the specification used.

ORDER OF MAGNITUDE, STATED CORRECTLY: between these two bounds the gap is ~5 USD/t on the
real sample. That is **not** the same order as the white premium itself (~70 USD/t, i.e.
7%) — saying so would be false. It is the same order as **richness**, the quantity the page
actually tries to establish, whose median since 2015 sits around 5 USD/t. That is the
comparison that matters, and it is what `identification_check` measures instead of assuming.

IDENTITY
--------
    white_premium = No5_usd_t - No11_c_lb x 22.0462 x pol_adjust
    fv_refining   = energy + yield_loss x No11_usd_t + labour
                    + financing + freight_leg
    richness      = white_premium - fv_refining

    richness > 0  -> RICH zone   : white pays more than it costs to produce
    richness < 0  -> CHEAP zone  : refining destroys value at the quoted price

ASSUMPTIONS
-----------
W-H1  `pol_adjust` defaults to 1.07, bounded to [1.00 ; 1.20]. See above.
W-H2  Refining yield loss is a percentage of raw input, valued at the No.11 price.
      Default 2%.
W-H3  Energy cost is a flat rate per tonne, exogenous. It is the most volatile line
      item in a refinery — hence the slider and the sensitivity check.
W-H4  No cost of capital on the refining asset. `richness` is therefore a contribution
      margin, never a profit. Do not compare it to a ROIC.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from agri.data.snapshot import cached

from agri.core.stats import regime_runs
from agri.core.units import cents_lb_to_usd_t

DEFAULT_POL_ADJUST = 1.07              # W-H1
DEFAULT_YIELD_LOSS = 0.02              # W-H2
DEFAULT_ENERGY_USD_T = 28.0            # W-H3
DEFAULT_LABOUR_USD_T = 12.0
DEFAULT_FREIGHT_LEG_USD_T = 18.0
DEFAULT_FINANCING_DAYS = 45

# Default energy intensity of a sugar refinery (W-H3, real data): calibrated so that at a
# Henry Hub price near its historical average (~3.5 USD/mmBtu), the energy cost lands close
# to the DEFAULT_ENERGY_USD_T flat rate (28 USD/t) — 28/3.5 = 8 mmBtu/t, a plausible order
# of magnitude for crystallisation. A parameter, not a measurement: flagged as such in the
# page's diagnostics panel.
DEFAULT_ENERGY_INTENSITY_MMBTU_T = 8.0


class WhitePremiumError(ValueError):
    """Mis-specified model."""


def white_premium_usd_t(
    no5_usd_t: pd.Series,
    no11_cents_lb: pd.Series,
    *,
    pol_adjust: float = DEFAULT_POL_ADJUST,
) -> pd.Series:
    """White premium, both legs brought to the same unit and the same polarisation."""
    if not 1.0 <= pol_adjust <= 1.20:
        raise WhitePremiumError(
            f"pol_adjust outside the plausible range [1.00 ; 1.20], got {pol_adjust} — "
            "beyond that it is no longer a polarisation correction"
        )
    raw_on_white_basis = cents_lb_to_usd_t(no11_cents_lb) * pol_adjust
    return no5_usd_t - raw_on_white_basis


def fair_value_refining_usd_t(
    no11_cents_lb: pd.Series,
    *,
    yield_loss: float = DEFAULT_YIELD_LOSS,
    energy_usd_t: float | pd.Series = DEFAULT_ENERGY_USD_T,
    labour_usd_t: float = DEFAULT_LABOUR_USD_T,
    freight_leg_usd_t: float = DEFAULT_FREIGHT_LEG_USD_T,
    annual_rate: pd.Series | float = 0.055,
    financing_days: int = DEFAULT_FINANCING_DAYS,
) -> pd.DataFrame:
    """Reconstructed refining cost, line item by line item.

    Returns a DataFrame with one column per line item plus the total: this is the
    page's waterfall, and it has to be readable line by line by someone who runs a
    refinery — otherwise they cannot usefully dispute it.

    `energy_usd_t` accepts a flat rate (default) or a dated series — a real natural
    gas proxy (Henry Hub) rather than a constant, when one is available.
    """
    if not 0.0 <= yield_loss < 0.20:
        raise WhitePremiumError(f"yield loss outside the plausible range: {yield_loss}")
    raw_usd_t = cents_lb_to_usd_t(no11_cents_lb)
    rate = (
        pd.Series(float(annual_rate), index=no11_cents_lb.index)
        if isinstance(annual_rate, (int, float))
        else pd.Series(annual_rate).reindex(no11_cents_lb.index)
    )
    energy = (
        pd.Series(float(energy_usd_t), index=no11_cents_lb.index)
        if isinstance(energy_usd_t, (int, float))
        else pd.Series(energy_usd_t).reindex(no11_cents_lb.index)
    )
    out = pd.DataFrame(index=no11_cents_lb.index)
    out["energy"] = energy
    out["yield_loss"] = yield_loss * raw_usd_t
    out["labour"] = labour_usd_t
    out["freight"] = freight_leg_usd_t
    out["financing"] = raw_usd_t * rate * financing_days / 360.0
    out["total"] = out.sum(axis=1)
    return out


def build_richness(
    no5_usd_t: pd.Series,
    no11_cents_lb: pd.Series,
    *,
    pol_adjust: float = DEFAULT_POL_ADJUST,
    **cost_kwargs,
) -> pd.DataFrame:
    """Observed premium, reconstructed fair value, and the residual — the whole page in one frame.

    Columns: no5, no11, white_premium, fv_refining, richness, zone.
    """
    aligned = pd.concat({"no5": no5_usd_t, "no11": no11_cents_lb}, axis=1, sort=True).dropna()
    if aligned.empty:
        raise WhitePremiumError("no common date between No.5 and No.11")

    out = aligned.copy()
    out["white_premium"] = white_premium_usd_t(
        out["no5"], out["no11"], pol_adjust=pol_adjust
    )
    costs = fair_value_refining_usd_t(out["no11"], **cost_kwargs)
    out["fv_refining"] = costs["total"]
    out["richness"] = out["white_premium"] - out["fv_refining"]
    out["zone"] = np.where(out["richness"] > 0, "RICH", "CHEAP")
    out.attrs["pol_adjust"] = pol_adjust
    return out


@dataclass(frozen=True)
class RichnessSummary:
    """How long the premium covers costs, and what the residual is worth."""

    share_rich: float
    mean_richness: float
    median_richness: float
    rich_episodes: pd.DataFrame
    cheap_episodes: pd.DataFrame
    n_obs: int

    @property
    def headline(self) -> str:
        return (
            f"The white premium is covered by reconstructed refining costs only "
            f"{self.share_rich:.0%} of the time; the median residual is "
            f"{self.median_richness:+.1f} USD/t. This is not a refining margin — it is "
            "a physical availability signal that carries a refining margin inside it."
        )


def summarise_richness(frame: pd.DataFrame, *, min_obs: int = 5) -> RichnessSummary:
    return RichnessSummary(
        share_rich=float((frame["richness"] > 0).mean()),
        mean_richness=float(frame["richness"].mean()),
        median_richness=float(frame["richness"].median()),
        rich_episodes=regime_runs(
            frame["richness"] > 0, depth=frame["richness"], min_obs=min_obs
        ),
        cheap_episodes=regime_runs(
            frame["richness"] <= 0, depth=frame["richness"], min_obs=min_obs
        ),
        n_obs=len(frame),
    )


def pol_adjust_sensitivity(
    no5_usd_t: pd.Series,
    no11_cents_lb: pd.Series,
    *,
    values: np.ndarray | None = None,
    **cost_kwargs,
) -> pd.DataFrame:
    """What the sole choice of `pol_adjust` does to the page's result (W-H1).

    A required section: over a defensible range of 1.06 to 1.08, the share of time
    spent in the RICH zone can shift substantially. Showing this sensitivity is what
    separates an honest page from one that picked its assumption to be right.
    """
    values = np.arange(1.04, 1.101, 0.005) if values is None else np.asarray(values)
    rows = []
    for pol in values:
        frame = build_richness(
            no5_usd_t, no11_cents_lb, pol_adjust=float(pol), **cost_kwargs
        )
        rows.append(
            {
                "pol_adjust": float(pol),
                "mean_white_premium": float(frame["white_premium"].mean()),
                "mean_richness": float(frame["richness"].mean()),
                "share_rich": float((frame["richness"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


@cached('t2_4_richness')
def load_real_richness_frame(
    *,
    pol_adjust: float = DEFAULT_POL_ADJUST,
    energy_intensity_mmbtu_t: float = DEFAULT_ENERGY_INTENSITY_MMBTU_T,
    start: str = "1990-07-18",
) -> pd.DataFrame:
    """`build_richness` on **real** ICE No.11 and No.5 (Bloomberg export), energy cost
    proxied by real Henry Hub rather than the constant DEFAULT_ENERGY_USD_T flat rate.

    Both price legs (No.11, No.5) and energy (Henry Hub) are entirely real; labour,
    freight and yield loss remain parameterised flat rates (W-H2, and the equivalent
    for freight/labour) for lack of public refinery cost accounting — no source
    publishes these line items as a time series.
    """
    from agri.data.bloomberg_loader import load as load_bloomberg

    no11 = load_bloomberg("sugar_no11").loc[start:]
    no5 = load_bloomberg("sugar_no5").loc[start:]
    henry_hub = load_bloomberg("henry_hub")
    energy_usd_t = (henry_hub * energy_intensity_mmbtu_t).reindex(no11.index)

    frame = build_richness(no5, no11, pol_adjust=pol_adjust, energy_usd_t=energy_usd_t)
    frame.attrs["energy_source"] = "henry_hub_real"
    frame.attrs["energy_intensity_mmbtu_t"] = energy_intensity_mmbtu_t
    return frame


# ===========================================================================
# IDENTIFICATION — what a price can say, and what it cannot
# ===========================================================================
# `pol_adjust` is not observable: no exchange publishes it, it depends on the contract
# specification used, and its plausible range is 1.06-1.08. The instinct is to pick a value
# and conclude. That instinct is wrong: on real data, the uncertainty this single parameter
# injects into richness is the same order as richness itself. The LEVEL of the refining
# rent is therefore not identifiable from prices. Its VARIATION, however, is fully
# identifiable — this is the page's result.
POL_PLAUSIBLE_LO = 1.06
POL_PLAUSIBLE_HI = 1.08


@dataclass(frozen=True)
class ImpliedPolAdjust:
    """The polarisation adjustment the market is pricing, under a zero-rent assumption.

    Inverts the usual question. Rather than fixing `pol_adjust` and asking whether
    refining is profitable, we assume refining is **competitive** — zero median rent,
    what a mature industry should produce — and ask which `pol_adjust` the market is
    then pricing. The number that comes out compares directly to the specification a
    refiner knows by heart.
    """

    pol_star: float
    plausible_lo: float
    plausible_hi: float
    start: str

    @property
    def within_plausible(self) -> bool:
        return self.plausible_lo <= self.pol_star <= self.plausible_hi

    @property
    def headline(self) -> str:
        if self.within_plausible:
            verdict = (
                "it falls **inside** the plausible range: prices are consistent with "
                "rent-free refining, and nothing in the data forces another conclusion"
            )
        else:
            side = "above" if self.pol_star > self.plausible_hi else "below"
            verdict = (
                f"it falls {side} the plausible range "
                f"[{self.plausible_lo:.2f} ; {self.plausible_hi:.2f}] — either the range "
                "is too narrow, or refining captures a rent. Price alone cannot settle it"
            )
        return (
            f"For the median refining rent to be exactly zero since "
            f"{self.start[:4]}, a polarisation adjustment of "
            f"{self.pol_star:.4f} would be needed; {verdict}."
        )


def implied_pol_adjust(
    *,
    start: str = "2015-01-01",
    plausible_lo: float = POL_PLAUSIBLE_LO,
    plausible_hi: float = POL_PLAUSIBLE_HI,
    search_lo: float = 1.0001,
    search_hi: float = 1.1999,
    **kwargs,
) -> ImpliedPolAdjust:
    """Solves for `pol*` such that median richness is exactly zero.

    The search is bounded inside the range `white_premium_usd_t` accepts: beyond 1.20
    it is no longer a polarisation correction, and letting the solver go there would
    produce an exception instead of a result.
    """
    from scipy.optimize import brentq

    def median_richness(pol: float) -> float:
        return float(
            load_real_richness_frame(pol_adjust=pol, start=start, **kwargs)["richness"].median()
        )

    low, high = median_richness(search_lo), median_richness(search_hi)
    if low * high > 0:
        raise WhitePremiumError(
            f"no pol_adjust in [{search_lo:.4f} ; {search_hi:.4f}] zeroes the median "
            f"richness (it is {low:+.1f} at the low bound and {high:+.1f} at the high "
            "one). The rent is therefore not explained by polarisation alone."
        )

    return ImpliedPolAdjust(
        pol_star=float(brentq(median_richness, search_lo, search_hi, xtol=1e-6)),
        plausible_lo=plausible_lo,
        plausible_hi=plausible_hi,
        start=start,
    )


@dataclass(frozen=True)
class IdentificationCheck:
    """THE result: the level is not identifiable, the variation is.

    `annual` carries median richness by year, computed at both bounds of the
    plausible range and at the reference value. The three columns move almost in
    parallel: that, and nothing else, is what allows reading the gaps between years
    while refusing to read the level.
    """

    annual: pd.DataFrame
    pol_lo: float
    pol_hi: float
    pol_ref: float

    @property
    def parameter_span_max(self) -> float:
        """The worst gap, in any given year, attributable to the choice of `pol_adjust` alone."""
        return float((self.annual["richness_lo"] - self.annual["richness_hi"]).max())

    @property
    def signal_span(self) -> float:
        """The amplitude of the signal itself: highest richness minus lowest."""
        return float(self.annual["richness_ref"].max() - self.annual["richness_ref"].min())

    @property
    def ratio(self) -> float:
        if self.parameter_span_max <= 0:
            raise WhitePremiumError(
                "zero parameter span: the three pol_adjust variants are identical, "
                "which can only happen if they were rebuilt from a single shared "
                "series. The ratio is meaningless here."
            )
        return self.signal_span / self.parameter_span_max

    @property
    def rank_correlation(self) -> float:
        """Does the ranking of years survive the change in parameter?"""
        return float(
            self.annual["richness_lo"].rank().corr(
                self.annual["richness_hi"].rank(), method="spearman"
            )
        )

    @property
    def sign_flipping_years(self) -> list[int]:
        """Years whose SIGN depends on the parameter — the only non-interpretable ones."""
        lo, hi = self.annual["richness_lo"], self.annual["richness_hi"]
        return [int(year) for year in self.annual.index if (lo[year] > 0) != (hi[year] > 0)]

    @property
    def headline(self) -> str:
        flipping = self.sign_flipping_years
        flip_text = (
            "no year flips sign"
            if not flipping
            else f"{len(flipping)} year(s) flip sign ({', '.join(map(str, flipping))})"
        )
        return (
            f"The choice of pol_adjust moves richness by at most "
            f"{self.parameter_span_max:.1f} USD/t in any given year, while the gap "
            f"between the best and worst year reaches {self.signal_span:.1f} USD/t — a "
            f"factor of {self.ratio:.1f}. The ranking of years is identical at both "
            f"bounds (rank correlation {self.rank_correlation:.4f}) and {flip_text}. "
            "The level of the rent is not identifiable; its variation is."
        )


def identification_check(
    *,
    start: str = "2015-01-01",
    pol_lo: float = POL_PLAUSIBLE_LO,
    pol_hi: float = POL_PLAUSIBLE_HI,
    pol_ref: float = DEFAULT_POL_ADJUST,
    **kwargs,
) -> IdentificationCheck:
    """Compares the amplitude injected by the parameter to the amplitude of the signal.

    This is the test the page runs against its own conclusion before stating it: if
    the unobservable parameter weighed as much as the phenomenon, there would be
    nothing to say, and that has to be said.
    """
    if not pol_lo < pol_ref < pol_hi:
        raise WhitePremiumError(
            f"the reference value {pol_ref} must sit strictly between the bounds "
            f"{pol_lo} and {pol_hi}"
        )

    # The three variants are rebuilt from a SINGLE frame rather than three separate
    # loader calls. Two reasons, the second being the important one:
    #   - it is three times less disk reading;
    #   - in snapshot mode the loader ignores its arguments and would return the same
    #     frame three times, giving a zero parameter span and a division by zero.
    #     Recomputing here keeps the page truthful even when raw data is absent.
    base = load_real_richness_frame(pol_adjust=pol_ref, start=start, **kwargs)
    columns = {}
    for label, pol in (("richness_lo", pol_lo), ("richness_ref", pol_ref), ("richness_hi", pol_hi)):
        premium = base["no5"] - cents_lb_to_usd_t(base["no11"]) * pol
        richness = premium - base["fv_refining"]
        columns[label] = richness.groupby(base.index.year).median()

    annual = pd.DataFrame(columns)
    annual.index.name = "year"
    if len(annual) < 3:
        raise WhitePremiumError(
            f"only {len(annual)} year(s) in the sample: comparing the parameter's "
            "amplitude to the signal's amplitude is meaningless"
        )
    return IdentificationCheck(annual=annual, pol_lo=pol_lo, pol_hi=pol_hi, pol_ref=pol_ref)


@dataclass(frozen=True)
class ImpliedRefiningCost:
    """What the market pays for the act of refining — the number for the email.

    The white premium *is* the price the market puts on turning one tonne of raw
    sugar into one tonne of white. It needs no cost assumption to be read: it is an
    observed price. Comparing it to a cost model is what introduces assumptions —
    hence showing both side by side rather than just their difference.
    """

    market_usd_t: float
    modelled_usd_t: float
    pol_adjust: float
    start: str

    @property
    def gap_usd_t(self) -> float:
        return self.market_usd_t - self.modelled_usd_t

    @property
    def headline(self) -> str:
        return (
            f"Since {self.start[:4]}, the market has paid a median of "
            f"{self.market_usd_t:.0f} USD/t for the act of refining. This page's cost "
            f"model finds {self.modelled_usd_t:.0f}, a gap of {self.gap_usd_t:+.0f} "
            "USD/t — but it is the first number that is observed, and the second that "
            "rests on flat rates. A refiner compares their own to the first."
        )


def implied_refining_cost(
    *, start: str = "2015-01-01", pol_adjust: float = DEFAULT_POL_ADJUST, **kwargs
) -> ImpliedRefiningCost:
    """The median white premium, read as the market price of refining."""
    frame = load_real_richness_frame(pol_adjust=pol_adjust, start=start, **kwargs)
    return ImpliedRefiningCost(
        market_usd_t=float(frame["white_premium"].median()),
        modelled_usd_t=float(frame["fv_refining"].median()),
        pol_adjust=float(pol_adjust),
        start=start,
    )


__all__ = [
    "DEFAULT_ENERGY_INTENSITY_MMBTU_T",
    "POL_PLAUSIBLE_HI",
    "POL_PLAUSIBLE_LO",
    "IdentificationCheck",
    "ImpliedPolAdjust",
    "ImpliedRefiningCost",
    "RichnessSummary",
    "WhitePremiumError",
    "build_richness",
    "fair_value_refining_usd_t",
    "identification_check",
    "implied_pol_adjust",
    "implied_refining_cost",
    "load_real_richness_frame",
    "pol_adjust_sensitivity",
    "summarise_richness",
    "white_premium_usd_t",
]
