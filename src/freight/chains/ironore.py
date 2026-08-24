"""Project A — decomposing the 65-62% Fe premium into a freight share and a residual.

THESIS
------
The 62% Fe and 65% Fe iron ore indices are both quoted **CFR China**, so freight is
already inside both prices. The 65% grade is mostly Brazilian (Tubarão -> Qingdao,
~11,000 nm, Baltic route C3), the 62% grade mostly Australian (Port Hedland -> Qingdao,
~1,600 nm, route C5). The observed premium therefore mechanically contains the C3 - C5
freight differential.

    observed_premium = P65_CFR - P62_CFR
    fair_value_freight = C3_per_dmt - C5_per_dmt
    residual            = observed_premium - fair_value_freight

The residual is NOT "physical tension." It's "quality + use-value + tension + the
FFA-vs-index basis." It doesn't get renamed anything else: no public data lets these
four terms be separated, and an invented use-value model would turn the result into an
artefact.

THE UNIT TRAP, WHICH IS THE TECHNICAL CORE
-------------------------------------------
Ore indices are quoted in USD per **dry** metric tonne (dmt). Freight is paid on the
**loaded** weight, i.e. wet (wmt, bill-of-lading weight). So:

    freight_per_dmt = freight_per_wmt / (1 - moisture)

Brazilian fines come out around 9% moisture, Pilbara Blend around 8%. Ignoring the
correction systematically understates the premium's freight share.

It's the same move as `TC_per_t_zinc = TC_per_dmt_conc / (grade × recovery)`: the
quoted unit is not the economic unit.

ASSUMPTIONS (all explicit, all parameterised)
-----------------------------------------------
A-H1  The 65% CFR grade is mostly Brazilian, the 62% grade mostly Australian. The 62%
      grade actually also contains Brazilian and Indian ore -> the freight share is
      UNDER-estimated. Conservative bias, so in the right direction.
A-H2  Moisture: 9.0% Brazil, 8.0% Australia. Parameterised, swept in sensitivity.
A-H2b The C3/C5 freight is paid on the wet weight. True for a standard voyage charter
      on loaded weight; to double-check if a counterparty pays on a dry basis.
A-H3  The front-month FFA is a proxy for the spot route. A basis exists. It's shown as
      a data warning, never hidden.
A-H4  No use-value premium modelled (set to zero). The residual absorbs it.
A-H5  No carry cost on the ~25-day voyage-time gap. Quantified separately in the
      sensitivity section, not in the main identity.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from agri.data.snapshot import cached

# Default values for assumptions A-H2. Documented, never hard-coded anywhere else.
DEFAULT_MOISTURE_BRAZIL = 0.090
DEFAULT_MOISTURE_AUSTRALIA = 0.080


def freight_per_dry_tonne(freight_per_wet_tonne: pd.Series | float, moisture: float) -> pd.Series | float:
    """Converts freight paid on a wet tonne into a cost per dry tonne (A-H2b).

    moisture is a fraction (0.09 = 9%), not a percentage.
    """
    if not 0.0 <= moisture < 1.0:
        raise ValueError(f"moisture must be in [0, 1), got {moisture}")
    return freight_per_wet_tonne / (1.0 - moisture)


def decompose_premium(
    p65: pd.Series,
    p62: pd.Series,
    c3: pd.Series,
    c5: pd.Series,
    *,
    moisture_brazil: float = DEFAULT_MOISTURE_BRAZIL,
    moisture_australia: float = DEFAULT_MOISTURE_AUSTRALIA,
) -> pd.DataFrame:
    """Decomposes the 65-62 premium into a freight share and a residual.

    All series are indexed by date. Prices are in USD/dmt, freight in USD/wmt.
    Calendar alignment is done here by intersection: no forward-fill, a gap stays a
    gap (the data contract's rule).

    Returns a DataFrame with, per date:
        p65, p62, premium_observed,
        c3_wmt, c5_wmt, c3_dmt, c5_dmt,
        freight_fair_value, residual, freight_share,
        premium_naive_freight  (the differential uncorrected for moisture, to show
                                exactly what the correction changes)
    """
    aligned = pd.concat(
        {"p65": p65, "p62": p62, "c3_wmt": c3, "c5_wmt": c5}, axis=1
    ).dropna()
    if aligned.empty:
        raise ValueError(
            "no common date across the four series — check the calendars before "
            "going further, don't fill the gaps"
        )
    aligned = aligned.sort_index()

    out = aligned.copy()
    out["c3_dmt"] = freight_per_dry_tonne(aligned["c3_wmt"], moisture_brazil)
    out["c5_dmt"] = freight_per_dry_tonne(aligned["c5_wmt"], moisture_australia)
    out["premium_observed"] = aligned["p65"] - aligned["p62"]
    out["freight_fair_value"] = out["c3_dmt"] - out["c5_dmt"]
    out["residual"] = out["premium_observed"] - out["freight_fair_value"]
    out["premium_naive_freight"] = aligned["c3_wmt"] - aligned["c5_wmt"]

    # freight share of the premium: undefined when the premium is zero, and misleading
    # when it's negative -> left as NaN rather than producing an absurd percentage.
    share = out["freight_fair_value"] / out["premium_observed"]
    share[out["premium_observed"] <= 0] = np.nan
    out["freight_share"] = share

    return out


@dataclass(frozen=True)
class ExplainedVariance:
    """Result of the premium ~ a + b * fair_value_freight regression."""

    slope: float
    intercept: float
    r_squared: float
    correlation: float
    n_obs: int

    @property
    def summary(self) -> str:
        return (
            f"premium = {self.intercept:.2f} + {self.slope:.2f} × freight  "
            f"(R² = {self.r_squared:.3f}, rho = {self.correlation:.3f}, n = {self.n_obs})"
        )


def explained_variance(
    premium: pd.Series, freight_fair_value: pd.Series, *, on_changes: bool = False
) -> ExplainedVariance:
    """Share of the premium's variance explained by the freight share.

    on_changes=True regresses the changes rather than the levels. Preferable for the
    statistical reading: two non-stationary level series produce a flattering R² that
    means nothing. Both are exposed because the level speaks to a trader and the
    change speaks to an econometrician — and the gap between the two is itself
    information worth showing.
    """
    aligned = pd.concat({"y": premium, "x": freight_fair_value}, axis=1).dropna()
    if on_changes:
        aligned = aligned.diff().dropna()
    if len(aligned) < 3:
        raise ValueError(f"not enough observations to regress (n={len(aligned)})")

    x = aligned["x"].to_numpy(dtype=float)
    y = aligned["y"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    corr = float(np.corrcoef(x, y)[0, 1])
    return ExplainedVariance(
        slope=float(slope),
        intercept=float(intercept),
        r_squared=r_squared,
        correlation=corr,
        n_obs=len(aligned),
    )


def negative_residual_episodes(
    decomposition: pd.DataFrame, *, min_days: int = 5
) -> pd.DataFrame:
    """Episodes where the residual is negative: the high-grade premium is below the
    distance surcharge alone. In other words, quality is being sold for free or at a
    loss.

    This is the anomaly that carries the conversation. min_days filters out one-day
    noise.

    Returns a DataFrame: start, end, n_obs, residual_min, residual_mean.
    """
    flag = decomposition["residual"] < 0
    if not flag.any():
        return pd.DataFrame(
            columns=["start", "end", "n_obs", "residual_min", "residual_mean"]
        )

    # group consecutive True values
    group_id = (flag != flag.shift()).cumsum()
    rows = []
    for _, chunk in decomposition[flag].groupby(group_id[flag]):
        if len(chunk) < min_days:
            continue
        rows.append(
            {
                "start": chunk.index.min(),
                "end": chunk.index.max(),
                "n_obs": len(chunk),
                "residual_min": float(chunk["residual"].min()),
                "residual_mean": float(chunk["residual"].mean()),
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class HedgeResult:
    """Effect of hedging the freight leg on the premium trade's volatility."""

    beta: float
    vol_unhedged: float
    vol_hedged: float
    n_obs: int

    @property
    def vol_reduction_pct(self) -> float:
        if self.vol_unhedged == 0:
            return float("nan")
        return 100.0 * (1.0 - self.vol_hedged / self.vol_unhedged)


def freight_hedge_effect(
    premium: pd.Series,
    freight_fair_value: pd.Series,
    *,
    beta: float | None = None,
) -> HedgeResult:
    """Volatility of the premium's daily changes, before and after hedging the
    freight share with C3/C5 FFAs.

    beta=None estimates the hedge ratio by OLS on changes (the minimum-variance
    ratio). beta=1.0 forces a unit hedge, which is what a desk would do naively — the
    gap between the two is itself a result.

    What this calculation does NOT say: that the hedge is executable. C3/C5 FFAs are
    monthly average-of-route contracts; hedging a daily exposure with them leaves a
    non-trivial basis. To be shown as a caveat.
    """
    aligned = pd.concat({"prem": premium, "fret": freight_fair_value}, axis=1).dropna()
    changes = aligned.diff().dropna()
    if len(changes) < 3:
        raise ValueError(f"not enough observations (n={len(changes)})")

    if beta is None:
        var_f = float(changes["fret"].var())
        if var_f == 0:
            raise ValueError("the freight share doesn't vary — beta is undetermined")
        beta = float(changes[["prem", "fret"]].cov().iloc[0, 1] / var_f)

    residual_changes = changes["prem"] - beta * changes["fret"]
    return HedgeResult(
        beta=float(beta),
        vol_unhedged=float(changes["prem"].std()),
        vol_hedged=float(residual_changes.std()),
        n_obs=len(changes),
    )


def carry_cost_of_extra_voyage_days(
    cargo_value_per_dmt: pd.Series | float,
    extra_days: float,
    annual_rate: float,
) -> pd.Series | float:
    """A-H5, quantified: carry cost of Brazil's extra voyage days.

    Tubarão -> Qingdao against Port Hedland -> Qingdao is on the order of 25 extra
    days at sea. On a cargo at 100 $/dmt and 6% annual, that's ~0.41 $/dmt: small
    against a freight differential of 10-14 $/dmt, but not zero against a residual of
    2-3 $/dmt. Showing it means refusing to pretend the term doesn't exist.
    """
    if extra_days < 0:
        raise ValueError("extra_days must be positive")
    return cargo_value_per_dmt * annual_rate * (extra_days / 365.0)


# ===========================================================================
# REAL DATA — the shorthand tested, and found arithmetically impossible
# ===========================================================================
# The whole freight-spread story rests on a shorthand: 65 % Fe is Brazilian and travels on
# route C3, 62 % Fe is Australian and travels on C5, so the C3 - C5 differential sits inside
# the CFR premium. The export now carries all four series. Testing the shorthand at full
# strength turns out to break it — and the moisture correction, which is this project's
# signature unit trap, makes the contradiction worse rather than resolving it.
IODEX_62_SHEET = "SGX IODEX (61%) Iron Ore Future"
MB_65_SHEET = "SGX MB IronOre 65 Sep26 Comdty"
C3_SHEET = "SGX Baltic C3 Futures FSP Index"
C5_SHEET = "C5 FFA USD MT M1 Index"


def _read_sheet(sheet: str) -> pd.Series:
    """Read one raw sheet of the Bloomberg export.

    These four series are not in `agri.data.bloomberg_loader` because they belong to the
    freight side of the portfolio, and because two of them carry defects the loader's
    contract would have to describe one by one — see `load_real_premium_frame`.
    """
    from agri.data.bloomberg_loader import DEFAULT_PATH

    raw = pd.read_excel(DEFAULT_PATH, sheet_name=sheet, header=None)
    values = pd.to_numeric(raw.iloc[:, 1], errors="coerce")
    dates = pd.to_datetime(raw.iloc[:, 0], errors="coerce", format="mixed")
    return pd.Series(values.values, index=dates).dropna().sort_index()


@cached('a_iron_ore')
def load_real_premium_frame() -> pd.DataFrame:
    """The four legs on a common **monthly** grid, and the reason it has to be monthly.

    Columns: io65, io62, c3, c5, premium, freight_spread.

    FREQUENCY, AND WHY IT IS NOT A DETAIL. C5 is daily with 3 130 observations; the C3 route
    in this export has **64 points at a monthly step**. Forward-filling C3 onto a daily grid
    would create 600 observations out of 31 real ones and make every standard error in the
    page meaningless — the portfolio's own resampling rules forbid exactly that. So both are
    taken down to the monthly grid the coarser series actually supports. Thirty-one months
    is a small sample and the page says so rather than manufacturing a large one.
    """
    monthly = {
        name: _read_sheet(sheet).resample("ME").median()
        for name, sheet in (
            ("io65", MB_65_SHEET),
            ("io62", IODEX_62_SHEET),
            ("c3", C3_SHEET),
            ("c5", C5_SHEET),
        )
    }
    frame = pd.concat(monthly, axis=1, sort=True).dropna()
    if frame.empty:
        raise ValueError("no common month across the four iron ore legs")
    frame["premium"] = frame["io65"] - frame["io62"]
    frame["freight_spread"] = frame["c3"] - frame["c5"]
    return frame


@dataclass(frozen=True)
class ShorthandTest:
    """What the origin shorthand implies about the FOB quality differential.

    Both indices are quoted CFR China, so

        premium_CFR = (FOB_65 - FOB_62) + (C3 - C5)

    Taken at full strength, the shorthand therefore pins the FOB quality differential to
    `premium - freight_spread`. If that comes out negative, high-grade ore would be cheaper
    at the loadport than low-grade ore — which nobody believes, and which is the point.
    """

    frame: pd.DataFrame
    moisture: float

    @property
    def implied_quality_median(self) -> float:
        return float(self.frame["implied_quality"].median())

    @property
    def share_negative(self) -> float:
        return float((self.frame["implied_quality"] < 0).mean())

    @property
    def freight_share_of_premium(self) -> float:
        return float(
            self.frame["freight_dry"].median() / self.frame["premium"].median()
        )

    @property
    def shorthand_survives(self) -> bool:
        """The shorthand only survives if it implies a non-negative quality differential
        most of the time."""
        return self.share_negative < 0.5

    @property
    def headline(self) -> str:
        return (
            f"At {self.moisture:.0%} moisture, the freight differential covers "
            f"{self.freight_share_of_premium:.0%} of the 65-62 premium, which leaves an "
            f"implied FOB quality differential of {self.implied_quality_median:+.2f} USD/t "
            f"— negative in {self.share_negative:.0%} of months. Taken at full strength the "
            "origin shorthand would make high-grade ore cheaper at the loadport than "
            "low-grade ore. It cannot be right as stated."
        )


def evaluate_origin_shorthand(
    frame: pd.DataFrame, *, moisture: float = DEFAULT_MOISTURE_BRAZIL
) -> ShorthandTest:
    """Apply the shorthand at full strength and look at what it implies.

    The moisture correction is applied to the freight leg and **only** to the freight leg:
    freight is paid on the wet tonne shipped, the index is quoted on the dry tonne
    delivered, so one dry tonne costs `freight / (1 - moisture)` to move. This is the
    correction the project exists to make — and here it deepens the contradiction instead of
    resolving it, which is worth more than a correction that had tidied things up.
    """
    if not 0.0 <= moisture < 0.30:
        raise ValueError(f"moisture outside the plausible range [0, 0.30): {moisture}")

    out = frame.copy()
    out["freight_dry"] = out["freight_spread"] / (1.0 - moisture)
    out["implied_quality"] = out["premium"] - out["freight_dry"]
    return ShorthandTest(frame=out, moisture=float(moisture))


@dataclass(frozen=True)
class ImpliedOriginWeight:
    """The inversion: how much of the freight spread the premium can actually carry.

    Rather than asserting the shorthand and finding a contradiction, assume the FOB quality
    differential a practitioner believes in and solve for the weight at which the freight
    spread enters:

        premium = quality_FOB + w x freight_dry     =>     w = (premium - quality) / freight_dry

    `w = 1` is the full-strength shorthand. `w = 0` is a pure quality premium with no freight
    content. What comes out is neither — and an iron ore desk reads its own answer straight
    off the curve, because it knows its FOB differential.
    """

    curve: pd.DataFrame
    moisture: float

    def weight_at(self, quality_usd_t: float) -> float:
        row = self.curve.iloc[(self.curve["quality_usd_t"] - quality_usd_t).abs().argmin()]
        return float(row["implied_weight"])

    @property
    def headline(self) -> str:
        return (
            f"Assuming a 6 USD/t FOB quality differential, the freight spread enters the "
            f"premium at a weight of {self.weight_at(6.0):.2f} rather than 1 — the two "
            "indices are not single-origin, and the shorthand overstates the freight content "
            "by roughly a factor of two."
        )


def implied_origin_weight(
    frame: pd.DataFrame,
    *,
    moisture: float = DEFAULT_MOISTURE_BRAZIL,
    quality_grid: np.ndarray | None = None,
) -> ImpliedOriginWeight:
    """The weight at which the freight spread enters, as a function of assumed FOB quality."""
    tested = evaluate_origin_shorthand(frame, moisture=moisture)
    grid = np.arange(0.0, 14.1, 0.5) if quality_grid is None else np.asarray(quality_grid)

    rows = []
    for quality in grid:
        weight = (tested.frame["premium"] - quality) / tested.frame["freight_dry"]
        rows.append(
            {
                "quality_usd_t": float(quality),
                "implied_weight": float(weight.median()),
                "weight_low": float(weight.quantile(0.10)),
                "weight_high": float(weight.quantile(0.90)),
            }
        )
    return ImpliedOriginWeight(curve=pd.DataFrame(rows), moisture=float(moisture))


# ===========================================================================
# THE THIRD ORIGIN — what Simandou does to a two-origin decomposition
# ===========================================================================
# Everything above assumes a two-origin world: 65 % Fe travels on C3 (Brazil), 62 % Fe on
# C5 (Australia), and the premium carries the difference at some weight. That world ended
# in January 2026, when the first Simandou cargo reached China.
#
# The distances below are not looked up in a routing table — no Guinea-China Capesize
# benchmark exists yet, which is itself the point. They are DERIVED from two published
# facts and cross-checked against the two distances the portfolio already uses:
#
#   Drewry: the Guinea-China round voyage runs beyond 90 days, against 30-35 days for
#   Australia-China, and the Guinea distance is roughly three times the Western Australian
#   one. At 12 knots with ~8 round-trip port days:
#       Australia:  (32.5 - 8) x 12 x 24 / 2 = ~3 530 nm   vs NAMED_ROUTES' 3 500  [check]
#       Guinea:     (90.0 - 8) x 12 x 24 / 2 = ~11 800 nm
#   and 3 500 x 3 = 10 500 nm, the same order. Two independent published statements agree
#   with each other and one of them reproduces a distance already in the repository.
#
# THE CONSEQUENCE, WHICH IS NOT THE OBVIOUS ONE. Simandou is not cheap high-grade ore. At
# ~11 800 nm it sits at or beyond the Brazilian haul (11 000 nm), so it arrives in China
# carrying a freight bill comparable to Vale's, not a smaller one. What it does do is add
# an enormous amount of tonne-mileage to a fleet that has to serve it: roughly 180
# Capesizes to move 120 Mtpa from Guinea against 64 for the same tonnage from Australia.
GUINEA_QINGDAO_NM = 11_800.0        # derived above, not a published benchmark
BRAZIL_QINGDAO_NM = 11_000.0        # NAMED_ROUTES C3 laden leg
AUSTRALIA_QINGDAO_NM = 3_500.0      # NAMED_ROUTES C5 laden leg


@dataclass(frozen=True)
class PremiumAttribution:
    """Split a move in the observed premium into a freight part and a residual part.

    The identity is the same one the whole page runs on, applied to a *change* rather than
    a level:

        d(premium) = d(quality_FOB) + w x d(freight_dry)

    so the residual — everything not explained by freight at weight `w` — is what the
    underlying quality differential must have done. The weight is an assumption, so the
    honest object is not a single number but `threshold_weight`: the weight at which the
    residual changes sign. Where that threshold sits relative to the plausible range for
    `w` is what decides whether the answer is knowable at all.
    """

    label_from: str
    label_to: str
    premium_change: float
    freight_change: float
    weight: float

    @property
    def freight_part(self) -> float:
        return self.weight * self.freight_change

    @property
    def residual_part(self) -> float:
        return self.premium_change - self.freight_part

    @property
    def threshold_weight(self) -> float:
        """The weight at which the residual flips sign. Undefined if freight did not move."""
        if self.freight_change == 0:
            return float("nan")
        return self.premium_change / self.freight_change

    @property
    def quality_fell(self) -> bool:
        return self.residual_part < 0

    @property
    def headline(self) -> str:
        direction = "fell" if self.quality_fell else "rose"
        return (
            f"From {self.label_from} to {self.label_to} the premium rose "
            f"{self.premium_change:+.2f} USD/t while the moisture-corrected freight spread "
            f"rose {self.freight_change:+.2f}. At a weight of {self.weight:.2f}, freight "
            f"accounts for {self.freight_part:+.2f} of the move and the underlying quality "
            f"differential {direction} by {abs(self.residual_part):.2f}. The sign of that "
            f"residual flips at a weight of {self.threshold_weight:.3f} — so the answer "
            "depends on a parameter this page can only bound, not measure."
        )


def premium_attribution(
    frame: pd.DataFrame,
    *,
    weight: float,
    year_from: int,
    year_to: int,
    moisture: float = DEFAULT_MOISTURE_BRAZIL,
) -> PremiumAttribution:
    """Attribute the change in the premium between two years to freight versus quality.

    Annual means rather than endpoints: with a monthly series of this length, two endpoint
    observations would be a coin flip dressed as a measurement.
    """
    if not 0.0 <= weight <= 1.5:
        raise ValueError(f"weight outside the plausible range [0, 1.5]: {weight}")

    working = frame.copy()
    working["freight_dry"] = working["freight_spread"] / (1.0 - moisture)
    a = working[working.index.year == year_from]
    b = working[working.index.year == year_to]
    if a.empty or b.empty:
        raise ValueError(f"no observations for {year_from} or {year_to}")

    return PremiumAttribution(
        label_from=str(year_from),
        label_to=str(year_to),
        premium_change=float(b["premium"].mean() - a["premium"].mean()),
        freight_change=float(b["freight_dry"].mean() - a["freight_dry"].mean()),
        weight=float(weight),
    )


@dataclass(frozen=True)
class ThirdOriginArithmetic:
    """Simandou's two opposing effects on the same observed number.

    Both follow from one fact — Guinea is a Brazil-length haul, not an Australia-length one
    — and they push the 65-62 premium in opposite directions:

      * **supply / quality**: more 65 % Fe ore competing for the same buyers compresses the
        FOB quality differential, which pushes the premium DOWN;
      * **freight**: serving that tonnage consumes far more Capesize capacity per tonne than
        Australian ore does, which tightens the fleet, widens C3 - C5, and — at any weight
        above zero — pushes the observed premium UP.

    A desk watching only the headline premium sees the net of the two and cannot tell them
    apart. That is what the decomposition is for, and it is the reason this page exists
    beyond the arithmetic curiosity of its own weight estimate.
    """

    guinea_nm: float
    brazil_nm: float
    australia_nm: float

    @property
    def guinea_vs_australia(self) -> float:
        return self.guinea_nm / self.australia_nm

    @property
    def guinea_vs_brazil(self) -> float:
        return self.guinea_nm / self.brazil_nm

    @property
    def is_long_haul_like_brazil(self) -> bool:
        """Guinea within 25 % of the Brazilian haul — i.e. not a cheaper source of 65 % Fe."""
        return abs(self.guinea_vs_brazil - 1.0) < 0.25

    @property
    def headline(self) -> str:
        return (
            f"Guinea to Qingdao is about {self.guinea_nm:,.0f} nm against "
            f"{self.brazil_nm:,.0f} for Brazil and {self.australia_nm:,.0f} for Australia — "
            f"{self.guinea_vs_australia:.1f} times the Australian haul and "
            f"{self.guinea_vs_brazil:.2f} times the Brazilian one. Simandou is not cheap "
            "high-grade ore: it lands in China carrying a Brazil-sized freight bill, while "
            "consuming far more fleet capacity per tonne than the Australian ore it displaces."
        )


def third_origin_arithmetic(
    *,
    guinea_nm: float = GUINEA_QINGDAO_NM,
    brazil_nm: float = BRAZIL_QINGDAO_NM,
    australia_nm: float = AUSTRALIA_QINGDAO_NM,
) -> ThirdOriginArithmetic:
    """The distance comparison the two-origin decomposition does not have a slot for."""
    for value, label in ((guinea_nm, "guinea"), (brazil_nm, "brazil"), (australia_nm, "australia")):
        if value <= 0:
            raise ValueError(f"{label} distance must be > 0 nm, got {value}")
    return ThirdOriginArithmetic(
        guinea_nm=float(guinea_nm),
        brazil_nm=float(brazil_nm),
        australia_nm=float(australia_nm),
    )
