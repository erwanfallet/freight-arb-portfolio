"""Project H — the index tracks the route worst on the days that matter least.

THESIS
------
A Panamax charterer hedging the specific Santos-Qingdao route with a BPI-linked FFA is
implicitly betting that the global index moves with the route closely enough, often
enough, to be worth the basis it accepts. The natural fear is that the index decouples
exactly when a hedge is needed most — on the route's biggest moves.

MEASURED ON THE REAL BPI AND P8 ROUTE RATE, 624 OVERLAPPING DAYS (2021-2022, 2025-2026):
that fear is the wrong one. Split by the size of the route's own daily move, the index
explains **more** of the route on the biggest-move days (R² = 10.4%, p = 0.011), not
less — against next to nothing on calm days (R² = 0.3%, p = 0.183). The naive "it
decouples in the tails" story is rejected by the data it was meant to explain.

THE STEP THAT MATTERS IS THE ONE UNDERNEATH IT
---------------------------------------------------
A higher R² does not mean a better hedge in the units that matter to a book. What R²
measures is a *share* of variance; what a hedger carries is *dollars* of unexplained
move. The residual left over after regressing the route on the index is **0.10 USD/t on
calm days and 3.92 USD/t on the biggest-move days — a 39x increase**, almost exactly in
line with the 42x increase in the route's own raw move size over the same split. The
index explaining a larger *share* of a much larger number still leaves a much larger
*absolute* number unexplained. Statistically the hedge gets better in the tails;
practically, in the units a P&L is kept in, it does not protect any more than it does on
a quiet day.

WHY THIS IS BUILT AS ITS OWN PROJECT, NOT AN EXTENSION OF D
------------------------------------------------------------------
Project D already states that the BPI is "too global to show a single-basin flow" as a
limit on what it can prove. This project measures that limit directly, on daily
frequency, as a tracking-error question rather than a seasonality one, and reaches a
different desk: not a chartering desk fixing physical cargo, but a derivatives desk
sizing how much of its route exposure an index-linked hedge actually removes.

ASSUMPTIONS
-----------
H-H1  Both series are daily and tested on changes, never levels (Rule A/B) — a levels
      regression on two trending series would describe a shared trend, not co-movement.
H-H2  Buckets are defined by quantiles of |change in P8|, computed from the same sample
      being split — this is a description of realized history, not a forecast, and is
      stated as such.
H-H3  A verdict requires at least 60 observations in a bucket (matching project E's
      threshold) — the finest slice tried (90th-95th percentile of moves, n=32) shows a
      striking R² but is too thin to trust and is reported without a verdict drawn.
H-H4  Inherits the P8 route's 2023-2024 gap (documented in project D) — this test runs
      on the 2021-2022 and 2025-2026 windows only.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from agri.core.stats import hac_ols
from agri.data.snapshot import cached

MIN_OBS_FOR_VERDICT = 60  # H-H3, matching project E's threshold


class IndexBasisError(ValueError):
    """Mis-specified basis test — always a caller error."""


# ===========================================================================
# Data
# ===========================================================================
@cached("h_index_basis")
def load_index_basis_frame(start: str | None = None) -> pd.DataFrame:
    """Real BPI and real P8 route rate on their common calendar. Columns: bpi, p8."""
    from agri.data.bloomberg_loader import load

    frame = pd.concat({"bpi": load("bpi"), "p8": load("p8_route_usd_t")}, axis=1, sort=True).dropna()
    if start is not None:
        frame = frame[frame.index >= pd.Timestamp(start)]
    if frame.empty:
        raise IndexBasisError(f"no common dates between BPI and the P8 route rate after {start}")
    return frame


# ===========================================================================
# One regression, always on changes (H-H1)
# ===========================================================================
@dataclass(frozen=True)
class BucketResult:
    label: str
    n_obs: int
    beta: float
    r_squared: float
    pvalue: float
    raw_std: float
    resid_std: float

    @property
    def has_power(self) -> bool:
        return self.n_obs >= MIN_OBS_FOR_VERDICT

    @property
    def significant(self) -> bool:
        return self.has_power and self.pvalue < 0.05


def _bucket_regression(label: str, changes: pd.DataFrame) -> BucketResult:
    regression = hac_ols(changes["p8"], changes[["bpi"]])
    fitted = regression.params["const"] + regression.params["bpi"] * changes["bpi"]
    residual = changes["p8"] - fitted
    return BucketResult(
        label=label,
        n_obs=int(len(changes)),
        beta=float(regression.params["bpi"]),
        r_squared=float(regression.r_squared),
        pvalue=float(regression.pvalues["bpi"]),
        raw_std=float(changes["p8"].std()),
        resid_std=float(residual.std()),
    )


# ===========================================================================
# THE TAIL SPLIT
# ===========================================================================
@dataclass(frozen=True)
class TailDecomposition:
    full: BucketResult
    calm: BucketResult
    top_half: BucketResult
    top_quarter: BucketResult
    top_decile: BucketResult

    @property
    def r_squared_rises_with_move_size(self) -> bool:
        return (
            self.calm.r_squared
            < self.top_half.r_squared
            < self.top_quarter.r_squared
            < self.top_decile.r_squared
        )

    @property
    def headline(self) -> str:
        return (
            f"Full sample: R²={self.full.r_squared:.1%} (p={self.full.pvalue:.2f}, not "
            f"significant). Split by the size of the route's own move: R² rises from "
            f"{self.calm.r_squared:.1%} on calm days to {self.top_decile.r_squared:.1%} "
            f"on the biggest-move decile ({'significant, p=' + f'{self.top_decile.pvalue:.3f}' if self.top_decile.significant else 'not significant'})"
            " — the index tracks the route better, not worse, when the route moves hardest."
        )


def tail_decomposition(frame: pd.DataFrame) -> TailDecomposition:
    """Split the route's daily changes by the size of its own move (H-H2) and regress
    the route on the index in each bucket. Buckets are nested (top_decile subset of
    top_quarter subset of top_half) so the pattern can be read as a single gradient."""
    changes = frame.diff().dropna()
    abs_move = changes["p8"].abs()
    q50, q75, q90 = abs_move.quantile([0.5, 0.75, 0.9])

    return TailDecomposition(
        full=_bucket_regression("full sample", changes),
        calm=_bucket_regression("calm (bottom 50%)", changes[abs_move <= q50]),
        top_half=_bucket_regression("top 50%", changes[abs_move > q50]),
        top_quarter=_bucket_regression("top 25%", changes[abs_move > q75]),
        top_decile=_bucket_regression("top 10%", changes[abs_move > q90]),
    )


# ===========================================================================
# THE CORRECTION — absolute risk, not R²
# ===========================================================================
@dataclass(frozen=True)
class AbsoluteRiskScaling:
    calm_resid_std: float
    extreme_resid_std: float
    calm_raw_std: float
    extreme_raw_std: float

    @property
    def resid_scaling(self) -> float:
        return self.extreme_resid_std / self.calm_resid_std

    @property
    def raw_scaling(self) -> float:
        return self.extreme_raw_std / self.calm_raw_std

    @property
    def hedge_gets_proportionally_no_better(self) -> bool:
        """True when the unexplained residual scales up in line with the raw move —
        i.e. the higher tail R² does not translate into better absolute protection."""
        ratio = self.resid_scaling / self.raw_scaling
        return 0.7 <= ratio <= 1.3

    @property
    def headline(self) -> str:
        return (
            f"The unexplained residual grows {self.resid_scaling:.0f}x from calm to "
            f"extreme days, almost exactly matching the {self.raw_scaling:.0f}x growth "
            "in the route's own raw move. The index explains a larger share of a much "
            "larger number — in the dollars a book actually carries, it does not "
            "protect any more than on a quiet day."
        )


def absolute_risk_scaling(frame: pd.DataFrame) -> AbsoluteRiskScaling:
    decomposition = tail_decomposition(frame)
    return AbsoluteRiskScaling(
        calm_resid_std=decomposition.calm.resid_std,
        extreme_resid_std=decomposition.top_decile.resid_std,
        calm_raw_std=decomposition.calm.raw_std,
        extreme_raw_std=decomposition.top_decile.raw_std,
    )
