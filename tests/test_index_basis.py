"""Golden tests for project H — index-vs-route basis, tested where it matters most.

Three tests carry this page and none of them may be weakened by a later rework.

`test_the_naive_decoupling_fear_is_rejected` is the first result: the index does not
track the route worse in the tails, it tracks it better.

`test_the_residual_scales_with_the_raw_move_not_with_r_squared` is the deeper result
underneath it — the one that keeps the first result from being read as good news.

`test_the_thinnest_slice_is_not_read_as_a_verdict` is the honesty guard on H-H3.
"""
from __future__ import annotations

import pandas as pd
import pytest

from agri.data.bloomberg_loader import DEFAULT_PATH
from freight.chains.index_basis import (
    MIN_OBS_FOR_VERDICT,
    IndexBasisError,
    absolute_risk_scaling,
    load_index_basis_frame,
    tail_decomposition,
)

pytestmark = pytest.mark.skipif(
    not DEFAULT_PATH.exists(), reason=f"Bloomberg file absent: {DEFAULT_PATH}"
)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return load_index_basis_frame()


# ===========================================================================
# THE FIRST RESULT — the naive fear is wrong
# ===========================================================================
def test_the_full_sample_relationship_is_weak_and_not_significant(frame):
    """Before any split: on a typical day, BPI barely explains the P8 route at all."""
    d = tail_decomposition(frame)
    assert d.full.r_squared < 0.05
    assert not d.full.significant


def test_the_naive_decoupling_fear_is_rejected(frame):
    """The index tracks the route BETTER, not worse, on the route's biggest-move days
    — the opposite of the intuitive fear this project sets out to test."""
    d = tail_decomposition(frame)
    assert d.r_squared_rises_with_move_size
    assert d.top_decile.r_squared > 3 * d.full.r_squared
    assert d.top_decile.significant
    assert not d.calm.significant
    assert "better, not worse" in d.headline


# ===========================================================================
# THE DEEPER RESULT — R² is the wrong unit for a hedger
# ===========================================================================
def test_the_residual_scales_with_the_raw_move_not_with_r_squared(frame):
    """The correction to the first result: a bigger R² in the tail does not mean better
    absolute protection, because the unexplained dollars grow right along with the
    move itself."""
    scaling = absolute_risk_scaling(frame)
    assert scaling.resid_scaling > 20.0
    assert scaling.hedge_gets_proportionally_no_better
    assert "does not protect any more" in scaling.headline


def test_calm_and_extreme_residuals_are_both_locked(frame):
    """Golden values — a future data refresh changing these substantially should be
    caught here rather than silently changing the page's headline numbers."""
    scaling = absolute_risk_scaling(frame)
    assert scaling.calm_resid_std == pytest.approx(0.10, abs=0.03)
    assert scaling.extreme_resid_std == pytest.approx(3.9, abs=0.5)


# ===========================================================================
# THE HONESTY GUARD — H-H3
# ===========================================================================
def test_the_thinnest_slice_is_not_read_as_a_verdict(frame):
    """A finer split than the top decile (e.g. the 90th-95th percentile alone, ~32 obs)
    would fall under MIN_OBS_FOR_VERDICT and must not be exposed as a bucket with a
    verdict attached — the coarser top-decile bucket is the finest one reported."""
    changes = frame.diff().dropna()
    abs_move = changes["p8"].abs()
    q90, q95 = abs_move.quantile([0.90, 0.95])
    thinnest = changes[(abs_move > q90) & (abs_move <= q95)]
    assert len(thinnest) < MIN_OBS_FOR_VERDICT


def test_every_reported_bucket_meets_the_power_threshold(frame):
    d = tail_decomposition(frame)
    for bucket in (d.full, d.calm, d.top_half, d.top_quarter, d.top_decile):
        assert bucket.has_power, f"{bucket.label} has only {bucket.n_obs} obs"


# ===========================================================================
# Method discipline
# ===========================================================================
def test_buckets_are_built_on_changes_not_levels(frame):
    """H-H1. A levels regression on two trending series would describe the shared trend,
    not co-movement — the check is that using levels gives a very different R²."""
    from agri.core.stats import hac_ols

    on_levels = hac_ols(frame["p8"], frame[["bpi"]]).r_squared
    on_changes = tail_decomposition(frame).full.r_squared
    assert abs(on_levels - on_changes) > 0.1


# ===========================================================================
# Guardrails
# ===========================================================================
def test_an_impossible_start_date_raises():
    with pytest.raises(IndexBasisError, match="no common dates"):
        load_index_basis_frame("2099-01-01")
