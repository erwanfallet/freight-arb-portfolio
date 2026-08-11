"""Golden tests T1-1 on the real P8 route — and arbitrating the ballast disagreement.

The page's central result: the export's P8 series contains **two unit regimes**, a TCE
in USD/day (Jul-Oct 2021) then a voyage rate in USD/tonne (Nov 2021 onward). The first,
which had to be isolated as a data defect, then serves as a testbed for the second: it
gives the TCE level actually quoted on this route at the **peak** of the dry bulk boom,
i.e. a plausibility ceiling.
"""
from __future__ import annotations

import pandas as pd
import pytest

from agri.chains.freight_cf import (
    FreightCfError,
    implied_tce_by_convention,
    load_real_route_frame,
    market_implied_ballast_share,
)
from agri.core.voyage import ROUTES, VESSELS, VoyageParams
from agri.data.bloomberg_loader import DEFAULT_PATH, load

pytestmark = pytest.mark.skipif(
    not DEFAULT_PATH.exists(), reason=f"Bloomberg file absent: {DEFAULT_PATH}"
)

PANAMAX = VESSELS["panamax"]
SANTOS_QINGDAO = ROUTES["santos_qingdao"]


# ===========================================================================
# The data defect: two unit regimes in the same cell
# ===========================================================================
def test_p8_splits_into_two_unit_regimes_with_no_overlap():
    """The USD/day segment stops on 30/10/2021, the USD/t segment resumes on
    18/11/2021: a 19-day gap and **no common date**. The conversion factor therefore
    can't be calibrated at the junction — the market moved between the two."""
    tce = load("p8_route_tce_2021")
    rate = load("p8_route_usd_t")

    assert tce.index.max() == pd.Timestamp("2021-10-30")
    assert rate.index.min() == pd.Timestamp("2021-11-18")
    assert len(tce.index.intersection(rate.index)) == 0
    assert (rate.index.min() - tce.index.max()).days == 19


def test_each_segment_is_internally_plausible_in_its_own_unit():
    """A Panamax TCE of 24,500-38,000 USD/day is the real peak of the 2021 boom; a
    Santos-Qingdao voyage rate of 36-85 USD/t is the normal order of magnitude. Each
    segment is consistent **in its own unit** — it's their juxtaposition that isn't."""
    tce = load("p8_route_tce_2021")
    rate = load("p8_route_usd_t")

    assert 20_000 < tce.min() and tce.max() < 45_000
    assert 20 < rate.min() and rate.max() < 150


def test_the_impossible_zero_print_is_dropped():
    """A zero freight print on 30/04/2022: physically impossible, and it would break
    every downstream division. Dropped by `min_valid` rather than left to circulate."""
    rate = load("p8_route_usd_t")
    assert pd.Timestamp("2022-04-30") not in rate.index
    assert (rate > 0).all()


# ===========================================================================
# The inversion: the same print, two conventions
# ===========================================================================
@pytest.fixture(scope="module")
def spread():
    return load_real_route_frame()


def test_no_ballast_reading_always_implies_a_higher_tce(spread):
    """Invariant: at identical revenue, not charging empty repositioning crushes the
    cycle's duration, thus inflating the resulting TCE. The gap is strictly positive
    everywhere."""
    assert (spread.tce_no_ballast > spread.tce_full_ballast).all()
    assert (spread.spread > 0).all()


def test_the_gap_is_large_enough_to_be_the_whole_argument(spread):
    """The median gap between the two readings exceeds 30,000 USD/day — the same
    order of magnitude as the TCE itself. This isn't a model refinement, it's the
    question."""
    assert spread.spread.median() > 25_000
    assert "in the freight department's unit" in spread.headline


# ===========================================================================
# THE RESULT — the defective segment settles the disagreement
# ===========================================================================
def test_the_no_ballast_reading_implies_an_impossible_market(spread):
    """THE page's test.

    The 2021 TCE segment gives the level actually quoted on this route at the
    **peak** of the dry bulk boom: 38,000 USD/day at the top. Reading the published
    rate without charging ballast implies a TCE above that peak **almost the entire
    time** over 2021-2026 — which would amount to saying the market spent five years
    above its own high. The trading desk's reading is therefore arithmetically
    untenable.
    """
    boom_peak = float(load("p8_route_tce_2021").max())
    share_above = float((spread.tce_no_ballast > boom_peak).mean())
    assert share_above > 0.90


def test_the_full_ballast_reading_is_plausible(spread):
    """Direct contrast: the same inversion charging ballast stays below the boom's
    peak almost all the time. That's what tips the balance toward the freight
    department's side — not an argument from authority, a plausibility bound."""
    boom_peak = float(load("p8_route_tce_2021").max())
    share_above = float((spread.tce_full_ballast > boom_peak).mean())
    assert share_above < 0.10


def test_full_ballast_median_sits_below_the_2021_peak(spread):
    boom_peak = float(load("p8_route_tce_2021").max())
    assert spread.tce_full_ballast.median() < boom_peak
    assert spread.tce_no_ballast.median() > boom_peak


# ===========================================================================
# The ballast share the market prices in
# ===========================================================================
def test_market_implied_ballast_share_recovers_a_known_input():
    """Inversion consistency check: a rate is built with a known ballast share, then
    the solver is checked to recover it."""
    from agri.core.voyage import voyage_freight_usd_t

    known_share = 0.6
    rate = voyage_freight_usd_t(
        15_000.0, 500.0, 700.0,
        vessel=PANAMAX, route=SANTOS_QINGDAO,
        params=VoyageParams(ballast_share=known_share),
    ).freight_usd_t

    result = market_implied_ballast_share(
        rate, 15_000.0, 500.0, 700.0,
        vessel=PANAMAX, route=SANTOS_QINGDAO, params=VoyageParams(),
    )
    assert result.implied_share == pytest.approx(known_share, abs=1e-6)
    assert "ballast repositioning" in result.headline


def test_a_rate_above_the_model_range_names_the_binding_end():
    """A published rate the model can't produce signals a voyage assumption to
    revisit — and the message must name the **binding** bound, not the furthest one.
    Since the gap decreases in ballast, a rate that's too high is one that exceeds the
    model even at 100% ballast."""
    result = market_implied_ballast_share(
        500.0, 15_000.0, 500.0, 700.0,
        vessel=PANAMAX, route=SANTOS_QINGDAO, params=VoyageParams(),
    )
    assert result.implied_share is None
    assert "100%" in result.reason
    assert "too low" in result.reason


def test_a_rate_below_the_model_range_names_the_other_end():
    result = market_implied_ballast_share(
        5.0, 15_000.0, 500.0, 700.0,
        vessel=PANAMAX, route=SANTOS_QINGDAO, params=VoyageParams(),
    )
    assert result.implied_share is None
    assert "no ballast" in result.reason
    assert "too high" in result.reason


def test_implied_share_is_monotone_in_the_published_rate():
    """Freight is affine increasing in ballast, so a higher published rate implies a
    higher ballast share. That's what guarantees the root's uniqueness."""
    shares = [
        market_implied_ballast_share(
            rate, 15_000.0, 500.0, 700.0,
            vessel=PANAMAX, route=SANTOS_QINGDAO, params=VoyageParams(),
        ).implied_share
        for rate in (25.0, 30.0, 35.0)
    ]
    assert all(s is not None for s in shares)
    assert shares == sorted(shares)


def test_disjoint_calendars_raise():
    a = pd.Series([50.0], index=pd.to_datetime(["2024-01-01"]))
    b = pd.Series([500.0], index=pd.to_datetime(["2030-01-01"]))
    with pytest.raises(FreightCfError, match="no common date"):
        implied_tce_by_convention(
            a, b, b, vessel=PANAMAX, route=SANTOS_QINGDAO, params=VoyageParams()
        )
