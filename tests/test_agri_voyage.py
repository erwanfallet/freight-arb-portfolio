"""Golden tests for the voyage model.

The reference voyage, used in almost every test below:
Panamax 66 kt on Santos -> Qingdao, TCE 15,000 $/day, VLSFO 500 $/t, MGO 700 $/t,
default parameters (100% ballast, 12.5 kn laden, 13.0 kn on ballast, 6 days of port,
3.75% commission).

    laden_days    = 11,000 / (12.5 x 24)       = 36.666667
    ballast_days  = 11,000 / (13.0 x 24)       = 35.256410
    total_days    = 36.666667 + 35.256410 + 6  = 77.923077

    ballast consumption = 24 x (13.0/12.5)^3 = 24 x 1.124864 = 26.996736 t/day
    sea bunkers   = 500 x (26 x 36.666667 + 26.996736 x 35.256410)
                  = 500 x 1,905.141333 = 952,570.67 $
    port bunkers  = 700 x 3.0 x 6 = 12,600 $
    hire          = 15,000 x 77.923077 = 1,168,846.15 $
    total cost    = 1,168,846.15 + 952,570.67 + 12,600 + 180,000 = 2,314,016.82 $
    payable cargo = 66,000 x (1 - 0.0375) = 63,525 t
    freight       = 2,314,016.82 / 63,525 = 36.426868 $/t
"""
from __future__ import annotations

import pandas as pd
import pytest

from agri.core.voyage import (
    ROUTES,
    VESSELS,
    VoyageError,
    VoyageParams,
    implied_tce_from_freight,
    plausibility_warnings,
    voyage_freight_series,
    voyage_freight_usd_t,
)

PANAMAX = VESSELS["panamax"]
SANTOS_QINGDAO = ROUTES["santos_qingdao"]


def _reference(**overrides):
    params = VoyageParams(**overrides)
    return voyage_freight_usd_t(
        15_000.0, 500.0, 700.0, vessel=PANAMAX, route=SANTOS_QINGDAO, params=params
    )


# ===========================================================================
# The reference voyage, term by term
# ===========================================================================
def test_laden_days():
    assert _reference().laden_days == pytest.approx(36.666667, abs=1e-6)


def test_ballast_days_at_full_ballast():
    assert _reference().ballast_days == pytest.approx(35.256410, abs=1e-6)


def test_total_days():
    assert _reference().total_days == pytest.approx(77.923077, abs=1e-6)


def test_hire_cost():
    assert _reference().hire_usd == pytest.approx(1_168_846.153846, abs=1e-4)


def test_sea_bunker_cost_uses_cube_law_on_the_ballast_leg():
    # base 24 t/day at 12.5 kn, but ballast runs at 13.0 -> 24 x 1.124864 = 26.996736
    assert _reference().bunker_sea_usd == pytest.approx(952_570.666667, abs=1e-4)


def test_port_bunker_uses_mgo_not_vlsfo():
    # 700 (MGO) x 3 t/day x 6 days = 12,600, and definitely NOT 500 x 3 x 6 = 9,000
    assert _reference().bunker_port_usd == pytest.approx(12_600.0)


def test_payable_cargo_is_net_of_commission():
    # 66,000 x (1 - 0.0375) = 63,525: the commission reduces the payable cargo,
    # it doesn't add to the cost
    assert _reference().cargo_t == pytest.approx(63_525.0)


def test_total_freight_per_tonne():
    assert _reference().freight_usd_t == pytest.approx(36.426868, abs=1e-6)


def test_waterfall_sums_back_to_the_freight_rate():
    breakdown = _reference()
    assert sum(breakdown.waterfall.values()) == pytest.approx(breakdown.freight_usd_t, abs=1e-9)


# ===========================================================================
# The central disagreement: ballast_share (V-H1)
# ===========================================================================
def test_zero_ballast_removes_the_leg_entirely():
    zero = _reference(ballast_share=0.0)
    assert zero.ballast_days == 0.0
    # the voyage drops to 36.666667 + 6 = 42.666667 days
    assert zero.total_days == pytest.approx(42.666667, abs=1e-6)


def test_ballast_can_only_make_freight_more_expensive():
    """The invariant that makes the gap's distribution asymmetric by construction."""
    rates = [_reference(ballast_share=s).freight_usd_t for s in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert rates == sorted(rates)
    assert rates[0] < rates[-1]


def test_the_ballast_disagreement_is_worth_16_dollars_a_tonne():
    """The number that carries the page: what the disagreement costs, on this voyage.

    Santos -> Qingdao is an 11,000 nm one-way leg; charging the empty return or not
    nearly doubles the cycle's duration. The gap isn't a model refinement, it's the same
    order of magnitude as the arb itself.

    From the trading desk's view, ballast = 0:
        total_days   = 36.666667 + 6                            = 42.666667
        sea bunkers  = 500 x 26 x 36.666667                     = 476,666.67
        hire         = 15,000 x 42.666667                       = 640,000.00
        total cost   = 640,000 + 476,666.67 + 12,600 + 180,000  = 1,309,266.67
        freight      = 1,309,266.67 / 63,525                     = 20.610258 $/t

    From the freight department's view, ballast = 1: 36.426868 $/t (see the module
    docstring). The gap is therefore 15.816610 $/t.
    """
    index_view = _reference(ballast_share=0.0).freight_usd_t
    freight_view = _reference(ballast_share=1.0).freight_usd_t
    assert index_view == pytest.approx(20.610258, abs=1e-5)
    assert freight_view == pytest.approx(36.426868, abs=1e-5)
    assert freight_view - index_view == pytest.approx(15.816610, abs=1e-5)


def test_ballast_share_above_one_is_rejected():
    with pytest.raises(VoyageError, match="ballast_share"):
        VoyageParams(ballast_share=1.5)


def test_negative_ballast_share_is_rejected():
    with pytest.raises(VoyageError, match="ballast_share"):
        VoyageParams(ballast_share=-0.1)


# ===========================================================================
# Speed and the cube law (V-H2)
# ===========================================================================
def test_cube_law_on_consumption():
    # 26 t/day at 12.5 kn -> at 15 kn: 26 x (15/12.5)^3 = 26 x 1.728 = 44.928
    assert PANAMAX.sea_consumption(15.0, laden=True) == pytest.approx(44.928, abs=1e-6)


def test_consumption_at_reference_speed_is_the_base_figure():
    assert PANAMAX.sea_consumption(12.5, laden=True) == pytest.approx(26.0)


def test_slowing_down_saves_bunkers_but_costs_days():
    """The speed/bunkers trade-off, both ways — neither direction is free."""
    fast = _reference(speed_laden_kn=14.0)
    slow = _reference(speed_laden_kn=11.0)
    assert slow.bunker_sea_usd < fast.bunker_sea_usd
    assert slow.total_days > fast.total_days
    assert slow.hire_usd > fast.hire_usd


def test_zero_speed_is_rejected():
    with pytest.raises(VoyageError, match="speed"):
        PANAMAX.sea_consumption(0.0, laden=True)


# ===========================================================================
# Bunkers (V-H3) and invariants (V-H5)
# ===========================================================================
def test_bunker_price_moves_the_rate_materially():
    """The second disagreement: the bunker valuation date.

    On a 78-day voyage, +30% on VLSFO adds 4.50 $/t to the freight — nearly the entire
    width of the marginal decision band.
    """
    base = _reference().freight_usd_t
    expensive = voyage_freight_usd_t(
        15_000.0, 650.0, 700.0, vessel=PANAMAX, route=SANTOS_QINGDAO, params=VoyageParams()
    ).freight_usd_t
    assert expensive - base == pytest.approx(4.498, abs=1e-2)


def test_negative_bunker_price_is_rejected():
    with pytest.raises(VoyageError, match="bunker"):
        voyage_freight_usd_t(
            15_000.0, -500.0, 700.0, vessel=PANAMAX, route=SANTOS_QINGDAO, params=VoyageParams()
        )


def test_negative_tce_is_rejected_as_a_data_diagnostic():
    with pytest.raises(VoyageError, match="negative TCE"):
        voyage_freight_usd_t(
            -100.0, 500.0, 700.0, vessel=PANAMAX, route=SANTOS_QINGDAO, params=VoyageParams()
        )


def test_reference_voyage_raises_no_plausibility_warning():
    assert plausibility_warnings(_reference()) == []


def test_absurd_bunker_share_is_flagged():
    absurd = voyage_freight_usd_t(
        100.0, 3_000.0, 3_000.0, vessel=PANAMAX, route=SANTOS_QINGDAO, params=VoyageParams()
    )
    warnings = plausibility_warnings(absurd)
    assert any("bunkers weigh" in w for w in warnings)


# ===========================================================================
# The inversion — the T1-1 gate's fallback
# ===========================================================================
def test_implied_tce_round_trips():
    """When the Baltic $/t routes exist but not the TCE, one is derived from the other."""
    freight = _reference().freight_usd_t
    implied = implied_tce_from_freight(
        freight, 500.0, 700.0, vessel=PANAMAX, route=SANTOS_QINGDAO, params=VoyageParams()
    )
    assert implied == pytest.approx(15_000.0, abs=1e-6)


def test_a_published_rate_below_costs_implies_a_negative_tce():
    """A published route rate below the voyage cost implies a negative TCE.

    That's not absurd — it happens in a depressed market — but it's the kind of print
    that must be seen, not silently absorbed into an arb.
    """
    implied = implied_tce_from_freight(
        5.0, 500.0, 700.0, vessel=PANAMAX, route=SANTOS_QINGDAO, params=VoyageParams()
    )
    assert implied < 0


# ===========================================================================
# Series
# ===========================================================================
def test_series_intersects_without_filling():
    index = pd.date_range("2024-01-01", periods=10, freq="B")
    tce = pd.Series(15_000.0, index=index)
    vlsfo = pd.Series(500.0, index=index)
    mgo = pd.Series(700.0, index=index).iloc[:6]      # three days missing at the end
    out = voyage_freight_series(
        tce, vlsfo, mgo, vessel=PANAMAX, route=SANTOS_QINGDAO, params=VoyageParams()
    )
    assert len(out) == 6
    assert out.iloc[0] == pytest.approx(36.426868, abs=1e-6)


def test_series_with_no_common_date_raises():
    a = pd.Series(15_000.0, index=pd.date_range("2024-01-01", periods=5, freq="B"))
    b = pd.Series(500.0, index=pd.date_range("2025-01-01", periods=5, freq="B"))
    with pytest.raises(VoyageError, match="no common date"):
        voyage_freight_series(a, b, b, vessel=PANAMAX, route=SANTOS_QINGDAO, params=VoyageParams())


# ===========================================================================
# Other routes and classes
# ===========================================================================
def test_shorter_route_is_cheaper_per_tonne():
    pnw = voyage_freight_usd_t(
        15_000.0, 500.0, 700.0, vessel=PANAMAX, route=ROUTES["pnw_qingdao"], params=VoyageParams()
    )
    assert pnw.freight_usd_t < _reference().freight_usd_t


def test_panama_canal_dues_show_up_in_the_waterfall():
    usgulf = voyage_freight_usd_t(
        15_000.0, 500.0, 700.0, vessel=PANAMAX, route=ROUTES["usgulf_qingdao"], params=VoyageParams()
    )
    # 420,000 $ of dues / 63,525 t = 6.61 $/t: far from a detail
    assert usgulf.waterfall["canal dues"] == pytest.approx(6.611, abs=1e-2)


def test_bigger_vessel_dilutes_fixed_costs_per_tonne():
    kamsarmax = voyage_freight_usd_t(
        15_000.0, 500.0, 700.0, vessel=VESSELS["kamsarmax"], route=SANTOS_QINGDAO, params=VoyageParams()
    )
    assert kamsarmax.freight_usd_t < _reference().freight_usd_t
