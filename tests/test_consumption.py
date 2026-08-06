import pytest

from freight.voyage.config import VoyageParams
from freight.voyage.consumption import consumption_t_per_day, sea_days, leg_bunker_consumption_t


def test_cubic_law_doubling_speed_octuples_consumption():
    params = VoyageParams(reference_speed_kn=10.0, reference_consumption_t_per_day=20.0, consumption_exponent=3.0)
    at_reference = consumption_t_per_day(10.0, params)
    at_double = consumption_t_per_day(20.0, params)
    assert at_reference == pytest.approx(20.0)
    assert at_double == pytest.approx(20.0 * 8, rel=1e-9)


def test_slow_steaming_saves_disproportionately():
    """Slowing from 14 to 9 knots (the doc's example) should cut consumption far more
    than proportionally — this asymmetry is what makes speed a real decision variable.
    """
    params = VoyageParams(reference_speed_kn=14.0, reference_consumption_t_per_day=60.0, consumption_exponent=3.0)
    fast = consumption_t_per_day(14.0, params)
    slow = consumption_t_per_day(9.0, params)
    speed_drop_pct = (14.0 - 9.0) / 14.0
    consumption_drop_pct = (fast - slow) / fast
    assert consumption_drop_pct > speed_drop_pct  # cubic law: consumption falls faster than speed


def test_sea_days_and_total_leg_consumption():
    days = sea_days(distance_nm=2400.0, speed_kn=10.0)
    assert days == pytest.approx(10.0)  # 2400 / (10*24)

    params = VoyageParams(reference_speed_kn=10.0, reference_consumption_t_per_day=20.0, consumption_exponent=3.0)
    total = leg_bunker_consumption_t(distance_nm=2400.0, speed_kn=10.0, params=params)
    assert total == pytest.approx(200.0)  # 10 days * 20 t/day


def test_zero_or_negative_speed_rejected():
    params = VoyageParams()
    with pytest.raises(ValueError):
        consumption_t_per_day(0.0, params)
    with pytest.raises(ValueError):
        sea_days(1000.0, -5.0)
