"""Golden test against the brainstorm's worked Capesize C3 example (Partie 2.2)."""
import pytest

from freight.voyage.config import VoyageParams
from freight.voyage.tce import VoyageInputs, compute_tce


@pytest.fixture
def doc_example_params() -> VoyageParams:
    return VoyageParams(
        cargo_t=170_000,
        laden_speed_kn=12.0,
        ballast_speed_kn=13.0,
        reference_speed_kn=12.5,
        reference_consumption_t_per_day=40.0,
        consumption_exponent=0.0,  # doc's example uses a flat 40 t/day regardless of leg speed
        brokerage_commission=0.0375,
        port_costs_usd=180_000.0,
        port_consumption_t_per_day=0.0,
    )


def test_capesize_c3_worked_example(doc_example_params):
    inputs = VoyageInputs(
        cargo_t=170_000,
        freight_rate_usd_per_t=22.0,
        distance_laden_nm=11_000.0,
        distance_ballast_nm=10_500.0,
        bunker_price_usd_per_t=550.0,
        port_days=6.0,
        params=doc_example_params,
    )
    result = compute_tce(inputs)

    assert result.revenue_gross_usd == pytest.approx(3_740_000, rel=1e-6)
    assert result.commission_usd == pytest.approx(140_250, rel=1e-6)
    assert result.revenue_net_usd == pytest.approx(3_599_750, rel=1e-6)
    assert result.laden_days == pytest.approx(38.19, abs=0.05)
    assert result.ballast_days == pytest.approx(33.65, abs=0.05)
    assert result.total_days == pytest.approx(77.9, abs=0.5)
    assert result.port_costs_usd == pytest.approx(180_000)
    # doc rounds sea days to 72 for the illustrative bunker calc; we keep continuous days,
    # so allow a wider tolerance while still pinning the order of magnitude
    assert result.tce_usd_per_day == pytest.approx(23_565, rel=0.03)


def test_ballast_leg_matters(doc_example_params):
    """Dropping the ballast leg materially overstates TCE on a long route — Partie 2.2
    point 1, the single most-repeated warning in the brainstorm ("~40-50% on a long
    route" is the doc's illustrative order of magnitude, not a universal constant; the
    exact overstatement is route-specific, so this test only pins the direction and
    that it is large, not a precise doc-quoted bound).
    """
    with_ballast = VoyageInputs(
        cargo_t=170_000, freight_rate_usd_per_t=22.0,
        distance_laden_nm=11_000.0, distance_ballast_nm=10_500.0,
        bunker_price_usd_per_t=550.0, port_days=6.0, params=doc_example_params,
    )
    without_ballast = VoyageInputs(
        cargo_t=170_000, freight_rate_usd_per_t=22.0,
        distance_laden_nm=11_000.0, distance_ballast_nm=0.0,
        bunker_price_usd_per_t=550.0, port_days=6.0, params=doc_example_params,
    )
    tce_with = compute_tce(with_ballast).tce_usd_per_day
    tce_without = compute_tce(without_ballast).tce_usd_per_day
    overstatement = (tce_without - tce_with) / tce_with
    assert overstatement > 0.3  # ballast omission materially inflates TCE, direction + magnitude
