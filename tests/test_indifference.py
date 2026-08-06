import pytest

from freight.voyage.indifference import fair_value_c3


def test_indifference_is_self_consistent():
    """If C5's actual employment already equalizes TCE with a hypothetical C3 employment
    at rate R, then fair_value_c3 computed from that same C5 and cost structure must
    return R — this is the round-trip sanity check for the derivation in the module
    docstring (Partie 2.3 / 3.4).
    """
    cargo_t = 170_000.0
    commission = 0.0375
    d3_days, d5_days = 78.0, 27.0
    costs3_usd, costs5_usd = 1_760_000.0, 650_000.0
    c5 = 9.0

    # Solve TCE5 first, then back out the C3 rate that equalizes TCE3 to it directly
    # (independent of fair_value_c3), and check fair_value_c3 reproduces that rate.
    revenue5_net = cargo_t * c5 * (1 - commission)
    tce5 = (revenue5_net - costs5_usd) / d5_days
    revenue3_net_needed = tce5 * d3_days + costs3_usd
    c3_direct = revenue3_net_needed / (cargo_t * (1 - commission))

    c3_star = fair_value_c3(
        c5_usd_per_t=c5, d3_days=d3_days, d5_days=d5_days,
        costs3_usd=costs3_usd, costs5_usd=costs5_usd,
        cargo_t=cargo_t, commission=commission,
    )
    assert c3_star == pytest.approx(c3_direct, rel=1e-9)


def test_higher_c5_raises_fair_value_c3():
    kwargs = dict(d3_days=78.0, d5_days=27.0, costs3_usd=1_760_000.0,
                   costs5_usd=650_000.0, cargo_t=170_000.0, commission=0.0375)
    low = fair_value_c3(c5_usd_per_t=8.0, **kwargs)
    high = fair_value_c3(c5_usd_per_t=10.0, **kwargs)
    assert high > low
