from freight.voyage.config import VoyageParams
from freight.backtest.sensitivity import sensitivity_sweep


def test_sensitivity_ranks_by_elasticity():
    base = VoyageParams()

    def metric_fn(p: VoyageParams) -> float:
        # a synthetic metric strongly driven by cargo_t, weakly by port_costs_usd
        return p.cargo_t * 2.0 - p.port_costs_usd * 0.001

    table = sensitivity_sweep(base, metric_fn, pct=0.10)
    ranked_params = list(table["parameter"])
    assert ranked_params[0] == "cargo_t"
    assert "port_costs_usd" in ranked_params
    assert ranked_params.index("cargo_t") < ranked_params.index("port_costs_usd")


def test_sensitivity_elasticity_sign_matches_direction():
    base = VoyageParams()

    def metric_fn(p: VoyageParams) -> float:
        return p.cargo_t  # metric increases with cargo_t

    table = sensitivity_sweep(base, metric_fn, pct=0.10)
    row = table[table["parameter"] == "cargo_t"].iloc[0]
    assert row["elasticity"] > 0
