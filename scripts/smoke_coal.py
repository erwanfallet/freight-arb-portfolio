"""Smoke test for project B on the SYNTHETIC dataset, whose break is imposed by hand.

Proves nothing about the coal market. Proves that fixture -> arb -> regimes -> break
test with and without control -> ETS layer -> CV sensitivity chain together.
"""
from __future__ import annotations

import pandas as pd

from freight.chains.coal import (
    BENCHMARK_CV_KCAL_PER_KG,
    ets_cost_per_cargo_tonne,
    freight_binding_test,
    phase_in_series,
    reconstruct_ara_arb,
    regime_stats,
    to_energy_basis,
    voyage_emissions_t_co2,
)
from freight.ingest.fixture_coal import BREAKPOINT, SYNTHETIC_TICKERS, synthetic_coal
from freight.ingest.series import to_series


def main() -> None:
    print("=" * 78)
    print("SYNTHETIC DATA — the 2022 break is IMPOSED in the generator.")
    print("No economic reading of these numbers is valid.")
    print("=" * 78)

    raw = synthetic_coal()
    assert raw.attrs.get("synthetic") is True
    s = {role: to_series(raw, tk) for role, tk in SYNTHETIC_TICKERS.items()}

    emissions = voyage_emissions_t_co2(480.0)
    ets = ets_cost_per_cargo_tonne(
        eua_price_eur=s["eua"], eurusd=s["eurusd"], emissions_t_co2=emissions,
        cargo_t=150_000.0, phase_in=phase_in_series(s["api2"].index),
    )
    print(f"\nvoyage emissions: {emissions:,.2f} tCO2")

    arb = reconstruct_ara_arb(
        api2=s["api2"], api4=s["api4"], freight=s["freight"],
        voyage_days=20.0, annual_rate=0.06, ets_cost=ets.reindex(s["api2"].index),
    )
    print(f"arb computed over {len(arb)} dates")
    print(
        arb[["spread", "freight", "financing", "ets", "arb"]]
        .describe().round(2).to_string()
    )

    print("\n-- regimes -------------------------------------------------------------")
    for st_ in regime_stats(arb, BREAKPOINT):
        print(
            f"{st_.label:>20} | n={st_.n_obs:5d} | mean arb {st_.arb_mean:7.2f} "
            f"| open {100 * st_.share_open:5.1f}% | longest run "
            f"{st_.longest_open_run:4d} d"
        )

    print("\n-- break test, WITHOUT control ------------------------------------------")
    for label, res in freight_binding_test(arb["spread"], arb["freight"], BREAKPOINT).items():
        print(f"{label:>20} | {res.summary()}")

    print("\n-- break test, WITH TTF control ------------------------------------------")
    for label, res in freight_binding_test(
        arb["spread"], arb["freight"], BREAKPOINT,
        controls={"ttf": s["ttf"].reindex(arb.index)},
    ).items():
        print(f"{label:>20} | {res.summary()}")

    print("\n-- ETS layer -------------------------------------------------------------")
    for year in (2023, 2024, 2025, 2026):
        chunk = arb.loc[str(year)]
        if not chunk.empty:
            print(f"{year}: average ETS cost {chunk['ets'].mean():.3f} $/t")

    print("\n-- calorific value drift --------------------------------------------------")
    for cv in (5500, 5750, 6000):
        freight_energy = to_energy_basis(arb["freight"], float(cv))
        print(
            f"CV {cv}: factor {BENCHMARK_CV_KCAL_PER_KG / cv:.4f} | quoted freight "
            f"{arb['freight'].mean():.2f} -> per t-eq-6000 {freight_energy.mean():.2f} "
            f"(+{freight_energy.mean() - arb['freight'].mean():.2f} $/t)"
        )

    assert pd.notna(arb["arb"]).all()
    print("\nOK — project B's pipeline runs end to end.")


if __name__ == "__main__":
    main()
