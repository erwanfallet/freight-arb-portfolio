"""Smoke test du projet B sur le jeu SYNTHÉTIQUE, dont la rupture est imposée à la main.

Ne prouve rien sur le marché du charbon. Prouve que fixture -> arb -> régimes -> test de
rupture avec et sans contrôle -> couche ETS -> sensibilité CV s'enchaînent.
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
    print("DONNÉES SYNTHÉTIQUES — la rupture 2022 est IMPOSÉE dans le générateur.")
    print("Aucune lecture économique de ces chiffres n'est valide.")
    print("=" * 78)

    raw = synthetic_coal()
    assert raw.attrs.get("synthetic") is True
    s = {role: to_series(raw, tk) for role, tk in SYNTHETIC_TICKERS.items()}

    emissions = voyage_emissions_t_co2(480.0)
    ets = ets_cost_per_cargo_tonne(
        eua_price_eur=s["eua"], eurusd=s["eurusd"], emissions_t_co2=emissions,
        cargo_t=150_000.0, phase_in=phase_in_series(s["api2"].index),
    )
    print(f"\némissions du voyage : {emissions:,.2f} tCO2")

    arb = reconstruct_ara_arb(
        api2=s["api2"], api4=s["api4"], freight=s["freight"],
        voyage_days=20.0, annual_rate=0.06, ets_cost=ets.reindex(s["api2"].index),
    )
    print(f"arb calculé sur {len(arb)} dates")
    print(
        arb[["spread", "freight", "financing", "ets", "arb"]]
        .describe().round(2).to_string()
    )

    print("\n-- régimes ---------------------------------------------------------")
    for st_ in regime_stats(arb, BREAKPOINT):
        print(
            f"{st_.label:>20} | n={st_.n_obs:5d} | arb moyen {st_.arb_mean:7.2f} "
            f"| ouvert {100 * st_.share_open:5.1f} % | plus longue série "
            f"{st_.longest_open_run:4d} j"
        )

    print("\n-- test de rupture, SANS contrôle ---------------------------------")
    for label, res in freight_binding_test(arb["spread"], arb["freight"], BREAKPOINT).items():
        print(f"{label:>20} | {res.summary()}")

    print("\n-- test de rupture, AVEC contrôle TTF -----------------------------")
    for label, res in freight_binding_test(
        arb["spread"], arb["freight"], BREAKPOINT,
        controls={"ttf": s["ttf"].reindex(arb.index)},
    ).items():
        print(f"{label:>20} | {res.summary()}")

    print("\n-- couche ETS ------------------------------------------------------")
    for year in (2023, 2024, 2025, 2026):
        chunk = arb.loc[str(year)]
        if not chunk.empty:
            print(f"{year} : coût ETS moyen {chunk['ets'].mean():.3f} $/t")

    print("\n-- dérive du pouvoir calorifique ----------------------------------")
    for cv in (5500, 5750, 6000):
        fret_energy = to_energy_basis(arb["freight"], float(cv))
        print(
            f"CV {cv} : facteur {BENCHMARK_CV_KCAL_PER_KG / cv:.4f} | fret affiché "
            f"{arb['freight'].mean():.2f} -> par t-éq-6000 {fret_energy.mean():.2f} "
            f"(+{fret_energy.mean() - arb['freight'].mean():.2f} $/t)"
        )

    assert pd.notna(arb["arb"]).all()
    print("\nOK — le pipeline du projet B tourne de bout en bout.")


if __name__ == "__main__":
    main()
