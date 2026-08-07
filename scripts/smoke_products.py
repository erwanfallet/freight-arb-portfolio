"""Smoke test du projet C sur le jeu SYNTHÉTIQUE, dont les marches de flat rate sont
imposées à la main. Aucune lecture économique valide.
"""
from __future__ import annotations

import pandas as pd

from freight.chains.products import (
    DEFAULT_BBL_PER_TONNE,
    decompose_freight_change,
    density_sensitivity,
    flat_rate_step_series,
    freight_model_vs_quoted,
    freight_usd_per_tonne,
    implied_freight_from_tce,
    january_reset_effect,
    monthly_profile,
    open_days_comparison,
    reconstruct_arb,
)
from freight.ingest.fixture_products import (
    ROUTE,
    SYNTHETIC_FLAT_RATES,
    SYNTHETIC_TICKERS,
    synthetic_products,
)
from freight.ingest.series import to_series
from freight.voyage.config import VoyageParams
from freight.voyage.consumption import leg_bunker_consumption_t, sea_days


def main() -> None:
    print("=" * 78)
    print("DONNÉES SYNTHÉTIQUES — les marches de flat rate sont IMPOSÉES.")
    print("=" * 78)

    raw = synthetic_products()
    assert raw.attrs.get("synthetic") is True
    s = {role: to_series(raw, tk) for role, tk in SYNTHETIC_TICKERS.items()}

    flat_rates = flat_rate_step_series(s["ws"].index, ROUTE, SYNTHETIC_FLAT_RATES)
    freight = freight_usd_per_tonne(s["ws"], flat_rates)

    arb = reconstruct_arb(
        price_destination_usd_t=s["p_ara"],
        price_origin_usd_per_gallon=s["p_usgc"],
        freight_usd_t=freight,
        spec_bridge_usd_t=5.0, voyage_days=16.0, annual_rate=0.06, losses_usd_t=1.0,
    )
    print(f"\narb calculé sur {len(arb)} dates")
    print(
        arb[["p_origin_t", "spread_naif", "freight", "financing", "arb"]]
        .describe().round(2).to_string()
    )

    print("\n-- Worldscale : réinitialisations de flat rate ----------------------")
    resets = january_reset_effect(decompose_freight_change(s["ws"], flat_rates))
    print(resets.to_string(index=False))
    biggest = resets.iloc[resets["part_reglage"].abs().argmax()]
    print(
        f"\nplus grosse marche : {pd.Timestamp(biggest['date']).date()} | "
        f"saut total {biggest['saut_total']:.2f} $/t dont réglage "
        f"{biggest['part_reglage']:.2f} et marché {biggest['part_marche']:.2f}"
        f"\narb moyen sur l'échantillon : {arb['arb'].mean():.2f} $/t"
    )

    print("\n-- illusion des jours ouverts --------------------------------------")
    c = open_days_comparison(arb)
    print(
        f"{c.days_looking_open} jours ouverts sur le spread nu "
        f"({100 * c.share_looking_open:.1f} %), {c.days_really_open} réellement ouverts "
        f"({100 * c.share_really_open:.1f} %) -> illusion {100 * c.illusion_share:.1f} %"
    )

    print("\n-- densité ---------------------------------------------------------")
    print(density_sensitivity(float(arb['p_origin_gal'].iloc[-1])).to_string(index=False))

    print("\n-- C-2 : fret reconstruit par inversion du TCE ---------------------")
    params = VoyageParams(
        cargo_t=37_000.0, laden_speed_kn=13.0, ballast_speed_kn=14.0,
        reference_speed_kn=13.0, reference_consumption_t_per_day=25.0,
        port_costs_usd=120_000.0, brokerage_commission=0.0375,
    )
    distance = 5_000.0
    total_days = (
        sea_days(distance, params.laden_speed_kn)
        + sea_days(distance, params.ballast_speed_kn) + 4.0
    )
    bunker_t = (
        leg_bunker_consumption_t(distance, params.laden_speed_kn, params)
        + leg_bunker_consumption_t(distance, params.ballast_speed_kn, params)
    )
    costs = bunker_t * s["bunker"] + params.port_costs_usd
    modelled = pd.Series(
        [
            implied_freight_from_tce(float(t), total_days, float(c_), 37_000.0, 0.0375)
            for t, c_ in zip(s["tce_mr"], costs)
        ],
        index=s["tce_mr"].index,
    )
    print(f"voyage : {total_days:.1f} jours, {bunker_t:,.0f} t de soutes")
    for key, value in freight_model_vs_quoted(modelled, freight).items():
        print(f"  {key:>20} : {value:,.3f}")

    print("\n-- saisonnalité de l'arb (moyenne par mois) ------------------------")
    print(monthly_profile(arb["arb"])["mean"].round(2).to_string())

    assert pd.notna(arb["arb"]).all()
    print(f"\ndensité par défaut utilisée : {DEFAULT_BBL_PER_TONNE} bbl/t")
    print("OK — le pipeline du projet C tourne de bout en bout.")


if __name__ == "__main__":
    main()
