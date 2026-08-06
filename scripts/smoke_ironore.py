"""Smoke test du projet A de bout en bout, sur le jeu SYNTHÉTIQUE.

Ne prouve rien sur le marché. Prouve que fixture -> séries -> décomposition -> régression
-> épisodes -> hedge -> sensibilités s'enchaînent sans erreur et produisent des nombres
finis. `make smoke`.
"""
from __future__ import annotations

import pandas as pd

from freight.chains.ironore import (
    carry_cost_of_extra_voyage_days,
    decompose_premium,
    explained_variance,
    freight_hedge_effect,
    negative_residual_episodes,
)
from freight.ingest.fixture import SYNTHETIC_TICKERS, synthetic_ironore
from freight.ingest.series import coverage_report, to_series


def main() -> None:
    print("=" * 78)
    print("DONNÉES SYNTHÉTIQUES — aucun chiffre ci-dessous n'a de valeur économique")
    print("=" * 78)

    raw = synthetic_ironore()
    assert raw.attrs.get("synthetic") is True, "le drapeau synthetic doit être posé"
    series = {role: to_series(raw, tk) for role, tk in SYNTHETIC_TICKERS.items()}

    print("\n-- couverture ------------------------------------------------------")
    print(coverage_report(series).to_string(index=False))

    d = decompose_premium(
        p65=series["p65"], p62=series["p62"], c3=series["c3"], c5=series["c5"]
    )
    print(f"\n-- décomposition : {len(d)} dates après intersection des calendriers")
    print(
        d[["premium_observed", "freight_fair_value", "residual", "freight_share"]]
        .describe()
        .round(3)
        .to_string()
    )

    naive_share = (d["premium_naive_freight"] / d["premium_observed"]).mean()
    print(
        f"\npart fret moyenne corrigée   : {100 * d['freight_share'].mean():.1f} %"
        f"\npart fret moyenne non corrigée: {100 * naive_share:.1f} %"
        f"\nécart dû à l'humidité         : "
        f"{100 * (d['freight_share'].mean() - naive_share):.1f} point(s)"
    )

    print("\n-- variance expliquée ----------------------------------------------")
    print("niveau   :", explained_variance(d["premium_observed"], d["freight_fair_value"]).summary)
    print(
        "variation:",
        explained_variance(d["premium_observed"], d["freight_fair_value"], on_changes=True).summary,
    )

    print("\n-- épisodes de résidu négatif (>= 5 jours) -------------------------")
    ep = negative_residual_episodes(d, min_days=5)
    print("aucun" if ep.empty else ep.to_string(index=False))

    print("\n-- couverture de la jambe fret -------------------------------------")
    opt = freight_hedge_effect(d["premium_observed"], d["freight_fair_value"])
    unit = freight_hedge_effect(d["premium_observed"], d["freight_fair_value"], beta=1.0)
    print(
        f"beta optimal {opt.beta:.3f} | vol non couverte {opt.vol_unhedged:.3f} "
        f"| couverte {opt.vol_hedged:.3f} | réduction {opt.vol_reduction_pct:.1f} %"
    )
    print(
        f"beta = 1     {unit.beta:.3f} | vol couverte {unit.vol_hedged:.3f} "
        f"| réduction {unit.vol_reduction_pct:.1f} %"
    )

    carry = carry_cost_of_extra_voyage_days(d["p62"].mean(), 25.0, 0.06)
    print(
        f"\n-- portage : 25 j à 6 % sur {d['p62'].mean():.1f} $/dmt = {carry:.2f} $/dmt"
        f" | résidu moyen {d['residual'].mean():.2f} $/dmt"
    )

    for name, value in {
        "premium": d["premium_observed"],
        "fret": d["freight_fair_value"],
        "résidu": d["residual"],
    }.items():
        assert pd.notna(value).all(), f"{name} contient des NaN"
    print("\nOK — le pipeline tourne de bout en bout.")


if __name__ == "__main__":
    main()
