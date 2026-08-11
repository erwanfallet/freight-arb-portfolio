"""End-to-end smoke test for project A, on the SYNTHETIC dataset.

Proves nothing about the market. Proves that fixture -> series -> decomposition ->
regression -> episodes -> hedge -> sensitivities chain together without error and
produce finite numbers. `make smoke`.
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
    print("SYNTHETIC DATA — no number below has any economic value")
    print("=" * 78)

    raw = synthetic_ironore()
    assert raw.attrs.get("synthetic") is True, "the synthetic flag must be set"
    series = {role: to_series(raw, tk) for role, tk in SYNTHETIC_TICKERS.items()}

    print("\n-- coverage ----------------------------------------------------------")
    print(coverage_report(series).to_string(index=False))

    d = decompose_premium(
        p65=series["p65"], p62=series["p62"], c3=series["c3"], c5=series["c5"]
    )
    print(f"\n-- decomposition: {len(d)} dates after calendar intersection")
    print(
        d[["premium_observed", "freight_fair_value", "residual", "freight_share"]]
        .describe()
        .round(3)
        .to_string()
    )

    naive_share = (d["premium_naive_freight"] / d["premium_observed"]).mean()
    print(
        f"\naverage corrected freight share  : {100 * d['freight_share'].mean():.1f}%"
        f"\naverage uncorrected freight share: {100 * naive_share:.1f}%"
        f"\ngap due to moisture               : "
        f"{100 * (d['freight_share'].mean() - naive_share):.1f} point(s)"
    )

    print("\n-- explained variance --------------------------------------------------")
    print("level  :", explained_variance(d["premium_observed"], d["freight_fair_value"]).summary)
    print(
        "change :",
        explained_variance(d["premium_observed"], d["freight_fair_value"], on_changes=True).summary,
    )

    print("\n-- negative residual episodes (>= 5 days) ------------------------------")
    ep = negative_residual_episodes(d, min_days=5)
    print("none" if ep.empty else ep.to_string(index=False))

    print("\n-- freight leg hedge ----------------------------------------------------")
    opt = freight_hedge_effect(d["premium_observed"], d["freight_fair_value"])
    unit = freight_hedge_effect(d["premium_observed"], d["freight_fair_value"], beta=1.0)
    print(
        f"optimal beta {opt.beta:.3f} | unhedged vol {opt.vol_unhedged:.3f} "
        f"| hedged {opt.vol_hedged:.3f} | reduction {opt.vol_reduction_pct:.1f}%"
    )
    print(
        f"beta = 1     {unit.beta:.3f} | hedged vol {unit.vol_hedged:.3f} "
        f"| reduction {unit.vol_reduction_pct:.1f}%"
    )

    carry = carry_cost_of_extra_voyage_days(d["p62"].mean(), 25.0, 0.06)
    print(
        f"\n-- carry: 25 d at 6% on {d['p62'].mean():.1f} $/dmt = {carry:.2f} $/dmt"
        f" | average residual {d['residual'].mean():.2f} $/dmt"
    )

    for name, value in {
        "premium": d["premium_observed"],
        "freight": d["freight_fair_value"],
        "residual": d["residual"],
    }.items():
        assert pd.notna(value).all(), f"{name} contains NaNs"
    print("\nOK — the pipeline runs end to end.")


if __name__ == "__main__":
    main()
