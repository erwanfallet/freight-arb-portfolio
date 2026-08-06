"""H1 verdict — Partie 4.1 / PROJECT_NOTES.md.

Ranks chains two ways: by freight/cargo-value ratio (the conventional heuristic) and by
share of arb sign-flips attributable to freight (signals/switching.py). If the two
rankings correlate strongly (Spearman rho >= H1_FALSIFICATION_RHO) across
H1_FALSIFICATION_MIN_CHAINS or more chains, H1 is dead by its own pre-registered
criterion — this module reports that verdict, it does not adjudicate it after the fact.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from scipy import stats

H1_FALSIFICATION_RHO = 0.7
H1_FALSIFICATION_MIN_CHAINS = 4


@dataclass(frozen=True)
class ChainResult:
    chain: str
    freight_value_ratio: float
    freight_attributable_share: float
    n_flips: int


def build_ranking_table(chain_results: list[ChainResult]) -> pd.DataFrame:
    df = pd.DataFrame([r.__dict__ for r in chain_results])
    df["rank_by_freight_value"] = df["freight_value_ratio"].rank(ascending=False)
    df["rank_by_switching_share"] = df["freight_attributable_share"].rank(ascending=False)
    df["rank_delta"] = df["rank_by_freight_value"] - df["rank_by_switching_share"]
    return df.sort_values("rank_by_freight_value")


def h1_verdict(df: pd.DataFrame) -> dict:
    n_chains = len(df)
    if n_chains < 2:
        return {"n_chains": n_chains, "rho": None, "p_value": None, "h1_falsified": None,
                "reason": "fewer than 2 chains — cannot rank-correlate"}

    rho, p_value = stats.spearmanr(df["freight_value_ratio"], df["freight_attributable_share"])
    falsified = None
    reason = f"only {n_chains} chains — need >= {H1_FALSIFICATION_MIN_CHAINS} to invoke the pre-registered criterion"
    if n_chains >= H1_FALSIFICATION_MIN_CHAINS:
        falsified = bool(rho >= H1_FALSIFICATION_RHO)
        reason = (
            f"rho={rho:.2f} >= {H1_FALSIFICATION_RHO} on {n_chains} chains -> H1 dead, "
            "the rankings agree and freight/value was the right criterion all along"
            if falsified
            else f"rho={rho:.2f} < {H1_FALSIFICATION_RHO} on {n_chains} chains -> H1 survives, "
            "the rankings diverge, freight/value is not tracking what drives decisions"
        )
    return {"n_chains": n_chains, "rho": float(rho), "p_value": float(p_value),
            "h1_falsified": falsified, "reason": reason}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--out", default="report/h1_ranking.csv")
    args = parser.parse_args()

    processed = Path(args.processed)
    if not any(processed.glob("*.csv")):
        print(
            f"No processed data in {processed} yet — nothing to rank. "
            "This is expected until the Bloomberg dump is ingested; wire real per-chain "
            "series into freight.signals.arb.reconstruct_arb / signals.switching before "
            "calling build_ranking_table. See PROJECT_NOTES.md for the falsification rule."
        )
        return

    raise NotImplementedError(
        "Per-chain series assembly from data/processed/ is chain-specific (which ticker "
        "is price_destination/price_origin/freight/cargo_value per data_dictionary.csv "
        "'chain'/'leg' columns) and depends on the actual Bloomberg tickers received — "
        "wire this up once the dump lands, using arb.py and switching.py as building blocks."
    )


if __name__ == "__main__":
    main()
