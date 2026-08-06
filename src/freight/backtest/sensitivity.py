"""Parameter sensitivity sweep — Partie 7's mandatory "±10% on each parameter, ranked
by contribution to variance" section. This is what answers a senior's first objection:
"and if your speed assumption is wrong?"
"""
from __future__ import annotations

from dataclasses import fields
from typing import Callable

import pandas as pd

from freight.voyage.config import VoyageParams


def sensitivity_sweep(
    base_params: VoyageParams,
    metric_fn: Callable[[VoyageParams], float],
    pct: float = 0.10,
) -> pd.DataFrame:
    """For each numeric field of VoyageParams, perturb it +-pct and record the effect on
    metric_fn(params). Returns a table ranked by |elasticity| descending, where
    elasticity = (%change in metric) / (%change in parameter).
    """
    base_value = metric_fn(base_params)
    rows = []
    for f in fields(base_params):
        current = getattr(base_params, f.name)
        if not isinstance(current, (int, float)) or isinstance(current, bool):
            continue
        if current == 0:
            continue  # relative perturbation undefined at zero

        up_params = base_params.with_overrides(**{f.name: current * (1 + pct)})
        down_params = base_params.with_overrides(**{f.name: current * (1 - pct)})
        metric_up = metric_fn(up_params)
        metric_down = metric_fn(down_params)

        metric_range = metric_up - metric_down
        elasticity = (metric_range / base_value) / (2 * pct) if base_value else float("nan")

        rows.append(
            {
                "parameter": f.name,
                "base_value": current,
                "metric_base": base_value,
                "metric_at_plus_pct": metric_up,
                "metric_at_minus_pct": metric_down,
                "elasticity": elasticity,
            }
        )

    df = pd.DataFrame(rows)
    df["abs_elasticity"] = df["elasticity"].abs()
    return df.sort_values("abs_elasticity", ascending=False).drop(columns="abs_elasticity")
