"""T1-1 synthetic dataset — built to **impose** the phenomenon the thesis predicts.

This isn't plausible noise. The construction forces three properties, and the golden
tests verify the engine recovers them:

1. `freight_full > freight_index` **always** — ballast can only make it more expensive;
2. the arb under the full convention oscillates around zero with a standard deviation of
   a few dollars, so a substantial share of days fall in the borderline band;
3. on those days, the two conventions give opposite signs — that's the subject.

Property 2 is obtained by building CIF **from** the full freight plus centred noise,
rather than by drawing CIF and FOB independently and hoping the arb lands in the right
place. That's deliberate: a synthetic dataset is for testing the engine, not for
simulating a market.

Tickers prefixed `SYNTH_`. Nothing here should ever be read as a market number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from agri.chains.freight_cf import financing_cost_usd_t
from agri.core.voyage import ROUTES, VESSELS, VoyageParams, voyage_freight_series

DEFAULT_START = "2023-01-02"
DEFAULT_PERIODS = 600          # ~2.4 years of business days, 90 of which are eaten by smoothing

# Reference levels, credible orders of magnitude for a grain Panamax.
TCE_MEAN_USD_DAY = 15_000.0
VLSFO_MEAN_USD_T = 560.0
MGO_MEAN_USD_T = 780.0
FOB_MEAN_USD_T = 440.0         # Santos soybean, order of magnitude

# Standard deviation of the noise added to CIF: this is what decides the borderline
# band's width, and therefore the share of days where the convention is decisive.
CIF_WOBBLE_STD_USD_T = 6.0


def build(
    *,
    start: str = DEFAULT_START,
    periods: int = DEFAULT_PERIODS,
    seed: int = 0,
    vessel_key: str = "panamax",
    route_key: str = "santos_qingdao",
    annual_rate: float = 0.055,
    credit_days: float = 30.0,
    insurance_usd_t: float = 0.85,
) -> dict[str, pd.Series]:
    """Returns T1-1's five input series, as a dict ready for `build_conventions`."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=periods, freq="B")
    n = len(index)
    day_of_year = index.dayofyear.to_numpy()

    # --- TCE: slow mean-reversion + seasonality (northern-hemisphere winter trough) ---
    seasonal_tce = 1.0 + 0.18 * np.sin(2 * np.pi * (day_of_year - 60) / 365.25)
    shocks = rng.normal(scale=0.02, size=n)
    drift = np.zeros(n)
    for i in range(1, n):
        drift[i] = 0.97 * drift[i - 1] + shocks[i]
    tce = pd.Series(
        TCE_MEAN_USD_DAY * seasonal_tce * np.exp(drift), index=index, name="SYNTH_TCE_PANAMAX"
    )

    # --- Bunkers: common slow trend, MGO above VLSFO with a noisy spread ---
    bunker_drift = np.zeros(n)
    bunker_shocks = rng.normal(scale=0.012, size=n)
    for i in range(1, n):
        bunker_drift[i] = 0.985 * bunker_drift[i - 1] + bunker_shocks[i]
    vlsfo = pd.Series(
        VLSFO_MEAN_USD_T * np.exp(bunker_drift), index=index, name="SYNTH_VLSFO_SIN"
    )
    mgo = pd.Series(
        MGO_MEAN_USD_T * np.exp(bunker_drift) + rng.normal(scale=8.0, size=n),
        index=index,
        name="SYNTH_MGO_SIN",
    )

    # --- Origin FOB: gentle random walk around the reference level ---
    fob_drift = np.zeros(n)
    fob_shocks = rng.normal(scale=0.006, size=n)
    for i in range(1, n):
        fob_drift[i] = 0.995 * fob_drift[i - 1] + fob_shocks[i]
    fob = pd.Series(
        FOB_MEAN_USD_T * np.exp(fob_drift), index=index, name="SYNTH_FOB_SANTOS"
    )

    # --- Destination CIF: built SO THAT the full arb oscillates around zero ---
    vessel = VESSELS[vessel_key]
    route = ROUTES[route_key]
    params = VoyageParams()
    freight_full = voyage_freight_series(
        tce, vlsfo, mgo, vessel=vessel, route=route, params=params.with_ballast(1.0)
    )
    voyage_days = _reference_days(vessel, route, params)
    financing = financing_cost_usd_t(
        fob,
        freight_full,
        annual_rate=annual_rate,
        voyage_days=voyage_days,
        credit_days=credit_days,
    )
    wobble = pd.Series(rng.normal(scale=CIF_WOBBLE_STD_USD_T, size=n), index=index)
    cif = pd.Series(
        fob + freight_full + financing + insurance_usd_t + wobble,
        index=index,
        name="SYNTH_CIF_QINGDAO",
    )

    return {"tce": tce, "vlsfo": vlsfo, "mgo": mgo, "cif": cif, "fob": fob}


def _reference_days(vessel, route, params: VoyageParams) -> float:
    from agri.core.voyage import HOURS_PER_DAY

    laden = route.distance_laden_nm / (params.speed_laden_kn * HOURS_PER_DAY)
    ballast = route.distance_ballast_nm / (params.speed_ballast_kn * HOURS_PER_DAY)
    return laden + ballast + params.port_days + params.wait_days


def build_frame(**kwargs) -> pd.DataFrame:
    """Shortcut: the synthetic series already passed through `build_conventions`."""
    from agri.chains.freight_cf import build_conventions

    vessel_key = kwargs.pop("vessel_key", "panamax")
    route_key = kwargs.pop("route_key", "santos_qingdao")
    series = build(vessel_key=vessel_key, route_key=route_key, **kwargs)
    return build_conventions(
        series["tce"],
        series["vlsfo"],
        series["mgo"],
        series["cif"],
        series["fob"],
        vessel=VESSELS[vessel_key],
        route=ROUTES[route_key],
        params=VoyageParams(),
    )
