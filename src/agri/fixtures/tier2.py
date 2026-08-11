"""Synthetic datasets for the six Tier 2 projects.

Each generator **imposes** the phenomenon its thesis predicts, with the true parameters
exposed as constants so the golden tests verify the engine recovers them.
Tickers prefixed `SYNTH_`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# T2-3 — board crush against plant crush
# ---------------------------------------------------------------------------
DECOUPLING_WINDOWS = (("2022-09-01", "2022-11-15"), ("2024-03-01", "2024-05-01"))


def crush_tracking(*, periods: int = 1_400, seed: int = 2) -> dict[str, pd.Series]:
    """Board and cash prices, with two imposed decoupling episodes.

    Outside the episodes, the local basis is small and stable; inside them, the meal
    basis blows up — that's the mechanism plant people describe and the board doesn't
    see.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2021-01-04", periods=periods, freq="B")
    n = len(index)

    bean_board = pd.Series(
        13.20 * np.exp(np.cumsum(rng.normal(scale=0.010, size=n))),
        index=index,
        name="SYNTH_CBOT_BEAN_USD_BU",
    )
    meal_board = pd.Series(
        390.0 * np.exp(np.cumsum(rng.normal(scale=0.011, size=n))),
        index=index,
        name="SYNTH_CBOT_MEAL_USD_STON",
    )
    oil_board = pd.Series(
        58.0 * np.exp(np.cumsum(rng.normal(scale=0.013, size=n))),
        index=index,
        name="SYNTH_CBOT_OIL_C_LB",
    )

    stress = np.zeros(n)
    for start, end in DECOUPLING_WINDOWS:
        stress[(index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))] = 1.0

    # During the decoupling, the meal basis doesn't just shift: it acquires its OWN
    # dynamics, independent of the board. That's what moves the covariance, and
    # therefore the optimal hedge ratio. A simple level shift would leave h* stuck at 1
    # and the page would have nothing to show.
    independent_walk = np.cumsum(rng.normal(scale=9.0, size=n) * stress)
    bean_cash = bean_board + (-0.25 + rng.normal(scale=0.04, size=n))
    meal_cash = meal_board + (
        8.0 + 55.0 * stress + independent_walk + rng.normal(scale=3.0, size=n)
    )
    oil_cash = oil_board + (-0.6 + rng.normal(scale=0.25, size=n))

    return {
        "bean_board": bean_board,
        "meal_board": meal_board,
        "oil_board": oil_board,
        "bean_cash": pd.Series(bean_cash.to_numpy(), index=index, name="SYNTH_CASH_BEAN"),
        "meal_cash": pd.Series(meal_cash.to_numpy(), index=index, name="SYNTH_CASH_MEAL"),
        "oil_cash": pd.Series(oil_cash.to_numpy(), index=index, name="SYNTH_CASH_OIL"),
    }


# ---------------------------------------------------------------------------
# T2-4 — white premium
# ---------------------------------------------------------------------------
def white_premium(*, periods: int = 1_300, seed: int = 3) -> dict[str, pd.Series]:
    """No.11 and No.5, with a `richness` that switches between RICH and CHEAP.

    No.5 is built **from** the reconstructed refining cost plus a cyclical residual:
    without that, the page would have no zones to show.
    """
    from agri.chains.white_premium import (
        DEFAULT_POL_ADJUST,
        fair_value_refining_usd_t,
    )
    from agri.core.units import cents_lb_to_usd_t

    rng = np.random.default_rng(seed)
    index = pd.date_range("2021-01-04", periods=periods, freq="B")
    n = len(index)

    no11 = pd.Series(
        19.5 * np.exp(np.cumsum(rng.normal(scale=0.011, size=n))),
        index=index,
        name="SYNTH_NY11_C_LB",
    )
    costs = fair_value_refining_usd_t(no11)
    richness = 34.0 * np.sin(2 * np.pi * np.arange(n) / 300.0) + rng.normal(scale=6.0, size=n)
    no5 = pd.Series(
        (cents_lb_to_usd_t(no11) * DEFAULT_POL_ADJUST + costs["total"] + richness).to_numpy(),
        index=index,
        name="SYNTH_NO5_USD_T",
    )
    # keys named like `build_richness`'s parameters, so that `**fixture` works
    return {"no5_usd_t": no5, "no11_cents_lb": no11}


# ---------------------------------------------------------------------------
# T2-5 — the plant as an option
# ---------------------------------------------------------------------------
TRUE_OU_KAPPA = 0.035
TRUE_OU_THETA = 4.0
TRUE_OU_SIGMA = 3.2


def plant_margin(*, periods: int = 1_600, seed: int = 4) -> pd.Series:
    """Crush margin as an Ornstein-Uhlenbeck process, true parameters known.

    Exact simulation of the discrete process:
        M_{t+1} = theta + (M_t - theta) e^{-kappa dt} + eps,
        eps ~ N(0, sigma sqrt((1 - e^{-2 kappa dt}) / (2 kappa)))
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2019-01-02", periods=periods, freq="B")
    decay = np.exp(-TRUE_OU_KAPPA)
    innovation_std = TRUE_OU_SIGMA * np.sqrt((1.0 - decay**2) / (2.0 * TRUE_OU_KAPPA))

    values = np.empty(periods)
    values[0] = TRUE_OU_THETA
    shocks = rng.normal(scale=innovation_std, size=periods)
    for i in range(1, periods):
        values[i] = TRUE_OU_THETA + (values[i - 1] - TRUE_OU_THETA) * decay + shocks[i]
    return pd.Series(values, index=index, name="SYNTH_CRUSH_MARGIN_USD_BU")


# ---------------------------------------------------------------------------
# T2-6 — inter-oil substitution
# ---------------------------------------------------------------------------
# True half-lives: -ln(2) / ln(1 + beta).
# The narrow regime must stay reversive enough for the process to **oscillate** around
# zero within the sample. With a 173-day half-life over 1,600 days, a single long
# excursion dominated the path, the realised median fell to -88 instead of 0, and the
# regime classification flipped. A test fixture must be stationary at the sample's
# scale, not just in theory.
TRUE_BETA_NARROW = -0.015      # narrow spread: slow reversion    (half-life ~45.9 d)
TRUE_BETA_WIDE = -0.120        # wide spread:   fast reversion    (half-life ~5.4 d)
# Threshold calibrated so the wide regime holds enough observations to be estimable.
# At 90 $/t against a realised std dev of ~41, the wide regime covered only 2.8% of
# the sample (n=60): beta came out close (-0.125 for a true -0.120) but with p=0.30,
# so unusable. At 60 $/t it covers ~14%.
SUBSTITUTION_THRESHOLD_USD_T = 60.0
SUBSTITUTION_SHOCK_STD = 10.0


def oil_prices(*, periods: int = 2_500, seed: int = 5) -> dict[str, pd.Series]:
    """Palm, soy and canola in USD/t, with a threshold palm-soy spread.

    The spread follows a **threshold** AR(1): slow while it stays within a +/- 90 $/t
    band around its mean, fast beyond it. This is the substitution bound
    `substitution_bound` must recover.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2019-01-02", periods=periods, freq="B")

    soy = pd.Series(
        980.0 * np.exp(np.cumsum(rng.normal(scale=0.010, size=periods))),
        index=index,
        name="SYNTH_SOYOIL_USD_T",
    )

    spread = np.empty(periods)
    spread[0] = 0.0
    shocks = rng.normal(scale=SUBSTITUTION_SHOCK_STD, size=periods)
    for i in range(1, periods):
        beta = TRUE_BETA_WIDE if abs(spread[i - 1]) > SUBSTITUTION_THRESHOLD_USD_T else TRUE_BETA_NARROW
        spread[i] = spread[i - 1] + beta * spread[i - 1] + shocks[i]

    # No structural palm-soy discount here, deliberately: the process's regime switches
    # on |spread| > threshold around ZERO, and `build_spreads` returns `palm - soy`.
    # Adding a -120 offset would centre the observed series on -120 while the regime
    # stayed defined around 0: the two centres would diverge and the regime
    # classification would be wrong with nothing to flag it.
    palm = pd.Series((soy + spread).to_numpy(), index=index, name="SYNTH_PALM_USD_T")
    rape = pd.Series(
        (soy + 60.0 + np.cumsum(rng.normal(scale=3.0, size=periods))).to_numpy(),
        index=index,
        name="SYNTH_RAPEOIL_USD_T",
    )
    return {"palm": palm, "soy": soy, "canola": rape}
