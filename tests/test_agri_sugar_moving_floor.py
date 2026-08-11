"""Golden tests T3-2 — the cost floor that isn't one, on real NY11 and USDBRL.

The central test is `test_the_floor_moves_by_twenty_cents_with_no_cost_change`: the
production cost is **held constant by construction** throughout the calculation, so the
floor's amplitude in cents/lb can only come from FX. That's the page's result, and it's
all the more solid for resting on no estimation at all.

`test_czarnikow_claim_holds_on_real_prices` verifies a published, dated claim rather
than just citing it. If it stopped being true, the page's S2 section would need a
rewrite.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agri.chains.sugar_mix import (
    ATR_ETHANOL_HYDROUS_PER_L,
    ATR_SUGAR_VHP_PER_KG,
    CENTS_LB_TO_USD_T,
    CZARNIKOW_COST_BRL_T,
    DEFAULT_POL_FACTOR,
    KG_PER_LB,
    SugarMixError,
    floor_variance_decomposition,
    hydrous_sugar_equivalent_cents_lb,
    indifference_hydrous_brl_l,
    load_real_parity_frame,
    moving_floor,
    production_cost_check,
)
from agri.data.bloomberg_loader import DEFAULT_PATH

pytestmark = pytest.mark.skipif(
    not DEFAULT_PATH.exists(), reason=f"Bloomberg file absent: {DEFAULT_PATH}"
)

START = "2015-01-01"


@pytest.fixture(scope="module")
def frame():
    return load_real_parity_frame(START)


# ===========================================================================
# THE RESULT
# ===========================================================================
def test_the_floor_moves_by_twenty_cents_with_no_cost_change(frame):
    """THE page's test.

    The production cost fed into the calculation is a scalar: it doesn't vary from one
    day to the next, by construction. The floor's entire amplitude in cents/lb
    therefore comes from USDBRL alone. Twenty cents on a market that trades between 10
    and 25 cents isn't an adjustment — it's more than the market's own range.
    """
    floor = moving_floor(frame, cost_brl_t=CZARNIKOW_COST_BRL_T)
    assert floor.floor_range > 15.0
    assert floor.floor_min < 16.0 < floor.floor_max
    # the floor is EXACTLY proportional to the inverse exchange rate
    product = floor.frame["floor_c_lb"] * floor.frame["usdbrl"]
    assert product.std() == pytest.approx(0.0, abs=1e-9)


def test_the_floor_is_nothing_but_a_rescaled_exchange_rate(frame):
    """Strong statement of the same fact: the rank correlation between the floor and
    the inverse exchange rate is exactly 1. There is no Brazilian information in it."""
    floor = moving_floor(frame, cost_brl_t=CZARNIKOW_COST_BRL_T)
    inverse_fx = 1.0 / floor.frame["usdbrl"]
    assert floor.frame["floor_c_lb"].corr(inverse_fx, method="spearman") == pytest.approx(1.0)


def test_a_higher_cost_lifts_the_whole_floor_proportionally(frame):
    low = moving_floor(frame, cost_brl_t=1500.0)
    high = moving_floor(frame, cost_brl_t=3000.0)
    ratio = high.frame["floor_c_lb"] / low.frame["floor_c_lb"]
    assert ratio.std() == pytest.approx(0.0, abs=1e-12)
    assert ratio.iloc[0] == pytest.approx(2.0)


# ===========================================================================
# The sourced claim
# ===========================================================================
def test_czarnikow_claim_holds_on_real_prices(frame):
    """Czarnikow (June 2026): 2026/27 pricing stayed below BRL 2,000/t, below the cost
    of production. Verified rather than cited."""
    check = production_cost_check(frame, cost_brl_t=CZARNIKOW_COST_BRL_T)
    assert check.is_below_now
    assert check.last_brl_t < CZARNIKOW_COST_BRL_T
    recent = frame[frame.index >= "2026-01-01"]
    assert (recent["sugar_brl_t"] < CZARNIKOW_COST_BRL_T).mean() > 0.80


def test_sugar_in_brl_hand_computed(frame):
    """sugar_BRL_t = NY11 x 22.0462 x USDBRL — three numbers, no assumption."""
    row = frame.iloc[-1]
    assert row["sugar_brl_t"] == pytest.approx(
        row["ny11"] * CENTS_LB_TO_USD_T * row["usdbrl"], rel=1e-12
    )
    assert CENTS_LB_TO_USD_T == pytest.approx(22.0462, abs=1e-4)


def test_production_cost_check_rejects_a_frame_without_the_brl_leg(frame):
    with pytest.raises(SugarMixError, match="sugar_brl_t"):
        production_cost_check(frame[["ny11", "usdbrl"]])


def test_a_non_positive_cost_is_rejected(frame):
    with pytest.raises(SugarMixError, match="cost of production"):
        moving_floor(frame, cost_brl_t=0.0)


# ===========================================================================
# The inversion to ethanol
# ===========================================================================
def test_the_indifference_price_inverts_the_parity_exactly(frame):
    """Cross-check: feeding the indifference price back through the direct conversion
    must return the adjusted NY11. If the two don't match, one of them is wrong."""
    hydrous = indifference_hydrous_brl_l(frame["ny11"], frame["usdbrl"])
    back = hydrous_sugar_equivalent_cents_lb(hydrous, frame["usdbrl"])
    expected = frame["ny11"] * DEFAULT_POL_FACTOR
    pd.testing.assert_series_equal(
        back.rename(None), expected.rename(None), rtol=1e-10
    )


def test_the_indifference_price_is_a_plausible_ethanol_level(frame):
    """Brazilian hydrous ethanol trades between 1 and 4 BRL/litre depending on the era.
    An inversion landing outside this range would signal an error in the conversion
    chain."""
    hydrous = indifference_hydrous_brl_l(frame["ny11"], frame["usdbrl"])
    assert 0.5 < hydrous.median() < 5.0
    assert (hydrous > 0).all()


def test_the_indifference_price_hand_computed():
    """NY11 20 c/lb, pol 0.98, USDBRL 5.0:
    20 x 0.98 x 5.0 x 2.20462 x (1.6913 / 1.0495) / 100 = 3.4816 BRL/litre."""
    index = pd.date_range("2024-01-01", periods=1)
    hydrous = indifference_hydrous_brl_l(
        pd.Series([20.0], index=index), pd.Series([5.0], index=index)
    )
    expected = (
        20.0 * 0.98 * 5.0 * KG_PER_LB * (ATR_ETHANOL_HYDROUS_PER_L / ATR_SUGAR_VHP_PER_KG) / 100.0
    )
    assert hydrous.iloc[0] == pytest.approx(expected, rel=1e-12)
    assert hydrous.iloc[0] == pytest.approx(3.4816, abs=1e-3)


def test_a_negative_exchange_rate_is_rejected():
    index = pd.date_range("2024-01-01", periods=2)
    with pytest.raises(SugarMixError, match="quoting direction"):
        indifference_hydrous_brl_l(
            pd.Series([20.0, 21.0], index=index), pd.Series([-5.0, 5.0], index=index)
        )


# ===========================================================================
# S4's asymmetry
# ===========================================================================
def test_the_exchange_rate_partially_cushions_the_price_but_not_the_floor(frame):
    """S4's argument. The price received in reais benefits from a negative correlation
    between dollar sugar and the exchange rate; the floor, exactly proportional to the
    inverse exchange rate, benefits from none of it."""
    decomposition = floor_variance_decomposition(frame)
    assert decomposition["correlation"] < 0
    assert decomposition["share_covariance"] < 0
    assert decomposition["share_sugar"] > decomposition["share_fx"]
    assert decomposition[["share_sugar", "share_fx", "share_covariance"]].sum() == pytest.approx(1.0)


def test_the_decomposition_refuses_a_short_sample(frame):
    with pytest.raises(SugarMixError, match="too short"):
        floor_variance_decomposition(frame.head(10))


# ===========================================================================
# Loading
# ===========================================================================
def test_load_real_parity_frame_shape(frame):
    assert list(frame.columns) == ["ny11", "usdbrl", "sugar_brl_t"]
    assert len(frame) > 2_000
    assert frame.index.is_monotonic_increasing
    assert (frame["usdbrl"] > 0).all()


def test_an_impossible_start_date_raises():
    with pytest.raises(SugarMixError, match="no common date"):
        load_real_parity_frame("2099-01-01")


def test_the_brl_era_guard_is_active(frame):
    """USDBRL before July 1994 quotes cruzeiros — a different currency. The loader
    excludes it; this test guards that exclusion from downstream, where a cruzeiro
    would produce an absurd floor of several tens of thousands of cents."""
    full = load_real_parity_frame(None)
    assert full.index.min() >= pd.Timestamp("1994-07-01")
    assert moving_floor(full).floor_max < 1_000.0
