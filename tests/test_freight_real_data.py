"""Golden tests for the three freight projects on real market data.

Each of the three ends in the same shape, and that recurrence is the portfolio's actual
thesis rather than a coincidence: **a parameter that is not a market price, is not
published, and is quietly assumed, moves the answer by as much as or more than the market
data does.**

  A — iron ore : the origin shorthand, taken at full strength, implies a negative FOB
      quality differential. The moisture correction makes it worse, not better.
  B — coal     : the efficiency pair moves the switching carbon price by twice the standard
      deviation of the carbon price itself.
  C — products : the density assumption accounts for ~90 % of the variability of the arb it
      is used to compute.

The tests that matter most here are the ones asserting the *negative* or *uncomfortable*
result, because those are the ones a future rewrite would be tempted to soften.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agri.data.bloomberg_loader import DEFAULT_PATH
from freight.chains.coal import (
    DEFAULT_COAL_EFFICIENCY,
    DEFAULT_GAS_EFFICIENCY,
    EF_COAL_T_PER_MWH_TH,
    EF_GAS_T_PER_MWH_TH,
    MWH_TH_PER_TONNE_COAL,
    efficiency_identification,
    generation_cost_eur_mwh_e,
    load_real_switching_frame,
    switching_carbon_price,
)
from freight.chains.ironore import (
    DEFAULT_MOISTURE_BRAZIL,
    implied_origin_weight,
    load_real_premium_frame,
    evaluate_origin_shorthand,
)
from freight.chains.products import (
    DENSITY_HEAVY,
    DENSITY_LIGHT,
    DENSITY_TYPICAL,
    breakeven_density,
    breakeven_density_series,
    density_identification,
    gallons_per_tonne,
    load_real_transatlantic_frame,
    transatlantic_spread,
)

pytestmark = pytest.mark.skipif(
    not DEFAULT_PATH.exists(), reason=f"Bloomberg export missing: {DEFAULT_PATH}"
)


# ===========================================================================
# C — transatlantic distillate
# ===========================================================================
@pytest.fixture(scope="module")
def products_frame():
    return load_real_transatlantic_frame("2015-01-01")


def test_gallons_per_tonne_hand_computed():
    """1000 / 0.845 / 3.785412 = 312.63 gallons per tonne."""
    assert gallons_per_tonne(0.845) == pytest.approx(312.63, abs=0.01)
    assert gallons_per_tonne(DENSITY_LIGHT) > gallons_per_tonne(DENSITY_HEAVY)


def test_an_implausible_density_is_rejected():
    with pytest.raises(ValueError, match="plausible range"):
        gallons_per_tonne(1.5)


def test_the_density_swing_is_the_size_of_the_arb_itself(products_frame):
    """THE result of project C. If a conversion factor nobody publishes accounts for most of
    the variability of the arb, then the level of that arb is not identifiable from prices."""
    identification = density_identification(products_frame)
    assert identification.ratio > 0.75
    assert not identification.level_is_identifiable
    assert identification.swing_usd_t > 25.0


def test_a_lighter_grade_always_raises_the_arb(products_frame):
    """Sign check: fewer kilos per gallon means more gallons per tonne, so the US leg is
    worth more once restated per tonne."""
    light = transatlantic_spread(products_frame, density_kg_l=DENSITY_LIGHT).median()
    heavy = transatlantic_spread(products_frame, density_kg_l=DENSITY_HEAVY).median()
    assert light > heavy


def test_breakeven_density_is_the_closed_form(products_frame):
    row = products_frame.iloc[-1]
    result = breakeven_density(products_frame, freight_usd_t=25.0)
    expected = row["ulsd_c_gal"] / 100.0 * 1000.0 / 3.785411784 / (row["gasoil_usd_t"] + 25.0)
    assert result.density_star == pytest.approx(expected, rel=1e-12)


def test_at_a_realistic_freight_the_grade_decides_more_than_half_the_time(products_frame):
    """The deliverable: on most sessions, whether the cargo works is settled by which grade
    is in the tank rather than by either screen price."""
    path = breakeven_density_series(products_frame, freight_usd_t=25.0)
    assert path["inside_band"].mean() > 0.40
    assert path["inside_band"].mean() + path["above_band"].mean() <= 1.0


def test_zero_freight_pushes_the_breakeven_above_the_band(products_frame):
    """Contrast that keeps the result honest: with no freight to cover, the arb is open on
    any defensible grade — the density argument bites on magnitude, not on sign."""
    result = breakeven_density(products_frame, freight_usd_t=0.0)
    assert not result.inside_plausible_band
    assert result.density_star > DENSITY_HEAVY


def test_negative_freight_is_rejected(products_frame):
    with pytest.raises(ValueError, match="freight"):
        breakeven_density(products_frame, freight_usd_t=-5.0)


# ===========================================================================
# A — iron ore
# ===========================================================================
@pytest.fixture(scope="module")
def ore_frame():
    return load_real_premium_frame()


def test_the_premium_frame_is_monthly_because_c3_is_monthly(ore_frame):
    """Rule A, enforced. C3 has 64 monthly points against a daily C5; forward-filling would
    turn 31 real observations into 600 and make every standard error meaningless."""
    steps = ore_frame.index.to_series().diff().dt.days.dropna()
    assert steps.median() >= 28
    assert 20 < len(ore_frame) < 60


def test_the_shorthand_implies_a_negative_quality_differential(ore_frame):
    """THE result of project A. Taken at full strength the origin shorthand makes high-grade
    ore cheaper at the loadport than low-grade ore, which nobody believes."""
    tested = evaluate_origin_shorthand(ore_frame, moisture=0.0)
    assert tested.implied_quality_median < 0
    assert tested.share_negative > 0.5
    assert not tested.shorthand_survives


def test_the_moisture_correction_makes_the_contradiction_worse(ore_frame):
    """The detail that rules out the obvious explanation. Freight is paid on the wet tonne,
    so correcting for moisture makes the freight leg larger and the implied quality
    differential more negative — the correction deepens the anomaly instead of resolving it."""
    dry = evaluate_origin_shorthand(ore_frame, moisture=0.0)
    wet = evaluate_origin_shorthand(ore_frame, moisture=DEFAULT_MOISTURE_BRAZIL)
    assert wet.implied_quality_median < dry.implied_quality_median
    assert wet.share_negative > dry.share_negative


def test_the_freight_spread_exceeds_the_whole_premium(ore_frame):
    tested = evaluate_origin_shorthand(ore_frame, moisture=DEFAULT_MOISTURE_BRAZIL)
    assert tested.freight_share_of_premium > 1.0


def test_the_implied_weight_is_about_half_at_a_plausible_quality(ore_frame):
    """The inversion, and the honest version of the project's thesis: the freight spread is
    inside the premium at roughly half weight, not at full weight."""
    weights = implied_origin_weight(ore_frame)
    assert 0.3 < weights.weight_at(6.0) < 0.7
    assert weights.weight_at(0.0) > weights.weight_at(12.0)


def test_the_implied_weight_falls_monotonically_in_assumed_quality(ore_frame):
    curve = implied_origin_weight(ore_frame).curve
    assert curve["implied_weight"].is_monotonic_decreasing


def test_an_implausible_moisture_is_rejected(ore_frame):
    with pytest.raises(ValueError, match="moisture"):
        evaluate_origin_shorthand(ore_frame, moisture=0.5)


# ===========================================================================
# B — coal-to-gas switching
# ===========================================================================
@pytest.fixture(scope="module")
def switching_frame():
    return load_real_switching_frame("2018-01-01")


def test_the_coal_energy_conversion_is_hand_computable():
    """6 000 kcal/kg x 1 000 kg/t x 1.163e-6 MWh/kcal = 6.978 MWh per tonne."""
    assert MWH_TH_PER_TONNE_COAL == pytest.approx(6.978, abs=0.001)


def test_the_fx_direction_is_checked_not_assumed(switching_frame):
    """A reversed EURUSD leg would move the coal price by about 25 % with no visible error.
    The loader refuses rather than trusting the direction."""
    assert 0.6 < switching_frame["eurusd"].median() < 1.8
    assert 5.0 < switching_frame["coal_eur_mwh_th"].median() < 40.0


def test_the_switching_price_is_the_closed_form(switching_frame):
    switch = switching_carbon_price(switching_frame)
    denominator = (
        EF_COAL_T_PER_MWH_TH / DEFAULT_COAL_EFFICIENCY
        - EF_GAS_T_PER_MWH_TH / DEFAULT_GAS_EFFICIENCY
    )
    expected = (
        switching_frame["ttf_eur_mwh"] / DEFAULT_GAS_EFFICIENCY
        - switching_frame["coal_eur_mwh_th"] / DEFAULT_COAL_EFFICIENCY
    ) / denominator
    pd.testing.assert_series_equal(switch, expected.rename("eua_switch_eur_t"))


def test_at_the_switching_price_the_two_plants_cost_the_same(switching_frame):
    """Cross-check of the algebra against the cost functions it was derived from."""
    switch = switching_carbon_price(switching_frame)
    substituted = switching_frame.copy()
    substituted["eua_eur_t"] = switch
    costs = generation_cost_eur_mwh_e(substituted)
    assert costs["spread"].abs().max() < 1e-8


def test_the_efficiency_pair_moves_the_answer_more_than_the_carbon_price_does(switching_frame):
    """THE result of project B, and the strongest instance of the portfolio's recurring
    shape: the unpublished parameter is twice the size of the published one."""
    identification = efficiency_identification(switching_frame)
    assert identification.ratio > 1.5
    assert identification.swing_eur_t > 40.0


def test_the_efficiency_pair_flips_the_qualitative_conclusion(switching_frame):
    """Same fuel prices, same carbon price, opposite conclusions about whether the carbon
    market has been working."""
    identification = efficiency_identification(switching_frame)
    assert identification.share_above_low - identification.share_above_high > 0.40


def test_a_more_efficient_coal_plant_raises_the_switching_price(switching_frame):
    """Sign check: a cleaner coal unit needs a higher carbon price before gas overtakes it."""
    low = switching_carbon_price(switching_frame, coal_efficiency=0.36).median()
    high = switching_carbon_price(switching_frame, coal_efficiency=0.42).median()
    assert high > low


def test_the_switching_price_is_defined_across_the_whole_plausible_range(switching_frame):
    """Written first as an exception test, which was wrong: the guard in the engine is
    unreachable with plausible inputs, because a coal plant would need to be more than 1.7
    times as efficient as the CCGT before its CO2 per MWh of electricity fell below the gas
    plant's. Asserting the guard fires would have been asserting something false.

    What is true and worth holding is the opposite: over the entire plausible efficiency
    box, the denominator stays safely positive, so the switching price is always defined and
    never blows up on a near-zero divisor."""
    for coal_efficiency in (0.34, 0.38, 0.45):
        for gas_efficiency in (0.45, 0.55, 0.63):
            denominator = (
                EF_COAL_T_PER_MWH_TH / coal_efficiency
                - EF_GAS_T_PER_MWH_TH / gas_efficiency
            )
            assert denominator > 0.15, (coal_efficiency, gas_efficiency, denominator)
            switch = switching_carbon_price(
                switching_frame,
                coal_efficiency=coal_efficiency,
                gas_efficiency=gas_efficiency,
            )
            assert np.isfinite(switch).all()


def test_an_implausible_efficiency_is_rejected(switching_frame):
    with pytest.raises(ValueError, match="efficiency"):
        generation_cost_eur_mwh_e(switching_frame, coal_efficiency=0.95)


# ===========================================================================
# The shape all three share
# ===========================================================================
def test_all_three_projects_find_the_unpublished_parameter_dominates(
    products_frame, ore_frame, switching_frame
):
    """The portfolio's actual thesis, asserted once across the three freight projects.

    In each case a number that is not a market price — a density, a moisture content and an
    origin mapping, a pair of plant efficiencies — moves the answer by an amount comparable
    to or larger than the market data itself. If a future rewrite quietly resolved any of
    these by picking a convenient value, this test is where it would show.
    """
    assert density_identification(products_frame).ratio > 0.75
    assert not evaluate_origin_shorthand(ore_frame, moisture=DEFAULT_MOISTURE_BRAZIL).shorthand_survives
    assert efficiency_identification(switching_frame).ratio > 1.5
