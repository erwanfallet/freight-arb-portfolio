"""Golden tests T3-2 — le plancher de coût qui n'en est pas un, sur NY11 et USDBRL réels.

Le test central est `test_the_floor_moves_by_twenty_cents_with_no_cost_change` : le coût de
production est **tenu constant par construction** dans tout le calcul, donc l'amplitude du
plancher en cents/lb ne peut venir que du change. C'est le résultat de la page, et il est
d'autant plus solide qu'il ne repose sur aucune estimation.

`test_czarnikow_claim_holds_on_real_prices` vérifie une affirmation publiée et datée plutôt
que de la citer. Si elle cessait d'être vraie, la section S2 de la page devrait être relue.
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
    not DEFAULT_PATH.exists(), reason=f"fichier Bloomberg absent : {DEFAULT_PATH}"
)

START = "2015-01-01"


@pytest.fixture(scope="module")
def frame():
    return load_real_parity_frame(START)


# ===========================================================================
# LE RÉSULTAT
# ===========================================================================
def test_the_floor_moves_by_twenty_cents_with_no_cost_change(frame):
    """LE test de la page.

    Le coût de production passé au calcul est un scalaire : il ne varie pas d'un jour à
    l'autre, par construction. Toute l'amplitude du plancher en cents/lb vient donc du seul
    USDBRL. Vingt cents sur un marché qui cote entre 10 et 25 cents, ce n'est pas un
    ajustement — c'est plus que la fourchette du marché lui-même.
    """
    floor = moving_floor(frame, cost_brl_t=CZARNIKOW_COST_BRL_T)
    assert floor.floor_range > 15.0
    assert floor.floor_min < 16.0 < floor.floor_max
    # le plancher est EXACTEMENT proportionnel a l'inverse du change
    product = floor.frame["floor_c_lb"] * floor.frame["usdbrl"]
    assert product.std() == pytest.approx(0.0, abs=1e-9)


def test_the_floor_is_nothing_but_a_rescaled_exchange_rate(frame):
    """Formulation forte du même fait : la corrélation de rang entre le plancher et
    l'inverse du change vaut exactement 1. Il n'y a aucune information brésilienne dedans."""
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
# L'affirmation sourcée
# ===========================================================================
def test_czarnikow_claim_holds_on_real_prices(frame):
    """Czarnikow (juin 2026) : le pricing 2026/27 est resté sous BRL 2 000/t, sous le coût
    de production. Vérifié plutôt que cité."""
    check = production_cost_check(frame, cost_brl_t=CZARNIKOW_COST_BRL_T)
    assert check.is_below_now
    assert check.last_brl_t < CZARNIKOW_COST_BRL_T
    recent = frame[frame.index >= "2026-01-01"]
    assert (recent["sugar_brl_t"] < CZARNIKOW_COST_BRL_T).mean() > 0.80


def test_sugar_in_brl_hand_computed(frame):
    """sucre_BRL_t = NY11 x 22,0462 x USDBRL — trois nombres, aucune hypothèse."""
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
# L'inversion vers l'éthanol
# ===========================================================================
def test_the_indifference_price_inverts_the_parity_exactly(frame):
    """Contrôle croisé : repasser le prix d'indifférence dans la conversion directe doit
    redonner le NY11 ajusté. Si les deux ne se répondent pas, l'une des deux est fausse."""
    hydrous = indifference_hydrous_brl_l(frame["ny11"], frame["usdbrl"])
    back = hydrous_sugar_equivalent_cents_lb(hydrous, frame["usdbrl"])
    expected = frame["ny11"] * DEFAULT_POL_FACTOR
    pd.testing.assert_series_equal(
        back.rename(None), expected.rename(None), rtol=1e-10
    )


def test_the_indifference_price_is_a_plausible_ethanol_level(frame):
    """Un hydraté brésilien se traite entre 1 et 4 BRL/litre selon l'époque. Une inversion
    qui sortirait de cette plage signalerait une erreur dans la chaîne de conversion."""
    hydrous = indifference_hydrous_brl_l(frame["ny11"], frame["usdbrl"])
    assert 0.5 < hydrous.median() < 5.0
    assert (hydrous > 0).all()


def test_the_indifference_price_hand_computed():
    """NY11 20 c/lb, pol 0,98, USDBRL 5,0 :
    20 x 0,98 x 5,0 x 2,20462 x (1,6913 / 1,0495) / 100 = 3,4816 BRL/litre."""
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
# L'asymétrie de S4
# ===========================================================================
def test_the_exchange_rate_partially_cushions_the_price_but_not_the_floor(frame):
    """L'argument de S4. Le prix reçu en réaux bénéficie d'une corrélation négative entre le
    sucre en dollars et le change ; le plancher, exactement proportionnel à l'inverse du
    change, n'en bénéficie d'aucune."""
    decomposition = floor_variance_decomposition(frame)
    assert decomposition["correlation"] < 0
    assert decomposition["share_covariance"] < 0
    assert decomposition["share_sugar"] > decomposition["share_fx"]
    assert decomposition[["share_sugar", "share_fx", "share_covariance"]].sum() == pytest.approx(1.0)


def test_the_decomposition_refuses_a_short_sample(frame):
    with pytest.raises(SugarMixError, match="too short"):
        floor_variance_decomposition(frame.head(10))


# ===========================================================================
# Chargement
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
    """L'USDBRL d'avant juillet 1994 cote des cruzeiros — une autre monnaie. Le loader
    l'exclut ; ce test garde cette exclusion depuis l'aval, là où un cruzeiro produirait un
    plancher absurde de plusieurs dizaines de milliers de cents."""
    full = load_real_parity_frame(None)
    assert full.index.min() >= pd.Timestamp("1994-07-01")
    assert moving_floor(full).floor_max < 1_000.0
