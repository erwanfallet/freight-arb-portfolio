"""Golden tests T2-3 — le rendement implicite du board crush, sur données CBOT réelles.

Le test central est `test_the_precision_demanded_collapses_when_the_margin_tightens` : c'est
le résultat de la page, et il ne porte pas sur un niveau moyen mais sur la **dépendance au
régime**. Un test qui se contenterait de vérifier la médiane laisserait passer une refonte
qui casse précisément ce qui rend la page intéressante.

`test_the_exposure_is_largest_when_the_margin_is_widest` garde le résultat contre-intuitif
de S4 : mon intuition initiale (« la position grossit quand la marge se resserre, donc les
deux problèmes se composent ») était **fausse**, et la page le dit. Si une refonte future
rétablissait l'intuition confortable, ce test échouerait.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agri.chains.crush_tracking import (
    CBOT_MEAL_LB_BU,
    CBOT_OIL_LB_BU,
    CrushError,
    hedge_ratio_identity_bias,
    load_real_board_frame,
    required_yield_precision,
    yield_exposure,
)
from agri.data.bloomberg_loader import DEFAULT_PATH

pytestmark = pytest.mark.skipif(
    not DEFAULT_PATH.exists(), reason=f"fichier Bloomberg absent : {DEFAULT_PATH}"
)


@pytest.fixture(scope="module")
def frame():
    return load_real_board_frame("2015-01-01")


# ===========================================================================
# Le board crush lui-même
# ===========================================================================
def test_board_crush_on_2026_08_07_hand_computed(frame):
    """0,022 x 308,10 + 0,11 x 68,160 - 11,5650 = 6,7782 + 7,4976 - 11,5650 = 2,7108."""
    row = frame.loc[pd.Timestamp("2026-08-07")]
    assert row["meal"] == pytest.approx(308.10)
    assert row["oil"] == pytest.approx(68.160)
    assert row["bean"] == pytest.approx(11.5650)
    assert row["board"] == pytest.approx(2.7108, abs=1e-6)


def test_the_board_crush_never_goes_negative(frame):
    """Même signature qu'en T2-5 : le board ne porte aucun opex, donc il ne descend pas sous
    zéro. C'est une propriété du contrat, pas de l'économie d'une usine — et c'est ce qui
    justifie que toute la page raisonne en marge NETTE."""
    assert frame["board"].min() > 0
    assert (frame["board"] < 0).sum() == 0


def test_a_plausible_opex_pushes_the_margin_below_zero_sometimes(frame):
    """Contraste avec le test précédent : l'opex est ce qui transforme un board toujours
    positif en une marge qui passe réellement sous l'eau."""
    assert ((frame["board"] - 0.70) < 0).mean() > 0.05


# ===========================================================================
# LE LIVRABLE — la précision exigée
# ===========================================================================
def test_required_precision_hand_computed(frame):
    """Au 07/08/2026, opex 0,55 : (2,7108 - 0,55) / (308,10 / 2000) = 2,1608 / 0,15405."""
    precision = required_yield_precision(frame, opex_usd_bu=0.55)
    row = precision.frame.loc[pd.Timestamp("2026-08-07")]
    assert row["net_margin"] == pytest.approx(2.1608, abs=1e-6)
    assert row["position_per_lb"] == pytest.approx(0.15405, abs=1e-9)
    assert row["breakeven_lb"] == pytest.approx(2.1608 / 0.15405, rel=1e-12)


def test_the_precision_demanded_collapses_when_the_margin_tightens(frame):
    """LE test de la page.

    L'exigence de précision n'est pas un niveau, c'est une fonction du régime. Entre le
    décile de marge le plus large et le plus tendu, elle est divisée par plus de dix — et
    dans le décile tendu elle tombe sous 1 livre, soit moins de 2,5 % du rendement que le
    contrat suppose.
    """
    precision = required_yield_precision(frame, opex_usd_bu=0.55)
    assert precision.tight_decile_lb < 1.0
    assert precision.wide_decile_lb > 10.0
    assert precision.wide_decile_lb / precision.tight_decile_lb > 10.0
    assert precision.tight_decile_pct < 0.025


def test_one_pound_wipes_the_margin_on_a_material_share_of_days(frame):
    """Le chiffre du mail : une livre d'écart — 2,3 % du rendement du contrat — efface la
    marge nette entière sur une part non anecdotique de l'échantillon."""
    precision = required_yield_precision(frame, opex_usd_bu=0.55)
    assert precision.share_below(1.0) > 0.05
    assert precision.share_below(2.0) > precision.share_below(1.0)
    assert precision.share_below(20.0) > 0.90


def test_precision_is_monotone_in_opex(frame):
    """Plus l'opex est lourd, plus la marge nette est mince, donc plus l'exigence est dure."""
    medians = [
        required_yield_precision(frame, opex_usd_bu=value).median_lb
        for value in (0.30, 0.50, 0.70, 0.90)
    ]
    assert medians == sorted(medians, reverse=True)


def test_required_precision_rejects_a_frame_without_the_board(frame):
    with pytest.raises(CrushError, match="board"):
        required_yield_precision(frame[["bean", "meal"]], opex_usd_bu=0.55)


# ===========================================================================
# La position
# ===========================================================================
def test_yield_exposure_hand_computed(frame):
    """1 lb/bu au 07/08/2026 : 308,10 / 2000 = 0,15405 USD/bu."""
    exposure = yield_exposure(frame, meal_lb_gap=1.0, oil_lb_gap=0.0, opex_usd_bu=0.55)
    row = exposure.frame.loc[pd.Timestamp("2026-08-07")]
    assert row["meal_leg"] == pytest.approx(0.15405, abs=1e-9)
    assert row["oil_leg"] == 0.0
    assert row["position_usd_bu"] == pytest.approx(0.15405, abs=1e-9)


def test_the_oil_leg_uses_the_cents_to_dollar_conversion(frame):
    """Piège d'unité : l'huile cote en cents par livre. 1 lb/bu au 07/08/2026 vaut
    68,160 / 100 = 0,6816 USD/bu, pas 68,16."""
    exposure = yield_exposure(frame, meal_lb_gap=0.0, oil_lb_gap=1.0, opex_usd_bu=0.55)
    assert exposure.frame.loc[pd.Timestamp("2026-08-07"), "oil_leg"] == pytest.approx(0.6816)


def test_exposure_is_linear_in_the_gap(frame):
    """C'est ce qui autorise à parler de « position » : la valeur est le produit d'une
    quantité par un prix, donc strictement proportionnelle à la quantité."""
    one = yield_exposure(frame, meal_lb_gap=1.0, opex_usd_bu=0.55).position_median
    three = yield_exposure(frame, meal_lb_gap=3.0, opex_usd_bu=0.55).position_median
    assert three == pytest.approx(3.0 * one, rel=1e-12)


def test_a_negative_gap_flips_the_position(frame):
    positive = yield_exposure(frame, meal_lb_gap=1.0, opex_usd_bu=0.55).position_median
    negative = yield_exposure(frame, meal_lb_gap=-1.0, opex_usd_bu=0.55).position_median
    assert negative == pytest.approx(-positive, rel=1e-12)


def test_an_implausible_gap_is_rejected(frame):
    with pytest.raises(CrushError, match="implausible"):
        yield_exposure(frame, meal_lb_gap=60.0, opex_usd_bu=0.55)


def test_exposure_rejects_a_frame_missing_a_leg(frame):
    with pytest.raises(CrushError, match="missing column"):
        yield_exposure(frame[["bean", "board"]], meal_lb_gap=1.0)


# ===========================================================================
# LE RÉSULTAT CONTRE-INTUITIF — garde-fou contre le confort
# ===========================================================================
def test_the_exposure_is_largest_when_the_margin_is_widest(frame):
    """Mon intuition de départ était que la position nue grossit quand la marge se resserre,
    ce qui composerait les deux problèmes. La donnée dit l'inverse : le tourteau est la
    principale recette du crush, donc son prix est POSITIVEMENT corrélé à la marge nette.

    La page annonce ce résultat au lieu de le cacher. Ce test empêche qu'une refonte
    ultérieure rétablisse silencieusement l'histoire plus confortable.
    """
    net_margin = frame["board"] - 0.55
    correlation = net_margin.corr(frame["meal"], method="spearman")
    assert correlation > 0.15, (
        "la corrélation marge/tourteau est devenue négative ou nulle : la section S4 de la "
        "page affirme le contraire et doit être relue"
    )

    tight = frame.loc[net_margin <= net_margin.quantile(0.10), "meal"].median()
    wide = frame.loc[net_margin >= net_margin.quantile(0.90), "meal"].median()
    assert wide > tight


def test_what_degrades_in_the_tight_regime_is_the_ratio_not_the_position(frame):
    """La formulation exacte que la page retient : en régime tendu la position est petite en
    dollars, mais la marge l'est encore plus. Une usine qui surveille son exposition en
    dollars absolus ne voit rien venir."""
    net_margin = frame["board"] - 0.55
    exposure = yield_exposure(frame, meal_lb_gap=1.0, opex_usd_bu=0.55)
    position = exposure.frame["position_usd_bu"]

    tight = net_margin <= net_margin.quantile(0.10)
    wide = net_margin >= net_margin.quantile(0.90)

    assert position[tight].median() < position[wide].median()          # position plus petite
    assert (position / net_margin.clip(lower=0.05))[tight].median() > (
        position / net_margin.clip(lower=0.05)
    )[wide].median()                                                   # ratio bien pire


# ===========================================================================
# Le piège d'identité comptable, hérité de T2-1
# ===========================================================================
def test_the_identity_bias_is_positive_and_scales_with_the_gap(frame):
    betas = [
        hedge_ratio_identity_bias(frame, meal_lb_gap=gap, opex_usd_bu=0.55).beta_naive
        for gap in (0.5, 1.0, 2.0, 4.0)
    ]
    assert all(b > 1.0 for b in betas)
    assert betas == sorted(betas)
    # linearite : le biais double quand l'ecart double
    biases = [b - 1.0 for b in betas]
    assert biases[2] == pytest.approx(2 * biases[1], rel=1e-6)


def test_the_identity_bias_is_small_and_the_page_says_so(frame):
    """Garde-fou d'honnêteté. La page annonce explicitement que cette contamination est de
    l'ordre du pourcent et que ce n'est PAS l'argument. Si le biais devenait grand, la
    formulation de S5 serait à revoir — ce test le signalerait."""
    bias = hedge_ratio_identity_bias(frame, meal_lb_gap=1.0, opex_usd_bu=0.55)
    assert 0.0 < bias.bias < 0.05
    assert "only" in bias.headline
    assert "yield" in bias.headline


def test_a_zero_gap_leaves_no_bias_at_all(frame):
    """Contrôle : sans écart de rendement, la marge d'usine EST le board moins une
    constante, donc la régression rend exactement 1."""
    bias = hedge_ratio_identity_bias(frame, meal_lb_gap=0.0, oil_lb_gap=0.0, opex_usd_bu=0.55)
    assert bias.beta_naive == pytest.approx(1.0, abs=1e-12)
    assert bias.bias == pytest.approx(0.0, abs=1e-12)


def test_identity_bias_refuses_a_short_sample(frame):
    with pytest.raises(CrushError, match="too short"):
        hedge_ratio_identity_bias(frame.head(10), meal_lb_gap=1.0)


# ===========================================================================
# Chargement
# ===========================================================================
def test_load_real_board_frame_has_the_three_legs_and_the_board(frame):
    assert list(frame.columns) == ["bean", "meal", "oil", "board"]
    assert len(frame) > 2_000
    assert frame.index.is_monotonic_increasing
    assert not frame.isna().any().any()


def test_the_board_coefficients_are_the_cbot_yields():
    """Le cœur du sujet de la page : 0,022 et 0,11 ne sont pas des conversions d'unité mais
    des rendements. Ce test l'énonce en code."""
    assert CBOT_MEAL_LB_BU / 2000.0 == pytest.approx(0.022)
    assert CBOT_OIL_LB_BU / 100.0 == pytest.approx(0.11)


def test_an_impossible_start_date_raises():
    with pytest.raises(CrushError, match="no common date"):
        load_real_board_frame("2099-01-01")
