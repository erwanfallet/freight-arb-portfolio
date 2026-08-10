"""Golden tests des conversions d'unités.

Chaque valeur attendue est calculée à la main dans le commentaire qui la précède.
Aucune n'est un print du code relu après coup — c'est la règle du repo, et c'est
particulièrement critique ici : un facteur de conversion faux ne plante jamais, il
produit une page cohérente et fausse.
"""
from __future__ import annotations

import pandas as pd
import pytest

from agri.core.units import (
    LB_PER_TONNE,
    UnitError,
    add_vat,
    board_crush_usd_bu,
    bushels_per_tonne,
    cents_lb_to_usd_t,
    gallons_to_litres,
    kg_to_lb,
    lb_to_kg,
    local_to_usd_per_t,
    raw_sugar_to_white_basis,
    strip_vat,
    usd_bu_to_usd_t,
    usd_short_ton_to_usd_t,
    usd_t_to_cents_lb,
    usd_t_to_usd_bu,
    usd_t_to_usd_short_ton,
)


# ---------------------------------------------------------------------------
# Boisseaux : le facteur dépend de la graine (U-H1)
# ---------------------------------------------------------------------------
def test_bushels_per_tonne_soybean():
    # 2204.62262 lb/t / 60 lb/bu = 36.743710...
    assert bushels_per_tonne("soybean") == pytest.approx(36.7437103, abs=1e-6)


def test_bushels_per_tonne_corn_differs_from_soybean():
    # 2204.62262 / 56 = 39.368261...  Le maïs et le soja n'ont PAS le même facteur :
    # c'est exactement l'erreur que ce module existe pour rendre impossible.
    assert bushels_per_tonne("corn") == pytest.approx(39.3682611, abs=1e-6)
    assert bushels_per_tonne("corn") > bushels_per_tonne("soybean")


def test_bushels_per_tonne_oats():
    # 2204.62262 / 32 = 68.894457...
    assert bushels_per_tonne("oats") == pytest.approx(68.8944569, abs=1e-6)


def test_wheat_shares_the_soybean_test_weight():
    # blé et soja sont tous deux à 60 lb/bu — même facteur, pour de bonnes raisons
    assert bushels_per_tonne("wheat") == bushels_per_tonne("soybean")


def test_unknown_commodity_raises_rather_than_guessing():
    with pytest.raises(UnitError, match="poids-test inconnu"):
        bushels_per_tonne("canola")


def test_commodity_name_is_case_insensitive():
    assert bushels_per_tonne("SoyBean") == bushels_per_tonne("soybean")


def test_usd_bu_to_usd_t_soybean():
    # 10.00 USD/bu * 36.7437103 = 367.437103 USD/t
    assert usd_bu_to_usd_t(10.0, "soybean") == pytest.approx(367.437103, abs=1e-4)


def test_usd_bu_round_trip():
    assert usd_t_to_usd_bu(usd_bu_to_usd_t(10.5, "corn"), "corn") == pytest.approx(10.5)


# ---------------------------------------------------------------------------
# Cents/lb et short ton
# ---------------------------------------------------------------------------
def test_cents_lb_to_usd_t():
    # 1 c/lb = 0.01 USD/lb * 2204.62262 lb/t = 22.0462262 USD/t
    assert cents_lb_to_usd_t(1.0) == pytest.approx(22.0462262, abs=1e-6)


def test_sugar_20_cents_lb():
    # 20 c/lb * 22.0462262 = 440.924524 USD/t
    assert cents_lb_to_usd_t(20.0) == pytest.approx(440.924524, abs=1e-4)


def test_cents_lb_round_trip():
    assert usd_t_to_cents_lb(cents_lb_to_usd_t(45.0)) == pytest.approx(45.0)


def test_usd_short_ton_to_usd_t():
    # 2204.62262 / 2000 = 1.10231131
    assert usd_short_ton_to_usd_t(1.0) == pytest.approx(1.10231131, abs=1e-7)


def test_meal_300_usd_short_ton_in_metric():
    # 300 * 1.10231131 = 330.693393 USD/t métrique
    assert usd_short_ton_to_usd_t(300.0) == pytest.approx(330.693393, abs=1e-4)


def test_short_ton_round_trip():
    assert usd_t_to_usd_short_ton(usd_short_ton_to_usd_t(300.0)) == pytest.approx(300.0)


def test_short_ton_is_lighter_so_price_per_metric_tonne_is_higher():
    # contrôle de sens : une short ton pèse moins qu'une tonne, donc le même prix
    # rapporté à la tonne métrique est forcément PLUS élevé. Un signe inversé ici
    # passerait tous les tests numériques d'un board crush sans jamais planter.
    assert usd_short_ton_to_usd_t(100.0) > 100.0


# ---------------------------------------------------------------------------
# Board crush — les trois unités dans une seule formule
# ---------------------------------------------------------------------------
def test_board_crush_hand_computed():
    # fève 10.50 USD/bu, tourteau 300 USD/short ton, huile 45 c/lb
    #   tourteau : 44/2000 = 0.022 short ton/bu * 300 = 6.60 USD/bu
    #   huile    : 11 lb/bu * 0.45 USD/lb          = 4.95 USD/bu
    #   crush    : 6.60 + 4.95 - 10.50            = 1.05 USD/bu
    assert board_crush_usd_bu(10.50, 300.0, 45.0) == pytest.approx(1.05, abs=1e-10)


def test_board_crush_treating_meal_as_metric_is_materially_wrong():
    """Le piège, chiffré : traiter le tourteau comme s'il était en tonnes métriques.

    Coefficient faux : 44 lb/bu / 2204.62262 = 0.0199580 au lieu de 0.022.
    Sur 300 USD : 5.987 au lieu de 6.60, soit 0.613 USD/bu d'erreur — la moitié du
    crush lui-même (1.05). Une erreur qui ne plante pas et qui décide du trade.
    """
    correct = board_crush_usd_bu(10.50, 300.0, 45.0)
    wrong_meal_coefficient = 44.0 / LB_PER_TONNE
    wrong = wrong_meal_coefficient * 300.0 + 11.0 * 0.45 - 10.50
    assert correct - wrong == pytest.approx(0.6127, abs=1e-3)
    assert abs(correct - wrong) > 0.5 * correct


def test_board_crush_accepts_series():
    beans = pd.Series([10.50, 11.00])
    meal = pd.Series([300.0, 310.0])
    oil = pd.Series([45.0, 46.0])
    out = board_crush_usd_bu(beans, meal, oil)
    # deuxième date : 0.022*310 + 11*0.46 - 11.00 = 6.82 + 5.06 - 11.00 = 0.88
    assert out.iloc[0] == pytest.approx(1.05)
    assert out.iloc[1] == pytest.approx(0.88, abs=1e-10)


# ---------------------------------------------------------------------------
# Masse, volume
# ---------------------------------------------------------------------------
def test_lb_kg_round_trip():
    assert kg_to_lb(lb_to_kg(100.0)) == pytest.approx(100.0)


def test_gallon_to_litre():
    assert gallons_to_litres(1.0) == pytest.approx(3.785411784, abs=1e-9)


# ---------------------------------------------------------------------------
# Change et TVA
# ---------------------------------------------------------------------------
def test_local_to_usd_uses_local_per_usd_convention():
    # 5400 BRL/t à USDBRL = 5.4 -> 1000 USD/t
    assert local_to_usd_per_t(5400.0, 5.4) == pytest.approx(1000.0)


def test_negative_fx_raises():
    with pytest.raises(UnitError, match="taux de change"):
        local_to_usd_per_t(5400.0, -5.4)


def test_strip_vat_chinese_agricultural_rate():
    # 109 TTC à 9 % -> 100 HT, exactement
    assert strip_vat(109.0, 0.09) == pytest.approx(100.0)


def test_vat_round_trip():
    assert strip_vat(add_vat(100.0, 0.13), 0.13) == pytest.approx(100.0)


def test_vat_rate_given_as_percent_instead_of_fraction_raises():
    # 9 au lieu de 0.09 : l'erreur la plus probable, et elle doit être bruyante
    with pytest.raises(UnitError, match="taux de TVA"):
        strip_vat(109.0, 9.0)


# ---------------------------------------------------------------------------
# Sucre : polarisation (U-H2)
# ---------------------------------------------------------------------------
def test_raw_sugar_to_white_basis():
    # 20 c/lb -> 440.924524 USD/t, * 1.07 = 471.789241 USD/t base blanc
    assert raw_sugar_to_white_basis(20.0, pol_adjust=1.07) == pytest.approx(471.789241, abs=1e-4)


def test_pol_adjust_out_of_range_raises():
    # 1.5 n'est plus une correction de polarisation, c'est une faute de frappe
    with pytest.raises(UnitError, match="pol_adjust"):
        raw_sugar_to_white_basis(20.0, pol_adjust=1.5)


def test_pol_adjust_moves_white_premium_by_a_material_amount():
    """Pourquoi pol_adjust est un slider et pas une constante.

    Entre 1.06 et 1.08, sur du sucre à 20 c/lb, l'écart est de 8.82 USD/t — du même
    ordre que le white premium qu'on cherche à décomposer dans T2-4. Figer ce facteur
    déciderait du résultat de la page.
    """
    low = raw_sugar_to_white_basis(20.0, pol_adjust=1.06)
    high = raw_sugar_to_white_basis(20.0, pol_adjust=1.08)
    assert high - low == pytest.approx(8.8185, abs=1e-3)
