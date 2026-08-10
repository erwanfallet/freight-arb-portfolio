"""Golden tests T3-1 — l'inversion en décote, et l'invariance qui la rend difficile à récuser.

Le test central de ce fichier est `test_the_discount_is_invariant_to_diesel_rin_and_plant_costs` :
il vérifie sur le modèle COMPLET (valeur en sortie d'usine, RIN, opex, ROI) ce que la forme
fermée affirme, à savoir que ces termes s'annulent entre les deux filières. C'est l'argument
de la section S2 de la page ; sans ce test, ce n'est qu'une affirmation dans une docstring.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agri.chains.feedstock_lcfs import (
    CENTS_PER_USD,
    LCFS_PROGRAM_HIGH_USD_T,
    LCFS_PROGRAM_LOW_USD_T,
    SOYOIL_DOMESTIC,
    UCO_IMPORTED,
    Feedstock,
    FeedstockError,
    crush_from_soyoil_lb,
    discount_burden,
    feedstock_breakeven_usd_lb,
    import_penalty,
    lcfs_breakeven,
    lcfs_neutral_price,
    load_soyoil_usd_lb,
    penalty_bounds,
    structural_exit,
)
from agri.data.bloomberg_loader import DEFAULT_PATH

needs_bloomberg = pytest.mark.skipif(
    not DEFAULT_PATH.exists(), reason=f"fichier Bloomberg absent : {DEFAULT_PATH}"
)


# ===========================================================================
# L'ARGUMENT DE S2 — ce qui s'annule s'annule vraiment
# ===========================================================================
@pytest.mark.parametrize("ulsd", [2.00, 3.50, 5.00])
@pytest.mark.parametrize("rin", [0.40, 1.20])
@pytest.mark.parametrize("opex,roi", [(0.30, 0.10), (0.55, 0.25), (1.10, 0.60)])
def test_the_discount_is_invariant_to_diesel_rin_and_plant_costs(ulsd, rin, opex, roi):
    """LE test de la page.

    On calcule la décote d'indifférence par le modèle complet — valeur en sortie d'usine,
    RIN, opex, ROI, tout inclus — et on vérifie qu'elle coïncide avec la forme fermée qui
    n'en contient aucun. Dix-huit combinaisons de diesel, de RIN et de structure de coûts
    donnent exactement le même nombre : c'est ce qui autorise la page à dire qu'un lecteur
    ne peut pas la récuser en contestant sa prévision de diesel.
    """
    kwargs = dict(ulsd_usd_gal=ulsd, rin_d4_usd=rin, opex_usd_gal=opex, roi_usd_gal=roi)
    lcfs = 75.0

    # Décote d'indifférence par le modèle complet : à quel écart de prix les deux filières
    # dégagent-elles le même avantage net ?
    breakeven_domestic = feedstock_breakeven_usd_lb(
        SOYOIL_DOMESTIC, lcfs_usd_t=lcfs, **kwargs
    )
    breakeven_imported = feedstock_breakeven_usd_lb(
        UCO_IMPORTED, lcfs_usd_t=lcfs, **kwargs
    )
    discount_from_full_model = breakeven_domestic - breakeven_imported

    assert import_penalty(lcfs).discount_required_usd_lb == pytest.approx(
        discount_from_full_model, rel=1e-12
    )


def test_the_closed_form_matches_the_existing_threshold_at_price_parity():
    """Contrôle croisé contre `lcfs_breakeven`, écrit avant et indépendamment : à parité de
    prix feedstock, le seuil neutralisant doit être le même nombre."""
    threshold = lcfs_breakeven(
        price_domestic_usd_lb=0.50,
        price_imported_usd_lb=0.50,
        lcfs_current_usd_t=75.0,
    )
    assert lcfs_neutral_price() == pytest.approx(threshold.lcfs_star_usd_t, rel=1e-12)


def test_the_penalty_vanishes_exactly_at_the_neutral_price():
    neutral = lcfs_neutral_price()
    assert import_penalty(neutral).discount_required_usd_lb == pytest.approx(0.0, abs=1e-12)
    assert import_penalty(neutral).imports_win_outright


def test_above_the_neutral_price_imports_can_pay_a_premium():
    """Au-delà du seuil, l'avantage carbone dépasse le 45Z : la « décote requise » devient
    négative, ce qui se lit comme une prime que l'importé peut se permettre."""
    penalty = import_penalty(lcfs_neutral_price() + 100.0)
    assert penalty.discount_required_usd_lb < 0
    assert penalty.imports_win_outright
    assert "prime" in penalty.headline


def test_the_discount_decreases_monotonically_in_the_lcfs_price():
    discounts = [import_penalty(x).discount_required_usd_lb for x in (0.0, 50.0, 100.0, 200.0)]
    assert discounts == sorted(discounts, reverse=True)


def test_the_45z_credit_is_the_whole_penalty_when_the_lcfs_is_worthless():
    """À LCFS nul, la décote requise vaut exactement le crédit 45Z rapporté au rendement —
    borne haute du problème, et contrôle d'unité gallon -> livre."""
    penalty = import_penalty(0.0)
    assert penalty.lcfs_offset_usd_gal == 0.0
    assert penalty.discount_required_usd_lb == pytest.approx(0.46 / 7.6, rel=1e-12)


# ===========================================================================
# LE RÉSULTAT — la réponse est bornée
# ===========================================================================
def test_the_lcfs_has_never_traded_high_enough_to_offset_45z():
    """Le résultat de S3 : le prix neutralisant est hors de la plage que le programme a
    réalisée, donc aucun niveau historique du crédit n'annule le 45Z à parité de prix."""
    bounds = penalty_bounds()
    assert not bounds.reaches_neutral
    assert bounds.lcfs_neutral_usd_t > LCFS_PROGRAM_HIGH_USD_T
    assert bounds.lcfs_neutral_usd_t == pytest.approx(285.07, abs=0.01)


def test_the_whole_lcfs_range_moves_the_answer_by_only_a_few_cents():
    """L'argument « les deux camps se trompent de variable » : sur toute l'amplitude
    réalisée du crédit, la décote requise ne bouge que de trois cents et des poussières."""
    bounds = penalty_bounds()
    assert bounds.span_c_lb == pytest.approx(3.18, abs=0.01)
    assert 0.0 < bounds.span_c_lb < 4.0
    assert bounds.discount_at_low_usd_lb > bounds.discount_at_high_usd_lb


def test_bounds_reject_an_inverted_range():
    with pytest.raises(FeedstockError, match="borne haute"):
        penalty_bounds(lcfs_low_usd_t=200.0, lcfs_high_usd_t=50.0)


def test_a_richer_ci_gap_lowers_the_neutral_price():
    """Contrôle de sens : plus l'importé est propre devant le domestique, moins il faut de
    LCFS pour compenser le 45Z."""
    clean = Feedstock("UCO très propre", 5.0, north_american=False)
    assert lcfs_neutral_price(imported=clean) < lcfs_neutral_price()


def test_a_non_positive_ci_gap_raises_rather_than_returning_a_sign_flip():
    dirty = Feedstock("UCO plus carboné", 40.0, north_american=False)
    with pytest.raises(FeedstockError, match="deux tableaux"):
        import_penalty(75.0, imported=dirty)


# ===========================================================================
# Le poids relatif — et le garde-fou d'unité
# ===========================================================================
def test_discount_burden_rejects_a_series_quoted_in_cents():
    """Le piège d'unité du module, transformé en erreur : le soyoil CBOT cote en cents par
    livre, et un facteur 100 oublié produirait un pourcentage plausible et faux."""
    cents = pd.Series([45.0, 50.0, 55.0], index=pd.date_range("2024-01-01", periods=3))
    with pytest.raises(FeedstockError, match="cents par livre"):
        discount_burden(cents, lcfs_usd_t=75.0)


def test_discount_burden_is_countercyclical_to_the_oil_price():
    """Même décote en cents, poids relatif d'autant plus lourd que l'huile est bon marché."""
    prices = pd.Series([0.25, 0.50, 0.90], index=pd.date_range("2024-01-01", periods=3))
    burden = discount_burden(prices, lcfs_usd_t=75.0)
    shares = burden.frame["burden_share"].tolist()
    assert shares == sorted(shares, reverse=True)
    assert burden.burden_max / burden.burden_min == pytest.approx(0.90 / 0.25, rel=1e-9)


def test_discount_burden_rejects_an_empty_series():
    with pytest.raises(FeedstockError, match="vide"):
        discount_burden(pd.Series(dtype=float), lcfs_usd_t=75.0)


# ===========================================================================
# La sortie structurelle
# ===========================================================================
def test_structural_exit_is_the_floor_plus_the_required_discount():
    penalty = import_penalty(75.0)
    result = structural_exit(uco_floor_usd_lb=0.35, lcfs_usd_t=75.0)
    assert result.soyoil_critical_usd_lb == pytest.approx(
        0.35 + penalty.discount_required_usd_lb, rel=1e-12
    )
    assert result.share_below is None  # aucune série fournie


def test_structural_exit_counts_the_crossings():
    prices = pd.Series(
        [0.30, 0.35, 0.45, 0.60], index=pd.date_range("2024-01-01", periods=4)
    )
    result = structural_exit(prices, uco_floor_usd_lb=0.35, lcfs_usd_t=75.0)
    # seuil ~0,3946 : les deux premiers prints sont dessous, les deux derniers dessus
    assert result.share_below == pytest.approx(0.5)
    assert result.n_obs == 4


def test_structural_exit_rejects_a_non_positive_floor():
    with pytest.raises(FeedstockError, match="plancher"):
        structural_exit(uco_floor_usd_lb=0.0, lcfs_usd_t=75.0)


# ===========================================================================
# Le bilan de trituration
# ===========================================================================
def test_crush_from_soyoil_lb_arithmetic():
    """3,25 Md lb d'huile / 11 lb par boisseau / 365 jours = ~809 000 bu/jour."""
    balance = crush_from_soyoil_lb(3.25e9, installed_capacity_bu_day=6.8e6)
    assert balance.crush_required_bu_day == pytest.approx(809_464.5, rel=1e-4)
    assert not balance.is_short


def test_crush_from_soyoil_lb_rejects_a_non_positive_capacity():
    with pytest.raises(FeedstockError, match="capacité"):
        crush_from_soyoil_lb(1e9, installed_capacity_bu_day=0.0)


# ===========================================================================
# Sur la donnée réelle
# ===========================================================================
@needs_bloomberg
def test_loaded_soyoil_is_in_usd_per_pound_not_cents():
    """Régression sur le piège d'unité : un soyoil médian au-dessus de 5 signifierait que la
    conversion cents -> USD n'a pas été appliquée."""
    series = load_soyoil_usd_lb("2015")
    assert 0.10 < series.median() < 1.50
    assert series.max() < 2.0


@needs_bloomberg
def test_the_import_economics_work_today_because_oil_is_expensive():
    """Le renversement de S5, sur données réelles.

    Au même plancher de collecte et au même prix du LCFS, le soyoil a passé une grande part
    de 2015-2026 sous le prix critique, et une part quasi nulle depuis 2024. Ce qui a changé
    n'est pas la politique — c'est le niveau de l'huile végétale.
    """
    long_run = structural_exit(
        load_soyoil_usd_lb("2015"), uco_floor_usd_lb=0.35, lcfs_usd_t=75.0
    )
    recent = structural_exit(
        load_soyoil_usd_lb("2024"), uco_floor_usd_lb=0.35, lcfs_usd_t=75.0
    )
    assert long_run.share_below > 0.40
    assert recent.share_below < 0.05


@needs_bloomberg
def test_the_soyoil_range_dwarfs_the_lcfs_lever():
    """L'argument quantifié de S3 : le prix de l'huile parcourt un ordre de grandeur de plus
    que ce que le LCFS peut déplacer sur toute son histoire."""
    series = load_soyoil_usd_lb("2015")
    oil_range_c_lb = (series.max() - series.min()) * CENTS_PER_USD
    assert oil_range_c_lb > 10 * penalty_bounds().span_c_lb


@needs_bloomberg
def test_the_burden_swings_by_more_than_a_factor_three_on_real_prices():
    burden = discount_burden(load_soyoil_usd_lb("2015"), lcfs_usd_t=75.0)
    assert burden.burden_max / burden.burden_min > 3.0
    assert 0.03 < burden.burden_min < 0.08


@needs_bloomberg
def test_the_program_bounds_are_documented_not_loaded():
    """Garde-fou de posture : les bornes du programme LCFS sont des constantes documentées,
    pas des données de l'export. Si un jour la série CARB est ajoutée au loader, ce test
    doit échouer et forcer une relecture des sections qui s'appuient dessus."""
    from agri.data import bloomberg_loader

    assert "lcfs" not in bloomberg_loader.SERIES_SPECS
    assert LCFS_PROGRAM_LOW_USD_T < LCFS_PROGRAM_HIGH_USD_T < lcfs_neutral_price()
