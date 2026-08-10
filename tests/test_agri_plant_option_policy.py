"""Golden tests du simulateur de politique T2-5 et de l'inversion.

Le chemin de marge de référence, utilisé pour tous les calculs à la main :

    margin = [10, -5, -5, -5, -5, 10, 10, 10, 10]      (9 périodes)
    coûts  : redémarrage 100, arrêt 50, maintien 1/période

Il est construit pour que la règle de persistance N=4 déclenche exactement une fois, au
5e point : c'est la première date où les quatre valeurs qui précèdent (indice compris)
sont toutes négatives.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agri.chains.china_soy import load_real_crush_frame
from agri.chains.plant_option import (
    HysteresisBand,
    PlantOptionError,
    calibrate_ou,
    compare_policies,
    implied_switching_cost,
    run_always_on_policy,
    run_band_policy,
    run_heuristic_policy,
    solve_hysteresis,
    switching_cost_sensitivity,
)
from agri.data.bloomberg_loader import DEFAULT_PATH

PATH_VALUES = [10.0, -5.0, -5.0, -5.0, -5.0, 10.0, 10.0, 10.0, 10.0]
COSTS = dict(cost_restart=100.0, cost_shutdown=50.0, cost_idle=1.0)


@pytest.fixture
def path() -> pd.Series:
    return pd.Series(PATH_VALUES, index=pd.date_range("2024-01-31", periods=9, freq="ME"))


# ===========================================================================
# Le simulateur, calculé à la main
# ===========================================================================
def test_heuristic_policy_hand_computed(path):
    """Règle « marge < 0 pendant 4 périodes », symétrique au redémarrage.

    Arrêt au 5e point (indice 4) : c'est la première fenêtre de 4 toutes négatives.
    Redémarrage au 9e (indice 8) : première fenêtre de 4 toutes positives.

        exploitation = 10 - 5 - 5 - 5 - 5        = -10   (5 périodes en marche)
        switch       = 50 (arrêt) + 100 (redém.) = 150
        maintien     = 4 périodes x 1            =   4
        total        = -10 - 150 - 4             = -164
    """
    result = run_heuristic_policy(path, threshold=0.0, n_periods=4, **COSTS)
    assert result.n_stops == 1
    assert result.n_starts == 1
    assert result.periods_on == 5
    assert result.periods_off == 4
    assert result.operating_pnl == pytest.approx(-10.0)
    assert result.switching_cost == pytest.approx(150.0)
    assert result.idle_cost == pytest.approx(4.0)
    assert result.total_pnl == pytest.approx(-164.0)


def test_persistence_rule_does_not_fire_before_n_periods(path):
    """Le 4e point est négatif mais la fenêtre contient encore le +10 initial : pas d'arrêt.

    C'est exactement ce que la persistance achète — et c'est aussi pourquoi la règle
    arrête à un niveau de marge plus bas que son seuil affiché.
    """
    result = run_heuristic_policy(path, threshold=0.0, n_periods=4, **COSTS)
    assert result.stop_margins == [-5.0]
    # l'arret a lieu a la 5e periode, pas a la 4e
    assert bool(result.state.iloc[3]) is True
    assert bool(result.state.iloc[4]) is False


def test_band_policy_hand_computed(path):
    """Bande [-3, +5] : arrêt instantané au premier point sous -3, redémarrage au premier
    point au-dessus de +5.

        exploitation = 10 - 5 + 10 + 10 + 10 = 35
        switch       = 50 + 100              = 150
        maintien     = 4 x 1                 =   4
        total        = 35 - 150 - 4          = -119
    """
    band = HysteresisBand(
        m_off=-3.0, m_on=5.0, grid=np.linspace(-20, 20, 5),
        value_on=np.zeros(5), value_off=np.zeros(5), n_iterations=1, converged=True,
    )
    result = run_band_policy(path, band, **COSTS)
    assert result.n_stops == 1
    assert result.periods_on == 5
    assert result.operating_pnl == pytest.approx(35.0)
    assert result.total_pnl == pytest.approx(-119.0)


def test_always_on_is_the_counterfactual(path):
    """Contrefactuel : somme brute du chemin, aucun coût de switch ni de maintien.

        10 + 4 x (-5) + 4 x 10 = 10 - 20 + 40 = +30

    Il bat les deux règles sur ce chemin — un rappel que s'arrêter n'est pas
    gratuitement bon, et que la comparaison doit toujours inclure ce cas.
    """
    result = run_always_on_policy(path, **COSTS)
    assert result.n_stops == 0
    assert result.periods_off == 0
    assert result.total_pnl == pytest.approx(30.0)
    assert result.total_pnl == pytest.approx(sum(PATH_VALUES))


# ===========================================================================
# La bande dégénérée — refuser plutôt que substituer
# ===========================================================================
def test_degenerate_band_is_detected():
    band = HysteresisBand(
        m_off=10.0, m_on=-10.0, grid=np.linspace(-20, 20, 5),
        value_on=np.zeros(5), value_off=np.zeros(5), n_iterations=1, converged=True,
    )
    assert band.is_degenerate
    assert "never stop" in band.headline


def test_degenerate_band_refuses_to_run_rather_than_substituting(path):
    """Le point de discipline : appliquer M_on < M_off ferait osciller l'usine à chaque
    période, et lui substituer une politique de repli reviendrait à comparer une règle
    que le modèle n'a pas produite."""
    band = HysteresisBand(
        m_off=10.0, m_on=-10.0, grid=np.linspace(-20, 20, 5),
        value_on=np.zeros(5), value_off=np.zeros(5), n_iterations=1, converged=True,
    )
    with pytest.raises(PlantOptionError, match="degenerate"):
        run_band_policy(path, band, **COSTS)


def test_comparison_continues_without_the_band_when_degenerate(path):
    """La comparaison ne s'interrompt pas : règle à seuil contre contrefactuel reste
    informatif, et le headline dit pourquoi la bande manque."""
    band = HysteresisBand(
        m_off=10.0, m_on=-10.0, grid=np.linspace(-20, 20, 5),
        value_on=np.zeros(5), value_off=np.zeros(5), n_iterations=1, converged=True,
    )
    comparison = compare_policies(path, band, **COSTS)
    assert not comparison.band_is_available
    assert np.isnan(comparison.gap_vs_band)
    assert "No exercise boundary" in comparison.headline
    # le contrefactuel reste calculable : -164 (heuristique) - 30 (jamais d'arret)
    assert comparison.heuristic_flexibility_value == pytest.approx(-194.0)
    assert len(comparison.to_frame()) == 2


# ===========================================================================
# L'inversion, sur la vraie marge de crush chinoise
# ===========================================================================
@pytest.mark.skipif(not DEFAULT_PATH.exists(), reason=f"fichier Bloomberg absent : {DEFAULT_PATH}")
def test_chinese_crush_margin_is_the_right_anchor():
    """Pourquoi cette page s'ancre sur la marge chinoise et non sur le board crush US :
    elle passe réellement sous zéro (41 % du temps, jusqu'à -865 CNY/t) et elle est
    stationnaire, donc la calibration OU y est légitime et la question du curtailment
    y est vivante."""
    margin = load_real_crush_frame()["margin"]
    assert (margin < 0).mean() > 0.35
    assert margin.min() < -500
    assert calibrate_ou(margin, strict=False).stationarity.verdict == "stationary"


@pytest.mark.skipif(not DEFAULT_PATH.exists(), reason=f"fichier Bloomberg absent : {DEFAULT_PATH}")
def test_implied_switching_cost_is_the_deliverable():
    """LE chiffre du mail : la règle N=4 arrête et redémarre à des niveaux précis, donc
    elle suppose un coût d'aller-retour précis — ici ~143 CNY/t de fève triturée."""
    margin = load_real_crush_frame()["margin"]
    ou = calibrate_ou(margin, strict=False)
    implied = implied_switching_cost(margin, ou, cost_idle=2.0)
    assert implied.converged
    assert implied.n_stops_observed > 10
    assert implied.effective_m_off < 0 < implied.effective_m_on
    assert 100.0 < implied.implied_switching_cost < 200.0
    assert "assumes without saying so" in implied.headline


@pytest.mark.skipif(not DEFAULT_PATH.exists(), reason=f"fichier Bloomberg absent : {DEFAULT_PATH}")
def test_rule_stops_below_its_own_threshold():
    """La persistance déplace le point d'arrêt : la règle affiche un seuil de 0 mais
    arrête en réalité bien en dessous. C'est ce décalage qui la rend équivalente à une
    bande, donc traduisible en coût de switch."""
    margin = load_real_crush_frame()["margin"]
    ou = calibrate_ou(margin, strict=False)
    implied = implied_switching_cost(margin, ou, cost_idle=2.0)
    assert implied.effective_m_off < -10.0


@pytest.mark.skipif(not DEFAULT_PATH.exists(), reason=f"fichier Bloomberg absent : {DEFAULT_PATH}")
def test_sensitivity_marks_degenerate_rows_instead_of_negative_widths():
    """Régression du défaut trouvé en construisant la page : à faible coût de switch la
    bande s'inverse, et une largeur négative se lisait comme une bande étroite."""
    margin = load_real_crush_frame()["margin"]
    ou = calibrate_ou(margin, strict=False)
    table = switching_cost_sensitivity(
        margin, ou, cost_grid=np.array([5.0, 30.0, 143.0, 600.0]), cost_idle=2.0
    )
    assert table["degenerate"].any()
    widths = table.loc[~table["degenerate"], "band_width"]
    assert (widths > 0).all()
    assert table.loc[table["degenerate"], "band_width"].isna().all()


@pytest.mark.skipif(not DEFAULT_PATH.exists(), reason=f"fichier Bloomberg absent : {DEFAULT_PATH}")
def test_band_width_grows_with_switching_cost():
    """Monotonie attendue — c'est elle qui autorise l'interpolation de l'inversion."""
    margin = load_real_crush_frame()["margin"]
    ou = calibrate_ou(margin, strict=False)
    table = switching_cost_sensitivity(
        margin, ou, cost_grid=np.array([30.0, 60.0, 143.0, 300.0, 600.0]), cost_idle=2.0
    )
    valid = table[~table["degenerate"]]
    assert valid["band_width"].is_monotonic_increasing


@pytest.mark.skipif(not DEFAULT_PATH.exists(), reason=f"fichier Bloomberg absent : {DEFAULT_PATH}")
def test_flexibility_is_worth_far_more_than_the_rule_choice():
    """Hiérarchie des enjeux, à dire dans cet ordre : pouvoir s'arrêter vaut beaucoup
    (~125 000 CNY/t cumulés contre ne jamais s'arrêter), choisir la bonne règle d'arrêt
    vaut nettement moins (~3 000). Inverser cet ordre dans la présentation ferait passer
    un raffinement pour le sujet principal."""
    margin = load_real_crush_frame()["margin"]
    ou = calibrate_ou(margin, strict=False)
    band = solve_hysteresis(ou, cost_restart=96.0, cost_shutdown=47.0, cost_idle=2.0)
    comparison = compare_policies(margin, band, cost_restart=96.0, cost_shutdown=47.0, cost_idle=2.0)
    assert comparison.band_is_available
    assert comparison.flexibility_value > 50_000
    assert 0 < comparison.gap_vs_band < comparison.flexibility_value / 10
