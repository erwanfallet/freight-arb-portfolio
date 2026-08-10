"""Golden tests du solveur de point de bascule."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agri.core.breakeven import (
    BreakevenError,
    NoBreakevenInRange,
    solve_breakeven,
)


def test_linear_margin_root_and_sensitivity_hand_computed():
    # marge(theta) = 10 - 20*theta  ->  racine en theta = 0.5, pente -20 partout
    out = solve_breakeven(lambda t: 10.0 - 20.0 * t, 0.0, 1.0)
    assert out.theta_star == pytest.approx(0.5, abs=1e-9)
    assert out.sensitivity == pytest.approx(-20.0, abs=1e-6)


def test_nonlinear_margin():
    # marge(theta) = 4 - theta^2  ->  racine en theta = 2, pente = -2*theta = -4
    out = solve_breakeven(lambda t: 4.0 - t**2, 0.0, 5.0)
    assert out.theta_star == pytest.approx(2.0, abs=1e-8)
    assert out.sensitivity == pytest.approx(-4.0, abs=1e-4)


def test_ballast_share_breakeven_is_the_t1_1_deliverable():
    """La phrase attendue de T1-1 : « au-delà de 40 % de ballast facturé, l'arb se ferme ».

    arb(ballast) = 8 - 20 * ballast, en USD/t : l'arb vaut 8 USD/t sans ballast facturé
    et se ferme quand la part de ballast atteint 0.40. La sensibilité, -20 USD/t par
    unité de ballast, se lit « 10 points de ballast en plus coûtent 2 USD/t d'arb ».
    """
    out = solve_breakeven(
        lambda b: 8.0 - 20.0 * b,
        0.0,
        1.0,
        theta_label="ballast_share",
        margin_label="arb",
    )
    assert out.theta_star == pytest.approx(0.40, abs=1e-9)
    assert out.sensitivity / 10.0 == pytest.approx(-2.0, abs=1e-6)
    assert "ballast_share* = 0.4" in out.summary


def test_distance_in_sigmas_hand_computed():
    # racine en 0.5, niveau courant 0.3, écart-type historique 0.1 -> +2.0 sigmas
    history = pd.Series([0.2, 0.3, 0.4])  # écart-type d'échantillon = 0.1
    out = solve_breakeven(
        lambda t: 10.0 - 20.0 * t, 0.0, 1.0, theta_current=0.3, theta_history=history
    )
    assert out.theta_sigma == pytest.approx(0.1, abs=1e-9)
    assert out.distance_sigmas == pytest.approx(2.0, abs=1e-6)


def test_a_breakeven_two_sigmas_away_is_flagged_out_of_reach():
    """La protection centrale : ne pas annoncer un seuil lointain comme imminent."""
    history = pd.Series([0.2, 0.3, 0.4])
    out = solve_breakeven(
        lambda t: 10.0 - 20.0 * t, 0.0, 1.0, theta_current=0.3, theta_history=history
    )
    assert not out.is_within_reach
    assert "hors de portée" in out.summary


def test_a_close_breakeven_is_flagged_within_reach():
    history = pd.Series([0.1, 0.3, 0.5, 0.7])  # écart-type ~0.2582
    out = solve_breakeven(
        lambda t: 10.0 - 20.0 * t, 0.0, 1.0, theta_current=0.45, theta_history=history
    )
    assert out.distance_sigmas == pytest.approx(0.1936, abs=1e-3)
    assert out.is_within_reach
    assert "à portée" in out.summary


def test_distance_is_none_without_history_rather_than_a_number_without_scale():
    out = solve_breakeven(lambda t: 10.0 - 20.0 * t, 0.0, 1.0, theta_current=0.3)
    assert out.distance_sigmas is None
    assert not out.is_within_reach
    assert "écart-type" not in out.summary


def test_no_sign_change_is_a_result_not_a_crash():
    """« Sur toute la plage plausible, l'arb reste ouvert » est une affirmation publiable."""
    with pytest.raises(NoBreakevenInRange) as excinfo:
        solve_breakeven(lambda b: 8.0 - 2.0 * b, 0.0, 1.0)
    err = excinfo.value
    assert err.margin_lo == pytest.approx(8.0)
    assert err.margin_hi == pytest.approx(6.0)
    assert "reste positive" in str(err)
    assert "sans extrapoler" in str(err)


def test_no_sign_change_reports_a_persistently_negative_margin():
    with pytest.raises(NoBreakevenInRange, match="reste négative"):
        solve_breakeven(lambda b: -8.0 - 2.0 * b, 0.0, 1.0)


def test_root_exactly_on_the_lower_bound():
    out = solve_breakeven(lambda t: t, 0.0, 1.0)
    assert out.theta_star == pytest.approx(0.0)
    assert out.sensitivity == pytest.approx(1.0, abs=1e-6)


def test_root_exactly_on_the_upper_bound():
    out = solve_breakeven(lambda t: t - 1.0, 0.0, 1.0)
    assert out.theta_star == pytest.approx(1.0)


def test_sensitivity_stays_inside_the_bracket_at_a_boundary_root():
    """Un ballast négatif n'existe pas : la dérivée ne doit pas sortir des bornes."""
    calls: list[float] = []

    def margin(t: float) -> float:
        calls.append(t)
        return t

    solve_breakeven(margin, 0.0, 1.0)
    assert min(calls) >= 0.0
    assert max(calls) <= 1.0


def test_inverted_bounds_raise():
    with pytest.raises(BreakevenError, match="bornes incohérentes"):
        solve_breakeven(lambda t: t, 1.0, 0.0)


def test_non_finite_margin_raises():
    with pytest.raises(BreakevenError, match="non finie"):
        solve_breakeven(lambda t: np.inf, 0.0, 1.0)


def test_increasing_margin_lcfs_style():
    """T3-1 : la valeur en sortie d'usine croît avec le crédit LCFS.

    valeur(LCFS) = -30 + 0.25 * LCFS, en USD/t : le seuil est à 120 USD/t CO2e.
    Le solveur doit traiter une marge croissante exactement comme une décroissante.
    """
    out = solve_breakeven(
        lambda lcfs: -30.0 + 0.25 * lcfs,
        0.0,
        400.0,
        theta_current=95.0,
        theta_history=pd.Series([60.0, 80.0, 95.0, 110.0, 130.0]),
        theta_label="LCFS",
    )
    assert out.theta_star == pytest.approx(120.0, abs=1e-8)
    assert out.sensitivity == pytest.approx(0.25, abs=1e-6)
    # moyenne 95, écarts -35/-15/0/+15/+35, somme des carrés 2900,
    # variance d'échantillon 2900/4 = 725, écart-type sqrt(725) = 26.9258
    # distance = (120 - 95) / 26.9258 = 0.9285
    assert out.theta_sigma == pytest.approx(26.9258, abs=1e-3)
    assert out.distance_sigmas == pytest.approx(0.9285, abs=1e-3)
    assert out.is_within_reach
