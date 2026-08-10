"""Golden tests T1-2 — le prix auquel le bilan force la sortie.

Le livrable de la page est une forme fermée, donc entièrement vérifiable à la main :

    P* = (B/Q + P0) / (1 + im_rate)

Avec B = 250 M USD, Q = 100 kt, P0 = 2 572 USD/t (cacao au 03/01/2023) et un taux de
marge implicite de 4,4988 % :

    B/Q = 2 500
    P*  = (2 500 + 2 572) / 1,044988 = 5 072 / 1,044988 = 4 853,64 USD/t
"""
from __future__ import annotations

import pandas as pd
import pytest

from agri.chains.hedge_cost import (
    SHORT_HEDGE,
    HedgeCostError,
    HedgeParams,
    forced_exit_price,
    forced_exit_schedule,
    hedging_intensity,
    implied_margin_rate,
    load_real_hedge_frame,
)
from agri.data.bloomberg_loader import DEFAULT_PATH, load

pytestmark = pytest.mark.skipif(
    not DEFAULT_PATH.exists(), reason=f"fichier Bloomberg absent : {DEFAULT_PATH}"
)

BOOK_T = 100_000.0
LINE_USD = 250e6


@pytest.fixture(scope="module")
def cocoa() -> pd.Series:
    return load("cocoa_ny")


@pytest.fixture(scope="module")
def margin_rate() -> float:
    params = HedgeParams(side=SHORT_HEDGE, book_size_t=BOOK_T, credit_line_usd=LINE_USD)
    simulation = load_real_hedge_frame("cacao_ny", params=params)
    return implied_margin_rate(simulation, book_size_t=BOOK_T)


# ===========================================================================
# La forme fermée
# ===========================================================================
def test_closed_form_hand_computed():
    """Sur une série synthétique minimale, pour isoler l'algèbre du bruit de marché."""
    price = pd.Series(
        [2572.0, 3000.0, 6000.0], index=pd.to_datetime(["2023-01-03", "2023-06-01", "2024-02-01"])
    )
    result = forced_exit_price(
        price, inception="2023-01-03", book_size_t=BOOK_T,
        credit_line_usd=LINE_USD, im_rate=0.044988,
    )
    assert result.inception_price == pytest.approx(2572.0)
    assert result.exit_price == pytest.approx(4853.64, abs=0.01)
    assert result.crossed_on == pd.Timestamp("2024-02-01")


def test_exit_price_scales_with_the_credit_line():
    """Doubler la ligne recule le seuil de B/Q, pas de deux fois le seuil — la relation
    est affine en B/Q, pas proportionnelle au prix. C'est ce qui permet à un desk de
    refaire le calcul de tête."""
    price = pd.Series([2000.0], index=pd.to_datetime(["2023-01-03"]))
    small = forced_exit_price(price, inception="2023-01-03", book_size_t=BOOK_T,
                              credit_line_usd=100e6, im_rate=0.05)
    large = forced_exit_price(price, inception="2023-01-03", book_size_t=BOOK_T,
                              credit_line_usd=200e6, im_rate=0.05)
    delta = large.exit_price - small.exit_price
    assert delta == pytest.approx((200e6 - 100e6) / BOOK_T / 1.05, abs=1e-6)


def test_bigger_book_lowers_the_exit_price():
    """Même ligne, book deux fois plus gros : on est forcé de sortir plus tôt."""
    price = pd.Series([2000.0], index=pd.to_datetime(["2023-01-03"]))
    small_book = forced_exit_price(price, inception="2023-01-03", book_size_t=50_000.0,
                                   credit_line_usd=LINE_USD, im_rate=0.05)
    big_book = forced_exit_price(price, inception="2023-01-03", book_size_t=200_000.0,
                                 credit_line_usd=LINE_USD, im_rate=0.05)
    assert big_book.exit_price < small_book.exit_price


def test_never_crossed_is_reported_as_such():
    price = pd.Series(
        [2000.0, 2100.0], index=pd.to_datetime(["2023-01-03", "2023-06-01"])
    )
    result = forced_exit_price(price, inception="2023-01-03", book_size_t=BOOK_T,
                               credit_line_usd=LINE_USD, im_rate=0.05)
    assert result.crossed_on is None
    assert result.days_of_protection is None
    assert "jamais allé" in result.headline


def test_invalid_inputs_raise():
    price = pd.Series([2000.0], index=pd.to_datetime(["2023-01-03"]))
    with pytest.raises(HedgeCostError, match="book et ligne"):
        forced_exit_price(price, inception="2023-01-03", book_size_t=0.0,
                          credit_line_usd=LINE_USD, im_rate=0.05)
    with pytest.raises(HedgeCostError, match="aucun prix"):
        forced_exit_price(price, inception="2030-01-01", book_size_t=BOOK_T,
                          credit_line_usd=LINE_USD, im_rate=0.05)


# ===========================================================================
# LE FAIT QUE LA PAGE EXPLIQUE
# ===========================================================================
def test_hedging_six_years_early_bought_three_months(cocoa, margin_rate):
    """Le résultat qui porte la page.

    Sur le cacao réel, une maison couverte depuis janvier 2018 et une maison couverte
    depuis janvier 2024 sont forcées de sortir **à moins de trois mois d'écart**, alors
    que six ans séparent leurs points d'entrée. Le mouvement de début 2024 a été assez
    violent pour écraser la dispersion des dates d'ouverture — ce qui explique pourquoi
    tant de maisons ont heurté la contrainte de bilan en même temps plutôt que chacune
    à son tour.
    """
    schedule = forced_exit_schedule(
        cocoa, ["2018-01-02", "2022-01-03", "2023-01-03", "2023-09-01", "2024-01-02"],
        book_size_t=BOOK_T, credit_line_usd=LINE_USD, im_rate=margin_rate,
    )
    assert len(schedule) == 5
    assert schedule["franchi le"].notna().all()

    span_days = (schedule["franchi le"].max() - schedule["franchi le"].min()).days
    assert span_days < 100, f"les sorties forcées s'étalent sur {span_days} jours"

    # toutes tombent dans la fenetre nov. 2023 - fev. 2024
    assert schedule["franchi le"].min() >= pd.Timestamp("2023-11-01")
    assert schedule["franchi le"].max() <= pd.Timestamp("2024-03-31")


def test_earlier_inception_still_buys_some_protection(cocoa, margin_rate):
    """La dispersion est écrasée, pas nulle : se couvrir tôt reste meilleur, l'ordre est
    respecté. C'est ce qui distingue « le timing ne sert à rien » (faux) de « le timing
    n'a acheté que quelques semaines » (vrai)."""
    schedule = forced_exit_schedule(
        cocoa, ["2018-01-02", "2022-01-03", "2023-01-03", "2023-09-01", "2024-01-02"],
        book_size_t=BOOK_T, credit_line_usd=LINE_USD, im_rate=margin_rate,
    ).sort_values("ouverture")
    assert schedule["jours de protection"].is_monotonic_decreasing
    assert schedule["franchi le"].is_monotonic_increasing


def test_headroom_shrinks_as_the_market_runs(cocoa, margin_rate):
    """Plus on ouvre tard dans un marché haussier, moins la ligne laisse de marge en %."""
    schedule = forced_exit_schedule(
        cocoa, ["2018-01-02", "2023-01-03", "2024-01-02"],
        book_size_t=BOOK_T, credit_line_usd=LINE_USD, im_rate=margin_rate,
    ).sort_values("ouverture")
    assert schedule["marge de manœuvre"].is_monotonic_decreasing


# ===========================================================================
# L'intensité de couverture — l'ancrage Montesanto
# ===========================================================================
def test_hedging_intensity_climbs_from_margin_only_to_near_book_value():
    """Au départ on ne poste que la marge initiale (~4,5 % du prix) ; quand le marché
    court contre la couverture, la variation margin s'empile jusqu'à immobiliser presque
    autant que la valeur du stock protégé."""
    params = HedgeParams(side=SHORT_HEDGE, book_size_t=BOOK_T, credit_line_usd=LINE_USD)
    simulation = load_real_hedge_frame("cacao_ny", params=params)
    intensity = hedging_intensity(
        simulation, book_size_t=BOOK_T, calm_window=("2018-04-02", "2022-12-31")
    )
    assert intensity.calm_median < 0.15
    assert intensity.peak_ratio > 0.7
    assert intensity.peak_ratio > intensity.calm_median * 5
    assert "Montesanto" in intensity.headline


def test_hedging_intensity_peak_lands_in_the_crisis_window():
    params = HedgeParams(side=SHORT_HEDGE, book_size_t=BOOK_T, credit_line_usd=LINE_USD)
    simulation = load_real_hedge_frame("cacao_ny", params=params)
    intensity = hedging_intensity(simulation, book_size_t=BOOK_T)
    assert pd.Timestamp("2024-01-01") <= intensity.peak_date <= pd.Timestamp("2025-06-30")


def test_implied_margin_rate_is_plausible():
    """Un taux de marge initiale hors de [1 %, 15 %] du prix signalerait un proxy mal
    calibré plutôt qu'un barème de chambre."""
    params = HedgeParams(side=SHORT_HEDGE, book_size_t=BOOK_T, credit_line_usd=LINE_USD)
    simulation = load_real_hedge_frame("cacao_ny", params=params)
    rate = implied_margin_rate(simulation, book_size_t=BOOK_T)
    assert 0.01 < rate < 0.15
