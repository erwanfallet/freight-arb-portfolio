"""Golden tests T2-4 — ce qu'un prix peut identifier, et ce qu'il ne peut pas.

Le test central est `test_the_level_is_not_identifiable_but_the_variation_is`. C'est la
thèse de la page, et elle est de nature inhabituelle : elle affirme à la fois une limite
(le niveau de la rente ne se déduit pas des prix) et un résultat (sa variation, si). Les
deux moitiés doivent être testées, sinon la page pourrait dériver vers la conclusion
confortable — publier un niveau — sans que rien ne le signale.
"""
from __future__ import annotations

import pandas as pd
import pytest

from agri.chains.white_premium import (
    POL_PLAUSIBLE_HI,
    POL_PLAUSIBLE_LO,
    WhitePremiumError,
    identification_check,
    implied_pol_adjust,
    implied_refining_cost,
    load_real_richness_frame,
)
from agri.data.bloomberg_loader import DEFAULT_PATH

pytestmark = pytest.mark.skipif(
    not DEFAULT_PATH.exists(), reason=f"fichier Bloomberg absent : {DEFAULT_PATH}"
)

START = "2015-01-01"


@pytest.fixture(scope="module")
def check():
    return identification_check(start=START)


# ===========================================================================
# LA THÈSE
# ===========================================================================
def test_the_level_is_not_identifiable_but_the_variation_is(check):
    """LE test de la page, dans ses deux moitiés.

    Moitié 1 — le niveau n'est pas identifiable : l'incertitude qu'un seul paramètre
    inobservable injecte est du même ordre que la richness médiane elle-même.
    Moitié 2 — la variation l'est : l'écart entre la meilleure et la pire année dépasse
    cette incertitude d'un facteur qui ne laisse pas de place au doute.
    """
    median_richness = abs(check.annual["richness_ref"].median())
    assert check.parameter_span_max > median_richness, (
        "l'incertitude de paramètre est devenue petite devant la richness médiane : la "
        "section S3 de la page affirme le contraire et doit être relue"
    )
    assert check.ratio > 3.0
    assert check.signal_span > 40.0


def test_the_year_ranking_survives_the_parameter_entirely(check):
    """Ce qui autorise à lire les écarts entre années : le paramètre les déplace toutes dans
    le même sens, donc il ne réordonne rien."""
    assert check.rank_correlation == pytest.approx(1.0, abs=1e-9)


def test_only_years_near_zero_can_flip_sign(check):
    """Les seules années non interprétables sont celles dont la richness est déjà proche de
    zéro. Si une année franchement positive ou négative changeait de signe, la conclusion de
    la page ne tiendrait plus."""
    for year in check.sign_flipping_years:
        assert abs(check.annual.loc[year, "richness_ref"]) < check.parameter_span_max


def test_the_regime_shift_is_larger_than_the_parameter_can_explain(check):
    """Le basculement de S5 : la richness est majoritairement négative jusqu'en 2021 et
    franchement positive à partir de 2023, et l'écart entre les deux périodes dépasse ce que
    le choix de pol_adjust peut produire."""
    reference = check.annual["richness_ref"]
    before = reference.loc[2017:2021].median()
    after = reference.loc[2023:].median()
    assert before < 0 < after
    assert after - before > 2 * check.parameter_span_max


# ===========================================================================
# L'inversion
# ===========================================================================
def test_implied_pol_zeroes_the_median_richness():
    """Contrôle de cohérence du solveur : au pol* qu'il rend, la richness médiane doit être
    nulle à la tolérance près."""
    implied = implied_pol_adjust(start=START)
    frame = load_real_richness_frame(pol_adjust=implied.pol_star, start=START)
    assert frame["richness"].median() == pytest.approx(0.0, abs=0.05)


def test_implied_pol_sits_just_above_the_plausible_band():
    """Le résultat de S3 : l'ajustement qui annulerait la rente est hors de la plage
    plausible, mais de peu — ce qui est précisément ce qui rend les deux lectures ouvertes."""
    implied = implied_pol_adjust(start=START)
    assert not implied.within_plausible
    assert POL_PLAUSIBLE_HI < implied.pol_star < POL_PLAUSIBLE_HI + 0.02
    assert "ne permet pas de trancher" in implied.headline


def test_implied_pol_stays_inside_the_guarded_range():
    """Le solveur ne doit jamais sortir de la plage que `white_premium_usd_t` accepte —
    sinon il lève une exception au lieu de renvoyer un résultat (rencontré en développement)."""
    implied = implied_pol_adjust(start=START)
    assert 1.00 <= implied.pol_star <= 1.20


def test_a_range_with_no_root_raises_rather_than_returning_a_bound():
    """Sur une plage où la richness garde le même signe aux deux bornes, il n'y a pas de
    pol* — et le moteur le dit au lieu de renvoyer une borne comme si c'était une solution.
    Au-delà de 1,10 la richness médiane est négative partout : aucun ajustement de
    polarisation ne l'annule dans cette fenêtre."""
    with pytest.raises(WhitePremiumError, match="aucun pol_adjust"):
        implied_pol_adjust(start=START, search_lo=1.10, search_hi=1.1999)


# ===========================================================================
# Le nombre du mail
# ===========================================================================
def test_implied_refining_cost_is_the_white_premium_itself():
    """Le prix payé pour raffiner est le white premium, sans hypothèse de coût. C'est ce qui
    en fait le nombre présentable : il est observé, pas modélisé."""
    cost = implied_refining_cost(start=START)
    frame = load_real_richness_frame(start=START)
    assert cost.market_usd_t == pytest.approx(frame["white_premium"].median())
    assert cost.gap_usd_t == pytest.approx(cost.market_usd_t - cost.modelled_usd_t)


def test_the_market_pays_a_plausible_order_of_magnitude_for_refining():
    cost = implied_refining_cost(start=START)
    assert 40.0 < cost.market_usd_t < 120.0


# ===========================================================================
# Garde-fous
# ===========================================================================
def test_identification_check_rejects_a_reference_outside_the_bounds():
    with pytest.raises(WhitePremiumError, match="strictement entre"):
        identification_check(start=START, pol_ref=1.15)


def test_identification_check_rejects_a_window_too_short_to_compare_years():
    with pytest.raises(WhitePremiumError, match="année"):
        identification_check(start="2025-06-01")


def test_the_parameter_moves_every_year_in_the_same_direction(check):
    """La propriété qui fonde toute la section S4 : pol_adjust est un multiplicateur du prix
    du brut, donc il déplace toutes les années dans le même sens. Si ce n'était pas le cas,
    les écarts entre années ne survivraient pas."""
    differences = check.annual["richness_lo"] - check.annual["richness_hi"]
    assert (differences > 0).all()


def test_a_higher_pol_adjust_always_lowers_the_richness():
    """Contrôle de sens : plus l'ajustement est élevé, plus le brut converti coûte cher, donc
    plus la richness est faible."""
    medians = [
        load_real_richness_frame(pol_adjust=pol, start=START)["richness"].median()
        for pol in (1.02, 1.06, 1.10, 1.14)
    ]
    assert medians == sorted(medians, reverse=True)


def test_the_plausible_band_brackets_the_default():
    assert POL_PLAUSIBLE_LO < 1.07 < POL_PLAUSIBLE_HI
