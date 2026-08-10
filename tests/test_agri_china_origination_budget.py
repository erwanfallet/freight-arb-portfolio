"""Golden tests T3-4 — le budget d'origination, et les fenêtres où aucune origine ne marche.

L'intérêt du budget tient à ce qu'il **ne** contient pas : ni basis d'origine, ni fret. Deux
tests le vérifient explicitement (`test_the_budget_does_not_depend_on_the_freight_assumption`
et `..._on_the_basis_assumption`), parce que c'est exactement la propriété qui permet à la
page de conclure sans les deux séries que l'export ne fournit pas. Si une refonte future
réintroduisait l'une des deux dans le calcul, la conclusion deviendrait conditionnelle à un
forfait sans que rien ne le signale.
"""
from __future__ import annotations

import pandas as pd
import pytest

from agri.chains.china_soy import (
    BUSHELS_PER_TONNE_SOYBEAN,
    DEFAULT_BASIS_CENTS_BU,
    DEFAULT_FREIGHT_USD_T,
    DEFAULT_IMPORT_DUTY,
    DEFAULT_PROCESSING_CNY_T,
    ChinaSoyError,
    affordable_origination_budget,
    impossible_windows,
    load_real_crush_frame,
)
from agri.data.bloomberg_loader import DEFAULT_PATH, load

pytestmark = pytest.mark.skipif(
    not DEFAULT_PATH.exists(), reason=f"fichier Bloomberg absent : {DEFAULT_PATH}"
)


@pytest.fixture(scope="module")
def budget():
    return affordable_origination_budget(start="2018-01-01")


# ===========================================================================
# LA PROPRIÉTÉ QUI FONDE LA PAGE
# ===========================================================================
def test_the_budget_does_not_depend_on_the_freight_assumption():
    """LE test de la page.

    Le fret de référence sert de **seuil de lecture**, jamais d'entrée du calcul. Deux
    valeurs très différentes doivent produire exactement le même budget — sinon la
    conclusion serait conditionnelle à un forfait que l'export ne fournit pas.
    """
    low = affordable_origination_budget(start="2018-01-01", freight_reference_usd_t=25.0)
    high = affordable_origination_budget(start="2018-01-01", freight_reference_usd_t=85.0)
    pd.testing.assert_series_equal(
        low.frame["budget_usd_t"], high.frame["budget_usd_t"]
    )
    # seule la LECTURE change
    assert low.share_below_freight < high.share_below_freight


def test_the_budget_does_not_depend_on_the_basis_assumption():
    """Même exigence pour le basis FOB : il est passé à `load_real_crush_frame` en aval mais
    ne doit pas atteindre le budget, qui part de la recette et du CBOT nu."""
    low = affordable_origination_budget(start="2018-01-01", basis_cents_bu=0.0)
    high = affordable_origination_budget(start="2018-01-01", basis_cents_bu=150.0)
    pd.testing.assert_series_equal(
        low.frame["budget_usd_t"], high.frame["budget_usd_t"]
    )


def test_budget_hand_computed(budget):
    """budget = (recette_HT - transformation)/(1 + droit)/USDCNY - CBOT x 36,7437."""
    crush = load_real_crush_frame(start="2018-01-01")
    row = budget.frame.iloc[-1]
    revenue = float(crush.loc[budget.frame.index[-1], "revenue_ex_vat"])

    expected_cnf = (revenue - DEFAULT_PROCESSING_CNY_T) / (1.0 + DEFAULT_IMPORT_DUTY) / row["usdcny"]
    assert row["cnf_max_usd_t"] == pytest.approx(expected_cnf, rel=1e-12)
    assert row["cbot_usd_t"] == pytest.approx(
        row["cbot_usd_bu"] * BUSHELS_PER_TONNE_SOYBEAN, rel=1e-12
    )
    assert row["budget_usd_t"] == pytest.approx(
        row["cnf_max_usd_t"] - row["cbot_usd_t"], rel=1e-12
    )


def test_the_bushel_conversion_is_derived_not_hardcoded():
    """60 lb par boisseau de soja -> 36,7437 boisseaux par tonne métrique."""
    assert BUSHELS_PER_TONNE_SOYBEAN == pytest.approx(36.7437, abs=1e-4)


# ===========================================================================
# LE RÉSULTAT
# ===========================================================================
def test_a_material_share_of_sessions_admits_no_origin_at_all(budget):
    """Le résultat de S2 : sur une part non anecdotique des séances, le budget est négatif —
    une fève gratuite, transportée gratuitement, ne rendrait pas le crush rentable."""
    assert budget.share_impossible > 0.005
    assert (budget.frame["budget_usd_t"] < 0).any()
    assert "négatif" in budget.headline


def test_freight_alone_eats_the_whole_budget_far_more_often(budget):
    """Contraste : le budget passe sous le seul coût du fret bien plus souvent qu'il ne
    devient négatif. Entre les deux, il faudrait acheter la fève SOUS le CBOT à l'origine."""
    assert budget.share_below_freight > budget.share_impossible
    assert budget.share_below_freight > 0.05


def test_the_impossible_windows_are_concentrated_in_2023(budget):
    """La concentration temporelle est le fait saillant de S3 : ce n'est pas du bruit autour
    de zéro réparti sur huit ans, c'est un épisode daté."""
    windows = impossible_windows(budget, threshold_usd_t=0.0, min_obs=3)
    assert len(windows) > 0
    years = {pd.Timestamp(value).year for value in windows["start"]}
    assert years == {2023}
    assert windows["duration_days"].max() >= 20


def test_the_windows_calendar_carries_dates_not_just_a_count(budget):
    """Le livrable est un calendrier confrontable à un carnet d'arrivées."""
    windows = impossible_windows(budget, threshold_usd_t=0.0, min_obs=3)
    assert {"start", "end", "duration_days"} <= set(windows.columns)
    assert (pd.to_datetime(windows["end"]) >= pd.to_datetime(windows["start"])).all()


def test_a_higher_threshold_can_only_add_windows(budget):
    """Monotonie : relever le seuil ne peut pas faire disparaître de jours sous le seuil."""
    strict = impossible_windows(budget, threshold_usd_t=0.0, min_obs=3)
    loose = impossible_windows(budget, threshold_usd_t=45.0, min_obs=3)
    assert loose["duration_days"].sum() > strict["duration_days"].sum()


# ===========================================================================
# Cohérence et garde-fous
# ===========================================================================
def test_the_budget_median_is_a_plausible_origination_cost(budget):
    """Contrôle de plausibilité : un basis Gulf plus un fret Chine tournent autour de
    60-100 USD/t. Un budget médian très en dehors signalerait une erreur de conversion."""
    assert 40.0 < budget.median_budget < 140.0


def test_the_module_default_assumption_sits_near_the_median(budget):
    """Le forfait retenu ailleurs dans le module (70 c/bu de basis + 45 USD/t de fret) doit
    tomber dans la plage que le budget autorise en médiane — sinon l'un des deux est faux."""
    from agri.chains.china_soy import DEFAULT_BASIS_CENTS_BU

    assumed = DEFAULT_BASIS_CENTS_BU / 100.0 * BUSHELS_PER_TONNE_SOYBEAN + DEFAULT_FREIGHT_USD_T
    assert abs(assumed - budget.median_budget) < 25.0


def test_budget_frame_has_the_reading_flags(budget):
    assert {"budget_usd_t", "cnf_max_usd_t", "cbot_usd_t", "impossible", "below_freight"} <= set(
        budget.frame.columns
    )
    assert budget.frame["impossible"].equals(budget.frame["budget_usd_t"] < 0)


def test_an_impossible_start_date_raises():
    with pytest.raises(ChinaSoyError, match="aucune date commune"):
        affordable_origination_budget(start="2099-01-01")


def test_the_budget_is_the_margin_stripped_of_its_two_forfaits(budget):
    """Ce que le budget est **exactement**, énoncé plutôt que suggéré.

    Écrit en développement pour vérifier que budget et marge n'étaient pas la même chose ;
    la donnée a répondu qu'ils l'étaient à une transformation affine près, et l'identité est
    exacte au flottant :

        marge = (1 + droit) x USDCNY x (budget - basis_forfait - fret_forfait)

    Le budget n'apporte donc **aucune information nouvelle** — il retire deux paramètres
    arbitraires. C'est précisément ce qui rend son passage à zéro interprétable là où celui
    de la marge ne l'est pas : le zéro de la marge dépend du forfait retenu, celui du budget
    ne dépend de rien. La page dit cela explicitement plutôt que de laisser croire à une
    grandeur indépendante.
    """
    crush = load_real_crush_frame(start="2018-01-01")
    aligned = pd.concat(
        {
            "budget": budget.frame["budget_usd_t"],
            "usdcny": budget.frame["usdcny"],
            "margin": crush["margin"],
        },
        axis=1,
        sort=True,
    ).dropna()

    forfait = DEFAULT_BASIS_CENTS_BU / 100.0 * BUSHELS_PER_TONNE_SOYBEAN + DEFAULT_FREIGHT_USD_T
    predicted = (1.0 + DEFAULT_IMPORT_DUTY) * aligned["usdcny"] * (aligned["budget"] - forfait)
    assert (predicted - aligned["margin"]).abs().max() < 1e-8
