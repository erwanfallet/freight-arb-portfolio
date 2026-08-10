"""Golden tests T3-1 — le seuil LCFS.

Constantes du modèle, utilisées dans tous les calculs à la main ci-dessous :
    CI soyoil = 27, CI UCO = 15            -> différentiel de CI = 12 gCO2e/MJ
    crédit 45Z soyoil = (50 - 27)/50       = 0,46 $/gal
    rendement                              = 7,6 lb/gal
    conversion LCFS = 134,47e-6            (134,47 MJ/gal x 1e-6 t/g)

    dénominateur du seuil = 12 x 1,0 x 134,47e-6 = 1,61364e-3
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agri.chains.feedstock_lcfs import (
    SOYOIL_DOMESTIC,
    UCO_IMPORTED,
    Feedstock,
    FeedstockError,
    NoBreakevenInRange,
    calibration_gap_45z,
    chow_break_test,
    crush_balance,
    feedstock_breakeven_usd_lb,
    gate_value,
    lcfs_breakeven,
    lcfs_breakeven_numeric,
    lcfs_value_usd_gal,
    rolling_energy_beta,
    winner_grid,
)
from agri.fixtures.feedstock_lcfs import (
    BETA_AFTER_BREAK,
    BETA_BEFORE_BREAK,
    RVO_BREAK_DATE,
    build,
)

LCFS_DENOMINATOR = 12 * 1.0 * 134.47e-6      # 1,61364e-3


@pytest.fixture(scope="module")
def series() -> dict[str, pd.Series]:
    return build()


# ===========================================================================
# Le crédit 45Z (L-H4)
# ===========================================================================
def test_45z_credit_on_soyoil():
    # 1,00 x (50 - 27)/50 = 0,46 $/gal
    assert SOYOIL_DOMESTIC.credit_45z_usd_gal() == pytest.approx(0.46, abs=1e-12)


def test_45z_credit_is_zero_for_imported_feedstock_however_clean():
    """Le point réglementaire central : 45Z est une règle d'origine, pas une règle de CI.

    L'UCO importé a un CI de 15, bien meilleur que le soyoil à 27, et touche pourtant zéro.
    C'est précisément ce que le LCFS doit compenser — donc c'est tout le sujet de la page.
    """
    assert UCO_IMPORTED.carbon_intensity < SOYOIL_DOMESTIC.carbon_intensity
    assert UCO_IMPORTED.credit_45z_usd_gal() == 0.0


def test_45z_credit_floors_at_zero_for_dirty_feedstock():
    dirty = Feedstock("filière carbonée", 65.0, north_american=True)
    assert dirty.credit_45z_usd_gal() == 0.0


def test_calibration_gap_is_shown_not_absorbed():
    """L-H4 : l'écart de 3 c/gal avec la valeur publiée est affiché, pas corrigé en douce."""
    gap = calibration_gap_45z()
    assert gap["modelled_usd_gal"] == pytest.approx(0.46)
    assert gap["published_usd_gal"] == pytest.approx(0.49)
    assert gap["gap_usd_gal"] == pytest.approx(0.03, abs=1e-9)
    assert gap["gap_pct"] == pytest.approx(0.061224, abs=1e-5)


# ===========================================================================
# La jambe LCFS — le piège d'unité
# ===========================================================================
def test_lcfs_leg_hand_computed():
    # 200 $/t x (95 - 27) x 1,0 x 134,47e-6 = 200 x 68 x 134,47e-6 = 1,828792 $/gal
    assert lcfs_value_usd_gal(200.0, 27.0, ci_std=95.0) == pytest.approx(1.828792, abs=1e-6)


def test_cleaner_feedstock_earns_more_lcfs():
    clean = lcfs_value_usd_gal(200.0, 15.0, ci_std=95.0)
    dirty = lcfs_value_usd_gal(200.0, 27.0, ci_std=95.0)
    # l'écart vaut 200 x 12 x 134,47e-6 = 0,322728 $/gal
    assert clean - dirty == pytest.approx(0.322728, abs=1e-6)


def test_gate_value_stack_sums_to_total():
    value = gate_value(
        SOYOIL_DOMESTIC, ulsd_usd_gal=2.55, rin_d4_usd=0.62, lcfs_usd_t=200.0
    )
    # diesel 2,55 + RIN 0,62 x 1,7 = 1,054 + LCFS 1,828792 + 45Z 0,46 = 5,892792
    assert value.total_usd_gal == pytest.approx(5.892792, abs=1e-6)
    assert sum(value.stack.values()) == pytest.approx(value.total_usd_gal)


def test_feedstock_breakeven_hand_computed():
    # (5,892792 - 0,55 opex - 0,25 roi) / 7,6 = 5,092792 / 7,6 = 0,670104 $/lb
    out = feedstock_breakeven_usd_lb(
        SOYOIL_DOMESTIC, ulsd_usd_gal=2.55, rin_d4_usd=0.62, lcfs_usd_t=200.0
    )
    assert out == pytest.approx(0.670104, abs=1e-6)


def test_zero_yield_is_rejected():
    with pytest.raises(FeedstockError, match="yield_lb_gal"):
        feedstock_breakeven_usd_lb(
            SOYOIL_DOMESTIC, ulsd_usd_gal=2.55, rin_d4_usd=0.62, lcfs_usd_t=200.0, yield_lb_gal=0.0
        )


# ===========================================================================
# LE POINT DE BASCULE — le livrable
# ===========================================================================
def test_lcfs_threshold_at_price_parity():
    """À prix de feedstock égaux, le LCFS doit compenser tout le 45Z à lui seul.

        LCFS* = 0,46 / 1,61364e-3 = 285,07 $/t CO2e

    C'est le chiffre nu du désaccord : sans avantage de prix, il faut un crédit LCFS de
    285 $/t pour que l'UCO importé égale le soyoil domestique.
    """
    out = lcfs_breakeven(
        price_domestic_usd_lb=0.50,
        price_imported_usd_lb=0.50,
        lcfs_current_usd_t=95.0,
    )
    assert out.lcfs_star_usd_t == pytest.approx(0.46 / LCFS_DENOMINATOR, abs=1e-6)
    assert out.lcfs_star_usd_t == pytest.approx(285.070, abs=1e-2)
    assert out.ci_gap == 12.0


def test_a_cheaper_import_collapses_the_threshold():
    """L'UCO décoté de 6 c/lb : le seuil tombe de 285 à 2,5 $/t.

        numérateur = 0,46 + (-0,06 x 7,6) = 0,46 - 0,456 = 0,004
        LCFS*      = 0,004 / 1,61364e-3   = 2,479 $/t

    C'est le résultat qui rend la page utile : le seuil est **extrêmement sensible** au
    différentiel de prix feedstock. Six cents la livre suffisent à renverser 46 c/gal de
    crédit fiscal. C'est ce chiffre-là qu'un insider peut confirmer ou démolir.
    """
    out = lcfs_breakeven(
        price_domestic_usd_lb=0.52,
        price_imported_usd_lb=0.46,
        lcfs_current_usd_t=95.0,
    )
    assert out.price_gap_usd_lb == pytest.approx(-0.06)
    assert out.lcfs_star_usd_t == pytest.approx(2.479, abs=1e-3)


def test_a_pricier_import_raises_the_threshold():
    # numérateur = 0,46 + 0,02 x 7,6 = 0,612 ; 0,612 / 1,61364e-3 = 379,27 $/t
    out = lcfs_breakeven(
        price_domestic_usd_lb=0.50,
        price_imported_usd_lb=0.52,
        lcfs_current_usd_t=95.0,
    )
    assert out.lcfs_star_usd_t == pytest.approx(379.267, abs=1e-2)


def test_threshold_does_not_depend_on_diesel_rin_or_plant_costs():
    """L'argument qui rend le seuil robuste, et qu'il faut faire passer dans le mail.

    Les termes diesel, RIN, opex et ROI sont identiques pour les deux filières et
    disparaissent de la différence. Le seuil ne bouge que par le différentiel de CI et le
    différentiel de prix feedstock — donc il survit à l'ignorance sur la structure de
    coûts d'une usine qu'on ne connaît pas.
    """
    reference = lcfs_breakeven(
        price_domestic_usd_lb=0.52, price_imported_usd_lb=0.46, lcfs_current_usd_t=95.0
    )
    # le seuil analytique ne prend meme pas ces arguments ; on verifie via la forme
    # numerique, qui les prend, que le resultat est identique quels qu'ils soient
    cheap_diesel = lcfs_breakeven_numeric(
        price_domestic_usd_lb=0.52,
        price_imported_usd_lb=0.46,
        ulsd_usd_gal=1.80,
        rin_d4_usd=0.30,
        lcfs_current_usd_t=95.0,
    )
    rich_diesel = lcfs_breakeven_numeric(
        price_domestic_usd_lb=0.52,
        price_imported_usd_lb=0.46,
        ulsd_usd_gal=4.20,
        rin_d4_usd=1.40,
        lcfs_current_usd_t=95.0,
    )
    assert cheap_diesel.theta_star == pytest.approx(rich_diesel.theta_star, abs=1e-6)
    assert cheap_diesel.theta_star == pytest.approx(reference.lcfs_star_usd_t, abs=1e-4)


def test_closed_form_and_numeric_solver_agree():
    """Contrôle croisé : si les deux divergent, la forme fermée a une erreur d'algèbre."""
    analytic = lcfs_breakeven(
        price_domestic_usd_lb=0.50, price_imported_usd_lb=0.50, lcfs_current_usd_t=95.0
    )
    numeric = lcfs_breakeven_numeric(
        price_domestic_usd_lb=0.50,
        price_imported_usd_lb=0.50,
        ulsd_usd_gal=2.55,
        rin_d4_usd=0.62,
        lcfs_current_usd_t=95.0,
        hi=600.0,
    )
    assert analytic.lcfs_star_usd_t == pytest.approx(numeric.theta_star, abs=1e-4)


def test_headline_names_the_threshold_and_the_distance():
    history = pd.Series([60.0, 80.0, 95.0, 110.0, 130.0])
    out = lcfs_breakeven(
        price_domestic_usd_lb=0.52,
        price_imported_usd_lb=0.50,
        lcfs_current_usd_t=95.0,
        lcfs_history=history,
    )
    headline = out.headline
    assert "$/t CO2e" in headline
    assert "standard deviations" in headline
    assert "soy takes the share" in headline


def test_an_import_dirtier_than_the_domestic_has_no_threshold():
    """S'il perd sur les deux tableaux, il n'y a pas de seuil — et l'erreur doit le dire."""
    dirty_import = Feedstock("UCO importé sale", 35.0, north_american=False)
    with pytest.raises(FeedstockError, match="no threshold"):
        lcfs_breakeven(
            imported=dirty_import,
            price_domestic_usd_lb=0.50,
            price_imported_usd_lb=0.50,
            lcfs_current_usd_t=95.0,
        )


def test_numeric_solver_reports_no_crossing_in_range():
    with pytest.raises(NoBreakevenInRange):
        lcfs_breakeven_numeric(
            price_domestic_usd_lb=0.90,      # soyoil hors de prix : l'UCO gagne partout
            price_imported_usd_lb=0.20,
            ulsd_usd_gal=2.55,
            rin_d4_usd=0.62,
            lcfs_current_usd_t=95.0,
        )


# ===========================================================================
# S4 — la heatmap
# ===========================================================================
def test_winner_grid_has_both_zones():
    grid = winner_grid()
    assert set(grid["winner"].unique()) == {"imported UCO", "domestic soyoil"}


def test_imports_win_at_high_lcfs_and_low_ci():
    grid = winner_grid(
        ci_imported_values=np.array([12.0]),
        lcfs_values=np.array([0.0, 400.0]),
        price_domestic_usd_lb=0.50,
        price_imported_usd_lb=0.50,
    )
    low, high = grid.sort_values("lcfs_usd_t")["winner"].tolist()
    assert low == "domestic soyoil"
    assert high == "imported UCO"


def test_advantage_is_monotonic_in_lcfs():
    grid = winner_grid(ci_imported_values=np.array([15.0]))
    ordered = grid.sort_values("lcfs_usd_t")
    assert ordered["advantage_usd_lb"].is_monotonic_increasing


# ===========================================================================
# Le bilan de trituration
# ===========================================================================
def test_crush_balance_hand_computed():
    """5 Md gal de RD, 40 % de part soyoil, 7,6 lb/gal, 11 lb d'huile par boisseau.

        soyoil requis = 5e9 x 0,40 x 7,6      = 1,52e10 lb
        crush requis  = 1,52e10 / 11          = 1 381 818 182 bu
        par jour      = / 365                 = 3 785 803 bu/jour
        gap           = 3 785 803 - 2 500 000 = 1 285 803 bu/jour
    """
    out = crush_balance(
        rvo_gallons=5e9, soyoil_share=0.40, installed_capacity_bu_day=2_500_000.0
    )
    assert out.soyoil_required_lb == pytest.approx(1.52e10)
    assert out.crush_required_bu == pytest.approx(1_381_818_181.8, rel=1e-9)
    assert out.crush_required_bu_day == pytest.approx(3_785_803.24, abs=1e-2)
    assert out.gap_bu_day == pytest.approx(1_285_803.24, abs=1e-2)
    assert out.is_short


def test_crush_balance_when_capacity_is_sufficient():
    out = crush_balance(
        rvo_gallons=1e9, soyoil_share=0.20, installed_capacity_bu_day=2_500_000.0
    )
    assert not out.is_short
    assert "to spare" in out.headline


def test_crush_headline_quantifies_the_shortfall():
    out = crush_balance(
        rvo_gallons=5e9, soyoil_share=0.40, installed_capacity_bu_day=2_500_000.0
    )
    assert "short" in out.headline
    assert "bu/day" in out.headline


def test_soyoil_share_out_of_range_is_rejected():
    with pytest.raises(FeedstockError, match="soyoil_share"):
        crush_balance(rvo_gallons=5e9, soyoil_share=1.4, installed_capacity_bu_day=2.5e6)


# ===========================================================================
# T3-5 — bêta énergie et rupture de politique
# ===========================================================================
def test_rolling_beta_recovers_the_two_regimes(series):
    out = rolling_energy_beta(series["soyoil"], series["brent"], window=120)
    # bien avant la rupture, la fenetre ne contient que le regime bas
    early = out[out.index < RVO_BREAK_DATE - pd.Timedelta(days=200)]["beta"].mean()
    # bien apres, elle ne contient que le regime haut
    late = out[out.index > RVO_BREAK_DATE + pd.Timedelta(days=200)]["beta"].mean()
    assert early == pytest.approx(BETA_BEFORE_BREAK, abs=0.10)
    assert late == pytest.approx(BETA_AFTER_BREAK, abs=0.10)
    assert late > early


def test_chow_detects_the_policy_break(series):
    out = chow_break_test(series["soyoil"], series["brent"], RVO_BREAK_DATE)
    assert out.rejects_stability
    assert out.beta_after > out.beta_before
    assert "significant break" in out.summary


def test_chow_finds_nothing_within_a_single_regime(series):
    """Contrôle négatif, sur un échantillon qui ne contient aucune rupture.

    On coupe au milieu de la période PRÉ-rupture, des deux côtés dans le même régime.
    Sans ce test, on ne saurait pas si le Chow détecte la politique ou s'il détecte
    n'importe quoi.
    """
    pre_break = slice(None, RVO_BREAK_DATE - pd.Timedelta(days=1))
    out = chow_break_test(
        series["soyoil"].loc[pre_break], series["brent"].loc[pre_break], "2024-08-01"
    )
    assert not out.rejects_stability


def test_a_split_that_straddles_the_real_break_also_fires(series):
    """Pourquoi la date de rupture doit être choisie a priori, jamais cherchée.

    Couper en juin 2024 met la vraie rupture de mars 2026 à l'intérieur du sous-échantillon
    « après », dont le bêta moyen (~0,29) diffère donc réellement du bêta d'avant (~0,19).
    Le test rejette — à juste titre, il y a bien une différence — mais la date qu'il
    désigne n'est pas celle de l'événement. Balayer toutes les dates et retenir le F
    maximal produirait un « point de rupture » qui n'est qu'un artefact de recherche.
    Les seules dates légitimes ici sont celles du calendrier réglementaire.
    """
    out = chow_break_test(series["soyoil"], series["brent"], "2024-06-03")
    assert out.rejects_stability
    at_policy_date = chow_break_test(series["soyoil"], series["brent"], RVO_BREAK_DATE)
    # la vraie date de politique donne un F nettement plus fort
    assert at_policy_date.f_stat > out.f_stat


def test_chow_refuses_a_date_too_close_to_the_edge(series):
    with pytest.raises(FeedstockError, match="too short"):
        chow_break_test(series["soyoil"], series["brent"], "2023-01-20")


def test_rolling_beta_refuses_a_window_longer_than_the_sample(series):
    with pytest.raises(FeedstockError, match="not enough"):
        rolling_energy_beta(series["soyoil"].head(50), series["brent"].head(50), window=120)


# ===========================================================================
# Le fixture impose bien le phénomène
# ===========================================================================
def test_fixture_lcfs_crosses_the_parity_threshold(series):
    lcfs = series["lcfs"]
    assert lcfs.min() < 285.07 < lcfs.max()


def test_fixture_uco_trades_at_a_discount(series):
    assert (series["uco"] < series["soyoil"]).mean() > 0.95


def test_fixture_is_deterministic():
    a = build(seed=3)["lcfs"]
    b = build(seed=3)["lcfs"]
    pd.testing.assert_series_equal(a, b)
