"""Golden tests des six moteurs Tier 2.

Rappel de posture, vérifié en fin de fichier : chaque moteur T2 repose sur une tension
**inférée**, pas sur une citation. Les docstrings doivent dire « il me semble », jamais
« j'ai lu que » — un test le contrôle, parce que c'est la ligne qui se fait démonter en
une réponse si elle dérape.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from agri.chains import crush_tracking as ct
from agri.chains import oil_substitution as os_
from agri.chains import plant_option as po
from agri.chains import white_premium as wp
from agri.fixtures import tier2


# ===========================================================================
# T2-1 — basis contre flat price
# ===========================================================================
def test_plant_crush_hand_computed():
    """Fève 13,00 $/bu, tourteau 400 $/short ton, huile 55 c/lb,
    rendements 43,5 et 10,8 lb/bu, opex 0,42 :

        tourteau : 43,5/2000 x 400 = 8,70
        huile    : 10,8 x 0,55     = 5,94
        crush    : 8,70 + 5,94 - 13,00 - 0,42 = 1,22 $/bu
    """
    index = pd.to_datetime(["2024-01-01"])
    out = ct.plant_crush_usd_bu(
        pd.Series([13.00], index=index),
        pd.Series([400.0], index=index),
        pd.Series([55.0], index=index),
    )
    assert out.iloc[0] == pytest.approx(1.22, abs=1e-10)


def test_board_crush_beats_plant_crush_on_cbot_yields():
    """Aux mêmes prix, le board utilise 44/11 lb et l'usine 43,5/10,8 : l'écart est le
    tracking error minimal, celui qui existe même sans basis.

        board = 0,022 x 400 + 0,11 x 55 - 13,00 = 1,85
        usine = 1,22 (ci-dessus)
        écart = 0,63 $/bu
    """
    from agri.core.units import board_crush_usd_bu

    board = board_crush_usd_bu(13.00, 400.0, 55.0)
    assert board == pytest.approx(1.85, abs=1e-10)
    assert board - 1.22 == pytest.approx(0.63, abs=1e-10)


def test_impossible_yields_are_rejected():
    index = pd.to_datetime(["2024-01-01"])
    args = (
        pd.Series([13.0], index=index),
        pd.Series([400.0], index=index),
        pd.Series([55.0], index=index),
    )
    with pytest.raises(ct.CrushError, match="60 lb"):
        ct.plant_crush_usd_bu(*args, yield_meal_lb_bu=55.0, yield_oil_lb_bu=12.0)


def test_negative_yield_is_rejected():
    index = pd.to_datetime(["2024-01-01"])
    with pytest.raises(ct.CrushError, match="physical range"):
        ct.plant_crush_usd_bu(
            pd.Series([13.0], index=index),
            pd.Series([400.0], index=index),
            pd.Series([55.0], index=index),
            yield_meal_lb_bu=-1.0,
        )


@pytest.fixture(scope="module")
def crush_frame() -> pd.DataFrame:
    return ct.build_tracking(**tier2.crush_tracking())


def test_tracking_error_is_the_difference(crush_frame):
    assert (
        crush_frame["tracking_error"] == crush_frame["board_crush"] - crush_frame["plant_crush"]
    ).all()


def test_optimal_hedge_ratio_is_near_one_overall(crush_frame):
    out = ct.optimal_hedge_ratio(crush_frame)
    assert 0.85 < out.h_star < 1.15
    assert out.variance_reduction_at_h_star >= out.variance_reduction_at_one


def test_rolling_hedge_ratio_moves_away_from_one(crush_frame):
    """Le produit de la page : `h*` n'est pas constant, et il s'éloigne de 1 par épisodes."""
    rolling = ct.rolling_hedge_ratio(crush_frame, window=120)
    assert rolling["h_star"].min() < 0.92
    assert rolling["h_star"].max() > 1.08
    assert rolling.attrs["n_eff"] == pytest.approx(len(rolling) / 120)


def test_decoupling_episodes_are_found(crush_frame):
    episodes = ct.decoupling_episodes(crush_frame, threshold_usd_bu=0.35)
    assert len(episodes) > 0
    assert (episodes["n_obs"] >= 5).all()


def test_decomposition_is_exact(crush_frame):
    """L'identité doit se refermer au flottant près, sur toutes les dates.

    C'est ce qui distingue une décomposition d'une régression : rien n'est estimé, donc
    rien ne peut être biaisé.
    """
    components = ct.decompose_tracking_error(crush_frame)
    assert np.allclose(components["total"], crush_frame["tracking_error"], atol=1e-10)


def test_the_yield_mismatch_term_is_not_a_basis(crush_frame):
    """Le terme que « le décrochage vient du basis » rate complètement.

    L'écart entre les rendements CBOT (44/11 lb) et les rendements réels (43,5/10,8 lb)
    crée un terme proportionnel au **niveau** du board, qui existe même quand tous les
    basis sont nuls. Sur ce jeu, sa dispersion dépasse celle du basis fève.
    """
    components = ct.decompose_tracking_error(crush_frame)
    assert components["oil_yield"].std() > components["bean_basis"].std()
    assert components["oil_yield"].abs().mean() > 0


def test_meal_basis_dominates_the_variability(crush_frame):
    """Le mécanisme que décrivent les gens d'usine : c'est le basis tourteau qui casse le
    hedge, pas la fève."""
    contributions = ct.basis_contributions(crush_frame)
    assert contributions.loc[0, "term"] == "meal_basis"
    assert contributions.loc[0, "share"] > 0.5
    assert contributions["share"].sum() == pytest.approx(1.0)


def test_opex_moves_the_level_but_not_the_variability(crush_frame):
    """Distinction qui compte pour une couverture : un coût fixe décale, il ne fait pas
    trembler."""
    contributions = ct.basis_contributions(crush_frame).set_index("term")
    assert contributions.loc["opex", "std_usd_bu"] == pytest.approx(0.0)
    assert contributions.loc["opex", "mean_usd_bu"] == pytest.approx(0.42)


def test_a_regression_on_the_basis_alone_is_biased(crush_frame):
    """Pourquoi la décomposition exacte remplace la régression.

    Régresser le tracking error sur les trois basis omet les deux termes de rendement, qui
    sont aussi dispersés que le basis fève. Le coefficient de la fève — dont la valeur
    structurelle est exactement +1 — sort biaisé à ~0,99.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        regression = ct.explain_tracking_error(crush_frame)
    assert regression.params["bean_basis"] != pytest.approx(1.0, abs=1e-3)
    assert regression.params["bean_basis"] == pytest.approx(1.0, abs=0.05)
    # les deux basis a forte dispersion sont, eux, bien identifies
    assert regression.params["meal_basis"] == pytest.approx(-0.02175, abs=1e-3)
    assert regression.params["oil_basis"] == pytest.approx(-0.108, abs=1e-3)


# ===========================================================================
# T2-4 — white premium
# ===========================================================================
def test_white_premium_hand_computed():
    """No.5 à 520 $/t, No.11 à 20 c/lb, pol_adjust 1,07 :
        20 x 22,0462262 = 440,924524 $/t
        x 1,07          = 471,789241 $/t sur base blanc
        premium         = 520 - 471,789241 = 48,210759 $/t
    """
    index = pd.to_datetime(["2024-01-01"])
    out = wp.white_premium_usd_t(
        pd.Series([520.0], index=index), pd.Series([20.0], index=index)
    )
    assert out.iloc[0] == pytest.approx(48.210759, abs=1e-6)


def test_fair_value_refining_hand_computed():
    """No.11 à 20 c/lb = 440,924524 $/t de brut :
        énergie          28,000000
        perte 2 %         8,818490
        main d'oeuvre    12,000000
        fret             18,000000
        financement      440,924524 x 0,055 x 45/360 = 3,031356
        total            69,849846
    """
    index = pd.to_datetime(["2024-01-01"])
    costs = wp.fair_value_refining_usd_t(pd.Series([20.0], index=index))
    assert costs["yield_loss"].iloc[0] == pytest.approx(8.818490, abs=1e-6)
    assert costs["financing"].iloc[0] == pytest.approx(3.031356, abs=1e-6)
    assert costs["total"].iloc[0] == pytest.approx(69.849846, abs=1e-6)


def test_richness_hand_computed():
    # 48,210759 - 69,849846 = -21,639087 -> zone CHEAP
    index = pd.to_datetime(["2024-01-01"])
    frame = wp.build_richness(
        pd.Series([520.0], index=index), pd.Series([20.0], index=index)
    )
    assert frame["richness"].iloc[0] == pytest.approx(-21.639087, abs=1e-6)
    assert frame["zone"].iloc[0] == "CHEAP"


def test_pol_adjust_out_of_range_is_rejected():
    index = pd.to_datetime(["2024-01-01"])
    with pytest.raises(wp.WhitePremiumError, match="pol_adjust"):
        wp.white_premium_usd_t(
            pd.Series([520.0], index=index), pd.Series([20.0], index=index), pol_adjust=1.4
        )


def test_pol_adjust_sensitivity_is_material():
    """W-H1 : entre 1,06 et 1,08 la part du temps en zone RICH bouge assez pour qu'on ne
    puisse pas figer le paramètre. C'est ce qui justifie le slider."""
    data = tier2.white_premium()
    table = wp.pol_adjust_sensitivity(
        data["no5_usd_t"], data["no11_cents_lb"], values=np.array([1.06, 1.08])
    )
    low, high = table["share_rich"].tolist()
    assert abs(high - low) > 0.05
    # un pol_adjust plus eleve renchérit la jambe brute, donc comprime le premium
    assert table["mean_white_premium"].is_monotonic_decreasing


def test_richness_summary_produces_the_headline():
    data = tier2.white_premium()
    frame = wp.build_richness(**data)
    summary = wp.summarise_richness(frame)
    assert 0.0 < summary.share_rich < 1.0
    assert len(summary.rich_episodes) > 0
    assert len(summary.cheap_episodes) > 0
    assert "physical availability" in summary.headline


# ===========================================================================
# T2-4 sur données réelles (export Bloomberg) — No.11/No.5 + Henry Hub
# ===========================================================================
from agri.data.bloomberg_loader import DEFAULT_PATH as _BBG_PATH  # noqa: E402

pytestmark_real_t2_4 = pytest.mark.skipif(
    not _BBG_PATH.exists(), reason=f"fichier Bloomberg absent : {_BBG_PATH}"
)


@pytest.mark.skipif(not _BBG_PATH.exists(), reason=f"fichier Bloomberg absent : {_BBG_PATH}")
def test_real_richness_on_2026_08_07_hand_computed():
    """No.11 = 16,45 c/lb, No.5 = 503,4 USD/t, Henry Hub = 2,662 USD/mmBtu :
        raw_usd_t       = 16,45 x 22,0462262           = 362,660421
        white_premium   = 503,4 - 362,660421 x 1,07     = 115,353350
        energie         = 2,662 x 8,0                   = 21,296
        perte_rendement = 0,02 x 362,660421              = 7,253208
        financement     = 362,660421 x 0,055 x 45/360    = 2,493290
        fv_refining     = 21,296 + 7,253208 + 12 + 18 + 2,493290 = 61,042499
        richness        = 115,353350 - 61,042499         = 54,310851
    """
    frame = wp.load_real_richness_frame()
    row = frame.loc["2026-08-07"]
    assert row["no11"] == pytest.approx(16.45)
    assert row["no5"] == pytest.approx(503.4)
    assert row["white_premium"] == pytest.approx(115.353350, abs=1e-4)
    assert row["fv_refining"] == pytest.approx(61.042499, abs=1e-4)
    assert row["richness"] == pytest.approx(54.310851, abs=1e-4)
    assert row["zone"] == "RICH"


@pytest.mark.skipif(not _BBG_PATH.exists(), reason=f"fichier Bloomberg absent : {_BBG_PATH}")
def test_real_energy_cost_tracks_henry_hub_not_a_constant():
    """Le coût énergie doit varier dans le temps (proxy Henry Hub), contrairement au
    forfait DEFAULT_ENERGY_USD_T qu'il remplace — sinon l'upgrade « données réelles »
    serait cosmétique."""
    frame = wp.load_real_richness_frame()
    from agri.chains.white_premium import fair_value_refining_usd_t
    from agri.data.bloomberg_loader import load as load_bloomberg

    energy_leg = fair_value_refining_usd_t(
        frame["no11"], energy_usd_t=(load_bloomberg("henry_hub") * 8.0).reindex(frame.index)
    )["energy"]
    assert energy_leg.std() > 0.5
    assert energy_leg.nunique() > 100


@pytest.mark.skipif(not _BBG_PATH.exists(), reason=f"fichier Bloomberg absent : {_BBG_PATH}")
def test_real_richness_summary_and_headline_run_on_real_data():
    frame = wp.load_real_richness_frame()
    summary = wp.summarise_richness(frame)
    assert 0.0 < summary.share_rich < 1.0
    assert "physical availability" in summary.headline
    assert frame.attrs["energy_source"] == "henry_hub_real"


# ===========================================================================
# T2-5 — l'usine comme option
# ===========================================================================
@pytest.fixture(scope="module")
def ou_params() -> po.OUParams:
    return po.calibrate_ou(tier2.plant_margin())


def test_ou_calibration_recovers_kappa_and_sigma(ou_params):
    """theta est volontairement testé plus large : c'est le paramètre le plus lent à
    estimer. Avec kappa = 0,035 la demi-vie est de 20 périodes, donc 1 600 observations
    n'en valent qu'une trentaine d'indépendantes pour la moyenne de long terme."""
    assert ou_params.kappa == pytest.approx(tier2.TRUE_OU_KAPPA, abs=0.008)
    assert ou_params.sigma == pytest.approx(tier2.TRUE_OU_SIGMA, abs=0.4)
    assert ou_params.theta == pytest.approx(tier2.TRUE_OU_THETA, abs=3.0)


def test_half_life_is_consistent_with_kappa(ou_params):
    assert ou_params.half_life == pytest.approx(np.log(2) / ou_params.kappa)


def test_calibration_refuses_a_random_walk():
    """O-H1 : calibrer un OU sur une marche aléatoire donnerait un kappa quasi nul et une
    valeur d'option absurde, sans jamais planter. Le refus est explicite."""
    rng = np.random.default_rng(0)
    walk = pd.Series(
        np.cumsum(rng.normal(size=400)),
        index=pd.date_range("2020-01-01", periods=400, freq="B"),
    )
    with pytest.raises(po.PlantOptionError, match="stationary"):
        po.calibrate_ou(walk)


def test_calibration_can_be_forced_with_a_warning_path():
    rng = np.random.default_rng(0)
    walk = pd.Series(
        np.cumsum(rng.normal(size=400)),
        index=pd.date_range("2020-01-01", periods=400, freq="B"),
    )
    forced = po.calibrate_ou(walk, strict=False)
    assert forced.stationarity.verdict != "stationary"


@pytest.fixture(scope="module")
def band(ou_params) -> po.HysteresisBand:
    return po.solve_hysteresis(
        ou_params, cost_restart=120.0, cost_shutdown=60.0, cost_idle=0.5
    )


def test_value_iteration_converges(band):
    assert band.converged
    assert band.n_iterations < 5_000


def test_the_frontier_is_a_band_not_a_threshold(band):
    """LE résultat de la page : M_off < 0 < M_on, avec une hystérésis strictement positive.

    Une règle « marge < 0 » suppose M_off = M_on = 0. La frontière optimale ne l'est
    jamais dès que l'arrêt et le redémarrage coûtent quelque chose.
    """
    assert band.m_off < 0.0 < band.m_on
    assert band.width > 0.0
    assert "hysteresis" in band.headline


def test_higher_switching_costs_widen_the_band(ou_params):
    cheap = po.solve_hysteresis(ou_params, cost_restart=20.0, cost_shutdown=10.0, cost_idle=0.5)
    expensive = po.solve_hysteresis(
        ou_params, cost_restart=400.0, cost_shutdown=200.0, cost_idle=0.5
    )
    assert expensive.width > cheap.width


def test_plant_value_increases_with_volatility(ou_params):
    """La démonstration contre-intuitive, et c'est elle qui fait la page.

    À moyenne de marge égale, une usine dont la marge est **plus volatile** vaut plus,
    parce que la flexibilité d'arrêt tronque la queue basse. C'est un chiffre posé sur un
    débat qui se tient d'habitude en slogans.
    """
    table = po.volatility_sensitivity(
        ou_params, cost_restart=120.0, cost_shutdown=60.0, cost_idle=0.5
    )
    assert table["value_at_theta"].is_monotonic_increasing
    assert table["band_width"].is_monotonic_increasing
    assert table["value_at_theta"].iloc[-1] > table["value_at_theta"].iloc[0]


def test_heuristic_rule_shuts_down_more_often_than_the_frontier(band):
    margin = tier2.plant_margin()
    comparison = po.compare_to_heuristic(margin, band, threshold=0.0, consecutive_periods=4)
    assert comparison.n_shutdowns_heuristic >= comparison.n_shutdowns_optimal
    assert "shutdowns" in comparison.headline


def test_negative_switching_costs_are_rejected(ou_params):
    with pytest.raises(po.PlantOptionError, match="costs"):
        po.solve_hysteresis(ou_params, cost_restart=-1.0, cost_shutdown=60.0, cost_idle=0.5)


# ===========================================================================
# T2-5 sur données réelles — marge de crush CBOT, entièrement réelle
# ===========================================================================
@pytest.mark.skipif(not _BBG_PATH.exists(), reason=f"fichier Bloomberg absent : {_BBG_PATH}")
def test_real_crush_margin_on_2026_08_07_hand_computed():
    """Soja 11,565 USD/bu, tourteau 308,1 USD/short ton, huile 68,16 c/lb :
        tourteau : 44/2000 x 308,1 = 6,7782
        huile    : 11 x 0,6816    = 7,4976
        crush    : 6,7782 + 7,4976 - 11,565 = 2,7108 USD/bu
    """
    margin = po.real_board_crush_margin()
    assert margin.loc["2026-08-07"] == pytest.approx(2.7108, abs=1e-4)


@pytest.mark.skipif(not _BBG_PATH.exists(), reason=f"fichier Bloomberg absent : {_BBG_PATH}")
def test_real_margin_uses_only_real_legs_no_parameters():
    """Contrairement à T1-2 (roll omis) ou T2-4 (labeur/fret paramétrés), les trois
    jambes ici sont entièrement réelles — aucun terme constant injecté."""
    margin = po.real_board_crush_margin(start="2020-01-01")
    assert margin.std() > 0.1
    assert margin.nunique() > 500


@pytest.mark.skipif(not _BBG_PATH.exists(), reason=f"fichier Bloomberg absent : {_BBG_PATH}")
def test_real_margin_fails_stationarity_and_that_is_the_finding():
    """Résultat vérifié en session : aucune fenêtre testée (36 ans complets, ni les
    sous-périodes depuis 2005) ne passe le verdict conjoint ADF+KPSS. Ce test verrouille
    que le diagnostic le dit clairement plutôt que de masquer l'échec."""
    margin = po.real_board_crush_margin()
    diagnostic = po.diagnose_real_margin_stationarity(margin)
    assert diagnostic.stationarity.verdict != "stationary"
    assert "does not behave like a homogeneous OU" in diagnostic.headline
    assert diagnostic.n_obs == len(margin)


@pytest.mark.skipif(not _BBG_PATH.exists(), reason=f"fichier Bloomberg absent : {_BBG_PATH}")
def test_indicative_calibration_still_produces_a_usable_band():
    """La calibration indicative (strict=False) doit rester utilisable — bande valide,
    convergence de l'itération de la valeur — même si le verdict de stationnarité est
    defavorable. C'est ce qui la rend affichable comme resultat illustratif."""
    margin = po.real_board_crush_margin(start="2018-01-01")
    ou = po.calibrate_real_ou_indicative(margin)
    assert ou.stationarity.verdict != "stationary"
    band = po.solve_hysteresis(ou, cost_restart=0.30, cost_shutdown=0.15, cost_idle=0.02)
    assert band.converged
    assert band.m_off < 0 < band.m_on


# ===========================================================================
# T2-6 — substitution inter-huiles
# ===========================================================================
@pytest.fixture(scope="module")
def spreads() -> pd.DataFrame:
    return os_.build_spreads(tier2.oil_prices())


def test_spreads_are_built_once_per_pair(spreads):
    assert set(spreads.columns) == {
        "canola_minus_palm",
        "canola_minus_soy",
        "palm_minus_soy",
    }


def test_half_life_formula():
    # b = -0,10 -> demi-vie = -ln(2)/ln(0,90) = 6,579
    assert -np.log(2) / np.log(0.90) == pytest.approx(6.5788, abs=1e-3)


def test_substitution_bound_finds_the_regime_split(spreads):
    """Le jeu impose un AR à seuil : lent sous 60 $/t d'écart, rapide au-delà.

    On ne teste pas la restitution exacte des demi-vies — un modèle à seuil estimé par
    partition est une approximation (S-H4) — mais la **séparation**, qui est ce que la
    page affirme.
    """
    bound = os_.substitution_bound(
        spreads["palm_minus_soy"],
        pair="palm-soy",
        threshold_usd_t=tier2.SUBSTITUTION_THRESHOLD_USD_T,
    )
    assert bound.narrow.is_mean_reverting
    assert bound.wide.is_mean_reverting
    assert bound.wide.half_life_days < bound.narrow.half_life_days
    assert bound.substitution_kicks_in
    assert "substitution bound" in bound.headline


def test_lags_are_computed_before_the_regime_filter(spreads):
    """Le bug que ce test verrouille.

    Filtrer le sous-échantillon **avant** de calculer `.diff()` et `.shift()` calcule des
    écarts entre observations non adjacentes dans le temps, ce qui fabrique une fausse
    moyenne-réversion : deux points distants de trois semaines paraissent avoir convergé
    en un pas. Sur ce jeu, l'erreur ramenait une demi-vie de 173 jours à 10.
    """
    series = spreads["palm_minus_soy"]
    mask = (series.shift(1).abs() < 20.0)

    correct = os_.estimate_half_life(series, mask=mask)
    naive = os_.estimate_half_life(series[mask])       # l'erreur : filtrer d'abord
    assert correct.beta != pytest.approx(naive.beta, abs=1e-6)
    assert abs(naive.beta) > abs(correct.beta)          # l'erreur exagère la réversion


def test_screen_flags_non_stationary_pairs(spreads):
    """Le contrôle qui empêche de lire une demi-vie sur une racine unitaire.

    Le jeu contient un couple construit comme une vraie relation (palme-soja) et deux
    couples qui n'en sont pas. Le screen doit les distinguer, sinon la page publierait
    trois « bornes de substitution » dont deux n'existent pas.
    """
    table = os_.screen_all_pairs(spreads).set_index("pair")
    assert table.loc["palm_minus_soy", "stationarity"] == "stationary"
    assert table.loc["palm_minus_soy", "substitution_kicks_in"]
    assert table.loc["canola_minus_soy", "stationarity"] != "stationary"
    assert "do not read" in table.loc["canola_minus_soy", "note"]


def test_a_pure_random_walk_has_no_half_life():
    rng = np.random.default_rng(1)
    walk = pd.Series(
        np.cumsum(rng.normal(size=400)),
        index=pd.date_range("2020-01-01", periods=400, freq="B"),
    )
    out = os_.estimate_half_life(walk)
    assert not out.is_mean_reverting
    assert "no detectable mean reversion" in out.summary


def test_too_few_oils_is_rejected():
    index = pd.date_range("2024-01-01", periods=100, freq="B")
    with pytest.raises(os_.SubstitutionError, match="two oils"):
        os_.build_spreads({"palm": pd.Series(900.0, index=index)})


# ===========================================================================
# Posture Tier 2 — la règle absolue
# ===========================================================================
# `plant_option` est volontairement hors de cette liste depuis sa refonte : la page ne
# repose plus sur une tension inférée du marché (« il me semble que les desks se
# disputent ») mais sur la critique d'une règle **réellement utilisée** — le
# `consecutive_below(margin, 0, N=4)` des pages zinc et lithium — dont elle calcule
# l'implication. Ce n'est pas un affaiblissement du garde-fou : c'est un statut
# épistémique différent, couvert par son propre test juste en dessous.
#
# `basis_flat` (T2-1) et `grain_carry` (T2-2) ont quitté le portefeuille le 10/08/2026 :
# l'export Bloomberg ne contient que des premiers mois génériques, donc ni série cash ni
# spread calendaire. Leur livrable n'était pas calculable et serait resté synthétique.
# Code conservé dans `_archive/`, pas supprimé.
INFERRED_TENSION_MODULES = [ct, wp, os_]


@pytest.mark.parametrize("module", INFERRED_TENSION_MODULES)
def test_tier2_modules_frame_the_tension_as_inferred(module):
    """« Il me semble que », jamais « j'ai lu que ».

    Présenter une tension inférée comme une citation se fait démonter en une ligne, parce
    que c'est faux. Ce test garde la frontière au niveau du code plutôt qu'au niveau de la
    discipline.
    """
    import re

    doc = module.__doc__ or ""
    lowered = doc.lower()
    assert "INFERRED" in doc, f"{module.__name__} does not mark its tension as inferred"
    assert "it seems to me" in lowered, f"{module.__name__} does not use the cautious phrasing"

    # "I read that" is acceptable only when preceded by "never" — i.e. cited as the
    # phrasing to NOT use. Any other occurrence is a claim presented as sourced when
    # it is not.
    for match in re.finditer(r"i read that", lowered):
        preceding = lowered[max(0, match.start() - 30) : match.start()]
        assert "never" in preceding, (
            f"{module.__name__} uses \"I read that\" outside the disclaimer: "
            "an inferred tension presented as a citation falls apart in one line"
        )


def test_plant_option_rests_on_a_rule_it_can_point_at_not_an_inferred_tension():
    """Le garde-fou qui remplace celui d'au-dessus, pour le seul module refondu.

    La page ne revendique aucune dispute de marché — elle vise une règle qu'on peut
    montrer du doigt, et elle doit dire ce qu'elle en fait : nommer la règle, annoncer
    qu'elle en calcule l'implication, et exposer le contrefactuel qui empêche de
    présenter un raffinement comme le sujet principal.
    """
    doc = po.__doc__ or ""
    lowered = doc.lower()

    assert "consecutive_below" in lowered, "the targeted rule is not named"
    assert "zinc" in lowered and "lithium" in lowered, "the targeted pages are not named"
    assert "inverting the question" in lowered
    assert "counterfactual" in lowered, "the never-stop counterfactual is not announced"

    # Same rule as for the inferred-tension modules: no claim presented as sourced
    # that is not.
    import re

    for match in re.finditer(r"i read that", lowered):
        preceding = lowered[max(0, match.start() - 30) : match.start()]
        assert "never" in preceding
