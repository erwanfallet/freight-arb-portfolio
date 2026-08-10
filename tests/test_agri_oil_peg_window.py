"""Golden tests T2-6 — la fenêtre de parité fixe, et le résultat négatif qu'elle produit.

Deux choses sont gardées ici, et la seconde compte autant que la première.

1. `test_the_substitution_band_does_not_exist` : le résultat de la page est **négatif** —
   les grands écarts ne reviennent pas. Un résultat négatif est fragile parce qu'il est
   toujours tentant de le convertir en résultat positif en changeant un paramètre. Le test
   le fige, et il le fige sur plusieurs réglages de fenêtre et de quantile.

2. `test_splitting_on_the_raw_level_manufactures_the_expected_answer` : montre d'où vient
   réellement l'artefact. Ma première hypothèse — « comparer à une constante plutôt qu'à une
   médiane glissante » — était fausse, et ce test l'a établi avant que la page ne l'affirme.
   L'artefact vient de découper sur le **niveau absolu** du spread : comme la médiane vaut
   environ -83 USD/t, |spread| grand sélectionne l'ère 2004-2005 au lieu d'écarts anormaux.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agri.chains.oil_substitution import (
    CENTS_LB_TO_USD_T,
    MYR_PEG_END,
    MYR_PEG_RATE,
    MYR_PEG_START,
    SubstitutionError,
    estimate_half_life,
    load_peg_window_spread,
    rolling_deviation,
    structural_drift,
    substitution_verdict,
)
from agri.data.bloomberg_loader import DEFAULT_PATH, load

pytestmark = pytest.mark.skipif(
    not DEFAULT_PATH.exists(), reason=f"fichier Bloomberg absent : {DEFAULT_PATH}"
)


@pytest.fixture(scope="module")
def frame():
    return load_peg_window_spread()


# ===========================================================================
# La fenêtre et sa conversion
# ===========================================================================
def test_the_window_is_exactly_the_peg_period(frame):
    assert frame.index.min() >= pd.Timestamp(MYR_PEG_START)
    assert frame.index.max() <= pd.Timestamp(MYR_PEG_END)
    assert len(frame) > 1_500


def test_the_palm_conversion_is_a_division_by_a_constant(frame):
    """Tout l'intérêt de la fenêtre : la devise n'est pas estimée, elle est décrétée."""
    pd.testing.assert_series_equal(
        frame["palm_usd"], (frame["palm_myr"] / MYR_PEG_RATE).rename("palm_usd")
    )
    assert MYR_PEG_RATE == pytest.approx(3.80)


def test_the_soy_leg_uses_the_cents_to_tonne_conversion(frame):
    soy_raw = load("cbot_soyoil").reindex(frame.index)
    pd.testing.assert_series_equal(
        frame["soy_usd"], (soy_raw * CENTS_LB_TO_USD_T).rename("soy_usd")
    )


def test_both_legs_land_in_a_plausible_usd_range(frame):
    """Une huile végétale se traite entre 200 et 900 USD/t sur cette période. Une conversion
    ratée sortirait de cette plage d'un facteur visible."""
    assert 200 < frame["palm_usd"].median() < 900
    assert 200 < frame["soy_usd"].median() < 900


def test_palm_trades_below_soy_most_of_the_time(frame):
    assert (frame["spread"] < 0).mean() > 0.75


# ===========================================================================
# LE RÉSULTAT NÉGATIF
# ===========================================================================
@pytest.mark.parametrize("window", [125, 250, 375])
@pytest.mark.parametrize("quantile", [0.65, 0.70, 0.80])
def test_the_substitution_band_does_not_exist(frame, window, quantile):
    """LE test de la page, et il est répété sur neuf réglages.

    La thèse prédit que les grands écarts reviennent PLUS vite que les petits. Sur toutes
    les combinaisons de fenêtre glissante et de quantile de séparation, c'est l'inverse qui
    sort. Un résultat négatif qui ne tient qu'à un réglage n'en est pas un.
    """
    verdict = substitution_verdict(frame["spread"], window=window, quantile=quantile)
    assert not verdict.substitution_band_exists
    assert np.isfinite(verdict.narrow.half_life_days)


def test_narrow_deviations_do_revert_quickly(frame):
    """Le contraste qui rend le résultat lisible : ce n'est pas que rien ne revient jamais.
    Les petits écarts reviennent vite — c'est de la microstructure, pas de la substitution."""
    verdict = substitution_verdict(frame["spread"])
    assert verdict.narrow.half_life_days < 30
    assert verdict.narrow.pvalue < 0.01


def test_the_wide_regime_coefficient_is_not_merely_small(frame):
    """Distinguer « pas de puissance » de « pas d'effet » : l'échantillon large est ample et
    le coefficient n'est pas seulement petit, il est du mauvais signe."""
    verdict = substitution_verdict(frame["spread"])
    assert verdict.wide.n_obs > 300
    assert verdict.wide.pvalue > 0.10
    assert verdict.wide.beta >= -0.005


def test_the_headline_states_the_negative_result_plainly(frame):
    verdict = substitution_verdict(frame["spread"])
    assert "contrary to the thesis" in verdict.headline
    assert "no mean reversion" in verdict.headline


# ===========================================================================
# L'ARTEFACT QUE S4 DÉNONCE
# ===========================================================================
def test_splitting_on_the_raw_level_manufactures_the_expected_answer(frame):
    """L'erreur évitée, gardée en test pour ne pas y retomber.

    Écrit d'abord pour montrer que le test « contre une constante » créait l'artefact — la
    donnée a répondu que non, et il a fallu chercher l'origine réelle. La voici : l'artefact
    vient de découper sur le **niveau absolu** du spread plutôt que sur un écart à un
    centre. Comme le spread médian vaut environ -83 USD/t, sélectionner |spread| grand ne
    sélectionne pas des écarts anormaux : cela sélectionne l'ère 2004-2005, où la palme est
    profondément décotée. On mesure alors une dynamique d'époque et on l'appelle
    substitution.

    Découper sur un écart — même à une constante — donne déjà la bonne réponse ; la médiane
    glissante ne fait que la rendre plus nette.
    """
    level = frame["spread"].abs()
    threshold = level.quantile(0.70)
    wide = estimate_half_life(frame["spread"], mask=level >= threshold, label="large")
    narrow = estimate_half_life(frame["spread"], mask=level < threshold, label="étroit")

    # le decoupage naif sur le NIVEAU "trouve" la bande...
    assert wide.half_life_days < narrow.half_life_days

    # ...alors que le decoupage sur un ECART, meme a une constante, ne la trouve pas
    deviation = frame["spread"] - frame["spread"].median()
    threshold_deviation = deviation.abs().quantile(0.70)
    wide_deviation = estimate_half_life(
        frame["spread"], mask=deviation.abs() >= threshold_deviation, label="large"
    )
    narrow_deviation = estimate_half_life(
        frame["spread"], mask=deviation.abs() < threshold_deviation, label="étroit"
    )
    assert wide_deviation.half_life_days > narrow_deviation.half_life_days

    # ...et le test de la page non plus
    assert not substitution_verdict(frame["spread"]).substitution_band_exists


def test_the_spread_drifts_rather_than_oscillates(frame):
    """La raison de l'artefact : sur sept ans le spread se déplace de près de 200 USD/t, ce
    qui interdit de le traiter comme stationnaire autour d'une constante."""
    drift = structural_drift(frame["spread"])
    assert abs(drift.attrs["drift_usd_t"]) > 150
    assert drift.attrs["range_usd_t"] > 200
    assert drift["median_spread"].iloc[0] > 0 > drift["median_spread"].iloc[-1]


def test_the_two_tails_are_separated_in_time(frame):
    """Le détail qui tranche en S4 : palme chère au début, très décotée à la fin. Ce ne sont
    pas deux excursions autour d'un équilibre, ce sont deux époques."""
    deviation = frame["spread"] - frame["spread"].median()
    threshold = deviation.abs().quantile(0.70)
    expensive_years = deviation[deviation > threshold].index.year
    cheap_years = deviation[deviation < -threshold].index.year
    assert expensive_years.min() < cheap_years.min()
    assert pd.Series(expensive_years).median() < pd.Series(cheap_years).median()


# ===========================================================================
# Garde-fous
# ===========================================================================
def test_rolling_deviation_refuses_a_degenerate_window(frame):
    with pytest.raises(SubstitutionError, match="too short"):
        rolling_deviation(frame["spread"], window=5)


def test_substitution_verdict_refuses_an_implausible_quantile(frame):
    with pytest.raises(SubstitutionError, match="quantile"):
        substitution_verdict(frame["spread"], quantile=0.30)


def test_the_loader_marks_palm_as_quoted_in_ringgit():
    """Garde-fou d'unité : si quelqu'un ajoute un jour un `scale` à la palme ou la
    redéclare en USD, ce test doit échouer avant que la page ne produise des spreads faux."""
    from agri.data.bloomberg_loader import SERIES_SPECS

    spec = SERIES_SPECS["palm_oil_myr"]
    assert spec.unit == "MYR/t"
    assert getattr(spec, "scale", 1.0) in (1.0, None)
    assert "USDMYR" in (spec.note or "")


def test_no_usdmyr_series_exists_in_the_export():
    """La contrainte qui justifie toute la construction de la page. Si l'USDMYR est ajouté
    un jour, ce test échoue et force à rouvrir le test sur trente ans au lieu de sept."""
    from agri.data.bloomberg_loader import SERIES_SPECS

    assert not any("myr" in key.lower() and "palm" not in key.lower() for key in SERIES_SPECS)
