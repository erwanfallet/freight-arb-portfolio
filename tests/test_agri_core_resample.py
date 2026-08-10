"""Golden tests des trois règles de rééchantillonnage.

Valeurs attendues calculées à la main dans le commentaire qui les précède. Ce fichier
teste surtout des **refus** : la valeur de ce module est dans ce qu'il empêche, pas dans
ce qu'il calcule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agri.core.resample import (
    EffectiveSample,
    ResampleError,
    StaleDataInTest,
    align_for_test,
    average_concurrency,
    downsample,
    effective_n,
    effective_n_from_trades,
    effective_n_rolling,
    ffill_with_provenance,
    for_pnl,
    for_test,
    infer_frequency,
    slowest_frequency,
)


def _daily(start: str, periods: int, start_value: float = 100.0) -> pd.Series:
    idx = pd.date_range(start, periods=periods, freq="B")
    return pd.Series(np.arange(periods, dtype=float) + start_value, index=idx)


def _monthly(start: str, periods: int, start_value: float = 10.0) -> pd.Series:
    idx = pd.date_range(start, periods=periods, freq="MS")  # datée au 1er du mois
    return pd.Series(np.arange(periods, dtype=float) + start_value, index=idx)


# ===========================================================================
# RÈGLE A
# ===========================================================================
def test_infer_frequency_daily_survives_weekends():
    # jours ouvrés : écart médian de 1 jour malgré les sauts de week-end à 3 jours
    assert infer_frequency(_daily("2024-01-01", 60)) == "daily"


def test_infer_frequency_weekly():
    s = pd.Series(1.0, index=pd.date_range("2024-01-01", periods=30, freq="W"))
    assert infer_frequency(s) == "weekly"


def test_infer_frequency_fortnightly_unica_style():
    # quinzaine UNICA : écart médian de 15 jours
    s = pd.Series(1.0, index=pd.date_range("2024-01-01", periods=30, freq="15D"))
    assert infer_frequency(s) == "fortnightly"


def test_infer_frequency_monthly_despite_uneven_month_lengths():
    # mois de 28 à 31 jours : la médiane traverse, un mode ou un min se ferait piéger
    assert infer_frequency(_monthly("2024-01-01", 24)) == "monthly"


def test_infer_frequency_quarterly():
    s = pd.Series(1.0, index=pd.date_range("2020-01-01", periods=16, freq="QE"))
    assert infer_frequency(s) == "quarterly"


def test_infer_frequency_needs_three_observations():
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-01-01", "2024-02-01"]))
    with pytest.raises(ResampleError, match="3 observations"):
        infer_frequency(s)


def test_slowest_frequency_picks_the_binding_one():
    assert slowest_frequency(
        {"px": _daily("2024-01-01", 60), "grind": _monthly("2024-01-01", 24)}
    ) == "monthly"


def test_downsample_daily_to_monthly_takes_last_print():
    # série quotidienne 100, 101, 102, ... sur jours ouvrés depuis le 2024-01-01.
    # Janvier 2024 compte 23 jours ouvrés -> valeurs 100..122, dernier print = 122.
    s = _daily("2024-01-01", 60)
    out = downsample(s, "monthly", how="last")
    assert out.index[0] == pd.Timestamp("2024-01-31")
    assert out.iloc[0] == pytest.approx(122.0)


def test_downsample_daily_to_monthly_mean_differs_from_last():
    s = _daily("2024-01-01", 60)
    # moyenne de 100..122 = (100+122)/2 = 111.0
    assert downsample(s, "monthly", how="mean").iloc[0] == pytest.approx(111.0)


def test_downsample_refuses_to_go_up_rule_a():
    """Le cœur de la Règle A : remonter une série lente doit lever, pas s'exécuter."""
    with pytest.raises(ResampleError, match="RÈGLE A violée"):
        downsample(_monthly("2024-01-01", 24), "daily")


def test_downsample_normalises_labels_even_at_same_frequency():
    """Le piège de l'intersection vide.

    Une série mensuelle datée au 1er et une quotidienne descendue en mensuel (datée en
    fin de mois) ont la même fréquence et **zéro date commune**. Sans normalisation,
    align_for_test sortirait vide sans rien signaler.
    """
    monthly = _monthly("2024-01-01", 6)
    assert monthly.index[0] == pd.Timestamp("2024-01-01")
    out = downsample(monthly, "monthly")
    assert out.index[0] == pd.Timestamp("2024-01-31")
    assert out.iloc[0] == pytest.approx(10.0)  # la valeur est conservée, seule l'étiquette bouge


def test_downsample_rejects_unknown_aggregation():
    with pytest.raises(ResampleError, match="agrégation"):
        downsample(_daily("2024-01-01", 60), "monthly", how="median")


def test_align_for_test_runs_at_the_slowest_frequency():
    aligned = align_for_test(
        {"px": _daily("2024-01-01", 120), "grind": _monthly("2024-01-01", 6)}
    )
    assert aligned.attrs["test_frequency"] == "monthly"
    # 120 jours ouvrés depuis le 2024-01-01 couvrent janvier à mi-juin -> 6 mois pleins
    # côté quotidien, et la série mensuelle en fournit 6 : l'intersection en fait 6.
    assert len(aligned) == 6
    assert list(aligned.columns) == ["px", "grind"]


def test_align_for_test_intersects_without_filling_gaps():
    # la série mensuelle démarre trois mois après la quotidienne : les trois premiers
    # mois disparaissent du test au lieu d'être comblés
    aligned = align_for_test(
        {"px": _daily("2024-01-01", 250), "grind": _monthly("2024-04-01", 6)}
    )
    assert len(aligned) == 6
    assert aligned.index[0] == pd.Timestamp("2024-04-30")


def test_align_for_test_raises_when_no_common_date():
    aligned_a = _monthly("2020-01-01", 6)
    aligned_b = _monthly("2024-01-01", 6)
    with pytest.raises(ResampleError, match="aucune date commune"):
        align_for_test({"a": aligned_a, "b": aligned_b})


# ===========================================================================
# RÈGLE B
# ===========================================================================
@pytest.fixture
def filled_monthly_on_daily() -> pd.DataFrame:
    """TC mensuel (10 en janvier, 20 en février) reporté sur un calendrier quotidien."""
    source = pd.Series(
        [10.0, 20.0], index=pd.to_datetime(["2024-01-01", "2024-02-01"])
    )
    target = pd.date_range("2024-01-01", "2024-02-05", freq="D")
    return ffill_with_provenance(source, target)


def test_ffill_marks_true_prints(filled_monthly_on_daily):
    df = filled_monthly_on_daily
    assert bool(df.loc["2024-01-01", "is_true_print"]) is True
    assert bool(df.loc["2024-02-01", "is_true_print"]) is True
    assert bool(df.loc["2024-01-15", "is_true_print"]) is False


def test_ffill_carries_the_value_forward(filled_monthly_on_daily):
    df = filled_monthly_on_daily
    assert df.loc["2024-01-15", "value"] == pytest.approx(10.0)
    assert df.loc["2024-02-05", "value"] == pytest.approx(20.0)


def test_staleness_days_hand_computed(filled_monthly_on_daily):
    df = filled_monthly_on_daily
    # un vrai print a un âge nul
    assert df.loc["2024-01-01", "staleness_days"] == pytest.approx(0.0)
    # 2024-01-15 est à 14 jours du print du 1er janvier
    assert df.loc["2024-01-15", "staleness_days"] == pytest.approx(14.0)
    # 2024-02-05 est à 4 jours du print du 1er février, PAS à 35 du print de janvier
    assert df.loc["2024-02-05", "staleness_days"] == pytest.approx(4.0)


def test_no_value_and_no_age_before_the_first_print():
    source = pd.Series([10.0], index=pd.to_datetime(["2024-01-10"]))
    target = pd.date_range("2024-01-01", "2024-01-15", freq="D")
    df = ffill_with_provenance(source, target)
    assert np.isnan(df.loc["2024-01-05", "value"])
    assert np.isnan(df.loc["2024-01-05", "staleness_days"])
    assert df.loc["2024-01-12", "value"] == pytest.approx(10.0)


def test_max_staleness_kills_dead_data(filled_monthly_on_daily):
    source = pd.Series(
        [10.0, 20.0], index=pd.to_datetime(["2024-01-01", "2024-02-01"])
    )
    target = pd.date_range("2024-01-01", "2024-02-05", freq="D")
    df = ffill_with_provenance(source, target, max_staleness_days=10)
    # à 14 jours d'âge, la valeur est retirée : un TC de deux semaines n'est plus une
    # hypothèse conservatrice
    assert np.isnan(df.loc["2024-01-15", "value"])
    # à 4 jours, elle reste
    assert df.loc["2024-02-05", "value"] == pytest.approx(20.0)


def test_for_pnl_accepts_filled_data(filled_monthly_on_daily):
    """Règle B, côté autorisé : un smelter paie bien un TC mensuel tous les jours."""
    values = for_pnl(filled_monthly_on_daily)
    assert len(values) == 36  # 2024-01-01 au 2024-02-05 inclus
    assert values.loc["2024-01-15"] == pytest.approx(10.0)


def test_for_test_refuses_filled_data(filled_monthly_on_daily):
    """Règle B, côté bloqué. C'est le test le plus important du fichier.

    La règle cesse d'être une consigne de méthode : le chemin de code qui alimenterait
    une régression avec du forward-fill lève une exception.
    """
    with pytest.raises(StaleDataInTest, match="RÈGLE B violée"):
        for_test(filled_monthly_on_daily)


def test_for_test_error_names_the_fallback(filled_monthly_on_daily):
    # l'erreur doit dire quoi faire à la place, sinon elle sera contournée
    with pytest.raises(StaleDataInTest, match="align_for_test"):
        for_test(filled_monthly_on_daily)


def test_for_test_accepts_data_that_was_never_filled():
    source = pd.Series([10.0, 11.0, 12.0], index=pd.date_range("2024-01-01", periods=3, freq="D"))
    df = ffill_with_provenance(source, pd.date_range("2024-01-01", periods=3, freq="D"))
    assert for_test(df).tolist() == [10.0, 11.0, 12.0]


def test_provenance_frame_shape_is_enforced():
    with pytest.raises(ResampleError, match="ffill_with_provenance"):
        for_test(pd.DataFrame({"value": [1.0]}))


# ===========================================================================
# RÈGLE C
# ===========================================================================
def test_effective_n_basic():
    # 18 observations, chevauchement 3 -> n_eff = 6
    out = effective_n(18, 3)
    assert out.n_eff == pytest.approx(6.0)
    assert out.shrinkage == pytest.approx(3.0)


def test_effective_n_rolling_window_is_the_overlap():
    # 250 jours en fenêtre glissante de 30 -> n_eff = 8.33
    assert effective_n_rolling(250, 30).n_eff == pytest.approx(8.3333, abs=1e-3)


def test_effective_n_refuses_overlap_below_one():
    with pytest.raises(ResampleError, match="chevauchement"):
        effective_n(18, 0.5)


def test_average_concurrency_hand_computed():
    """Trois entrées espacées de 10 jours, tenues 30 jours.

    Couverture jour par jour depuis la première entrée :
        jours  0-9  : 1 position ouverte
        jours 10-19 : 2
        jours 20-29 : 3
        jours 30-39 : 2
        jours 40-49 : 1
    Somme = 3 trades x 30 jours = 90 positions-jours sur 50 jours -> moyenne 1.8.
    """
    entries = pd.to_datetime(["2024-01-01", "2024-01-11", "2024-01-21"])
    assert average_concurrency(entries, hold_days=30) == pytest.approx(1.8)


def test_effective_n_from_trades_hand_computed():
    # 3 trades / chevauchement moyen 1.8 = 1.667
    entries = pd.to_datetime(["2024-01-01", "2024-01-11", "2024-01-21"])
    assert effective_n_from_trades(entries, hold_days=30).n_eff == pytest.approx(1.6667, abs=1e-3)


def test_non_overlapping_trades_keep_their_full_sample():
    # entrées espacées de 30 jours, tenues 30 jours : aucun chevauchement, n_eff = n
    entries = pd.to_datetime(["2024-01-01", "2024-01-31", "2024-03-01"])
    out = effective_n_from_trades(entries, hold_days=30)
    assert out.overlap == pytest.approx(1.0)
    assert out.n_eff == pytest.approx(3.0)


def test_copper_backtest_shape_lands_in_the_predicted_range():
    """Le chiffre du module T1-M, sur la forme du backtest cuivre.

    18 positions tenues 30 jours, entrées tous les 11 jours :
        étendue  = 17 x 11 + 30 = 217 jours couverts
        somme    = 18 x 30 = 540 positions-jours
        moyenne  = 540 / 217 = 2.4885 positions ouvertes en même temps
        n_eff    = 18 / 2.4885 = 7.23

    La spec annonçait « de l'ordre de 6-8, pas 18 ». C'est ce qui change la lecture d'un
    win rate de 100 % : sur ~7 tirages indépendants, il n'est pas distinguable d'un
    processus à 70 % de réussite.
    """
    entries = pd.date_range("2024-01-01", periods=18, freq="11D")
    out = effective_n_from_trades(entries, hold_days=30)
    assert out.overlap == pytest.approx(2.4885, abs=1e-3)
    assert out.n_eff == pytest.approx(7.233, abs=1e-2)
    assert 6.0 <= out.n_eff <= 8.0


def test_effective_sample_summary_is_readable():
    summary = EffectiveSample(n_obs=18, overlap=2.5, n_eff=7.2).summary
    assert "n_eff = 7.2" in summary
    assert "2.5x moins" in summary
