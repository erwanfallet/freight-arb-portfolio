"""Golden tests du module T1-M (régime ou skill).

Le test central est `test_copper_backtest_self_criticism_sentence` : il produit
littéralement la phrase que la spec demande d'écrire dans le mail.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agri.core.regime import (
    RegimeError,
    attribute_pnl_to_regime,
    honest_win_rate,
)


# ===========================================================================
# Win rate honnête — le chiffre du mail
# ===========================================================================
def test_copper_backtest_self_criticism_sentence():
    """La forme du backtest cuivre : 18 positions, hold 30 jours, entrées tous les 11 jours.

    Chevauchement moyen 2.49 -> n_eff = 7.23 -> arrondi conservateur à 7 tirages.
    18 succès sur 18 devient 7 succès sur 7, dont la borne basse exacte est 59.0 %.
    """
    entries = pd.date_range("2024-01-01", periods=18, freq="11D")
    out = honest_win_rate(entries, [True] * 18, hold_days=30)

    assert out.n_trades == 18
    assert out.sample.n_eff == pytest.approx(7.233, abs=1e-2)
    assert out.naive.point == pytest.approx(1.0)
    assert out.naive.lo == pytest.approx(0.81470, abs=1e-4)
    assert out.honest.lo == pytest.approx(0.59043, abs=1e-4)
    # ignorer le chevauchement surestimerait la borne basse de 22 points
    assert out.lower_bound_cost == pytest.approx(0.2243, abs=1e-3)

    sentence = out.mail_sentence
    assert "18 positions" in sentence
    assert "7.2 tirages indépendants" in sentence
    assert "6 fois sur 10" in sentence
    assert "22 points" in sentence


def test_non_overlapping_trades_cost_nothing():
    # entrées espacées d'exactement la période de hold : aucun chevauchement à corriger
    entries = pd.date_range("2024-01-01", periods=10, freq="30D")
    out = honest_win_rate(entries, [True] * 10, hold_days=30)
    assert out.sample.overlap == pytest.approx(1.0)
    assert out.sample.n_eff == pytest.approx(10.0)
    assert out.lower_bound_cost == pytest.approx(0.0, abs=1e-9)


def test_win_share_is_preserved_not_the_raw_count():
    """Le chevauchement réduit le nombre d'observations, pas le taux de réussite."""
    entries = pd.date_range("2024-01-01", periods=18, freq="11D")
    wins = [True] * 12 + [False] * 6  # 2/3 de réussite
    out = honest_win_rate(entries, wins, hold_days=30)
    assert out.naive.point == pytest.approx(12 / 18)
    # 2/3 reporté sur 7 tirages -> round(0.667 * 7) = 5 succès sur 7
    assert out.honest.successes == 5
    assert out.honest.point == pytest.approx(5 / 7)


def test_shorter_hold_preserves_more_information():
    entries = pd.date_range("2024-01-01", periods=18, freq="11D")
    long_hold = honest_win_rate(entries, [True] * 18, hold_days=30)
    short_hold = honest_win_rate(entries, [True] * 18, hold_days=5)
    assert short_hold.sample.n_eff > long_hold.sample.n_eff
    assert short_hold.honest.lo > long_hold.honest.lo


def test_mismatched_lengths_raise():
    entries = pd.date_range("2024-01-01", periods=5, freq="D")
    with pytest.raises(RegimeError, match="trade à trade"):
        honest_win_rate(entries, [True] * 3, hold_days=10)


# ===========================================================================
# Attribution du P&L au régime
# ===========================================================================
def _synthetic_trades(n: int, *, alpha: float, seed: int, freq: str = "11D") -> pd.DataFrame:
    """P&L fabriqué pour être EXACTEMENT du régime plus une constante connue.

    pnl = alpha + 2*vol + 1.5*width - 0.5*dispersion + 0.1*duration + bruit
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq=freq)
    vol = rng.uniform(0.1, 0.5, n)
    width = rng.uniform(-2.0, 2.0, n)
    dispersion = rng.uniform(0.0, 3.0, n)
    duration = rng.uniform(20.0, 40.0, n)
    pnl = alpha + 2.0 * vol + 1.5 * width - 0.5 * dispersion + 0.1 * duration
    pnl = pnl + rng.normal(scale=0.2, size=n)
    return pd.DataFrame(
        {
            "pnl": pnl,
            "vol_realised": vol,
            "term_structure_width": width,
            "dispersion": dispersion,
            "duration": duration,
        },
        index=idx,
    )


def test_regime_regression_recovers_known_coefficients():
    trades = _synthetic_trades(400, alpha=0.0, seed=20, freq="3D")
    out = attribute_pnl_to_regime(trades, hold_days=3)
    assert out.regression.params["vol_realised"] == pytest.approx(2.0, abs=0.2)
    assert out.regression.params["term_structure_width"] == pytest.approx(1.5, abs=0.1)
    assert out.regression.params["dispersion"] == pytest.approx(-0.5, abs=0.1)
    assert out.r_squared > 0.9


def test_pure_regime_pnl_yields_an_alpha_indistinguishable_from_zero():
    """Un P&L intégralement fabriqué par le régime ne doit pas produire d'alpha."""
    trades = _synthetic_trades(400, alpha=0.0, seed=21, freq="3D")
    out = attribute_pnl_to_regime(trades, hold_days=3, n_iter=2000)
    assert not out.alpha_is_distinguishable_from_zero
    assert out.verdict == "indistinguable du régime"
    assert "je ne peux pas distinguer ce résultat du régime" in out.mail_sentence


def test_genuine_alpha_is_detected():
    trades = _synthetic_trades(400, alpha=5.0, seed=22, freq="3D")
    out = attribute_pnl_to_regime(trades, hold_days=3, n_iter=2000)
    assert out.alpha == pytest.approx(5.0, abs=0.3)
    assert out.alpha_is_distinguishable_from_zero
    assert out.verdict == "alpha présumé"


def test_eighteen_overlapping_trades_are_flagged_uninterpretable():
    """Le cas du backtest cuivre : 5 paramètres pour ~7 observations indépendantes.

    Le R² sera flatteur et ne veut rien dire. Le module doit le dire lui-même plutôt que
    d'afficher le chiffre — c'est la ligne « le résultat est affiché même s'il est
    défavorable » de la checklist.
    """
    trades = _synthetic_trades(18, alpha=3.0, seed=23, freq="11D")
    out = attribute_pnl_to_regime(trades, hold_days=30, n_iter=1000)
    assert out.sample.n_eff == pytest.approx(7.233, abs=1e-2)
    assert out.is_overfit
    assert out.verdict == "non interprétable"
    assert "souplesse du modèle" in out.mail_sentence


def test_missing_regime_column_raises_with_guidance():
    trades = _synthetic_trades(50, alpha=1.0, seed=24, freq="3D").drop(columns="dispersion")
    with pytest.raises(RegimeError, match="regime_columns"):
        attribute_pnl_to_regime(trades, hold_days=3)


def test_custom_regime_columns_are_supported():
    """Chaque page nomme ses propres variables : dispersion inter-origines pour un grain."""
    trades = _synthetic_trades(200, alpha=1.0, seed=25, freq="3D").rename(
        columns={"dispersion": "dispersion_inter_origines"}
    )
    out = attribute_pnl_to_regime(
        trades,
        hold_days=3,
        regime_columns=("vol_realised", "term_structure_width", "dispersion_inter_origines"),
        n_iter=1000,
    )
    assert "dispersion_inter_origines" in out.regime_columns
    assert out.r_squared > 0.8


def test_missing_pnl_column_raises():
    trades = _synthetic_trades(50, alpha=1.0, seed=26, freq="3D").rename(columns={"pnl": "profit"})
    with pytest.raises(RegimeError, match="colonne P&L absente"):
        attribute_pnl_to_regime(trades, hold_days=3)
