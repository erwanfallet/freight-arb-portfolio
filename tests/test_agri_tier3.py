"""Golden tests T3-2 (sucre), T3-3 (EUDR), T3-4 (Chine soja).

Les valeurs attendues viennent des memes jeux synthetiques que ceux verifies en session
(smoke tests manuels) — reportees ici en assertions formelles pour que la suite les
verrouille.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from agri.chains.china_soy import (
    ChinaSoyError,
    bean_cnf_usd_t,
    crush_margin_cny_t,
    load_real_crush_frame,
    purchases_by_margin_quintile,
    signature_test,
)
from agri.data.bloomberg_loader import DEFAULT_PATH
from agri.chains.sugar_mix import (
    SugarMixError,
    consecana_sensitivity,
    estimate_mix_elasticity,
    hydrous_sugar_equivalent_cents_lb,
    parity_gap_cents_lb,
)
from agri.fixtures import tier3


# ===========================================================================
# T3-2 — sucre
# ===========================================================================
def test_hydrous_equivalent_hand_computed():
    """2,95 BRL/L, USDBRL 5,10, coefficients Consecana par defaut :
        BRL/kg  = 2,95 x (1,0495/1,6913) = 1,830559...
        c/lb    = 1,830559 x 100 / (5,10 x 2,20462) = 16,280956...
    """
    index = pd.to_datetime(["2024-01-01"])
    out = hydrous_sugar_equivalent_cents_lb(
        pd.Series([2.95], index=index), pd.Series([5.10], index=index)
    )
    assert out.iloc[0] == pytest.approx(16.280936, abs=1e-4)


def test_parity_gap_sign():
    index = pd.to_datetime(["2024-01-01"])
    out = parity_gap_cents_lb(
        pd.Series([20.0], index=index), pd.Series([2.95], index=index), pd.Series([5.10], index=index)
    )
    # ny11_adjusted = 20 x 0,98 = 19,6 ; equivalent = 16,280956 ; gap = 3,319044
    assert out["parity_gap"].iloc[0] == pytest.approx(3.319064, abs=1e-4)
    assert bool(out["sugar_favoured"].iloc[0]) is True


def test_consecana_sensitivity_moves_the_parity():
    data = tier3.sugar_prices()
    table = consecana_sensitivity(data["ny11"], data["hydrous"], data["usdbrl"])
    assert table["atr_sugar"].is_monotonic_increasing
    # a higher sugar ATR raises the ethanol equivalent -> compresses the gap
    assert table["mean_parity_gap"].iloc[0] > table["mean_parity_gap"].iloc[-1]


def test_mix_elasticity_recovers_true_betas_after_decorrelation():
    """Apres decorrelation de hedge_ratio et dist_port (correlation -0,98 -> -0,21), b1
    et b2 sont tous deux correctement identifies — c'etait le bug qui rendait b2
    indetectable (sortait a ~1e-6 au lieu de -0,0028)."""
    out = estimate_mix_elasticity(tier3.sugar_panel())
    assert out.beta_parity == pytest.approx(tier3.TRUE_BETA_PARITY, abs=5e-4)
    assert out.beta_interaction == pytest.approx(tier3.TRUE_BETA_INTERACTION, abs=5e-4)
    assert out.hedging_matters


def test_mix_elasticity_headline_is_readable_at_small_magnitude():
    """Regression du bug d'affichage : inverser une elasticite minuscule produit un
    nombre absurde (des milliers de cents/lb). Le headline doit rester lisible."""
    out = estimate_mix_elasticity(tier3.sugar_panel())
    headline = out.headline()
    assert "100 points of parity gap" in headline
    assert "point" in headline


def test_missing_panel_columns_raise():
    with pytest.raises(SugarMixError, match="missing columns"):
        estimate_mix_elasticity(pd.DataFrame({"region": ["A"], "d_mix": [0.1]}))


# ===========================================================================
# T3-3 — EUDR
# ===========================================================================
def test_bean_cnf_hand_computed():
    """CBOT 12,00 $/bu, basis 70 c/bu, fret 50 $/t :
        fob = 12,00 + 0,70 = 12,70 $/bu
        cnf = 12,70 x 36,7437 + 50 = 466,84 + 50 = 516,84 $/t
    """
    index = pd.to_datetime(["2024-01-01"])
    out = bean_cnf_usd_t(
        pd.Series([12.0], index=index), pd.Series([70.0], index=index), pd.Series([50.0], index=index)
    )
    assert out.iloc[0] == pytest.approx(516.845, abs=0.5)


def test_crush_margin_yields_over_one_is_rejected():
    index = pd.to_datetime(["2024-01-01"])
    series = pd.Series([1.0], index=index)
    with pytest.raises(ChinaSoyError, match="yields"):
        crush_margin_cny_t(series, series, series, series, meal_yield=0.9, oil_yield=0.3)


def test_signature_test_detects_the_imposed_political_signature():
    """Le jeu impose une probabilite d'achat qui DECROIT avec la marge retardee ->
    signature politique attendue, avec un beta negatif et significatif."""
    data = tier3.china_soy()
    out = signature_test(data["purchases"], data["margin"], data["stock_days"], data["cbot"])
    assert out.beta_margin < 0
    assert out.is_significant
    assert out.signature == "political"


def test_quintiles_are_monotonically_decreasing():
    data = tier3.china_soy()
    table = purchases_by_margin_quintile(data["purchases"], data["margin"])
    assert table["purchase_rate"].is_monotonic_decreasing


def test_signature_test_needs_enough_variation():
    index = pd.date_range("2020-01-01", periods=50, freq="ME")
    all_zero = pd.Series(0, index=index)
    margin = pd.Series(np.random.default_rng(0).normal(size=50), index=index)
    with pytest.raises(ChinaSoyError, match="too little variation"):
        signature_test(all_zero, margin, margin, margin)


# ===========================================================================
# T3-4 on real data — Chinese crush margin, 3 of 4 legs real
# ===========================================================================
@pytest.mark.skipif(not DEFAULT_PATH.exists(), reason=f"Bloomberg file absent: {DEFAULT_PATH}")
def test_real_china_crush_margin_on_2026_08_07_hand_computed():
    """CBOT 11.565 USD/bu, DCE meal 3109 CNY/t, DCE oil 8372 CNY/t, USDCNY 6.7454,
    parameterised basis 70 c/bu and freight 45 USD/t:
        fob_usd_bu    = 11.565 + 0.70                    = 12.265
        bean_cnf      = 12.265 x 36.7437103 + 45          = 495.661607
        revenue_gross = 0.785 x 3109 + 0.185 x 8372       = 3989.385
        revenue_ex_vat= 3989.385 / 1.09                    = 3659.986239
        bean_cost     = 495.661607 x 6.7454 x 1.03         = 3443.738880
        margin        = 3659.986239 - 3443.738880 - 120    = 96.247359
    """
    frame = load_real_crush_frame()
    assert frame.loc["2026-08-07", "margin"] == pytest.approx(96.247359, abs=1e-3)
    assert frame.loc["2026-08-07", "revenue_gross"] == pytest.approx(3989.385, abs=1e-3)
    assert frame.loc["2026-08-07", "bean_cost"] == pytest.approx(3443.738880, abs=1e-3)


@pytest.mark.skipif(not DEFAULT_PATH.exists(), reason=f"Bloomberg file absent: {DEFAULT_PATH}")
def test_real_crush_frame_documents_which_legs_are_real():
    """Three real legs (CBOT, DCE x2, USDCNY), one parameterised (basis+freight) — the
    split must be declared in the attrs, not only in the docstring."""
    frame = load_real_crush_frame()
    assert set(frame.attrs["real_legs"]) == {"cbot_soybean", "dce_soymeal", "dce_soyoil", "usdcny"}
    assert "basis_cents_bu" in frame.attrs["parametrized_legs"]
    assert "freight_usd_t" in frame.attrs["parametrized_legs"]


@pytest.mark.skipif(not DEFAULT_PATH.exists(), reason=f"fichier Bloomberg absent : {DEFAULT_PATH}")
def test_real_crush_margin_has_genuine_variation():
    frame = load_real_crush_frame(start="2020-01-01")
    assert frame["margin"].std() > 50.0
    assert frame["margin"].nunique() > 500
