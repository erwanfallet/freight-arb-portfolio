"""Golden tests du projet A — valeurs calculées à la main, indépendamment du code.

Chaque valeur attendue ci-dessous est dérivée à la main dans le commentaire qui la
précède. Si le code change et qu'un test casse, c'est le code qui a tort jusqu'à preuve
du contraire.
"""
import numpy as np
import pandas as pd
import pytest

from freight.chains.ironore import (
    carry_cost_of_extra_voyage_days,
    decompose_premium,
    explained_variance,
    freight_hedge_effect,
    freight_per_dry_tonne,
    negative_residual_episodes,
)


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="B")


# --------------------------------------------------------------------------- humidité


def test_moisture_conversion_golden():
    """20,00 $/wmt à 9 % d'humidité = 20 / 0,91 = 21,978021978... $/dmt."""
    assert freight_per_dry_tonne(20.0, 0.09) == pytest.approx(21.978021978021978, rel=1e-12)
    # 10,00 $/wmt à 8 % = 10 / 0,92 = 10,869565217...
    assert freight_per_dry_tonne(10.0, 0.08) == pytest.approx(10.869565217391305, rel=1e-12)


def test_moisture_zero_is_identity():
    assert freight_per_dry_tonne(17.5, 0.0) == pytest.approx(17.5)


def test_moisture_out_of_range_is_rejected():
    with pytest.raises(ValueError):
        freight_per_dry_tonne(20.0, 1.0)
    with pytest.raises(ValueError):
        freight_per_dry_tonne(20.0, -0.01)


# ---------------------------------------------------------------------- décomposition


def test_decomposition_golden_single_date():
    """Cas chiffré à la main, le seul qui compte vraiment.

    P65 = 120, P62 = 100          -> premium observé = 20,00 $/dmt
    C3  = 20 $/wmt, humidité 9 %  -> 21,978021978 $/dmt
    C5  = 10 $/wmt, humidité 8 %  -> 10,869565217 $/dmt
    fair value fret               = 11,108456761 $/dmt
    résidu                        = 20 - 11,108456761 = 8,891543239 $/dmt
    part fret                     = 11,108456761 / 20 = 55,542 %

    Et le point pédagogique : le différentiel brut non corrigé vaut 10,00, soit une part
    fret de 50,0 %. La correction d'humidité déplace la part fret de 5,5 points.
    """
    idx = _dates(1)
    d = decompose_premium(
        p65=pd.Series([120.0], index=idx),
        p62=pd.Series([100.0], index=idx),
        c3=pd.Series([20.0], index=idx),
        c5=pd.Series([10.0], index=idx),
        moisture_brazil=0.09,
        moisture_australia=0.08,
    )
    row = d.iloc[0]
    assert row["premium_observed"] == pytest.approx(20.0)
    assert row["c3_dmt"] == pytest.approx(21.978021978021978, rel=1e-12)
    assert row["c5_dmt"] == pytest.approx(10.869565217391305, rel=1e-12)
    assert row["freight_fair_value"] == pytest.approx(11.108456760630673, rel=1e-12)
    assert row["residual"] == pytest.approx(8.891543239369327, rel=1e-12)
    assert row["freight_share"] == pytest.approx(0.5554228380315337, rel=1e-12)
    # le naïf, pour mesurer ce que la correction change
    assert row["premium_naive_freight"] == pytest.approx(10.0)
    assert row["premium_naive_freight"] / row["premium_observed"] == pytest.approx(0.50)


def test_moisture_correction_always_increases_freight_share():
    """La correction d'humidité ne peut pas réduire la part fret quand l'humidité
    brésilienne est supérieure à l'australienne : c'est une propriété de l'identité, pas
    un accident d'échantillon.
    """
    idx = _dates(50)
    rng = np.random.default_rng(0)
    d = decompose_premium(
        p65=pd.Series(120 + rng.normal(0, 3, 50), index=idx),
        p62=pd.Series(100 + rng.normal(0, 3, 50), index=idx),
        c3=pd.Series(20 + rng.normal(0, 1, 50), index=idx),
        c5=pd.Series(10 + rng.normal(0, 1, 50), index=idx),
        moisture_brazil=0.09,
        moisture_australia=0.08,
    )
    assert (d["freight_fair_value"] >= d["premium_naive_freight"]).all()


def test_negative_premium_yields_nan_share_not_nonsense():
    """Un premium négatif ne doit pas produire une part fret négative interprétable."""
    idx = _dates(1)
    d = decompose_premium(
        p65=pd.Series([98.0], index=idx),
        p62=pd.Series([100.0], index=idx),
        c3=pd.Series([20.0], index=idx),
        c5=pd.Series([10.0], index=idx),
    )
    assert np.isnan(d.iloc[0]["freight_share"])
    # le résidu, lui, reste défini et très négatif : -2 - 11,1 = -13,1
    assert d.iloc[0]["residual"] < -13.0


def test_no_common_dates_raises_rather_than_filling():
    """Règle du contrat de données : un trou est une information, jamais un forward-fill."""
    with pytest.raises(ValueError, match="aucune date commune"):
        decompose_premium(
            p65=pd.Series([120.0], index=pd.DatetimeIndex(["2024-01-01"])),
            p62=pd.Series([100.0], index=pd.DatetimeIndex(["2024-01-02"])),
            c3=pd.Series([20.0], index=pd.DatetimeIndex(["2024-01-03"])),
            c5=pd.Series([10.0], index=pd.DatetimeIndex(["2024-01-04"])),
        )


# ------------------------------------------------------------------------ régression


def test_explained_variance_on_exact_linear_relation():
    """premium = 2 + 1,5 × fret exactement -> pente 1,5, constante 2, R² = 1."""
    idx = _dates(40)
    fret = pd.Series(np.linspace(8.0, 16.0, 40), index=idx)
    premium = 2.0 + 1.5 * fret
    ev = explained_variance(premium, fret)
    assert ev.slope == pytest.approx(1.5, rel=1e-9)
    assert ev.intercept == pytest.approx(2.0, rel=1e-9)
    assert ev.r_squared == pytest.approx(1.0, rel=1e-9)
    assert ev.n_obs == 40


def test_explained_variance_on_changes_differs_from_levels():
    """Deux séries tendancielles donnent un R² flatteur en niveau et honnête en
    variation. L'écart entre les deux est le résultat à afficher, pas à cacher.
    """
    idx = _dates(120)
    rng = np.random.default_rng(7)
    trend = np.linspace(0, 10, 120)
    fret = pd.Series(trend + rng.normal(0, 0.5, 120), index=idx)
    premium = pd.Series(trend + rng.normal(0, 3.0, 120), index=idx)
    levels = explained_variance(premium, fret, on_changes=False)
    changes = explained_variance(premium, fret, on_changes=True)
    assert levels.r_squared > changes.r_squared


# --------------------------------------------------------------------------- épisodes


def test_negative_residual_episodes_respects_min_days():
    idx = _dates(20)
    residual = np.ones(20) * 5.0
    residual[3:9] = -1.0   # 6 jours consécutifs -> retenu
    residual[15:17] = -2.0  # 2 jours -> filtré
    d = pd.DataFrame({"residual": residual}, index=idx)
    episodes = negative_residual_episodes(d, min_days=5)
    assert len(episodes) == 1
    assert episodes.iloc[0]["n_obs"] == 6
    assert episodes.iloc[0]["residual_min"] == pytest.approx(-1.0)


def test_no_negative_residual_returns_empty_frame():
    idx = _dates(10)
    d = pd.DataFrame({"residual": np.ones(10)}, index=idx)
    assert negative_residual_episodes(d).empty


# ------------------------------------------------------------------------------ hedge


def test_perfect_hedge_removes_all_volatility():
    """Si le premium bouge exactement 2× la part fret, beta = 2 et la vol résiduelle
    est nulle : réduction de 100 %.
    """
    idx = _dates(60)
    rng = np.random.default_rng(3)
    fret = pd.Series(np.cumsum(rng.normal(0, 1, 60)) + 12.0, index=idx)
    premium = 2.0 * fret + 5.0
    h = freight_hedge_effect(premium, fret)
    assert h.beta == pytest.approx(2.0, rel=1e-9)
    assert h.vol_hedged == pytest.approx(0.0, abs=1e-9)
    assert h.vol_reduction_pct == pytest.approx(100.0, abs=1e-6)


def test_unit_hedge_can_be_worse_than_optimal_hedge():
    """Le hedge unitaire naïf n'est pas le hedge de variance minimale. L'écart est en
    soi un résultat de desk.
    """
    idx = _dates(200)
    rng = np.random.default_rng(11)
    fret = pd.Series(np.cumsum(rng.normal(0, 1, 200)) + 12.0, index=idx)
    premium = 0.4 * fret + pd.Series(np.cumsum(rng.normal(0, 1, 200)), index=idx)
    optimal = freight_hedge_effect(premium, fret)
    naive = freight_hedge_effect(premium, fret, beta=1.0)
    assert optimal.vol_hedged <= naive.vol_hedged


def test_zero_variance_freight_is_rejected():
    idx = _dates(30)
    fret = pd.Series(np.ones(30) * 11.0, index=idx)
    premium = pd.Series(np.linspace(1, 5, 30), index=idx)
    with pytest.raises(ValueError, match="ne varie pas"):
        freight_hedge_effect(premium, fret)


# ------------------------------------------------------------------------ coût portage


def test_carry_cost_golden():
    """100 $/dmt, 25 jours de plus, 6 % annuel -> 100 × 0,06 × 25/365 = 0,410958904 $/dmt."""
    assert carry_cost_of_extra_voyage_days(100.0, 25.0, 0.06) == pytest.approx(
        0.410958904109589, rel=1e-12
    )


def test_carry_cost_rejects_negative_days():
    with pytest.raises(ValueError):
        carry_cost_of_extra_voyage_days(100.0, -1.0, 0.06)
