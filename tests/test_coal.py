"""Golden tests du projet B — valeurs calculées à la main dans les commentaires."""
import numpy as np
import pandas as pd
import pytest

from freight.chains.coal import (
    BENCHMARK_CV_KCAL_PER_KG,
    DEFAULT_EMISSION_FACTOR,
    EXTRA_EU_SCOPE_FACTOR,
    ets_cost_per_cargo_tonne,
    financing_cost,
    freight_binding_test,
    ols,
    phase_in_factor,
    phase_in_series,
    reconstruct_ara_arb,
    regime_stats,
    to_energy_basis,
    voyage_emissions_t_co2,
)


def _dates(n: int, start: str = "2021-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


# ------------------------------------------------------------- pouvoir calorifique
def test_energy_basis_golden():
    """5 700 kcal/kg contre une référence 6 000 : facteur 6000/5700 = 1,052631578...

    Un prix de 90 $/t devient 94,736842105 $ par tonne-équivalent-6 000.
    Un fret de 12 $/t devient 12,631578947 par tonne-équivalent-6 000, soit +5,3 %.
    """
    assert to_energy_basis(90.0, 5700.0) == pytest.approx(94.73684210526316, rel=1e-12)
    assert to_energy_basis(12.0, 5700.0) == pytest.approx(12.631578947368421, rel=1e-12)


def test_energy_basis_at_benchmark_is_identity():
    assert to_energy_basis(88.5, BENCHMARK_CV_KCAL_PER_KG) == pytest.approx(88.5)


def test_energy_basis_penalises_low_cv():
    """Plus le CV réel est bas, plus le coût par unité d'énergie est élevé."""
    assert to_energy_basis(12.0, 5500.0) > to_energy_basis(12.0, 5800.0) > 12.0


def test_energy_basis_rejects_bad_cv():
    with pytest.raises(ValueError):
        to_energy_basis(90.0, 0.0)
    with pytest.raises(ValueError):
        to_energy_basis(90.0, -5700.0)


# ---------------------------------------------------------------------------- ETS
def test_phase_in_factor_golden():
    """Montée en charge de l'ETS maritime : rien avant 2024, puis 40 %, 70 %, 100 %."""
    assert phase_in_factor(2023) == 0.0
    assert phase_in_factor(2024) == pytest.approx(0.40)
    assert phase_in_factor(2025) == pytest.approx(0.70)
    assert phase_in_factor(2026) == pytest.approx(1.00)
    assert phase_in_factor(2030) == pytest.approx(1.00)


def test_effective_extra_eu_coverage_is_20_35_50():
    """Pour un voyage dont une extrémité est hors UE — Richards Bay -> Rotterdam — la
    couverture effective est portée × montée en charge, soit 20 % en 2024, 35 % en 2025
    et 50 % à partir de 2026.
    """
    assert EXTRA_EU_SCOPE_FACTOR * phase_in_factor(2024) == pytest.approx(0.20)
    assert EXTRA_EU_SCOPE_FACTOR * phase_in_factor(2025) == pytest.approx(0.35)
    assert EXTRA_EU_SCOPE_FACTOR * phase_in_factor(2026) == pytest.approx(0.50)


def test_phase_in_series_follows_calendar_year():
    idx = pd.DatetimeIndex(["2023-12-31", "2024-01-01", "2025-06-30", "2026-01-01"])
    s = phase_in_series(idx)
    assert list(s.values) == [0.0, 0.40, 0.70, 1.00]


def test_voyage_emissions_golden():
    """480 t de combustible × 3,114 tCO2/t = 1 494,72 tCO2."""
    assert voyage_emissions_t_co2(480.0) == pytest.approx(1494.72, rel=1e-12)
    assert voyage_emissions_t_co2(480.0, DEFAULT_EMISSION_FACTOR) == pytest.approx(1494.72)


def test_ets_cost_golden():
    """Calcul complet, à la main :

    émissions            1 494,72 tCO2
    × portée 0,50        =   747,36
    × montée 0,40 (2024) =   298,944 quotas dus
    × 70 EUR             = 20 926,08 EUR
    × 1,10 EURUSD        = 23 018,688 USD
    / 150 000 t          =     0,15345792 USD/t
    """
    cost = ets_cost_per_cargo_tonne(
        eua_price_eur=70.0,
        eurusd=1.10,
        emissions_t_co2=1494.72,
        cargo_t=150_000.0,
        phase_in=0.40,
    )
    assert cost == pytest.approx(0.15345792, rel=1e-12)


def test_ets_cost_scales_with_phase_in():
    kwargs = dict(
        eua_price_eur=70.0, eurusd=1.10, emissions_t_co2=1494.72, cargo_t=150_000.0
    )
    c2024 = ets_cost_per_cargo_tonne(phase_in=0.40, **kwargs)
    c2026 = ets_cost_per_cargo_tonne(phase_in=1.00, **kwargs)
    # 100 % / 40 % = 2,5x
    assert c2026 / c2024 == pytest.approx(2.5, rel=1e-12)


def test_ets_cost_rejects_bad_inputs():
    kwargs = dict(eua_price_eur=70.0, eurusd=1.1, emissions_t_co2=1000.0, phase_in=0.4)
    with pytest.raises(ValueError):
        ets_cost_per_cargo_tonne(cargo_t=0.0, **kwargs)
    with pytest.raises(ValueError):
        ets_cost_per_cargo_tonne(cargo_t=150_000.0, scope_factor=1.5, **kwargs)


# ---------------------------------------------------------------------------- arb
def test_financing_cost_golden():
    """90 $/t × 6 % × 20/365 = 0,295890410958904 $/t."""
    assert financing_cost(90.0, 20.0, 0.06) == pytest.approx(0.2958904109589041, rel=1e-12)


def test_arb_reconstruction_golden():
    """API2 110, API4 90, fret 12, 20 jours à 6 %, ETS 0,15345792 :

    spread     = 20,00
    financement=  0,295890411
    arb        = 20 − 12 − 0,295890411 − 0,15345792 = 7,550651669
    """
    idx = _dates(1)
    frame = reconstruct_ara_arb(
        api2=pd.Series([110.0], index=idx),
        api4=pd.Series([90.0], index=idx),
        freight=pd.Series([12.0], index=idx),
        voyage_days=20.0,
        annual_rate=0.06,
        ets_cost=0.15345792,
    )
    row = frame.iloc[0]
    assert row["spread"] == pytest.approx(20.0)
    assert row["financing"] == pytest.approx(0.2958904109589041, rel=1e-12)
    assert row["arb"] == pytest.approx(7.550651669041096, rel=1e-12)
    assert bool(row["is_open"]) is True


def test_arb_requires_common_dates():
    with pytest.raises(ValueError, match="aucune date commune"):
        reconstruct_ara_arb(
            api2=pd.Series([110.0], index=pd.DatetimeIndex(["2024-01-01"])),
            api4=pd.Series([90.0], index=pd.DatetimeIndex(["2024-01-02"])),
            freight=pd.Series([12.0], index=pd.DatetimeIndex(["2024-01-03"])),
        )


def test_arb_rejects_ets_series_with_gaps():
    """Une série ETS qui ne couvre pas tout l'arb doit lever, pas trouer le calcul."""
    idx = _dates(5)
    with pytest.raises(ValueError, match="ne couvre pas toutes les dates"):
        reconstruct_ara_arb(
            api2=pd.Series(110.0, index=idx),
            api4=pd.Series(90.0, index=idx),
            freight=pd.Series(12.0, index=idx),
            ets_cost=pd.Series(0.15, index=idx[:3]),
        )


# --------------------------------------------------------------------------- MCO
def test_ols_recovers_exact_coefficients():
    """y = 1 + 2·x1 + 3·x2 exactement -> constante 1, pentes 2 et 3, R² = 1.

    Ajustement parfait : l'écart-type est nul et le t n'est pas défini. Le code doit
    renvoyer NaN plutôt qu'un infini qui contaminerait un tableau.
    """
    idx = _dates(50)
    x1 = pd.Series(np.linspace(1, 10, 50), index=idx)
    x2 = pd.Series(np.linspace(-3, 4, 50) ** 2, index=idx)
    y = 1.0 + 2.0 * x1 + 3.0 * x2
    r = ols(y, {"x1": x1, "x2": x2})
    assert r.coefficients["const"] == pytest.approx(1.0, abs=1e-8)
    assert r.coefficients["x1"] == pytest.approx(2.0, abs=1e-8)
    assert r.coefficients["x2"] == pytest.approx(3.0, abs=1e-8)
    assert r.r_squared == pytest.approx(1.0, abs=1e-12)
    assert np.isnan(r.t_stats["x1"])
    assert r.n_obs == 50


def test_ols_needs_enough_observations():
    idx = _dates(2)
    with pytest.raises(ValueError, match="pas assez d'observations"):
        ols(pd.Series([1.0, 2.0], index=idx), {"x": pd.Series([1.0, 2.0], index=idx)})


def test_ols_requires_a_regressor():
    idx = _dates(10)
    with pytest.raises(ValueError, match="au moins un régresseur"):
        ols(pd.Series(np.arange(10.0), index=idx), {})


def test_omitting_the_control_biases_the_freight_coefficient():
    """LE test qui justifie l'existence de `ols` à plusieurs régresseurs.

    Construction : un facteur commun z, `freight = z + a`, `ttf = z + b`, avec a et b
    indépendants et de même variance que z. La vérité est y = 1·freight + 5·ttf.

    Régression simple sur le fret seul :
        coef = 1 + 5 · cov(ttf, freight)/var(freight) = 1 + 5 · (1/2) = 3,5
    Le fret hérite de la moitié de l'effet du TTF. Avec le contrôle, on retrouve 1 et 5
    exactement.

    Autrement dit : sans contrôler par le TTF, on attribuerait au fret un effet 3,5 fois
    trop grand — et le raisonnement symétrique s'applique à l'attribution du décrochage
    post-2022 à l'Inde.
    """
    idx = _dates(600)
    rng = np.random.default_rng(5)
    z = rng.normal(0, 1, 600)
    freight = pd.Series(z + rng.normal(0, 1, 600), index=idx)
    ttf = pd.Series(z + rng.normal(0, 1, 600), index=idx)
    y = 1.0 * freight + 5.0 * ttf

    naive = ols(y, {"freight": freight})
    controlled = ols(y, {"freight": freight, "ttf": ttf})

    assert naive.coefficients["freight"] > 2.5  # biaisé, ~3,5
    assert controlled.coefficients["freight"] == pytest.approx(1.0, abs=1e-8)
    assert controlled.coefficients["ttf"] == pytest.approx(5.0, abs=1e-8)


# ------------------------------------------------------------------------ régimes
def test_regime_stats_splits_and_measures_persistence():
    """Avant la rupture : arb alterné, jamais deux jours de suite ouverts.
    Après : arb positif tous les jours, donc une séquence ouverte de 10 jours.
    """
    idx = _dates(20, start="2021-12-20")
    arb = np.where(np.arange(20) % 2 == 0, 1.0, -1.0)
    arb[10:] = 3.0
    frame = pd.DataFrame({"arb": arb}, index=idx)
    frame["is_open"] = frame["arb"] > 0
    bp = idx[10]
    before, after = regime_stats(frame, bp)

    assert before.n_obs == 10
    assert after.n_obs == 10
    assert before.share_open == pytest.approx(0.5)
    assert after.share_open == pytest.approx(1.0)
    assert before.longest_open_run == 1
    assert after.longest_open_run == 10
    assert after.arb_mean > before.arb_mean


def test_regime_stats_handles_empty_side():
    idx = _dates(5, start="2024-01-01")
    frame = pd.DataFrame({"arb": np.ones(5), "is_open": [True] * 5}, index=idx)
    before, after = regime_stats(frame, "2020-01-01")
    assert before.n_obs == 0
    assert after.n_obs == 5


def test_freight_binding_test_detects_a_collapsing_coefficient():
    """Avant la rupture, spread = 1,0 × fret (le fret est la contrainte).
    Après, spread = 0,1 × fret + un niveau (le fret ne contraint plus).
    """
    idx = _dates(400, start="2021-01-01")
    rng = np.random.default_rng(9)
    freight = pd.Series(12.0 + rng.normal(0, 2, 400), index=idx)
    bp = idx[200]
    spread = pd.Series(index=idx, dtype=float)
    pre = idx < bp
    spread[pre] = 1.0 * freight[pre]
    spread[~pre] = 0.1 * freight[~pre] + 25.0

    results = freight_binding_test(spread, freight, bp)
    labels = list(results)
    assert results[labels[0]].coefficients["freight"] == pytest.approx(1.0, abs=1e-6)
    assert results[labels[1]].coefficients["freight"] == pytest.approx(0.1, abs=1e-6)
