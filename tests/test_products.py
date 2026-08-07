"""Golden tests du projet C — valeurs calculées à la main dans les commentaires."""
import numpy as np
import pandas as pd
import pytest

from freight.chains.products import (
    DEFAULT_BBL_PER_TONNE,
    decompose_freight_change,
    density_sensitivity,
    flat_rate_step_series,
    freight_model_vs_quoted,
    freight_usd_per_tonne,
    implied_freight_from_tce,
    january_reset_effect,
    monthly_profile,
    open_days_comparison,
    reconstruct_arb,
    usd_per_gallon_to_usd_per_tonne,
)
from freight.signals.worldscale import FlatRateMissing, FlatRateTable
from freight.voyage.config import VoyageParams
from freight.voyage.tce import VoyageInputs, compute_tce

ROUTE = "TC14"
TABLE = FlatRateTable(rates={ROUTE: {2023: 20.0, 2024: 24.0, 2025: 26.0}})


def _dates(n: int, start: str = "2023-12-20") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


# ----------------------------------------------------------------- volume vs masse
def test_gallon_to_tonne_golden():
    """2,50 $/gal × 42 gal/bbl = 105,00 $/bbl × 7,45 bbl/t = 782,25 $/t."""
    assert usd_per_gallon_to_usd_per_tonne(2.50) == pytest.approx(782.25, rel=1e-12)
    assert usd_per_gallon_to_usd_per_tonne(2.50, DEFAULT_BBL_PER_TONNE) == pytest.approx(782.25)


def test_density_error_is_bigger_than_a_typical_arb():
    """LE résultat central du projet C, chiffré à la main.

    À 2,50 $/gal : 7,45 bbl/t -> 782,25 $/t ; 7,50 bbl/t -> 787,50 $/t.
    Écart : 5,25 $/t pour une variation de densité de 0,67 %.

    Un arb distillat transatlantique vit souvent entre 0 et 15 $/t. Le facteur de
    conversion que tout le monde traite comme une constante pèse donc autant que le
    signal.
    """
    at_745 = usd_per_gallon_to_usd_per_tonne(2.50, 7.45)
    at_750 = usd_per_gallon_to_usd_per_tonne(2.50, 7.50)
    assert at_750 - at_745 == pytest.approx(5.25, rel=1e-12)
    assert at_745 == pytest.approx(782.25)
    assert at_750 == pytest.approx(787.50)


def test_density_sensitivity_table_is_centred_on_the_default():
    table = density_sensitivity(2.50)
    default_row = table[table["bbl_par_tonne"] == DEFAULT_BBL_PER_TONNE].iloc[0]
    assert default_row["écart au défaut ($/t)"] == pytest.approx(0.0)
    # monotone croissante en densité
    assert table["prix ($/t)"].is_monotonic_increasing


def test_gallon_conversion_rejects_bad_density():
    with pytest.raises(ValueError):
        usd_per_gallon_to_usd_per_tonne(2.50, 0.0)


# --------------------------------------------------------------------- Worldscale
def test_flat_rate_step_series_is_a_staircase():
    idx = pd.DatetimeIndex(["2023-12-29", "2024-01-02", "2024-06-03", "2025-01-02"])
    s = flat_rate_step_series(idx, ROUTE, TABLE)
    assert list(s.values) == [20.0, 24.0, 24.0, 26.0]


def test_missing_flat_rate_is_loud():
    """Un repli silencieux sur l'année précédente est l'erreur que le projet dénonce."""
    idx = pd.DatetimeIndex(["2026-01-02"])
    with pytest.raises(FlatRateMissing):
        flat_rate_step_series(idx, ROUTE, TABLE)


def test_freight_conversion_golden():
    """WS 150 avec un flat rate de 20 $/t -> 150/100 × 20 = 30,00 $/t."""
    idx = _dates(1)
    f = freight_usd_per_tonne(pd.Series([150.0], index=idx), pd.Series([20.0], index=idx))
    assert f.iloc[0] == pytest.approx(30.0)


def test_freight_change_decomposition_is_an_exact_identity():
    """WS 150 -> 160, flat rate 20 -> 24.

    fret_prev = 150/100 × 20 = 30,00
    fret      = 160/100 × 24 = 38,40
    Δfret     = 8,40

    part marché  = ΔWS × FR_prev / 100 = 10 × 20 / 100 = 2,00
    part réglage = WS_prev × ΔFR / 100 = 150 × 4 / 100 = 6,00
    part croisée = ΔWS × ΔFR / 100     = 10 × 4 / 100  = 0,40
    somme        = 8,40  ->  identité exacte, pas une approximation
    """
    idx = pd.DatetimeIndex(["2023-12-29", "2024-01-02"])
    d = decompose_freight_change(
        pd.Series([150.0, 160.0], index=idx), pd.Series([20.0, 24.0], index=idx)
    )
    row = d.iloc[0]
    assert row["freight_usd_t"] == pytest.approx(38.40, rel=1e-12)
    assert row["d_freight"] == pytest.approx(8.40, rel=1e-12)
    assert row["market_component"] == pytest.approx(2.00, rel=1e-12)
    assert row["reset_component"] == pytest.approx(6.00, rel=1e-12)
    assert row["cross_component"] == pytest.approx(0.40, rel=1e-12)
    assert row["market_component"] + row["reset_component"] + row["cross_component"] == (
        pytest.approx(row["d_freight"], rel=1e-12)
    )


def test_reset_moves_cost_with_zero_market_move():
    """LE résultat Worldscale, dans sa forme la plus nue.

    Points WS strictement inchangés à 150, flat rate 20 -> 24 au 1er janvier :
      Δfret = 150 × 4 / 100 = 6,00 $/t
      part marché = 0, part croisée = 0

    Six dollars la tonne de coût supplémentaire sans qu'un seul point de fret n'ait bougé.
    Un modèle qui suit les points WS ne voit rien.
    """
    idx = pd.DatetimeIndex(["2023-12-29", "2024-01-02"])
    d = decompose_freight_change(
        pd.Series([150.0, 150.0], index=idx), pd.Series([20.0, 24.0], index=idx)
    )
    row = d.iloc[0]
    assert row["market_component"] == pytest.approx(0.0)
    assert row["cross_component"] == pytest.approx(0.0)
    assert row["reset_component"] == pytest.approx(6.0, rel=1e-12)
    assert row["d_freight"] == pytest.approx(6.0, rel=1e-12)


def test_january_reset_effect_finds_only_the_reset_dates():
    idx = pd.bdate_range("2023-12-27", "2024-01-05")
    ws = pd.Series(np.linspace(150, 155, len(idx)), index=idx)
    fr = flat_rate_step_series(idx, ROUTE, TABLE)
    resets = january_reset_effect(decompose_freight_change(ws, fr))
    assert len(resets) == 1
    assert pd.Timestamp(resets.iloc[0]["date"]).year == 2024
    assert resets.iloc[0]["part_reglage"] > 0


def test_decomposition_needs_two_observations():
    idx = _dates(1)
    with pytest.raises(ValueError, match="au moins deux observations"):
        decompose_freight_change(
            pd.Series([150.0], index=idx), pd.Series([20.0], index=idx)
        )


# --------------------------------------------------------------------------- arb
def test_arb_reconstruction_golden():
    """P_ARA = 800 $/t, P_USGC = 2,50 $/gal, fret = 30 $/t, spec = 4 $/t,
    16 jours à 6 %, pertes 1 $/t.

    p_origin_t  = 782,25
    spread_naif = 800 − 782,25 = 17,75
    financement = 782,25 × 0,06 = 46,935 ; × 16 = 750,96 ; / 365 = 2,0574246575...
    arb         = 17,75 − 30 − 4 − 2,0574246575 − 1 = −19,3074246575

    Autrement dit : le spread nu affiche +17,75 et l'arb réel est très négatif. C'est
    exactement l'illusion que la section S4 quantifie.
    """
    idx = _dates(1)
    frame = reconstruct_arb(
        price_destination_usd_t=pd.Series([800.0], index=idx),
        price_origin_usd_per_gallon=pd.Series([2.50], index=idx),
        freight_usd_t=pd.Series([30.0], index=idx),
        spec_bridge_usd_t=4.0,
        voyage_days=16.0,
        annual_rate=0.06,
        losses_usd_t=1.0,
    )
    row = frame.iloc[0]
    assert row["p_origin_t"] == pytest.approx(782.25, rel=1e-12)
    assert row["spread_naif"] == pytest.approx(17.75, rel=1e-12)
    assert row["financing"] == pytest.approx(2.0574246575342465, rel=1e-12)
    assert row["arb"] == pytest.approx(-19.307424657534247, rel=1e-12)
    assert bool(row["looks_open_naif"]) is True
    assert bool(row["is_open"]) is False


def test_arb_requires_common_dates():
    with pytest.raises(ValueError, match="aucune date commune"):
        reconstruct_arb(
            price_destination_usd_t=pd.Series([800.0], index=pd.DatetimeIndex(["2024-01-01"])),
            price_origin_usd_per_gallon=pd.Series([2.5], index=pd.DatetimeIndex(["2024-01-02"])),
            freight_usd_t=pd.Series([30.0], index=pd.DatetimeIndex(["2024-01-03"])),
        )


def test_open_days_comparison_measures_the_illusion():
    """10 jours dont 8 ont l'air ouverts et 2 le restent -> illusion de 75 %."""
    idx = _dates(10)
    frame = pd.DataFrame(
        {
            "looks_open_naif": [True] * 8 + [False] * 2,
            "is_open": [True] * 2 + [False] * 8,
        },
        index=idx,
    )
    c = open_days_comparison(frame)
    assert c.days_looking_open == 8
    assert c.days_really_open == 2
    assert c.share_looking_open == pytest.approx(0.8)
    assert c.illusion_share == pytest.approx(0.75)


def test_open_days_comparison_handles_never_open():
    idx = _dates(5)
    frame = pd.DataFrame(
        {"looks_open_naif": [False] * 5, "is_open": [False] * 5}, index=idx
    )
    c = open_days_comparison(frame)
    assert c.days_looking_open == 0
    assert np.isnan(c.illusion_share)


def test_monthly_profile_groups_by_calendar_month():
    idx = pd.date_range("2024-01-01", "2024-12-31", freq="MS")
    s = pd.Series(range(12), index=idx, dtype=float)
    profile = monthly_profile(s)
    assert len(profile) == 12
    assert profile.loc[1, "mean"] == pytest.approx(0.0)
    assert profile.loc[12, "mean"] == pytest.approx(11.0)


# ------------------------------------------------------------- C-2 : fret calculé
def test_implied_freight_round_trips_through_the_tce_engine():
    """Le test qui valide la variante C-2.

    On construit un voyage MR, on calcule son TCE à un taux de fret donné, puis on inverse
    et on doit retrouver exactement le taux de départ. Si cette boucle ne ferme pas, le
    fret reconstruit est faux et toute la variante C-2 s'écroule.
    """
    params = VoyageParams(
        cargo_t=37_000.0,
        laden_speed_kn=13.0,
        ballast_speed_kn=14.0,
        reference_speed_kn=13.0,
        reference_consumption_t_per_day=25.0,
        port_costs_usd=120_000.0,
        brokerage_commission=0.0375,
    )
    inputs = VoyageInputs(
        cargo_t=37_000.0,
        freight_rate_usd_per_t=32.0,
        distance_laden_nm=5_000.0,
        distance_ballast_nm=5_000.0,
        bunker_price_usd_per_t=600.0,
        port_days=4.0,
        params=params,
    )
    result = compute_tce(inputs)
    recovered = implied_freight_from_tce(
        target_tce_usd_per_day=result.tce_usd_per_day,
        total_days=result.total_days,
        total_voyage_costs_usd=result.total_voyage_costs_usd,
        cargo_t=37_000.0,
        commission=0.0375,
    )
    assert recovered == pytest.approx(32.0, rel=1e-9)


def test_implied_freight_increases_with_target_tce():
    kwargs = dict(
        total_days=25.0, total_voyage_costs_usd=900_000.0, cargo_t=37_000.0, commission=0.0375
    )
    low = implied_freight_from_tce(10_000.0, **kwargs)
    high = implied_freight_from_tce(30_000.0, **kwargs)
    assert high > low


def test_implied_freight_rejects_bad_inputs():
    kwargs = dict(total_voyage_costs_usd=900_000.0, cargo_t=37_000.0, commission=0.0375)
    with pytest.raises(ValueError):
        implied_freight_from_tce(10_000.0, total_days=0.0, **kwargs)
    with pytest.raises(ValueError):
        implied_freight_from_tce(
            10_000.0, total_days=25.0, total_voyage_costs_usd=1.0, cargo_t=0.0,
            commission=0.0375,
        )
    with pytest.raises(ValueError):
        implied_freight_from_tce(
            10_000.0, total_days=25.0, total_voyage_costs_usd=1.0, cargo_t=37_000.0,
            commission=1.0,
        )


def test_freight_model_vs_quoted_reports_a_systematic_gap():
    """Un modèle systématiquement 10 % au-dessus du coté doit ressortir comme tel."""
    idx = _dates(40)
    quoted = pd.Series(np.linspace(25, 40, 40), index=idx)
    modelled = quoted * 1.10
    stats = freight_model_vs_quoted(modelled, quoted)
    assert stats["ecart_moyen_pct"] == pytest.approx(10.0, rel=1e-9)
    assert stats["correlation"] == pytest.approx(1.0, rel=1e-9)
    assert stats["ecart_moyen_usd_t"] > 0
