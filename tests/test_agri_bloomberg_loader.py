"""Golden tests du loader Bloomberg reel.

Ces tests lisent le vrai fichier de l'utilisateur (~/Desktop/Data Bloomberg.xlsx) : ils
sont skippes proprement si le fichier est absent, pour que la suite reste executable sur
une autre machine.
"""
from __future__ import annotations

import pandas as pd
import pytest

from agri.data.bloomberg_loader import (
    DEFAULT_PATH,
    SERIES_SPECS,
    BloombergLoaderError,
    detect_unit_jumps,
    load,
    load_raw_series,
)

pytestmark = pytest.mark.skipif(
    not DEFAULT_PATH.exists(), reason=f"fichier Bloomberg absent : {DEFAULT_PATH}"
)


def test_all_registered_series_load_without_error():
    for key in SERIES_SPECS:
        series = load(key)
        assert len(series) > 100, f"{key} : trop peu d'observations ({len(series)})"
        assert isinstance(series.index, pd.DatetimeIndex)


def test_unknown_key_raises():
    with pytest.raises(BloombergLoaderError, match="cle inconnue"):
        load("ce_ticker_n_existe_pas")


def test_dropna_removes_leading_blank_prints():
    """Bloomberg exporte parfois une premiere ligne datee sans valeur — un futur non
    encore fixe. `dropna=True` (defaut) doit l'ecarter."""
    with_na = load_raw_series("henry_hub", dropna=False)
    without_na = load_raw_series("henry_hub", dropna=True)
    assert len(without_na) <= len(with_na)
    assert without_na.isna().sum() == 0


def test_detect_unit_jumps_catches_the_known_jet_swap_defect():
    """Le controle qui a permis de trouver le defaut reel : jet_swap_m1 alterne entre
    USD/gal et c/gal plusieurs fois dans son historique. Ce test verrouille la
    detection — s'il devient vert par accident (donnee corrigee cote Bloomberg), le
    module doit etre reevalue, pas le test supprime en silence."""
    jumps = detect_unit_jumps(load("jet_swap_m1"))
    assert len(jumps) >= 5


def test_detect_unit_jumps_is_clean_on_jet_spot():
    """Contraste direct : la serie propre ne declenche rien — sinon le controle serait
    un test qui crie tout le temps, donc inutile."""
    jumps = detect_unit_jumps(load("jet_spot"))
    assert len(jumps) == 0


def test_detect_unit_jumps_is_clean_on_ulsd_ttf_henry_hub_eurusd():
    for key in ("ulsd", "ttf", "henry_hub", "eurusd"):
        jumps = detect_unit_jumps(load(key))
        assert len(jumps) == 0, f"{key} porte {len(jumps)} saut(s) suspect(s) inattendu(s)"


def test_eurusd_is_in_the_usd_per_eur_convention():
    """Verification de sens de cotation (L-H4 de lng_netback) : les valeurs recentes
    doivent etre de l'ordre de 1,0-1,3 (USD par EUR), pas 0,7-0,95 (EUR par USD)."""
    recent = load("eurusd").tail(250)
    assert recent.between(0.9, 1.5).mean() > 0.95


# ===========================================================================
# Extension : softs, grains CBOT, DCE, change (piege cents/boisseau + cruzeiro)
# ===========================================================================
def test_detect_unit_jumps_is_clean_on_all_new_series():
    new_keys = [
        "cocoa_ny", "cocoa_london", "coffee_arabica", "coffee_robusta",
        "sugar_no11", "sugar_no5", "cbot_soybean", "cbot_corn", "cbot_wheat",
        "cbot_soymeal", "cbot_soyoil", "dce_soymeal", "dce_soyoil", "usdbrl", "usdcny",
    ]
    for key in new_keys:
        jumps = detect_unit_jumps(load(key))
        assert len(jumps) == 0, f"{key} porte {len(jumps)} saut(s) suspect(s) inattendu(s)"


def test_cbot_grains_are_converted_from_cents_to_dollars_per_bushel():
    """Le piege trouve en construisant cette extension : les grains CBOT sont cotes en
    CENTS par boisseau dans cet export (1156,50 = 11,565 USD/bu), pas en dollars. Sans le
    scale=0.01, toute formule attendant du USD/bu (board_crush_usd_bu, financing_cost...)
    sortirait un resultat cent fois trop grand."""
    soy = load("cbot_soybean")
    corn = load("cbot_corn")
    wheat = load("cbot_wheat")
    # ordres de grandeur reels 2026 : soja ~9-14 USD/bu, mais ~3-6, ble ~5-9
    assert 8.0 < soy.iloc[-1] < 16.0
    assert 3.0 < corn.iloc[-1] < 7.0
    assert 4.0 < wheat.iloc[-1] < 10.0


def test_cbot_soymeal_and_soyoil_need_no_scaling():
    """Contraste direct avec le test precedent : ces deux-la sont deja dans l'unite
    economique native (USD/short ton, c/lb) — un scale=0.01 par reflexe les casserait."""
    meal = load("cbot_soymeal")
    oil = load("cbot_soyoil")
    assert 100.0 < meal.iloc[-1] < 700.0        # USD/short ton, ordre de grandeur reel
    assert 10.0 < oil.iloc[-1] < 100.0          # c/lb, ordre de grandeur reel


def test_usdbrl_excludes_the_pre_plano_real_era():
    """Avant juillet 1994 : cruzeiro/cruzeiro real d'avant la reforme monetaire
    bresilienne (hyperinflation, valeurs ~0,0004). Monnaie differente, pas une valeur
    aberrante — exclue via valid_from plutot que reechelonnee."""
    usdbrl = load("usdbrl")
    assert usdbrl.index.min() >= pd.Timestamp("1994-07-01")
    assert (usdbrl > 0.3).all()  # plus de valeurs quasi nulles de l'ere cruzeiro


def test_cocoa_ny_peak_matches_the_real_2024_crisis():
    """Controle de coherence externe : le cacao a reellement culmine autour de
    12 000 USD/t en avril 2024 (Barry Callebaut, source T1-2). Si ce pic n'apparaissait
    pas ici, ce serait le signe d'un re-echelonnage ou d'une serie tronquee."""
    cocoa = load("cocoa_ny")
    peak = cocoa.loc["2024-01-01":"2024-12-31"]
    assert peak.max() > 9000.0


def test_sofr_is_a_decimal_fraction_not_a_percent():
    """Défaut trouvé en construisant T1-2, et corrigé ici.

    Bloomberg cote le SOFR en POURCENTS (5,40 au pic du resserrement 2023). Additionné
    tel quel à un spread déjà exprimé en décimal (250 bps → 0,025) puis utilisé comme un
    décimal, il produisait un taux tout compris de 243 % et gonflait le coût de
    financement d'un facteur ~100 — sans jamais lever d'erreur.

    Le contrat du loader est désormais : **tout taux sort en fraction décimale**, prêt à
    multiplier un montant.
    """
    sofr = load("sofr")
    assert sofr.max() < 0.15, "le SOFR ressort en pourcents — le scale=0.01 a sauté"
    assert 0.03 < sofr.max() < 0.08, "le pic du resserrement 2023 doit être autour de 5,4 %"
    assert (sofr >= 0).all()


def test_financing_cost_is_plausible_once_sofr_is_decimal():
    """Contrôle de bout en bout : le taux tout compris appliqué dans la simulation doit
    rester dans une plage de marché, pas à 243 %."""
    from agri.chains.hedge_cost import (
        SHORT_HEDGE,
        HedgeParams,
        load_real_hedge_frame,
    )

    params = HedgeParams(side=SHORT_HEDGE, book_size_t=100_000.0, credit_line_usd=250e6)
    simulation = load_real_hedge_frame("cacao_ny", params=params)
    implied_rate = (simulation["financing_usd"] / simulation["cash_usd"] * 360).median()
    assert 0.01 < implied_rate < 0.12, f"taux tout compris implicite de {implied_rate:.1%}"


def test_dce_series_are_in_cny_thousands_not_usd():
    """Controle d'ordre de grandeur : le tourteau/huile DCE cotent en CNY/t (milliers),
    pas en USD/t (centaines) — une confusion frequente entre les deux marches."""
    meal = load("dce_soymeal")
    oil = load("dce_soyoil")
    assert 1000.0 < meal.iloc[-1] < 8000.0
    assert 3000.0 < oil.iloc[-1] < 20000.0
