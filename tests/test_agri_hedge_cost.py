"""Golden tests T1-2 — le coût complet de la couverture.

Le test le plus important du fichier est `test_roll_sign_*` : un roll mal signé transforme
un coût de couverture en revenu de couverture **sans rien casser**, et fait mentir toute
la page dans le sens le plus flatteur.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agri.chains.hedge_cost import (
    LONG_HEDGE,
    SHORT_HEDGE,
    HedgeCostError,
    HedgeParams,
    calibrate_im_k,
    compare_sides,
    hedge_capacity,
    initial_margin_proxy,
    load_real_hedge_frame,
    margin_breakeven_im_usd_t,
    procyclicality,
    roll_cost_usd_t,
    roll_pnl_usd_t,
    simulate_hedge,
)
from agri.data.bloomberg_loader import DEFAULT_PATH
from agri.fixtures.hedge_cost import PEAK_DATE, WINDOWS, build


# ===========================================================================
# LE SIGNE DU ROLL — le test qui protège toute la page
# ===========================================================================
def test_roll_sign_backwardation_costs_the_short():
    """Front 10 000, déféré 9 600 : le short rachète haut et se retrouve court à 9 600.

    Le contrat converge ensuite vers le spot, donc remonte : il perd.
        coût = +1 x (10 000 - 9 600) = +400 $/t
    C'est le cas Barry Callebaut : short futures contre physique long, backwardation
    rapportée comme un coût.
    """
    assert roll_cost_usd_t(10_000.0, 9_600.0, side=SHORT_HEDGE) == pytest.approx(400.0)
    assert roll_pnl_usd_t(10_000.0, 9_600.0, side=SHORT_HEDGE) == pytest.approx(-400.0)


def test_roll_sign_backwardation_pays_the_long():
    """Symétrique exact : ce que le short paie, le long l'encaisse."""
    assert roll_cost_usd_t(10_000.0, 9_600.0, side=LONG_HEDGE) == pytest.approx(-400.0)


def test_roll_sign_contango_pays_the_short():
    # front 2 400, déféré 2 450 : le short est court plus haut, le prix redescend -> il gagne
    assert roll_cost_usd_t(2_400.0, 2_450.0, side=SHORT_HEDGE) == pytest.approx(-50.0)


def test_roll_sign_contango_costs_the_long():
    assert roll_cost_usd_t(2_400.0, 2_450.0, side=LONG_HEDGE) == pytest.approx(50.0)


def test_the_two_sides_are_exactly_mirrored():
    for front, deferred in ((10_000.0, 9_600.0), (2_400.0, 2_450.0), (5_000.0, 5_000.0)):
        short = roll_cost_usd_t(front, deferred, side=SHORT_HEDGE)
        long = roll_cost_usd_t(front, deferred, side=LONG_HEDGE)
        assert short == pytest.approx(-long)


def test_flat_curve_costs_nothing_to_roll():
    assert roll_cost_usd_t(5_000.0, 5_000.0, side=SHORT_HEDGE) == 0.0


def test_invalid_side_is_rejected():
    with pytest.raises(HedgeCostError, match="side"):
        roll_cost_usd_t(5_000.0, 4_900.0, side=0)


# ===========================================================================
# Marge initiale
# ===========================================================================
def test_im_proxy_scales_with_volatility_and_price():
    index = pd.date_range("2024-01-01", periods=60, freq="B")
    calm = pd.Series(1_000.0 * np.exp(np.linspace(0, 0.01, 60)), index=index)
    im_calm = initial_margin_proxy(calm).dropna()
    assert (im_calm > 0).all()

    rng = np.random.default_rng(0)
    wild = pd.Series(
        1_000.0 * np.exp(np.cumsum(rng.normal(scale=0.05, size=60))), index=index
    )
    im_wild = initial_margin_proxy(wild).dropna()
    assert im_wild.mean() > im_calm.mean()


def test_im_proxy_calibration_recovers_a_known_k():
    """H-H2 : `k` est calibré sur des points publiés, jamais posé au doigt mouillé."""
    rng = np.random.default_rng(1)
    index = pd.date_range("2024-01-01", periods=200, freq="B")
    price = pd.Series(2_000.0 * np.exp(np.cumsum(rng.normal(scale=0.02, size=200))), index=index)
    true_k = 3.1
    observed = initial_margin_proxy(price, k=true_k)
    assert calibrate_im_k(price, observed) == pytest.approx(true_k, abs=1e-9)


def test_negative_k_is_rejected():
    index = pd.date_range("2024-01-01", periods=40, freq="B")
    with pytest.raises(HedgeCostError, match="k must"):
        initial_margin_proxy(pd.Series(1_000.0, index=index), k=-1.0)


# ===========================================================================
# La simulation, terme à terme
# ===========================================================================
@pytest.fixture
def tiny_simulation() -> pd.DataFrame:
    """Trois jours, tout calculé à la main.

        front     = [1 000, 1 100, 1 050]
        déféré    = [  990, 1 080, 1 040]        (backwardation partout)
        im_usd_t  = [   50,    60,    55]
        Q = 1 000 t, lot 10 t -> 100 lots

        im_usd     = im_usd_t x 10 x 100        = [50 000, 60 000, 55 000]
        vm_usd     = -1 x 1 000 x diff(front)   = [0, -100 000, +50 000]
        cumul      =                              [0, -100 000,  -50 000]
        cum_loss   = max(0, -cumul)             = [0,  100 000,   50 000]
        cash       = im + cum_loss              = [50 000, 160 000, 105 000]

        taux 5 % + 250 bps = 7,5 %
        financement = cash x 0,075 / 360        = [10,4167, 33,3333, 21,875]

        roll au jour 2 : +1 x (1 100 - 1 080) x 1 000 = 20 000
        liquidité au jour 2 : 1 000 x 4              =  4 000
    """
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    front = pd.Series([1_000.0, 1_100.0, 1_050.0], index=index)
    deferred = pd.Series([990.0, 1_080.0, 1_040.0], index=index)
    im_usd_t = pd.Series([50.0, 60.0, 55.0], index=index)
    params = HedgeParams(
        side=SHORT_HEDGE,
        book_size_t=1_000.0,
        lot_size_t=10.0,
        funding_spread_bps=250.0,
        liquidity_spread_usd_t=4.0,
    )
    return simulate_hedge(
        front, deferred, im_usd_t, 0.05, pd.DatetimeIndex([index[1]]), params=params
    )


def test_variation_margin_hand_computed(tiny_simulation):
    assert tiny_simulation["vm_usd"].tolist() == [0.0, -100_000.0, 50_000.0]


def test_cumulative_loss_never_goes_negative(tiny_simulation):
    assert tiny_simulation["cum_loss_usd"].tolist() == [0.0, 100_000.0, 50_000.0]


def test_initial_margin_is_charged_in_whole_lots(tiny_simulation):
    assert tiny_simulation["im_usd"].tolist() == [50_000.0, 60_000.0, 55_000.0]


def test_cash_is_margin_plus_unrecovered_losses(tiny_simulation):
    assert tiny_simulation["cash_usd"].tolist() == [50_000.0, 160_000.0, 105_000.0]


def test_financing_hand_computed(tiny_simulation):
    expected = [10.416667, 33.333333, 21.875]
    assert tiny_simulation["financing_usd"].tolist() == pytest.approx(expected, abs=1e-6)


def test_roll_and_liquidity_hit_only_on_roll_dates(tiny_simulation):
    assert tiny_simulation["roll_usd"].tolist() == pytest.approx([0.0, 20_000.0, 0.0])
    assert tiny_simulation["liquidity_usd"].tolist() == pytest.approx([0.0, 4_000.0, 0.0])


def test_cumulative_cost_hand_computed(tiny_simulation):
    # 10,4167 + (33,3333 + 20 000 + 4 000) + 21,875 = 24 065,625 sur 1 000 t
    assert tiny_simulation["hedge_cost_cum_usd"].iloc[-1] == pytest.approx(24_065.625, abs=1e-3)
    assert tiny_simulation["hedge_cost_cum_usd_t"].iloc[-1] == pytest.approx(24.065625, abs=1e-6)


def test_variation_margin_is_not_counted_as_a_cost(tiny_simulation):
    """La VM est un transfert, pas un coût. Ce qui coûte, c'est de la financer."""
    total = tiny_simulation["hedge_cost_cum_usd"].iloc[-1]
    assert abs(total) < abs(tiny_simulation["vm_usd"]).sum()


def test_a_rate_series_starting_late_is_rejected():
    """Un taux qui démarre après les prix laisse un trou en tête, que le report ne comble pas.

    C'est le cas dangereux : reporter en arrière un taux qu'on ne connaissait pas encore
    inventerait un coût de financement. Le report ne va que vers l'avant.
    """
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    rate = pd.Series([0.05], index=index[-1:])
    with pytest.raises(HedgeCostError, match="financing rate"):
        simulate_hedge(
            pd.Series(1_000.0, index=index),
            pd.Series(990.0, index=index),
            pd.Series(50.0, index=index),
            rate,
            pd.DatetimeIndex([]),
            params=HedgeParams(book_size_t=1_000.0),
        )


def test_simulation_rejects_disjoint_calendars():
    a = pd.Series(1_000.0, index=pd.date_range("2024-01-01", periods=3, freq="D"))
    b = pd.Series(990.0, index=pd.date_range("2025-01-01", periods=3, freq="D"))
    with pytest.raises(HedgeCostError, match="no common date"):
        simulate_hedge(a, b, b, 0.05, pd.DatetimeIndex([]), params=HedgeParams())


# ===========================================================================
# Capacité de couverture — le cœur du projet
# ===========================================================================
def test_hedge_capacity_hand_computed(tiny_simulation):
    """Q*(t) = B / cash_t(1 tonne).

    Au jour 2, cash = 160 000 $ pour 1 000 t, soit 160 $/t. Avec 10 M$ de lignes :
        Q* = 10 000 000 / 160 = 62 500 t
    """
    out = hedge_capacity(tiny_simulation, credit_line_usd=10_000_000.0, book_size_t=1_000.0)
    assert out.capacity_t.iloc[1] == pytest.approx(62_500.0)
    assert out.capacity_t.iloc[0] == pytest.approx(200_000.0)   # 10e6 / 50
    assert out.peak_cash_usd == pytest.approx(160_000.0)


def test_capacity_contracts_exactly_when_cash_peaks(tiny_simulation):
    out = hedge_capacity(tiny_simulation, credit_line_usd=10_000_000.0, book_size_t=1_000.0)
    assert out.capacity_t.idxmin() == tiny_simulation["cash_usd"].idxmax()


def test_capacity_is_binding_when_the_line_is_too_small(tiny_simulation):
    """Au pic, le cash vaut 160 $/t. La ligne contraint dès qu'elle passe sous 160 k$
    pour un book de 1 000 t.

        100 000 / 160 = 625 t couvrables < 1 000 t de book  -> contraignante
        1 000 000 / 160 = 6 250 t                            -> non contraignante
    """
    tight = hedge_capacity(tiny_simulation, credit_line_usd=100_000.0, book_size_t=1_000.0)
    assert tight.min_capacity_t == pytest.approx(625.0)
    assert tight.is_binding

    loose = hedge_capacity(tiny_simulation, credit_line_usd=1_000_000.0, book_size_t=1_000.0)
    assert loose.min_capacity_t == pytest.approx(6_250.0)
    assert not loose.is_binding


def test_margin_breakeven_hand_computed():
    """IM* = lignes / book. 250 M$ pour 100 kt -> 2 500 $/t de marge initiale."""
    assert margin_breakeven_im_usd_t(
        credit_line_usd=250_000_000.0, book_size_t=100_000.0
    ) == pytest.approx(2_500.0)


def test_margin_breakeven_falls_as_losses_accumulate():
    """Les pertes cumulées mangent la ligne : le seuil de marge tolérable s'effondre."""
    fresh = margin_breakeven_im_usd_t(credit_line_usd=250e6, book_size_t=100_000.0)
    bruised = margin_breakeven_im_usd_t(
        credit_line_usd=250e6, book_size_t=100_000.0, cumulative_loss_usd=200e6
    )
    assert bruised == pytest.approx(500.0)
    assert bruised < fresh


def test_margin_breakeven_floors_at_zero_when_the_line_is_exhausted():
    assert margin_breakeven_im_usd_t(
        credit_line_usd=250e6, book_size_t=100_000.0, cumulative_loss_usd=300e6
    ) == 0.0


def test_zero_credit_line_is_rejected(tiny_simulation):
    with pytest.raises(HedgeCostError, match="credit line"):
        hedge_capacity(tiny_simulation, credit_line_usd=0.0, book_size_t=1_000.0)


# ===========================================================================
# Sur le jeu synthétique complet
# ===========================================================================
@pytest.fixture(scope="module")
def full():
    data = build()
    im = initial_margin_proxy(data["front"])
    params = HedgeParams()
    simulation = simulate_hedge(
        data["front"], data["deferred"], im, data["rate"], data["roll_dates"], params=params
    )
    return data, im, params, simulation


def test_hedging_the_2024_cocoa_market_costs_money_not_makes_it(full):
    """Contrôle de sens global. Un coût cumulé négatif signalerait un roll mal signé."""
    _, _, _, simulation = full
    assert simulation["hedge_cost_cum_usd_t"].iloc[-1] > 0


def test_all_three_cost_components_are_material(full):
    _, _, params, simulation = full
    per_tonne = lambda column: simulation[column].sum() / params.book_size_t
    assert per_tonne("financing_usd") > 0
    assert per_tonne("roll_usd") > 0
    assert per_tonne("liquidity_usd") > 0


def test_margin_is_procyclical(full):
    """La marge initiale monte quand le prix monte : le coût explose quand la couverture
    est le plus nécessaire. C'est l'insulte finale de la thèse."""
    _, _, _, simulation = full
    stats = procyclicality(simulation)
    assert stats["corr_delta_im_delta_price"] > 0.2


def test_cash_peaks_near_the_price_peak(full):
    _, _, _, simulation = full
    gap_days = abs((simulation["cash_usd"].idxmax() - PEAK_DATE).days)
    assert gap_days < 120


def test_the_credit_line_binds_on_the_2024_book(full):
    _, _, params, simulation = full
    capacity = hedge_capacity(
        simulation, credit_line_usd=params.credit_line_usd, book_size_t=params.book_size_t
    )
    assert capacity.is_binding
    assert capacity.min_capacity_t < params.book_size_t
    assert "binding" in capacity.headline


def test_both_sides_are_punished_in_their_own_window(full):
    """LA thèse du projet : la couverture a puni les deux camps, à quatorze mois d'écart.

    Sur la hausse, c'est le short qui mobilise le plus de trésorerie ; sur la baisse,
    c'est le long. Le coût par tonne ne suffit pas à le voir — c'est le **cash au pic**
    qui contraint, et c'est lui qui bascule d'un camp à l'autre.
    """
    data, im, params, _ = full
    table = compare_sides(
        data["front"],
        data["deferred"],
        im,
        data["rate"],
        data["roll_dates"],
        params=params,
        windows=WINDOWS,
    )
    rise = table[table["window"].str.startswith("hausse")].set_index("side")
    fall = table[table["window"].str.startswith("baisse")].set_index("side")

    assert rise.loc["short (trader)", "peak_cash_usd"] > rise.loc["long (manufacturer)", "peak_cash_usd"]
    assert fall.loc["long (manufacturer)", "peak_cash_usd"] > fall.loc["short (trader)", "peak_cash_usd"]


def test_fixture_imposes_backwardation_around_the_peak(full):
    data, _, _, _ = full
    around_peak = slice("2024-09-01", "2025-03-01")
    assert (data["deferred"].loc[around_peak] < data["front"].loc[around_peak]).mean() > 0.95


def test_fixture_imposes_a_price_collapse(full):
    data, _, _, _ = full
    assert data["front"].max() / data["front"].iloc[0] > 3.0
    assert data["front"].iloc[-1] < data["front"].max() / 2.0


# ===========================================================================
# Sur données réelles (export Bloomberg de l'utilisateur) — crise cacao 2022-2024
# ===========================================================================
pytestmark_real = pytest.mark.skipif(
    not DEFAULT_PATH.exists(), reason=f"fichier Bloomberg absent : {DEFAULT_PATH}"
)


@pytest.fixture(scope="module")
def real_cacao():
    params = HedgeParams(side=SHORT_HEDGE, book_size_t=100_000.0, credit_line_usd=250_000_000.0)
    return load_real_hedge_frame("cacao_ny", params=params), params


@pytest.mark.skipif(not DEFAULT_PATH.exists(), reason=f"fichier Bloomberg absent : {DEFAULT_PATH}")
def test_real_cocoa_peak_matches_the_actual_2024_crisis(real_cacao):
    """Le prix reel du cacao ICE NY a culmine a 12 565 USD/t le 18 decembre 2024 —
    verifie a la main contre le fichier source, pas contre le code qui le lit."""
    sim, _ = real_cacao
    assert sim["front"].max() == pytest.approx(12565.0)
    assert sim["front"].idxmax() == pd.Timestamp("2024-12-18")


@pytest.mark.skipif(not DEFAULT_PATH.exists(), reason=f"fichier Bloomberg absent : {DEFAULT_PATH}")
def test_real_cash_mobilised_exceeds_one_billion_at_the_peak(real_cacao):
    """Au pic reel, le cash mobilise depasse 1 milliard USD sur un book de 100 kt — un
    chiffre que Barry Callebaut n'a pas invente, et que ce calcul retrouve independamment."""
    sim, _ = real_cacao
    peak_cash = sim.loc[sim["front"].idxmax(), "cash_usd"]
    assert peak_cash == pytest.approx(1_079_501_592.69, rel=1e-6)
    assert peak_cash > 1e9


@pytest.mark.skipif(not DEFAULT_PATH.exists(), reason=f"fichier Bloomberg absent : {DEFAULT_PATH}")
def test_real_roll_cost_is_explicitly_zero_not_estimated(real_cacao):
    """Limite de donnee assumee : sans echeance differee reelle, le roll est neutralise
    (deferred = front), pas approxime sur une hypothese de structure par terme inventee."""
    sim, _ = real_cacao
    assert (sim["roll_usd"] == 0.0).all()
    assert (sim["liquidity_usd"] == 0.0).all()
    assert sim.attrs["roll_cost_omitted"] is True


@pytest.mark.skipif(not DEFAULT_PATH.exists(), reason=f"fichier Bloomberg absent : {DEFAULT_PATH}")
def test_real_credit_line_binds_at_the_actual_peak(real_cacao):
    sim, params = real_cacao
    capacity = hedge_capacity(sim, credit_line_usd=params.credit_line_usd, book_size_t=params.book_size_t)
    assert capacity.is_binding
    assert capacity.peak_cash_date == pd.Timestamp("2024-12-18")


@pytest.mark.skipif(not DEFAULT_PATH.exists(), reason=f"fichier Bloomberg absent : {DEFAULT_PATH}")
def test_real_coffee_price_hits_its_genuine_historical_record():
    """Le cafe arabica ICE a reellement depasse 400 c/lb en 2025 — record historique, pas
    un artefact de synthese."""
    sim = load_real_hedge_frame("cafe_arabica")
    assert sim["front"].max() > 400.0


@pytest.mark.skipif(not DEFAULT_PATH.exists(), reason=f"fichier Bloomberg absent : {DEFAULT_PATH}")
def test_real_frame_starts_at_sofr_coverage():
    """start par defaut cale sur le debut de couverture SOFR (2018-04-02) : au-dela, le
    taux de financement reel ne couvre pas la periode."""
    sim = load_real_hedge_frame("cacao_ny")
    assert sim.index.min() >= pd.Timestamp("2018-04-02")


@pytest.mark.skipif(not DEFAULT_PATH.exists(), reason=f"fichier Bloomberg absent : {DEFAULT_PATH}")
def test_unknown_commodity_alias_falls_back_to_raw_bloomberg_key():
    """Un alias non reconnu est essaye tel quel comme cle bloomberg_loader — permet de
    passer 'coffee_robusta' directement sans devoir l'ajouter a REAL_COMMODITY_KEYS."""
    sim = load_real_hedge_frame("coffee_robusta")
    assert len(sim) > 100


def test_fixture_is_deterministic():
    a = build(seed=5)["front"]
    b = build(seed=5)["front"]
    pd.testing.assert_series_equal(a, b)
