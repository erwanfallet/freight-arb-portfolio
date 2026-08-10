"""Golden tests T1-1 — le fret dans le calcul C&F.

Le fret sur le voyage de référence (Panamax, Santos -> Qingdao, TCE 15 000, VLSFO 500,
MGO 700) est **linéaire en ballast_share** :

    freight(b) = 20,610258 + b x 15,816610   $/t

C'est cette linéarité qui rend le point de bascule calculable à la main, et c'est elle qui
est vérifiée en premier ci-dessous.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agri.chains.freight_cf import (
    NoBreakevenInRange,
    arb_usd_t,
    ballast_breakeven,
    build_conventions,
    disagreement_episodes,
    financing_cost_usd_t,
    marginal_decision_zone,
    pnl_attribution,
    sensitivity_grid,
    sign_flip_rate,
    spread_distribution,
    spread_seasonality,
)
from agri.core.voyage import ROUTES, VESSELS, VoyageParams
from agri.fixtures.freight_cf import build, build_frame

PANAMAX = VESSELS["panamax"]
SANTOS_QINGDAO = ROUTES["santos_qingdao"]

FREIGHT_AT_ZERO_BALLAST = 20.610258
FREIGHT_SLOPE_PER_BALLAST = 15.816610


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return build_frame()


# ===========================================================================
# L'identité comptable
# ===========================================================================
def test_financing_uses_a_360_day_base():
    # (440 + 36) x 5,5 % x (78 + 30) / 360 = 476 x 0,055 x 0,3 = 7,854
    assert financing_cost_usd_t(
        440.0, 36.0, annual_rate=0.055, voyage_days=78.0, credit_days=30.0
    ) == pytest.approx(7.854, abs=1e-3)


def test_arb_identity_is_a_plain_subtraction():
    # 500 - 440 - 36 - financing - 0,85
    # financing = (440 + 36) x 0,055 x (78 + 30)/360 = 7,854
    # arb = 500 - 440 - 36 - 7,854 - 0,85 = 15,296
    index = pd.date_range("2024-01-01", periods=3, freq="B")
    out = arb_usd_t(
        pd.Series(500.0, index=index),
        pd.Series(440.0, index=index),
        pd.Series(36.0, index=index),
        annual_rate=0.055,
        voyage_days=78.0,
        credit_days=30.0,
        insurance_usd_t=0.85,
    )
    assert out.iloc[0] == pytest.approx(15.296, abs=1e-3)


def test_negative_financing_rate_is_rejected():
    with pytest.raises(Exception, match="negative"):
        financing_cost_usd_t(440.0, 36.0, annual_rate=-0.01, voyage_days=78.0, credit_days=30.0)


# ===========================================================================
# Les trois conventions
# ===========================================================================
def test_freight_is_linear_in_ballast_share():
    """La propriété qui rend le point de bascule calculable à la main."""
    from agri.core.voyage import voyage_freight_usd_t

    for share in (0.0, 0.25, 0.5, 0.75, 1.0):
        expected = FREIGHT_AT_ZERO_BALLAST + share * FREIGHT_SLOPE_PER_BALLAST
        actual = voyage_freight_usd_t(
            15_000.0,
            500.0,
            700.0,
            vessel=PANAMAX,
            route=SANTOS_QINGDAO,
            params=VoyageParams(ballast_share=share),
        ).freight_usd_t
        assert actual == pytest.approx(expected, abs=1e-5)


def test_full_is_always_above_index(frame):
    """L'invariant central : le ballast ne peut que renchérir, jamais l'inverse.

    Un `freight_full < freight_index` persistant serait une erreur de modèle de voyage,
    pas une opportunité — c'est le premier mode de défaillance listé par la spec.
    """
    assert (frame["freight_full"] > frame["freight_index"]).all()
    assert (frame["spread_full_index"] > 0).all()


def test_internal_convention_lags_the_full_one(frame):
    """La convention interne est stable *donc* en retard — c'est tout son intérêt et tout son défaut."""
    assert frame["freight_internal"].std() < frame["freight_full"].std()
    correlation = frame["freight_internal"].corr(frame["freight_full"])
    assert 0.0 < correlation < 0.99


def test_internal_convention_refuses_a_partial_window():
    """Une moyenne « interne » sur trois points n'est pas une moyenne interne."""
    series = build(periods=200, seed=1)
    out = build_conventions(
        series["tce"],
        series["vlsfo"],
        series["mgo"],
        series["cif"],
        series["fob"],
        vessel=PANAMAX,
        route=SANTOS_QINGDAO,
        params=VoyageParams(),
        internal_window_days=90,
    )
    # 200 jours d'entree - 89 manges par la fenetre = 111 jours exploitables
    assert len(out) == 111


def test_arb_is_more_generous_under_the_index_convention(frame):
    """Le desk trading voit systématiquement un arb plus ouvert. C'est le désaccord."""
    assert (frame["arb_index"] > frame["arb_full"]).all()
    assert frame["arb_index"].mean() - frame["arb_full"].mean() == pytest.approx(
        frame["spread_full_index"].mean(), abs=0.5
    )


def test_disagreement_flag_marks_sign_conflicts(frame):
    signs = np.sign(frame[["arb_index", "arb_full", "arb_internal"]].to_numpy())
    expected = (signs != signs[:, [0]]).any(axis=1)
    assert (frame["disagreement"].to_numpy() == expected).all()
    assert frame["disagreement"].any()


# ===========================================================================
# S4 — le panneau de désaccord
# ===========================================================================
def test_sign_flip_rate_carries_an_exact_interval(frame):
    out = sign_flip_rate(frame)
    assert 0.0 < out.point < 1.0
    assert out.lo < out.point < out.hi
    assert "IC 95" in out.summary


def test_sign_flip_rate_is_zero_when_conventions_agree():
    """Sur un arb largement ouvert, les conventions s'accordent — et il faut le dire."""
    index = pd.date_range("2024-01-01", periods=120, freq="B")
    wide = pd.DataFrame(
        {"arb_index": pd.Series(80.0, index=index), "arb_full": pd.Series(60.0, index=index)}
    )
    assert sign_flip_rate(wide).point == 0.0


def test_spread_distribution_reports_median_and_iqr_not_just_the_mean(frame):
    """La distribution est bornée à gauche par construction : une moyenne seule ment."""
    stats = spread_distribution(frame)
    assert stats["min"] > 0
    assert stats["q1"] < stats["median"] < stats["q3"]
    assert stats["iqr"] == pytest.approx(stats["q3"] - stats["q1"])


def test_spread_seasonality_covers_twelve_months(frame):
    seasonal = spread_seasonality(frame)
    assert set(seasonal.index) <= set(range(1, 13))
    assert len(seasonal) == 12


def test_disagreement_episodes_are_dated_and_measurable(frame):
    episodes = disagreement_episodes(frame, min_days=3)
    assert len(episodes) > 0
    assert (episodes["n_obs"] >= 3).all()
    assert "duration_days" in episodes.columns


# ===========================================================================
# S5 — la zone de décision marginale : le chiffre du mail
# ===========================================================================
def test_marginal_zone_produces_the_headline(frame):
    zone = marginal_decision_zone(frame, band_usd_t=5.0)
    assert zone.n_in_band > 0
    assert 0.0 < zone.share_of_sample <= 1.0
    headline = zone.headline
    assert "5 USD/t of breakeven" in headline
    assert "95% exact CI" in headline
    assert "freight convention" in headline


def test_a_wider_band_captures_more_days(frame):
    narrow = marginal_decision_zone(frame, band_usd_t=2.0)
    wide = marginal_decision_zone(frame, band_usd_t=10.0)
    assert wide.n_in_band > narrow.n_in_band


def test_marginal_zone_is_empty_when_the_arb_is_never_close():
    index = pd.date_range("2024-01-01", periods=50, freq="B")
    wide = pd.DataFrame(
        {"arb_index": pd.Series(80.0, index=index), "arb_full": pd.Series(60.0, index=index)}
    )
    zone = marginal_decision_zone(wide, band_usd_t=5.0)
    assert zone.n_in_band == 0
    assert zone.share_of_sample == 0.0


def test_zero_band_is_rejected():
    frame = pd.DataFrame({"arb_index": [1.0], "arb_full": [1.0]})
    with pytest.raises(Exception, match="band"):
        marginal_decision_zone(frame, band_usd_t=0.0)


# ===========================================================================
# Point de bascule — ballast_share*
# ===========================================================================
def test_ballast_breakeven_hand_computed():
    """Sans financement ni assurance, l'arb est affine en ballast et le seuil est exact.

        arb(b) = (CIF - FOB) - 20,610258 - b x 15,816610

    En posant CIF - FOB = 28,518563, la racine tombe exactement à b* = 0,50 :
        (28,518563 - 20,610258) / 15,816610 = 7,908305 / 15,816610 = 0,500000
    """
    out = ballast_breakeven(
        15_000.0,
        500.0,
        700.0,
        cif_usd_t=468.518563,
        fob_usd_t=440.0,
        vessel=PANAMAX,
        route=SANTOS_QINGDAO,
        params=VoyageParams(),
        annual_rate=0.0,
        insurance_usd_t=0.0,
    )
    assert out.theta_star == pytest.approx(0.5, abs=1e-6)
    # la sensibilité est la pente du fret, au signe près : -15,8166 $/t par unité de ballast
    assert out.sensitivity == pytest.approx(-FREIGHT_SLOPE_PER_BALLAST, abs=1e-3)


def test_breakeven_summary_is_the_mail_sentence():
    out = ballast_breakeven(
        15_000.0,
        500.0,
        700.0,
        cif_usd_t=468.518563,
        fob_usd_t=440.0,
        vessel=PANAMAX,
        route=SANTOS_QINGDAO,
        params=VoyageParams(),
        annual_rate=0.0,
        insurance_usd_t=0.0,
    )
    assert "ballast_share* = 0.5" in out.summary


def test_financing_pushes_the_breakeven_down():
    """Le financement est un coût de plus : il ferme l'arb plus tôt, donc à moins de ballast."""
    kwargs = dict(
        cif_usd_t=468.518563,
        fob_usd_t=440.0,
        vessel=PANAMAX,
        route=SANTOS_QINGDAO,
        params=VoyageParams(),
        insurance_usd_t=0.0,
    )
    free = ballast_breakeven(15_000.0, 500.0, 700.0, annual_rate=0.0, **kwargs)
    costly = ballast_breakeven(15_000.0, 500.0, 700.0, annual_rate=0.055, **kwargs)
    assert costly.theta_star < free.theta_star


def test_a_wide_open_arb_has_no_breakeven_and_that_is_the_result():
    """« Même en facturant 100 % du ballast, l'arb reste ouvert » est une affirmation publiable."""
    with pytest.raises(NoBreakevenInRange) as excinfo:
        ballast_breakeven(
            15_000.0,
            500.0,
            700.0,
            cif_usd_t=560.0,      # 120 $/t d'écart : très au-dessus de tout fret plausible
            fob_usd_t=440.0,
            vessel=PANAMAX,
            route=SANTOS_QINGDAO,
            params=VoyageParams(),
        )
    assert excinfo.value.margin_lo > 0
    assert "reste positive" in str(excinfo.value)


def test_a_shut_arb_has_no_breakeven_either():
    with pytest.raises(NoBreakevenInRange, match="reste négative"):
        ballast_breakeven(
            15_000.0,
            500.0,
            700.0,
            cif_usd_t=445.0,      # 5 $/t d'écart : fermé même sans facturer de ballast
            fob_usd_t=440.0,
            vessel=PANAMAX,
            route=SANTOS_QINGDAO,
            params=VoyageParams(),
        )


def test_breakeven_reports_distance_in_sigmas_when_history_is_given():
    history = pd.Series([0.3, 0.5, 0.7, 0.9])
    out = ballast_breakeven(
        15_000.0,
        500.0,
        700.0,
        cif_usd_t=468.518563,
        fob_usd_t=440.0,
        vessel=PANAMAX,
        route=SANTOS_QINGDAO,
        params=VoyageParams(ballast_share=0.8),
        annual_rate=0.0,
        insurance_usd_t=0.0,
        ballast_history=history,
    )
    assert out.theta_current == 0.8
    assert out.distance_sigmas is not None
    assert out.distance_sigmas < 0      # le seuil est en dessous du niveau retenu


# ===========================================================================
# S6 — sensibilité croisée
# ===========================================================================
def test_sensitivity_grid_shape_and_monotonicity():
    grid = sensitivity_grid(
        15_000.0,
        500.0,
        700.0,
        cif_usd_t=468.5,
        fob_usd_t=440.0,
        vessel=PANAMAX,
        route=SANTOS_QINGDAO,
        params=VoyageParams(),
        ballast_values=np.array([0.0, 0.5, 1.0]),
        bunker_shifts_pct=np.array([-0.2, 0.0, 0.2]),
    )
    assert len(grid) == 9
    # a decalage de soutes fixe, plus de ballast = arb plus ferme
    at_zero_shift = grid[grid["bunker_shift_pct"] == 0.0].sort_values("ballast_share")
    assert at_zero_shift["arb_usd_t"].is_monotonic_decreasing
    # a ballast fixe, soutes plus cheres = arb plus ferme
    at_full_ballast = grid[grid["ballast_share"] == 1.0].sort_values("bunker_shift_pct")
    assert at_full_ballast["arb_usd_t"].is_monotonic_decreasing


# ===========================================================================
# S7 — attribution de P&L
# ===========================================================================
def test_pnl_attribution_sign_convention(frame):
    """Signe positif = le département fret a facturé au-dessus de son coût du jour."""
    out = pnl_attribution(frame, cargo_t=66_000.0)
    assert (out["gap_usd_t"] * 66_000.0).equals(out["pnl_shifted_usd"])
    positive = out[out["gap_usd_t"] > 0]
    assert (positive["credited"] == "freight department").all()
    negative = out[out["gap_usd_t"] <= 0]
    assert (negative["credited"] == "trading desk").all()


def test_pnl_attribution_cumulates(frame):
    out = pnl_attribution(frame, cargo_t=66_000.0)
    assert out["pnl_shifted_cum_usd"].iloc[-1] == pytest.approx(out["pnl_shifted_usd"].sum())


def test_zero_cargo_is_rejected(frame):
    with pytest.raises(Exception, match="cargo_t"):
        pnl_attribution(frame, cargo_t=0.0)


# ===========================================================================
# Le jeu synthétique impose bien le phénomène
# ===========================================================================
def test_fixture_puts_a_large_share_of_days_in_the_marginal_band(frame):
    """Sans cette propriété, la page n'aurait rien à montrer — c'est le point du fixture."""
    zone = marginal_decision_zone(frame, band_usd_t=5.0)
    assert zone.share_of_sample > 0.3


def test_fixture_produces_genuine_sign_conflicts(frame):
    zone = marginal_decision_zone(frame, band_usd_t=5.0)
    assert zone.n_decided_by_convention > 0


def test_fixture_is_deterministic():
    a = build_frame(seed=7)
    b = build_frame(seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_give_different_data():
    a = build_frame(seed=1)
    b = build_frame(seed=2)
    assert not a["arb_full"].equals(b["arb_full"])
