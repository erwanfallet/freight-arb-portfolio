"""Golden tests T1-1 sur la route P8 réelle — et l'arbitrage du désaccord de ballast.

Le résultat central de la page : la série P8 de l'export contient **deux régimes
d'unité**, un TCE en USD/jour (jul-oct 2021) puis un taux de voyage en USD/tonne
(nov 2021 →). Le premier, qu'il a fallu isoler comme défaut de données, sert ensuite de
banc d'essai au second : il donne le niveau de TCE réellement coté sur cette route au
**pic** du boom vraquier, donc un plafond de plausibilité.
"""
from __future__ import annotations

import pandas as pd
import pytest

from agri.chains.freight_cf import (
    FreightCfError,
    implied_tce_by_convention,
    load_real_route_frame,
    market_implied_ballast_share,
)
from agri.core.voyage import ROUTES, VESSELS, VoyageParams
from agri.data.bloomberg_loader import DEFAULT_PATH, load

pytestmark = pytest.mark.skipif(
    not DEFAULT_PATH.exists(), reason=f"fichier Bloomberg absent : {DEFAULT_PATH}"
)

PANAMAX = VESSELS["panamax"]
SANTOS_QINGDAO = ROUTES["santos_qingdao"]


# ===========================================================================
# Le défaut de donnée : deux régimes d'unité dans la même cellule
# ===========================================================================
def test_p8_splits_into_two_unit_regimes_with_no_overlap():
    """Le segment USD/jour s'arrête le 30/10/2021, le segment USD/t reprend le 18/11/2021 :
    19 jours de trou et **aucune date commune**. Le facteur de conversion ne peut donc pas
    être calibré sur la jonction — le marché a bougé entre les deux."""
    tce = load("p8_route_tce_2021")
    rate = load("p8_route_usd_t")

    assert tce.index.max() == pd.Timestamp("2021-10-30")
    assert rate.index.min() == pd.Timestamp("2021-11-18")
    assert len(tce.index.intersection(rate.index)) == 0
    assert (rate.index.min() - tce.index.max()).days == 19


def test_each_segment_is_internally_plausible_in_its_own_unit():
    """Un TCE Panamax de 24 500-38 000 USD/jour est le pic réel du boom 2021 ; un taux de
    voyage Santos-Qingdao de 36-85 USD/t est l'ordre de grandeur normal. Chaque segment
    est cohérent **dans son unité** — c'est leur juxtaposition qui ne l'est pas."""
    tce = load("p8_route_tce_2021")
    rate = load("p8_route_usd_t")

    assert 20_000 < tce.min() and tce.max() < 45_000
    assert 20 < rate.min() and rate.max() < 150


def test_the_impossible_zero_print_is_dropped():
    """Un fret nul au 30/04/2022 : physiquement impossible, et il casserait toute division
    en aval. Écarté par `min_valid` plutôt que laissé circuler."""
    rate = load("p8_route_usd_t")
    assert pd.Timestamp("2022-04-30") not in rate.index
    assert (rate > 0).all()


# ===========================================================================
# L'inversion : le même print, deux conventions
# ===========================================================================
@pytest.fixture(scope="module")
def spread():
    return load_real_route_frame()


def test_no_ballast_reading_always_implies_a_higher_tce(spread):
    """Invariant : à revenu identique, ne pas facturer le repositionnement à vide écrase
    la durée du cycle, donc gonfle le TCE dégagé. L'écart est strictement positif partout."""
    assert (spread.tce_no_ballast > spread.tce_full_ballast).all()
    assert (spread.spread > 0).all()


def test_the_gap_is_large_enough_to_be_the_whole_argument(spread):
    """L'écart médian entre les deux lectures dépasse 30 000 USD/jour — du même ordre que
    le TCE lui-même. Ce n'est pas un raffinement de modèle, c'est la question."""
    assert spread.spread.median() > 25_000
    assert "in the freight department's unit" in spread.headline


# ===========================================================================
# LE RÉSULTAT — le segment défectueux arbitre le désaccord
# ===========================================================================
def test_the_no_ballast_reading_implies_an_impossible_market(spread):
    """LE test de la page.

    Le segment TCE de 2021 donne le niveau réellement coté sur cette route au **pic** du
    boom vraquier : 38 000 USD/jour au plus haut. Lire le taux publié sans facturer le
    ballast implique un TCE supérieur à ce pic **quasiment tout le temps** sur 2021-2026 —
    ce qui reviendrait à dire que le marché a passé cinq ans au-dessus de son propre
    sommet. La lecture du desk trading est donc arithmétiquement intenable.
    """
    boom_peak = float(load("p8_route_tce_2021").max())
    share_above = float((spread.tce_no_ballast > boom_peak).mean())
    assert share_above > 0.90


def test_the_full_ballast_reading_is_plausible(spread):
    """Contraste direct : la même inversion en facturant le ballast reste sous le pic du
    boom presque tout le temps. C'est ce qui fait pencher la balance du côté du
    département fret — pas un argument d'autorité, une borne de plausibilité."""
    boom_peak = float(load("p8_route_tce_2021").max())
    share_above = float((spread.tce_full_ballast > boom_peak).mean())
    assert share_above < 0.10


def test_full_ballast_median_sits_below_the_2021_peak(spread):
    boom_peak = float(load("p8_route_tce_2021").max())
    assert spread.tce_full_ballast.median() < boom_peak
    assert spread.tce_no_ballast.median() > boom_peak


# ===========================================================================
# La part de ballast que le marché price
# ===========================================================================
def test_market_implied_ballast_share_recovers_a_known_input():
    """Contrôle de cohérence de l'inversion : on fabrique un taux avec une part de
    ballast connue, puis on vérifie que le solveur la retrouve."""
    from agri.core.voyage import voyage_freight_usd_t

    known_share = 0.6
    rate = voyage_freight_usd_t(
        15_000.0, 500.0, 700.0,
        vessel=PANAMAX, route=SANTOS_QINGDAO,
        params=VoyageParams(ballast_share=known_share),
    ).freight_usd_t

    result = market_implied_ballast_share(
        rate, 15_000.0, 500.0, 700.0,
        vessel=PANAMAX, route=SANTOS_QINGDAO, params=VoyageParams(),
    )
    assert result.implied_share == pytest.approx(known_share, abs=1e-6)
    assert "ballast repositioning" in result.headline


def test_a_rate_above_the_model_range_names_the_binding_end():
    """Un taux publié que le modèle ne peut pas produire est un signal d'hypothèse de
    voyage à revoir — et le message doit nommer la borne **contraignante**, pas la plus
    éloignée. L'écart étant décroissant en ballast, un taux trop haut est celui qui
    dépasse le modèle même à 100 % de ballast."""
    result = market_implied_ballast_share(
        500.0, 15_000.0, 500.0, 700.0,
        vessel=PANAMAX, route=SANTOS_QINGDAO, params=VoyageParams(),
    )
    assert result.implied_share is None
    assert "100%" in result.reason
    assert "too low" in result.reason


def test_a_rate_below_the_model_range_names_the_other_end():
    result = market_implied_ballast_share(
        5.0, 15_000.0, 500.0, 700.0,
        vessel=PANAMAX, route=SANTOS_QINGDAO, params=VoyageParams(),
    )
    assert result.implied_share is None
    assert "no ballast" in result.reason
    assert "too high" in result.reason


def test_implied_share_is_monotone_in_the_published_rate():
    """Le fret est affine croissant en ballast, donc un taux publié plus élevé implique
    une part de ballast plus élevée. C'est ce qui garantit l'unicité de la racine."""
    shares = [
        market_implied_ballast_share(
            rate, 15_000.0, 500.0, 700.0,
            vessel=PANAMAX, route=SANTOS_QINGDAO, params=VoyageParams(),
        ).implied_share
        for rate in (25.0, 30.0, 35.0)
    ]
    assert all(s is not None for s in shares)
    assert shares == sorted(shares)


def test_disjoint_calendars_raise():
    a = pd.Series([50.0], index=pd.to_datetime(["2024-01-01"]))
    b = pd.Series([500.0], index=pd.to_datetime(["2030-01-01"]))
    with pytest.raises(FreightCfError, match="no common date"):
        implied_tce_by_convention(
            a, b, b, vessel=PANAMAX, route=SANTOS_QINGDAO, params=VoyageParams()
        )
