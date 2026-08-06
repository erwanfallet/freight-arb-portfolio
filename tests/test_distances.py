import pytest

from freight.voyage.distances import route_distance_nm, great_circle_nm, NAMED_ROUTES


def test_named_route_returns_documented_figure():
    assert route_distance_nm("TUBARAO", "QINGDAO") == NAMED_ROUTES[("TUBARAO", "QINGDAO")]


def test_named_route_c3_c5_ratio_matches_doc_geographic_constant():
    d3 = route_distance_nm("TUBARAO", "QINGDAO")
    d5 = route_distance_nm("PORT_HEDLAND", "QINGDAO")
    # Partie 3.4: "le ratio D3/D5 ~= 2.9 est une constante geographique"
    assert d3 / d5 == pytest.approx(11_000 / 3_500, rel=1e-6)


def test_great_circle_falls_back_for_unnamed_pair():
    dist = route_distance_nm("ROTTERDAM", "NEW_YORK")
    assert dist == pytest.approx(great_circle_nm("ROTTERDAM", "NEW_YORK"))
    assert dist > 0


def test_unknown_port_raises():
    with pytest.raises(KeyError):
        route_distance_nm("ATLANTIS", "QINGDAO")
