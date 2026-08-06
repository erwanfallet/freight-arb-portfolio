import pytest

from freight.signals.worldscale import FlatRateTable, FlatRateMissing


def test_ws_to_usd_conversion():
    table = FlatRateTable(rates={"TD3C": {2021: 12.0, 2022: 18.0}})
    assert table.ws_to_usd_per_t("TD3C", 2021, 100) == pytest.approx(12.0)
    assert table.ws_to_usd_per_t("TD3C", 2021, 200) == pytest.approx(24.0)


def test_same_ws_points_different_years_are_not_the_same_usd_rate():
    """The exact trap from Partie 2.7: WS180 in 2021 != WS180 in 2022 because the flat
    rate itself was recalculated on a different bunker/cost base.
    """
    table = FlatRateTable(rates={"TD3C": {2021: 12.0, 2022: 18.0}})
    usd_2021 = table.ws_to_usd_per_t("TD3C", 2021, 180)
    usd_2022 = table.ws_to_usd_per_t("TD3C", 2022, 180)
    assert usd_2021 != usd_2022
    assert usd_2022 == pytest.approx(usd_2021 * 1.5)


def test_missing_flat_rate_raises_loudly_instead_of_silently_guessing():
    table = FlatRateTable(rates={"TD3C": {2021: 12.0}})
    with pytest.raises(FlatRateMissing):
        table.ws_to_usd_per_t("TD3C", 2022, 180)
    with pytest.raises(FlatRateMissing):
        table.ws_to_usd_per_t("TC2", 2021, 180)


def test_round_trip_usd_to_ws_and_back():
    table = FlatRateTable(rates={"TC2": {2023: 20.0}})
    usd = table.ws_to_usd_per_t("TC2", 2023, 145)
    ws = table.usd_per_t_to_ws("TC2", 2023, usd)
    assert ws == pytest.approx(145)
