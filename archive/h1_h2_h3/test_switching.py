import pandas as pd

from freight.signals.switching import detect_and_attribute_flips, freight_attributable_share

DATES = pd.date_range("2024-01-01", periods=5, freq="D")


def _series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=DATES)


def test_attribution_of_each_cause():
    # t0 baseline: price=10, freight=9  -> arb=+1
    # t1: freight jumps to 11           -> arb=-1  (freight-caused flip)
    # t2: price jumps to 13             -> arb=+2  (price-caused flip)
    # t3: price drops to 9, freight to 13 -> arb=-4 (both-caused flip)
    # t4: price=12, freight=9.5         -> arb=+2.5 (ambiguous: neither alone flips it)
    price = _series([10, 10, 13, 9, 12])
    freight = _series([9, 11, 11, 13, 9.5])

    flips = detect_and_attribute_flips(price, freight)

    assert list(flips["attribution"]) == ["freight", "price", "both", "ambiguous"]
    assert list(flips["prev_sign"]) == [1, -1, 1, -1]
    assert list(flips["new_sign"]) == [-1, 1, -1, 1]


def test_freight_attributable_share_weighting():
    price = _series([10, 10, 13, 9, 12])
    freight = _series([9, 11, 11, 13, 9.5])
    flips = detect_and_attribute_flips(price, freight)
    # freight=1.0, price=0.0, both=0.5, ambiguous=0.5 -> mean = 0.5
    assert freight_attributable_share(flips) == 0.5


def test_no_flips_when_arb_never_changes_sign():
    price = _series([10, 10.5, 11, 10.8, 10.2])
    freight = _series([1, 1, 1, 1, 1])  # arb stays comfortably positive throughout
    flips = detect_and_attribute_flips(price, freight)
    assert flips.empty


def test_freight_attributable_share_nan_when_no_flips():
    import math

    flips = detect_and_attribute_flips(_series([10] * 5), _series([1] * 5))
    assert math.isnan(freight_attributable_share(flips))
