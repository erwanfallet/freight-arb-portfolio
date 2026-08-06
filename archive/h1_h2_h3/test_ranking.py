from freight.signals.ranking import ChainResult, build_ranking_table, h1_verdict


def test_h1_falsified_when_rankings_agree_on_4plus_chains():
    # both rankings agree exactly: freight/value and switching-share ordered the same way
    results = [
        ChainResult("iron_ore", freight_value_ratio=0.25, freight_attributable_share=0.20, n_flips=10),
        ChainResult("coal", freight_value_ratio=0.18, freight_attributable_share=0.15, n_flips=8),
        ChainResult("grains", freight_value_ratio=0.12, freight_attributable_share=0.10, n_flips=6),
        ChainResult("copper", freight_value_ratio=0.015, freight_attributable_share=0.01, n_flips=4),
    ]
    df = build_ranking_table(results)
    verdict = h1_verdict(df)
    assert verdict["n_chains"] == 4
    assert verdict["h1_falsified"] is True


def test_h1_survives_when_rankings_diverge_on_4plus_chains():
    # deliberately inverted: highest freight/value has the LOWEST switching share, and
    # vice versa — the exact pattern the doc's "two columns, arrows crossing" graph wants
    results = [
        ChainResult("iron_ore", freight_value_ratio=0.25, freight_attributable_share=0.05, n_flips=10),
        ChainResult("coal", freight_value_ratio=0.18, freight_attributable_share=0.10, n_flips=8),
        ChainResult("grains", freight_value_ratio=0.12, freight_attributable_share=0.30, n_flips=6),
        ChainResult("copper", freight_value_ratio=0.015, freight_attributable_share=0.60, n_flips=4),
    ]
    df = build_ranking_table(results)
    verdict = h1_verdict(df)
    assert verdict["n_chains"] == 4
    assert verdict["h1_falsified"] is False


def test_verdict_withheld_below_minimum_chain_count():
    results = [
        ChainResult("iron_ore", freight_value_ratio=0.25, freight_attributable_share=0.20, n_flips=10),
        ChainResult("coal", freight_value_ratio=0.18, freight_attributable_share=0.15, n_flips=8),
    ]
    df = build_ranking_table(results)
    verdict = h1_verdict(df)
    assert verdict["n_chains"] == 2
    assert verdict["h1_falsified"] is None
