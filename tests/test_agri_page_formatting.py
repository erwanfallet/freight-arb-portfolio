"""Tests du formatage francais applique a toutes les pages.

Ce n'est pas de la cosmetique. Python formate `1,843.5` ; un lecteur francophone lit
« 1,843 » comme mille huit cent quarante-trois virgule cinq — un facteur mille d'erreur sur
un prix. Sur un portefeuille dont chaque page repose sur des unites correctement traitees,
laisser passer une erreur de lecture des nombres serait cocasse.

Les tests couvrent surtout ce que le formateur ne doit PAS toucher : le code, les dates, les
numeros de section, et le texte deja francais.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from page_template import THIN_NBSP, _localise_numbers, _prose  # noqa: E402


# ===========================================================================
# Ce que le formateur doit convertir
# ===========================================================================
def test_decimal_point_becomes_a_comma():
    assert _localise_numbers("La décote vaut 4.46 c/lb.") == "La décote vaut 4,46 c/lb."


def test_thousands_comma_becomes_a_thin_space():
    assert _localise_numbers("1,843 BRL/t") == f"1{THIN_NBSP}843 BRL/t"


def test_both_conversions_apply_to_the_same_number():
    assert _localise_numbers("12,565.50 USD/t") == f"12{THIN_NBSP}565,50 USD/t"


def test_several_numbers_in_one_sentence():
    result = _localise_numbers("De 14.42 à 35.26 c/lb, soit 20.85 cents.")
    assert result == "De 14,42 à 35,26 c/lb, soit 20,85 cents."


# ===========================================================================
# Ce que le formateur ne doit PAS toucher
# ===========================================================================
def test_inline_code_is_protected():
    """Un appel de fonction est du code, pas du texte : son point décimal doit survivre."""
    text = "Appelle `substitution_verdict(spread, quantile=0.70)` puis lis 12.5 jours."
    result = _localise_numbers(text)
    assert "quantile=0.70" in result
    assert "12,5 jours" in result


def test_dates_are_untouched():
    assert _localise_numbers("Le 30/12/2026 et le 2005-07-21.") == "Le 30/12/2026 et le 2005-07-21."


def test_section_numbering_is_untouched():
    """« S3. Le reste » : le point suit un chiffre mais est suivi d'une espace, pas d'un
    chiffre — la lookahead l'exclut."""
    assert _localise_numbers("Voir S3. Le reste suit.") == "Voir S3. Le reste suit."


def test_already_french_numbers_survive_unchanged():
    assert _localise_numbers("Un facteur 134,47e-6 et 21 000 USD.") == "Un facteur 134,47e-6 et 21 000 USD."


def test_a_four_digit_group_is_not_a_thousands_separator():
    """`0,7000` n'est pas un séparateur de milliers : la lookahead exige exactement trois
    chiffres non suivis d'un quatrième."""
    assert _localise_numbers("beta = -0,05391") == "beta = -0,05391"


def test_percentages_and_ratios_convert_cleanly():
    assert _localise_numbers("soit 65.3 % du temps") == "soit 65,3 % du temps"


# ===========================================================================
# Le traitement complet applique à la prose
# ===========================================================================
def test_prose_escapes_dollars_and_localises_numbers():
    result = _prose("Le seuil est à 285.5 $/t contre 21,000 $/jour.")
    assert r"\$" in result
    assert "285,5" in result
    assert f"21{THIN_NBSP}000" in result


def test_prose_leaves_a_dollarless_sentence_alone_apart_from_numbers():
    assert _prose("Douze jours contre trente-neuf.") == "Douze jours contre trente-neuf."


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Aucun chiffre ici.",
        "`tout est du code`",
    ],
)
def test_degenerate_inputs_do_not_raise(text):
    assert isinstance(_prose(text), str)


def test_the_formatter_is_idempotent():
    """Appliqué deux fois, le résultat ne doit plus bouger — sinon une page qui formate deux
    fois (via un helper imbriqué) corromprait ses nombres."""
    once = _prose("De 1,843.50 à 12,565.00 $/t")
    assert _prose(once) == once
