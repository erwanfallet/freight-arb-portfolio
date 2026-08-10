"""Tests for the dollar-escaping helper shared by every page.

This is not cosmetic. Streamlit reads `$...$` as a math formula, so a sentence with two
literal dollar amounts — "55 USD/t" and "21,000 USD/t" — would have everything between
the two signs swallowed and rendered in math italics. Every page's prose runs through
this escape before reaching `st.markdown`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from page_template import _escape_dollars, _prose  # noqa: E402


def test_a_single_dollar_sign_is_escaped():
    assert _prose("The threshold sits at $285.5/t.") == r"The threshold sits at \$285.5/t."


def test_two_dollar_amounts_are_both_escaped():
    """The failure mode this guards against: Streamlit would otherwise treat everything
    between the two `$` as a math expression and swallow it."""
    result = _prose("From $14.42 to $35.26, a $20.85 swing.")
    assert result == r"From \$14.42 to \$35.26, a \$20.85 swing."


def test_text_without_a_dollar_sign_is_unchanged():
    assert _prose("Twelve days against thirty-nine.") == "Twelve days against thirty-nine."


def test_already_escaped_dollars_are_left_alone():
    assert _prose(r"Already escaped: \$100.") == r"Already escaped: \$100."


def test_the_escape_is_idempotent():
    """Applying it twice must not add a second backslash — a page that composes two
    template helpers (e.g. `_prose` inside `kpi_banner`) will run text through this more
    than once, and a naive `.replace('$', r'\\$')` would produce a visible `\\\\$`."""
    once = _prose("From $1,843.50 to $12,565.00 /t")
    assert _prose(once) == once


@pytest.mark.parametrize("text", ["", "No dollar amount here.", "`all code, no prose`"])
def test_degenerate_inputs_do_not_raise(text):
    assert isinstance(_prose(text), str)


def test_escape_dollars_is_the_same_function_as_prose():
    """Several pages import `_escape_dollars` directly under its old name."""
    assert _escape_dollars is _prose
