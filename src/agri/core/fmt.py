"""Formatage numérique français, pour que les chiffres se lisent comme ils se prononcent.

Python formate en convention anglo-saxonne : `f"{1843.5:,.1f}"` donne `1,843.5`. Sur une
page rédigée en français, un lecteur lit « 1,843 » comme mille huit cent quarante-trois
**virgule** cinq — c'est-à-dire un facteur mille d'erreur sur un prix. Ce n'est pas une
question de style, c'est une erreur de lecture, et sur un portefeuille dont tout l'argument
tient à des unités correctement traitées, la laisser passer serait cocasse.

Convention retenue : espace fine insécable pour les milliers, virgule pour les décimales.
"""
from __future__ import annotations

THIN_NBSP = " "


def fr(value: float, decimals: int = 2, *, sign: bool = False) -> str:
    """Formate un nombre à la française : `fr(1843.5, 1)` -> `1 843,5`.

    `sign=True` force le signe explicite, pour les grandeurs dont la direction est le
    message (une marge, un écart, un P&L).
    """
    spec = f"{'+' if sign else ''},.{decimals}f"
    formatted = format(float(value), spec)
    integer, _, fraction = formatted.partition(".")
    integer = integer.replace(",", THIN_NBSP)
    return f"{integer},{fraction}" if fraction else integer


def fr_pct(value: float, decimals: int = 0) -> str:
    """Formate une fraction en pourcentage français : `fr_pct(0.653)` -> `65 %`.

    L'espace avant le signe pourcent est la règle typographique française, et elle est
    insécable pour éviter un retour à la ligne entre le nombre et son unité.
    """
    return f"{fr(value * 100.0, decimals)}{THIN_NBSP}%"


__all__ = ["THIN_NBSP", "fr", "fr_pct"]
